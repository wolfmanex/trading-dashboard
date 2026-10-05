"""Trade journal: the trades you actually take, their live P&L and stop/target alerts.

The journal is a CSV next to the signal log on the repo's `signal-log` branch, so the app and the
GitHub Actions jobs (intraday alerts, evening recap) read the same file. The app writes it through
the GitHub contents API, which needs a GITHUB_JOURNAL_TOKEN secret: a fine-grained token with
read and write access to this repository's contents. Without the token the app keeps the journal in
a local file (JOURNAL_PATH, default journal.csv), which is fine when running locally but is lost
whenever a hosted app restarts.
"""
import base64
import io
import os
import uuid
import pandas as pd
import requests

from secrets_config import get_configured_secret
from signal_log import LOG_BRANCH


JOURNAL_FILE = "journal.csv"
GITHUB_REPO = "wolfmanex/trading-dashboard"
CONTENTS_URL = "https://api.github.com/repos/{repo}/contents/{path}"

LONG, SHORT = "Long", "Short"
OPEN, CLOSED = "Open", "Closed"
JOURNAL_COLUMNS = [
    "ID", "Ticker", "Side", "Entry Date", "Entry", "Shares", "Stop", "Initial Stop", "Target", "Setup", "Notes",
    "Status", "Exit Date", "Exit", "Exit Reason", "P&L", "R",
]
ALERT_COLUMNS = ["Date", "ID", "Ticker", "Kind", "Price"]


def empty_journal() -> pd.DataFrame:
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def parse_journal(text: str) -> pd.DataFrame:
    if not text.strip():
        return empty_journal()
    journal = pd.read_csv(io.StringIO(text), dtype={"ID": str, "Ticker": str, "Notes": str, "Setup": str})
    for column in JOURNAL_COLUMNS:
        if column not in journal:
            journal[column] = None
    return journal[JOURNAL_COLUMNS]


def read_journal_file(path: str) -> pd.DataFrame:
    try:
        with open(path, encoding="utf-8") as handle:
            return parse_journal(handle.read())
    except FileNotFoundError:
        return empty_journal()


class JournalStore:
    """Load and save the journal on GitHub when a token is configured, else in a local file."""

    def __init__(self, token: str = None, repo: str = GITHUB_REPO, branch: str = LOG_BRANCH, path: str = None):
        self.token = token if token is not None else get_configured_secret("GITHUB_JOURNAL_TOKEN")
        self.repo, self.branch = repo, branch
        self.path = path or os.getenv("JOURNAL_PATH") or JOURNAL_FILE
        self.sha = None

    @property
    def remote(self) -> bool:
        return bool(self.token)

    @property
    def description(self) -> str:
        if self.remote:
            return f"saved to {JOURNAL_FILE} on the {self.branch} branch"
        return f"saved to the local file {self.path}; set GITHUB_JOURNAL_TOKEN to keep it on GitHub"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}

    def _url(self) -> str:
        return CONTENTS_URL.format(repo=self.repo, path=JOURNAL_FILE)

    def load(self) -> pd.DataFrame:
        if not self.remote:
            return read_journal_file(self.path)
        response = requests.get(self._url(), params={"ref": self.branch}, headers=self._headers(), timeout=15)
        if response.status_code == 404:
            self.sha = None
            return empty_journal()
        response.raise_for_status()
        payload = response.json()
        self.sha = payload.get("sha")
        return parse_journal(base64.b64decode(payload.get("content", "")).decode("utf-8"))

    def save(self, journal: pd.DataFrame, message: str) -> None:
        text = journal[JOURNAL_COLUMNS].to_csv(index=False)
        if not self.remote:
            with open(self.path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return
        body = {"message": message, "branch": self.branch, "content": base64.b64encode(text.encode()).decode()}
        if self.sha:
            body["sha"] = self.sha
        response = requests.put(self._url(), json=body, headers=self._headers(), timeout=20)
        if response.status_code == 409:
            raise RuntimeError("The journal changed elsewhere since it was loaded. Reload the page and try again.")
        response.raise_for_status()
        self.sha = response.json().get("content", {}).get("sha")


def _risk_per_share(side: str, entry: float, stop: float) -> float:
    return entry - stop if side == LONG else stop - entry


def add_trade(journal: pd.DataFrame, ticker: str, side: str, entry_date, entry: float, shares: float,
              stop: float, target: float = None, setup: str = "", notes: str = "") -> pd.DataFrame:
    """Append an open trade; the stop must be on the losing side of the entry."""
    ticker = str(ticker).strip().upper()
    if not ticker:
        raise ValueError("Ticker is required")
    if side not in (LONG, SHORT):
        raise ValueError(f"Side must be {LONG} or {SHORT}")
    if entry <= 0 or shares <= 0:
        raise ValueError("Entry price and shares must be positive")
    if _risk_per_share(side, entry, stop) <= 0:
        raise ValueError("The stop must be below the entry for a long and above it for a short")
    if target and (target - entry) * (1 if side == LONG else -1) <= 0:
        raise ValueError("The target must be above the entry for a long and below it for a short")
    row = {
        "ID": uuid.uuid4().hex[:8], "Ticker": ticker, "Side": side,
        "Entry Date": pd.Timestamp(entry_date).strftime("%Y-%m-%d"), "Entry": round(float(entry), 4),
        "Shares": float(shares), "Stop": round(float(stop), 4), "Initial Stop": round(float(stop), 4), "Target": round(float(target), 4) if target else None,
        "Setup": setup or "", "Notes": notes or "", "Status": OPEN,
    }
    new = pd.DataFrame([row], columns=JOURNAL_COLUMNS)
    return new if journal.empty else pd.concat([journal, new], ignore_index=True)


def initial_stop(row) -> float:
    """The stop the trade was opened with, which R is measured from even after the stop is moved."""
    value = row.get("Initial Stop")
    return float(value) if pd.notna(value) else float(row["Stop"])


def trade_result(side: str, entry: float, stop: float, shares: float, price: float) -> tuple:
    """(P&L in USD, result in R against the risk from entry to `stop`) for a position marked at `price`."""
    direction = 1 if side == LONG else -1
    pnl = (price - entry) * shares * direction
    risk = _risk_per_share(side, entry, stop)
    return round(pnl, 2), round((price - entry) * direction / risk, 2) if risk > 0 else None


def close_trade(journal: pd.DataFrame, trade_id: str, exit_price: float, exit_date, reason: str = "") -> pd.DataFrame:
    journal = journal.copy().astype({column: object for column in JOURNAL_COLUMNS})
    match = journal.index[(journal["ID"] == trade_id) & (journal["Status"] == OPEN)]
    if match.empty:
        raise ValueError("No open trade with that ID")
    if exit_price <= 0:
        raise ValueError("Exit price must be positive")
    index = match[0]
    row = journal.loc[index]
    pnl, r = trade_result(row["Side"], float(row["Entry"]), initial_stop(row), float(row["Shares"]), float(exit_price))
    journal.loc[index, ["Status", "Exit Date", "Exit", "Exit Reason", "P&L", "R"]] = [
        CLOSED, pd.Timestamp(exit_date).strftime("%Y-%m-%d"), round(float(exit_price), 4), reason or "", pnl, r,
    ]
    return journal


def update_stop(journal: pd.DataFrame, trade_id: str, stop: float) -> pd.DataFrame:
    """Move an open trade's stop (e.g. to breakeven). R stays measured from the initial stop."""
    journal = journal.copy()
    match = journal.index[(journal["ID"] == trade_id) & (journal["Status"] == OPEN)]
    if match.empty:
        raise ValueError("No open trade with that ID")
    journal.loc[match[0], "Stop"] = round(float(stop), 4)
    return journal


def open_positions(journal: pd.DataFrame) -> pd.DataFrame:
    return journal[journal["Status"] == OPEN] if not journal.empty else journal


def mark_positions(positions: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Open positions with the latest price, P&L, R and distance to stop and target."""
    rows = []
    for _, row in positions.iterrows():
        price = prices.get(row["Ticker"])
        entry, stop, shares = float(row["Entry"]), float(row["Stop"]), float(row["Shares"])
        target = float(row["Target"]) if pd.notna(row["Target"]) else None
        marked = {"ID": row["ID"], "Ticker": row["Ticker"], "Side": row["Side"], "Entry Date": row["Entry Date"],
                  "Entry": entry, "Shares": shares, "Stop": stop, "Target": target, "Price": price,
                  "P&L": None, "R": None, "To Stop %": None, "To Target %": None}
        if price:
            marked["P&L"], marked["R"] = trade_result(row["Side"], entry, initial_stop(row), shares, price)
            marked["To Stop %"] = round(abs(price - stop) / price * 100, 2)
            if target:
                marked["To Target %"] = round(abs(target - price) / price * 100, 2)
        rows.append(marked)
    return pd.DataFrame(rows)


def summarize_journal(journal: pd.DataFrame) -> dict:
    closed = journal[journal["Status"] == CLOSED] if not journal.empty else journal
    pnl = pd.to_numeric(closed["P&L"], errors="coerce").dropna() if not closed.empty else pd.Series(dtype=float)
    r = pd.to_numeric(closed["R"], errors="coerce").dropna() if not closed.empty else pd.Series(dtype=float)
    return {
        "open": int((journal["Status"] == OPEN).sum()) if not journal.empty else 0,
        "closed": len(closed),
        "win_rate_pct": round(float((pnl > 0).mean() * 100), 1) if len(pnl) else None,
        "total_pnl": round(float(pnl.sum()), 2) if len(pnl) else 0.0,
        "average_r": round(float(r.mean()), 2) if len(r) else None,
    }


def position_alerts(positions: pd.DataFrame, prices: dict) -> list:
    """Stop and target hits for open positions at the given prices."""
    alerts = []
    for _, row in positions.iterrows():
        price = prices.get(row["Ticker"])
        if not price:
            continue
        direction = 1 if row["Side"] == LONG else -1
        stop, target = float(row["Stop"]), row["Target"]
        kind = None
        if (price - stop) * direction <= 0:
            kind = "stop"
        elif pd.notna(target) and (price - float(target)) * direction >= 0:
            kind = "target"
        if kind:
            pnl, r = trade_result(row["Side"], float(row["Entry"]), initial_stop(row), float(row["Shares"]), price)
            alerts.append({"ID": row["ID"], "Ticker": row["Ticker"], "Side": row["Side"], "Kind": kind,
                           "Price": round(price, 2), "Stop": stop, "Target": target, "P&L": pnl, "R": r})
    return alerts
