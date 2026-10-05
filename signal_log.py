"""Forward track record for the breakout scanner.

The scheduled scan appends each day's setups to a CSV log and re-checks every unfinished signal
against fresh daily bars, using the same entry and exit rules as the scanner backtest: buy-stop at
the pivot within ENTRY_WINDOW trading days, then exit at the stop, the target, or after MAX_HOLD
days. The log lives on the repo's `signal-log` branch so daily updates don't touch main, and the
app reads it from there.
"""
import io
from datetime import date

import pandas as pd
import requests

from breakout_backtest import ENTRY_WINDOW, simulate_trade, summarize_trades


LOG_BRANCH = "signal-log"
LOG_FILE = "signal_log.csv"
LOG_URL = f"https://raw.githubusercontent.com/wolfmanex/trading-dashboard/{LOG_BRANCH}/{LOG_FILE}"

LOG_COLUMNS = [
    "Scan Date", "Ticker", "Rank", "Price", "Pivot", "Stop", "Target", "Reward/Risk", "Breakout Score",
    "IWM Trend", "Universe", "Status", "Entry Date", "Entry", "Exit Date", "Exit", "R", "Return %",
    "IWM %", "AI Grade", "AI Risk", "Updated",
]
UNGRADED = "Ungraded"

WAITING = "Waiting"            # not triggered yet, still inside the entry window
NOT_TRIGGERED = "Not triggered"
OPEN = "Open"
FINAL_STATUSES = {NOT_TRIGGERED, "Stop", "Target", "Time exit"}
LIVE_STATUSES = {WAITING, OPEN}


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def read_log(source) -> pd.DataFrame:
    """Read a log from a path or file-like object; a missing or empty file gives an empty log."""
    try:
        log = pd.read_csv(source, dtype={"Ticker": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return empty_log()
    for column in LOG_COLUMNS:
        if column not in log:
            log[column] = None
    return log[LOG_COLUMNS]


def fetch_published_log(url: str = LOG_URL, timeout: int = 15) -> pd.DataFrame:
    """Download the log the scheduled scan publishes; a 404 (nothing logged yet) gives an empty log."""
    response = requests.get(url, timeout=timeout)
    if response.status_code == 404:
        return empty_log()
    response.raise_for_status()
    return read_log(io.StringIO(response.text))


def append_signals(log: pd.DataFrame, results: pd.DataFrame, scan_date: date, iwm_trend: str = "") -> pd.DataFrame:
    """Add today's scan results to the log, one row per new setup.

    A ticker that already has a live signal (waiting or open) is skipped, so a setup that stays on
    the list for several days counts once, like the backtest's one-position-per-ticker rule. A rerun
    on the same day adds nothing new.
    """
    if results is None or results.empty:
        return log
    scan_day = pd.Timestamp(scan_date).strftime("%Y-%m-%d")
    live = set(log.loc[log["Status"].isin(LIVE_STATUSES), "Ticker"])
    logged_today = set(log.loc[log["Scan Date"] == scan_day, "Ticker"])
    universe = results.attrs.get("universe_source", "")

    rows = []
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        ticker = str(row["Ticker"])
        if ticker in live or ticker in logged_today:
            continue
        rows.append({
            "Scan Date": scan_day,
            "Ticker": ticker,
            "Rank": rank,
            "Price": row["Price"],
            "Pivot": row["Pivot"],
            "Stop": row["Stop"],
            "Target": row["Target"],
            "Reward/Risk": row["Reward/Risk"],
            "Breakout Score": row["Breakout Score"],
            "IWM Trend": iwm_trend,
            "Universe": universe,
            "AI Grade": row.get("AI Grade") or None,
            "AI Risk": row.get("AI Risk") or None,
            "Status": WAITING,
        })
    if not rows:
        return log
    new_rows = pd.DataFrame(rows, columns=LOG_COLUMNS)
    return new_rows if log.empty else pd.concat([log, new_rows], ignore_index=True)


def evaluate_signal(df: pd.DataFrame, scan_date, pivot: float, stop: float, target: float) -> dict:
    """Play one logged signal forward over daily bars.

    The scan runs before the open, so the setup is measured on bars before scan_date and the entry
    window starts with the scan date's own bar.
    """
    status = {"Status": WAITING}
    if df is None or df.empty:
        return status
    df = df[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    before = df.index < pd.Timestamp(scan_date)
    if not before.any():
        return status
    signal_index = int(before.sum()) - 1

    trade = simulate_trade(df, signal_index, pivot, stop, target)
    if trade is None:
        after = df.iloc[signal_index + 1:signal_index + 1 + ENTRY_WINDOW]
        broke_stop = ((after["Open"] <= stop) | ((after["Low"] <= stop) & (after["High"] < pivot))).any()
        if broke_stop or len(after) >= ENTRY_WINDOW:
            return {"Status": NOT_TRIGGERED}
        return status
    return {
        "Status": trade["Outcome"],
        "Entry Date": trade["Entry Date"].strftime("%Y-%m-%d"),
        "Entry": trade["Entry"],
        "Exit Date": trade["Exit Date"].strftime("%Y-%m-%d"),
        "Exit": trade["Exit"],
        "R": trade["R"],
        "Return %": round((trade["Exit"] / trade["Entry"] - 1) * 100, 2),
    }


def benchmark_return(benchmark_close: pd.Series, entry_date: str, exit_date: str) -> float | None:
    """IWM's % change from the close before entry to the exit day's close, for the same holding period."""
    if benchmark_close is None or benchmark_close.empty:
        return None
    closes = benchmark_close.copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    start = closes[closes.index < pd.Timestamp(entry_date)]
    end = closes[closes.index <= pd.Timestamp(exit_date)]
    if start.empty or end.empty or start.iloc[-1] <= 0:
        return None
    return round((float(end.iloc[-1]) / float(start.iloc[-1]) - 1) * 100, 2)


def update_outcomes(log: pd.DataFrame, price_frames: dict, benchmark_close: pd.Series = None, today: date = None) -> pd.DataFrame:
    """Re-evaluate every live signal with fresh daily bars; finished signals are left as they are."""
    if log.empty:
        return log
    log = log.copy().astype({column: object for column in LOG_COLUMNS})
    updated = pd.Timestamp(today or date.today()).strftime("%Y-%m-%d")
    for index, row in log[log["Status"].isin(LIVE_STATUSES)].iterrows():
        frame = price_frames.get(row["Ticker"])
        if frame is None or frame.empty:
            continue
        result = evaluate_signal(frame, row["Scan Date"], float(row["Pivot"]), float(row["Stop"]), float(row["Target"]))
        if "Entry Date" in result:
            result["IWM %"] = benchmark_return(benchmark_close, result["Entry Date"], result["Exit Date"])
        for column in ["Entry Date", "Entry", "Exit Date", "Exit", "R", "Return %", "IWM %"]:
            log.at[index, column] = result.get(column)
        log.at[index, "Status"] = result["Status"]
        log.at[index, "Updated"] = updated
    return log


def summarize_log(log: pd.DataFrame, max_rank: int = None) -> dict:
    """Track-record summary: signal counts by status plus win rate and R for closed trades."""
    if max_rank is not None and not log.empty:
        log = log[pd.to_numeric(log["Rank"], errors="coerce") <= max_rank]
    triggered = log[~log["Status"].isin({WAITING, NOT_TRIGGERED})].copy() if not log.empty else log
    if not triggered.empty:
        triggered = triggered.rename(columns={"Status": "Outcome"})
        triggered["R"] = pd.to_numeric(triggered["R"], errors="coerce")
        trades = summarize_trades(triggered)
    else:
        trades = summarize_trades(pd.DataFrame(columns=["Outcome", "R"]))

    closed = triggered[triggered["Outcome"] != OPEN] if not triggered.empty else triggered
    returns = pd.to_numeric(closed["Return %"], errors="coerce") if not closed.empty else pd.Series(dtype=float)
    iwm = pd.to_numeric(closed["IWM %"], errors="coerce") if not closed.empty else pd.Series(dtype=float)
    return {
        **trades,
        "signals": len(log),
        "waiting": int((log["Status"] == WAITING).sum()) if not log.empty else 0,
        "not_triggered": int((log["Status"] == NOT_TRIGGERED).sum()) if not log.empty else 0,
        "average_return_pct": round(float(returns.mean()), 2) if returns.notna().any() else None,
        "average_iwm_pct": round(float(iwm.mean()), 2) if iwm.notna().any() else None,
        "first_scan": log["Scan Date"].min() if not log.empty else None,
        "last_scan": log["Scan Date"].max() if not log.empty else None,
    }


def summarize_by_grade(log: pd.DataFrame, max_rank: int = None) -> pd.DataFrame:
    """Track record split by the AI grade the scheduled scan gave each setup before the open.

    One row per grade (A, B, C, then setups the AI didn't grade), so the grades can be compared with each
    other and with the scanner's own ranking.
    """
    columns = ["AI Grade", "Signals", "Closed Trades", "Win Rate %", "Average R", "Total R"]
    if log.empty:
        return pd.DataFrame(columns=columns)
    if max_rank is not None:
        log = log[pd.to_numeric(log["Rank"], errors="coerce") <= max_rank]
    grades = log["AI Grade"].fillna("").astype(str).str.strip().str.upper().replace("", UNGRADED)
    rows = []
    for grade in ["A", "B", "C", UNGRADED]:
        subset = log[grades == grade]
        if subset.empty:
            continue
        summary = summarize_log(subset)
        rows.append({
            "AI Grade": grade,
            "Signals": summary["signals"],
            "Closed Trades": summary["trades"],
            "Win Rate %": summary["win_rate_pct"],
            "Average R": summary["average_r"],
            "Total R": summary["total_r"],
        })
    return pd.DataFrame(rows, columns=columns)
