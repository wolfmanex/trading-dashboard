"""Duplicate-run guard for the morning scan and the evening recap, sent to Telegram.

Used by .github/workflows/scheduled-scan.yml and market-session.yml. GitHub's own cron never fired in
this repo, so a daily outside trigger starts the scan, the scan starts the market session (intraday
checks every 15 minutes), and the session ends with the recap. GitHub's cron stays on the scan as a
backup, which is why `ran-today` exists: a second scan the same day would repeat the alerts. It also reports
market holidays as already run, so nothing starts on a closed day. `session-started` keeps a rerun scan
from starting a second market session. Days are New York dates.

`recap` re-checks every live signal against the day's bars, saves the log, and reports which setups
triggered, stopped out, hit target or timed out today, how the open ones stand, how your journal
positions closed, and whether today's scan and intraday checks actually ran.
"""
import argparse
import os
import sys
from datetime import date, datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from journal import mark_positions, open_positions, read_journal_file
from market_calendar import is_trading_day
from scheduled_scan import download_outcome_prices, send_telegram
from signal_log import OPEN, WAITING, read_log, update_outcomes
from smallcap_screener import _ticker_frame


EASTERN = ZoneInfo("America/New_York")
API = "https://api.github.com"
SCAN_WORKFLOW = "scheduled-scan.yml"
SESSION_WORKFLOW = "market-session.yml"
ACTIVE_STATUSES = {"queued", "in_progress", "waiting", "pending", "requested"}
EXPECTED_CHECKS = 25   # every 15 minutes, 9:45-15:45 New York
EXIT_STATUSES = {"Stop", "Target", "Time exit"}


def github(method: str, path: str, **kwargs) -> requests.Response:
    token, repo = os.getenv("GITHUB_TOKEN"), os.getenv("GITHUB_REPOSITORY")
    if not (token and repo):
        raise RuntimeError("GITHUB_TOKEN and GITHUB_REPOSITORY must be set")
    response = requests.request(
        method, f"{API}/repos/{repo}{path}", timeout=20,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, **kwargs,
    )
    response.raise_for_status()
    return response


def new_york_day(timestamp: str) -> date | None:
    """New York date of a GitHub UTC timestamp like '2026-10-05T23:30:00Z', or None if unreadable."""
    try:
        return datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(EASTERN).date()
    except ValueError:
        return None


def runs_today(workflow: str, day: date) -> list:
    """Runs of `workflow` created on `day` (New York date), any trigger, newest first."""
    # created_at is UTC and an evening New York run falls on the next UTC day, so ask from the day before.
    since = (day - timedelta(days=1)).isoformat()
    payload = github("GET", f"/actions/workflows/{workflow}/runs", params={"created": f">={since}", "per_page": 100}).json()
    return [run for run in payload.get("workflow_runs", []) if new_york_day(run.get("created_at", "")) == day]


def run_line(runs: list) -> str:
    """'ran at 12:47 UTC (scheduled), success' for the newest run, or a warning when there was none."""
    if not runs:
        return "did NOT run today"
    run = runs[0]
    started = str(run.get("run_started_at") or run.get("created_at") or "")[11:16]
    trigger = "GitHub's own schedule" if run.get("event") == "schedule" else "daily trigger or by hand"
    outcome = run.get("conclusion") or run.get("status") or "unknown"
    return f"ran at {started} UTC ({trigger}), {outcome}"


def already_ran(workflow: str, day: date, current_run_id: str = None) -> bool:
    """True when an earlier run of `workflow` on `day` finished successfully or is still going.

    Only runs started before this one count, so two runs queued together don't both stand down.
    """
    for run in runs_today(workflow, day):
        if current_run_id and int(run.get("id", 0)) >= int(current_run_id):
            continue
        if run.get("conclusion") == "success" or run.get("status") in ACTIVE_STATUSES:
            return True
    return False


def session_started(day: date, workflow: str = SESSION_WORKFLOW) -> bool:
    """True when a market session already ran successfully on `day` or is queued or running.

    The scan checks this before starting the session, so a rerun of the scan doesn't start a second
    session (and a second recap).
    """
    return any(run.get("conclusion") == "success" or run.get("status") in ACTIVE_STATUSES
               for run in runs_today(workflow, day))


def write_output(name: str, value: bool) -> None:
    output = os.getenv("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={'true' if value else 'false'}\n")


def format_r(value) -> str:
    value = pd.to_numeric(value, errors="coerce")
    return f"{float(value):+.2f}R" if pd.notna(value) else "R n/a"


def outcome_lines(log: pd.DataFrame, today: str) -> list:
    """What happened to the logged signals today, plus the open ones' marks."""
    if log.empty:
        return ["No signals logged yet."]
    lines = []
    entered = log[log["Entry Date"] == today]
    exited = log[(log["Exit Date"] == today) & log["Status"].isin(EXIT_STATUSES)]
    for _, row in entered.iterrows():
        lines.append(f"Triggered: <b>{escape(str(row['Ticker']))}</b> at {float(row['Entry']):.2f}")
    for _, row in exited.iterrows():
        lines.append(
            f"{escape(str(row['Status']))}: <b>{escape(str(row['Ticker']))}</b> at {float(row['Exit']):.2f}, "
            f"{format_r(row['R'])}"
        )
    open_rows = log[(log["Status"] == OPEN) & ~log.index.isin(entered.index)]
    if not open_rows.empty:
        marks = ", ".join(f"{escape(str(row['Ticker']))} {format_r(row['R'])}" for _, row in open_rows.iterrows())
        lines.append(f"Open: {marks}")
    waiting = int((log["Status"] == WAITING).sum())
    if not lines:
        lines.append("No setup triggered or exited today.")
    lines.append(f"{waiting} setups still waiting for their pivot.")
    return lines


def daily_closes(tickers: list) -> dict:
    if not tickers:
        return {}
    batch = yf.download(tickers=tickers, period="5d", interval="1d", group_by="ticker", auto_adjust=False, progress=False)
    prices = {}
    for ticker in tickers:
        close = _ticker_frame(batch, ticker).get("Close")
        if close is not None and close.dropna().size:
            prices[ticker] = float(close.dropna().iloc[-1])
    return prices


def journal_lines(journal_path: str, today: str) -> list:
    journal = read_journal_file(journal_path)
    if journal.empty:
        return []
    lines = []
    closed_today = journal[(journal["Status"] == "Closed") & (journal["Exit Date"] == today)]
    for _, row in closed_today.iterrows():
        lines.append(f"Closed <b>{escape(str(row['Ticker']))}</b>: {float(row['P&L']):+,.0f} USD ({format_r(row['R'])})")
    positions = open_positions(journal)
    if not positions.empty:
        marked = mark_positions(positions, daily_closes(sorted(set(positions["Ticker"]))))
        total = pd.to_numeric(marked["P&L"], errors="coerce").sum()
        for _, row in marked.iterrows():
            if row["Price"] is None or pd.isna(row["Price"]):
                lines.append(f"<b>{escape(row['Ticker'])}</b>: no price")
                continue
            side = "above" if row["Side"] == "Long" else "below"
            lines.append(
                f"<b>{escape(row['Ticker'])}</b> {row['Price']:.2f}: {row['P&L']:+,.0f} USD ({format_r(row['R'])}), "
                f"{row['To Stop %']:.1f}% {side} stop"
            )
        lines.append(f"Open P&amp;L {total:+,.0f} USD across {len(marked)} positions")
    return lines


def recap(log_path: str, journal_path: str, notify, checks: int = None) -> int:
    now = datetime.now(EASTERN)
    today = now.date().isoformat()
    log = read_log(log_path)
    if not log.empty:
        frames, benchmark_close = download_outcome_prices(log)
        log = update_outcomes(log, frames, benchmark_close, today=now.date())
        log.to_csv(log_path, index=False)

    message = [f"<b>Evening recap</b> {now:%a %d %b}", "", "<b>Scanner signals</b>"] + outcome_lines(log, today)
    positions = journal_lines(journal_path, today) if journal_path else []
    if positions:
        message += ["", "<b>Your positions</b>"] + positions
    try:
        scan_runs = runs_today(SCAN_WORKFLOW, now.date())
        lines = [f"Morning scan {run_line(scan_runs)}."]
        if checks is not None:
            lines.append(f"Intraday checks ran {checks} times (every 15 minutes from 9:45 is {EXPECTED_CHECKS}).")
            if checks < EXPECTED_CHECKS * 0.75:
                lines[-1] += " Some breakouts may have gone unalerted."
    except Exception as error:
        lines = [f"Could not check today's runs: {escape(str(error))}"]
    message += ["", "<b>Checks</b>"] + lines
    notify("\n".join(message))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Duplicate-run guard and evening recap.")
    parser.add_argument("mode", choices=["ran-today", "session-started", "recap"])
    parser.add_argument("--workflow", default=SCAN_WORKFLOW, help="Workflow file (ran-today).")
    parser.add_argument("--checks", type=int, help="Intraday checks the session ran today (recap).")
    parser.add_argument("--log", default="signal_log.csv", help="Signal log CSV (recap).")
    parser.add_argument("--journal", help="Trade journal CSV (recap).")
    args = parser.parse_args(argv)

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")

    def notify(message: str) -> None:
        print(message)
        if token and chat_id:
            send_telegram(token, chat_id, message)
            print("Sent to Telegram.")
        else:
            print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; message is in the job log only.")

    today = datetime.now(EASTERN).date()
    if args.mode == "ran-today":
        if not is_trading_day(today):
            # Reported as already run so neither the scan nor the market session starts.
            print(f"Market closed today ({today:%a %d %b}); nothing to do.")
            write_output("ran", True)
            return 0
        try:
            ran = already_ran(args.workflow, today, os.getenv("GITHUB_RUN_ID"))
            print(f"{args.workflow} {'already ran' if ran else 'has not run'} today.")
        except Exception as error:
            # Fail open: a GitHub API hiccup shouldn't cost the day's scan.
            print(f"Could not check today's runs ({type(error).__name__}: {error}); running anyway.")
            ran = False
        write_output("ran", ran)
        return 0
    if args.mode == "session-started":
        try:
            started = session_started(today)
            print(f"{SESSION_WORKFLOW} {'already started' if started else 'has not started'} today.")
        except Exception as error:
            # Fail open: a missed session costs more than a duplicate one.
            print(f"Could not check today's sessions ({type(error).__name__}: {error}); starting one.")
            started = False
        write_output("started", started)
        return 0
    if not is_trading_day(today):
        print(f"Market closed today ({today:%a %d %b}); no recap.")
        return 0
    return recap(args.log, args.journal, notify, args.checks)


if __name__ == "__main__":
    sys.exit(main())
