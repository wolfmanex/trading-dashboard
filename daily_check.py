"""Missed-run watchdog and the evening recap, both sent to Telegram.

Used by .github/workflows/daily-check.yml.

`watchdog` runs shortly after the morning scan should have started. GitHub sometimes drops scheduled
runs without a word, so when no scan run exists for today it starts one through the API and says so.

`recap` runs after the close. It re-checks every live signal against the day's bars, saves the log,
and reports which setups triggered, stopped out, hit target or timed out today, how the open ones
stand, how your journal positions closed, and whether today's scan and intraday checks actually ran.
"""
import argparse
import os
import sys
from datetime import date, datetime, timezone
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from journal import mark_positions, open_positions, read_journal_file
from scheduled_scan import download_outcome_prices, send_telegram
from signal_log import OPEN, WAITING, read_log, update_outcomes
from smallcap_screener import _ticker_frame


EASTERN = ZoneInfo("America/New_York")
API = "https://api.github.com"
SCAN_WORKFLOW = "scheduled-scan.yml"
ALERTS_WORKFLOW = "intraday-alerts.yml"
EXPECTED_ALERT_RUNS = 32   # 4 per hour, 13:07-20:52 UTC
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


def runs_today(workflow: str, day: date) -> list:
    """Runs of `workflow` created on `day` (UTC), any trigger, newest first."""
    payload = github("GET", f"/actions/workflows/{workflow}/runs", params={"created": f">={day.isoformat()}", "per_page": 100}).json()
    return [run for run in payload.get("workflow_runs", []) if str(run.get("created_at", "")).startswith(day.isoformat())]


def run_line(runs: list) -> str:
    """'ran at 12:47 UTC (scheduled), success' for the newest run, or a warning when there was none."""
    if not runs:
        return "did NOT run today"
    run = runs[0]
    started = str(run.get("run_started_at") or run.get("created_at") or "")[11:16]
    trigger = "scheduled" if run.get("event") == "schedule" else "started by hand or by the watchdog"
    outcome = run.get("conclusion") or run.get("status") or "unknown"
    return f"ran at {started} UTC ({trigger}), {outcome}"


def watchdog(notify) -> int:
    today = datetime.now(timezone.utc).date()
    if today.weekday() >= 5:
        print("Weekend; nothing to check.")
        return 0
    runs = runs_today(SCAN_WORKFLOW, today)
    if runs:
        print(f"Morning scan {run_line(runs)}.")
        return 0
    github("POST", f"/actions/workflows/{SCAN_WORKFLOW}/dispatches", json={"ref": os.getenv("GITHUB_REF_NAME") or "main"})
    notify("<b>Watchdog</b>\nGitHub did not start this morning's scan on schedule, so I started it now. "
           "The scan results follow in a few minutes.")
    return 0


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


def recap(log_path: str, journal_path: str, notify) -> int:
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
        utc_day = datetime.now(timezone.utc).date()
        scan_runs = runs_today(SCAN_WORKFLOW, utc_day)
        alert_runs = [run for run in runs_today(ALERTS_WORKFLOW, utc_day) if run.get("event") == "schedule"]
        checks = [f"Morning scan {run_line(scan_runs)}.",
                  f"Intraday checks ran {len(alert_runs)} of {EXPECTED_ALERT_RUNS} times."]
        if len(alert_runs) < EXPECTED_ALERT_RUNS * 0.75:
            checks[-1] += " GitHub skipped many of them, so some breakouts may have gone unalerted."
    except Exception as error:
        checks = [f"Could not check today's runs: {escape(str(error))}"]
    message += ["", "<b>Checks</b>"] + checks
    notify("\n".join(message))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Missed-run watchdog and evening recap.")
    parser.add_argument("mode", choices=["watchdog", "recap"])
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

    if args.mode == "watchdog":
        return watchdog(notify)
    return recap(args.log, args.journal, notify)


if __name__ == "__main__":
    sys.exit(main())
