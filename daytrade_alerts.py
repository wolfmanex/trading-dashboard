"""Telegram alert when a large-cap day-trading breakout triggers, plus the close-of-day outcome.

Run every 15 minutes in the session by session_loop.sh (.github/workflows/market-session.yml). Each run
scans the large-cap universe (largecap_scanner.py) and alerts the top setups that have broken their
trigger, once per ticker per day, between FIRST_ALERT and LAST_ALERT New York time. Sent alerts are
kept in a CSV on the signal-log branch; the evening recap (daily_check.py) plays each of the day's
alerts forward through the session's 5-minute bars and writes its outcome back, so the CSV doubles as
the day-trade track record.
"""
import argparse
import io
import os
import sys
from datetime import datetime, time
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd

from daytrade_engine import BREAKING_OUT, evaluate_daytrade, split_sessions
from market_calendar import is_trading_day
from signal_log import GITHUB_REPO, LOG_BRANCH, fetch_branch_file
from smallcap_screener import _ticker_frame


EASTERN = ZoneInfo("America/New_York")
FIRST_ALERT = time(9, 45)   # once the 15-minute opening range is in
LAST_ALERT = time(15, 30)   # too little session left to work a trade after this
TOP_N = 5
MIN_SCORE = 60.0
DAYTRADE_COLUMNS = [
    "Date", "Time", "Ticker", "Price", "Trigger", "Stop", "Target", "RVOL", "Day Score",
    "Outcome", "Exit", "R",
]


DAYTRADE_FILE = "daytrade_alerts.csv"
DAYTRADE_LOG_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{LOG_BRANCH}/{DAYTRADE_FILE}"


def in_alert_window(now: datetime) -> bool:
    return is_trading_day(now.date()) and FIRST_ALERT <= now.time() < LAST_ALERT


def read_alerts(path) -> pd.DataFrame:
    try:
        alerts = pd.read_csv(path, dtype={"Ticker": str, "Date": str, "Time": str, "Outcome": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=DAYTRADE_COLUMNS)
    return alerts.reindex(columns=DAYTRADE_COLUMNS)


def fetch_published_alerts(url: str = DAYTRADE_LOG_URL, timeout: int = 15, token: str = None) -> pd.DataFrame:
    """The alerts the market session publishes on the signal-log branch; a 404 (none sent yet) is empty."""
    text = fetch_branch_file(url, DAYTRADE_FILE, token=token, timeout=timeout)
    return pd.DataFrame(columns=DAYTRADE_COLUMNS) if text is None else read_alerts(io.StringIO(text))


def pick_alerts(results: pd.DataFrame, already: set, top_n: int = TOP_N, min_score: float = MIN_SCORE) -> list:
    """Setups through their trigger with a strong enough score that haven't been alerted today."""
    if results is None or results.empty:
        return []
    fresh = results[
        (results["Status"] == BREAKING_OUT)
        & (results["Day Score"] >= min_score)
        & ~results["Ticker"].isin(already)
    ]
    return fresh.head(top_n).to_dict("records")


def format_alerts(picks: list, now: datetime, benchmark_change=None) -> str:
    market = f" | SPY {benchmark_change:+.2f}%" if isinstance(benchmark_change, (int, float)) else ""
    lines = [f"<b>Large-cap day-trade breakout</b> {now.strftime('%H:%M')} New York{market}"]
    for pick in picks:
        lines.append(
            f"<b>{escape(str(pick['Ticker']))}</b> {pick['Price']:.2f} over {escape(str(pick['Trigger Level']).lower())} "
            f"{pick['Trigger']:.2f} on {pick['RVOL']:.1f}x volume | stop {pick['Stop']:.2f} "
            f"target {pick['Target']:.2f} | score {pick['Day Score']:.0f}"
        )
    lines.append("Intraday trades: flat by the close.")
    return "\n".join(lines)


def check_daytrades(sent_path: str, now: datetime, notify, scan=None) -> None:
    if scan is None:
        from largecap_scanner import scan_daytrade_setups as scan
    sent = read_alerts(sent_path)
    already = set(sent.loc[sent["Date"] == now.date().isoformat(), "Ticker"])
    results = scan()
    picks = pick_alerts(results, already)
    print(f"Scanned {results.attrs.get('analyzed', 'N/A')} large caps at {now:%H:%M}: "
          f"{len(results)} setups, {len(picks)} new breakouts to alert.")
    if picks and notify(format_alerts(picks, now, results.attrs.get("benchmark_change"))):
        rows = pd.DataFrame([{**pick, "Date": now.date().isoformat(), "Time": now.strftime("%H:%M")} for pick in picks])
        rows = rows.reindex(columns=DAYTRADE_COLUMNS)
        (rows if sent.empty else pd.concat([sent, rows], ignore_index=True)).to_csv(sent_path, index=False)


def score_alerts(alerts: pd.DataFrame, day: str, intraday: pd.DataFrame) -> pd.DataFrame:
    """Fill in Outcome, Exit and R for the alerts sent on `day` from that day's 5-minute bars."""
    alerts = alerts.copy()
    alerts["Outcome"] = alerts["Outcome"].astype(object)
    for index, row in alerts[alerts["Date"] == day].iterrows():
        sessions = split_sessions(_ticker_frame(intraday, str(row["Ticker"])))
        regular = next((parts["regular"] for session_day, parts in sessions.items() if session_day.isoformat() == day), None)
        if regular is None or regular.empty:
            continue
        hour, minute = (int(part) for part in str(row["Time"]).split(":")[:2])
        # Bars that start at or after the alert's quarter hour (the alert saw the bar before it close).
        after = regular[[ts.hour * 60 + ts.minute >= hour * 60 + minute for ts in regular.index]]
        result = evaluate_daytrade(float(row["Price"]), float(row["Stop"]), float(row["Target"]), after)
        for column, value in result.items():
            alerts.loc[index, column] = value
    return alerts


def summarize_daytrades(alerts: pd.DataFrame) -> dict:
    scored = alerts[pd.to_numeric(alerts["R"], errors="coerce").notna()] if not alerts.empty else alerts
    if scored.empty:
        return {"trades": 0}
    r = pd.to_numeric(scored["R"], errors="coerce")
    return {
        "trades": int(len(scored)),
        "win_rate_pct": round(float((r > 0).mean() * 100), 1),
        "average_r": round(float(r.mean()), 2),
        "total_r": round(float(r.sum()), 2),
        "target_pct": round(float((scored["Outcome"] == "Target").mean() * 100), 1),
        "stop_pct": round(float((scored["Outcome"] == "Stop").mean() * 100), 1),
    }


def recap_lines(alerts: pd.DataFrame, day: str) -> list:
    """Evening-recap lines for the day's alerted day trades."""
    today = alerts[alerts["Date"] == day] if not alerts.empty else alerts
    if today.empty:
        return ["No large-cap day-trade alerts today."]
    lines = []
    for _, row in today.iterrows():
        r = pd.to_numeric(row["R"], errors="coerce")
        result = f"{row['Outcome']} {float(row['Exit']):.2f}, {float(r):+.2f}R" if pd.notna(r) else "no bars to score"
        lines.append(f"<b>{escape(str(row['Ticker']))}</b> {row['Time']} at {float(row['Price']):.2f}: {escape(result)}")
    summary = summarize_daytrades(today)
    if summary["trades"]:
        lines.append(f"Today {summary['total_r']:+.2f}R over {summary['trades']} trades.")
    overall = summarize_daytrades(alerts)
    if overall["trades"]:
        lines.append(
            f"All day-trade alerts: {overall['trades']} trades, {overall['win_rate_pct']}% winners, "
            f"{overall['average_r']:+.2f}R average."
        )
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Alert on large-cap day-trading breakouts.")
    parser.add_argument("--sent", required=True, help="CSV of day-trade alerts already sent; appended to.")
    parser.add_argument("--force", action="store_true", help="Run outside the alert window (for testing).")
    args = parser.parse_args(argv)

    now = datetime.now(EASTERN)
    if not args.force and not in_alert_window(now):
        print(f"Outside the day-trade alert window ({now:%a %H:%M} New York); nothing to do.")
        return 0

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")

    def notify(message: str) -> bool:
        print(message)
        if not (token and chat_id):
            print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; alerts are in the log only.")
            return False
        from scheduled_scan import send_telegram
        send_telegram(token, chat_id, message)
        print("Sent to Telegram.")
        return True

    # Exit 0 even on failure so the session loop keeps going.
    try:
        check_daytrades(args.sent, now, notify)
    except Exception as error:
        print(f"Day-trade check FAILED: {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
