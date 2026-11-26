"""Telegram alert when a large-cap day-trading breakout triggers, plus its outcome once it closes.

Run every 15 minutes in the session by session_loop.sh (.github/workflows/market-session.yml). Each run
scans the large-cap universe (largecap_scanner.py) and alerts the top setups that have broken their
trigger, between FIRST_ALERT and LAST_ALERT New York time: once per ticker per day, and not again while
an earlier alert on the ticker is still open. Setups are ranked by their day score adjusted by the
track record (daytrade_feedback.py). Sent alerts are kept in a CSV on the signal-log branch.

A trade closes only at its stop or its target, so it can stay open for days. Every evening the recap
(daily_check.py) plays each still-open alert forward from its alert time through the regular-session
bars since, writes the outcome back (or the open trade's mark), and the CSV doubles as the day-trade
track record. "Close R" keeps what the trade would have made if flattened at the alert day's close.
"""
import argparse
import io
import os
import sys
from datetime import datetime, time
from html import escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from daytrade_engine import (
    BREAKING_OUT, CLOSED_OUTCOMES, MAX_HOLD_DAYS, OPEN_TRADE, TIME_EXIT, evaluate_daytrade, split_sessions,
)
from daytrade_feedback import apply_feedback, build_feedback, feedback_summary
from market_calendar import is_trading_day
from signal_log import GITHUB_REPO, LOG_BRANCH, fetch_branch_file
from smallcap_screener import _ticker_frame


EASTERN = ZoneInfo("America/New_York")
FIRST_ALERT = time(10, 0)   # once the 30-minute opening range is in
LAST_ALERT = time(15, 30)   # too little session left to work a trade after this
TOP_N = 5
MIN_SCORE = 60.0
MAX_ALERTS_PER_DAY = 6       # 2-25 Nov 2026 sent up to 42 a day; the later ones didn't add anything
# Market filter: no new longs when SPY is down this much on the day, and a higher bar while it trades
# under its VWAP (20 Nov 2026, a weak market day, cost three quarters of the month's losses).
SPY_BLOCK_CHANGE = -0.5
SPY_UNDER_VWAP_EXTRA_SCORE = 10.0
MAX_OPEN_LISTED = 12   # carried-over open trades named in the recap; the rest are counted
DAYTRADE_COLUMNS = [
    "Date", "Time", "Ticker", "Price", "Trigger", "Stop", "Target", "RVOL", "Day Score",
    "Outcome", "Exit", "R",
    # Added with trades that carry over: when it closed, the result had it been flattened at the alert
    # day's close, and what was known at the alert (for the feedback loop).
    "Exit Date", "Days Held", "Close R", "Trigger Level", "Risk %", "Change %", "RS vs SPY", "SPY %",
    "History Adj", "Adj Score", "ATR %", "AI Grade", "AI Reason",
]
LEGACY_CLOSE = "Close"   # the old rule's flat-at-the-bell outcome; such trades are reopened and replayed


DAYTRADE_FILE = "daytrade_alerts.csv"
DAYTRADE_LOG_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{LOG_BRANCH}/{DAYTRADE_FILE}"


def in_alert_window(now: datetime) -> bool:
    return is_trading_day(now.date()) and FIRST_ALERT <= now.time() < LAST_ALERT


def read_alerts(path) -> pd.DataFrame:
    try:
        alerts = pd.read_csv(
            path, dtype={"Ticker": str, "Date": str, "Time": str, "Outcome": str, "Exit Date": str, "Trigger Level": str},
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=DAYTRADE_COLUMNS)
    return alerts.reindex(columns=DAYTRADE_COLUMNS)


def is_closed(alerts: pd.DataFrame) -> pd.Series:
    return alerts["Outcome"].isin(CLOSED_OUTCOMES)


def open_tickers(alerts: pd.DataFrame) -> set:
    """Tickers with an alert that hasn't reached its stop or target yet."""
    if alerts.empty:
        return set()
    return set(alerts.loc[~is_closed(alerts) & (alerts["Outcome"] != "Invalid"), "Ticker"].astype(str))


def fetch_published_alerts(url: str = DAYTRADE_LOG_URL, timeout: int = 15, token: str = None) -> pd.DataFrame:
    """The alerts the market session publishes on the signal-log branch; a 404 (none sent yet) is empty."""
    text = fetch_branch_file(url, DAYTRADE_FILE, token=token, timeout=timeout)
    return pd.DataFrame(columns=DAYTRADE_COLUMNS) if text is None else read_alerts(io.StringIO(text))


def pick_alerts(results: pd.DataFrame, already: set, top_n: int = TOP_N, min_score: float = MIN_SCORE) -> list:
    """Setups through their trigger with a strong enough score that aren't already alerted or open.

    Ranked by the track-record-adjusted score ("Adj Score") when the results have one.
    """
    if results is None or results.empty:
        return []
    score = "Adj Score" if "Adj Score" in results else "Day Score"
    fresh = results[
        (results["Status"] == BREAKING_OUT)
        & (results[score] >= min_score)
        & ~results["Ticker"].isin(already)
    ]
    return fresh.sort_values([score, "RVOL"], ascending=False).head(top_n).to_dict("records")


def format_alerts(picks: list, now: datetime, benchmark_change=None) -> str:
    market = f" | SPY {benchmark_change:+.2f}%" if isinstance(benchmark_change, (int, float)) else ""
    lines = [f"<b>Large-cap day-trade breakout</b> {now.strftime('%H:%M')} New York{market}"]
    for pick in picks:
        score = f"score {pick['Day Score']:.0f}"
        adjustment = pick.get("History Adj")
        if isinstance(adjustment, (int, float)) and abs(adjustment) >= 0.5:
            score += f" ({adjustment:+.0f} from track record: {escape(str(pick.get('History Note') or ''))})"
        if pick.get("AI Grade"):
            score += f" | AI {escape(str(pick['AI Grade']))}: {escape(str(pick.get('AI Reason') or ''))}"
        lines.append(
            f"<b>{escape(str(pick['Ticker']))}</b> {pick['Price']:.2f} over {escape(str(pick['Trigger Level']).lower())} "
            f"{pick['Trigger']:.2f} on {pick['RVOL']:.1f}x volume | stop {pick['Stop']:.2f} "
            f"target {pick['Target']:.2f} | {score}"
        )
    lines.append(f"Held until the stop or the target, overnight if needed; out at the close after {MAX_HOLD_DAYS} days otherwise.")
    return "\n".join(lines)


def market_gate(spy_change, spy_vwap_gap) -> tuple:
    """(minimum score, reason) for new alerts given SPY; a minimum of None means no new alerts."""
    if isinstance(spy_change, (int, float)) and spy_change <= SPY_BLOCK_CHANGE:
        return None, f"SPY {spy_change:+.2f}% on the day, no new long alerts"
    if isinstance(spy_vwap_gap, (int, float)) and spy_vwap_gap < 0:
        return MIN_SCORE + SPY_UNDER_VWAP_EXTRA_SCORE, f"SPY under its VWAP, alerts need {MIN_SCORE + SPY_UNDER_VWAP_EXTRA_SCORE:.0f}+"
    return MIN_SCORE, ""


def grade_picks(picks: list, feedback: dict, market_note: str) -> list:
    """Add the AI grade to the picks and drop the ones it grades C; ungraded picks pass when AI fails."""
    from llm_engine import is_ai_configured, review_daytrade_candidates
    if not picks or not is_ai_configured():
        return picks
    outcome = review_daytrade_candidates(picks, feedback_summary(feedback), market_note)
    if outcome["error"]:
        print(f"AI day-trade grading failed, alerting ungraded: {outcome['error']}")
        return picks
    graded = []
    for pick in picks:
        review = outcome["reviews"].get(str(pick["Ticker"]).upper())
        if review:
            pick = {**pick, "AI Grade": review["grade"], "AI Reason": review["reason"]}
        if pick.get("AI Grade") == "C":
            print(f"AI graded {pick['Ticker']} C, not alerted: {pick.get('AI Reason')}")
            continue
        graded.append(pick)
    return graded


def check_daytrades(sent_path: str, now: datetime, notify, scan=None, grade=grade_picks) -> None:
    if scan is None:
        from largecap_scanner import scan_daytrade_setups as scan
    sent = read_alerts(sent_path)
    today = now.date().isoformat()
    sent_today = sent.loc[sent["Date"] == today, "Ticker"]
    remaining = MAX_ALERTS_PER_DAY - len(sent_today)
    if remaining <= 0:
        print(f"Already sent {len(sent_today)} day-trade alerts today (cap {MAX_ALERTS_PER_DAY}); nothing to do.")
        return
    already = set(sent_today) | open_tickers(sent)
    results = scan()
    spy_change = results.attrs.get("benchmark_change")
    min_score, market_reason = market_gate(spy_change, results.attrs.get("benchmark_vwap_gap"))
    if min_score is None:
        print(f"Market filter at {now:%H:%M}: {market_reason}.")
        return
    feedback = build_feedback(sent)
    attrs = dict(results.attrs)
    results = apply_feedback(results, feedback, now_minute=now.hour * 60 + now.minute, spy_change=spy_change)
    results.attrs.update(attrs)
    picks = pick_alerts(results, already, top_n=min(TOP_N, remaining), min_score=min_score)
    market_note = f"SPY {spy_change:+.2f}% on the day" if isinstance(spy_change, (int, float)) else ""
    if market_reason:
        market_note += f"; {market_reason}"
    picks = grade(picks, feedback, market_note)
    print(f"Scanned {results.attrs.get('analyzed', 'N/A')} large caps at {now:%H:%M}: "
          f"{len(results)} setups, {len(picks)} new breakouts to alert "
          f"(track record: {feedback['trades']} closed trades{', adjusting scores' if feedback['active'] else ''}"
          f"{'; ' + market_reason if market_reason else ''}).")
    if picks and notify(format_alerts(picks, now, spy_change)):
        rows = pd.DataFrame([
            {**pick, "Date": today, "Time": now.strftime("%H:%M"), "Outcome": OPEN_TRADE, "SPY %": spy_change}
            for pick in picks
        ])
        rows = rows.reindex(columns=DAYTRADE_COLUMNS)
        (rows if sent.empty else pd.concat([sent, rows], ignore_index=True)).to_csv(sent_path, index=False)


def _regular_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """A ticker's regular-session bars across all its days, in New York time, oldest first."""
    sessions = split_sessions(bars)
    frames = [parts["regular"] for parts in sessions.values() if not parts["regular"].empty]
    return pd.concat(frames) if frames else pd.DataFrame()


def _trading_days_between(start: str, end: str) -> int:
    from market_calendar import is_trading_day
    days = pd.date_range(start, end, freq="D")
    return max(sum(1 for day in days if is_trading_day(day.date())) - 1, 0)


def score_alerts(alerts: pd.DataFrame, today: str, bars_by_ticker) -> pd.DataFrame:
    """Replay every alert that hasn't closed from its alert time through the bars since, up to `today`.

    `bars_by_ticker` is a yfinance batch download or a {ticker: bars} dict; bars must reach back to the
    alert's day (5-minute or hourly bars both work). Fills Outcome ("Open" until the stop or target),
    Exit (the mark while open), R, Exit Date, Days Held and, once the alert day's bars are in, Close R.
    """
    alerts = alerts.copy()
    for column in ("Outcome", "Exit Date"):
        alerts[column] = alerts[column].astype(object)
    # Old-rule results: what the trade made flat at the bell is kept as Close R, and the trade reopened.
    legacy = alerts["Outcome"] == LEGACY_CLOSE
    alerts.loc[legacy & alerts["Close R"].isna(), "Close R"] = alerts.loc[legacy & alerts["Close R"].isna(), "R"]
    alerts.loc[legacy, "Outcome"] = OPEN_TRADE
    # Trades the old rule already closed at a stop or target closed the same way under the new one.
    settled = is_closed(alerts) & alerts["Exit Date"].isna()
    alerts.loc[settled, "Exit Date"] = alerts.loc[settled, "Date"]
    alerts.loc[settled & alerts["Days Held"].isna(), "Days Held"] = 0
    alerts.loc[settled & alerts["Close R"].isna(), "Close R"] = alerts.loc[settled & alerts["Close R"].isna(), "R"]
    pending = alerts[~is_closed(alerts) & (alerts["Outcome"] != "Invalid") & (alerts["Date"] <= today)]
    cache = {}
    for index, row in pending.iterrows():
        ticker = str(row["Ticker"])
        if ticker not in cache:
            frame = bars_by_ticker.get(ticker) if isinstance(bars_by_ticker, dict) else _ticker_frame(bars_by_ticker, ticker)
            cache[ticker] = _regular_bars(frame if frame is not None else pd.DataFrame())
        regular = cache[ticker]
        if regular.empty:
            continue
        hour, minute = (int(part) for part in str(row["Time"]).split(":")[:2])
        dates = np.array([ts.date().isoformat() for ts in regular.index])
        minutes = np.array([ts.hour * 60 + ts.minute for ts in regular.index])
        alert_day = str(row["Date"])
        if alert_day not in set(dates):
            continue  # the download doesn't reach back to the alert; leave it for a fuller one
        # Bars that start at or after the alert's quarter hour (the alert saw the bar before it close).
        after = ((dates == alert_day) & (minutes >= hour * 60 + minute)) | ((dates > alert_day) & (dates <= today))
        result = evaluate_daytrade(float(row["Price"]), float(row["Stop"]), float(row["Target"]), regular[after])
        if result["Outcome"] == "Invalid":
            alerts.loc[index, "Outcome"] = "Invalid"
            continue
        if pd.isna(row["Close R"]):
            same_day = regular[after & (dates == alert_day)]
            alerts.loc[index, "Close R"] = evaluate_daytrade(float(row["Price"]), float(row["Stop"]), float(row["Target"]), same_day)["R"]
        last_day = result["Exit Date"] or alert_day
        days_held = _trading_days_between(alert_day, last_day)
        if result["Outcome"] == OPEN_TRADE and days_held >= MAX_HOLD_DAYS and last_day == today:
            result["Outcome"] = TIME_EXIT   # the holding period ended at today's close
        alerts.loc[index, "Outcome"] = result["Outcome"]
        alerts.loc[index, "Exit"] = result["Exit"]
        alerts.loc[index, "R"] = result["R"]
        alerts.loc[index, "Exit Date"] = last_day if result["Outcome"] in CLOSED_OUTCOMES else None
        alerts.loc[index, "Days Held"] = days_held
    return alerts


def summarize_daytrades(alerts: pd.DataFrame) -> dict:
    """Closed trades (stop or target) for the stats; open trades counted separately at their mark."""
    if alerts.empty:
        return {"trades": 0, "open": 0}
    r = pd.to_numeric(alerts["R"], errors="coerce")
    closed = is_closed(alerts) & r.notna()
    # A legacy "Close" row is open under the new rule until the next recap replays it.
    open_rows = alerts["Outcome"].isin([OPEN_TRADE, LEGACY_CLOSE])
    still_open = open_rows & r.notna()
    summary = {
        "trades": int(closed.sum()),
        "open": int(open_rows.sum()),
        "open_r": round(float(r[still_open].sum()), 2),
    }
    if not summary["trades"]:
        return summary
    days = pd.to_numeric(alerts.loc[closed, "Days Held"], errors="coerce")
    close_r = pd.to_numeric(alerts.loc[closed, "Close R"], errors="coerce")
    summary.update({
        "win_rate_pct": round(float((r[closed] > 0).mean() * 100), 1),
        "average_r": round(float(r[closed].mean()), 2),
        "total_r": round(float(r[closed].sum()), 2),
        "target_pct": round(float((alerts.loc[closed, "Outcome"] == "Target").mean() * 100), 1),
        "stop_pct": round(float((alerts.loc[closed, "Outcome"] == "Stop").mean() * 100), 1),
        "time_pct": round(float((alerts.loc[closed, "Outcome"] == TIME_EXIT).mean() * 100), 1),
        "average_days": round(float(days.mean()), 1) if days.notna().any() else None,
        "close_rule_total_r": round(float(close_r.sum()), 2) if close_r.notna().any() else None,
    })
    return summary


def _r_text(value) -> str:
    r = pd.to_numeric(value, errors="coerce")
    return f"{float(r):+.2f}R" if pd.notna(r) else "N/A"


def recap_lines(alerts: pd.DataFrame, day: str) -> list:
    """Evening-recap lines: today's alerts, trades that closed today, and what is still open."""
    if alerts.empty:
        return ["No large-cap day-trade alerts today."]
    lines = []
    new = alerts[alerts["Date"] == day]
    if new.empty:
        lines.append("No new large-cap day-trade alerts today.")
    for _, row in new.iterrows():
        if row["Outcome"] in CLOSED_OUTCOMES:
            result = f"{row['Outcome']} {float(row['Exit']):.2f}, {_r_text(row['R'])}"
        elif pd.notna(pd.to_numeric(row["R"], errors="coerce")):
            result = f"open, {float(row['Exit']):.2f} at the close ({_r_text(row['R'])})"
        else:
            result = "no bars to score"
        lines.append(f"<b>{escape(str(row['Ticker']))}</b> {row['Time']} at {float(row['Price']):.2f}: {escape(result)}")

    carried_closed = alerts[(alerts["Date"] < day) & (alerts["Exit Date"] == day) & is_closed(alerts)]
    if not carried_closed.empty:
        lines.append("Closed today from earlier days:")
        for _, row in carried_closed.iterrows():
            lines.append(
                f"<b>{escape(str(row['Ticker']))}</b> from {row['Date']}: {row['Outcome']} {float(row['Exit']):.2f}, "
                f"{_r_text(row['R'])}"
            )
    carried_open = alerts[(alerts["Date"] < day) & (alerts["Outcome"] == OPEN_TRADE)]
    if not carried_open.empty:
        shown = carried_open.sort_values(["Date", "Time"]).head(MAX_OPEN_LISTED)
        marks = ", ".join(f"{escape(str(row['Ticker']))} {_r_text(row['R'])}" for _, row in shown.iterrows())
        more = f" and {len(carried_open) - len(shown)} more" if len(carried_open) > len(shown) else ""
        lines.append(f"Still open from earlier days ({len(carried_open)}, oldest first): {marks}{more}")

    closed_today = alerts[(alerts["Exit Date"] == day) & is_closed(alerts)]
    if not closed_today.empty:
        lines.append(f"Closed today {pd.to_numeric(closed_today['R'], errors='coerce').sum():+.2f}R over {len(closed_today)} trades.")
    overall = summarize_daytrades(alerts)
    if overall["trades"]:
        lines.append(
            f"All closed day trades: {overall['trades']}, {overall['win_rate_pct']}% winners, "
            f"{overall['average_r']:+.2f}R average; {overall['open']} still open ({overall['open_r']:+.2f}R marked)."
        )
    elif overall["open"]:
        lines.append(f"{overall['open']} day trades open ({overall['open_r']:+.2f}R marked), none closed yet.")
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
