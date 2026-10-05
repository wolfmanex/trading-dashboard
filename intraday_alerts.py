"""Telegram alert when a scanner setup breaks its pivot on volume during market hours.

Used by .github/workflows/intraday-alerts.yml, which runs every 15 minutes on weekdays. It watches the
top setups the pre-market scan logged that are still waiting to trigger (see signal_log.py), and
alerts once per ticker per day when the price is at or above the pivot and the day's volume so far
runs at least MIN_RVOL times the 20-day average for this point in the session. Sent alerts are kept
in a CSV next to the signal log so later runs don't repeat them.
"""
import argparse
import os
import sys
from datetime import datetime, time
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from scheduled_scan import TOP_N, send_telegram
from signal_log import WAITING, read_log
from smallcap_screener import _ticker_frame


EASTERN = ZoneInfo("America/New_York")
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
FIRST_CHECK = time(9, 45)   # the opening minutes' volume makes RVOL meaningless
SESSION_MINUTES = 390
MIN_RVOL = 1.5
EXTENDED_PCT = 5.0          # more than this far above the pivot is flagged as extended (chasing)
AVERAGE_DAYS = 20
SENT_COLUMNS = ["Date", "Time", "Ticker", "Price", "Pivot", "Stop", "Target", "RVOL"]


def in_alert_window(now: datetime) -> bool:
    return now.weekday() < 5 and FIRST_CHECK <= now.time() < SESSION_CLOSE


def session_fraction(now: datetime) -> float:
    """Share of the regular session that has passed, between 0 and 1."""
    opened = datetime.combine(now.date(), SESSION_OPEN, tzinfo=now.tzinfo)
    minutes = (now - opened).total_seconds() / 60
    return min(max(minutes / SESSION_MINUTES, 0.0), 1.0)


def watched_signals(log: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """Logged top-N setups that haven't traded through their pivot yet."""
    if log.empty:
        return log
    ranked = pd.to_numeric(log["Rank"], errors="coerce") <= top_n
    return log[(log["Status"] == WAITING) & ranked]


def read_sent(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype={"Ticker": str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=SENT_COLUMNS)


def bars_for_day(bars: pd.DataFrame, day) -> pd.DataFrame:
    """Rows of `bars` that fall on `day` in New York time."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    index = bars.index if bars.index.tz is not None else bars.index.tz_localize("UTC")
    return bars[index.tz_convert(EASTERN).date == day]


def average_daily_volume(daily: pd.DataFrame, day, days: int = AVERAGE_DAYS) -> float | None:
    """Mean volume of the last `days` complete sessions before `day`."""
    if daily is None or daily.empty or "Volume" not in daily:
        return None
    dates = pd.DatetimeIndex(daily.index).date
    volume = pd.to_numeric(daily.loc[dates < day, "Volume"], errors="coerce").dropna().iloc[-days:]
    if volume.empty or volume.mean() <= 0:
        return None
    return float(volume.mean())


def check_breakout(signal: dict, today_bars: pd.DataFrame, average_volume: float | None, fraction: float) -> dict | None:
    """Return alert details when the price is at or above the pivot on enough volume, else None."""
    if today_bars is None or today_bars.empty or not average_volume or fraction <= 0:
        return None
    price = float(today_bars["Close"].dropna().iloc[-1])
    pivot = float(signal["Pivot"])
    if price < pivot:
        return None
    rvol = float(today_bars["Volume"].sum()) / (average_volume * fraction)
    if rvol < MIN_RVOL:
        return None
    return {
        "Ticker": str(signal["Ticker"]),
        "Price": round(price, 2),
        "Pivot": round(pivot, 2),
        "Stop": round(float(signal["Stop"]), 2),
        "Target": round(float(signal["Target"]), 2),
        "RVOL": round(rvol, 1),
        "Above Pivot %": round((price / pivot - 1) * 100, 1),
        "AI Grade": signal.get("AI Grade") if isinstance(signal.get("AI Grade"), str) else "",
    }


def format_alerts(alerts: list, now: datetime) -> str:
    lines = [f"<b>Breakout alert</b> {now.strftime('%H:%M')} New York"]
    for alert in alerts:
        extended = alert["Above Pivot %"] > EXTENDED_PCT
        lines.append(
            f"<b>{escape(alert['Ticker'])}</b> {alert['Price']:.2f} is {alert['Above Pivot %']:.1f}% above pivot "
            f"{alert['Pivot']:.2f} on {alert['RVOL']:.1f}x volume | stop {alert['Stop']:.2f} target {alert['Target']:.2f}"
            + (f" | AI {escape(alert['AI Grade'])}" if alert["AI Grade"] else "")
            + (" | extended, don't chase" if extended else "")
        )
    return "\n".join(lines)


def download(tickers: list, **kwargs) -> pd.DataFrame:
    # Unadjusted prices, matching the levels in the signal log.
    return yf.download(tickers=tickers, group_by="ticker", auto_adjust=False, progress=False, threads=True, **kwargs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Alert on scanner setups breaking their pivot on volume.")
    parser.add_argument("--log", required=True, help="Signal log CSV written by the scheduled scan.")
    parser.add_argument("--sent", required=True, help="CSV of alerts already sent; appended to.")
    parser.add_argument("--force", action="store_true", help="Run outside market hours (for testing).")
    args = parser.parse_args(argv)

    now = datetime.now(EASTERN)
    if not args.force and not in_alert_window(now):
        print(f"Outside the alert window ({now:%a %H:%M} New York); nothing to do.")
        return 0

    sent = read_sent(args.sent)
    already = set(sent.loc[sent["Date"] == now.date().isoformat(), "Ticker"])
    watch = watched_signals(read_log(args.log))
    watch = watch[~watch["Ticker"].isin(already)]
    if watch.empty:
        print("No waiting setups to watch.")
        return 0

    tickers = sorted(set(watch["Ticker"]))
    intraday = download(tickers, period="1d", interval="5m", prepost=False)
    daily = download(tickers, period="3mo", interval="1d")
    fraction = session_fraction(now)

    alerts = []
    for signal in watch.to_dict("records"):
        ticker = signal["Ticker"]
        alert = check_breakout(
            signal,
            bars_for_day(_ticker_frame(intraday, ticker), now.date()),
            average_daily_volume(_ticker_frame(daily, ticker), now.date()),
            fraction,
        )
        if alert:
            alerts.append(alert)
    print(f"Checked {len(tickers)} setups at {now:%H:%M}: {len(alerts)} breaking out on volume.")
    if not alerts:
        return 0

    message = format_alerts(alerts, now)
    print(message)
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; alerts are in the log only.")
        return 0
    send_telegram(token, chat_id, message)
    new_rows = pd.DataFrame(
        [{**alert, "Date": now.date().isoformat(), "Time": now.strftime("%H:%M")} for alert in alerts]
    )[SENT_COLUMNS]
    (new_rows if sent.empty else pd.concat([sent, new_rows], ignore_index=True)).to_csv(args.sent, index=False)
    print("Sent to Telegram.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
