import re
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st

from secrets_config import get_configured_secret
from smallcap_screener import NASDAQ_HEADERS


EARNINGS_WINDOW_DAYS = 14
NASDAQ_MAX_CONSECUTIVE_FAILURES = 3   # stop walking the window once Nasdaq is clearly unreachable
NASDAQ_EARNINGS_URL = "https://api.nasdaq.com/api/calendar/earnings"
FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"


def parse_nasdaq_earnings(payload) -> list:
    """Symbols on one day of Nasdaq's earnings calendar; tolerates the null rows it returns for empty days."""
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    return [
        str(row["symbol"]).strip().upper()
        for row in rows
        if isinstance(row, dict) and row.get("symbol")
    ]


def parse_finnhub_earnings(payload) -> dict:
    """{symbol: earliest date} from Finnhub's earnings calendar response."""
    entries = payload.get("earningsCalendar") if isinstance(payload, dict) else None
    dates = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("symbol") or not entry.get("date"):
            continue
        symbol = str(entry["symbol"]).strip().upper()
        dates[symbol] = min(dates.get(symbol, entry["date"]), entry["date"])
    return dates


def _finnhub_calendar(start: date, end: date) -> dict:
    api_key = get_configured_secret("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY is not configured")
    # The key goes in a header, never the URL, so it can't leak into error messages shown in the UI,
    # Telegram or the GitHub step summary.
    response = requests.get(
        FINNHUB_EARNINGS_URL,
        params={"from": start.isoformat(), "to": end.isoformat()},
        headers={"X-Finnhub-Token": api_key},
        timeout=20,
    )
    response.raise_for_status()
    dates = parse_finnhub_earnings(response.json())
    if not dates:
        raise RuntimeError("Finnhub returned no earnings dates")
    return dates


def _safe_error(error: Exception) -> str:
    """Error text without any URL (a request URL can carry query parameters such as API tokens)."""
    return re.sub(r"https?://\S+", "<url>", str(error))


def _nasdaq_calendar(start: date, end: date) -> dict:
    """Read Nasdaq's calendar one weekday at a time.

    Raises when any day fails, so a partial calendar (missing reporters) is never cached as complete,
    and stops after NASDAQ_MAX_CONSECUTIVE_FAILURES failures in a row to bound the cost of an outage.
    """
    dates = {}
    failures = consecutive = 0
    day = start
    while day <= end:
        if day.weekday() < 5:
            try:
                response = requests.get(
                    NASDAQ_EARNINGS_URL, params={"date": day.isoformat()}, headers=NASDAQ_HEADERS, timeout=15
                )
                response.raise_for_status()
                for symbol in parse_nasdaq_earnings(response.json()):
                    dates.setdefault(symbol, day.isoformat())
                consecutive = 0
            except Exception as error:
                print(f"Nasdaq earnings calendar failed for {day}: {_safe_error(error)}")
                failures += 1
                consecutive += 1
                if consecutive >= NASDAQ_MAX_CONSECUTIVE_FAILURES:
                    break
        day += timedelta(days=1)
    if failures:
        raise RuntimeError(f"Nasdaq earnings calendar failed for {failures} day(s)")
    return dates


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_upcoming_earnings(days: int = EARNINGS_WINDOW_DAYS) -> tuple:
    """Return (source, {symbol: 'YYYY-MM-DD'}) for every US company reporting in the next `days` days.

    Finnhub answers in one call when FINNHUB_API_KEY is set (an empty answer counts as a failure);
    otherwise Nasdaq's public calendar is read one weekday at a time. Raises RuntimeError when no
    source returns a complete calendar, so neither a failure nor a partial result is cached.
    """
    start = date.today()
    end = start + timedelta(days=days)
    errors = []
    for name, fetch in (("Finnhub", _finnhub_calendar), ("Nasdaq", _nasdaq_calendar)):
        try:
            return name, fetch(start, end)
        except Exception as error:
            errors.append(f"{name}: {_safe_error(error)}")
    raise RuntimeError("; ".join(errors))


def add_earnings_columns(candidates: pd.DataFrame, earnings: dict, today: date = None) -> pd.DataFrame:
    """Add 'Earnings' (date or empty) and 'Days to ER' columns to scanner rows."""
    today = today or date.today()
    result = candidates.copy()
    dates = [earnings.get(str(ticker).upper(), "") for ticker in result["Ticker"]]
    result["Earnings"] = dates
    result["Days to ER"] = [
        (datetime.strptime(value, "%Y-%m-%d").date() - today).days if value else None
        for value in dates
    ]
    return result
