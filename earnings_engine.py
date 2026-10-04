from datetime import date, datetime, timedelta

import pandas as pd
import requests
import streamlit as st

from secrets_config import get_configured_secret
from smallcap_screener import NASDAQ_HEADERS


EARNINGS_WINDOW_DAYS = 14
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
    response = requests.get(
        FINNHUB_EARNINGS_URL,
        params={"from": start.isoformat(), "to": end.isoformat(), "token": api_key},
        timeout=20,
    )
    response.raise_for_status()
    return parse_finnhub_earnings(response.json())


def _nasdaq_calendar(start: date, end: date) -> dict:
    dates = {}
    failures = 0
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
            except Exception as error:
                print(f"Nasdaq earnings calendar failed for {day}: {error}")
                failures += 1
        day += timedelta(days=1)
    if failures and not dates:
        raise RuntimeError(f"Nasdaq earnings calendar failed for {failures} day(s)")
    return dates


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_upcoming_earnings(days: int = EARNINGS_WINDOW_DAYS) -> tuple:
    """Return (source, {symbol: 'YYYY-MM-DD'}) for every US company reporting in the next `days` days.

    Finnhub answers in one call when FINNHUB_API_KEY is set; otherwise Nasdaq's public calendar
    is read one weekday at a time. Raises RuntimeError when no source answers, so the failure
    is not cached.
    """
    start = date.today()
    end = start + timedelta(days=days)
    errors = []
    for name, fetch in (("Finnhub", _finnhub_calendar), ("Nasdaq", _nasdaq_calendar)):
        try:
            return name, fetch(start, end)
        except Exception as error:
            errors.append(f"{name}: {error}")
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
