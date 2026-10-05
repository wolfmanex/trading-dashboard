"""Market conditions: index trends, breadth, VIX and the macro calendar, rolled into one label.

Breakouts fail far more often in a weak market, so the scanner, the AI reviewer and the morning scan
all read this. The label is Risk-on, Neutral or Risk-off from a simple vote:
- IWM and SPY trend (classify_regime): +1 for an uptrend, -1 for a downtrend, each
- breadth, the share of MOVER_UNIVERSE above its 50-day average: +1 above 60%, -1 below 40%
- VIX: +1 below 16, -1 above 25
A total of 2 or more is Risk-on, -2 or less Risk-off. Upcoming FOMC, CPI, jobs and similar releases
come from Nasdaq's economic calendar, with the Fed's published FOMC dates as a fallback.
"""
import re
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from mover_universe import MOVER_UNIVERSE
from smallcap_screener import NASDAQ_HEADERS, _ticker_frame, classify_regime


NASDAQ_ECONOMIC_URL = "https://api.nasdaq.com/api/calendar/economicevents"
MACRO_WINDOW_DAYS = 14
EVENT_WARNING_DAYS = 2   # a major release this close makes a fresh breakout entry a coin flip on the number
RISK_OFF_SIZE = 0.5      # share of the normal position size suggested in a Risk-off market

# Second (decision) day of each FOMC meeting, from the Fed's published 2026 calendar. Extend yearly.
FOMC_DECISIONS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]

# US releases that move the whole market; matched case-insensitively against Nasdaq's event names.
MAJOR_EVENTS = [
    (r"\bFOMC\b|Fed Interest Rate Decision|Federal Funds Rate", "FOMC rate decision"),
    (r"\bCPI\b|Consumer Price Index", "CPI"),
    (r"Nonfarm Payrolls|Non-Farm Payrolls|Employment Situation|Unemployment Rate", "Jobs report"),
    (r"\bPCE\b", "PCE inflation"),
    (r"\bPPI\b|Producer Price Index", "PPI"),
    (r"\bGDP\b", "GDP"),
    (r"Retail Sales", "Retail sales"),
]


def classify_vix(level: float | None) -> str:
    if level is None:
        return "Unknown"
    if level < 16:
        return "Calm"
    if level <= 22:
        return "Normal"
    if level <= 30:
        return "Elevated"
    return "Stressed"


def breadth_pct(frames: dict) -> float | None:
    """Share (0-100) of the given daily frames whose last close is above their 50-day average."""
    above = total = 0
    for frame in frames.values():
        close = pd.to_numeric(frame.get("Close"), errors="coerce").dropna() if frame is not None and not frame.empty else None
        if close is None or len(close) < 50:
            continue
        total += 1
        above += int(close.iloc[-1] > close.iloc[-50:].mean())
    return round(100 * above / total, 1) if total else None


def classify_market(iwm: dict, spy: dict, breadth: float | None, vix: float | None) -> dict:
    """Vote the inputs into Risk-on / Neutral / Risk-off and list the reasons."""
    score, reasons = 0, []
    for name, regime in (("IWM", iwm), ("SPY", spy)):
        label = (regime or {}).get("label", "Unknown")
        if label == "Uptrend":
            score += 1
        elif label == "Downtrend":
            score -= 1
        reasons.append(f"{name} {label.lower()}")
    if breadth is not None:
        if breadth > 60:
            score += 1
        elif breadth < 40:
            score -= 1
        reasons.append(f"{breadth:.0f}% of large caps above their 50-day")
    if vix is not None:
        if vix < 16:
            score += 1
        elif vix > 25:
            score -= 1
        reasons.append(f"VIX {vix:.1f} ({classify_vix(vix).lower()})")
    label = "Risk-on" if score >= 2 else "Risk-off" if score <= -2 else "Neutral"
    return {"label": label, "score": score, "reasons": reasons}


def parse_economic_events(payload, day: str) -> list:
    """Major US releases from one day of Nasdaq's economic calendar."""
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("rows") if isinstance(data, dict) else None
    events = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        country = str(row.get("country", "")).strip().lower()
        if country and country not in {"united states", "us", "usa"}:
            continue
        name = str(row.get("eventName") or row.get("event") or "").strip()
        for pattern, label in MAJOR_EVENTS:
            if re.search(pattern, name, flags=re.IGNORECASE):
                events.append({"date": day, "event": label, "detail": name, "time": str(row.get("gmt") or "").strip()})
                break
    return events


def merge_events(events: list, today: date, days: int = MACRO_WINDOW_DAYS) -> list:
    """Add the FOMC fallback dates, drop duplicates (one per event per day) and keep the window, sorted."""
    end = today + timedelta(days=days)
    combined = list(events) + [
        {"date": day, "event": "FOMC rate decision", "detail": "FOMC decision (Fed calendar)", "time": "18:00"}
        for day in FOMC_DECISIONS
    ]
    unique = {}
    for event in combined:
        if today.isoformat() <= event["date"] <= end.isoformat():
            unique.setdefault((event["date"], event["event"]), event)
    return sorted(unique.values(), key=lambda event: (event["date"], event["event"]))


def upcoming_macro_events(today: date = None, days: int = MACRO_WINDOW_DAYS) -> tuple:
    """(events, source note). Nasdaq per weekday, merged with the FOMC list; never raises."""
    today = today or date.today()
    events, failures, checked = [], 0, 0
    day = today
    while day <= today + timedelta(days=days):
        if day.weekday() < 5:
            checked += 1
            try:
                response = requests.get(NASDAQ_ECONOMIC_URL, params={"date": day.isoformat()}, headers=NASDAQ_HEADERS, timeout=10)
                response.raise_for_status()
                events += parse_economic_events(response.json(), day.isoformat())
            except Exception as error:
                failures += 1
                if failures == 1:
                    print(f"Nasdaq economic calendar failed for {day}: {error}")
                if failures >= 3 and not events:
                    break   # unreachable; don't spend the whole window timing out
        day += timedelta(days=1)
    source = "Nasdaq economic calendar + Fed FOMC dates" if failures < checked else "Fed FOMC dates only (Nasdaq calendar unavailable)"
    return merge_events(events, today, days), source


def events_soon(events: list, today: date = None, days: int = EVENT_WARNING_DAYS) -> list:
    today = today or date.today()
    limit = (today + timedelta(days=days)).isoformat()
    return [event for event in events if event["date"] <= limit]


def format_events(events: list) -> str:
    if not events:
        return "No major US releases in the next two weeks."
    return "; ".join(f"{event['event']} {event['date']}" for event in events)


@st.cache_data(ttl=1800, show_spinner=False)
def get_market_conditions() -> dict:
    """Everything the market panel shows; parts that fail come back as None/Unknown, never an exception."""
    conditions = {"iwm": classify_regime(None), "spy": classify_regime(None), "breadth": None, "vix": None,
                  "vix_change_5d": None, "events": [], "events_source": "", "error": None}
    try:
        batch = yf.download(
            tickers=["IWM", "SPY", "^VIX"], period="2y", interval="1d", group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
        conditions["iwm"] = classify_regime(_ticker_frame(batch, "IWM").get("Close"))
        conditions["spy"] = classify_regime(_ticker_frame(batch, "SPY").get("Close"))
        vix_frame = _ticker_frame(batch, "^VIX")
        vix_close = pd.to_numeric(vix_frame["Close"], errors="coerce").dropna() if "Close" in vix_frame else pd.Series(dtype=float)
        if len(vix_close) > 5:
            conditions["vix"] = round(float(vix_close.iloc[-1]), 2)
            conditions["vix_change_5d"] = round(float(vix_close.iloc[-1] - vix_close.iloc[-6]), 2)
        stocks = yf.download(
            tickers=MOVER_UNIVERSE, period="6mo", interval="1d", group_by="ticker",
            auto_adjust=True, progress=False, threads=True,
        )
        conditions["breadth"] = breadth_pct({ticker: _ticker_frame(stocks, ticker) for ticker in MOVER_UNIVERSE})
    except Exception as error:
        conditions["error"] = str(error)
        print(f"Market conditions download failed: {error}")
    conditions["events"], conditions["events_source"] = upcoming_macro_events()
    conditions.update(classify_market(conditions["iwm"], conditions["spy"], conditions["breadth"], conditions["vix"]))
    return conditions


def summary_line(conditions: dict) -> str:
    """One line for Telegram and the AI prompts."""
    line = f"Market: {conditions['label']} ({', '.join(conditions['reasons'])})"
    soon = events_soon(conditions["events"])
    if soon:
        line += f". Coming up: {format_events(soon)}"
    return line
