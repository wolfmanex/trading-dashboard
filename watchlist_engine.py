import pandas as pd
import streamlit as st

from bot import calculate_ta_score
from news_engine import get_ticker_news_sentiment
from technical_engine import get_technical_data


def normalize_watchlist(tickers, max_items: int = 8) -> list:
    normalized = []
    for ticker in tickers or []:
        symbol = str(ticker).strip().upper()
        if symbol and symbol not in normalized:
            normalized.append(symbol)
    return normalized[:max_items]


@st.cache_data(ttl=300, show_spinner=False)
def get_watchlist_snapshot(tickers, timeframe: str = "5m") -> pd.DataFrame:
    rows = []
    for ticker in normalize_watchlist(tickers):
        prices = get_technical_data(ticker, timeframe=timeframe)
        if prices is None or prices.empty:
            rows.append({
                "Ticker": ticker,
                "Price": None,
                "RSI": None,
                "TA Score": None,
                "News": "Unavailable",
            })
            continue

        sentiment, _ = get_ticker_news_sentiment(ticker)
        latest = prices.iloc[-1]
        rsi = latest.get("RSI")
        rows.append({
            "Ticker": ticker,
            "Price": round(float(latest["Close"]), 2),
            "RSI": round(float(rsi), 1) if pd.notna(rsi) else "N/A",
            "TA Score": calculate_ta_score(prices),
            "News": sentiment,
        })

    snapshot = pd.DataFrame(rows)
    for column in ("Price", "RSI", "TA Score"):
        snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")
    return snapshot