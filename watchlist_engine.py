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
                "Focus": "",
                "Price": None,
                "Change": None,
                "RSI": None,
                "TA Score": None,
                "News": "N/A",
            })
            continue

        sentiment, _ = get_ticker_news_sentiment(ticker)
        latest = prices.iloc[-1]
        rsi = latest.get("RSI")
        daily_prices = get_technical_data(ticker, timeframe="1d")
        if daily_prices is not None and len(daily_prices) >= 2:
            daily_change = (daily_prices["Close"].iloc[-1] / daily_prices["Close"].iloc[-2] - 1) * 100
        else:
            daily_change = None
        if sentiment.startswith("Bullish"):
            news_label = "Bullish"
        elif sentiment.startswith("Bearish"):
            news_label = "Bearish"
        else:
            news_label = "Neutral"
        rows.append({
            "Ticker": ticker,
            "Focus": "",
            "Price": round(float(latest["Close"]), 2),
            "Change": round(float(daily_change), 2) if daily_change is not None else None,
            "RSI": round(float(rsi), 1) if pd.notna(rsi) else "N/A",
            "TA Score": calculate_ta_score(prices),
            "News": news_label,
        })

    snapshot = pd.DataFrame(rows)
    for column in ("Price", "Change", "RSI", "TA Score"):
        snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce")
    return snapshot