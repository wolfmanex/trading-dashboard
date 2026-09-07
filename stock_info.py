import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_profile(ticker: str) -> dict:
    """Fetch a compact company profile with graceful Yahoo Finance fallbacks."""
    profile = {
        "ticker": ticker,
        "name": ticker,
        "sector": "Unavailable",
        "industry": "Unavailable",
        "exchange": "Unavailable",
        "country": "Unavailable",
        "summary": "",
    }

    try:
        info = yf.Ticker(ticker).info
        profile.update({
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "Unavailable",
            "industry": info.get("industry") or "Unavailable",
            "exchange": info.get("fullExchangeName") or info.get("exchange") or "Unavailable",
            "country": info.get("country") or "Unavailable",
            "summary": info.get("longBusinessSummary") or "",
        })
    except Exception as error:
        print(f"Stock profile lookup failed for {ticker}: {error}")

    return profile


def format_profile_summary(profile: dict) -> str:
    """Return the complete business description for the profile section."""
    summary = str(profile.get("summary", "")).strip()
    if not summary:
        return "Business description unavailable."
    return summary
