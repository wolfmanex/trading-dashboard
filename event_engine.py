import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=1800)
def get_upcoming_events(ticker: str) -> dict:
    """Fetches upcoming corporate earnings, macro indicators (VIX, 10Y Yield), and news context."""
    events = {
        "earnings_date": "N/A",
        "macro_vix": 0.0,
        "macro_tnx": 0.0,
        "news_headlines": []
    }
    
    # 1. Fetch Ticker Earnings Date & Specific News
    try:
        t = yf.Ticker(ticker)
        
        # Check calendar for next earnings date
        if hasattr(t, 'calendar') and t.calendar is not None:
            cal = t.calendar
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                earnings_dates = cal['Earnings Date']
                if earnings_dates:
                    events["earnings_date"] = str(earnings_dates[0])
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                if 'Earnings Date' in cal.index:
                    events["earnings_date"] = str(cal.loc['Earnings Date'].values[0])
                    
        # Extract recent news headlines
        if t.news:
            events["news_headlines"] = [
                f"- {item.get('title', '')} ({item.get('publisher', 'News')})" 
                for item in t.news[:5]
            ]
    except Exception as e:
        print(f"Error fetching ticker events for {ticker}: {e}")

    # 2. Fetch Macro Environment Indicators (VIX & 10-Year Yield)
    try:
        vix = yf.Ticker("^VIX").history(period="1d")
        if not vix.empty:
            events["macro_vix"] = round(float(vix['Close'].iloc[-1]), 2)
            
        tnx = yf.Ticker("^TNX").history(period="1d")
        if not tnx.empty:
            events["macro_tnx"] = round(float(tnx['Close'].iloc[-1]), 2)
    except Exception as e:
        print(f"Error fetching macro climate indicators: {e}")

    return events