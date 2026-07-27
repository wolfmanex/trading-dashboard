import os
import requests
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

# Grab Finnhub key from streamlit secrets
FINNHUB_API_KEY = st.secrets.get("FINNHUB_API_KEY") or os.getenv("FINNHUB_API_KEY", "")

@st.cache_data(ttl=1800)
def get_upcoming_events(ticker: str) -> dict:
    """Fetches upcoming corporate earnings, macro indicators (VIX, 10Y Yield), and news context."""
    events = {
        "earnings_date": "N/A",
        "macro_vix": 0.0,
        "macro_tnx": 0.0,
        "news_headlines": []
    }
    
    # 1. Fetch Ticker Earnings Date (FINNHUB PRIMARY, YFINANCE FALLBACK)
    try:
        if FINNHUB_API_KEY:
            # Finnhub API: Get earnings calendar from today to 90 days out
            start_date = datetime.today().strftime('%Y-%m-%d')
            end_date = (datetime.today() + timedelta(days=90)).strftime('%Y-%m-%d')
            url = f"https://finnhub.io/api/v1/calendar/earnings?from={start_date}&to={end_date}&symbol={ticker}&token={FINNHUB_API_KEY}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "earningsCalendar" in data and len(data["earningsCalendar"]) > 0:
                    events["earnings_date"] = data["earningsCalendar"][0].get("date", "N/A")

        # Fallback to yfinance if Finnhub fails or key is missing
        if events["earnings_date"] == "N/A":
            t = yf.Ticker(ticker)
            today_utc = pd.Timestamp(datetime.today()).tz_localize('UTC')
            
            if hasattr(t, 'earnings_dates') and t.earnings_dates is not None:
                df = t.earnings_dates
                if not df.empty:
                    future_dates = df[df.index >= today_utc]
                    if not future_dates.empty:
                        events["earnings_date"] = future_dates.index.min().strftime("%Y-%m-%d")
            
            # Extract recent news headlines while we have the yfinance object
            if t.news:
                events["news_headlines"] = [
                    f"- {item.get('title', '')} ({item.get('publisher', 'News')})" 
                    for item in t.news[:5]
                ]
    except Exception as e:
        print(f"Error fetching ticker events for {ticker}: {e}")

    # 2. Calculate earnings proximity
    def calculate_earnings_proximity(earnings_date_str: str) -> dict:
        """Calculates days until earnings and assigns a temporal risk tag."""
        if not earnings_date_str or earnings_date_str in ["N/A", "N/A (API Unavailable)"]:
            return {"days_until_earnings": None, "proximity_flag": "NO_EARNINGS_DATA"}
        
        try:
            # Finnhub returns 'YYYY-MM-DD', safely parse it
            earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d").date()
            today_date = datetime.today().date()
            days_left = (earnings_dt - today_date).days
            
            if days_left < 0:
                flag = "PAST_EARNINGS"
            elif days_left <= 2:
                flag = "IMMEDIATE_BINARY_RISK"
            elif days_left <= 5:
                flag = "SWING_WINDOW_OVERLAP"
            else:
                flag = "OUTSIDE_SWING_WINDOW"
                
            return {
                "days_until_earnings": days_left,
                "proximity_flag": flag
            }
        except Exception as e:
            return {"days_until_earnings": None, "proximity_flag": "PARSING_ERROR"}
            
    # Execute the proximity calculation and add it to the events dict
    proximity_data = calculate_earnings_proximity(events["earnings_date"])
    events["days_until_earnings"] = proximity_data["days_until_earnings"]
    events["proximity_flag"] = proximity_data["proximity_flag"]

    # 3. Fetch Macro Environment Indicators (VIX & 10-Year Yield)
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