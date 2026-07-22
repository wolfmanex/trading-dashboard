import os
import re
import requests
import pandas as pd
import yfinance as yf
import streamlit as st

# Safely import Alpaca if installed
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Applies technical indicators (EMA_9, EMA_21, RSI, MACD, BB) to a price DataFrame."""
    if df.empty or len(df) < 14:
        return df

    # Exponential Moving Averages
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (std * 2)
    df['BB_Lower'] = df['BB_Mid'] - (std * 2)

    return df

# Clean bad ticks
def clean_bad_ticks(df: pd.DataFrame, window: int = 14, atr_multiplier: float = 4.0) -> pd.DataFrame:
    """
    Dynamically filters out API glitches using Average True Range (ATR).
    Real crashes move the candle body; glitches are massive, isolated wicks 
    that far exceed recent baseline volatility.
    """
    if df.empty or len(df) < window:
        return df
        
    df_clean = df.copy()
    
    # 1. Calculate True Range (TR) for volatility baseline
    prev_close = df_clean['Close'].shift(1)
    tr1 = df_clean['High'] - df_clean['Low']
    tr2 = (df_clean['High'] - prev_close).abs()
    tr3 = (df_clean['Low'] - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).median()
    
    body_top = df_clean[['Open', 'Close']].max(axis=1)
    body_bottom = df_clean[['Open', 'Close']].min(axis=1)
    
    upper_wick = df_clean['High'] - body_top
    lower_wick = body_bottom - df_clean['Low']
    
    min_move = df_clean['Close'] * 0.01 
    
    fake_upper = (upper_wick > (atr * atr_multiplier)) & (upper_wick > min_move)
    fake_lower = (lower_wick > (atr * atr_multiplier)) & (lower_wick > min_move)
    
    return df_clean[~(fake_upper | fake_lower)]

@st.cache_data(ttl=300)
def get_technical_data(ticker: str, timeframe: str = "5m") -> pd.DataFrame:
    """Fetches market data for a given ticker and timeframe, then calculates technical indicators."""
    tf_params = {
        "5m": {"period": "5d", "interval": "5m"},
        "15m": {"period": "1mo", "interval": "15m"},
        "1h": {"period": "3mo", "interval": "1h"},
        "1d": {"period": "1y", "interval": "1d"},
    }

    params = tf_params.get(timeframe, {"period": "5d", "interval": "5m"})

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=params["period"], interval=params["interval"], prepost=True)

        if not df.empty:
            # Handle Yahoo's MultiIndex column format
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure index is datetime without timezone for clean Plotly rendering
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # --- SCRUB BAD API TICKS HERE ---
            df = clean_bad_ticks(df)
            
            # Calculate technical indicators on the clean data
            df = add_technical_indicators(df)
            return df
            
    except Exception as e:
        print(f"Error fetching technical data for {ticker} ({timeframe}): {e}")

    return pd.DataFrame()


@st.cache_data(ttl=600)
def get_multi_timeframe_data(ticker: str):
    """Fetches 5m, 4h (resampled from 1h), and 1d data for LLM multi-timeframe analysis."""
    df_5m = get_technical_data(ticker, timeframe="5m")
    df_1h = get_technical_data(ticker, timeframe="1h")
    df_1d = get_technical_data(ticker, timeframe="1d")

    if not df_1h.empty:
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        df_4h = add_technical_indicators(df_4h)
    else:
        df_4h = pd.DataFrame()

    return df_5m, df_4h, df_1d


def get_live_price(ticker: str) -> float:
    """
    Bulletproof live price fetcher abandoning Yahoo.
    1. CNBC Public Quote API (Highly reliable, bypasses Cloudflare)
    2. yfinance 1m history fallback
    3. Alpaca Snapshot fallback
    """
    ticker = ticker.strip().upper()

    # 1. CNBC API (Bypasses Yahoo entirely)
    try:
        # Map common Yahoo tickers to CNBC formats
        cnbc_ticker = ticker
        if ticker == "BTC-USD": 
            cnbc_ticker = "BTC.CM="
        elif ticker == "EURUSD=X": 
            cnbc_ticker = "EUR="
        elif "-" in ticker:
            cnbc_ticker = ticker.replace("-", "")

        url = f"https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?symbols={cnbc_ticker}&requestMethod=itv&noform=1&fund=1&exthrs=1&output=json&events=1"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get('FormattedQuoteResult', {}).get('FormattedQuote', [])
            
            if quotes:
                q = quotes[0]
                # Check extended hours (Pre/Post) first
                ext_price = q.get('ExtendedMktQuote', {}).get('last')
                reg_price = q.get('last')
                
                # CNBC returns prices as strings with commas (e.g., "1,234.56")
                if ext_price and ext_price != "NA":
                    return float(ext_price.replace(',', ''))
                elif reg_price and reg_price != "NA":
                    return float(reg_price.replace(',', ''))
    except Exception as e:
        print(f"CNBC API failed for {ticker}: {e}")

    # 2. yfinance 1-Minute History Fallback
    try:
        t = yf.Ticker(ticker)
        df_1m = t.history(period="1d", interval="1m", prepost=True)
        if not df_1m.empty:
            last_close = df_1m['Close'].iloc[-1]
            if isinstance(last_close, pd.Series):
                last_close = last_close.iloc[-1]
            if not pd.isna(last_close) and last_close > 0:
                return float(last_close)
    except Exception:
        pass

    # 3. Alpaca Snapshot Fallback
    if "-" not in ticker and "=" not in ticker and ALPACA_AVAILABLE:
        try:
            api_key = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
            api_secret = st.secrets.get("ALPACA_API_SECRET") or os.getenv("ALPACA_API_SECRET")
            if api_key and api_secret:
                client = StockHistoricalDataClient(api_key, api_secret)
                req = StockSnapshotRequest(symbol_or_symbols=ticker, feed=DataFeed.IEX)
                res = client.get_stock_snapshot(req)
                if ticker in res:
                    snapshot = res[ticker]
                    if snapshot.latest_trade and snapshot.latest_trade.price > 0:
                        return float(snapshot.latest_trade.price)
        except Exception:
            pass

    return 0.0