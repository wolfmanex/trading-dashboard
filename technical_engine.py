import os
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta

# Safely import Alpaca if installed
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockSnapshotRequest, StockBarsRequest
    from alpaca.data.enums import DataFeed
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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


def clean_bad_ticks(df: pd.DataFrame, window: int = 14, atr_multiplier: float = 4.0) -> pd.DataFrame:
    """Filters out API glitches using Average True Range (ATR), protecting recent candles."""
    if df.empty or len(df) < window + 20:
        return df
        
    df_clean = df.copy()
    
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
    
    fake_upper.iloc[-20:] = False
    fake_lower.iloc[-20:] = False
    
    return df_clean[~(fake_upper | fake_lower)]


@st.cache_data(ttl=300)
def get_technical_data(ticker: str, timeframe: str = "5m") -> pd.DataFrame:
    """Fetches chart data supporting Intraday (5m, 15m, 1h) and Swing/Weekly (1d, 1w) timeframes."""
    
    # --- 1. PRIMARY ENGINE: YFINANCE ---
    tf_params = {
        "5m": {"period": "7d", "interval": "5m"},
        "15m": {"period": "1mo", "interval": "15m"},
        "1h": {"period": "3mo", "interval": "1h"},
        "1d": {"period": "1y", "interval": "1d"},
        "1w": {"period": "5y", "interval": "1wk"},  # <--- Added Weekly Support
    }
    params = tf_params.get(timeframe, {"period": "7d", "interval": "5m"})

    try:
        t = yf.Ticker(ticker)
        df = t.history(period=params["period"], interval=params["interval"], prepost=True)

        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            if str(df.index.tz) == 'UTC':
                df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
            elif df.index.tz is not None:
                try:
                    df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
                except Exception:
                    df.index = df.index.tz_localize(None)
            else:
                df.index = pd.to_datetime(df.index)
            
            df = clean_bad_ticks(df)
            df = add_technical_indicators(df)
            return df
            
    except Exception as e:
        print(f"YFinance failed for {ticker} ({timeframe}): {e}. Trying Twelve Data...")

    # --- 2. SECONDARY ENGINE: TWELVE DATA ---
    twelve_key = st.secrets.get("TWELVEDATA_API_KEY") or os.getenv("TWELVEDATA_API_KEY")
    if twelve_key:
        try:
            td_ticker = ticker.replace("-", "/") 
            twelve_tf_map = {
                "5m": "5min", 
                "15m": "15min", 
                "1h": "1h", 
                "1d": "1day", 
                "1w": "1week"  # <--- Added Weekly Support
            }
            interval = twelve_tf_map.get(timeframe, "5min")
            
            url = f"https://api.twelvedata.com/time_series?symbol={td_ticker}&interval={interval}&apikey={twelve_key}&outputsize=1000&format=JSON&timezone=America/New_York&prepost=true"
            resp = requests.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                if 'values' in data:
                    df = pd.DataFrame(data['values'])
                    df = df.iloc[::-1].copy()
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                    df.index.name = 'Date'
                    
                    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df = df.astype(float)
                    
                    df = clean_bad_ticks(df)
                    df = add_technical_indicators(df)
                    return df
        except Exception as e:
            print(f"Twelve Data failed for {ticker}: {e}. Trying Alpaca...")

    # --- 3. TERTIARY ENGINE: ALPACA ---
    if ALPACA_AVAILABLE:
        alpaca_key = st.secrets.get("ALPACA_API_KEY") or os.getenv("ALPACA_API_KEY")
        alpaca_secret = st.secrets.get("ALPACA_API_SECRET") or os.getenv("ALPACA_API_SECRET")
        
        if alpaca_key and alpaca_secret:
            try:
                client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
                tf_map = {
                    "5m": (TimeFrame(5, TimeFrameUnit.Minute), 7),
                    "15m": (TimeFrame(15, TimeFrameUnit.Minute), 30),
                    "1h": (TimeFrame(1, TimeFrameUnit.Hour), 90),
                    "1d": (TimeFrame(1, TimeFrameUnit.Day), 365),
                    "1w": (TimeFrame(1, TimeFrameUnit.Week), 730)  # <--- Added Weekly Support
                }
                alpaca_tf, days_back = tf_map.get(timeframe, (TimeFrame(5, TimeFrameUnit.Minute), 7))
                
                start_date = datetime.utcnow() - timedelta(days=days_back)
                req = StockBarsRequest(
                    symbol_or_symbols=ticker,
                    timeframe=alpaca_tf,
                    start=start_date,
                    feed=DataFeed.IEX 
                )
                bars = client.get_stock_bars(req)
                
                if bars and ticker in bars.data:
                    df = bars.df.loc[ticker].copy()
                    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df.index = df.index.tz_convert('US/Eastern').tz_localize(None)
                    
                    df = clean_bad_ticks(df)
                    df = add_technical_indicators(df)
                    return df
            except Exception as e:
                print(f"Alpaca fallback failed for {ticker}: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=600)
def get_multi_timeframe_data(ticker: str):
    """Fetches 5m, 4h (resampled from 1h), and 1d data for LLM analysis."""
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
    """Fetches live price using yfinance (Primary), CNBC (Secondary), and Twelve Data (Tertiary)."""
    ticker = ticker.strip().upper()

    # 1. YFINANCE (Primary)
    try:
        t = yf.Ticker(ticker)
        df_1m = t.history(period="1d", interval="1m", prepost=True)
        if not df_1m.empty:
            last_close = df_1m['Close'].iloc[-1]
            if isinstance(last_close, pd.Series):
                last_close = last_close.iloc[-1]
            if not pd.isna(last_close) and last_close > 0:
                return float(last_close)
    except Exception as e:
        print(f"yfinance live price failed for {ticker}: {e}")

    # 2. CNBC API (Secondary)
    try:
        cnbc_ticker = ticker
        if ticker == "BTC-USD": 
            cnbc_ticker = "BTC.CM="
        elif ticker == "EURUSD=X": 
            cnbc_ticker = "EUR="
        elif "-" in ticker:
            cnbc_ticker = ticker.replace("-", "")

        url = f"https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol?symbols={cnbc_ticker}&requestMethod=itv&noform=1&fund=1&exthrs=1&output=json&events=1"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get('FormattedQuoteResult', {}).get('FormattedQuote', [])
            
            if quotes:
                q = quotes[0]
                reg_price = q.get('last')
                ext_price = q.get('ExtendedMktQuote', {}).get('last')
                market_state = str(q.get('curmktstate', 'REG_MKT')).upper()
                
                if "REG" not in market_state and ext_price and ext_price != "NA":
                    return float(str(ext_price).replace(',', ''))
                
                if reg_price and reg_price != "NA":
                    return float(str(reg_price).replace(',', ''))
    except Exception as e:
        print(f"CNBC API failed for {ticker}: {e}")

    # 3. TWELVE DATA API (Tertiary)
    try:
        twelve_key = st.secrets.get("TWELVEDATA_API_KEY") or os.getenv("TWELVEDATA_API_KEY")
        if twelve_key:
            td_ticker = ticker.replace("-", "/")
            url = f"https://api.twelvedata.com/price?symbol={td_ticker}&apikey={twelve_key}&prepost=true"
            
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "price" in data:
                    return float(data["price"])
    except Exception as e:
        print(f"Twelve Data live price failed for {ticker}: {e}")

    return 0.0