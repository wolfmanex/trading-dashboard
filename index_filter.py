import pandas as pd
import yfinance as yf
import requests

def get_macro_market_trend(index_ticker: str = "^GSPC") -> str:
    """
    Evaluate macro market regime (e.g., S&P 500 ^GSPC) relative to its 200-day SMA.
    Returns a human-readable trend string: Bullish, Bearish, or Neutral/Unavailable.
    """
    try:
        # 1. Add the session disguise to prevent Yahoo from blocking the request
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # 2. Pass the session into the download call
        df = yf.download(
            index_ticker, 
            period="1y", 
            interval="1d", 
            progress=False,
            session=session
        )

        # Handle yfinance multi-index column structures if present
        if isinstance(df.columns, pd.MultiIndex):
            # droplevel(1) safely removes the ticker name from the column headers
            df.columns = df.columns.droplevel(1)

        if df.empty or len(df) < 200:
            return "Neutral (Insufficient Data)"

        latest_close = df['Close'].iloc[-1]
        sma_200 = df['Close'].rolling(window=200).mean().iloc[-1]

        # Handle Series case if close returns as Series
        if isinstance(latest_close, pd.Series):
            latest_close = latest_close.item()
        if isinstance(sma_200, pd.Series):
            sma_200 = sma_200.item()

        if latest_close > sma_200:
            percent_above = ((latest_close - sma_200) / sma_200) * 100
            return f"Bullish (+{percent_above:.1f}% > 200 SMA)"
        else:
            percent_below = ((sma_200 - latest_close) / sma_200) * 100
            return f"Bearish (-{percent_below:.1f}% < 200 SMA)"

    except Exception as e:
        return f"Neutral (Error: {str(e)})"