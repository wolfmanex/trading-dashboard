import yfinance as yf
import pandas as pd
import requests

def get_options_sentiment(ticker: str) -> dict:
    """Fetches Options Open Interest, PCR, and Strike Walls.
       Iterates through expirations if the nearest one is missing data."""
    
    fallback = {
        "pcr_oi": "N/A",
        "call_wall": "N/A",
        "put_wall": "N/A",
        "expiration": "N/A",
        "positioning_summary": "Exchange data currently unavailable or corrupted."
    }
    
    try:
        # 1. Inject a custom User-Agent to prevent yfinance rate-limiting/blocking
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        })
        
        tk = yf.Ticker(ticker, session=session)
        expirations = tk.options
        
        if not expirations:
            return fallback
            
        # 2. Iterate through the first 3 expirations to find valid data
        for target_exp in expirations[:3]:
            chain = tk.option_chain(target_exp)
            
            # Clean the data: Drop rows where Open Interest is NaN or exactly 0
            calls = chain.calls.dropna(subset=['openInterest', 'strike'])
            calls = calls[calls['openInterest'] > 0]
            
            puts = chain.puts.dropna(subset=['openInterest', 'strike'])
            puts = puts[puts['openInterest'] > 0]
            
            # If this chain is empty/corrupted, skip to the next expiration date
            if calls.empty or puts.empty:
                continue
                
            # Calculate PCR-OI (Put/Call Open Interest Ratio)
            total_call_oi = calls['openInterest'].sum()
            total_put_oi = puts['openInterest'].sum()
            
            if total_call_oi > 0:
                pcr_oi = round(total_put_oi / total_call_oi, 2)
                
                # Identify Institutional Walls (Strikes with Maximum Open Interest)
                call_wall_idx = calls['openInterest'].idxmax()
                put_wall_idx = puts['openInterest'].idxmax()
                
                call_wall = float(calls.loc[call_wall_idx]['strike'])
                put_wall = float(puts.loc[put_wall_idx]['strike'])
                
                # Generate AI Positioning Context
                if pcr_oi > 1.2:
                    summary = "Heavy Put Bias (Institutional Hedging / Bearish)"
                elif pcr_oi < 0.8:
                    summary = "Heavy Call Bias (Speculative / Bullish)"
                else:
                    summary = "Neutral / Balanced Positioning"
                    
                # Successfully found data, return immediately
                return {
                    "pcr_oi": pcr_oi,
                    "call_wall": call_wall,
                    "put_wall": put_wall,
                    "expiration": target_exp,
                    "positioning_summary": summary
                }
        
        # If we exhausted the first 3 expirations and found nothing valid:
        return fallback
        
    except Exception as e:
        print(f"Options Engine Error for {ticker}: {e}")
        return fallback