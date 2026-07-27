import yfinance as yf
import pandas as pd

def get_options_sentiment(ticker: str) -> dict:
    """Fetches Options Open Interest, PCR, and Strike Walls.
       Includes strict data sanitization to prevent 0.0 PCR or phantom strikes."""
    
    fallback = {
        "pcr_oi": "N/A",
        "call_wall": "N/A",
        "put_wall": "N/A",
        "expiration": "N/A",
        "positioning_summary": "Exchange data currently unavailable or corrupted."
    }
    
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        
        if not expirations:
            return fallback
            
        # Target the nearest active expiration date
        target_exp = expirations[0]
        chain = tk.option_chain(target_exp)
        
        # 1. Clean the data: Drop rows where Open Interest is NaN or exactly 0
        calls = chain.calls.dropna(subset=['openInterest', 'strike'])
        calls = calls[calls['openInterest'] > 0]
        
        puts = chain.puts.dropna(subset=['openInterest', 'strike'])
        puts = puts[puts['openInterest'] > 0]
        
        if calls.empty or puts.empty:
            return fallback
            
        # 2. Calculate PCR-OI (Put/Call Open Interest Ratio)
        total_call_oi = calls['openInterest'].sum()
        total_put_oi = puts['openInterest'].sum()
        
        if total_call_oi > 0:
            pcr_oi = round(total_put_oi / total_call_oi, 2)
        else:
            pcr_oi = "N/A"
            
        # 3. Identify Institutional Walls (Strikes with Maximum Open Interest)
        call_wall_idx = calls['openInterest'].idxmax()
        put_wall_idx = puts['openInterest'].idxmax()
        
        call_wall = float(calls.loc[call_wall_idx]['strike'])
        put_wall = float(puts.loc[put_wall_idx]['strike'])
        
        # 4. Generate AI Positioning Context
        if isinstance(pcr_oi, float):
            if pcr_oi > 1.2:
                summary = "Heavy Put Bias (Institutional Hedging / Bearish)"
            elif pcr_oi < 0.8:
                summary = "Heavy Call Bias (Speculative / Bullish)"
            else:
                summary = "Neutral / Balanced Positioning"
        else:
            summary = "Insufficient OI for sentiment summary."
            
        return {
            "pcr_oi": pcr_oi,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "expiration": target_exp,
            "positioning_summary": summary
        }
        
    except Exception as e:
        print(f"Options Engine Error for {ticker}: {e}")
        return fallback