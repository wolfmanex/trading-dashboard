import yfinance as yf
import pandas as pd
import requests
from datetime import date, datetime


def select_expiration_candidates(expirations, analysis_mode: str = "Intra-Day (Scalp/Day Trade)", as_of=None):
    """Rank future expirations near the requested strategy horizon."""
    as_of = as_of or date.today()
    target_days = 7 if "Intra-Day" in analysis_mode else 14
    candidates = []

    for expiration in expirations:
        try:
            expiration_date = datetime.strptime(str(expiration), "%Y-%m-%d").date()
        except ValueError:
            continue
        days_out = (expiration_date - as_of).days
        if days_out >= 0:
            candidates.append((abs(days_out - target_days), days_out, str(expiration)))

    candidates.sort()
    return [expiration for _, _, expiration in candidates]


def option_midpoint(option) -> float | None:
    """Return a valid bid/ask midpoint, falling back to last traded price."""
    bid = option.get("bid")
    ask = option.get("ask")
    if has_valid_bid_ask(option):
        return float((bid + ask) / 2)

    last_price = option.get("lastPrice")
    if pd.notna(last_price) and last_price > 0:
        return float(last_price)
    return None


def has_valid_bid_ask(option) -> bool:
    bid = option.get("bid")
    ask = option.get("ask")
    return bool(pd.notna(bid) and pd.notna(ask) and bid >= 0 and ask >= bid and ask > 0)


def get_options_sentiment(ticker: str, analysis_mode: str = "Intra-Day (Scalp/Day Trade)") -> dict:
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
        # 1. Inject a custom User-Agent and extended headers to prevent yfinance rate-limiting/blocking
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # 2. Ping the Yahoo Finance homepage to grab the required session cookies
        try:
            session.get("https://finance.yahoo.com", timeout=5)
        except Exception as e:
            print(f"Cookie ping failed, continuing anyway: {e}")
        
        tk = yf.Ticker(ticker, session=session)
        expirations = tk.options
        
        if not expirations:
            return fallback
            
        # 3. Prefer expirations near the selected strategy horizon.
        candidate_expirations = select_expiration_candidates(expirations, analysis_mode)[:3]
        for target_exp in candidate_expirations:
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