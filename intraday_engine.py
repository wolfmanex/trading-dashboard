import yfinance as yf
import pandas as pd

def get_intraday_metrics(ticker: str) -> dict:
    """Calculates Session VWAP, Prior Day Levels, Pre-Market Levels, and RVOL.
       Includes off-hours fallback for weekends and pre-4AM EST dead zones."""
    metrics = {
        "vwap": "N/A",
        "pdh": "N/A",
        "pdl": "N/A",
        "pmh": "N/A",
        "pml": "N/A",
        "rvol": "N/A"
    }
    
    try:
        tk = yf.Ticker(ticker)
        # Fetch up to 5 days to ensure we have at least 2 valid trading days over long weekends
        df = tk.history(period="5d", interval="5m", prepost=True)
        if df.empty or len(df) < 10:
            return metrics
        
        # Ensure timezone is correctly localized to NY (Market Hours)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
        else:
            df.index = df.index.tz_convert('America/New_York')

        # Filter out days with absolutely zero volume (common bug in off-hours yfinance pulls)
        valid_df = df[df['Volume'] > 0]
        if valid_df.empty:
            return metrics

        unique_days = valid_df.index.normalize().unique()
        if len(unique_days) < 2:
            return metrics

        # Assume the last day with volume is "today", and the one before is "yesterday"
        today_date = unique_days[-1]
        yesterday_date = unique_days[-2]

        # 1. PDH / PDL (Prior Day High / Low during regular hours 09:30 - 16:00)
        yesterday_df = valid_df[(valid_df.index.normalize() == yesterday_date) & 
                                (valid_df.index.time >= pd.to_datetime('09:30').time()) & 
                                (valid_df.index.time < pd.to_datetime('16:00').time())]
        if not yesterday_df.empty:
            metrics['pdh'] = round(yesterday_df['High'].max(), 2)
            metrics['pdl'] = round(yesterday_df['Low'].min(), 2)

        # 2. PMH / PML (Pre-Market High / Low for 'Today' 04:00 - 09:30)
        premarket_df = valid_df[(valid_df.index.normalize() == today_date) & 
                                (valid_df.index.time >= pd.to_datetime('04:00').time()) & 
                                (valid_df.index.time < pd.to_datetime('09:30').time())]
        if not premarket_df.empty:
            metrics['pmh'] = round(premarket_df['High'].max(), 2)
            metrics['pml'] = round(premarket_df['Low'].min(), 2)
        else:
            metrics['pmh'] = "Awaiting Market"
            metrics['pml'] = "Awaiting Market"
        
        # 3. Session VWAP & RVOL (Today's Regular Hours)
        today_rh_df = valid_df[(valid_df.index.normalize() == today_date) & 
                               (valid_df.index.time >= pd.to_datetime('09:30').time())]
        
        if not today_rh_df.empty:
            typical_price = (today_rh_df['High'] + today_rh_df['Low'] + today_rh_df['Close']) / 3
            vol = today_rh_df['Volume']
            
            # VWAP Calculation
            vwap = (typical_price * vol).cumsum() / vol.cumsum()
            metrics['vwap'] = round(vwap.iloc[-1], 2)
            
            # RVOL (Current Volume vs Average Session Volume)
            current_vol = vol.iloc[-1]
            avg_vol = vol.mean()
            if avg_vol > 0:
                metrics['rvol'] = round(current_vol / avg_vol, 2)
            else:
                metrics['rvol'] = 1.0
        else:
             metrics['vwap'] = "Market Closed"
             metrics['rvol'] = "Market Closed"

    except Exception as e:
        print(f"Intraday Metrics Engine Error for {ticker}: {e}")
        
    return metrics