import yfinance as yf
import pandas as pd
from options_engine import has_valid_bid_ask, option_midpoint, select_expiration_candidates

# Standard Institutional Sector Mapping
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "QCOM": "XLK", "AMD": "XLK",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
    "XOM": "XLE", "CVX": "XLE",
    "TSLA": "XLY", "AMZN": "XLY",
    "JNJ": "XLV", "UNH": "XLV", "LLY": "XLV",
    "GOOGL": "XLC", "META": "XLC"
}

def get_swing_metrics(ticker: str, analysis_mode: str = "Intra-Day (Scalp/Day Trade)") -> dict:
    """Calculates Options Expected Move (ATM Straddle) and Sector Relative Strength."""
    metrics = {
        "expected_move_usd": "N/A",
        "expected_move_pct": "N/A",
        "upper_expected_bound": "N/A",
        "lower_expected_bound": "N/A",
        "sector_etf": "SPY",
        "relative_strength_1w": "N/A",
        "rs_rating": "Neutral / Data Unavailable"
        ,"expected_move_pricing": "N/A"
    }
    
    try:
        tk = yf.Ticker(ticker)
        # 1. Fetch Current Price
        hist = tk.history(period="1d")
        if hist.empty:
            return metrics
        current_price = hist['Close'].iloc[-1]
        
        # 2. Calculate Weekly Expected Move via ATM Straddle
        expirations = tk.options
        if expirations:
            candidate_expirations = select_expiration_candidates(expirations, analysis_mode)
            if not candidate_expirations:
                candidate_expirations = []
            if not candidate_expirations:
                raise ValueError("No valid future option expirations")
            target_exp = candidate_expirations[0]
            opt = tk.option_chain(target_exp)
            calls, puts = opt.calls, opt.puts
            
            # Find the At-The-Money (ATM) Strike
            calls['strike_dist'] = abs(calls['strike'] - current_price)
            atm_call = calls.loc[calls['strike_dist'].idxmin()]
            atm_put = puts.loc[puts['strike'] == atm_call['strike']]
            
            if not atm_put.empty:
                atm_put = atm_put.iloc[0]
                call_price = option_midpoint(atm_call)
                put_price = option_midpoint(atm_put)
                if call_price is None or put_price is None:
                    call_price = put_price = None
                else:
                    expected_move = call_price + put_price

                if call_price is not None and put_price is not None:
                    if has_valid_bid_ask(atm_call) and has_valid_bid_ask(atm_put):
                        metrics["expected_move_pricing"] = "Bid/ask midpoint"
                    else:
                        metrics["expected_move_pricing"] = "Last traded price fallback"
                else:
                    expected_move = None
                
                if expected_move is not None:
                    metrics["expected_move_usd"] = round(expected_move, 2)
                    metrics["expected_move_pct"] = round((expected_move / current_price) * 100, 2)
                    metrics["upper_expected_bound"] = round(current_price + expected_move, 2)
                    metrics["lower_expected_bound"] = round(current_price - expected_move, 2)

        # 3. Calculate Sector Relative Strength (1-Week)
        sector_etf = SECTOR_MAP.get(ticker, "SPY") # Default to S&P 500 if not mapped
        metrics["sector_etf"] = sector_etf
        
        tk_hist = tk.history(period="5d")
        etf_hist = yf.Ticker(sector_etf).history(period="5d")
        
        if len(tk_hist) >= 5 and len(etf_hist) >= 5:
            tk_perf = ((tk_hist['Close'].iloc[-1] - tk_hist['Close'].iloc[0]) / tk_hist['Close'].iloc[0]) * 100
            etf_perf = ((etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[0]) / etf_hist['Close'].iloc[0]) * 100
            
            rs_score = round(tk_perf - etf_perf, 2)
            metrics["relative_strength_1w"] = rs_score
            
            if rs_score > 2.0:
                metrics["rs_rating"] = "Strong Relative Strength (Institutional Accumulation)"
            elif rs_score < -2.0:
                metrics["rs_rating"] = "Strong Relative Weakness (Institutional Distribution)"
            else:
                metrics["rs_rating"] = "Market Performing (Neutral Flow)"

    except Exception as e:
        print(f"Swing Metrics Engine Error for {ticker}: {e}")
        
    return metrics