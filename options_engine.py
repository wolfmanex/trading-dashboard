import yfinance as yf
import pandas as pd


def get_options_sentiment(ticker: str) -> dict:
    """Fetches near-term options chain metrics including Put/Call OI ratio,

    Call Wall, and Put Wall to gauge institutional positioning.
    """
    default_data = {
        "pcr_oi": "N/A",
        "call_wall": "N/A",
        "put_wall": "N/A",
        "expiration": "N/A",
        "positioning_summary": "No options data available.",
    }

    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options

        if not expirations:
            return default_data

        # Use the front-month/nearest monthly or weekly expiration
        target_exp = expirations[0]
        opt = tk.option_chain(target_exp)

        calls = opt.calls.dropna(subset=["openInterest"])
        puts = opt.puts.dropna(subset=["openInterest"])

        if calls.empty or puts.empty:
            return default_data

        total_call_oi = calls["openInterest"].sum()
        total_put_oi = puts["openInterest"].sum()

        pcr_oi = (
            round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 0.0
        )

        # Find strike prices with highest Open Interest
        call_wall_row = calls.loc[calls["openInterest"].idxmax()]
        put_wall_row = puts.loc[puts["openInterest"].idxmax()]

        call_wall = float(call_wall_row["strike"])
        put_wall = float(put_wall_row["strike"])

        # Determine qualitative smart money bias
        if pcr_oi > 1.2:
            bias = "Heavy Bearish Hedging / Put-Heavy"
        elif pcr_oi < 0.65:
            bias = "Strong Bullish Bias / Call Accumulation"
        else:
            bias = "Neutral / Balanced Hedging"

        return {
            "expiration": target_exp,
            "pcr_oi": pcr_oi,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "total_call_oi": int(total_call_oi),
            "total_put_oi": int(total_put_oi),
            "positioning_summary": f"PCR-OI at {pcr_oi} ({bias}). Call Wall at USD {call_wall:.2f}, Put Wall at USD {put_wall:.2f} (Exp: {target_exp}).",
        }

    except Exception as e:
        print(f"Error fetching options metrics for {ticker}: {e}")
        return default_data