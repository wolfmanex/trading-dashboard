import pandas as pd
import yfinance as yf
import warnings
from technical_engine import TechnicalEngine
from index_filter import IndexTrendFilter
from news_engine import NewsSentimentEngine

warnings.filterwarnings("ignore")

def fetch_intraday_data(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="5d", interval="5m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

if __name__ == "__main__":
    TARGET_TICKER = "AMD"
    
    ta_engine = TechnicalEngine(symbol=TARGET_TICKER)
    market_filter = IndexTrendFilter()
    news_engine = NewsSentimentEngine(symbol=TARGET_TICKER)
    
    print(f"[*] Initializing Intraday Agent for {TARGET_TICKER}...")
    
    market_context = market_filter.get_market_bias()
    news_context = news_engine.analyze_sentiment()
    
    raw_data = fetch_intraday_data(TARGET_TICKER)
    processed_data = ta_engine.add_indicators(raw_data)
    latest_candle = processed_data.iloc[-1]
    base_score = ta_engine.generate_ta_signal(latest_candle)
    
    final_action = "WAIT"
    
    if base_score >= 50:
        if market_context["bias"] == "BULLISH" and news_context["status"] != "BEARISH_NEWS":
            final_action = "🟢 EXECUTING LONG TRADE (Full Alignment)"
        else:
            final_action = f"🟡 ABORT LONG: Conflict found. (Market: {market_context['bias']}, News: {news_context['status']})"
            
    elif base_score <= -50:
        if market_context["bias"] == "BEARISH" and news_context["status"] != "BULLISH_NEWS":
            final_action = "🔴 EXECUTING SHORT TRADE (Full Alignment)"
        else:
            final_action = f"🟠 ABORT SHORT: Conflict found. (Market: {market_context['bias']}, News: {news_context['status']})"
    
    print("=" * 60)
    print(f"📊 INTRADAY TRADING AGENT: {TARGET_TICKER}")
    print("=" * 60)
    print(f"📰 NEWS SENTIMENT : {news_context['status']} (Score: {news_context['score']} across {news_context['count']} headlines)")
    print(f"🌍 MARKET TREND   : {market_context['bias']}")
    print(f"📈 TA ALGO SCORE  : {base_score} / 100")
    print("-" * 60)
    print(f"🤖 ACTION         : {final_action}")
    print("=" * 60)