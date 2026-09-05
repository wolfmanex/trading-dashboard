import argparse
import pandas as pd
import warnings
from technical_engine import get_technical_data
from index_filter import get_macro_market_trend
from news_engine import get_ticker_news_sentiment

warnings.filterwarnings("ignore")


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Run the advisory intraday TA agent.")
    parser.add_argument("--ticker", default="AMD", help="Ticker symbol to analyze.")
    parser.add_argument(
        "--timeframe",
        choices=["5m", "15m", "1h", "1d", "1w"],
        default="5m",
        help="Historical timeframe used for the technical analysis.",
    )
    return parser.parse_args(args)


def calculate_ta_score(df: pd.DataFrame) -> int:
    """Calculate a simple advisory score from the shared technical indicators."""
    if df is None or df.empty:
        return 0

    latest = df.iloc[-1]
    score = 0

    if latest["EMA_9"] > latest["EMA_21"]:
        score += 35
    else:
        score -= 35

    if latest["MACD"] > latest["Signal_Line"]:
        score += 25
    else:
        score -= 25

    if latest["Close"] > latest["EMA_21"]:
        score += 20
    else:
        score -= 20

    rsi = latest["RSI"]
    if pd.notna(rsi):
        if rsi <= 30:
            score += 20
        elif rsi >= 70:
            score -= 20

    return score


def get_market_bias(trend: str) -> str:
    """Map the shared macro trend response to the bot's compact status format."""
    if trend.startswith("Bullish"):
        return "BULLISH"
    if trend.startswith("Bearish"):
        return "BEARISH"
    return "NEUTRAL"


def get_news_status(sentiment: str) -> str:
    """Map the shared news sentiment response to the bot's compact status format."""
    if sentiment.startswith("Bullish"):
        return "BULLISH_NEWS"
    if sentiment.startswith("Bearish"):
        return "BEARISH_NEWS"
    return "NEUTRAL_NEWS"

if __name__ == "__main__":
    cli_args = parse_args()
    target_ticker = cli_args.ticker.strip().upper()
    
    print(f"[*] Initializing Advisory Agent for {target_ticker} ({cli_args.timeframe})...")

    market_trend = get_macro_market_trend()
    news_sentiment, _ = get_ticker_news_sentiment(target_ticker)
    processed_data = get_technical_data(target_ticker, timeframe=cli_args.timeframe)

    if processed_data.empty:
        raise RuntimeError(f"No technical data available for {target_ticker}")

    market_bias = get_market_bias(market_trend)
    news_status = get_news_status(news_sentiment)
    base_score = calculate_ta_score(processed_data)
    
    final_action = "WAIT"
    
    if base_score >= 50:
        if market_bias == "BULLISH" and news_status != "BEARISH_NEWS":
            final_action = "ADVISORY LONG BIAS (Full Alignment)"
        else:
            final_action = f"ABORT LONG: Conflict found. (Market: {market_bias}, News: {news_status})"
            
    elif base_score <= -50:
        if market_bias == "BEARISH" and news_status != "BULLISH_NEWS":
            final_action = "ADVISORY SHORT BIAS (Full Alignment)"
        else:
            final_action = f"ABORT SHORT: Conflict found. (Market: {market_bias}, News: {news_status})"
    
    print("=" * 60)
    print(f"📊 ADVISORY TRADING AGENT: {target_ticker}")
    print("=" * 60)
    print(f"NEWS SENTIMENT : {news_sentiment}")
    print(f"MARKET TREND   : {market_trend}")
    print(f"TA ALGO SCORE  : {base_score} / 100")
    print("-" * 60)
    print(f"ACTION         : {final_action}")
    print("=" * 60)