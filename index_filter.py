import yfinance as yf
import pandas as pd
import pandas_ta as ta

class IndexTrendFilter:
    def __init__(self, tickers=["QQQ", "SPY", "SMH"]):
        self.tickers = tickers

    def fetch_index_data(self) -> pd.DataFrame:
        df = yf.download(self.tickers, period="2d", interval="5m", group_by='ticker', prepost=True, progress=False)
        return df

    def analyze_trend(self, df: pd.DataFrame, ticker: str) -> int:
        ticker_data = df[ticker].copy()
        ticker_data.ta.ema(length=9, append=True)
        ticker_data.ta.vwap(append=True)
        ticker_data.dropna(inplace=True)
        
        latest = ticker_data.iloc[-1]
        score = 1 if latest['Close'] > latest.get('VWAP_D', 0) else -1
        return score

    def get_market_bias(self) -> dict:
        df = self.fetch_index_data()
        qqq_score = self.analyze_trend(df, "QQQ")
        spy_score = self.analyze_trend(df, "SPY")
        smh_score = self.analyze_trend(df, "SMH")
        
        total_score = qqq_score + spy_score + smh_score
        
        if total_score >= 2: bias = "BULLISH"
        elif total_score <= -2: bias = "BEARISH"
        else: bias = "CHOPPY / MIXED"
            
        return {
            "bias": bias,
            "qqq_score": qqq_score,
            "spy_score": spy_score,
            "smh_score": smh_score
        }