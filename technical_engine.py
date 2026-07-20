import pandas as pd
import pandas_ta as ta

class TechnicalEngine:
    def __init__(self, symbol="AMD"):
        self.symbol = symbol

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.vwap(append=True)
        df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['RVOL'] = df['Volume'] / df['VOL_SMA_20']
        df.dropna(inplace=True)
        return df

    def generate_ta_signal(self, current_row: pd.Series) -> int:
        score = 0
        vwap = current_row.get('VWAP_D', current_row['Close'])
        ema9 = current_row['EMA_9']
        ema21 = current_row['EMA_21']
        cci = current_row.get('CCI_14_0.015', 0)
        rvol = current_row['RVOL']
        
        if current_row['Close'] > vwap: score += 30
        else: score -= 30
            
        if ema9 > ema21: score += 20
        else: score -= 20
            
        if cci < -100: score += 20
        elif cci > 100: score -= 20
            
        if rvol > 1.5 and score > 0: score += 15 
        elif rvol > 1.5 and score < 0: score -= 15 
            
        return score