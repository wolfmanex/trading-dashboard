from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf


def _ticker_frame(batch_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if batch_data.empty:
        return pd.DataFrame()
    if isinstance(batch_data.columns, pd.MultiIndex):
        if ticker not in batch_data.columns.get_level_values(0):
            return pd.DataFrame()
        return batch_data[ticker]
    return batch_data


def _percentile_rank(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    if values.nunique() <= 1:
        return pd.Series(1.0, index=values.index)
    return values.rank(pct=True)


def rank_movers(raw_rows: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Rank positive movers by relative gain, range volatility, and volume."""
    columns = [
        "Ticker", "Price", "Change", "Relative Gain", "Day Range", "RVOL", "Mover Score"
    ]
    if raw_rows is None or raw_rows.empty:
        return pd.DataFrame(columns=columns)

    rows = raw_rows.copy()
    rows = rows.dropna(subset=["Price", "Change", "Relative Gain", "Day Range", "RVOL"])
    rows = rows[(rows["Price"] > 0) & (rows["RVOL"] >= 0)]
    rows = rows[rows["Relative Gain"] > 0]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    rows["Mover Score"] = (
        0.50 * _percentile_rank(rows["Relative Gain"])
        + 0.30 * _percentile_rank(rows["Day Range"])
        + 0.20 * _percentile_rank(rows["RVOL"])
    ) * 100
    rows = rows.sort_values(["Mover Score", "Relative Gain"], ascending=False).head(limit)
    return rows[columns].reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def get_market_movers(universe, benchmark: str = "SPY", limit: int = 5) -> pd.DataFrame:
    """Find top positive relative movers from one batched daily Yahoo request."""
    tickers = list(dict.fromkeys(str(ticker).strip().upper() for ticker in universe if str(ticker).strip()))
    if not tickers or benchmark in tickers:
        tickers = [ticker for ticker in tickers if ticker != benchmark]

    try:
        batch = yf.download(
            tickers=tickers + [benchmark],
            period="30d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
        benchmark_frame = _ticker_frame(batch, benchmark)
        if benchmark_frame.empty or len(benchmark_frame) < 2:
            return pd.DataFrame()

        benchmark_close = pd.to_numeric(benchmark_frame["Close"], errors="coerce").dropna()
        if len(benchmark_close) < 2:
            return pd.DataFrame()
        benchmark_return = float(benchmark_close.iloc[-1] / benchmark_close.iloc[-2] - 1)

        rows = []
        for ticker in tickers:
            frame = _ticker_frame(batch, ticker)
            if frame.empty:
                continue
            frame = frame.copy()
            numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
            frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna(subset=numeric_columns)
            if len(frame) < 2:
                continue

            latest = frame.iloc[-1]
            previous = frame.iloc[-2]
            if previous["Close"] <= 0 or latest["Close"] <= 0:
                continue
            daily_return = float(latest["Close"] / previous["Close"] - 1)
            day_range = float((latest["High"] - latest["Low"]) / latest["Close"])
            average_volume = frame["Volume"].iloc[:-1].tail(20).mean()
            rvol = float(latest["Volume"] / average_volume) if average_volume > 0 else 0.0
            rows.append({
                "Ticker": ticker,
                "Price": float(latest["Close"]),
                "Change": daily_return * 100,
                "Relative Gain": (daily_return - benchmark_return) * 100,
                "Day Range": day_range * 100,
                "RVOL": rvol,
            })

        result = rank_movers(pd.DataFrame(rows), limit=limit)
        result.attrs["scan_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result.attrs["universe_size"] = len(tickers)
        result.attrs["benchmark"] = benchmark
        return result
    except Exception as error:
        print(f"Market movers scan failed: {error}")
        return pd.DataFrame()
