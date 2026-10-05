import numpy as np
import pandas as pd

from breakout_engine import (
    BASE_WINDOWS, MAX_BASE_DEPTH, MAX_EXTENSION_ABOVE_PIVOT, MAX_PIVOT_DISTANCE, MIN_ATR_PCT, MIN_STOP_PCT,
    TIGHT_WINDOW,
    analyze_breakout_setup, qualifies,
)


ENTRY_WINDOW = 10   # bars a setup has to trigger (trade through the pivot) before it is dropped
MAX_HOLD = 40       # bars a triggered trade is held before a time exit at the close
LOOKBACK = 260      # bars of history handed to the setup analysis, matching the live scan's 1y download
MIN_HISTORY = 120   # bars needed before the first signal is evaluated

TRADE_COLUMNS = ["Ticker", "Signal Date", "Entry Date", "Entry", "Stop", "Target", "Exit Date", "Exit", "Outcome", "R"]


def _candidate_days(df: pd.DataFrame, min_reward_risk: float, min_dollar_volume: float) -> np.ndarray:
    """Vectorized copy of the `qualifies` rules, used to find the days worth a full analysis.

    It mirrors analyze_breakout_setup's base, stop and target maths over every day at once, so
    the (much slower) per-day analysis only runs where a setup is likely; that analysis stays the
    authority on whether a day qualifies.
    """
    close, high, low = df["Close"], df["High"], df["Low"]

    pivot = pd.Series(np.nan, index=df.index)
    base_low = pd.Series(np.nan, index=df.index)
    for window in BASE_WINDOWS:  # longest first, like find_base; the base ends the day before
        window_high = high.shift(1).rolling(window).max()
        window_low = low.shift(1).rolling(window).min()
        fits = ((window_high - window_low) / window_high <= MAX_BASE_DEPTH) & pivot.isna()
        pivot = pivot.where(~fits, window_high)
        base_low = base_low.where(~fits, window_low)

    prev_close = close.shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().fillna(0.0)
    stop = np.minimum(np.minimum(low.rolling(TIGHT_WINDOW).min(), pivot - atr), pivot * (1 - MIN_STOP_PCT))
    risk = pivot - stop
    reward_risk = ((pivot - base_low) / risk).where(risk > 0, 0.0)
    distance = (pivot - close) / pivot

    mask = (
        distance.between(-MAX_EXTENSION_ABOVE_PIVOT, MAX_PIVOT_DISTANCE)
        & (close > close.rolling(50).mean())
        & (atr / close >= MIN_ATR_PCT * 0.98)
        & (reward_risk >= min_reward_risk * 0.98)
        & ((close * df["Volume"]).rolling(20).mean() >= min_dollar_volume * 0.98)
    ).to_numpy(copy=True)
    mask[:MIN_HISTORY] = False
    return np.flatnonzero(mask)


def simulate_trade(df: pd.DataFrame, signal_index: int, pivot: float, stop: float, target: float) -> dict | None:
    """Play one setup forward: buy stop at the pivot, then exit at the stop, the target, or after MAX_HOLD bars.

    Gaps fill at the open. When a bar touches both the stop and the target, the stop is assumed
    to fill first, which keeps the result conservative. Returns None when the setup never
    triggers (or breaks the stop first) within ENTRY_WINDOW bars.
    """
    opens, highs, lows, closes = (df[column].to_numpy(dtype=float) for column in ("Open", "High", "Low", "Close"))
    last = len(df) - 1

    entry_index = None
    for i in range(signal_index + 1, min(signal_index + ENTRY_WINDOW, last) + 1):
        if opens[i] <= stop or (lows[i] <= stop and highs[i] < pivot):
            return None
        if highs[i] >= pivot:
            entry_index, entry = i, max(opens[i], pivot)
            break
    if entry_index is None or entry <= stop:
        return None

    for i in range(entry_index, min(entry_index + MAX_HOLD, last) + 1):
        same_bar = i == entry_index
        if lows[i] <= stop:
            exit_price = stop if same_bar else min(opens[i], stop)
            return _trade(df, signal_index, entry_index, entry, stop, target, i, exit_price, "Stop")
        if highs[i] >= target:
            exit_price = target if same_bar else max(opens[i], target)
            return _trade(df, signal_index, entry_index, entry, stop, target, i, exit_price, "Target")

    exit_index = min(entry_index + MAX_HOLD, last)
    outcome = "Time exit" if exit_index == entry_index + MAX_HOLD else "Open"
    return _trade(df, signal_index, entry_index, entry, stop, target, exit_index, closes[exit_index], outcome)


def _trade(df, signal_index, entry_index, entry, stop, target, exit_index, exit_price, outcome) -> dict:
    return {
        "Signal Date": df.index[signal_index],
        "Entry Date": df.index[entry_index],
        "Entry": round(float(entry), 2),
        "Stop": round(float(stop), 2),
        "Target": round(float(target), 2),
        "Exit Date": df.index[exit_index],
        "Exit": round(float(exit_price), 2),
        "Outcome": outcome,
        "R": round(float((exit_price - entry) / (entry - stop)), 2),
        "_exit_index": exit_index,
    }


def backtest_ticker(
    df: pd.DataFrame,
    benchmark_close: pd.Series = None,
    min_reward_risk: float = 3.0,
    min_dollar_volume: float = 5_000_000,
) -> list:
    """Replay the scanner's rules day by day over one ticker's daily history, one position at a time."""
    required = ["Open", "High", "Low", "Close", "Volume"]
    if df is None or len(df) <= MIN_HISTORY or not set(required).issubset(df.columns):
        return []
    df = df[required].apply(pd.to_numeric, errors="coerce").dropna()

    trades = []
    busy_until = -1
    for t in _candidate_days(df, min_reward_risk, min_dollar_volume):
        if t <= busy_until or t >= len(df) - 1:
            continue
        setup = analyze_breakout_setup(df.iloc[max(0, t + 1 - LOOKBACK):t + 1], benchmark_close)
        if not qualifies(setup, min_reward_risk, min_dollar_volume):
            continue
        trade = simulate_trade(df, t, setup["pivot"], setup["stop"], setup["target"])
        if trade is None:
            continue
        busy_until = trade.pop("_exit_index")
        trades.append(trade)
    return trades


def summarize_trades(trades: pd.DataFrame) -> dict:
    """Win rate, average R and outcome counts for closed trades (trades still open at the end are excluded)."""
    closed = trades[trades["Outcome"] != "Open"] if not trades.empty else trades
    summary = {
        "trades": len(closed),
        "still_open": len(trades) - len(closed),
        "win_rate_pct": None,
        "average_r": None,
        "total_r": None,
        "target_pct": None,
        "stop_pct": None,
        "time_exit_pct": None,
    }
    if closed.empty:
        return summary
    r = closed["R"]
    summary.update({
        "win_rate_pct": round(float((r > 0).mean() * 100), 1),
        "average_r": round(float(r.mean()), 2),
        "total_r": round(float(r.sum()), 1),
        "target_pct": round(float((closed["Outcome"] == "Target").mean() * 100), 1),
        "stop_pct": round(float((closed["Outcome"] == "Stop").mean() * 100), 1),
        "time_exit_pct": round(float((closed["Outcome"] == "Time exit").mean() * 100), 1),
    })
    return summary
