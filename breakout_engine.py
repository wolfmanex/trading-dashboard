import numpy as np
import pandas as pd


BASE_WINDOWS = (40, 35, 30, 25, 20, 15)  # 3-8 weeks of daily bars, longest first
MAX_BASE_DEPTH = 0.30
TIGHT_WINDOW = 10
MAX_PIVOT_DISTANCE = 0.08
MAX_EXTENSION_ABOVE_PIVOT = 0.02
# A stop never sits closer than this share of the pivot, even when the ATR is tiny.
MIN_STOP_PCT = 0.03
# Below this daily ATR (as a share of price) the stock is pinned, typically by a pending takeover, and a
# "base" is just the deal spread: skip it. Real small-cap breakout candidates move 2-6% a day.
MIN_ATR_PCT = 0.015

SCORE_WEIGHTS = {
    "proximity": 0.20,
    "contraction": 0.20,
    "squeeze": 0.15,
    "volume": 0.10,
    "trend": 0.15,
    "relative_strength": 0.20,
}

CANDIDATE_COLUMNS = [
    "Ticker", "Name", "Price", "Market Cap", "Pivot", "To Pivot", "Stop", "Target",
    "Reward/Risk", "Base Weeks", "Base Depth", "RVOL", "RS vs IWM", "Breakout Score",
]


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def _average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(window).mean()


def find_base(df: pd.DataFrame, max_depth: float = MAX_BASE_DEPTH):
    """Return the longest recent consolidation whose high-to-low depth stays within max_depth."""
    for window in BASE_WINDOWS:
        if len(df) < window:
            continue
        recent = df.iloc[-window:]
        base_high = float(recent["High"].max())
        base_low = float(recent["Low"].min())
        if base_high <= 0:
            continue
        depth = (base_high - base_low) / base_high
        if depth <= max_depth:
            return {"window": window, "high": base_high, "low": base_low, "depth": depth}
    return None


def analyze_breakout_setup(df: pd.DataFrame, benchmark_close: pd.Series = None) -> dict | None:
    """Measure a daily OHLCV series for a base-breakout setup and derive entry, stop and target."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    if df is None or len(df) < 60 or not required.issubset(df.columns):
        return None

    df = df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df) < 60:
        return None

    close = df["Close"]
    price = float(close.iloc[-1])
    if price <= 0:
        return None

    # The base is measured before today so a breakout bar shows as extension, not a new pivot.
    base = find_base(df.iloc[:-1])
    if base is None:
        return None

    pivot = base["high"]
    distance_to_pivot = (pivot - price) / pivot

    # Stop sits under the most recent tight area, but never closer than one ATR or MIN_STOP_PCT to the pivot.
    atr = _average_true_range(df)
    latest_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    recent_low = float(df["Low"].iloc[-TIGHT_WINDOW:].min())
    stop = min(recent_low, pivot - latest_atr, pivot * (1 - MIN_STOP_PCT))
    risk = pivot - stop
    # Measured move: the full base depth projected above the pivot.
    target = pivot + (pivot - base["low"])
    reward_risk = (target - pivot) / risk if risk > 0 else 0.0

    recent_range = (df["High"].iloc[-TIGHT_WINDOW:].max() - recent_low) / pivot
    contraction_ratio = recent_range / base["depth"] if base["depth"] > 0 else 1.0

    bb_mid = close.rolling(20).mean()
    bb_width = (4 * close.rolling(20).std()) / bb_mid
    bb_history = bb_width.dropna().iloc[-252:]
    squeeze_rank = float((bb_history <= bb_history.iloc[-1]).mean()) if len(bb_history) > 20 else 0.5

    volume = df["Volume"]
    volume_50 = float(volume.iloc[-50:].mean())
    dryup_ratio = float(volume.iloc[-TIGHT_WINDOW:].mean()) / volume_50 if volume_50 > 0 else 1.0
    rvol = float(volume.iloc[-1]) / float(volume.iloc[-51:-1].mean()) if volume.iloc[-51:-1].mean() > 0 else 0.0

    sma_50 = float(close.rolling(50).mean().iloc[-1])
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else np.nan
    sma_50_prior = float(close.rolling(50).mean().iloc[-11])
    trend_points = 0.0
    if price > sma_50:
        trend_points += 0.4
    if sma_50 > sma_50_prior:
        trend_points += 0.3
    if pd.notna(sma_200) and sma_50 > sma_200:
        trend_points += 0.3

    relative_strength = np.nan
    if len(close) >= 64:
        stock_return = price / float(close.iloc[-64]) - 1
        relative_strength = stock_return * 100
        if benchmark_close is not None:
            aligned = benchmark_close.reindex(close.index).ffill().dropna()
            if len(aligned) >= 64 and aligned.iloc[-64] > 0:
                relative_strength = (stock_return - (aligned.iloc[-1] / aligned.iloc[-64] - 1)) * 100

    dollar_volume = float((close.iloc[-20:] * volume.iloc[-20:]).mean())

    return {
        "price": price,
        "pivot": pivot,
        "distance_to_pivot": distance_to_pivot,
        "stop": stop,
        "target": target,
        "reward_risk": reward_risk,
        "base_bars": base["window"],
        "base_depth": base["depth"],
        "contraction_ratio": contraction_ratio,
        "squeeze_rank": squeeze_rank,
        "dryup_ratio": dryup_ratio,
        "rvol": rvol,
        "above_sma_50": price > sma_50,
        "trend_points": trend_points,
        "relative_strength": relative_strength,
        "avg_dollar_volume": dollar_volume,
        "atr_pct": latest_atr / price,
    }


def score_setup(setup: dict, rs_percentile: float = 0.5) -> float:
    """Combine setup measurements into a 0-100 breakout score."""
    components = {
        "proximity": 1 - _clip01(max(setup["distance_to_pivot"], 0.0) / MAX_PIVOT_DISTANCE),
        "contraction": 1 - _clip01(setup["contraction_ratio"]),
        "squeeze": 1 - _clip01(setup["squeeze_rank"]),
        "volume": _clip01((1.2 - setup["dryup_ratio"]) / 0.8),
        "trend": _clip01(setup["trend_points"]),
        "relative_strength": _clip01(rs_percentile),
    }
    return round(100 * sum(SCORE_WEIGHTS[key] * value for key, value in components.items()), 1)


def qualifies(setup: dict, min_reward_risk: float, min_dollar_volume: float) -> bool:
    return bool(
        setup
        and -MAX_EXTENSION_ABOVE_PIVOT <= setup["distance_to_pivot"] <= MAX_PIVOT_DISTANCE
        and setup["above_sma_50"]
        and setup.get("atr_pct", MIN_ATR_PCT) >= MIN_ATR_PCT
        and setup["reward_risk"] >= min_reward_risk
        and setup["avg_dollar_volume"] >= min_dollar_volume
    )


def rank_breakout_candidates(
    setups: dict,
    metadata: dict = None,
    min_reward_risk: float = 3.0,
    min_dollar_volume: float = 5_000_000,
    limit: int = 25,
) -> pd.DataFrame:
    """Filter analyzed setups and rank them by breakout score."""
    metadata = metadata or {}
    qualified = {
        ticker: setup for ticker, setup in setups.items()
        if qualifies(setup, min_reward_risk, min_dollar_volume)
    }
    if not qualified:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    # Relative strength is ranked against the whole analyzed universe, not only survivors.
    all_rs = pd.Series({
        ticker: setup["relative_strength"] for ticker, setup in setups.items()
        if setup and pd.notna(setup["relative_strength"])
    })
    rs_percentiles = all_rs.rank(pct=True) if not all_rs.empty else pd.Series(dtype=float)

    rows = []
    for ticker, setup in qualified.items():
        info = metadata.get(ticker, {})
        rows.append({
            "Ticker": ticker,
            "Name": info.get("name", ticker),
            "Price": round(setup["price"], 2),
            "Market Cap": info.get("market_cap"),
            "Pivot": round(setup["pivot"], 2),
            "To Pivot": round(setup["distance_to_pivot"] * 100, 2),
            "Stop": round(setup["stop"], 2),
            "Target": round(setup["target"], 2),
            "Reward/Risk": round(setup["reward_risk"], 2),
            "Base Weeks": round(setup["base_bars"] / 5, 1),
            "Base Depth": round(setup["base_depth"] * 100, 1),
            "RVOL": round(setup["rvol"], 2),
            "RS vs IWM": round(setup["relative_strength"], 2) if pd.notna(setup["relative_strength"]) else None,
            "Breakout Score": score_setup(setup, float(rs_percentiles.get(ticker, 0.5))),
        })

    ranked = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    ranked = ranked.sort_values(["Breakout Score", "Reward/Risk"], ascending=False).head(limit)
    return ranked.reset_index(drop=True)
