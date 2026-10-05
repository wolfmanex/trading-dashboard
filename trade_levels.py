"""Trade levels for the AI trade reviewer, calculated in code so the model never invents prices.

A ticker loaded from the breakout scanner keeps the scanner's pivot, stop and target (long only).
Any other ticker gets a long and a short plan from the current price and the daily ATR: the stop sits
a mode-dependent number of ATRs away and the target at TARGET_R times that risk. Recent swing
support and resistance are reported alongside so the reviewer can judge where the stop sits.
"""
import math

import pandas as pd


ATR_PERIOD = 14
STRUCTURE_BARS = 20   # daily bars used for recent support (lowest low) and resistance (highest high)
TARGET_R = 2.0
STOP_ATR = {"intraday": 0.5, "swing": 1.5}


def horizon(analysis_mode: str) -> str:
    return "intraday" if "Intra-Day" in str(analysis_mode) else "swing"


def average_true_range(df: pd.DataFrame, period: int = ATR_PERIOD) -> float | None:
    """Simple-average true range of the last `period` bars, or None without enough data."""
    if df is None or df.empty or len(df) < period + 1 or not {"High", "Low", "Close"} <= set(df.columns):
        return None
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(true_range.iloc[-period:].mean())
    return atr if math.isfinite(atr) and atr > 0 else None


def structure_levels(df: pd.DataFrame, bars: int = STRUCTURE_BARS) -> dict:
    """Recent support (lowest low) and resistance (highest high) over the last `bars` daily bars."""
    if df is None or df.empty or not {"High", "Low"} <= set(df.columns):
        return {"support": None, "resistance": None}
    recent = df.iloc[-bars:]
    return {"support": round(float(recent["Low"].min()), 2), "resistance": round(float(recent["High"].max()), 2)}


def make_plan(entry: float, stop: float, target: float) -> dict:
    risk = abs(entry - stop)
    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": round(abs(target - entry) / risk, 1) if risk else None,
    }


def compute_trade_plans(price: float, df_1d: pd.DataFrame, analysis_mode: str, scanner_row: dict = None) -> dict:
    """Return the levels the reviewer works from.

    {"source": "scanner" | "atr" | None, "atr": float | None, "support", "resistance",
     "long": plan | None, "short": plan | None}. Both plans are None when there is no price or ATR.
    """
    levels = {"source": None, "atr": average_true_range(df_1d), **structure_levels(df_1d), "long": None, "short": None}
    if levels["atr"] is not None:
        levels["atr"] = round(levels["atr"], 2)

    if scanner_row:
        levels["source"] = "scanner"
        levels["long"] = make_plan(float(scanner_row["Pivot"]), float(scanner_row["Stop"]), float(scanner_row["Target"]))
        return levels

    if not price or price <= 0 or levels["atr"] is None:
        return levels
    risk = STOP_ATR[horizon(analysis_mode)] * levels["atr"]
    if risk >= price:
        return levels
    levels["source"] = "atr"
    levels["long"] = make_plan(price, price - risk, price + TARGET_R * risk)
    levels["short"] = make_plan(price, price + risk, price - TARGET_R * risk)
    return levels


def format_recent_bars(df: pd.DataFrame, bars: int, label: str) -> str:
    """Compact OHLCV table of the last `bars` rows for the prompt."""
    if df is None or df.empty:
        return f"{label}: no data."
    recent = df.iloc[-bars:]
    lines = [f"{label} (oldest first): time | open | high | low | close | volume"]
    for timestamp, row in recent.iterrows():
        stamp = timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(timestamp, "strftime") else str(timestamp)
        volume = row.get("Volume")
        volume_text = f"{volume:,.0f}" if pd.notna(volume) else "N/A"
        lines.append(
            f"{stamp} | {row['Open']:.2f} | {row['High']:.2f} | {row['Low']:.2f} | {row['Close']:.2f} | {volume_text}"
        )
    return "\n".join(lines)
