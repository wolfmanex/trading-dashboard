"""Intraday breakout setups for liquid large caps, measured from 5-minute bars (pre-market included).

The trigger is the opening-range high once the first 30 minutes have traded, and the pre-market high
(or, without pre-market trading, the prior day's high) before that. A setup is listed when the price
sits just under the trigger or has broken it by no more than MAX_EXTENSION, on at least MIN_RVOL times
the normal volume for this point in the day and, during the session, above VWAP.

Risk is measured from the entry: the current price once it is through the trigger, the trigger before.
Stops sit under the tighter of the opening-range low and VWAP, kept between MIN_STOP_ATR and
MAX_STOP_ATR of the stock's average daily range (ATR, from the prior sessions' bars) and never more
than MAX_RISK_PCT under the entry; the target is TARGET_R times that risk above the entry. Stocks whose
ATR is under MIN_ATR_PCT of the price rarely travel far enough in a day and are not listed. Everything
here is pure (no downloads) so the scanner, the alerts and the tests share it.
"""
from datetime import time

import numpy as np
import pandas as pd


EASTERN = "America/New_York"
PREMARKET_OPEN = time(4, 0)
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)
BAR_MINUTES = 5
# 30 minutes rather than 15: the first quarter hour's swings set off too many false breakouts and
# leave stops inside normal noise.
OPENING_RANGE_MINUTES = 30
MAX_BELOW_TRIGGER = 0.02        # list setups up to 2% under the trigger
# Up to 0.5% over it. Entries further through the trigger had a worse record (2-25 Nov 2026 review):
# the risk from such an entry is far bigger than the setup's own.
MAX_EXTENSION = 0.005
MIN_RVOL = 2.0                  # under 2x lost the most in the same review
MIN_ATR_PCT = 0.015             # defensive names with smaller daily ranges almost never reached target
MIN_STOP_ATR = 0.5              # a stop never sits closer than half a day's average range under the entry
MAX_STOP_ATR = 1.0              # nor further than a full day's range
MIN_RISK_PCT = 0.005            # and in any case 0.5%-3% under the entry
MAX_RISK_PCT = 0.03
TARGET_R = 2.0

PHASE_PREMARKET = "Pre-market"
PHASE_OPENING = "Opening range"
PHASE_SESSION = "Session"
BREAKING_OUT = "Breaking out"
NEAR_TRIGGER = "Near trigger"

SCORE_WEIGHTS = {
    "rvol": 0.30,
    "proximity": 0.20,
    "relative_strength": 0.20,
    "vwap": 0.10,
    "above_pdh": 0.10,
    "momentum": 0.10,
}

DAYTRADE_COLUMNS = [
    "Ticker", "Name", "Price", "Market Cap", "Status", "Trigger", "Trigger Level", "To Trigger", "Stop",
    "Target", "Risk %", "ATR %", "RVOL", "VWAP", "OR High", "PM High", "Prior High", "Change %", "RS vs SPY",
    "Dollar Volume", "Day Score",
]


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def to_eastern(bars: pd.DataFrame) -> pd.DataFrame:
    """The bars with a New York time index (yfinance gives exchange time; tz-naive is taken as UTC)."""
    if bars is None or bars.empty:
        return pd.DataFrame()
    index = pd.DatetimeIndex(bars.index)
    index = index.tz_localize("UTC") if index.tz is None else index
    bars = bars.copy()
    bars.index = index.tz_convert(EASTERN)
    return bars


def split_sessions(bars: pd.DataFrame) -> dict:
    """Group bars by New York date into {"premarket": frame, "regular": frame} per day, oldest first."""
    bars = to_eastern(bars)
    if bars.empty:
        return {}
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(bars.columns):
        return {}
    bars = bars[required].apply(pd.to_numeric, errors="coerce").dropna(subset=["High", "Low", "Close"])
    bars["Volume"] = bars["Volume"].fillna(0)
    times = bars.index.time
    sessions = {}
    for day in sorted(set(bars.index.date)):
        on_day = bars.index.date == day
        sessions[day] = {
            "premarket": bars[on_day & (times >= PREMARKET_OPEN) & (times < SESSION_OPEN)],
            "regular": bars[on_day & (times >= SESSION_OPEN) & (times < SESSION_CLOSE)],
        }
    return sessions


def _minutes(timestamp) -> int:
    return timestamp.hour * 60 + timestamp.minute


def _volume_through(frame: pd.DataFrame, minute: int) -> float:
    if frame.empty:
        return 0.0
    minutes = np.array([_minutes(ts) for ts in frame.index])
    return float(frame["Volume"][minutes <= minute].sum())


def average_daily_range(sessions: dict, prior_days: list) -> float | None:
    """Average true range of the prior regular sessions (up to the last five)."""
    ranges, previous_close = [], None
    for day in prior_days[-6:]:
        regular = sessions[day]["regular"]
        high, low, close = float(regular["High"].max()), float(regular["Low"].min()), float(regular["Close"].iloc[-1])
        if previous_close is not None:
            ranges.append(max(high, previous_close) - min(low, previous_close))
        elif len(prior_days) == 1:
            ranges.append(high - low)
        previous_close = close
    ranges = ranges[-5:]
    return float(np.mean(ranges)) if ranges else None


def session_vwap(regular: pd.DataFrame) -> float | None:
    if regular.empty or regular["Volume"].sum() <= 0:
        return None
    typical = (regular["High"] + regular["Low"] + regular["Close"]) / 3
    return float((typical * regular["Volume"]).sum() / regular["Volume"].sum())


def day_change_pct(bars: pd.DataFrame) -> float | None:
    """Percent change from the prior regular-session close to the latest price (pre-market included)."""
    sessions = split_sessions(bars)
    days = list(sessions)
    if len(days) < 2:
        return None
    prior = next((sessions[day]["regular"] for day in reversed(days[:-1]) if not sessions[day]["regular"].empty), None)
    today = sessions[days[-1]]
    latest = pd.concat([today["premarket"], today["regular"]])
    if prior is None or latest.empty:
        return None
    prior_close = float(prior["Close"].iloc[-1])
    return (float(latest["Close"].iloc[-1]) / prior_close - 1) * 100 if prior_close > 0 else None


def analyze_daytrade_setup(
    bars: pd.DataFrame, benchmark_change: float | None = None, opening_range_minutes: int = OPENING_RANGE_MINUTES,
) -> dict | None:
    """Measure today's 5-minute bars (with pre-market) against the opening range, VWAP and prior day.

    Needs today plus at least one prior day with regular-session bars; the relative-volume baseline is
    every prior day in `bars` (5 days from the scanner's download).
    """
    sessions = split_sessions(bars)
    days = list(sessions)
    if len(days) < 2:
        return None
    today_day = days[-1]
    today = sessions[today_day]
    prior_days = [day for day in days[:-1] if not sessions[day]["regular"].empty]
    if not prior_days:
        return None
    prior = sessions[prior_days[-1]]["regular"]
    premarket, regular = today["premarket"], today["regular"]
    latest = pd.concat([premarket, regular])
    if latest.empty:
        return None

    price = float(latest["Close"].iloc[-1])
    prior_high, prior_low = float(prior["High"].max()), float(prior["Low"].min())
    prior_close = float(prior["Close"].iloc[-1])
    if price <= 0 or prior_close <= 0:
        return None
    pm_high = float(premarket["High"].max()) if not premarket.empty else None
    pm_low = float(premarket["Low"].min()) if not premarket.empty else None

    range_bars = max(opening_range_minutes // BAR_MINUTES, 1)
    if regular.empty:
        phase = PHASE_PREMARKET
    elif len(regular) < range_bars:
        phase = PHASE_OPENING
    else:
        phase = PHASE_SESSION

    opening = regular.iloc[:range_bars]
    or_high = float(opening["High"].max()) if phase == PHASE_SESSION else None
    or_low = float(opening["Low"].min()) if phase == PHASE_SESSION else None
    vwap = session_vwap(regular)

    if phase == PHASE_SESSION:
        trigger, trigger_level, stop_candidates = or_high, "OR high", [or_low, vwap]
    elif pm_high is not None:
        trigger, trigger_level, stop_candidates = pm_high, "PM high", [pm_low, vwap]
    else:
        trigger, trigger_level, stop_candidates = prior_high, "Prior high", [vwap]

    # Risk is taken from where the trade would be entered: the price once through the trigger.
    entry = max(price, trigger)
    atr = average_daily_range(sessions, prior_days)
    min_risk = max(MIN_STOP_ATR * atr, entry * MIN_RISK_PCT) if atr else entry * MIN_RISK_PCT
    max_risk = min(max(MAX_STOP_ATR * atr, min_risk), entry * MAX_RISK_PCT) if atr else entry * MAX_RISK_PCT
    min_risk = min(min_risk, max_risk)
    # The tightest candidate stop that leaves at least min_risk, never wider than max_risk.
    tightest_allowed, loosest_allowed = entry - min_risk, entry - max_risk
    valid = [level for level in stop_candidates if level is not None and level <= tightest_allowed]
    stop = max(max(valid) if valid else tightest_allowed, loosest_allowed)
    risk = entry - stop
    target = entry + TARGET_R * risk

    # Relative volume: today's volume so far against the prior days' volume through the same minute
    # (pre-market against pre-market, the session from the open against the same stretch of the session).
    window = "regular" if not regular.empty else "premarket"
    minute = _minutes(latest.index[-1])
    today_volume = float(today[window]["Volume"].sum())
    baseline = [_volume_through(sessions[day][window], minute) for day in prior_days]
    baseline = [volume for volume in baseline if volume > 0]
    rvol = today_volume / float(np.mean(baseline)) if baseline else 0.0

    dollar_volume = float(np.mean([
        (sessions[day]["regular"]["Close"] * sessions[day]["regular"]["Volume"]).sum() for day in prior_days
    ]))
    change_pct = (price / prior_close - 1) * 100
    relative_strength = change_pct - benchmark_change if benchmark_change is not None else change_pct
    distance = (trigger - price) / trigger

    return {
        "phase": phase,
        "price": price,
        "trigger": trigger,
        "trigger_level": trigger_level,
        "distance_to_trigger": distance,
        "status": BREAKING_OUT if distance <= 0 else NEAR_TRIGGER,
        "stop": stop,
        "target": target,
        "entry": entry,
        "risk_pct": risk / entry,
        "atr_pct": atr / price if atr else None,
        "rvol": rvol,
        "vwap": vwap,
        "above_vwap": vwap is None or price > vwap,
        "or_high": or_high,
        "or_low": or_low,
        "pm_high": pm_high,
        "prior_high": prior_high,
        "prior_low": prior_low,
        "prior_close": prior_close,
        "above_prior_high": price > prior_high,
        "change_pct": change_pct,
        "relative_strength": relative_strength,
        "avg_dollar_volume": dollar_volume,
    }


def score_daytrade(setup: dict, rs_percentile: float = 0.5) -> float:
    """0-100: volume first, then closeness to the trigger and strength against the market."""
    distance = setup["distance_to_trigger"]
    vwap = setup.get("vwap")
    components = {
        # 1x is nothing special; 3x or more earns the full weight.
        "rvol": _clip01((setup["rvol"] - 1) / 2),
        # Right at (or just through) the trigger is best; the far edge of the band scores zero.
        "proximity": 1.0 if distance <= 0 else 1 - _clip01(distance / MAX_BELOW_TRIGGER),
        "relative_strength": _clip01(rs_percentile),
        # Up to 1% above VWAP; before the open there's no VWAP, so half credit.
        "vwap": _clip01((setup["price"] / vwap - 1) / 0.01) if vwap else 0.5,
        "above_pdh": 1.0 if setup["above_prior_high"] else 0.0,
        # Up 3% or more on the day earns the full weight.
        "momentum": _clip01(setup["change_pct"] / 3),
    }
    return round(100 * sum(SCORE_WEIGHTS[key] * value for key, value in components.items()), 1)


def qualifies_daytrade(setup: dict, min_rvol: float, min_dollar_volume: float, min_atr_pct: float = MIN_ATR_PCT) -> bool:
    return bool(
        setup
        and -MAX_EXTENSION <= setup["distance_to_trigger"] <= MAX_BELOW_TRIGGER
        and setup["above_vwap"]
        and setup["rvol"] >= min_rvol
        and setup["avg_dollar_volume"] >= min_dollar_volume
        and (setup.get("atr_pct") is None or setup["atr_pct"] >= min_atr_pct)
    )


def _round(value, digits: int = 2):
    return round(float(value), digits) if value is not None and pd.notna(value) else None


def rank_daytrade_candidates(
    setups: dict,
    metadata: dict = None,
    min_rvol: float = MIN_RVOL,
    min_dollar_volume: float = 100_000_000,
    limit: int = 25,
) -> pd.DataFrame:
    """Filter analyzed setups to day-trade breakouts and rank them by day score."""
    metadata = metadata or {}
    qualified = {
        ticker: setup for ticker, setup in setups.items()
        if qualifies_daytrade(setup, min_rvol, min_dollar_volume)
    }
    if not qualified:
        return pd.DataFrame(columns=DAYTRADE_COLUMNS)

    # Strength against the market is ranked across the whole analyzed universe.
    all_rs = pd.Series({ticker: setup["relative_strength"] for ticker, setup in setups.items() if setup})
    rs_percentiles = all_rs.rank(pct=True)

    rows = []
    for ticker, setup in qualified.items():
        info = metadata.get(ticker, {})
        rows.append({
            "Ticker": ticker,
            "Name": info.get("name", ticker),
            "Price": _round(setup["price"]),
            "Market Cap": info.get("market_cap"),
            "Status": setup["status"],
            "Trigger": _round(setup["trigger"]),
            "Trigger Level": setup["trigger_level"],
            "To Trigger": _round(setup["distance_to_trigger"] * 100),
            "Stop": _round(setup["stop"]),
            "Target": _round(setup["target"]),
            "Risk %": _round(setup["risk_pct"] * 100),
            "ATR %": _round(setup["atr_pct"] * 100) if setup.get("atr_pct") is not None else None,
            "RVOL": _round(setup["rvol"]),
            "VWAP": _round(setup["vwap"]),
            "OR High": _round(setup["or_high"]),
            "PM High": _round(setup["pm_high"]),
            "Prior High": _round(setup["prior_high"]),
            "Change %": _round(setup["change_pct"]),
            "RS vs SPY": _round(setup["relative_strength"]),
            "Dollar Volume": setup["avg_dollar_volume"],
            "Day Score": score_daytrade(setup, float(rs_percentiles.get(ticker, 0.5))),
        })
    ranked = pd.DataFrame(rows, columns=DAYTRADE_COLUMNS)
    return ranked.sort_values(["Day Score", "RVOL"], ascending=False).head(limit).reset_index(drop=True)


OPEN_TRADE = "Open"
TIME_EXIT = "Time"
CLOSED_OUTCOMES = ("Target", "Stop", TIME_EXIT)
# A carried trade that has reached neither level by the close of this many trading days after the
# alert day is closed there, so capital isn't tied up in trades that went nowhere.
MAX_HOLD_DAYS = 5


def evaluate_daytrade(
    entry: float, stop: float, target: float, bars_after: pd.DataFrame, max_hold_days: int = MAX_HOLD_DAYS,
) -> dict:
    """Play an alerted trade forward through the regular-session bars since the alert, across days.

    The trade closes at the stop or the target, whichever the bars reach first (the stop when one bar
    spans both, to stay conservative). A bar that opens through a level, such as a gap the next morning,
    fills at that bar's open. With neither reached by the close of the max_hold_days-th trading day after
    the alert day, it closes there ("Time"). Until then it is "Open", marked at the last close.
    R is measured from the alert price.
    """
    risk = entry - stop
    if risk <= 0:
        return {"Outcome": "Invalid", "Exit": None, "R": None, "Exit Date": None}
    bars = bars_after if bars_after is not None else pd.DataFrame()
    outcome, exit_price, exit_date = OPEN_TRADE, entry, None
    days_seen = []
    for timestamp, bar in bars.iterrows():
        bar_open = float(bar["Open"]) if "Open" in bar and pd.notna(bar["Open"]) else None
        bar_date = timestamp.date().isoformat() if hasattr(timestamp, "date") else None
        if bar_date is not None and bar_date not in days_seen:
            if max_hold_days is not None and len(days_seen) > max_hold_days:
                outcome = TIME_EXIT   # the previous bar was the last close of the holding period
                break
            days_seen.append(bar_date)
        exit_date = bar_date
        if bar["Low"] <= stop:
            outcome, exit_price = "Stop", min(stop, bar_open) if bar_open is not None else stop
            break
        if bar["High"] >= target:
            outcome, exit_price = "Target", max(target, bar_open) if bar_open is not None else target
            break
        exit_price = float(bar["Close"])
    return {
        "Outcome": outcome, "Exit": round(float(exit_price), 2), "R": round((exit_price - entry) / risk, 2),
        "Exit Date": exit_date,
    }
