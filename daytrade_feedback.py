"""Feedback loop for the large-cap day trades: what past alerts did shapes how new ones are ranked.

Every closed alert (stopped out or at target) is bucketed by what was known when it was sent: time of
day, relative volume, day score, how far it was already through its trigger, the trigger type and how
SPY was doing. For each bucket the average R is shrunk toward the overall average with PRIOR_TRADES
imaginary trades, so a bucket with five trades barely moves anything and one with a hundred speaks for
itself. A new setup's score is then nudged by how much better or worse than average its buckets have
done, by at most MAX_ADJUSTMENT points, and only once MIN_TRADES alerts have closed. The nudged score
is what the Telegram alerts rank by and must clear MIN_SCORE with, so weak buckets drop out and strong
ones move up without hard-coded rules that a small sample could get wrong.
"""
import numpy as np
import pandas as pd

from daytrade_engine import CLOSED_OUTCOMES


MIN_TRADES = 30          # below this the whole record is noise; no adjustment at all
PRIOR_TRADES = 20        # shrinkage: each bucket starts as 20 trades at the overall average
POINTS_PER_R = 25        # 0.1R better than average across a setup's buckets is worth 2.5 score points
MAX_ADJUSTMENT = 10.0    # never move a score by more than this

# (dimension, column, upper bucket edges, labels). Values come from the alert row or the scan row.
DIMENSIONS = [
    ("Time of day", "Minute", [11 * 60, 13 * 60, 24 * 60], ["10:00-10:59", "11:00-12:59", "13:00 on"]),
    ("Relative volume", "RVOL", [2, 3, np.inf], ["under 2x", "2-3x", "3x and up"]),
    ("Day score", "Day Score", [70, 80, np.inf], ["under 70", "70-79", "80 and up"]),
    ("Through trigger", "Extension %", [0.25, 0.75, np.inf], ["under 0.25%", "0.25-0.75%", "over 0.75%"]),
    ("SPY on the day", "SPY %", [-0.3, 0.3, np.inf], ["down over 0.3%", "flat", "up over 0.3%"]),
]
CATEGORICAL = [("Trigger level", "Trigger Level")]


def _minute(value) -> float:
    try:
        hour, minute = str(value).split(":")[:2]
        return int(hour) * 60 + int(minute)
    except (ValueError, AttributeError):
        return np.nan


def features(rows: pd.DataFrame, now_minute: int = None) -> pd.DataFrame:
    """The bucketing inputs from either alert-log rows or scan rows (which have no Time; pass now_minute)."""
    frame = pd.DataFrame(index=rows.index)
    if "Time" in rows:
        frame["Minute"] = rows["Time"].map(_minute)
    else:
        frame["Minute"] = float(now_minute) if now_minute is not None else np.nan
    for column in ("RVOL", "Day Score", "SPY %"):
        frame[column] = pd.to_numeric(rows[column], errors="coerce") if column in rows else np.nan
    price = pd.to_numeric(rows.get("Price"), errors="coerce")
    trigger = pd.to_numeric(rows.get("Trigger"), errors="coerce")
    frame["Extension %"] = (price / trigger - 1) * 100
    frame["Trigger Level"] = rows["Trigger Level"] if "Trigger Level" in rows else None
    return frame


def _bucket(values: pd.Series, edges: list, labels: list) -> pd.Series:
    return pd.cut(values, [-np.inf, *edges], labels=labels, right=False).astype(object)


def bucketed(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for name, column, edges, labels in DIMENSIONS:
        out[name] = _bucket(frame[column], edges, labels)
    for name, column in CATEGORICAL:
        out[name] = frame[column].where(frame[column].notna() & (frame[column].astype(str) != ""), None)
    return out


def closed_trades(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts is None or alerts.empty:
        return pd.DataFrame()
    r = pd.to_numeric(alerts["R"], errors="coerce")
    return alerts[alerts["Outcome"].isin(CLOSED_OUTCOMES) & r.notna()]


def build_feedback(alerts: pd.DataFrame) -> dict:
    """Per-bucket record of the closed alerts: {"trades", "average_r", "active", "table"}."""
    closed = closed_trades(alerts)
    columns = ["Dimension", "Bucket", "Trades", "Win Rate %", "Average R", "Shrunk R", "Edge R", "Points"]
    if closed.empty:
        return {"trades": 0, "average_r": None, "active": False, "table": pd.DataFrame(columns=columns)}
    r = pd.to_numeric(closed["R"], errors="coerce")
    overall = float(r.mean())
    active = len(closed) >= MIN_TRADES
    buckets = bucketed(features(closed))
    rows = []
    for name in buckets.columns:
        for bucket, group in r.groupby(buckets[name]):
            shrunk = (group.sum() + PRIOR_TRADES * overall) / (len(group) + PRIOR_TRADES)
            rows.append({
                "Dimension": name, "Bucket": bucket, "Trades": int(len(group)),
                "Win Rate %": round(float((group > 0).mean() * 100), 1), "Average R": round(float(group.mean()), 2),
                "Shrunk R": round(float(shrunk), 3), "Edge R": round(float(shrunk - overall), 3),
                "Points": round(float(np.clip((shrunk - overall) * POINTS_PER_R, -MAX_ADJUSTMENT, MAX_ADJUSTMENT)), 1)
                if active else 0.0,
            })
    return {
        "trades": int(len(closed)), "average_r": round(overall, 2), "active": active,
        "table": pd.DataFrame(rows, columns=columns),
    }


def apply_feedback(results: pd.DataFrame, feedback: dict, now_minute: int = None, spy_change: float = None) -> pd.DataFrame:
    """Add "History Adj", "Adj Score" and "History Note" to scan results; unchanged scores until active."""
    results = results.copy()
    if results.empty:
        for column in ("History Adj", "Adj Score", "History Note"):
            results[column] = pd.Series(dtype=object)
        return results
    scan_rows = results.copy()
    if "SPY %" not in scan_rows and spy_change is not None:
        scan_rows["SPY %"] = spy_change
    buckets = bucketed(features(scan_rows, now_minute))
    table = feedback.get("table", pd.DataFrame())
    edges = {(row["Dimension"], row["Bucket"]): row["Edge R"] for _, row in table.iterrows()} if not table.empty else {}
    adjustments, notes = [], []
    for index in results.index:
        contributions = {
            name: edges[(name, buckets.at[index, name])]
            for name in buckets.columns if (name, buckets.at[index, name]) in edges
        }
        points = float(np.clip(sum(contributions.values()) * POINTS_PER_R, -MAX_ADJUSTMENT, MAX_ADJUSTMENT))
        if not feedback.get("active"):
            points, contributions = 0.0, {}
        adjustments.append(round(points, 1))
        if contributions and abs(points) >= 0.5:
            name = max(contributions, key=lambda key: abs(contributions[key]))
            notes.append(f"{name.lower()} {buckets.at[index, name]}: {contributions[name]:+.2f}R vs average")
        else:
            notes.append("")
    results["History Adj"] = adjustments
    results["Adj Score"] = (pd.to_numeric(results["Day Score"], errors="coerce") + results["History Adj"]).round(1)
    results["History Note"] = notes
    return results


def feedback_line(feedback: dict) -> str:
    """One line on whether past alerts are shaping the ranking yet."""
    if not feedback["trades"]:
        return "No closed day trades yet, so past results don't change the ranking."
    if not feedback["active"]:
        return (f"{feedback['trades']} closed day trades; the track record starts adjusting scores at {MIN_TRADES}.")
    return (f"Scores adjusted by up to {MAX_ADJUSTMENT:.0f} points from {feedback['trades']} closed day trades "
            f"(average {feedback['average_r']:+.2f}R).")


def feedback_summary(feedback: dict, max_rows: int = 12) -> str:
    """Plain-text track record for the AI prompt: overall, then the buckets furthest from average."""
    if not feedback["trades"]:
        return ""
    lines = [f"{feedback['trades']} closed alerts, average {feedback['average_r']:+.2f}R per trade."]
    table = feedback["table"]
    if not table.empty:
        table = table.assign(distance=table["Edge R"].abs()).sort_values("distance", ascending=False).head(max_rows)
        for _, row in table.iterrows():
            lines.append(
                f"- {row['Dimension']} {row['Bucket']}: {row['Trades']} trades, {row['Win Rate %']:.0f}% winners, "
                f"{row['Average R']:+.2f}R average"
            )
    return "\n".join(lines)
