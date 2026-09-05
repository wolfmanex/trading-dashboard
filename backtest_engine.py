import pandas as pd

from bot import calculate_ta_score


def run_ta_backtest(
    df: pd.DataFrame,
    holding_period: int = 5,
    cost_per_trade_pct: float = 0.0,
) -> dict:
    """Evaluate the deterministic TA score with next-bar entry and fixed exit."""
    result = {
        "total_trades": 0,
        "win_rate_pct": "N/A",
        "average_return_pct": "N/A",
        "cumulative_return_pct": "N/A",
        "max_drawdown_pct": "N/A",
    }

    if df is None or df.empty or holding_period < 1 or cost_per_trade_pct < 0:
        return result

    required_columns = {"Close", "EMA_9", "EMA_21", "MACD", "Signal_Line", "RSI"}
    if not required_columns.issubset(df.columns) or len(df) <= holding_period:
        return result

    returns = []
    for index in range(len(df) - holding_period):
        signal_row = df.iloc[index]
        if signal_row[list(required_columns)].isna().any():
            continue

        score = calculate_ta_score(df.iloc[[index]])
        if abs(score) < 50:
            continue

        entry_price = float(df["Close"].iloc[index + 1])
        exit_price = float(df["Close"].iloc[index + holding_period])
        if entry_price <= 0 or exit_price <= 0:
            continue

        direction = 1 if score >= 50 else -1
        gross_return = direction * ((exit_price - entry_price) / entry_price)
        net_return = gross_return - (cost_per_trade_pct / 100)
        returns.append(net_return)

    if not returns:
        return result

    returns_series = pd.Series(returns)
    equity = (1 + returns_series).cumprod()
    drawdown = (equity / equity.cummax()) - 1

    result.update({
        "total_trades": len(returns),
        "win_rate_pct": round(float((returns_series > 0).mean() * 100), 2),
        "average_return_pct": round(float(returns_series.mean() * 100), 2),
        "cumulative_return_pct": round(float((equity.iloc[-1] - 1) * 100), 2),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
    })
    return result