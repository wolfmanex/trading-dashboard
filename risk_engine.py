import math


def position_size(account_size: float, risk_pct: float, entry: float, stop: float) -> dict | None:
    """Shares to buy (or short) so that a stop-out loses risk_pct of the account.

    The position is capped at the account size (no margin), in which case the dollar risk is
    smaller than the budget. Returns None when the inputs can't produce a position.
    """
    values = (account_size, risk_pct, entry, stop)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return None
    if account_size <= 0 or risk_pct <= 0 or entry <= 0 or stop <= 0:
        return None
    risk_per_share = abs(entry - stop)
    if risk_per_share == 0:
        return None

    risk_budget = account_size * risk_pct / 100
    shares = math.floor(risk_budget / risk_per_share)
    capped = shares * entry > account_size
    if capped:
        shares = math.floor(account_size / entry)
    if shares <= 0:
        return None
    return {
        "shares": shares,
        "position_value": round(shares * entry, 2),
        "dollar_risk": round(shares * risk_per_share, 2),
        "risk_per_share": round(risk_per_share, 4),
        "capped_by_account": capped,
    }
