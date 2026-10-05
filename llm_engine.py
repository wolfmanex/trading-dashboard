import json
import pandas as pd
from google import genai

from secrets_config import get_configured_secret
from trade_levels import STOP_ATR, TARGET_R, format_recent_bars, horizon


# Initialize Google Generative AI Client
api_key = get_configured_secret("GEMINI_API_KEY")
# "gemini-flash-latest" follows Google's newest Flash model, so output can change without a code
# change; set GEMINI_MODEL in secrets (e.g. a dated model name) to pin it.
GEMINI_MODEL = get_configured_secret("GEMINI_MODEL", "gemini-flash-latest")
gemini_client = genai.Client(api_key=api_key) if api_key else None


def is_ai_configured() -> bool:
    return gemini_client is not None


def format_dataframe_summary(df: pd.DataFrame, tf_label: str = "5m") -> str:
    """Extracts key latest technical metrics from a DataFrame into readable text for the LLM."""
    if df is None or df.empty or len(df) < 5:
        return f"Timeframe [{tf_label}]: No sufficient technical data available."
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    close_price = last.get('Close', 0.0)
    ema_9 = last.get('EMA_9', 0.0)
    ema_21 = last.get('EMA_21', 0.0)
    rsi = last.get('RSI', 0.0)
    macd = last.get('MACD', 0.0)
    signal_line = last.get('Signal_Line', 0.0)
    bb_upper = last.get('BB_Upper', 0.0)
    bb_lower = last.get('BB_Lower', 0.0)
    
    trend = "Bullish" if ema_9 > ema_21 else "Bearish"
    momentum = "Bullish" if macd > signal_line else "Bearish"
    
    summary = f"""Timeframe [{tf_label}]:
    - Last Price: {close_price:.2f} USD (Prev Close: {prev.get('Close', 0.0):.2f} USD)
    - EMA 9: {ema_9:.2f} USD | EMA 21: {ema_21:.2f} USD ({trend} Alignment)
    - RSI (14): {rsi:.1f}
    - MACD Line: {macd:.3f} | Signal Line: {signal_line:.3f} ({momentum} Momentum)
    - Bollinger Bands: Upper {bb_upper:.2f} USD | Lower {bb_lower:.2f} USD"""
    return summary


def generate_json(prompt: str):
    """Send a prompt in JSON mode and parse the reply; raises on a missing key or unparseable output."""
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.2, "response_mime_type": "application/json"},
    )
    clean_text = (response.text or "").replace("```json", "").replace("```", "").strip()
    if not clean_text:
        raise ValueError("The model returned an empty response")
    return json.loads(clean_text, strict=False)


def generate_text(prompt: str) -> str:
    """Send a prompt and return the plain-text reply; raises on a missing key or an empty reply."""
    if gemini_client is None:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config={"temperature": 0.3})
    text = (response.text or "").strip()
    if not text:
        raise ValueError("The model returned an empty response")
    return text


TRADE_VERDICTS = {"GO", "WAIT", "SKIP"}
TRADE_DIRECTIONS = {"LONG", "SHORT", "NONE"}
TRADE_GRADES = {"A", "B", "C"}
RISK_LEVELS = {"LOW", "MEDIUM", "HIGH", "EXTREME"}


def _bullets(value) -> list:
    """Accept a list or a newline/bullet string and return clean bullet texts."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.splitlines()
    else:
        items = []
    cleaned = [str(item).strip().lstrip("-*• ").strip() for item in items]
    return [item for item in cleaned if item]


def validate_trade_review(result: dict, levels: dict) -> dict:
    """Validate and normalize the reviewer's JSON against the plans that were actually offered."""
    if not isinstance(result, dict):
        raise ValueError("AI response must be a JSON object")

    direction = str(result.get("direction", "")).strip().upper()
    verdict = str(result.get("verdict", "")).strip().upper()
    grade = str(result.get("grade", "")).strip().upper()
    if direction not in TRADE_DIRECTIONS:
        raise ValueError(f"Unsupported AI direction: {direction or 'missing'}")
    if verdict not in TRADE_VERDICTS:
        raise ValueError(f"Unsupported AI verdict: {verdict or 'missing'}")
    if grade not in TRADE_GRADES:
        raise ValueError(f"Unsupported AI grade: {grade or 'missing'}")

    notes = []
    if direction != "NONE" and not levels.get(direction.lower()):
        notes.append(f"The reviewer chose {direction}, but no {direction} plan was offered, so it is shown as no trade.")
        direction = "NONE"
    if direction == "NONE" and verdict == "GO":
        verdict = "SKIP"

    risk = str(result.get("risk_level", "")).strip().upper()
    return {
        "direction": direction,
        "verdict": verdict,
        "grade": grade,
        "risk_level": risk if risk in RISK_LEVELS else "HIGH",
        "summary": str(result.get("summary", "")).strip(),
        "trigger": str(result.get("trigger", "")).strip(),
        "technical_notes": _bullets(result.get("technical_notes")),
        "catalyst_notes": _bullets(result.get("catalyst_notes")),
        "risks": _bullets(result.get("risks")),
        "scenarios": _bullets(result.get("scenarios")),
        "validation_notes": notes,
    }


def format_plan(name: str, plan: dict) -> str:
    if not plan:
        return f"- {name}: not offered."
    return (
        f"- {name}: entry {plan['entry']:.2f} | stop {plan['stop']:.2f} | target {plan['target']:.2f} "
        f"| reward/risk {plan['reward_risk']}:1"
    )


def format_scanner_context(scanner_row: dict, context: dict) -> str:
    if not scanner_row:
        return "Not from the breakout scanner."
    context = context or {}
    return (
        f"- Found by the small-cap breakout scanner: {scanner_row.get('Base Weeks', 'N/A')}-week base, "
        f"{scanner_row.get('Base Depth', 'N/A')}% deep, latest RVOL {scanner_row.get('RVOL', 'N/A')}, "
        f"3-month RS vs IWM {scanner_row.get('RS vs IWM', 'N/A')}%, breakout score {scanner_row.get('Breakout Score', 'N/A')}/100\n"
        f"- Market cap: {scanner_row.get('Market Cap', 'N/A')} USD | Float: {context.get('float_shares', 'N/A')} shares "
        f"| Short % of float: {context.get('short_pct_float', 'N/A')} | Insider %: {context.get('insider_pct', 'N/A')}\n"
        f"- Sector / Industry: {context.get('sector', 'N/A')} / {context.get('industry', 'N/A')}"
    )


def review_trade_setup(
    ticker: str,
    price: float,
    levels: dict,
    analysis_mode: str,
    df_5m: pd.DataFrame = None,
    df_4h: pd.DataFrame = None,
    df_1d: pd.DataFrame = None,
    df_1w: pd.DataFrame = None,
    event_data: dict = None,
    options_data: dict = None,
    swing_metrics: dict = None,
    intraday_metrics: dict = None,
    scanner_row: dict = None,
    scanner_context: dict = None,
    market_note: str = "",
) -> dict:
    """Ask the model to judge a trade whose levels were calculated in code (see trade_levels.py).

    Returns {"review": dict | None, "error": str | None, "prompt": str}. The model picks a direction among
    the offered plans and grades the setup; it never sets prices. The prompt is returned so follow-up
    questions (ask_follow_up) see the same data.
    """
    event_data = event_data or {}
    options_data = options_data or {}
    swing_metrics = swing_metrics or {}
    intraday_metrics = intraday_metrics or {}
    is_intraday = "Intra-Day" in str(analysis_mode)

    headlines = event_data.get("news_headlines") or (scanner_context or {}).get("headlines") or []
    headlines_str = "\n".join(f"  {headline}" for headline in headlines) or "  No recent headlines."
    timeframes = [format_dataframe_summary(df_1d, "1d"), format_dataframe_summary(df_4h, "4h")]
    if is_intraday:
        timeframes.insert(0, format_dataframe_summary(df_5m, "5m"))
    else:
        timeframes.append(format_dataframe_summary(df_1w, "1w"))
    recent_bars = [format_recent_bars(df_1d, 20, "Last 20 daily bars")]
    if is_intraday:
        recent_bars.append(format_recent_bars(df_5m, 24, "Last 24 five-minute bars"))
    atr = levels.get("atr")
    level_source = (
        "the breakout scanner (pivot buy-stop, base stop, measured-move target; long only)"
        if levels.get("source") == "scanner" else
        f"the daily ATR ({atr} USD): stop {STOP_ATR[horizon(analysis_mode)]} ATR from the current price, "
        f"target {TARGET_R:.0f}R"
    )

    prompt = f"""
You are reviewing a possible trade in **{ticker}** for an individual trader. Horizon: **{analysis_mode}**.
Current price: {price:.2f} USD.

The trade levels below were calculated in code from {level_source}. Do not change them or invent other
prices. Your job is to judge whether either plan is worth taking now, should wait for a trigger, or
should be skipped, and to point out what the numbers cannot show.

### TRADE PLANS
{format_plan("LONG", levels.get("long"))}
{format_plan("SHORT", levels.get("short"))}
- Recent 20-day support (lowest low): {levels.get("support", "N/A")} | resistance (highest high): {levels.get("resistance", "N/A")}
- Daily ATR(14): {atr if atr is not None else "N/A"} USD

### PRICE ACTION
{chr(10).join(recent_bars)}

### INDICATORS
{chr(10).join(timeframes)}

### SESSION LEVELS
- VWAP {intraday_metrics.get("vwap", "N/A")} | RVOL {intraday_metrics.get("rvol", "N/A")}
- Prior day high/low {intraday_metrics.get("pdh", "N/A")} / {intraday_metrics.get("pdl", "N/A")}
- Pre-market high/low {intraday_metrics.get("pmh", "N/A")} / {intraday_metrics.get("pml", "N/A")}

### SCANNER CONTEXT
{format_scanner_context(scanner_row, scanner_context)}

### CONTEXT
- Relative strength vs {swing_metrics.get("sector_etf", "SPY")} over 1 week: {swing_metrics.get("relative_strength_1w", "N/A")}% ({swing_metrics.get("rs_rating", "N/A")})
- Options-implied weekly expected move: +/- {swing_metrics.get("expected_move_usd", "N/A")} USD
- Options put/call OI ratio {options_data.get("pcr_oi", "N/A")}, call wall {options_data.get("call_wall", "N/A")}, put wall {options_data.get("put_wall", "N/A")}
- VIX {event_data.get("macro_vix", "N/A")} | US 10-year yield {event_data.get("macro_tnx", "N/A")}%
- {market_note or "Market conditions and the macro calendar are unavailable."}
- Next earnings: {event_data.get("earnings_date", "N/A")} ({event_data.get("days_until_earnings", "N/A")} days away)
- Today: {pd.Timestamp.today().strftime("%Y-%m-%d")}
- Headlines:
{headlines_str}

### HOW TO JUDGE
- Pick the direction that fits the trend and structure, or NONE if neither plan has an edge. Only choose a
  direction whose plan is offered above.
- Verdict GO: the setup is valid now. WAIT: valid but needs a trigger first (say exactly what in `trigger`,
  e.g. "a close above 12.40 on volume above the 20-day average"). SKIP: no edge or the risk is not worth it.
- Grade A: clean setup, stop below real structure, supportive trend and catalyst. B: valid with minor concerns.
  C: undermined by structure, trend, liquidity or event risk.
- Check whether the stop sits beyond the recent support/resistance or inside the noise (compare with ATR),
  and whether the target runs into nearby resistance/support.
- Earnings within 5 days are binary risk: say so, and fill `scenarios` with a bull case, a bear case and how
  to handle the position. Otherwise leave `scenarios` empty.
- Options data is often missing or thin for small caps; mention it only when it is present and meaningful.
- Use only the macro dates listed under CONTEXT; never state FOMC, CPI or other release dates from memory. A
  Risk-off market or a major release within two days counts against a fresh entry.
- If information is missing, say so instead of guessing. Never use the '$' symbol; write prices as numbers.

Output strictly one JSON object:
{{
  "direction": "LONG",
  "verdict": "WAIT",
  "grade": "B",
  "risk_level": "MEDIUM",
  "summary": "One sentence verdict a trader can act on.",
  "trigger": "What must happen before entering, or what confirms the entry.",
  "technical_notes": ["Short bullet on trend, structure, stop and target placement"],
  "catalyst_notes": ["Short bullet on news, earnings or the lack of a catalyst"],
  "risks": ["Each concrete risk as a short phrase"],
  "scenarios": []
}}
"""

    try:
        review = validate_trade_review(generate_json(prompt), levels)
        return {"review": review, "error": None, "prompt": prompt}
    except Exception as e:
        print(f"Trade Review Engine Error: {e}")
        return {"review": None, "error": str(e), "prompt": prompt}


def ask_follow_up(review_prompt: str, review: dict, history: list, question: str) -> dict:
    """Answer a question about a finished trade review, with the review's data and the conversation so far.

    `history` is a list of {"question", "answer"}. Returns {"answer": str | None, "error": str | None}.
    """
    question = str(question or "").strip()
    if not question:
        return {"answer": None, "error": "Ask a question first."}
    conversation = "\n\n".join(f"Trader: {turn['question']}\nYou: {turn['answer']}" for turn in history[-6:])
    prompt = f"""
Earlier you reviewed a trade from the data below and gave the verdict that follows it. The trader now has
a follow-up question. Answer it directly in at most about 150 words of plain text (no JSON, no headings).
Use only the data below and your review. The entry, stop and target were calculated in code; you may
discuss other levels only if they appear in the data (e.g. support, resistance, the recent bars), and say
which. If the data can't answer the question, say so. Never use the '$' symbol.

=== DATA AND INSTRUCTIONS YOU REVIEWED ===
{review_prompt}

=== YOUR REVIEW ===
{json.dumps(review, ensure_ascii=False)}

=== CONVERSATION SO FAR ===
{conversation or "None yet."}

Trader: {question}
You:"""
    try:
        return {"answer": generate_text(prompt), "error": None}
    except Exception as e:
        print(f"Follow-up Engine Error: {e}")
        return {"answer": None, "error": str(e)}


BREAKOUT_GRADES = {"A", "B", "C"}
BREAKOUT_RISK_LEVELS = RISK_LEVELS


def validate_breakout_reviews(result: dict, candidate_tickers) -> dict:
    """Keep only well-formed reviews for tickers that were actually sent to the model."""
    if not isinstance(result, dict) or not isinstance(result.get("reviews"), list):
        raise ValueError("AI breakout response must contain a 'reviews' list")

    allowed = {str(ticker).upper() for ticker in candidate_tickers}
    reviews = {}
    for review in result["reviews"]:
        if not isinstance(review, dict):
            continue
        ticker = str(review.get("ticker", "")).strip().upper()
        grade = str(review.get("grade", "")).strip().upper()
        risk = str(review.get("risk_level", "")).strip().upper()
        if ticker not in allowed or grade not in BREAKOUT_GRADES:
            continue
        red_flags = review.get("red_flags", [])
        if isinstance(red_flags, str):
            red_flags = [red_flags] if red_flags.strip() else []
        elif not isinstance(red_flags, list):
            red_flags = []
        reviews[ticker] = {
            "grade": grade,
            "risk_level": risk if risk in BREAKOUT_RISK_LEVELS else "HIGH",
            "catalyst": str(review.get("catalyst", "")).strip() or "None identified",
            "thesis": str(review.get("thesis", "")).strip(),
            "red_flags": [str(flag).strip() for flag in red_flags if str(flag).strip()],
        }
    return reviews


def format_breakout_candidate(candidate: dict, context: dict) -> str:
    headlines = context.get("headlines") or []
    headlines_str = "\n".join(f"    {headline}" for headline in headlines) or "    No recent headlines."
    to_pivot = candidate["To Pivot"]
    pivot_position = f"{to_pivot}% above price" if to_pivot >= 0 else f"price already {abs(to_pivot)}% above it"
    return f"""#### {candidate['Ticker']} ({candidate.get('Name', '')})
- Sector / Industry: {context.get('sector', 'N/A')} / {context.get('industry', 'N/A')}
- Price: {candidate['Price']} USD | Market Cap: {candidate.get('Market Cap', 'N/A')} USD
- Pivot (buy-stop): {candidate['Pivot']} USD, {pivot_position}
- Stop: {candidate['Stop']} USD | Measured-move Target: {candidate['Target']} USD | Reward/Risk: {candidate['Reward/Risk']}
- Base: {candidate['Base Weeks']} weeks, {candidate['Base Depth']}% deep | Latest RVOL: {candidate['RVOL']}
- 3-month Relative Strength vs IWM: {candidate['RS vs IWM']}% | Breakout Score: {candidate['Breakout Score']}/100
- Float: {context.get('float_shares', 'N/A')} shares | Short % of Float: {context.get('short_pct_float', 'N/A')} | Insider %: {context.get('insider_pct', 'N/A')}
- Next Earnings Date: {context.get('earnings_date', 'N/A')} (today is {pd.Timestamp.today().strftime('%Y-%m-%d')})
- Recent Headlines:
{headlines_str}"""


def review_breakout_candidates(candidates: list, contexts: dict, market_note: str = "") -> dict:
    """Ask the model to grade pre-computed small-cap breakout setups and surface catalysts and red flags.

    Returns {"reviews": {ticker: review}, "error": str | None}.
    """
    if not candidates:
        return {"reviews": {}, "error": None}

    candidate_blocks = "\n\n".join(
        format_breakout_candidate(candidate, contexts.get(candidate["Ticker"], {}))
        for candidate in candidates
    )
    prompt = f"""
You are a small-cap equity analyst reviewing breakout setups that a quantitative screener has already found.
The pivot, stop, target and reward/risk below were calculated in code from daily price data. Do not recompute
or invent price levels. Your job is to judge the quality of each setup and the risks the numbers cannot show.

For each candidate assess:
1. Catalyst: is there a fundamental or news reason the stock could break out (earnings momentum, contract,
   FDA decision, guidance raise, sector rotation)? Use only the headlines and data provided.
2. Red flags typical of small caps: share offerings or dilution (ATM programs, S-3 shelves, convertible notes),
   reverse splits, going-concern or delisting notices, promotional or pump-style news, very low float combined
   with high short interest (squeeze-driven, unstable), or earnings falling within the next 10 trading days.
3. Grade: A = clean setup with a supportive catalyst and no material red flags; B = valid setup with minor
   concerns or no clear catalyst; C = setup undermined by red flags or binary event risk.
4. Risk level: LOW, MEDIUM, HIGH or EXTREME.

If information is missing, say so instead of guessing. Never use the '$' symbol; write prices as numbers or USD.

### MARKET
{market_note or "Market conditions unavailable."}
A Risk-off market or a major release (FOMC, CPI, jobs report) within two days makes any breakout less reliable;
factor that into the grade and name it as a red flag when it applies.

### CANDIDATES
{candidate_blocks}

Output strictly one JSON object, with one review per candidate, matching this schema:
{{
  "reviews": [
    {{
      "ticker": "ABCD",
      "grade": "B",
      "risk_level": "HIGH",
      "catalyst": "Short phrase naming the catalyst, or 'None identified'",
      "thesis": "Two or three sentences on why the setup is or is not worth watching.",
      "red_flags": ["Each concrete concern as a short phrase"]
    }}
  ]
}}
"""

    try:
        reviews = validate_breakout_reviews(generate_json(prompt), [c["Ticker"] for c in candidates])
        return {"reviews": reviews, "error": None}
    except Exception as e:
        print(f"Breakout Review Engine Error: {e}")
        return {"reviews": {}, "error": str(e)}
