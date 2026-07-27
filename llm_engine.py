import os
import json
import pandas as pd
import streamlit as st
import google.generativeai as genai

# Initialize Google Generative AI Client
api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


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


def synthesize_signals(
    ticker: str, 
    df_5m: pd.DataFrame = None, 
    df_4h: pd.DataFrame = None, 
    df_1d: pd.DataFrame = None, 
    df_1w: pd.DataFrame = None,
    sentiment_summary: str = "",
    event_data: dict = None,
    options_data: dict = None,
    swing_metrics: dict = None,
    intraday_metrics: dict = None,
    analysis_mode: str = "Intra-Day (Scalp/Day Trade)"
) -> dict:
    """Synthesizes technical, macro, options, swing, and intraday metrics into an institutional trade decision."""
    
    if event_data is None:
        event_data = {
            "earnings_date": "N/A", 
            "days_until_earnings": "N/A", 
            "proximity_flag": "N/A", 
            "macro_vix": 0.0, 
            "macro_tnx": 0.0, 
            "news_headlines": []
        }

    if options_data is None:
        options_data = {"pcr_oi": "N/A", "call_wall": "N/A", "put_wall": "N/A", "positioning_summary": "N/A"}

    if swing_metrics is None:
        swing_metrics = {"expected_move_usd": "N/A", "upper_expected_bound": "N/A", "lower_expected_bound": "N/A", "sector_etf": "SPY", "relative_strength_1w": "N/A", "rs_rating": "N/A"}

    if intraday_metrics is None:
        intraday_metrics = {"vwap": "N/A", "pdh": "N/A", "pdl": "N/A", "pmh": "N/A", "pml": "N/A", "rvol": "N/A"}

    tech_5m = format_dataframe_summary(df_5m, "5m")
    tech_4h = format_dataframe_summary(df_4h, "4h")
    tech_1d = format_dataframe_summary(df_1d, "1d")
    tech_1w = format_dataframe_summary(df_1w, "1w") if df_1w is not None else "Timeframe [1w]: Omitted for Intraday Mode."

    headlines = event_data.get("news_headlines", [])
    headlines_str = "\n".join(headlines) if headlines else "No recent high-impact headlines."

    prompt = f"""
You are an institutional Lead Quantitative Strategist analyzing **{ticker}**.
Execution Strategy Horizon: **{analysis_mode}**

### 1. TECHNICAL INDICATORS MATRIX:
{tech_5m}
{tech_4h}
{tech_1d}
{tech_1w}

### 2. INTRADAY LIQUIDITY & SESSION LEVELS:
- Session VWAP: USD {intraday_metrics.get('vwap', 'N/A')}
- Relative Volume (RVOL): {intraday_metrics.get('rvol', 'N/A')}
- Prior Day High (PDH) / Low (PDL): USD {intraday_metrics.get('pdh', 'N/A')} / USD {intraday_metrics.get('pdl', 'N/A')}
- Pre-Market High (PMH) / Low (PML): USD {intraday_metrics.get('pmh', 'N/A')} / USD {intraday_metrics.get('pml', 'N/A')}

### 3. OPTIONS POSITIONING & WEEKLY EXPECTED MOVE:
- Put/Call Open Interest Ratio (PCR-OI): {options_data.get('pcr_oi', 'N/A')}
- Major Resistance (Call Wall Strike): USD {options_data.get('call_wall', 'N/A')}
- Major Support (Put Wall Strike): USD {options_data.get('put_wall', 'N/A')}
- **Weekly Expected Move:** +/- USD {swing_metrics.get('expected_move_usd', 'N/A')}
- **Expected Upper Bound:** USD {swing_metrics.get('upper_expected_bound', 'N/A')} | **Expected Lower Bound:** USD {swing_metrics.get('lower_expected_bound', 'N/A')}

### 4. MACRO, SECTOR RELATIVE STRENGTH & EVENT ENVIRONMENT:
- Sector Benchmark Used: {swing_metrics.get('sector_etf', 'SPY')}
- 1-Week Sector Relative Strength: {swing_metrics.get('relative_strength_1w', 'N/A')}% ({swing_metrics.get('rs_rating', 'N/A')})
- Market Volatility Index (VIX): {event_data.get('macro_vix', 'N/A')}
- 10-Year Treasury Yield (^TNX): {event_data.get('macro_tnx', 'N/A')}%
- Upcoming Corporate Earnings Date: {event_data.get('earnings_date', 'N/A')}
- Days Until Earnings: {event_data.get('days_until_earnings', 'N/A')}
- Earnings Proximity Flag: {event_data.get('proximity_flag', 'N/A')}

### 5. SENTIMENT & CATALYST HEADLINES:
- Overall Sentiment Summary: {sentiment_summary}
- News Headlines:
{headlines_str}

---

### STRATEGIC & FORMATTING INSTRUCTIONS:
- **Intra-Day Mode**: Prioritize Session VWAP, PDH/PDL sweeps, RVOL, 5m momentum, and tight stop losses. If RVOL is < 0.8, reduce confidence due to low liquidity. Avoid taking trades directly into PMH/PML resistance/support.
- **Weekly Mode**: Heavy emphasis on 1d/1w structural trend, options Call/Put Walls, Sector Relative Strength (RS), and Expected Move bounds. 
- **CRITICAL**: If your Take Profit or Stop Loss targets exceed the "Weekly Expected Move Bounds", explicitly state that the trade represents an outlier volatility bet.
- **MANDATORY INCLUSION**: You MUST explicitly analyze the Put/Call Open Interest Ratio (PCR-OI) and Options Strike Walls inside your `macro_analysis` or `higher_tf_breakdown` text fields. Do not omit this options data.
- **No Dollar Signs ($)**: NEVER use the '$' symbol in descriptive text explanations or reasoning fields, as it triggers UI rendering bugs. Write prices as numbers or USD.
- **Bullet Points**: Write all breakdown fields ('higher_tf_breakdown', 'intraday_tf_breakdown', 'macro_analysis', 'news_catalyst_analysis') as clean, bulleted markdown points starting with '- '.

🚨 CRITICAL PROXIMITY RULES (EARNINGS EVENT HORIZON) 🚨
You must inspect the `days_until_earnings` and `proximity_flag` fields:

1. IF `proximity_flag` == 'IMMEDIATE_BINARY_RISK' (0-2 Days):
   - RISK LEVEL: Must be set to 'HIGH' or 'EXTREME'.
   - SWING STRATEGY: Explicitly state that holding shares/naked options into the release is a BINARY GAMBLE. Recommend closing positions before the bell or hedging via options spreads (e.g., defined-risk structures).

2. IF `proximity_flag` == 'SWING_WINDOW_OVERLAP' (3-5 Days):
   - SWING STRATEGY: You MUST factor earnings into the trade duration. 
   - Mandatory note: State whether this setup is a "Pre-Earnings Run Up Play" (exiting BEFORE the report) or if the binary risk outweighs the technical setup.
   - IV CRUSH WARNING: Warn that IV expansion will inflate option premiums and IV crush will destroy post-earnings long options.

3. IF `proximity_flag` == 'OUTSIDE_SWING_WINDOW' (> 5 Days):
   - Standard technical/swing rules apply.

🚨 CRITICAL RULE: CATALYST COLLISION CHECK 🚨
Cross-reference the `earnings_date` with macro events and `news_headlines`. 
If a stock's earnings report or proximity window occurs near or on a major macroeconomic catalyst (e.g., FOMC Rate Decision, Fed decision, CPI release, NFP), you MUST:
1. Elevate overall risk assessment to HIGH or EXTREME.
2. Explicitly flag the "Macro/Micro Catalyst Collision" inside your `news_catalyst_analysis` field.
3. Adjust your execution plan to account for binary, multi-directional volatility (e.g., wider stop losses, defined-risk structures, or exiting before the catalyst).

🚨 CATALYST SCENARIO REASONING 🚨
If `days_until_earnings` is <= 5, or a major macro event is imminent, you MUST generate predictive scenarios in the `catalyst_scenarios` field:
1. Bull Case Reaction: What structural levels must break for a sustained rally? (Reference Call Walls and Resistance).
2. Bear Case Reaction: Where is the ultimate capitulation level if the catalyst fails? (Reference Put Walls and Support).
3. Tactical Play: Suggest the optimal institutional approach (e.g., "Wait for T+1 post-earnings drift", "Delta-neutral straddle", or "Close 80% of position pre-market").

Synthesize all data and output strictly a SINGLE valid JSON object matching this schema without any outer explanation or extra text:

{{
  "signal": "BUY",
  "confidence": 0.85,
  "timeframe_confluence": "Bullish Confluence Across 5m/1d holding above Session VWAP",
  "execution_plan": {{
    "entry_zone": "145.20 - 145.80",
    "take_profit": 152.00,
    "stop_loss": 142.50,
    "risk_reward_ratio": "1:2.3",
    "key_support": 142.00,
    "key_resistance": 153.50,
    "swing_upper_limit": 155.00,
    "swing_lower_limit": 138.00
  }},
  "higher_tf_breakdown": "- Daily structural analysis remains bullish above 21 EMA.\n- Put Wall at 140.00 USD provides strong institutional hedging floor.",
  "intraday_tf_breakdown": "- Price cleared PMH and is retesting Session VWAP as support.\n- RVOL at 1.4 confirms strong participation.",
  "macro_analysis": "- Options PCR-OI at 0.58 indicates strong smart money call accumulation.\n- Stock showing Strong Relative Strength vs XLK (+2.4%).",
  "news_catalyst_analysis": "- Upcoming earnings catalyst presents low immediate risk.",
  "catalyst_scenarios": "- Bull Case: Price gaps above 153.50 Call Wall, triggering a gamma squeeze toward 160.\n- Bear Case: Forward guidance misses, breaking 142 support and dropping to 138 swing limit.\n- Tactical Play: Do not hold directional calls through the bell due to IV Crush. Wait for T+1 morning settlement to trade the post-earnings drift.",
  "detailed_reasoning": "Comprehensive thesis unifying technicals, options OI walls, relative strength, VWAP levels, and expected move guidelines."
}}
"""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2}
        )
        
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        result_json = json.loads(clean_text, strict=False)
        return result_json

    except Exception as e:
        print(f"LLM Synthesis Engine Error: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "timeframe_confluence": "Error generating synthesis",
            "execution_plan": {
                "entry_zone": "N/A", "take_profit": 0.0, "stop_loss": 0.0,
                "risk_reward_ratio": "N/A", "key_support": 0.0, "key_resistance": 0.0,
                "swing_upper_limit": 0.0, "swing_lower_limit": 0.0
            },
            "higher_tf_breakdown": "- Higher timeframe synthesis unavailable.",
            "intraday_tf_breakdown": "- Intraday timeframe synthesis unavailable.",
            "macro_analysis": f"- API Exception: {str(e)}",
            "news_catalyst_analysis": "- N/A",
            "catalyst_scenarios": "- Scenario modeling unavailable due to API error.",
            "detailed_reasoning": f"An error occurred during AI analysis generation: {str(e)}"
        }

def generate_ai_analysis(ticker: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Insufficient data for AI analysis."
    data_summary = format_dataframe_summary(df, "Primary")
    prompt = f"Provide a brief 3-bullet technical breakdown for {ticker} based on:\n{data_summary}"
    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Analysis unavailable: {e}"