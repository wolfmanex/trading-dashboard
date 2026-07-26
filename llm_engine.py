import json
import os
import google.generativeai as genai
import pandas as pd
import streamlit as st

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

    close_price = last.get("Close", 0.0)
    ema_9 = last.get("EMA_9", 0.0)
    ema_21 = last.get("EMA_21", 0.0)
    rsi = last.get("RSI", 0.0)
    macd = last.get("MACD", 0.0)
    signal_line = last.get("Signal_Line", 0.0)
    bb_upper = last.get("BB_Upper", 0.0)
    bb_lower = last.get("BB_Lower", 0.0)

    trend = "Bullish" if ema_9 > ema_21 else "Bearish"
    momentum = "Bullish" if macd > signal_line else "Bearish"

    summary = f"""Timeframe [{tf_label}]:
    - Last Price: ${close_price:.2f} (Prev Close: ${prev.get('Close', 0.0):.2f})
    - EMA 9: ${ema_9:.2f} | EMA 21: ${ema_21:.2f} ({trend} Alignment)
    - RSI (14): {rsi:.1f}
    - MACD Line: {macd:.3f} | Signal Line: {signal_line:.3f} ({momentum} Momentum)
    - Bollinger Bands: Upper ${bb_upper:.2f} | Lower ${bb_lower:.2f}"""
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
    analysis_mode: str = "Intra-Day (Scalp/Day Trade)",
) -> dict:
    """Synthesizes technical indicators, macro environment, options smart-money metrics,

    and event catalysts into an institutional trade decision.
    """

    if event_data is None:
        event_data = {
            "earnings_date": "N/A",
            "macro_vix": 0.0,
            "macro_tnx": 0.0,
            "news_headlines": [],
        }

    if options_data is None:
        options_data = {
            "pcr_oi": "N/A",
            "call_wall": "N/A",
            "put_wall": "N/A",
            "positioning_summary": "Options positioning data unavailable.",
        }

    # 1. Format Technical Data Summaries Across Horizons
    tech_5m = format_dataframe_summary(df_5m, "5m")
    tech_4h = format_dataframe_summary(df_4h, "4h")
    tech_1d = format_dataframe_summary(df_1d, "1d")
    tech_1w = (
        format_dataframe_summary(df_1w, "1w")
        if df_1w is not None
        else "Timeframe [1w]: Omitted for Intraday Mode."
    )

    # 2. Format News & Catalyst Headings
    headlines = event_data.get("news_headlines", [])
    headlines_str = (
        "\n".join(headlines) if headlines else "No recent high-impact headlines."
    )

    # 3. Construct Institutional AI Prompt
    prompt = f"""
You are an institutional Lead Quantitative Strategist analyzing **{ticker}**.
Execution Strategy Horizon: **{analysis_mode}**

### 1. TECHNICAL INDICATORS MATRIX:
{tech_5m}
{tech_4h}
{tech_1d}
{tech_1w}

### 2. OPTIONS OPEN INTEREST & SMART MONEY POSITIONING:
- Put/Call Open Interest Ratio (PCR-OI): {options_data.get('pcr_oi', 'N/A')}
- Major Resistance (Call Wall Strike): USD {options_data.get('call_wall', 'N/A')}
- Major Support (Put Wall Strike): USD {options_data.get('put_wall', 'N/A')}
- Options Chain Expiration Date: {options_data.get('expiration', 'N/A')}
- Smart Money Summary: {options_data.get('positioning_summary', 'N/A')}

### 3. MACRO & EVENT ENVIRONMENT:
- Market Volatility Index (VIX): {event_data.get('macro_vix', 'N/A')}
- 10-Year Treasury Yield (^TNX): {event_data.get('macro_tnx', 'N/A')}%
- Upcoming Corporate Earnings Date: {event_data.get('earnings_date', 'N/A')}

### 4. SENTIMENT & CATALYST HEADLINES:
- Overall Sentiment Summary: {sentiment_summary}
- News Headlines:
{headlines_str}

---

### STRATEGIC & FORMATTING INSTRUCTIONS:
- **Intra-Day Mode**: Prioritize 5m/1h momentum, micro EMA crossovers, and tight stop losses. Warn explicitly if earnings are within 24 hours.
- **Weekly Mode**: Heavy emphasis on 1d/1w structural trend, options Call/Put Walls as hard magnetic boundaries, PCR-OI positioning bias, VIX climate, and multi-week target setting.
- **No Dollar Signs ($)**: NEVER use the '$' symbol in descriptive text explanations or reasoning fields, as it triggers UI rendering bugs. Write prices as numbers or USD (e.g., "166.50" or "166.50 USD").
- **Bullet Points**: Write all breakdown fields ('higher_tf_breakdown', 'intraday_tf_breakdown', 'macro_analysis', 'news_catalyst_analysis') as clean, bulleted markdown points starting with '- '. Keep them concise and scannable.

Synthesize all data and output strictly a SINGLE valid JSON object matching this schema without any outer explanation or extra text:

{{
  "signal": "BUY",
  "confidence": 0.85,
  "timeframe_confluence": "Bullish Confluence Across 5m/1d with Call Wall Clearance",
  "execution_plan": {{
    "entry_zone": "145.20 - 145.80",
    "take_profit": 152.00,
    "stop_loss": 142.50,
    "risk_reward_ratio": "1:2.3",
    "key_support": 142.00,
    "key_resistance": 153.50
  }},
  "higher_tf_breakdown": "- Daily and weekly structural analysis remains bullish above 21 EMA.\n- Put Wall at 140.00 USD provides strong institutional hedging floor.",
  "intraday_tf_breakdown": "- Short-term momentum shows oversold RSI turning upward.\n- Immediate entry timing favored on 5m EMA crossover.",
  "macro_analysis": "- Options PCR-OI at 0.58 indicates strong smart money call accumulation.\n- VIX volatility risk remains muted under 18.",
  "news_catalyst_analysis": "- Upcoming earnings catalyst presents low immediate risk.\n- Headline sentiment leans slightly positive.",
  "detailed_reasoning": "Comprehensive thesis unifying technicals, options OI walls, macro context, and risk management guidelines."
}}
"""

    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(
            prompt, generation_config={"temperature": 0.2}
        )

        clean_text = (
            response.text.replace("```json", "").replace("```", "").strip()
        )
        result_json = json.loads(clean_text)
        return result_json

    except Exception as e:
        print(f"LLM Synthesis Engine Error: {e}")
        return {
            "signal": "HOLD",
            "confidence": 0.0,
            "timeframe_confluence": "Error generating synthesis",
            "execution_plan": {
                "entry_zone": "N/A",
                "take_profit": 0.0,
                "stop_loss": 0.0,
                "risk_reward_ratio": "N/A",
                "key_support": 0.0,
                "key_resistance": 0.0,
            },
            "higher_tf_breakdown": "- Higher timeframe synthesis unavailable.",
            "intraday_tf_breakdown": "- Intraday timeframe synthesis unavailable.",
            "macro_analysis": f"- API Exception: {str(e)}",
            "news_catalyst_analysis": "- N/A",
            "detailed_reasoning": f"An error occurred during AI analysis generation: {str(e)}",
        }


def generate_ai_analysis(ticker: str, df: pd.DataFrame) -> str:
    """Legacy helper function maintained for backward compatibility."""
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