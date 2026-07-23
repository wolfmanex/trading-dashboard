import json
import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
import yfinance as yf

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

SYSTEM_PROMPT = """
You are an elite quantitative multi-factor trading AI. You perform institutional-grade analysis by evaluating Macro Market Regimes, News Catalysts, and Multi-Timeframe Technical Structures (Daily/4H vs. 5M Intraday).

You must analyze all provided metrics and output a rich, actionable trade plan strictly following this JSON schema:

{
  "signal": "BUY | SELL | HOLD",
  "confidence": 0.85,
  "timeframe_confluence": "Aligned Bullish | Conflicting | Neutral",
  "macro_analysis": "Detailed synthesis of SPY trend and VIX impact on this specific asset.",
  "news_catalyst_analysis": "Evaluation of recent headlines and news sentiment impact.",
  "higher_tf_breakdown": "Daily/4H trend structure, key support/resistance, and moving average alignment.",
  "intraday_tf_breakdown": "5m execution setup, RSI momentum, MACD state, and Bollinger Band compression/expansion.",
  "execution_plan": {
    "entry_zone": "$150.00 - $151.20",
    "stop_loss": 148.50,
    "take_profit": 156.00,
    "risk_reward_ratio": "1:2.5",
    "key_support": 149.00,
    "key_resistance": 155.50
  },
  "detailed_reasoning": "Comprehensive institutional synthesis explaining WHY this trade setup exists, potential failure points, and overall trade execution advice."
}

CRITICAL RULES:
- Output strictly raw JSON. Do NOT wrap output in markdown code fences like ```json ... ```.
- Numbers in execution_plan must be raw floats or formatted strings as shown.
- Do not abbreviate or give short answers—provide complete, thorough professional commentary.
"""

def fetch_macro_context() -> dict:
    try:
        spy = yf.Ticker("SPY").history(period="2d")
        vix = yf.Ticker("^VIX").history(period="1d")
        spy_change = 0.0
        if len(spy) >= 2:
            prev_close = spy['Close'].iloc[-2]
            curr_close = spy['Close'].iloc[-1]
            spy_change = ((curr_close - prev_close) / prev_close) * 100
        vix_level = float(vix['Close'].iloc[-1]) if not vix.empty else 20.0
        return {"spy_change_pct": round(spy_change, 2), "vix": round(vix_level, 2)}
    except Exception as e:
        return {"spy_change_pct": 0.0, "vix": 20.0}

def fetch_recent_news(ticker: str, limit: int = 3) -> list:
    try:
        t = yf.Ticker(ticker)
        news_items = t.news or []
        headlines = []
        for item in news_items[:limit]:
            title = item.get('title') or item.get('content', {}).get('title')
            if title:
                headlines.append(title)
        return headlines if headlines else ["No recent major headlines found."]
    except Exception as e:
        return ["News unavailable."]

def _extract_tf_metrics(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {}
    latest = df.iloc[-1]
    return {
        "price": float(latest.get('Close', 0.0)),
        "sma_20": float(latest.get('SMA_20', 0.0)),
        "sma_50": float(latest.get('SMA_50', 0.0)),
        "sma_200": float(latest.get('SMA_200', 0.0)),
        "rsi": float(latest.get('RSI', 0.0)),
        "macd": float(latest.get('MACD', 0.0)),
        "macd_signal": float(latest.get('MACD_Signal', 0.0)),
        "atr": float(latest.get('ATR', 0.0)),
        "bb_upper": float(latest.get('BB_Upper', 0.0)),
        "bb_lower": float(latest.get('BB_Lower', 0.0))
    }

def generate_ai_analysis(
    ticker: str, 
    df_5m: pd.DataFrame = None, 
    df_4h: pd.DataFrame = None, 
    df_1d: pd.DataFrame = None, 
    sentiment_summary: str = "", 
    **kwargs
) -> dict:
    default_response = {
        "signal": "HOLD",
        "confidence": 0.0,
        "timeframe_confluence": "Unknown",
        "macro_analysis": "Data unavailable.",
        "news_catalyst_analysis": "Data unavailable.",
        "higher_tf_breakdown": "Data unavailable.",
        "intraday_tf_breakdown": "Data unavailable.",
        "execution_plan": {
            "entry_zone": "N/A", "stop_loss": 0.0, "take_profit": 0.0,
            "risk_reward_ratio": "N/A", "key_support": 0.0, "key_resistance": 0.0
        },
        "detailed_reasoning": "Unable to generate full analysis due to API or formatting error."
    }

    if not api_key:
        default_response["detailed_reasoning"] = "Gemini API key is missing."
        return default_response

    try:
        data_5m = _extract_tf_metrics(df_5m)
        data_4h = _extract_tf_metrics(df_4h)
        data_1d = _extract_tf_metrics(df_1d)
        
        macro = fetch_macro_context()
        news = fetch_recent_news(ticker)
        news_formatted = "\n".join([f"- {h}" for h in news])

        model = genai.GenerativeModel("gemini-flash-latest")

        user_prompt = f"""
        Perform complete institutional multi-factor analysis for ticker: {ticker}

        === 1. MACRO CONTEXT ===
        - SPY 1-Day Return: {macro['spy_change_pct']}%
        - Volatility Index (VIX): {macro['vix']}

        === 2. NEWS & SENTIMENT ===
        Sentiment Context: {sentiment_summary if sentiment_summary else "N/A"}
        Headlines:
        {news_formatted}

        === 3. HIGHER TIMEFRAME (DAILY & 4H MACRO TREND) ===
        - Daily Price: ${data_1d.get('price', 0.0):.2f}, 20 SMA: {data_1d.get('sma_20', 0.0)}, 50 SMA: {data_1d.get('sma_50', 0.0)}, 200 SMA: {data_1d.get('sma_200', 0.0)}         - Daily RSI: {data_1d.get('rsi', 0.0)}, MACD: {data_1d.get('macd', 0.0)}         - 4H Price:${data_4h.get('price', 0.0):.2f}, 4H RSI: {data_4h.get('rsi', 0.0)}, 4H MACD: {data_4h.get('macd', 0.0)}

        === 4. INTRADAY TIMEFRAME (5M TACTICAL EXECUTION) ===
        - 5m Price: ${data_5m.get('price', 0.0):.2f}
        - 5m RSI: {data_5m.get('rsi', 0.0)}
        - 5m MACD Line: {data_5m.get('macd', 0.0)}, Signal: {data_5m.get('macd_signal', 0.0)}
        - 5m Bollinger Bands: Upper {data_5m.get('bb_upper', 0.0)}, Lower {data_5m.get('bb_lower', 0.0)}

        Synthesize all layers into a complete trade execution report in raw JSON.
        """

        response = model.generate_content(
            contents=[SYSTEM_PROMPT, user_prompt],
            generation_config={"temperature": 0.2}
        )

        clean_text = response.text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.strip("`").replace("json\n", "").replace("json", "").strip()

        return json.loads(clean_text)

    except Exception as e:
        print(f"Error generating AI analysis: {e}")
        return default_response

synthesize_signals = generate_ai_analysis