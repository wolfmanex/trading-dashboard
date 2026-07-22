import os
import pandas as pd
import streamlit as st
import google.generativeai as genai

def configure_llm():
    """Initializes the Gemini API using Streamlit secrets or OS env vars."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return False
    
    genai.configure(api_key=api_key)
    return True

def safe_get_latest(df: pd.DataFrame) -> dict:
    """Safely extracts the last row of a DataFrame to prevent index or empty table crashes."""
    if df is None or df.empty:
        return {}
    return df.iloc[-1].to_dict()

def synthesize_signals(ticker: str, df_5m: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, sentiment_summary: str) -> str:
    """Passes multi-timeframe data to the LLM to generate a unified trade decision."""
    
    if not configure_llm():
        return "⚠️ **API Key Missing:** Please add your `GEMINI_API_KEY` to your environment variables or Streamlit secrets."

    # Safely convert the last row of each timeframe into a dictionary
    latest_5m = safe_get_latest(df_5m)
    latest_4h = safe_get_latest(df_4h)
    latest_1d = safe_get_latest(df_1d)

    # Build the prompt. 
    # All .get() fallbacks are strict 0.0 floats to guarantee formatting safety.
    prompt = f"""
    You are an elite quantitative trading assistant. Analyze the following multi-timeframe technical data and news sentiment for {ticker}.
    Provide a concise, professional trading recommendation (Bullish, Bearish, or Neutral) with key price targets.
    
    === ASSET: {ticker} ===
    
    [ 5-Minute Timeframe (Micro / Entry) ]
    - Close Price: ${latest_5m.get('Close', 0.0):.2f}
    - RSI (14): {latest_5m.get('RSI', 0.0):.2f}
    - MACD: {latest_5m.get('MACD', 0.0):.2f} | Signal: {latest_5m.get('Signal_Line', 0.0):.2f}
    - EMA 9: ${latest_5m.get('EMA_9', 0.0):.2f} | EMA 21: ${latest_5m.get('EMA_21', 0.0):.2f}
    
    [ 4-Hour Timeframe (Macro / Trend) ]
    - Close Price: ${latest_4h.get('Close', 0.0):.2f}
    - RSI (14): {latest_4h.get('RSI', 0.0):.2f}
    - MACD: {latest_4h.get('MACD', 0.0):.2f} | Signal: {latest_4h.get('Signal_Line', 0.0):.2f}
    
    [ 1-Day Timeframe (Overarching Trend) ]
    - Close Price: ${latest_1d.get('Close', 0.0):.2f}
    - RSI (14): {latest_1d.get('RSI', 0.0):.2f}
    
    [ News Sentiment Summary ]
    {sentiment_summary if sentiment_summary else "No recent news available."}
    
    Based on this data:
    1. What is the immediate trend?
    2. Are there any divergences across timeframes?
    3. What is the recommended action (Long/Short/Hold)?
    4. What are the logical support and resistance levels?
    
    Format your response cleanly using markdown headings and bullet points. Keep it professional, objective, and dense with insight.
    """
    
    try:
        # Initializing the modern standard Gemini model
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🚨 **AI Synthesis Failed:** {e}"