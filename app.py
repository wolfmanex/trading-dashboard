import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from google import genai

# Download NLTK VADER lexicon quietly if not present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ==============================================================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==============================================================================
st.set_page_config(
    page_title="AI Multi-Signal US Stock Trading Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2a2e39;
        text-align: center;
    }
    .signal-box-bullish {
        background-color: #132e22;
        color: #26a69a;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #26a69a;
        font-weight: bold;
        text-align: center;
    }
    .signal-box-bearish {
        background-color: #381e25;
        color: #ef5350;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #ef5350;
        font-weight: bold;
        text-align: center;
    }
    .signal-box-neutral {
        background-color: #2a2e39;
        color: #b2b5be;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #787b86;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. SECRETS & SIDEBAR CONTROLS (US STOCK SELECTOR)
# ==============================================================================
gemini_api_key = ""
try:
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    gemini_api_key = ""

st.sidebar.header("🕹️ Control Panel")

# Popular US Stocks Preset List
POPULAR_US_STOCKS = [
    "AMD", "NVDA", "AAPL", "MSFT", "TSLA", "GOOGL", "AMZN", "META",
    "NFLX", "PLTR", "INTC", "SPY", "QQQ", "IWM", "COIN", "DIS"
]

stock_mode = st.sidebar.radio("Select Stock Input Method:", ["Popular US Stocks", "Custom Ticker Search"])

if stock_mode == "Popular US Stocks":
    symbol = st.sidebar.selectbox("Choose US Stock:", POPULAR_US_STOCKS, index=0)
else:
    custom_symbol = st.sidebar.text_input("Enter Any US Ticker Symbol (e.g., BAC, JPM, UNH):", value="AMD")
    symbol = custom_symbol.strip().upper()

timeframe = st.sidebar.selectbox("Select Timeframe", options=["5m", "15m", "1h", "4h"], index=1)

if not gemini_api_key:
    st.sidebar.warning("⚠️ Local `GEMINI_API_KEY` not found in `.streamlit/secrets.toml`.")
else:
    st.sidebar.success("🔑 Gemini API Key Active")

# ==============================================================================
# 3. ENGINES: TECHNICAL, NEWS SENTIMENT, & MARKET TREND
# ==============================================================================

@st.cache_data(ttl=60)
def fetch_technical_engine(ticker: str, tf: str) -> pd.DataFrame:
    """Component 1: Technical Analysis & Charting Data"""
    t_obj = yf.Ticker(ticker)
    
    if tf in ["5m", "15m"]:
        df = t_obj.history(period="5d", interval=tf)
    elif tf == "1h":
        df = t_obj.history(period="1mo", interval="1h")
    elif tf == "4h":
        df_1h = t_obj.history(period="3mo", interval="1h")
        if df_1h.empty: return pd.DataFrame()
        df = df_1h.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()
    else:
        return pd.DataFrame()

    if df.empty: return df

    # Technical Indicators
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, 1e-9))
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Cross Markers
    df['EMA_Cross_Bull'] = (df['EMA_9'] > df['EMA_21']) & (df['EMA_9'].shift(1) <= df['EMA_21'].shift(1))
    df['EMA_Cross_Bear'] = (df['EMA_9'] < df['EMA_21']) & (df['EMA_9'].shift(1) >= df['EMA_21'].shift(1))
    
    return df

@st.cache_data(ttl=300)
def fetch_news_sentiment(ticker: str):
    """Component 2: Robust News Sentiment Engine (Handles new & old yfinance formats)"""
    try:
        t_obj = yf.Ticker(ticker)
        news_items = t_obj.news
        if not news_items:
            return {"score": 0.0, "label": "NEUTRAL", "headlines": []}
        
        sia = SentimentIntensityAnalyzer()
        scores = []
        headlines = []

        for item in news_items[:8]:
            title = ""
            # Handle new nested yfinance format (v0.2.50+)
            if isinstance(item, dict):
                if 'content' in item and isinstance(item['content'], dict):
                    title = item['content'].get('title', '')
                elif 'title' in item:
                    title = item.get('title', '')

            if title:
                headlines.append(title)
                score = sia.polarity_scores(title)['compound']
                scores.append(score)

        if not scores:
            return {"score": 0.0, "label": "NEUTRAL", "headlines": []}

        avg_score = float(np.mean(scores))
        label = "BULLISH" if avg_score >= 0.05 else ("BEARISH" if avg_score <= -0.05 else "NEUTRAL")
        return {"score": avg_score, "label": label, "headlines": headlines}
    except Exception as e:
        return {"score": 0.0, "label": "NEUTRAL", "headlines": []}

@st.cache_data(ttl=120)
def fetch_market_trend():
    """Component 3: Macro Market Trend Engine (S&P 500 & Nasdaq)"""
    try:
        sp500 = yf.Ticker("^GSPC").history(period="2d", interval="1d")
        nasdaq = yf.Ticker("^IXIC").history(period="2d", interval="1d")

        sp_pct = ((sp500['Close'].iloc[-1] - sp500['Close'].iloc[-2]) / sp500['Close'].iloc[-2]) * 100
        nas_pct = ((nasdaq['Close'].iloc[-1] - nasdaq['Close'].iloc[-2]) / nasdaq['Close'].iloc[-2]) * 100

        avg_market_change = (sp_pct + nas_pct) / 2
        trend_label = "BULLISH" if avg_market_change > 0.1 else ("BEARISH" if avg_market_change < -0.1 else "NEUTRAL")
        return {"sp_pct": sp_pct, "nas_pct": nas_pct, "trend_label": trend_label}
    except Exception:
        return {"sp_pct": 0.0, "nas_pct": 0.0, "trend_label": "NEUTRAL"}

# Fetch all 3 data sources
with st.spinner(f"Aggregating 4-component signal engine for {symbol}..."):
    tech_df = fetch_technical_engine(symbol, timeframe)
    news_data = fetch_news_sentiment(symbol)
    market_data = fetch_market_trend()

if tech_df.empty:
    st.error(f"Failed to pull ticker data for '{symbol}'. Please verify the ticker symbol and try again.")
    st.stop()

# ==============================================================================
# 4. DASHBOARD KPI & 4-SIGNAL SUMMARY
# ==============================================================================
latest = tech_df.iloc[-1]
prev_close = tech_df.iloc[-2]['Close']
price_diff = latest['Close'] - prev_close
pct_diff = (price_diff / prev_close) * 100

st.title(f"⚡ {symbol} Multi-Signal Trading Dashboard")
st.caption(f"Timeframe: {timeframe} | Last Update: {tech_df.index[-1].strftime('%Y-%m-%d %H:%M')}")

# Top Metric Row
col_p, col_ema, col_rsi, col_macd = st.columns(4)
col_p.metric("Price", f"${latest['Close']:.2f}", f"{price_diff:+.2f} ({pct_diff:+.2f}%)")
col_ema.metric("EMA 9 / 21", f"${latest['EMA_9']:.2f}", f"EMA 21: ${latest['EMA_21']:.2f}")
col_rsi.metric("RSI (14)", f"{latest['RSI_14']:.1f}", "Overbought" if latest['RSI_14'] > 70 else ("Oversold" if latest['RSI_14'] < 30 else "Neutral"))
col_macd.metric("MACD Hist", f"{latest['MACD_Hist']:+.2f}", "Bullish" if latest['MACD_Hist'] > 0 else "Bearish")

st.divider()

# 4-Component Summary Row
st.subheader("🧩 The 4 Signal Components")
sig1, sig2, sig3, sig4 = st.columns(4)

# 1. Technical Signal
tech_sig = "BULLISH" if (latest['EMA_9'] > latest['EMA_21'] and latest['MACD_Hist'] > 0) else ("BEARISH" if (latest['EMA_9'] < latest['EMA_21'] and latest['MACD_Hist'] < 0) else "NEUTRAL")
with sig1:
    st.markdown("**1. Technical Engine**")
    if tech_sig == "BULLISH": st.markdown('<div class="signal-box-bullish">🟢 BULLISH</div>', unsafe_allow_html=True)
    elif tech_sig == "BEARISH": st.markdown('<div class="signal-box-bearish">🔴 BEARISH</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="signal-box-neutral">⚪ NEUTRAL</div>', unsafe_allow_html=True)

# 2. News Sentiment
with sig2:
    st.markdown(f"**2. News Sentiment** ({news_data['score']:+.2f})")
    n_lbl = news_data['label']
    if n_lbl == "BULLISH": st.markdown('<div class="signal-box-bullish">🟢 BULLISH</div>', unsafe_allow_html=True)
    elif n_lbl == "BEARISH": st.markdown('<div class="signal-box-bearish">🔴 BEARISH</div>', unsafe_allow_html=True)
    else: st.markdown('<div class="signal-box-neutral">⚪ NEUTRAL</div>', unsafe_allow_html=True)

# 3. Market Trend
with sig3:
    st.markdown("**3. Market Trend**")
    m_lbl = market_data['trend_label']
    if m_lbl == "BULLISH": st.markdown(f'<div class="signal-box-bullish">🟢 BULLISH ({market_data["sp_pct"]:+.1f}%)</div>', unsafe_allow_html=True)
    elif m_lbl == "BEARISH": st.markdown(f'<div class="signal-box-bearish">🔴 BEARISH ({market_data["sp_pct"]:+.1f}%)</div>', unsafe_allow_html=True)
    else: st.markdown(f'<div class="signal-box-neutral">⚪ NEUTRAL ({market_data["sp_pct"]:+.1f}%)</div>', unsafe_allow_html=True)

# 4. LLM Synthesis Trigger
with sig4:
    st.markdown("**4. LLM Synthesis**")
    st.markdown('<div class="signal-box-neutral">⏳ AWAITING RUN</div>', unsafe_allow_html=True)

st.divider()

# ==============================================================================
# 5. PLOTLY CHART
# ==============================================================================
def draw_chart(df: pd.DataFrame, symbol_str: str, tf_str: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.8, 0.2])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_9'], mode='lines', name='EMA 9', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_21'], mode='lines', name='EMA 21', line=dict(color='#2196f3', width=1.5)), row=1, col=1)

    # Cross Markers
    bulls = df[df['EMA_Cross_Bull']]
    bears = df[df['EMA_Cross_Bear']]
    if not bulls.empty:
        fig.add_trace(go.Scatter(x=bulls.index, y=bulls['Low'] * 0.998, mode='markers', name='Bull Cross', marker=dict(symbol='triangle-up', size=11, color='#00e676')), row=1, col=1)
    if not bears.empty:
        fig.add_trace(go.Scatter(x=bears.index, y=bears['High'] * 1.002, mode='markers', name='Bear Cross', marker=dict(symbol='triangle-down', size=11, color='#ff5252')), row=1, col=1)

    # Volume Subplot
    colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors), row=2, col=1)

    fig.update_layout(
        title=f"📊 {symbol_str} ({tf_str}) Chart Analysis",
        template="plotly_dark", height=560, xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

chart = draw_chart(tech_df, symbol, timeframe)
st.plotly_chart(chart, use_container_width=True)

# ==============================================================================
# 6. COMPONENT 4: GEMINI LLM ANALYSIS ENGINE (gemini-flash-lite-latest)
# ==============================================================================
st.subheader("🤖 AI Synthesis (4-Factor Model)")

def request_ai_analysis(df, news, market, ticker, tf_val, api_key):
    try:
        client = genai.Client(api_key=api_key)
        curr = df.iloc[-1]
        
        prompt = f"""
        Act as an elite intraday portfolio manager evaluating {ticker} ({tf_val} timeframe). Synthesize all 4 signals into a final trading recommendation:
        
        1. TECHNICAL ENGINE:
           - Price: ${curr['Close']:.2f}
           - EMA 9: ${curr['EMA_9']:.2f} | EMA 21: ${curr['EMA_21']:.2f}
           - RSI (14): {curr['RSI_14']:.2f}
           - MACD Hist: {curr['MACD_Hist']:.2f}
           - Tech Signal: {tech_sig}
        
        2. NEWS SENTIMENT:
           - Label: {news['label']} (Compound Score: {news['score']:.2f})
           - Headlines: {news['headlines'][:3]}
        
        3. MARKET TREND CONTEXT:
           - S&P 500 Daily Move: {market['sp_pct']:+.2f}%
           - Nasdaq Daily Move: {market['nas_pct']:+.2f}%
           - Macro Context: {market['trend_label']}
        
        INSTRUCTIONS:
        Synthesize these 4 factors into a 3-part structured trade plan:
        - **Multi-Factor Confluence:** How well do technicals, news, and market trend align?
        - **Risk/Reward Level:** Key stop-loss invalidation price and target level.
        - **Final Trade Signal:** Explicit decision [BUY / SELL / WAIT] with 1-sentence justification.
        """
        
        # Updated model target to gemini-flash-lite-latest
        res = client.models.generate_content(
            model='gemini-flash-lite-latest',
            contents=prompt
        )
        return res.text
    except Exception as err:
        return f"AI Generation Error: {str(err)}"

if st.button("🚀 Synthesize All 4 Signals with Gemini AI", use_container_width=True):
    if not gemini_api_key:
        st.error("Missing Gemini API Key. Please add `GEMINI_API_KEY` to `.streamlit/secrets.toml` or your deployment platform's environment variables.")
    else:
        with st.spinner(f"Synthesizing Chart Technicals, News Sentiment, and Market Macro for {symbol}..."):
            ai_out = request_ai_analysis(tech_df, news_data, market_data, symbol, timeframe, gemini_api_key)
            st.info(ai_out)

# Expanders for News Headlines & Raw Data
exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    with st.expander("📰 Scraped Headlines & Sentiment"):
        if news_data['headlines']:
            for h in news_data['headlines']:
                st.markdown(f"• {h}")
        else:
            st.write(f"No recent news headlines available for {symbol}.")

with exp_col2:
    with st.expander("📄 Raw Technical Data Table"):
        st.dataframe(tech_df[['Close', 'EMA_9', 'EMA_21', 'RSI_14', 'MACD_Hist']].sort_index(ascending=False), use_container_width=True)