import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from technical_engine import get_technical_data, get_multi_timeframe_data, get_live_price, add_technical_indicators
from news_engine import get_ticker_news_sentiment
from index_filter import get_macro_market_trend
from llm_engine import synthesize_signals


st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Professional Dashboard Custom Styling (Typography & Glow)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        .stApp { 
            background-color: #0B0E14; 
            color: #E2E8F0; 
            font-family: 'Inter', sans-serif;
        }
        
        .metric-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 900px) {
            .metric-container {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        @media (max-width: 600px) {
            .metric-container {
                grid-template-columns: 1fr;
            }
        }

        .kpi-card {
            background: #11151F;
            border: 1px solid #1E2532;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            transition: all 0.3s ease;
        }
        .kpi-card:hover {
            border-color: #00F0FF;
            transform: translateY(-3px);
            box-shadow: 0 8px 24px rgba(0, 240, 255, 0.1);
        }
        .kpi-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #718096;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .kpi-badge {
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Neon glows */
        .badge-bullish { background: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; box-shadow: 0 0 8px rgba(0,255,136,0.2); }
        .badge-bearish { background: rgba(255, 0, 85, 0.1); color: #ff0055; border: 1px solid #ff0055; box-shadow: 0 0 8px rgba(255,0,85,0.2); }
        .badge-neutral { background: rgba(160, 174, 192, 0.1); color: #A0AEC0; border: 1px solid #A0AEC0; }
        .badge-cyan    { background: rgba(0, 240, 255, 0.1); color: #00F0FF; border: 1px solid #00F0FF; box-shadow: 0 0 8px rgba(0,240,255,0.2); }
    </style>
""", unsafe_allow_html=True)


# Initialize Session State
if "llm_analysis" not in st.session_state:
    st.session_state.llm_analysis = None
if "last_analyzed_ticker" not in st.session_state:
    st.session_state.last_analyzed_ticker = None


st.title("📈 AI Trading Dashboard")

# Sidebar Controls
st.sidebar.header("Control Panel")
preset_tickers = ["AMD", "AAPL", "NVDA", "MSFT", "TSLA", "BTC-USD", "EURUSD=X"]
select_mode = st.sidebar.radio("Ticker Mode", ["Preset List", "Custom Input"])

if select_mode == "Preset List":
    selected_ticker = st.sidebar.selectbox("Select Asset", preset_tickers)
else:
    selected_ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()

# Timeframe Selection for the Chart
timeframe = st.sidebar.selectbox("Chart Timeframe", ["5m", "15m", "1h", "1d"], index=0)

# Reset analysis state if user changes the ticker
if selected_ticker != st.session_state.last_analyzed_ticker:
    st.session_state.llm_analysis = None
    st.session_state.last_analyzed_ticker = selected_ticker

# Load Chart & Indicator Data
with st.spinner(f"Loading market data for {selected_ticker}..."):
    df_chart = get_technical_data(selected_ticker, timeframe=timeframe)
    news_sentiment, sentiment_summary = get_ticker_news_sentiment(selected_ticker)
    macro_trend = get_macro_market_trend()

if df_chart.empty:
    st.error(f"No price data available for {selected_ticker} on timeframe {timeframe}. Check ticker or market hours.")
    st.stop()

# Helper function to assign badge color classes
def get_badge_class(text_str: str) -> str:
    lower_s = str(text_str).lower()
    if "bullish" in lower_s:
        return "badge-bullish"
    elif "bearish" in lower_s:
        return "badge-bearish"
    return "badge-neutral"

# --- Live Price Logic & Feed Status ---
raw_live_price = get_live_price(selected_ticker)

if raw_live_price > 0 and not pd.isna(raw_live_price):
    latest_price = raw_live_price
    price_badge_text = f"LIVE • {selected_ticker}"
    price_badge_class = "badge-cyan"
    
    # Bind live price into df_chart so the Candlestick chart and indicators update
    df_chart.iloc[-1, df_chart.columns.get_loc('Close')] = latest_price
    df_chart.iloc[-1, df_chart.columns.get_loc('High')] = max(df_chart['High'].iloc[-1], latest_price)
    df_chart.iloc[-1, df_chart.columns.get_loc('Low')] = min(df_chart['Low'].iloc[-1], latest_price)
    
    # Recalculate indicators so RSI & EMAs on chart match the live price
    df_chart = add_technical_indicators(df_chart)
else:
    latest_price = float(df_chart['Close'].iloc[-1])
    price_badge_text = f"CLOSED • {selected_ticker}"
    price_badge_class = "badge-neutral"

rsi_val = df_chart['RSI'].iloc[-1] if 'RSI' in df_chart and not df_chart['RSI'].isna().all() else 0.0

rsi_badge = "badge-neutral"
rsi_state = "Neutral"
if rsi_val >= 70:
    rsi_badge = "badge-bearish"
    rsi_state = "Overbought"
elif rsi_val <= 30:
    rsi_badge = "badge-bullish"
    rsi_state = "Oversold"

# Render Custom KPI Cards Top Row
st.markdown(f"""
<div class="metric-container">
    <div class="kpi-card">
        <div class="kpi-title">Price ({timeframe.upper()})</div>
        <div class="kpi-value">
            ${latest_price:,.2f}
            <span class="kpi-badge {price_badge_class}">{price_badge_text}</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">RSI (14)</div>
        <div class="kpi-value">
            {rsi_val:.1f}
            <span class="kpi-badge {rsi_badge}">{rsi_state}</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">News Sentiment</div>
        <div class="kpi-value" style="font-size: 1.25rem;">
            {news_sentiment.split(' ')[0]}
            <span class="kpi-badge {get_badge_class(news_sentiment)}">{news_sentiment}</span>
        </div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Macro Trend (^GSPC)</div>
        <div class="kpi-value" style="font-size: 1.25rem;">
            {macro_trend.split(' ')[0]}
            <span class="kpi-badge {get_badge_class(macro_trend)}">{macro_trend}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# Interactive Candlestick Chart with Volume
st.subheader(f"📊 Technical Chart ({timeframe}) — {selected_ticker}")

fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.03, 
    row_heights=[0.75, 0.25]
)

# Row 1: Candlesticks
fig.add_trace(go.Candlestick(
    x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
    low=df_chart['Low'], close=df_chart['Close'], name="Price",
    increasing_line_color='#00ff88', decreasing_line_color='#ff0055'
), row=1, col=1)

# Row 1: EMAs
if 'EMA_9' in df_chart:
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_9'], line=dict(color='#00F0FF', width=1.5), name="EMA 9"), row=1, col=1)
if 'EMA_21' in df_chart:
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_21'], line=dict(color='#FF007A', width=1.5), name="EMA 21"), row=1, col=1)

# Row 2: Volume Bar Chart
colors = ['#00ff88' if row.Close >= row.Open else '#ff0055' for index, row in df_chart.iterrows()]
fig.add_trace(go.Bar(
    x=df_chart.index, y=df_chart['Volume'], name="Volume", marker_color=colors, opacity=0.8
), row=2, col=1)

# Pro-TradingView Styling
fig.update_layout(
    template="plotly_dark",
    height=650,
    margin=dict(l=10, r=10, t=20, b=20),
    xaxis_rangeslider_visible=False,
    plot_bgcolor='rgba(11, 14, 20, 1)',
    paper_bgcolor='rgba(11, 14, 20, 1)',
    showlegend=False
)

# Identify missing dates to remove gaps (weekends, off-hours)
freq_map = {"5m": "5min", "15m": "15min", "1h": "1h", "1d": "D"}
# Dvalue dictates the width of the gap in milliseconds (e.g., 5 mins = 300,000 ms)
dvalue_map = {"5m": 300000, "15m": 900000, "1h": 3600000, "1d": 86400000}

if timeframe in freq_map:
    # Build a complete continuous timeline from start to end
    full_idx = pd.date_range(start=df_chart.index.min(), end=df_chart.index.max(), freq=freq_map[timeframe])
    # Find the timestamps that are missing from our actual data
    missing_dt = full_idx.difference(df_chart.index)
    
    # Instruct Plotly to hide these specific timestamps
    fig.update_xaxes(
        rangebreaks=[dict(values=missing_dt, dvalue=dvalue_map[timeframe])]
    )

# Subdued gridlines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E2532', row=1, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E2532', row=1, col=1)
fig.update_xaxes(showgrid=False, row=2, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Callback to run multi-timeframe LLM synthesis cleanly
def run_synthesis_callback():
    with st.spinner("Fetching multi-timeframe data & synthesizing signals..."):
        df_5m, df_4h, df_1d = get_multi_timeframe_data(selected_ticker)
        st.session_state.llm_analysis = synthesize_signals(
            ticker=selected_ticker,
            df_5m=df_5m,
            df_4h=df_4h,
            df_1d=df_1d,
            sentiment_summary=sentiment_summary
        )

# Section: AI Synthesis Control
col_title, col_btn = st.columns([3, 1])

with col_title:
    st.subheader("🤖 Multi-Timeframe AI Synthesis")

with col_btn:
    btn_label = "🔄 Regenerate Analysis" if st.session_state.llm_analysis else "🚀 Run AI Analysis"
    st.button(btn_label, on_click=run_synthesis_callback, use_container_width=True)

# Render results
if st.session_state.llm_analysis:
    st.markdown(st.session_state.llm_analysis)
else:
    st.info("Click 'Run AI Analysis' above to generate a multi-timeframe unified trade decision.")