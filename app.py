import pandas as pd
import streamlit as st
from html import escape
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from technical_engine import (
    get_technical_data, get_multi_timeframe_data, get_live_price, add_technical_indicators,
    get_market_session, should_apply_live_price,
)
from news_engine import get_ticker_news_sentiment
from index_filter import get_macro_market_trend
from llm_engine import (
    generate_ai_analysis, synthesize_signals, is_ai_configured, review_breakout_candidates, check_execution_plan,
    parse_plan_level,
)
from event_engine import get_upcoming_events
from options_engine import get_options_sentiment
from swing_engine import get_swing_metrics
from intraday_engine import get_intraday_metrics
from backtest_engine import run_ta_backtest
from watchlist_engine import get_watchlist_snapshot, normalize_watchlist
from mover_universe import MOVER_UNIVERSE
from movers_engine import get_market_movers
from stock_info import get_stock_profile, format_profile_summary
from risk_engine import position_size
from smallcap_screener import (
    get_candidate_context, get_smallcap_universe, scan_smallcap_breakouts,
    get_smallcap_regime, backtest_smallcap_breakouts,
    MIN_MARKET_CAP, MAX_MARKET_CAP, MIN_REWARD_RISK, MIN_DOLLAR_VOLUME,
)


st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Professional Dashboard Custom Styling (Typography & Glow)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        
        .stApp { 
            background:
                radial-gradient(circle at 85% 0%, rgba(0, 240, 255, 0.08), transparent 34%),
                linear-gradient(135deg, #090C12 0%, #0B0E14 52%, #0E121B 100%);
            color: #E2E8F0;
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #0D1119;
            border-right: 1px solid #202938;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #F8FAFC;
            letter-spacing: -0.02em;
        }

        h1 {
            font-weight: 700;
            letter-spacing: -0.04em;
            margin-bottom: 0.25rem;
        }

        h2, h3 {
            letter-spacing: -0.025em;
        }
        
        .metric-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 1.25rem 0 1.75rem;
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
            background: rgba(17, 21, 31, 0.88);
            border: 1px solid #202938;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 8px 28px rgba(0, 0, 0, 0.24);
            transition: all 0.3s ease;
        }
        .kpi-card:hover {
            border-color: #00F0FF;
            transform: translateY(-2px);
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
            font-size: 1.65rem;
            font-weight: 700;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        [data-testid="stMetric"] {
            background: rgba(17, 21, 31, 0.72);
            border: 1px solid #202938;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
        }

        .status-label {
            color: #718096;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-ready { color: #00ff88; }
        .status-warning { color: #FFD166; }
        .status-error { color: #ff6b8a; }

        .profile-label {
            color: #8FA1B5;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.07em;
            text-transform: uppercase;
        }

        .profile-value {
            color: #F3F7FC;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1rem;
            font-weight: 500;
            line-height: 1.35;
            margin-top: 0.25rem;
        }

        .profile-description-spacer {
            height: 1rem;
        }

        .system-status-bar {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 1rem;
            background: rgba(17, 21, 31, 0.48);
            border: 1px solid rgba(32, 41, 56, 0.72);
            border-radius: 6px;
            padding: 0.55rem 0.8rem;
            margin: 0.75rem 0 1.35rem;
        }

        .system-status-bar .status-label {
            font-size: 0.62rem;
            letter-spacing: 0.07em;
        }

        .system-status-bar .status-value {
            color: #D8E1EC;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.72rem;
            white-space: nowrap;
        }

        @media (max-width: 768px) {
            .system-status-bar {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.75rem;
            }
        }

        @media (max-width: 480px) {
            .system-status-bar {
                grid-template-columns: 1fr;
            }
        }

        .section-nav {
            background: rgba(17, 21, 31, 0.72);
            border: 1px solid #202938;
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0 1.5rem;
        }

        .section-nav a {
            color: #A9B6C8;
            margin-right: 1rem;
            text-decoration: none;
            font-size: 0.86rem;
        }

        .section-nav a:hover { color: #00F0FF; }

        .view-button {
            display: inline-block;
            width: 100%;
            box-sizing: border-box;
            background: #151C28;
            border: 1px solid #2B394D;
            border-radius: 6px;
            color: #D8E1EC !important;
            padding: 0.55rem 0.7rem;
            text-align: center;
            text-decoration: none !important;
            font-size: 0.78rem;
            font-weight: 600;
            transition: border-color 0.2s ease, color 0.2s ease, background 0.2s ease;
        }

        .view-button:hover {
            background: #1B2635;
            border-color: #00F0FF;
            color: #00F0FF !important;
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 0.75rem 2rem;
            }

            h1 { font-size: 1.75rem; }
            h2 { font-size: 1.35rem; }
            h3 { font-size: 1.1rem; }

            .metric-container {
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 8px;
                margin: 0.75rem 0 1.25rem;
            }

            .kpi-card {
                min-width: 0;
                padding: 12px;
            }

            .kpi-value {
                align-items: flex-start;
                flex-direction: column;
                gap: 8px;
                font-size: 1.25rem;
            }

            .kpi-badge {
                max-width: 100%;
                overflow-wrap: anywhere;
                white-space: normal;
            }

            .section-nav a {
                display: inline-block;
                margin: 0 0.7rem 0.4rem 0;
            }

            [data-testid="stMetric"] {
                min-width: 0;
                padding: 0.6rem 0.7rem;
            }

            [data-testid="stMetricLabel"] {
                overflow-wrap: anywhere;
            }

            [data-testid="stDataFrame"] {
                overflow-x: auto;
            }
        }

        @media (max-width: 480px) {
            .metric-container {
                grid-template-columns: 1fr;
            }

            .kpi-value {
                flex-direction: row;
                align-items: center;
                justify-content: space-between;
                font-size: 1.35rem;
            }
        }

        code, .stCode {
            font-family: 'IBM Plex Mono', monospace;
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
if "ticker_mode" not in st.session_state:
    st.session_state.ticker_mode = "Preset List"
if "custom_ticker" not in st.session_state:
    st.session_state.custom_ticker = "AMD"
if "scanner_results" not in st.session_state:
    st.session_state.scanner_results = None
if "scanner_reviews" not in st.session_state:
    st.session_state.scanner_reviews = {}
if "scanner_review_error" not in st.session_state:
    st.session_state.scanner_review_error = None


# Helper to prevent Streamlit from treating dollar signs in AI text as LaTeX
def sanitize_ai_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Escapes unescaped $ signs so Streamlit won't parse them as LaTeX math formulas
    return text.replace("$", r"\$")


DASHBOARD_TAB = "📊 Dashboard"
SCANNER_TAB = "🚀 Breakout Scanner"


def load_scanner_ticker():
    """Switch the dashboard to the breakout candidate clicked in the scanner table."""
    selection = st.session_state.get("breakout_table")
    rows = selection.selection.rows if selection else []
    tickers = st.session_state.get("scanner_display_tickers", [])
    if rows and rows[0] < len(tickers):
        st.session_state.ticker_mode = "Custom Input"
        st.session_state.custom_ticker = tickers[rows[0]]
        st.session_state.main_view = DASHBOARD_TAB


st.title("📈 AI Trading Dashboard")

# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("Trading Dashboard Controls")

preset_tickers = ["AMD", "QCOM", "AAPL", "NVDA", "MSFT", "TSLA", "BTC-USD", "EURUSD=X"]

with st.sidebar.expander("Asset", expanded=True):
    select_mode = st.radio("Ticker Mode", ["Preset List", "Custom Input"], key="ticker_mode")
    if select_mode == "Preset List":
        selected_ticker = st.selectbox("Select Asset", preset_tickers)
    else:
        selected_ticker = st.text_input("Enter Ticker Symbol", key="custom_ticker").strip().upper()

    watchlist_options = list(dict.fromkeys(preset_tickers + [selected_ticker]))
    default_watchlist = [selected_ticker] if select_mode == "Custom Input" else preset_tickers[:5]
    watchlist_selection = st.multiselect(
        "Watchlist",
        watchlist_options,
        default=default_watchlist,
        max_selections=8,
    )

with st.sidebar.expander("Strategy", expanded=True):
    analysis_mode = st.radio(
        "Horizon",
        ["Weekly (Swing/Position)", "Intra-Day (Scalp/Day Trade)"],
        index=0,
    )
    default_tf_index = 0 if "Intra-Day" in analysis_mode else 3
    timeframe = st.selectbox(
        "Chart Timeframe",
        ["5m", "15m", "1h", "1d", "1w"],
        index=default_tf_index,
    )

with st.sidebar.expander("Backtest", expanded=False):
    backtest_holding_period = st.number_input(
        "Holding Period (bars)", min_value=1, max_value=100, value=5
    )
    backtest_cost_pct = st.number_input(
        "Cost + Slippage (%)", min_value=0.0, max_value=5.0, value=0.0, step=0.05
    )

with st.sidebar.expander("Position Sizing", expanded=False):
    account_size = st.number_input(
        "Account Size (USD)", min_value=0.0, value=25_000.0, step=1_000.0, key="account_size"
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)", min_value=0.0, max_value=10.0, value=1.0, step=0.25, key="risk_pct",
        help="Share counts are sized so a stop-out loses this percentage of the account.",
    )

with st.sidebar.expander("Data", expanded=False):
    st.caption("Cached market data refreshes automatically. Use this control after a provider outage.")
    refresh_data = st.button("Refresh Market Data", width="stretch")

if refresh_data:
    get_technical_data.clear()
    get_multi_timeframe_data.clear()
    get_ticker_news_sentiment.clear()
    get_macro_market_trend.clear()
    get_upcoming_events.clear()
    get_watchlist_snapshot.clear()
    get_market_movers.clear()
    get_stock_profile.clear()
    get_live_price.clear()
    get_smallcap_universe.clear()
    scan_smallcap_breakouts.clear()
    get_candidate_context.clear()
    get_smallcap_regime.clear()
    st.rerun()

# Reset analysis state if user changes the ticker
if selected_ticker != st.session_state.last_analyzed_ticker:
    st.session_state.llm_analysis = None
    st.session_state.last_analyzed_ticker = selected_ticker

# Load Chart, Sentiment, Macro Trend & Upcoming Events Data
with st.spinner(f"Loading market data & events for {selected_ticker}..."):
    df_chart = get_technical_data(selected_ticker, timeframe=timeframe)
    news_sentiment, sentiment_summary = get_ticker_news_sentiment(selected_ticker)
    macro_trend = get_macro_market_trend()
    event_data = get_upcoming_events(selected_ticker)

if df_chart is None or df_chart.empty:
    st.error(f"No price data available for {selected_ticker} on timeframe {timeframe}. Check ticker or market hours.")
    st.stop()

data_source = df_chart.attrs.get("data_source", "Unknown provider")
latest_candle_timestamp = df_chart.index[-1]
backtest_data = df_chart.copy(deep=True)
if hasattr(latest_candle_timestamp, "strftime"):
    latest_candle_timestamp = latest_candle_timestamp.strftime("%Y-%m-%d %H:%M:%S")

news_status = "READY" if not news_sentiment.startswith("Neutral (Error") and not news_sentiment.startswith("Neutral (No News") else "NO DATA"
macro_status = "READY" if not macro_trend.startswith("Neutral (Error") and not macro_trend.startswith("Neutral (Insufficient") else "NO DATA"
events_status = "READY" if event_data.get("earnings_date") != "N/A" or event_data.get("news_headlines") else "NO DATA"
ai_status = "READY" if is_ai_configured() else "CONFIGURE KEY"

status_items = [
    ("Price Feed", data_source, "status-ready"),
    ("Latest Candle", latest_candle_timestamp, "status-ready"),
    ("News Feed", news_status, "status-ready" if news_status == "READY" else "status-warning"),
    ("Macro & Events", "READY" if macro_status == "READY" and events_status == "READY" else "PARTIAL", "status-ready" if macro_status == "READY" and events_status == "READY" else "status-warning"),
    ("AI Analysis", ai_status, "status-ready" if ai_status == "READY" else "status-warning"),
]
status_markup = "".join(
    f'<div><div class="status-label">{label}</div><strong class="status-value {css_class}">{value}</strong></div>'
    for label, value, css_class in status_items
)
st.markdown(
    f'<div class="system-status-bar">{status_markup}</div>',
    unsafe_allow_html=True,
)


dashboard_tab, scanner_tab = st.tabs(
    [DASHBOARD_TAB, SCANNER_TAB], key="main_view", on_change="rerun"
)

# The scanner renders first so a fresh scan feeds the chart overlay in the dashboard tab
with scanner_tab:
    scan_title_col, scan_btn_col = st.columns([3, 1])
    with scan_title_col:
        st.subheader("🚀 Small-Cap Breakout Scanner")
        st.caption(
            f"US stocks with a {MIN_MARKET_CAP / 1e6:,.0f}M-{MAX_MARKET_CAP / 1e9:,.0f}B USD market cap and at least "
            f"{MIN_DOLLAR_VOLUME / 1e6:,.0f}M USD average daily dollar volume, in a 3-8 week base within 8% of the pivot, "
            f"above the 50-day average, with a measured-move reward/risk of {MIN_REWARD_RISK:.0f}:1 or better."
        )
    with scan_btn_col:
        if st.button("🔎 Run Breakout Scan", width="stretch"):
            with st.spinner("Screening small caps and measuring bases (this can take up to a minute)..."):
                try:
                    st.session_state.scanner_results = scan_smallcap_breakouts()
                    st.session_state.scanner_error = None
                except Exception as error:
                    st.session_state.scanner_results = None
                    st.session_state.scanner_error = str(error)
            st.session_state.scanner_reviews = {}
            st.session_state.scanner_review_error = None

    regime = get_smallcap_regime()
    regime_detail = (
        f"IWM {regime['price']:.2f} | 50-day {regime['sma_50']:.2f} ({'rising' if regime['sma_50_rising'] else 'falling'}) "
        f"| 200-day {regime['sma_200']:.2f}"
        if regime["label"] != "Unknown" else "IWM history unavailable"
    )
    if regime["label"] == "Uptrend":
        st.success(f"**Small-cap trend: Uptrend.** Breakouts have the market behind them. {regime_detail}")
    elif regime["label"] == "Downtrend":
        st.error(
            f"**Small-cap trend: Downtrend.** Most breakouts fail in a falling small-cap market; "
            f"consider smaller size or waiting. {regime_detail}"
        )
    elif regime["label"] == "Mixed":
        st.warning(f"**Small-cap trend: Mixed.** Be selective and keep size modest. {regime_detail}")
    else:
        st.caption(f"Small-cap trend unavailable: {regime_detail}.")

    scan_results = st.session_state.scanner_results
    if scan_results is not None and scan_results.attrs.get("universe_source") == "fallback list":
        st.warning(
            "No small-cap screener answered, so this scan used a built-in list of "
            f"{scan_results.attrs.get('universe_size', 'N/A')} small caps whose market caps are not re-checked. "
            f"Screener error: {scan_results.attrs.get('screener_error')}"
        )
    if scan_results is None and st.session_state.get("scanner_error"):
        st.error(f"Breakout scan failed: {st.session_state.scanner_error}")
    elif scan_results is None:
        st.info("Run the scan to find small caps setting up for a breakout. Results are cached for 15 minutes.")
    elif scan_results.empty:
        st.warning(scan_results.attrs.get("error") or "No small caps currently meet the breakout and reward/risk criteria.")
    else:
        st.caption(
            f"{len(scan_results)} setups | {scan_results.attrs.get('analyzed', 'N/A')} of "
            f"{scan_results.attrs.get('universe_size', 'N/A')} stocks analyzed "
            f"(from {scan_results.attrs.get('universe_source', 'N/A')}) | "
            f"{scan_results.attrs.get('scan_timestamp', '')} | Click a row to load it into the dashboard."
        )
        reviews = st.session_state.scanner_reviews
        display_df = scan_results.copy()
        display_df.insert(1, "AI Grade", [reviews.get(t, {}).get("grade", "") for t in display_df["Ticker"]])
        display_df.insert(2, "AI Risk", [reviews.get(t, {}).get("risk_level", "") for t in display_df["Ticker"]])
        display_df["Market Cap"] = display_df["Market Cap"].apply(
            lambda value: value / 1e6 if isinstance(value, (int, float)) else None
        )
        sizes = [
            position_size(account_size, risk_pct, row["Pivot"], row["Stop"])
            for _, row in scan_results.iterrows()
        ]
        display_df["Shares"] = [size["shares"] if size else None for size in sizes]
        display_df["Position"] = [size["position_value"] if size else None for size in sizes]
        st.session_state.scanner_display_tickers = list(display_df["Ticker"])
        st.dataframe(
            display_df,
            key="breakout_table",
            on_select=load_scanner_ticker,
            selection_mode="single-row",
            hide_index=True,
            width="stretch",
            height=min(39 * (len(display_df) + 1), 460),
            column_config={
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Market Cap": st.column_config.NumberColumn("Mkt Cap", format="$%.0fM"),
                "Pivot": st.column_config.NumberColumn("Pivot", format="$%.2f"),
                "To Pivot": st.column_config.NumberColumn("To Pivot", format="%.2f%%"),
                "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
                "Reward/Risk": st.column_config.NumberColumn("R:R", format="%.1f"),
                "Base Depth": st.column_config.NumberColumn("Base Depth", format="%.1f%%"),
                "RVOL": st.column_config.NumberColumn("RVOL", format="%.2fx"),
                "RS vs IWM": st.column_config.NumberColumn("RS vs IWM (3M)", format="%.1f%%"),
                "Breakout Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                "Shares": st.column_config.NumberColumn(
                    "Shares", format="%d",
                    help=f"Buy-stop at the pivot, sized to risk {risk_pct:.2f}% of a {account_size:,.0f} USD account.",
                ),
                "Position": st.column_config.NumberColumn("Position", format="$%.0f"),
            },
        )
        st.caption(
            f"Shares risk {risk_pct:.2f}% of a {account_size:,.0f} USD account from pivot to stop "
            "(change it under Position Sizing in the sidebar)."
        )

        review_count = min(8, len(scan_results))
        review_col, review_note_col = st.columns([1, 3])
        with review_col:
            run_review = st.button(
                f"🤖 AI Review Top {review_count}",
                width="stretch",
                disabled=not is_ai_configured(),
            )
        with review_note_col:
            st.caption(
                "Grades the top setups for catalysts and small-cap red flags (dilution, reverse splits, "
                "earnings inside the breakout window) in a single Gemini request."
                if is_ai_configured() else "AI review needs GEMINI_API_KEY in Streamlit secrets or the environment."
            )
        if run_review:
            candidates = scan_results.head(review_count).to_dict("records")
            with st.spinner("Collecting float, short interest and headlines for the top candidates..."):
                contexts = {c["Ticker"]: get_candidate_context(c["Ticker"]) for c in candidates}
            with st.spinner("Running AI review..."):
                review_result = review_breakout_candidates(candidates, contexts)
            st.session_state.scanner_reviews = review_result["reviews"]
            st.session_state.scanner_review_error = review_result["error"]
            st.rerun()

        if st.session_state.scanner_review_error:
            st.error(f"AI review failed: {st.session_state.scanner_review_error}")
        for ticker in scan_results["Ticker"]:
            review = reviews.get(ticker)
            if not review:
                continue
            with st.expander(f"{ticker} | Grade {review['grade']} | Risk {review['risk_level']} | {review['catalyst']}"):
                st.markdown(sanitize_ai_text(review["thesis"]) or "No thesis provided.")
                if review["red_flags"]:
                    st.markdown("**Red flags:**\n" + "\n".join(f"- {sanitize_ai_text(flag)}" for flag in review["red_flags"]))
                else:
                    st.caption("No red flags identified from the available data.")
        st.caption("Screening output is informational, not investment advice. Small-cap breakouts fail often; size positions from the stop.")

    st.divider()
    st.subheader("🧪 Scanner Rules Backtest")
    st.caption(
        "Replays the scanner's rules day by day over the last 3 years for the 150 most liquid stocks in the "
        "current universe: buy-stop at the pivot within 10 days, exit at the stop, the target, or after 40 days. "
        "Results are in R (multiples of the initial risk). Today's universe leaves out stocks that have since "
        "been delisted, so the numbers flatter the rules somewhat."
    )
    if st.button("🧪 Run Backtest", key="run_scanner_backtest"):
        with st.spinner("Downloading 3 years of daily data and replaying the rules (this can take a minute or two)..."):
            try:
                st.session_state.scanner_backtest = backtest_smallcap_breakouts()
                st.session_state.scanner_backtest_error = None
            except Exception as error:
                st.session_state.scanner_backtest = None
                st.session_state.scanner_backtest_error = str(error)
    if st.session_state.get("scanner_backtest_error"):
        st.error(f"Backtest failed: {st.session_state.scanner_backtest_error}")
    backtest_result = st.session_state.get("scanner_backtest")
    if backtest_result:
        summary = backtest_result["summary"]
        if summary["trades"] == 0:
            st.info("No closed trades: the rules did not trigger in the tested history.")
        else:
            bt1, bt2, bt3, bt4 = st.columns(4)
            bt1.metric("Closed Trades", summary["trades"])
            bt2.metric("Win Rate", f"{summary['win_rate_pct']}%")
            bt3.metric("Average R", f"{summary['average_r']:+.2f}R")
            bt4.metric("Total R", f"{summary['total_r']:+.1f}R")
            st.caption(
                f"Exits: {summary['target_pct']}% target, {summary['stop_pct']}% stop, "
                f"{summary['time_exit_pct']}% time exit | {summary['still_open']} trades still open | "
                f"{backtest_result['tickers_tested']} stocks from {backtest_result['universe_source']} | "
                f"run {backtest_result['run_timestamp']}"
            )
            if backtest_result["by_regime"]:
                regime_rows = [
                    {"IWM on signal day": name, "Trades": stats["trades"], "Win Rate": stats["win_rate_pct"],
                     "Average R": stats["average_r"], "Total R": stats["total_r"]}
                    for name, stats in backtest_result["by_regime"].items()
                ]
                st.dataframe(
                    pd.DataFrame(regime_rows), hide_index=True, width="stretch",
                    column_config={
                        "Win Rate": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                        "Average R": st.column_config.NumberColumn("Average R", format="%+.2f"),
                        "Total R": st.column_config.NumberColumn("Total R", format="%+.1f"),
                    },
                )
            with st.expander(f"All {len(backtest_result['trades'])} trades", expanded=False):
                st.dataframe(backtest_result["trades"], hide_index=True, width="stretch")

with dashboard_tab:
    watchlist = normalize_watchlist(watchlist_selection or [selected_ticker])
    with st.spinner("Loading watchlist and market movers..."):
        watchlist_df = get_watchlist_snapshot(watchlist, timeframe=timeframe)
        mover_universe = list(dict.fromkeys(MOVER_UNIVERSE + [selected_ticker]))
        movers_df = get_market_movers(mover_universe)

    st.markdown("### Dashboard Views")
    view_columns = st.columns(5, gap="small")
    view_links = [
        ("Watchlist", "#watchlist-overview"),
        ("Technicals", "#technical-chart"),
        ("Catalysts", "#catalysts"),
        ("Backtest", "#backtest"),
        ("AI Analysis", "#ai-analysis"),
    ]
    for column, (label, anchor) in zip(view_columns, view_links):
        with column:
            st.markdown(f'<a class="view-button" href="{anchor}">{label}</a>', unsafe_allow_html=True)

    watchlist_col, movers_col = st.columns(2, gap="large")
    with watchlist_col:
        st.markdown('<div id="watchlist-overview"></div>', unsafe_allow_html=True)
        st.subheader("📋 Watchlist Overview")
        st.caption(f"Selected asset: {selected_ticker} | {len(watchlist)} symbols tracked")
        watchlist_height = max(74, 39 * (len(watchlist_df) + 1))
        st.dataframe(
            watchlist_df.drop(columns=["Focus"], errors="ignore"),
            hide_index=True,
            height=watchlist_height,
            width="stretch",
            column_config={
                "Focus": st.column_config.TextColumn("", width="small"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "Change": st.column_config.NumberColumn("Today", format="%.2f%%"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "TA Score": st.column_config.NumberColumn("TA Score", format="%d"),
            },
        )

    with movers_col:
        st.markdown('<div id="market-movers"></div>', unsafe_allow_html=True)
        st.subheader("⚡ Top Market Movers")
        scan_timestamp = movers_df.attrs.get("scan_timestamp", "Unavailable")
        universe_size = movers_df.attrs.get("universe_size", len(mover_universe))
        st.caption(f"Positive relative movers | {universe_size} stocks scanned | {scan_timestamp}")
        if movers_df.empty:
            st.info("No positive relative movers are available for the current market session.")
        else:
            movers_height = max(74, 39 * (len(movers_df) + 1))
            st.dataframe(
                movers_df,
                hide_index=True,
                height=movers_height,
                width="stretch",
                column_config={
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Change": st.column_config.NumberColumn("Change", format="%.2f%%"),
                    "Relative Gain": st.column_config.NumberColumn("Vs SPY", format="%.2f%%"),
                    "Day Range": st.column_config.NumberColumn("Day Range", format="%.2f%%"),
                    "RVOL": st.column_config.NumberColumn("RVOL", format="%.2fx"),
                    "Mover Score": st.column_config.NumberColumn("Score", format="%.1f"),
                },
            )

    st.divider()
    st.markdown('<div id="selected-asset"></div>', unsafe_allow_html=True)
    st.subheader("Selected Asset")
    stock_profile = get_stock_profile(selected_ticker)
    profile_columns = st.columns([1.2, 2.0, 1.3, 1.5, 1.2])
    profile_items = [
        ("Ticker", stock_profile["ticker"]),
        ("Company", stock_profile["name"]),
        ("Sector", stock_profile["sector"]),
        ("Industry", stock_profile["industry"]),
        ("Exchange", stock_profile["exchange"]),
    ]
    for column, (label, value) in zip(profile_columns, profile_items):
        with column:
            st.markdown(
                f'<div class="profile-label">{escape(label)}</div>'
                f'<div class="profile-value">{escape(str(value))}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="profile-description-spacer"></div>', unsafe_allow_html=True)
    with st.expander("Business description", expanded=False):
        st.write(format_profile_summary(stock_profile))

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
    market_session = get_market_session(selected_ticker)
    session_labels = {
        "REGULAR": ("LIVE", "badge-cyan"),
        "PRE": ("PRE-MARKET", "badge-neutral"),
        "POST": ("AFTER-HOURS", "badge-neutral"),
        "CLOSED": ("CLOSED", "badge-neutral"),
    }
    has_live_quote = raw_live_price > 0 and not pd.isna(raw_live_price) and market_session != "CLOSED"

    if has_live_quote:
        latest_price = raw_live_price
        session_label, price_badge_class = session_labels[market_session]
        price_badge_text = f"{session_label} • {selected_ticker}"

        # Only a current candle takes the quote, and daily/weekly bars only during the regular session
        if should_apply_live_price(df_chart.index[-1], timeframe, market_session):
            df_chart.iloc[-1, df_chart.columns.get_loc('Close')] = latest_price
            df_chart.iloc[-1, df_chart.columns.get_loc('High')] = max(df_chart['High'].iloc[-1], latest_price)
            df_chart.iloc[-1, df_chart.columns.get_loc('Low')] = min(df_chart['Low'].iloc[-1], latest_price)

            # Recalculate indicators so RSI & EMAs on chart match the live price
            df_chart = add_technical_indicators(df_chart)
    else:
        latest_price = float(df_chart['Close'].iloc[-1])
        price_badge_text = f"CLOSED • {selected_ticker}"
        price_badge_class = "badge-neutral"

    st.caption(
        f"Historical feed: {data_source} | Latest candle: {latest_candle_timestamp} "
        f"| Price status: {price_badge_text}"
    )

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

    st.markdown('<div id="technical-chart"></div>', unsafe_allow_html=True)
    # Interactive Candlestick Chart with Volume
    st.subheader(f"📊 Technical Chart ({timeframe}) — {selected_ticker} [{analysis_mode}]")

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

    # Identify missing dates to remove gaps (ONLY for Intraday timeframes)
    freq_map = {"5m": "5min", "15m": "15min", "1h": "1h"}
    dvalue_map = {"5m": 300000, "15m": 900000, "1h": 3600000}

    if timeframe in freq_map:
        full_idx = pd.date_range(start=df_chart.index.min(), end=df_chart.index.max(), freq=freq_map[timeframe])
        missing_dt = full_idx.difference(df_chart.index)
    
        fig.update_xaxes(
            rangebreaks=[dict(values=missing_dt, dvalue=dvalue_map[timeframe])]
        )

    # Subdued gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#1E2532', row=1, col=1)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#1E2532', row=1, col=1)
    fig.update_xaxes(showgrid=False, row=2, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)

    # Overlay breakout levels when the selected asset came from the scanner
    if scan_results is not None and not scan_results.empty and selected_ticker in set(scan_results["Ticker"]):
        breakout_row = scan_results.loc[scan_results["Ticker"] == selected_ticker].iloc[0]
        for level, label, color in (
            ("Pivot", "Pivot", "#00F0FF"),
            ("Stop", "Stop", "#ff0055"),
            ("Target", "Target", "#00ff88"),
        ):
            fig.add_hline(
                y=breakout_row[level], line_dash="dash", line_color=color, line_width=1,
                annotation_text=f"{label} {breakout_row[level]:.2f}", annotation_font_color=color,
                row=1, col=1,
            )

    st.plotly_chart(fig, width="stretch")

    # ==============================================================================
    # UPCOMING EVENTS & MACRO CATALYST SECTION
    # ==============================================================================
    st.markdown('<div id="catalysts"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("🌐 Catalysts & Macro Economic Environment")

    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Upcoming Earnings Date", event_data.get("earnings_date", "N/A"))
    with col_e2:
        st.metric("Market Volatility (VIX)", f"{event_data.get('macro_vix', 0.0)}")
    with col_e3:
        st.metric("10Y Treasury Yield (^TNX)", f"{event_data.get('macro_tnx', 0.0)}%")

    days_until_earnings = event_data.get("days_until_earnings")
    proximity_flag = event_data.get("proximity_flag")
    if proximity_flag == "IMMEDIATE_BINARY_RISK":
        st.error(
            f"⚠️ Earnings in {days_until_earnings} day(s) ({event_data.get('earnings_date')}). "
            "Holding through the release is a binary event: consider closing or hedging with defined-risk structures."
        )
    elif proximity_flag == "SWING_WINDOW_OVERLAP":
        st.warning(
            f"⏳ Earnings in {days_until_earnings} days ({event_data.get('earnings_date')}) overlap a typical swing window. "
            "Plan the exit before the report and expect IV expansion followed by IV crush."
        )

    with st.expander("📰 Recent Catalyst Headlines", expanded=False):
        if event_data.get("news_headlines"):
            for headline in event_data["news_headlines"]:
                st.markdown(headline)
        else:
            st.caption("No recent headlines are available for this ticker.")

    st.divider()
    st.markdown('<div id="backtest"></div>', unsafe_allow_html=True)
    st.subheader("📈 Historical TA Signal Check")
    st.caption(
        "Fixed-horizon backtest of the deterministic TA score. "
        f"Holding period: {backtest_holding_period} bars | Estimated costs: {backtest_cost_pct:.2f}% | AI decisions excluded."
    )
    backtest = run_ta_backtest(
        backtest_data,
        holding_period=backtest_holding_period,
        cost_per_trade_pct=backtest_cost_pct,
    )
    if backtest["total_trades"] == 0:
        st.info("No qualifying historical TA signals were found in the loaded timeframe.")
    else:
        bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
        bt_col1.metric("Trades", backtest["total_trades"])
        bt_col2.metric("Win Rate", f"{backtest['win_rate_pct']}%")
        bt_col3.metric("Cumulative Return", f"{backtest['cumulative_return_pct']}%")
        bt_col4.metric("Max Drawdown", f"{backtest['max_drawdown_pct']}%")

    st.divider()

    # Callback to run multi-timeframe LLM synthesis cleanly with Macro Context
    def run_synthesis_callback():
        with st.spinner("Fetching multi-horizon data, options OI & macro signals..."):
            df_5m, df_4h, df_1d = get_multi_timeframe_data(selected_ticker)
            df_1w = get_technical_data(selected_ticker, timeframe="1w")
            options_data = get_options_sentiment(selected_ticker, analysis_mode=analysis_mode)
            swing_metrics = get_swing_metrics(selected_ticker, analysis_mode=analysis_mode)
            intraday_metrics = get_intraday_metrics(selected_ticker)

            st.session_state.llm_analysis = synthesize_signals(
                ticker=selected_ticker,
                df_5m=df_5m,
                df_4h=df_4h,
                df_1d=df_1d,
                df_1w=df_1w,
                sentiment_summary=sentiment_summary,
                event_data=event_data,
                options_data=options_data,
                swing_metrics=swing_metrics,
                intraday_metrics=intraday_metrics,
                analysis_mode=analysis_mode,
            )

    # Section: AI Synthesis Control
    st.markdown('<div id="ai-analysis"></div>', unsafe_allow_html=True)
    col_title, col_btn = st.columns([3, 1])

    with col_title:
        st.subheader("🤖 Multi-Timeframe AI Synthesis")

        if not is_ai_configured():
            st.warning("AI analysis is disabled: configure GEMINI_API_KEY in Streamlit secrets or the environment.")

    with col_btn:
        btn_label = "🔄 Regenerate Analysis" if st.session_state.llm_analysis else "🚀 Run AI Analysis"
        st.button(btn_label, on_click=run_synthesis_callback, width="stretch")

    # Render Multi-Factor Deep AI Results
    if st.session_state.llm_analysis:
        res = st.session_state.llm_analysis
    
        if isinstance(res, dict):
            signal = str(res.get('signal', 'HOLD')).upper()
            confidence = int(res.get('confidence', 0) * 100) if res.get('confidence', 0) <= 1 else int(res.get('confidence', 0))
            alignment = res.get('timeframe_confluence', 'N/A')
        
            # Color coding for Signal Banner
            if signal == "BUY":
                badge_class = "badge-bullish"
                glow_color = "rgba(0, 255, 136, 0.15)"
                border_color = "#00ff88"
                icon = "🟢"
            elif signal == "SELL":
                badge_class = "badge-bearish"
                glow_color = "rgba(255, 0, 85, 0.15)"
                border_color = "#ff0055"
                icon = "🔴"
            else:
                badge_class = "badge-neutral"
                glow_color = "rgba(160, 174, 192, 0.15)"
                border_color = "#A0AEC0"
                icon = "🟡"

            # --- Top Level Signal Summary Banner ---
            st.markdown(f"""
        <div style="background: {glow_color}; border: 1px solid {border_color}; border-radius: 12px; padding: 18px 24px; margin-bottom: 25px;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">
                    {icon} Signal: <span style="color: {border_color};">{signal}</span> 
                    <span style="font-size: 1rem; color: #A0AEC0; font-weight: 400; margin-left: 15px;">Confidence: <strong>{confidence}%</strong></span>
                </div>
                <div>
                    <span class="kpi-badge {badge_class}" style="font-size: 0.85rem; padding: 6px 14px;">Alignment: {alignment}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
            ## --- Institutional Execution Plan Cards ---
            plan = res.get('execution_plan', {})
            st.markdown("### 🎯 Trade Execution Plan")
            plan_issues = check_execution_plan(res, latest_price)
            if plan_issues:
                st.warning(
                    "**Plan check failed.** Treat these levels with caution:\n"
                    + "\n".join(f"- {issue}" for issue in plan_issues)
                )
            elif signal in ("BUY", "SELL"):
                st.caption(f"Plan check passed: levels are ordered for a {signal} and sit near the current price.")
        
            tp_val = plan.get('take_profit', 0.0)
            sl_val = plan.get('stop_loss', 0.0)
            up_limit = plan.get('swing_upper_limit', 'N/A')
            low_limit = plan.get('swing_lower_limit', 'N/A')
        
            # Row 1: Core Trade Targets
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.metric("Target Entry Zone", f"${plan.get('entry_zone', 'N/A')}")
            with p_col2:
                st.metric("Take Profit Target", f"${tp_val:.2f}" if isinstance(tp_val, (int, float)) else str(tp_val))
            with p_col3:
                st.metric("Stop Loss Level", f"${sl_val:.2f}" if isinstance(sl_val, (int, float)) else str(sl_val))
            with p_col4:
                st.metric("Risk / Reward Ratio", str(plan.get('risk_reward_ratio', 'N/A')))

            plan_entry, plan_stop = parse_plan_level(plan.get('entry_zone')), parse_plan_level(plan.get('stop_loss'))
            plan_size = (
                position_size(account_size, risk_pct, plan_entry, plan_stop)
                if signal in ("BUY", "SELL") and not plan_issues and plan_entry and plan_stop else None
            )
            if plan_size:
                st.caption(
                    f"Position size: **{plan_size['shares']:,} shares** ({plan_size['position_value']:,.0f} USD), "
                    f"risking {plan_size['dollar_risk']:,.0f} USD ({risk_pct:.2f}% of {account_size:,.0f} USD) "
                    f"from entry {plan_entry:.2f} to stop {plan_stop:.2f}"
                    + (". Capped at the account size, so the risk is below budget." if plan_size["capped_by_account"] else ".")
                )
            
            # Row 2: Expected Move Swing Limits
            l_col1, l_col2, l_col3, l_col4 = st.columns(4)
            with l_col1:
                st.metric("Swing Lower Bound (1SD)", f"${low_limit:.2f}" if isinstance(low_limit, (int, float)) else str(low_limit))
            with l_col2:
                st.metric("Swing Upper Bound (1SD)", f"${up_limit:.2f}" if isinstance(up_limit, (int, float)) else str(up_limit))
        
            st.markdown("<br>", unsafe_allow_html=True)
        
            # --- Multi-Factor Breakdown Section ---
            st.markdown("### 🔬 Multi-Factor Analysis Breakdown")
        
            tab_tech, tab_macro, tab_news, tab_scenarios = st.tabs([
                "📊 Technical Structure",
                "🌐 Macro Regime & Risk",
                "📰 Catalysts & Headlines",
                "🎲 Catalyst Scenarios"
            ])
        
            with tab_tech:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.markdown("**Higher Timeframe (Macro Trend):**")
                    st.markdown(sanitize_ai_text(res.get('higher_tf_breakdown', 'N/A')))
                with t_col2:
                    st.markdown("**Intraday Setup (Trigger):**")
                    st.markdown(sanitize_ai_text(res.get('intraday_tf_breakdown', 'N/A')))
            
                st.markdown("---")
                s_col1, s_col2 = st.columns(2)
                supp_val = plan.get('key_support', 0.0)
                rest_val = plan.get('key_resistance', 0.0)
                s_col1.metric("Key Technical Support", f"${supp_val:.2f}" if isinstance(supp_val, (int, float)) else str(supp_val))
                s_col2.metric("Key Technical Resistance", f"${rest_val:.2f}" if isinstance(rest_val, (int, float)) else str(rest_val))

            with tab_macro:
                st.markdown(sanitize_ai_text(res.get('macro_analysis', 'N/A')))

            with tab_news:
                st.markdown(sanitize_ai_text(res.get('news_catalyst_analysis', 'N/A')))

            with tab_scenarios:
                st.markdown(sanitize_ai_text(res.get('catalyst_scenarios', 'N/A')))

            st.markdown("<br>", unsafe_allow_html=True)

            # --- Comprehensive Thesis ---
            with st.expander("📝 View Complete AI Thesis & Strategic Commentary", expanded=True):
                st.markdown(sanitize_ai_text(res.get('detailed_reasoning', 'N/A')))

        else:
            st.markdown(sanitize_ai_text(str(res)))
    else:
        st.info("Click 'Run AI Analysis' above to generate a multi-timeframe unified trade decision.")