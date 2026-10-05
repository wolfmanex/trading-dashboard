import re
from pathlib import Path

import pandas as pd
import streamlit as st
from html import escape
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots

from technical_engine import (
    get_technical_data, get_multi_timeframe_data, get_live_price, add_technical_indicators,
    get_market_session, should_apply_live_price,
)
from news_engine import get_ticker_news_sentiment
from index_filter import get_macro_market_trend
from llm_engine import ask_follow_up, is_ai_configured, review_breakout_candidates, review_trade_setup
from market_conditions import RISK_OFF_SIZE, events_soon, get_market_conditions, summary_line
from journal import (
    LONG, SHORT, JournalStore, add_trade, close_trade, mark_positions, open_positions, summarize_journal, update_stop,
)
from trade_levels import STOP_ATR, TARGET_R, compute_trade_plans
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
from earnings_engine import EARNINGS_WINDOW_DAYS, add_earnings_columns, get_upcoming_earnings
from secrets_config import get_configured_secret
from signal_log import (
    LOG_URL, OPEN, WAITING, fetch_published_log, live_pick_states, summarize_by_grade, summarize_log,
)
from scheduled_scan import TOP_N as ALERTED_TOP_N
from breakout_engine import MIN_ATR_PCT, MIN_STOP_PCT
from smallcap_screener import (
    _ticker_frame, get_candidate_context, get_smallcap_universe, scan_smallcap_breakouts,
    get_smallcap_regime, backtest_smallcap_breakouts,
    MIN_MARKET_CAP, MAX_MARKET_CAP, MIN_REWARD_RISK, MIN_DOLLAR_VOLUME,
)


st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Professional Dashboard Custom Styling (Typography & Glow)
st.markdown(
    f"<style>{(Path(__file__).parent / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


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
if "llm_chat" not in st.session_state:
    st.session_state.llm_chat = []


# Helper to prevent Streamlit from treating dollar signs in AI text as LaTeX
def sanitize_ai_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Escapes unescaped $ signs so Streamlit won't parse them as LaTeX math formulas
    return text.replace("$", r"\$")


# Chart colours, kept in step with styles.css and .streamlit/config.toml
CHART_BG = "#FFFFFF"
CHART_INK = "#111418"
CHART_GRID = "#E3E7EC"
CHART_UP = "#00A85A"
CHART_DOWN = "#E8173A"
CHART_BLUE = "#1F3DFF"
CHART_AMBER = "#F59E0B"

DASHBOARD_TAB = "Dashboard"
SCANNER_TAB = "Breakout Scanner"
TRACK_RECORD_TAB = "Track Record"
JOURNAL_TAB = "Journal"


def market_note_or_empty() -> str:
    try:
        return summary_line(get_market_conditions())
    except Exception as error:
        print(f"Market conditions unavailable: {error}")
        return ""


def render_market_panel(conditions: dict) -> None:
    """Market label with its inputs and the macro releases coming up."""
    reasons = ", ".join(conditions["reasons"])
    if conditions["label"] == "Risk-on":
        st.success(f"**Market: Risk-on.** Breakouts have the market behind them. {reasons}.")
    elif conditions["label"] == "Risk-off":
        st.error(
            f"**Market: Risk-off.** Most breakouts fail in this kind of market, so suggested position sizes are cut to "
            f"{RISK_OFF_SIZE:.0%}. {reasons}."
        )
    else:
        st.warning(f"**Market: Neutral.** Be selective and keep size modest. {reasons}.")
    iwm, spy = conditions["iwm"], conditions["spy"]
    m1, m2, m3, m4 = st.columns(4)
    for column, name, regime in ((m1, "IWM", iwm), (m2, "SPY", spy)):
        column.metric(
            f"{name} trend", regime["label"],
            help=(f"{regime['price']} vs 50-day {regime['sma_50']} ({'rising' if regime['sma_50_rising'] else 'falling'}) "
                  f"and 200-day {regime['sma_200']}") if regime["label"] != "Unknown" else "History unavailable",
        )
    m3.metric("Breadth", f"{conditions['breadth']:.0f}%" if conditions["breadth"] is not None else "N/A",
              help="Share of the market-movers universe (large caps) above its 50-day average.")
    m4.metric("VIX", conditions["vix"] if conditions["vix"] is not None else "N/A",
              delta=f"{conditions['vix_change_5d']:+.2f} in 5 days" if conditions["vix_change_5d"] is not None else None,
              delta_color="inverse")
    soon = {(event["date"], event["event"]) for event in events_soon(conditions["events"])}
    if conditions["events"]:
        st.markdown("**Macro calendar, next two weeks:** " + " | ".join(
            (f"**{event['event']} {event['date']}**" if (event["date"], event["event"]) in soon else f"{event['event']} {event['date']}")
            for event in conditions["events"]
        ))
        if soon:
            st.caption("Bold releases are within two days: a breakout bought now rides on the number.")
    else:
        st.caption("No major US releases in the next two weeks.")
    st.caption(f"Calendar: {conditions['events_source']}." + (f" Price data error: {conditions['error']}" if conditions["error"] else ""))


@st.cache_data(ttl=1800, show_spinner=False)
def load_signal_log():
    """The forward signal log the scheduled scan publishes on the signal-log branch."""
    return fetch_published_log(get_configured_secret("SIGNAL_LOG_URL") or LOG_URL)


@st.cache_data(ttl=60, show_spinner=False)
def latest_closes(tickers: tuple) -> dict:
    """Last price per ticker from daily bars (today's bar moves with the market during the session)."""
    if not tickers:
        return {}
    batch = yf.download(tickers=list(tickers), period="5d", interval="1d", group_by="ticker", auto_adjust=False, progress=False)
    prices = {}
    for ticker in tickers:
        frame = _ticker_frame(batch, ticker)
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna() if "Close" in frame else pd.Series(dtype=float)
        if not close.empty:
            prices[ticker] = float(close.iloc[-1])
    return prices


def load_scanner_ticker():
    """Switch the dashboard to the breakout candidate clicked in the scanner table."""
    selection = st.session_state.get("breakout_table")
    rows = selection.selection.rows if selection else []
    tickers = st.session_state.get("scanner_display_tickers", [])
    if rows and rows[0] < len(tickers):
        st.session_state.ticker_mode = "Custom Input"
        st.session_state.custom_ticker = tickers[rows[0]]
        st.session_state.main_view = DASHBOARD_TAB


st.title("AI Trading Dashboard")

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

    # The watchlist lives in the page URL (?watchlist=AMD,NVDA), so a bookmark brings it back
    if "watchlist" not in st.session_state:
        saved_watchlist = normalize_watchlist(
            symbol for symbol in str(st.query_params.get("watchlist", "")).upper().split(",")
            if re.fullmatch(r"[A-Z0-9.^=\-]{1,12}", symbol.strip())
        )
        st.session_state.watchlist = saved_watchlist or preset_tickers[:5]
    watchlist_options = list(dict.fromkeys(preset_tickers + st.session_state.watchlist + [selected_ticker]))
    watchlist_selection = st.multiselect(
        "Watchlist",
        watchlist_options,
        key="watchlist",
        max_selections=8,
        help="Saved in the page address: bookmark it to keep this watchlist.",
    )
    if ",".join(watchlist_selection) != st.query_params.get("watchlist", ""):
        st.query_params["watchlist"] = ",".join(watchlist_selection)

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
    get_upcoming_earnings.clear()
    get_market_conditions.clear()
    st.rerun()

# Reset analysis state if user changes the ticker
if selected_ticker != st.session_state.last_analyzed_ticker:
    st.session_state.llm_analysis = None
    st.session_state.llm_chat = []
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


dashboard_tab, scanner_tab, track_tab, journal_tab = st.tabs(
    [DASHBOARD_TAB, SCANNER_TAB, TRACK_RECORD_TAB, JOURNAL_TAB], key="main_view", on_change="rerun"
)

# Only the open tab runs, so working in the scanner doesn't reload the dashboard's data (and vice versa)
scan_results = st.session_state.scanner_results

with scanner_tab:
    if scanner_tab.open:
        scan_title_col, scan_btn_col = st.columns([3, 1])
        with scan_title_col:
            st.subheader("Small-Cap Breakout Scanner")
            st.caption(
                f"US stocks with a {MIN_MARKET_CAP / 1e6:,.0f}M-{MAX_MARKET_CAP / 1e9:,.0f}B USD market cap and at least "
                f"{MIN_DOLLAR_VOLUME / 1e6:,.0f}M USD average daily dollar volume, in a 3-8 week base within 8% of the pivot, "
                f"above the 50-day average, with a measured-move reward/risk of {MIN_REWARD_RISK:.0f}:1 or better. "
                f"Stops sit at least 1 ATR and {MIN_STOP_PCT:.0%} under the pivot, and stocks moving less than "
                f"{MIN_ATR_PCT:.1%} a day (usually pinned by a pending takeover) are left out."
            )
        with scan_btn_col:
            if st.button("Run Breakout Scan", width="stretch"):
                with st.spinner("Screening small caps and measuring bases (this can take up to a minute)..."):
                    try:
                        st.session_state.scanner_results = scan_smallcap_breakouts()
                        st.session_state.scanner_error = None
                    except Exception as error:
                        st.session_state.scanner_results = None
                        st.session_state.scanner_error = str(error)
                st.session_state.scanner_reviews = {}
                st.session_state.scanner_review_error = None

        with st.spinner("Checking market conditions..."):
            market = get_market_conditions()
        render_market_panel(market)
        size_factor = RISK_OFF_SIZE if market["label"] == "Risk-off" else 1.0

        # Picks from the pre-market scans stay here until their signal finishes, so a stock that breaks
        # out (and so drops off a fresh scan) is still in view.
        try:
            picks_log = load_signal_log()
        except Exception as error:
            picks_log = None
            st.caption(f"Logged picks unavailable: {error}")
        if picks_log is not None and not picks_log.empty:
            live_tickers = tuple(sorted(set(picks_log.loc[picks_log["Status"].isin([WAITING, OPEN]), "Ticker"])))
            picks = live_pick_states(picks_log, latest_closes(live_tickers))
            if not picks.empty:
                st.markdown("**Live picks from the morning scans**")
                st.dataframe(
                    picks, hide_index=True, width="stretch",
                    column_config={
                        "Pivot": st.column_config.NumberColumn("Pivot", format="$%.2f"),
                        "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                        "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
                        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "vs Pivot %": st.column_config.NumberColumn("vs Pivot", format="%+.1f%%"),
                    },
                )
                st.caption(
                    "A fresh scan only lists stocks still under or just over their pivot, so a pick that breaks out "
                    "leaves the scan table but stays here until it stops out, hits target or times out. "
                    "Prices refresh every minute."
                )

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
            try:
                earnings_source, earnings_dates = get_upcoming_earnings()
                earnings_error = None
            except Exception as error:
                earnings_source, earnings_dates, earnings_error = None, {}, str(error)
            scan_view = add_earnings_columns(scan_results, earnings_dates)
            hide_earnings = st.checkbox(
                f"Hide setups that report earnings within {EARNINGS_WINDOW_DAYS} days",
                value=True,
                key="hide_earnings",
                help="A report inside the breakout window turns the trade into a gamble on the numbers.",
            )
            if earnings_error:
                st.caption(f"Earnings dates unavailable, so nothing is hidden: {earnings_error}")
            else:
                reporting = scan_view["Days to ER"].between(0, EARNINGS_WINDOW_DAYS)
                if hide_earnings:
                    scan_view = scan_view[~reporting].reset_index(drop=True)
                st.caption(
                    f"{int(reporting.sum())} of {len(reporting)} setups report earnings within "
                    f"{EARNINGS_WINDOW_DAYS} days{' and are hidden' if hide_earnings else ''} "
                    f"(dates from {earnings_source})."
                )

            if scan_view.empty:
                st.info("Every setup reports earnings soon. Untick the box above to see them.")
            else:
                reviews = st.session_state.scanner_reviews
                display_df = scan_view.copy()
                display_df.insert(1, "AI Grade", [reviews.get(t, {}).get("grade", "") for t in display_df["Ticker"]])
                display_df.insert(2, "AI Risk", [reviews.get(t, {}).get("risk_level", "") for t in display_df["Ticker"]])
                display_df["Market Cap"] = display_df["Market Cap"].apply(
                    lambda value: value / 1e6 if isinstance(value, (int, float)) else None
                )
                sizes = [
                    position_size(account_size, risk_pct * size_factor, row["Pivot"], row["Stop"])
                    for _, row in scan_view.iterrows()
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
                            help=f"Buy-stop at the pivot, sized to risk {risk_pct * size_factor:.2f}% of a {account_size:,.0f} USD account.",
                        ),
                        "Position": st.column_config.NumberColumn("Position", format="$%.0f"),
                        "Earnings": st.column_config.TextColumn("Earnings"),
                        "Days to ER": st.column_config.NumberColumn("Days to ER", format="%d"),
                    },
                )
                st.caption(
                    f"Shares risk {risk_pct * size_factor:.2f}% of a {account_size:,.0f} USD account from pivot to stop "
                    + (f"(your {risk_pct:.2f}% cut to {RISK_OFF_SIZE:.0%} in a Risk-off market; " if size_factor < 1 else "(")
                    + "change it under Position Sizing in the sidebar)."
                )

                review_count = min(8, len(scan_view))
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
                    candidates = scan_view.head(review_count).to_dict("records")
                    with st.spinner("Collecting float, short interest and headlines for the top candidates..."):
                        contexts = {c["Ticker"]: get_candidate_context(c["Ticker"]) for c in candidates}
                    with st.spinner("Running AI review..."):
                        review_result = review_breakout_candidates(candidates, contexts, market_note=summary_line(market))
                    st.session_state.scanner_reviews = review_result["reviews"]
                    st.session_state.scanner_review_error = review_result["error"]
                    st.rerun()

                if st.session_state.scanner_review_error:
                    st.error(f"AI review failed: {st.session_state.scanner_review_error}")
                for ticker in scan_view["Ticker"]:
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
        st.subheader("Scanner Rules Backtest")
        st.caption(
            "Replays the scanner's rules day by day over the last 3 years for the 150 most liquid stocks in the "
            "current universe: buy-stop at the pivot within 10 days, exit at the stop, the target, or after 40 days. "
            "Results are in R (multiples of the initial risk). Today's universe leaves out stocks that have since "
            "been delisted, so the numbers flatter the rules somewhat."
        )
        if st.button("Run Backtest", key="run_scanner_backtest"):
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
    if dashboard_tab.open:
        watchlist = normalize_watchlist(watchlist_selection or [selected_ticker])
        with st.spinner("Loading watchlist and market movers..."):
            watchlist_df = get_watchlist_snapshot(watchlist, timeframe=timeframe)
            mover_universe = list(dict.fromkeys(MOVER_UNIVERSE + [selected_ticker]))
            movers_df = get_market_movers(mover_universe)

        st.subheader("Dashboard Views")
        view_columns = st.columns(5, gap="small")
        view_links = [
            ("Watchlist", "#watchlist-overview"),
            ("Technicals", "#technical-chart"),
            ("Catalysts", "#catalysts"),
            ("Backtest", "#backtest"),
            ("AI Review", "#ai-analysis"),
        ]
        for column, (label, anchor) in zip(view_columns, view_links):
            with column:
                st.markdown(f'<a class="view-button" href="{anchor}">{label}</a>', unsafe_allow_html=True)

        watchlist_col, movers_col = st.columns(2, gap="large")
        with watchlist_col:
            st.markdown('<div id="watchlist-overview"></div>', unsafe_allow_html=True)
            st.subheader("Watchlist Overview")
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
            st.subheader("Top Market Movers")
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
        st.subheader(f"Technical Chart · {selected_ticker} · {timeframe} · {analysis_mode}")

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
            increasing_line_color=CHART_UP, decreasing_line_color=CHART_DOWN,
            increasing_fillcolor=CHART_UP, decreasing_fillcolor=CHART_DOWN
        ), row=1, col=1)

        # Row 1: EMAs
        if 'EMA_9' in df_chart:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_9'], line=dict(color=CHART_BLUE, width=1.5), name="EMA 9"), row=1, col=1)
        if 'EMA_21' in df_chart:
            fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['EMA_21'], line=dict(color=CHART_AMBER, width=2), name="EMA 21"), row=1, col=1)

        # Row 2: Volume Bar Chart
        colors = [CHART_UP if row.Close >= row.Open else CHART_DOWN for index, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(
            x=df_chart.index, y=df_chart['Volume'], name="Volume", marker_color=colors, opacity=0.55
        ), row=2, col=1)

        # Flat light chart to match the boxed page styling
        fig.update_layout(
            template="plotly_white",
            height=650,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis_rangeslider_visible=False,
            plot_bgcolor=CHART_BG,
            paper_bgcolor=CHART_BG,
            font=dict(family="JetBrains Mono, monospace", color=CHART_INK, size=11),
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
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=CHART_GRID, row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=CHART_GRID, row=1, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor=CHART_INK, mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor=CHART_INK, mirror=True)
        fig.update_xaxes(showgrid=False, row=2, col=1)
        fig.update_yaxes(showgrid=False, row=2, col=1)

        # Overlay breakout levels when the selected asset came from the scanner
        scanner_row = None
        if scan_results is not None and not scan_results.empty and selected_ticker in set(scan_results["Ticker"]):
            scanner_row = scan_results.loc[scan_results["Ticker"] == selected_ticker].iloc[0].to_dict()
        if scanner_row:
            for level, label, color in (
                ("Pivot", "Pivot", CHART_BLUE),
                ("Stop", "Stop", CHART_DOWN),
                ("Target", "Target", CHART_UP),
            ):
                fig.add_hline(
                    y=scanner_row[level], line_dash="dash", line_color=color, line_width=1,
                    annotation_text=f"{label} {scanner_row[level]:.2f}", annotation_font_color=color,
                    row=1, col=1,
                )

        st.plotly_chart(fig, width="stretch")

        # ==============================================================================
        # UPCOMING EVENTS & MACRO CATALYST SECTION
        # ==============================================================================
        st.markdown('<div id="catalysts"></div>', unsafe_allow_html=True)
        st.divider()
        st.subheader("Catalysts & Macro Environment")

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

        if st.toggle("Show market conditions and macro calendar", key="show_market_panel"):
            render_market_panel(get_market_conditions())

        with st.expander("Recent Catalyst Headlines", expanded=False):
            if event_data.get("news_headlines"):
                for headline in event_data["news_headlines"]:
                    st.markdown(headline)
            else:
                st.caption("No recent headlines are available for this ticker.")

        st.divider()
        st.markdown('<div id="backtest"></div>', unsafe_allow_html=True)
        st.subheader("Historical TA Signal Check")
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
            bt_col1, bt_col2, bt_col3, bt_col4, bt_col5 = st.columns(5)
            bt_col1.metric("Trades", backtest["total_trades"])
            bt_col2.metric("Win Rate", f"{backtest['win_rate_pct']}%")
            buy_hold = backtest["buy_hold_return_pct"]
            bt_col3.metric(
                "Cumulative Return", f"{backtest['cumulative_return_pct']}%",
                delta=(
                    f"{backtest['cumulative_return_pct'] - buy_hold:+.2f} pts vs buy & hold"
                    if isinstance(buy_hold, (int, float)) else None
                ),
            )
            bt_col4.metric("Buy & Hold", f"{buy_hold}%")
            bt_col5.metric("Max Drawdown", f"{backtest['max_drawdown_pct']}%")
            st.caption(
                f"Average per trade: {backtest['average_return_pct']}% | "
                f"Long: {backtest['long_trades']} trades, {backtest['long_average_return_pct']}% avg | "
                f"Short: {backtest['short_trades']} trades, {backtest['short_average_return_pct']}% avg | "
                "Buy & hold covers the same loaded bars."
            )

        st.divider()

        # The reviewer judges levels calculated in code (trade_levels.py); it never sets prices itself
        def run_review_callback():
            with st.spinner("Fetching multi-timeframe data, options and session levels for the AI review..."):
                df_5m, df_4h, df_1d = get_multi_timeframe_data(selected_ticker)
                df_1w = get_technical_data(selected_ticker, timeframe="1w")
                options_data = get_options_sentiment(selected_ticker, analysis_mode=analysis_mode)
                swing_metrics = get_swing_metrics(selected_ticker, analysis_mode=analysis_mode)
                intraday_metrics = get_intraday_metrics(selected_ticker)
                scanner_context = get_candidate_context(selected_ticker) if scanner_row else None
                levels = compute_trade_plans(latest_price, df_1d, analysis_mode, scanner_row)
                market_note = market_note_or_empty()

            with st.spinner("Running AI review..."):
                result = review_trade_setup(
                    ticker=selected_ticker,
                    price=latest_price,
                    levels=levels,
                    analysis_mode=analysis_mode,
                    df_5m=df_5m,
                    df_4h=df_4h,
                    df_1d=df_1d,
                    df_1w=df_1w,
                    event_data=event_data,
                    options_data=options_data,
                    swing_metrics=swing_metrics,
                    intraday_metrics=intraday_metrics,
                    scanner_row=scanner_row,
                    scanner_context=scanner_context,
                    market_note=market_note,
                )
            st.session_state.llm_analysis = {**result, "levels": levels, "price": latest_price, "mode": analysis_mode}
            st.session_state.llm_chat = []

        def ask_follow_up_callback():
            question = st.session_state.get("follow_up_question", "").strip()
            analysis = st.session_state.llm_analysis
            if not question or not analysis or not analysis.get("review"):
                return
            with st.spinner("Asking the AI..."):
                answer = ask_follow_up(analysis["prompt"], analysis["review"], st.session_state.llm_chat, question)
            st.session_state.llm_chat.append({"question": question, "answer": answer["answer"], "error": answer["error"]})
            st.session_state.follow_up_question = ""

        # Section: AI Trade Review
        st.markdown('<div id="ai-analysis"></div>', unsafe_allow_html=True)
        col_title, col_btn = st.columns([3, 1])

        with col_title:
            st.subheader("AI Trade Review")
            st.caption(
                "Entry, stop and target are calculated in code: the scanner's levels for a scanner pick, otherwise "
                f"{STOP_ATR['swing']} daily ATR (swing) or {STOP_ATR['intraday']} ATR (intraday) stops with a "
                f"{TARGET_R:.0f}R target. The AI picks a direction, grades the setup and says go, wait or skip."
            )
            if not is_ai_configured():
                st.warning("AI review is disabled: configure GEMINI_API_KEY in Streamlit secrets or the environment.")

        with col_btn:
            btn_label = "Re-run AI Review" if st.session_state.llm_analysis else "Run AI Review"
            st.button(btn_label, on_click=run_review_callback, width="stretch", disabled=not is_ai_configured())

        analysis = st.session_state.llm_analysis
        if analysis and analysis.get("error"):
            st.error(f"AI review failed, so there is no verdict: {analysis['error']}")
        elif analysis:
            review, levels = analysis["review"], analysis["levels"]
            direction, verdict = review["direction"], review["verdict"]
            if verdict == "GO" and direction == "LONG":
                badge_class, banner_class = "badge-bullish", "signal-bullish"
            elif verdict == "GO" and direction == "SHORT":
                badge_class, banner_class = "badge-bearish", "signal-bearish"
            else:
                badge_class, banner_class = "badge-neutral", "signal-neutral"
            verdict_text = verdict if direction == "NONE" else f"{verdict} {direction}"

            st.markdown(f"""
        <div class="signal-banner {banner_class}">
            <div class="signal-title">
                Verdict: {escape(verdict_text)}
                <span class="signal-confidence">Grade <strong>{escape(review['grade'])}</strong> | Risk <strong>{escape(review['risk_level'])}</strong></span>
            </div>
            <span class="kpi-badge {badge_class}">{escape(review['summary'] or 'No summary provided.')}</span>
        </div>
        """, unsafe_allow_html=True)
            if review["trigger"]:
                st.markdown(f"**Trigger:** {sanitize_ai_text(review['trigger'])}")
            for note in review["validation_notes"]:
                st.caption(note)

            st.subheader("Trade Plan")
            plan = levels.get(direction.lower()) if direction != "NONE" else None
            if plan:
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                p_col1.metric("Entry", f"${plan['entry']:.2f}")
                p_col2.metric("Stop Loss", f"${plan['stop']:.2f}")
                p_col3.metric("Target", f"${plan['target']:.2f}")
                p_col4.metric("Reward / Risk", f"{plan['reward_risk']}:1" if plan["reward_risk"] else "N/A")
                plan_size = position_size(account_size, risk_pct, plan["entry"], plan["stop"])
                if plan_size:
                    st.caption(
                        f"Position size: **{plan_size['shares']:,} shares** ({plan_size['position_value']:,.0f} USD), "
                        f"risking {plan_size['dollar_risk']:,.0f} USD ({risk_pct:.2f}% of {account_size:,.0f} USD) "
                        f"from entry {plan['entry']:.2f} to stop {plan['stop']:.2f}"
                        + (". Capped at the account size, so the risk is below budget." if plan_size["capped_by_account"] else ".")
                    )
            elif levels.get("source") is None:
                st.caption("No plan: there was not enough daily history to calculate the ATR.")
            else:
                st.caption("No trade: the reviewer did not pick a direction.")
            level_source = "breakout scanner" if levels.get("source") == "scanner" else "daily ATR"
            s_col1, s_col2, s_col3 = st.columns(3)
            for column, label, value in (
                (s_col1, "20-Day Support", levels.get("support")),
                (s_col2, "20-Day Resistance", levels.get("resistance")),
                (s_col3, "Daily ATR (14)", levels.get("atr")),
            ):
                column.metric(label, f"${value:.2f}" if isinstance(value, (int, float)) else "N/A")
            st.caption(
                f"Levels from the {level_source}, reviewed at {analysis['price']:.2f} in "
                f"{analysis['mode']} mode."
            )

            tab_labels = ["Technical", "Catalysts", "Risks"] + (["Scenarios"] if review["scenarios"] else [])
            review_tabs = st.tabs(tab_labels)
            for tab, key in zip(review_tabs, ["technical_notes", "catalyst_notes", "risks", "scenarios"]):
                with tab:
                    items = review[key]
                    st.markdown(
                        "\n".join(f"- {sanitize_ai_text(item)}" for item in items) if items else "Nothing noted."
                    )

            st.markdown("**Ask a follow-up**")
            for turn in st.session_state.llm_chat:
                st.markdown(f"**You:** {sanitize_ai_text(turn['question'])}")
                if turn["error"]:
                    st.error(f"No answer: {turn['error']}")
                else:
                    st.markdown(f"**AI:** {sanitize_ai_text(turn['answer'])}")
            q_col, a_col = st.columns([4, 1], vertical_alignment="bottom")
            with q_col:
                st.text_input(
                    "Question about this review", key="follow_up_question",
                    placeholder="e.g. Why not wait for a retest of the pivot?", label_visibility="collapsed",
                )
            with a_col:
                st.button("Ask", on_click=ask_follow_up_callback, width="stretch", key="ask_follow_up")
            st.caption("The AI answers from the same data and review as above; nothing new is downloaded.")
        else:
            st.info("Click 'Run AI Review' to have the AI grade a trade plan for this ticker.")

with track_tab:
    if track_tab.open:
        st.subheader("Scanner Track Record")
        st.caption(
            "Every setup the weekday scheduled scan finds (after the earnings filter) is logged before the open and "
            "followed forward with the backtest's rules: buy-stop at the pivot within 10 trading days, then exit at "
            "the stop, the target, or after 40 days. Unlike the backtest, nothing here is replayed after the fact. "
            "Results are in R (multiples of the initial risk)."
        )
        try:
            signal_log = load_signal_log()
        except Exception as error:
            signal_log = None
            st.error(f"Signal log unavailable: {error}")

        if signal_log is not None and signal_log.empty:
            st.info("No signals logged yet. The first ones appear after the next scheduled scan (weekdays, before the US open).")
        elif signal_log is not None:
            scope = st.radio(
                "Signals", [f"Top {ALERTED_TOP_N} (sent as alerts)", "All logged setups"],
                horizontal=True, key="track_record_scope",
            )
            summary = summarize_log(signal_log, max_rank=ALERTED_TOP_N if scope.startswith("Top") else None)

            tr1, tr2, tr3, tr4, tr5 = st.columns(5)
            tr1.metric("Signals", summary["signals"])
            tr2.metric("Closed Trades", summary["trades"])
            tr3.metric("Win Rate", f"{summary['win_rate_pct']}%" if summary["trades"] else "N/A")
            tr4.metric("Average R", f"{summary['average_r']:+.2f}R" if summary["trades"] else "N/A")
            tr5.metric("Total R", f"{summary['total_r']:+.1f}R" if summary["trades"] else "N/A")

            details = [
                f"Logged since {summary['first_scan']} (last scan {summary['last_scan']})",
                f"{summary['waiting']} waiting to trigger",
                f"{summary['still_open']} open",
                f"{summary['not_triggered']} never triggered",
            ]
            if summary["average_return_pct"] is not None and summary["average_iwm_pct"] is not None:
                details.append(
                    f"closed trades averaged {summary['average_return_pct']:+.2f}% vs IWM "
                    f"{summary['average_iwm_pct']:+.2f}% over the same days"
                )
            st.caption(" | ".join(details))
            if summary["trades"] < 30:
                st.caption("Fewer than 30 closed trades so far, so these numbers are still mostly noise.")

            by_grade = summarize_by_grade(signal_log, max_rank=ALERTED_TOP_N if scope.startswith("Top") else None)
            if not by_grade.empty and (by_grade["AI Grade"] != "Ungraded").any():
                st.markdown("**By AI grade**")
                st.dataframe(
                    by_grade, hide_index=True, width="stretch",
                    column_config={
                        "Win Rate %": st.column_config.NumberColumn("Win Rate", format="%.1f%%"),
                        "Average R": st.column_config.NumberColumn("Average R", format="%+.2f"),
                        "Total R": st.column_config.NumberColumn("Total R", format="%+.1f"),
                    },
                )
                st.caption(
                    "The scheduled scan grades the top setups with the AI reviewer before the open, so these grades "
                    "were set before the outcome was known. If A-grades don't beat C-grades over time, the AI review "
                    "isn't adding anything."
                )

            table = signal_log.copy()
            if scope.startswith("Top"):
                table = table[pd.to_numeric(table["Rank"], errors="coerce") <= ALERTED_TOP_N]
            table = table.sort_values(["Scan Date", "Rank"], ascending=[False, True])
            st.dataframe(
                table.drop(columns=["Universe", "Updated"]), hide_index=True, width="stretch",
                column_config={
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "Pivot": st.column_config.NumberColumn("Pivot", format="$%.2f"),
                    "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                    "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
                    "Reward/Risk": st.column_config.NumberColumn("R:R", format="%.1f"),
                    "Breakout Score": st.column_config.NumberColumn("Score", format="%.0f"),
                    "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Exit": st.column_config.NumberColumn("Exit", format="$%.2f"),
                    "R": st.column_config.NumberColumn("R", format="%+.2f"),
                    "Return %": st.column_config.NumberColumn("Return", format="%+.2f%%"),
                    "IWM %": st.column_config.NumberColumn("IWM", format="%+.2f%%"),
                },
            )
            st.caption(
                f"{OPEN} trades show their mark at the latest close; {WAITING} signals haven't traded through the pivot yet. "
                "Screening output is informational, not investment advice."
            )

with journal_tab:
    if journal_tab.open:
        st.subheader("Trade Journal")
        store = JournalStore()
        if "journal" not in st.session_state or st.button("Reload journal", key="reload_journal"):
            try:
                st.session_state.journal = store.load()
                st.session_state.journal_sha = store.sha
                st.session_state.journal_error = None
            except Exception as error:
                st.session_state.journal = None
                st.session_state.journal_error = str(error)
        store.sha = st.session_state.get("journal_sha")
        journal = st.session_state.journal
        st.caption(
            f"The trades you actually take, {store.description}. Open positions get a Telegram alert during market "
            "hours when they reach their stop or target, and the evening recap lists them."
        )
        if not store.remote:
            st.warning(
                "The journal is in a local file, which a hosted app loses when it restarts. Add a GITHUB_JOURNAL_TOKEN "
                "secret (a fine-grained GitHub token with read and write access to this repository's contents) to keep it "
                "on GitHub, where the alerts can see it."
            )
        if journal is None:
            st.error(f"Journal unavailable: {st.session_state.journal_error}")
            st.stop()

        def save(updated, message):
            try:
                store.save(updated, message)
                st.session_state.journal = updated
                st.session_state.journal_sha = store.sha
                return True
            except Exception as error:
                st.error(f"Not saved: {error}")
                return False

        positions = open_positions(journal)
        summary = summarize_journal(journal)
        prices = latest_closes(tuple(sorted(set(positions["Ticker"])))) if not positions.empty else {}
        marked = mark_positions(positions, prices) if not positions.empty else pd.DataFrame()
        open_pnl = pd.to_numeric(marked["P&L"], errors="coerce").sum() if not marked.empty else 0.0

        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Open Positions", summary["open"])
        j2.metric("Open P&L", f"{open_pnl:+,.0f}", help="USD, at the latest prices.")
        j3.metric("Realized P&L", f"{summary['total_pnl']:+,.0f}", help="USD, from closed trades.",
                  delta=f"{summary['average_r']:+.2f}R average" if summary["average_r"] is not None else None)
        j4.metric("Win Rate", f"{summary['win_rate_pct']:.0f}%" if summary["win_rate_pct"] is not None else "N/A",
                  help=f"{summary['closed']} closed trades.")

        st.markdown("**Open positions**")
        if marked.empty:
            st.info("No open positions. Add a trade below when you take one.")
        else:
            st.dataframe(
                marked, hide_index=True, width="stretch",
                column_config={
                    "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Shares": st.column_config.NumberColumn("Shares", format="%g"),
                    "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                    "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
                    "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                    "P&L": st.column_config.NumberColumn("P&L", format="$%+.0f"),
                    "R": st.column_config.NumberColumn("R", format="%+.2f"),
                    "To Stop %": st.column_config.NumberColumn("To Stop", format="%.1f%%"),
                    "To Target %": st.column_config.NumberColumn("To Target", format="%.1f%%"),
                },
            )
            st.caption("Prices are the latest daily bar, refreshed every minute; R is measured from the stop the trade was opened with.")

            labels = {f"{row['Ticker']} {row['Side'].lower()} from {float(row['Entry']):.2f} ({row['ID']})": row["ID"]
                      for _, row in positions.iterrows()}
            st.markdown("**Close a trade or move its stop**")
            chosen = st.selectbox("Trade", list(labels), key="manage_trade_choice")
            trade = positions[positions["ID"] == labels[chosen]].iloc[0]
            with st.form(f"manage_trade_{labels[chosen]}"):
                c1, c2, c3 = st.columns(3)
                exit_price = c1.number_input("Exit price", min_value=0.0, value=float(prices.get(trade["Ticker"], trade["Entry"])), step=0.01, format="%.2f")
                exit_date = c2.date_input("Exit date", value=pd.Timestamp.today())
                reason = c3.selectbox("Reason", ["Target", "Stop", "Manual exit", "Time exit"])
                new_stop = st.number_input("New stop", min_value=0.0, value=float(trade["Stop"]), step=0.01, format="%.2f")
                close_col, stop_col = st.columns(2)
                do_close = close_col.form_submit_button("Close trade", width="stretch")
                do_stop = stop_col.form_submit_button("Move stop", width="stretch")
            if do_close:
                if save(close_trade(journal, labels[chosen], exit_price, exit_date, reason), f"Journal: close {trade['Ticker']}"):
                    st.rerun()
            if do_stop:
                if save(update_stop(journal, labels[chosen], new_stop), f"Journal: move {trade['Ticker']} stop to {new_stop:.2f}"):
                    st.rerun()

        with st.expander("Add a trade", expanded=positions.empty):
            journal_scan = st.session_state.scanner_results
            prefill = None
            if journal_scan is not None and not journal_scan.empty and selected_ticker in set(journal_scan["Ticker"]):
                prefill = journal_scan.loc[journal_scan["Ticker"] == selected_ticker].iloc[0]
            if prefill is not None:
                st.caption(f"Prefilled from the scanner's {selected_ticker} setup: entry at the pivot, its stop and target.")
            with st.form("add_trade", clear_on_submit=True):
                a1, a2, a3 = st.columns(3)
                ticker_in = a1.text_input("Ticker", value=selected_ticker)
                side_in = a2.selectbox("Side", [LONG, SHORT])
                date_in = a3.date_input("Entry date", value=pd.Timestamp.today())
                b1, b2, b3, b4 = st.columns(4)
                entry_in = b1.number_input("Entry", min_value=0.0, value=float(prefill["Pivot"]) if prefill is not None else 0.0, step=0.01, format="%.2f")
                stop_in = b2.number_input("Stop", min_value=0.0, value=float(prefill["Stop"]) if prefill is not None else 0.0, step=0.01, format="%.2f")
                target_in = b3.number_input("Target (optional)", min_value=0.0, value=float(prefill["Target"]) if prefill is not None else 0.0, step=0.01, format="%.2f")
                suggested = position_size(account_size, risk_pct, float(prefill["Pivot"]), float(prefill["Stop"])) if prefill is not None else None
                shares_in = b4.number_input("Shares", min_value=0.0, value=float(suggested["shares"]) if suggested else 0.0, step=1.0)
                setup_in = st.text_input("Setup", value="Breakout scanner" if prefill is not None else "")
                notes_in = st.text_input("Notes")
                submitted = st.form_submit_button("Add trade")
            if submitted:
                try:
                    updated = add_trade(journal, ticker_in, side_in, date_in, entry_in, shares_in, stop_in,
                                        target_in or None, setup_in, notes_in)
                except ValueError as error:
                    st.error(str(error))
                else:
                    if save(updated, f"Journal: add {ticker_in.strip().upper()}"):
                        st.rerun()

        closed = journal[journal["Status"] == "Closed"] if not journal.empty else journal
        if not closed.empty:
            st.markdown("**Closed trades**")
            st.dataframe(
                closed.sort_values("Exit Date", ascending=False).drop(columns=["ID", "Status"]),
                hide_index=True, width="stretch",
                column_config={
                    "Entry": st.column_config.NumberColumn("Entry", format="$%.2f"),
                    "Exit": st.column_config.NumberColumn("Exit", format="$%.2f"),
                    "Stop": st.column_config.NumberColumn("Stop", format="$%.2f"),
                    "Initial Stop": st.column_config.NumberColumn("Initial Stop", format="$%.2f"),
                    "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
                    "P&L": st.column_config.NumberColumn("P&L", format="$%+.0f"),
                    "R": st.column_config.NumberColumn("R", format="%+.2f"),
                },
            )
