from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery

from breakout_engine import analyze_breakout_setup, rank_breakout_candidates
from event_engine import parse_news_headlines


MIN_MARKET_CAP = 300_000_000
MAX_MARKET_CAP = 2_000_000_000
MIN_PRICE = 2.0
MIN_AVG_SHARE_VOLUME = 300_000
MIN_DOLLAR_VOLUME = 5_000_000
MIN_REWARD_RISK = 3.0
BENCHMARK = "IWM"
US_EXCHANGES = ["NMS", "NYQ", "NGM", "NCM", "ASE"]


def build_smallcap_query() -> EquityQuery:
    return EquityQuery("and", [
        EquityQuery("btwn", ["intradaymarketcap", MIN_MARKET_CAP, MAX_MARKET_CAP]),
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", *US_EXCHANGES]),
        EquityQuery("gt", ["intradayprice", MIN_PRICE]),
        EquityQuery("gt", ["avgdailyvol3m", MIN_AVG_SHARE_VOLUME]),
    ])


def parse_screener_quotes(response) -> dict:
    """Map Yahoo screener quotes to {ticker: {name, market_cap}}, skipping malformed rows."""
    universe = {}
    quotes = response.get("quotes") if isinstance(response, dict) else None
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        symbol = str(quote.get("symbol", "")).strip().upper()
        if not symbol or "." in symbol or "^" in symbol:
            continue
        universe[symbol] = {
            "name": quote.get("shortName") or quote.get("longName") or symbol,
            "market_cap": quote.get("marketCap"),
        }
    return universe


@st.cache_data(ttl=3600, show_spinner=False)
def get_smallcap_universe(pages: int = 2, page_size: int = 250) -> dict:
    """Most liquid US small caps from the Yahoo screener, sorted by 3-month average volume."""
    universe = {}
    query = build_smallcap_query()
    for page in range(pages):
        try:
            response = yf.screen(
                query,
                offset=page * page_size,
                size=page_size,
                sortField="avgdailyvol3m",
                sortAsc=False,
            )
        except Exception as error:
            print(f"Small-cap screener page {page} failed: {error}")
            break
        page_universe = parse_screener_quotes(response)
        universe.update(page_universe)
        if len(page_universe) < page_size:
            break
    return universe


def _ticker_frame(batch: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if batch.empty:
        return pd.DataFrame()
    if isinstance(batch.columns, pd.MultiIndex):
        if ticker not in batch.columns.get_level_values(0):
            return pd.DataFrame()
        return batch[ticker].dropna(how="all")
    return batch


@st.cache_data(ttl=900, show_spinner=False)
def scan_smallcap_breakouts(
    min_reward_risk: float = MIN_REWARD_RISK,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    limit: int = 25,
) -> pd.DataFrame:
    """Screen the small-cap universe for base-breakout setups using one batched daily download."""
    universe = get_smallcap_universe()
    tickers = sorted(universe)
    if not tickers:
        result = rank_breakout_candidates({})
        result.attrs["error"] = "Small-cap universe unavailable from the Yahoo screener."
        return result

    try:
        batch = yf.download(
            tickers=tickers + [BENCHMARK],
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as error:
        print(f"Small-cap price download failed: {error}")
        batch = pd.DataFrame()

    benchmark_frame = _ticker_frame(batch, BENCHMARK)
    benchmark_close = benchmark_frame["Close"].dropna() if "Close" in benchmark_frame else None

    setups = {}
    for ticker in tickers:
        try:
            setups[ticker] = analyze_breakout_setup(_ticker_frame(batch, ticker), benchmark_close)
        except Exception as error:
            print(f"Breakout analysis failed for {ticker}: {error}")

    result = rank_breakout_candidates(
        setups,
        metadata=universe,
        min_reward_risk=min_reward_risk,
        min_dollar_volume=min_dollar_volume,
        limit=limit,
    )
    result.attrs["scan_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result.attrs["universe_size"] = len(tickers)
    result.attrs["analyzed"] = sum(1 for setup in setups.values() if setup)
    return result


def _format_earnings_date(info: dict) -> str:
    timestamp = info.get("earningsTimestampStart") or info.get("earningsTimestamp")
    if not timestamp:
        return "N/A"
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return "N/A"


@st.cache_data(ttl=1800, show_spinner=False)
def get_candidate_context(ticker: str) -> dict:
    """Fundamental and news context the AI reviewer needs to spot catalysts and red flags."""
    context = {
        "sector": "N/A",
        "industry": "N/A",
        "float_shares": "N/A",
        "short_pct_float": "N/A",
        "insider_pct": "N/A",
        "earnings_date": "N/A",
        "headlines": [],
    }
    try:
        ticker_data = yf.Ticker(ticker)
        info = ticker_data.info or {}
        short_pct = info.get("shortPercentOfFloat")
        insider_pct = info.get("heldPercentInsiders")
        context.update({
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "float_shares": info.get("floatShares") or "N/A",
            "short_pct_float": round(short_pct * 100, 1) if isinstance(short_pct, (int, float)) else "N/A",
            "insider_pct": round(insider_pct * 100, 1) if isinstance(insider_pct, (int, float)) else "N/A",
            "earnings_date": _format_earnings_date(info),
        })
        context["headlines"] = parse_news_headlines(ticker_data.news, limit=6)
    except Exception as error:
        print(f"Candidate context lookup failed for {ticker}: {error}")
    return context
