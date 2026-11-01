"""Large-cap day-trading breakout scanner: universe, intraday download and ranking.

Built on the small-cap scanner's pieces: the same universe sources in the same order (Yahoo's custom
screen, Nasdaq's public screener, a predefined Yahoo screen, then a built-in list), with large-cap
bounds, and the batched yfinance download. The setup rules live in daytrade_engine.py.
"""
from datetime import datetime

import pandas as pd
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery

from daytrade_engine import MIN_RVOL, analyze_daytrade_setup, day_change_pct, rank_daytrade_candidates
from smallcap_screener import (
    US_EXCHANGES, _ticker_frame, fetch_nasdaq_screener, first_universe, parse_nasdaq_rows, parse_screener_quotes,
)


MIN_MARKET_CAP = 10_000_000_000
MIN_PRICE = 10.0
MIN_AVG_SHARE_VOLUME = 1_000_000
# Average daily traded value over the prior sessions: enough that a day trader's fills don't move the price.
MIN_DOLLAR_VOLUME = 100_000_000
# Five days of 5-minute bars with pre-market is ~1,000 bars a stock, so the universe is kept to the
# most traded names to keep the download well under a minute.
MAX_UNIVERSE = 150
BENCHMARK = "SPY"

FALLBACK_LARGECAPS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "AVGO", "NFLX",
    "COST", "JPM", "BAC", "WFC", "C", "GS", "MS", "SCHW", "XOM", "CVX",
    "UNH", "LLY", "JNJ", "PFE", "MRK", "ABBV", "V", "MA", "PYPL", "INTC",
    "MU", "QCOM", "ORCL", "CRM", "ADBE", "NOW", "UBER", "SHOP", "PLTR", "COIN",
    "PANW", "CRWD", "DIS", "NKE", "SBUX", "MCD", "WMT", "HD", "TGT", "BA",
    "CAT", "GE", "F", "GM", "KO", "PEP", "T", "VZ", "CSCO", "TXN",
    "AMAT", "LRCX", "ARM", "SMCI", "DELL", "ANET", "MRVL", "ABNB", "DASH", "HOOD",
]


def build_largecap_query() -> EquityQuery:
    return EquityQuery("and", [
        EquityQuery("gt", ["intradaymarketcap", MIN_MARKET_CAP]),
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", *US_EXCHANGES]),
        EquityQuery("gt", ["intradayprice", MIN_PRICE]),
        EquityQuery("gt", ["avgdailyvol3m", MIN_AVG_SHARE_VOLUME]),
    ])


def _yahoo_custom_universe() -> dict:
    response = yf.screen(
        build_largecap_query(), offset=0, size=MAX_UNIVERSE, count=MAX_UNIVERSE,
        sortField="avgdailyvol3m", sortAsc=False,
    )
    return parse_screener_quotes(response)


def _nasdaq_universe() -> dict:
    return parse_nasdaq_rows(
        fetch_nasdaq_screener(), limit=MAX_UNIVERSE, min_market_cap=MIN_MARKET_CAP, max_market_cap=float("inf"),
        min_price=MIN_PRICE, min_volume=MIN_AVG_SHARE_VOLUME,
    )


def _yahoo_predefined_universe() -> dict:
    # Without an offset yfinance uses the GET endpoint for predefined screens, which cloud hosts can reach more often.
    response = yf.screen("most_actives", count=250)
    universe = parse_screener_quotes(
        response, apply_bounds=True, min_market_cap=MIN_MARKET_CAP, max_market_cap=float("inf"), min_price=MIN_PRICE,
    )
    return dict(list(universe.items())[:MAX_UNIVERSE])


UNIVERSE_SOURCES = (
    ("Yahoo screener", _yahoo_custom_universe),
    ("Nasdaq screener", _nasdaq_universe),
    ("Yahoo most-active screen", _yahoo_predefined_universe),
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_largecap_universe() -> tuple:
    """(source name, {ticker: {name, market_cap}}); raises when every source fails, so failures aren't cached."""
    return first_universe(UNIVERSE_SOURCES)


def load_universe() -> tuple:
    """The screened universe, or the built-in list (with the screeners' error) when none answers."""
    try:
        source, universe = get_largecap_universe()
        return source, universe, None
    except Exception as error:
        print(f"Large-cap screener unavailable, using fallback list: {error}")
        return "fallback list", {ticker: {"name": ticker, "market_cap": None} for ticker in FALLBACK_LARGECAPS}, str(error)


def download_intraday(tickers: list) -> pd.DataFrame:
    # Unadjusted, with pre-market, so the levels match live quotes and the pre-market high is there.
    return yf.download(
        tickers=tickers, period="5d", interval="5m", prepost=True, group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )


def scan_daytrade_setups(
    min_rvol: float = MIN_RVOL,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
    limit: int = 25,
) -> pd.DataFrame:
    """Screen the large-cap universe for intraday breakouts using one batched 5-minute download."""
    universe_source, universe, screener_error = load_universe()
    tickers = sorted(universe)
    try:
        batch = download_intraday(tickers + [BENCHMARK])
    except Exception as error:
        print(f"Large-cap intraday download failed: {error}")
        batch = pd.DataFrame()

    benchmark_bars = _ticker_frame(batch, BENCHMARK)
    benchmark_change = day_change_pct(benchmark_bars)
    setups = {}
    for ticker in tickers:
        try:
            setups[ticker] = analyze_daytrade_setup(_ticker_frame(batch, ticker), benchmark_change)
        except Exception as error:
            print(f"Day-trade analysis failed for {ticker}: {error}")

    analyzed = [setup for setup in setups.values() if setup]
    if not analyzed:
        raise RuntimeError(f"No intraday data downloaded for the {universe_source} universe.")
    result = rank_daytrade_candidates(
        setups, metadata=universe, min_rvol=min_rvol, min_dollar_volume=min_dollar_volume, limit=limit,
    )
    phases = pd.Series([setup["phase"] for setup in analyzed]).value_counts()
    result.attrs.update({
        "scan_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "universe_size": len(tickers),
        "analyzed": len(analyzed),
        "universe_source": universe_source,
        "screener_error": screener_error,
        "phase": phases.index[0],
        "benchmark_change": round(benchmark_change, 2) if benchmark_change is not None else None,
    })
    return result


# The app caches the scan for 5 minutes; the alert script calls the uncached one every 15 minutes.
cached_scan_daytrade_setups = st.cache_data(ttl=300, show_spinner=False)(scan_daytrade_setups)
