from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from yfinance import EquityQuery

from breakout_backtest import TRADE_COLUMNS, backtest_ticker, summarize_trades
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


# Used only when no screener source answers. Market caps drift, so these are re-checked by the scan's dollar-volume and setup filters.
FALLBACK_SMALLCAPS = [
    "AAOI", "AEHR", "AEO", "AMC", "AMPL", "ARCT", "ARRY", "BBAI", "BLDP", "BMBL",
    "BTBT", "CARS", "CLOV", "CRSR", "DNUT", "EVGO", "FIGS", "FUBO", "GOGO", "GPRE",
    "GRPN", "HLIT", "INDI", "JBLU", "KSS", "LC", "LMND", "LUNR", "NTLA", "NVAX",
    "PLUG", "RDW", "ROOT", "RUN", "RVLV", "RXRX", "SABR", "SANA", "SEDG", "SHLS",
    "TDOC", "VSAT", "BEAM", "WULF",
]

MAX_UNIVERSE = 500
NASDAQ_SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
NASDAQ_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def parse_screener_quotes(response, apply_bounds: bool = False) -> dict:
    """Map Yahoo screener quotes to {ticker: {name, market_cap}}, skipping malformed rows."""
    universe = {}
    quotes = response.get("quotes") if isinstance(response, dict) else None
    for quote in quotes or []:
        if not isinstance(quote, dict):
            continue
        symbol = str(quote.get("symbol", "")).strip().upper()
        if not symbol or "." in symbol or "^" in symbol:
            continue
        market_cap = quote.get("marketCap")
        if apply_bounds:
            price = quote.get("regularMarketPrice")
            if not isinstance(market_cap, (int, float)) or not MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP:
                continue
            if isinstance(price, (int, float)) and price < MIN_PRICE:
                continue
        universe[symbol] = {
            "name": quote.get("shortName") or quote.get("longName") or symbol,
            "market_cap": market_cap,
        }
    return universe


def _parse_number(value):
    """Turn Nasdaq display strings like "$12.34", "1,234,567" or "" into floats (None when blank)."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace("$", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_nasdaq_rows(payload, limit: int = MAX_UNIVERSE) -> dict:
    """Filter Nasdaq's all-US-stocks screener to small caps, keeping the most traded by dollar volume."""
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else {}
    rows = data.get("rows") or (data.get("table") or {}).get("rows") or []
    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        # Skip preferreds, units, warrants and share classes ("ABC^A", "ABC/WS", "BRK.B").
        if not symbol or any(char in symbol for char in "^/. "):
            continue
        market_cap = _parse_number(row.get("marketCap"))
        price = _parse_number(row.get("lastsale"))
        volume = _parse_number(row.get("volume"))
        if market_cap is None or not MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP:
            continue
        if price is None or price < MIN_PRICE or volume is None or volume < MIN_AVG_SHARE_VOLUME:
            continue
        candidates.append((price * volume, symbol, row.get("name") or symbol, market_cap))
    candidates.sort(reverse=True)
    return {
        symbol: {"name": name, "market_cap": market_cap}
        for _, symbol, name, market_cap in candidates[:limit]
    }


def _yahoo_custom_universe(pages: int = 2, page_size: int = 250) -> dict:
    universe = {}
    for page in range(pages):
        response = yf.screen(
            build_smallcap_query(),
            offset=page * page_size,
            size=page_size,
            count=page_size,
            sortField="avgdailyvol3m",
            sortAsc=False,
        )
        universe.update(parse_screener_quotes(response))
        if len(response.get("quotes") or []) < page_size:
            break
    return universe


def _nasdaq_universe() -> dict:
    # Nasdaq's public screener needs no cookie or crumb, unlike Yahoo's, which returns 401 from cloud hosts.
    response = requests.get(
        NASDAQ_SCREENER_URL,
        params={"tableonly": "true", "download": "true"},
        headers=NASDAQ_HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    return parse_nasdaq_rows(response.json())


def _yahoo_predefined_universe() -> dict:
    # Without an offset yfinance uses the GET endpoint for predefined screens rather than the POST one.
    return parse_screener_quotes(yf.screen("small_cap_gainers", count=250), apply_bounds=True)


UNIVERSE_SOURCES = (
    ("Yahoo screener", _yahoo_custom_universe),
    ("Nasdaq screener", _nasdaq_universe),
    ("Yahoo small-cap screen", _yahoo_predefined_universe),
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_smallcap_universe() -> tuple:
    """Return (source name, {ticker: {name, market_cap}}) from the first small-cap source that answers.

    Raises RuntimeError when every source fails, so a failure is never cached for an hour.
    """
    errors = []
    for name, fetch in UNIVERSE_SOURCES:
        try:
            universe = fetch()
        except Exception as error:
            errors.append(f"{name}: {error}")
            continue
        if universe:
            return name, universe
        errors.append(f"{name}: no stocks returned")
    raise RuntimeError("; ".join(errors))


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
    screener_error = None
    try:
        universe_source, universe = get_smallcap_universe()
    except Exception as error:
        print(f"Small-cap screener unavailable, using fallback list: {error}")
        screener_error = str(error)
        universe_source = "fallback list"
        universe = {ticker: {"name": ticker, "market_cap": None} for ticker in FALLBACK_SMALLCAPS}
    tickers = sorted(universe)

    try:
        # Unadjusted prices, so pivot/stop/target match the real quotes the outcome checks and live
        # prices use (auto_adjust=True shifts history around ex-dividend dates).
        batch = yf.download(
            tickers=tickers + [BENCHMARK],
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
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
    result.attrs["universe_source"] = universe_source
    result.attrs["screener_error"] = screener_error
    if result.attrs["analyzed"] == 0:
        raise RuntimeError(f"No price data downloaded for the {universe_source} universe.")
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


def classify_regime(close: pd.Series) -> dict:
    """Describe the small-cap market trend from IWM's daily closes.

    Uptrend: price above a rising 50-day average that sits above the 200-day.
    Downtrend: price below a falling 50-day average that sits below the 200-day.
    Anything else is mixed.
    """
    close = pd.to_numeric(close, errors="coerce").dropna() if close is not None else pd.Series(dtype=float)
    regime = {"label": "Unknown", "price": None, "sma_50": None, "sma_200": None, "sma_50_rising": None}
    if len(close) < 200:
        return regime

    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    price, latest_50, latest_200 = float(close.iloc[-1]), float(sma_50.iloc[-1]), float(sma_200.iloc[-1])
    rising = bool(latest_50 > float(sma_50.iloc[-11]))
    if price > latest_50 > latest_200 and rising:
        label = "Uptrend"
    elif price < latest_50 < latest_200 and not rising:
        label = "Downtrend"
    else:
        label = "Mixed"
    regime.update({
        "label": label,
        "price": round(price, 2),
        "sma_50": round(latest_50, 2),
        "sma_200": round(latest_200, 2),
        "sma_50_rising": rising,
    })
    return regime


@st.cache_data(ttl=900, show_spinner=False)
def get_smallcap_regime() -> dict:
    """IWM trend for the scanner header; returns label "Unknown" when the download fails."""
    try:
        history = yf.Ticker(BENCHMARK).history(period="2y", interval="1d", auto_adjust=True)
        return classify_regime(history["Close"] if "Close" in history else None)
    except Exception as error:
        print(f"Small-cap regime lookup failed: {error}")
        return classify_regime(None)


def label_trade_regimes(trades: pd.DataFrame, benchmark_close: pd.Series) -> pd.DataFrame:
    """Tag each trade with whether IWM was above its 50-day average on the signal date."""
    if trades.empty or benchmark_close is None or benchmark_close.empty:
        trades["IWM > 50d"] = None
        return trades
    above = (benchmark_close > benchmark_close.rolling(50).mean()).reindex(
        pd.to_datetime(trades["Signal Date"]), method="ffill"
    )
    trades["IWM > 50d"] = above.to_numpy()
    return trades


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def backtest_smallcap_breakouts(
    max_tickers: int = 150,
    period: str = "3y",
    min_reward_risk: float = MIN_REWARD_RISK,
    min_dollar_volume: float = MIN_DOLLAR_VOLUME,
) -> dict:
    """Replay the scanner's rules over the current small-cap universe's daily history.

    The universe is today's list, so stocks that were delisted or fell out of the small-cap range
    are missing (survivorship bias); results flatter the rules somewhat.
    """
    try:
        universe_source, universe = get_smallcap_universe()
    except Exception as error:
        print(f"Small-cap screener unavailable for backtest, using fallback list: {error}")
        universe_source = "fallback list"
        universe = {ticker: {"name": ticker, "market_cap": None} for ticker in FALLBACK_SMALLCAPS}
    tickers = list(universe)[:max_tickers]

    batch = yf.download(
        tickers=tickers + [BENCHMARK],
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    benchmark_frame = _ticker_frame(batch, BENCHMARK)
    benchmark_close = benchmark_frame["Close"].dropna() if "Close" in benchmark_frame else None

    rows = []
    tested = 0
    for ticker in tickers:
        frame = _ticker_frame(batch, ticker)
        if frame.empty:
            continue
        tested += 1
        try:
            for trade in backtest_ticker(frame, benchmark_close, min_reward_risk, min_dollar_volume):
                rows.append({"Ticker": ticker, **trade})
        except Exception as error:
            print(f"Breakout backtest failed for {ticker}: {error}")
    if tested == 0:
        raise RuntimeError(f"No price history downloaded for the {universe_source} universe.")

    trades = label_trade_regimes(pd.DataFrame(rows, columns=TRADE_COLUMNS), benchmark_close)
    trades = trades.sort_values("Signal Date", ascending=False).reset_index(drop=True)

    by_regime = {}
    if "IWM > 50d" in trades and trades["IWM > 50d"].notna().any():
        by_regime = {
            "IWM above 50-day": summarize_trades(trades[trades["IWM > 50d"] == True]),  # noqa: E712
            "IWM below 50-day": summarize_trades(trades[trades["IWM > 50d"] == False]),  # noqa: E712
        }
    return {
        "trades": trades,
        "summary": summarize_trades(trades),
        "by_regime": by_regime,
        "tickers_tested": tested,
        "universe_source": universe_source,
        "period": period,
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
