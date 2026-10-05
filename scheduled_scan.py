"""Run the small-cap breakout scan outside Streamlit and report the results.

Used by .github/workflows/scheduled-scan.yml. Results always go to stdout and, on GitHub Actions,
to the job summary. When TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set they are also sent to
Telegram. When GEMINI_API_KEY is set the top setups are graded by the AI reviewer before anything is
sent, the grade is logged with the signal so the Track Record can compare grades, and a second
message (the AI brief) gives the catalyst and red flags for the top BRIEF_N. With --log PATH
the setups are also added to the signal log and every live signal's outcome is re-checked (see
signal_log.py).
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from earnings_engine import EARNINGS_WINDOW_DAYS, add_earnings_columns, get_upcoming_earnings
from signal_log import LIVE_STATUSES, append_signals, read_log, summarize_log, update_outcomes
from llm_engine import GEMINI_MODEL, is_ai_configured, review_breakout_candidates
from market_conditions import get_market_conditions, summary_line
from smallcap_screener import (
    BENCHMARK, _ticker_frame, get_candidate_context, get_smallcap_regime, scan_smallcap_breakouts,
)


TOP_N = 10
BRIEF_N = 3   # setups that get the AI's catalyst and red flags in the morning brief
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


def grade_top_setups(results, top_n: int = TOP_N, market_note: str = "") -> tuple:
    """Add the AI reviewer's grade and risk level to the top setups (in place).

    Returns (one-line note, {ticker: review}) so the morning brief can quote the reviews.
    """
    if results is None or results.empty:
        return "", {}
    if not is_ai_configured():
        return "AI grading skipped: GEMINI_API_KEY is not set.", {}
    candidates = results.head(top_n).to_dict("records")
    contexts = {candidate["Ticker"]: get_candidate_context(candidate["Ticker"]) for candidate in candidates}
    outcome = review_breakout_candidates(candidates, contexts, market_note=market_note)
    if outcome["error"]:
        return f"AI grading failed, setups are logged ungraded: {outcome['error']}", {}
    reviews = outcome["reviews"]
    results["AI Grade"] = [reviews.get(ticker, {}).get("grade", "") for ticker in results["Ticker"]]
    results["AI Risk"] = [reviews.get(ticker, {}).get("risk_level", "") for ticker in results["Ticker"]]
    return f"AI graded {len(reviews)} of the top {len(candidates)} setups ({GEMINI_MODEL}).", reviews


def build_brief(results, reviews: dict, brief_n: int = BRIEF_N) -> str:
    """Telegram HTML with the AI's catalyst, thesis and red flags for the top graded setups; empty without reviews."""
    if results is None or results.empty or not reviews:
        return ""
    lines = ["<b>AI brief</b>"]
    for ticker in [ticker for ticker in results["Ticker"] if ticker in reviews][:brief_n]:
        review = reviews[ticker]
        lines.append(
            f"\n<b>{escape(ticker)}</b> grade {escape(review['grade'])}, {escape(review['risk_level'].lower())} risk. "
            f"Catalyst: {escape(review['catalyst'])}"
        )
        if review.get("thesis"):
            lines.append(escape(review["thesis"]))
        if review.get("red_flags"):
            lines.append("Red flags: " + escape("; ".join(review["red_flags"])))
    return "\n".join(lines)


def build_report(results, regime: dict, earnings_note: str, top_n: int = TOP_N, ai_note: str = "", market_note: str = "") -> tuple:
    """Return (markdown, telegram_html) for the top setups."""
    regime_line = market_note or f"Small-cap trend (IWM): {regime.get('label', 'Unknown')}"
    source = results.attrs.get("universe_source", "N/A")
    header = (
        f"{len(results)} setups from {results.attrs.get('analyzed', 'N/A')} stocks ({source}), "
        f"{results.attrs.get('scan_timestamp', '')}"
    )
    rows = results.head(top_n)

    notes = [earnings_note] + ([ai_note] if ai_note else [])
    markdown = [f"## Breakout scan\n", f"{regime_line}  ", f"{header}  "] + [f"{note}  " for note in notes] + [""]
    html = [f"<b>Breakout scan</b>", escape(regime_line), escape(header)] + [escape(note) for note in notes] + [""]
    if rows.empty:
        markdown.append("No setups meet the criteria today.")
        html.append("No setups meet the criteria today.")
        return "\n".join(markdown), "\n".join(html)

    markdown.append("| Ticker | Price | Pivot | Stop | Target | R:R | Score | AI | Earnings |")
    markdown.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in rows.iterrows():
        earnings = row.get("Earnings") or ""
        grade = row.get("AI Grade") or ""
        ai = f"{grade} ({row.get('AI Risk') or 'N/A'} risk)" if grade else ""
        markdown.append(
            f"| {row['Ticker']} | {row['Price']:.2f} | {row['Pivot']:.2f} | {row['Stop']:.2f} | "
            f"{row['Target']:.2f} | {row['Reward/Risk']:.1f} | {row['Breakout Score']:.0f} | {ai} | {earnings} |"
        )
        html.append(
            f"<b>{escape(str(row['Ticker']))}</b> {row['Price']:.2f} | pivot {row['Pivot']:.2f} "
            f"stop {row['Stop']:.2f} target {row['Target']:.2f} | R:R {row['Reward/Risk']:.1f} "
            f"| score {row['Breakout Score']:.0f}" + (f" | AI {escape(ai)}" if ai else "")
            + (f" | ER {escape(earnings)}" if earnings else "")
        )
    return "\n".join(markdown), "\n".join(html)


def send_telegram(token: str, chat_id: str, html: str) -> None:
    response = requests.post(
        TELEGRAM_URL.format(token=token),
        json={"chat_id": chat_id, "text": html[:4000], "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()


def download_outcome_prices(log: pd.DataFrame) -> tuple:
    """Daily bars for every live signal's ticker plus IWM, from a week before the oldest live scan."""
    live = log[log["Status"].isin(LIVE_STATUSES)]
    if live.empty:
        return {}, None
    tickers = sorted(set(live["Ticker"]))
    start = (pd.Timestamp(live["Scan Date"].min()) - timedelta(days=10)).strftime("%Y-%m-%d")
    # Unadjusted prices, so later dividend adjustments don't move bars away from the logged levels.
    batch = yf.download(
        tickers=tickers + [BENCHMARK], start=start, interval="1d", group_by="ticker",
        auto_adjust=False, progress=False, threads=True,
    )
    frames = {ticker: _ticker_frame(batch, ticker) for ticker in tickers}
    benchmark = _ticker_frame(batch, BENCHMARK)
    return frames, benchmark["Close"].dropna() if "Close" in benchmark else None


def record_signals(log_path: str, results, regime: dict, scan_date) -> str:
    """Append today's setups to the log, re-check live signals, save, and return a one-line summary."""
    log = append_signals(read_log(log_path), results, scan_date, regime.get("label", ""))
    frames, benchmark_close = download_outcome_prices(log)
    log = update_outcomes(log, frames, benchmark_close, today=scan_date)
    log.to_csv(log_path, index=False)
    summary = summarize_log(log)
    line = f"Signal log: {summary['signals']} signals since {summary['first_scan']}, {summary['trades']} closed trades"
    if summary["trades"]:
        line += f", win rate {summary['win_rate_pct']}%, average {summary['average_r']:+.2f}R"
    return line


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the breakout scan and report the results.")
    parser.add_argument("--log", help="CSV signal log to append to and update.")
    args = parser.parse_args(argv)
    scan_date = datetime.now(ZoneInfo("America/New_York")).date()

    regime = get_smallcap_regime()
    try:
        results = scan_smallcap_breakouts()
    except Exception:
        if args.log:
            # Still re-check open signals so a failed scan doesn't stall the track record.
            print(record_signals(args.log, None, regime, scan_date))
        raise

    try:
        earnings_source, earnings_dates = get_upcoming_earnings()
        results = add_earnings_columns(results, earnings_dates)
        reporting = results["Days to ER"].between(0, EARNINGS_WINDOW_DAYS)
        results = results[~reporting].reset_index(drop=True)
        earnings_note = (
            f"{int(reporting.sum())} setups reporting earnings within {EARNINGS_WINDOW_DAYS} days were "
            f"left out (dates from {earnings_source})."
        )
    except Exception as error:
        earnings_note = f"Earnings dates unavailable, nothing left out: {error}"

    try:
        market_note = summary_line(get_market_conditions())
    except Exception as error:
        market_note = ""
        print(f"Market conditions unavailable: {error}")
    ai_note, reviews = grade_top_setups(results, market_note=market_note)
    markdown, html = build_report(results, regime, earnings_note, ai_note=ai_note, market_note=market_note)
    brief = build_brief(results, reviews)
    log_error = None
    if args.log:
        try:
            markdown += "\n\n" + record_signals(args.log, results, regime, scan_date)
        except Exception as error:
            log_error = error
            markdown += f"\n\nSignal log not updated: {error}"
    print(markdown)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(markdown + "\n")

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        send_telegram(token, chat_id, html)
        if brief:
            send_telegram(token, chat_id, brief)
        print("Sent to Telegram.")
    else:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; results are in the job summary only.")
    # The alerts went out, but fail the job so a broken signal log gets noticed.
    return 1 if log_error else 0


if __name__ == "__main__":
    sys.exit(main())
