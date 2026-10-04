"""Run the small-cap breakout scan outside Streamlit and report the results.

Used by .github/workflows/scheduled-scan.yml. Results always go to stdout and, on GitHub Actions,
to the job summary. When TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set they are also sent to
Telegram.
"""
import os
import sys
from html import escape

import requests

from earnings_engine import EARNINGS_WINDOW_DAYS, add_earnings_columns, get_upcoming_earnings
from smallcap_screener import get_smallcap_regime, scan_smallcap_breakouts


TOP_N = 10
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"


def build_report(results, regime: dict, earnings_note: str, top_n: int = TOP_N) -> tuple:
    """Return (markdown, telegram_html) for the top setups."""
    regime_line = f"Small-cap trend (IWM): {regime.get('label', 'Unknown')}"
    source = results.attrs.get("universe_source", "N/A")
    header = (
        f"{len(results)} setups from {results.attrs.get('analyzed', 'N/A')} stocks ({source}), "
        f"{results.attrs.get('scan_timestamp', '')}"
    )
    rows = results.head(top_n)

    markdown = [f"## Breakout scan\n", f"{regime_line}  ", f"{header}  ", f"{earnings_note}\n"]
    html = [f"<b>Breakout scan</b>", escape(regime_line), escape(header), escape(earnings_note), ""]
    if rows.empty:
        markdown.append("No setups meet the criteria today.")
        html.append("No setups meet the criteria today.")
        return "\n".join(markdown), "\n".join(html)

    markdown.append("| Ticker | Price | Pivot | Stop | Target | R:R | Score | Earnings |")
    markdown.append("|---|---|---|---|---|---|---|---|")
    for _, row in rows.iterrows():
        earnings = row.get("Earnings") or ""
        markdown.append(
            f"| {row['Ticker']} | {row['Price']:.2f} | {row['Pivot']:.2f} | {row['Stop']:.2f} | "
            f"{row['Target']:.2f} | {row['Reward/Risk']:.1f} | {row['Breakout Score']:.0f} | {earnings} |"
        )
        html.append(
            f"<b>{escape(str(row['Ticker']))}</b> {row['Price']:.2f} | pivot {row['Pivot']:.2f} "
            f"stop {row['Stop']:.2f} target {row['Target']:.2f} | R:R {row['Reward/Risk']:.1f} "
            f"| score {row['Breakout Score']:.0f}" + (f" | ER {escape(earnings)}" if earnings else "")
        )
    return "\n".join(markdown), "\n".join(html)


def send_telegram(token: str, chat_id: str, html: str) -> None:
    response = requests.post(
        TELEGRAM_URL.format(token=token),
        json={"chat_id": chat_id, "text": html[:4000], "parse_mode": "HTML", "disable_web_page_preview": True},
        timeout=20,
    )
    response.raise_for_status()


def main() -> int:
    results = scan_smallcap_breakouts()
    regime = get_smallcap_regime()

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

    markdown, html = build_report(results, regime, earnings_note)
    print(markdown)

    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(markdown + "\n")

    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        send_telegram(token, chat_id, html)
        print("Sent to Telegram.")
    else:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set; results are in the job summary only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
