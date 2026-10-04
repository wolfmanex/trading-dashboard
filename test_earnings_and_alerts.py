import unittest
from datetime import date
from unittest import mock

import numpy as np
import pandas as pd

import earnings_engine
from backtest_engine import run_ta_backtest
from earnings_engine import add_earnings_columns, parse_finnhub_earnings, parse_nasdaq_earnings
from scheduled_scan import build_report
from technical_engine import add_technical_indicators


class EarningsParsingTests(unittest.TestCase):
    def test_nasdaq_rows_give_symbols_and_tolerate_empty_days(self):
        payload = {"data": {"rows": [{"symbol": "abc"}, {"symbol": ""}, None, {"name": "x"}]}}
        self.assertEqual(parse_nasdaq_earnings(payload), ["ABC"])
        self.assertEqual(parse_nasdaq_earnings({"data": {"rows": None}}), [])
        self.assertEqual(parse_nasdaq_earnings(None), [])

    def test_finnhub_keeps_the_earliest_date_per_symbol(self):
        payload = {"earningsCalendar": [
            {"symbol": "ABC", "date": "2026-10-20"},
            {"symbol": "ABC", "date": "2026-10-08"},
            {"symbol": "XYZ", "date": "2026-10-09"},
            {"symbol": "BAD"},
        ]}
        self.assertEqual(parse_finnhub_earnings(payload), {"ABC": "2026-10-08", "XYZ": "2026-10-09"})
        self.assertEqual(parse_finnhub_earnings({}), {})

    def test_earnings_columns_count_days_from_today(self):
        rows = pd.DataFrame({"Ticker": ["ABC", "XYZ"]})
        result = add_earnings_columns(rows, {"ABC": "2026-10-08"}, today=date(2026, 10, 5))
        self.assertEqual(list(result["Earnings"]), ["2026-10-08", ""])
        self.assertEqual(result["Days to ER"].iloc[0], 3)
        self.assertTrue(pd.isna(result["Days to ER"].iloc[1]))

    def test_upcoming_earnings_falls_back_to_nasdaq_without_a_finnhub_key(self):
        earnings_engine.get_upcoming_earnings.clear()
        response = mock.Mock()
        response.json.return_value = {"data": {"rows": [{"symbol": "ABC"}]}}
        response.raise_for_status.return_value = None
        with mock.patch.object(earnings_engine, "get_configured_secret", return_value=""), \
                mock.patch.object(earnings_engine.requests, "get", return_value=response) as fake_get:
            source, dates = earnings_engine.get_upcoming_earnings(days=3)
        self.assertEqual(source, "Nasdaq")
        self.assertIn("ABC", dates)
        self.assertTrue(all("api.nasdaq.com" in call.args[0] for call in fake_get.call_args_list))

    def test_upcoming_earnings_raises_when_every_source_fails(self):
        earnings_engine.get_upcoming_earnings.clear()
        with mock.patch.object(earnings_engine, "get_configured_secret", return_value=""), \
                mock.patch.object(earnings_engine.requests, "get", side_effect=OSError("blocked")):
            with self.assertRaises(RuntimeError):
                earnings_engine.get_upcoming_earnings(days=3)


class BuyHoldBenchmarkTests(unittest.TestCase):
    def test_backtest_reports_buy_and_hold_and_sides(self):
        closes = np.concatenate([np.linspace(100, 150, 60), np.linspace(150, 110, 40)])
        df = add_technical_indicators(pd.DataFrame({
            "Open": closes, "High": closes + 1, "Low": closes - 1, "Close": closes, "Volume": 1_000,
        }))
        result = run_ta_backtest(df, holding_period=5)
        self.assertEqual(result["buy_hold_return_pct"], 10.0)
        self.assertEqual(result["long_trades"] + result["short_trades"], result["total_trades"])
        self.assertGreater(result["total_trades"], 0)


class ScheduledReportTests(unittest.TestCase):
    def test_report_lists_top_setups_and_escapes_html(self):
        results = pd.DataFrame([{
            "Ticker": "A<B", "Price": 19.5, "Pivot": 20.1, "Stop": 19.0, "Target": 24.0,
            "Reward/Risk": 3.5, "Breakout Score": 72.0, "Earnings": "",
        }])
        results.attrs.update(universe_source="Nasdaq screener", analyzed=90, scan_timestamp="2026-10-05 09:00:00")
        markdown, html = build_report(results, {"label": "Uptrend"}, "0 left out.")
        self.assertIn("| A<B | 19.50 | 20.10 | 19.00 | 24.00 | 3.5 | 72 |", markdown)
        self.assertIn("<b>A&lt;B</b>", html)
        self.assertIn("Uptrend", html)

    def test_report_says_when_nothing_qualifies(self):
        markdown, html = build_report(pd.DataFrame(columns=["Ticker"]), {"label": "Mixed"}, "")
        self.assertIn("No setups", markdown)
        self.assertIn("No setups", html)


if __name__ == "__main__":
    unittest.main()
