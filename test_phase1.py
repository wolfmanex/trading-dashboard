import unittest

import numpy as np
import pandas as pd

from bot import calculate_ta_score, get_market_bias, get_news_status, parse_args
from backtest_engine import run_ta_backtest
from event_engine import parse_news_headlines
from watchlist_engine import normalize_watchlist
from movers_engine import rank_movers
from stock_info import format_profile_summary
from intraday_engine import calculate_relative_volume
from llm_engine import validate_trade_review
from options_engine import has_valid_bid_ask, option_midpoint, select_expiration_candidates
from technical_engine import add_technical_indicators, tag_data_source, validate_ohlcv_data


def valid_trade_review():
    return {
        "direction": "long",
        "verdict": "wait",
        "grade": "b",
        "risk_level": "medium",
        "summary": "Valid base, wait for the pivot.",
        "trigger": "A close above 12.40 on volume.",
        "technical_notes": "- Above the rising 50-day.\n- Stop sits under the base low.",
        "catalyst_notes": ["No catalyst in the headlines."],
        "risks": ["Thin float"],
        "scenarios": [],
    }


LONG_ONLY_LEVELS = {"long": {"entry": 12.4, "stop": 11.5, "target": 14.2, "reward_risk": 2.0}, "short": None}


class TechnicalEngineTests(unittest.TestCase):
    def test_indicators_are_added_to_sufficient_data(self):
        close = pd.Series(np.linspace(100, 130, 40))
        prices = pd.DataFrame({
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000,
        })

        result = add_technical_indicators(prices)

        for column in ("EMA_9", "EMA_21", "RSI", "MACD", "Signal_Line", "BB_Upper", "BB_Lower"):
            self.assertIn(column, result.columns)
        self.assertTrue(result["EMA_9"].iloc[-1] > result["EMA_21"].iloc[-1])

    def test_rsi_handles_flat_prices_as_neutral(self):
        prices = pd.DataFrame({"Close": [100.0] * 40})

        result = add_technical_indicators(prices)

        self.assertEqual(result["RSI"].iloc[-1], 50)

    def test_rsi_handles_continuous_gain_as_overbought(self):
        prices = pd.DataFrame({"Close": np.linspace(100, 140, 40)})

        result = add_technical_indicators(prices)

        self.assertEqual(result["RSI"].iloc[-1], 100)

    def test_empty_data_has_a_neutral_score(self):
        self.assertEqual(calculate_ta_score(pd.DataFrame()), 0)

    def test_data_source_is_recorded_on_price_data(self):
        data = pd.DataFrame({"Close": [100.0]})

        result = tag_data_source(data, "Test Provider")

        self.assertEqual(result.attrs["data_source"], "Test Provider")

    def test_invalid_ohlcv_data_is_rejected(self):
        data = pd.DataFrame({
            "Open": [100.0],
            "High": [99.0],
            "Low": [98.0],
            "Close": [98.5],
            "Volume": [1000],
        })

        self.assertFalse(validate_ohlcv_data(data))

    def test_valid_ohlcv_data_is_accepted(self):
        data = pd.DataFrame({
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000],
        })

        self.assertTrue(validate_ohlcv_data(data))

    def test_incomplete_rows_can_be_removed_before_validation(self):
        data = pd.DataFrame({
            "Open": [None, 100.0],
            "High": [None, 102.0],
            "Low": [None, 99.0],
            "Close": [None, 101.0],
            "Volume": [None, 1000],
        }).dropna()

        self.assertTrue(validate_ohlcv_data(data))

    def test_bullish_indicators_produce_positive_score(self):
        data = pd.DataFrame({
            "Close": [110.0],
            "EMA_9": [109.0],
            "EMA_21": [105.0],
            "MACD": [2.0],
            "Signal_Line": [1.0],
            "RSI": [55.0],
        })

        self.assertEqual(calculate_ta_score(data), 80)

    def test_backtest_returns_empty_metrics_without_trades(self):
        data = pd.DataFrame({"Close": [100.0] * 10})

        result = run_ta_backtest(data)

        self.assertEqual(result["total_trades"], 0)
        self.assertEqual(result["win_rate_pct"], "N/A")

    def test_backtest_uses_next_bar_for_entry(self):
        data = pd.DataFrame({
            "Close": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0],
            "EMA_9": [101.0] * 7,
            "EMA_21": [100.0] * 7,
            "MACD": [2.0] * 7,
            "Signal_Line": [1.0] * 7,
            "RSI": [50.0] * 7,
        })

        result = run_ta_backtest(data, holding_period=2)

        # Signals at bars 1 and 3 (bar 0 closes on EMA 21); the second waits for the first exit
        self.assertEqual(result["total_trades"], 2)
        self.assertEqual(result["win_rate_pct"], 100.0)

    def test_backtest_does_not_open_overlapping_trades(self):
        data = pd.DataFrame({
            "Close": [100.0, 100.0, 110.0, 121.0, 133.1, 146.41, 161.051],
            "EMA_9": [100.0] * 7,
            "EMA_21": [99.0] * 7,
            "MACD": [2.0] * 7,
            "Signal_Line": [1.0] * 7,
            "RSI": [50.0] * 7,
        })

        result = run_ta_backtest(data, holding_period=3)

        # Signals at bars 0 and 3 only: entry 1 -> exit 3, then entry 4 -> exit 6
        self.assertEqual(result["total_trades"], 2)
        self.assertAlmostEqual(result["cumulative_return_pct"], (1.21 * 1.21 - 1) * 100, places=2)

    def test_backtest_applies_cost_per_trade(self):
        data = pd.DataFrame({
            "Close": [100.0, 110.0, 120.0, 130.0],
            "EMA_9": [101.0] * 4,
            "EMA_21": [100.0] * 4,
            "MACD": [2.0] * 4,
            "Signal_Line": [1.0] * 4,
            "RSI": [50.0] * 4,
        })

        result = run_ta_backtest(data, holding_period=2, cost_per_trade_pct=1.0)

        self.assertEqual(result["total_trades"], 1)
        self.assertAlmostEqual(result["average_return_pct"], 7.33, places=2)


class MappingTests(unittest.TestCase):
    def test_market_trend_mapping(self):
        self.assertEqual(get_market_bias("Bullish (+4.0% > 200 SMA)"), "BULLISH")
        self.assertEqual(get_market_bias("Bearish (-2.0% < 200 SMA)"), "BEARISH")
        self.assertEqual(get_market_bias("Neutral (Error)"), "NEUTRAL")

    def test_news_sentiment_mapping(self):
        self.assertEqual(get_news_status("Bullish (+0.25)"), "BULLISH_NEWS")
        self.assertEqual(get_news_status("Bearish (-0.25)"), "BEARISH_NEWS")
        self.assertEqual(get_news_status("Neutral (No News)"), "NEUTRAL_NEWS")

    def test_bot_arguments_support_custom_ticker_and_timeframe(self):
        args = parse_args(["--ticker", "nvda", "--timeframe", "1h"])

        self.assertEqual(args.ticker, "nvda")
        self.assertEqual(args.timeframe, "1h")

    def test_news_parser_handles_nested_yahoo_records(self):
        records = [{
            "content": {
                "title": "Company reports strong earnings",
                "provider": {"displayName": "Example News"},
            }
        }]

        self.assertEqual(
            parse_news_headlines(records),
            ["- Company reports strong earnings (Example News)"],
        )

    def test_news_parser_handles_legacy_records(self):
        records = [{"title": "Markets open higher", "publisher": "Wire Service"}]

        self.assertEqual(
            parse_news_headlines(records),
            ["- Markets open higher (Wire Service)"],
        )

    def test_watchlist_normalization_deduplicates_and_limits_symbols(self):
        tickers = [" amd ", "NVDA", "AMD", "QCOM", "AAPL"]

        self.assertEqual(
            normalize_watchlist(tickers, max_items=3),
            ["AMD", "NVDA", "QCOM"],
        )

    def test_movers_rank_positive_relative_gains(self):
        data = pd.DataFrame([
            {"Ticker": "SLOW", "Price": 100, "Change": 1, "Relative Gain": 0.5, "Day Range": 2, "RVOL": 1},
            {"Ticker": "FAST", "Price": 100, "Change": 5, "Relative Gain": 4.5, "Day Range": 4, "RVOL": 2},
            {"Ticker": "WEAK", "Price": 100, "Change": -2, "Relative Gain": -2.5, "Day Range": 8, "RVOL": 3},
        ])

        result = rank_movers(data, limit=2)

        self.assertEqual(result["Ticker"].tolist(), ["FAST", "SLOW"])

    def test_movers_return_empty_for_no_positive_relative_gains(self):
        data = pd.DataFrame([
            {"Ticker": "WEAK", "Price": 100, "Change": -2, "Relative Gain": -2.5, "Day Range": 8, "RVOL": 3},
        ])

        self.assertTrue(rank_movers(data).empty)

    def test_profile_summary_handles_missing_business_description(self):
        self.assertEqual(
            format_profile_summary({"summary": ""}),
            "Business description unavailable.",
        )

    def test_profile_summary_is_not_truncated(self):
        summary = format_profile_summary({"summary": "x" * 300})

        self.assertEqual(len(summary), 300)


class IntradayMetricTests(unittest.TestCase):
    def test_rvol_uses_matching_prior_day_slots(self):
        timestamps = pd.to_datetime([
            "2026-09-01 09:30",
            "2026-09-02 09:30",
            "2026-09-03 09:30",
        ])
        data = pd.DataFrame({"Volume": [100, 200, 600]}, index=timestamps)

        self.assertEqual(calculate_relative_volume(data), 4.0)

    def test_rvol_returns_na_without_prior_session_data(self):
        timestamps = pd.to_datetime(["2026-09-03 09:30"])
        data = pd.DataFrame({"Volume": [600]}, index=timestamps)

        self.assertEqual(calculate_relative_volume(data), "N/A")


class OptionsMetricTests(unittest.TestCase):
    def test_expirations_are_ranked_near_strategy_horizon(self):
        expirations = ["2026-09-08", "2026-09-12", "2026-09-26"]

        result = select_expiration_candidates(
            expirations,
            analysis_mode="Intra-Day (Scalp/Day Trade)",
            as_of=pd.Timestamp("2026-09-05").date(),
        )

        self.assertEqual(result[0], "2026-09-12")

    def test_option_midpoint_prefers_valid_bid_ask(self):
        self.assertEqual(option_midpoint({"bid": 2.0, "ask": 3.0, "lastPrice": 9.0}), 2.5)

    def test_option_midpoint_falls_back_to_last_price(self):
        option = {"bid": 0.0, "ask": 0.0, "lastPrice": 4.0}

        self.assertEqual(option_midpoint(option), 4.0)
        self.assertFalse(has_valid_bid_ask(option))


class TradeReviewValidationTests(unittest.TestCase):
    def test_valid_response_is_normalized(self):
        review = validate_trade_review(valid_trade_review(), LONG_ONLY_LEVELS)

        self.assertEqual((review["direction"], review["verdict"], review["grade"]), ("LONG", "WAIT", "B"))
        self.assertEqual(review["risk_level"], "MEDIUM")
        self.assertEqual(review["technical_notes"], ["Above the rising 50-day.", "Stop sits under the base low."])
        self.assertEqual(review["validation_notes"], [])

    def test_invalid_verdict_is_rejected(self):
        result = valid_trade_review()
        result["verdict"] = "MAYBE"

        with self.assertRaises(ValueError):
            validate_trade_review(result, LONG_ONLY_LEVELS)

    def test_missing_grade_is_rejected(self):
        result = valid_trade_review()
        del result["grade"]

        with self.assertRaises(ValueError):
            validate_trade_review(result, LONG_ONLY_LEVELS)

    def test_direction_without_an_offered_plan_becomes_no_trade(self):
        result = valid_trade_review()
        result.update(direction="SHORT", verdict="GO")

        review = validate_trade_review(result, LONG_ONLY_LEVELS)

        self.assertEqual((review["direction"], review["verdict"]), ("NONE", "SKIP"))
        self.assertIn("no SHORT plan", review["validation_notes"][0])

    def test_unknown_risk_level_defaults_to_high(self):
        result = valid_trade_review()
        result["risk_level"] = "spicy"

        self.assertEqual(validate_trade_review(result, LONG_ONLY_LEVELS)["risk_level"], "HIGH")


if __name__ == "__main__":
    unittest.main()