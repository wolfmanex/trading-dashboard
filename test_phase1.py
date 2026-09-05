import unittest

import numpy as np
import pandas as pd

from bot import calculate_ta_score, get_market_bias, get_news_status, parse_args
from backtest_engine import run_ta_backtest
from event_engine import parse_news_headlines
from watchlist_engine import normalize_watchlist
from intraday_engine import calculate_relative_volume
from llm_engine import validate_synthesis_result
from options_engine import has_valid_bid_ask, option_midpoint, select_expiration_candidates
from technical_engine import add_technical_indicators, tag_data_source, validate_ohlcv_data


def valid_synthesis_result():
    return {
        "signal": "buy",
        "confidence": 85,
        "timeframe_confluence": "Bullish alignment",
        "execution_plan": {},
        "higher_tf_breakdown": "- Trend is bullish.",
        "intraday_tf_breakdown": "- Momentum is improving.",
        "macro_analysis": "- Macro trend is supportive.",
        "news_catalyst_analysis": "- No immediate catalyst risk.",
        "catalyst_scenarios": "- Bull case remains valid.",
        "detailed_reasoning": "The technical inputs are aligned.",
    }


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

        self.assertEqual(result["total_trades"], 4)
        self.assertEqual(result["win_rate_pct"], 100.0)

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


class SynthesisValidationTests(unittest.TestCase):
    def test_valid_response_is_normalized(self):
        result = validate_synthesis_result(valid_synthesis_result())

        self.assertEqual(result["signal"], "BUY")
        self.assertEqual(result["confidence"], 0.85)

    def test_invalid_signal_is_rejected(self):
        result = valid_synthesis_result()
        result["signal"] = "MAYBE"

        with self.assertRaises(ValueError):
            validate_synthesis_result(result)

    def test_missing_required_field_is_rejected(self):
        result = valid_synthesis_result()
        del result["execution_plan"]

        with self.assertRaises(ValueError):
            validate_synthesis_result(result)

    def test_out_of_range_confidence_is_rejected(self):
        result = valid_synthesis_result()
        result["confidence"] = 150

        with self.assertRaises(ValueError):
            validate_synthesis_result(result)


if __name__ == "__main__":
    unittest.main()