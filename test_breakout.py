import unittest
from unittest import mock

import numpy as np
import pandas as pd

from breakout_engine import analyze_breakout_setup, find_base, qualifies, rank_breakout_candidates, score_setup
from llm_engine import format_breakout_candidate, validate_breakout_reviews
import smallcap_screener
from smallcap_screener import FALLBACK_SMALLCAPS, parse_screener_quotes


def make_frame(closes, volumes=None, spread=0.01):
    closes = np.asarray(closes, dtype=float)
    index = pd.bdate_range("2025-01-01", periods=len(closes))
    volumes = np.full(len(closes), 1_000_000.0) if volumes is None else np.asarray(volumes, dtype=float)
    return pd.DataFrame({
        "Open": closes,
        "High": closes * (1 + spread),
        "Low": closes * (1 - spread),
        "Close": closes,
        "Volume": volumes,
    }, index=index)


def breakout_setup_frame():
    """Uptrend, a 30-bar base with a 20% pullback, then a tight final 10 bars just under the pivot."""
    uptrend = np.linspace(10, 20, 200)
    pullback = np.concatenate([np.linspace(20, 16, 10), np.linspace(16, 19.4, 10)])
    tight = 19.4 + 0.1 * np.sin(np.arange(10))
    closes = np.concatenate([uptrend, pullback, tight, [19.5]])
    volumes = np.concatenate([np.full(220, 1_000_000), np.full(10, 400_000), [900_000]])
    return make_frame(closes, volumes, spread=0.005)


def make_setup(**overrides):
    setup = {
        "price": 19.5, "pivot": 20.1, "distance_to_pivot": 0.03, "stop": 19.0, "target": 24.0,
        "reward_risk": 3.5, "base_bars": 30, "base_depth": 0.2, "contraction_ratio": 0.2,
        "squeeze_rank": 0.1, "dryup_ratio": 0.5, "rvol": 1.2, "above_sma_50": True,
        "trend_points": 1.0, "relative_strength": 10.0, "avg_dollar_volume": 20_000_000,
    }
    setup.update(overrides)
    return setup


class BaseDetectionTests(unittest.TestCase):
    def test_longest_base_within_depth_is_selected(self):
        frame = make_frame(np.concatenate([np.linspace(10, 30, 60), np.full(40, 30.0)]))

        base = find_base(frame)

        self.assertEqual(base["window"], 40)
        self.assertLess(base["depth"], 0.05)

    def test_no_base_when_every_window_is_too_deep(self):
        frame = make_frame(np.concatenate([np.full(20, 30.0), np.linspace(30, 10, 40)]))

        self.assertIsNone(find_base(frame, max_depth=0.10))


class BreakoutSetupTests(unittest.TestCase):
    def test_setup_levels_follow_measured_move(self):
        setup = analyze_breakout_setup(breakout_setup_frame())

        self.assertIsNotNone(setup)
        self.assertGreater(setup["pivot"], setup["price"])
        self.assertLess(setup["distance_to_pivot"], 0.08)
        self.assertLess(setup["stop"], setup["price"])
        # Target projects the base depth above the pivot
        base_low = setup["pivot"] * (1 - setup["base_depth"])
        self.assertAlmostEqual(setup["target"], 2 * setup["pivot"] - base_low, places=6)
        self.assertGreaterEqual(setup["reward_risk"], 3.0)
        self.assertTrue(setup["above_sma_50"])
        self.assertLess(setup["dryup_ratio"], 1.0)

    def test_stop_is_at_least_one_atr_below_pivot(self):
        setup = analyze_breakout_setup(breakout_setup_frame())
        frame = breakout_setup_frame()
        true_range = (frame["High"] - frame["Low"]).rolling(14).mean().iloc[-1]

        self.assertLessEqual(setup["stop"], setup["pivot"] - true_range * 0.99)

    def test_relative_strength_uses_benchmark(self):
        frame = breakout_setup_frame()
        flat_benchmark = pd.Series(100.0, index=frame.index)
        rising_benchmark = pd.Series(np.linspace(50, 200, len(frame)), index=frame.index)

        flat = analyze_breakout_setup(frame, flat_benchmark)
        rising = analyze_breakout_setup(frame, rising_benchmark)

        self.assertGreater(flat["relative_strength"], rising["relative_strength"])

    def test_insufficient_history_returns_none(self):
        self.assertIsNone(analyze_breakout_setup(make_frame(np.linspace(10, 12, 40))))

    def test_missing_columns_return_none(self):
        self.assertIsNone(analyze_breakout_setup(pd.DataFrame({"Close": np.linspace(10, 12, 100)})))


class RankingTests(unittest.TestCase):
    def test_score_rewards_tighter_setups(self):
        tight = score_setup(make_setup(contraction_ratio=0.1, distance_to_pivot=0.01))
        loose = score_setup(make_setup(contraction_ratio=0.9, distance_to_pivot=0.07))

        self.assertGreater(tight, loose)
        self.assertLessEqual(tight, 100)

    def test_qualification_filters(self):
        self.assertTrue(qualifies(make_setup(), 3.0, 5_000_000))
        self.assertFalse(qualifies(make_setup(reward_risk=2.0), 3.0, 5_000_000))
        self.assertFalse(qualifies(make_setup(avg_dollar_volume=1_000_000), 3.0, 5_000_000))
        self.assertFalse(qualifies(make_setup(distance_to_pivot=0.15), 3.0, 5_000_000))
        self.assertFalse(qualifies(make_setup(distance_to_pivot=-0.05), 3.0, 5_000_000))
        self.assertFalse(qualifies(make_setup(above_sma_50=False), 3.0, 5_000_000))
        self.assertFalse(qualifies(None, 3.0, 5_000_000))

    def test_rank_orders_by_score_and_drops_unqualified(self):
        setups = {
            "TIGHT": make_setup(contraction_ratio=0.1),
            "LOOSE": make_setup(contraction_ratio=0.8),
            "LOWRR": make_setup(reward_risk=1.5),
            "NODATA": None,
        }

        ranked = rank_breakout_candidates(setups, {"TIGHT": {"name": "Tight Co", "market_cap": 1e9}})

        self.assertEqual(list(ranked["Ticker"]), ["TIGHT", "LOOSE"])
        self.assertEqual(ranked.loc[0, "Name"], "Tight Co")
        self.assertEqual(ranked.loc[0, "Base Weeks"], 6.0)

    def test_rank_returns_empty_frame_with_columns(self):
        ranked = rank_breakout_candidates({})

        self.assertTrue(ranked.empty)
        self.assertIn("Breakout Score", ranked.columns)


class ScreenerParsingTests(unittest.TestCase):
    def test_quotes_are_parsed_and_foreign_or_index_symbols_skipped(self):
        response = {"quotes": [
            {"symbol": "abcd", "shortName": "ABCD Inc", "marketCap": 900_000_000},
            {"symbol": "XYZ.TO", "shortName": "Foreign"},
            {"symbol": "^RUT"},
            {"shortName": "No symbol"},
            "not a dict",
        ]}

        universe = parse_screener_quotes(response)

        self.assertEqual(universe, {"ABCD": {"name": "ABCD Inc", "market_cap": 900_000_000}})

    def test_malformed_response_returns_empty_universe(self):
        self.assertEqual(parse_screener_quotes(None), {})
        self.assertEqual(parse_screener_quotes({"quotes": None}), {})


class ScreenerFallbackTests(unittest.TestCase):
    def setUp(self):
        smallcap_screener.get_smallcap_universe.clear()
        smallcap_screener.scan_smallcap_breakouts.clear()

    def test_bounds_drop_quotes_outside_small_cap_range(self):
        response = {"quotes": [
            {"symbol": "IN", "marketCap": 900_000_000, "regularMarketPrice": 12.0},
            {"symbol": "BIG", "marketCap": 9_000_000_000, "regularMarketPrice": 50.0},
            {"symbol": "PENNY", "marketCap": 500_000_000, "regularMarketPrice": 0.8},
            {"symbol": "NOCAP", "regularMarketPrice": 10.0},
        ]}
        self.assertEqual(list(parse_screener_quotes(response, apply_bounds=True)), ["IN"])

    def nasdaq_payload(self):
        return {"data": {"rows": [
            {"symbol": "LIQ", "name": "Liquid Co", "lastsale": "$20.00", "volume": "2,000,000", "marketCap": "1,500,000,000.00"},
            {"symbol": "THIN", "name": "Thin Co", "lastsale": "$5.00", "volume": "400,000", "marketCap": "800,000,000.00"},
            {"symbol": "LOWVOL", "name": "Quiet Co", "lastsale": "$30.00", "volume": "1,000", "marketCap": "900,000,000.00"},
            {"symbol": "MEGA", "name": "Mega Co", "lastsale": "$300.00", "volume": "9,000,000", "marketCap": "900,000,000,000.00"},
            {"symbol": "PENNY", "name": "Penny Co", "lastsale": "$1.20", "volume": "9,000,000", "marketCap": "400,000,000.00"},
            {"symbol": "ABC^A", "name": "ABC Preferred", "lastsale": "$25.00", "volume": "900,000", "marketCap": "1,000,000,000.00"},
            {"symbol": "NOCAP", "name": "No Cap", "lastsale": "$10.00", "volume": "900,000", "marketCap": ""},
        ]}}

    def nasdaq_response(self):
        response = mock.Mock()
        response.json.return_value = self.nasdaq_payload()
        response.raise_for_status.return_value = None
        return response

    def test_nasdaq_rows_keep_liquid_small_caps_ranked_by_dollar_volume(self):
        universe = smallcap_screener.parse_nasdaq_rows(self.nasdaq_payload())
        self.assertEqual(list(universe), ["LIQ", "THIN"])
        self.assertEqual(universe["LIQ"], {"name": "Liquid Co", "market_cap": 1_500_000_000.0})
        self.assertEqual(list(smallcap_screener.parse_nasdaq_rows(self.nasdaq_payload(), limit=1)), ["LIQ"])

    def test_nasdaq_rows_tolerate_malformed_payloads(self):
        self.assertEqual(smallcap_screener.parse_nasdaq_rows(None), {})
        self.assertEqual(smallcap_screener.parse_nasdaq_rows({"data": None}), {})
        self.assertEqual(smallcap_screener.parse_nasdaq_rows({"data": {"rows": [None, "x"]}}), {})

    def test_universe_uses_nasdaq_when_yahoo_returns_401(self):
        with mock.patch("yfinance.screen", side_effect=Exception("401 Client Error: Unauthorized")) as screen, \
                mock.patch("requests.get", return_value=self.nasdaq_response()):
            source, universe = smallcap_screener.get_smallcap_universe()
        self.assertEqual(source, "Nasdaq screener")
        self.assertEqual(list(universe), ["LIQ", "THIN"])
        self.assertEqual(screen.call_count, 1)

    def test_predefined_yahoo_screen_uses_get_endpoint_without_offset(self):
        quotes = {"quotes": [{"symbol": "ABC", "marketCap": 800_000_000, "regularMarketPrice": 9.0}]}
        with mock.patch("yfinance.screen", side_effect=[Exception("401"), quotes]) as screen, \
                mock.patch("requests.get", side_effect=Exception("403 Forbidden")):
            source, universe = smallcap_screener.get_smallcap_universe()
        self.assertEqual(source, "Yahoo small-cap screen")
        self.assertEqual(list(universe), ["ABC"])
        self.assertEqual(screen.call_args.args, ("small_cap_gainers",))
        self.assertNotIn("offset", screen.call_args.kwargs)

    def test_universe_raises_when_every_source_fails_so_failure_is_not_cached(self):
        with mock.patch("yfinance.screen", side_effect=Exception("401 Invalid Crumb")), \
                mock.patch("requests.get", side_effect=Exception("403 Forbidden")):
            with self.assertRaisesRegex(RuntimeError, "Invalid Crumb.*403 Forbidden"):
                smallcap_screener.get_smallcap_universe()
        quotes = {"quotes": [{"symbol": "ABC", "marketCap": 800_000_000}]}
        with mock.patch("yfinance.screen", return_value=quotes):
            self.assertEqual(smallcap_screener.get_smallcap_universe(), ("Yahoo screener", {
                "ABC": {"name": "ABC", "market_cap": 800_000_000},
            }))

    def test_scan_uses_fallback_list_when_every_source_fails(self):
        frames = {ticker: make_frame(np.linspace(10, 12, 260)) for ticker in FALLBACK_SMALLCAPS + ["IWM"]}
        batch = pd.concat(frames, axis=1)
        with mock.patch("yfinance.screen", side_effect=Exception("401 Invalid Crumb")), \
                mock.patch("requests.get", side_effect=Exception("403 Forbidden")), \
                mock.patch("yfinance.download", return_value=batch) as download:
            result = smallcap_screener.scan_smallcap_breakouts()
        self.assertEqual(result.attrs["universe_source"], "fallback list")
        self.assertIn("Invalid Crumb", result.attrs["screener_error"])
        self.assertEqual(result.attrs["universe_size"], len(FALLBACK_SMALLCAPS))
        self.assertEqual(set(download.call_args.kwargs["tickers"]), set(FALLBACK_SMALLCAPS) | {"IWM"})

    def test_scan_records_nasdaq_as_universe_source(self):
        batch = pd.concat({ticker: make_frame(np.linspace(10, 12, 260)) for ticker in ["LIQ", "THIN", "IWM"]}, axis=1)
        with mock.patch("yfinance.screen", side_effect=Exception("401")), \
                mock.patch("requests.get", return_value=self.nasdaq_response()), \
                mock.patch("yfinance.download", return_value=batch):
            result = smallcap_screener.scan_smallcap_breakouts()
        self.assertEqual(result.attrs["universe_source"], "Nasdaq screener")
        self.assertIsNone(result.attrs["screener_error"])
        self.assertEqual(result.attrs["universe_size"], 2)

    def test_scan_raises_when_no_prices_download(self):
        quotes = {"quotes": [{"symbol": "ABC", "marketCap": 800_000_000}]}
        with mock.patch("yfinance.screen", return_value=quotes), \
                mock.patch("yfinance.download", return_value=pd.DataFrame()):
            with self.assertRaisesRegex(RuntimeError, "No price data"):
                smallcap_screener.scan_smallcap_breakouts()


class BreakoutReviewValidationTests(unittest.TestCase):
    def test_valid_reviews_are_normalized(self):
        result = {"reviews": [
            {"ticker": "abcd", "grade": "a", "risk_level": "medium", "catalyst": "Contract win",
             "thesis": "Clean base.", "red_flags": "Earnings in 6 days"},
        ]}

        reviews = validate_breakout_reviews(result, ["ABCD"])

        self.assertEqual(reviews["ABCD"]["grade"], "A")
        self.assertEqual(reviews["ABCD"]["risk_level"], "MEDIUM")
        self.assertEqual(reviews["ABCD"]["red_flags"], ["Earnings in 6 days"])

    def test_unknown_tickers_and_grades_are_dropped(self):
        result = {"reviews": [
            {"ticker": "ZZZZ", "grade": "A"},
            {"ticker": "ABCD", "grade": "F"},
            {"ticker": "EFGH", "grade": "B", "risk_level": "unknown"},
        ]}

        reviews = validate_breakout_reviews(result, ["ABCD", "EFGH"])

        self.assertEqual(list(reviews), ["EFGH"])
        self.assertEqual(reviews["EFGH"]["risk_level"], "HIGH")
        self.assertEqual(reviews["EFGH"]["catalyst"], "None identified")

    def test_missing_reviews_list_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_breakout_reviews({"signal": "BUY"}, ["ABCD"])


class BreakoutCandidateFormatTests(unittest.TestCase):
    CANDIDATE = {
        "Ticker": "ABCD", "Name": "Abcd Inc", "Price": 20.3, "Market Cap": 900_000_000, "Pivot": 20.0,
        "To Pivot": 3.5, "Stop": 18.5, "Target": 25.0, "Reward/Risk": 3.3, "Base Weeks": 6.0,
        "Base Depth": 20.0, "RVOL": 1.2, "RS vs IWM": 8.0, "Breakout Score": 70.0,
    }

    def test_pivot_below_price_is_described_as_distance_to_go(self):
        block = format_breakout_candidate(self.CANDIDATE, {})

        self.assertIn("20.0 USD, 3.5% above price", block)

    def test_extended_price_is_not_described_as_negative_distance(self):
        block = format_breakout_candidate({**self.CANDIDATE, "To Pivot": -1.5}, {})

        self.assertIn("20.0 USD, price already 1.5% above it", block)
        self.assertNotIn("-1.5%", block)


if __name__ == "__main__":
    unittest.main()
