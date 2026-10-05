import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from breakout_backtest import backtest_ticker, simulate_trade, summarize_trades, TRADE_COLUMNS
from risk_engine import position_size
from smallcap_screener import classify_regime, label_trade_regimes
from technical_engine import EASTERN, get_market_session, should_apply_live_price
from test_breakout import breakout_setup_frame
from trade_levels import average_true_range, compute_trade_plans, structure_levels


def eastern(*args):
    return datetime(*args, tzinfo=EASTERN)


def daily_bars(count=30, close=100.0, spread=2.0):
    index = pd.bdate_range("2026-08-03", periods=count)
    closes = np.full(count, close)
    return pd.DataFrame({"Open": closes, "High": closes + spread / 2, "Low": closes - spread / 2, "Close": closes}, index=index)


class TradeLevelTests(unittest.TestCase):
    def test_atr_of_steady_bars_is_their_range(self):
        self.assertAlmostEqual(average_true_range(daily_bars(spread=2.0)), 2.0)

    def test_atr_needs_enough_history(self):
        self.assertIsNone(average_true_range(daily_bars(count=10)))

    def test_structure_levels_use_recent_extremes(self):
        df = daily_bars()
        df.iloc[-5, df.columns.get_loc("High")] = 110.0
        df.iloc[-25, df.columns.get_loc("Low")] = 80.0  # outside the 20-bar window
        self.assertEqual(structure_levels(df), {"support": 99.0, "resistance": 110.0})

    def test_swing_plans_use_atr_stops_and_2r_targets(self):
        levels = compute_trade_plans(100.0, daily_bars(spread=2.0), "Weekly (Swing/Position)")
        self.assertEqual(levels["source"], "atr")
        self.assertEqual(levels["long"], {"entry": 100.0, "stop": 97.0, "target": 106.0, "reward_risk": 2.0})
        self.assertEqual(levels["short"], {"entry": 100.0, "stop": 103.0, "target": 94.0, "reward_risk": 2.0})

    def test_intraday_plans_use_tighter_stops(self):
        levels = compute_trade_plans(100.0, daily_bars(spread=2.0), "Intra-Day (Scalp/Day Trade)")
        self.assertEqual(levels["long"]["stop"], 99.0)

    def test_scanner_levels_win_and_are_long_only(self):
        row = {"Pivot": 12.4, "Stop": 11.5, "Target": 15.1}
        levels = compute_trade_plans(12.0, daily_bars(), "Weekly (Swing/Position)", scanner_row=row)
        self.assertEqual(levels["source"], "scanner")
        self.assertEqual(levels["long"], {"entry": 12.4, "stop": 11.5, "target": 15.1, "reward_risk": 3.0})
        self.assertIsNone(levels["short"])

    def test_no_plans_without_history(self):
        levels = compute_trade_plans(100.0, pd.DataFrame(), "Weekly (Swing/Position)")
        self.assertIsNone(levels["source"])
        self.assertIsNone(levels["long"])


class MarketSessionTests(unittest.TestCase):
    def test_stock_sessions_follow_the_eastern_clock(self):
        self.assertEqual(get_market_session("AMD", eastern(2026, 10, 5, 10, 0)), "REGULAR")
        self.assertEqual(get_market_session("AMD", eastern(2026, 10, 5, 8, 0)), "PRE")
        self.assertEqual(get_market_session("AMD", eastern(2026, 10, 5, 17, 0)), "POST")
        self.assertEqual(get_market_session("AMD", eastern(2026, 10, 5, 22, 0)), "CLOSED")
        self.assertEqual(get_market_session("AMD", eastern(2026, 10, 4, 12, 0)), "CLOSED")

    def test_crypto_and_fx_hours(self):
        self.assertEqual(get_market_session("BTC-USD", eastern(2026, 10, 4, 3, 0)), "REGULAR")
        self.assertEqual(get_market_session("EURUSD=X", eastern(2026, 10, 3, 12, 0)), "CLOSED")
        self.assertEqual(get_market_session("EURUSD=X", eastern(2026, 10, 4, 18, 0)), "REGULAR")
        self.assertEqual(get_market_session("EURUSD=X", eastern(2026, 10, 9, 17, 30)), "CLOSED")

    def test_live_price_only_updates_a_current_candle(self):
        now = eastern(2026, 10, 5, 10, 2)
        self.assertTrue(should_apply_live_price(pd.Timestamp("2026-10-05 10:00"), "5m", "REGULAR", now))
        self.assertFalse(should_apply_live_price(pd.Timestamp("2026-10-05 09:50"), "5m", "REGULAR", now))
        self.assertTrue(should_apply_live_price(pd.Timestamp("2026-10-05"), "1d", "REGULAR", now))
        self.assertFalse(should_apply_live_price(pd.Timestamp("2026-10-02"), "1d", "REGULAR", now))

    def test_extended_hours_quotes_skip_daily_bars(self):
        now = eastern(2026, 10, 5, 17, 2)
        self.assertFalse(should_apply_live_price(pd.Timestamp("2026-10-05"), "1d", "POST", now))
        self.assertTrue(should_apply_live_price(pd.Timestamp("2026-10-05 17:00"), "5m", "POST", now))
        self.assertFalse(should_apply_live_price(pd.Timestamp("2026-10-05 17:00"), "5m", "CLOSED", now))


class PositionSizeTests(unittest.TestCase):
    def test_shares_are_sized_from_the_stop(self):
        size = position_size(25_000, 1.0, entry=20.0, stop=19.0)
        self.assertEqual(size["shares"], 250)
        self.assertEqual(size["dollar_risk"], 250.0)
        self.assertFalse(size["capped_by_account"])

    def test_short_positions_use_the_absolute_risk(self):
        self.assertEqual(position_size(10_000, 1.0, entry=50.0, stop=52.0)["shares"], 50)

    def test_tight_stop_is_capped_by_account_size(self):
        size = position_size(10_000, 2.0, entry=100.0, stop=99.9)
        self.assertEqual(size["shares"], 100)
        self.assertTrue(size["capped_by_account"])

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(position_size(10_000, 1.0, entry=20.0, stop=20.0))
        self.assertIsNone(position_size(0, 1.0, entry=20.0, stop=19.0))
        self.assertIsNone(position_size(10_000, 1.0, entry=None, stop=19.0))
        self.assertIsNone(position_size(100, 0.1, entry=20.0, stop=10.0))


def bars(rows):
    index = pd.bdate_range("2026-01-01", periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)


class TradeSimulationTests(unittest.TestCase):
    def test_trade_hits_target(self):
        df = bars([(10, 10, 10, 10), (10, 10.6, 9.9, 10.5), (10.5, 12.5, 10.4, 12.2)])
        trade = simulate_trade(df, 0, pivot=10.5, stop=9.5, target=12.0)
        self.assertEqual((trade["Entry"], trade["Exit"], trade["Outcome"], trade["R"]), (10.5, 12.0, "Target", 1.5))

    def test_gap_down_through_stop_fills_at_open(self):
        df = bars([(10, 10, 10, 10), (10.6, 10.8, 10.4, 10.7), (9.0, 9.2, 8.8, 9.0)])
        trade = simulate_trade(df, 0, pivot=10.5, stop=9.5, target=12.0)
        self.assertEqual((trade["Entry"], trade["Exit"], trade["Outcome"]), (10.6, 9.0, "Stop"))

    def test_bar_touching_stop_and_target_counts_as_stop(self):
        df = bars([(10, 10, 10, 10), (10, 10.6, 9.9, 10.5), (10.5, 12.5, 9.0, 11.0)])
        self.assertEqual(simulate_trade(df, 0, pivot=10.5, stop=9.5, target=12.0)["Outcome"], "Stop")

    def test_setup_that_never_triggers_is_skipped(self):
        df = bars([(10, 10, 10, 10)] + [(10, 10.2, 9.8, 10)] * 12)
        self.assertIsNone(simulate_trade(df, 0, pivot=10.5, stop=9.5, target=12.0))

    def test_setup_that_breaks_its_stop_before_triggering_is_skipped(self):
        df = bars([(10, 10, 10, 10), (10, 10.1, 9.4, 9.6), (9.6, 11, 9.6, 10.8)])
        self.assertIsNone(simulate_trade(df, 0, pivot=10.5, stop=9.5, target=12.0))

    def test_backtest_finds_the_known_setup_and_summarizes(self):
        frame = breakout_setup_frame()
        # Add a breakout leg after the setup bar so the trade can trigger and run to the target.
        last = frame.iloc[-1]
        extra_index = pd.bdate_range(frame.index[-1] + pd.offsets.BDay(), periods=20)
        closes = np.linspace(last["Close"] * 1.03, last["Close"] * 1.4, 20)
        extra = pd.DataFrame({
            "Open": closes * 0.99, "High": closes * 1.01, "Low": closes * 0.98, "Close": closes, "Volume": 2_000_000.0,
        }, index=extra_index)
        trades = backtest_ticker(pd.concat([frame, extra]), min_reward_risk=1.0, min_dollar_volume=1_000_000)
        self.assertGreaterEqual(len(trades), 1)
        self.assertEqual(trades[0]["Outcome"], "Target")

        summary = summarize_trades(pd.DataFrame([{"Ticker": "X", **t} for t in trades], columns=TRADE_COLUMNS))
        self.assertEqual(summary["win_rate_pct"], 100.0)
        self.assertGreater(summary["average_r"], 0)

    def test_summary_excludes_open_trades(self):
        trades = pd.DataFrame({"R": [2.0, -1.0, 0.5], "Outcome": ["Target", "Stop", "Open"]})
        summary = summarize_trades(trades)
        self.assertEqual((summary["trades"], summary["still_open"], summary["average_r"]), (2, 1, 0.5))
        self.assertEqual(summarize_trades(pd.DataFrame(columns=TRADE_COLUMNS))["trades"], 0)


class RegimeTests(unittest.TestCase):
    def test_rising_series_is_an_uptrend(self):
        self.assertEqual(classify_regime(pd.Series(np.linspace(100, 200, 260)))["label"], "Uptrend")

    def test_falling_series_is_a_downtrend(self):
        self.assertEqual(classify_regime(pd.Series(np.linspace(200, 100, 260)))["label"], "Downtrend")

    def test_rebound_inside_a_downtrend_is_mixed(self):
        closes = np.concatenate([np.linspace(200, 100, 240), np.linspace(100, 160, 20)])
        self.assertEqual(classify_regime(pd.Series(closes))["label"], "Mixed")

    def test_short_or_missing_history_is_unknown(self):
        self.assertEqual(classify_regime(pd.Series(np.linspace(100, 110, 50)))["label"], "Unknown")
        self.assertEqual(classify_regime(None)["label"], "Unknown")

    def test_trades_are_tagged_with_the_signal_day_regime(self):
        index = pd.bdate_range("2026-01-01", periods=120)
        benchmark = pd.Series(np.concatenate([np.linspace(100, 150, 80), np.linspace(150, 90, 40)]), index=index)
        trades = pd.DataFrame({"Signal Date": [index[70], index[119]]})
        tagged = label_trade_regimes(trades, benchmark)
        self.assertEqual(list(tagged["IWM > 50d"]), [True, False])


if __name__ == "__main__":
    unittest.main()
