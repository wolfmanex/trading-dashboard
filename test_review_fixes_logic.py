import io
import unittest
from unittest import mock

import pandas as pd

import swing_engine
from backtest_engine import run_ta_backtest
from breakout_backtest import simulate_trade
from journal import LONG, add_trade, close_trade, empty_journal, parse_journal, update_stop
from signal_log import read_log
from technical_engine import resample_to_4h


def ta_frame(closes):
    n = len(closes)
    return pd.DataFrame({
        "Close": closes, "EMA_9": [101.0] * n, "EMA_21": [100.0] * n,
        "MACD": [2.0] * n, "Signal_Line": [1.0] * n, "RSI": [50.0] * n,
    })


class BacktestHoldingPeriodTests(unittest.TestCase):
    def test_holding_period_one_holds_one_bar(self):
        # Signal at bar 1, entry at bar 2's close, exit one bar later
        result = run_ta_backtest(ta_frame([100.0, 110.0, 120.0, 132.0]), holding_period=1)
        self.assertEqual(result["total_trades"], 1)
        self.assertAlmostEqual(result["average_return_pct"], 10.0, places=2)

    def test_too_short_history_has_no_trades(self):
        self.assertEqual(run_ta_backtest(ta_frame([100.0, 110.0, 120.0]), holding_period=2)["total_trades"], 0)


def ohlc(rows):
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"],
                        index=pd.date_range("2026-01-01", periods=len(rows), freq="D"))


class BreakoutGapTests(unittest.TestCase):
    def test_gap_above_target_on_entry_bar_fills_at_open(self):
        df = ohlc([[10, 10, 9.5, 10], [13, 13.5, 12.8, 13.2]])
        trade = simulate_trade(df, 0, pivot=10.5, stop=9.0, target=12.0)
        self.assertEqual(trade["Outcome"], "Target")
        self.assertEqual(trade["Entry"], 13.0)
        self.assertEqual(trade["Exit"], 13.0)
        self.assertGreaterEqual(trade["R"], 0)

    def test_gap_above_target_after_entry_fills_at_open(self):
        df = ohlc([[10, 10, 9.5, 10], [10.2, 10.8, 10.1, 10.7], [12.5, 12.6, 12.4, 12.5]])
        trade = simulate_trade(df, 0, pivot=10.5, stop=9.0, target=12.0)
        self.assertEqual(trade["Exit"], 12.5)
        self.assertEqual(trade["R"], 1.33)


class NaTickerTests(unittest.TestCase):
    def test_signal_log_keeps_na_ticker_and_numeric_blanks(self):
        log = read_log(io.StringIO("Ticker,Pivot,R\nNA,10.5,\nNULL,,1.5\n"))
        self.assertEqual(list(log["Ticker"]), ["NA", "NULL"])
        self.assertEqual(log["Pivot"].iloc[0], 10.5)
        self.assertTrue(pd.isna(log["Pivot"].iloc[1]))
        self.assertTrue(pd.isna(log["R"].iloc[0]))

    def test_journal_keeps_na_ticker(self):
        journal = parse_journal("ID,Ticker,Entry,Stop\nabc,N/A,10,\n")
        self.assertEqual(journal["Ticker"].iloc[0], "N/A")
        self.assertEqual(journal["Entry"].iloc[0], 10)
        self.assertTrue(pd.isna(journal["Stop"].iloc[0]))


class JournalValidationTests(unittest.TestCase):
    def setUp(self):
        self.journal = add_trade(empty_journal(), "ABC", LONG, "2026-10-05", 10.0, 100, 9.0, 13.0)
        self.trade_id = self.journal.iloc[0]["ID"]

    def test_add_trade_rejects_non_positive_stop(self):
        with self.assertRaises(ValueError):
            add_trade(empty_journal(), "ABC", LONG, "2026-10-05", 10.0, 100, 0.0)

    def test_update_stop_rejects_non_positive_stop_but_allows_profit_lock(self):
        with self.assertRaises(ValueError):
            update_stop(self.journal, self.trade_id, -1.0)
        self.assertEqual(update_stop(self.journal, self.trade_id, 11.0).iloc[0]["Stop"], 11.0)

    def test_close_trade_rejects_exit_before_entry(self):
        with self.assertRaises(ValueError):
            close_trade(self.journal, self.trade_id, 12.0, "2026-10-04")
        self.assertEqual(close_trade(self.journal, self.trade_id, 12.0, "2026-10-05").iloc[0]["Exit"], 12.0)


class SwingRelativeStrengthTests(unittest.TestCase):
    def test_one_week_change_spans_five_sessions(self):
        closes = {"AAPL": [100.0] * 5 + [100.0, 101.0, 102.0, 103.0, 104.0, 110.0], "XLK": [100.0] * 11}

        def ticker(symbol):
            tk = mock.Mock(options=[])
            tk.history.side_effect = lambda period: pd.DataFrame({"Close": closes[symbol][-1:] if period == "1d" else closes[symbol]})
            return tk

        with mock.patch.object(swing_engine.yf, "Ticker", side_effect=ticker):
            metrics = swing_engine.get_swing_metrics("AAPL")
        self.assertEqual(metrics["relative_strength_1w"], 10.0)


class FourHourResampleTests(unittest.TestCase):
    def frame(self, index):
        n = len(index)
        return pd.DataFrame({"Open": range(n), "High": range(n), "Low": range(n), "Close": range(n), "Volume": [1] * n},
                            index=index, dtype=float)

    def test_stock_bars_use_regular_session_buckets(self):
        index = pd.date_range("2026-10-05 04:30", "2026-10-05 19:30", freq="h")
        bars = resample_to_4h(self.frame(index), "AAPL")
        self.assertEqual([t.strftime("%H:%M") for t in bars.index], ["09:30", "13:30"])
        self.assertEqual(list(bars["Volume"]), [4.0, 3.0])  # 9:30-12:30 and 13:30-15:30

    def test_tz_aware_index_is_filtered_in_eastern_time(self):
        index = pd.date_range("2026-10-05 08:30", "2026-10-05 23:30", freq="h", tz="UTC")
        bars = resample_to_4h(self.frame(index), "AAPL")
        self.assertEqual(int(bars["Volume"].sum()), 7)  # 13:30-19:30 UTC is 9:30-15:30 EDT

    def test_crypto_keeps_all_hours(self):
        index = pd.date_range("2026-10-05 00:00", periods=24, freq="h")
        self.assertEqual(int(resample_to_4h(self.frame(index), "BTC-USD")["Volume"].sum()), 24)


if __name__ == "__main__":
    unittest.main()
