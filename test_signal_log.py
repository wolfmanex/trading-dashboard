import io
import os
import tempfile
import unittest
from datetime import date
from unittest import mock

import pandas as pd

import scheduled_scan
from signal_log import (
    LOG_COLUMNS, NOT_TRIGGERED, OPEN, WAITING, append_signals, benchmark_return, empty_log,
    evaluate_signal, read_log, summarize_by_grade, summarize_log, update_outcomes,
)


def bars(rows, start="2026-10-01"):
    """Daily OHLC bars from (open, high, low, close) tuples on consecutive business days."""
    index = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=index)


def scan_results(tickers):
    results = pd.DataFrame({
        "Ticker": tickers,
        "Price": [9.5] * len(tickers),
        "Pivot": [10.0] * len(tickers),
        "Stop": [9.0] * len(tickers),
        "Target": [13.0] * len(tickers),
        "Reward/Risk": [3.0] * len(tickers),
        "Breakout Score": [70.0] * len(tickers),
    })
    results.attrs["universe_source"] = "Nasdaq screener"
    return results


FLAT = (9.5, 9.7, 9.4, 9.6)  # inside the base: below the pivot, above the stop


class AppendSignalsTests(unittest.TestCase):
    def test_new_setups_are_logged_with_rank_and_waiting_status(self):
        log = append_signals(empty_log(), scan_results(["AAA", "BBB"]), date(2026, 10, 5), "Uptrend")
        self.assertEqual(list(log.columns), LOG_COLUMNS)
        self.assertEqual(list(log["Ticker"]), ["AAA", "BBB"])
        self.assertEqual(list(log["Rank"]), [1, 2])
        self.assertEqual(set(log["Status"]), {WAITING})
        self.assertEqual(log["Scan Date"].iloc[0], "2026-10-05")
        self.assertEqual(log["Universe"].iloc[0], "Nasdaq screener")

    def test_live_signals_and_same_day_reruns_are_not_logged_twice(self):
        log = append_signals(empty_log(), scan_results(["AAA", "BBB"]), date(2026, 10, 5))
        log.loc[log["Ticker"] == "BBB", "Status"] = "Stop"
        rerun = append_signals(log, scan_results(["AAA", "BBB"]), date(2026, 10, 5))
        self.assertEqual(len(rerun), 2)
        next_day = append_signals(log, scan_results(["AAA", "BBB", "CCC"]), date(2026, 10, 6))
        self.assertEqual(list(next_day["Ticker"]), ["AAA", "BBB", "BBB", "CCC"])

    def test_no_results_leave_the_log_unchanged(self):
        log = append_signals(empty_log(), scan_results(["AAA"]), date(2026, 10, 5))
        self.assertIs(append_signals(log, None, date(2026, 10, 6)), log)
        self.assertIs(append_signals(log, scan_results([]), date(2026, 10, 6)), log)


class EvaluateSignalTests(unittest.TestCase):
    # Scan on Mon 2026-10-05: bars before it are the setup, bars from it on are the entry window.
    SETUP = [FLAT] * 2  # Thu 10-01, Fri 10-02

    def evaluate(self, after):
        return evaluate_signal(bars(self.SETUP + after), "2026-10-05", 10.0, 9.0, 13.0)

    def test_waits_until_the_pivot_trades(self):
        self.assertEqual(self.evaluate([FLAT] * 3), {"Status": WAITING})
        self.assertEqual(evaluate_signal(bars(self.SETUP), "2026-10-05", 10.0, 9.0, 13.0), {"Status": WAITING})

    def test_not_triggered_after_the_entry_window_or_a_stop_break(self):
        self.assertEqual(self.evaluate([FLAT] * 10)["Status"], NOT_TRIGGERED)
        self.assertEqual(self.evaluate([FLAT, (9.2, 9.3, 8.8, 8.9)])["Status"], NOT_TRIGGERED)

    def test_entry_then_target(self):
        result = self.evaluate([(9.8, 10.2, 9.7, 10.1), (10.1, 13.5, 10.0, 13.2)])
        self.assertEqual(result["Status"], "Target")
        self.assertEqual(result["Entry Date"], "2026-10-05")
        self.assertEqual(result["Entry"], 10.0)
        self.assertEqual(result["Exit Date"], "2026-10-06")
        self.assertEqual(result["R"], 3.0)
        self.assertEqual(result["Return %"], 30.0)

    def test_triggered_trade_still_running_is_open(self):
        result = self.evaluate([(9.8, 10.2, 9.7, 10.1), (10.1, 10.6, 10.0, 10.5)])
        self.assertEqual(result["Status"], OPEN)
        self.assertEqual(result["Exit"], 10.5)
        self.assertEqual(result["R"], 0.5)

    def test_bars_before_the_first_setup_day_mean_nothing_to_evaluate(self):
        self.assertEqual(evaluate_signal(bars([FLAT] * 3, start="2026-10-05"), "2026-10-05", 10, 9, 13), {"Status": WAITING})
        self.assertEqual(evaluate_signal(pd.DataFrame(), "2026-10-05", 10, 9, 13), {"Status": WAITING})


class UpdateOutcomesTests(unittest.TestCase):
    def test_live_signals_are_updated_and_finished_ones_kept(self):
        log = append_signals(empty_log(), scan_results(["AAA", "BBB", "CCC"]), date(2026, 10, 5))
        log.loc[log["Ticker"] == "CCC", ["Status", "R"]] = ["Stop", -1.0]
        frames = {
            "AAA": bars([FLAT, FLAT, (9.8, 10.2, 9.7, 10.1), (10.1, 13.5, 10.0, 13.2)]),
            "CCC": bars([FLAT, FLAT, (9.8, 13.5, 9.7, 13.2)]),
        }
        iwm = bars([(100, 100, 100, 100), (100, 100, 100, 100), (101, 101, 101, 101), (102, 102, 102, 102)])["Close"]
        updated = update_outcomes(log, frames, iwm, today=date(2026, 10, 7)).set_index("Ticker")
        self.assertEqual(updated.at["AAA", "Status"], "Target")
        self.assertEqual(updated.at["AAA", "IWM %"], 2.0)
        self.assertEqual(updated.at["AAA", "Updated"], "2026-10-07")
        self.assertEqual(updated.at["BBB", "Status"], WAITING)  # no prices: left alone
        self.assertEqual(updated.at["CCC", "Status"], "Stop")
        self.assertEqual(updated.at["CCC", "R"], -1.0)

    def test_benchmark_return_covers_the_holding_days(self):
        closes = bars([(0, 0, 0, 100), (0, 0, 0, 104), (0, 0, 0, 110)])["Close"]
        self.assertEqual(benchmark_return(closes, "2026-10-02", "2026-10-05"), 10.0)
        self.assertIsNone(benchmark_return(closes, "2026-10-01", "2026-10-05"))
        self.assertIsNone(benchmark_return(None, "2026-10-02", "2026-10-05"))


class SummaryAndStorageTests(unittest.TestCase):
    def make_log(self):
        log = append_signals(empty_log(), scan_results(["A", "B", "C", "D", "E"]), date(2026, 10, 5))
        log["Status"] = ["Target", "Stop", OPEN, WAITING, NOT_TRIGGERED]
        log["R"] = [3.0, -1.0, 0.5, None, None]
        log["Return %"] = [30.0, -10.0, 5.0, None, None]
        log["IWM %"] = [2.0, -1.0, 1.0, None, None]
        return log

    def test_summary_counts_only_closed_trades_in_the_stats(self):
        summary = summarize_log(self.make_log())
        self.assertEqual(summary["signals"], 5)
        self.assertEqual(summary["trades"], 2)
        self.assertEqual(summary["still_open"], 1)
        self.assertEqual(summary["waiting"], 1)
        self.assertEqual(summary["not_triggered"], 1)
        self.assertEqual(summary["win_rate_pct"], 50.0)
        self.assertEqual(summary["average_r"], 1.0)
        self.assertEqual(summary["average_return_pct"], 10.0)
        self.assertEqual(summary["average_iwm_pct"], 0.5)

    def test_summary_can_be_limited_to_the_top_ranks_and_handles_an_empty_log(self):
        self.assertEqual(summarize_log(self.make_log(), max_rank=1)["trades"], 1)
        empty = summarize_log(empty_log())
        self.assertEqual((empty["signals"], empty["trades"], empty["win_rate_pct"]), (0, 0, None))

    def test_log_round_trips_through_csv_and_missing_files_are_empty(self):
        buffer = io.StringIO()
        self.make_log().to_csv(buffer, index=False)
        buffer.seek(0)
        restored = read_log(buffer)
        self.assertEqual(list(restored.columns), LOG_COLUMNS)
        self.assertEqual(list(restored["Status"]), ["Target", "Stop", OPEN, WAITING, NOT_TRIGGERED])
        self.assertTrue(read_log("/nonexistent/signal_log.csv").empty)


class GradeTests(unittest.TestCase):
    def test_ai_grades_are_logged_and_summarized_per_grade(self):
        results = scan_results(["A", "B", "C", "D"])
        results["AI Grade"] = ["A", "A", "C", ""]
        results["AI Risk"] = ["LOW", "MEDIUM", "HIGH", ""]
        log = append_signals(empty_log(), results, date(2026, 10, 5))
        log["Status"] = ["Target", "Stop", "Stop", WAITING]
        log["R"] = [3.0, -1.0, -1.0, None]

        self.assertEqual(list(log["AI Risk"].iloc[:3]), ["LOW", "MEDIUM", "HIGH"])
        table = summarize_by_grade(log).set_index("AI Grade")
        self.assertEqual(list(table.index), ["A", "C", "Ungraded"])
        self.assertEqual(table.loc["A", "Closed Trades"], 2)
        self.assertEqual(table.loc["A", "Average R"], 1.0)
        self.assertEqual(table.loc["C", "Win Rate %"], 0.0)
        self.assertEqual(table.loc["Ungraded", "Signals"], 1)

    def test_old_logs_without_grade_columns_read_as_ungraded(self):
        buffer = io.StringIO("Scan Date,Ticker,Rank,Status\n2026-10-01,OLD,1,Waiting\n")
        log = read_log(buffer)
        self.assertEqual(list(summarize_by_grade(log)["AI Grade"]), ["Ungraded"])
        self.assertTrue(summarize_by_grade(empty_log()).empty)

    def test_scan_grades_the_top_setups(self):
        results = scan_results(["AAA", "BBB", "CCC"])
        reviews = {"AAA": {"grade": "A", "risk_level": "LOW"}, "BBB": {"grade": "C", "risk_level": "HIGH"}}
        with mock.patch.object(scheduled_scan, "is_ai_configured", return_value=True), \
                mock.patch.object(scheduled_scan, "get_candidate_context", return_value={}), \
                mock.patch.object(scheduled_scan, "review_breakout_candidates",
                                  return_value={"reviews": reviews, "error": None}) as review:
            note, graded = scheduled_scan.grade_top_setups(results, top_n=2)
        self.assertEqual(graded, reviews)
        self.assertEqual(len(review.call_args.args[0]), 2)
        self.assertEqual(list(results["AI Grade"]), ["A", "C", ""])
        self.assertIn("AI graded 2 of the top 2", note)
        markdown, html = scheduled_scan.build_report(results, {"label": "Uptrend"}, "", ai_note=note)
        self.assertIn("A (LOW risk)", markdown)
        self.assertIn("AI C (HIGH risk)", html)

    def test_scan_without_a_key_or_with_an_ai_error_leaves_setups_ungraded(self):
        results = scan_results(["AAA"])
        with mock.patch.object(scheduled_scan, "is_ai_configured", return_value=False):
            self.assertIn("GEMINI_API_KEY", scheduled_scan.grade_top_setups(results)[0])
        with mock.patch.object(scheduled_scan, "is_ai_configured", return_value=True), \
                mock.patch.object(scheduled_scan, "get_candidate_context", return_value={}), \
                mock.patch.object(scheduled_scan, "review_breakout_candidates",
                                  return_value={"reviews": {}, "error": "quota"}):
            self.assertIn("quota", scheduled_scan.grade_top_setups(results)[0])
        self.assertNotIn("AI Grade", results)


class ScheduledScanLogTests(unittest.TestCase):
    def test_record_signals_writes_the_log_and_reports_a_summary(self):
        frames = {"AAA": bars([FLAT, FLAT, (9.8, 10.2, 9.7, 10.1), (10.1, 13.5, 10.0, 13.2)])}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(scheduled_scan, "download_outcome_prices", return_value=(frames, None)):
            path = os.path.join(folder, "signal_log.csv")
            line = scheduled_scan.record_signals(path, scan_results(["AAA"]), {"label": "Uptrend"}, date(2026, 10, 5))
            log = read_log(path)
        self.assertEqual(log["Status"].iloc[0], "Target")
        self.assertEqual(log["IWM Trend"].iloc[0], "Uptrend")
        self.assertIn("1 signals since 2026-10-05, 1 closed trades, win rate 100.0%", line)


if __name__ == "__main__":
    unittest.main()
