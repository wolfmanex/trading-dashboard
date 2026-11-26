"""The day-trade feedback loop: past closed alerts nudge new setups' scores, gently with few trades."""
import unittest

import pandas as pd

from daytrade_alerts import BREAKING_OUT, pick_alerts
from daytrade_feedback import MAX_ADJUSTMENT, MIN_TRADES, apply_feedback, build_feedback, feedback_line, feedback_summary


def history(early_r: float, late_r: float, per_bucket: int) -> pd.DataFrame:
    """Closed alerts: `per_bucket` at 10:15 that made early_r and as many at 13:30 that made late_r."""
    rows = []
    for time, r in (("10:15", early_r), ("13:30", late_r)):
        for _ in range(per_bucket):
            rows.append({
                "Date": "2026-11-02", "Time": time, "Ticker": "AAA", "Price": 100.5, "Trigger": 100.0, "Stop": 99.0,
                "Target": 103.0, "RVOL": 2.5, "Day Score": 75, "Outcome": "Target" if r > 0 else "Stop", "R": r,
                "Trigger Level": "OR high", "SPY %": 0.1,
            })
    return pd.DataFrame(rows)


def scan_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"Ticker": "AAA", "Price": 100.5, "Trigger": 100.0, "RVOL": 2.5, "Day Score": 68.0, "Status": BREAKING_OUT,
         "Trigger Level": "OR high"},
        {"Ticker": "BBB", "Price": 50.2, "Trigger": 50.0, "RVOL": 2.5, "Day Score": 62.0, "Status": BREAKING_OUT,
         "Trigger Level": "OR high"},
    ])


class FeedbackTests(unittest.TestCase):
    def test_no_adjustment_below_the_minimum_sample(self):
        feedback = build_feedback(history(-1.0, 2.0, per_bucket=MIN_TRADES // 2 - 1))
        self.assertFalse(feedback["active"])
        adjusted = apply_feedback(scan_rows(), feedback, now_minute=10 * 60 + 15)
        self.assertEqual(list(adjusted["History Adj"]), [0.0, 0.0])
        self.assertEqual(list(adjusted["Adj Score"]), [68.0, 62.0])
        self.assertIn(f"starts adjusting scores at {MIN_TRADES}", feedback_line(feedback))

    def test_weak_time_of_day_lowers_scores_and_strong_raises_them(self):
        feedback = build_feedback(history(-1.0, 1.0, per_bucket=30))
        self.assertTrue(feedback["active"])
        table = feedback["table"].set_index(["Dimension", "Bucket"])
        self.assertLess(table.loc[("Time of day", "10:00-10:59"), "Points"], 0)
        self.assertGreater(table.loc[("Time of day", "13:00 on"), "Points"], 0)
        # Shrinkage: 30 trades at -1R against a 0R average count as (30 * -1) / (30 + 20).
        self.assertAlmostEqual(table.loc[("Time of day", "10:00-10:59"), "Edge R"], -0.6, places=3)

        morning = apply_feedback(scan_rows(), feedback, now_minute=10 * 60 + 15)
        afternoon = apply_feedback(scan_rows(), feedback, now_minute=13 * 60 + 30)
        self.assertEqual(morning.loc[0, "History Adj"], -MAX_ADJUSTMENT)   # capped
        self.assertEqual(afternoon.loc[0, "History Adj"], MAX_ADJUSTMENT)
        self.assertIn("time of day 10:00-10:59", morning.loc[0, "History Note"])
        # The adjusted score is what has to clear the alert threshold.
        self.assertEqual(pick_alerts(morning, already=set()), [])
        self.assertEqual([pick["Ticker"] for pick in pick_alerts(afternoon, already=set())], ["AAA", "BBB"])

    def test_small_buckets_barely_move(self):
        closed = pd.concat([history(0.0, 0.0, per_bucket=30), history(-1.0, 0.0, per_bucket=1).head(1)])
        closed.iloc[-1, closed.columns.get_loc("RVOL")] = 6.0
        table = build_feedback(closed)["table"].set_index(["Dimension", "Bucket"])
        self.assertGreater(table.loc[("Relative volume", "3x and up"), "Edge R"], -0.05)

    def test_summary_for_the_ai_prompt_leads_with_the_strongest_buckets(self):
        summary = feedback_summary(build_feedback(history(-1.0, 1.0, per_bucket=30)))
        lines = summary.splitlines()
        self.assertEqual(lines[0], "60 closed alerts, average +0.00R per trade.")
        self.assertIn("Time of day", lines[1])
        self.assertEqual(feedback_summary(build_feedback(pd.DataFrame())), "")

    def test_open_and_old_close_rows_are_not_counted(self):
        closed = history(1.0, 1.0, per_bucket=1)
        closed.loc[0, "Outcome"] = "Open"
        closed.loc[1, "Outcome"] = "Close"
        self.assertEqual(build_feedback(closed)["trades"], 0)
        self.assertEqual(build_feedback(pd.DataFrame())["trades"], 0)


if __name__ == "__main__":
    unittest.main()
