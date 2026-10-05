import os
import tempfile
import unittest
from datetime import date, datetime
from unittest import mock

import pandas as pd

import intraday_alerts
from intraday_alerts import (
    EASTERN, average_daily_volume, bars_for_day, check_breakout, format_alerts, in_alert_window,
    session_fraction, watched_signals,
)
from signal_log import OPEN, WAITING, empty_log


def eastern(*args):
    return datetime(*args, tzinfo=EASTERN)


SIGNAL = {"Ticker": "ABC", "Pivot": 10.0, "Stop": 9.0, "Target": 13.0, "AI Grade": "A"}


def five_minute_bars(closes, volume=10_000, day="2026-10-05"):
    index = pd.date_range(f"{day} 09:30", periods=len(closes), freq="5min", tz=EASTERN)
    return pd.DataFrame({"Close": closes, "Volume": volume}, index=index)


class AlertWindowTests(unittest.TestCase):
    def test_alerts_run_on_weekdays_after_the_opening_minutes(self):
        self.assertTrue(in_alert_window(eastern(2026, 10, 5, 10, 0)))
        self.assertFalse(in_alert_window(eastern(2026, 10, 5, 9, 35)))
        self.assertFalse(in_alert_window(eastern(2026, 10, 5, 16, 0)))
        self.assertFalse(in_alert_window(eastern(2026, 10, 4, 12, 0)))

    def test_session_fraction(self):
        self.assertEqual(session_fraction(eastern(2026, 10, 5, 9, 0)), 0.0)
        self.assertAlmostEqual(session_fraction(eastern(2026, 10, 5, 12, 45)), 0.5)
        self.assertEqual(session_fraction(eastern(2026, 10, 5, 17, 0)), 1.0)


class BreakoutCheckTests(unittest.TestCase):
    def test_only_waiting_top_setups_are_watched(self):
        log = empty_log().reindex(range(3))
        log["Ticker"], log["Rank"], log["Status"] = ["A", "B", "C"], [1, 2, 15], [WAITING, OPEN, WAITING]
        self.assertEqual(list(watched_signals(log, top_n=10)["Ticker"]), ["A"])

    def test_breakout_on_volume_alerts(self):
        # 300k traded by a quarter of the session against a 400k average day: 3x the usual pace
        bars = five_minute_bars([9.8, 10.1, 10.3], volume=100_000)
        alert = check_breakout(SIGNAL, bars, average_volume=400_000, fraction=0.25)
        self.assertEqual((alert["Price"], alert["RVOL"], alert["Above Pivot %"]), (10.3, 3.0, 3.0))

    def test_no_alert_below_the_pivot_or_on_light_volume(self):
        self.assertIsNone(check_breakout(SIGNAL, five_minute_bars([9.9]), 10_000, 0.5))
        self.assertIsNone(check_breakout(SIGNAL, five_minute_bars([10.2], volume=10_000), 400_000, 0.5))
        self.assertIsNone(check_breakout(SIGNAL, pd.DataFrame(), 400_000, 0.5))

    def test_extended_breakouts_are_flagged(self):
        alert = check_breakout(SIGNAL, five_minute_bars([10.8], volume=1_000_000), 400_000, 0.5)
        message = format_alerts([alert], eastern(2026, 10, 5, 11, 0))
        self.assertIn("8.0% above pivot", message)
        self.assertIn("don't chase", message)
        self.assertIn("AI A", message)

    def test_day_filter_and_average_volume_exclude_today(self):
        bars = pd.concat([five_minute_bars([9.0], day="2026-10-02"), five_minute_bars([10.0], day="2026-10-05")])
        self.assertEqual(list(bars_for_day(bars, date(2026, 10, 5))["Close"]), [10.0])
        daily = pd.DataFrame({"Volume": [100, 200, 900]}, index=pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"]))
        self.assertEqual(average_daily_volume(daily, date(2026, 10, 5)), 150.0)


class AlertRunTests(unittest.TestCase):
    def test_alerts_are_sent_once_per_ticker_per_day(self):
        log = empty_log().reindex(range(1))
        log["Ticker"], log["Rank"], log["Status"] = ["ABC"], [1], [WAITING]
        log["Pivot"], log["Stop"], log["Target"] = [10.0], [9.0], [13.0]
        alert = {**SIGNAL, "Price": 10.2, "RVOL": 2.0, "Above Pivot %": 2.0}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}), \
                mock.patch.object(intraday_alerts, "download", return_value=pd.DataFrame()), \
                mock.patch.object(intraday_alerts, "check_breakout", return_value=alert), \
                mock.patch.object(intraday_alerts, "send_telegram") as send:
            log_path, sent_path = os.path.join(folder, "log.csv"), os.path.join(folder, "sent.csv")
            log.to_csv(log_path, index=False)
            args = ["--log", log_path, "--sent", sent_path, "--force"]
            intraday_alerts.main(args)
            intraday_alerts.main(args)
            sent = pd.read_csv(sent_path)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(list(sent["Ticker"]), ["ABC"])


if __name__ == "__main__":
    unittest.main()
