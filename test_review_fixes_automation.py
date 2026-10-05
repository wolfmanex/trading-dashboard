import os
import tempfile
import unittest
from datetime import date, datetime
from unittest import mock

import pandas as pd

import daily_check
import intraday_alerts
import scheduled_scan
from journal import LONG, add_trade, empty_journal
from market_calendar import is_trading_day
from signal_log import WAITING, empty_log


EASTERN = intraday_alerts.EASTERN


class MarketCalendarTests(unittest.TestCase):
    def test_holidays_and_weekends_are_closed(self):
        self.assertTrue(is_trading_day(date(2026, 10, 5)))
        self.assertFalse(is_trading_day(date(2026, 10, 4)))       # Sunday
        self.assertFalse(is_trading_day(date(2026, 11, 26)))      # Thanksgiving
        self.assertFalse(is_trading_day(date(2027, 3, 26)))       # Good Friday
        self.assertTrue(is_trading_day(date(2026, 11, 27)))       # early close counts as open

    def test_warns_past_the_list(self):
        with mock.patch("builtins.print") as printed:
            self.assertTrue(is_trading_day(date(2030, 1, 2)))
        self.assertIn("no NYSE holiday list for 2030", printed.call_args.args[0])

    def test_no_alerts_on_a_holiday(self):
        self.assertFalse(intraday_alerts.in_alert_window(datetime(2026, 11, 26, 11, 0, tzinfo=EASTERN)))


class IntradayIsolationTests(unittest.TestCase):
    def write_files(self, folder, ticker="AAA"):
        log = empty_log().reindex(range(1))
        log["Ticker"], log["Rank"], log["Status"] = ["ABC"], [1], [WAITING]
        log["Pivot"], log["Stop"], log["Target"] = [10.0], [9.0], [13.0]
        log_path = os.path.join(folder, "log.csv")
        log.to_csv(log_path, index=False)
        journal_path = os.path.join(folder, "journal.csv")
        add_trade(empty_journal(), ticker, LONG, "2026-10-05", 10.0, 100, 9.0, 12.0).to_csv(journal_path, index=False)
        return ["--log", log_path, "--sent", os.path.join(folder, "sent.csv"), "--journal", journal_path,
                "--position-sent", os.path.join(folder, "pos.csv"), "--force"]

    def test_failed_position_check_still_runs_setups(self):
        alert = {"Ticker": "ABC", "Price": 10.2, "Pivot": 10.0, "Stop": 9.0, "Target": 13.0, "RVOL": 2.0,
                 "Above Pivot %": 2.0, "AI Grade": ""}
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}), \
                mock.patch.object(intraday_alerts, "download", return_value=pd.DataFrame()), \
                mock.patch.object(intraday_alerts, "latest_prices", side_effect=RuntimeError("429 Too Many Requests")), \
                mock.patch.object(intraday_alerts, "check_breakout", return_value=alert), \
                mock.patch.object(intraday_alerts, "send_telegram") as send:
            self.assertEqual(intraday_alerts.main(self.write_files(folder)), 0)
        self.assertEqual(send.call_count, 1)
        self.assertIn("Breakout alert", send.call_args.args[2])

    def test_failed_setup_check_exits_zero(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(intraday_alerts, "download", side_effect=ValueError("yfinance down")), \
                mock.patch.object(intraday_alerts, "latest_prices", return_value={}):
            self.assertEqual(intraday_alerts.main(self.write_files(folder)), 0)

    def test_blank_journal_ticker_is_skipped(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}), \
                mock.patch.object(intraday_alerts, "download", return_value=pd.DataFrame()) as download, \
                mock.patch.object(intraday_alerts, "send_telegram"):
            args = self.write_files(folder)
            journal_path = args[5]
            journal = pd.read_csv(journal_path)
            journal["Ticker"] = journal["Ticker"].astype(object)
            journal.loc[0, "Ticker"] = None
            journal.to_csv(journal_path, index=False)
            intraday_alerts.check_positions(journal_path, os.path.join(folder, "pos.csv"),
                                            datetime(2026, 10, 5, 11, 0, tzinfo=EASTERN), mock.Mock())
        download.assert_not_called()


class DailyCheckDateTests(unittest.TestCase):
    def test_runs_today_uses_the_new_york_date(self):
        runs = [
            {"id": 3, "created_at": "2026-10-06T01:30:00Z"},   # 21:30 New York on the 5th
            {"id": 2, "created_at": "2026-10-05T12:05:00Z"},   # 08:05 New York on the 5th
            {"id": 1, "created_at": "2026-10-05T02:00:00Z"},   # 22:00 New York on the 4th
        ]
        response = mock.Mock()
        response.json.return_value = {"workflow_runs": runs}
        with mock.patch.object(daily_check, "github", return_value=response) as github:
            found = daily_check.runs_today("scheduled-scan.yml", date(2026, 10, 5))
        self.assertEqual([run["id"] for run in found], [3, 2])
        self.assertEqual(github.call_args.kwargs["params"]["created"], ">=2026-10-04")

    def test_session_started(self):
        cases = [([], False), ([{"id": 1, "conclusion": "failure", "status": "completed"}], False),
                 ([{"id": 1, "status": "in_progress"}], True), ([{"id": 1, "conclusion": "success"}], True)]
        for runs, expected in cases:
            with mock.patch.object(daily_check, "runs_today", return_value=runs):
                self.assertEqual(daily_check.session_started(date(2026, 10, 5)), expected)

    def run_main(self, mode, today, **patches):
        with tempfile.TemporaryDirectory() as folder:
            output = os.path.join(folder, "out")
            fake_now = mock.Mock(wraps=datetime)
            fake_now.now.return_value = datetime.combine(today, datetime.min.time(), tzinfo=EASTERN).replace(hour=8)
            with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": output}), \
                    mock.patch.object(daily_check, "datetime", fake_now):
                with mock.patch.multiple(daily_check, **patches):
                    daily_check.main([mode])
            with open(output, encoding="utf-8") as handle:
                return handle.read()

    def test_ran_today_fails_open(self):
        out = self.run_main("ran-today", date(2026, 10, 5), already_ran=mock.Mock(side_effect=RuntimeError("502")))
        self.assertEqual(out, "ran=false\n")

    def test_ran_today_reports_closed_days_as_run(self):
        already = mock.Mock()
        self.assertEqual(self.run_main("ran-today", date(2026, 11, 26), already_ran=already), "ran=true\n")
        already.assert_not_called()

    def test_session_started_output(self):
        self.assertEqual(self.run_main("session-started", date(2026, 10, 5),
                                       session_started=mock.Mock(return_value=True)), "started=true\n")
        self.assertEqual(self.run_main("session-started", date(2026, 10, 5),
                                       session_started=mock.Mock(side_effect=RuntimeError("500"))), "started=false\n")

    def test_no_recap_on_a_holiday(self):
        fake_now = mock.Mock(wraps=datetime)
        fake_now.now.return_value = datetime(2026, 12, 25, 16, 10, tzinfo=EASTERN)
        with mock.patch.object(daily_check, "datetime", fake_now), \
                mock.patch.object(daily_check, "recap") as recap:
            self.assertEqual(daily_check.main(["recap"]), 0)
        recap.assert_not_called()


class TelegramChunkTests(unittest.TestCase):
    def test_splits_on_lines_without_breaking_tags(self):
        lines = [f"<b>T{i}</b> " + "x" * 90 for i in range(100)]
        chunks = scheduled_scan.telegram_chunks("\n".join(lines))
        self.assertGreater(len(chunks), 1)
        self.assertEqual("\n".join(chunks), "\n".join(lines))
        for chunk in chunks:
            self.assertLess(len(chunk), 4001)
            self.assertEqual(chunk.count("<b>"), chunk.count("</b>"))

    def test_overlong_line_loses_its_tags(self):
        chunks = scheduled_scan.telegram_chunks("head\n<b>" + "y" * 5000 + "</b>")
        self.assertEqual(chunks[0], "head")
        self.assertEqual(chunks[1], "y" * 4000)

    def test_send_telegram_posts_each_chunk(self):
        with mock.patch.object(scheduled_scan.requests, "post") as post:
            scheduled_scan.send_telegram("t", "c", "\n".join(["z" * 3000] * 3))
        self.assertEqual(post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
