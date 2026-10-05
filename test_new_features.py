import os
import tempfile
import unittest
from datetime import date, datetime
from unittest import mock

import numpy as np
import pandas as pd

import daily_check
import intraday_alerts
import llm_engine
import scheduled_scan
from breakout_engine import MIN_ATR_PCT, MIN_STOP_PCT, analyze_breakout_setup, qualifies
from intraday_alerts import EASTERN
from journal import (
    CLOSED, LONG, OPEN, SHORT, JournalStore, add_trade, close_trade, empty_journal, mark_positions,
    open_positions, position_alerts, read_journal_file, summarize_journal, update_stop,
)
from market_conditions import (
    breadth_pct, classify_market, classify_vix, events_soon, merge_events, parse_economic_events, summary_line,
)
from signal_log import empty_log
from test_breakout import breakout_setup_frame, make_frame, make_setup


class StopFloorTests(unittest.TestCase):
    def test_stop_is_never_closer_than_the_percent_floor(self):
        setup = analyze_breakout_setup(breakout_setup_frame())
        self.assertLessEqual(setup["stop"], setup["pivot"] * (1 - MIN_STOP_PCT) + 1e-9)

    def test_pinned_stock_with_a_tiny_atr_is_left_out(self):
        # A takeover target: jumps once, then barely moves around the deal price.
        closes = np.concatenate([np.linspace(5, 5.3, 200), [7.3], 7.35 + 0.01 * np.sin(np.arange(60))])
        setup = analyze_breakout_setup(make_frame(closes, spread=0.002))
        self.assertIsNotNone(setup)
        self.assertLess(setup["atr_pct"], MIN_ATR_PCT)
        self.assertFalse(qualifies(setup, min_reward_risk=0.0, min_dollar_volume=0.0))

    def test_normal_setup_still_qualifies(self):
        self.assertTrue(qualifies(make_setup(atr_pct=0.04), 3.0, 5_000_000))
        self.assertFalse(qualifies(make_setup(atr_pct=0.005), 3.0, 5_000_000))


class MarketConditionsTests(unittest.TestCase):
    def test_vote(self):
        up, down, mixed = {"label": "Uptrend"}, {"label": "Downtrend"}, {"label": "Mixed"}
        self.assertEqual(classify_market(up, up, 70, 14)["label"], "Risk-on")
        self.assertEqual(classify_market(down, down, 30, 28)["label"], "Risk-off")
        self.assertEqual(classify_market(up, mixed, 50, 20)["label"], "Neutral")
        self.assertEqual(classify_market(mixed, mixed, None, None)["label"], "Neutral")
        self.assertEqual(classify_vix(35), "Stressed")

    def test_breadth(self):
        rising = pd.DataFrame({"Close": np.linspace(10, 20, 60)})
        falling = pd.DataFrame({"Close": np.linspace(20, 10, 60)})
        self.assertEqual(breadth_pct({"A": rising, "B": falling, "C": pd.DataFrame()}), 50.0)
        self.assertIsNone(breadth_pct({}))

    def test_economic_calendar_keeps_major_us_releases(self):
        payload = {"data": {"rows": [
            {"country": "United States", "eventName": "CPI (YoY) (Sep)", "gmt": "08:30"},
            {"country": "United States", "eventName": "Crude Oil Inventories"},
            {"country": "Euro Zone", "eventName": "CPI (YoY)"},
            {"country": "United States", "eventName": "Nonfarm Payrolls (Sep)"},
        ]}}
        events = parse_economic_events(payload, "2026-10-14")
        self.assertEqual([event["event"] for event in events], ["CPI", "Jobs report"])
        self.assertEqual(parse_economic_events({"data": None}, "2026-10-14"), [])

    def test_merge_adds_fomc_and_drops_duplicates(self):
        today = date(2026, 10, 20)
        events = [{"date": "2026-10-28", "event": "FOMC rate decision", "detail": "x", "time": ""},
                  {"date": "2026-10-21", "event": "CPI", "detail": "y", "time": ""}]
        merged = merge_events(events, today)
        self.assertEqual([(e["date"], e["event"]) for e in merged], [("2026-10-21", "CPI"), ("2026-10-28", "FOMC rate decision")])
        self.assertEqual([e["event"] for e in events_soon(merged, today)], ["CPI"])
        conditions = {"label": "Neutral", "reasons": ["IWM mixed"], "events": merged}
        with mock.patch("market_conditions.date") as fake_date:
            fake_date.today.return_value = today
            self.assertIn("Coming up: CPI 2026-10-21", summary_line(conditions))


class JournalTests(unittest.TestCase):
    def test_add_mark_and_close(self):
        journal = add_trade(empty_journal(), "abc", LONG, "2026-10-05", 10.0, 100, 9.0, 13.0, "Breakout scanner")
        trade_id = journal.iloc[0]["ID"]
        marked = mark_positions(open_positions(journal), {"ABC": 11.0})
        self.assertEqual(marked.iloc[0]["P&L"], 100.0)
        self.assertEqual(marked.iloc[0]["R"], 1.0)
        journal = update_stop(journal, trade_id, 10.0)   # breakeven stop keeps R measured from the initial risk
        journal = close_trade(journal, trade_id, 12.0, "2026-10-08", "Manual exit")
        row = journal.iloc[0]
        self.assertEqual((row["Status"], row["P&L"], row["R"]), (CLOSED, 200.0, 2.0))
        summary = summarize_journal(journal)
        self.assertEqual((summary["closed"], summary["win_rate_pct"], summary["total_pnl"]), (1, 100.0, 200.0))

    def test_short_trades_and_validation(self):
        journal = add_trade(empty_journal(), "XYZ", SHORT, "2026-10-05", 20.0, 50, 21.0, 18.0)
        self.assertEqual(mark_positions(journal, {"XYZ": 19.0}).iloc[0]["R"], 1.0)
        with self.assertRaises(ValueError):
            add_trade(empty_journal(), "XYZ", LONG, "2026-10-05", 20.0, 50, 21.0)
        with self.assertRaises(ValueError):
            add_trade(empty_journal(), "XYZ", SHORT, "2026-10-05", 20.0, 50, 21.0, 22.0)
        with self.assertRaises(ValueError):
            close_trade(journal, "missing", 19.0, "2026-10-06")

    def test_stop_and_target_alerts(self):
        journal = add_trade(empty_journal(), "AAA", LONG, "2026-10-05", 10.0, 100, 9.0, 12.0)
        journal = add_trade(journal, "BBB", SHORT, "2026-10-05", 20.0, 10, 21.0, 18.0)
        alerts = position_alerts(open_positions(journal), {"AAA": 8.9, "BBB": 17.5})
        self.assertEqual([(a["Ticker"], a["Kind"]) for a in alerts], [("AAA", "stop"), ("BBB", "target")])
        self.assertEqual(position_alerts(open_positions(journal), {"AAA": 10.5}), [])

    def test_local_store_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            store = JournalStore(token="", path=os.path.join(folder, "journal.csv"))
            self.assertFalse(store.remote)
            self.assertTrue(store.load().empty)
            store.save(add_trade(empty_journal(), "ABC", LONG, "2026-10-05", 10.0, 1, 9.0), "add")
            loaded = store.load()
        self.assertEqual(list(loaded["Ticker"]), ["ABC"])
        self.assertEqual(loaded.iloc[0]["Status"], OPEN)

    def test_github_store_sends_the_sha(self):
        store = JournalStore(token="t")
        get = mock.Mock(status_code=200, json=lambda: {"sha": "abc", "content": ""})
        put = mock.Mock(status_code=200, json=lambda: {"content": {"sha": "def"}})
        with mock.patch("journal.requests.get", return_value=get), mock.patch("journal.requests.put", return_value=put) as sent:
            store.load()
            store.save(empty_journal(), "msg")
        self.assertEqual(sent.call_args.kwargs["json"]["sha"], "abc")
        self.assertEqual(sent.call_args.kwargs["json"]["branch"], "signal-log")
        self.assertEqual(store.sha, "def")


class PositionAlertRunTests(unittest.TestCase):
    def test_position_alerts_are_sent_once_per_day(self):
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "t", "TELEGRAM_CHAT_ID": "c"}), \
                mock.patch.object(intraday_alerts, "download", return_value=pd.DataFrame()), \
                mock.patch.object(intraday_alerts, "latest_prices", return_value={"AAA": 8.5}), \
                mock.patch.object(intraday_alerts, "send_telegram") as send:
            journal_path = os.path.join(folder, "journal.csv")
            add_trade(empty_journal(), "AAA", LONG, "2026-10-05", 10.0, 100, 9.0, 12.0).to_csv(journal_path, index=False)
            args = ["--log", os.path.join(folder, "missing.csv"), "--sent", os.path.join(folder, "sent.csv"),
                    "--journal", journal_path, "--position-sent", os.path.join(folder, "pos.csv"), "--force"]
            intraday_alerts.main(args)
            intraday_alerts.main(args)
            sent = pd.read_csv(os.path.join(folder, "pos.csv"))
        self.assertEqual(send.call_count, 1)
        self.assertIn("hit its stop 9.00", send.call_args.args[2])
        self.assertEqual(list(sent["Kind"]), ["stop"])


class DailyCheckTests(unittest.TestCase):
    def test_outcome_lines(self):
        log = empty_log().reindex(range(3))
        log["Ticker"] = ["AAA", "BBB", "CCC"]
        log["Status"] = ["Open", "Stop", "Waiting"]
        log["Entry Date"] = ["2026-10-05", "2026-10-01", None]
        log["Entry"] = [10.0, 5.0, None]
        log["Exit Date"] = ["2026-10-05", "2026-10-05", None]
        log["Exit"] = [10.5, 4.6, None]
        log["R"] = [0.5, -1.0, None]
        text = "\n".join(daily_check.outcome_lines(log, "2026-10-05"))
        self.assertIn("Triggered: <b>AAA</b> at 10.00", text)
        self.assertIn("Stop: <b>BBB</b> at 4.60, -1.00R", text)
        self.assertIn("1 setups still waiting", text)

    def test_run_line(self):
        self.assertEqual(daily_check.run_line([]), "did NOT run today")
        run = {"run_started_at": "2026-10-05T12:47:03Z", "event": "schedule", "conclusion": "success"}
        self.assertEqual(daily_check.run_line([run]), "ran at 12:47 UTC (GitHub's own schedule), success")

    def test_already_ran_counts_only_earlier_runs(self):
        runs = [{"id": 30, "status": "queued"}, {"id": 20, "conclusion": "success"}, {"id": 10, "conclusion": "failure"}]
        with mock.patch.object(daily_check, "runs_today", return_value=runs):
            self.assertTrue(daily_check.already_ran("scheduled-scan.yml", date(2026, 10, 6), "25"))
            self.assertFalse(daily_check.already_ran("scheduled-scan.yml", date(2026, 10, 6), "15"))
        with mock.patch.object(daily_check, "runs_today", return_value=[{"id": 9, "status": "in_progress"}]):
            self.assertTrue(daily_check.already_ran("scheduled-scan.yml", date(2026, 10, 6), "12"))

    def test_recap_reports_the_check_count(self):
        notify = mock.Mock()
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(daily_check, "runs_today", return_value=[]):
            daily_check.recap(os.path.join(folder, "log.csv"), None, notify, checks=10)
        text = notify.call_args.args[0]
        self.assertIn("Morning scan did NOT run today", text)
        self.assertIn("Intraday checks ran 10 times", text)
        self.assertIn("may have gone unalerted", text)


class LivePickTests(unittest.TestCase):
    def test_states_against_the_levels(self):
        from signal_log import live_pick_states
        log = empty_log().reindex(range(5))
        log["Ticker"] = ["A", "B", "C", "D", "E"]
        log["Status"] = ["Waiting", "Waiting", "Open", "Waiting", "Stop"]
        log["Pivot"], log["Stop"], log["Target"] = 10.0, 9.0, 14.0
        prices = {"A": 9.5, "B": 10.3, "C": 11.0, "D": 8.8, "E": 12.0}
        picks = live_pick_states(log, prices)
        self.assertEqual(list(picks["Ticker"]), ["A", "B", "C", "D"])
        self.assertEqual(list(picks["State"]), ["Below pivot", "Breaking out", "Extended, don't chase", "At or below stop"])
        self.assertEqual(picks.iloc[2]["vs Pivot %"], 10.0)


class BriefAndFollowUpTests(unittest.TestCase):
    def test_brief_lists_top_graded_setups(self):
        results = pd.DataFrame({"Ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"]})
        reviews = {t: {"grade": "B", "risk_level": "MEDIUM", "catalyst": "Contract win", "thesis": "Tight base.",
                       "red_flags": ["ATM offering"]} for t in ["BBB", "CCC", "DDD", "EEE"]}
        brief = scheduled_scan.build_brief(results, reviews)
        self.assertIn("<b>BBB</b> grade B, medium risk. Catalyst: Contract win", brief)
        self.assertIn("Red flags: ATM offering", brief)
        self.assertNotIn("EEE", brief)
        self.assertEqual(scheduled_scan.build_brief(results, {}), "")

    def test_follow_up_includes_review_and_history(self):
        with mock.patch.object(llm_engine, "generate_text", return_value="Because volume is light.") as generate:
            answer = llm_engine.ask_follow_up("DATA", {"verdict": "WAIT"}, [{"question": "Q1", "answer": "A1"}], "Why wait?")
        prompt = generate.call_args.args[0]
        self.assertEqual(answer, {"answer": "Because volume is light.", "error": None})
        for part in ("DATA", '"verdict": "WAIT"', "Trader: Q1", "Trader: Why wait?"):
            self.assertIn(part, prompt)
        self.assertIsNotNone(llm_engine.ask_follow_up("DATA", {}, [], " ")["error"])


if __name__ == "__main__":
    unittest.main()
