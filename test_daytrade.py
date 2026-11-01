import unittest
from datetime import datetime
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import daytrade_alerts
import largecap_scanner
import smallcap_screener
from daytrade_engine import (
    BREAKING_OUT, MAX_RISK_PCT, MIN_RISK_PCT, NEAR_TRIGGER, PHASE_PREMARKET, PHASE_SESSION, analyze_daytrade_setup,
    day_change_pct, evaluate_daytrade, qualifies_daytrade, rank_daytrade_candidates, score_daytrade, split_sessions,
)

EASTERN = ZoneInfo("America/New_York")


def day_bars(day: str, start: str, end: str, price: float, volume: float, high_pad: float = 0.2, low_pad: float = 0.2):
    index = pd.date_range(f"{day} {start}", f"{day} {end}", freq="5min", tz=EASTERN, inclusive="left")
    closes = np.full(len(index), price, dtype=float)
    return pd.DataFrame({
        "Open": closes, "High": closes + high_pad, "Low": closes - low_pad, "Close": closes,
        "Volume": np.full(len(index), volume, dtype=float),
    }, index=index)


def history(today_regular=None, today_premarket=None, prior_price=100.0, prior_volume=10_000.0):
    """Two quiet prior days (100 +/- 0.2, so the prior high is 100.2), then today's bars."""
    frames = []
    for day in ("2026-10-28", "2026-10-29"):
        frames.append(day_bars(day, "04:00", "09:30", prior_price, prior_volume / 10))
        frames.append(day_bars(day, "09:30", "16:00", prior_price, prior_volume))
    if today_premarket is not None:
        frames.append(today_premarket)
    if today_regular is not None:
        frames.append(today_regular)
    return pd.concat(frames)


def breakout_day():
    """Opening range 101-102, then a push to 102.3 on heavy volume, above VWAP."""
    opening = day_bars("2026-10-30", "09:30", "09:45", 101.5, 40_000, high_pad=0.5, low_pad=0.5)
    push = day_bars("2026-10-30", "09:45", "10:15", 102.1, 30_000, high_pad=0.2, low_pad=0.1)
    push.iloc[-1, push.columns.get_loc("Close")] = 102.3
    push.iloc[-1, push.columns.get_loc("High")] = 102.4
    premarket = day_bars("2026-10-30", "08:00", "09:30", 101.0, 3_000, high_pad=0.3, low_pad=0.3)
    return history(pd.concat([opening, push]), premarket)


def make_setup(**overrides):
    setup = {
        "phase": PHASE_SESSION, "price": 102.3, "trigger": 102.0, "trigger_level": "OR high",
        "distance_to_trigger": -0.003, "status": BREAKING_OUT, "stop": 101.0, "target": 104.0, "risk_pct": 0.01,
        "rvol": 3.0, "vwap": 101.6, "above_vwap": True, "or_high": 102.0, "or_low": 101.0, "pm_high": 101.3,
        "prior_high": 100.2, "prior_low": 99.8, "prior_close": 100.0, "above_prior_high": True, "change_pct": 2.3,
        "relative_strength": 1.8, "avg_dollar_volume": 500_000_000,
    }
    setup.update(overrides)
    return setup


class DayTradeSetupTests(unittest.TestCase):
    def test_sessions_split_premarket_and_regular_in_new_york_time(self):
        bars = breakout_day().tz_convert("UTC")
        sessions = split_sessions(bars)
        today = sessions[list(sessions)[-1]]
        self.assertEqual(len(today["premarket"]), 18)
        self.assertEqual(today["regular"].index[0].strftime("%H:%M"), "09:30")

    def test_opening_range_breakout_levels(self):
        setup = analyze_daytrade_setup(breakout_day(), benchmark_change=0.5)

        self.assertEqual(setup["phase"], PHASE_SESSION)
        self.assertEqual(setup["trigger_level"], "OR high")
        self.assertAlmostEqual(setup["trigger"], 102.0)
        self.assertAlmostEqual(setup["or_low"], 101.0)
        self.assertEqual(setup["status"], BREAKING_OUT)
        self.assertTrue(setup["above_vwap"])
        self.assertTrue(setup["above_prior_high"])
        self.assertAlmostEqual(setup["change_pct"], 2.3)
        self.assertAlmostEqual(setup["relative_strength"], 1.8)
        # VWAP (~101.9) is the tighter stop candidate but sits within 0.5% of the trigger, so the
        # opening-range low is used instead; risk stays inside the allowed band.
        self.assertGreaterEqual(setup["risk_pct"], MIN_RISK_PCT - 1e-9)
        self.assertLessEqual(setup["risk_pct"], MAX_RISK_PCT + 1e-9)
        self.assertAlmostEqual(setup["target"], setup["trigger"] + 2 * (setup["trigger"] - setup["stop"]))

    def test_relative_volume_compares_the_same_stretch_of_prior_sessions(self):
        setup = analyze_daytrade_setup(breakout_day())
        # Today: 3 bars x 40k + 6 bars x 30k = 300k by 10:10; prior days: 9 bars x 10k = 90k.
        self.assertAlmostEqual(setup["rvol"], 300_000 / 90_000, places=3)
        self.assertAlmostEqual(setup["avg_dollar_volume"], 100.0 * 10_000 * 78)

    def test_premarket_uses_premarket_high_as_trigger(self):
        premarket = day_bars("2026-10-30", "07:00", "09:00", 101.0, 5_000, high_pad=0.3, low_pad=0.3)
        setup = analyze_daytrade_setup(history(today_premarket=premarket))

        self.assertEqual(setup["phase"], PHASE_PREMARKET)
        self.assertEqual(setup["trigger_level"], "PM high")
        self.assertAlmostEqual(setup["trigger"], 101.3)
        self.assertEqual(setup["status"], NEAR_TRIGGER)
        self.assertIsNone(setup["vwap"])
        self.assertTrue(setup["above_vwap"])

    def test_stop_is_capped_at_max_risk(self):
        opening = day_bars("2026-10-30", "09:30", "09:45", 100.0, 40_000, high_pad=5.0, low_pad=5.0)
        later = day_bars("2026-10-30", "09:45", "10:00", 104.8, 40_000, high_pad=0.1, low_pad=0.1)
        setup = analyze_daytrade_setup(history(pd.concat([opening, later])))
        self.assertAlmostEqual(setup["stop"], setup["trigger"] * (1 - MAX_RISK_PCT))

    def test_needs_a_prior_session(self):
        today = day_bars("2026-10-30", "09:30", "10:30", 100.0, 10_000)
        self.assertIsNone(analyze_daytrade_setup(today))
        self.assertIsNone(analyze_daytrade_setup(pd.DataFrame()))
        self.assertIsNone(analyze_daytrade_setup(None))

    def test_day_change_for_benchmark(self):
        self.assertAlmostEqual(day_change_pct(breakout_day()), 2.3)
        self.assertIsNone(day_change_pct(pd.DataFrame()))


class DayTradeRankingTests(unittest.TestCase):
    def test_score_rewards_volume_and_closeness(self):
        strong = score_daytrade(make_setup())
        self.assertGreater(strong, score_daytrade(make_setup(rvol=1.2)))
        self.assertGreater(strong, score_daytrade(make_setup(distance_to_trigger=0.012, status=NEAR_TRIGGER)))
        self.assertLessEqual(score_daytrade(make_setup(), rs_percentile=1.0), 100)

    def test_qualification_filters(self):
        self.assertTrue(qualifies_daytrade(make_setup(), 1.5, 100e6))
        self.assertFalse(qualifies_daytrade(make_setup(rvol=1.2), 1.5, 100e6))
        self.assertFalse(qualifies_daytrade(make_setup(above_vwap=False), 1.5, 100e6))
        self.assertFalse(qualifies_daytrade(make_setup(distance_to_trigger=-0.02), 1.5, 100e6))   # extended
        self.assertFalse(qualifies_daytrade(make_setup(distance_to_trigger=0.03), 1.5, 100e6))    # too far below
        self.assertFalse(qualifies_daytrade(make_setup(avg_dollar_volume=50e6), 1.5, 100e6))
        self.assertFalse(qualifies_daytrade(None, 1.5, 100e6))

    def test_rank_orders_by_score_and_drops_unqualified(self):
        setups = {
            "AAA": make_setup(),
            "BBB": make_setup(rvol=1.8, relative_strength=-1.0),
            "CCC": make_setup(rvol=1.0),
            "DDD": None,
        }
        ranked = rank_daytrade_candidates(setups, metadata={"AAA": {"name": "Alpha", "market_cap": 50e9}})
        self.assertEqual(list(ranked["Ticker"]), ["AAA", "BBB"])
        self.assertEqual(ranked.loc[0, "Name"], "Alpha")
        self.assertEqual(ranked.loc[0, "Status"], BREAKING_OUT)
        self.assertTrue(rank_daytrade_candidates({"CCC": make_setup(rvol=1.0)}).empty)


class DayTradeOutcomeTests(unittest.TestCase):
    def bars(self, rows):
        index = pd.date_range("2026-10-30 10:00", periods=len(rows), freq="5min", tz=EASTERN)
        return pd.DataFrame(rows, columns=["High", "Low", "Close"], index=index)

    def test_target_first(self):
        result = evaluate_daytrade(100, 99, 102, self.bars([(100.5, 99.5, 100.2), (102.1, 100.1, 102.0)]))
        self.assertEqual(result, {"Outcome": "Target", "Exit": 102.0, "R": 2.0})

    def test_stop_wins_when_one_bar_spans_both(self):
        result = evaluate_daytrade(100, 99, 102, self.bars([(102.5, 98.5, 100.0)]))
        self.assertEqual(result["Outcome"], "Stop")
        self.assertEqual(result["R"], -1.0)

    def test_closes_at_the_bell_otherwise(self):
        result = evaluate_daytrade(100, 99, 102, self.bars([(100.6, 99.6, 100.5)]))
        self.assertEqual(result, {"Outcome": "Close", "Exit": 100.5, "R": 0.5})


class LargeCapUniverseTests(unittest.TestCase):
    def nasdaq_payload(self):
        return {"data": {"rows": [
            {"symbol": "MEGA", "name": "Mega", "marketCap": "2,000,000,000,000", "lastsale": "$400", "volume": "20,000,000"},
            {"symbol": "BIG", "name": "Big", "marketCap": "50,000,000,000", "lastsale": "$100", "volume": "5,000,000"},
            {"symbol": "SMALL", "name": "Small", "marketCap": "1,000,000,000", "lastsale": "$20", "volume": "9,000,000"},
            {"symbol": "CHEAP", "name": "Cheap", "marketCap": "20,000,000,000", "lastsale": "$5", "volume": "90,000,000"},
            {"symbol": "THIN", "name": "Thin", "marketCap": "20,000,000,000", "lastsale": "$50", "volume": "500,000"},
        ]}}

    def nasdaq_response(self):
        response = mock.Mock()
        response.json.return_value = self.nasdaq_payload()
        response.raise_for_status.return_value = None
        return response

    def setUp(self):
        largecap_scanner.get_largecap_universe.clear()

    def test_nasdaq_fallback_keeps_liquid_large_caps(self):
        with mock.patch("yfinance.screen", side_effect=Exception("401 Unauthorized")), \
                mock.patch("requests.get", return_value=self.nasdaq_response()):
            source, universe = largecap_scanner.get_largecap_universe()
        self.assertEqual(source, "Nasdaq screener")
        self.assertEqual(list(universe), ["MEGA", "BIG"])

    def test_small_cap_parsing_defaults_are_unchanged(self):
        self.assertEqual(list(smallcap_screener.parse_nasdaq_rows(self.nasdaq_payload())), ["SMALL"])

    def test_predefined_screen_applies_large_cap_bounds(self):
        quotes = {"quotes": [
            {"symbol": "BIG", "marketCap": 50e9, "regularMarketPrice": 100.0},
            {"symbol": "SMALL", "marketCap": 1e9, "regularMarketPrice": 20.0},
        ]}
        with mock.patch("yfinance.screen", side_effect=[Exception("401"), quotes]) as screen, \
                mock.patch("requests.get", side_effect=Exception("403")):
            source, universe = largecap_scanner.get_largecap_universe()
        self.assertEqual(source, "Yahoo most-active screen")
        self.assertEqual(list(universe), ["BIG"])
        self.assertEqual(screen.call_args.args, ("most_actives",))

    def test_scan_uses_fallback_list_and_spy(self):
        frames = {ticker: breakout_day() for ticker in ["AAPL", "SPY"]}
        batch = pd.concat(frames, axis=1)
        with mock.patch("yfinance.screen", side_effect=Exception("401")), \
                mock.patch("requests.get", side_effect=Exception("403")), \
                mock.patch("yfinance.download", return_value=batch) as download:
            result = largecap_scanner.scan_daytrade_setups(min_dollar_volume=1)
        self.assertEqual(result.attrs["universe_source"], "fallback list")
        self.assertEqual(result.attrs["analyzed"], 1)   # only AAPL of the fallback list has bars
        self.assertEqual(result.attrs["benchmark_change"], 2.3)
        self.assertEqual(list(result["Ticker"]), ["AAPL"])
        self.assertAlmostEqual(result.loc[0, "RS vs SPY"], 0.0)
        self.assertTrue(download.call_args.kwargs["prepost"])
        self.assertIn("SPY", download.call_args.kwargs["tickers"])

    def test_scan_raises_without_data(self):
        with mock.patch("yfinance.screen", side_effect=Exception("401")), \
                mock.patch("requests.get", side_effect=Exception("403")), \
                mock.patch("yfinance.download", return_value=pd.DataFrame()):
            with self.assertRaisesRegex(RuntimeError, "No intraday data"):
                largecap_scanner.scan_daytrade_setups()


class DayTradeAlertTests(unittest.TestCase):
    def results(self):
        ranked = rank_daytrade_candidates({
            "AAA": make_setup(),
            "BBB": make_setup(distance_to_trigger=0.005, status=NEAR_TRIGGER),
            "CCC": make_setup(rvol=1.6, change_pct=0.1, above_prior_high=False, relative_strength=-2),
        })
        ranked.attrs.update({"analyzed": 3, "benchmark_change": 0.4})
        return ranked

    def test_alert_window(self):
        self.assertTrue(daytrade_alerts.in_alert_window(datetime(2026, 10, 30, 9, 45, tzinfo=EASTERN)))
        self.assertFalse(daytrade_alerts.in_alert_window(datetime(2026, 10, 30, 15, 30, tzinfo=EASTERN)))
        self.assertFalse(daytrade_alerts.in_alert_window(datetime(2026, 10, 31, 11, 0, tzinfo=EASTERN)))  # Saturday

    def test_only_fresh_high_scoring_breakouts_are_picked(self):
        results = self.results()
        picked = [pick["Ticker"] for pick in daytrade_alerts.pick_alerts(results, already=set())]
        self.assertEqual(picked, ["AAA"])
        self.assertEqual(daytrade_alerts.pick_alerts(results, already={"AAA"}), [])

    def test_alerts_are_sent_once_and_saved(self):
        now = datetime(2026, 10, 30, 10, 15, tzinfo=EASTERN)
        sent = []
        import tempfile, os
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "daytrade_alerts.csv")
            for _ in range(2):
                daytrade_alerts.check_daytrades(path, now, lambda message: sent.append(message) or True, scan=self.results)
            saved = daytrade_alerts.read_alerts(path)
        self.assertEqual(len(sent), 1)
        self.assertIn("<b>AAA</b>", sent[0])
        self.assertIn("SPY +0.40%", sent[0])
        self.assertEqual(list(saved["Ticker"]), ["AAA"])
        self.assertEqual(saved.loc[0, "Time"], "10:15")

    def test_recap_scores_the_days_alerts(self):
        alerts = pd.DataFrame([{
            "Date": "2026-10-30", "Time": "10:15", "Ticker": "AAA", "Price": 102.3, "Trigger": 102.0,
            "Stop": 101.0, "Target": 104.0, "RVOL": 3.0, "Day Score": 80,
        }]).reindex(columns=daytrade_alerts.DAYTRADE_COLUMNS)
        before = day_bars("2026-10-30", "09:30", "10:15", 101.5, 1000, high_pad=5)   # would hit target, but before the alert
        after = day_bars("2026-10-30", "10:15", "11:00", 102.5, 1000, high_pad=0.1, low_pad=0.1)
        after.iloc[3, after.columns.get_loc("High")] = 104.5
        intraday = pd.concat({"AAA": pd.concat([before, after])}, axis=1)

        scored = daytrade_alerts.score_alerts(alerts, "2026-10-30", intraday)

        self.assertEqual(scored.loc[0, "Outcome"], "Target")
        self.assertAlmostEqual(scored.loc[0, "R"], round((104.0 - 102.3) / 1.3, 2))
        lines = daytrade_alerts.recap_lines(scored, "2026-10-30")
        self.assertIn("Target", lines[0])
        self.assertTrue(any("1 trades" in line for line in lines))
        self.assertEqual(daytrade_alerts.recap_lines(scored, "2026-10-31"), ["No large-cap day-trade alerts today."])

    def test_missing_alert_file_is_empty(self):
        self.assertTrue(daytrade_alerts.read_alerts("/nonexistent/daytrade_alerts.csv").empty)
        self.assertEqual(daytrade_alerts.summarize_daytrades(daytrade_alerts.read_alerts("/nonexistent.csv")), {"trades": 0})


class PrivateRepoFetchTests(unittest.TestCase):
    def response(self, status=200, text=""):
        response = mock.Mock(status_code=status, text=text)
        response.raise_for_status.return_value = None
        return response

    def test_token_reads_through_the_contents_api(self):
        import signal_log
        csv = "Date,Time,Ticker,Price\n2026-10-30,10:15,AAA,102.3\n"
        with mock.patch("requests.get", return_value=self.response(text=csv)) as get:
            alerts = daytrade_alerts.fetch_published_alerts(token="secret")
        url = get.call_args.args[0]
        self.assertEqual(url, f"https://api.github.com/repos/{signal_log.GITHUB_REPO}/contents/daytrade_alerts.csv")
        self.assertEqual(get.call_args.kwargs["params"], {"ref": "signal-log"})
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(list(alerts["Ticker"]), ["AAA"])

    def test_without_token_uses_the_raw_url_and_404_is_empty(self):
        import signal_log
        with mock.patch("requests.get", return_value=self.response(status=404)) as get:
            log = signal_log.fetch_published_log()
        self.assertEqual(get.call_args.args[0], signal_log.LOG_URL)
        self.assertTrue(log.empty)
        self.assertEqual(list(log.columns), signal_log.LOG_COLUMNS)


if __name__ == "__main__":
    unittest.main()
