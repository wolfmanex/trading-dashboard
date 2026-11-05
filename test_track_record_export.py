"""Excel export of the Track Record tab."""
import io
import unittest
from datetime import datetime

import pandas as pd
from openpyxl import load_workbook

from daytrade_alerts import DAYTRADE_COLUMNS, read_alerts
from signal_log import LOG_COLUMNS, empty_log, read_log
from track_record_export import build_track_record_workbook


def signal_log():
    rows = [
        {"Scan Date": "2026-10-05", "Ticker": "AAA", "Rank": 1, "Price": 10, "Pivot": 10.5, "Stop": 9.5,
         "Target": 13, "Status": "Target", "Entry": 10.5, "Exit": 13, "R": 2.5, "Return %": 23.8, "AI Grade": "A"},
        {"Scan Date": "2026-10-06", "Ticker": "BBB", "Rank": 12, "Price": 20, "Pivot": 21, "Stop": 19,
         "Target": 25, "Status": "Stop", "Entry": 21, "Exit": 19, "R": -1, "Return %": -9.5, "AI Grade": ""},
        {"Scan Date": "2026-10-07", "Ticker": "CCC", "Rank": 2, "Status": "Waiting", "AI Grade": "B"},
    ]
    # Round-trip through CSV like the app does, so numbers arrive the way the published log delivers them.
    return read_log(io.StringIO(pd.DataFrame(rows).reindex(columns=LOG_COLUMNS).to_csv(index=False)))


def daytrade_log():
    rows = [{"Date": "2026-11-03", "Time": "10:15", "Ticker": "MSFT", "Price": 500, "Trigger": 499, "Stop": 495,
             "Target": 510, "RVOL": 1.8, "Day Score": 72, "Outcome": "Target", "Exit": 510, "R": 2.0}]
    return read_alerts(io.StringIO(pd.DataFrame(rows).reindex(columns=DAYTRADE_COLUMNS).to_csv(index=False)))


def open_book(data):
    return load_workbook(io.BytesIO(data))


class TrackRecordExportTests(unittest.TestCase):
    def test_one_sheet_per_table_with_real_numbers(self):
        book = open_book(build_track_record_workbook(signal_log(), daytrade_log(), top_n=10, generated=datetime(2026, 11, 5)))
        self.assertEqual(book.sheetnames, ["Summary", "Swing Signals", "By AI Grade", "Day Trades"])

        signals = pd.read_excel(io.BytesIO(build_track_record_workbook(signal_log(), daytrade_log())), sheet_name="Swing Signals")
        self.assertEqual(list(signals["Ticker"]), ["CCC", "BBB", "AAA"])   # newest scan first, like the tab
        self.assertEqual(signals.loc[signals["Ticker"] == "AAA", "R"].item(), 2.5)
        r_cell = book["Swing Signals"].cell(row=4, column=LOG_COLUMNS.index("R") + 1)
        self.assertIsInstance(r_cell.value, float)
        self.assertEqual(book["Swing Signals"].freeze_panes, "A2")

        summary = pd.read_excel(io.BytesIO(build_track_record_workbook(signal_log(), daytrade_log(), top_n=10)), sheet_name="Summary")
        scopes = summary.set_index("Scope")
        self.assertEqual(scopes.loc["Swing: top 10 (sent as alerts)", "Signals"], 2)
        self.assertEqual(scopes.loc["Swing: all logged setups", "Closed Trades"], 2)
        self.assertEqual(scopes.loc["Large-cap day trades", "Total R"], 2.0)

        trades = pd.read_excel(io.BytesIO(build_track_record_workbook(signal_log(), daytrade_log())), sheet_name="Day Trades")
        self.assertEqual(trades.loc[0, "R"], 2.0)

    def test_a_missing_or_empty_log_still_exports_the_rest(self):
        book = open_book(build_track_record_workbook(None, daytrade_log()))
        self.assertEqual(book.sheetnames, ["Summary", "Day Trades"])
        book = open_book(build_track_record_workbook(empty_log(), read_alerts("/nonexistent.csv")))
        self.assertEqual(book.sheetnames, ["Summary", "Swing Signals", "Day Trades"])


if __name__ == "__main__":
    unittest.main()
