"""Excel export of the Track Record tab: summary, swing signal log, AI-grade split and large-cap day trades."""
import io
from datetime import datetime

import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from daytrade_alerts import summarize_daytrades
from signal_log import summarize_by_grade, summarize_log

PRICE_FORMAT = "$#,##0.00"
R_FORMAT = "+0.00;-0.00;0.00"
PCT_FORMAT = '+0.00"%";-0.00"%";0.00"%"'
NUMBER_FORMATS = {
    "Price": PRICE_FORMAT, "Pivot": PRICE_FORMAT, "Stop": PRICE_FORMAT, "Target": PRICE_FORMAT,
    "Entry": PRICE_FORMAT, "Exit": PRICE_FORMAT, "Trigger": PRICE_FORMAT,
    "R": R_FORMAT, "Average R": R_FORMAT, "Total R": R_FORMAT,
    "Return %": PCT_FORMAT, "IWM %": PCT_FORMAT,
    "Win Rate %": '0.0"%"', "Reward/Risk": "0.0", "Breakout Score": "0", "Day Score": "0", "RVOL": '0.0"x"',
}
TEXT_COLUMNS = {
    "Scan Date", "Ticker", "IWM Trend", "Universe", "Status", "Entry Date", "Exit Date", "AI Grade", "AI Risk",
    "Updated", "Date", "Time", "Outcome", "Scope",
}
HEADER_FILL = PatternFill("solid", fgColor="1F3A5F")
HEADER_FONT = Font(bold=True, color="FFFFFF")


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """The logs are read from CSV, so numbers can arrive as text; Excel should get real numbers it can sum."""
    frame = frame.copy()
    for column in frame.columns:
        if column not in TEXT_COLUMNS:
            converted = pd.to_numeric(frame[column], errors="coerce")
            # Only convert columns that really are numeric, so stray text columns keep their values.
            if converted.notna().sum() >= frame[column].replace("", pd.NA).notna().sum():
                frame[column] = converted
    return frame


def _summary_rows(signal_log: pd.DataFrame, daytrade_log: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows = []
    if signal_log is not None:
        for scope, max_rank in [(f"Swing: top {top_n} (sent as alerts)", top_n), ("Swing: all logged setups", None)]:
            summary = summarize_log(signal_log, max_rank=max_rank)
            rows.append({
                "Scope": scope,
                "Signals": summary["signals"],
                "Closed Trades": summary["trades"],
                "Open": summary["still_open"],
                "Waiting": summary["waiting"],
                "Not Triggered": summary["not_triggered"],
                "Win Rate %": summary["win_rate_pct"],
                "Average R": summary["average_r"],
                "Total R": summary["total_r"],
                "Avg Return %": summary["average_return_pct"],
                "Avg IWM %": summary["average_iwm_pct"],
                "First Scan": summary["first_scan"],
                "Last Scan": summary["last_scan"],
            })
    if daytrade_log is not None:
        summary = summarize_daytrades(daytrade_log)
        dates = daytrade_log["Date"].dropna() if not daytrade_log.empty else pd.Series(dtype=str)
        rows.append({
            "Scope": "Large-cap day trades",
            "Signals": len(daytrade_log),
            "Closed Trades": summary["trades"],
            "Win Rate %": summary.get("win_rate_pct"),
            "Average R": summary.get("average_r"),
            "Total R": summary.get("total_r"),
            "First Scan": dates.min() if not dates.empty else None,
            "Last Scan": dates.max() if not dates.empty else None,
        })
    return pd.DataFrame(rows)


def _write_sheet(writer, frame: pd.DataFrame, name: str, formats: dict = None) -> None:
    frame.to_excel(writer, sheet_name=name, index=False)
    sheet = writer.sheets[name]
    sheet.freeze_panes = "A2"
    formats = {**NUMBER_FORMATS, **(formats or {})}
    for index, column in enumerate(frame.columns, start=1):
        letter = get_column_letter(index)
        header = sheet.cell(row=1, column=index)
        header.font, header.fill = HEADER_FONT, HEADER_FILL
        values = [str(column)] + [str(value) for value in frame[column].head(500) if pd.notna(value)]
        sheet.column_dimensions[letter].width = min(max(len(value) for value in values) + 3, 45)
        if column in formats:
            for cell in sheet[letter][1:]:
                cell.number_format = formats[column]
    if len(frame.columns):
        sheet.auto_filter.ref = sheet.dimensions


def build_track_record_workbook(signal_log: pd.DataFrame = None, daytrade_log: pd.DataFrame = None,
                                top_n: int = 10, generated: datetime = None) -> bytes:
    """An .xlsx with one sheet per Track Record table. Either log may be None when it couldn't be loaded."""
    generated = generated or datetime.now()
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary = _summary_rows(signal_log, daytrade_log, top_n)
        _write_sheet(writer, summary, "Summary", {"Avg Return %": PCT_FORMAT, "Avg IWM %": PCT_FORMAT})
        note_row = len(summary) + 3
        sheet = writer.sheets["Summary"]
        sheet.cell(row=note_row, column=1, value=f"Exported {generated:%Y-%m-%d %H:%M}. Results are in R (multiples of the initial risk).")
        sheet.cell(row=note_row + 1, column=1, value="Screening output is informational, not investment advice.")

        if signal_log is not None:
            signals = signal_log.copy()
            if not signals.empty:
                signals = signals.sort_values(["Scan Date", "Rank"], ascending=[False, True])
            _write_sheet(writer, _numeric(signals), "Swing Signals")

            by_grade = []
            for scope, max_rank in [(f"Top {top_n}", top_n), ("All setups", None)]:
                grades = summarize_by_grade(signal_log, max_rank=max_rank)
                if not grades.empty:
                    by_grade.append(grades.assign(Scope=scope)[["Scope", *grades.columns]])
            if by_grade:
                _write_sheet(writer, pd.concat(by_grade, ignore_index=True), "By AI Grade")

        if daytrade_log is not None:
            trades = daytrade_log.copy()
            if not trades.empty:
                trades = trades.sort_values(["Date", "Time"], ascending=False)
            _write_sheet(writer, _numeric(trades), "Day Trades")
    return buffer.getvalue()
