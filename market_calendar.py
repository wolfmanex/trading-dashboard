"""NYSE trading days: weekdays that aren't full-day exchange holidays.

Used by the scan's duplicate-run guard, the intraday alerts and the evening recap so nothing runs or
alerts on a day the market is closed. Early-close days (e.g. the day after Thanksgiving) count as
normal trading days. The holiday list has to be extended each year; past its last year only weekends
count as closed and a warning is printed.
"""
from datetime import date

NYSE_HOLIDAYS = {
    2026: {
        date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25),
        date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
    },
    2027: {
        date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31),
        date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24),
    },
}


def is_trading_day(day: date) -> bool:
    """True on weekdays that aren't NYSE full-day holidays."""
    if day.weekday() >= 5:
        return False
    holidays = NYSE_HOLIDAYS.get(day.year)
    if holidays is None:
        print(f"Warning: no NYSE holiday list for {day.year} (market_calendar.py); only weekends count as closed.")
        return True
    return day not in holidays
