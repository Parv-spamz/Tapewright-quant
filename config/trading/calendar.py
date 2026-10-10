from datetime import date, datetime
from functools import lru_cache


def _get_easter_sunday(year: int) -> date:
    """Calculate Easter Sunday using the Meeus/Jones/Butcher Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _get_nth_weekday(year: int, month: int, target_weekday: int, n: int) -> date:
    """Find the n-th target_weekday of a given month (0=Monday, 6=Sunday)."""
    first_day = date(year, month, 1)
    offset = (target_weekday - first_day.weekday()) % 7
    day_num = 1 + offset + (n - 1) * 7
    return date(year, month, day_num)


def _get_last_weekday(year: int, month: int, last_day_num: int, target_weekday: int) -> date:
    """Find the last target_weekday of a given month."""
    last_day = date(year, month, last_day_num)
    offset = (last_day.weekday() - target_weekday) % 7
    return date(year, month, last_day_num - offset)


def _observe_fixed_holiday(dt: date, allow_saturday_prior_friday: bool = True) -> date | None:
    """Apply weekend observation rule: Saturday -> Friday, Sunday -> Monday.

    If allow_saturday_prior_friday is False and dt is Saturday, returns None (unobserved).
    """
    weekday = dt.weekday()
    if weekday == 5:  # Saturday
        if allow_saturday_prior_friday:
            return date.fromordinal(dt.toordinal() - 1)
        return None
    elif weekday == 6:  # Sunday
        return date.fromordinal(dt.toordinal() + 1)
    return dt


@lru_cache(maxsize=128)
def get_nyse_holidays(year: int) -> frozenset[date]:
    """Compute the rule-based set of observed NYSE holidays for a given year (2000-2040).

    Holidays included:
    1. New Year's Day (Jan 1, Sat not observed on prior Friday, Sun -> Mon Jan 2)
    2. Martin Luther King Jr. Day (3rd Mon in Jan)
    3. Washington's Birthday / Presidents' Day (3rd Mon in Feb)
    4. Good Friday (2 days before Easter Sunday)
    5. Memorial Day (Last Mon in May)
    6. Juneteenth (Jun 19, observed starting 2022)
    7. Independence Day (Jul 4, Sat -> Fri, Sun -> Mon)
    8. Labor Day (1st Mon in Sep)
    9. Thanksgiving Day (4th Thu in Nov)
    10. Christmas Day (Dec 25, Sat -> Fri, Sun -> Mon)
    """
    holidays: set[date] = set()

    # 1. New Year's Day (Sat -> not observed on prior Friday; Sun -> Mon Jan 2)
    new_years = _observe_fixed_holiday(date(year, 1, 1), allow_saturday_prior_friday=False)
    if new_years is not None and new_years.year == year:
        holidays.add(new_years)

    # 2. Martin Luther King Jr. Day (3rd Monday in January)
    holidays.add(_get_nth_weekday(year, 1, target_weekday=0, n=3))

    # 3. Presidents' Day (3rd Monday in February)
    holidays.add(_get_nth_weekday(year, 2, target_weekday=0, n=3))

    # 4. Good Friday (2 days before Easter Sunday)
    easter = _get_easter_sunday(year)
    good_friday = date.fromordinal(easter.toordinal() - 2)
    holidays.add(good_friday)

    # 5. Memorial Day (Last Monday in May)
    holidays.add(_get_last_weekday(year, 5, last_day_num=31, target_weekday=0))

    # 6. Juneteenth (June 19, observed from 2022 onwards)
    if year >= 2022:
        juneteenth = _observe_fixed_holiday(date(year, 6, 19), allow_saturday_prior_friday=True)
        if juneteenth is not None and juneteenth.year == year:
            holidays.add(juneteenth)

    # 7. Independence Day (July 4)
    july4 = _observe_fixed_holiday(date(year, 7, 4), allow_saturday_prior_friday=True)
    if july4 is not None and july4.year == year:
        holidays.add(july4)

    # 8. Labor Day (1st Monday in September)
    holidays.add(_get_nth_weekday(year, 9, target_weekday=0, n=1))

    # 9. Thanksgiving Day (4th Thursday in November, Thursday = 3)
    holidays.add(_get_nth_weekday(year, 11, target_weekday=3, n=4))

    # 10. Christmas Day (December 25)
    xmas = _observe_fixed_holiday(date(year, 12, 25), allow_saturday_prior_friday=True)
    if xmas is not None and xmas.year == year:
        holidays.add(xmas)

    return frozenset(holidays)


def is_trading_day(d: date | datetime | str, calendar: str = "NYSE") -> bool:
    """Check whether a given date is an active trading day for the specified calendar.

    Supported calendars:
    - 'NYSE': Monday-Friday minus rule-based NYSE holidays.
    - 'CRYPTO_24_7': Every calendar day is a trading day.
    """
    cal_upper = calendar.strip().upper()
    if cal_upper == "CRYPTO_24_7":
        return True
    elif cal_upper != "NYSE":
        raise ValueError(f"Unsupported trading calendar '{calendar}'. Must be 'NYSE' or 'CRYPTO_24_7'.")

    # Normalize to datetime.date
    if isinstance(d, datetime):
        day_date = d.date()
    elif isinstance(d, date):
        day_date = d
    elif isinstance(d, str):
        clean_str = d.strip().split("T")[0].split(" ")[0]
        day_date = date.fromisoformat(clean_str)
    else:
        raise TypeError(f"Expected date, datetime, or ISO string, got {type(d)}")

    # Check weekend (5=Saturday, 6=Sunday)
    if day_date.weekday() >= 5:
        return False

    # Check holidays
    holidays = get_nyse_holidays(day_date.year)
    return day_date not in holidays
