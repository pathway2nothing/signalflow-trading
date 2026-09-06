"""signalflow._time: the one place durations, dates, intervals and bar widths are parsed."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from signalflow._time import advance, bar_seconds, interval_seconds, parse_datetime, parse_duration, retreat, to_epoch


def test_parse_duration():
    assert parse_duration("1d") == timedelta(days=1)
    assert parse_duration("365d") == timedelta(days=365)
    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("30m") == timedelta(minutes=30)
    assert parse_duration("1.5h") == timedelta(minutes=90)
    with pytest.raises(ValueError):
        parse_duration("1mo")
    with pytest.raises(ValueError):
        parse_duration("abc")


def test_advance_and_retreat_calendar_months():
    jan31 = datetime(2024, 1, 31)
    assert advance(jan31, "1mo") == datetime(2024, 2, 29)
    assert retreat(datetime(2024, 3, 31), "1mo") == datetime(2024, 2, 29)
    assert advance(jan31, "2d") == datetime(2024, 2, 2)
    assert retreat(jan31, "12h") == datetime(2024, 1, 30, 12)


def test_parse_datetime_and_epoch_are_utc():
    assert parse_datetime("2024-01-02") == datetime(2024, 1, 2)
    assert parse_datetime("2024-01-02 03:04") == datetime(2024, 1, 2, 3, 4)
    assert parse_datetime("2024-01-02T03:04:05") == datetime(2024, 1, 2, 3, 4, 5)
    assert to_epoch("1970-01-01 00:00:10") == 10
    assert to_epoch(1_700_000_000) == 1_700_000_000
    assert parse_datetime(86400) == datetime(1970, 1, 2)
    with pytest.raises(ValueError):
        parse_datetime("02/01/2024")


def test_interval_seconds():
    assert interval_seconds("1m") == 60
    assert interval_seconds("1w") == 604800
    with pytest.raises(ValueError):
        interval_seconds("1M")


def test_bar_seconds_median_of_positive_steps():
    ts = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10)]
    assert bar_seconds(pl.Series(ts)) == 3600.0
    assert bar_seconds(ts + ts) == 3600.0  # duplicates (several pairs) are ignored
    assert bar_seconds([datetime(2024, 1, 1)]) == 0.0
    assert bar_seconds(pl.Series([1.0, 2.0, 4.0, 7.0])) == 2.0
