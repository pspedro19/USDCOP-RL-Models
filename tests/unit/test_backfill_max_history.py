from datetime import datetime, timezone
import sys
import types

import pytest

from scripts.ops.backfill_max_history import (
    _upsert_daily_with_partial_repair,
    _upsert_m5_with_partial_repair,
    bar_complete_after,
    bar_is_complete,
)


UTC = timezone.utc


@pytest.mark.parametrize(
    ("interval", "opened", "expected"),
    [
        ("5min", "2026-07-22T12:00:00", "2026-07-22T12:10:00"),
        ("1h", "2026-07-22T12:00:00", "2026-07-22T13:05:00"),
        ("4h", "2026-07-22T12:00:00", "2026-07-22T16:05:00"),
        ("1day", "2026-07-22T00:00:00", "2026-07-23T00:05:00"),
        ("1month", "2026-12-01T00:00:00", "2027-01-01T00:05:00"),
    ],
)
def test_bar_complete_after_respects_open_timestamp(interval, opened, expected):
    start = datetime.fromisoformat(opened).replace(tzinfo=UTC)
    end = datetime.fromisoformat(expected).replace(tzinfo=UTC)
    assert bar_complete_after(start, interval) == end


def test_bar_is_complete_rejects_partial_hour():
    opened = datetime(2026, 7, 22, 12, tzinfo=UTC)
    assert not bar_is_complete(
        opened, "1h", datetime(2026, 7, 22, 12, 20, tzinfo=UTC)
    )
    assert bar_is_complete(
        opened, "1h", datetime(2026, 7, 22, 13, 5, tzinfo=UTC)
    )


def test_bar_complete_after_rejects_unknown_interval():
    with pytest.raises(ValueError, match="Unsupported"):
        bar_complete_after(datetime(2026, 7, 22, tzinfo=UTC), "2h")


class _ExecuteValuesCursor:
    rowcount = 1

    def mogrify(self, _template, _args):
        return b"(NULL)"

    def execute(self, query):
        self.query = query


@pytest.mark.parametrize(
    ("upsert", "row"),
    [
        (
            _upsert_m5_with_partial_repair,
            (
                datetime(2026, 7, 22, tzinfo=UTC),
                "USD/MXN",
                1.0,
                1.0,
                1.0,
                1.0,
                0,
                "twelvedata_backfill",
                datetime(2026, 7, 22, 0, 10, tzinfo=UTC),
            ),
        ),
        (
            _upsert_daily_with_partial_repair,
            (
                datetime(2026, 7, 22, tzinfo=UTC),
                "USD/MXN",
                1.0,
                1.0,
                1.0,
                1.0,
                0,
                "twelvedata_daily_deep",
                datetime(2026, 7, 23, 0, 5, tzinfo=UTC),
            ),
        ),
    ],
)
def test_repair_upserts_escape_like_wildcard_for_execute_values(
    upsert, row, monkeypatch
):
    def execute_values(cursor, sql, _rows, page_size):
        assert page_size == 2000
        assert "%" not in sql.replace("%%", "").replace("%s", "")
        cursor.execute(sql.replace("%%", "%").replace("%s", "(NULL)"))

    psycopg2 = types.ModuleType("psycopg2")
    extras = types.ModuleType("psycopg2.extras")
    extras.execute_values = execute_values
    psycopg2.extras = extras
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)
    cursor = _ExecuteValuesCursor()
    assert upsert(cursor, [row]) == 1
    assert "LIKE 'twelvedata%'" in cursor.query
