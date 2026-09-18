from __future__ import annotations

from datetime import datetime, timezone

from src.research.llm_forward.session_date import session_date_from_context


def test_data_interval_end_has_precedence_over_logical_date() -> None:
    result = session_date_from_context({
        "data_interval_end": datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc),
        "logical_date": datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc),
    })
    assert result == "2026-09-10"


def test_iso_interval_converts_to_bogota_date() -> None:
    assert session_date_from_context({
        "data_interval_end": "2026-09-14T04:00:00+00:00"
    }) == "2026-09-13"


def test_legacy_logical_date_is_only_a_fallback() -> None:
    assert session_date_from_context({
        "logical_date": datetime(2026, 9, 14, 13, 0, tzinfo=timezone.utc)
    }) == "2026-09-14"
