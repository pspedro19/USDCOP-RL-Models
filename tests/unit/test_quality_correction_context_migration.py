from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "database" / "migrations" / "084_quality_correction_context.sql"
APPLIED_073 = ROOT / "database" / "migrations" / "073_market_quality.sql"


def _sql() -> str:
    return re.sub(r"\s+", " ", MIGRATION.read_text(encoding="utf-8")).lower()


def test_084_is_additive_and_keeps_applied_073_untouched() -> None:
    sql = _sql()

    assert MIGRATION.exists()
    assert "alter table quality.quarantine_event" in sql
    assert "add column if not exists provider_id text" in sql
    assert "add column if not exists provider_symbol text" in sql
    assert "add column if not exists interval_id text references reference.bar_interval" in sql
    assert "add column if not exists observed_at timestamptz" in sql
    assert "add column if not exists source_uri text" in sql
    assert "add column if not exists context_version smallint" in sql
    assert "alter table market.raw_bar" not in sql
    assert "alter table market.canonical_bar" not in sql
    assert APPLIED_073.read_text(encoding="utf-8").startswith(
        "-- Migration 073: immutable raw/canonical market bars"
    )


def test_every_new_ohlcv_quarantine_requires_complete_typed_context() -> None:
    sql = _sql()

    assert "before insert on quality.quarantine_event" in sql
    assert "new.entity_type = 'ohlcv_bar'" in sql
    for field in (
        "new.provider_id",
        "new.provider_symbol",
        "new.interval_id",
        "new.observed_at",
        "new.source_uri",
        "new.context_version",
    ):
        assert field in sql
    assert "split_part" not in sql
    assert "string_to_array" not in sql


def test_one_quarantine_has_at_most_one_correction_and_consistent_state() -> None:
    sql = _sql()

    assert "having count(*) > 1" in sql
    assert "create unique index if not exists uq_quality_correction_quarantine" in sql
    assert "on quality.correction_event (quarantine_id)" in sql
    assert "(status = 'corrected') = (correction_event_id is not null)" in sql
    assert "not valid" in sql, "legacy evidence must remain loadable without invented backfill"
