from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "database/migrations/086_lineage_last_verified_at.sql"


def _sql() -> str:
    return re.sub(r"\s+", " ", MIGRATION.read_text(encoding="utf-8")).lower()


def _migrator():
    path = ROOT / "scripts/ops/db_migrate.py"
    spec = importlib.util.spec_from_file_location("db_migrate_c033", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_086_adds_a_non_null_verification_clock_without_editing_076() -> None:
    sql = _sql()
    assert "alter table lineage.node add column if not exists last_verified_at timestamptz" in sql
    assert "set last_verified_at = created_at" in sql
    assert "where last_verified_at is null" in sql
    assert "alter column last_verified_at set default now()" in sql
    assert "alter column last_verified_at set not null" in sql
    assert "comment on column lineage.node.last_verified_at" in sql


def test_086_declares_legacy_backfill_is_inferred_not_observed() -> None:
    sql = _sql()
    comment = sql.split("comment on column lineage.node.last_verified_at", 1)[1]
    assert "inferred" in comment
    assert "created_at" in comment
    assert "not an observed verification" in comment


def test_086_is_additive_and_does_not_redefine_revision_semantics() -> None:
    sql = _sql()
    assert "drop table" not in sql
    assert "delete from" not in sql
    assert "truncate" not in sql
    assert "create or replace function lineage.apply_revision_semantics" not in sql


def test_086_has_a_scoped_review_gated_unpinned_plan() -> None:
    migrator = _migrator()
    plan = "lineage-verification-v1"
    assert [path.name for path in migrator.get_migration_files(plan)] == [
        "086_lineage_last_verified_at.sql"
    ]
    assert plan in migrator.REVIEW_GATED_PLANS
    assert plan not in migrator.PINNED_PLAN_DIGESTS
    assert migrator.PLAN_PREREQUISITE_TABLES[plan] == ("lineage.node",)
    assert migrator.REQUIRED_TABLES_BY_PLAN[plan] == {
        "lineage.node": "Lineage nodes with producer verification clock"
    }
    assert migrator.REQUIRED_COLUMNS_BY_PLAN[plan] == {
        "lineage.node": {
            "last_verified_at": "Latest producer-observed verification time"
        }
    }
    assert not migrator.plan_is_authorized(plan, None)
    assert not migrator.plan_is_authorized(plan, migrator.get_plan_digest(plan))
