from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "database/migrations/085_feature_status_provenance.sql"


def _sql() -> str:
    return re.sub(r"\s+", " ", MIGRATION.read_text(encoding="utf-8")).lower()


def _migrator():
    path = ROOT / "scripts/ops/db_migrate.py"
    spec = importlib.util.spec_from_file_location("db_migrate_c031", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_085_preserves_legacy_rows_and_db_owns_new_creation_time() -> None:
    sql = _sql()
    assert "add column if not exists created_at timestamptz" in sql
    assert "update quality.feature_status" not in sql
    assert "alter column created_at set default clock_timestamp()" in sql
    assert "new.created_at := clock_timestamp()" in sql
    assert "new.observed_at > new.created_at" in sql
    assert "before insert on quality.feature_status" in sql
    assert "not valid" in sql


def test_085_has_a_dedicated_pinned_review_gated_plan() -> None:
    migrator = _migrator()
    plan = "feature-status-provenance-v1"
    assert [path.name for path in migrator.get_migration_files(plan)] == [
        "085_feature_status_provenance.sql"
    ]
    assert plan in migrator.REVIEW_GATED_PLANS
    assert migrator.PINNED_PLAN_DIGESTS[plan] == (
        "sha256:29b3f7dc2dcff3c558057567a4033de30797058f361f801dae357e0ae185fb0b"
    )
    assert migrator.PLAN_PREREQUISITE_TABLES[plan] == ("quality.feature_status",)
    assert migrator.REQUIRED_COLUMNS_BY_PLAN[plan] == {
        "quality.feature_status": {
            "created_at": "Database-owned feature-status creation seal"
        }
    }
    digest = migrator.get_plan_digest(plan)
    assert not migrator.plan_is_authorized(plan, None)
    assert migrator.plan_is_authorized(plan, digest)
