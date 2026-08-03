from __future__ import annotations

import asyncio
import importlib.util
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_path(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MigrationConnection:
    def __init__(self, row):
        self.row = row
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.lock_ids: list[int] = []

    async def fetchval(self, sql: str, *args):
        if "pg_advisory_xact_lock" in sql:
            self.lock_ids.append(args[0])
            return None
        raise AssertionError(f"unexpected fetchval: {sql}")

    async def fetchrow(self, sql: str, *args):
        assert "FOR UPDATE" in sql
        return self.row

    async def execute(self, sql: str, *args):
        self.executed.append((sql, args))
        return "DELETE 1"


def test_failed_migration_record_is_recoverable_and_attempt_is_fenced() -> None:
    migrator = _load_path("db_migrate_phase2", "scripts/ops/db_migrate.py")
    conn = _MigrationConnection(
        {"checksum": "old-failed-checksum", "success": False}
    )

    should_execute = asyncio.run(
        migrator.claim_migration_attempt(conn, "080_example.sql", "new-checksum")
    )

    assert should_execute is True
    assert len(conn.lock_ids) == 1
    assert any(
        "DELETE FROM _migrations" in sql and args == ("080_example.sql",)
        for sql, args in conn.executed
    )
    assert migrator.migration_lock_id("080_example.sql") == conn.lock_ids[0]
    assert migrator.migration_lock_id("080_example.sql") != migrator.migration_lock_id(
        "081_example.sql"
    )


def test_applied_migration_is_idempotent_but_drift_is_fatal() -> None:
    migrator = _load_path("db_migrate_phase2_applied", "scripts/ops/db_migrate.py")
    same = _MigrationConnection({"checksum": "same", "success": True})
    assert (
        asyncio.run(migrator.claim_migration_attempt(same, "080.sql", "same"))
        is False
    )

    drift = _MigrationConnection({"checksum": "old", "success": True})
    with pytest.raises(migrator.MigrationDriftError, match="080.sql"):
        asyncio.run(migrator.claim_migration_attempt(drift, "080.sql", "new"))


def test_metric_annualization_comes_from_asset_profile_and_return_interval() -> None:
    from src.metrics.annualization import AnnualizationRegistry
    from src.metrics.engine import MetricCatalog, MetricEngine

    registry = AnnualizationRegistry.load(REPO_ROOT / "config" / "assets")
    assert registry.periods_per_year("usdcop", "PT5M") == 15_660
    assert registry.periods_per_year("usdcop", "P1D") == 261
    assert registry.periods_per_year("usdcop", "P1W") == 52
    with pytest.raises(ValueError, match="unsupported"):
        registry.periods_per_year("usdcop", "PT7M")

    engine = MetricEngine.from_asset_registry(
        MetricCatalog.load(REPO_ROOT / "config" / "metrics" / "catalog.yaml"),
        assets_dir=REPO_ROOT / "config" / "assets",
    )
    end = datetime(2026, 7, 27, tzinfo=timezone.utc)
    event = engine.compute(
        entity_type="strategy",
        entity_id="s1",
        metric="strategy.sharpe",
        window="26w",
        env="held_out",
        as_of=end,
        asset_id="usdcop",
        context={
            "returns": [0.01, -0.004, 0.003, -0.002] * 5,
            "return_interval": "P1W",
            "n_trades": 20,
            "window_start": end - timedelta(weeks=26),
            "window_end": end,
        },
    )
    assert event.dimensions["return_interval"] == "P1W"
    assert event.dimensions["annualization_periods"] == 52


def _source_bar(index: int, *, omit_available_delay: bool = False):
    from src.market.resampling import SourceBar

    start = datetime(2026, 7, 27, 13, 0, tzinfo=timezone.utc)
    event_time = start + timedelta(minutes=5 * index)
    return SourceBar(
        raw_bar_id=f"raw-{index:02d}",
        instrument_id="usdcop-spot",
        event_time=event_time,
        available_at=event_time
        + (timedelta(0) if omit_available_delay else timedelta(seconds=index)),
        open=Decimal(4_000 + index),
        high=Decimal(4_002 + index),
        low=Decimal(3_999 + index),
        close=Decimal("4000.5") + index,
        volume=Decimal(10 + index),
    )


def test_resampler_emits_only_complete_session_anchored_buckets() -> None:
    from src.contracts.asset_profile import load_asset_profile
    from src.market.resampling import ResamplePolicy, resample_complete_bars

    policy = ResamplePolicy.for_asset(
        load_asset_profile("usdcop"),
        source_interval="PT5M",
        target_interval="PT1H",
        compute_latency=timedelta(seconds=2),
    )
    complete = resample_complete_bars([_source_bar(i) for i in range(12)], policy)
    assert len(complete) == 1
    result = complete[0]
    assert result.bar_method == "resampled"
    assert result.open == Decimal(4_000)
    assert result.high == Decimal(4_013)
    assert result.low == Decimal(3_999)
    assert result.close == Decimal("4011.5")
    assert result.volume == sum(Decimal(10 + i) for i in range(12))
    assert result.source_raw_bar_ids == tuple(f"raw-{i:02d}" for i in range(12))
    assert result.available_at == max(
        _source_bar(i).available_at for i in range(12)
    ) + timedelta(seconds=2)

    assert resample_complete_bars([_source_bar(i) for i in range(11)], policy) == ()


def test_synthetic_model_is_fail_closed_outside_demo_domain() -> None:
    from src.governance.synthetic_isolation import (
        SyntheticIsolationError,
        validate_model_boundary,
    )

    good = {
        "model_id": "investor_demo",
        "algorithm": "SYNTHETIC",
        "environment": "demo",
        "surface": "synthetic",
        "execution_eligible": False,
    }
    validate_model_boundary(good, relation="demo.synthetic_model")

    for mutation in (
        {"environment": "paper"},
        {"surface": "action"},
        {"execution_eligible": True},
    ):
        with pytest.raises(SyntheticIsolationError):
            validate_model_boundary(good | mutation, relation="demo.synthetic_model")
    with pytest.raises(SyntheticIsolationError):
        validate_model_boundary(good, relation="config.models")

    production_markers = good | {
        "environment": "production",
        "surface": "action",
        "execution_eligible": True,
    }
    with pytest.raises(
        SyntheticIsolationError,
        match="synthetic model cannot be stored in real relation",
    ):
        validate_model_boundary(production_markers, relation="config.models")


def test_physical_and_synthetic_migrations_have_executable_static_smoke() -> None:
    physical = (
        REPO_ROOT / "database" / "migrations" / "080_market_physical_profile.sql"
    ).read_text(encoding="utf-8").lower()
    synthetic = (
        REPO_ROOT / "database" / "migrations" / "081_synthetic_demo_isolation.sql"
    ).read_text(encoding="utf-8").lower()

    physical_sql = re.sub(r"--[^\n]*", "", physical)
    synthetic_sql = re.sub(r"--[^\n]*", "", synthetic)
    assert "create_hypertable" in physical_sql
    assert "chunk_time_interval" in physical_sql
    assert "compress_segmentby" in physical_sql
    assert "add_retention_policy" not in physical_sql
    assert "cold archive restore evidence" in physical
    assert "create table if not exists demo.synthetic_model" in synthetic_sql
    assert "constraint synthetic_demo_only check" in synthetic_sql
    assert "execution_eligible = false" in synthetic_sql
    assert "create or replace function demo.reject_synthetic_performance" in synthetic_sql
    assert "raise exception" in synthetic_sql


def test_ci_and_readiness_matrix_are_executable_honest_contracts() -> None:
    validator = (
        REPO_ROOT / "scripts" / "validation" / "validate_fabric_contracts.py"
    ).read_text(encoding="utf-8")
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "fabric-contracts.yml"
    ).read_text(encoding="utf-8")
    matrix = (
        REPO_ROOT / ".claude" / "specs" / "planes" / "04b-readiness-matrix.md"
    ).read_text(encoding="utf-8")
    profiler = (
        REPO_ROOT / "scripts" / "diagnostics" / "timescale_profile_v2.py"
    ).read_text(encoding="utf-8")

    assert "legacy_bypass_allowlist" in validator
    assert "test_codex_phase2_backlog.py" in workflow
    for domain in (
        "Técnica",
        "Riesgo",
        "Ejecución",
        "Seguridad",
        "Compliance",
        "Operaciones",
        "Investor",
    ):
        assert domain in matrix
    assert "timescaledb_information.hypertables" in profiler
    assert "timescaledb_information.continuous_aggregates" in profiler
