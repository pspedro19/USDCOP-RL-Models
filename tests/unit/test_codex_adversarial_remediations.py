from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import ast
import importlib.util
import sys
import threading
import types

import pytest


if "src" not in sys.modules:
    package = types.ModuleType("src")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src")]
    sys.modules["src"] = package


def _load_migrator():
    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_adversarial", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modified_migration_is_drift_not_pending() -> None:
    migrator = _load_migrator()
    first = migrator.get_migration_files("fabric-v1")[0]
    executed = {first.name: "definitely-not-the-current-checksum"}

    with pytest.raises(migrator.MigrationDriftError, match=first.name):
        migrator.classify_migrations("fabric-v1", executed)


def test_fabric_validation_has_plan_specific_required_tables() -> None:
    migrator = _load_migrator()
    required = migrator.REQUIRED_TABLES_BY_PLAN["fabric-v1"]

    assert "control.artifact_identity" in required
    assert "forecast.forecast_output" in required
    assert "portfolio.target" in required
    assert "exec.reconciliation_event" in required


def test_review_gate_cannot_self_authorize_modified_sql(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrator = _load_migrator()
    migration = tmp_path / "001_reviewed.sql"
    migration.write_text("SELECT 1;\n", encoding="utf-8")
    monkeypatch.setitem(migrator.MIGRATION_PLANS, "fabric-v1", (migration,))
    reviewed = migrator.get_plan_digest("fabric-v1")
    monkeypatch.setitem(migrator.PINNED_PLAN_DIGESTS, "fabric-v1", reviewed)

    assert migrator.plan_is_authorized("fabric-v1", reviewed)

    migration.write_text("SELECT 2;\n", encoding="utf-8")
    attacker_digest = migrator.get_plan_digest("fabric-v1")
    assert attacker_digest != reviewed
    assert not migrator.plan_is_authorized("fabric-v1", attacker_digest)
    assert not migrator.plan_is_authorized("fabric-v1", reviewed)


def test_makefile_exposes_explicit_review_gated_fabric_invoker() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    target = makefile.split("db-migrate-fabric:", 1)[1]
    target = target.split("db-status-fabric:", 1)[0]

    assert "FABRIC_REVIEWED_DIGEST is required" in target
    assert "--plan fabric-v1" in target
    assert '--reviewed-digest "$$FABRIC_REVIEWED_DIGEST"' in target
    assert (
        "$(PYTHON) scripts/ops/db_migrate.py --plan fabric-v1 --plan-digest"
        in makefile
    )


def test_wired_migration_callers_select_legacy_plan_and_fail_closed() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    entrypoint = Path("services/inference_api/entrypoint.sh").read_text(
        encoding="utf-8"
    )
    health = Path("services/inference_api/routers/health.py").read_text(
        encoding="utf-8"
    )
    main = Path("services/inference_api/main.py").read_text(encoding="utf-8")
    fresh_install = Path(
        "scripts/validation/validate_fresh_install.py"
    ).read_text(encoding="utf-8")

    make_callers = [
        line.strip()
        for line in makefile.splitlines()
        if "$(PYTHON) scripts/ops/db_migrate.py" in line
        and "--plan legacy-init" in line
    ]
    entrypoint_callers = [
        line.strip()
        for line in entrypoint.splitlines()
        if "python /app/scripts/ops/db_migrate.py" in line
    ]

    assert len(make_callers) == 4
    assert len(entrypoint_callers) == 2
    assert all("--plan legacy-init" in line for line in make_callers)
    assert all("--plan legacy-init" in line for line in entrypoint_callers)
    assert "continuing anyway" not in entrypoint
    assert "Migration script not found, skipping" not in entrypoint
    migration_section = entrypoint.split("Running database migrations...", 1)[1]
    migration_section = migration_section.split("# Start API", 1)[0]
    assert migration_section.count("exit 1") == 3

    expected_run = "python scripts/ops/db_migrate.py --plan legacy-init"
    expected_validate = (
        "python scripts/ops/db_migrate.py --plan legacy-init --validate"
    )
    health_strings = {
        node.value
        for node in ast.walk(ast.parse(health))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    main_strings = {
        node.value
        for node in ast.walk(ast.parse(main))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    fresh_install_strings = {
        node.value
        for node in ast.walk(ast.parse(fresh_install))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert any(expected_run in value for value in health_strings)
    assert any(expected_validate in value for value in health_strings)
    assert "validate_tables(plan=\"legacy-init\")" in main
    assert any(expected_run in value for value in main_strings)
    assert any(expected_run in value for value in fresh_install_strings)
    assert "python scripts/db_migrate.py" not in fresh_install


def test_integrity_remediation_is_append_only_and_serializes_derivations() -> None:
    sql = Path(
        "database/migrations/079_fabric_integrity_remediation.sql"
    ).read_text(encoding="utf-8").lower()
    normalized = " ".join(sql.split())

    assert "pg_advisory_xact_lock" in sql
    assert "return null" in sql
    assert "before insert on control.artifact_identity" in normalized
    assert "trg_artifact_identity_append_only" in sql
    assert "trg_artifact_identity_no_truncate" in sql
    for table in (
        "forecast.forecast_output",
        "forecast.forecast_score",
        "forecast.model_horizon_result",
        "forecast.calibration_result",
    ):
        assert f"before update or delete on {table}" in normalized
        assert f"before truncate on {table}" in normalized


def test_governance_identity_and_capital_escalation_are_database_enforced() -> None:
    sql = Path(
        "database/migrations/079_fabric_integrity_remediation.sql"
    ).read_text(encoding="utf-8").lower()

    assert "strategy identity fields are immutable" in sql
    assert "paper -> champion must enter at canary" in sql
    assert "n_trials" in sql
    assert "trial_ledger_fingerprint" in sql
    assert "novelty_gate_passed" in sql
    assert "canary -> full requires canary evidence and vote-2" in sql


def test_canonical_artifact_atomic_publication_has_exactly_one_winner(
    tmp_path: Path,
) -> None:
    from src.identity.canonical import CanonicalArtifact, CanonicalizationError

    destination = tmp_path / "artifact.json"
    artifacts = [CanonicalArtifact.build({"publisher": index}) for index in range(8)]
    barrier = threading.Barrier(len(artifacts))

    def publish(artifact: CanonicalArtifact) -> str:
        barrier.wait()
        try:
            artifact.write(destination)
            return "published"
        except CanonicalizationError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=len(artifacts)) as pool:
        outcomes = list(pool.map(publish, artifacts))

    assert outcomes.count("published") == 1
    assert outcomes.count("conflict") == len(artifacts) - 1
    assert destination.read_bytes() in {artifact.content for artifact in artifacts}


def test_direct_portfolio_target_cannot_lie_about_its_identity() -> None:
    from src.portfolio.target import TargetBuilder, TargetError, TargetExposure

    now = datetime(2026, 1, 5, tzinfo=timezone.utc)
    target = TargetBuilder().build(
        target_version="v1",
        snapshot_id="snapshot-1",
        account_id="account-1",
        environment="paper",
        allocator_version="allocator-v1",
        valid_from=now,
        valid_until=now + timedelta(hours=1),
        rebalance_cutoff=now,
        decision_fingerprint="sha256:" + "a" * 64,
        constraints_snapshot={"gross_cap": Decimal("1")},
        exposures=[
            TargetExposure(
                allocation_id="allocation-1",
                strategy_id="strategy-1",
                sleeve_id="sleeve-1",
                instrument_id="instrument-1",
                instrument="USD/COP",
                side="LONG",
                risk_budget=Decimal("0.1"),
                target_weight=Decimal("0.1"),
                currency="COP",
            )
        ],
    )

    with pytest.raises(TargetError, match="semantic_hash"):
        replace(target, semantic_hash="sha256:" + "f" * 64)
    with pytest.raises(TargetError, match="target_id"):
        replace(target, target_id="fabricated-id")


def test_metric_environment_is_closed_and_implausible_value_is_not_ok() -> None:
    from src.metrics.engine import (
        FORMULAS,
        MetricCatalog,
        MetricContractError,
        MetricEngine,
    )

    catalog = MetricCatalog.load("config/metrics/catalog.yaml")
    formulas = dict(FORMULAS)
    formulas["strategy.sharpe"] = lambda _context, _annualization: 100.0
    engine = MetricEngine(
        catalog,
        formulas=formulas,
        annualization_by_asset={"usdcop": 52},
    )
    end = datetime(2026, 1, 5, tzinfo=timezone.utc)
    context = {
        "returns": [0.01] * 20,
        "n_trades": 20,
        "window_start": end - timedelta(weeks=26),
        "window_end": end,
    }
    kwargs = dict(
        entity_type="strategy",
        entity_id="strategy-1",
        metric="strategy.sharpe",
        window="26w",
        as_of=end,
        asset_id="usdcop",
        context=context,
    )

    with pytest.raises(MetricContractError, match="environment"):
        engine.compute(env="replay", **kwargs)
    event = engine.compute(env="backtest", **kwargs)
    assert event.status == "CRITICAL"
    assert event.dimensions["plausibility_violation"] is True
