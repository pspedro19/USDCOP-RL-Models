"""
CXD-057 — el estado de aprobación es PRIVADO y el consumidor del Voto 2 lo sigue leyendo.
==========================================================================================

Hallazgo (CODEX): ``usdcop-trading-dashboard/public/data/production/approval_state*.json``
publicaba ``gates``, el gate ``deflated_sharpe`` (DSR trial-aware) y ``backtest_metrics``.
El middleware del dashboard solo exige SESIÓN para ``/data/**``, así que cualquier
``free``/``subscriber`` autenticado se saltaba el ``research:read`` que el SSOT reserva a
Backtest/Experimentos (``frontend-backend-contract.md`` §6, ``ux-navigation.md`` P3,
``docs/rbac/VISUAL-SPEC-CHECKLIST.md`` §B).

Estos tests cubren el lado Python del remedio, que es lo que NO puede romperse:

  1. Ausencia física bajo ``public/`` + presencia en ``<repo>/data/approvals/``.
  2. El SSOT de la ruta (``src/contracts/approval_store.py``) resuelve scoped→singleton
     con la MISMA regla que el DAG H5-L4b y el approve del dashboard.
  3. **Fail-closed**: sin artefacto ⇒ ``None``, jamás un estado fabricado.
  4. La proyección pública es por ALLOWLIST y no deja pasar campos futuros.
  5. Los consumidores del Voto 2 (DAG H5-L4b guard_approved, DAG H5-L4 validate_output,
     watchdog, runner diario) apuntan al store privado y NO a ``public/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.contracts import approval_store as store

REPO = Path(__file__).resolve().parents[2]
PUBLIC_PROD = REPO / "usdcop-trading-dashboard" / "public" / "data" / "production"
APPROVALS = REPO / "data" / "approvals"


# ─────────────────────────────────────────────────────────── 1 · ubicación física


def test_no_approval_artifact_survives_under_public():
    leftovers = sorted(p.name for p in PUBLIC_PROD.glob("approval_state*.json"))
    assert leftovers == [], f"approval_state bajo public/: {leftovers}"


def test_artifacts_live_in_the_private_root():
    names = sorted(p.name for p in APPROVALS.glob("approval_state*.json"))
    assert "approval_state.json" in names
    assert len(names) >= 5, names


def test_no_json_under_public_production_carries_internal_keys():
    leak = re.compile(r'"(gates|deflated_sharpe|backtest_metrics)"')
    offenders = [
        str(p.relative_to(REPO))
        for p in PUBLIC_PROD.rglob("*.json")
        if leak.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == [], offenders


# ───────────────────────────────────────────── 2/3 · resolución + fail-closed


def test_resolution_prefers_scoped_then_owning_singleton(tmp_path, monkeypatch):
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    (tmp_path / "approval_state.json").write_text(
        json.dumps({"status": "PENDING_APPROVAL", "strategy": "smart_simple_v11"}), encoding="utf-8")
    (tmp_path / "approval_state_btc_trend_b2.json").write_text(
        json.dumps({"status": "APPROVED", "strategy": "btc_trend_b2"}), encoding="utf-8")

    assert store.resolve_approval_path("btc_trend_b2").name == "approval_state_btc_trend_b2.json"
    # singleton solo cuando ES esa estrategia
    assert store.resolve_approval_path("smart_simple_v11").name == "approval_state.json"
    # una id ajena JAMÁS resuelve al bundle de otro
    assert store.resolve_approval_path("gold_trend_simple") is None
    assert store.resolve_approval_path(None).name == "approval_state.json"


@pytest.mark.parametrize("sid", ["../../etc/passwd", "a/b", "a\\b", ".hidden", ""])
def test_traversal_ids_never_escape_the_private_root(tmp_path, monkeypatch, sid):
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    assert not store.is_valid_strategy_id(sid)
    # una id inválida degrada al singleton, nunca a una ruta compuesta
    assert store.approval_path(sid) == tmp_path / "approval_state.json"


def test_fail_closed_when_artifact_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path / "nope"))
    assert store.resolve_approval_path(None) is None
    assert store.read_approval(None) is None  # None, no un estado fabricado


def test_fail_closed_when_artifact_is_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path))
    (tmp_path / "approval_state.json").write_text("{not json{{{", encoding="utf-8")
    assert store.read_approval(None) is None


def test_round_trip_write_read(tmp_path, monkeypatch):
    monkeypatch.setenv("APPROVALS_DATA_DIR", str(tmp_path / "fresh"))
    doc = {"status": "PENDING_APPROVAL", "strategy": "x_1", "gates": [{"gate": "deflated_sharpe"}]}
    p = store.write_approval(doc, "x_1")
    assert p.parent.name == "fresh"
    assert store.read_approval("x_1")["gates"][0]["gate"] == "deflated_sharpe"


# ────────────────────────────────────────────────────── 4 · allowlist pública


def test_public_projection_is_an_allowlist_and_drops_future_fields():
    assert store.PUBLIC_APPROVAL_FIELDS == (
        "status", "strategy", "strategy_name", "approved_at", "created_at", "last_updated")
    pub = store.to_public_approval({
        "status": "APPROVED", "strategy": "s", "strategy_name": "S",
        "created_at": "c", "last_updated": "u", "approved_at": "a",
        "gates": [{"gate": "deflated_sharpe"}], "backtest_metrics": {"sharpe": 1},
        "backtest_recommendation": "REVIEW", "deploy_manifest": {}, "campo_futuro": "secreto",
    })
    assert set(pub) == set(store.PUBLIC_APPROVAL_FIELDS)
    blob = json.dumps(pub)
    for k in ("gates", "deflated_sharpe", "backtest_metrics", "deploy_manifest",
              "backtest_recommendation", "campo_futuro", "secreto"):
        assert k not in blob


def test_ts_and_python_allowlists_are_mirrors():
    """El allowlist vive en DOS lenguajes: si divergen, el cliente ve algo distinto."""
    ts = (REPO / "usdcop-trading-dashboard" / "lib" / "approvals" / "store.ts").read_text(encoding="utf-8")
    block = ts.split("export const PUBLIC_APPROVAL_FIELDS = [")[1].split("] as const;")[0]
    assert tuple(re.findall(r"'([a-z_]+)'", block)) == store.PUBLIC_APPROVAL_FIELDS


# ───────────────────────────────── 5 · los consumidores del Voto 2 apuntan al privado


CONSUMERS = [
    "airflow/dags/forecast_h5_l4b_production_deploy.py",   # guard_approved (hard gate Voto 2)
    "airflow/dags/forecast_h5_l4_backtest_promotion.py",   # validate_output / report_metrics
    "airflow/dags/core_watchdog.py",                       # frescura del artefacto
    "scripts/pipeline/train_and_export_smart_simple.py",   # export + --reset-approval
    "scripts/pipeline/run_daily_production.py",            # check_approval (dry-run gate)
    "scripts/pipeline/replay_backtest_universal.py",
    "scripts/pipeline/run_btc_pipeline.py",
    "scripts/pipeline/publish_gold_dynexit.py",
    "scripts/pipeline/publish_gold_trend_simple.py",
    "scripts/analysis/backtest_2026_production.py",
    "scripts/ops/log_training_to_mlflow.py",
    "scripts/validation/run_e2e_suite.py",
    "src/contracts/strategy_manifest.py",
]

# Rutas que reintroducirían la fuga: el artefacto colgando de public/.
_PUBLIC_APPROVAL = re.compile(
    r"(public[/\\]data[/\\]production[^\n]*approval_state|"
    r"(?:DASHBOARD_DATA_DIR|DASHBOARD_DIR|PROD_DIR|OUTPUT_DIR|PRODUCTION_DIR|PUBLIC_DATA|prod_dir|prod)\s*/\s*"
    r"[\"']?(?:f\")?approval_state)")


@pytest.mark.parametrize("rel", CONSUMERS)
def test_consumer_does_not_read_approval_state_from_public(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    # se ignoran comentarios/docstrings: solo importa el código que construye rutas
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    hit = _PUBLIC_APPROVAL.search(code)
    assert hit is None, f"{rel} sigue resolviendo approval_state bajo public/: {hit.group(0)!r}"


@pytest.mark.parametrize("rel", CONSUMERS)
def test_consumer_points_at_the_private_root(rel):
    src = (REPO / rel).read_text(encoding="utf-8")
    assert ("approval_store" in src or "APPROVALS_DIR" in src
            or "data" in src and "approvals" in src), f"{rel} no referencia el store privado"
