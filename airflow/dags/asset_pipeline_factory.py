"""
DAG FACTORY: asset_<asset_id>_pipeline_weekly
=============================================
Emits ONE data-science-lifecycle pipeline DAG per asset declared in
`config/assets/pipelines.yaml`, making every tradeable index/pair (Gold, BTC,
future additions) fully **DAG-driven** — not just a manual script-runner.

Each generated DAG's tasks map to the DS-cycle:

    l0_ingest            ->  L0  Data ingestion (daily OHLCV -> seed / DB)
    l4_backtest_publish  ->  L2+L4+L5  features -> regime -> backtest (honest gate) -> publish bundle
    l6_verify_registry   ->  L6  Verify/Monitor (registry.json has the asset + fresh strategy bundles)

USD/COP is intentionally NOT here — it runs the richer bespoke H5 weekly chain
(forecast_h5_l3..l7). Adding an asset = ONE entry in the SSOT yaml (no code here).

Graceful degradation: a stage marked `graceful: true` (the ingest refresh) does
not block the science stage — if the live feed is down, the backtest still runs
on the last good seed. The failed ingest task remains visible in the run so the
staleness is surfaced, never hidden (`trigger_rule=ALL_DONE` on the next stage).

Contract: CTR-ASSET-PIPELINE-001
Version: 1.0.0
Date: 2026-07-05
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from src.orchestration.dataset_uri import DatasetContractError, validate_dataset_edges

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/airflow")
CONFIG_PATH = PROJECT_ROOT / "config" / "assets" / "pipelines.yaml"

DEFAULT_ARGS = {
    "owner": "asset-pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=45),
}


def _load_config() -> dict:
    """Load the SSOT, degrading only on availability/serialization failures."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        validate_dataset_edges(config.get("dataset_edges") or [])
        return config
    except DatasetContractError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("asset_pipeline_factory: could not read %s: %s", CONFIG_PATH, e)
        return {}


def _run_stage(script: str, args: list[str], stage_name: str, **context) -> None:
    """Run a pipeline stage as a subprocess from the repo root."""
    script_path = PROJECT_ROOT / script
    if not script_path.exists():
        raise FileNotFoundError(f"[{stage_name}] script not found: {script_path}")

    cmd = [sys.executable, str(script_path), *args]
    logger.info("[%s] running: %s", stage_name, " ".join(cmd))
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=40 * 60,
    )
    for line in (result.stdout or "").splitlines()[-80:]:
        logger.info("[%s] %s", stage_name, line)
    for line in (result.stderr or "").splitlines()[-30:]:
        logger.warning("[%s:err] %s", stage_name, line)

    if result.returncode != 0:
        raise RuntimeError(f"[{stage_name}] exited {result.returncode}")
    logger.info("[%s] DONE", stage_name)


def _make_verify(registry_root: str, registry_asset: str, strategy_ids: list[str]):
    """Build the L6 verify callable: registry has the asset + all strategy bundles."""

    def _verify(**context) -> None:
        registry_path = PROJECT_ROOT / registry_root / "registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(f"registry.json not found: {registry_path}")
        with open(registry_path, encoding="utf-8") as f:
            reg = json.load(f)

        asset_ids = {a.get("asset_id") for a in reg.get("assets", [])}
        if registry_asset not in asset_ids:
            raise RuntimeError(f"registry missing asset '{registry_asset}' (have: {sorted(asset_ids)})")

        published = {s.get("strategy_id") for s in reg.get("strategies", [])
                     if s.get("asset_id") == registry_asset}
        missing = [sid for sid in strategy_ids if sid not in published]
        if missing:
            raise RuntimeError(
                f"registry missing strategies for {registry_asset}: {missing} "
                f"(published: {sorted(published)})"
            )

        # Confirm each strategy's manifest bundle is present on disk.
        for sid in strategy_ids:
            manifest = PROJECT_ROOT / registry_root / "strategies" / sid / "manifest.json"
            if not manifest.exists():
                raise RuntimeError(f"bundle manifest missing for {sid}: {manifest}")

        logger.info("[verify] %s OK — %d strategies published + bundles present",
                    registry_asset, len(strategy_ids))

    return _verify



# --- C-010 R3: cadena gobernada de politica (aditiva, fail-closed) -----------
#
# Solo se emite para referencias declaradas en `policy_runs` cuyo
# `migration.status` sea PARITY_GREEN o CUTOVER. `SPEC_ONLY`/`PARITY_PENDING`
# producen CERO tareas -- no un skip verde. Con el arbol actual no hay ninguna
# entrada declarada, asi que el grafo de tareas queda IDENTICO; esa ausencia de
# delta no depende de criterio, sino de que no hay nada que declarar.
#
# Promover un `migration.status` es acto EXCLUSIVO del operador (C-010 amendment):
# este codigo solo lee el estado, nunca lo escribe ni lo infiere.
ELIGIBLE_MIGRATION_STATES = frozenset({"PARITY_GREEN", "CUTOVER"})

#: Ramificacion permitida: por `engine.type`, jamas por `strategy_id`
#: (`strategy-engines.md` invariante 1).
SUPPORTED_ENGINE_TYPES = frozenset({"rule_based"})


class PolicyRunConfigError(RuntimeError):
    """Declaracion de `policy_runs` invalida: falla al parsear el DAG, no en runtime."""


def resolve_policy_runs(spec: dict) -> list[dict]:
    """Resolver `policy_runs` a las referencias ELEGIBLES, o fallar cerrado.

    Devuelve `[]` sin importar nada cuando no hay `policy_runs`, para que el
    grafo actual no cambie ni adquiera dependencias nuevas.
    """
    declared = spec.get("policy_runs") or []
    if not declared:
        return []

    seen: set[str] = set()
    wanted: list[str] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict) or not isinstance(entry.get("policy_id"), str):
            raise PolicyRunConfigError(
                f"policy_runs[{index}] debe ser un mapping con `policy_id` de texto"
            )
        policy_id = entry["policy_id"].strip()
        if not policy_id:
            raise PolicyRunConfigError(f"policy_runs[{index}]: `policy_id` vacio")
        if policy_id in seen:
            raise PolicyRunConfigError(f"policy_runs: `policy_id` duplicado: {policy_id}")
        seen.add(policy_id)
        wanted.append(policy_id)

    from src.strategies.policies.loader import load_all_policy_specs

    by_id = {str(item["id"]): item for item in load_all_policy_specs()}
    unknown = [pid for pid in wanted if pid not in by_id]
    if unknown:
        raise PolicyRunConfigError(
            f"policy_runs referencia policies que el loader SSOT no conoce: {unknown}"
        )

    eligible: list[dict] = []
    for policy_id in wanted:
        policy_spec = by_id[policy_id]
        status = (policy_spec.get("migration") or {}).get("status")
        if status not in ELIGIBLE_MIGRATION_STATES:
            continue  # inerte: cero tareas, nunca un skip verde
        engine_type = (policy_spec.get("engine") or {}).get("type")
        if engine_type not in SUPPORTED_ENGINE_TYPES:
            raise PolicyRunConfigError(
                f"{policy_id}: engine.type {engine_type!r} elegible pero no soportado "
                f"por la cadena gobernada (soportados: {sorted(SUPPORTED_ENGINE_TYPES)})"
            )
        eligible.append({"policy_id": policy_id, "engine_type": engine_type})
    return eligible


def make_resolve_snapshot(policy_id: str):
    """Tarea 1: materializar el snapshot causal. Aqui vive el cutoff."""

    def _resolve(**context):
        from src.orchestration.feature_snapshot import resolve_feature_snapshot

        ti = context["ti"]
        observations = ti.xcom_pull(key=f"observations::{policy_id}")
        decision_cutoff = ti.xcom_pull(key=f"decision_cutoff::{policy_id}")
        if not observations or not decision_cutoff:
            raise PolicyRunConfigError(
                f"{policy_id}: faltan observations/decision_cutoff; no se evalua a ciegas"
            )
        return resolve_feature_snapshot(observations, decision_cutoff=decision_cutoff)

    return _resolve


def make_evaluate_policy(policy_id: str):
    """Tarea 2: evaluar la politica sobre el snapshot ya acotado por cutoff."""

    def _evaluate(**context):
        from src.policy_engine import evaluate_policy
        from src.strategies.policies.loader import build_policy

        ti = context["ti"]
        snapshot = ti.xcom_pull(task_ids=f"policy_{policy_id}_resolve_snapshot")
        if snapshot is None:
            raise PolicyRunConfigError(f"{policy_id}: sin snapshot resuelto; no se evalua")
        return evaluate_policy(build_policy(policy_id), snapshot, context.get("ctx"))

    return _evaluate


def make_publish_signal(policy_id: str):
    """Tarea 3: publicar la decision. Sin decision no se publica nada."""

    def _publish(**context):
        from src.policy_engine import publish_signal

        ti = context["ti"]
        decision = ti.xcom_pull(task_ids=f"policy_{policy_id}_evaluate")
        if decision is None:
            raise PolicyRunConfigError(f"{policy_id}: sin decision; no se publica")
        return publish_signal(decision)

    return _publish


def _build_asset_dag(asset_id: str, spec: dict, registry_root: str) -> DAG:
    """Build a single DS-cycle pipeline DAG for one asset."""
    stages = spec.get("stages") or []
    verify_spec = spec.get("verify") or {}

    dag = DAG(
        dag_id=f"asset_{asset_id}_pipeline_weekly",
        default_args=DEFAULT_ARGS,
        description=f"DS-cycle pipeline for {spec.get('display_name', asset_id)} "
                    f"(ingest -> backtest/publish -> verify)",
        schedule=spec.get("schedule"),
        start_date=days_ago(1),
        catchup=False,
        tags=["asset-pipeline", "multi-asset", asset_id, "weekly",
              "ds-cycle", "ctr-asset-pipeline-001"],
        max_active_runs=1,
    )

    with dag:
        prev = None
        for stage in stages:
            graceful = bool(stage.get("graceful", False))
            task = PythonOperator(
                task_id=stage["id"],
                python_callable=_run_stage,
                op_kwargs={
                    "script": stage["script"],
                    "args": stage.get("args", []),
                    "stage_name": f"{asset_id}:{stage['id']}",
                },
                # If the *previous* stage was graceful, still run this one so a
                # stale-feed ingest failure never blocks the science stage.
                trigger_rule=(TriggerRule.ALL_DONE if (prev is not None and prev.get("graceful"))
                              else TriggerRule.ALL_SUCCESS),
                # A graceful stage (e.g. best-effort ingest refresh) fails fast:
                # retrying it only stalls the downstream science stage, which runs
                # on the last good seed anyway.
                retries=(0 if graceful else DEFAULT_ARGS["retries"]),
            )
            if prev is not None:
                dag.get_task(prev["id"]) >> task
            prev = {"id": stage["id"], "graceful": graceful}

        verify = PythonOperator(
            task_id="l6_verify_registry",
            python_callable=_make_verify(
                registry_root=registry_root,
                registry_asset=verify_spec.get("registry_asset", asset_id),
                strategy_ids=verify_spec.get("strategy_ids", []),
            ),
            # Verify only after the (non-graceful) publish stage actually succeeds.
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )
        if prev is not None:
            dag.get_task(prev["id"]) >> verify

        # C-010 R3: cadena gobernada por `engine.type`, solo para elegibles.
        # Sin `policy_runs` declarados esto es un bucle vacio y el grafo no cambia.
        for run in resolve_policy_runs(spec):
            policy_id = run["policy_id"]
            chain = [
                PythonOperator(
                    task_id=f"policy_{policy_id}_resolve_snapshot",
                    python_callable=make_resolve_snapshot(policy_id),
                ),
                PythonOperator(
                    task_id=f"policy_{policy_id}_evaluate",
                    python_callable=make_evaluate_policy(policy_id),
                ),
                PythonOperator(
                    task_id=f"policy_{policy_id}_publish",
                    python_callable=make_publish_signal(policy_id),
                ),
            ]
            verify >> chain[0] >> chain[1] >> chain[2]

    return dag


# --- Factory: register one DAG per enabled asset in the module globals --------
_config = _load_config()
_registry_root = _config.get("registry_root", "usdcop-trading-dashboard/public/data")

for _asset_id, _spec in (_config.get("assets") or {}).items():
    if not _spec.get("enabled", False):
        continue
    _dag = _build_asset_dag(_asset_id, _spec, _registry_root)
    globals()[f"asset_{_asset_id}_pipeline_weekly"] = _dag
