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
    """Load the per-asset pipeline SSOT. Never raise at DAG-parse time."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
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

    return dag


# --- Factory: register one DAG per enabled asset in the module globals --------
_config = _load_config()
_registry_root = _config.get("registry_root", "usdcop-trading-dashboard/public/data")

for _asset_id, _spec in (_config.get("assets") or {}).items():
    if not _spec.get("enabled", False):
        continue
    _dag = _build_asset_dag(_asset_id, _spec, _registry_root)
    globals()[f"asset_{_asset_id}_pipeline_weekly"] = _dag
