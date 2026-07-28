"""
H5-L4b — Production Deploy (post-Vote-2)
========================================
Container-native deploy path for the two-vote promotion system (closes QA ledger #40:
the dashboard's in-process `spawn('python3')` cannot run inside the node container).

Flow:
  guard_approved   — HARD GATE: approval_state.json must be status=APPROVED (Vote 2/2 cast).
  run_production   — execute the deploy_manifest command (default:
                     train_and_export_smart_simple.py --phase production --no-png --seed-db)
  validate_output  — summary.json must exist and be freshly written.

Progress is mirrored into deploy_status.json so the dashboard deploy panel tracks the
DAG-driven deploy exactly like the legacy in-process one.

Trigger: dashboard deploy API via Airflow REST (basic auth), or manual:
  airflow dags trigger forecast_h5_l4b_production_deploy
Schedule: None (event-driven only — safe to leave unpaused).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import json
import logging
import os
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L4B_PRODUCTION_DEPLOY,
    get_dag_tags,
)

DAG_ID = FORECAST_H5_L4B_PRODUCTION_DEPLOY
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')
DASHBOARD_DATA_DIR = PROJECT_ROOT / 'usdcop-trading-dashboard' / 'public' / 'data' / 'production'
# CXD-057: el approval_state salió de `public/` (gates + DSR + backtest_metrics son
# research:read; bajo `public/` el estático /data/** los servía a cualquier sesión).
# SSOT de la ruta: `src/contracts/approval_store.py` (`./data` está montado en el
# contenedor de Airflow como /opt/airflow/data, ver docker-compose).
APPROVALS_DIR = Path(os.getenv('APPROVALS_DATA_DIR') or (PROJECT_ROOT / 'data' / 'approvals'))
APPROVAL_FILE = APPROVALS_DIR / 'approval_state.json'
DEPLOY_STATUS_FILE = DASHBOARD_DATA_DIR / 'deploy_status.json'
# El comando por defecto (fallback sin manifiesto) y la allowlist de script/args viven en
# `src/contracts/deploy_manifest.py` (CTR-DEPLOY-CMD-001), espejo de
# `usdcop-trading-dashboard/lib/security/deploy-command.ts`. No se duplican aquí.


def _write_deploy_status(patch: Dict[str, Any]) -> None:
    """Best-effort merge-write of deploy_status.json (dashboard panel mirror)."""
    try:
        current: Dict[str, Any] = {}
        if DEPLOY_STATUS_FILE.exists():
            current = json.loads(DEPLOY_STATUS_FILE.read_text(encoding='utf-8'))
        current.update(patch)
        current['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        DEPLOY_STATUS_FILE.write_text(
            json.dumps(current, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception as e:  # noqa: BLE001 — mirror only, never fail the deploy for it
        logger.warning(f"[H5-L4b] deploy_status mirror failed (non-blocking): {e}")


def _approval_file(context) -> Path:
    """Multi-strategy: dag_run.conf.strategy_id selects approval_state_<sid>.json
    (e.g. btc_trend_b2 paper); default = the COP singleton.

    The singleton fallback is load-bearing and must match the dashboard's approve/deploy
    routes: the export pipeline writes the ACTIVE strategy to the unsuffixed
    `approval_state.json`, while the dashboard always sends a `strategy_id`. Without it,
    `smart_simple_v11` resolved to a file that is never written and this hard gate failed
    on a bundle that WAS approved. Falls back only when the singleton IS that strategy, so
    a stale id can never deploy somebody else's bundle.
    """
    sid = ((context.get('dag_run') and context['dag_run'].conf) or {}).get('strategy_id')
    if not (sid and str(sid).replace('_', '').replace('-', '').isalnum()):
        return APPROVAL_FILE
    scoped = APPROVALS_DIR / f"approval_state_{sid}.json"
    if scoped.is_file():
        return scoped
    try:
        singleton = json.loads(APPROVAL_FILE.read_text(encoding='utf-8'))
        if singleton.get('strategy') == sid:
            return APPROVAL_FILE
    except Exception:  # noqa: BLE001 — unreadable singleton: keep the scoped path so the gate fails honestly
        pass
    return scoped


def guard_approved(**context) -> Dict[str, Any]:
    """HARD GATE: refuse to deploy unless the human Vote 2/2 is APPROVED.

    BL-25 (FABRIC §23): además revalida que el reloj de MODELO no haya
    congelado las promociones DESPUÉS del Vote 2 — el deploy re-valida
    server-side, la UI no es la autoridad (approval-gates invariante 4).
    """
    from src.monitoring.system_health import (
        PromotionsFrozenError,
        assert_promotions_not_frozen,
    )

    try:
        assert_promotions_not_frozen(DASHBOARD_DATA_DIR / 'system_health.json')
    except PromotionsFrozenError as exc:
        raise ValueError(f"[H5-L4b] Deploy REFUSED — BL-25 health gate: {exc}") from exc

    approval_file = _approval_file(context)
    if not approval_file.exists():
        raise ValueError(f"[H5-L4b] approval file not found: {approval_file}")
    state = json.loads(approval_file.read_text(encoding='utf-8'))
    status = state.get('status')
    if status != 'APPROVED':
        raise ValueError(
            f"[H5-L4b] Deploy REFUSED — approval status is '{status}', requires APPROVED "
            f"(Vote 2/2 on /dashboard first)."
        )
    # GATE DE EJECUCIÓN (CTR-DEPLOY-CMD-001): el manifiesto NO elige qué programa corre.
    # Antes, `run_production` unía PROJECT_ROOT con el script del plan, y pathlib DESCARTA
    # la raíz si el operando derecho es absoluto (`Path('/opt/airflow') / '/etc/x.py'` →
    # `/etc/x.py`): un manifiesto manipulado ejecutaba cualquier fichero del contenedor.
    # Ahora `script`/`args` pasan por la misma allowlist que el dashboard
    # (`lib/security/deploy-command.ts` ↔ `src/contracts/deploy_manifest.py`), y el
    # rechazo ocurre ANTES de tocar nada: fail-closed.
    from src.contracts.deploy_manifest import (
        DeployManifestRejected,
        describe_rejection,
        resolve_deploy_command,
    )

    manifest = state.get('deploy_manifest') or {}
    try:
        command = resolve_deploy_command(PROJECT_ROOT, manifest or None)
    except DeployManifestRejected as exc:
        logger.error(f"[H5-L4b] SECURITY — manifiesto RECHAZADO: {describe_rejection(exc, manifest)}")
        _write_deploy_status({
            'status': 'failed', 'phase': None, 'runner': 'airflow', 'dag_id': DAG_ID,
            'error': f"Deploy manifest rejected ({exc.field}): {exc.reason}",
            'completed_at': datetime.utcnow().isoformat() + 'Z',
        })
        raise ValueError(f"[H5-L4b] Deploy REFUSED — {exc}") from exc

    result = {
        'strategy': state.get('strategy'),
        'is_default': approval_file == APPROVAL_FILE,
        'approved_by': state.get('approved_by'),
        'approved_at': state.get('approved_at'),
        'script': command.script_rel,
        'args': list(command.args),
    }
    logger.info(f"[H5-L4b] APPROVED by {result['approved_by']} at {result['approved_at']} — "
                f"deploying {result['strategy']} via {result['script']} {' '.join(result['args'])}")
    _write_deploy_status({
        'status': 'running', 'phase': 'retraining', 'runner': 'airflow',
        'dag_id': DAG_ID, 'run_id': context['run_id'],
        'strategy': result['strategy'],
        'started_at': datetime.utcnow().isoformat() + 'Z',
        'error': None, 'completed_at': None,
    })
    context['ti'].xcom_push(key='deploy_plan', value=result)
    return result


def run_production(**context) -> Dict[str, Any]:
    """Execute the manifest deploy command (retrain full window + export + seed DB)."""
    from src.contracts.deploy_manifest import (
        DeployManifestRejected,
        describe_rejection,
        resolve_deploy_command,
    )

    plan = context['ti'].xcom_pull(key='deploy_plan', task_ids='guard_approved')
    # Se RE-RESUELVE aquí (no se confía en la XCom, que es estado mutable entre tareas):
    # misma allowlist, mismo fail-closed. El argv es explícito y jamás pasa por un shell.
    proposed = {'script': plan.get('script'), 'args': plan.get('args') or []}
    try:
        command = resolve_deploy_command(PROJECT_ROOT, proposed, interpreter=sys.executable)
    except DeployManifestRejected as exc:
        logger.error(f"[H5-L4b] SECURITY — manifiesto RECHAZADO en ejecución: "
                     f"{describe_rejection(exc, proposed)}")
        _write_deploy_status({
            'status': 'failed',
            'error': f"Deploy manifest rejected ({exc.field}): {exc.reason}",
            'completed_at': datetime.utcnow().isoformat() + 'Z',
        })
        raise ValueError(f"[H5-L4b] Deploy REFUSED — {exc}") from exc

    cmd = command.argv()
    logger.info(f"[H5-L4b] Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1500, cwd=str(PROJECT_ROOT))

    if result.stdout:
        for line in result.stdout.strip().split('\n')[-25:]:
            logger.info(f"[H5-L4b] stdout: {line}")
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-10:]:
            logger.warning(f"[H5-L4b] stderr: {line}")

    if result.returncode != 0:
        _write_deploy_status({
            'status': 'failed', 'error': (result.stderr or '')[-500:] or f"exit {result.returncode}",
            'completed_at': datetime.utcnow().isoformat() + 'Z',
        })
        raise RuntimeError(f"[H5-L4b] Deploy script failed (exit {result.returncode})")

    _write_deploy_status({'phase': 'exporting'})
    return {'exit_code': result.returncode}


def validate_output(**context) -> Dict[str, Any]:
    """summary.json must exist, parse, and be freshly written by this run."""
    summary_path = DASHBOARD_DATA_DIR / 'summary.json'
    if not summary_path.exists():
        _write_deploy_status({'status': 'failed', 'error': 'summary.json missing after deploy',
                              'completed_at': datetime.utcnow().isoformat() + 'Z'})
        raise ValueError(f"[H5-L4b] summary.json missing: {summary_path}")
    doc = json.loads(summary_path.read_text(encoding='utf-8'))
    age_min = (datetime.now().timestamp() - summary_path.stat().st_mtime) / 60
    if age_min > 60:
        _write_deploy_status({'status': 'failed',
                              'error': f'summary.json stale ({age_min:.0f} min old) — deploy did not export',
                              'completed_at': datetime.utcnow().isoformat() + 'Z'})
        raise ValueError(f"[H5-L4b] summary.json is {age_min:.0f} min old — not written by this deploy")
    _write_deploy_status({'status': 'completed', 'phase': 'done',
                          'completed_at': datetime.utcnow().isoformat() + 'Z'})
    logger.info(f"[H5-L4b] Deploy VALIDATED — strategy={doc.get('strategy_name')}, "
                f"summary.json {age_min:.1f} min old")
    return {'strategy_name': doc.get('strategy_name'), 'summary_age_min': round(age_min, 1)}


default_args = {
    'owner': 'forecast-h5-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 7, 1),
    'retries': 0,  # a deploy is not idempotent-cheap; the operator re-triggers explicitly
    'execution_timeout': timedelta(minutes=30),
}

with DAG(
    DAG_ID,
    default_args=default_args,
    description='H5 Smart Simple — production deploy after human Vote 2/2 (dashboard-triggered)',
    schedule_interval=None,  # event-driven only (REST trigger from the deploy API)
    catchup=False,
    max_active_runs=1,
    # Event-driven only (schedule=None) — MUST survive cold boot unpaused, else the
    # dashboard-triggered deploy queues forever (found live 2026-07-07: fresh Airflow
    # metadata re-pauses everything and the operator's Vote-2 deploy sat in 'queued').
    is_paused_upon_creation=False,
    tags=DAG_TAGS_LIST,
) as dag:

    def _register_bundle(**context):
        """DAG EXIT gate (CTR-STRAT-REGISTRY-001 §6.2, audit A4-02): refresh the
        registry from the production artifacts this deploy just wrote."""
        from utils.register_bundle import register_strategy_bundle
        return register_strategy_bundle()

    t_guard = PythonOperator(task_id='guard_approved', python_callable=guard_approved)
    t_run = PythonOperator(task_id='run_production', python_callable=run_production)
    t_validate = PythonOperator(task_id='validate_output', python_callable=validate_output)
    t_register = PythonOperator(task_id='register_bundle', python_callable=_register_bundle)

    t_guard >> t_run >> t_validate >> t_register
