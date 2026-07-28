"""
DAG: control_system_health — Monitoreo en tres relojes (BL-25, FABRIC §23)
==========================================================================

Un solo motor (src/monitoring/system_health.py), tres latencias:

  DATA  (cada run, minutos) — freshness/paridad por serie; rojo binario,
        fail-closed sobre componentes ACTIVOS (jamás reusar el último valor).
  MODEL (diaria)            — PSI de features vs train + drift de la
        distribución de predicción; PSI > 0.25 => congelar promociones.
  PNL   (semanal, viernes)  — tracking error live-vs-paper (desviación
        acumulada, 3σ) + decay de Sharpe; dispara withdrawal_protocol/REDUCED.

Salidas:
  - Snapshot JSON => usdcop-trading-dashboard/public/data/production/system_health.json
    (semáforos de /production; runtime-written, gitignored como deploy_status.json).
  - Eventos JSONL => data/health/metric_events.jsonl (costura hacia
    control.metric_event de BL-18).
  - El gate de promoción (forecast_h5_l4_backtest_promotion::validate_data)
    lee el snapshot y BLOQUEA si promotions_frozen.

Diferencia con core_watchdog: el watchdog AUTO-SANA staleness operativa
(re-dispara DAGs); este DAG EVALÚA salud y publica acciones automáticas de
gobierno (freeze/withdrawal). No se pisan: corren a :30 vs :00.

Contract: CTR-SYSTEM-HEALTH-001
Version: 1.0.0
Date: 2026-07-27
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

from contracts.dag_registry import CONTROL_SYSTEM_HEALTH, get_dag_tags

logger = logging.getLogger(__name__)

DAG_ID = CONTROL_SYSTEM_HEALTH

PROJECT_ROOT = Path("/opt/airflow")
DASHBOARD_DIR = PROJECT_ROOT / "usdcop-trading-dashboard" / "public"
SNAPSHOT_PATH = DASHBOARD_DIR / "data" / "production" / "system_health.json"
EVENTS_PATH = PROJECT_ROOT / "data" / "health" / "metric_events.jsonl"
DAILY_SEED = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
MACRO_CLEAN = (
    PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
)
BACKTEST_SUMMARY = DASHBOARD_DIR / "data" / "production" / "summary_2025.json"

COT = timezone(timedelta(hours=-5))
#: Cutoff de train del track H5 (metodología: entrenado <= Dec-2024; 2025=OOS).
TRAIN_CUTOFF = "2024-12-31"
#: Ventana "actual" del reloj de modelo (días calendario).
MODEL_CURRENT_WINDOW_DAYS = 90


def _engine():
    from src.monitoring.system_health import JsonlMetricEventSink, SystemHealthEngine

    return SystemHealthEngine(event_sink=JsonlMetricEventSink(EVENTS_PATH))


def _get_db_connection():
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def _is_market_hours() -> bool:
    now = datetime.now(COT)
    return now.weekday() < 5 and 8 <= now.hour < 13


# =============================================================================
# RELOJ DE DATOS — cada run (minutos)
# =============================================================================

def evaluate_data_clock(**context):
    """Probes de freshness. Componentes ACTIVOS (insumos de la estrategia en
    producción) fallan cerrado; superficies diagnósticas degradan elegante."""
    from src.monitoring.system_health_contract import DataProbe

    probes: list[DataProbe] = []

    # -- DB: OHLCV m5 (activo: alimenta executor/monitor intradía) ----------
    try:
        conn = _get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT EXTRACT(EPOCH FROM (NOW() - MAX(time)))/60 "
                "FROM usdcop_m5_ohlcv WHERE symbol = 'USD/COP'"
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                probes.append(DataProbe("db_ohlcv_m5", "missing", True))
            else:
                minutes = float(row[0])
                stale = _is_market_hours() and minutes > 15
                probes.append(
                    DataProbe(
                        "db_ohlcv_m5", "stale" if stale else "ok", True,
                        {"minutes_ago": round(minutes, 1)},
                    )
                )

            # -- DB: macro daily (activo: features T-1 de H5) ---------------
            cur.execute("SELECT (CURRENT_DATE - MAX(fecha)) FROM macro_indicators_daily")
            row = cur.fetchone()
            days = None
            if row and row[0] is not None:
                days = row[0].days if hasattr(row[0], "days") else int(row[0])
            if days is None:
                probes.append(DataProbe("db_macro_daily", "missing", True))
            else:
                probes.append(
                    DataProbe(
                        "db_macro_daily", "stale" if days > 7 else "ok", True,
                        {"days_ago": days},
                    )
                )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — DB caída = probe en error => fail-closed
        logger.error("[SystemHealth] DB probe error: %s", exc)
        probes.append(DataProbe("db_connection", "error", True, {"error": str(exc)}))

    # -- Archivos (activos: insumos de training/inferencia H5) --------------
    for name, path, max_age_days in (
        ("seed_daily_ohlcv", DAILY_SEED, 7),
        ("macro_daily_clean", MACRO_CLEAN, 7),
    ):
        if not path.exists():
            probes.append(DataProbe(name, "missing", True))
        else:
            age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).days
            probes.append(
                DataProbe(name, "stale" if age > max_age_days else "ok", True, {"age_days": age})
            )

    # -- Superficies diagnósticas (degradación elegante, §23.1) -------------
    fc_csv = DASHBOARD_DIR / "forecasting" / "bi_dashboard_unified.csv"
    probes.append(
        DataProbe("forecasting_csv", "ok" if fc_csv.exists() else "missing", False)
    )

    status = _engine().evaluate_data_clock(probes)
    context["ti"].xcom_push(key="data_clock", value=status.to_dict())
    logger.info("[SystemHealth] DATA clock: %s (%d probes)", status.signal.value, len(probes))
    return status.to_dict()


# =============================================================================
# RELOJ DE MODELO — diario (primer run del día)
# =============================================================================

def evaluate_model_clock(**context):
    """PSI de features vs train (<= TRAIN_CUTOFF) + drift de predicción.

    Cadencia diaria: solo evalúa en el primer run del día; los demás runs
    arrastran el estado previo vía build_snapshot(previous_path=...).
    Features monitoreadas (wiring, no modelado — 0 trials): log-return diario
    y vol realizada 20d del seed diario, las mismas primitivas de la
    metodología H5 (train <= Dec-2024).
    """
    import numpy as np
    import pandas as pd

    from src.monitoring.system_health import load_snapshot

    # ¿Ya evaluado hoy? => skip (carry-forward).
    previous = load_snapshot(SNAPSHOT_PATH)
    today = datetime.now(COT).date().isoformat()
    if previous is not None:
        model_prev = previous.clocks.get("model")
        if model_prev and (model_prev.evaluated_at or "").startswith(today):
            logger.info("[SystemHealth] MODEL clock already evaluated today — carry-forward")
            context["ti"].xcom_push(key="model_clock", value=None)
            return None

    if not DAILY_SEED.exists():
        logger.warning("[SystemHealth] Daily seed missing — MODEL clock N/A this run")
        context["ti"].xcom_push(key="model_clock", value=None)
        return None

    df = pd.read_parquet(DAILY_SEED)
    if "time" in df.columns:
        df = df.set_index("time")
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.sort_index()

    close = df["close"]
    ret = np.log(close / close.shift(1)).dropna()
    vol20 = ret.rolling(20).std().dropna()

    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    current_start = pd.Timestamp.now() - pd.Timedelta(days=MODEL_CURRENT_WINDOW_DAYS)

    reference = {
        "log_return_1d": ret[ret.index <= cutoff].to_numpy(),
        "realized_vol_20d": vol20[vol20.index <= cutoff].to_numpy(),
    }
    current = {
        "log_return_1d": ret[ret.index >= current_start].to_numpy(),
        "realized_vol_20d": vol20[vol20.index >= current_start].to_numpy(),
    }

    # Drift de la distribución de predicción (forecast_h5_predictions).
    pred_ref = pred_cur = None
    try:
        conn = _get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT predicted_return_pct FROM forecast_h5_predictions "
                "WHERE inference_date < CURRENT_DATE - INTERVAL '90 days'"
            )
            pred_ref = np.array([r[0] for r in cur.fetchall()], dtype=float)
            cur.execute(
                "SELECT predicted_return_pct FROM forecast_h5_predictions "
                "WHERE inference_date >= CURRENT_DATE - INTERVAL '90 days'"
            )
            pred_cur = np.array([r[0] for r in cur.fetchall()], dtype=float)
            if len(pred_ref) < 10 or len(pred_cur) < 1:
                pred_ref = pred_cur = None
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — sin predicciones => solo PSI de features
        logger.warning("[SystemHealth] prediction-drift source unavailable: %s", exc)

    status = _engine().evaluate_model_clock(
        reference_features=reference,
        current_features=current,
        predictions_reference=pred_ref,
        predictions_current=pred_cur,
    )
    context["ti"].xcom_push(key="model_clock", value=status.to_dict())
    logger.info(
        "[SystemHealth] MODEL clock: %s (psi_max=%s)",
        status.signal.value, status.metrics.get("psi_max"),
    )
    if status.signal.value == "yellow":
        logger.warning("[SystemHealth] MODEL DRIFT — promociones CONGELADAS")
    return status.to_dict()


# =============================================================================
# RELOJ DE PnL — semanal (viernes)
# =============================================================================

def evaluate_pnl_clock(**context):
    """TE live-vs-paper (3σ acumulada) + decay de Sharpe vs backtest.

    Cadencia semanal: evalúa solo los viernes (tras el cierre H5 12:50 COT).
    live  = forecast_h5_executions.week_pnl_pct (join por signal_date)
    paper = forecast_h5_paper_trading.week_pnl_pct
    Sin filas apareadas => N/A honesto (jamás imputar, §23.1).
    """
    import numpy as np

    if datetime.now(COT).weekday() != 4:  # viernes
        logger.info("[SystemHealth] PNL clock only runs Fridays — carry-forward")
        context["ti"].xcom_push(key="pnl_clock", value=None)
        return None

    live = paper = None
    rolling_sharpe = None
    try:
        conn = _get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT e.week_pnl_pct, p.week_pnl_pct
                FROM forecast_h5_executions e
                JOIN forecast_h5_paper_trading p USING (signal_date)
                WHERE e.week_pnl_pct IS NOT NULL AND e.status = 'closed'
                ORDER BY e.signal_date
                """
            )
            rows = cur.fetchall()
            if rows:
                live = np.array([r[0] for r in rows], dtype=float) / 100.0
                paper = np.array([r[1] for r in rows], dtype=float) / 100.0
            cur.execute(
                "SELECT running_sharpe FROM forecast_h5_paper_trading "
                "ORDER BY signal_date DESC LIMIT 1"
            )
            row = cur.fetchone()
            rolling_sharpe = float(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — fuente ausente => N/A
        logger.warning("[SystemHealth] PnL sources unavailable: %s", exc)

    backtest_sharpe = None
    try:
        if BACKTEST_SUMMARY.exists():
            summary = json.loads(BACKTEST_SUMMARY.read_text(encoding="utf-8"))
            sid = summary.get("strategy_id")
            backtest_sharpe = (summary.get("strategies", {}).get(sid, {}) or {}).get("sharpe")
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[SystemHealth] backtest summary unreadable: %s", exc)

    status = _engine().evaluate_pnl_clock(
        live_returns=live,
        paper_returns=paper,
        rolling_sharpe=rolling_sharpe,
        backtest_sharpe=backtest_sharpe,
        # Slippage realizado vs modelado: fuente aún no persistida (BL-22
        # fact_pnl.cost_slippage) => N/A honesto, no un número inventado.
        realized_slippage_bps=None,
        modeled_slippage_bps=None,
    )
    context["ti"].xcom_push(key="pnl_clock", value=status.to_dict())
    logger.info("[SystemHealth] PNL clock: %s", status.signal.value)
    if status.metrics.get("tracking_error_zscore") is not None and status.signal.value == "orange":
        logger.warning("[SystemHealth] TE>3σ — evento withdrawal_protocol_triggered emitido")
    return status.to_dict()


# =============================================================================
# PUBLISH — consolida y escribe el snapshot (semáforos de /production)
# =============================================================================

def publish_snapshot(**context):
    from src.monitoring.system_health import write_snapshot
    from src.monitoring.system_health_contract import (
        Clock,
        ClockStatus,
        HealthAction,
        HealthEvent,
        HealthSignal,
    )

    def _revive(d):
        if not d:
            return None
        return ClockStatus(
            clock=Clock(d["clock"]),
            signal=HealthSignal(d["signal"]),
            actions=[HealthAction(a) for a in d.get("actions", [])],
            events=[
                HealthEvent(
                    kind=e["kind"], clock=Clock(e["clock"]), signal=HealthSignal(e["signal"]),
                    action=HealthAction(e["action"]), message=e.get("message", ""),
                    metric_value=e.get("metric_value"), threshold=e.get("threshold"),
                    timestamp=e.get("timestamp"),
                )
                for e in d.get("events", [])
            ],
            metrics=d.get("metrics", {}),
            evaluated_at=d.get("evaluated_at"),
        )

    ti = context["ti"]
    data = _revive(ti.xcom_pull(key="data_clock", task_ids="evaluate_data_clock"))
    model = _revive(ti.xcom_pull(key="model_clock", task_ids="evaluate_model_clock"))
    pnl = _revive(ti.xcom_pull(key="pnl_clock", task_ids="evaluate_pnl_clock"))

    snapshot = _engine().build_snapshot(
        data=data, model=model, pnl=pnl, previous_path=SNAPSHOT_PATH
    )
    write_snapshot(snapshot, SNAPSHOT_PATH)

    logger.info("=" * 60)
    logger.info("[SystemHealth] SNAPSHOT %s", snapshot.generated_at)
    for name, clock in snapshot.clocks.items():
        logger.info("  %s: %s %s", name.upper(), clock.signal.value,
                    [a.value for a in clock.actions] or "")
    logger.info("  promotions_frozen=%s withdrawal_triggered=%s",
                snapshot.promotions_frozen, snapshot.withdrawal_triggered)
    logger.info("=" * 60)
    return snapshot.to_dict()


# =============================================================================
# DAG
# =============================================================================

default_args = {
    "owner": "system-watchdog",
    "depends_on_past": False,
    "start_date": datetime(2026, 7, 27),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="BL-25 — tres relojes (datos/modelo/PnL): semáforos + freeze/withdrawal",
    schedule_interval="30 13-18 * * 1-5",  # :30 cada hora 8:30-13:30 COT (no pisa watchdog :00)
    catchup=False,
    max_active_runs=1,
    # Monitoreo de salud: debe sobrevivir un cold boot ACTIVO (como el deploy
    # L4b) — un reloj pausado no congela nada.
    is_paused_upon_creation=False,
    tags=get_dag_tags(DAG_ID),
) as dag:

    t_data = PythonOperator(task_id="evaluate_data_clock", python_callable=evaluate_data_clock)
    t_model = PythonOperator(task_id="evaluate_model_clock", python_callable=evaluate_model_clock)
    t_pnl = PythonOperator(task_id="evaluate_pnl_clock", python_callable=evaluate_pnl_clock)
    t_publish = PythonOperator(task_id="publish_snapshot", python_callable=publish_snapshot)

    [t_data, t_model, t_pnl] >> t_publish
