"""
DAG: forecast_h5_l5_vol_targeting
====================================
Track B — Monday vol-targeting + adaptive stops + confidence multiplier for H=5 signal.

Schedule: Monday 13:45 UTC = 08:45 COT (15 min after Track A L5c, 30 min after H5-L5b)
Input: forecast_h5_signals row from L5b (with confidence scoring)
Output: Updated forecast_h5_signals with leverage + adaptive stop levels
Downstream: forecast_h5_l7_multiday_executor reads adjusted_leverage, hard_stop_pct, take_profit_pct

Contract: FC-H5-L5-002
Version: 2.0.0 (Smart Simple v1.0 — adaptive stops + confidence multiplier)
Date: 2026-02-16
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L5_VOL_TARGETING,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H5_L5_VOL_TARGETING
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')

# Asymmetric sizing defaults (from smart_executor_h5_v1.yaml)
LONG_MULTIPLIER = 0.5
SHORT_MULTIPLIER = 1.0


# =============================================================================
# TASK 1: LOAD SIGNAL FROM DB
# =============================================================================

def load_signal(**context) -> Dict[str, Any]:
    """Load today's H5 signal from forecast_h5_signals."""
    import pandas as pd

    signal_date = pd.Timestamp(context['ds']).date()

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, signal_date, direction, ensemble_return,
                   sizing_multiplier, skip_trade, confidence_tier
            FROM forecast_h5_signals
            WHERE signal_date = %s
        """, (signal_date,))
        row = cur.fetchone()

        if not row:
            raise ValueError(f"[H5-L5c] No signal found for {signal_date}")

        result = {
            "signal_id": row[0],
            "signal_date": str(row[1]),
            "direction": row[2],
            "ensemble_return": float(row[3]),
            "sizing_multiplier": float(row[4]) if row[4] is not None else 1.0,
            "skip_trade": bool(row[5]) if row[5] is not None else False,
            "confidence_tier": row[6],
        }
        logger.info(
            f"[H5-L5c] Loaded signal: direction={row[2]}, "
            f"ensemble_return={row[3]:+.6f}, "
            f"confidence={row[6]}, sizing_mult={result['sizing_multiplier']:.2f}, "
            f"skip={result['skip_trade']}"
        )
        context['ti'].xcom_push(key='signal', value=result)
        return result
    finally:
        conn.close()


# =============================================================================
# TASK 2: COMPUTE REALIZED VOL
# =============================================================================

def compute_vol(**context) -> Dict[str, Any]:
    """Compute 21-day annualized realized vol from daily returns."""
    import numpy as np
    import pandas as pd

    ohlcv_path = PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_daily_ohlcv.parquet'
    df = pd.read_parquet(ohlcv_path)
    df = df.reset_index()
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date")

    returns = df["close"].pct_change().dropna().values

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.vol_targeting import compute_realized_vol

    realized_vol = compute_realized_vol(returns, lookback=21, annualization=252.0)

    logger.info(f"[H5-L5c] Realized vol (21d): {realized_vol:.4f}")

    result = {"realized_vol_21d": realized_vol}
    context['ti'].xcom_push(key='vol', value=result)
    return result


# =============================================================================
# TASK 3: COMPUTE LEVERAGE + ASYMMETRIC SIZING
# =============================================================================

def compute_leverage(**context) -> Dict[str, Any]:
    """
    Smart Simple v1.0:
    1. Vol-target base leverage
    2. Asymmetric direction scaling (SHORT full, LONG half)
    3. Confidence multiplier (HIGH=2x, MEDIUM=1.5x, etc.)
    4. Adaptive stops from realized vol
    """
    import numpy as np
    import yaml

    ti = context['ti']
    signal = ti.xcom_pull(key='signal', task_ids='load_signal')
    vol_data = ti.xcom_pull(key='vol', task_ids='compute_vol')

    if not signal or not vol_data:
        raise ValueError("[H5-L5c] Missing upstream data")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.vol_targeting import (
        VolTargetConfig,
        compute_vol_target_signal,
        apply_asymmetric_sizing,
    )
    from src.forecasting.adaptive_stops import (
        AdaptiveStopsConfig,
        compute_adaptive_stops,
    )

    # Load Smart Simple config for adaptive stops params
    conf_path = PROJECT_ROOT / 'config' / 'execution' / 'smart_simple_v1.yaml'
    with open(conf_path) as f:
        ss_config = yaml.safe_load(f)

    vt_cfg = ss_config.get("vol_targeting", {})
    config = VolTargetConfig(
        target_vol=vt_cfg.get("target_vol", 0.15),
        max_leverage=vt_cfg.get("max_leverage", 2.0),
        min_leverage=vt_cfg.get("min_leverage", 0.5),
        vol_lookback=vt_cfg.get("vol_lookback", 21),
    )

    vt_signal = compute_vol_target_signal(
        forecast_direction=signal["direction"],
        forecast_return=signal["ensemble_return"],
        realized_vol_21d=vol_data["realized_vol_21d"],
        config=config,
    )

    asymmetric_lev = apply_asymmetric_sizing(
        leverage=vt_signal.clipped_leverage,
        direction=signal["direction"],
        long_mult=LONG_MULTIPLIER,
        short_mult=SHORT_MULTIPLIER,
    )

    # Apply confidence multiplier
    sizing_mult = signal.get("sizing_multiplier", 1.0)
    adjusted_lev = asymmetric_lev * sizing_mult
    adjusted_lev = max(config.min_leverage, min(adjusted_lev, config.max_leverage))

    # Compute adaptive stops
    as_cfg = ss_config.get("adaptive_stops", {})
    stops_config = AdaptiveStopsConfig(
        vol_multiplier=as_cfg.get("vol_multiplier", 1.5),
        hard_stop_min_pct=as_cfg.get("hard_stop_min_pct", 0.01),
        hard_stop_max_pct=as_cfg.get("hard_stop_max_pct", 0.03),
        tp_ratio=as_cfg.get("tp_ratio", 0.5),
    )
    stops = compute_adaptive_stops(
        realized_vol_annualized=vol_data["realized_vol_21d"],
        config=stops_config,
    )

    dir_str = "LONG" if signal["direction"] == 1 else "SHORT"
    mult = LONG_MULTIPLIER if signal["direction"] == 1 else SHORT_MULTIPLIER
    logger.info(
        f"[H5-L5c] {dir_str}: raw_lev={vt_signal.raw_leverage:.3f}, "
        f"clipped={vt_signal.clipped_leverage:.3f}, "
        f"x{mult} -> asymmetric={asymmetric_lev:.3f}, "
        f"x{sizing_mult:.2f} confidence -> adjusted={adjusted_lev:.3f}"
    )
    logger.info(
        f"[H5-L5c] ADAPTIVE STOPS: vol_daily={stops.realized_vol_daily:.4f}, "
        f"vol_weekly={stops.realized_vol_weekly:.4f}, "
        f"HS={stops.hard_stop_pct:.4f} ({stops.hard_stop_pct*100:.2f}%), "
        f"TP={stops.take_profit_pct:.4f} ({stops.take_profit_pct*100:.2f}%)"
    )

    result = {
        "realized_vol_21d": vol_data["realized_vol_21d"],
        "raw_leverage": vt_signal.raw_leverage,
        "clipped_leverage": vt_signal.clipped_leverage,
        "asymmetric_leverage": asymmetric_lev,
        "long_multiplier": LONG_MULTIPLIER,
        "short_multiplier": SHORT_MULTIPLIER,
        "sizing_multiplier": sizing_mult,
        "adjusted_leverage": adjusted_lev,
        "hard_stop_pct": stops.hard_stop_pct,
        "take_profit_pct": stops.take_profit_pct,
    }
    context['ti'].xcom_push(key='leverage', value=result)
    return result


# =============================================================================
# TASK 4: PERSIST LEVERAGE TO DB
# =============================================================================

def persist_leverage(**context) -> Dict[str, Any]:
    """Update forecast_h5_signals with vol-targeting and asymmetric leverage."""
    ti = context['ti']
    signal = ti.xcom_pull(key='signal', task_ids='load_signal')
    leverage = ti.xcom_pull(key='leverage', task_ids='compute_leverage')

    if not signal or not leverage:
        return {"persisted": False}

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE forecast_h5_signals
            SET realized_vol_21d = %s,
                raw_leverage = %s,
                clipped_leverage = %s,
                asymmetric_leverage = %s,
                long_multiplier = %s,
                short_multiplier = %s,
                adjusted_leverage = %s,
                hard_stop_pct = %s,
                take_profit_pct = %s
            WHERE id = %s
        """, (
            leverage["realized_vol_21d"],
            leverage["raw_leverage"],
            leverage["clipped_leverage"],
            leverage["asymmetric_leverage"],
            leverage["long_multiplier"],
            leverage["short_multiplier"],
            leverage["adjusted_leverage"],
            leverage["hard_stop_pct"],
            leverage["take_profit_pct"],
            signal["signal_id"],
        ))
        conn.commit()
        logger.info(
            f"[H5-L5c] Updated signal {signal['signal_id']} with leverage + stops: "
            f"adj_lev={leverage['adjusted_leverage']:.3f}, "
            f"HS={leverage['hard_stop_pct']*100:.2f}%, "
            f"TP={leverage['take_profit_pct']*100:.2f}%"
        )
        return {"persisted": True}
    finally:
        conn.close()


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-h5-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 16),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Track B: Monday H=5 vol-targeting + adaptive stops + confidence (08:45 COT)',
    schedule_interval='45 13 * * 1',  # Lun 13:45 UTC = 08:45 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_load = PythonOperator(
        task_id='load_signal',
        python_callable=load_signal,
    )

    t_vol = PythonOperator(
        task_id='compute_vol',
        python_callable=compute_vol,
    )

    t_leverage = PythonOperator(
        task_id='compute_leverage',
        python_callable=compute_leverage,
    )

    t_persist = PythonOperator(
        task_id='persist_leverage',
        python_callable=persist_leverage,
    )

    [t_load, t_vol] >> t_leverage >> t_persist
