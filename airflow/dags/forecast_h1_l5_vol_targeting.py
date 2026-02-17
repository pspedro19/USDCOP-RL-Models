"""
DAG: forecast_h1_l5_vol_targeting
=================================
USD/COP Trading System - Daily Vol-Targeting Signal Generation

Generates daily vol-targeting signals from the forecasting pipeline.
Reads the latest H=1 forecast from bi.fact_forecasts, computes realized
volatility from daily OHLCV, and produces a position-sized signal.

Architecture:
    bi.fact_forecasts (H=1)          bi.dim_daily_usdcop (last 22 days)
              |                                    |
              v                                    v
      load_latest_forecast()            load_recent_prices()
              |                                    |
              +------------------+-----------------+
                                 v
                     compute_vol_target_signal()
                                 |
                                 v
                        persist_signal()
                                 |
                                 v
                       signal_summary()

Temporal Offset:
    The target variable is log(close[T+1] / close[T]).
    A signal generated on day T predicts TOMORROW's return.

    Day T (today):
      12:55 COT -> Market closes, close[T] available
      13:30 COT -> L5c runs: generates signal for T->T+1

    Day T+1 (tomorrow):
      12:55 COT -> Market closes, close[T+1] available
      19:00 COT -> L6 runs: evaluates signal(T) vs actual return

Schedule: Daily Mon-Fri at 18:30 UTC (13:30 COT), after market close (12:55 COT)
Output: 1 row in forecast_vol_targeting_signals per trading day
Depends on: forecast_h1_l3_weekly_training (L3, runs Sundays)

Author: Trading Team
Version: 1.0.0
Date: 2026-02-15
Contract: FC-SIZE-001
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import math
import sys

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

from contracts.dag_registry import (
    FORECAST_H1_L5_VOL_TARGETING,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H1_L5_VOL_TARGETING
DAG_TAGS = get_dag_tags(DAG_ID)

# Vol-targeting module
try:
    from src.forecasting.vol_targeting import (
        VolTargetConfig,
        VolTargetSignal,
        compute_vol_target_signal,
        compute_realized_vol,
    )
    VOL_TARGETING_AVAILABLE = True
except ImportError as e:
    VOL_TARGETING_AVAILABLE = False
    logging.error(f"[L5c] vol_targeting module not available: {e}")

logger = logging.getLogger(__name__)


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "target_vol": 0.15,
    "max_leverage": 2.0,
    "min_leverage": 0.5,
    "vol_lookback": 21,
    "vol_floor": 0.05,
    "annualization_factor": 252.0,
    "config_version": "vol_target_v1",
    "ensemble_strategy": "top_3",
    "n_top_models": 3,
    "max_forecast_age_days": 14,
    "warn_forecast_age_days": 7,
}

# Colombia holidays 2026 (extend as needed)
COLOMBIA_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-12", "2026-03-23", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-18", "2026-06-08", "2026-06-15", "2026-06-29",
    "2026-07-20", "2026-08-07", "2026-08-17", "2026-10-12", "2026-11-02",
    "2026-11-16", "2026-12-08", "2026-12-25",
}


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def get_vol_target_config(**context) -> Dict[str, Any]:
    """Load vol-targeting config: DEFAULT -> Variable -> dag_run.conf."""
    config = DEFAULT_CONFIG.copy()

    # Override from Airflow Variable
    try:
        var_config = Variable.get("forecast_vol_targeting_config", default_var=None)
        if var_config:
            config.update(json.loads(var_config))
    except Exception as e:
        logger.warning(f"[L5c] Could not load Variable: {e}")

    # Override from dag_run.conf
    if context.get('dag_run') and context['dag_run'].conf:
        config.update(context['dag_run'].conf)

    return config


# =============================================================================
# TASK 1: CHECK MARKET DAY
# =============================================================================

def check_market_day(**context) -> bool:
    """
    Check if today is a trading day. Skip holidays and weekends.
    Override with dag_run.conf: {"force_run": true}
    Returns True to continue, False to short-circuit.
    """
    conf = context.get('dag_run', None)
    if conf and conf.conf and conf.conf.get('force_run', False):
        logger.info("[L5c] Force run enabled, skipping market day check")
        return True

    today = datetime.utcnow().date()
    today_str = today.strftime("%Y-%m-%d")

    # Weekend check (Saturday=5, Sunday=6)
    if today.weekday() >= 5:
        logger.info(f"[L5c] {today_str} is a weekend, skipping")
        return False

    # Holiday check
    if today_str in COLOMBIA_HOLIDAYS_2026:
        logger.info(f"[L5c] {today_str} is a Colombia holiday, skipping")
        return False

    logger.info(f"[L5c] {today_str} is a trading day, proceeding")
    return True


# =============================================================================
# TASK 2: LOAD LATEST FORECAST
# =============================================================================

def load_latest_forecast(**context) -> Dict[str, Any]:
    """
    Load latest H=1 forecast from bi.fact_forecasts.
    Gets latest predictions from up to 9 models, selects top-3 by historical DA.
    Forecasts up to 14 days old are valid (handles L5b failure).
    """
    config = get_vol_target_config(**context)
    today = datetime.utcnow().date()
    today_str = today.strftime("%Y-%m-%d")
    max_age = config["max_forecast_age_days"]
    warn_age = config["warn_forecast_age_days"]
    n_top = config["n_top_models"]

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Get the latest inference_date (may be days ago if L5b runs weekly)
        cur.execute("""
            SELECT DISTINCT inference_date
            FROM bi.fact_forecasts
            WHERE horizon_id = 1
              AND inference_date <= %s
            ORDER BY inference_date DESC
            LIMIT 1
        """, (today_str,))

        row = cur.fetchone()
        if row is None:
            raise ValueError("[L5c] No H=1 forecasts found in bi.fact_forecasts")

        latest_date = row[0]
        forecast_age = (today - latest_date).days
        logger.info(f"[L5c] Latest forecast date: {latest_date} (age: {forecast_age} days)")

        if forecast_age > max_age:
            logger.error(
                f"[L5c] Forecast is {forecast_age} days old (max={max_age}). "
                f"Proceeding with stale forecast but check L5b."
            )
        elif forecast_age > warn_age:
            logger.warning(
                f"[L5c] Forecast is {forecast_age} days old (warn={warn_age}). "
                f"Consider re-running L5b."
            )

        # Get all model predictions for that date
        cur.execute("""
            SELECT model_id, predicted_return_pct / 100.0, direction
            FROM bi.fact_forecasts
            WHERE horizon_id = 1
              AND inference_date = %s
            ORDER BY model_id
        """, (latest_date,))

        predictions = cur.fetchall()
        if not predictions:
            raise ValueError(f"[L5c] No predictions for {latest_date}")

        logger.info(f"[L5c] Found {len(predictions)} model predictions")

        # Select top-N models by predicted return magnitude (proxy for confidence)
        # In production, this could use historical DA from walk-forward
        preds_sorted = sorted(predictions, key=lambda p: abs(p[1]) if p[1] else 0, reverse=True)
        top_models = preds_sorted[:n_top]

        model_ids = [p[0] for p in top_models]
        predicted_returns = [p[1] for p in top_models if p[1] is not None]

        if not predicted_returns:
            raise ValueError("[L5c] All predicted returns are NULL")

        # Ensemble: arithmetic mean of top-N predicted returns
        ensemble_return = sum(predicted_returns) / len(predicted_returns)
        ensemble_direction = 1 if ensemble_return > 0 else -1

        result = {
            "inference_date": str(latest_date),
            "forecast_age_days": forecast_age,
            "ensemble_return": ensemble_return,
            "ensemble_direction": ensemble_direction,
            "ensemble_models": ",".join(model_ids),
            "n_models_used": len(model_ids),
            "ensemble_strategy": config["ensemble_strategy"],
        }

        logger.info(
            f"[L5c] Ensemble: direction={ensemble_direction}, "
            f"return={ensemble_return:.6f}, models={model_ids}"
        )

        context['ti'].xcom_push(key='forecast', value=result)
        return result

    finally:
        conn.close()


# =============================================================================
# TASK 3: LOAD RECENT PRICES
# =============================================================================

def load_recent_prices(**context) -> Dict[str, Any]:
    """
    Load last 22 trading days of daily OHLCV from bi.dim_daily_usdcop.
    Compute daily log-returns and realized volatility.
    """
    config = get_vol_target_config(**context)
    lookback = config["vol_lookback"]
    annualization = config["annualization_factor"]

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Get last lookback+1 days (need N+1 prices for N returns)
        cur.execute("""
            SELECT date, close
            FROM bi.dim_daily_usdcop
            WHERE close IS NOT NULL
            ORDER BY date DESC
            LIMIT %s
        """, (lookback + 5,))  # +5 buffer for weekends/holidays

        rows = cur.fetchall()
        if len(rows) < lookback + 1:
            raise ValueError(
                f"[L5c] Not enough daily prices: got {len(rows)}, "
                f"need {lookback + 1}"
            )

        # Sort chronologically
        rows = sorted(rows, key=lambda r: r[0])
        dates = [r[0] for r in rows]
        closes = [r[1] for r in rows]

        # Compute log-returns
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i] > 0 and closes[i - 1] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))

        # Compute realized vol using the module function
        import numpy as np
        returns_array = np.array(log_returns)
        realized_vol = compute_realized_vol(
            returns_array,
            lookback=lookback,
            annualization=annualization,
        )

        latest_close = closes[-1]
        latest_date = dates[-1]

        result = {
            "realized_vol_21d": realized_vol,
            "latest_close": latest_close,
            "latest_date": str(latest_date),
            "n_returns": len(log_returns),
        }

        logger.info(
            f"[L5c] Realized vol (21d): {realized_vol:.4f}, "
            f"latest close: {latest_close:.2f} on {latest_date}"
        )

        context['ti'].xcom_push(key='prices', value=result)
        return result

    finally:
        conn.close()


# =============================================================================
# TASK 4: COMPUTE VOL-TARGET SIGNAL
# =============================================================================

def compute_signal(**context) -> Dict[str, Any]:
    """
    Compute vol-targeting signal from forecast + realized volatility.
    Uses src/forecasting/vol_targeting.compute_vol_target_signal().
    """
    if not VOL_TARGETING_AVAILABLE:
        raise ImportError("[L5c] vol_targeting module not available")

    config = get_vol_target_config(**context)
    ti = context['ti']

    forecast = ti.xcom_pull(key='forecast', task_ids='load_latest_forecast')
    prices = ti.xcom_pull(key='prices', task_ids='load_recent_prices')

    if not forecast or not prices:
        raise ValueError("[L5c] Missing forecast or prices from upstream tasks")

    # Build VolTargetConfig from DAG config
    vt_config = VolTargetConfig(
        target_vol=config["target_vol"],
        max_leverage=config["max_leverage"],
        min_leverage=config["min_leverage"],
        vol_lookback=config["vol_lookback"],
        vol_floor=config["vol_floor"],
        annualization_factor=config["annualization_factor"],
    )

    today_str = datetime.utcnow().date().strftime("%Y-%m-%d")

    signal = compute_vol_target_signal(
        forecast_direction=forecast["ensemble_direction"],
        forecast_return=forecast["ensemble_return"],
        realized_vol_21d=prices["realized_vol_21d"],
        config=vt_config,
        date=today_str,
    )

    result = {
        "signal_date": today_str,
        "forecast_direction": signal.forecast_direction,
        "forecast_return": signal.forecast_return,
        "ensemble_strategy": forecast["ensemble_strategy"],
        "ensemble_models": forecast["ensemble_models"],
        "realized_vol_21d": signal.realized_vol_21d,
        "raw_leverage": signal.raw_leverage,
        "clipped_leverage": signal.clipped_leverage,
        "position_size": signal.position_size,
        "target_vol": vt_config.target_vol,
        "max_leverage": vt_config.max_leverage,
        "min_leverage": vt_config.min_leverage,
        "vol_floor": vt_config.vol_floor,
        "config_version": config["config_version"],
        "forecast_age_days": forecast["forecast_age_days"],
    }

    logger.info(
        f"[L5c] Signal: date={today_str}, dir={signal.forecast_direction}, "
        f"lev={signal.clipped_leverage:.3f}, pos={signal.position_size:.3f}, "
        f"vol={signal.realized_vol_21d:.4f}"
    )

    ti.xcom_push(key='signal', value=result)
    return result


# =============================================================================
# TASK 5: PERSIST SIGNAL
# =============================================================================

def persist_signal(**context) -> Dict[str, Any]:
    """
    UPSERT signal into forecast_vol_targeting_signals.
    ON CONFLICT (signal_date) updates the existing row.
    """
    ti = context['ti']
    signal = ti.xcom_pull(key='signal', task_ids='compute_signal')

    if not signal:
        raise ValueError("[L5c] No signal from compute_signal task")

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO forecast_vol_targeting_signals (
                signal_date, forecast_direction, forecast_return,
                ensemble_strategy, ensemble_models,
                realized_vol_21d, raw_leverage, clipped_leverage, position_size,
                target_vol, max_leverage, min_leverage, vol_floor,
                config_version
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (signal_date) DO UPDATE SET
                forecast_direction = EXCLUDED.forecast_direction,
                forecast_return = EXCLUDED.forecast_return,
                ensemble_strategy = EXCLUDED.ensemble_strategy,
                ensemble_models = EXCLUDED.ensemble_models,
                realized_vol_21d = EXCLUDED.realized_vol_21d,
                raw_leverage = EXCLUDED.raw_leverage,
                clipped_leverage = EXCLUDED.clipped_leverage,
                position_size = EXCLUDED.position_size,
                target_vol = EXCLUDED.target_vol,
                max_leverage = EXCLUDED.max_leverage,
                min_leverage = EXCLUDED.min_leverage,
                vol_floor = EXCLUDED.vol_floor,
                config_version = EXCLUDED.config_version,
                created_at = NOW()
            RETURNING id
        """, (
            signal["signal_date"],
            signal["forecast_direction"],
            signal["forecast_return"],
            signal["ensemble_strategy"],
            signal["ensemble_models"],
            signal["realized_vol_21d"],
            signal["raw_leverage"],
            signal["clipped_leverage"],
            signal["position_size"],
            signal["target_vol"],
            signal["max_leverage"],
            signal["min_leverage"],
            signal["vol_floor"],
            signal["config_version"],
        ))

        row = cur.fetchone()
        conn.commit()

        signal_id = row[0] if row else None
        logger.info(f"[L5c] Persisted signal id={signal_id} for {signal['signal_date']}")

        return {"signal_id": signal_id, "signal_date": signal["signal_date"]}

    finally:
        conn.close()


# =============================================================================
# TASK 6: SIGNAL SUMMARY
# =============================================================================

def signal_summary(**context) -> None:
    """
    Log a structured summary of the generated signal.
    Fires pg_notify via the INSERT trigger (trg_notify_forecast_signal).
    """
    ti = context['ti']
    signal = ti.xcom_pull(key='signal', task_ids='compute_signal')

    if not signal:
        logger.warning("[L5c] No signal to summarize")
        return

    direction_str = "LONG" if signal["forecast_direction"] == 1 else "SHORT"
    age = signal.get("forecast_age_days", "?")

    logger.info("=" * 60)
    logger.info("[L5c] VOL-TARGETING SIGNAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Date:         {signal['signal_date']}")
    logger.info(f"  Direction:    {direction_str}")
    logger.info(f"  Leverage:     {signal['clipped_leverage']:.3f}x")
    logger.info(f"  Position:     {signal['position_size']:+.3f}")
    logger.info(f"  Vol (21d):    {signal['realized_vol_21d']:.4f}")
    logger.info(f"  Raw Leverage: {signal['raw_leverage']:.3f}x")
    logger.info(f"  Forecast Ret: {signal['forecast_return']:.6f}")
    logger.info(f"  Forecast Age: {age} day(s)")
    logger.info(f"  Ensemble:     {signal['ensemble_strategy']}")
    logger.info(f"  Models:       {signal['ensemble_models']}")
    logger.info(f"  Config:       {signal['config_version']}")
    logger.info("=" * 60)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 15),  # Creation date
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Daily vol-targeting signal from forecasting pipeline',
    schedule_interval='30 18 * * 1-5',  # 18:30 UTC = 13:30 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS,
) as dag:

    t_check_market = ShortCircuitOperator(
        task_id='check_market_day',
        python_callable=check_market_day,
        provide_context=True,
    )

    t_load_forecast = PythonOperator(
        task_id='load_latest_forecast',
        python_callable=load_latest_forecast,
        provide_context=True,
    )

    t_load_prices = PythonOperator(
        task_id='load_recent_prices',
        python_callable=load_recent_prices,
        provide_context=True,
    )

    t_compute_signal = PythonOperator(
        task_id='compute_signal',
        python_callable=compute_signal,
        provide_context=True,
    )

    t_persist = PythonOperator(
        task_id='persist_signal',
        python_callable=persist_signal,
        provide_context=True,
    )

    t_summary = PythonOperator(
        task_id='signal_summary',
        python_callable=signal_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # DAG flow: check market → [load forecast, load prices] → compute → persist → summary
    t_check_market >> [t_load_forecast, t_load_prices]
    [t_load_forecast, t_load_prices] >> t_compute_signal >> t_persist >> t_summary
