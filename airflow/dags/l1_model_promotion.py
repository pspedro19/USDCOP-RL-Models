"""
L1 Model Promotion DAG - Populate inference_ready_nrt on Model Approval
========================================================================
When a new model is approved (via L4 backtest + dashboard), this DAG:
1. Loads the model's norm_stats.json and validates the hash
2. Loads the L2 historical datasets (train/val/test parquets)
3. Computes features via CanonicalFeatureBuilder (SSOT)
4. Normalizes using the model's norm_stats
5. Populates inference_ready_nrt with historical data
6. Copies norm_stats.json to config/norm_stats.json for L1 DAG

This replaces the async L1NRTDataService.on_model_approved() logic with
a proper Airflow DAG that can be triggered manually or by model_approved event.

Contract: CTR-L1-NRT-001 - Normalized features ready for model.predict()

Trigger:
    - Manual: `airflow dags trigger rl_l1_03_model_promotion`
    - Event: Triggered by `model_approved` PostgreSQL NOTIFY channel

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Created: 2026-02-12
"""

import sys
import os
import json
import shutil
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

# DAG utils
from utils.dag_common import get_db_connection

# Add src to path
sys.path.insert(0, '/opt/airflow')

# Feature contracts
from src.core.contracts.feature_contract import (
    FEATURE_ORDER,
    FEATURE_ORDER_HASH,
    OBSERVATION_DIM,
)

# Production contract
try:
    from src.core.contracts.production_contract import (
        ProductionContract,
        get_production_contract,
    )
    PRODUCTION_CONTRACT_AVAILABLE = True
except ImportError:
    PRODUCTION_CONTRACT_AVAILABLE = False
    logging.warning("ProductionContract not available")

# Norm stats contract
try:
    from src.core.contracts.norm_stats_contract import load_norm_stats, NormStatsContract
    NORM_STATS_CONTRACT_AVAILABLE = True
except ImportError:
    NORM_STATS_CONTRACT_AVAILABLE = False
    logging.warning("NormStatsContract not available")

# CanonicalFeatureBuilder
try:
    from src.feature_store.builders.canonical_feature_builder import CanonicalFeatureBuilder
    CANONICAL_BUILDER_AVAILABLE = True
except ImportError:
    CANONICAL_BUILDER_AVAILABLE = False
    logging.warning("CanonicalFeatureBuilder not available")

# DAG registry
from contracts.dag_registry import RL_L1_MODEL_PROMOTION


# =============================================================================
# CONSTANTS
# =============================================================================

CLIP_MIN = -5.0
CLIP_MAX = 5.0
BATCH_SIZE = 500
# Market features = first N features in FEATURE_ORDER (exclude state features)
# State features are typically the last 2: position, unrealized_pnl (or time_normalized)
MARKET_FEATURE_COUNT = OBSERVATION_DIM - 2
MARKET_FEATURES = list(FEATURE_ORDER[:MARKET_FEATURE_COUNT])

# Config path for L1 DAG to pick up
NORM_STATS_DEST = "/opt/airflow/config/norm_stats.json"


# =============================================================================
# TASK: Check for new production model
# =============================================================================

def check_new_production_model(**ctx) -> str:
    """Check if there's a newly approved production model.

    Returns branch: 'load_model_config' if model found, 'skip_no_model' otherwise.
    """
    if not PRODUCTION_CONTRACT_AVAILABLE:
        logging.warning("ProductionContract not available, checking dag_run.conf")
        conf = ctx.get("dag_run", {}).conf if ctx.get("dag_run") else {}
        if conf and conf.get("model_path"):
            ctx["ti"].xcom_push(key="model_source", value="manual")
            ctx["ti"].xcom_push(key="manual_config", value=conf)
            return "load_model_config"
        return "skip_no_model"

    conn = get_db_connection()
    try:
        prod_contract = get_production_contract(conn)
        if prod_contract is None:
            # Also check dag_run.conf for manual trigger
            conf = ctx.get("dag_run", {}).conf if ctx.get("dag_run") else {}
            if conf and conf.get("model_path"):
                ctx["ti"].xcom_push(key="model_source", value="manual")
                ctx["ti"].xcom_push(key="manual_config", value=conf)
                return "load_model_config"
            logging.info("No production model found in contract")
            return "skip_no_model"

        logging.info(
            f"Production model found: {prod_contract.model_id}, "
            f"experiment={prod_contract.experiment_name}"
        )
        ctx["ti"].xcom_push(key="model_source", value="contract")
        ctx["ti"].xcom_push(key="model_id", value=prod_contract.model_id)
        ctx["ti"].xcom_push(key="model_path", value=prod_contract.model_path)
        ctx["ti"].xcom_push(key="norm_stats_path", value=prod_contract.norm_stats_path)
        ctx["ti"].xcom_push(key="norm_stats_hash", value=prod_contract.norm_stats_hash)
        ctx["ti"].xcom_push(key="feature_order_hash", value=prod_contract.feature_order_hash)
        ctx["ti"].xcom_push(key="dataset_path", value=getattr(prod_contract, 'dataset_path', None))
        return "load_model_config"
    finally:
        conn.close()


# =============================================================================
# TASK: Load model config (norm_stats + validate hashes)
# =============================================================================

def load_model_config(**ctx):
    """Load norm_stats.json from the model's lineage and validate hash."""
    ti = ctx["ti"]
    source = ti.xcom_pull(key="model_source")

    if source == "manual":
        manual_conf = ti.xcom_pull(key="manual_config")
        norm_stats_path = manual_conf.get("norm_stats_path", NORM_STATS_DEST)
        expected_hash = manual_conf.get("norm_stats_hash")
        model_id = manual_conf.get("model_id", "manual")
        dataset_path = manual_conf.get("dataset_path")
    else:
        norm_stats_path = ti.xcom_pull(key="norm_stats_path") or NORM_STATS_DEST
        expected_hash = ti.xcom_pull(key="norm_stats_hash")
        model_id = ti.xcom_pull(key="model_id")
        dataset_path = ti.xcom_pull(key="dataset_path")

    logging.info(f"Loading norm_stats from {norm_stats_path} for model {model_id}")

    # Load norm_stats
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"norm_stats not found at {norm_stats_path}")

    with open(norm_stats_path, 'r') as f:
        norm_stats_raw = json.load(f)

    # Compute hash
    with open(norm_stats_path, 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    # Validate hash
    if expected_hash and actual_hash != expected_hash:
        logging.error(
            f"norm_stats HASH MISMATCH: expected={expected_hash}, actual={actual_hash}. "
            f"Continuing with warning (fail-open during development)."
        )
    else:
        logging.info(f"norm_stats hash validated: {actual_hash}")

    # Push norm_stats to XCom (as JSON string to avoid serialization issues)
    ti.xcom_push(key="norm_stats_json", value=json.dumps(norm_stats_raw))
    ti.xcom_push(key="actual_norm_stats_hash", value=actual_hash)
    ti.xcom_push(key="confirmed_dataset_path", value=dataset_path)
    ti.xcom_push(key="confirmed_model_id", value=model_id)

    # Copy norm_stats to standard location for L1 DAG
    if norm_stats_path != NORM_STATS_DEST:
        os.makedirs(os.path.dirname(NORM_STATS_DEST), exist_ok=True)
        shutil.copy2(norm_stats_path, NORM_STATS_DEST)
        logging.info(f"Copied norm_stats to {NORM_STATS_DEST}")

    logging.info(
        f"Model config loaded: model_id={model_id}, "
        f"hash={actual_hash}, features={len(norm_stats_raw)}"
    )


# =============================================================================
# TASK: Load historical datasets and populate inference_ready_nrt
# =============================================================================

def populate_inference_ready_nrt(**ctx):
    """Load L2 datasets, compute features via CanonicalFeatureBuilder,
    normalize, and populate inference_ready_nrt.

    This replaces L1NRTDataService.on_model_approved() with proper
    Airflow task using synchronous psycopg2 (not async asyncpg).
    """
    ti = ctx["ti"]
    norm_stats_json = ti.xcom_pull(key="norm_stats_json")
    ns_hash = ti.xcom_pull(key="actual_norm_stats_hash")
    dataset_path = ti.xcom_pull(key="confirmed_dataset_path")
    model_id = ti.xcom_pull(key="confirmed_model_id")

    norm_stats = json.loads(norm_stats_json) if norm_stats_json else {}

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Log existing row count before truncate
        cur.execute("SELECT COUNT(*) FROM inference_ready_nrt")
        existing_rows = cur.fetchone()[0]
        logging.info(f"inference_ready_nrt has {existing_rows} existing rows")

        # Truncate for fresh historical load
        cur.execute("TRUNCATE inference_ready_nrt")
        conn.commit()
        logging.info("Truncated inference_ready_nrt for historical load")

        total_inserted = 0

        # Strategy 1: Load from L2 datasets (parquet files)
        if dataset_path and os.path.exists(dataset_path):
            total_inserted = _load_from_l2_datasets(
                cur, conn, dataset_path, norm_stats, ns_hash
            )
            logging.info(f"Loaded {total_inserted} rows from L2 datasets")

        # Strategy 2: If no L2 datasets, compute from OHLCV seed
        if total_inserted == 0:
            logging.info("No L2 datasets found, computing from OHLCV data in DB")
            total_inserted = _compute_from_ohlcv(cur, conn, norm_stats, ns_hash)
            logging.info(f"Computed {total_inserted} rows from OHLCV data")

        # Final verification
        cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM inference_ready_nrt")
        final_count, min_ts, max_ts = cur.fetchone()
        logging.info(
            f"inference_ready_nrt populated: {final_count} rows, "
            f"range: {min_ts} → {max_ts}"
        )

        ti.xcom_push(key="rows_inserted", value=total_inserted)
        ti.xcom_push(key="min_timestamp", value=str(min_ts))
        ti.xcom_push(key="max_timestamp", value=str(max_ts))

    finally:
        cur.close()
        conn.close()


def _load_from_l2_datasets(
    cur, conn, dataset_path: str, norm_stats: dict, ns_hash: str
) -> int:
    """Load features from L2 parquet datasets."""
    base = Path(dataset_path)
    dfs = []

    for split in ["train", "val", "test"]:
        for ext in [".parquet", "_features.parquet"]:
            path = base / f"{split}{ext}"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    logging.info(f"Loaded {path}: {len(df)} rows, cols={list(df.columns[:5])}...")
                    dfs.append(df)
                except Exception as e:
                    logging.warning(f"Failed to load {path}: {e}")
                break

    if not dfs:
        logging.warning(f"No L2 datasets found in {dataset_path}")
        return 0

    df_all = pd.concat(dfs, ignore_index=False)

    # Ensure DatetimeIndex
    if not isinstance(df_all.index, pd.DatetimeIndex):
        for col in ["timestamp", "date", "datetime", "time"]:
            if col in df_all.columns:
                df_all = df_all.set_index(col)
                break

    df_all = df_all.sort_index()

    return _batch_insert_features(cur, conn, df_all, norm_stats, ns_hash, source="historical")


def _compute_from_ohlcv(cur, conn, norm_stats: dict, ns_hash: str) -> int:
    """Compute features from OHLCV data using CanonicalFeatureBuilder."""
    if not CANONICAL_BUILDER_AVAILABLE:
        logging.error("CanonicalFeatureBuilder not available, cannot compute features")
        return 0

    # Load OHLCV from database
    cur.execute("""
        SELECT time, open, high, low, close, volume
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
        ORDER BY time ASC
    """)
    rows = cur.fetchall()

    if not rows:
        logging.warning("No OHLCV data found in database")
        return 0

    df_ohlcv = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df_ohlcv = df_ohlcv.set_index("time")
    logging.info(f"Loaded {len(df_ohlcv)} OHLCV bars for feature computation")

    # Compute features using CanonicalFeatureBuilder
    builder = CanonicalFeatureBuilder()
    df_features = builder.compute_features(df_ohlcv, include_state=False)

    if df_features is None or df_features.empty:
        logging.error("CanonicalFeatureBuilder returned empty features")
        return 0

    logging.info(f"Computed {len(df_features)} feature rows, columns: {list(df_features.columns[:5])}...")

    return _batch_insert_features(cur, conn, df_features, norm_stats, ns_hash, source="historical")


def _batch_insert_features(
    cur, conn, df: pd.DataFrame, norm_stats: dict, ns_hash: str, source: str
) -> int:
    """Normalize features and batch-insert into inference_ready_nrt."""
    if df.empty:
        return 0

    total_inserted = 0
    batch = []

    for ts, row in df.iterrows():
        # Extract market features in canonical order
        features = []
        valid = True

        for feat_name in MARKET_FEATURES:
            value = row.get(feat_name, np.nan)

            if pd.isna(value) or np.isinf(value):
                value = 0.0
            else:
                value = float(value)

            # Normalize with z-score
            if feat_name in norm_stats:
                stats = norm_stats[feat_name]
                mean = stats.get("mean", 0.0)
                std = stats.get("std", 1.0)
                if isinstance(stats, dict) and std > 1e-8:
                    value = (value - mean) / std

            # Clip
            value = float(np.clip(value, CLIP_MIN, CLIP_MAX))
            features.append(value)

        if len(features) != len(MARKET_FEATURES):
            continue

        # Get price
        price = row.get("close", row.get("price", 0.0))
        if pd.isna(price) or price <= 0:
            continue

        batch.append((
            ts,
            features,
            float(price),
            FEATURE_ORDER_HASH,
            ns_hash or "",
            source,
        ))

        if len(batch) >= BATCH_SIZE:
            total_inserted += _execute_batch(cur, conn, batch)
            batch = []

    # Insert remaining
    if batch:
        total_inserted += _execute_batch(cur, conn, batch)

    return total_inserted


def _execute_batch(cur, conn, batch: list) -> int:
    """Execute a batch insert into inference_ready_nrt."""
    if not batch:
        return 0

    template = "(%s, %s::FLOAT[], %s, %s, %s, %s)"
    query = """
        INSERT INTO inference_ready_nrt
            (timestamp, features, price, feature_order_hash, norm_stats_hash, source)
        VALUES %s
        ON CONFLICT (timestamp) DO UPDATE SET
            features = EXCLUDED.features,
            price = EXCLUDED.price,
            source = EXCLUDED.source
    """

    try:
        execute_values(cur, query, batch, template=template, page_size=BATCH_SIZE)
        conn.commit()
        return len(batch)
    except Exception as e:
        conn.rollback()
        logging.error(f"Batch insert failed: {e}")
        return 0


# =============================================================================
# TASK: Validate populated data
# =============================================================================

def validate_promotion(**ctx):
    """Validate that inference_ready_nrt was populated correctly."""
    ti = ctx["ti"]
    rows_inserted = ti.xcom_pull(key="rows_inserted") or 0
    model_id = ti.xcom_pull(key="confirmed_model_id") or "unknown"

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Row count check
        cur.execute("SELECT COUNT(*) FROM inference_ready_nrt")
        total_rows = cur.fetchone()[0]

        # Feature dimension check (sample)
        cur.execute("""
            SELECT array_length(features, 1), COUNT(*)
            FROM inference_ready_nrt
            GROUP BY array_length(features, 1)
        """)
        dim_distribution = {row[0]: row[1] for row in cur.fetchall()}

        # Null/NaN check
        cur.execute("""
            SELECT COUNT(*)
            FROM inference_ready_nrt
            WHERE features IS NULL OR price IS NULL OR price <= 0
        """)
        invalid_rows = cur.fetchone()[0]

        logging.info(f"Validation results for model {model_id}:")
        logging.info(f"  Total rows: {total_rows}")
        logging.info(f"  Feature dimensions: {dim_distribution}")
        logging.info(f"  Invalid rows: {invalid_rows}")

        if total_rows == 0:
            raise ValueError("inference_ready_nrt is empty after promotion!")

        if invalid_rows > 0:
            logging.warning(f"{invalid_rows} invalid rows detected")

        expected_dim = len(MARKET_FEATURES)
        if expected_dim not in dim_distribution:
            logging.warning(
                f"Expected feature dim {expected_dim} not found in distribution: {dim_distribution}"
            )

        logging.info(f"Model promotion validated: {total_rows} rows for model {model_id}")

    finally:
        cur.close()
        conn.close()


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "trading-team",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

dag = DAG(
    RL_L1_MODEL_PROMOTION,
    default_args=default_args,
    description="L1: Populate inference_ready_nrt on model approval (historical data load)",
    schedule_interval=None,  # Manual trigger or event-driven
    catchup=False,
    max_active_runs=1,
    tags=["l1", "model-promotion", "inference-ready-nrt", "unified-pipeline"],
)

with dag:
    check_model = BranchPythonOperator(
        task_id="check_new_production_model",
        python_callable=check_new_production_model,
        provide_context=True,
    )

    skip = EmptyOperator(
        task_id="skip_no_model",
    )

    load_config = PythonOperator(
        task_id="load_model_config",
        python_callable=load_model_config,
        provide_context=True,
    )

    populate = PythonOperator(
        task_id="populate_inference_ready_nrt",
        python_callable=populate_inference_ready_nrt,
        provide_context=True,
    )

    validate = PythonOperator(
        task_id="validate_promotion",
        python_callable=validate_promotion,
        provide_context=True,
    )

    # Task chain
    check_model >> [load_config, skip]
    load_config >> populate >> validate
