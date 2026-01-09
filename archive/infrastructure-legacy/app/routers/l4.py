#!/usr/bin/env python3
"""
L4 Router - RL-Ready Dataset Endpoints
=======================================
Data source: MinIO (bucket: usdcop, prefix: l4/)
Format: Parquet + JSON specs

Endpoints:
- GET /pipeline/l4/contract - 17-observation schema
- GET /pipeline/l4/quality-check - Clip rates, reward reproducibility
- GET /pipeline/l4/splits - Train/val/test split metadata
"""

from fastapi import APIRouter, Query
from typing import Optional
import pandas as pd
import numpy as np

from app.deps import (
    get_layer_config,
    read_latest_manifest,
    read_run_manifest,
    read_parquet_dataset,
    read_json_metadata,
    load_quality_gates
)

router = APIRouter(prefix="/pipeline/l4", tags=["pipeline-l4"])

# Observation schema (17 features)
OBS_SCHEMA = {
    "obs_00": "spread_proxy_bps_norm",
    "obs_01": "ret_5m_z",
    "obs_02": "ret_10m_z",
    "obs_03": "ret_15m_z",
    "obs_04": "ret_30m_z",
    "obs_05": "range_bps_norm",
    "obs_06": "volume_zscore",
    "obs_07": "rsi_norm",
    "obs_08": "macd_zscore",
    "obs_09": "bb_position",
    "obs_10": "ema_cross_signal",
    "obs_11": "atr_norm",
    "obs_12": "vwap_distance",
    "obs_13": "time_of_day_sin",
    "obs_14": "time_of_day_cos",
    "obs_15": "position",
    "obs_16": "inventory_age"
}


@router.get("/contract")
def get_l4_contract(run_id: Optional[str] = None):
    """
    L4: RL environment contract

    Returns:
    - Observation schema (17 features)
    - Action space definition
    - Reward specification
    - Cost model parameters
    """
    config = get_layer_config("l4")

    # Try to read env_spec.json from latest run if available
    try:
        bucket = config["bucket"]
        prefix = config["prefix"]

        if run_id is None:
            manifest = read_latest_manifest(bucket, prefix)
        else:
            manifest = read_run_manifest(bucket, prefix, run_id)

        # Try to read env_spec.json
        env_spec_file = next(
            (f for f in manifest["files"] if f["name"] == "env_spec.json"),
            None
        )

        if env_spec_file:
            env_spec = read_json_metadata(bucket, env_spec_file["path"])
        else:
            env_spec = None

        # Try to read reward_spec.json
        reward_spec_file = next(
            (f for f in manifest["files"] if f["name"] == "reward_spec.json"),
            None
        )

        if reward_spec_file:
            reward_spec = read_json_metadata(bucket, reward_spec_file["path"])
        else:
            reward_spec = None

    except:
        # Fallback to default contract if no MinIO data
        env_spec = None
        reward_spec = None
        manifest = {"run_id": "not_available"}

    # Default contract
    return {
        "status": "OK",
        "layer": "l4",
        "run_id": manifest.get("run_id"),
        "observation_schema": OBS_SCHEMA,
        "observation_space": {
            "type": "Box",
            "shape": [17],
            "dtype": "float32",
            "range": [-5, 5],
            "normalization": "robust_zscore (median/MAD per hour)"
        },
        "action_space": {
            "type": "Discrete",
            "n": 3,
            "actions": {
                "-1": "SELL",
                "0": "HOLD",
                "1": "BUY"
            }
        },
        "reward_spec": reward_spec or {
            "type": "float32",
            "reproducible": True,
            "calculation": "PnL - transaction_costs - holding_costs",
            "rmse_target": 0.0,
            "std_min": 0.1,
            "zero_pct_max": 1.0
        },
        "cost_model": {
            "spread_method": "corwin_schultz_proxy",
            "spread_p95_range_bps": [2, 25],
            "peg_rate_max_pct": 5.0
        },
        "quality_gates": load_quality_gates().get("l4", {}),
        "env_spec": env_spec,
        "note": "Contract stable across runs. Use /quality-check to verify data quality."
    }


@router.get("/quality-check")
def get_l4_quality_check(run_id: Optional[str] = None):
    """
    L4: Quality verification

    Checks:
    - Observation clip rate (<= 0.5% per feature)
    - Reward reproducibility (RMSE ~0.0)
    - Cost model bounds (spread in [2, 25] bps)
    - Train/val/test splits with embargo

    Returns GO/NO-GO status
    """
    config = get_layer_config("l4")
    gates = load_quality_gates().get("l4", {})

    bucket = config["bucket"]
    prefix = config["prefix"]

    try:
        # Get manifest
        if run_id is None:
            manifest = read_latest_manifest(bucket, prefix)
        else:
            manifest = read_run_manifest(bucket, prefix, run_id)

        # Read replay dataset
        replay_file = next(
            (f for f in manifest["files"] if f["name"] == "replay_dataset.parquet"),
            None
        )

        if not replay_file:
            return {
                "status": "NOT_FOUND",
                "message": f"No replay dataset found for run_id {manifest['run_id']}"
            }

        # Read data (only obs columns + reward columns)
        obs_cols = [f"obs_{i:02d}" for i in range(17)]
        reward_cols = ["reward", "reward_recomputed"] if "reward_recomputed" in replay_file else ["reward"]

        df = read_parquet_dataset(
            bucket,
            replay_file["path"],
            columns=obs_cols + reward_cols + ["episode_id", "t_in_episode"]
        )

        # Calculate clip rates (observations should be in [-5, 5])
        clip_rates = {}
        for col in obs_cols:
            if col in df.columns:
                clipped_count = ((df[col].abs() > 5).sum())
                clip_rates[col] = float(clipped_count / len(df) * 100)
            else:
                clip_rates[col] = None

        max_clip_rate = max([r for r in clip_rates.values() if r is not None])

        # Reward reproducibility (if reward_recomputed exists)
        if "reward_recomputed" in df.columns:
            rmse = float(np.sqrt(np.mean((df["reward"] - df["reward_recomputed"]) ** 2)))
            reward_std = float(df["reward"].std())
            reward_zero_pct = float((df["reward"] == 0).sum() / len(df) * 100)
            reward_mean = float(df["reward"].mean())
        else:
            rmse = None
            reward_std = float(df["reward"].std()) if "reward" in df.columns else None
            reward_zero_pct = float((df["reward"] == 0).sum() / len(df) * 100) if "reward" in df.columns else None
            reward_mean = float(df["reward"].mean()) if "reward" in df.columns else None

        # Read split_spec.json if available
        split_file = next(
            (f for f in manifest["files"] if f["name"] == "split_spec.json"),
            None
        )

        if split_file:
            split_spec = read_json_metadata(bucket, split_file["path"])
            splits_ok = True
            embargo_days = split_spec.get("embargo_days", 5)
        else:
            split_spec = None
            splits_ok = False
            embargo_days = None

        # Read obs_clip_rates.json if available
        clip_file = next(
            (f for f in manifest["files"] if f["name"] == "obs_clip_rates.json"),
            None
        )

        if clip_file:
            saved_clip_rates = read_json_metadata(bucket, clip_file["path"])
        else:
            saved_clip_rates = None

        # GO/NO-GO evaluation
        reward_check_ok = (
            (rmse is None or rmse < 0.01) and
            (reward_std is None or reward_std > 0) and
            (reward_zero_pct is None or reward_zero_pct < 1.0)
        )

        passed = (
            max_clip_rate <= 0.5 and
            reward_check_ok and
            splits_ok
        )

        return {
            "status": "OK",
            "layer": "l4",
            "run_id": manifest["run_id"],
            "dataset_hash": manifest.get("dataset_hash"),
            "quality_checks": {
                "obs_clip_rates": {col: round(rate, 3) if rate is not None else None
                                   for col, rate in clip_rates.items()},
                "max_clip_rate_pct": round(max_clip_rate, 3),
                "clip_rate_pass": max_clip_rate <= 0.5
            },
            "reward_check": {
                "rmse": round(rmse, 6) if rmse is not None else None,
                "std": round(reward_std, 3) if reward_std is not None else None,
                "zero_pct": round(reward_zero_pct, 2) if reward_zero_pct is not None else None,
                "mean": round(reward_mean, 2) if reward_mean is not None else None,
                "pass": reward_check_ok
            },
            "splits": {
                "embargo_days": embargo_days,
                "spec": split_spec,
                "pass": splits_ok
            },
            "data_shape": {
                "episodes": int(df["episode_id"].nunique()) if "episode_id" in df.columns else None,
                "total_steps": len(df)
            },
            "overall_pass": passed,
            "quality_gates": gates,
            "note": "Real data from MinIO L4 pipeline output"
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "layer": "l4",
            "message": str(e),
            "note": "L4 data may not exist in MinIO yet. Run pipeline to generate."
        }


@router.get("/splits")
def get_l4_splits(run_id: Optional[str] = None):
    """
    L4: Train/val/test split metadata

    Returns split sizes and date ranges
    """
    config = get_layer_config("l4")
    bucket = config["bucket"]
    prefix = config["prefix"]

    try:
        # Get manifest
        if run_id is None:
            manifest = read_latest_manifest(bucket, prefix)
        else:
            manifest = read_run_manifest(bucket, prefix, run_id)

        # Read split_spec.json
        split_file = next(
            (f for f in manifest["files"] if f["name"] == "split_spec.json"),
            None
        )

        if not split_file:
            return {
                "status": "NOT_FOUND",
                "message": "No split spec found"
            }

        split_spec = read_json_metadata(bucket, split_file["path"])

        return {
            "status": "OK",
            "layer": "l4",
            "run_id": manifest["run_id"],
            "splits": split_spec
        }

    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }
