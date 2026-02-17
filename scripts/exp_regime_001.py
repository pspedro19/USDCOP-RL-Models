"""
EXP-REGIME-001: Regime Detection Features for Forecasting
==========================================================

Pre-registered experiment runner.
Tests whether regime-aware features improve directional accuracy.

Protocol:
  GATE 1 — Walk-forward on 2019-2025 (5 expanding windows, 30-day gap)
  GATE 2 — 2026 holdout (one shot, only if GATE 1 passes)

Treatment Groups:
  CONTROL:  19 baseline features (no regime)
  TREAT-A:  + F1 (trend_slope_60d)
  TREAT-B:  + F1, F2 (trend_slope_60d, range_ratio_20d)
  TREAT-C:  + F1, F2, F3 (trend_slope_60d, range_ratio_20d, return_20d_sign)
  TREAT-D:  + F1, F2, F3, F4 (all non-HMM regime features)
  TREAT-E:  + F1, F4 (trend + vol ratio — minimal regime pair)
  TREAT-F:  + F5 (HMM only — 3 cols: prob_calm, prob_crisis, entropy)
  TREAT-G:  + F1, F4, F5 (trend + vol_ratio + HMM)

Significance:
  Paired t-test DA(treatment) vs DA(CONTROL) per fold.
  Bonferroni correction: p < 0.05/7 = 0.00714 for 7 comparisons.

Usage:
  python scripts/exp_regime_001.py                     # GATE 1 only
  python scripts/exp_regime_001.py --gate2             # GATE 1 + GATE 2
  python scripts/exp_regime_001.py --groups CONTROL TREAT-A TREAT-E  # Subset
  python scripts/exp_regime_001.py --skip-hmm          # Skip F5 (slow)

@experiment EXP-REGIME-001
@pre-registered 2026-02-16
@status READY
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.contracts import MODEL_IDS, get_horizon_config, MODEL_DEFINITIONS
from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.regime_features import (
    TREATMENT_GROUPS,
    FEATURE_COLUMNS as REGIME_FEATURE_COLUMNS,
    build_regime_features,
    get_feature_columns,
    hmm_regime_features,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

BONFERRONI_ALPHA = 0.05 / 7  # 7 treatment groups
N_FOLDS = 5
INITIAL_TRAIN_RATIO = 0.6
GAP_DAYS = 30  # 30-day gap between train/test
HORIZON = 1  # H=1 only (production horizon)

GATE1_MIN_DA_IMPROVEMENT = 0.5  # +0.5pp over control
GATE2_MIN_DA = 50.0  # Absolute DA threshold on 2026

HOLDOUT_START = pd.Timestamp("2026-01-01")

OUTPUT_DIR = PROJECT_ROOT / "results" / "exp_regime_001"


# =============================================================================
# DATA LOADING (mirrors backtest_2025_10k.py)
# =============================================================================

def load_daily_data() -> pd.DataFrame:
    """Load daily OHLCV + macro, build 19 baseline features."""
    logger.info("Loading daily OHLCV...")
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path)

    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)
    logger.info(
        f"  OHLCV: {len(df_ohlcv)} rows, "
        f"{df_ohlcv['date'].min().date()} to {df_ohlcv['date'].max().date()}"
    )

    # Load macro (DXY + WTI with T-1 lag)
    logger.info("Loading macro data (DXY, WTI)...")
    macro_path = (
        PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
        / "MACRO_DAILY_CLEAN.parquet"
    )
    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
    }
    df_macro_subset = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_subset.rename(columns=macro_cols, inplace=True)
    df_macro_subset = df_macro_subset.sort_values("date")
    df_macro_subset["dxy_close_lag1"] = df_macro_subset["dxy_close_lag1"].shift(1)
    df_macro_subset["oil_close_lag1"] = df_macro_subset["oil_close_lag1"].shift(1)

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_subset.sort_values("date"),
        on="date",
        direction="backward",
    )

    df = _build_baseline_features(df)
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    target_mask = df["target_return_1d"].notna()
    df = df[feature_mask & target_mask].reset_index(drop=True)

    logger.info(f"  After cleanup: {len(df)} rows with complete features")
    return df


def _build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 19 SSOT features from raw OHLCV + macro."""
    df = df.copy()

    # Returns (4)
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # Volatility (3)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Technical (3)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

    # Calendar (3)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    # Macro (2) — already in df
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()

    return df


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def get_walk_forward_splits(
    n_samples: int,
    n_folds: int = N_FOLDS,
    initial_ratio: float = INITIAL_TRAIN_RATIO,
    gap: int = GAP_DAYS,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate expanding window splits with gap (purging)."""
    initial_train_size = int(n_samples * initial_ratio)
    remaining = n_samples - initial_train_size
    test_size = remaining // n_folds

    splits = []
    for fold in range(n_folds):
        train_end = initial_train_size + (fold * test_size)
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)

        if test_start >= n_samples or test_end <= test_start:
            continue

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def run_walk_forward_for_group(
    df: pd.DataFrame,
    group: str,
    feature_cols: List[str],
    models_to_use: List[str],
) -> Dict:
    """
    Run walk-forward validation for one treatment group.

    Returns dict with per-model, per-fold DA results.
    """
    horizon_config = get_horizon_config(HORIZON)
    X = df[feature_cols].values.astype(np.float64)
    y = df["target_return_1d"].values.astype(np.float64)

    splits = get_walk_forward_splits(len(X))
    logger.info(f"  [{group}] {len(splits)} folds, {len(feature_cols)} features")

    boosting_models = {"xgboost_pure", "lightgbm_pure"}
    catboost_models = {"catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    results = {
        "group": group,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "n_folds": len(splits),
        "models": {},
    }

    for model_id in models_to_use:
        fold_das = []
        fold_details = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Scale for linear models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                # Create model with correct params
                if model_id in linear_models:
                    params = None
                elif model_id in catboost_models:
                    params = {
                        "iterations": horizon_config.get("n_estimators", 50),
                        "depth": horizon_config.get("max_depth", 3),
                        "learning_rate": horizon_config.get("learning_rate", 0.05),
                        "l2_leaf_reg": horizon_config.get("reg_alpha", 0.5),
                        "verbose": False,
                        "allow_writing_files": False,
                    }
                elif model_id in hybrid_models:
                    if "catboost" in model_id:
                        params = {
                            "iterations": horizon_config.get("n_estimators", 50),
                            "depth": horizon_config.get("max_depth", 3),
                            "learning_rate": horizon_config.get("learning_rate", 0.05),
                            "verbose": False,
                            "allow_writing_files": False,
                        }
                    else:
                        params = horizon_config
                else:
                    params = horizon_config

                model = ModelFactory.create(model_id, params=params, horizon=HORIZON)

                if model.requires_scaling:
                    model.fit(X_train_scaled, y_train)
                    preds = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                # Direction accuracy
                correct = np.sum(np.sign(preds) == np.sign(y_test))
                da = (correct / len(y_test)) * 100

                fold_das.append(da)
                fold_details.append({
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                    "da": round(da, 2),
                })

            except Exception as e:
                logger.warning(f"    {model_id} fold {fold_idx} failed: {e}")
                fold_das.append(np.nan)
                fold_details.append({
                    "fold": fold_idx,
                    "da": None,
                    "error": str(e),
                })

        # Aggregate
        valid_das = [d for d in fold_das if not np.isnan(d)]
        results["models"][model_id] = {
            "fold_das": fold_das,
            "fold_details": fold_details,
            "da_mean": round(np.mean(valid_das), 2) if valid_das else None,
            "da_std": round(np.std(valid_das), 2) if valid_das else None,
            "da_min": round(np.min(valid_das), 2) if valid_das else None,
            "da_max": round(np.max(valid_das), 2) if valid_das else None,
            "n_valid_folds": len(valid_das),
        }

    return results


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def compute_paired_tests(
    control_results: Dict,
    treatment_results: Dict,
) -> Dict:
    """
    Paired t-test: DA(treatment) vs DA(control) per model, averaged.

    Uses the per-fold DA as paired observations.
    """
    tests = {}

    for model_id in control_results["models"]:
        if model_id not in treatment_results["models"]:
            continue

        ctrl_das = control_results["models"][model_id]["fold_das"]
        treat_das = treatment_results["models"][model_id]["fold_das"]

        # Filter NaN pairs
        pairs = [
            (c, t) for c, t in zip(ctrl_das, treat_das)
            if not (np.isnan(c) or np.isnan(t))
        ]

        if len(pairs) < 3:
            tests[model_id] = {"n_pairs": len(pairs), "insufficient_data": True}
            continue

        ctrl_arr = np.array([p[0] for p in pairs])
        treat_arr = np.array([p[1] for p in pairs])

        diff = treat_arr - ctrl_arr
        t_stat, p_value = stats.ttest_rel(treat_arr, ctrl_arr)

        tests[model_id] = {
            "n_pairs": len(pairs),
            "ctrl_mean": round(float(np.mean(ctrl_arr)), 2),
            "treat_mean": round(float(np.mean(treat_arr)), 2),
            "improvement_pp": round(float(np.mean(diff)), 2),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_value), 4),
            "significant_bonferroni": p_value < BONFERRONI_ALPHA,
            "significant_nominal": p_value < 0.05,
        }

    # Ensemble test: average DA across all models per fold
    ctrl_ensemble = []
    treat_ensemble = []

    for fold_idx in range(N_FOLDS):
        ctrl_fold_das = []
        treat_fold_das = []
        for model_id in control_results["models"]:
            if model_id in treatment_results["models"]:
                c = control_results["models"][model_id]["fold_das"][fold_idx]
                t = treatment_results["models"][model_id]["fold_das"][fold_idx]
                if not (np.isnan(c) or np.isnan(t)):
                    ctrl_fold_das.append(c)
                    treat_fold_das.append(t)
        if ctrl_fold_das:
            ctrl_ensemble.append(np.mean(ctrl_fold_das))
            treat_ensemble.append(np.mean(treat_fold_das))

    if len(ctrl_ensemble) >= 3:
        ctrl_arr = np.array(ctrl_ensemble)
        treat_arr = np.array(treat_ensemble)
        diff = treat_arr - ctrl_arr
        t_stat, p_value = stats.ttest_rel(treat_arr, ctrl_arr)

        tests["_ensemble_avg"] = {
            "n_pairs": len(ctrl_ensemble),
            "ctrl_mean": round(float(np.mean(ctrl_arr)), 2),
            "treat_mean": round(float(np.mean(treat_arr)), 2),
            "improvement_pp": round(float(np.mean(diff)), 2),
            "t_stat": round(float(t_stat), 3),
            "p_value": round(float(p_value), 4),
            "significant_bonferroni": p_value < BONFERRONI_ALPHA,
            "significant_nominal": p_value < 0.05,
        }

    return tests


# =============================================================================
# GATE 2: 2026 HOLDOUT
# =============================================================================

def run_gate2_holdout(
    df: pd.DataFrame,
    group: str,
    feature_cols: List[str],
    models_to_use: List[str],
) -> Dict:
    """
    Train on all pre-2026 data, test on 2026.
    One shot — no re-running.
    """
    df_train = df[df["date"] < HOLDOUT_START].copy()
    df_test = df[df["date"] >= HOLDOUT_START].copy()

    if len(df_test) == 0:
        logger.warning(f"  [{group}] No 2026 data available for GATE 2")
        return {"error": "No 2026 data", "n_test": 0}

    logger.info(
        f"  [{group}] GATE 2: train={len(df_train)}, test={len(df_test)} "
        f"({df_test['date'].min().date()} to {df_test['date'].max().date()})"
    )

    horizon_config = get_horizon_config(HORIZON)
    X_train = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train["target_return_1d"].values.astype(np.float64)
    X_test = df_test[feature_cols].values.astype(np.float64)
    y_test = df_test["target_return_1d"].values.astype(np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    boosting_models = {"xgboost_pure", "lightgbm_pure"}
    catboost_models = {"catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    results = {
        "group": group,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "test_start": str(df_test["date"].min().date()),
        "test_end": str(df_test["date"].max().date()),
        "models": {},
    }

    for model_id in models_to_use:
        try:
            if model_id in linear_models:
                params = None
            elif model_id in catboost_models:
                params = {
                    "iterations": horizon_config.get("n_estimators", 50),
                    "depth": horizon_config.get("max_depth", 3),
                    "learning_rate": horizon_config.get("learning_rate", 0.05),
                    "l2_leaf_reg": horizon_config.get("reg_alpha", 0.5),
                    "verbose": False,
                    "allow_writing_files": False,
                }
            elif model_id in hybrid_models:
                if "catboost" in model_id:
                    params = {
                        "iterations": horizon_config.get("n_estimators", 50),
                        "depth": horizon_config.get("max_depth", 3),
                        "learning_rate": horizon_config.get("learning_rate", 0.05),
                        "verbose": False,
                        "allow_writing_files": False,
                    }
                else:
                    params = horizon_config
            else:
                params = horizon_config

            model = ModelFactory.create(model_id, params=params, horizon=HORIZON)

            if model.requires_scaling:
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

            correct = np.sum(np.sign(preds) == np.sign(y_test))
            da = (correct / len(y_test)) * 100

            results["models"][model_id] = {
                "da": round(da, 2),
                "n_correct": int(correct),
                "n_total": len(y_test),
            }

        except Exception as e:
            logger.warning(f"    {model_id} GATE 2 failed: {e}")
            results["models"][model_id] = {"error": str(e)}

    # Ensemble DA
    model_das = [
        v["da"] for v in results["models"].values() if "da" in v
    ]
    if model_das:
        results["ensemble_da_mean"] = round(float(np.mean(model_das)), 2)

    return results


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(
    groups: Optional[List[str]] = None,
    run_gate2: bool = False,
    skip_hmm: bool = False,
) -> Dict:
    """Run full EXP-REGIME-001 experiment."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine groups to run
    if groups is None:
        groups = list(TREATMENT_GROUPS.keys())

    if skip_hmm:
        groups = [g for g in groups if g not in ("TREAT-F", "TREAT-G")]
        logger.info("Skipping HMM groups (TREAT-F, TREAT-G) as requested")

    if "CONTROL" not in groups:
        groups = ["CONTROL"] + groups

    logger.info("=" * 70)
    logger.info(f"EXP-REGIME-001: Regime Detection Features")
    logger.info(f"Groups: {groups}")
    logger.info(f"Folds: {N_FOLDS}, Initial ratio: {INITIAL_TRAIN_RATIO}, Gap: {GAP_DAYS}d")
    logger.info(f"Bonferroni alpha: {BONFERRONI_ALPHA:.4f} (7 comparisons)")
    logger.info(f"GATE 2 (2026 holdout): {'YES' if run_gate2 else 'NO'}")
    logger.info("=" * 70)

    # Load data
    df = load_daily_data()

    # Check available models
    models_to_use = list(MODEL_IDS)
    try:
        ModelFactory.create("ard")
    except Exception:
        models_to_use = [m for m in models_to_use if m != "ard"]
        logger.warning("ARD model not available, using 8 models")

    # Pre-compute HMM features once (expensive, shared across groups)
    hmm_cache = None
    hmm_groups = [g for g in groups if "F5" in TREATMENT_GROUPS.get(g, [])]
    if hmm_groups:
        logger.info("Pre-computing HMM features (may take a few minutes)...")
        t0 = time.time()
        hmm_cache = hmm_regime_features(df["close"])
        logger.info(f"  HMM computed in {time.time() - t0:.1f}s")

    # Run walk-forward for each group
    all_results = {}
    gate1_results = {}

    for group in groups:
        logger.info(f"\n{'='*50}")
        logger.info(f"GATE 1: Walk-Forward for {group}")
        logger.info(f"{'='*50}")

        # Build regime features for this group
        df_group = build_regime_features(df, group, hmm_cache=hmm_cache)

        # Build feature column list
        baseline_cols = list(FEATURE_COLUMNS)
        regime_cols = get_feature_columns(group)
        feature_cols = baseline_cols + regime_cols

        # Drop rows with NaN in regime features
        valid_mask = df_group[feature_cols + ["target_return_1d"]].notna().all(axis=1)
        df_valid = df_group[valid_mask].reset_index(drop=True)

        # For GATE 1, use only pre-2026 data
        df_gate1 = df_valid[df_valid["date"] < HOLDOUT_START].reset_index(drop=True)

        logger.info(
            f"  {group}: {len(df_gate1)} rows for walk-forward "
            f"({len(feature_cols)} features: {len(baseline_cols)} base + {len(regime_cols)} regime)"
        )

        wf_result = run_walk_forward_for_group(
            df_gate1, group, feature_cols, models_to_use
        )
        gate1_results[group] = wf_result

    # Statistical comparison: each treatment vs CONTROL
    logger.info(f"\n{'='*70}")
    logger.info("STATISTICAL COMPARISONS (paired t-test, Bonferroni corrected)")
    logger.info(f"{'='*70}")

    comparisons = {}
    passing_groups = []

    for group in groups:
        if group == "CONTROL":
            continue

        tests = compute_paired_tests(gate1_results["CONTROL"], gate1_results[group])
        comparisons[group] = tests

        # Check if ensemble average passes
        ens = tests.get("_ensemble_avg", {})
        improvement = ens.get("improvement_pp", 0)
        p_val = ens.get("p_value", 1.0)
        sig_bonf = ens.get("significant_bonferroni", False)

        status = "PASS" if improvement >= GATE1_MIN_DA_IMPROVEMENT and sig_bonf else "FAIL"
        if status == "PASS":
            passing_groups.append(group)

        logger.info(
            f"  {group} vs CONTROL: "
            f"improvement={improvement:+.2f}pp, "
            f"p={p_val:.4f}, "
            f"Bonferroni={'YES' if sig_bonf else 'NO'} "
            f"→ {status}"
        )

        # Per-model details
        for model_id, test in sorted(tests.items()):
            if model_id.startswith("_"):
                continue
            if isinstance(test, dict) and "improvement_pp" in test:
                logger.info(
                    f"    {model_id:20s}: "
                    f"ctrl={test['ctrl_mean']:.1f}%, "
                    f"treat={test['treat_mean']:.1f}%, "
                    f"Δ={test['improvement_pp']:+.1f}pp, "
                    f"p={test['p_value']:.4f}"
                )

    # GATE 2: 2026 Holdout (only for passing groups)
    gate2_results = {}
    if run_gate2:
        logger.info(f"\n{'='*70}")
        logger.info("GATE 2: 2026 Holdout (one shot)")
        logger.info(f"{'='*70}")

        # Always run CONTROL for comparison
        gate2_groups = ["CONTROL"] + passing_groups
        if not passing_groups:
            logger.info("  No groups passed GATE 1. Running CONTROL anyway for reference.")
            gate2_groups = ["CONTROL"]

        for group in gate2_groups:
            logger.info(f"\n  Running GATE 2 for {group}...")
            df_group = build_regime_features(df, group, hmm_cache=hmm_cache)

            baseline_cols = list(FEATURE_COLUMNS)
            regime_cols = get_feature_columns(group)
            feature_cols = baseline_cols + regime_cols

            valid_mask = df_group[feature_cols + ["target_return_1d"]].notna().all(axis=1)
            df_valid = df_group[valid_mask].reset_index(drop=True)

            g2_result = run_gate2_holdout(
                df_valid, group, feature_cols, models_to_use
            )
            gate2_results[group] = g2_result

            if "ensemble_da_mean" in g2_result:
                da = g2_result["ensemble_da_mean"]
                status = "PASS" if da >= GATE2_MIN_DA else "FAIL"
                logger.info(
                    f"    {group} GATE 2: ensemble DA={da:.1f}% → {status}"
                )

    # Summary
    duration = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*70}")

    summary = {
        "experiment_id": "EXP-REGIME-001",
        "timestamp": timestamp,
        "duration_seconds": round(duration, 1),
        "config": {
            "n_folds": N_FOLDS,
            "initial_train_ratio": INITIAL_TRAIN_RATIO,
            "gap_days": GAP_DAYS,
            "horizon": HORIZON,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "gate1_min_improvement": GATE1_MIN_DA_IMPROVEMENT,
            "gate2_min_da": GATE2_MIN_DA,
            "n_models": len(models_to_use),
            "models": models_to_use,
        },
        "groups_tested": groups,
        "gate1_results": {},
        "gate1_comparisons": comparisons,
        "gate1_passing_groups": passing_groups,
        "gate2_results": gate2_results,
    }

    # Build GATE 1 summary table
    logger.info(f"\n  {'Group':<12} {'Features':>8} {'DA Mean':>8} {'DA Std':>7} {'vs CTRL':>8} {'p-value':>8} {'Status':>8}")
    logger.info(f"  {'-'*60}")

    for group in groups:
        gr = gate1_results[group]
        # Ensemble DA across models
        model_das = [
            m["da_mean"] for m in gr["models"].values()
            if m["da_mean"] is not None
        ]
        da_mean = np.mean(model_das) if model_das else 0
        da_std = np.std(model_das) if model_das else 0

        summary["gate1_results"][group] = {
            "n_features": gr["n_features"],
            "da_mean_across_models": round(da_mean, 2),
            "da_std_across_models": round(da_std, 2),
            "models": gr["models"],
        }

        if group == "CONTROL":
            logger.info(
                f"  {group:<12} {gr['n_features']:>8} "
                f"{da_mean:>7.2f}% {da_std:>6.2f}% "
                f"{'---':>8} {'---':>8} {'BASE':>8}"
            )
        else:
            comp = comparisons.get(group, {})
            ens = comp.get("_ensemble_avg", {})
            improvement = ens.get("improvement_pp", 0)
            p_val = ens.get("p_value", 1.0)
            status = "PASS" if group in passing_groups else "FAIL"
            logger.info(
                f"  {group:<12} {gr['n_features']:>8} "
                f"{da_mean:>7.2f}% {da_std:>6.2f}% "
                f"{improvement:>+7.2f}pp {p_val:>8.4f} {status:>8}"
            )

    # Decision
    logger.info(f"\n  Duration: {duration:.1f}s")
    logger.info(f"  Groups passing GATE 1: {passing_groups or 'NONE'}")

    if passing_groups:
        # Parsimony rule: if multiple pass, prefer fewest features
        best = min(
            passing_groups,
            key=lambda g: len(get_feature_columns(g))
        )
        logger.info(f"  Recommended (parsimony): {best} ({len(get_feature_columns(best))} regime features)")
        summary["recommended_group"] = best
    else:
        logger.info("  No treatment group improved significantly over CONTROL.")
        logger.info("  Decision: KEEP baseline 19 features (no regime features needed).")
        summary["recommended_group"] = "CONTROL"

    # Save results
    results_path = OUTPUT_DIR / f"exp_regime_001_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\n  Results saved to: {results_path}")

    # Save latest symlink
    latest_path = OUTPUT_DIR / "latest.json"
    with open(latest_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EXP-REGIME-001: Regime Detection Features Experiment"
    )
    parser.add_argument(
        "--gate2",
        action="store_true",
        help="Run GATE 2 (2026 holdout) for groups passing GATE 1",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Specific treatment groups to test (default: all)",
    )
    parser.add_argument(
        "--skip-hmm",
        action="store_true",
        help="Skip HMM-based groups (TREAT-F, TREAT-G) to save time",
    )
    args = parser.parse_args()

    run_experiment(
        groups=args.groups,
        run_gate2=args.gate2,
        skip_hmm=args.skip_hmm,
    )


if __name__ == "__main__":
    main()
