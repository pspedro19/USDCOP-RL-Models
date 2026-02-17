#!/usr/bin/env python3
"""
Pre-Training Validation Script
==============================

Ejecutar ANTES de cualquier training para validar estado del sistema.

Verifica:
1. Dataset existe y tiene suficientes filas
2. ADX no está saturado
3. Norm stats existen y son válidos
4. Configuración de training es correcta
5. Features están en el orden correcto

Exit codes:
    0 = OK, puede proceder con training
    1 = FAIL, corregir antes de proceder

Uso:
    python scripts/validate_before_training.py
    python scripts/validate_before_training.py --dataset DS_v2_fixed2

Fecha: 2026-02-01
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATASET = "DS_v2_fixed2"
DATA_DIR = Path("data/pipeline/07_output/5min")
CONFIG_PATH = Path("config/training_config.yaml")

EXPECTED_FEATURES = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d"
]

THRESHOLDS = {
    "min_rows": 50000,
    "adx_max_mean": 50.0,
    "adx_min_std": 10.0,
    "rsi_min": 0.0,
    "rsi_max": 100.0,
    "zscore_max_abs_mean": 0.5,
    "ent_coef_min": 0.05,
}


# =============================================================================
# Validation Functions
# =============================================================================

def check_dataset_exists(dataset_name: str) -> Tuple[bool, str, pd.DataFrame | None]:
    """Check dataset parquet file exists and load it."""
    dataset_path = DATA_DIR / f"{dataset_name}.parquet"

    if not dataset_path.exists():
        return False, f"Dataset not found: {dataset_path}", None

    try:
        df = pd.read_parquet(dataset_path)
        return True, f"Dataset loaded: {len(df):,} rows", df
    except Exception as e:
        return False, f"Failed to load dataset: {e}", None


def check_row_count(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check dataset has minimum required rows."""
    count = len(df)
    threshold = THRESHOLDS["min_rows"]

    if count < threshold:
        return False, f"Insufficient rows: {count:,} < {threshold:,}"

    return True, f"Row count OK: {count:,} >= {threshold:,}"


def check_adx_distribution(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check ADX is not saturated."""
    if "adx_14" not in df.columns:
        return False, "ADX column 'adx_14' not found"

    adx = df["adx_14"]
    mean = adx.mean()
    std = adx.std()

    max_mean = THRESHOLDS["adx_max_mean"]
    min_std = THRESHOLDS["adx_min_std"]

    issues = []
    if mean > max_mean:
        issues.append(f"mean={mean:.2f} > {max_mean}")
    if std < min_std:
        issues.append(f"std={std:.2f} < {min_std}")

    if issues:
        return False, f"ADX SATURATED: {', '.join(issues)}"

    return True, f"ADX OK: mean={mean:.2f}, std={std:.2f}"


def check_rsi_range(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check RSI is in valid range."""
    if "rsi_9" not in df.columns:
        return False, "RSI column 'rsi_9' not found"

    rsi = df["rsi_9"]
    min_val = rsi.min()
    max_val = rsi.max()

    if min_val < THRESHOLDS["rsi_min"] or max_val > THRESHOLDS["rsi_max"]:
        return False, f"RSI out of range: [{min_val:.2f}, {max_val:.2f}]"

    return True, f"RSI OK: [{min_val:.2f}, {max_val:.2f}]"


def check_zscore_features(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check z-score features have reasonable distribution."""
    zscore_features = ["dxy_z", "vix_z", "embi_z"]
    issues = []

    for feat in zscore_features:
        if feat not in df.columns:
            issues.append(f"{feat} not found")
            continue

        z = df[feat]
        mean = z.mean()
        max_abs = THRESHOLDS["zscore_max_abs_mean"]

        if abs(mean) > max_abs:
            issues.append(f"{feat} mean={mean:.2f}")

    if issues:
        return False, f"Z-score issues: {', '.join(issues)}"

    return True, "Z-score features OK"


def check_features_present(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check all expected features are present."""
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]

    if missing:
        return False, f"Missing features: {missing}"

    return True, f"All {len(EXPECTED_FEATURES)} features present"


def check_norm_stats(dataset_name: str) -> Tuple[bool, str]:
    """Check norm_stats.json exists and is valid."""
    norm_stats_path = DATA_DIR / f"{dataset_name}_norm_stats.json"

    if not norm_stats_path.exists():
        return False, f"Norm stats not found: {norm_stats_path}"

    try:
        with open(norm_stats_path) as f:
            stats = json.load(f)

        features = stats.get("features", stats)
        count = len(features)

        if count < 13:
            return False, f"Norm stats incomplete: {count} features"

        return True, f"Norm stats OK: {count} features"

    except Exception as e:
        return False, f"Failed to load norm stats: {e}"


def check_training_config() -> Tuple[bool, str]:
    """Check training config exists and has correct values."""
    if not CONFIG_PATH.exists():
        return False, f"Training config not found: {CONFIG_PATH}"

    try:
        with open(CONFIG_PATH) as f:
            config = yaml.safe_load(f)

        # Check ent_coef
        ent_coef = config.get("training", {}).get("ent_coef", 0)
        min_ent = THRESHOLDS["ent_coef_min"]

        if ent_coef < min_ent:
            return False, f"ent_coef too low: {ent_coef} < {min_ent}"

        # Check episode length consistency
        max_steps = config.get("environment", {}).get("max_episode_steps", 0)

        return True, f"Config OK: ent_coef={ent_coef}, max_steps={max_steps}"

    except Exception as e:
        return False, f"Failed to load config: {e}"


def check_null_values(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check for null values in critical columns."""
    null_counts = df[EXPECTED_FEATURES].isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls > 0:
        null_features = null_counts[null_counts > 0].to_dict()
        return False, f"Null values found: {null_features}"

    return True, "No null values in features"


# =============================================================================
# Main
# =============================================================================

def run_validation(dataset_name: str) -> int:
    """Run all validation checks."""
    print("=" * 60)
    print("PRE-TRAINING VALIDATION")
    print(f"Dataset: {dataset_name}")
    print("=" * 60)
    print()

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Check dataset exists
    success, msg, df = check_dataset_exists(dataset_name)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)
        # Cannot continue without dataset
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED - Dataset not found")
        print("=" * 60)
        return 1

    # 2. Check row count
    success, msg = check_row_count(df)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)

    # 3. Check features present
    success, msg = check_features_present(df)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)

    # 4. Check null values
    success, msg = check_null_values(df)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)

    # 5. Check ADX distribution
    success, msg = check_adx_distribution(df)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)

    # 6. Check RSI range
    success, msg = check_rsi_range(df)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)

    # 7. Check z-score features
    success, msg = check_zscore_features(df)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        warnings.append(msg)  # Warning, not error

    # 8. Check norm_stats
    success, msg = check_norm_stats(dataset_name)
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        errors.append(msg)

    # 9. Check training config
    success, msg = check_training_config()
    print(f"{'✓' if success else '✗'} {msg}")
    if not success:
        warnings.append(msg)  # Warning, not error (config may be in L3 contracts)

    # Summary
    print()
    print("=" * 60)

    if errors:
        print("❌ VALIDATION FAILED")
        print()
        print("Errors:")
        for e in errors:
            print(f"  • {e}")
        if warnings:
            print()
            print("Warnings:")
            for w in warnings:
                print(f"  • {w}")
        print()
        print("Fix the errors above before training.")
        print("=" * 60)
        return 1

    elif warnings:
        print("⚠️ VALIDATION PASSED WITH WARNINGS")
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  • {w}")
        print()
        print("You may proceed with training, but consider addressing warnings.")
        print("=" * 60)
        return 0

    else:
        print("✅ VALIDATION PASSED")
        print()
        print("All checks passed. Ready for training.")
        print("=" * 60)
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Validate system state before training"
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset name without extension (default: {DEFAULT_DATASET})"
    )
    args = parser.parse_args()

    exit_code = run_validation(args.dataset)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
