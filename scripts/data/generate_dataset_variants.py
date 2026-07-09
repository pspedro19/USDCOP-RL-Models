#!/usr/bin/env python3
"""
Generate Dataset Variants for A/B Testing
==========================================

Creates multiple dataset variants with different feature sets for A/B experiments.
Automatically versions datasets with DVC and pushes to MinIO.

Variants:
1. RL_DS3_MACRO_FULL.csv - All 7 macro features (baseline)
2. RL_DS3_MACRO_CORE.csv - 4 core macro features (reduced)

Usage:
    python scripts/generate_dataset_variants.py

Output:
    data/pipeline/07_output/datasets_5min/
    ├── RL_DS3_MACRO_FULL.csv      (15 features: 6 tech + 7 macro + 2 state)
    ├── RL_DS3_MACRO_FULL.csv.dvc  (DVC tracking file)
    ├── RL_DS3_MACRO_CORE.csv      (12 features: 6 tech + 4 macro + 2 state)
    └── RL_DS3_MACRO_CORE.csv.dvc  (DVC tracking file)

DVC:
    - Automatically runs `dvc add` for each dataset
    - Pushes to MinIO remote
    - Creates git tags: dataset-exp/{variant}_v{timestamp}

Author: Trading Team
Date: 2026-01-17
"""

import os
import sys
import json
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Feature Definitions
# =============================================================================

# Technical features (same for all variants)
TECHNICAL_FEATURES = [
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    "rsi_9",
    "atr_pct",
    "adx_14",
]

# Macro features - FULL set (7 features)
MACRO_FEATURES_FULL = [
    "dxy_z",           # Dollar Index z-score
    "dxy_change_1d",   # DXY daily change
    "vix_z",           # VIX z-score
    "embi_z",          # EMBI Colombia z-score
    "brent_change_1d", # Brent oil daily change
    "rate_spread",     # COL-USA rate differential
    "usdmxn_change_1d", # USD/MXN daily change (EM proxy)
]

# Macro features - CORE set (4 features)
MACRO_FEATURES_CORE = [
    "dxy_z",           # Dollar strength (highest correlation)
    "vix_z",           # Global risk sentiment
    "embi_z",          # Colombia sovereign risk
    "brent_change_1d", # Oil price (Colombia = exporter)
]

# State features (added by environment, not in CSV)
STATE_FEATURES = [
    "position",
    "time_normalized",
]

# Dataset variants configuration
DATASET_VARIANTS = {
    "RL_DS3_MACRO_FULL": {
        "macro_features": MACRO_FEATURES_FULL,
        "description": "Full macro features (7 variables)",
        "observation_dim": len(TECHNICAL_FEATURES) + len(MACRO_FEATURES_FULL) + len(STATE_FEATURES),
    },
    "RL_DS3_MACRO_CORE": {
        "macro_features": MACRO_FEATURES_CORE,
        "description": "Core macro features (4 variables)",
        "observation_dim": len(TECHNICAL_FEATURES) + len(MACRO_FEATURES_CORE) + len(STATE_FEATURES),
    },
}


# =============================================================================
# Database Connection
# =============================================================================

def get_db_connection():
    """Get PostgreSQL connection."""
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        database=os.environ.get("POSTGRES_DB", "usdcop"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
    )


# =============================================================================
# Data Loading
# =============================================================================

def load_ohlcv_data(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV data from database."""
    query = f"""
        SELECT time, open, high, low, close, volume
        FROM usdcop_m5_ohlcv
        WHERE time >= '{start_date}' AND time <= '{end_date}'
        ORDER BY time
    """
    df = pd.read_sql(query, conn)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    logger.info(f"Loaded {len(df):,} OHLCV bars from {start_date} to {end_date}")
    return df


def load_macro_data(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Load macro indicators from database."""
    query = f"""
        SELECT
            fecha as date,
            fxrt_index_dxy_usa_d_dxy as dxy,
            volt_vix_usa_d_vix as vix,
            crsk_spread_embi_col_d_embi as embi,
            comm_oil_brent_glb_d_brent as brent,
            finc_bond_yield10y_usa_d_ust10y as ust10y,
            finc_bond_yield10y_col_d_tes10y as tes10y,
            fxrt_spot_usdmxn_mex_d_usdmxn as usdmxn
        FROM macro_indicators_daily
        WHERE fecha >= '{start_date}' AND fecha <= '{end_date}'
        ORDER BY fecha
    """
    df = pd.read_sql(query, conn)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    logger.info(f"Loaded {len(df):,} macro rows from {start_date} to {end_date}")
    return df


# =============================================================================
# Feature Calculation
# =============================================================================

def calculate_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features using SSOT methods."""
    df = ohlcv.copy()

    # Log returns
    df['log_ret_5m'] = np.log(df['close'] / df['close'].shift(1))
    df['log_ret_1h'] = np.log(df['close'] / df['close'].shift(12))
    df['log_ret_4h'] = np.log(df['close'] / df['close'].shift(48))

    # RSI (Wilder's EMA)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    alpha = 1 / 9  # RSI period = 9
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi_9'] = 100 - (100 / (1 + rs))

    # ATR % (Wilder's EMA)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    alpha_atr = 1 / 10  # ATR period = 10
    atr = tr.ewm(alpha=alpha_atr, adjust=False).mean()
    df['atr_pct'] = atr / df['close']

    # ADX (Wilder's smoothing)
    df['adx_14'] = calculate_adx(df, period=14)

    return df[TECHNICAL_FEATURES]


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX using Wilder's smoothing."""
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilder smoothing
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


def calculate_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    """Calculate macro z-scores and changes."""
    df = macro.copy()

    # Z-scores (rolling 252-day window for daily data)
    window = 252

    # DXY z-score and change
    df['dxy_z'] = (df['dxy'] - df['dxy'].rolling(window).mean()) / df['dxy'].rolling(window).std()
    df['dxy_change_1d'] = df['dxy'].pct_change()

    # VIX z-score
    df['vix_z'] = (df['vix'] - df['vix'].rolling(window).mean()) / df['vix'].rolling(window).std()

    # EMBI z-score
    df['embi_z'] = (df['embi'] - df['embi'].rolling(window).mean()) / df['embi'].rolling(window).std()

    # Brent change
    df['brent_change_1d'] = df['brent'].pct_change()

    # Rate spread (COL 10Y - USA 10Y)
    df['rate_spread'] = df['tes10y'] - df['ust10y']

    # USD/MXN change (EM proxy)
    df['usdmxn_change_1d'] = df['usdmxn'].pct_change()

    # Select only the derived features
    feature_cols = MACRO_FEATURES_FULL  # All possible macro features
    return df[feature_cols]


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_dataset_variant(
    ohlcv: pd.DataFrame,
    macro_features: pd.DataFrame,
    variant_name: str,
    macro_columns: List[str],
    output_dir: Path
) -> Path:
    """Generate a specific dataset variant."""
    logger.info(f"Generating variant: {variant_name}")

    # Calculate technical features
    tech_features = calculate_technical_features(ohlcv)

    # Resample macro to 5min (forward fill from daily)
    macro_5min = macro_features[macro_columns].resample('5T').ffill()

    # Merge
    df = tech_features.join(macro_5min, how='left')

    # Forward fill macro (daily -> 5min)
    df[macro_columns] = df[macro_columns].ffill()

    # Drop NaN rows (warmup period)
    initial_rows = len(df)
    df = df.dropna()
    dropped = initial_rows - len(df)
    logger.info(f"  Dropped {dropped:,} rows with NaN (warmup period)")

    # Add metadata columns
    df['timestamp'] = df.index
    df['bar_idx'] = range(len(df))

    # Reorder columns: technical, macro, metadata
    feature_columns = TECHNICAL_FEATURES + macro_columns
    df = df[['timestamp', 'bar_idx'] + feature_columns]

    # Save to CSV
    output_path = output_dir / f"{variant_name}.csv"
    df.to_csv(output_path, index=False)

    # Calculate hash
    with open(output_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    logger.info(f"  Saved: {output_path}")
    logger.info(f"  Rows: {len(df):,}")
    logger.info(f"  Features: {len(feature_columns)} ({len(TECHNICAL_FEATURES)} tech + {len(macro_columns)} macro)")
    logger.info(f"  Hash: {file_hash}")

    return output_path


def generate_norm_stats(
    df: pd.DataFrame,
    variant_name: str,
    output_dir: Path
) -> Path:
    """Generate normalization statistics for a variant."""
    # Calculate stats only on training portion (first 85%)
    train_size = int(len(df) * 0.85)
    train_df = df.iloc[:train_size]

    # Get feature columns (exclude timestamp and bar_idx)
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'bar_idx']]

    norm_stats = {}
    for col in feature_cols:
        values = train_df[col].dropna()
        norm_stats[col] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    # Add metadata
    norm_stats["_metadata"] = {
        "variant": variant_name,
        "created_at": datetime.now().isoformat(),
        "feature_count": len(feature_cols),
        "train_samples": train_size,
    }

    # Determine output filename
    if variant_name == "RL_DS3_MACRO_CORE":
        filename = "norm_stats_reduced.json"
    else:
        filename = "norm_stats.json"

    output_path = output_dir / filename

    with open(output_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)

    logger.info(f"  Norm stats saved: {output_path}")
    return output_path


# =============================================================================
# DVC Integration
# =============================================================================

def run_dvc_command(args: List[str], cwd: Path = None) -> Tuple[bool, str]:
    """Run a DVC command and return success status and output."""
    try:
        result = subprocess.run(
            ["dvc"] + args,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "DVC command timed out"
    except FileNotFoundError:
        return False, "DVC not installed or not in PATH"


def version_dataset_with_dvc(
    dataset_path: Path,
    variant_name: str,
    file_hash: str
) -> Dict[str, str]:
    """
    Version a dataset with DVC.

    Steps:
    1. dvc add <dataset>
    2. dvc push
    3. git tag dataset-exp/<variant>_v<timestamp>

    Returns:
        Dict with dvc_file, md5_hash, git_tag
    """
    logger.info(f"  Versioning with DVC: {dataset_path.name}")

    result = {
        "dvc_file": None,
        "md5_hash": None,
        "git_tag": None,
        "pushed": False,
        "errors": []
    }

    # Step 1: DVC add
    success, output = run_dvc_command(["add", str(dataset_path)])
    if success:
        dvc_file = dataset_path.with_suffix(dataset_path.suffix + ".dvc")
        result["dvc_file"] = str(dvc_file)
        logger.info(f"    ✓ dvc add: {dvc_file.name}")

        # Extract MD5 from .dvc file
        if dvc_file.exists():
            with open(dvc_file) as f:
                dvc_content = f.read()
                import re
                md5_match = re.search(r"md5:\s*([a-f0-9]+)", dvc_content)
                if md5_match:
                    result["md5_hash"] = md5_match.group(1)
    else:
        logger.warning(f"    ✗ dvc add failed: {output}")
        result["errors"].append(f"dvc add: {output}")

    # Step 2: DVC push
    success, output = run_dvc_command(["push", str(dataset_path)])
    if success:
        result["pushed"] = True
        logger.info(f"    ✓ dvc push: uploaded to MinIO")
    else:
        logger.warning(f"    ✗ dvc push failed: {output}")
        result["errors"].append(f"dvc push: {output}")

    # Step 3: Create git tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_name = f"dataset-exp/{variant_name.lower()}_v{timestamp}"

    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Dataset variant: {variant_name}\nHash: {file_hash}"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        result["git_tag"] = tag_name
        logger.info(f"    ✓ git tag: {tag_name}")
    except Exception as e:
        logger.warning(f"    ✗ git tag failed: {e}")
        result["errors"].append(f"git tag: {str(e)}")

    return result


def generate_dvc_manifest(
    variants_info: Dict[str, Dict],
    output_dir: Path
) -> Path:
    """Generate a manifest file with all DVC versioning info."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "variants": variants_info,
        "dvc_remote": "minio",
        "instructions": {
            "checkout": "dvc checkout",
            "pull": "dvc pull",
            "diff": "dvc diff <tag1> <tag2>"
        }
    }

    manifest_path = output_dir / "dvc_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"  DVC manifest saved: {manifest_path}")
    return manifest_path


def commit_dvc_files_to_git(
    output_dir: Path,
    variants_info: Dict[str, Dict]
) -> bool:
    """
    Commit .dvc files to git for traceability.

    Implements GAP Q6: Auto git commit of .dvc files after generation.

    Args:
        output_dir: Directory containing .dvc files
        variants_info: Dict with variant metadata for commit message

    Returns:
        True if commit successful, False otherwise
    """
    logger.info("Committing .dvc files to git...")

    # Find all .dvc files
    dvc_files = list(output_dir.glob("*.dvc"))
    if not dvc_files:
        logger.warning("No .dvc files found to commit")
        return False

    # Also include .gitignore updates from DVC
    gitignore_path = output_dir / ".gitignore"

    try:
        # Step 1: git add .dvc files
        files_to_add = [str(f) for f in dvc_files]
        if gitignore_path.exists():
            files_to_add.append(str(gitignore_path))

        result = subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.warning(f"git add failed: {result.stderr}")
            return False

        logger.info(f"  ✓ git add: {len(dvc_files)} .dvc files")

        # Step 2: Create commit message
        variant_names = list(variants_info.keys())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_msg = f"""Add DVC tracking for A/B test dataset variants

Variants: {', '.join(variant_names)}
Generated: {timestamp}

Files tracked:
{chr(10).join(f'  - {f.name}' for f in dvc_files)}

This commit enables:
- Dataset versioning via DVC
- Reproducible experiment comparisons
- Artifact traceability in MLflow
"""

        # Step 3: git commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("  ✓ git commit: .dvc files committed")
            return True
        elif "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            logger.info("  ✓ No changes to commit (files already tracked)")
            return True
        else:
            logger.warning(f"git commit failed: {result.stderr}")
            return False

    except Exception as e:
        logger.warning(f"Git operations failed: {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all dataset variants with DVC versioning."""
    logger.info("=" * 60)
    logger.info("DATASET VARIANT GENERATOR (with DVC)")
    logger.info("=" * 60)

    # Configuration
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    output_dir = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "datasets_5min"
    config_dir = PROJECT_ROOT / "config"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    logger.info("Connecting to database...")
    conn = get_db_connection()

    # Track DVC versioning info for all variants
    dvc_versions: Dict[str, Dict] = {}

    try:
        # Load raw data
        ohlcv = load_ohlcv_data(conn, start_date, end_date)
        macro = load_macro_data(conn, start_date, end_date)

        # Calculate macro features
        macro_features = calculate_macro_features(macro)

        # Generate each variant
        for variant_name, config in DATASET_VARIANTS.items():
            logger.info("-" * 40)
            logger.info(f"Variant: {variant_name}")
            logger.info(f"  {config['description']}")
            logger.info(f"  Observation dim: {config['observation_dim']}")

            # Generate dataset
            output_path = generate_dataset_variant(
                ohlcv=ohlcv,
                macro_features=macro_features,
                variant_name=variant_name,
                macro_columns=config['macro_features'],
                output_dir=output_dir,
            )

            # Calculate file hash for versioning
            with open(output_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()[:16]

            # Load and generate norm stats
            df = pd.read_csv(output_path)
            generate_norm_stats(df, variant_name, config_dir)

            # Version with DVC
            logger.info("-" * 40)
            logger.info(f"DVC Versioning: {variant_name}")
            dvc_result = version_dataset_with_dvc(
                dataset_path=output_path,
                variant_name=variant_name,
                file_hash=file_hash
            )

            dvc_versions[variant_name] = {
                "dataset_path": str(output_path),
                "file_hash": file_hash,
                "row_count": len(df),
                "feature_count": len(config['macro_features']) + len(TECHNICAL_FEATURES),
                "date_range": {
                    "start": start_date,
                    "end": end_date
                },
                "dvc": dvc_result
            }

    finally:
        conn.close()

    # Generate DVC manifest
    logger.info("-" * 40)
    generate_dvc_manifest(dvc_versions, output_dir)

    # Commit .dvc files to git (Q6: Auto git commit)
    logger.info("-" * 40)
    commit_dvc_files_to_git(output_dir, dvc_versions)

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Generated files:")
    for f in output_dir.glob("*.csv"):
        logger.info(f"  • {f}")
    logger.info("")
    logger.info("DVC files:")
    for f in output_dir.glob("*.dvc"):
        logger.info(f"  • {f}")
    logger.info("")
    logger.info("Norm stats:")
    for f in config_dir.glob("norm_stats*.json"):
        logger.info(f"  • {f}")
    logger.info("")
    logger.info("Git tags created:")
    for variant, info in dvc_versions.items():
        if info["dvc"].get("git_tag"):
            logger.info(f"  • {info['dvc']['git_tag']}")


if __name__ == "__main__":
    main()
