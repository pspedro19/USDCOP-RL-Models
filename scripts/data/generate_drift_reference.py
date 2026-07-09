#!/usr/bin/env python3
"""
Generate Drift Reference Data
=============================

Extracts historical observations from training data to initialize
the multivariate drift detector.

This script:
1. Loads training data (parquet/csv)
2. Extracts features in FEATURE_ORDER
3. Normalizes using norm_stats.json
4. Saves reference observations as JSON
5. Optionally uploads to API

Usage:
    python scripts/generate_drift_reference.py
    python scripts/generate_drift_reference.py --samples 1000 --upload
    python scripts/generate_drift_reference.py --input data/training.parquet

Author: Trading Team
Date: 2026-01-17
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_feature_order() -> List[str]:
    """Load FEATURE_ORDER from SSOT."""
    try:
        from src.core.contracts import FEATURE_ORDER
        return list(FEATURE_ORDER)
    except ImportError:
        logger.warning("Could not import FEATURE_ORDER, using default")
        return [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14", "dxy_z",
            "vix_z", "wti_z", "hour_sin", "hour_cos",
            "dow_sin", "dow_cos", "is_ny_session", "is_london_overlap"
        ]


def load_norm_stats(path: Path) -> Dict:
    """Load normalization statistics."""
    if not path.exists():
        raise FileNotFoundError(f"Norm stats not found: {path}")

    with open(path, 'r') as f:
        return json.load(f)


def normalize_features(
    data: np.ndarray,
    feature_names: List[str],
    norm_stats: Dict
) -> np.ndarray:
    """
    Normalize features using z-score normalization.

    Args:
        data: 2D array of shape (n_samples, n_features)
        feature_names: List of feature names
        norm_stats: Dictionary with mean/std per feature

    Returns:
        Normalized array
    """
    normalized = np.zeros_like(data)

    for i, feat in enumerate(feature_names):
        if feat in norm_stats:
            mean = norm_stats[feat].get('mean', 0.0)
            std = norm_stats[feat].get('std', 1.0)
            if std == 0:
                std = 1.0
            normalized[:, i] = (data[:, i] - mean) / std
        else:
            logger.warning(f"Feature {feat} not in norm_stats, using raw values")
            normalized[:, i] = data[:, i]

    return normalized


def load_training_data(
    path: Path,
    feature_names: List[str],
    n_samples: int = 1000
) -> np.ndarray:
    """
    Load training data from parquet or CSV.

    Args:
        path: Path to data file
        feature_names: Features to extract
        n_samples: Number of samples to extract

    Returns:
        2D array of observations
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    logger.info(f"Loading data from {path}")

    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Check which features are available
    available = [f for f in feature_names if f in df.columns]
    missing = [f for f in feature_names if f not in df.columns]

    if missing:
        logger.warning(f"Missing features: {missing}")

    if not available:
        raise ValueError("No features found in data")

    logger.info(f"Using {len(available)} features: {available[:5]}...")

    # Extract features
    data = df[available].values

    # Remove NaN rows
    valid_mask = ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    logger.info(f"After NaN removal: {len(data)} rows")

    # Sample if needed
    if len(data) > n_samples:
        # Random sample from across the dataset
        indices = np.random.choice(len(data), n_samples, replace=False)
        indices.sort()  # Keep temporal order
        data = data[indices]
        logger.info(f"Sampled {n_samples} observations")

    # Pad missing features with zeros
    if len(available) < len(feature_names):
        full_data = np.zeros((len(data), len(feature_names)))
        for i, feat in enumerate(feature_names):
            if feat in available:
                full_data[:, i] = data[:, available.index(feat)]
        data = full_data

    return data


def generate_synthetic_reference(
    n_samples: int,
    feature_names: List[str],
    norm_stats: Dict
) -> np.ndarray:
    """
    Generate synthetic reference data based on norm_stats.

    Use this when training data is not available.
    """
    logger.info(f"Generating synthetic reference with {n_samples} samples")

    data = np.zeros((n_samples, len(feature_names)))

    for i, feat in enumerate(feature_names):
        if feat in norm_stats:
            mean = norm_stats[feat].get('mean', 0.0)
            std = norm_stats[feat].get('std', 1.0)
            data[:, i] = np.random.normal(mean, std, n_samples)
        else:
            # Standard normal for unknown features
            data[:, i] = np.random.randn(n_samples)

    return data


def save_reference_data(
    data: np.ndarray,
    output_path: Path,
    feature_names: List[str]
) -> None:
    """Save reference data to JSON file."""
    # Convert to list format for JSON
    reference = {
        "observations": data.tolist(),
        "feature_order": feature_names,
        "n_samples": len(data),
        "n_features": len(feature_names),
        "generated_at": __import__('datetime').datetime.utcnow().isoformat(),
    }

    with open(output_path, 'w') as f:
        json.dump(reference, f, indent=2)

    logger.info(f"Saved reference data to {output_path}")


def upload_to_api(
    data: np.ndarray,
    api_url: str = "http://localhost:8000"
) -> bool:
    """Upload reference data to drift detection API."""
    try:
        import requests
    except ImportError:
        logger.error("requests is required for upload: pip install requests")
        return False

    endpoint = f"{api_url}/api/v1/monitoring/drift/reference/multivariate"

    logger.info(f"Uploading to {endpoint}")

    try:
        response = requests.post(
            endpoint,
            json=data.tolist(),
            headers={"Content-Type": "application/json"},
            timeout=60,
        )

        if response.ok:
            result = response.json()
            logger.info(f"Upload successful: {result}")
            return True
        else:
            logger.error(f"Upload failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate drift reference data from training data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to training data (parquet/csv). If not provided, generates synthetic data."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="config/drift_reference.json",
        help="Output path for reference data"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=500,
        help="Number of reference samples (default: 500)"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features using norm_stats.json"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to drift detection API"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API base URL for upload"
    )

    args = parser.parse_args()

    # Load feature order
    feature_names = load_feature_order()
    logger.info(f"Using {len(feature_names)} features from FEATURE_ORDER")

    # Load norm stats
    norm_stats_path = PROJECT_ROOT / "config" / "norm_stats.json"
    norm_stats = {}
    if norm_stats_path.exists():
        norm_stats = load_norm_stats(norm_stats_path)
        logger.info(f"Loaded norm stats for {len(norm_stats)} features")

    # Load or generate data
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)

        data = load_training_data(input_path, feature_names, args.samples)
    else:
        logger.info("No input file provided, generating synthetic reference data")
        data = generate_synthetic_reference(args.samples, feature_names, norm_stats)

    # Normalize if requested
    if args.normalize and norm_stats:
        logger.info("Normalizing features")
        data = normalize_features(data, feature_names, norm_stats)

    # Save to file
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_reference_data(data, output_path, feature_names)

    # Upload if requested
    if args.upload:
        success = upload_to_api(data, args.api_url)
        if not success:
            logger.warning("Upload failed, but reference file was saved")

    # Print summary
    print("\n" + "=" * 50)
    print("DRIFT REFERENCE DATA GENERATED")
    print("=" * 50)
    print(f"Samples: {len(data)}")
    print(f"Features: {len(feature_names)}")
    print(f"Output: {output_path}")
    print(f"Normalized: {args.normalize}")
    if args.upload:
        print(f"Uploaded: {'Yes' if success else 'No'}")
    print("=" * 50)

    # Show usage instructions
    print("\nTo load into API manually:")
    print(f"  curl -X POST {args.api_url}/api/v1/monitoring/drift/reference/multivariate \\")
    print(f"    -H 'Content-Type: application/json' \\")
    print(f"    -d @{output_path}")


if __name__ == "__main__":
    main()
