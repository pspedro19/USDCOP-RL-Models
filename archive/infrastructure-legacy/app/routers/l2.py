#!/usr/bin/env python3
"""
L2 Router - Prepared Data Endpoints
====================================
Data source: MinIO (bucket: usdcop, prefix: l2/)
Format: Parquet (strict/flex variants)

Endpoints:
- GET /pipeline/l2/prepared - Winsorization, HOD, NaN stats
- GET /pipeline/l2/contract - L2 data contract
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
    load_quality_gates,
    calculate_etag
)

router = APIRouter(prefix="/pipeline/l2", tags=["pipeline-l2"])


@router.get("/prepared")
def get_l2_prepared_data(
    run_id: Optional[str] = Query(None, description="Specific run ID (default: latest)"),
    variant: str = Query("strict", description="strict or flex")
):
    """
    L2: Prepared data statistics

    Returns:
    - Winsorization rate
    - HOD deseasonalization stats (median, MAD)
    - NaN rate
    - Indicator count
    - GO/NO-GO status

    Quality Gates:
    - Winsorization rate <= 1.0%
    - HOD MAD in [0.8, 1.2]
    - HOD median abs <= 0.05
    - NaN rate <= 0.5%
    """
    config = get_layer_config("l2")
    gates = load_quality_gates().get("l2", {})

    # Get manifest
    bucket = config["bucket"]
    prefix = config["prefix"]

    if run_id is None:
        manifest = read_latest_manifest(bucket, prefix)
    else:
        manifest = read_run_manifest(bucket, prefix, run_id)

    # Read data file
    filename = f"data_premium_{variant}.parquet"
    data_file = next(
        (f for f in manifest["files"] if f["name"] == filename),
        None
    )

    if not data_file:
        return {
            "status": "NOT_FOUND",
            "message": f"No {variant} variant found for run_id {manifest['run_id']}"
        }

    # Read with selected columns for efficiency
    df = read_parquet_dataset(
        bucket,
        data_file["path"],
        columns=["ret_z", "winsor_flag", "hod_median", "hod_mad"]
        if variant == "strict" else None
    )

    # Calculate statistics
    winsor_rate = float(df["winsor_flag"].mean() * 100) if "winsor_flag" in df.columns else 0.0

    hod_median_abs = float(df["hod_median"].abs().median()) if "hod_median" in df.columns else 0.0
    hod_mad_mean = float(df["hod_mad"].mean()) if "hod_mad" in df.columns else 0.0

    nan_rate = float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)

    # Indicator count (columns that aren't basic OHLCV or metadata)
    basic_cols = {'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
                  'datetime', 'ret_z', 'winsor_flag', 'hod_median', 'hod_mad'}
    indicator_count = len([col for col in df.columns if col not in basic_cols])

    # GO/NO-GO
    passed = (
        winsor_rate <= 1.0 and
        0.8 <= hod_mad_mean <= 1.2 and
        hod_median_abs <= 0.05 and
        nan_rate <= 0.5
    )

    return {
        "status": "OK",
        "layer": "l2",
        "run_id": manifest["run_id"],
        "variant": variant,
        "dataset_hash": manifest.get("dataset_hash"),
        "quality_metrics": {
            "winsorization_rate_pct": round(winsor_rate, 3),
            "hod_median_abs": round(hod_median_abs, 4),
            "hod_mad_mean": round(hod_mad_mean, 3),
            "nan_rate_pct": round(nan_rate, 3),
            "indicator_count": indicator_count
        },
        "data_shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "quality_gates": gates,
        "pass": passed,
        "note": "Proxy data - winsorization & HOD based on OHLCV"
    }


@router.get("/contract")
def get_l2_contract():
    """
    L2: Data contract specification

    Returns schema and quality requirements
    """
    config = get_layer_config("l2")

    return {
        "layer": "l2",
        "description": config["description"],
        "backend": config["backend"],
        "location": {
            "bucket": config["bucket"],
            "prefix": config["prefix"]
        },
        "file_format": config["file_format"],
        "variants": config.get("variants", {}),
        "required_columns": [
            "timestamp",
            "open", "high", "low", "close", "volume",
            "ret_z",
            "winsor_flag",
            "hod_median",
            "hod_mad"
        ],
        "technical_indicators": {
            "count": "60+",
            "categories": [
                "Momentum (RSI, CCI, Williams %R, ROC)",
                "Trend (ADX, AROON, DI+/-)",
                "Moving Averages (SMA/EMA 5,10,20,50,100,200)",
                "Oscillators (MACD, Stochastic)",
                "Volatility (ATR, NATR, Bollinger Bands)",
                "Volume (OBV, AD)"
            ]
        },
        "quality_gates": load_quality_gates().get("l2", {}),
        "metadata_files": config.get("metadata_files", []),
        "typical_file_size_mb": "10-100 (depends on date range)"
    }


@router.get("/indicators")
def get_l2_indicators(run_id: Optional[str] = None, limit: int = Query(100, le=10000)):
    """
    Get sample of L2 data with all technical indicators

    Args:
        run_id: Specific run ID (default: latest)
        limit: Number of rows to return

    Returns:
        Sample data with all calculated indicators
    """
    config = get_layer_config("l2")
    bucket = config["bucket"]
    prefix = config["prefix"]

    # Get manifest
    if run_id is None:
        manifest = read_latest_manifest(bucket, prefix)
    else:
        manifest = read_run_manifest(bucket, prefix, run_id)

    # Read strict variant
    data_file = next(
        (f for f in manifest["files"] if f["name"] == "data_premium_strict.parquet"),
        None
    )

    if not data_file:
        return {"status": "NOT_FOUND", "message": "No strict variant found"}

    # Read all columns (no filter)
    df = read_parquet_dataset(bucket, data_file["path"])

    # Take sample
    sample = df.head(limit)

    return {
        "status": "OK",
        "layer": "l2",
        "run_id": manifest["run_id"],
        "columns": list(df.columns),
        "column_count": len(df.columns),
        "sample_size": len(sample),
        "total_rows": len(df),
        "data": sample.to_dict(orient="records")
    }
