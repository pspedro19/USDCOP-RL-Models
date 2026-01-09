#!/usr/bin/env python3
"""
L6 Router - Backtest Results Endpoints
=======================================
Data source: MinIO (bucket: usdcop, prefix: l6/) OR PostgreSQL (l6_backtest_results table)
Format: Parquet (trades, returns) + JSON (KPIs)

Endpoints:
- GET /backtest/l6/results - Backtest performance metrics
- GET /backtest/l6/trades - Individual trade details
- GET /backtest/l6/equity-curve - Equity curve time series
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
    execute_sql_query,
    load_quality_gates
)

router = APIRouter(prefix="/backtest/l6", tags=["backtest-l6"])


@router.get("/results")
def get_l6_results(
    model_id: str = Query(..., description="Model ID (e.g., 'ppo_v1.2.3')"),
    split: str = Query("test", description="train|val|test"),
    run_id: Optional[str] = None
):
    """
    L6: Backtest results for a specific model

    Returns:
    - Performance metrics (Sortino, Sharpe, Calmar, MaxDD)
    - Trade statistics (win rate, profit factor)
    - Cost analysis (total costs, % of PnL)

    Data source: MinIO (preferred) or PostgreSQL (fallback)
    """
    config = get_layer_config("l6")
    gates = load_quality_gates().get("l6", {})

    # Try MinIO first
    if config["backend"] == "s3":
        try:
            bucket = config["bucket"]
            prefix = config["prefix"]

            # Get manifest
            if run_id is None:
                manifest = read_latest_manifest(bucket, prefix)
            else:
                manifest = read_run_manifest(bucket, prefix, run_id)

            # Read KPIs JSON
            kpi_file = next(
                (f for f in manifest["files"] if f["name"] == "kpis.json"),
                None
            )

            if kpi_file:
                kpis = read_json_metadata(bucket, kpi_file["path"])

                # Filter by model_id and split
                if model_id in kpis and split in kpis[model_id]:
                    result = kpis[model_id][split]

                    return {
                        "status": "OK",
                        "layer": "l6",
                        "source": "minio",
                        "run_id": manifest["run_id"],
                        "model_id": model_id,
                        "split": split,
                        "performance": {
                            "sortino": result.get("sortino"),
                            "sharpe": result.get("sharpe"),
                            "calmar": result.get("calmar"),
                            "max_drawdown": result.get("max_drawdown"),
                            "total_return": result.get("total_return")
                        },
                        "trades": {
                            "total": result.get("total_trades"),
                            "winning": result.get("winning_trades"),
                            "losing": result.get("losing_trades"),
                            "win_rate": result.get("win_rate"),
                            "profit_factor": result.get("profit_factor")
                        },
                        "costs": {
                            "total_paid": result.get("total_costs"),
                            "pct_of_pnl": result.get("costs_pct_pnl")
                        },
                        "quality_gates": gates,
                        "pass": result.get("sortino", 0) >= gates.get("sortino", 1.3)
                    }

        except:
            pass  # Fallback to PostgreSQL

    # Fallback to PostgreSQL
    if "fallback" in config and config["fallback"]["backend"] == "postgres":
        table = config["fallback"]["table"]

        query = f"""
            SELECT
                split, model_id, sortino, sharpe, calmar, maxdd as max_drawdown,
                trades, winning_trades, losing_trades,
                win_rate, profit_factor,
                costs_paid as total_costs,
                (costs_paid / NULLIF(total_pnl, 0) * 100) as costs_pct_pnl
            FROM {table}
            WHERE model_id = :model_id AND split = :split
            ORDER BY created_at DESC
            LIMIT 1
        """

        df = execute_sql_query(query, {"model_id": model_id, "split": split})

        if df.empty:
            return {
                "status": "NOT_FOUND",
                "message": f"No backtest results for model {model_id}, split {split}"
            }

        row = df.iloc[0].to_dict()

        return {
            "status": "OK",
            "layer": "l6",
            "source": "postgres",
            "model_id": model_id,
            "split": split,
            "performance": {
                "sortino": float(row["sortino"]) if row["sortino"] else None,
                "sharpe": float(row["sharpe"]) if row["sharpe"] else None,
                "calmar": float(row["calmar"]) if row["calmar"] else None,
                "max_drawdown": float(row["max_drawdown"]) if row["max_drawdown"] else None
            },
            "trades": {
                "total": int(row["trades"]) if row["trades"] else None,
                "winning": int(row["winning_trades"]) if row["winning_trades"] else None,
                "losing": int(row["losing_trades"]) if row["losing_trades"] else None,
                "win_rate": float(row["win_rate"]) if row["win_rate"] else None,
                "profit_factor": float(row["profit_factor"]) if row["profit_factor"] else None
            },
            "costs": {
                "total_paid": float(row["total_costs"]) if row["total_costs"] else None,
                "pct_of_pnl": float(row["costs_pct_pnl"]) if row["costs_pct_pnl"] else None
            },
            "quality_gates": gates,
            "pass": (row["sortino"] or 0) >= gates.get("sortino", 1.3)
        }

    # No data found
    return {
        "status": "NOT_FOUND",
        "message": "No backtest results found in MinIO or PostgreSQL"
    }


@router.get("/trades")
def get_l6_trades(
    model_id: str,
    split: str = "test",
    run_id: Optional[str] = None,
    limit: int = Query(100, le=10000)
):
    """
    L6: Individual trade details

    Returns list of all trades with entry/exit/pnl
    """
    config = get_layer_config("l6")

    if config["backend"] == "s3":
        try:
            bucket = config["bucket"]
            prefix = config["prefix"]

            # Get manifest
            if run_id is None:
                manifest = read_latest_manifest(bucket, prefix)
            else:
                manifest = read_run_manifest(bucket, prefix, run_id)

            # Read trades.parquet
            trades_file = next(
                (f for f in manifest["files"] if f["name"] == "trades.parquet"),
                None
            )

            if trades_file:
                df = read_parquet_dataset(bucket, trades_file["path"])

                # Filter by model_id and split if columns exist
                if "model_id" in df.columns:
                    df = df[df["model_id"] == model_id]
                if "split" in df.columns:
                    df = df[df["split"] == split]

                # Limit
                df = df.head(limit)

                return {
                    "status": "OK",
                    "layer": "l6",
                    "model_id": model_id,
                    "split": split,
                    "total_trades": len(df),
                    "trades": df.to_dict(orient="records")
                }

        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    return {
        "status": "NOT_FOUND",
        "message": "Trade details not available"
    }


@router.get("/equity-curve")
def get_l6_equity_curve(
    model_id: str,
    split: str = "test",
    run_id: Optional[str] = None
):
    """
    L6: Equity curve time series

    Returns cumulative PnL over time
    """
    config = get_layer_config("l6")

    if config["backend"] == "s3":
        try:
            bucket = config["bucket"]
            prefix = config["prefix"]

            # Get manifest
            if run_id is None:
                manifest = read_latest_manifest(bucket, prefix)
            else:
                manifest = read_run_manifest(bucket, prefix, run_id)

            # Read equity_curve.json
            equity_file = next(
                (f for f in manifest["files"] if f["name"] == "equity_curve.json"),
                None
            )

            if equity_file:
                equity_data = read_json_metadata(bucket, equity_file["path"])

                # Filter by model_id and split
                if model_id in equity_data and split in equity_data[model_id]:
                    curve = equity_data[model_id][split]

                    return {
                        "status": "OK",
                        "layer": "l6",
                        "model_id": model_id,
                        "split": split,
                        "equity_curve": curve,
                        "data_points": len(curve.get("timestamps", []))
                    }

        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    return {
        "status": "NOT_FOUND",
        "message": "Equity curve not available"
    }
