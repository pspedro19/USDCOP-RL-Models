#!/usr/bin/env python3
"""
Migrate Forecasting CSV Data to PostgreSQL
==========================================

This script migrates historical forecasting data from CSV files to PostgreSQL.
It's designed to be run once to bootstrap the database with existing predictions.

Usage:
    python scripts/migrate_forecasting_csv_to_db.py
    python scripts/migrate_forecasting_csv_to_db.py --csv-path /path/to/data.csv
    python scripts/migrate_forecasting_csv_to_db.py --dry-run

Tables Created/Populated:
    - bi.dim_models: Model catalog
    - bi.dim_horizons: Horizon catalog
    - bi.fact_forecasts: Historical predictions
    - bi.fact_consensus: Consensus by horizon
    - bi.fact_model_metrics: Walk-forward metrics (if available)

Author: Trading Team
Date: 2026-01-22
Contract: CTR-FORECASTING-001
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# SSOT IMPORTS
# =============================================================================

try:
    from src.forecasting.contracts import (
        HORIZONS,
        MODEL_IDS,
        MODEL_DEFINITIONS,
        HORIZON_LABELS,
        FORECASTING_CONTRACT_VERSION,
    )
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    HORIZONS = (1, 5, 10, 15, 20, 25, 30)
    MODEL_IDS = ("ridge", "bayesian_ridge", "ard", "xgboost_pure", "lightgbm_pure",
                 "catboost_pure", "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost")
    MODEL_DEFINITIONS = {}
    HORIZON_LABELS = {1: "1 day", 5: "5 days", 10: "10 days", 15: "15 days",
                      20: "20 days", 25: "25 days", 30: "30 days"}
    FORECASTING_CONTRACT_VERSION = "1.0.0"


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Get database connection from environment."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable not set. "
            "Set it to your PostgreSQL connection string."
        )
    return psycopg2.connect(database_url)


def create_schema_and_tables(conn) -> None:
    """Create the bi schema and required tables if they don't exist."""
    cur = conn.cursor()

    logger.info("Creating schema and tables...")

    cur.execute("""
        -- Create schema
        CREATE SCHEMA IF NOT EXISTS bi;

        -- Dimension: Models
        CREATE TABLE IF NOT EXISTS bi.dim_models (
            model_id VARCHAR(100) PRIMARY KEY,
            model_name VARCHAR(200) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            version VARCHAR(50),
            requires_scaling BOOLEAN DEFAULT FALSE,
            supports_early_stopping BOOLEAN DEFAULT FALSE,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Dimension: Horizons
        CREATE TABLE IF NOT EXISTS bi.dim_horizons (
            horizon INT PRIMARY KEY,
            horizon_label VARCHAR(50) NOT NULL,
            category VARCHAR(20),  -- short, medium, long
            description TEXT
        );

        -- Fact: Forecasts (individual predictions)
        CREATE TABLE IF NOT EXISTS bi.fact_forecasts (
            id SERIAL PRIMARY KEY,
            inference_date DATE NOT NULL,
            inference_week INT,
            inference_year INT,
            model_id VARCHAR(100) NOT NULL,
            horizon INT NOT NULL,
            base_price DECIMAL(12,4),
            predicted_price DECIMAL(12,4),
            predicted_return_pct DECIMAL(10,6),
            direction VARCHAR(10),  -- UP, DOWN
            signal INT,  -- -1=SELL, 0=HOLD, 1=BUY
            confidence DECIMAL(5,4),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(inference_date, model_id, horizon)
        );

        -- Fact: Consensus (aggregated by horizon)
        CREATE TABLE IF NOT EXISTS bi.fact_consensus (
            id SERIAL PRIMARY KEY,
            inference_date DATE NOT NULL,
            horizon INT NOT NULL,
            bullish_count INT,
            bearish_count INT,
            consensus_direction VARCHAR(10),  -- BULLISH, BEARISH, NEUTRAL
            consensus_strength DECIMAL(5,4),
            avg_predicted_return DECIMAL(10,6),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(inference_date, horizon)
        );

        -- Fact: Model Metrics (walk-forward performance)
        CREATE TABLE IF NOT EXISTS bi.fact_model_metrics (
            id SERIAL PRIMARY KEY,
            model_id VARCHAR(100) NOT NULL,
            horizon INT NOT NULL,
            version VARCHAR(50),
            metric_date DATE,
            direction_accuracy DECIMAL(5,4),
            rmse DECIMAL(10,6),
            mae DECIMAL(10,6),
            mape DECIMAL(10,6),
            sharpe_ratio DECIMAL(10,6),
            max_drawdown DECIMAL(10,6),
            profit_factor DECIMAL(10,6),
            sample_count INT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_fact_forecasts_date
            ON bi.fact_forecasts (inference_date DESC);
        CREATE INDEX IF NOT EXISTS idx_fact_forecasts_model
            ON bi.fact_forecasts (model_id, inference_date DESC);
        CREATE INDEX IF NOT EXISTS idx_fact_consensus_date
            ON bi.fact_consensus (inference_date DESC);
        CREATE INDEX IF NOT EXISTS idx_fact_model_metrics_model
            ON bi.fact_model_metrics (model_id, horizon);

        -- View: Latest forecasts
        CREATE OR REPLACE VIEW bi.v_latest_forecasts AS
        SELECT f.*, m.model_name, h.horizon_label
        FROM bi.fact_forecasts f
        LEFT JOIN bi.dim_models m ON f.model_id = m.model_id
        LEFT JOIN bi.dim_horizons h ON f.horizon = h.horizon
        WHERE f.inference_date = (
            SELECT MAX(inference_date) FROM bi.fact_forecasts
        );

        -- View: Latest consensus
        CREATE OR REPLACE VIEW bi.v_latest_consensus AS
        SELECT c.*, h.horizon_label
        FROM bi.fact_consensus c
        LEFT JOIN bi.dim_horizons h ON c.horizon = h.horizon
        WHERE c.inference_date = (
            SELECT MAX(inference_date) FROM bi.fact_consensus
        );
    """)

    conn.commit()
    logger.info("Schema and tables created successfully")


def seed_dimensions(conn) -> None:
    """Seed dimension tables with SSOT data."""
    cur = conn.cursor()

    logger.info("Seeding dimension tables...")

    # Seed horizons
    for horizon in HORIZONS:
        category = "short" if horizon <= 5 else ("medium" if horizon <= 15 else "long")
        cur.execute("""
            INSERT INTO bi.dim_horizons (horizon, horizon_label, category)
            VALUES (%s, %s, %s)
            ON CONFLICT (horizon) DO UPDATE SET
                horizon_label = EXCLUDED.horizon_label,
                category = EXCLUDED.category
        """, (
            horizon,
            HORIZON_LABELS.get(horizon, f"{horizon} days"),
            category,
        ))

    # Seed models
    for model_id in MODEL_IDS:
        info = MODEL_DEFINITIONS.get(model_id, {})
        cur.execute("""
            INSERT INTO bi.dim_models
            (model_id, model_name, model_type, version, requires_scaling, supports_early_stopping)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO UPDATE SET
                model_name = EXCLUDED.model_name,
                model_type = EXCLUDED.model_type,
                updated_at = NOW()
        """, (
            model_id,
            info.get("name", model_id),
            info.get("type", "unknown").value if hasattr(info.get("type"), "value") else str(info.get("type", "unknown")),
            FORECASTING_CONTRACT_VERSION,
            info.get("requires_scaling", False),
            info.get("supports_early_stopping", False),
        ))

    conn.commit()
    logger.info(f"Seeded {len(HORIZONS)} horizons and {len(MODEL_IDS)} models")


# =============================================================================
# CSV LOADING
# =============================================================================

def find_csv_files() -> List[Path]:
    """Find all CSV files with forecasting data."""
    search_paths = [
        PROJECT_ROOT / "data" / "forecasting",
        PROJECT_ROOT / "NewFeature" / "consolidated_backend" / "outputs",
        PROJECT_ROOT / "NewFeature" / "consolidated_backend" / "data",
    ]

    csv_files = []
    for path in search_paths:
        if path.exists():
            csv_files.extend(path.glob("**/bi_dashboard*.csv"))
            csv_files.extend(path.glob("**/forecasts*.csv"))
            csv_files.extend(path.glob("**/predictions*.csv"))

    return list(set(csv_files))  # Remove duplicates


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load and normalize CSV data."""
    logger.info(f"Loading CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize column names
    column_mapping = {
        "date": "inference_date",
        "model": "model_id",
        "pred_return": "predicted_return_pct",
        "return_pct": "predicted_return_pct",
        "dir": "direction",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Ensure required columns
    required = ["model_id", "horizon", "inference_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Parse dates
    if "inference_date" in df.columns:
        df["inference_date"] = pd.to_datetime(df["inference_date"]).dt.date

    # Add week/year if not present
    if "inference_week" not in df.columns and "inference_date" in df.columns:
        df["inference_week"] = pd.to_datetime(df["inference_date"]).dt.isocalendar().week
    if "inference_year" not in df.columns and "inference_date" in df.columns:
        df["inference_year"] = pd.to_datetime(df["inference_date"]).dt.year

    logger.info(f"Loaded {len(df)} rows from CSV")
    return df


# =============================================================================
# MIGRATION
# =============================================================================

def migrate_forecasts(conn, df: pd.DataFrame, dry_run: bool = False) -> int:
    """Migrate forecast data to PostgreSQL."""
    cur = conn.cursor()

    logger.info(f"Migrating {len(df)} forecasts...")

    if dry_run:
        logger.info("[DRY RUN] Would insert {} rows".format(len(df)))
        return 0

    inserted = 0
    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT INTO bi.fact_forecasts
                (inference_date, inference_week, inference_year, model_id, horizon,
                 base_price, predicted_return_pct, direction, signal, confidence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (inference_date, model_id, horizon) DO UPDATE SET
                    predicted_return_pct = EXCLUDED.predicted_return_pct,
                    direction = EXCLUDED.direction,
                    signal = EXCLUDED.signal
            """, (
                row.get("inference_date"),
                row.get("inference_week"),
                row.get("inference_year"),
                row.get("model_id"),
                row.get("horizon"),
                row.get("base_price"),
                row.get("predicted_return_pct"),
                row.get("direction"),
                row.get("signal"),
                row.get("confidence"),
            ))
            inserted += 1
        except Exception as e:
            logger.warning(f"Error inserting row: {e}")
            continue

    conn.commit()
    logger.info(f"Inserted/updated {inserted} forecasts")
    return inserted


def compute_and_insert_consensus(conn, dry_run: bool = False) -> int:
    """Compute consensus from forecasts and insert."""
    cur = conn.cursor()

    logger.info("Computing consensus from forecasts...")

    if dry_run:
        logger.info("[DRY RUN] Would compute consensus")
        return 0

    # Get distinct dates
    cur.execute("SELECT DISTINCT inference_date FROM bi.fact_forecasts ORDER BY inference_date")
    dates = [row[0] for row in cur.fetchall()]

    inserted = 0
    for inf_date in dates:
        for horizon in HORIZONS:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE direction = 'UP') as bullish,
                    COUNT(*) FILTER (WHERE direction = 'DOWN') as bearish,
                    AVG(predicted_return_pct) as avg_return
                FROM bi.fact_forecasts
                WHERE inference_date = %s AND horizon = %s
            """, (inf_date, horizon))

            result = cur.fetchone()
            if not result or (result[0] == 0 and result[1] == 0):
                continue

            bullish, bearish, avg_return = result
            total = bullish + bearish

            if bullish > bearish:
                direction = "BULLISH"
                strength = bullish / total if total > 0 else 0.5
            elif bearish > bullish:
                direction = "BEARISH"
                strength = bearish / total if total > 0 else 0.5
            else:
                direction = "NEUTRAL"
                strength = 0.5

            cur.execute("""
                INSERT INTO bi.fact_consensus
                (inference_date, horizon, bullish_count, bearish_count,
                 consensus_direction, consensus_strength, avg_predicted_return)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (inference_date, horizon) DO UPDATE SET
                    bullish_count = EXCLUDED.bullish_count,
                    bearish_count = EXCLUDED.bearish_count,
                    consensus_direction = EXCLUDED.consensus_direction,
                    consensus_strength = EXCLUDED.consensus_strength,
                    avg_predicted_return = EXCLUDED.avg_predicted_return
            """, (inf_date, horizon, bullish, bearish, direction, strength, avg_return))
            inserted += 1

    conn.commit()
    logger.info(f"Inserted/updated {inserted} consensus records")
    return inserted


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Migrate forecasting CSV to PostgreSQL")
    parser.add_argument("--csv-path", type=Path, help="Path to specific CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--skip-schema", action="store_true", help="Skip schema creation")
    parser.add_argument("--skip-dimensions", action="store_true", help="Skip dimension seeding")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Forecasting CSV to PostgreSQL Migration")
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN MODE - No changes will be made]")

    try:
        conn = get_db_connection()
        logger.info("Connected to database")

        # Step 1: Create schema and tables
        if not args.skip_schema and not args.dry_run:
            create_schema_and_tables(conn)

        # Step 2: Seed dimensions
        if not args.skip_dimensions and not args.dry_run:
            seed_dimensions(conn)

        # Step 3: Find and load CSV files
        if args.csv_path:
            csv_files = [args.csv_path]
        else:
            csv_files = find_csv_files()

        if not csv_files:
            logger.warning("No CSV files found to migrate")
            return

        logger.info(f"Found {len(csv_files)} CSV files")

        # Step 4: Migrate each CSV
        total_migrated = 0
        for csv_path in csv_files:
            try:
                df = load_csv(csv_path)
                migrated = migrate_forecasts(conn, df, args.dry_run)
                total_migrated += migrated
            except Exception as e:
                logger.error(f"Error migrating {csv_path}: {e}")
                continue

        # Step 5: Compute consensus
        if total_migrated > 0 or not args.dry_run:
            compute_and_insert_consensus(conn, args.dry_run)

        conn.close()

        logger.info("=" * 60)
        logger.info(f"Migration complete: {total_migrated} forecasts migrated")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
