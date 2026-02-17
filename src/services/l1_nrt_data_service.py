"""
L1 NRT Data Service - LOADS L2 DATASETS + MAINTAINS NRT UPDATES
================================================================

DEPRECATED (2026-02-12): This standalone async service has been absorbed into
the Airflow DAG pipeline:
  - Model promotion logic → airflow/dags/l1_model_promotion.py
  - NRT feature updates → airflow/dags/l1_feature_refresh.py (v5.0.0+)

The Airflow DAGs use CanonicalFeatureBuilder (Wilder's EMA) for feature
computation, whereas this service used pandas ewm(span=9) which produces
different RSI/ATR values. The DAG versions are authoritative.

This file is kept for reference and backward compatibility only.

Contract ID: CTR-L1-NRT-001
Version: 1.0.0
Created: 2026-02-04
Deprecated: 2026-02-12

On model approval:
1. Load the EXACT train/val/test datasets that L2 generated
2. Upload ALL historical data to inference_ready_nrt (model trained on this!)
3. Then just MAINTAIN: every 5 min, append new bars using same preprocessing

This ensures 100% consistency: L5 uses the SAME features the model saw during training.

L5 NEVER touches raw data. L5 only reads this table.

Architecture:
    model_approved
          |
          v
    +------------------+
    | L1NRTDataService |
    +------------------+
          |
          |--> Load norm_stats (validate hash)
          |--> Load L2 datasets (train/val/test)
          |--> Upload historical features to DB
          |
          v (every 5 min)
    +------------------+
    | _append_new_bars |
    +------------------+
          |
          |--> Read new OHLCV from L0
          |--> Read macro data from L0
          |--> Compute features (CanonicalFeatureBuilder)
          |--> Normalize with norm_stats
          |--> Upsert to inference_ready_nrt
          |--> NOTIFY 'features_ready'
          |
          v
    inference_ready_nrt (100% ready for model.predict)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Async PostgreSQL
try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

# Feature contracts and builder
from src.core.contracts.feature_contract import (
    FEATURE_ORDER,
    FEATURE_ORDER_HASH,
    OBSERVATION_DIM,
)
from src.core.contracts.norm_stats_contract import load_norm_stats, NormStatsContract

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class L1NRTConfig:
    """Configuration for L1 NRT Data Service."""
    update_interval_seconds: int = 300  # 5 minutes
    batch_size: int = 100  # Rows per DB insert batch
    clip_min: float = -5.0
    clip_max: float = 5.0
    market_features_count: int = 18  # First 18 features (excluding state)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class L1NRTError(Exception):
    """Base exception for L1 NRT service."""
    pass


class NormStatsHashMismatchError(L1NRTError):
    """Raised when norm_stats hash doesn't match expected."""
    def __init__(self, expected: str, actual: str):
        super().__init__(
            f"norm_stats hash mismatch: expected {expected[:12]}..., got {actual[:12]}... "
            f"Model cannot produce correct predictions without matching stats."
        )
        self.expected = expected
        self.actual = actual


class DatasetLoadError(L1NRTError):
    """Raised when L2 datasets cannot be loaded."""
    pass


# =============================================================================
# L1 NRT DATA SERVICE
# =============================================================================

class L1NRTDataService:
    """
    L1 Data Layer for NRT (Near Real-Time) Inference.

    Responsibilities:
    - Load norm_stats and validate hashes on model approval
    - Load L2 datasets (train/val/test) and upload to inference_ready_nrt
    - Maintain NRT updates every 5 minutes
    - Use CanonicalFeatureBuilder for feature computation
    - Apply z-score normalization using training norm_stats

    L5 NEVER does preprocessing. This service produces a table 100% ready for model.predict().

    Usage:
        service = L1NRTDataService(pool)
        await service.on_model_approved(payload)
        await service.run_update_loop()
    """

    # Market feature names (first 18, excluding state features)
    MARKET_FEATURES: Tuple[str, ...] = FEATURE_ORDER[:18]

    def __init__(
        self,
        db_pool: "asyncpg.Pool",
        config: Optional[L1NRTConfig] = None,
    ):
        """
        Initialize L1 NRT Data Service.

        Args:
            db_pool: asyncpg connection pool
            config: Optional configuration
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required for L1NRTDataService")

        self.pool = db_pool
        self.config = config or L1NRTConfig()

        # State
        self._norm_stats: Optional[NormStatsContract] = None
        self._norm_stats_hash: Optional[str] = None
        self._feature_order_hash: str = FEATURE_ORDER_HASH
        self._last_historical_ts: Optional[datetime] = None
        self._running: bool = False
        self._model_id: Optional[str] = None

        logger.info("L1NRTDataService initialized")

    # =========================================================================
    # MODEL APPROVAL HANDLER
    # =========================================================================

    async def on_model_approved(self, payload: Dict[str, Any]) -> None:
        """
        Handle model approval event.

        Steps:
        1. Load norm_stats from lineage
        2. Validate norm_stats hash
        3. Load L2 datasets (train/val/test)
        4. Upload all historical features to inference_ready_nrt

        Args:
            payload: model_approved notification payload containing:
                - model_id: Model identifier
                - norm_stats_path: Path to norm_stats.json
                - dataset_path: Path to L2 datasets directory
                - norm_stats_hash: Expected hash for validation
                - feature_order_hash: Expected feature order hash

        Raises:
            NormStatsHashMismatchError: If hash doesn't match
            DatasetLoadError: If datasets cannot be loaded
        """
        self._model_id = payload.get("model_id", "unknown")
        norm_stats_path = payload.get("norm_stats_path", "")
        dataset_path = payload.get("dataset_path", "")
        expected_norm_hash = payload.get("norm_stats_hash", "")
        expected_feature_hash = payload.get("feature_order_hash", "")

        logger.info(
            f"L1: Processing model approval for {self._model_id}, "
            f"norm_stats={norm_stats_path}, dataset={dataset_path}"
        )

        # Step 1: Load and validate norm_stats
        if norm_stats_path:
            await self._load_norm_stats(norm_stats_path, expected_norm_hash)

        # Step 2: Validate feature order hash
        if expected_feature_hash and expected_feature_hash != self._feature_order_hash:
            logger.warning(
                f"L1: Feature order hash mismatch: expected {expected_feature_hash[:12]}..., "
                f"got {self._feature_order_hash[:12]}..."
            )

        # Step 3: Load L2 datasets and upload to DB
        if dataset_path:
            await self._load_historical_datasets(dataset_path)

        logger.info(
            f"L1: Model approval processed. Historical data up to {self._last_historical_ts}"
        )

    async def _load_norm_stats(self, path: str, expected_hash: str) -> None:
        """
        Load norm_stats.json and validate hash.

        Args:
            path: Path to norm_stats.json
            expected_hash: Expected hash from training

        Raises:
            NormStatsHashMismatchError: If hash doesn't match
        """
        try:
            self._norm_stats, actual_hash = load_norm_stats(path)
            self._norm_stats_hash = actual_hash

            if expected_hash and actual_hash != expected_hash:
                raise NormStatsHashMismatchError(expected_hash, actual_hash)

            logger.info(
                f"L1: Loaded norm_stats from {path}, hash={actual_hash[:12]}..."
            )
        except FileNotFoundError:
            logger.error(f"L1: norm_stats not found at {path}")
            raise L1NRTError(f"norm_stats not found at {path}")

    async def _load_historical_datasets(self, dataset_path: str) -> None:
        """
        Load train/val/test parquet files from L2 and upload to DB.

        These are the EXACT features the model was trained/validated on.

        Args:
            dataset_path: Path to L2 dataset directory containing train.parquet, etc.
        """
        base = Path(dataset_path)
        dfs: List[pd.DataFrame] = []

        # Load all splits
        for split in ["train", "val", "test"]:
            for ext in [".parquet", "_features.parquet"]:
                path = base / f"{split}{ext}"
                if path.exists():
                    try:
                        df = pd.read_parquet(path)
                        df["_source"] = "historical"
                        df["_split"] = split
                        dfs.append(df)
                        logger.info(f"L1: Loaded {path} with {len(df)} rows")
                    except Exception as e:
                        logger.warning(f"L1: Failed to load {path}: {e}")
                    break

        if not dfs:
            logger.warning(f"L1: No datasets found in {dataset_path}")
            return

        # Combine and sort
        df_all = pd.concat(dfs, ignore_index=False)
        if not isinstance(df_all.index, pd.DatetimeIndex):
            # Try to find timestamp column
            ts_cols = ["timestamp", "date", "datetime"]
            for col in ts_cols:
                if col in df_all.columns:
                    df_all = df_all.set_index(col)
                    break

        df_all = df_all.sort_index()
        self._last_historical_ts = df_all.index.max()

        # Upload to database
        await self._bulk_upsert_features(df_all, source="historical")

        logger.info(
            f"L1: Uploaded {len(df_all)} historical rows, "
            f"last timestamp: {self._last_historical_ts}"
        )

    async def _bulk_upsert_features(
        self,
        df: pd.DataFrame,
        source: str = "nrt",
    ) -> int:
        """
        Bulk upsert features to inference_ready_nrt.

        Args:
            df: DataFrame with features indexed by timestamp
            source: 'historical' or 'nrt'

        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0

        # Clear existing data for historical load
        if source == "historical":
            async with self.pool.acquire() as conn:
                await conn.execute("TRUNCATE inference_ready_nrt")
                logger.info("L1: Cleared inference_ready_nrt for historical load")

        inserted = 0
        rows_data = []

        for ts, row in df.iterrows():
            # Extract and normalize features
            features = self._extract_features(row)
            if features is None:
                continue

            # Get price
            price = row.get("close", row.get("price", 0.0))
            if pd.isna(price) or price <= 0:
                continue

            rows_data.append((
                ts,
                features,
                float(price),
                self._feature_order_hash,
                self._norm_stats_hash or "",
                source,
            ))

            # Batch insert
            if len(rows_data) >= self.config.batch_size:
                inserted += await self._insert_batch(rows_data)
                rows_data = []

        # Insert remaining
        if rows_data:
            inserted += await self._insert_batch(rows_data)

        return inserted

    async def _insert_batch(
        self,
        rows: List[Tuple],
    ) -> int:
        """Insert a batch of rows to inference_ready_nrt."""
        if not rows:
            return 0

        async with self.pool.acquire() as conn:
            # Use executemany for efficiency
            result = await conn.executemany(
                """
                INSERT INTO inference_ready_nrt
                    (timestamp, features, price, feature_order_hash, norm_stats_hash, source)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (timestamp) DO UPDATE SET
                    features = EXCLUDED.features,
                    price = EXCLUDED.price,
                    source = EXCLUDED.source
                """,
                rows,
            )
            return len(rows)

    def _extract_features(self, row: pd.Series) -> Optional[List[float]]:
        """
        Extract and normalize 18 market features from a row.

        Args:
            row: DataFrame row with feature values

        Returns:
            List of 18 normalized floats, or None if extraction fails
        """
        features = []

        for feat_name in self.MARKET_FEATURES:
            value = row.get(feat_name, np.nan)

            # Handle missing/invalid values
            if pd.isna(value) or np.isinf(value):
                value = 0.0
            else:
                value = float(value)

            # Normalize if norm_stats available
            if self._norm_stats and feat_name in self._norm_stats.features:
                stats = self._norm_stats.features[feat_name]
                if stats.std > 1e-8:
                    value = (value - stats.mean) / stats.std

            # Clip to valid range
            value = np.clip(value, self.config.clip_min, self.config.clip_max)
            features.append(float(value))

        return features if len(features) == 18 else None

    # =========================================================================
    # NRT UPDATE LOOP
    # =========================================================================

    async def run_update_loop(self) -> None:
        """
        Main loop: append NEW data every 5 minutes.

        Only processes bars AFTER the historical data ends.
        Uses SAME CanonicalFeatureBuilder and norm_stats as L2.
        """
        self._running = True
        logger.info(
            f"L1: Starting NRT update loop, interval={self.config.update_interval_seconds}s"
        )

        while self._running:
            try:
                if self._norm_stats and self._last_historical_ts:
                    await self._append_new_bars()
            except Exception as e:
                logger.error(f"L1: Error in update loop: {e}", exc_info=True)

            await asyncio.sleep(self.config.update_interval_seconds)

    async def stop(self) -> None:
        """Stop the update loop."""
        self._running = False
        logger.info("L1: Stopping NRT update loop")

    async def _append_new_bars(self) -> None:
        """
        Compute features for NEW bars only (after historical data).

        Uses SAME feature computation and norm_stats as L2.
        """
        async with self.pool.acquire() as conn:
            # Get latest timestamp in our table
            last_ts = await conn.fetchval(
                "SELECT MAX(timestamp) FROM inference_ready_nrt"
            )

            if last_ts is None:
                logger.warning("L1: No data in inference_ready_nrt, skipping append")
                return

            # Read new OHLCV from L0 (after last_ts)
            ohlcv_rows = await conn.fetch(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_5min
                WHERE timestamp > $1
                ORDER BY timestamp
                LIMIT 100
                """,
                last_ts,
            )

            if not ohlcv_rows:
                logger.debug("L1: No new OHLCV data since %s", last_ts)
                return

            # Read macro data for context
            macro_rows = await conn.fetch(
                """
                SELECT timestamp, dxy, vix, embi, brent,
                       fed_rate, col_rate, usdmxn, us_10y, gold
                FROM macro_fill_5min
                WHERE timestamp > $1
                ORDER BY timestamp
                LIMIT 100
                """,
                last_ts,
            )

            # Convert to DataFrames and merge
            df_ohlcv = self._rows_to_dataframe(ohlcv_rows)
            df_macro = self._rows_to_dataframe(macro_rows) if macro_rows else None

            if df_ohlcv.empty:
                return

            # Merge OHLCV and macro
            df = df_ohlcv.copy()
            if df_macro is not None and not df_macro.empty:
                df = df.join(df_macro, how="left", rsuffix="_macro")
                # Forward fill macro values
                macro_cols = ["dxy", "vix", "embi", "brent", "fed_rate", "col_rate", "usdmxn", "us_10y", "gold"]
                for col in macro_cols:
                    if col in df.columns:
                        df[col] = df[col].ffill()

            # Compute features using CanonicalFeatureBuilder pattern
            features_df = self._compute_features(df)

            if features_df.empty:
                return

            # Bulk upsert new features
            inserted = await self._bulk_upsert_features(features_df, source="nrt")
            logger.info(f"L1: Appended {inserted} new NRT bars")

    def _rows_to_dataframe(self, rows: List) -> pd.DataFrame:
        """Convert asyncpg rows to pandas DataFrame."""
        if not rows:
            return pd.DataFrame()

        data = [dict(row) for row in rows]
        df = pd.DataFrame(data)

        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

        return df

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical and macro features for a DataFrame.

        This implements the same feature computation as CanonicalFeatureBuilder
        but optimized for batch processing.

        Args:
            df: DataFrame with OHLCV and macro columns

        Returns:
            DataFrame with computed features
        """
        if df.empty:
            return df

        result = df.copy()

        # Log returns
        if "close" in result.columns:
            result["log_ret_5m"] = np.log(result["close"] / result["close"].shift(1))
            result["log_ret_1h"] = np.log(result["close"] / result["close"].shift(12))
            result["log_ret_4h"] = np.log(result["close"] / result["close"].shift(48))
            result["log_ret_1d"] = np.log(result["close"] / result["close"].shift(288))

        # RSI (simplified calculation)
        if "close" in result.columns:
            delta = result["close"].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            avg_gain_9 = gain.ewm(span=9, adjust=False).mean()
            avg_loss_9 = loss.ewm(span=9, adjust=False).mean()
            rs_9 = avg_gain_9 / (avg_loss_9 + 1e-10)
            result["rsi_9"] = 100 - (100 / (1 + rs_9))
            result["rsi_9"] = (result["rsi_9"] - 50) / 25  # Normalize to ~[-2, 2]

            avg_gain_21 = gain.ewm(span=21, adjust=False).mean()
            avg_loss_21 = loss.ewm(span=21, adjust=False).mean()
            rs_21 = avg_gain_21 / (avg_loss_21 + 1e-10)
            result["rsi_21"] = 100 - (100 / (1 + rs_21))
            result["rsi_21"] = (result["rsi_21"] - 50) / 25

        # Volatility (close-based)
        if "close" in result.columns:
            returns = np.log(result["close"] / result["close"].shift(1))
            result["volatility_pct"] = returns.rolling(20).std() * np.sqrt(288) * 100

        # Trend (simplified)
        if "close" in result.columns:
            sma_20 = result["close"].rolling(20).mean()
            sma_50 = result["close"].rolling(50).mean()
            result["trend_z"] = (sma_20 - sma_50) / (sma_50 + 1e-10) * 100

        # Macro z-scores (rolling window)
        for col, target in [("dxy", "dxy_z"), ("vix", "vix_z"), ("embi", "embi_z")]:
            if col in result.columns:
                rolling_mean = result[col].rolling(60, min_periods=1).mean()
                rolling_std = result[col].rolling(60, min_periods=1).std()
                result[target] = (result[col] - rolling_mean) / (rolling_std + 1e-10)

        # Macro changes
        for col, target in [
            ("dxy", "dxy_change_1d"),
            ("brent", "brent_change_1d"),
            ("usdmxn", "usdmxn_change_1d"),
            ("gold", "gold_change_1d"),
        ]:
            if col in result.columns:
                result[target] = result[col].pct_change(288) * 100

        # Rate spread
        if "fed_rate" in result.columns and "col_rate" in result.columns:
            spread = result["col_rate"] - result["fed_rate"]
            rolling_mean = spread.rolling(60, min_periods=1).mean()
            rolling_std = spread.rolling(60, min_periods=1).std()
            result["rate_spread_z"] = (spread - rolling_mean) / (rolling_std + 1e-10)
            result["rate_spread_change"] = spread.diff(288)

        # Yield curve
        if "us_10y" in result.columns:
            us_2y = result.get("us_2y", result["us_10y"] * 0.8)  # Approximate if missing
            curve = result["us_10y"] - us_2y
            rolling_mean = curve.rolling(60, min_periods=1).mean()
            rolling_std = curve.rolling(60, min_periods=1).std()
            result["yield_curve_z"] = (curve - rolling_mean) / (rolling_std + 1e-10)

        # Fill NaN with 0 for new features
        for feat in self.MARKET_FEATURES:
            if feat in result.columns:
                result[feat] = result[feat].fillna(0.0)

        # Drop rows with NaN in required columns
        result = result.dropna(subset=["close"])

        return result

    # =========================================================================
    # STATUS & HEALTH
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if update loop is running."""
        return self._running

    @property
    def model_id(self) -> Optional[str]:
        """Get current model ID."""
        return self._model_id

    @property
    def norm_stats_hash(self) -> Optional[str]:
        """Get current norm_stats hash."""
        return self._norm_stats_hash

    async def get_status(self) -> Dict[str, Any]:
        """Get service status for health checks."""
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM inference_ready_nrt"
            )
            latest_ts = await conn.fetchval(
                "SELECT MAX(timestamp) FROM inference_ready_nrt"
            )

        return {
            "running": self._running,
            "model_id": self._model_id,
            "norm_stats_hash": self._norm_stats_hash[:12] if self._norm_stats_hash else None,
            "feature_order_hash": self._feature_order_hash[:12],
            "last_historical_ts": str(self._last_historical_ts) if self._last_historical_ts else None,
            "rows_in_db": count,
            "latest_timestamp": str(latest_ts) if latest_ts else None,
            "update_interval_seconds": self.config.update_interval_seconds,
        }
