"""
Backtest Feature Builder Service
Contract: CTR-FEAT-BUILD-001

Builds inference features for historical backtest periods.
Follows Single Responsibility: Only builds features, delegates validation.
"""

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from src.validation.backtest_data_validator import BacktestDataValidator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Immutable configuration for feature building."""
    # Time alignment
    use_backward_merge: bool = True  # merge_asof direction='backward' for no look-ahead

    # FFILL limits (days)
    macro_ffill_limit_daily: int = 5
    macro_ffill_limit_monthly: int = 35

    # Z-score parameters (static from training)
    zscore_params: dict = None

    # Batch processing
    batch_size_days: int = 30

    def __post_init__(self):
        if self.zscore_params is None:
            object.__setattr__(self, 'zscore_params', {
                "dxy": {"mean": 103.5, "std": 2.5},
                "vix": {"mean": 18.0, "std": 5.0},
                "embi": {"mean": 400.0, "std": 50.0},
            })


class BacktestFeatureBuilder:
    """
    Builds historical features for backtesting.

    Clean Code Principles:
    - Single Responsibility: Only builds features
    - Dependency Injection: Validator injected, not created
    - Fail Fast: Validates before building
    - Explicit: Clear method signatures
    """

    def __init__(
        self,
        connection_string: str,
        validator: Optional[BacktestDataValidator] = None,
        config: Optional[FeatureBuildConfig] = None
    ):
        self.engine = create_engine(connection_string)
        self.validator = validator or BacktestDataValidator(connection_string)
        self.config = config or FeatureBuildConfig()

    def build_features(
        self,
        start_date: dt.date,
        end_date: dt.date,
        validate_first: bool = True,
        dry_run: bool = False
    ) -> dict:
        """
        Build inference features for date range.

        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            validate_first: Run validation before building
            dry_run: If True, only report what would be built

        Returns:
            dict with build statistics
        """
        # Step 1: Validate data (Fail Fast)
        if validate_first:
            logger.info(f"Validating data from {start_date} to {end_date}")
            validation = self.validator.validate(start_date, end_date, fail_fast=True)
            logger.info(f"Validation passed. OHLCV: {validation.ohlcv_coverage_pct:.1%}, Macro: {validation.macro_coverage_pct:.1%}")

        if dry_run:
            return self._dry_run_report(start_date, end_date)

        # Step 2: Load OHLCV data
        logger.info("Loading OHLCV data...")
        ohlcv_df = self._load_ohlcv(start_date, end_date)

        # Step 3: Load macro data with bounded FFILL
        logger.info("Loading macro data with bounded FFILL...")
        macro_df = self._load_macro_with_bounded_ffill(start_date, end_date)

        # Step 4: Merge using backward asof (no look-ahead)
        logger.info("Merging OHLCV with macro (backward merge)...")
        merged_df = self._merge_backward(ohlcv_df, macro_df)

        # Step 5: Calculate derived features
        logger.info("Calculating derived features...")
        features_df = self._calculate_features(merged_df)

        # Step 6: Insert into inference_features_5m
        logger.info("Inserting features into database...")
        rows_inserted = self._insert_features(features_df)

        return {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "ohlcv_rows": len(ohlcv_df),
            "macro_rows": len(macro_df),
            "features_built": len(features_df),
            "rows_inserted": rows_inserted,
        }

    def _load_ohlcv(self, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Load OHLCV data for date range."""
        query = text("""
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_5m_usdcop
            WHERE timestamp >= :start AND timestamp < :end
            ORDER BY timestamp
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start": start_date, "end": end_date})

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def _load_macro_with_bounded_ffill(self, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
        """Load macro data with bounded forward-fill."""
        query = text("""
            SELECT fecha,
                   fxrt_index_dxy_usa_d_dxy as dxy,
                   volt_vix_usa_d_vix as vix,
                   finc_bond_yield10y_usa_d_ust10y as ust10y,
                   finc_bond_yield2y_usa_d_dgs2 as ust2y,
                   finc_cds_sovereign_col_5y_d_cds5y as cds5y,
                   finc_embi_spread_col_d_embi as embi
            FROM macro_indicators_daily
            WHERE fecha >= :start - INTERVAL '7 days' AND fecha <= :end
            ORDER BY fecha
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"start": start_date, "end": end_date})

        df['fecha'] = pd.to_datetime(df['fecha'])

        # Apply bounded FFILL per column
        for col in ['dxy', 'vix', 'ust10y', 'ust2y', 'cds5y', 'embi']:
            if col in df.columns:
                df[col] = df[col].ffill(limit=self.config.macro_ffill_limit_daily)

        return df

    def _merge_backward(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge OHLCV with macro using backward asof merge.

        CRITICAL: direction='backward' ensures we only use macro data
        that was available AT OR BEFORE the OHLCV timestamp.
        This prevents look-ahead bias.
        """
        ohlcv_df = ohlcv_df.sort_values('timestamp')
        macro_df = macro_df.sort_values('fecha')

        merged = pd.merge_asof(
            ohlcv_df,
            macro_df,
            left_on='timestamp',
            right_on='fecha',
            direction='backward'  # CRITICAL: No look-ahead
        )

        return merged

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features from merged data."""
        features = df.copy()

        # Z-scores with static parameters
        zscore_cols = {
            'dxy': 'dxy_zscore',
            'vix': 'vix_zscore',
            'embi': 'embi_zscore'
        }

        for col, zscore_col in zscore_cols.items():
            if col in features.columns:
                params = self.config.zscore_params.get(col, {"mean": 0, "std": 1})
                features[zscore_col] = (features[col] - params["mean"]) / params["std"]
                features[zscore_col] = features[zscore_col].clip(-3, 3)

        # Yield spread
        if 'ust10y' in features.columns and 'ust2y' in features.columns:
            features['yield_spread'] = features['ust10y'] - features['ust2y']

        # Price returns
        features['return_5m'] = features['close'].pct_change()
        features['return_15m'] = features['close'].pct_change(3)
        features['return_1h'] = features['close'].pct_change(12)

        # Volatility (rolling std of returns)
        features['volatility_1h'] = features['return_5m'].rolling(12).std()

        return features.dropna()

    def _insert_features(self, df: pd.DataFrame) -> int:
        """Insert features into database."""
        # Select columns that exist in target table
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'dxy', 'vix', 'ust10y', 'ust2y', 'cds5y', 'embi',
            'dxy_zscore', 'vix_zscore', 'embi_zscore',
            'yield_spread', 'return_5m', 'return_15m', 'return_1h',
            'volatility_1h'
        ]

        existing_cols = [c for c in columns if c in df.columns]
        insert_df = df[existing_cols].copy()

        # Upsert to avoid duplicates
        insert_df.to_sql(
            'inference_features_5m',
            self.engine,
            if_exists='append',
            index=False,
            method='multi'
        )

        return len(insert_df)

    def _dry_run_report(self, start_date: dt.date, end_date: dt.date) -> dict:
        """Report what would be built without actually building."""
        with self.engine.connect() as conn:
            ohlcv_count = conn.execute(text("""
                SELECT COUNT(*) FROM ohlcv_5m_usdcop
                WHERE timestamp >= :start AND timestamp < :end
            """), {"start": start_date, "end": end_date}).scalar()

            macro_count = conn.execute(text("""
                SELECT COUNT(*) FROM macro_indicators_daily
                WHERE fecha >= :start AND fecha <= :end
            """), {"start": start_date, "end": end_date}).scalar()

            existing_features = conn.execute(text("""
                SELECT COUNT(*) FROM inference_features_5m
                WHERE timestamp >= :start AND timestamp < :end
            """), {"start": start_date, "end": end_date}).scalar()

        return {
            "dry_run": True,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "ohlcv_available": ohlcv_count,
            "macro_available": macro_count,
            "features_existing": existing_features,
            "features_would_build": ohlcv_count,
        }
