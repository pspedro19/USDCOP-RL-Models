"""
SSOT Dataset Builder - Dynamic Feature Generation from pipeline_ssot.yaml
=========================================================================

This module provides the production dataset builder that reads feature
definitions from the SSOT and generates datasets dynamically.

L2 DAG should use this module to generate training/val/test datasets.

Usage:
    from src.data.ssot_dataset_builder import SSOTDatasetBuilder

    builder = SSOTDatasetBuilder()
    result = builder.build(df_ohlcv, df_macro)

    # Access outputs
    result.train_df  # Training dataset
    result.val_df    # Validation dataset
    result.test_df   # Test dataset
    result.norm_stats  # Normalization statistics

Contract: CTR-L2-DATASET-BUILDER-001
Version: 1.0.0
Date: 2026-02-03
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    """Result of dataset building process."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    full_df: pd.DataFrame
    norm_stats: Dict[str, Any]
    lineage: Dict[str, Any]
    feature_columns: List[str]

    @property
    def observation_dim(self) -> int:
        """Get observation dimension (market features only)."""
        return len(self.feature_columns)


class SSOTDatasetBuilder:
    """
    SSOT-driven dataset builder for L2.

    Reads feature definitions from pipeline_ssot.yaml and generates
    train/val/test datasets with proper normalization.

    Key features:
    - Dynamic feature calculation from SSOT
    - Train-only normalization statistics
    - Anti-leakage T-1 macro shift
    - Automatic data quality validation
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize builder with SSOT configuration.

        Args:
            config_path: Path to pipeline_ssot.yaml (uses default if None)
        """
        from src.config.pipeline_config import load_pipeline_config
        self.config = load_pipeline_config(str(config_path) if config_path else None)

        # Import calculator registry
        from src.features.calculator_registry import (
            get_calculator_registry,
            calculate_log_returns,
            calculate_rsi_wilders,
            calculate_volatility_pct,
            calculate_trend_z,
            calculate_macro_zscore,
            calculate_spread_zscore,
            calculate_pct_change,
        )
        self.registry = get_calculator_registry()

        logger.info(
            f"SSOTDatasetBuilder initialized with config v{self.config.version}, "
            f"{len(self.config.get_market_features())} market features"
        )

    def build(
        self,
        df_ohlcv: pd.DataFrame,
        df_macro: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        dataset_prefix: Optional[str] = None,
        df_aux_ohlcv: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> DatasetBuildResult:
        """
        Build complete dataset from OHLCV and macro data.

        Args:
            df_ohlcv: DataFrame with OHLCV data (must have 'close', datetime index or column)
            df_macro: DataFrame with macro data (optional)
            output_dir: Directory to save outputs (optional)
            dataset_prefix: Prefix for output files (optional, uses SSOT default)
            df_aux_ohlcv: Dict of auxiliary OHLCV DataFrames keyed by pair name
                          (e.g., {"usdmxn": df_mxn, "usdbrl": df_brl})

        Returns:
            DatasetBuildResult with train/val/test datasets and metadata
        """
        start_time = datetime.now()
        logger.info("Starting SSOT dataset build...")

        # Store aux OHLCV for feature calculation
        self._df_aux_ohlcv = df_aux_ohlcv or {}

        # 1. Prepare data
        df_ohlcv, df_macro = self._prepare_data(df_ohlcv, df_macro)

        # 2. Calculate all features
        df_features = self._calculate_features(df_ohlcv, df_macro)

        # 3. Merge with OHLCV
        df_full = self._merge_data(df_ohlcv, df_features, df_macro)

        # 4. Add auxiliary features (raw returns for PnL)
        df_full = self._add_auxiliary_features(df_full, df_ohlcv)

        # 5. Split into train/val/test
        train_df, val_df, test_df = self._split_data(df_full)

        # 6. Compute normalization stats (train only)
        feature_cols = [f.name for f in self.config.get_market_features()]
        norm_stats = self._compute_norm_stats(train_df, feature_cols)

        # 7. Apply normalization
        train_df = self._apply_normalization(train_df, feature_cols, norm_stats)
        val_df = self._apply_normalization(val_df, feature_cols, norm_stats)
        test_df = self._apply_normalization(test_df, feature_cols, norm_stats)

        # 8. Clean and validate
        train_df = self._clean_dataset(train_df)
        val_df = self._clean_dataset(val_df)
        test_df = self._clean_dataset(test_df)

        # 9. Compute and log descriptive statistics
        descriptive_stats = self._compute_descriptive_stats(train_df, val_df, test_df, feature_cols)
        self._log_descriptive_stats(descriptive_stats, feature_cols)

        # 10. Build lineage
        lineage = self._build_lineage(df_ohlcv, df_macro, train_df, val_df, test_df)

        # 11. Save if output_dir provided
        if output_dir:
            self._save_outputs(
                output_dir,
                dataset_prefix or self.config.paths.l2_dataset_prefix,
                train_df, val_df, test_df,
                norm_stats, lineage, descriptive_stats
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Dataset build complete in {elapsed:.1f}s: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        return DatasetBuildResult(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            full_df=df_full,
            norm_stats=norm_stats,
            lineage=lineage,
            feature_columns=feature_cols
        )

    def _prepare_data(
        self,
        df_ohlcv: pd.DataFrame,
        df_macro: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Prepare and validate input data."""
        # Ensure datetime index
        df_ohlcv = df_ohlcv.copy()
        if 'datetime' in df_ohlcv.columns:
            df_ohlcv['datetime'] = pd.to_datetime(df_ohlcv['datetime'])
            df_ohlcv = df_ohlcv.set_index('datetime')
        elif 'time' in df_ohlcv.columns:
            df_ohlcv['time'] = pd.to_datetime(df_ohlcv['time'])
            df_ohlcv = df_ohlcv.rename(columns={'time': 'datetime'}).set_index('datetime')

        # Remove timezone
        if df_ohlcv.index.tz is not None:
            df_ohlcv.index = df_ohlcv.index.tz_localize(None)

        # Validate OHLCV
        required_cols = ['close']
        missing = [c for c in required_cols if c not in df_ohlcv.columns]
        if missing:
            raise ValueError(f"OHLCV missing required columns: {missing}")

        # Sort by datetime
        df_ohlcv = df_ohlcv.sort_index()

        # Prepare macro if provided
        if df_macro is not None:
            df_macro = df_macro.copy()
            if 'datetime' in df_macro.columns:
                df_macro['datetime'] = pd.to_datetime(df_macro['datetime'])
                df_macro = df_macro.set_index('datetime')
            elif 'time' in df_macro.columns:
                df_macro['time'] = pd.to_datetime(df_macro['time'])
                df_macro = df_macro.rename(columns={'time': 'datetime'}).set_index('datetime')

            if df_macro.index.tz is not None:
                df_macro.index = df_macro.index.tz_localize(None)

            df_macro = df_macro.sort_index()

        logger.info(
            f"Data prepared: OHLCV={len(df_ohlcv)} rows, "
            f"Macro={len(df_macro) if df_macro is not None else 0} rows"
        )

        return df_ohlcv, df_macro

    def _calculate_features(
        self,
        df_ohlcv: pd.DataFrame,
        df_macro: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate all market features from SSOT definitions."""
        # Separate OHLCV and macro features
        ohlcv_features = {}
        macro_features = {}

        for feature_def in self.config.get_market_features():
            try:
                value = self._calculate_single_feature(feature_def, df_ohlcv, df_macro)
                if value is not None:
                    if feature_def.source in ("macro_daily", "macro_monthly"):
                        macro_features[feature_def.name] = value
                    else:
                        ohlcv_features[feature_def.name] = value
                    logger.debug(f"Calculated: {feature_def.name}")
            except Exception as e:
                logger.warning(f"Failed to calculate {feature_def.name}: {e}")

        # Build result from OHLCV features (already aligned to OHLCV index)
        result = pd.DataFrame(ohlcv_features, index=df_ohlcv.index)

        # Merge macro features using merge_asof (efficient for different freq data)
        if macro_features:
            macro_df = pd.DataFrame(macro_features)
            macro_df = macro_df.sort_index()

            # Reset index for merge_asof
            result_reset = result.reset_index()
            result_reset = result_reset.rename(columns={result_reset.columns[0]: 'datetime'})

            macro_reset = macro_df.reset_index()
            macro_reset = macro_reset.rename(columns={macro_reset.columns[0]: 'datetime'})

            # Use merge_asof: for each OHLCV row, find the most recent macro row
            merged = pd.merge_asof(
                result_reset.sort_values('datetime'),
                macro_reset.sort_values('datetime'),
                on='datetime',
                direction='backward'  # Use most recent available macro data
            )

            # Restore index
            merged = merged.set_index('datetime')
            result = merged

        return result

    def _calculate_single_feature(
        self,
        feature_def: "FeatureDefinition",
        df_ohlcv: pd.DataFrame,
        df_macro: Optional[pd.DataFrame]
    ) -> Optional[pd.Series]:
        """Calculate a single feature based on its definition."""
        calculator_name = feature_def.calculator
        params = feature_def.params.copy()

        # Determine source data
        if feature_def.source == "ohlcv":
            source_df = df_ohlcv
        elif feature_def.source in ("macro_daily", "macro_monthly"):
            if df_macro is None:
                logger.warning(f"No macro data for feature {feature_def.name}")
                return None
            source_df = df_macro
        elif feature_def.source == "aux_ohlcv":
            # Cross-pair features from auxiliary OHLCV DataFrames
            if not getattr(self, '_df_aux_ohlcv', None):
                logger.warning(f"No aux OHLCV data for feature {feature_def.name}")
                return None
            return self._calculate_aux_feature(feature_def, df_ohlcv)
        elif feature_def.source == "runtime":
            # State features are computed at runtime
            return None
        else:
            logger.warning(f"Unknown source {feature_def.source} for {feature_def.name}")
            return None

        # Special case: temporal features computed from datetime index
        if (calculator_name is None
                and feature_def.input_columns == ["time"]
                and feature_def.category == "temporal"
                and hasattr(source_df, 'index')
                and isinstance(source_df.index, pd.DatetimeIndex)):
            return self._compute_temporal_feature(feature_def, source_df)

        # Get input series
        input_cols = feature_def.input_columns
        missing_cols = [c for c in input_cols if c not in source_df.columns]
        if missing_cols:
            logger.warning(
                f"Missing columns {missing_cols} for {feature_def.name}, "
                f"available: {list(source_df.columns)[:10]}..."
            )
            return None

        # Apply preprocessing (T-1 shift for macro)
        series_list = []
        for col in input_cols:
            s = source_df[col].copy()

            # Forward fill with limit
            if feature_def.preprocessing.ffill_limit:
                s = s.ffill(limit=feature_def.preprocessing.ffill_limit)

            # Anti-leakage shift
            if feature_def.preprocessing.shift > 0:
                s = s.shift(feature_def.preprocessing.shift)

            # Floor (for volatility, etc)
            if feature_def.preprocessing.floor is not None:
                s = s.clip(lower=feature_def.preprocessing.floor)

            series_list.append(s)

        # Handle custom formula
        if feature_def.custom_formula:
            return self._eval_formula(feature_def.custom_formula, df_ohlcv, df_macro)

        # Execute calculator
        if calculator_name is None:
            # Direct pass-through
            return series_list[0] if series_list else None

        calc_info = self.registry.get(calculator_name)
        if calc_info is None:
            logger.warning(f"Unknown calculator: {calculator_name}")
            return None

        if calc_info.requires_multiple_series:
            return self.registry.calculate(calculator_name, *series_list, **params)
        else:
            return self.registry.calculate(calculator_name, series_list[0], **params)

    def _compute_temporal_feature(
        self,
        feature_def: "FeatureDefinition",
        source_df: pd.DataFrame,
    ) -> pd.Series:
        """Compute temporal (sin/cos) features from the datetime index.

        Matches the same encoding used by TradingEnvironment._get_observation():
        - hour_sin_market: sin(2π * hour_frac), where hour_frac normalized [0,1] within session
        - hour_cos_market: cos(2π * hour_frac)
        - dow_sin_market: sin(2π * dow / 5)  (5 trading days)

        Hour normalization: USDCOP trades 8:00-12:55 COT.
        After L2 tz-strip, hours are 8-12 (COT). Auto-detect: if median hour < 13,
        assume COT (session_start=8); else assume pre-fix UTC (session_start=13).
        """
        idx = source_df.index
        name = feature_def.name

        # Auto-detect session start based on actual hour values
        median_hour = int(np.median(idx.hour))
        session_start = 8.0 if median_hour < 13 else 13.0

        if "hour_sin" in name:
            hour_frac = np.clip((idx.hour + idx.minute / 60.0 - session_start) / 5.0, 0.0, 1.0)
            return pd.Series(np.sin(2 * np.pi * hour_frac), index=idx, name=name)
        elif "hour_cos" in name:
            hour_frac = np.clip((idx.hour + idx.minute / 60.0 - session_start) / 5.0, 0.0, 1.0)
            return pd.Series(np.cos(2 * np.pi * hour_frac), index=idx, name=name)
        elif "dow_sin" in name:
            dow_frac = idx.dayofweek / 5.0  # /5.0 for trading days, NOT /7.0
            return pd.Series(np.sin(2 * np.pi * dow_frac), index=idx, name=name)
        elif "dow_cos" in name:
            dow_frac = idx.dayofweek / 5.0
            return pd.Series(np.cos(2 * np.pi * dow_frac), index=idx, name=name)
        else:
            logger.warning(f"Unknown temporal feature: {name}")
            return pd.Series(0.0, index=idx, name=name)

    def _calculate_aux_feature(
        self,
        feature_def: "FeatureDefinition",
        df_ohlcv: pd.DataFrame,
    ) -> Optional[pd.Series]:
        """Calculate a feature from auxiliary OHLCV data.

        Handles "pairname.column" notation in input_columns.
        """
        calculator_name = feature_def.calculator
        params = feature_def.params.copy()

        series_list = []
        for col_spec in feature_def.input_columns:
            if "." in col_spec:
                pair_name, col_name = col_spec.split(".", 1)
                if pair_name not in self._df_aux_ohlcv:
                    logger.warning(
                        f"Aux pair '{pair_name}' not found for {feature_def.name}. "
                        f"Available: {list(self._df_aux_ohlcv.keys())}"
                    )
                    return None
                aux_df = self._df_aux_ohlcv[pair_name]
                if col_name not in aux_df.columns:
                    logger.warning(f"Column '{col_name}' not in {pair_name} data")
                    return None
                s = aux_df[col_name].copy()
                # Reindex to primary OHLCV index
                s = s.reindex(df_ohlcv.index, method="ffill")
            else:
                # Plain column from primary OHLCV
                if col_spec not in df_ohlcv.columns:
                    logger.warning(f"Column '{col_spec}' not in OHLCV for {feature_def.name}")
                    return None
                s = df_ohlcv[col_spec].copy()

            # Apply preprocessing
            if feature_def.preprocessing.ffill_limit:
                s = s.ffill(limit=feature_def.preprocessing.ffill_limit)
            if feature_def.preprocessing.shift > 0:
                s = s.shift(feature_def.preprocessing.shift)
            if feature_def.preprocessing.floor is not None:
                s = s.clip(lower=feature_def.preprocessing.floor)

            series_list.append(s)

        if not series_list:
            return None

        # Execute calculator
        if calculator_name is None:
            return series_list[0]

        calc_info = self.registry.get(calculator_name)
        if calc_info is None:
            logger.warning(f"Unknown calculator: {calculator_name}")
            return None

        if calc_info.requires_multiple_series:
            return self.registry.calculate(calculator_name, *series_list, **params)
        else:
            return self.registry.calculate(calculator_name, series_list[0], **params)

    def _eval_formula(
        self,
        formula: str,
        df_ohlcv: pd.DataFrame,
        df_macro: Optional[pd.DataFrame]
    ) -> pd.Series:
        """Evaluate a custom formula."""
        # Build context
        context = {}
        for col in df_ohlcv.columns:
            context[col] = df_ohlcv[col]
        if df_macro is not None:
            for col in df_macro.columns:
                context[col] = df_macro[col]
        context['pd'] = pd
        context['np'] = np

        return eval(formula, {"__builtins__": {}}, context)

    def _merge_data(
        self,
        df_ohlcv: pd.DataFrame,
        df_features: pd.DataFrame,
        df_macro: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge OHLCV, features, and optionally macro into single DataFrame."""
        # Start with features (has same index as OHLCV)
        df = df_features.copy()

        # Add timestamp column
        df['timestamp'] = df.index

        return df

    def _add_auxiliary_features(
        self,
        df: pd.DataFrame,
        df_ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add auxiliary features needed for PnL calculation and chart display.

        Includes:
        - raw_{first_feature_name}: Unnormalized log return for PnL calculation
          (e.g., raw_log_ret_5m for 5-min, raw_log_ret_1h for hourly)
        - close: Actual close price for chart display (JOIN by timestamp with OHLCV)
        """
        from src.features.calculator_registry import calculate_log_returns

        # Dynamic raw return column name based on first market feature
        first_feature = self.config.get_market_features()[0]
        raw_ret_col = f"raw_{first_feature.name}"

        # Raw log return (unnormalized) for PnL calculation
        df[raw_ret_col] = calculate_log_returns(df_ohlcv['close'], periods=1)

        # Close price for backtest visualization and accurate price display
        # This allows trades to be overlaid on the price chart by timestamp
        df['close'] = df_ohlcv['close']

        return df

    def _split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test based on SSOT config.

        Respects use_fixed_dates flag:
          - True: split by explicit date boundaries (default)
          - False: split by ratio (train_ratio, val_ratio, test_ratio)
        """
        dates = self.config.date_ranges
        time_col = 'timestamp' if 'timestamp' in df.columns else None

        if dates.use_fixed_dates:
            # Fixed date splits (default, recommended for reproducibility)
            train_end = pd.Timestamp(dates.train_end)
            val_end = pd.Timestamp(dates.val_end)
            test_end = pd.Timestamp(dates.test_end) if hasattr(dates, 'test_end') and dates.test_end else None

            if time_col:
                train_df = df[df[time_col] <= train_end].copy()
                val_df = df[(df[time_col] > train_end) & (df[time_col] <= val_end)].copy()
                test_mask = df[time_col] > val_end
                if test_end is not None:
                    test_mask &= df[time_col] <= test_end
                test_df = df[test_mask].copy()
            else:
                train_df = df[df.index <= train_end].copy()
                val_df = df[(df.index > train_end) & (df.index <= val_end)].copy()
                test_mask = df.index > val_end
                if test_end is not None:
                    test_mask &= df.index <= test_end
                test_df = df[test_mask].copy()

            logger.info(
                f"Split (fixed dates): train={len(train_df)} (until {dates.train_end}), "
                f"val={len(val_df)} (until {dates.val_end}), "
                f"test={len(test_df)}"
            )
        else:
            # Ratio-based splits
            n = len(df)
            train_end_idx = int(n * dates.train_ratio)
            val_end_idx = int(n * (dates.train_ratio + dates.val_ratio))

            train_df = df.iloc[:train_end_idx].copy()
            val_df = df.iloc[train_end_idx:val_end_idx].copy()
            test_df = df.iloc[val_end_idx:].copy()

            logger.info(
                f"Split (ratios {dates.train_ratio}/{dates.val_ratio}/{dates.test_ratio}): "
                f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
            )

        return train_df, val_df, test_df

    def _compute_norm_stats(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Compute normalization statistics from training data only."""
        stats = {}

        for feature_def in self.config.get_market_features():
            col = feature_def.name
            if col not in train_df.columns:
                continue

            norm = feature_def.normalization
            values = train_df[col].dropna()

            if len(values) == 0:
                stats[col] = {"mean": 0.0, "std": 1.0, "method": norm.method}
                continue

            if norm.method == "zscore":
                mean = float(values.mean())
                std = float(values.std())
                if std < 1e-8:
                    std = 1.0
                stats[col] = {"mean": mean, "std": std, "method": "zscore"}

            elif norm.method == "minmax":
                input_range = norm.input_range or (float(values.min()), float(values.max()))
                output_range = norm.output_range or (0.0, 1.0)
                stats[col] = {
                    "input_min": input_range[0],
                    "input_max": input_range[1],
                    "output_min": output_range[0],
                    "output_max": output_range[1],
                    "method": "minmax"
                }

            else:
                stats[col] = {"method": norm.method}

        # Add metadata
        stats["_meta"] = {
            "version": self.config.version,
            "ssot_model": self.config.based_on_model,
            "train_rows": len(train_df),
            "feature_count": len(feature_cols),
            "created_at": datetime.now().isoformat(),
        }

        return stats

    def _apply_normalization(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        norm_stats: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply normalization to features using pre-computed stats."""
        df = df.copy()

        for feature_def in self.config.get_market_features():
            col = feature_def.name
            if col not in df.columns or col not in norm_stats:
                continue

            stats = norm_stats[col]
            norm = feature_def.normalization
            values = df[col]

            if stats.get("method") == "zscore":
                normalized = (values - stats["mean"]) / max(stats["std"], 1e-8)
            elif stats.get("method") == "minmax":
                in_min, in_max = stats["input_min"], stats["input_max"]
                out_min, out_max = stats["output_min"], stats["output_max"]
                normalized = (values - in_min) / max(in_max - in_min, 1e-8)
                normalized = normalized * (out_max - out_min) + out_min
            else:
                normalized = values

            # Apply clipping
            if norm.clip:
                low, high = norm.clip
                normalized = normalized.clip(low, high)

            df[col] = normalized

        return df

    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by dropping NaN rows."""
        initial_len = len(df)

        # Get feature columns
        feature_cols = [f.name for f in self.config.get_market_features()]
        available_cols = [c for c in feature_cols if c in df.columns]

        # Drop rows with NaN in feature columns
        df = df.dropna(subset=available_cols)

        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Cleaned: dropped {dropped} rows with NaN ({dropped/initial_len:.1%})")

        return df

    def _compute_descriptive_stats(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Compute descriptive statistics for each dataset split.

        Returns dict with per-split, per-feature stats:
        mean, std, min, max, median, q25, q75, skew, kurtosis, nan_count, zero_pct
        Plus dataset-level summaries: date range, shape, correlations.
        """
        stats = {}

        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            available = [c for c in feature_cols if c in df.columns]
            split_stats = {
                "shape": list(df.shape),
                "date_range": [str(df.index.min()), str(df.index.max())],
                "features": {},
            }

            for col in available:
                s = df[col]
                split_stats["features"][col] = {
                    "mean": round(float(s.mean()), 6),
                    "std": round(float(s.std()), 6),
                    "min": round(float(s.min()), 6),
                    "q25": round(float(s.quantile(0.25)), 6),
                    "median": round(float(s.median()), 6),
                    "q75": round(float(s.quantile(0.75)), 6),
                    "max": round(float(s.max()), 6),
                    "skew": round(float(s.skew()), 4),
                    "kurtosis": round(float(s.kurtosis()), 4),
                    "nan_count": int(s.isna().sum()),
                    "zero_pct": round(float((s == 0).mean() * 100), 2),
                }

            # Aux columns stats (raw_log_ret_*, close)
            raw_ret_col = next((c for c in df.columns if c.startswith("raw_log_ret_")), None)
            aux_cols = [raw_ret_col, "close"] if raw_ret_col else ["close"]
            for aux_col in aux_cols:
                if aux_col in df.columns:
                    s = df[aux_col]
                    split_stats["features"][f"_aux_{aux_col}"] = {
                        "mean": round(float(s.mean()), 6),
                        "std": round(float(s.std()), 6),
                        "min": round(float(s.min()), 6),
                        "max": round(float(s.max()), 6),
                        "nan_count": int(s.isna().sum()),
                    }

            # Top-5 correlations with raw return (proxy for predictive power)
            if raw_ret_col and raw_ret_col in df.columns and len(available) > 0:
                corrs = df[available].corrwith(df[raw_ret_col]).abs().sort_values(ascending=False)
                split_stats["top_return_correlations"] = {
                    k: round(float(v), 4) for k, v in corrs.head(5).items()
                }

            stats[split_name] = split_stats

        # Cross-split distribution drift check (train vs val means)
        drift = {}
        for col in [c for c in feature_cols if c in train_df.columns and c in val_df.columns]:
            train_mean = train_df[col].mean()
            val_mean = val_df[col].mean()
            train_std = train_df[col].std()
            if train_std > 1e-8:
                drift[col] = round(float(abs(val_mean - train_mean) / train_std), 4)
        # Only report features with drift > 0.5 std
        stats["distribution_drift"] = {k: v for k, v in sorted(drift.items(), key=lambda x: -x[1]) if v > 0.5}

        return stats

    def _log_descriptive_stats(self, stats: Dict[str, Any], feature_cols: List[str]) -> None:
        """Log descriptive statistics summary to console."""
        logger.info("=" * 70)
        logger.info("L2 DESCRIPTIVE STATISTICS")
        logger.info("=" * 70)

        for split_name in ["train", "val", "test"]:
            s = stats[split_name]
            logger.info(f"\n  [{split_name.upper()}] Shape: {s['shape']}, Period: {s['date_range'][0]} → {s['date_range'][1]}")
            logger.info(f"  {'Feature':<25s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Median':>10s} {'Max':>10s} {'Skew':>8s} {'Kurt':>8s} {'NaN':>5s}")
            logger.info(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*5}")

            for col in feature_cols:
                if col in s["features"]:
                    f = s["features"][col]
                    logger.info(
                        f"  {col:<25s} {f['mean']:>10.4f} {f['std']:>10.4f} {f['min']:>10.4f} "
                        f"{f['median']:>10.4f} {f['max']:>10.4f} {f['skew']:>8.2f} {f['kurtosis']:>8.2f} {f['nan_count']:>5d}"
                    )

            # Aux columns (dynamic raw_log_ret_* + close)
            for aux_key in sorted(k for k in s["features"] if k.startswith("_aux_")):
                f = s["features"][aux_key]
                label = aux_key.replace("_aux_", "")
                logger.info(f"  {label:<25s} {f['mean']:>10.6f} {f['std']:>10.6f} {f['min']:>10.4f} {'':>10s} {f['max']:>10.4f} {'':>8s} {'':>8s} {f['nan_count']:>5d}")

            # Top correlations
            if "top_return_correlations" in s:
                corrs = s["top_return_correlations"]
                corr_str = ", ".join([f"{k}={v:.3f}" for k, v in corrs.items()])
                logger.info(f"  Top |corr| with return: {corr_str}")

        # Drift warnings
        drift = stats.get("distribution_drift", {})
        if drift:
            logger.info(f"\n  DISTRIBUTION DRIFT (train→val, >0.5 std):")
            for feat, d in list(drift.items())[:10]:
                level = "WARNING" if d > 1.0 else "info"
                logger.info(f"    {feat}: {d:.2f} std {'⚠️' if d > 1.0 else ''}")
        else:
            logger.info(f"\n  Distribution drift: None detected (all features <0.5 std)")

        logger.info("=" * 70)

    def _build_lineage(
        self,
        df_ohlcv: pd.DataFrame,
        df_macro: Optional[pd.DataFrame],
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Build lineage tracking information."""
        feature_cols = [f.name for f in self.config.get_market_features()]
        feature_order_hash = hashlib.md5(",".join(feature_cols).encode()).hexdigest()

        return {
            "ssot_version": self.config.version,
            "ssot_model": self.config.based_on_model,
            "feature_count": len(feature_cols),
            "feature_order": feature_cols,
            "feature_order_hash": feature_order_hash,
            "observation_dim": len(feature_cols) + len(self.config.get_state_features()),
            "date_ranges": {
                "train_start": self.config.date_ranges.train_start,
                "train_end": self.config.date_ranges.train_end,
                "val_start": self.config.date_ranges.val_start,
                "val_end": self.config.date_ranges.val_end,
                "test_start": self.config.date_ranges.test_start,
                "test_end": self.config.date_ranges.test_end,
            },
            "row_counts": {
                "ohlcv_input": len(df_ohlcv),
                "macro_input": len(df_macro) if df_macro is not None else 0,
                "train_output": len(train_df),
                "val_output": len(val_df),
                "test_output": len(test_df),
            },
            "created_at": datetime.now().isoformat(),
        }

    def _save_outputs(
        self,
        output_dir: Path,
        prefix: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        norm_stats: Dict[str, Any],
        lineage: Dict[str, Any],
        descriptive_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save all outputs to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save datasets
        train_df.to_parquet(output_dir / f"{prefix}_train.parquet")
        val_df.to_parquet(output_dir / f"{prefix}_val.parquet")
        test_df.to_parquet(output_dir / f"{prefix}_test.parquet")

        # Save norm stats
        with open(output_dir / f"{prefix}_norm_stats.json", 'w') as f:
            json.dump(norm_stats, f, indent=2)

        # Save lineage
        with open(output_dir / f"{prefix}_lineage.json", 'w') as f:
            json.dump(lineage, f, indent=2)

        # Save descriptive statistics
        if descriptive_stats:
            with open(output_dir / f"{prefix}_descriptive_stats.json", 'w') as f:
                json.dump(descriptive_stats, f, indent=2)

        logger.info(f"Saved outputs to {output_dir}/{prefix}_*")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def build_production_dataset(
    df_ohlcv: pd.DataFrame,
    df_macro: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None
) -> DatasetBuildResult:
    """
    Convenience function to build production dataset.

    Args:
        df_ohlcv: OHLCV DataFrame
        df_macro: Macro DataFrame (optional)
        output_dir: Output directory (optional)

    Returns:
        DatasetBuildResult
    """
    builder = SSOTDatasetBuilder()
    return builder.build(df_ohlcv, df_macro, output_dir)


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build datasets from SSOT")
    parser.add_argument("--ohlcv", type=str, help="Path to OHLCV parquet file")
    parser.add_argument("--macro", type=str, help="Path to macro parquet file")
    parser.add_argument("--output", type=str, default="data/pipeline/07_output/5min")
    args = parser.parse_args()

    if args.ohlcv:
        df_ohlcv = pd.read_parquet(args.ohlcv)
        df_macro = pd.read_parquet(args.macro) if args.macro else None

        result = build_production_dataset(
            df_ohlcv,
            df_macro,
            Path(args.output)
        )

        print(f"\nDataset built successfully!")
        print(f"  Train: {len(result.train_df)} rows")
        print(f"  Val: {len(result.val_df)} rows")
        print(f"  Test: {len(result.test_df)} rows")
        print(f"  Features: {len(result.feature_columns)}")
    else:
        print("No input files provided. Run with --ohlcv and optionally --macro")
