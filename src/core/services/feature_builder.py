"""
USD/COP Trading System - Feature Builder (Consolidated)
========================================================

Single Source of Truth (SSOT) for feature calculation and normalization.
Consolidates ~1,200 lines of duplicated code from 7 locations.

References:
- ARQUITECTURA_INTEGRAL_V3.md (Sections 11.0.1, 11.0.3, 12.2)
- MAPEO_MIGRACION_BIDIRECCIONAL.md (Parts 1-4)
- config/feature_config.json (v3.1.0)

Source files consolidated:
1. data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py:217-303
2. data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py
3. data/pipeline/03_processing/scripts/03_create_rl_datasets.py
4. notebooks/pipeline entrenamiento/src/utils.py:13-90
5. airflow/dags/usdcop_m5__06_l5_realtime_inference.py (inline)
6. services/trading_api_realtime.py (inline)
7. services/feature_calculator.py

Author: Pedro @ Lean Tech Solutions
Version: 2.1.0
Date: 2025-12-17

CRITICAL UPDATES IN v2.1:
- Macro z-score stats now read from config (SSOT compliance - no hardcoding)
- Added _load_macro_zscore_stats() method for DIP compliance

CRITICAL UPDATES IN v2.0:
- _calc_rsi: Fixed period=9 (was 14)
- _calc_atr_pct: Fixed period=10 (was 14)
- calc_adx: Fixed period=14 (confirmed)
- calculate_time_normalized: Now uses (bar_number - 1) / 60 formula → [0, 0.983]
- build_observation: Generates exactly 15 dimensions with proper validation
- validate_observation: Verifies shape, NaN, and value ranges
"""

import numpy as np
import pandas as pd
import sys
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Add src to path for flexible imports (works both as package and standalone)
_src_path = Path(__file__).parent.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Import from shared - use flexible import strategy
def _import_shared_modules():
    """Import shared modules with fallback for different import contexts."""
    import importlib.util

    # Try to find modules in src/shared
    shared_path = _src_path / "shared"

    # Import config_loader
    config_spec = importlib.util.spec_from_file_location(
        "config_loader", shared_path / "config_loader.py"
    )
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)

    # Import exceptions
    exc_spec = importlib.util.spec_from_file_location(
        "exceptions", shared_path / "exceptions.py"
    )
    exc_module = importlib.util.module_from_spec(exc_spec)
    exc_spec.loader.exec_module(exc_module)

    return config_module, exc_module

try:
    # Try relative imports first (when used as installed package)
    from ...shared.config_loader import get_config, ConfigLoader
    from ...shared.exceptions import (
        ValidationError,
        FeatureCalculationError,
        ObservationDimensionError,
        FeatureMissingError
    )
except (ImportError, ValueError):
    # Fallback: direct file imports (avoids __init__.py circular deps)
    _config_mod, _exc_mod = _import_shared_modules()
    get_config = _config_mod.get_config
    ConfigLoader = _config_mod.ConfigLoader
    ValidationError = _exc_mod.ValidationError
    FeatureCalculationError = _exc_mod.FeatureCalculationError
    ObservationDimensionError = _exc_mod.ObservationDimensionError
    FeatureMissingError = _exc_mod.FeatureMissingError


class FeatureBuilder:
    """
    Consolidated feature calculator for USD/COP Trading System.

    Provides unified methods for:
    - Technical indicator calculation (RSI, ATR, ADX)
    - Log return calculations
    - Feature normalization (z-score with clipping)
    - Observation vector construction for RL model

    ALL calculations are self-contained - no external dependencies on
    services/feature_calculator.py or other files.

    Usage:
        # Initialization
        builder = FeatureBuilder()  # Uses default config
        builder = FeatureBuilder('path/to/feature_config.json')  # Custom config

        # Batch processing (training)
        features_df = builder.build_batch(ohlcv_df, macro_df)

        # Single observation (inference)
        obs = builder.build_observation(features_dict, position=0.0, bar_number=30)
    """

    # EXACT indicator periods from MAPEO (CRITICAL - DO NOT CHANGE)
    RSI_PERIOD = 9   # feature_config.json:122
    ATR_PERIOD = 10  # feature_config.json:131
    ADX_PERIOD = 14  # feature_config.json:139

    # Episode configuration
    BARS_PER_SESSION = 60

    # Global clip bounds after z-score normalization
    GLOBAL_CLIP_MIN = -4.0
    GLOBAL_CLIP_MAX = 4.0

    def __init__(self, config_path: Optional[str] = None, config_loader: Optional[ConfigLoader] = None):
        """
        Initialize FeatureBuilder.

        Args:
            config_path: Path to feature_config.json (optional, uses default if None)
            config_loader: Optional ConfigLoader instance for dependency injection (DIP compliance).
                          If provided, config_path is ignored and this loader is used instead.

        DIP Compliance:
            For full dependency injection, pass a ConfigLoader instance:
                loader = ConfigLoader('/custom/config/path')
                builder = FeatureBuilder(config_loader=loader)

            Or use the default singleton:
                builder = FeatureBuilder()  # Uses global singleton
        """
        # Load config via dependency injection or shared singleton
        if config_loader is not None:
            self.config = config_loader
        else:
            self.config = get_config(config_path)

        # Cache frequently used values
        self._feature_order = self.config.get_feature_order()
        self._obs_dim = self.config.get_obs_dim()
        trading_params = self.config.get_trading_params()
        self._episode_length = trading_params.get('bars_per_session', self.BARS_PER_SESSION)

        # Load normalization stats
        self._norm_stats = self._load_norm_stats()

        # Load macro z-score stats from config (SSOT compliance - no hardcoding)
        self._macro_zscore_stats = self._load_macro_zscore_stats()

    def _load_norm_stats(self) -> Dict[str, Dict[str, float]]:
        """Load normalization statistics for all features."""
        stats = {}
        for feature in self._feature_order:
            norm = self.config.get_norm_stats(feature)
            clip = self.config.get_clip_bounds(feature)
            stats[feature] = {
                'mean': norm.get('mean', 0.0) if norm else 0.0,
                'std': norm.get('std', 1.0) if norm else 1.0,
                'clip_min': clip[0] if clip else self.GLOBAL_CLIP_MIN,
                'clip_max': clip[1] if clip else self.GLOBAL_CLIP_MAX,
            }
        return stats

    def _load_macro_zscore_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Load macro z-score statistics from config (SSOT compliance).

        Instead of hardcoding mean/std values for DXY, VIX, EMBI, this method
        reads them from feature_config.json to maintain single source of truth.

        SOLID: Fail Fast - raises error if config is missing required values.

        Returns:
            Dict with macro z-score stats: {feature_name: {mean, std, clip_min, clip_max}}

        Raises:
            ValueError: If required macro z-score stats are missing from config
        """
        macro_features = ['dxy_z', 'vix_z', 'embi_z']
        stats = {}

        for feature in macro_features:
            norm = self.config.get_norm_stats(feature)
            clip = self.config.get_clip_bounds(feature)

            if norm:
                stats[feature] = {
                    'mean': norm.get('mean', 0.0),
                    'std': norm.get('std', 1.0),
                    'clip_min': clip[0] if clip else -4.0,
                    'clip_max': clip[1] if clip else 4.0,
                }
            else:
                # SOLID: Fail Fast - no hardcoded fallbacks
                raise ValueError(
                    f"Missing norm_stats for '{feature}' in feature_config.json. "
                    f"SSOT requires all macro z-score stats to be defined in config. "
                    f"Add to validation.norm_stats_validation section."
                )

        return stats

    # =========================================================================
    # TECHNICAL INDICATOR CALCULATIONS
    # Source: 01_build_5min_datasets.py lines 217-303
    # FORMULAS: EXACT per MAPEO_MIGRACION_BIDIRECCIONAL.md Part 3
    # =========================================================================

    def _calc_log_return(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """
        Calculate logarithmic returns.
        Source: 01_build_5min_datasets.py:217-219
        Formula: ln(close / close[-periods])
        """
        return np.log(series / series.shift(periods))

    def _calc_rsi(self, close: pd.Series, period: int = 9) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index) - range 0-100.
        Source: 01_build_5min_datasets.py:221-227

        CRITICAL: Default period=9 per MAPEO Part 3, Line 360

        Args:
            close: Close prices
            period: RSI period (default 9 per MAPEO)

        Returns:
            RSI values (0-100)
        """
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calc_atr(self, high: pd.Series, low: pd.Series,
                  close: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Average True Range.
        Source: 01_build_5min_datasets.py:229-235

        CRITICAL: Default period=10 per MAPEO Part 3, Line 361
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calc_atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 10) -> pd.Series:
        """
        Calculate ATR as percentage of close price.
        Source: 01_build_5min_datasets.py:237-240
        Formula: (ATR / close) * 100

        CRITICAL: Default period=10 per MAPEO Part 3, Line 361

        Args:
            high, low, close: OHLC prices
            period: ATR period (default 10 per MAPEO)

        Returns:
            ATR% values
        """
        atr = self._calc_atr(high, low, close, period)
        return (atr / close) * 100

    def _calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
        """
        Calculate ADX (Average Directional Index) - range 0-100.
        Source: 01_build_5min_datasets.py:242-261

        CRITICAL: Default period=14 per MAPEO Part 3, Line 362

        Args:
            high, low, close: OHLC prices
            period: ADX period (default 14 per MAPEO)

        Returns:
            ADX values (0-100)
        """
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Directional Indicators
        plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=period).mean() / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=period).mean() / (atr + 1e-10)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        return adx

    # =========================================================================
    # UTILITY FUNCTIONS
    # Source: 01_build_5min_datasets.py lines 285-294
    # =========================================================================

    def _z_score(self, value: float, mean: float, std: float) -> float:
        """Calculate z-score for a single value."""
        if std <= 0:
            return 0.0
        return (value - mean) / std

    def _z_score_series(self, series: pd.Series, mean: float, std: float) -> pd.Series:
        """Calculate z-score for a series with fixed statistics."""
        if std <= 0:
            return pd.Series(0.0, index=series.index)
        return (series - mean) / std

    def _pct_change_safe(self, series: pd.Series, periods: int = 1,
                         clip_min: float = -0.1, clip_max: float = 0.1) -> pd.Series:
        """
        Calculate percentage change with clipping.
        Source: 01_build_5min_datasets.py:292-294
        """
        return series.pct_change(periods).clip(clip_min, clip_max)

    def _clip(self, value: float, min_val: float, max_val: float) -> float:
        """Clip a value to specified bounds."""
        return max(min_val, min(max_val, value))

    # =========================================================================
    # NORMALIZATION
    # Source: notebooks/pipeline entrenamiento/src/utils.py lines 44-90
    # =========================================================================

    def normalize_feature(self, name: str, value: float) -> float:
        """
        Normalize a single feature value using stored statistics.

        Args:
            name: Feature name (must be in feature_order)
            value: Raw feature value

        Returns:
            Normalized value (z-score + clip)

        Raises:
            FeatureMissingError: If feature not found in config
        """
        if name not in self._norm_stats:
            raise FeatureMissingError(feature_name=name)

        stats = self._norm_stats[name]

        # Handle NaN
        if pd.isna(value):
            return 0.0

        # Z-score normalization
        normalized = self._z_score(value, stats['mean'], stats['std'])

        # Apply feature-specific clip
        normalized = self._clip(normalized, stats['clip_min'], stats['clip_max'])

        # Apply global clip
        normalized = self._clip(normalized, self.GLOBAL_CLIP_MIN, self.GLOBAL_CLIP_MAX)

        return normalized

    def normalize_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize all features in a dataframe using config stats.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with normalized features
        """
        result = df.copy()

        # Normalize each feature that has stats defined
        for feat in self._feature_order:
            if feat in result.columns:
                stats = self.config.get_norm_stats(feat)
                if stats:
                    mean = stats['mean']
                    std = stats['std']
                    result[feat] = (result[feat] - mean) / (std + 1e-10)
                    result[feat] = result[feat].clip(-4.0, 4.0)

        return result

    # =========================================================================
    # TIME NORMALIZATION
    # CRITICAL: EXACT formula per environment.py:117
    # =========================================================================

    def calculate_time_normalized(self, bar_number: int, episode_length: Optional[int] = None) -> float:
        """
        Calculate time_normalized value for observation.

        CRITICAL FORMULA: time_normalized = (bar_number - 1) / episode_length
        This gives range [0, 0.983] for 60-bar episodes, NOT [0, 1].

        Source: MAPEO_MIGRACION_BIDIRECCIONAL.md Line 370
        Reference: notebooks/pipeline entrenamiento/src/environment.py:117

        Args:
            bar_number: Current bar number in episode (1-60)
            episode_length: Total bars per episode (default: 60)

        Returns:
            time_normalized value in range [0, 0.983] for 60 bars

        Example:
            >>> calculate_time_normalized(1)   # First bar
            0.0
            >>> calculate_time_normalized(30)  # Middle bar
            0.483333...
            >>> calculate_time_normalized(60)  # Last bar
            0.983333...
        """
        if episode_length is None:
            episode_length = self._episode_length

        # CRITICAL: (bar_number - 1) / episode_length
        # For bar_number in [1, 60] and episode_length=60:
        # - bar 1:  (1-1)/60 = 0/60 = 0.0
        # - bar 30: (30-1)/60 = 29/60 = 0.483
        # - bar 60: (60-1)/60 = 59/60 = 0.983
        return (bar_number - 1) / episode_length

    # =========================================================================
    # OBSERVATION CONSTRUCTION
    # =========================================================================

    def build_observation(self,
                         features_dict: Dict[str, float],
                         position: float,
                         bar_number: int,
                         episode_length: Optional[int] = None) -> np.ndarray:
        """
        Build complete 15-dim observation vector for the model.

        CRITICAL FORMULA: time_normalized = (bar_number - 1) / episode_length
        This gives range [0, 0.983] for 60-bar episodes, NOT [0, 1].

        EXACT implementation per MAPEO Part 3, Lines 301-364

        Args:
            features_dict: Dict with 13 feature values
            position: Current position (-1 to 1)
            bar_number: Current bar number in episode (1-60)
            episode_length: Total bars per episode (default: 60)

        Returns:
            np.ndarray of shape (15,) ready for model.predict()

        Raises:
            ObservationDimensionError: If observation dimension != 15
            FeatureMissingError: If required feature is missing

        Example:
            >>> features = {
            ...     'log_ret_5m': 0.0002,
            ...     'log_ret_1h': 0.0005,
            ...     'log_ret_4h': 0.0012,
            ...     'rsi_9': 55.0,
            ...     'atr_pct': 0.08,
            ...     'adx_14': 25.0,
            ...     'dxy_z': -0.5,
            ...     'dxy_change_1d': 0.001,
            ...     'vix_z': 0.3,
            ...     'embi_z': 0.1,
            ...     'brent_change_1d': 0.02,
            ...     'rate_spread': -0.5,
            ...     'usdmxn_ret_1h': 0.0003
            ... }
            >>> obs = builder.build_observation(features, position=0.5, bar_number=30)
            >>> obs.shape
            (15,)
        """
        if episode_length is None:
            episode_length = self._episode_length

        # Extract 13 features in correct order
        feature_values = []
        for feat in self._feature_order:
            if feat not in features_dict:
                raise FeatureMissingError(feature_name=feat)

            val = features_dict.get(feat, 0.0)
            if pd.isna(val) or np.isnan(val):
                val = 0.0
            feature_values.append(float(val))

        # Add state variables
        # CRITICAL: time_normalized = (bar_number - 1) / episode_length
        # For bar_number in [1, 60]: time_normalized in [0, 0.983]
        time_normalized = self.calculate_time_normalized(bar_number, episode_length)

        # Construct observation: [13 features] + [position] + [time_normalized]
        obs = np.array(
            feature_values + [position, time_normalized],
            dtype=np.float32
        )

        # Final clip to [-5, 5] as in environment
        obs = np.clip(obs, -5.0, 5.0)

        # Validate dimension
        if obs.shape[0] != self._obs_dim:
            raise ObservationDimensionError(
                expected=self._obs_dim,
                actual=obs.shape[0]
            )

        return obs

    def build_batch(self,
                   ohlcv_df: pd.DataFrame,
                   macro_df: Optional[pd.DataFrame] = None,
                   normalize: bool = True) -> pd.DataFrame:
        """
        Build complete feature set for a batch of OHLCV data.

        This is the main entry point for dataset construction, combining:
        1. Technical features from OHLCV (log returns, RSI, ATR%, ADX)
        2. Macro features from daily data (DXY, VIX, EMBI, etc.)
        3. Normalization (optional)

        ALL calculations are self-contained - no external dependencies.

        EXACT formulas per MAPEO_MIGRACION_BIDIRECCIONAL.md Part 3

        Args:
            ohlcv_df: DataFrame with [time, open, high, low, close]
            macro_df: DataFrame with [date, dxy, vix, embi, brent, ...] (optional)
            normalize: Apply z-score normalization (default: True)

        Returns:
            DataFrame with all 13 features computed and normalized
        """
        df = ohlcv_df.copy()

        # =====================================================================
        # 1. Calculate Returns (log)
        # Source: 01_build_5min_datasets.py:329-332
        # MAPEO Part 3, Lines 357-359
        # =====================================================================
        df['log_ret_5m'] = self._calc_log_return(df['close'], periods=1)
        df['log_ret_1h'] = self._calc_log_return(df['close'], periods=12)
        df['log_ret_4h'] = self._calc_log_return(df['close'], periods=48)

        # Save raw returns for reward calculation
        df['_raw_ret_5m'] = df['log_ret_5m'].copy()

        # =====================================================================
        # 2. Calculate Technical Indicators
        # Source: 01_build_5min_datasets.py:343-346
        # MAPEO Part 3, Lines 360-362
        # CRITICAL: RSI period=9, ATR period=10, ADX period=14
        # =====================================================================
        df['rsi_9'] = self._calc_rsi(df['close'], period=self.RSI_PERIOD)
        df['atr_pct'] = self._calc_atr_pct(
            df['high'], df['low'], df['close'],
            period=self.ATR_PERIOD
        )
        df['adx_14'] = self._calc_adx(
            df['high'], df['low'], df['close'],
            period=self.ADX_PERIOD
        )

        # =====================================================================
        # 3. Merge Macro Data if provided
        # Source: 01_build_5min_datasets.py:387-441
        # MAPEO Part 3, Lines 363-369
        # =====================================================================
        if macro_df is not None:
            macro = macro_df.copy()

            # Ensure macro has date index for merge
            if 'date' in macro.columns:
                macro['date'] = pd.to_datetime(macro['date']).dt.date
                macro = macro.set_index('date')

            # Create date column from time for merging
            if 'time' in df.columns:
                df['_date'] = pd.to_datetime(df['time']).dt.date
            elif 'datetime' in df.columns:
                df['_date'] = pd.to_datetime(df['datetime']).dt.date

            # Forward-fill macro data to 5-min bars
            if '_date' in df.columns:
                for col in ['dxy', 'vix', 'embi', 'brent', 'treasury_2y',
                            'treasury_10y', 'usdmxn']:
                    if col in macro.columns:
                        df[col] = df['_date'].map(macro[col].to_dict())

            # === Calculate Macro Z-scores (SSOT: read from config) ===
            # DXY z-score - Line 363
            if 'dxy' in df.columns:
                dxy_stats = self._macro_zscore_stats['dxy_z']
                df['dxy_z'] = (df['dxy'] - dxy_stats['mean']) / dxy_stats['std']
                df['dxy_z'] = df['dxy_z'].clip(dxy_stats['clip_min'], dxy_stats['clip_max'])

            # VIX z-score - Line 364
            if 'vix' in df.columns:
                vix_stats = self._macro_zscore_stats['vix_z']
                df['vix_z'] = (df['vix'] - vix_stats['mean']) / vix_stats['std']
                df['vix_z'] = df['vix_z'].clip(vix_stats['clip_min'], vix_stats['clip_max'])

            # EMBI z-score - Line 365
            if 'embi' in df.columns:
                embi_stats = self._macro_zscore_stats['embi_z']
                df['embi_z'] = (df['embi'] - embi_stats['mean']) / embi_stats['std']
                df['embi_z'] = df['embi_z'].clip(embi_stats['clip_min'], embi_stats['clip_max'])

            # === Calculate Macro Changes ===
            # DXY daily change (clip to [-0.03, 0.03]) - Line 364
            if 'dxy' in df.columns:
                df['dxy_change_1d'] = self._pct_change_safe(
                    df['dxy'], periods=288, clip_min=-0.03, clip_max=0.03
                )

            # Brent daily change (clip to [-0.10, 0.10]) - Line 367
            if 'brent' in df.columns:
                df['brent_change_1d'] = self._pct_change_safe(
                    df['brent'], periods=288, clip_min=-0.10, clip_max=0.10
                )

            # Rate spread - Line 368
            if 'treasury_10y' in df.columns and 'treasury_2y' in df.columns:
                df['rate_spread'] = df['treasury_10y'] - df['treasury_2y']

            # USDMXN hourly return (clip to [-0.1, 0.1]) - Line 369
            # CRITICAL: periods=12 per feature_config.json:204
            if 'usdmxn' in df.columns:
                df['usdmxn_ret_1h'] = self._pct_change_safe(
                    df['usdmxn'], periods=12, clip_min=-0.1, clip_max=0.1
                )

            # Clean up temp column
            if '_date' in df.columns:
                df = df.drop(columns=['_date'])

        # =====================================================================
        # 4. Apply normalization if requested
        # =====================================================================
        if normalize:
            df = self.normalize_batch(df)

        return df

    # =========================================================================
    # VALIDATION & UTILITIES
    # =========================================================================

    def validate_observation(self, obs: np.ndarray, check_range: bool = True) -> bool:
        """
        Validate that observation meets model requirements.

        Args:
            obs: Observation array
            check_range: Check if values are within expected range (default: True)

        Returns:
            True if valid

        Raises:
            ObservationDimensionError: If shape is incorrect
            ValidationError: If contains NaN/Inf or values out of range
        """
        # Check shape
        if obs.shape != (self._obs_dim,):
            raise ObservationDimensionError(
                expected=self._obs_dim,
                actual=obs.shape[0]
            )

        # Check for NaN/Inf
        if np.any(np.isnan(obs)):
            raise ValidationError("Observation contains NaN values")

        if np.any(np.isinf(obs)):
            raise ValidationError("Observation contains infinite values")

        # Check value range (observations should be clipped to [-5, 5])
        if check_range:
            if np.any(obs < -5.0) or np.any(obs > 5.0):
                raise ValidationError(
                    f"Observation values out of range [-5, 5]: min={obs.min():.3f}, max={obs.max():.3f}"
                )

        return True

    def get_feature_info(self) -> Dict[str, Dict]:
        """
        Get information about all features.

        Returns:
            Dict with feature metadata (periods, norm stats, etc.)
        """
        info = {}

        for feat in self._feature_order:
            feat_info = {
                'name': feat,
                'norm_stats': self.config.get_norm_stats(feat),
                'clip_bounds': self.config.get_clip_bounds(feat)
            }

            # Add period for technical indicators
            if feat in ['rsi_9', 'atr_pct', 'adx_14']:
                feat_info['period'] = self.config.get_technical_period(feat)

            info[feat] = feat_info

        return info

    def get_feature_order(self) -> List[str]:
        """Get ordered list of 13 feature names."""
        return self._feature_order.copy()

    def get_observation_dim(self) -> int:
        """Get total observation dimension (13 features + 2 state = 15)."""
        return self._obs_dim

    def get_norm_stats(self) -> Dict[str, Dict[str, float]]:
        """Get normalization statistics for all features."""
        return self._norm_stats.copy()

    @property
    def feature_order(self) -> List[str]:
        """Get ordered list of 13 features (property alias)"""
        return self._feature_order

    @property
    def obs_dim(self) -> int:
        """Get total observation dimension (property alias)"""
        return self._obs_dim

    @property
    def version(self) -> str:
        """Get configuration version"""
        return self.config.version

    # =========================================================================
    # Public API for Testing Compatibility (Aliases to private methods)
    # =========================================================================

    def calc_log_return(self, series: pd.Series, periods: int = 1) -> pd.Series:
        """Public alias for _calc_log_return (for testing compatibility)"""
        return self._calc_log_return(series, periods)

    def calc_rsi(self, close: pd.Series, period: int = 9) -> pd.Series:
        """Public alias for _calc_rsi (for testing compatibility)"""
        return self._calc_rsi(close, period)

    def calc_atr(self, high: pd.Series, low: pd.Series,
                 close: pd.Series, period: int = 10) -> pd.Series:
        """Public alias for _calc_atr (for testing compatibility)"""
        return self._calc_atr(high, low, close, period)

    def calc_atr_pct(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 10) -> pd.Series:
        """Public alias for _calc_atr_pct (for testing compatibility)"""
        return self._calc_atr_pct(high, low, close, period)

    def calc_adx(self, high: pd.Series, low: pd.Series, close: pd.Series,
                 period: int = 14) -> pd.Series:
        """Public alias for _calc_adx (for testing compatibility)"""
        return self._calc_adx(high, low, close, period)

    def calc_pct_change(self, series: pd.Series, periods: int = 1,
                        clip_range: Optional[Tuple[float, float]] = None) -> pd.Series:
        """Public alias for _pct_change_safe (for testing compatibility)"""
        return self._pct_change_safe(series, periods, clip_range)

    def normalize_zscore(self, series: Union[pd.Series, float], mean: float,
                        std: float, clip: Optional[float] = None) -> Union[pd.Series, float]:
        """
        Public alias for z-score normalization (for testing compatibility).

        Args:
            series: Series or single value to normalize
            mean: Mean for normalization
            std: Standard deviation for normalization
            clip: Optional clipping bound (symmetric, e.g., 10.0 means [-10, 10])

        Returns:
            Normalized series or value
        """
        if isinstance(series, pd.Series):
            result = self._z_score_series(series, mean, std)
            if clip is not None:
                result = result.clip(-abs(clip), abs(clip))
            return result
        else:
            result = self._z_score(series, mean, std)
            if clip is not None:
                result = self._clip(result, -abs(clip), abs(clip))
            return result

    def compute_technical_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Public alias for computing technical indicators (for testing compatibility).

        Args:
            ohlcv_df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with technical indicators: rsi_9, atr_pct, adx_14
        """
        result = ohlcv_df.copy()

        result['rsi_9'] = self._calc_rsi(ohlcv_df['close'], period=self.RSI_PERIOD)
        result['atr_pct'] = self._calc_atr_pct(
            ohlcv_df['high'],
            ohlcv_df['low'],
            ohlcv_df['close'],
            period=self.ATR_PERIOD
        )
        result['adx_14'] = self._calc_adx(
            ohlcv_df['high'],
            ohlcv_df['low'],
            ohlcv_df['close'],
            period=self.ADX_PERIOD
        )

        return result

    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Public alias for normalize_batch (for testing compatibility).

        Args:
            features_df: DataFrame with features to normalize

        Returns:
            DataFrame with normalized features
        """
        return self.normalize_batch(features_df)

    def get_norm_stats(self, feature_name: str) -> Dict[str, float]:
        """
        Get normalization statistics for a specific feature (for testing compatibility).

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with 'mean' and 'std' keys, or {'mean': 0.0, 'std': 1.0} if not found
        """
        return self._norm_stats.get(feature_name, {'mean': 0.0, 'std': 1.0})


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_feature_builder(
    config_path: Optional[str] = None,
    config_loader: Optional[ConfigLoader] = None
) -> FeatureBuilder:
    """
    Create a FeatureBuilder instance (factory function).

    Args:
        config_path: Path to feature_config.json (optional)
        config_loader: ConfigLoader instance for dependency injection (optional)

    Returns:
        Configured FeatureBuilder instance

    DIP Usage:
        # Default (uses singleton):
        builder = create_feature_builder()

        # Custom config path:
        builder = create_feature_builder('/path/to/config')

        # Dependency injection:
        loader = ConfigLoader('/custom/path')
        builder = create_feature_builder(config_loader=loader)
    """
    return FeatureBuilder(config_path=config_path, config_loader=config_loader)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FeatureBuilder v2.0.0 - USD/COP Trading System")
    print("=" * 80)

    # Initialize builder
    builder = FeatureBuilder()

    print(f"\nConfiguration version: {builder.version}")
    print(f"Observation dimension: {builder.obs_dim}")
    print(f"\nFeatures (in order):")
    for i, feat in enumerate(builder.feature_order, 1):
        period = builder.config.get_technical_period(feat)
        if period:
            print(f"  {i:2d}. {feat} (period={period})")
        else:
            print(f"  {i:2d}. {feat}")

    # Show critical periods
    print("\n" + "-" * 80)
    print("CRITICAL INDICATOR PERIODS:")
    print(f"  RSI:  period={builder.RSI_PERIOD} (EXACT per MAPEO)")
    print(f"  ATR:  period={builder.ATR_PERIOD} (EXACT per MAPEO)")
    print(f"  ADX:  period={builder.ADX_PERIOD} (EXACT per MAPEO)")
    print("-" * 80)

    # Test time normalization
    print("\nTesting time_normalized calculation:")
    print(f"  Bar 1:  {builder.calculate_time_normalized(1):.6f}  (expected: 0.000000)")
    print(f"  Bar 30: {builder.calculate_time_normalized(30):.6f}  (expected: 0.483333)")
    print(f"  Bar 60: {builder.calculate_time_normalized(60):.6f}  (expected: 0.983333)")

    # Test observation building
    print("\n" + "-" * 80)
    print("Testing observation construction...")
    print("-" * 80)

    test_features = {feat: 0.0 for feat in builder.feature_order}
    test_features['log_ret_5m'] = 0.0002
    test_features['rsi_9'] = 55.0

    obs = builder.build_observation(
        test_features,
        position=0.5,
        bar_number=30
    )

    print(f"Observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Sample values (first 5): {obs[:5]}")
    print(f"State values: position={obs[13]:.3f}, time_normalized={obs[14]:.6f}")

    # Validate
    try:
        builder.validate_observation(obs)
        print("\nValidation: PASSED [OK]")
    except (ObservationDimensionError, ValidationError) as e:
        print(f"\nValidation: FAILED [ERROR] - {e}")

    print("\n" + "=" * 80)
