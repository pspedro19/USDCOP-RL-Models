"""
USD/COP Trading System - Feature Builder (Refactored with SOLID & Design Patterns)
===================================================================================

Refactored version applying:
- SOLID principles
- Dependency Injection
- Factory Pattern
- Strategy Pattern
- Builder Pattern
- Template Method Pattern

Author: Pedro @ Lean Tech Solutions
Version: 3.0.0
Date: 2025-12-17

IMPROVEMENTS:
- Single Responsibility: Each calculator handles one feature type
- Open/Closed: New calculators can be added without modifying existing code
- Liskov Substitution: All calculators implement IFeatureCalculator
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depends on abstractions, not concretions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path

# Core interfaces
from ..interfaces.feature_calculator import IFeatureCalculator
from ..interfaces.normalizer import INormalizer
from ..interfaces.observation_builder import IObservationBuilder
from ..interfaces.config_loader import IConfigLoader

# Factories
from ..factories.feature_calculator_factory import FeatureCalculatorFactory
from ..factories.normalizer_factory import NormalizerFactory

# Calculators
from ..calculators import (
    RSICalculator,
    ATRCalculator,
    ADXCalculator,
    ReturnsCalculator,
    MacroZScoreCalculator,
    MacroChangeCalculator
)

# Normalizers
from ..normalizers import (
    ZScoreNormalizer,
    ClipNormalizer,
    NoOpNormalizer,
    CompositeNormalizer
)

# Builders
from ..builders import ObservationBuilder

# Shared
from ...shared.config_loader_adapter import ConfigLoaderAdapter
from ...shared.exceptions import (
    ValidationError,
    FeatureCalculationError,
    ObservationDimensionError,
    FeatureMissingError
)


class FeatureBuilderRefactored:
    """
    Refactored feature builder using SOLID principles and Design Patterns.

    DEPENDENCY INJECTION:
    - Accepts IConfigLoader for configuration
    - Accepts calculators via factory
    - Accepts normalizers via strategy pattern

    SINGLE RESPONSIBILITY:
    - Only orchestrates feature calculation
    - Delegates actual calculations to specialized calculators
    - Delegates normalization to normalizer strategies
    - Delegates observation building to ObservationBuilder

    Usage:
        # With default configuration
        builder = FeatureBuilderRefactored()
        features_df = builder.build_batch(ohlcv_df, macro_df)
        obs = builder.build_observation(features_dict, position=0.0, bar_number=30)

        # With custom configuration (Dependency Injection)
        config = ConfigLoaderAdapter('path/to/config')
        builder = FeatureBuilderRefactored(config=config)
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

    def __init__(self, config: Optional[IConfigLoader] = None):
        """
        Initialize FeatureBuilder with Dependency Injection.

        Args:
            config: Configuration loader (injected dependency)
        """
        # Inject or create config dependency
        self._config = config if config is not None else ConfigLoaderAdapter()

        # Cache frequently used values
        self._feature_order = self._config.get_feature_order()
        self._obs_dim = self._config.get_obs_dim()
        trading_params = self._config.get_trading_params()
        self._episode_length = trading_params.get('bars_per_session', self.BARS_PER_SESSION)

        # Initialize factories
        self._setup_factories()

        # Create calculators using factory pattern
        self._calculators = self._create_calculators()

        # Create observation builder
        self._obs_builder = ObservationBuilder(
            feature_order=self._feature_order,
            obs_dim=self._obs_dim,
            global_clip_min=-5.0,  # Final clip for observations
            global_clip_max=5.0
        )

    def _setup_factories(self) -> None:
        """
        Register calculator and normalizer types in factories.

        This enables the Factory Pattern for creating instances.
        """
        # Register feature calculators
        FeatureCalculatorFactory.register('rsi', RSICalculator)
        FeatureCalculatorFactory.register('atr', ATRCalculator)
        FeatureCalculatorFactory.register('adx', ADXCalculator)
        FeatureCalculatorFactory.register('returns', ReturnsCalculator)
        FeatureCalculatorFactory.register('macro_zscore', MacroZScoreCalculator)
        FeatureCalculatorFactory.register('macro_change', MacroChangeCalculator)

        # Register normalizers
        NormalizerFactory.register('zscore', ZScoreNormalizer)
        NormalizerFactory.register('clip', ClipNormalizer)
        NormalizerFactory.register('noop', NoOpNormalizer)
        NormalizerFactory.register('composite', CompositeNormalizer)

    def _create_calculators(self) -> Dict[str, IFeatureCalculator]:
        """
        Create all feature calculators using Factory Pattern.

        Returns:
            Dictionary mapping feature names to calculator instances
        """
        calculators = {}

        # OHLCV-based calculators
        for feature in self._feature_order:
            norm_stats = self._config.get_norm_stats(feature)
            clip_bounds = self._config.get_clip_bounds(feature)

            # Create normalizer using factory
            if norm_stats:
                normalizer = NormalizerFactory.create_from_config(
                    feature, norm_stats, clip_bounds
                )
            else:
                normalizer = None

            # Create calculator based on feature type
            if feature.startswith('log_ret_'):
                # Returns calculator
                if '5m' in feature:
                    calc = FeatureCalculatorFactory.create(
                        'returns',
                        periods=1,
                        name=feature,
                        normalizer=normalizer,
                        clip_bounds=clip_bounds
                    )
                elif '1h' in feature:
                    calc = FeatureCalculatorFactory.create(
                        'returns',
                        periods=12,
                        name=feature,
                        normalizer=normalizer,
                        clip_bounds=clip_bounds
                    )
                elif '4h' in feature:
                    calc = FeatureCalculatorFactory.create(
                        'returns',
                        periods=48,
                        name=feature,
                        normalizer=normalizer,
                        clip_bounds=clip_bounds
                    )
                else:
                    continue

            elif feature == 'rsi_9':
                calc = FeatureCalculatorFactory.create(
                    'rsi',
                    period=self.RSI_PERIOD,
                    normalizer=normalizer,
                    clip_bounds=clip_bounds
                )

            elif feature == 'atr_pct':
                calc = FeatureCalculatorFactory.create(
                    'atr',
                    period=self.ATR_PERIOD,
                    as_percentage=True,
                    normalizer=normalizer,
                    clip_bounds=clip_bounds
                )

            elif feature == 'adx_14':
                calc = FeatureCalculatorFactory.create(
                    'adx',
                    period=self.ADX_PERIOD,
                    normalizer=normalizer,
                    clip_bounds=clip_bounds
                )

            # Macro z-score calculators (SSOT: read mean/std from config)
            elif feature == 'dxy_z':
                if not norm_stats:
                    raise ValueError(f"Missing norm_stats for '{feature}' in config (SSOT)")
                calc = FeatureCalculatorFactory.create(
                    'macro_zscore',
                    indicator='dxy',
                    mean=norm_stats.get('mean'),
                    std=norm_stats.get('std'),
                    name=feature,
                    clip_bounds=clip_bounds or (-4.0, 4.0)
                )

            elif feature == 'vix_z':
                if not norm_stats:
                    raise ValueError(f"Missing norm_stats for '{feature}' in config (SSOT)")
                calc = FeatureCalculatorFactory.create(
                    'macro_zscore',
                    indicator='vix',
                    mean=norm_stats.get('mean'),
                    std=norm_stats.get('std'),
                    name=feature,
                    clip_bounds=clip_bounds or (-4.0, 4.0)
                )

            elif feature == 'embi_z':
                if not norm_stats:
                    raise ValueError(f"Missing norm_stats for '{feature}' in config (SSOT)")
                calc = FeatureCalculatorFactory.create(
                    'macro_zscore',
                    indicator='embi',
                    mean=norm_stats.get('mean'),
                    std=norm_stats.get('std'),
                    name=feature,
                    clip_bounds=clip_bounds or (-4.0, 4.0)
                )

            # Macro change calculators
            elif feature == 'dxy_change_1d':
                calc = FeatureCalculatorFactory.create(
                    'macro_change',
                    indicator='dxy',
                    periods=288,
                    name=feature,
                    clip_bounds=(-0.03, 0.03)
                )

            elif feature == 'brent_change_1d':
                calc = FeatureCalculatorFactory.create(
                    'macro_change',
                    indicator='brent',
                    periods=288,
                    name=feature,
                    clip_bounds=(-0.10, 0.10)
                )

            elif feature == 'usdmxn_ret_1h':
                calc = FeatureCalculatorFactory.create(
                    'macro_change',
                    indicator='usdmxn',
                    periods=12,
                    name=feature,
                    clip_bounds=(-0.1, 0.1)
                )

            elif feature == 'rate_spread':
                # Rate spread is calculated directly in build_batch
                continue

            else:
                continue

            calculators[feature] = calc

        return calculators

    # =========================================================================
    # BATCH PROCESSING (for training datasets)
    # =========================================================================

    def build_batch(self,
                   ohlcv_df: pd.DataFrame,
                   macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Build complete feature set for a batch of OHLCV data.

        Uses dependency-injected calculators for each feature.

        Args:
            ohlcv_df: DataFrame with [time, open, high, low, close]
            macro_df: DataFrame with [date, dxy, vix, embi, brent, ...] (optional)

        Returns:
            DataFrame with all 13 features computed
        """
        df = ohlcv_df.copy()

        # Calculate OHLCV-based features using calculators
        for feature_name, calculator in self._calculators.items():
            try:
                # Check if calculator can run on current data
                if all(dep in df.columns for dep in calculator.get_dependencies()):
                    df[feature_name] = calculator.calculate(df)
            except Exception as e:
                # Log error but continue with other features
                print(f"Warning: Failed to calculate {feature_name}: {e}")

        # Merge macro data if provided
        if macro_df is not None:
            df = self._merge_macro_data(df, macro_df)

            # Calculate rate spread (derived feature)
            if 'treasury_10y' in df.columns and 'treasury_2y' in df.columns:
                df['rate_spread'] = df['treasury_10y'] - df['treasury_2y']

        # Save raw returns for reward calculation
        if 'log_ret_5m' in df.columns:
            df['_raw_ret_5m'] = df['log_ret_5m'].copy()

        return df

    def _merge_macro_data(self, df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge macro data into OHLCV dataframe.

        Args:
            df: OHLCV dataframe
            macro_df: Macro indicators dataframe

        Returns:
            Merged dataframe
        """
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

            df = df.drop(columns=['_date'])

        return df

    # =========================================================================
    # OBSERVATION CONSTRUCTION (for inference)
    # =========================================================================

    def build_observation(self,
                         features_dict: Dict[str, float],
                         position: float,
                         bar_number: int,
                         episode_length: Optional[int] = None) -> np.ndarray:
        """
        Build complete 15-dim observation vector using Builder Pattern.

        Args:
            features_dict: Dict with 13 feature values
            position: Current position (-1 to 1)
            bar_number: Current bar number in episode (1-60)
            episode_length: Total bars per episode (default: 60)

        Returns:
            np.ndarray of shape (15,) ready for model.predict()

        Example:
            >>> obs = builder.build_observation(
            ...     features={'log_ret_5m': 0.0002, 'rsi_9': 55.0, ...},
            ...     position=0.5,
            ...     bar_number=30
            ... )
        """
        if episode_length is None:
            episode_length = self._episode_length

        # Use Builder Pattern for clean construction
        obs = (self._obs_builder
               .reset()
               .with_features(features_dict)
               .with_position(position)
               .with_time_normalized(bar_number, episode_length)
               .build())

        return obs

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
        Get information about all calculators.

        Returns:
            Dict with calculator metadata
        """
        info = {}
        for feat, calc in self._calculators.items():
            info[feat] = calc.get_params()
        return info

    def get_feature_order(self) -> List[str]:
        """Get ordered list of 13 feature names."""
        return self._feature_order.copy()

    def get_observation_dim(self) -> int:
        """Get total observation dimension (13 features + 2 state = 15)."""
        return self._obs_dim

    def get_calculators(self) -> Dict[str, IFeatureCalculator]:
        """
        Get all feature calculators (for testing/inspection).

        Returns:
            Dictionary of feature_name -> calculator
        """
        return self._calculators.copy()

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
        return self._config.version

    @property
    def config(self) -> IConfigLoader:
        """Get configuration loader (for testing)"""
        return self._config


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_feature_builder(config: Optional[IConfigLoader] = None) -> FeatureBuilderRefactored:
    """
    Create a FeatureBuilder instance (factory function).

    Args:
        config: Optional configuration loader (injected dependency)

    Returns:
        Configured FeatureBuilderRefactored instance
    """
    return FeatureBuilderRefactored(config=config)
