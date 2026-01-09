#!/usr/bin/env python
"""
Feature Builder for USD/COP RL Trading System V19
==================================================

Transforms raw OHLCV and macro data into normalized observation vectors
that exactly match what was used during model training.

This is the Single Source of Truth (SSOT) for feature engineering
during inference to ensure training-production parity.

Author: Claude Code
Version: 1.0.0
Date: 2025-12-26
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class NormStats:
    """Normalization statistics for a single feature."""
    mean: Optional[float]
    std: Optional[float]
    clip_min: Optional[float]
    clip_max: Optional[float]
    method: str  # 'z_score', 'clip_only', 'none'

    def normalize(self, value: float) -> float:
        """Apply normalization to a value."""
        if self.method == 'z_score' and self.mean is not None and self.std is not None:
            normalized = (value - self.mean) / self.std
            if self.clip_min is not None and self.clip_max is not None:
                normalized = np.clip(normalized, self.clip_min, self.clip_max)
            return normalized
        elif self.method == 'clip_only':
            if self.clip_min is not None and self.clip_max is not None:
                return np.clip(value, self.clip_min, self.clip_max)
            return value
        else:  # 'none'
            return value

    def denormalize(self, value: float) -> float:
        """Reverse normalization to get original value."""
        if self.method == 'z_score' and self.mean is not None and self.std is not None:
            return value * self.std + self.mean
        return value


# ==============================================================================
# MAIN FEATURE BUILDER CLASS
# ==============================================================================

class FeatureBuilderV19:
    """
    Feature builder for V19 environment.

    Transforms raw market data into observation vectors matching training.

    Usage:
        builder = FeatureBuilderV19()
        obs = builder.build_observation(ohlcv_df, macro_df)

        # For validation
        is_valid = builder.validate_observation(obs)

        # Get feature names
        names = builder.get_feature_names()
    """

    # Feature order EXACTLY as in training (state features + market features)
    MARKET_FEATURE_ORDER = [
        'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
        'rsi_9', 'atr_pct', 'adx_14', 'bb_position',
        'dxy_z', 'dxy_change_1d', 'dxy_mom_5d',
        'vix_z', 'embi_z',
        'brent_change_1d', 'brent_vol_5d',
        'rate_spread', 'usdmxn_change_1d',
        'hour_sin', 'hour_cos'
    ]

    STATE_FEATURE_ORDER = [
        'position', 'unrealized_pnl', 'cumulative_return',
        'current_drawdown', 'max_drawdown_episode', 'regime_encoded',
        'session_phase', 'volatility_regime', 'cost_regime',
        'position_duration', 'trade_count_normalized', 'time_remaining'
    ]

    # Normalization stats from training
    NORM_STATS = {
        'log_ret_5m': NormStats(mean=2.0e-06, std=0.001138, clip_min=-0.05, clip_max=0.05, method='z_score'),
        'log_ret_1h': NormStats(mean=2.3e-05, std=0.003776, clip_min=-0.05, clip_max=0.05, method='z_score'),
        'log_ret_4h': NormStats(mean=5.2e-05, std=0.007768, clip_min=-0.05, clip_max=0.05, method='z_score'),
        'rsi_9': NormStats(mean=49.27, std=23.07, clip_min=None, clip_max=None, method='z_score'),
        'atr_pct': NormStats(mean=0.062, std=0.0446, clip_min=None, clip_max=None, method='z_score'),
        'adx_14': NormStats(mean=32.01, std=16.36, clip_min=None, clip_max=None, method='z_score'),
        'bb_position': NormStats(mean=None, std=None, clip_min=0.0, clip_max=1.0, method='clip_only'),
        'dxy_z': NormStats(mean=100.21, std=5.60, clip_min=-4, clip_max=4, method='z_score'),
        'dxy_change_1d': NormStats(mean=None, std=None, clip_min=-0.03, clip_max=0.03, method='clip_only'),
        'dxy_mom_5d': NormStats(mean=None, std=None, clip_min=-0.05, clip_max=0.05, method='clip_only'),
        'vix_z': NormStats(mean=21.16, std=7.89, clip_min=-4, clip_max=4, method='z_score'),
        'embi_z': NormStats(mean=322.01, std=62.68, clip_min=-4, clip_max=4, method='z_score'),
        'brent_change_1d': NormStats(mean=None, std=None, clip_min=-0.10, clip_max=0.10, method='clip_only'),
        'brent_vol_5d': NormStats(mean=None, std=None, clip_min=0.0, clip_max=0.05, method='clip_only'),
        'rate_spread': NormStats(mean=7.03, std=1.41, clip_min=None, clip_max=None, method='z_score'),
        'usdmxn_change_1d': NormStats(mean=None, std=None, clip_min=-0.10, clip_max=0.10, method='clip_only'),
        'hour_sin': NormStats(mean=None, std=None, clip_min=None, clip_max=None, method='none'),
        'hour_cos': NormStats(mean=None, std=None, clip_min=None, clip_max=None, method='none'),
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        include_state_features: bool = False,
    ):
        """
        Initialize feature builder.

        Args:
            config_path: Path to normalization_stats_v19.json (optional)
            include_state_features: If True, include environment state features
        """
        self.include_state_features = include_state_features

        # Load config from JSON if provided
        if config_path is not None:
            self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Load normalization stats from JSON config."""
        path = Path(config_path)
        if not path.exists():
            warnings.warn(f"Config file not found: {config_path}. Using defaults.")
            return

        with open(path, 'r') as f:
            config = json.load(f)

        # Update NORM_STATS from config
        if 'market_features' in config:
            for name, stats in config['market_features'].items():
                if name in self.NORM_STATS:
                    self.NORM_STATS[name] = NormStats(
                        mean=stats.get('mean'),
                        std=stats.get('std'),
                        clip_min=stats.get('clip_min'),
                        clip_max=stats.get('clip_max'),
                        method=stats.get('method', 'none'),
                    )

    # ==========================================================================
    # TECHNICAL INDICATOR CALCULATIONS
    # ==========================================================================

    @staticmethod
    def calc_rsi(close: pd.Series, period: int = 9) -> float:
        """Calculate RSI for the last value."""
        if len(close) < period + 1:
            return 50.0  # Default neutral value

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    @staticmethod
    def calc_atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> float:
        """Calculate ATR as percentage of close."""
        if len(close) < period + 1:
            return 0.05  # Default value

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        atr_pct = (atr / close) * 100

        return float(atr_pct.iloc[-1]) if not np.isnan(atr_pct.iloc[-1]) else 0.05

    @staticmethod
    def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate ADX."""
        if len(close) < period * 2:
            return 25.0  # Default value

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 25.0

    @staticmethod
    def calc_bb_position(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> float:
        """Calculate position within Bollinger Bands."""
        if len(close) < period:
            return 0.5  # Default middle position

        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()

        upper = sma + std_dev * std
        lower = sma - std_dev * std

        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]

        if current_upper == current_lower:
            return 0.5

        position = (current_close - current_lower) / (current_upper - current_lower)
        return float(np.clip(position, 0.0, 1.0))

    @staticmethod
    def calc_log_return(close: pd.Series, periods: int = 1) -> float:
        """Calculate log return."""
        if len(close) < periods + 1:
            return 0.0

        current = close.iloc[-1]
        previous = close.iloc[-periods - 1]

        if previous <= 0:
            return 0.0

        return float(np.log(current / previous))

    # ==========================================================================
    # TIME ENCODING
    # ==========================================================================

    @staticmethod
    def encode_hour_cyclical(timestamp: pd.Timestamp) -> Tuple[float, float]:
        """Encode hour as cyclical (sin, cos)."""
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        return float(hour_sin), float(hour_cos)

    # ==========================================================================
    # MACRO FEATURE CALCULATIONS
    # ==========================================================================

    @staticmethod
    def calc_pct_change(series: pd.Series, periods: int = 1) -> float:
        """Calculate percent change."""
        if len(series) < periods + 1:
            return 0.0

        current = series.iloc[-1]
        previous = series.iloc[-periods - 1]

        if previous == 0:
            return 0.0

        return float((current - previous) / previous)

    @staticmethod
    def calc_rolling_std(series: pd.Series, period: int = 5) -> float:
        """Calculate rolling standard deviation."""
        if len(series) < period:
            return 0.01

        return float(series.tail(period).std())

    # ==========================================================================
    # MAIN OBSERVATION BUILDER
    # ==========================================================================

    def build_market_features(
        self,
        ohlcv_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """
        Build market feature vector from OHLCV and macro data.

        Args:
            ohlcv_df: DataFrame with columns [timestamp, open, high, low, close]
                      Must have at least 50 rows for technical indicators
            macro_df: DataFrame with macro indicators (optional, can be pre-merged)
            timestamp: Timestamp for time encoding (default: last row)

        Returns:
            np.array of shape (18,) with normalized market features
        """
        features = {}

        # Ensure we have the required columns
        required_cols = ['close']
        for col in required_cols:
            if col not in ohlcv_df.columns:
                raise ValueError(f"Missing required column: {col}")

        close = ohlcv_df['close']
        high = ohlcv_df.get('high', close)
        low = ohlcv_df.get('low', close)

        # === RETURNS ===
        features['log_ret_5m'] = self.calc_log_return(close, periods=1)
        features['log_ret_1h'] = self.calc_log_return(close, periods=12)
        features['log_ret_4h'] = self.calc_log_return(close, periods=48)

        # === TECHNICAL INDICATORS ===
        features['rsi_9'] = self.calc_rsi(close, period=9)
        features['atr_pct'] = self.calc_atr_pct(high, low, close, period=10)
        features['adx_14'] = self.calc_adx(high, low, close, period=14)
        features['bb_position'] = self.calc_bb_position(close, period=20)

        # === MACRO FEATURES ===
        if macro_df is not None and len(macro_df) > 0:
            # DXY features
            if 'dxy' in macro_df.columns:
                dxy = macro_df['dxy']
                features['dxy_z'] = dxy.iloc[-1]  # Will be normalized
                features['dxy_change_1d'] = self.calc_pct_change(dxy, periods=1)
                features['dxy_mom_5d'] = self.calc_pct_change(dxy, periods=5)
            else:
                features['dxy_z'] = 100.0  # Default
                features['dxy_change_1d'] = 0.0
                features['dxy_mom_5d'] = 0.0

            # VIX
            if 'vix' in macro_df.columns:
                features['vix_z'] = macro_df['vix'].iloc[-1]
            else:
                features['vix_z'] = 20.0

            # EMBI
            if 'embi' in macro_df.columns:
                features['embi_z'] = macro_df['embi'].iloc[-1]
            else:
                features['embi_z'] = 300.0

            # Brent
            if 'brent' in macro_df.columns:
                brent = macro_df['brent']
                features['brent_change_1d'] = self.calc_pct_change(brent, periods=1)
                features['brent_vol_5d'] = self.calc_rolling_std(brent.pct_change().dropna(), period=5)
            else:
                features['brent_change_1d'] = 0.0
                features['brent_vol_5d'] = 0.01

            # Rate spread
            if 'treasury_10y' in macro_df.columns:
                features['rate_spread'] = 10.0 - macro_df['treasury_10y'].iloc[-1]
            else:
                features['rate_spread'] = 6.0

            # USD/MXN
            if 'usdmxn' in macro_df.columns:
                features['usdmxn_change_1d'] = self.calc_pct_change(macro_df['usdmxn'], periods=1)
            else:
                features['usdmxn_change_1d'] = 0.0
        else:
            # Check if macro features are pre-computed in ohlcv_df
            macro_cols = ['dxy_z', 'dxy_change_1d', 'dxy_mom_5d', 'vix_z', 'embi_z',
                         'brent_change_1d', 'brent_vol_5d', 'rate_spread', 'usdmxn_change_1d']
            for col in macro_cols:
                if col in ohlcv_df.columns:
                    features[col] = ohlcv_df[col].iloc[-1]
                else:
                    # Defaults
                    defaults = {
                        'dxy_z': 100.0, 'dxy_change_1d': 0.0, 'dxy_mom_5d': 0.0,
                        'vix_z': 20.0, 'embi_z': 300.0,
                        'brent_change_1d': 0.0, 'brent_vol_5d': 0.01,
                        'rate_spread': 6.0, 'usdmxn_change_1d': 0.0
                    }
                    features[col] = defaults.get(col, 0.0)

        # === TIME ENCODING ===
        if timestamp is None:
            if 'timestamp' in ohlcv_df.columns:
                timestamp = pd.Timestamp(ohlcv_df['timestamp'].iloc[-1])
            else:
                timestamp = pd.Timestamp.now()

        hour_sin, hour_cos = self.encode_hour_cyclical(timestamp)
        features['hour_sin'] = hour_sin
        features['hour_cos'] = hour_cos

        # === NORMALIZE AND ORDER ===
        obs = []
        for feature_name in self.MARKET_FEATURE_ORDER:
            raw_value = features.get(feature_name, 0.0)

            # Handle NaN
            if np.isnan(raw_value):
                raw_value = 0.0

            # Normalize
            if feature_name in self.NORM_STATS:
                normalized = self.NORM_STATS[feature_name].normalize(raw_value)
            else:
                normalized = raw_value

            obs.append(normalized)

        return np.array(obs, dtype=np.float32)

    def build_observation(
        self,
        ohlcv_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
        state_dict: Optional[Dict] = None,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """
        Build complete observation vector.

        Args:
            ohlcv_df: OHLCV DataFrame
            macro_df: Macro indicators DataFrame (optional)
            state_dict: Environment state dictionary (for state features)
            timestamp: Timestamp for time encoding

        Returns:
            np.array with complete observation
        """
        market_features = self.build_market_features(ohlcv_df, macro_df, timestamp)

        if self.include_state_features and state_dict is not None:
            state_features = self._build_state_features(state_dict)
            return np.concatenate([state_features, market_features])

        return market_features

    def _build_state_features(self, state_dict: Dict) -> np.ndarray:
        """Build state features from environment state dictionary."""
        features = []

        max_dd_pct = state_dict.get('max_drawdown_pct', 0.15)

        defaults = {
            'position': 0.0,
            'unrealized_pnl': 0.0,
            'cumulative_return': 0.0,
            'current_drawdown': 0.0,
            'max_drawdown_episode': 0.0,
            'regime_encoded': 0.0,
            'session_phase': 0.5,
            'volatility_regime': 0.5,
            'cost_regime': 1.0,
            'position_duration': 0,
            'trade_count': 0,
            'current_step': 0,
            'episode_length': 400,
        }

        for key, default in defaults.items():
            state_dict.setdefault(key, default)

        # Build normalized state features
        features.append(state_dict['position'])  # Already [-1, 1]
        features.append(np.clip(state_dict['unrealized_pnl'] / 0.05, -1, 1))
        features.append(np.clip(state_dict['cumulative_return'] / 0.10, -1, 1))
        features.append(-state_dict['current_drawdown'] / max_dd_pct)
        features.append(-state_dict['max_drawdown_episode'] / max_dd_pct)
        features.append(state_dict['regime_encoded'])
        features.append(state_dict['session_phase'])
        features.append(state_dict['volatility_regime'])
        features.append(state_dict['cost_regime'])
        features.append(min(state_dict['position_duration'] / 100, 1.0))
        features.append(min(state_dict['trade_count'] / 50, 1.0))
        features.append(1.0 - (state_dict['current_step'] / state_dict['episode_length']))

        return np.array(features, dtype=np.float32)

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    def validate_observation(self, obs: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Validate observation vector.

        Args:
            obs: Observation array

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        # Check dimension
        expected_dim = len(self.MARKET_FEATURE_ORDER)
        if self.include_state_features:
            expected_dim += len(self.STATE_FEATURE_ORDER)

        if len(obs) != expected_dim:
            issues.append(f"Dimension mismatch: got {len(obs)}, expected {expected_dim}")

        # Check for NaN/Inf
        if np.any(np.isnan(obs)):
            nan_indices = np.where(np.isnan(obs))[0]
            issues.append(f"NaN values at indices: {nan_indices.tolist()}")

        if np.any(np.isinf(obs)):
            inf_indices = np.where(np.isinf(obs))[0]
            issues.append(f"Inf values at indices: {inf_indices.tolist()}")

        # Check extreme values
        if np.any(np.abs(obs) > 10):
            extreme_indices = np.where(np.abs(obs) > 10)[0]
            issues.append(f"Extreme values (|x| > 10) at indices: {extreme_indices.tolist()}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        if self.include_state_features:
            return self.STATE_FEATURE_ORDER + self.MARKET_FEATURE_ORDER
        return self.MARKET_FEATURE_ORDER.copy()

    def get_market_feature_names(self) -> List[str]:
        """Get market feature names only."""
        return self.MARKET_FEATURE_ORDER.copy()

    def get_state_feature_names(self) -> List[str]:
        """Get state feature names only."""
        return self.STATE_FEATURE_ORDER.copy()

    def denormalize_feature(self, name: str, value: float) -> float:
        """
        Denormalize a feature value back to original scale.

        Args:
            name: Feature name
            value: Normalized value

        Returns:
            Original scale value
        """
        if name in self.NORM_STATS:
            return self.NORM_STATS[name].denormalize(value)
        return value

    def get_normalization_stats(self, name: str) -> Optional[Dict]:
        """Get normalization stats for a feature."""
        if name in self.NORM_STATS:
            stats = self.NORM_STATS[name]
            return {
                'mean': stats.mean,
                'std': stats.std,
                'clip_min': stats.clip_min,
                'clip_max': stats.clip_max,
                'method': stats.method,
            }
        return None


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def load_feature_builder(
    config_dir: Optional[str] = None,
    include_state_features: bool = False,
) -> FeatureBuilderV19:
    """
    Load feature builder with config from standard location.

    Args:
        config_dir: Directory containing config files
        include_state_features: Whether to include environment state

    Returns:
        Configured FeatureBuilderV19
    """
    if config_dir is None:
        # Default location
        config_dir = Path(__file__).parent.parent.parent / "config" / "features"

    config_path = Path(config_dir) / "normalization_stats_v19.json"

    if config_path.exists():
        return FeatureBuilderV19(
            config_path=str(config_path),
            include_state_features=include_state_features,
        )

    return FeatureBuilderV19(include_state_features=include_state_features)


def build_observation_from_row(
    row: pd.Series,
    builder: Optional[FeatureBuilderV19] = None,
) -> np.ndarray:
    """
    Build observation from a single row (for pre-computed datasets).

    Args:
        row: Pandas Series with all feature columns
        builder: FeatureBuilderV19 instance (creates new if None)

    Returns:
        Normalized observation array
    """
    if builder is None:
        builder = FeatureBuilderV19()

    # Convert row to DataFrame for builder
    df = pd.DataFrame([row])
    return builder.build_market_features(df)


# ==============================================================================
# CLI / TESTING
# ==============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature Builder V19')
    parser.add_argument('--test', action='store_true', help='Run self-test')
    parser.add_argument('--csv', type=str, help='Path to CSV file to process')

    args = parser.parse_args()

    if args.test:
        print("Running self-test...")

        # Create sample data
        np.random.seed(42)
        n_rows = 100

        dates = pd.date_range(start='2025-01-01 13:00', periods=n_rows, freq='5min')
        close = 4200 + np.cumsum(np.random.randn(n_rows) * 5)

        ohlcv = pd.DataFrame({
            'timestamp': dates,
            'open': close - np.random.rand(n_rows) * 2,
            'high': close + np.random.rand(n_rows) * 3,
            'low': close - np.random.rand(n_rows) * 3,
            'close': close,
        })

        macro = pd.DataFrame({
            'date': dates.date,
            'dxy': 100 + np.random.randn(n_rows) * 0.5,
            'vix': 20 + np.random.randn(n_rows) * 2,
            'embi': 300 + np.random.randn(n_rows) * 10,
            'brent': 80 + np.random.randn(n_rows) * 2,
            'treasury_10y': 4.5 + np.random.randn(n_rows) * 0.1,
            'usdmxn': 17 + np.random.randn(n_rows) * 0.1,
        })

        # Build features
        builder = FeatureBuilderV19()
        obs = builder.build_market_features(ohlcv, macro)

        print(f"Feature names: {builder.get_feature_names()}")
        print(f"Observation shape: {obs.shape}")
        print(f"Observation values:\n{obs}")

        # Validate
        is_valid, issues = builder.validate_observation(obs)
        print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            for issue in issues:
                print(f"  - {issue}")

        # Test denormalization
        print("\nDenormalization test:")
        for i, name in enumerate(builder.get_feature_names()[:5]):
            original = builder.denormalize_feature(name, obs[i])
            print(f"  {name}: normalized={obs[i]:.4f} -> original={original:.4f}")

    elif args.csv:
        print(f"Processing {args.csv}...")
        df = pd.read_csv(args.csv, parse_dates=['timestamp'])

        builder = FeatureBuilderV19()
        obs = builder.build_market_features(df)

        print(f"Features computed: {len(obs)}")
        for name, value in zip(builder.get_feature_names(), obs):
            print(f"  {name}: {value:.6f}")

    else:
        print("Usage: python feature_builder.py --test")
        print("       python feature_builder.py --csv path/to/data.csv")
