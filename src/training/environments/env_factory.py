"""
Environment Factory - Creates Training Environments
===================================================
Factory pattern for creating configured trading environments.

SOLID Principles:
- Single Responsibility: Only creates environments
- Open/Closed: New env types via registration
- Dependency Inversion: Depends on TradingEnvConfig abstraction

Design Patterns:
- Factory Method: Creates environments from config
- Builder Pattern: Builds complex env configurations
- Registry Pattern: Stores env type registrations
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Type, Callable, Any, List, Tuple
import pandas as pd
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from .trading_env import (
    TradingEnvironment,
    TradingEnvConfig,
    DefaultRewardStrategy,
    RewardStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Reward Strategy Registry
# =============================================================================

class RewardStrategyRegistry:
    """Registry of available reward strategies"""

    _strategies: Dict[str, Type[RewardStrategy]] = {
        "default": DefaultRewardStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: Type[RewardStrategy]) -> None:
        """Register a new reward strategy"""
        cls._strategies[name] = strategy_class
        logger.info(f"Registered reward strategy: {name}")

    @classmethod
    def get(cls, name: str, **kwargs) -> RewardStrategy:
        """Get reward strategy by name"""
        if name not in cls._strategies:
            raise ValueError(
                f"Unknown reward strategy: {name}. "
                f"Available: {list(cls._strategies.keys())}"
            )
        return cls._strategies[name](**kwargs)

    @classmethod
    def list_strategies(cls) -> List[str]:
        """List available strategies"""
        return list(cls._strategies.keys())


# =============================================================================
# Environment Factory
# =============================================================================

class EnvironmentFactory:
    """
    Factory for creating trading environments.

    Handles:
    - Loading and validating datasets
    - Loading normalization statistics
    - Creating configured environments
    - Creating vectorized environments for training

    Usage:
        factory = EnvironmentFactory(project_root=Path("."))

        # Create single environment
        env = factory.create(
            dataset_path=Path("data/training.csv"),
            norm_stats_path=Path("config/norm_stats.json"),
            config=TradingEnvConfig(),
        )

        # Create vectorized environment
        vec_env = factory.create_vec_env(
            dataset_path=Path("data/training.csv"),
            norm_stats_path=Path("config/norm_stats.json"),
            config=TradingEnvConfig(),
            n_envs=4,
        )
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._dataset_cache: Dict[str, pd.DataFrame] = {}
        self._norm_stats_cache: Dict[str, Dict] = {}

    def create(
        self,
        dataset_path: Path,
        norm_stats_path: Path,
        config: Optional[TradingEnvConfig] = None,
        reward_strategy: Optional[str] = "default",
        reward_kwargs: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> TradingEnvironment:
        """
        Create a single trading environment.

        Args:
            dataset_path: Path to training dataset CSV
            norm_stats_path: Path to normalization stats JSON
            config: Environment configuration (default if None)
            reward_strategy: Name of reward strategy to use
            reward_kwargs: Kwargs for reward strategy
            use_cache: Whether to cache loaded data

        Returns:
            Configured TradingEnvironment
        """
        config = config or TradingEnvConfig()
        reward_kwargs = reward_kwargs or {}

        # Load data
        df = self._load_dataset(dataset_path, use_cache)
        norm_stats = self._load_norm_stats(norm_stats_path, use_cache)

        # Get reward strategy
        strategy = RewardStrategyRegistry.get(reward_strategy, **reward_kwargs)

        # Create environment
        env = TradingEnvironment(
            df=df,
            norm_stats=norm_stats,
            config=config,
            reward_strategy=strategy,
        )

        logger.info(
            f"Created environment: {len(df)} bars, "
            f"{config.observation_dim} dims, "
            f"reward={reward_strategy}"
        )

        return env

    def create_vec_env(
        self,
        dataset_path: Path,
        norm_stats_path: Path,
        config: Optional[TradingEnvConfig] = None,
        n_envs: int = 1,
        use_subproc: bool = False,
        reward_strategy: str = "default",
        reward_kwargs: Optional[Dict] = None,
    ) -> VecEnv:
        """
        Create vectorized environment for parallel training.

        Args:
            dataset_path: Path to training dataset
            norm_stats_path: Path to normalization stats
            config: Environment configuration
            n_envs: Number of parallel environments
            use_subproc: Use subprocess environments (for CPU parallelism)
            reward_strategy: Reward strategy name
            reward_kwargs: Kwargs for reward strategy

        Returns:
            Vectorized environment (DummyVecEnv or SubprocVecEnv)
        """
        config = config or TradingEnvConfig()
        reward_kwargs = reward_kwargs or {}

        # Pre-load data to share across environments
        df = self._load_dataset(dataset_path, use_cache=True)
        norm_stats = self._load_norm_stats(norm_stats_path, use_cache=True)

        def make_env():
            """Factory function for creating a single env"""
            strategy = RewardStrategyRegistry.get(reward_strategy, **reward_kwargs)
            return TradingEnvironment(
                df=df.copy(),  # Copy for thread safety
                norm_stats=norm_stats,
                config=config,
                reward_strategy=strategy,
            )

        # Create vectorized environment
        env_fns = [make_env for _ in range(n_envs)]

        if use_subproc and n_envs > 1:
            vec_env = SubprocVecEnv(env_fns)
            logger.info(f"Created SubprocVecEnv with {n_envs} environments")
        else:
            vec_env = DummyVecEnv(env_fns)
            logger.info(f"Created DummyVecEnv with {n_envs} environments")

        return vec_env

    def _split_by_dates(
        self,
        df: pd.DataFrame,
        date_ranges: Dict[str, str],
        timestamp_col: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe by date ranges from SSOT.

        This ensures train/test split is based on calendar dates, not row indices.
        Critical for preventing data leakage and ensuring proper OOS evaluation.

        Args:
            df: Full dataframe with timestamp column or DatetimeIndex
            date_ranges: Dict with train_end, test_start from SSOT
            timestamp_col: Name of timestamp column

        Returns:
            (train_df, test_df) - validation is handled internally during training
        """
        # Handle DatetimeIndex (L2 datasets use DatetimeIndex named 'time')
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Using DatetimeIndex for date-based split (index name: {df.index.name})")
            timestamps = df.index
        else:
            # Try to find timestamp column
            if timestamp_col not in df.columns:
                # Check common alternatives
                for alt_col in ['datetime', 'time', 'date']:
                    if alt_col in df.columns:
                        timestamp_col = alt_col
                        break
                else:
                    raise KeyError(f"No timestamp column found. Tried: {timestamp_col}, datetime, time, date. Available: {list(df.columns)}")

            timestamps = pd.to_datetime(df[timestamp_col])

        train_end = pd.to_datetime(date_ranges.get('train_end', '2024-12-31'))
        test_start = pd.to_datetime(date_ranges.get('test_start', '2025-01-01'))

        # Handle timezone-aware timestamps
        if timestamps.tz is not None:
            train_end = train_end.tz_localize('UTC')
            test_start = test_start.tz_localize('UTC')

        train_df = df[timestamps <= train_end].copy()
        test_df = df[timestamps >= test_start].copy()

        logger.info(f"Date-based split: train={len(train_df)} (until {train_end.date()}), test={len(test_df)} (from {test_start.date()})")

        return train_df, test_df

    def create_train_eval_envs(
        self,
        dataset_path: Path,
        norm_stats_path: Path,
        config: Optional[TradingEnvConfig] = None,
        date_ranges: Optional[Dict[str, str]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        n_train_envs: int = 1,
        n_eval_envs: int = 1,
        reward_strategy: str = "default",
        reward_kwargs: Optional[Dict] = None,
    ) -> Dict[str, VecEnv]:
        """
        Create train, validation, and test environments from a single dataset.

        Args:
            dataset_path: Path to full dataset
            norm_stats_path: Path to normalization stats
            config: Environment configuration
            date_ranges: Dict with train_end, test_start for date-based splits (from SSOT)
            train_ratio: Fraction for training (default 70%) - used if date_ranges is None
            val_ratio: Fraction for validation (default 15%)
            n_train_envs: Number of training environments
            n_eval_envs: Number of eval environments
            reward_strategy: Reward strategy name
            reward_kwargs: Kwargs for reward strategy

        Returns:
            Dict with keys: "train", "val", "test"
        """
        config = config or TradingEnvConfig()
        reward_kwargs = reward_kwargs or {}

        # Load full dataset
        df = self._load_dataset(dataset_path, use_cache=True)
        norm_stats = self._load_norm_stats(norm_stats_path, use_cache=True)

        # Split data - prefer date-based split from SSOT
        if date_ranges:
            # Date-based split from SSOT (fixes distribution shift)
            train_full_df, test_df = self._split_by_dates(df, date_ranges)

            # Use last 10% of train for validation (eval callback during training)
            n_train = len(train_full_df)
            val_start = int(n_train * 0.9)
            val_df = train_full_df.iloc[val_start:].reset_index(drop=True)
            train_df = train_full_df.iloc[:val_start].reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            logger.info(f"  -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        else:
            # Legacy ratio-based split (fallback)
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_df = df.iloc[:train_end].reset_index(drop=True)
            val_df = df.iloc[train_end:val_end].reset_index(drop=True)
            test_df = df.iloc[val_end:].reset_index(drop=True)

            logger.info(
                f"Ratio-based split: train={len(train_df)}, "
                f"val={len(val_df)}, test={len(test_df)}"
            )

        # Calculate minimum rows needed for an environment
        min_rows = config.episode_length + getattr(config, 'warmup_bars', 100) + 10

        # Create environments
        def make_env_fn(data: pd.DataFrame):
            def _make():
                strategy = RewardStrategyRegistry.get(reward_strategy, **reward_kwargs)
                return TradingEnvironment(
                    df=data.copy(),
                    norm_stats=norm_stats,
                    config=config,
                    reward_strategy=strategy,
                )
            return _make

        train_env = DummyVecEnv([make_env_fn(train_df) for _ in range(n_train_envs)])
        val_env = DummyVecEnv([make_env_fn(val_df) for _ in range(n_eval_envs)])

        # Test env is optional - may not have enough data for OOS testing
        # (e.g., training dataset doesn't extend into test period)
        test_env = None
        if len(test_df) >= min_rows:
            test_env = DummyVecEnv([make_env_fn(test_df) for _ in range(n_eval_envs)])
        else:
            logger.warning(
                f"Test dataset too small ({len(test_df)} rows, need {min_rows}). "
                f"Test env will be None. Run L4 backtest on separate test dataset."
            )

        return {
            "train": train_env,
            "val": val_env,
            "test": test_env,  # May be None if insufficient test data
            "splits": {
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
            }
        }

    def _load_dataset(self, path: Path, use_cache: bool = True) -> pd.DataFrame:
        """Load dataset with optional caching"""
        path_str = str(path)

        if use_cache and path_str in self._dataset_cache:
            logger.debug(f"Using cached dataset: {path}")
            return self._dataset_cache[path_str]

        full_path = path if path.is_absolute() else self.project_root / path

        if not full_path.exists():
            raise FileNotFoundError(f"Dataset not found: {full_path}")

        logger.info(f"Loading dataset: {full_path}")
        # Support both CSV and Parquet formats
        if str(full_path).endswith('.parquet'):
            df = pd.read_parquet(full_path)
        else:
            df = pd.read_csv(full_path)

        if use_cache:
            self._dataset_cache[path_str] = df

        return df

    def _load_norm_stats(self, path: Path, use_cache: bool = True) -> Dict:
        """Load normalization stats with optional caching"""
        path_str = str(path)

        if use_cache and path_str in self._norm_stats_cache:
            logger.debug(f"Using cached norm_stats: {path}")
            return self._norm_stats_cache[path_str]

        full_path = path if path.is_absolute() else self.project_root / path

        if not full_path.exists():
            raise FileNotFoundError(f"Norm stats not found: {full_path}")

        logger.info(f"Loading norm_stats: {full_path}")
        with open(full_path, 'r') as f:
            stats = json.load(f)

        if use_cache:
            self._norm_stats_cache[path_str] = stats

        return stats

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self._dataset_cache.clear()
        self._norm_stats_cache.clear()
        logger.info("Environment factory cache cleared")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_training_env(
    project_root: Path,
    dataset_path: Path,
    norm_stats_path: Path,
    config: Optional[TradingEnvConfig] = None,
    n_envs: int = 1,
) -> VecEnv:
    """
    Convenience function to create training environment.

    Args:
        project_root: Project root directory
        dataset_path: Path to training data
        norm_stats_path: Path to normalization stats
        config: Environment configuration
        n_envs: Number of parallel environments

    Returns:
        Vectorized training environment
    """
    factory = EnvironmentFactory(project_root)
    return factory.create_vec_env(
        dataset_path=dataset_path,
        norm_stats_path=norm_stats_path,
        config=config,
        n_envs=n_envs,
    )
