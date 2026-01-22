"""
BaseDAGFactory - DRY Factory for Airflow DAGs
==============================================

Centralizes common patterns across all trading DAGs:
- SSOT imports with fallbacks
- Project root path setup
- Default DAG arguments
- Trading calendar integration
- Logging configuration
- Database connection utilities

Usage:
    from common.dag_factory import BaseDAGFactory, SSOTImports

    # Get SSOT imports
    ssot = SSOTImports.load()

    # Create DAG with factory defaults
    dag = BaseDAGFactory.create_dag(
        dag_id='v3.l1_feature_refresh',
        schedule_interval='*/5 13-17 * * 1-5',
        tags=['features', 'trading'],
    )

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from airflow import DAG
from airflow.models import Variable

# =============================================================================
# PATH SETUP
# =============================================================================

# Standard project root for Airflow deployments
AIRFLOW_PROJECT_ROOT = Path('/opt/airflow')
LOCAL_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Use local if not in Airflow environment
PROJECT_ROOT = AIRFLOW_PROJECT_ROOT if AIRFLOW_PROJECT_ROOT.exists() else LOCAL_PROJECT_ROOT

# Add to path once
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


# =============================================================================
# SSOT IMPORTS - Centralized Import Logic
# =============================================================================

@dataclass
class SSOTImports:
    """
    Container for SSOT imports with availability flags.

    Centralizes all SSOT imports in one place to avoid duplicating
    try/except blocks across every DAG.
    """

    # Availability flags
    constants_available: bool = False
    feature_builder_available: bool = False
    training_engine_available: bool = False
    feature_contract_available: bool = False

    # SSOT objects (None if not available)
    FEATURE_ORDER: Optional[Tuple[str, ...]] = None
    OBSERVATION_DIM: int = 15
    CLIP_MIN: float = -5.0
    CLIP_MAX: float = 5.0
    THRESHOLD_LONG: float = 0.33
    THRESHOLD_SHORT: float = -0.33
    RSI_PERIOD: int = 9
    ATR_PERIOD: int = 10
    ADX_PERIOD: int = 14

    # SSOT Classes (None if not available)
    CanonicalFeatureBuilder: Any = None
    TrainingEngine: Any = None
    TrainingRequest: Any = None
    PPO_HYPERPARAMETERS: Any = None

    @classmethod
    def load(cls) -> 'SSOTImports':
        """
        Load all SSOT imports with graceful fallbacks.

        Returns:
            SSOTImports instance with all available imports
        """
        imports = cls()

        # 1. Load constants from SSOT
        try:
            from src.core.constants import (
                FEATURE_ORDER,
                OBSERVATION_DIM,
                CLIP_MIN,
                CLIP_MAX,
                THRESHOLD_LONG,
                THRESHOLD_SHORT,
                RSI_PERIOD,
                ATR_PERIOD,
                ADX_PERIOD,
            )
            imports.FEATURE_ORDER = FEATURE_ORDER
            imports.OBSERVATION_DIM = OBSERVATION_DIM
            imports.CLIP_MIN = CLIP_MIN
            imports.CLIP_MAX = CLIP_MAX
            imports.THRESHOLD_LONG = THRESHOLD_LONG
            imports.THRESHOLD_SHORT = THRESHOLD_SHORT
            imports.RSI_PERIOD = RSI_PERIOD
            imports.ATR_PERIOD = ATR_PERIOD
            imports.ADX_PERIOD = ADX_PERIOD
            imports.constants_available = True
            logger.info("[SSOT] Constants loaded from src.core.constants")
        except ImportError as e:
            logger.warning(f"[SSOT] Constants not available: {e}")

        # 2. Load CanonicalFeatureBuilder
        try:
            from src.feature_store.builders import CanonicalFeatureBuilder
            imports.CanonicalFeatureBuilder = CanonicalFeatureBuilder
            imports.feature_builder_available = True
            logger.info(f"[SSOT] CanonicalFeatureBuilder loaded (v{CanonicalFeatureBuilder.VERSION})")
        except ImportError as e:
            logger.warning(f"[SSOT] CanonicalFeatureBuilder not available: {e}")

        # 3. Load TrainingEngine
        try:
            from src.training.engine import TrainingEngine, TrainingRequest
            from src.training.config import PPO_HYPERPARAMETERS
            imports.TrainingEngine = TrainingEngine
            imports.TrainingRequest = TrainingRequest
            imports.PPO_HYPERPARAMETERS = PPO_HYPERPARAMETERS
            imports.training_engine_available = True
            logger.info("[SSOT] TrainingEngine loaded")
        except ImportError as e:
            logger.warning(f"[SSOT] TrainingEngine not available: {e}")

        # 4. Load Feature Contract (fallback source for FEATURE_ORDER)
        if imports.FEATURE_ORDER is None:
            try:
                from src.core.contracts.feature_contract import FEATURE_ORDER
                imports.FEATURE_ORDER = FEATURE_ORDER
                imports.feature_contract_available = True
                logger.info("[SSOT] FEATURE_ORDER loaded from contracts (fallback)")
            except ImportError:
                pass

        return imports


# =============================================================================
# BASE DAG FACTORY
# =============================================================================

class BaseDAGFactory:
    """
    Factory for creating Airflow DAGs with consistent configuration.

    Provides:
    - Standard default_args for all trading DAGs
    - Trading calendar integration
    - SSOT imports
    - Database connection utilities
    """

    # Standard owner for all DAGs
    DEFAULT_OWNER = 'trading_team'

    # Standard retry configuration
    DEFAULT_RETRIES = 2
    DEFAULT_RETRY_DELAY = timedelta(minutes=5)

    # Standard email configuration
    DEFAULT_EMAIL_ON_FAILURE = True
    DEFAULT_EMAIL_ON_RETRY = False

    @classmethod
    def get_default_args(
        cls,
        owner: str = None,
        retries: int = None,
        retry_delay: timedelta = None,
        start_date: datetime = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get default_args for DAG with sensible defaults.

        Args:
            owner: DAG owner (default: trading_team)
            retries: Number of retries (default: 2)
            retry_delay: Delay between retries (default: 5 minutes)
            start_date: DAG start date (default: yesterday)
            **kwargs: Additional default_args

        Returns:
            Dictionary of default_args
        """
        args = {
            'owner': owner or cls.DEFAULT_OWNER,
            'depends_on_past': False,
            'email_on_failure': cls.DEFAULT_EMAIL_ON_FAILURE,
            'email_on_retry': cls.DEFAULT_EMAIL_ON_RETRY,
            'retries': retries if retries is not None else cls.DEFAULT_RETRIES,
            'retry_delay': retry_delay or cls.DEFAULT_RETRY_DELAY,
            'start_date': start_date or datetime(2024, 1, 1),
        }
        args.update(kwargs)
        return args

    @classmethod
    def create_dag(
        cls,
        dag_id: str,
        schedule_interval: Optional[str] = None,
        description: str = None,
        tags: List[str] = None,
        catchup: bool = False,
        max_active_runs: int = 1,
        default_args: Dict[str, Any] = None,
        **kwargs
    ) -> DAG:
        """
        Create a DAG with standard configuration.

        Args:
            dag_id: Unique DAG identifier
            schedule_interval: Cron expression or preset
            description: DAG description
            tags: List of tags for filtering
            catchup: Whether to backfill (default: False)
            max_active_runs: Max concurrent runs (default: 1)
            default_args: Override default arguments
            **kwargs: Additional DAG arguments

        Returns:
            Configured DAG instance
        """
        final_args = cls.get_default_args()
        if default_args:
            final_args.update(default_args)

        base_tags = ['trading', 'usdcop']
        if tags:
            base_tags.extend(tags)

        return DAG(
            dag_id=dag_id,
            default_args=final_args,
            schedule_interval=schedule_interval,
            description=description or f"USD/COP Trading DAG: {dag_id}",
            tags=base_tags,
            catchup=catchup,
            max_active_runs=max_active_runs,
            **kwargs
        )

    @classmethod
    def get_config_from_variable(
        cls,
        variable_name: str,
        default: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Get configuration from Airflow Variable with fallback.

        Args:
            variable_name: Name of the Airflow Variable
            default: Default configuration if Variable not found

        Returns:
            Configuration dictionary
        """
        import json

        config = default.copy() if default else {}

        try:
            var_value = Variable.get(variable_name, default_var=None)
            if var_value:
                config.update(json.loads(var_value))
        except Exception as e:
            logger.warning(f"Could not load Variable '{variable_name}': {e}")

        return config


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_ssot() -> SSOTImports:
    """Convenience function to get SSOT imports."""
    return SSOTImports.load()


def create_trading_dag(
    dag_id: str,
    schedule: str = None,
    tags: List[str] = None,
    **kwargs
) -> DAG:
    """
    Convenience function to create a trading DAG.

    Args:
        dag_id: DAG identifier
        schedule: Cron schedule
        tags: Additional tags
        **kwargs: Additional DAG arguments

    Returns:
        Configured DAG
    """
    return BaseDAGFactory.create_dag(
        dag_id=dag_id,
        schedule_interval=schedule,
        tags=tags,
        **kwargs
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BaseDAGFactory',
    'SSOTImports',
    'PROJECT_ROOT',
    'get_ssot',
    'create_trading_dag',
]
