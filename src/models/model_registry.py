"""
Model Registry for managing multiple RL models.

This module provides a centralized registry for tracking, loading, and managing
multiple reinforcement learning models for the USD/COP trading system.
"""

import json
import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a registered model."""
    model_id: str
    model_name: str
    algorithm: str
    version: str
    status: str  # 'production', 'testing', 'deprecated'
    hyperparameters: Dict[str, Any]
    policy_config: Dict[str, Any]
    environment_config: Dict[str, Any]
    model_path: str
    feature_dim: int
    created_at: datetime
    updated_at: datetime
    validation_metrics: Optional[Dict[str, Any]] = None
    risk_limits: Optional[Dict[str, Any]] = None


class ModelStaleError(Exception):
    """Exception raised when a model exceeds its maximum allowed age."""

    def __init__(self, model_id: str, age_days: int, max_age_days: int):
        self.model_id = model_id
        self.age_days = age_days
        self.max_age_days = max_age_days
        super().__init__(
            f"Model {model_id} is {age_days} days old, maximum allowed is {max_age_days} days. "
            "Please retrain or update the model."
        )


class ModelRegistry:
    """
    Registry for managing multiple RL models.

    Provides functionality for:
    - Registering new models with their configurations
    - Loading models from storage with caching
    - Tracking model status (production/testing/deprecated)
    - Retrieving live performance metrics
    - TTL validation to prevent stale models in production (FASE 9)

    Attributes:
        db_connection: PostgreSQL database connection or connection string
        model_storage_path: Base path for model file storage
        max_model_age_days: Maximum allowed model age in days (default: 30)

    Contract: CTR-DEPLOY-001 (Model TTL Validation)
    """

    VALID_STATUSES = ['production', 'testing', 'deprecated', 'training']

    # Default maximum model age in days
    DEFAULT_MAX_MODEL_AGE_DAYS = 30

    def __init__(
        self,
        db_connection,
        model_storage_path: str,
        max_model_age_days: int = None,
        enforce_ttl: bool = True,
    ):
        """
        Initialize the ModelRegistry.

        Args:
            db_connection: Either a psycopg2 connection object or a connection string
            model_storage_path: Base path where model files are stored
            max_model_age_days: Maximum allowed model age in days (default: 30)
            enforce_ttl: Whether to enforce TTL validation (default: True)

        Contract: CTR-DEPLOY-001 (Model TTL Validation)
        """
        if isinstance(db_connection, str):
            self._conn_string = db_connection
            self._conn = None
        else:
            self._conn_string = None
            self._conn = db_connection

        self.model_storage_path = model_storage_path
        self._model_cache: Dict[str, Any] = {}
        self._config_cache: Dict[str, ModelConfig] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: Dict[str, datetime] = {}

        # TTL Configuration (FASE 9: Deployment)
        self.max_model_age_days = max_model_age_days or self.DEFAULT_MAX_MODEL_AGE_DAYS
        self.enforce_ttl = enforce_ttl

        # Lazy import to avoid circular dependencies
        self._model_loader = None

    def _get_connection(self):
        """Get database connection, creating if necessary."""
        if self._conn is None and self._conn_string:
            self._conn = psycopg2.connect(self._conn_string)
        return self._conn

    def _get_model_loader(self):
        """Lazy load the ModelLoader to avoid circular imports."""
        if self._model_loader is None:
            from .model_loader import ModelLoader
            self._model_loader = ModelLoader()
        return self._model_loader

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached item is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        elapsed = (datetime.now() - self._cache_timestamps[cache_key]).total_seconds()
        return elapsed < self._cache_ttl

    def validate_model_ttl(self, config: ModelConfig, raise_on_stale: bool = True) -> bool:
        """
        Validate that a model has not exceeded its maximum allowed age.

        This is part of FASE 9 (Deployment) - Model TTL Validation.
        Prevents stale models from being used in production to ensure
        models are regularly retrained with fresh data.

        Args:
            config: Model configuration containing created_at timestamp
            raise_on_stale: If True, raise ModelStaleError on stale model

        Returns:
            True if model is within TTL, False if stale

        Raises:
            ModelStaleError: If raise_on_stale=True and model exceeds max age

        Contract: CTR-DEPLOY-001
        """
        if not self.enforce_ttl:
            return True

        model_age = datetime.now() - config.created_at
        age_days = model_age.days

        if age_days > self.max_model_age_days:
            logger.warning(
                f"Model {config.model_id} is {age_days} days old "
                f"(max: {self.max_model_age_days} days)"
            )

            if raise_on_stale:
                raise ModelStaleError(
                    model_id=config.model_id,
                    age_days=age_days,
                    max_age_days=self.max_model_age_days
                )

            return False

        # Log warning if model is approaching TTL (>80% of max age)
        warning_threshold = self.max_model_age_days * 0.8
        if age_days > warning_threshold:
            logger.info(
                f"Model {config.model_id} is {age_days} days old, "
                f"approaching TTL limit of {self.max_model_age_days} days"
            )

        return True

    def get_model_age_days(self, model_id: str) -> Optional[int]:
        """
        Get the age of a model in days.

        Args:
            model_id: Unique identifier for the model

        Returns:
            Age in days, or None if model not found
        """
        config = self.get_model_config(model_id)
        if config is None:
            return None

        age = datetime.now() - config.created_at
        return age.days

    def get_expiring_models(self, days_until_expiry: int = 7) -> List[ModelConfig]:
        """
        Get models that will expire within the specified number of days.

        Useful for proactive model retraining alerts.

        Args:
            days_until_expiry: Number of days to look ahead

        Returns:
            List of models expiring soon
        """
        expiry_threshold = self.max_model_age_days - days_until_expiry
        expiring = []

        for config in self.get_enabled_models():
            age_days = (datetime.now() - config.created_at).days
            if age_days >= expiry_threshold:
                expiring.append(config)

        return expiring

    def get_enabled_models(self) -> List[ModelConfig]:
        """
        Get all models with status != 'deprecated'.

        Returns:
            List of ModelConfig objects for all enabled models
        """
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        model_id,
                        model_name,
                        algorithm,
                        version,
                        status,
                        hyperparameters,
                        policy_config,
                        environment_config,
                        model_path,
                        feature_dim,
                        validation_metrics,
                        risk_limits,
                        created_at,
                        updated_at
                    FROM models.model_registry
                    WHERE status != 'deprecated'
                    ORDER BY
                        CASE status
                            WHEN 'production' THEN 1
                            WHEN 'testing' THEN 2
                            ELSE 3
                        END,
                        created_at DESC
                """)
                rows = cur.fetchall()

                models = []
                for row in rows:
                    config = ModelConfig(
                        model_id=row['model_id'],
                        model_name=row['model_name'],
                        algorithm=row['algorithm'],
                        version=row['version'],
                        status=row['status'],
                        hyperparameters=row['hyperparameters'] or {},
                        policy_config=row['policy_config'] or {},
                        environment_config=row['environment_config'] or {},
                        model_path=row['model_path'],
                        feature_dim=row['feature_dim'],
                        validation_metrics=row['validation_metrics'],
                        risk_limits=row['risk_limits'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at']
                    )
                    models.append(config)
                    # Cache the config
                    self._config_cache[row['model_id']] = config
                    self._cache_timestamps[f"config_{row['model_id']}"] = datetime.now()

                return models

        except Exception as e:
            logger.error(f"Error fetching enabled models: {e}")
            raise

    def get_production_model(self) -> Optional[ModelConfig]:
        """
        Get the model with status == 'production'.

        Returns:
            ModelConfig for the production model, or None if no production model exists

        Raises:
            ValueError: If multiple production models are found
        """
        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        model_id,
                        model_name,
                        algorithm,
                        version,
                        status,
                        hyperparameters,
                        policy_config,
                        environment_config,
                        model_path,
                        feature_dim,
                        validation_metrics,
                        risk_limits,
                        created_at,
                        updated_at
                    FROM models.model_registry
                    WHERE status = 'production'
                """)
                rows = cur.fetchall()

                if len(rows) == 0:
                    logger.warning("No production model found in registry")
                    return None

                if len(rows) > 1:
                    model_ids = [r['model_id'] for r in rows]
                    raise ValueError(
                        f"Multiple production models found: {model_ids}. "
                        "Only one model can have 'production' status."
                    )

                row = rows[0]
                config = ModelConfig(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    algorithm=row['algorithm'],
                    version=row['version'],
                    status=row['status'],
                    hyperparameters=row['hyperparameters'] or {},
                    policy_config=row['policy_config'] or {},
                    environment_config=row['environment_config'] or {},
                    model_path=row['model_path'],
                    feature_dim=row['feature_dim'],
                    validation_metrics=row['validation_metrics'],
                    risk_limits=row['risk_limits'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                # Cache the config
                self._config_cache[row['model_id']] = config
                self._cache_timestamps[f"config_{row['model_id']}"] = datetime.now()

                return config

        except Exception as e:
            logger.error(f"Error fetching production model: {e}")
            raise

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.

        Args:
            model_id: Unique identifier for the model

        Returns:
            ModelConfig if found, None otherwise
        """
        # Check cache first
        cache_key = f"config_{model_id}"
        if model_id in self._config_cache and self._is_cache_valid(cache_key):
            return self._config_cache[model_id]

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        model_id,
                        model_name,
                        algorithm,
                        version,
                        status,
                        hyperparameters,
                        policy_config,
                        environment_config,
                        model_path,
                        feature_dim,
                        validation_metrics,
                        risk_limits,
                        created_at,
                        updated_at
                    FROM models.model_registry
                    WHERE model_id = %s
                """, (model_id,))
                row = cur.fetchone()

                if row is None:
                    return None

                config = ModelConfig(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    algorithm=row['algorithm'],
                    version=row['version'],
                    status=row['status'],
                    hyperparameters=row['hyperparameters'] or {},
                    policy_config=row['policy_config'] or {},
                    environment_config=row['environment_config'] or {},
                    model_path=row['model_path'],
                    feature_dim=row['feature_dim'],
                    validation_metrics=row['validation_metrics'],
                    risk_limits=row['risk_limits'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                self._config_cache[model_id] = config
                self._cache_timestamps[cache_key] = datetime.now()

                return config

        except Exception as e:
            logger.error(f"Error fetching model config for {model_id}: {e}")
            raise

    def load_model(self, model_id: str, skip_ttl_check: bool = False):
        """
        Load a model from storage, with caching and TTL validation.

        Args:
            model_id: Unique identifier for the model
            skip_ttl_check: If True, skip TTL validation (use with caution)

        Returns:
            Loaded SB3 BaseAlgorithm model

        Raises:
            ValueError: If model is not found in registry
            FileNotFoundError: If model file doesn't exist
            ModelStaleError: If model exceeds maximum allowed age

        Contract: CTR-DEPLOY-001 (includes TTL validation)
        """
        # Check model cache first
        cache_key = f"model_{model_id}"
        if model_id in self._model_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached model for {model_id}")
            return self._model_cache[model_id]

        # Get model config
        config = self.get_model_config(model_id)
        if config is None:
            raise ValueError(f"Model {model_id} not found in registry")

        # TTL Validation (FASE 9: Deployment)
        if not skip_ttl_check:
            self.validate_model_ttl(config, raise_on_stale=True)

        # Resolve model path
        model_path = config.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(self.model_storage_path, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model using ModelLoader
        loader = self._get_model_loader()
        model = loader.load_from_file(model_path, config.algorithm)

        # Validate model dimensions
        if not loader.validate_model(model, config.feature_dim):
            logger.warning(
                f"Model {model_id} feature dimension mismatch. "
                f"Expected: {config.feature_dim}"
            )

        # Cache the loaded model
        self._model_cache[model_id] = model
        self._cache_timestamps[cache_key] = datetime.now()

        logger.info(f"Loaded model {model_id} ({config.algorithm} {config.version})")
        return model

    def register_model(
        self,
        model_id: str,
        config: dict,
        model_path: str,
        status: str = 'testing'
    ) -> ModelConfig:
        """
        Register a new model in the database.

        Args:
            model_id: Unique identifier for the model
            config: Dictionary containing model configuration
            model_path: Path to the model file
            status: Initial status (default: 'testing')

        Returns:
            ModelConfig for the newly registered model

        Raises:
            ValueError: If model_id already exists or status is invalid
        """
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status}. Must be one of {self.VALID_STATUSES}"
            )

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if model already exists
                cur.execute(
                    "SELECT model_id FROM models.model_registry WHERE model_id = %s",
                    (model_id,)
                )
                if cur.fetchone():
                    raise ValueError(f"Model {model_id} already exists in registry")

                # If registering as production, demote existing production model
                if status == 'production':
                    cur.execute("""
                        UPDATE models.model_registry
                        SET status = 'testing', updated_at = NOW()
                        WHERE status = 'production'
                    """)
                    if cur.rowcount > 0:
                        logger.info("Demoted existing production model to testing")

                # Insert new model
                cur.execute("""
                    INSERT INTO models.model_registry (
                        model_id,
                        model_name,
                        algorithm,
                        version,
                        status,
                        hyperparameters,
                        policy_config,
                        environment_config,
                        model_path,
                        feature_dim,
                        validation_metrics,
                        risk_limits,
                        created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                    RETURNING *
                """, (
                    model_id,
                    config.get('model_name', model_id),
                    config.get('algorithm', 'PPO'),
                    config.get('version', 'V1'),
                    status,
                    json.dumps(config.get('hyperparameters', {})),
                    json.dumps(config.get('policy', {})),
                    json.dumps(config.get('environment', {})),
                    model_path,
                    config.get('data', {}).get('features', 21),
                    json.dumps(config.get('validation_results', {})),
                    json.dumps(config.get('risk_limits', {}))
                ))

                row = cur.fetchone()
                conn.commit()

                model_config = ModelConfig(
                    model_id=row['model_id'],
                    model_name=row['model_name'],
                    algorithm=row['algorithm'],
                    version=row['version'],
                    status=row['status'],
                    hyperparameters=row['hyperparameters'] or {},
                    policy_config=row['policy_config'] or {},
                    environment_config=row['environment_config'] or {},
                    model_path=row['model_path'],
                    feature_dim=row['feature_dim'],
                    validation_metrics=row['validation_metrics'],
                    risk_limits=row['risk_limits'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )

                logger.info(f"Registered new model: {model_id} with status: {status}")
                return model_config

        except Exception as e:
            conn.rollback()
            logger.error(f"Error registering model {model_id}: {e}")
            raise

    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        Change model status (production/testing/deprecated).

        Args:
            model_id: Unique identifier for the model
            status: New status to set

        Returns:
            True if update was successful

        Raises:
            ValueError: If status is invalid or model not found
        """
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status}. Must be one of {self.VALID_STATUSES}"
            )

        conn = self._get_connection()

        try:
            with conn.cursor() as cur:
                # If promoting to production, demote existing production model
                if status == 'production':
                    cur.execute("""
                        UPDATE models.model_registry
                        SET status = 'testing', updated_at = NOW()
                        WHERE status = 'production' AND model_id != %s
                    """, (model_id,))
                    if cur.rowcount > 0:
                        logger.info("Demoted existing production model to testing")

                # Update the target model
                cur.execute("""
                    UPDATE models.model_registry
                    SET status = %s, updated_at = NOW()
                    WHERE model_id = %s
                """, (status, model_id))

                if cur.rowcount == 0:
                    raise ValueError(f"Model {model_id} not found in registry")

                conn.commit()

                # Invalidate cache
                if model_id in self._config_cache:
                    del self._config_cache[model_id]

                logger.info(f"Updated model {model_id} status to: {status}")
                return True

        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating model status for {model_id}: {e}")
            raise

    def get_model_metrics(self, model_id: str, period: str = '7d') -> Dict[str, Any]:
        """
        Get live performance metrics for a model.

        Args:
            model_id: Unique identifier for the model
            period: Time period for metrics ('1d', '7d', '30d', 'all')

        Returns:
            Dictionary containing performance metrics
        """
        # Parse period
        period_map = {
            '1d': timedelta(days=1),
            '7d': timedelta(days=7),
            '30d': timedelta(days=30),
            '90d': timedelta(days=90),
            'all': None
        }

        if period not in period_map:
            raise ValueError(f"Invalid period: {period}. Must be one of {list(period_map.keys())}")

        conn = self._get_connection()

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build date filter
                if period_map[period]:
                    date_filter = "AND timestamp >= NOW() - %s"
                    date_param = (period_map[period],)
                else:
                    date_filter = ""
                    date_param = ()

                # Get trade metrics
                cur.execute(f"""
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        AVG(pnl) as avg_pnl,
                        SUM(pnl) as total_pnl,
                        MAX(pnl) as max_profit,
                        MIN(pnl) as max_loss,
                        STDDEV(pnl) as pnl_stddev
                    FROM models.model_trades
                    WHERE model_id = %s {date_filter}
                """, (model_id,) + date_param)

                trade_metrics = cur.fetchone() or {}

                # Get signal distribution
                cur.execute(f"""
                    SELECT
                        action,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence
                    FROM models.model_signals
                    WHERE model_id = %s {date_filter}
                    GROUP BY action
                """, (model_id,) + date_param)

                signal_dist = {row['action']: {
                    'count': row['count'],
                    'avg_confidence': float(row['avg_confidence']) if row['avg_confidence'] else 0
                } for row in cur.fetchall()}

                # Calculate derived metrics
                total_trades = trade_metrics.get('total_trades', 0) or 0
                winning_trades = trade_metrics.get('winning_trades', 0) or 0

                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

                avg_pnl = float(trade_metrics.get('avg_pnl', 0) or 0)
                pnl_stddev = float(trade_metrics.get('pnl_stddev', 0) or 0)
                sharpe_approx = (avg_pnl / pnl_stddev) if pnl_stddev > 0 else 0

                return {
                    'model_id': model_id,
                    'period': period,
                    'trade_metrics': {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': trade_metrics.get('losing_trades', 0) or 0,
                        'win_rate': round(win_rate, 2),
                        'avg_pnl': round(avg_pnl, 4),
                        'total_pnl': float(trade_metrics.get('total_pnl', 0) or 0),
                        'max_profit': float(trade_metrics.get('max_profit', 0) or 0),
                        'max_loss': float(trade_metrics.get('max_loss', 0) or 0),
                        'sharpe_approx': round(sharpe_approx, 4)
                    },
                    'signal_distribution': signal_dist,
                    'generated_at': datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"Error fetching metrics for model {model_id}: {e}")
            raise

    def clear_cache(self):
        """Clear all cached models and configs."""
        self._model_cache.clear()
        self._config_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Model registry cache cleared")

    def close(self):
        """Close database connection."""
        if self._conn and self._conn_string:
            self._conn.close()
            self._conn = None
