"""
Promotion Service
=================

Service for promoting models from experiment zone to production.

This module provides:
- PromotionService: Handles model promotion workflow
- Copies artifacts from s3://experiments/ to s3://production/
- Registers promoted models in PostgreSQL model_registry

Contract: CTR-PROMOTION-001
- Only models passing validation gate can be promoted
- All promotions are logged for audit
- Atomic promotion (rollback on failure)

Author: Trading Team
Version: 1.0.0
Created: 2026-01-18
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.contracts.storage_contracts import ModelSnapshot, BacktestSnapshot
from src.core.factories.storage_factory import StorageFactory
from src.core.interfaces.storage import ObjectNotFoundError
from src.ml_workflow.experiment_manager import ExperimentManager

logger = logging.getLogger(__name__)


# =============================================================================
# PROMOTION STATUS
# =============================================================================


class PromotionStatus(str, Enum):
    """Status of a promotion request."""
    PENDING = "pending"
    VALIDATING = "validating"
    COPYING = "copying"
    REGISTERING = "registering"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class PromotionResult:
    """Result of a promotion attempt."""
    status: PromotionStatus
    model_id: Optional[str]
    experiment_id: str
    model_version: str
    production_uri: Optional[str]
    validation_passed: bool
    validation_errors: List[str]
    promoted_at: Optional[datetime]
    error_message: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "model_version": self.model_version,
            "production_uri": self.production_uri,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "error_message": self.error_message,
        }


# =============================================================================
# PROMOTION SERVICE
# =============================================================================


class PromotionService:
    """
    Service for promoting models to production.

    Workflow:
    1. Validate model meets promotion criteria
    2. Copy artifacts to production bucket
    3. Register in PostgreSQL model_registry
    4. Update status and notify

    Example:
        >>> service = PromotionService()
        >>> result = service.promote(
        ...     experiment_id="baseline_v2",
        ...     model_version="20260118_123456",
        ...     validation_config={"min_sharpe": 0.5},
        ... )
        >>> if result.status == PromotionStatus.COMPLETED:
        ...     print(f"Model promoted: {result.model_id}")
    """

    def __init__(
        self,
        storage_factory: Optional[StorageFactory] = None,
        db_connection=None,
    ):
        """
        Initialize promotion service.

        Args:
            storage_factory: Optional custom storage factory
            db_connection: PostgreSQL connection for model_registry
        """
        self._factory = storage_factory or StorageFactory.get_instance()
        self._db_conn = db_connection

    def promote(
        self,
        experiment_id: str,
        model_version: str,
        model_id: Optional[str] = None,
        validation_config: Optional[Dict[str, Any]] = None,
        skip_validation: bool = False,
        dry_run: bool = False,
    ) -> PromotionResult:
        """
        Promote model to production.

        Args:
            experiment_id: Source experiment ID
            model_version: Model version to promote
            model_id: Custom model ID (auto-generated if None)
            validation_config: Validation criteria
            skip_validation: Skip validation checks (use with caution)
            dry_run: Validate only, don't actually promote

        Returns:
            PromotionResult with status and details
        """
        logger.info(f"Starting promotion: {experiment_id}/{model_version}")

        # Initialize result
        result = PromotionResult(
            status=PromotionStatus.PENDING,
            model_id=None,
            experiment_id=experiment_id,
            model_version=model_version,
            production_uri=None,
            validation_passed=False,
            validation_errors=[],
            promoted_at=None,
            error_message=None,
        )

        try:
            # Get experiment manager
            manager = ExperimentManager(experiment_id, self._factory)

            # Get model snapshot
            result.status = PromotionStatus.VALIDATING
            model_snapshot = manager.get_model_snapshot(model_version)

            # Validate if not skipped
            if not skip_validation:
                validation_errors = self._validate_model(
                    manager, model_snapshot, validation_config or {}
                )
                result.validation_errors = validation_errors

                if validation_errors:
                    result.status = PromotionStatus.FAILED
                    result.error_message = f"Validation failed: {'; '.join(validation_errors)}"
                    logger.warning(f"Promotion validation failed: {result.error_message}")
                    return result

            result.validation_passed = True

            # Dry run stops here
            if dry_run:
                result.status = PromotionStatus.COMPLETED
                logger.info(f"Dry run completed: {experiment_id}/{model_version} passed validation")
                return result

            # Copy to production bucket
            result.status = PromotionStatus.COPYING
            promoted_model_id = manager.promote_model(model_version, model_id)
            result.model_id = promoted_model_id
            result.production_uri = f"s3://production/models/{promoted_model_id}/"

            # Register in PostgreSQL
            result.status = PromotionStatus.REGISTERING
            self._register_in_db(promoted_model_id, model_snapshot)

            # Success
            result.status = PromotionStatus.COMPLETED
            result.promoted_at = datetime.utcnow()

            logger.info(
                f"Promotion completed: {experiment_id}/{model_version} -> {promoted_model_id}"
            )

        except ObjectNotFoundError as e:
            result.status = PromotionStatus.FAILED
            result.error_message = f"Model not found: {e}"
            logger.error(f"Promotion failed: {result.error_message}")

        except Exception as e:
            result.status = PromotionStatus.FAILED
            result.error_message = str(e)
            logger.exception(f"Promotion failed with error: {e}")

            # Attempt rollback if we got far enough
            if result.model_id:
                self._rollback(result.model_id)
                result.status = PromotionStatus.ROLLED_BACK

        return result

    def _validate_model(
        self,
        manager: ExperimentManager,
        model_snapshot: ModelSnapshot,
        config: Dict[str, Any],
    ) -> List[str]:
        """
        Validate model meets promotion criteria.

        Args:
            manager: Experiment manager
            model_snapshot: Model to validate
            config: Validation criteria

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check model hash exists
        if not model_snapshot.model_hash:
            errors.append("Model hash is missing")

        # Check norm_stats hash exists
        if not model_snapshot.norm_stats_hash:
            errors.append("Norm stats hash is missing")

        # Check feature order
        if not model_snapshot.feature_order:
            errors.append("Feature order is missing")

        # Check observation dimension
        expected_dim = config.get("expected_observation_dim", 15)
        if model_snapshot.observation_dim != expected_dim:
            errors.append(
                f"Observation dim mismatch: expected {expected_dim}, "
                f"got {model_snapshot.observation_dim}"
            )

        # Check backtest performance if available
        backtests = manager.list_backtests()
        model_backtests = [b for b in backtests if b.model_version == model_snapshot.version]

        if model_backtests:
            latest_backtest = model_backtests[0]

            # Check Sharpe ratio
            min_sharpe = config.get("min_sharpe", 0.0)
            if latest_backtest.sharpe_ratio < min_sharpe:
                errors.append(
                    f"Sharpe ratio too low: {latest_backtest.sharpe_ratio:.2f} < {min_sharpe}"
                )

            # Check max drawdown
            max_drawdown_limit = config.get("max_drawdown_limit", 0.5)
            if latest_backtest.max_drawdown > max_drawdown_limit:
                errors.append(
                    f"Max drawdown too high: {latest_backtest.max_drawdown:.2%} > {max_drawdown_limit:.2%}"
                )

            # Check minimum trades
            min_trades = config.get("min_trades", 10)
            if latest_backtest.total_trades < min_trades:
                errors.append(
                    f"Too few trades: {latest_backtest.total_trades} < {min_trades}"
                )

        elif config.get("require_backtest", False):
            errors.append("No backtest found for model")

        return errors

    def _register_in_db(self, model_id: str, snapshot: ModelSnapshot) -> None:
        """
        Register promoted model in PostgreSQL.

        Args:
            model_id: Production model ID
            snapshot: Model snapshot
        """
        if self._db_conn is None:
            logger.warning("No DB connection, skipping model_registry update")
            return

        try:
            self._db_conn.execute("""
                INSERT INTO model_registry
                (model_id, model_version, model_path, model_hash, norm_stats_hash,
                 config_hash, observation_dim, action_space, feature_order,
                 test_sharpe, test_max_drawdown, test_win_rate, status, deployed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id) DO UPDATE SET
                    status = 'deployed',
                    deployed_at = CURRENT_TIMESTAMP
            """, (
                model_id,
                snapshot.version,
                snapshot.storage_uri,
                snapshot.model_hash,
                snapshot.norm_stats_hash,
                snapshot.config_hash,
                snapshot.observation_dim,
                snapshot.action_space,
                json.dumps(list(snapshot.feature_order)),
                snapshot.test_sharpe,
                snapshot.test_max_drawdown,
                snapshot.test_win_rate,
                'deployed',
                datetime.utcnow(),
            ))

            logger.info(f"Registered model {model_id} in model_registry")

        except Exception as e:
            logger.error(f"Failed to register model in DB: {e}")
            raise

    def _rollback(self, model_id: str) -> None:
        """
        Rollback a failed promotion.

        Args:
            model_id: Model ID to rollback
        """
        logger.warning(f"Rolling back promotion for {model_id}")

        try:
            # Delete from production bucket
            storage = self._factory.create_object_storage()
            prefix = f"models/{model_id}/"
            objects = storage.list_objects("production", prefix)

            for obj in objects:
                storage.delete_object("production", obj.artifact_id)

            # Update DB status if we have connection
            if self._db_conn:
                self._db_conn.execute(
                    "DELETE FROM model_registry WHERE model_id = %s",
                    (model_id,)
                )

            logger.info(f"Rollback completed for {model_id}")

        except Exception as e:
            logger.error(f"Rollback failed for {model_id}: {e}")

    def get_promotion_history(
        self,
        experiment_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get promotion history from model_registry.

        Args:
            experiment_id: Filter by experiment (optional)
            limit: Maximum records to return

        Returns:
            List of promotion records
        """
        if self._db_conn is None:
            return []

        query = """
            SELECT model_id, model_version, model_hash, status,
                   deployed_at, retired_at, test_sharpe
            FROM model_registry
            WHERE status IN ('deployed', 'retired')
        """
        params = []

        if experiment_id:
            query += " AND model_version LIKE %s"
            params.append(f"%{experiment_id}%")

        query += " ORDER BY deployed_at DESC LIMIT %s"
        params.append(limit)

        rows = self._db_conn.fetchall(query, params)
        return [dict(row) for row in rows]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PromotionStatus",
    "PromotionResult",
    "PromotionService",
]
