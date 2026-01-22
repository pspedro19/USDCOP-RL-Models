"""
Drift Persistence Service
=========================

Service for persisting drift detection results to PostgreSQL.
Uses the drift_checks and drift_alerts tables from migration 021.

Usage:
    persistence = DriftPersistenceService(db_pool)
    await persistence.save_drift_check(drift_report)

Author: Trading Team
Date: 2026-01-17
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncpg

logger = logging.getLogger(__name__)


class DriftPersistenceService:
    """
    Service for persisting drift detection results to database.

    Features:
    - Save drift check results
    - Auto-create alerts for medium/high severity
    - Query drift history
    - Manage alert lifecycle
    """

    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        """
        Initialize the persistence service.

        Args:
            db_pool: AsyncPG connection pool
        """
        self.db_pool = db_pool
        self._enabled = db_pool is not None

        if self._enabled:
            logger.info("DriftPersistenceService initialized with database connection")
        else:
            logger.warning("DriftPersistenceService initialized without database - persistence disabled")

    async def save_drift_check(
        self,
        check_type: str,
        features_checked: int,
        features_drifted: int,
        drift_score: float,
        max_severity: str,
        univariate_results: Optional[List[Dict]] = None,
        multivariate_results: Optional[Dict] = None,
        model_id: Optional[str] = None,
        triggered_by: str = "scheduled",
    ) -> Optional[Dict]:
        """
        Save a drift check result to the database.

        Automatically creates an alert if severity is medium or high.

        Args:
            check_type: 'univariate' or 'multivariate'
            features_checked: Number of features checked
            features_drifted: Number of features with detected drift
            drift_score: Overall drift score (0-1)
            max_severity: Maximum severity ('none', 'low', 'medium', 'high')
            univariate_results: List of per-feature results
            multivariate_results: Dict of multivariate method results
            model_id: Optional model identifier
            triggered_by: What triggered the check ('scheduled', 'manual', 'inference')

        Returns:
            Dict with check_id and alert_id (if created), or None if persistence disabled
        """
        if not self._enabled:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                # Use the stored function for atomic insert + alert creation
                result = await conn.fetchrow(
                    """
                    SELECT * FROM fn_insert_drift_check_with_alert(
                        $1, $2, $3, $4, $5, $6, $7, $8, $9
                    )
                    """,
                    check_type,
                    features_checked,
                    features_drifted,
                    drift_score,
                    max_severity,
                    json.dumps(univariate_results) if univariate_results else None,
                    json.dumps(multivariate_results) if multivariate_results else None,
                    model_id,
                    triggered_by,
                )

                return {
                    "check_id": result["check_id"],
                    "alert_id": result["alert_id"],
                    "alert_created": result["alert_id"] is not None,
                }

        except asyncpg.UndefinedFunctionError:
            # Function doesn't exist - use direct insert
            logger.warning("fn_insert_drift_check_with_alert not found, using direct insert")
            return await self._save_drift_check_direct(
                check_type, features_checked, features_drifted,
                drift_score, max_severity, univariate_results,
                multivariate_results, model_id, triggered_by
            )

        except Exception as e:
            logger.error(f"Failed to save drift check: {e}")
            return None

    async def _save_drift_check_direct(
        self,
        check_type: str,
        features_checked: int,
        features_drifted: int,
        drift_score: float,
        max_severity: str,
        univariate_results: Optional[List[Dict]],
        multivariate_results: Optional[Dict],
        model_id: Optional[str],
        triggered_by: str,
    ) -> Optional[Dict]:
        """Direct insert without stored function (fallback)."""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Insert drift check
                    check_id = await conn.fetchval(
                        """
                        INSERT INTO drift_checks (
                            check_type, features_checked, features_drifted,
                            overall_drift_score, max_severity, alert_triggered,
                            univariate_results, multivariate_results,
                            model_id, triggered_by
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        RETURNING id
                        """,
                        check_type,
                        features_checked,
                        features_drifted,
                        drift_score,
                        max_severity,
                        max_severity in ('medium', 'high'),
                        json.dumps(univariate_results) if univariate_results else None,
                        json.dumps(multivariate_results) if multivariate_results else None,
                        model_id,
                        triggered_by,
                    )

                    # Create alert if needed
                    alert_id = None
                    if max_severity in ('medium', 'high'):
                        alert_id = await conn.fetchval(
                            """
                            INSERT INTO drift_alerts (
                                severity, alert_type, message,
                                drift_check_id, drifted_features
                            ) VALUES ($1, $2, $3, $4, $5)
                            RETURNING id
                            """,
                            max_severity,
                            f"{check_type}_drift",
                            f"Drift detected: {features_drifted}/{features_checked} features drifted (score: {drift_score:.4f})",
                            check_id,
                            json.dumps([r["feature_name"] for r in (univariate_results or []) if r.get("is_drifted")]),
                        )

                    return {
                        "check_id": check_id,
                        "alert_id": alert_id,
                        "alert_created": alert_id is not None,
                    }

        except Exception as e:
            logger.error(f"Direct insert failed: {e}")
            return None

    async def get_drift_history(
        self,
        hours: int = 24,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Get drift check history.

        Args:
            hours: Number of hours of history to retrieve
            model_id: Optional model filter
            limit: Maximum number of records

        Returns:
            List of drift check records
        """
        if not self._enabled:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        check_timestamp,
                        check_type,
                        features_checked,
                        features_drifted,
                        overall_drift_score,
                        max_severity,
                        alert_triggered,
                        model_id
                    FROM drift_checks
                    WHERE check_timestamp >= NOW() - ($1 || ' hours')::INTERVAL
                      AND ($2::VARCHAR IS NULL OR model_id = $2)
                    ORDER BY check_timestamp DESC
                    LIMIT $3
                    """,
                    str(hours),
                    model_id,
                    limit,
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get drift history: {e}")
            return []

    async def get_active_alerts(self) -> List[Dict]:
        """Get all active drift alerts."""
        if not self._enabled:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT * FROM v_active_drift_alerts
                    """
                )
                return [dict(row) for row in rows]

        except asyncpg.UndefinedTableError:
            # View doesn't exist
            logger.warning("v_active_drift_alerts view not found")
            return []

        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []

    async def acknowledge_alert(
        self,
        alert_id: int,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge a drift alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User/system that acknowledged

        Returns:
            True if successful
        """
        if not self._enabled:
            return False

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE drift_alerts
                    SET state = 'acknowledged',
                        acknowledged_by = $2,
                        acknowledged_at = NOW()
                    WHERE id = $1 AND state = 'active'
                    """,
                    alert_id,
                    acknowledged_by,
                )
                return True

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    async def resolve_alert(
        self,
        alert_id: int,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """
        Resolve a drift alert.

        Args:
            alert_id: Alert ID to resolve
            resolved_by: User/system that resolved
            resolution_notes: Optional notes about resolution

        Returns:
            True if successful
        """
        if not self._enabled:
            return False

        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE drift_alerts
                    SET state = 'resolved',
                        resolved_by = $2,
                        resolved_at = NOW(),
                        resolution_notes = $3
                    WHERE id = $1 AND state IN ('active', 'acknowledged')
                    """,
                    alert_id,
                    resolved_by,
                    resolution_notes,
                )
                return True

        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False

    async def get_summary_24h(self) -> Optional[Dict]:
        """Get 24-hour drift summary."""
        if not self._enabled:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM v_drift_summary_24h
                    """
                )
                return dict(row) if row else None

        except asyncpg.UndefinedTableError:
            logger.warning("v_drift_summary_24h view not found")
            return None

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return None


__all__ = ["DriftPersistenceService"]
