"""
Experiment Registry
===================

Database persistence for experiment tracking and comparison.
Stores experiment runs, metrics, and enables historical analysis.

Usage:
    registry = ExperimentRegistry()
    await registry.register_run(result)
    runs = await registry.get_runs("my_experiment")

Author: Trading Team
Date: 2026-01-17
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

from .experiment_runner import ExperimentResult

logger = logging.getLogger(__name__)


class ExperimentRegistry:
    """
    Database registry for experiments.

    Stores experiment runs and their metrics in PostgreSQL
    for historical analysis and comparison.

    Example:
        registry = ExperimentRegistry()
        await registry.connect(database_url)

        # Register a run
        await registry.register_run(result)

        # Get runs for experiment
        runs = await registry.get_runs("my_experiment", limit=10)

        # Get best run
        best = await registry.get_best_run("my_experiment", metric="sharpe_ratio")
    """

    def __init__(self, db_pool: Optional[Any] = None):
        """
        Initialize registry.

        Args:
            db_pool: Optional existing database pool
        """
        self.db_pool = db_pool
        self._connected = db_pool is not None

    async def connect(self, database_url: str) -> None:
        """
        Connect to database.

        Args:
            database_url: PostgreSQL connection URL
        """
        if not ASYNCPG_AVAILABLE:
            logger.warning("asyncpg not available, registry disabled")
            return

        try:
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=1,
                max_size=5,
            )
            self._connected = True
            logger.info("Experiment registry connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False

    async def close(self) -> None:
        """Close database connection."""
        if self.db_pool:
            await self.db_pool.close()
            self._connected = False

    async def register_run(self, result: ExperimentResult) -> Optional[int]:
        """
        Register an experiment run.

        Args:
            result: ExperimentResult to register

        Returns:
            Run ID in database, or None if failed
        """
        if not self._connected:
            logger.warning("Registry not connected, skipping registration")
            return None

        try:
            async with self.db_pool.acquire() as conn:
                run_id = await conn.fetchval(
                    """
                    INSERT INTO experiment_runs (
                        experiment_name, experiment_version, run_id,
                        status, started_at, completed_at, duration_seconds,
                        training_metrics, eval_metrics, backtest_metrics,
                        model_path, config_path, mlflow_run_id, error
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    )
                    RETURNING id
                    """,
                    result.experiment_name,
                    result.experiment_version,
                    result.run_id,
                    result.status,
                    result.started_at,
                    result.completed_at,
                    result.duration_seconds,
                    json.dumps(result.training_metrics),
                    json.dumps(result.eval_metrics),
                    json.dumps(result.backtest_metrics),
                    result.model_path,
                    result.config_path,
                    result.mlflow_run_id,
                    result.error,
                )

                logger.info(f"Registered run {result.run_id} with ID {run_id}")
                return run_id

        except Exception as e:
            logger.error(f"Failed to register run: {e}")
            return None

    async def get_runs(
        self,
        experiment_name: str,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get runs for an experiment.

        Args:
            experiment_name: Experiment to query
            status: Optional status filter
            limit: Maximum runs to return

        Returns:
            List of run records
        """
        if not self._connected:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                if status:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM experiment_runs
                        WHERE experiment_name = $1 AND status = $2
                        ORDER BY started_at DESC
                        LIMIT $3
                        """,
                        experiment_name,
                        status,
                        limit,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT * FROM experiment_runs
                        WHERE experiment_name = $1
                        ORDER BY started_at DESC
                        LIMIT $2
                        """,
                        experiment_name,
                        limit,
                    )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get runs: {e}")
            return []

    async def get_best_run(
        self,
        experiment_name: str,
        metric: str = "sharpe_ratio",
        higher_is_better: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get best run for an experiment.

        Args:
            experiment_name: Experiment to query
            metric: Metric to optimize
            higher_is_better: True if higher metric is better

        Returns:
            Best run record, or None
        """
        if not self._connected:
            return None

        try:
            order = "DESC" if higher_is_better else "ASC"

            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT * FROM experiment_runs
                    WHERE experiment_name = $1
                      AND status = 'success'
                      AND backtest_metrics ? $2
                    ORDER BY (backtest_metrics->>$2)::FLOAT {order}
                    LIMIT 1
                    """,
                    experiment_name,
                    metric,
                )

                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get best run: {e}")
            return None

    async def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run by run_id.

        Args:
            run_id: Unique run identifier

        Returns:
            Run record, or None
        """
        if not self._connected:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM experiment_runs
                    WHERE run_id = $1
                    """,
                    run_id,
                )
                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get run: {e}")
            return None

    async def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Optional list of metrics to include

        Returns:
            List of run records with metrics
        """
        if not self._connected:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        run_id,
                        experiment_name,
                        experiment_version,
                        status,
                        duration_seconds,
                        backtest_metrics
                    FROM experiment_runs
                    WHERE run_id = ANY($1)
                    """,
                    run_ids,
                )

                results = []
                for row in rows:
                    record = dict(row)
                    backtest = json.loads(record.get("backtest_metrics", "{}"))

                    if metrics:
                        record["metrics"] = {m: backtest.get(m) for m in metrics}
                    else:
                        record["metrics"] = backtest

                    results.append(record)

                return results

        except Exception as e:
            logger.error(f"Failed to compare runs: {e}")
            return []

    async def get_experiment_summary(
        self,
        experiment_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get summary statistics for an experiment.

        Args:
            experiment_name: Experiment name

        Returns:
            Summary statistics
        """
        if not self._connected:
            return None

        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        experiment_name,
                        COUNT(*) as total_runs,
                        COUNT(*) FILTER (WHERE status = 'success') as successful_runs,
                        COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
                        AVG(duration_seconds) as avg_duration,
                        MIN(started_at) as first_run,
                        MAX(started_at) as last_run,
                        AVG((backtest_metrics->>'sharpe_ratio')::FLOAT)
                            FILTER (WHERE status = 'success') as avg_sharpe
                    FROM experiment_runs
                    WHERE experiment_name = $1
                    GROUP BY experiment_name
                    """,
                    experiment_name,
                )

                return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return None

    async def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments with run counts.

        Returns:
            List of experiments
        """
        if not self._connected:
            return []

        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        experiment_name,
                        COUNT(*) as run_count,
                        COUNT(DISTINCT experiment_version) as version_count,
                        MAX(started_at) as last_run
                    FROM experiment_runs
                    GROUP BY experiment_name
                    ORDER BY last_run DESC
                    """
                )

                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []


# File-based registry fallback
class FileBasedRegistry:
    """
    File-based registry when database not available.

    Stores experiment results in JSON files.
    """

    def __init__(self, base_dir: Path = Path("experiments")):
        """
        Initialize file-based registry.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def register_run(self, result: ExperimentResult) -> str:
        """Register run to file."""
        exp_dir = self.base_dir / result.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        result_path = exp_dir / f"{result.run_id}.json"
        result.save(result_path)

        return str(result_path)

    def get_runs(self, experiment_name: str) -> List[ExperimentResult]:
        """Get runs from files."""
        exp_dir = self.base_dir / experiment_name
        if not exp_dir.exists():
            return []

        results = []
        for json_file in exp_dir.glob("*.json"):
            try:
                result = ExperimentResult.load(json_file)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")

        return sorted(results, key=lambda x: x.started_at, reverse=True)

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "sharpe_ratio",
    ) -> Optional[ExperimentResult]:
        """Get best run by metric."""
        runs = self.get_runs(experiment_name)
        successful = [r for r in runs if r.status == "success"]

        if not successful:
            return None

        return max(
            successful,
            key=lambda r: r.backtest_metrics.get(metric, float("-inf")),
        )


__all__ = ["ExperimentRegistry", "FileBasedRegistry"]
