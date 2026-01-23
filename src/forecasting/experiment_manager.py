"""
Forecasting Experiment Manager
==============================

Manages the lifecycle of forecasting experiments including:
- Configuration loading and validation
- Training execution
- Backtest evaluation
- Result persistence
- A/B comparison orchestration

Design Patterns:
    - Repository Pattern: Database operations
    - Factory Pattern: Experiment creation from config
    - Command Pattern: Experiment operations
    - SSOT: All configs reference contracts

@version 1.0.0
@contract CTR-FORECAST-EXPERIMENT-001
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

try:
    from src.forecasting.contracts import (
        HORIZONS,
        MODEL_IDS,
        MODEL_DEFINITIONS,
        FORECASTING_CONTRACT_VERSION,
        ForecastingTrainingRequest,
        ForecastingTrainingResult,
    )
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    HORIZONS = (1, 5, 10, 15, 20, 25, 30)
    MODEL_IDS = ("ridge", "bayesian_ridge", "ard", "xgboost_pure", "lightgbm_pure",
                 "catboost_pure", "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost")
    FORECASTING_CONTRACT_VERSION = "1.0.0"

try:
    from src.forecasting.ab_statistics import (
        ForecastABStatistics,
        ExperimentComparisonResult,
        Recommendation,
    )
    AB_STATS_AVAILABLE = True
except ImportError:
    AB_STATS_AVAILABLE = False


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a forecasting experiment."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    hypothesis: str = ""
    baseline_experiment: Optional[str] = None

    # Models and horizons
    models: Optional[List[str]] = None  # None = all models
    horizons: Optional[List[int]] = None  # None = all horizons

    # Feature configuration
    feature_additions: List[str] = field(default_factory=list)
    feature_removals: List[str] = field(default_factory=list)
    feature_contract_version: str = "1.0.0"

    # Training configuration
    walk_forward_windows: int = 5
    min_train_pct: float = 0.4
    gap_days: int = 30

    # Evaluation configuration
    primary_metric: str = "direction_accuracy"
    secondary_metrics: List[str] = field(default_factory=lambda: ["rmse", "sharpe_ratio"])
    significance_level: float = 0.05
    bonferroni_correction: bool = True

    # MLflow configuration
    mlflow_enabled: bool = True
    mlflow_experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.models is None:
            self.models = list(MODEL_IDS)
        if self.horizons is None:
            self.horizons = list(HORIZONS)
        if self.mlflow_experiment_name is None:
            self.mlflow_experiment_name = f"forecast_exp_{self.name}"

    def compute_hash(self) -> str:
        """Compute unique hash for this configuration."""
        config_dict = {
            "name": self.name,
            "version": self.version,
            "models": sorted(self.models),
            "horizons": sorted(self.horizons),
            "feature_additions": sorted(self.feature_additions),
            "feature_removals": sorted(self.feature_removals),
            "walk_forward_windows": self.walk_forward_windows,
        }
        content = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "baseline_experiment": self.baseline_experiment,
            "models": self.models,
            "horizons": self.horizons,
            "feature_additions": self.feature_additions,
            "feature_removals": self.feature_removals,
            "feature_contract_version": self.feature_contract_version,
            "walk_forward_windows": self.walk_forward_windows,
            "min_train_pct": self.min_train_pct,
            "gap_days": self.gap_days,
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "significance_level": self.significance_level,
            "bonferroni_correction": self.bonferroni_correction,
            "mlflow_enabled": self.mlflow_enabled,
            "mlflow_experiment_name": self.mlflow_experiment_name,
        }

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Extract nested sections
        experiment = data.get("experiment", {})
        models_config = data.get("models", {})
        horizons_config = data.get("horizons", {})
        features_config = data.get("features", {})
        training_config = data.get("training", {})
        evaluation_config = data.get("evaluation", {})

        return cls(
            name=experiment.get("name", "unnamed"),
            version=experiment.get("version", "1.0.0"),
            description=experiment.get("description", ""),
            hypothesis=experiment.get("hypothesis", ""),
            baseline_experiment=experiment.get("baseline_experiment"),
            models=models_config.get("include"),
            horizons=horizons_config.get("include"),
            feature_additions=features_config.get("additions", []),
            feature_removals=features_config.get("removals", []),
            feature_contract_version=features_config.get("contract_version", "1.0.0"),
            walk_forward_windows=training_config.get("walk_forward_windows", 5),
            min_train_pct=training_config.get("min_train_pct", 0.4),
            gap_days=training_config.get("gap_days", 30),
            primary_metric=evaluation_config.get("primary_metric", "direction_accuracy"),
            secondary_metrics=evaluation_config.get("secondary_metrics", ["rmse"]),
            significance_level=evaluation_config.get("significance_level", 0.05),
            bonferroni_correction=evaluation_config.get("bonferroni_correction", True),
            mlflow_enabled=data.get("mlflow", {}).get("enabled", True),
        )


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""
    run_id: str
    experiment_name: str
    experiment_version: str
    config: ExperimentConfig
    config_hash: str

    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    training_metrics: Dict[str, Any] = field(default_factory=dict)
    backtest_metrics: Dict[str, Dict[int, Dict[str, float]]] = field(default_factory=dict)
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)

    model_artifacts_path: Optional[str] = None
    mlflow_run_ids: Dict[str, str] = field(default_factory=dict)

    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "config_hash": self.config_hash,
            "config_json": self.config.to_dict(),
            "models_included": self.config.models,
            "horizons_included": self.config.horizons,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "training_metrics": self.training_metrics,
            "backtest_metrics": self.backtest_metrics,
            "aggregate_metrics": self.aggregate_metrics,
            "model_artifacts_path": self.model_artifacts_path,
            "mlflow_run_ids": self.mlflow_run_ids,
            "error_message": self.error_message,
        }


# =============================================================================
# REPOSITORY PATTERN - Database Operations
# =============================================================================

class ExperimentRepository:
    """Repository for experiment persistence operations."""

    def __init__(self, db_connection_string: Optional[str] = None):
        """
        Initialize repository.

        Args:
            db_connection_string: PostgreSQL connection string
        """
        self.db_url = db_connection_string or os.environ.get("DATABASE_URL")
        self._connection = None

    def _get_connection(self):
        """Get database connection."""
        if self._connection is None or self._connection.closed:
            import psycopg2
            self._connection = psycopg2.connect(self.db_url)
        return self._connection

    def save_run(self, run: ExperimentRun) -> None:
        """Save experiment run to database."""
        conn = self._get_connection()
        cur = conn.cursor()

        data = run.to_dict()

        cur.execute("""
            INSERT INTO bi.forecast_experiment_runs (
                run_id, experiment_name, experiment_version, config_hash,
                config_json, models_included, horizons_included, status,
                started_at, completed_at, training_metrics, backtest_metrics,
                aggregate_metrics, model_artifacts_path, mlflow_run_ids,
                error_message
            ) VALUES (
                %(run_id)s, %(experiment_name)s, %(experiment_version)s, %(config_hash)s,
                %(config_json)s, %(models_included)s, %(horizons_included)s, %(status)s,
                %(started_at)s, %(completed_at)s, %(training_metrics)s, %(backtest_metrics)s,
                %(aggregate_metrics)s, %(model_artifacts_path)s, %(mlflow_run_ids)s,
                %(error_message)s
            )
            ON CONFLICT (experiment_name, run_id) DO UPDATE SET
                status = EXCLUDED.status,
                completed_at = EXCLUDED.completed_at,
                training_metrics = EXCLUDED.training_metrics,
                backtest_metrics = EXCLUDED.backtest_metrics,
                aggregate_metrics = EXCLUDED.aggregate_metrics,
                model_artifacts_path = EXCLUDED.model_artifacts_path,
                mlflow_run_ids = EXCLUDED.mlflow_run_ids,
                error_message = EXCLUDED.error_message,
                updated_at = NOW()
        """, {
            **data,
            "config_json": json.dumps(data["config_json"]),
            "training_metrics": json.dumps(data["training_metrics"]),
            "backtest_metrics": json.dumps(data["backtest_metrics"]),
            "aggregate_metrics": json.dumps(data["aggregate_metrics"]),
            "mlflow_run_ids": json.dumps(data["mlflow_run_ids"]),
        })

        conn.commit()
        cur.close()

    def get_run(self, experiment_name: str, run_id: str) -> Optional[ExperimentRun]:
        """Get experiment run by ID."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                run_id, experiment_name, experiment_version, config_hash,
                config_json, status, started_at, completed_at,
                training_metrics, backtest_metrics, aggregate_metrics,
                model_artifacts_path, mlflow_run_ids, error_message
            FROM bi.forecast_experiment_runs
            WHERE experiment_name = %s AND run_id = %s
        """, (experiment_name, run_id))

        row = cur.fetchone()
        cur.close()

        if row is None:
            return None

        config_dict = row[4] if isinstance(row[4], dict) else json.loads(row[4])
        config = ExperimentConfig(**config_dict)

        return ExperimentRun(
            run_id=row[0],
            experiment_name=row[1],
            experiment_version=row[2],
            config_hash=row[3],
            config=config,
            status=row[5],
            started_at=row[6],
            completed_at=row[7],
            training_metrics=row[8] if isinstance(row[8], dict) else json.loads(row[8] or "{}"),
            backtest_metrics=row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
            aggregate_metrics=row[10] if isinstance(row[10], dict) else json.loads(row[10] or "{}"),
            model_artifacts_path=row[11],
            mlflow_run_ids=row[12] if isinstance(row[12], dict) else json.loads(row[12] or "{}"),
            error_message=row[13],
        )

    def get_latest_run(self, experiment_name: str) -> Optional[ExperimentRun]:
        """Get latest successful run for an experiment."""
        conn = self._get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT run_id
            FROM bi.forecast_experiment_runs
            WHERE experiment_name = %s AND status = 'success'
            ORDER BY completed_at DESC
            LIMIT 1
        """, (experiment_name,))

        row = cur.fetchone()
        cur.close()

        if row is None:
            return None

        return self.get_run(experiment_name, row[0])

    def save_comparison(
        self,
        result: "ExperimentComparisonResult",
        baseline_run_id: str,
        treatment_run_id: str,
    ) -> str:
        """Save A/B comparison result to database."""
        conn = self._get_connection()
        cur = conn.cursor()

        comparison_uuid = str(uuid.uuid4())

        cur.execute("""
            INSERT INTO bi.forecast_experiment_comparisons (
                comparison_uuid, baseline_experiment, baseline_version, baseline_run_id,
                treatment_experiment, treatment_version, treatment_run_id,
                primary_metric, horizon_results, statistical_tests,
                aggregate_p_value, aggregate_significant,
                treatment_wins, baseline_wins, ties,
                recommendation, confidence_score, warnings
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING comparison_uuid
        """, (
            comparison_uuid,
            result.baseline_experiment,
            "1.0.0",
            baseline_run_id,
            result.treatment_experiment,
            "1.0.0",
            treatment_run_id,
            result.primary_metric,
            json.dumps({str(k): v.to_dict() for k, v in result.horizon_results.items()}),
            json.dumps(result.aggregate_result.to_dict()),
            result.aggregate_result.p_value,
            result.aggregate_result.significant,
            result.summary.get("treatment_wins", 0),
            result.summary.get("baseline_wins", 0),
            result.summary.get("ties", 0),
            result.recommendation.value,
            result.confidence_score,
            result.warnings,
        ))

        conn.commit()
        cur.close()

        return comparison_uuid

    def close(self):
        """Close database connection."""
        if self._connection and not self._connection.closed:
            self._connection.close()


# =============================================================================
# MAIN EXPERIMENT MANAGER
# =============================================================================

class ForecastExperimentManager:
    """
    Manages forecasting experiment lifecycle.

    Usage:
        manager = ForecastExperimentManager(config_path="config/experiments/my_exp.yaml")
        run = manager.create_run()
        manager.train(run)
        manager.backtest(run)
        manager.save(run)

        # Compare with baseline
        comparison = manager.compare_with_baseline(run)
    """

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        config_path: Optional[Union[str, Path]] = None,
        project_root: Optional[Path] = None,
        db_connection_string: Optional[str] = None,
    ):
        """
        Initialize experiment manager.

        Args:
            config: ExperimentConfig object
            config_path: Path to YAML config file (alternative to config)
            project_root: Project root directory
            db_connection_string: PostgreSQL connection string
        """
        if config is None and config_path is None:
            raise ValueError("Either config or config_path must be provided")

        self.config = config or ExperimentConfig.from_yaml(config_path)
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self.repository = ExperimentRepository(db_connection_string)

        self._engine = None
        self._ab_stats = None

    @property
    def engine(self):
        """Lazy load ForecastingEngine."""
        if self._engine is None:
            try:
                from src.forecasting.engine import ForecastingEngine
                self._engine = ForecastingEngine(project_root=self.project_root)
            except ImportError:
                logger.warning("ForecastingEngine not available")
        return self._engine

    @property
    def ab_stats(self) -> "ForecastABStatistics":
        """Lazy load A/B Statistics."""
        if self._ab_stats is None:
            if AB_STATS_AVAILABLE:
                self._ab_stats = ForecastABStatistics(
                    alpha=self.config.significance_level,
                    bonferroni_correction=self.config.bonferroni_correction,
                )
            else:
                raise RuntimeError("ForecastABStatistics not available")
        return self._ab_stats

    def create_run(self) -> ExperimentRun:
        """Create a new experiment run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        return ExperimentRun(
            run_id=run_id,
            experiment_name=self.config.name,
            experiment_version=self.config.version,
            config=self.config,
            config_hash=self.config.compute_hash(),
            status="pending",
        )

    def train(self, run: ExperimentRun, dataset_path: Optional[str] = None) -> ExperimentRun:
        """
        Train models for the experiment.

        Args:
            run: ExperimentRun to train
            dataset_path: Path to training dataset

        Returns:
            Updated ExperimentRun
        """
        run.status = "running"
        run.started_at = datetime.now()

        try:
            if self.engine is None:
                raise RuntimeError("ForecastingEngine not available")

            # Build training request
            request = ForecastingTrainingRequest(
                dataset_path=dataset_path or str(self.project_root / "data" / "forecasting" / "features.parquet"),
                version=f"{self.config.name}_{run.run_id}",
                experiment_name=self.config.mlflow_experiment_name,
                models=self.config.models,
                horizons=self.config.horizons,
                mlflow_enabled=self.config.mlflow_enabled,
                walk_forward_windows=self.config.walk_forward_windows,
            )

            logger.info(f"Training experiment {run.experiment_name} ({run.run_id})")
            result = self.engine.train(request)

            if not result.success:
                run.status = "failed"
                run.error_message = "; ".join(result.errors)
            else:
                run.training_metrics = result.metrics_summary
                run.model_artifacts_path = result.model_artifacts_path
                run.mlflow_run_ids = result.mlflow_run_ids

                # Compute aggregate metrics
                all_da = []
                for model_metrics in result.metrics_summary.values():
                    for horizon, da in model_metrics.items():
                        all_da.append(da)

                run.aggregate_metrics = {
                    "avg_direction_accuracy": sum(all_da) / len(all_da) if all_da else 0,
                    "best_model_per_horizon": result.best_model_per_horizon,
                    "models_trained": result.models_trained,
                    "training_duration_seconds": result.training_duration_seconds,
                }

                run.status = "success"
                run.completed_at = datetime.now()

        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            logger.exception(f"Training failed: {e}")

        return run

    def backtest(
        self,
        run: ExperimentRun,
        actual_prices: Optional[pd.DataFrame] = None,
    ) -> ExperimentRun:
        """
        Run backtest evaluation for the experiment.

        Args:
            run: ExperimentRun to backtest
            actual_prices: Actual price data

        Returns:
            Updated ExperimentRun with backtest metrics
        """
        if run.status != "success":
            logger.warning(f"Cannot backtest run with status {run.status}")
            return run

        try:
            # Load actual prices if not provided
            if actual_prices is None:
                actual_prices = self._load_actual_prices()

            # Run backtest for each model/horizon
            backtest_results = {}

            for model_id in self.config.models:
                model_results = {}

                for horizon in self.config.horizons:
                    # Load predictions for this model/horizon
                    predictions = self._load_predictions(run, model_id, horizon)

                    if predictions is None or len(predictions) == 0:
                        continue

                    # Calculate metrics
                    metrics = self._calculate_backtest_metrics(
                        predictions, actual_prices, horizon
                    )
                    model_results[horizon] = metrics

                backtest_results[model_id] = model_results

            run.backtest_metrics = backtest_results

            # Update aggregate metrics
            all_da = []
            for model_metrics in backtest_results.values():
                for horizon_metrics in model_metrics.values():
                    if "direction_accuracy" in horizon_metrics:
                        all_da.append(horizon_metrics["direction_accuracy"])

            if all_da:
                run.aggregate_metrics["backtest_avg_da"] = sum(all_da) / len(all_da)

        except Exception as e:
            logger.exception(f"Backtest failed: {e}")
            run.error_message = (run.error_message or "") + f"; Backtest error: {e}"

        return run

    def compare_with_baseline(
        self,
        treatment_run: ExperimentRun,
        baseline_name: Optional[str] = None,
    ) -> Optional["ExperimentComparisonResult"]:
        """
        Compare experiment with baseline using A/B statistics.

        Args:
            treatment_run: Treatment experiment run
            baseline_name: Baseline experiment name (default: from config)

        Returns:
            ExperimentComparisonResult or None
        """
        baseline_name = baseline_name or self.config.baseline_experiment

        if baseline_name is None:
            logger.warning("No baseline experiment specified")
            return None

        # Load baseline run
        baseline_run = self.repository.get_latest_run(baseline_name)
        if baseline_run is None:
            logger.error(f"Baseline experiment '{baseline_name}' not found")
            return None

        try:
            # Load actual prices
            actual_prices = self._load_actual_prices()

            # Prepare results by horizon
            baseline_results = self._prepare_results_for_comparison(baseline_run)
            treatment_results = self._prepare_results_for_comparison(treatment_run)

            # Run comparison
            result = self.ab_stats.compare_experiments(
                baseline_name=baseline_name,
                treatment_name=treatment_run.experiment_name,
                baseline_results=baseline_results,
                treatment_results=treatment_results,
                actual_prices=actual_prices,
                primary_metric=self.config.primary_metric,
            )

            # Save comparison
            comparison_uuid = self.repository.save_comparison(
                result,
                baseline_run.run_id,
                treatment_run.run_id,
            )

            logger.info(f"Comparison saved: {comparison_uuid}")
            logger.info(f"Recommendation: {result.recommendation.value}")

            return result

        except Exception as e:
            logger.exception(f"Comparison failed: {e}")
            return None

    def save(self, run: ExperimentRun) -> None:
        """Save experiment run to database."""
        self.repository.save_run(run)
        logger.info(f"Saved run {run.run_id} with status {run.status}")

    def _load_actual_prices(self) -> pd.DataFrame:
        """Load actual daily prices."""
        try:
            import psycopg2
            conn = psycopg2.connect(os.environ.get("DATABASE_URL"))

            df = pd.read_sql("""
                SELECT date, close
                FROM bi.dim_daily_usdcop
                ORDER BY date
            """, conn)
            conn.close()

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            return df

        except Exception as e:
            logger.error(f"Failed to load actual prices: {e}")
            return pd.DataFrame()

    def _load_predictions(
        self,
        run: ExperimentRun,
        model_id: str,
        horizon: int,
    ) -> Optional[pd.DataFrame]:
        """Load predictions for a model/horizon from the run."""
        # This would load from the model artifacts or database
        # For now, return None as placeholder
        return None

    def _calculate_backtest_metrics(
        self,
        predictions: pd.DataFrame,
        actual_prices: pd.DataFrame,
        horizon: int,
    ) -> Dict[str, float]:
        """Calculate backtest metrics for predictions."""
        # Placeholder implementation
        return {
            "direction_accuracy": 0.0,
            "rmse": 0.0,
            "mae": 0.0,
        }

    def _prepare_results_for_comparison(
        self,
        run: ExperimentRun,
    ) -> Dict[int, pd.DataFrame]:
        """Prepare run results for A/B comparison."""
        # Convert backtest_metrics to format expected by AB statistics
        results = {}

        for horizon in self.config.horizons:
            horizon_data = []

            for model_id, model_metrics in run.backtest_metrics.items():
                if horizon in model_metrics:
                    horizon_data.append({
                        "model_id": model_id,
                        "horizon": horizon,
                        **model_metrics[horizon],
                    })

            if horizon_data:
                results[horizon] = pd.DataFrame(horizon_data)

        return results


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_experiment_from_config(
    config_path: Union[str, Path],
    **kwargs,
) -> ForecastExperimentManager:
    """
    Factory function to create experiment manager from config file.

    Args:
        config_path: Path to YAML configuration
        **kwargs: Additional arguments for ForecastExperimentManager

    Returns:
        ForecastExperimentManager instance
    """
    return ForecastExperimentManager(config_path=config_path, **kwargs)


__all__ = [
    # Configuration
    "ExperimentConfig",
    "ExperimentRun",
    # Repository
    "ExperimentRepository",
    # Manager
    "ForecastExperimentManager",
    # Factory
    "create_experiment_from_config",
]
