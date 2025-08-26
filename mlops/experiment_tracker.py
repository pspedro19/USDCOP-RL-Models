"""
MLOps Experiment Tracking and Model Registry System
====================================================
Comprehensive experiment tracking, model versioning, and registry management.
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import yaml
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from datetime import datetime
import logging
import os
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    name: str
    algorithm: str
    market: str
    timeframe: str
    description: str
    tags: Dict[str, str]
    hyperparameters: Dict[str, Any]
    data_version: str
    feature_set_version: str

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    # Training metrics
    train_loss: float
    train_accuracy: Optional[float] = None
    
    # Validation metrics
    val_sharpe: float
    val_sortino: float
    val_max_drawdown: float
    val_calmar: float
    
    # Test metrics
    test_sharpe: Optional[float] = None
    test_sortino: Optional[float] = None
    test_max_drawdown: Optional[float] = None
    test_calmar: Optional[float] = None
    
    # Production metrics (if available)
    prod_daily_pnl: Optional[float] = None
    prod_trade_count: Optional[int] = None
    prod_win_rate: Optional[float] = None

class ExperimentTracker:
    """Manages experiment tracking and model registry"""
    
    def __init__(self, config_path: str = "mlops/config/master_pipeline.yml"):
        """Initialize experiment tracker"""
        self.config = self._load_config(config_path)
        self.tracking_config = self.config.get('experiment_tracking', {})
        self.registry_config = self.config.get('model_registry', {})
        
        # Setup MLflow
        self._setup_mlflow()
        self.client = MlflowClient()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_mlflow(self):
        """Setup MLflow tracking server"""
        tracking_uri = self.tracking_config.get('tracking_uri', 'http://localhost:5000')
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set default experiment
        default_experiment = self.tracking_config.get('default_experiment', 'usdcop-trading')
        mlflow.set_experiment(default_experiment)
        
        logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new experiment
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        experiment_name = self._generate_experiment_name(config)
        
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"s3://mlflow-artifacts/{experiment_name}",
                tags=config.tags
            )
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
    
    def start_run(self, 
                  experiment_config: ExperimentConfig,
                  run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run
        
        Args:
            experiment_config: Experiment configuration
            run_name: Optional run name
            
        Returns:
            Run ID
        """
        experiment_id = self.create_experiment(experiment_config)
        
        # Generate run name if not provided
        if run_name is None:
            run_name = f"{experiment_config.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start MLflow run
        mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
        
        # Log tags
        mlflow.set_tags(experiment_config.tags)
        mlflow.set_tag("algorithm", experiment_config.algorithm)
        mlflow.set_tag("market", experiment_config.market)
        mlflow.set_tag("timeframe", experiment_config.timeframe)
        mlflow.set_tag("data_version", experiment_config.data_version)
        mlflow.set_tag("feature_set_version", experiment_config.feature_set_version)
        
        # Log hyperparameters
        mlflow.log_params(experiment_config.hyperparameters)
        
        # Log experiment config
        mlflow.log_dict(asdict(experiment_config), "experiment_config.json")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
        
        return run_id
    
    def log_metrics(self, metrics: ModelMetrics, step: Optional[int] = None):
        """Log model metrics"""
        metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
        
        for metric_name, metric_value in metrics_dict.items():
            mlflow.log_metric(metric_name, metric_value, step=step)
        
        logger.info(f"Logged {len(metrics_dict)} metrics")
    
    def log_model(self,
                  model: Any,
                  model_type: str,
                  signature: Optional[Any] = None,
                  input_example: Optional[pd.DataFrame] = None,
                  artifacts: Optional[Dict[str, str]] = None):
        """
        Log model to MLflow
        
        Args:
            model: Model object
            model_type: Type of model (sklearn, pytorch, tensorflow, etc.)
            signature: Model signature
            input_example: Example input for model
            artifacts: Additional artifacts to log
        """
        # Log model based on type
        if model_type == "sklearn":
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                input_example=input_example
            )
        elif model_type == "pytorch":
            mlflow.pytorch.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example
            )
        elif model_type == "custom":
            # For custom RL models
            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                signature=signature,
                input_example=input_example
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Log additional artifacts
        if artifacts:
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(artifact_path, artifact_name)
        
        logger.info(f"Logged {model_type} model with {len(artifacts or {})} artifacts")
    
    def log_artifacts(self, artifacts: Dict[str, Any]):
        """Log various artifacts"""
        for name, artifact in artifacts.items():
            if isinstance(artifact, pd.DataFrame):
                # Log DataFrames as CSV
                artifact_path = f"/tmp/{name}.csv"
                artifact.to_csv(artifact_path, index=False)
                mlflow.log_artifact(artifact_path)
            elif isinstance(artifact, dict):
                # Log dictionaries as JSON
                mlflow.log_dict(artifact, f"{name}.json")
            elif isinstance(artifact, str) and os.path.exists(artifact):
                # Log file paths
                mlflow.log_artifact(artifact)
            else:
                # Log as text
                mlflow.log_text(str(artifact), f"{name}.txt")
        
        logger.info(f"Logged {len(artifacts)} artifacts")
    
    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run"""
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")
    
    def register_model(self,
                       run_id: str,
                       model_name: str,
                       tags: Optional[Dict[str, str]] = None) -> str:
        """
        Register model in MLflow Model Registry
        
        Args:
            run_id: MLflow run ID
            model_name: Name for registered model
            tags: Optional tags for model version
            
        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Add tags to model version
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version.version,
                    key=key,
                    value=value
                )
        
        logger.info(f"Registered model: {model_name} v{model_version.version}")
        return model_version.version
    
    def promote_model(self,
                      model_name: str,
                      version: str,
                      stage: ModelStage,
                      archive_existing: bool = True) -> bool:
        """
        Promote model to a new stage
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage
            archive_existing: Archive existing production model
            
        Returns:
            Success status
        """
        try:
            # Archive existing production model if needed
            if archive_existing and stage == ModelStage.PRODUCTION:
                existing_prod = self.get_model_by_stage(model_name, ModelStage.PRODUCTION)
                if existing_prod:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=existing_prod.version,
                        stage=ModelStage.ARCHIVED.value
                    )
                    logger.info(f"Archived existing production model v{existing_prod.version}")
            
            # Promote model to new stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage.value
            )
            
            logger.info(f"Promoted model {model_name} v{version} to {stage.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return False
    
    def get_model_by_stage(self,
                           model_name: str,
                           stage: ModelStage) -> Optional[Any]:
        """Get model version by stage"""
        try:
            versions = self.client.search_model_versions(
                filter_string=f"name='{model_name}'",
            )
            
            for version in versions:
                if version.current_stage == stage.value:
                    return version
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model by stage: {e}")
            return None
    
    def evaluate_promotion_criteria(self,
                                   run_id: str,
                                   target_stage: ModelStage) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if model meets promotion criteria
        
        Args:
            run_id: MLflow run ID
            target_stage: Target stage for promotion
            
        Returns:
            Tuple of (meets_criteria, evaluation_details)
        """
        # Get run metrics
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        # Get promotion criteria from config
        stage_config = None
        for stage in self.registry_config.get('stages', []):
            if stage['name'].lower() == target_stage.value.lower():
                stage_config = stage
                break
        
        if not stage_config:
            return False, {"error": f"No criteria defined for stage: {target_stage.value}"}
        
        criteria = stage_config.get('promotion_criteria', {})
        evaluation = {}
        all_passed = True
        
        # Evaluate each criterion
        for criterion_name, criterion_value in criteria.items():
            if criterion_name == "approval" and criterion_value == "manual":
                evaluation[criterion_name] = {"status": "pending", "requires": "manual approval"}
                continue
            
            # Parse criterion (e.g., ">= 1.3")
            if isinstance(criterion_value, str) and any(op in criterion_value for op in ['>=', '<=', '>', '<', '==']):
                for op in ['>=', '<=', '>', '<', '==']:
                    if op in criterion_value:
                        threshold = float(criterion_value.replace(op, '').strip())
                        metric_value = metrics.get(criterion_name.replace('_', ''))
                        
                        if metric_value is None:
                            evaluation[criterion_name] = {"status": "failed", "reason": "metric not found"}
                            all_passed = False
                            continue
                        
                        # Evaluate condition
                        passed = eval(f"{metric_value} {op} {threshold}")
                        evaluation[criterion_name] = {
                            "status": "passed" if passed else "failed",
                            "value": metric_value,
                            "threshold": threshold,
                            "operator": op
                        }
                        
                        if not passed:
                            all_passed = False
                        break
            else:
                # Direct value comparison
                metric_value = metrics.get(criterion_name)
                if metric_value == criterion_value:
                    evaluation[criterion_name] = {"status": "passed", "value": metric_value}
                else:
                    evaluation[criterion_name] = {"status": "failed", "value": metric_value, "expected": criterion_value}
                    all_passed = False
        
        return all_passed, evaluation
    
    def get_best_model(self,
                       experiment_name: str,
                       metric: str = "val_sortino",
                       higher_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get best model from an experiment
        
        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            higher_is_better: Whether higher metric values are better
            
        Returns:
            Best model information
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error(f"Experiment not found: {experiment_name}")
            return None
        
        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'DESC' if higher_is_better else 'ASC'}"],
            max_results=1
        )
        
        if runs.empty:
            logger.warning(f"No runs found in experiment: {experiment_name}")
            return None
        
        best_run = runs.iloc[0]
        
        return {
            "run_id": best_run["run_id"],
            "experiment_id": experiment.experiment_id,
            metric: best_run[f"metrics.{metric}"],
            "params": {col.replace("params.", ""): val 
                      for col, val in best_run.items() 
                      if col.startswith("params.")},
            "tags": {col.replace("tags.", ""): val 
                    for col, val in best_run.items() 
                    if col.startswith("tags.")}
        }
    
    def compare_models(self,
                       run_ids: List[str],
                       metrics: List[str]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "algorithm": run.data.tags.get("algorithm", ""),
                "status": run.info.status
            }
            
            # Add metrics
            for metric in metrics:
                run_data[metric] = run.data.metrics.get(metric)
            
            comparison_data.append(run_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by first metric (descending)
        if metrics:
            comparison_df = comparison_df.sort_values(metrics[0], ascending=False)
        
        return comparison_df
    
    def generate_experiment_report(self, experiment_name: str) -> Dict[str, Any]:
        """Generate comprehensive experiment report"""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return {"error": f"Experiment not found: {experiment_name}"}
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if runs.empty:
            return {"error": "No runs found in experiment"}
        
        # Generate report
        report = {
            "experiment_name": experiment_name,
            "experiment_id": experiment.experiment_id,
            "total_runs": len(runs),
            "successful_runs": len(runs[runs["status"] == "FINISHED"]),
            "failed_runs": len(runs[runs["status"] == "FAILED"]),
            "report_timestamp": datetime.utcnow().isoformat(),
            "best_models": {},
            "summary_statistics": {}
        }
        
        # Find best models for key metrics
        key_metrics = ["val_sortino", "val_sharpe", "val_calmar"]
        for metric in key_metrics:
            if f"metrics.{metric}" in runs.columns:
                best_run = runs.nlargest(1, f"metrics.{metric}").iloc[0]
                report["best_models"][metric] = {
                    "run_id": best_run["run_id"],
                    "value": best_run[f"metrics.{metric}"],
                    "algorithm": best_run.get("tags.algorithm", "unknown")
                }
        
        # Calculate summary statistics
        for col in runs.columns:
            if col.startswith("metrics."):
                metric_name = col.replace("metrics.", "")
                metric_values = runs[col].dropna()
                if not metric_values.empty:
                    report["summary_statistics"][metric_name] = {
                        "mean": float(metric_values.mean()),
                        "std": float(metric_values.std()),
                        "min": float(metric_values.min()),
                        "max": float(metric_values.max()),
                        "median": float(metric_values.median())
                    }
        
        return report
    
    def _generate_experiment_name(self, config: ExperimentConfig) -> str:
        """Generate experiment name from config"""
        template = self.tracking_config.get('experiment_config', {}).get(
            'name_template', 
            "{algorithm}_{market}_{timeframe}"
        )
        
        return template.format(
            algorithm=config.algorithm,
            market=config.market,
            timeframe=config.timeframe,
            date=datetime.now().strftime("%Y%m%d"),
            version=config.data_version
        )


class ModelRegistryManager:
    """Manages model lifecycle in registry"""
    
    def __init__(self, tracker: ExperimentTracker):
        """Initialize registry manager"""
        self.tracker = tracker
        self.client = tracker.client
    
    def automated_promotion_pipeline(self,
                                    model_name: str,
                                    run_id: str) -> Dict[str, Any]:
        """
        Automated model promotion pipeline
        
        Args:
            model_name: Model name for registry
            run_id: MLflow run ID
            
        Returns:
            Promotion results
        """
        results = {
            "run_id": run_id,
            "model_name": model_name,
            "promotions": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Register model
        version = self.tracker.register_model(run_id, model_name)
        results["model_version"] = version
        
        # Evaluate for staging
        can_promote_staging, staging_eval = self.tracker.evaluate_promotion_criteria(
            run_id, ModelStage.STAGING
        )
        
        results["staging_evaluation"] = staging_eval
        
        if can_promote_staging:
            # Promote to staging
            success = self.tracker.promote_model(
                model_name, version, ModelStage.STAGING
            )
            
            if success:
                results["promotions"].append({
                    "stage": "staging",
                    "status": "success",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Evaluate for production (may require manual approval)
                can_promote_prod, prod_eval = self.tracker.evaluate_promotion_criteria(
                    run_id, ModelStage.PRODUCTION
                )
                
                results["production_evaluation"] = prod_eval
                
                if can_promote_prod and prod_eval.get("approval", {}).get("status") != "pending":
                    # Auto-promote to production if criteria met and no manual approval needed
                    success = self.tracker.promote_model(
                        model_name, version, ModelStage.PRODUCTION
                    )
                    
                    if success:
                        results["promotions"].append({
                            "stage": "production",
                            "status": "success",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                else:
                    results["promotions"].append({
                        "stage": "production",
                        "status": "pending_approval" if prod_eval.get("approval", {}).get("status") == "pending" else "criteria_not_met",
                        "timestamp": datetime.utcnow().isoformat()
                    })
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize tracker
    tracker = ExperimentTracker()
    
    # Create experiment config
    config = ExperimentConfig(
        name="PPO-LSTM Trading",
        algorithm="ppo_lstm",
        market="usdcop",
        timeframe="m5",
        description="PPO-LSTM model for USDCOP 5-minute trading",
        tags={
            "team": "ml-trading",
            "project": "usdcop-rl",
            "environment": "development"
        },
        hyperparameters={
            "learning_rate": 0.0003,
            "lstm_units": 64,
            "clip_range": 0.2,
            "batch_size": 1024
        },
        data_version="v2.0.0",
        feature_set_version="v1.5.0"
    )
    
    # Start experiment
    run_id = tracker.start_run(config)
    
    # Log metrics
    metrics = ModelMetrics(
        train_loss=0.023,
        val_sharpe=1.85,
        val_sortino=2.31,
        val_max_drawdown=0.12,
        val_calmar=1.54
    )
    tracker.log_metrics(metrics)
    
    # Log artifacts
    tracker.log_artifacts({
        "feature_importance": {"obs_00": 0.15, "obs_01": 0.12},
        "backtest_results": pd.DataFrame({"date": ["2024-01-01"], "pnl": [1000]})
    })
    
    # End run
    tracker.end_run()
    
    print(f"Experiment completed. Run ID: {run_id}")