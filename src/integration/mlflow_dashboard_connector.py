#!/usr/bin/env python3
"""
MLflow Dashboard Connector
===========================
Integrates MLflow experiment tracking and model registry with dashboards.
Provides real-time model performance metrics and experiment comparison.
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import requests
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio
import aiohttp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelStage(Enum):
    """Model lifecycle stages"""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: Dict[str, str]
    timestamp: datetime
    run_id: str

class MLflowDashboardConnector:
    """Connects MLflow tracking server with dashboard components"""
    
    def __init__(self, 
                 tracking_uri: str = "http://localhost:5000",
                 experiment_name: str = "usdcop-trading",
                 dashboard_api: str = "http://localhost:5005/api"):
        """
        Initialize MLflow connector
        
        Args:
            tracking_uri: MLflow tracking server URI
            experiment_name: Default experiment name
            dashboard_api: Dashboard API endpoint
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.dashboard_api = dashboard_api
        
        # Setup MLflow
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)
        
        # Model registry
        self.registered_models = {}
        
        # Metrics cache
        self.metrics_cache = {}
        self.last_update = None
        
        # Performance thresholds
        self.performance_gates = {
            'sharpe_ratio': 1.5,
            'sortino_ratio': 2.0,
            'max_drawdown': -0.15,
            'win_rate': 0.55,
            'profit_factor': 1.2,
            'calmar_ratio': 1.0
        }
    
    def get_experiment_metrics(self, 
                              experiment_name: Optional[str] = None,
                              last_n_runs: int = 10) -> List[ModelMetrics]:
        """
        Get metrics from recent experiment runs
        
        Args:
            experiment_name: Experiment to query
            last_n_runs: Number of recent runs to fetch
            
        Returns:
            List of ModelMetrics objects
        """
        try:
            exp_name = experiment_name or self.experiment_name
            experiment = self.client.get_experiment_by_name(exp_name)
            
            if not experiment:
                logger.warning(f"Experiment {exp_name} not found")
                return []
            
            # Get recent runs
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="",
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=last_n_runs,
                order_by=["start_time DESC"]
            )
            
            metrics_list = []
            for run in runs:
                # Extract model info
                model_name = run.data.tags.get('mlflow.runName', 'unknown')
                version = run.data.tags.get('model_version', '1.0')
                stage = run.data.tags.get('model_stage', 'None')
                
                # Create metrics object
                metrics = ModelMetrics(
                    model_name=model_name,
                    version=version,
                    stage=stage,
                    metrics=run.data.metrics,
                    params=run.data.params,
                    tags=run.data.tags,
                    timestamp=datetime.fromtimestamp(run.info.start_time / 1000),
                    run_id=run.info.run_id
                )
                
                metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Error fetching experiment metrics: {e}")
            return []
    
    def get_production_model_metrics(self) -> Optional[ModelMetrics]:
        """Get metrics for current production model"""
        try:
            # Get all registered models
            models = self.client.list_registered_models()
            
            for model in models:
                # Find production version
                for version in model.latest_versions:
                    if version.current_stage == ModelStage.PRODUCTION.value:
                        # Get run details
                        run = self.client.get_run(version.run_id)
                        
                        return ModelMetrics(
                            model_name=model.name,
                            version=version.version,
                            stage=version.current_stage,
                            metrics=run.data.metrics,
                            params=run.data.params,
                            tags=run.data.tags,
                            timestamp=datetime.fromtimestamp(run.info.start_time / 1000),
                            run_id=version.run_id
                        )
            
            logger.warning("No production model found")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching production model metrics: {e}")
            return None
    
    def compare_models(self, 
                       run_ids: List[str]) -> pd.DataFrame:
        """
        Compare metrics across multiple model runs
        
        Args:
            run_ids: List of MLflow run IDs to compare
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                
                row = {
                    'run_id': run_id,
                    'model_name': run.data.tags.get('mlflow.runName', 'unknown'),
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                    'duration_min': (run.info.end_time - run.info.start_time) / 60000,
                    **run.data.metrics,
                    **{f"param_{k}": v for k, v in run.data.params.items()}
                }
                
                comparison_data.append(row)
                
            except Exception as e:
                logger.error(f"Error fetching run {run_id}: {e}")
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            # Sort by sharpe ratio descending
            if 'sharpe_ratio' in df.columns:
                df = df.sort_values('sharpe_ratio', ascending=False)
            return df
        
        return pd.DataFrame()
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary for dashboard"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'production_model': None,
            'staging_models': [],
            'recent_experiments': [],
            'performance_comparison': {},
            'alerts': []
        }
        
        try:
            # Get production model
            prod_model = self.get_production_model_metrics()
            if prod_model:
                summary['production_model'] = {
                    'name': prod_model.model_name,
                    'version': prod_model.version,
                    'metrics': prod_model.metrics,
                    'run_id': prod_model.run_id,
                    'deployed_at': prod_model.timestamp.isoformat()
                }
                
                # Check performance gates
                for metric, threshold in self.performance_gates.items():
                    if metric in prod_model.metrics:
                        value = prod_model.metrics[metric]
                        if metric == 'max_drawdown':
                            if value < threshold:
                                summary['alerts'].append({
                                    'severity': 'HIGH',
                                    'message': f"Production model {metric}: {value:.3f} below threshold {threshold}",
                                    'timestamp': datetime.now().isoformat()
                                })
                        else:
                            if value < threshold:
                                summary['alerts'].append({
                                    'severity': 'MEDIUM',
                                    'message': f"Production model {metric}: {value:.3f} below threshold {threshold}",
                                    'timestamp': datetime.now().isoformat()
                                })
            
            # Get staging models
            models = self.client.list_registered_models()
            for model in models:
                for version in model.latest_versions:
                    if version.current_stage == ModelStage.STAGING.value:
                        run = self.client.get_run(version.run_id)
                        summary['staging_models'].append({
                            'name': model.name,
                            'version': version.version,
                            'metrics': run.data.metrics,
                            'run_id': version.run_id
                        })
            
            # Get recent experiments
            recent_metrics = self.get_experiment_metrics(last_n_runs=5)
            for metric in recent_metrics:
                summary['recent_experiments'].append({
                    'model_name': metric.model_name,
                    'run_id': metric.run_id,
                    'timestamp': metric.timestamp.isoformat(),
                    'key_metrics': {
                        'sharpe': metric.metrics.get('sharpe_ratio', 0),
                        'sortino': metric.metrics.get('sortino_ratio', 0),
                        'max_dd': metric.metrics.get('max_drawdown', 0),
                        'total_return': metric.metrics.get('total_return', 0)
                    }
                })
            
            # Performance comparison
            if len(recent_metrics) > 1:
                run_ids = [m.run_id for m in recent_metrics[:3]]
                comparison_df = self.compare_models(run_ids)
                
                if not comparison_df.empty:
                    summary['performance_comparison'] = {
                        'best_sharpe': {
                            'model': comparison_df.iloc[0]['model_name'],
                            'value': comparison_df.iloc[0].get('sharpe_ratio', 0)
                        },
                        'metrics_table': comparison_df[['model_name', 'sharpe_ratio', 
                                                       'sortino_ratio', 'max_drawdown']].to_dict('records')
                    }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            summary['alerts'].append({
                'severity': 'HIGH',
                'message': f"Error fetching MLflow metrics: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
        
        return summary
    
    async def stream_metrics_to_dashboard(self, update_interval: int = 30):
        """
        Stream MLflow metrics to dashboard via WebSocket
        
        Args:
            update_interval: Seconds between updates
        """
        logger.info("Starting MLflow metrics streaming to dashboard")
        
        while True:
            try:
                # Get latest metrics
                summary = self.get_model_performance_summary()
                
                # Send to dashboard API
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.dashboard_api}/mlflow/metrics",
                        json=summary
                    ) as response:
                        if response.status == 200:
                            logger.info("MLflow metrics sent to dashboard")
                        else:
                            logger.warning(f"Failed to send metrics: {response.status}")
                
                # Cache metrics
                self.metrics_cache = summary
                self.last_update = datetime.now()
                
            except Exception as e:
                logger.error(f"Error streaming metrics: {e}")
            
            await asyncio.sleep(update_interval)
    
    def get_model_artifacts(self, run_id: str) -> Dict[str, Any]:
        """
        Get model artifacts for a specific run
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with artifact information
        """
        artifacts = {
            'run_id': run_id,
            'artifacts': []
        }
        
        try:
            # List artifacts
            artifact_list = self.client.list_artifacts(run_id)
            
            for artifact in artifact_list:
                artifacts['artifacts'].append({
                    'path': artifact.path,
                    'is_dir': artifact.is_dir,
                    'size': artifact.file_size if not artifact.is_dir else None
                })
            
            # Check for specific artifacts
            expected_artifacts = [
                'model/policy.onnx',
                'model/requirements.txt',
                'metrics/performance_report.json',
                'plots/learning_curves.png',
                'plots/reward_distribution.png'
            ]
            
            for expected in expected_artifacts:
                found = any(a['path'] == expected for a in artifacts['artifacts'])
                if not found:
                    logger.warning(f"Missing expected artifact: {expected}")
            
        except Exception as e:
            logger.error(f"Error fetching artifacts for run {run_id}: {e}")
        
        return artifacts
    
    def promote_model_to_production(self, 
                                   model_name: str,
                                   version: str) -> bool:
        """
        Promote a model version to production
        
        Args:
            model_name: Registered model name
            version: Model version to promote
            
        Returns:
            Success status
        """
        try:
            # Transition current production to archived
            models = self.client.list_registered_models()
            for model in models:
                if model.name == model_name:
                    for v in model.latest_versions:
                        if v.current_stage == ModelStage.PRODUCTION.value:
                            self.client.transition_model_version_stage(
                                name=model_name,
                                version=v.version,
                                stage=ModelStage.ARCHIVED.value
                            )
                            logger.info(f"Archived previous production version {v.version}")
            
            # Promote new version to production
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=ModelStage.PRODUCTION.value
            )
            
            logger.info(f"Promoted {model_name} version {version} to production")
            
            # Send notification to dashboard
            notification = {
                'event': 'model_promoted',
                'model_name': model_name,
                'version': version,
                'stage': 'Production',
                'timestamp': datetime.now().isoformat()
            }
            
            requests.post(f"{self.dashboard_api}/notifications", json=notification)
            
            return True
            
        except Exception as e:
            logger.error(f"Error promoting model to production: {e}")
            return False
    
    def get_experiment_tracking_url(self, run_id: str) -> str:
        """Get MLflow UI URL for a specific run"""
        return f"{self.tracking_uri}/#/experiments/0/runs/{run_id}"
    
    def validate_model_for_production(self, run_id: str) -> Tuple[bool, List[str]]:
        """
        Validate if a model meets production criteria
        
        Args:
            run_id: MLflow run ID to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            
            # Check performance gates
            for metric, threshold in self.performance_gates.items():
                if metric not in metrics:
                    issues.append(f"Missing required metric: {metric}")
                else:
                    value = metrics[metric]
                    if metric == 'max_drawdown':
                        if value < threshold:
                            issues.append(f"{metric}: {value:.3f} below threshold {threshold}")
                    else:
                        if value < threshold:
                            issues.append(f"{metric}: {value:.3f} below threshold {threshold}")
            
            # Check for required artifacts
            artifacts = self.get_model_artifacts(run_id)
            required = ['model/policy.onnx', 'model/requirements.txt']
            
            for req in required:
                if not any(a['path'] == req for a in artifacts['artifacts']):
                    issues.append(f"Missing required artifact: {req}")
            
            # Check tags
            required_tags = ['model_type', 'dataset_version', 'training_date']
            for tag in required_tags:
                if tag not in run.data.tags:
                    issues.append(f"Missing required tag: {tag}")
            
            is_valid = len(issues) == 0
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            return False, [f"Validation error: {str(e)}"]


def main():
    """Main execution for testing"""
    
    # Initialize connector
    connector = MLflowDashboardConnector()
    
    # Get performance summary
    summary = connector.get_model_performance_summary()
    print(json.dumps(summary, indent=2, default=str))
    
    # Start streaming (for production use)
    # asyncio.run(connector.stream_metrics_to_dashboard())


if __name__ == "__main__":
    main()