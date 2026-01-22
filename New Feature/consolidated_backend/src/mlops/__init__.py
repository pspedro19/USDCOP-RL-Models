# backend/src/mlops/__init__.py
"""
MLOps module for experiment tracking and model management.

This module provides:
- MLflowClient: For experiment tracking, metrics logging, and model registry
- MinioClient: For model artifact storage in S3-compatible object storage

Usage:
    from backend.src.mlops import MLflowClient, MinioClient

    # MLflow tracking
    mlflow_client = MLflowClient()
    mlflow_client.initialize("my_experiment")
    mlflow_client.start_run("training_run")
    mlflow_client.log_params({"learning_rate": 0.01})
    mlflow_client.log_metrics({"rmse": 0.5})
    mlflow_client.end_run()

    # MinIO storage
    minio_client = MinioClient()
    minio_client.upload_model("models", "model.pkl", "models/v1/model.pkl")
"""

from .mlflow_client import MLflowClient
from .minio_client import MinioClient

__all__ = ["MLflowClient", "MinioClient"]
