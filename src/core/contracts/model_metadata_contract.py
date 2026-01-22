"""
MODEL METADATA CONTRACT
=======================
Define metadata obligatoria para cada modelo.

Contract ID: CTR-MODEL-METADATA-001
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import hashlib


class ModelMetadata(BaseModel):
    """Pydantic model para metadata obligatoria del modelo."""

    # Required fields
    model_id: str = Field(..., description="Unique model identifier")
    model_hash: str = Field(..., description="SHA256 hash of model file")
    dataset_hash: str = Field(..., description="Hash of training dataset")
    norm_stats_hash: str = Field(..., description="Hash of normalization stats")
    feature_contract_version: str = Field(..., description="Version of feature contract")
    action_contract_version: str = Field(..., description="Version of action contract")
    observation_dim: int = Field(15, description="Observation dimension")
    action_dim: int = Field(3, description="Action dimension")

    # Optional but recommended
    training_date: Optional[datetime] = Field(None, description="When model was trained")
    framework_version: Optional[str] = Field(None, description="e.g., stable-baselines3==2.0.0")
    python_version: Optional[str] = Field(None, description="e.g., 3.11.5")
    training_sharpe: Optional[float] = Field(None, description="Sharpe from training")
    validation_sharpe: Optional[float] = Field(None, description="Sharpe from validation")

    @field_validator("observation_dim")
    @classmethod
    def validate_obs_dim(cls, v):
        if v != 15:
            raise ValueError(f"observation_dim must be 15, got {v}")
        return v

    @field_validator("action_dim")
    @classmethod
    def validate_action_dim(cls, v):
        if v != 3:
            raise ValueError(f"action_dim must be 3, got {v}")
        return v

    @field_validator("dataset_hash", "norm_stats_hash", "model_hash")
    @classmethod
    def validate_hash_format(cls, v):
        if len(v) != 64:
            raise ValueError(f"Hash must be 64 hex chars, got {len(v)}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_mlflow_run(cls, run_id: str) -> "ModelMetadata":
        import mlflow
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params

        return cls(
            model_id=params.get("model_id", run_id),
            model_hash=params.get("model_hash", "0" * 64),
            dataset_hash=params.get("dataset_hash", "0" * 64),
            norm_stats_hash=params.get("norm_stats_hash", "0" * 64),
            feature_contract_version=params.get("feature_contract_version", "2.0.0"),
            action_contract_version=params.get("action_contract_version", "1.0.0"),
            observation_dim=int(params.get("observation_dim", 15)),
            action_dim=int(params.get("action_dim", 3)),
            training_date=datetime.fromisoformat(params["training_date"]) if "training_date" in params else None,
            framework_version=params.get("framework_version"),
            python_version=params.get("python_version"),
        )


def validate_model_metadata(metadata: ModelMetadata) -> tuple:
    """Valida metadata completa del modelo."""
    errors = []

    required = ["model_id", "model_hash", "dataset_hash", "norm_stats_hash",
                "feature_contract_version", "action_contract_version"]

    for field_name in required:
        value = getattr(metadata, field_name, None)
        if not value:
            errors.append(f"Missing required field: {field_name}")

    if metadata.observation_dim != 15:
        errors.append(f"Invalid observation_dim: {metadata.observation_dim}")

    if metadata.action_dim != 3:
        errors.append(f"Invalid action_dim: {metadata.action_dim}")

    return len(errors) == 0, errors
