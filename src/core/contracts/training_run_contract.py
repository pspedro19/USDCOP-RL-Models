"""
TRAINING RUN CONTRACT
=====================
Define los params, metrics, y artifacts obligatorios para MLflow logging.

Contract ID: CTR-TRAINING-RUN-001
"""
from dataclasses import dataclass, field
from typing import Set, Dict, Any, List, TYPE_CHECKING

# Lazy import to avoid MLflow/PySpark issues at module load
if TYPE_CHECKING:
    import mlflow
    from mlflow.tracking import MlflowClient


@dataclass
class TrainingRunContract:
    """Contrato de lo que debe loguearse en cada training run."""

    required_params: Set[str] = field(default_factory=lambda: {
        "dataset_hash", "norm_stats_hash", "learning_rate", "batch_size",
        "n_epochs", "total_timesteps", "gamma", "gae_lambda", "clip_range",
        "ent_coef", "seed", "observation_dim", "action_dim", "policy_type",
        "feature_contract_version", "action_contract_version",
    })

    required_metrics: Set[str] = field(default_factory=lambda: {
        "final_mean_reward", "best_mean_reward", "total_episodes", "training_time_seconds",
    })

    required_artifacts: Set[str] = field(default_factory=lambda: {
        "model", "norm_stats.json",
    })

    required_tags: Set[str] = field(default_factory=lambda: {
        "mlflow.runName", "version", "environment", "framework",
    })

    def validate_run(self, run_id: str) -> tuple:
        """Valida que un run cumple el contrato."""
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        run = client.get_run(run_id)
        missing = []

        # Check params
        logged_params = set(run.data.params.keys())
        missing_params = self.required_params - logged_params
        missing.extend([f"param:{p}" for p in missing_params])

        # Check metrics
        logged_metrics = set(run.data.metrics.keys())
        for m in self.required_metrics:
            if not m.startswith("val_") and m not in logged_metrics:
                missing.append(f"metric:{m}")

        # Check artifacts
        artifacts = client.list_artifacts(run_id)
        artifact_paths = {a.path for a in artifacts}
        for required in self.required_artifacts:
            if not any(required in path for path in artifact_paths):
                missing.append(f"artifact:{required}")

        # Check tags
        logged_tags = set(run.data.tags.keys())
        missing_tags = self.required_tags - logged_tags
        missing.extend([f"tag:{t}" for t in missing_tags])

        return len(missing) == 0, missing


TRAINING_RUN_CONTRACT = TrainingRunContract()


class TrainingRunValidator:
    """Validador de training runs que enforce el contrato."""

    def __init__(self, strict: bool = True):
        self._contract = TRAINING_RUN_CONTRACT
        self._strict = strict
        self._logged_params: Set[str] = set()
        self._logged_metrics: Set[str] = set()
        self._logged_artifacts: Set[str] = set()
        self._logged_tags: Set[str] = set()

    def log_param(self, key: str, value: Any) -> None:
        import mlflow
        mlflow.log_param(key, value)
        self._logged_params.add(key)

    def log_params(self, params: Dict[str, Any]) -> None:
        import mlflow
        mlflow.log_params(params)
        self._logged_params.update(params.keys())

    def log_metric(self, key: str, value: float, step: int = None) -> None:
        import mlflow
        mlflow.log_metric(key, value, step=step)
        self._logged_metrics.add(key)

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
        self._logged_metrics.update(metrics.keys())

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        import mlflow
        mlflow.log_artifact(local_path, artifact_path)
        self._logged_artifacts.add(local_path.split("/")[-1])

    def set_tag(self, key: str, value: str) -> None:
        import mlflow
        mlflow.set_tag(key, value)
        self._logged_tags.add(key)

    def validate_before_end(self) -> None:
        """Valida antes de mlflow.end_run()."""
        missing = []

        missing_params = self._contract.required_params - self._logged_params
        missing.extend([f"param:{p}" for p in missing_params])

        for m in self._contract.required_metrics:
            if not m.startswith("val_") and m not in self._logged_metrics:
                missing.append(f"metric:{m}")

        missing_artifacts = self._contract.required_artifacts - self._logged_artifacts
        missing.extend([f"artifact:{a}" for a in missing_artifacts])

        missing_tags = self._contract.required_tags - self._logged_tags
        missing.extend([f"tag:{t}" for t in missing_tags])

        if missing and self._strict:
            raise TrainingContractError(f"Missing required items:\n" + "\n".join(f"  - {m}" for m in sorted(missing)))
        elif missing:
            import logging
            logging.warning(f"Training run missing items: {missing}")


class TrainingContractError(Exception):
    pass
