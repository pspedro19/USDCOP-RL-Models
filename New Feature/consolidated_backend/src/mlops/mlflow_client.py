# backend/src/mlops/mlflow_client.py
"""
MLflow Client for experiment tracking and model registry.

This module provides a wrapper around MLflow to standardize experiment tracking,
metrics logging, and model registry operations for the ML pipeline.

Environment Variables:
    MLFLOW_TRACKING_URI: URI for the MLflow tracking server (default: ./mlruns)
    MLFLOW_EXPERIMENT_NAME: Default experiment name (optional)
    MLFLOW_ARTIFACT_LOCATION: Default artifact storage location (optional)

Example:
    client = MLflowClient()
    client.initialize("usd_cop_forecasting")
    client.start_run("hybrid_ensemble_v1", tags={"version": "1.0"})
    client.log_params({"n_estimators": 100, "max_depth": 3})
    client.log_metrics({"da_test": 62.5, "rmse": 0.0125})
    client.log_model(model, "model", signature=signature)
    client.end_run()
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient as _MlflowClient
    from mlflow.models.signature import ModelSignature, infer_signature
    from mlflow.exceptions import MlflowException
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    _MlflowClient = None
    ModelSignature = None
    MlflowException = Exception

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowClient:
    """
    A wrapper client for MLflow experiment tracking and model registry.

    This class provides a simplified interface for:
    - Initializing experiments
    - Starting and managing runs
    - Logging parameters, metrics, and artifacts
    - Registering models to the MLflow Model Registry

    Attributes:
        tracking_uri (str): The MLflow tracking server URI
        experiment_name (str): Current experiment name
        experiment_id (str): Current experiment ID
        run_id (str): Current active run ID
        client (_MlflowClient): Low-level MLflow client

    Example:
        >>> client = MLflowClient()
        >>> client.initialize("my_experiment")
        >>> client.start_run("training_v1")
        >>> client.log_params({"lr": 0.01})
        >>> client.log_metrics({"accuracy": 0.95})
        >>> client.end_run()
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the MLflow client.

        Args:
            tracking_uri: Optional URI for MLflow tracking server.
                         If not provided, uses MLFLOW_TRACKING_URI env var
                         or defaults to './mlruns'
        """
        if not MLFLOW_AVAILABLE:
            logger.warning(
                "MLflow is not installed. Install with: pip install mlflow"
            )
            self._mock_mode = True
        else:
            self._mock_mode = False

        # Set tracking URI from parameter, env var, or default
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "./mlruns"
        )

        self.experiment_name: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self.run_id: Optional[str] = None
        self._active_run = None
        self._client: Optional[_MlflowClient] = None
        self._initialized = False

        logger.info(f"MLflow client created with tracking URI: {self.tracking_uri}")

    def initialize(
        self,
        experiment_name: str,
        artifact_location: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Initialize or get an MLflow experiment.

        Creates a new experiment if it doesn't exist, or retrieves the existing one.

        Args:
            experiment_name: Name of the experiment
            artifact_location: Optional S3/GCS/local path for artifacts
            tags: Optional dictionary of experiment-level tags

        Returns:
            The experiment ID

        Example:
            >>> experiment_id = client.initialize("usd_cop_forecast")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Initialized experiment: {experiment_name}")
            self.experiment_name = experiment_name
            self.experiment_id = "mock_experiment_id"
            self._initialized = True
            return self.experiment_id

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location,
                    tags=tags
                )
                logger.info(
                    f"Created new experiment: {experiment_name} "
                    f"(ID: {self.experiment_id})"
                )
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(
                    f"Using existing experiment: {experiment_name} "
                    f"(ID: {self.experiment_id})"
                )
        except MlflowException as e:
            logger.error(f"Error initializing experiment: {e}")
            raise

        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

        # Initialize low-level client
        self._client = _MlflowClient(self.tracking_uri)
        self._initialized = True

        return self.experiment_id

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
        nested: bool = False
    ) -> str:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags for the run
            description: Optional description for the run
            nested: If True, start a nested run within current run

        Returns:
            The run ID

        Example:
            >>> run_id = client.start_run(
            ...     "hybrid_training_v1",
            ...     tags={"model_type": "hybrid", "horizon": "15"}
            ... )
        """
        if self._mock_mode:
            self.run_id = f"mock_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"[MOCK] Started run: {run_name} (ID: {self.run_id})")
            return self.run_id

        if not self._initialized:
            raise RuntimeError(
                "Client not initialized. Call initialize() first."
            )

        # Prepare tags
        run_tags = tags or {}
        if description:
            run_tags["mlflow.note.content"] = description

        # Generate run name if not provided
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Start the run
        self._active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=run_tags,
            nested=nested
        )
        self.run_id = self._active_run.info.run_id

        logger.info(f"Started run: {run_name} (ID: {self.run_id})")
        return self.run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to the current run.

        Args:
            params: Dictionary of parameter names and values.
                   Values are automatically converted to strings.

        Example:
            >>> client.log_params({
            ...     "n_estimators": 100,
            ...     "max_depth": 3,
            ...     "learning_rate": 0.01
            ... })
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Logged params: {params}")
            return

        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        # MLflow requires string values for params
        # Also handle nested dicts by flattening
        flat_params = self._flatten_dict(params)

        # Log in batches (MLflow has a limit of 100 params per batch)
        param_items = list(flat_params.items())
        batch_size = 100

        for i in range(0, len(param_items), batch_size):
            batch = dict(param_items[i:i + batch_size])
            # Convert all values to strings
            string_batch = {k: str(v) for k, v in batch.items()}
            mlflow.log_params(string_batch)

        logger.debug(f"Logged {len(flat_params)} parameters")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ) -> None:
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric names and numeric values
            step: Optional step number for tracking metrics over time

        Example:
            >>> client.log_metrics({
            ...     "da_test": 62.5,
            ...     "rmse": 0.0125,
            ...     "var_ratio": 0.85
            ... }, step=1)
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Logged metrics: {metrics}")
            return

        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        # Filter out non-numeric values and NaN/Inf
        valid_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if not (np.isnan(value) or np.isinf(value)):
                    valid_metrics[key] = float(value)
                else:
                    logger.warning(
                        f"Skipping metric {key}: value is NaN or Inf"
                    )
            else:
                logger.warning(
                    f"Skipping metric {key}: value is not numeric ({type(value)})"
                )

        if step is not None:
            mlflow.log_metrics(valid_metrics, step=step)
        else:
            mlflow.log_metrics(valid_metrics)

        logger.debug(f"Logged {len(valid_metrics)} metrics")

    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Optional[Any] = None,
        input_example: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
        await_registration_for: int = 300,
        pip_requirements: Optional[List[str]] = None,
        extra_pip_requirements: Optional[List[str]] = None,
        conda_env: Optional[Dict] = None
    ) -> str:
        """
        Log a model to the current run.

        Automatically detects the model type and uses the appropriate
        MLflow flavor (sklearn, xgboost, lightgbm, catboost, etc.)

        Args:
            model: The model object to log
            artifact_path: Path within the run's artifact directory
            signature: Optional ModelSignature for input/output schema
            input_example: Optional example input for documentation
            registered_model_name: If provided, registers model in Model Registry
            await_registration_for: Seconds to wait for registration
            pip_requirements: Optional list of pip requirements
            extra_pip_requirements: Additional pip requirements
            conda_env: Optional conda environment specification

        Returns:
            The model URI

        Example:
            >>> model_uri = client.log_model(
            ...     sklearn_model,
            ...     "model",
            ...     registered_model_name="usd_cop_model"
            ... )
        """
        if self._mock_mode:
            model_uri = f"runs:/mock_run/{artifact_path}"
            logger.info(f"[MOCK] Logged model to: {model_uri}")
            return model_uri

        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        # Detect model type and use appropriate flavor
        model_type = type(model).__name__
        model_module = type(model).__module__

        try:
            # Try different MLflow flavors based on model type
            if "xgboost" in model_module.lower() or "XGB" in model_type:
                import mlflow.xgboost
                model_info = mlflow.xgboost.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    await_registration_for=await_registration_for,
                    pip_requirements=pip_requirements,
                    extra_pip_requirements=extra_pip_requirements,
                    conda_env=conda_env
                )
            elif "lightgbm" in model_module.lower() or "LGBM" in model_type:
                import mlflow.lightgbm
                model_info = mlflow.lightgbm.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    await_registration_for=await_registration_for,
                    pip_requirements=pip_requirements,
                    extra_pip_requirements=extra_pip_requirements,
                    conda_env=conda_env
                )
            elif "catboost" in model_module.lower() or "CatBoost" in model_type:
                import mlflow.catboost
                model_info = mlflow.catboost.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    await_registration_for=await_registration_for,
                    pip_requirements=pip_requirements,
                    extra_pip_requirements=extra_pip_requirements,
                    conda_env=conda_env
                )
            else:
                # Default to sklearn for Ridge, BayesianRidge, etc.
                import mlflow.sklearn
                model_info = mlflow.sklearn.log_model(
                    model,
                    artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    await_registration_for=await_registration_for,
                    pip_requirements=pip_requirements,
                    extra_pip_requirements=extra_pip_requirements,
                    conda_env=conda_env
                )

            model_uri = model_info.model_uri
            logger.info(f"Logged model ({model_type}) to: {model_uri}")
            return model_uri

        except Exception as e:
            logger.error(f"Error logging model: {e}")
            # Fallback to pyfunc
            import mlflow.pyfunc
            model_info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                await_registration_for=await_registration_for
            )
            return model_info.model_uri

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a local file or directory as an artifact.

        Args:
            local_path: Path to the local file or directory
            artifact_path: Optional subdirectory within the artifact directory

        Example:
            >>> client.log_artifact("results/metrics.csv", "results")
            >>> client.log_artifact("plots/", "visualizations")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Logged artifact: {local_path}")
            return

        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact path not found: {local_path}")

        if path.is_dir():
            mlflow.log_artifacts(str(path), artifact_path)
            logger.info(f"Logged directory artifacts: {local_path}")
        else:
            mlflow.log_artifact(str(path), artifact_path)
            logger.info(f"Logged artifact: {local_path}")

    def log_figure(
        self,
        figure: Any,
        artifact_file: str
    ) -> None:
        """
        Log a matplotlib or plotly figure.

        Args:
            figure: Matplotlib or Plotly figure object
            artifact_file: Name for the saved figure file

        Example:
            >>> import matplotlib.pyplot as plt
            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3])
            >>> client.log_figure(fig, "training_curve.png")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Logged figure: {artifact_file}")
            return

        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.log_figure(figure, artifact_file)
        logger.info(f"Logged figure: {artifact_file}")

    def log_dict(
        self,
        dictionary: Dict,
        artifact_file: str
    ) -> None:
        """
        Log a dictionary as a JSON or YAML artifact.

        Args:
            dictionary: Dictionary to log
            artifact_file: Name for the file (should end in .json or .yaml)

        Example:
            >>> client.log_dict({"config": "value"}, "config.json")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Logged dict: {artifact_file}")
            return

        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        mlflow.log_dict(dictionary, artifact_file)
        logger.info(f"Logged dict artifact: {artifact_file}")

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End the current run.

        Args:
            status: Run status - "FINISHED", "FAILED", or "KILLED"

        Example:
            >>> client.end_run()  # Success
            >>> client.end_run("FAILED")  # Failure
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Ended run with status: {status}")
            self.run_id = None
            return

        if self._active_run is None:
            logger.warning("No active run to end")
            return

        mlflow.end_run(status=status)
        logger.info(f"Ended run {self.run_id} with status: {status}")

        self._active_run = None
        self.run_id = None

    def register_model(
        self,
        model_uri: str,
        name: str,
        stage: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        await_registration_for: int = 300
    ) -> str:
        """
        Register a model in the MLflow Model Registry.

        Args:
            model_uri: URI of the model to register (e.g., "runs:/run_id/model")
            name: Name for the registered model
            stage: Optional stage to transition to ("Staging", "Production", "Archived")
            description: Optional model description
            tags: Optional tags for the model version
            await_registration_for: Seconds to wait for registration

        Returns:
            The model version number

        Example:
            >>> version = client.register_model(
            ...     "runs:/abc123/model",
            ...     "usd_cop_forecaster",
            ...     stage="Production"
            ... )
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Registered model: {name} from {model_uri}")
            return "1"

        if self._client is None:
            self._client = _MlflowClient(self.tracking_uri)

        # Register the model
        result = mlflow.register_model(
            model_uri=model_uri,
            name=name,
            await_registration_for=await_registration_for,
            tags=tags
        )

        version = result.version
        logger.info(f"Registered model '{name}' version {version}")

        # Update description if provided
        if description:
            self._client.update_model_version(
                name=name,
                version=version,
                description=description
            )

        # Transition to stage if provided
        if stage:
            valid_stages = ["Staging", "Production", "Archived", "None"]
            if stage not in valid_stages:
                raise ValueError(
                    f"Invalid stage: {stage}. Must be one of {valid_stages}"
                )

            self._client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=(stage == "Production")
            )
            logger.info(f"Transitioned model to stage: {stage}")

        return version

    def get_best_run(
        self,
        metric: str,
        ascending: bool = True,
        filter_string: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get the best run from the current experiment based on a metric.

        Args:
            metric: Metric name to optimize
            ascending: If True, lower is better; if False, higher is better
            filter_string: Optional MLflow filter string

        Returns:
            Dictionary with run info and metrics, or None if no runs found

        Example:
            >>> best = client.get_best_run("rmse", ascending=True)
            >>> print(f"Best RMSE: {best['metrics']['rmse']}")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Get best run for metric: {metric}")
            return None

        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        order = "ASC" if ascending else "DESC"

        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            order_by=[f"metrics.{metric} {order}"],
            max_results=1
        )

        if len(runs) == 0:
            return None

        run = runs.iloc[0]
        return {
            "run_id": run.run_id,
            "metrics": {
                col.replace("metrics.", ""): run[col]
                for col in runs.columns
                if col.startswith("metrics.")
            },
            "params": {
                col.replace("params.", ""): run[col]
                for col in runs.columns
                if col.startswith("params.")
            },
            "tags": {
                col.replace("tags.", ""): run[col]
                for col in runs.columns
                if col.startswith("tags.")
            }
        }

    def load_model(self, model_uri: str) -> Any:
        """
        Load a model from MLflow.

        Args:
            model_uri: URI of the model to load
                      - "runs:/run_id/artifact_path"
                      - "models:/model_name/version"
                      - "models:/model_name/stage"

        Returns:
            The loaded model

        Example:
            >>> model = client.load_model("models:/usd_cop_forecaster/Production")
        """
        if self._mock_mode:
            logger.info(f"[MOCK] Load model: {model_uri}")
            return None

        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model from: {model_uri}")
        return model

    def _flatten_dict(
        self,
        d: Dict,
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict:
        """Flatten nested dictionary for MLflow params."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def __enter__(self) -> 'MLflowClient':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures run is ended."""
        if self._active_run is not None:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.end_run(status=status)


# Utility function for creating model signatures
def create_signature(
    X_sample: np.ndarray,
    y_sample: Optional[np.ndarray] = None
) -> Optional[Any]:
    """
    Create an MLflow ModelSignature from sample data.

    Args:
        X_sample: Sample input features
        y_sample: Sample output predictions

    Returns:
        ModelSignature or None if MLflow not available
    """
    if not MLFLOW_AVAILABLE:
        return None

    return infer_signature(X_sample, y_sample)
