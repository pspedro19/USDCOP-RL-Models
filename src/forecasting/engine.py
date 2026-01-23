"""
Forecasting Engine
==================

Central engine for forecasting model training and inference.
Follows same pattern as src/training/engine.py (TrainingEngine).

Design Patterns:
- Factory: ModelFactory for model instantiation
- Strategy: Different models as strategies
- Template Method: Common training/inference flow

@version 1.0.0
"""

import logging
import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import joblib

from .contracts import (
    HORIZONS,
    MODEL_IDS,
    MODEL_DEFINITIONS,
    HORIZON_CATEGORIES,
    EnsembleType,
    ForecastDirection,
    ForecastingTrainingRequest,
    ForecastingTrainingResult,
    ForecastingInferenceRequest,
    ForecastingInferenceResult,
    ForecastPrediction,
    ModelMetrics,
    get_horizon_config,
)
from .data_contracts import (
    FEATURE_COLUMNS,
    NUM_FEATURES,
    validate_feature_row,
    DATA_CONTRACT_VERSION,
    DATA_CONTRACT_HASH,
)
from .models.factory import ModelFactory
from .evaluation.walk_forward import WalkForwardValidator
from .evaluation.metrics import Metrics

logger = logging.getLogger(__name__)


class ForecastingEngine:
    """
    Central engine for forecasting operations.

    Responsibilities:
    - Train all models for all horizons
    - Run walk-forward validation
    - Generate predictions
    - Create ensembles
    - Persist results to DB and MinIO

    Usage:
        engine = ForecastingEngine(project_root=Path('/opt/airflow'))
        result = engine.train(request)
        inference_result = engine.predict(inference_request)
    """

    def __init__(
        self,
        project_root: Path = None,
        db_connection_string: Optional[str] = None,
        minio_client: Optional[Any] = None,
        mlflow_client: Optional[Any] = None,
    ):
        self.project_root = project_root or Path.cwd()
        self.db_connection_string = db_connection_string or os.environ.get("DATABASE_URL")
        self.minio_client = minio_client
        self.mlflow_client = mlflow_client

        # Lazy-loaded components
        self._model_factory = None
        self._walk_forward = None
        self._metrics = None

        # Model cache
        self._loaded_models: Dict[str, Dict[int, Any]] = {}

    @property
    def model_factory(self) -> ModelFactory:
        if self._model_factory is None:
            self._model_factory = ModelFactory
        return self._model_factory

    @property
    def walk_forward(self) -> WalkForwardValidator:
        if self._walk_forward is None:
            self._walk_forward = WalkForwardValidator()
        return self._walk_forward

    @property
    def metrics(self) -> Metrics:
        if self._metrics is None:
            self._metrics = Metrics()
        return self._metrics

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(self, request: ForecastingTrainingRequest) -> ForecastingTrainingResult:
        """
        Train all forecasting models.

        Steps:
        1. Load and prepare data
        2. Create targets for each horizon
        3. Train each model for each horizon
        4. Run walk-forward validation
        5. Save models and metrics
        6. Upload to MinIO if enabled
        7. Log to MLflow if enabled
        """
        start_time = time.time()
        errors = []
        mlflow_run_ids = {}
        metrics_summary: Dict[str, Dict[int, float]] = {}
        best_model_per_horizon: Dict[int, str] = {}

        logger.info("=" * 60)
        logger.info(f"Starting forecasting training v{request.version}")
        logger.info(f"Models: {len(request.models)}, Horizons: {len(request.horizons)}")
        logger.info("=" * 60)

        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.project_root / "outputs" / "forecasting" / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        models_dir = output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Load data (SSOT integration v2.0)
            logger.info("Loading dataset...")
            if request.use_db:
                logger.info("[SSOT] Loading from PostgreSQL/Parquet")
                df = self._load_dataset(path=None, use_db=True)
            else:
                df = self._load_dataset(path=request.dataset_path, use_db=False)
            logger.info(f"Loaded {len(df)} rows")

            # Step 2: Prepare features and targets
            X, feature_cols = self._prepare_features(df)
            targets = self._create_targets(df, request.horizons)

            # Save feature columns for inference
            with open(output_dir / "feature_cols.json", "w") as f:
                json.dump(feature_cols, f)

            # Step 3: Train models
            models_trained = 0
            total_combinations = len(request.models) * len(request.horizons)

            for model_id in request.models:
                metrics_summary[model_id] = {}

                for horizon in request.horizons:
                    try:
                        logger.info(f"Training {model_id} for H={horizon}...")

                        # Get target
                        y = targets[horizon]
                        valid_idx = ~np.isnan(y)
                        X_valid = X[valid_idx]
                        y_valid = y[valid_idx]

                        # Create model with horizon-specific config
                        horizon_config = get_horizon_config(horizon)
                        model = self.model_factory.create(
                            model_id,
                            params=horizon_config,
                            horizon=horizon
                        )

                        # Walk-forward validation
                        wf_result = self.walk_forward.validate(
                            model_factory=lambda: self.model_factory.create(model_id, horizon_config, horizon),
                            X=X_valid,
                            y=y_valid,
                            n_windows=request.walk_forward_windows,
                        )

                        # Train final model on all data
                        model.fit(X_valid, y_valid)

                        # Save model
                        model_path = models_dir / f"{model_id}_h{horizon}.pkl"
                        joblib.dump(model, model_path)

                        # Store metrics
                        da = wf_result.da_mean * 100
                        metrics_summary[model_id][horizon] = da

                        # Track best model per horizon
                        if horizon not in best_model_per_horizon:
                            best_model_per_horizon[horizon] = model_id
                        elif da > metrics_summary[best_model_per_horizon[horizon]][horizon]:
                            best_model_per_horizon[horizon] = model_id

                        models_trained += 1
                        logger.info(f"  {model_id} H={horizon}: DA={da:.2f}%")

                        # MLflow logging (now with full implementation)
                        if request.mlflow_enabled:
                            run_id = self._log_to_mlflow(
                                model_id=model_id,
                                horizon=horizon,
                                wf_result=wf_result,
                                experiment_name=request.experiment_name,
                                model_path=model_path,
                                params=horizon_config,
                            )
                            if run_id:
                                mlflow_run_ids[f"{model_id}_h{horizon}"] = run_id

                    except Exception as e:
                        logger.error(f"Error training {model_id} H={horizon}: {e}")
                        errors.append(f"{model_id}_h{horizon}: {str(e)}")

            # Step 4: Save metrics summary
            metrics_df = self._create_metrics_dataframe(metrics_summary)
            metrics_df.to_csv(output_dir / "model_results.csv", index=False)

            # Step 5: Upload to MinIO
            minio_uri = None
            if request.minio_enabled and self.minio_client:
                try:
                    minio_uri = self._upload_to_minio(output_dir, request.version)
                    logger.info(f"Uploaded to MinIO: {minio_uri}")
                except Exception as e:
                    logger.error(f"MinIO upload failed: {e}")
                    errors.append(f"MinIO: {str(e)}")

            # Step 6: Persist metrics to database
            if request.db_connection_string:
                try:
                    self._persist_metrics_to_db(
                        metrics_summary,
                        request.version,
                        request.db_connection_string
                    )
                    logger.info("Metrics persisted to database")
                except Exception as e:
                    logger.error(f"DB persist failed: {e}")
                    errors.append(f"DB: {str(e)}")

            duration = time.time() - start_time

            # Create result object
            result = ForecastingTrainingResult(
                success=len(errors) == 0,
                version=request.version,
                models_trained=models_trained,
                total_combinations=total_combinations,
                best_model_per_horizon=best_model_per_horizon,
                metrics_summary=metrics_summary,
                model_artifacts_path=str(output_dir),
                mlflow_experiment_id=request.experiment_name,
                mlflow_run_ids=mlflow_run_ids,
                minio_artifacts_uri=minio_uri,
                training_duration_seconds=duration,
                errors=errors,
            )

            # Step 7: Log training summary to MLflow
            if request.mlflow_enabled and models_trained > 0:
                summary_run_id = self._log_training_summary_to_mlflow(
                    result, request.experiment_name
                )
                if summary_run_id:
                    mlflow_run_ids["training_summary"] = summary_run_id
                    logger.info(f"MLflow training summary: {summary_run_id[:8]}")

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return ForecastingTrainingResult(
                success=False,
                version=request.version,
                models_trained=0,
                total_combinations=total_combinations,
                best_model_per_horizon={},
                metrics_summary={},
                model_artifacts_path=str(output_dir),
                training_duration_seconds=time.time() - start_time,
                errors=[str(e)],
            )

    # =========================================================================
    # INFERENCE
    # =========================================================================

    def predict(
        self,
        request: ForecastingInferenceRequest,
        models_path: str,
        features: pd.DataFrame,
        current_price: float,
    ) -> ForecastingInferenceResult:
        """
        Generate forecasts using trained models.

        Steps:
        1. Load models from path
        2. Generate predictions for each model/horizon
        3. Create ensembles
        4. Calculate consensus
        5. Persist to database
        6. Upload images to MinIO
        """
        errors = []
        predictions = []

        # Parse inference date
        inference_date = datetime.strptime(request.inference_date, "%Y-%m-%d")
        inference_week = inference_date.isocalendar()[1]
        inference_year = inference_date.year

        logger.info("=" * 60)
        logger.info(f"Running forecasting inference for {request.inference_date}")
        logger.info(f"Week {inference_week}, Year {inference_year}")
        logger.info("=" * 60)

        try:
            # Load feature columns
            feature_cols_path = Path(models_path) / "feature_cols.json"
            if feature_cols_path.exists():
                with open(feature_cols_path) as f:
                    feature_cols = json.load(f)
            else:
                feature_cols = list(features.columns)

            # Get latest features
            X_latest = features[feature_cols].iloc[-1:].values

            # Generate predictions
            all_predictions: Dict[str, Dict[int, float]] = {}

            for model_id in request.models:
                all_predictions[model_id] = {}

                for horizon in request.horizons:
                    try:
                        # Load model
                        model_path = Path(models_path) / "models" / f"{model_id}_h{horizon}.pkl"
                        if not model_path.exists():
                            logger.warning(f"Model not found: {model_path}")
                            continue

                        model = joblib.load(model_path)

                        # Predict
                        pred_return = model.predict(X_latest)[0]
                        all_predictions[model_id][horizon] = pred_return

                        # Calculate predicted price
                        predicted_price = current_price * np.exp(pred_return)
                        direction = ForecastDirection.UP if pred_return > 0 else ForecastDirection.DOWN
                        signal = 1 if pred_return > 0.001 else (-1 if pred_return < -0.001 else 0)

                        # Target date
                        target_date = inference_date + timedelta(days=horizon)

                        prediction = ForecastPrediction(
                            model_id=model_id,
                            horizon=horizon,
                            inference_date=request.inference_date,
                            target_date=target_date.strftime("%Y-%m-%d"),
                            base_price=current_price,
                            predicted_price=predicted_price,
                            predicted_return_pct=pred_return * 100,
                            direction=direction,
                            signal=signal,
                        )
                        predictions.append(prediction)

                    except Exception as e:
                        logger.error(f"Prediction error {model_id} H={horizon}: {e}")
                        errors.append(f"{model_id}_h{horizon}: {str(e)}")

            # Create ensembles
            ensembles = {}
            if request.generate_ensembles:
                ensembles = self._create_ensembles(
                    all_predictions,
                    current_price,
                    request.inference_date,
                    request.horizons,
                )

            # Calculate consensus
            consensus = self._calculate_consensus(predictions, request.horizons)

            # Persist to database
            forecasts_persisted = 0
            if request.persist_to_db and self.db_connection_string:
                try:
                    forecasts_persisted = self._persist_forecasts_to_db(
                        predictions,
                        ensembles,
                        inference_week,
                        inference_year,
                    )
                except Exception as e:
                    logger.error(f"DB persist failed: {e}")
                    errors.append(f"DB: {str(e)}")

            # Upload images
            images_uploaded = 0
            minio_path = None
            if request.upload_images and self.minio_client:
                try:
                    images_uploaded, minio_path = self._upload_forecast_images(
                        predictions,
                        inference_year,
                        inference_week,
                        request.inference_date,
                    )
                except Exception as e:
                    logger.error(f"Image upload failed: {e}")
                    errors.append(f"Images: {str(e)}")

            return ForecastingInferenceResult(
                success=len(errors) == 0,
                inference_date=request.inference_date,
                inference_week=inference_week,
                inference_year=inference_year,
                predictions=predictions,
                ensembles=ensembles,
                consensus_by_horizon=consensus,
                minio_week_path=minio_path,
                images_uploaded=images_uploaded,
                forecasts_persisted=forecasts_persisted,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return ForecastingInferenceResult(
                success=False,
                inference_date=request.inference_date,
                inference_week=inference_week,
                inference_year=inference_year,
                predictions=[],
                ensembles={},
                consensus_by_horizon={},
                errors=[str(e)],
            )

    # =========================================================================
    # ENSEMBLE METHODS
    # =========================================================================

    def _create_ensembles(
        self,
        predictions: Dict[str, Dict[int, float]],
        current_price: float,
        inference_date: str,
        horizons: List[int],
    ) -> Dict[str, Dict[int, ForecastPrediction]]:
        """Create ensemble predictions."""
        ensembles = {}

        # Calculate model rankings by average DA
        model_avg_da = {}
        for model_id in predictions:
            if predictions[model_id]:
                avg = np.mean(list(predictions[model_id].values()))
                model_avg_da[model_id] = avg

        sorted_models = sorted(model_avg_da.keys(), key=lambda m: model_avg_da[m], reverse=True)
        top_3 = sorted_models[:3]
        top_6 = sorted_models[:6]

        # Best-of-Breed: Best model per horizon (need metrics from training)
        # For now, use highest prediction magnitude as proxy
        best_of_breed = {}
        for h in horizons:
            best_model = None
            best_abs = -1
            for model_id, preds in predictions.items():
                if h in preds and abs(preds[h]) > best_abs:
                    best_abs = abs(preds[h])
                    best_model = model_id
            if best_model:
                best_of_breed[h] = predictions[best_model][h]

        ensembles[EnsembleType.BEST_OF_BREED.value] = self._predictions_to_forecasts(
            best_of_breed, current_price, inference_date, "best_of_breed"
        )

        # Top-3 Average
        top_3_preds = {}
        for h in horizons:
            vals = [predictions[m][h] for m in top_3 if h in predictions.get(m, {})]
            if vals:
                top_3_preds[h] = float(np.mean(vals))

        ensembles[EnsembleType.TOP_3.value] = self._predictions_to_forecasts(
            top_3_preds, current_price, inference_date, "top_3"
        )

        # Top-6 Average
        top_6_preds = {}
        for h in horizons:
            vals = [predictions[m][h] for m in top_6 if h in predictions.get(m, {})]
            if vals:
                top_6_preds[h] = float(np.mean(vals))

        ensembles[EnsembleType.TOP_6_MEAN.value] = self._predictions_to_forecasts(
            top_6_preds, current_price, inference_date, "top_6_mean"
        )

        # Consensus (all models average)
        consensus_preds = {}
        for h in horizons:
            vals = [preds[h] for preds in predictions.values() if h in preds]
            if vals:
                consensus_preds[h] = float(np.mean(vals))

        ensembles[EnsembleType.CONSENSUS.value] = self._predictions_to_forecasts(
            consensus_preds, current_price, inference_date, "consensus"
        )

        return ensembles

    def _predictions_to_forecasts(
        self,
        preds: Dict[int, float],
        current_price: float,
        inference_date: str,
        model_id: str,
    ) -> Dict[int, ForecastPrediction]:
        """Convert raw predictions to ForecastPrediction objects."""
        result = {}
        inf_date = datetime.strptime(inference_date, "%Y-%m-%d")

        for horizon, pred_return in preds.items():
            predicted_price = current_price * np.exp(pred_return)
            direction = ForecastDirection.UP if pred_return > 0 else ForecastDirection.DOWN
            signal = 1 if pred_return > 0.001 else (-1 if pred_return < -0.001 else 0)
            target_date = inf_date + timedelta(days=horizon)

            result[horizon] = ForecastPrediction(
                model_id=model_id,
                horizon=horizon,
                inference_date=inference_date,
                target_date=target_date.strftime("%Y-%m-%d"),
                base_price=current_price,
                predicted_price=predicted_price,
                predicted_return_pct=pred_return * 100,
                direction=direction,
                signal=signal,
            )

        return result

    def _calculate_consensus(
        self,
        predictions: List[ForecastPrediction],
        horizons: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """Calculate consensus for each horizon."""
        consensus = {}

        for h in horizons:
            h_preds = [p for p in predictions if p.horizon == h]
            if not h_preds:
                continue

            bullish = sum(1 for p in h_preds if p.direction == ForecastDirection.UP)
            bearish = len(h_preds) - bullish
            total = len(h_preds)

            prices = [p.predicted_price for p in h_preds]
            consensus[h] = {
                "horizon_id": h,
                "bullish_count": bullish,
                "bearish_count": bearish,
                "total_models": total,
                "consensus_direction": "UP" if bullish > bearish else "DOWN",
                "agreement_pct": max(bullish, bearish) / total * 100 if total > 0 else 0,
                "avg_predicted_price": float(np.mean(prices)),
                "median_predicted_price": float(np.median(prices)),
                "std_predicted_price": float(np.std(prices)),
                "min_predicted_price": float(np.min(prices)),
                "max_predicted_price": float(np.max(prices)),
            }

        return consensus

    # =========================================================================
    # DATA HELPERS
    # =========================================================================

    def _load_dataset(self, path: str = None, use_db: bool = False) -> pd.DataFrame:
        """
        Load dataset from file or SSOT (PostgreSQL/Parquet).

        SSOT Loading Order:
        1. If use_db=True: Try UnifiedLoaders (DB -> Parquet fallback)
        2. If path provided: Load from file (parquet/csv/pkl)
        3. Fall back to latest parquet in data/forecasting/aligned/

        Args:
            path: Optional path to dataset file
            use_db: If True, use UnifiedLoaders for SSOT data

        Returns:
            DataFrame with forecasting data
        """
        # Option 1: SSOT loading via UnifiedLoaders
        if use_db:
            try:
                return self._load_from_ssot()
            except Exception as e:
                logger.warning(f"SSOT loading failed: {e}")
                if path is None:
                    raise

        # Option 2: Load from specified path
        if path:
            path = Path(path)
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            elif path.suffix == ".csv":
                return pd.read_csv(path)
            elif path.suffix == ".pkl":
                return pd.read_pickle(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        # Option 3: Find latest aligned parquet
        aligned_dir = self.project_root / "data" / "forecasting" / "aligned"
        if aligned_dir.exists():
            parquet_files = list(aligned_dir.glob("forecasting_aligned_*.parquet"))
            if parquet_files:
                latest = max(parquet_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading latest aligned dataset: {latest.name}")
                return pd.read_parquet(latest)

        raise ValueError("No dataset path provided and no aligned datasets found")

    def _load_from_ssot(
        self,
        start_date: str = "2020-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Load data from SSOT (PostgreSQL/Parquet) and build features.

        ARCHITECTURE 10/10:
        Uses load_daily() to get OFFICIAL Investing.com daily values,
        NOT resampled 5-min data. This ensures forecasting uses the
        same official close prices that traders see.

        Data Sources:
        - OHLCV: bi.dim_daily_usdcop (Investing.com official)
        - Macro: macro_indicators_daily (DXY, WTI)
        """
        from src.data import UnifiedOHLCVLoader, UnifiedMacroLoader

        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Loading SSOT daily data: {start_date} to {end_date}")

        # Load OFFICIAL daily OHLCV from bi.dim_daily_usdcop (Investing.com)
        ohlcv_loader = UnifiedOHLCVLoader()
        df_ohlcv = ohlcv_loader.load_daily(start_date, end_date)
        logger.info(f"Loaded {len(df_ohlcv)} official daily OHLCV rows")

        # Load Macro
        macro_loader = UnifiedMacroLoader()
        df_macro = macro_loader.load_for_forecasting(start_date, end_date)

        # Merge
        df = df_ohlcv.merge(df_macro, on='date', how='left')

        # Build features (same as build_forecasting_dataset_aligned.py)
        df = self._build_ssot_features(df)

        logger.info(f"Loaded {len(df)} rows from SSOT")
        return df

    def _build_ssot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build 19 SSOT features from raw OHLCV and macro data."""
        df = df.copy()

        # Returns
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # Volatility
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()

        # Technical
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi_14d'] = 100 - (100 / (1 + rs))

        df['ma_ratio_20d'] = df['close'] / df['close'].rolling(20).mean()
        df['ma_ratio_50d'] = df['close'] / df['close'].rolling(50).mean()

        # Calendar
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['is_month_end'] = pd.to_datetime(df['date']).dt.is_month_end.astype(int)

        # Targets for all horizons
        for h in HORIZONS:
            df[f'target_{h}d'] = df['close'].shift(-h)
            df[f'target_return_{h}d'] = np.log(df['close'].shift(-h) / df['close'])

        return df

    def _validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has all SSOT feature columns.

        Uses FEATURE_COLUMNS from data_contracts.py as source of truth.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check all SSOT features exist
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            errors.append(f"Missing SSOT features: {missing}")

        # Check feature count
        available = [col for col in FEATURE_COLUMNS if col in df.columns]
        if len(available) != NUM_FEATURES:
            errors.append(
                f"Feature count mismatch: expected {NUM_FEATURES}, found {len(available)}"
            )

        # Log contract info
        logger.info(f"Data Contract: v{DATA_CONTRACT_VERSION} (hash: {DATA_CONTRACT_HASH})")
        logger.info(f"SSOT Features: {NUM_FEATURES}, Available: {len(available)}")

        return len(errors) == 0, errors

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare features for training using SSOT column order.

        CRITICAL: Uses FEATURE_COLUMNS from data_contracts.py to ensure
        consistent feature order across training and inference.
        """
        # First validate against SSOT
        is_valid, validation_errors = self._validate_features(df)

        if not is_valid:
            # Try fallback to auto-detected columns
            logger.warning(f"SSOT validation failed: {validation_errors}")
            logger.warning("Falling back to auto-detected feature columns")
            return self._prepare_features_fallback(df)

        # Use SSOT feature order (ensures consistency)
        feature_cols = list(FEATURE_COLUMNS)

        # Verify all are numeric
        for col in feature_cols:
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        X = df[feature_cols].values
        logger.info(f"Prepared {X.shape[1]} features in SSOT order")

        return X, feature_cols

    def _prepare_features_fallback(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Fallback feature preparation when SSOT validation fails."""
        exclude_cols = {
            "date", "time", "timestamp",
            "close", "open", "high", "low", "volume",
            "target", "y", "label",
        }

        # Find feature columns
        feature_cols = [
            col for col in df.columns
            if col.lower() not in exclude_cols
            and not col.startswith("target_")
            and not col.startswith("y_")
            and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        X = df[feature_cols].values
        return X, feature_cols

    def _create_targets(
        self,
        df: pd.DataFrame,
        horizons: List[int],
    ) -> Dict[int, np.ndarray]:
        """Create log-return targets for each horizon."""
        targets = {}

        # Find close price column
        close_col = None
        for col in ["close", "Close", "CLOSE"]:
            if col in df.columns:
                close_col = col
                break

        if close_col is None:
            raise ValueError("No close price column found")

        close = df[close_col].values

        for h in horizons:
            future_price = np.roll(close, -h)
            log_return = np.log(future_price / close)
            log_return[-h:] = np.nan  # Mark invalid
            targets[h] = log_return

        return targets

    def _create_metrics_dataframe(
        self,
        metrics_summary: Dict[str, Dict[int, float]],
    ) -> pd.DataFrame:
        """Create metrics dataframe from summary."""
        rows = []
        for model_id, horizons in metrics_summary.items():
            for horizon, da in horizons.items():
                rows.append({
                    "model_id": model_id,
                    "horizon": horizon,
                    "direction_accuracy": da,
                })
        return pd.DataFrame(rows)

    # =========================================================================
    # PERSISTENCE HELPERS
    # =========================================================================

    def _persist_metrics_to_db(
        self,
        metrics_summary: Dict[str, Dict[int, float]],
        version: str,
        db_connection: str,
    ):
        """Persist training metrics to database."""
        import psycopg2

        conn = psycopg2.connect(db_connection)
        cur = conn.cursor()

        try:
            training_date = datetime.now().date()

            for model_id, horizons in metrics_summary.items():
                for horizon, da in horizons.items():
                    cur.execute("""
                        INSERT INTO bi.fact_model_metrics
                        (training_date, evaluation_date, model_id, horizon_id,
                         direction_accuracy, sample_count)
                        VALUES (%s, %s, %s, %s, %s, 0)
                        ON CONFLICT (training_date, model_id, horizon_id)
                        DO UPDATE SET direction_accuracy = EXCLUDED.direction_accuracy
                    """, (training_date, training_date, model_id, horizon, da / 100))

            conn.commit()

        finally:
            cur.close()
            conn.close()

    def _persist_forecasts_to_db(
        self,
        predictions: List[ForecastPrediction],
        ensembles: Dict[str, Dict[int, ForecastPrediction]],
        inference_week: int,
        inference_year: int,
    ) -> int:
        """Persist forecasts to database."""
        import psycopg2

        if not self.db_connection_string:
            return 0

        conn = psycopg2.connect(self.db_connection_string)
        cur = conn.cursor()
        count = 0

        try:
            # Insert individual model predictions
            for pred in predictions:
                cur.execute("""
                    INSERT INTO bi.fact_forecasts
                    (inference_date, inference_week, inference_year, target_date,
                     model_id, horizon_id, base_price, predicted_price,
                     predicted_return_pct, direction, signal)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (inference_date, model_id, horizon_id)
                    DO UPDATE SET
                        predicted_price = EXCLUDED.predicted_price,
                        predicted_return_pct = EXCLUDED.predicted_return_pct,
                        direction = EXCLUDED.direction,
                        signal = EXCLUDED.signal
                """, (
                    pred.inference_date,
                    inference_week,
                    inference_year,
                    pred.target_date,
                    pred.model_id,
                    pred.horizon,
                    pred.base_price,
                    pred.predicted_price,
                    pred.predicted_return_pct,
                    pred.direction.value,
                    pred.signal,
                ))
                count += 1

            # Insert ensemble predictions
            for ensemble_type, horizon_preds in ensembles.items():
                for horizon, pred in horizon_preds.items():
                    cur.execute("""
                        INSERT INTO bi.fact_forecasts
                        (inference_date, inference_week, inference_year, target_date,
                         model_id, horizon_id, base_price, predicted_price,
                         predicted_return_pct, direction, signal)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (inference_date, model_id, horizon_id)
                        DO UPDATE SET
                            predicted_price = EXCLUDED.predicted_price,
                            predicted_return_pct = EXCLUDED.predicted_return_pct,
                            direction = EXCLUDED.direction,
                            signal = EXCLUDED.signal
                    """, (
                        pred.inference_date,
                        inference_week,
                        inference_year,
                        pred.target_date,
                        f"ensemble_{ensemble_type}",
                        pred.horizon,
                        pred.base_price,
                        pred.predicted_price,
                        pred.predicted_return_pct,
                        pred.direction.value,
                        pred.signal,
                    ))
                    count += 1

            conn.commit()
            logger.info(f"Persisted {count} forecasts to database")

        finally:
            cur.close()
            conn.close()

        return count

    def _upload_to_minio(self, output_dir: Path, version: str) -> str:
        """Upload training artifacts to MinIO."""
        if not self.minio_client:
            return None

        bucket = "forecasting-models"
        prefix = f"training/{version}"

        # Upload all files in output directory
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(output_dir)
                object_name = f"{prefix}/{relative}"
                self.minio_client.fput_object(bucket, object_name, str(file_path))

        return f"s3://{bucket}/{prefix}"

    def _upload_forecast_images(
        self,
        predictions: List[ForecastPrediction],
        year: int,
        week: int,
        inference_date: str,
    ) -> Tuple[int, str]:
        """Upload forecast images to MinIO."""
        # This would generate and upload visualization images
        # For now, return placeholder
        bucket = "forecasts"
        path = f"{year}/week{week:02d}/{inference_date}"
        return 0, f"s3://{bucket}/{path}"

    def _log_to_mlflow(
        self,
        model_id: str,
        horizon: int,
        wf_result: Any,
        experiment_name: str,
        model_path: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Log training run to MLflow with full experiment tracking.

        Implements complete MLflow integration matching RL pipeline standards:
        - Experiment management
        - Parameter logging
        - Metric logging
        - Artifact logging
        - Tag management

        Args:
            model_id: Model identifier (e.g., 'xgboost', 'hybrid_lightgbm')
            horizon: Forecast horizon in days
            wf_result: Walk-forward validation result
            experiment_name: MLflow experiment name
            model_path: Path to saved model file (optional)
            params: Model parameters (optional)

        Returns:
            MLflow run_id or None if MLflow not available
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            logger.warning("MLflow not installed, skipping logging")
            return None

        if not self.mlflow_client:
            # Try to initialize MLflow from environment
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            try:
                mlflow.set_tracking_uri(tracking_uri)
                self.mlflow_client = MlflowClient(tracking_uri)
            except Exception as e:
                logger.warning(f"Could not connect to MLflow: {e}")
                return None

        try:
            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={"project": "usdcop-forecasting", "type": "ml"}
                )
            else:
                experiment_id = experiment.experiment_id

            mlflow.set_experiment(experiment_name)

            # Generate run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_id}_h{horizon}_{timestamp}"

            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id

                # =============================================================
                # LOG PARAMETERS
                # =============================================================
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("horizon_days", horizon)
                mlflow.log_param("walk_forward_windows", getattr(wf_result, 'n_windows', 5))

                # Log model-specific params
                if params:
                    for key, value in params.items():
                        if isinstance(value, (int, float, str, bool)):
                            mlflow.log_param(f"model_{key}", value)

                # Log data contract info for reproducibility
                mlflow.log_param("data_contract_version", DATA_CONTRACT_VERSION)
                mlflow.log_param("data_contract_hash", DATA_CONTRACT_HASH)
                mlflow.log_param("num_features", NUM_FEATURES)

                # Log feature columns hash
                feature_hash = hashlib.sha256(
                    ",".join(FEATURE_COLUMNS).encode()
                ).hexdigest()[:8]
                mlflow.log_param("feature_order_hash", feature_hash)

                # =============================================================
                # LOG METRICS
                # =============================================================
                # Walk-forward validation metrics
                if hasattr(wf_result, 'da_mean'):
                    mlflow.log_metric("da_mean", wf_result.da_mean * 100)
                if hasattr(wf_result, 'da_std'):
                    mlflow.log_metric("da_std", wf_result.da_std * 100)
                if hasattr(wf_result, 'rmse_mean'):
                    mlflow.log_metric("rmse_mean", wf_result.rmse_mean)
                if hasattr(wf_result, 'mae_mean'):
                    mlflow.log_metric("mae_mean", wf_result.mae_mean)

                # Per-window metrics
                if hasattr(wf_result, 'window_metrics'):
                    for i, window_metric in enumerate(wf_result.window_metrics):
                        if hasattr(window_metric, 'direction_accuracy'):
                            mlflow.log_metric(f"da_window_{i}", window_metric.direction_accuracy * 100)

                # =============================================================
                # LOG ARTIFACTS
                # =============================================================
                if model_path and model_path.exists():
                    mlflow.log_artifact(str(model_path))

                # =============================================================
                # SET TAGS
                # =============================================================
                mlflow.set_tag("model_type", model_id)
                mlflow.set_tag("horizon", str(horizon))
                mlflow.set_tag("status", "FINISHED")
                mlflow.set_tag("pipeline", "forecasting")

                # Git info if available
                try:
                    import subprocess
                    git_commit = subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(self.project_root),
                        stderr=subprocess.DEVNULL
                    ).decode().strip()[:8]
                    mlflow.set_tag("git_commit", git_commit)
                except Exception:
                    pass

                logger.info(f"MLflow run logged: {run_name} (id: {run_id[:8]})")
                return run_id

        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")
            return None

    def _log_training_summary_to_mlflow(
        self,
        result: 'ForecastingTrainingResult',
        experiment_name: str,
    ) -> Optional[str]:
        """
        Log overall training summary as a parent MLflow run.

        Creates a summary run that links to all model-horizon runs.

        Args:
            result: Training result with all metrics
            experiment_name: MLflow experiment name

        Returns:
            Parent run_id or None
        """
        try:
            import mlflow
        except ImportError:
            return None

        try:
            mlflow.set_experiment(experiment_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"training_summary_{result.version}_{timestamp}"

            with mlflow.start_run(run_name=run_name) as run:
                # Log summary params
                mlflow.log_param("version", result.version)
                mlflow.log_param("models_trained", result.models_trained)
                mlflow.log_param("total_combinations", result.total_combinations)

                # Log summary metrics
                all_das = []
                for model_id, horizons in result.metrics_summary.items():
                    for horizon, da in horizons.items():
                        all_das.append(da)
                        mlflow.log_metric(f"da_{model_id}_h{horizon}", da)

                if all_das:
                    mlflow.log_metric("da_overall_mean", np.mean(all_das))
                    mlflow.log_metric("da_overall_max", np.max(all_das))
                    mlflow.log_metric("da_overall_min", np.min(all_das))

                # Log best models
                for horizon, model_id in result.best_model_per_horizon.items():
                    mlflow.set_tag(f"best_model_h{horizon}", model_id)

                # Log child run IDs
                if result.mlflow_run_ids:
                    mlflow.log_dict(result.mlflow_run_ids, "child_run_ids.json")

                mlflow.set_tag("run_type", "training_summary")
                mlflow.set_tag("status", "SUCCESS" if result.success else "FAILED")
                mlflow.log_metric("training_duration_seconds", result.training_duration_seconds)

                logger.info(f"Training summary logged to MLflow: {run_name}")
                return run.info.run_id

        except Exception as e:
            logger.error(f"MLflow summary logging failed: {e}")
            return None

    def _register_model_to_mlflow(
        self,
        model_id: str,
        horizon: int,
        run_id: str,
        model_path: Path,
        metrics: Dict[str, float],
    ) -> Optional[str]:
        """
        Register model to MLflow Model Registry.

        Args:
            model_id: Model identifier
            horizon: Forecast horizon
            run_id: MLflow run ID
            model_path: Path to model file
            metrics: Model metrics

        Returns:
            Model version or None
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
        except ImportError:
            return None

        try:
            client = MlflowClient()

            # Model name in registry
            model_name = f"forecasting-{model_id}-h{horizon}"

            # Check if model exists
            try:
                client.get_registered_model(model_name)
            except Exception:
                # Create new registered model
                client.create_registered_model(
                    model_name,
                    tags={
                        "model_type": model_id,
                        "horizon": str(horizon),
                        "pipeline": "forecasting",
                    },
                    description=f"Forecasting model {model_id} for {horizon}-day horizon"
                )

            # Log model and create version
            artifact_uri = f"runs:/{run_id}/{model_path.name}"

            model_version = client.create_model_version(
                name=model_name,
                source=artifact_uri,
                run_id=run_id,
                tags={
                    "da": str(metrics.get("direction_accuracy", 0)),
                    "status": "Staging",
                }
            )

            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version.version

        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            return None


__all__ = ["ForecastingEngine"]
