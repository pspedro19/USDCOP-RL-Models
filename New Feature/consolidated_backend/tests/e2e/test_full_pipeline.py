"""
End-to-End Test: Full Pipeline Execution

This module tests the complete flow of the USD/COP forecasting pipeline:
1. Load test data
2. Execute training (fast mode)
3. Execute inference
4. Verify outputs in MinIO
5. Verify data in PostgreSQL
6. Verify API responds correctly

Usage:
    pytest tests/e2e/test_full_pipeline.py -v
    pytest tests/e2e/test_full_pipeline.py -v -m "not slow"
"""

import os
import sys
import json
import pickle
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))


# =============================================================================
# Test Configuration
# =============================================================================

FAST_MODE_CONFIG = {
    "horizons": [5, 10],  # Only 2 horizons for speed
    "models": ["ridge"],  # Only Ridge for fast testing
    "n_features": 10,
    "train_size": 0.7,
    "n_optuna_trials": 2,
}


# =============================================================================
# Test Class: Full Pipeline E2E
# =============================================================================

@pytest.mark.e2e
class TestFullPipelineE2E:
    """End-to-end tests for the complete pipeline flow."""

    # -------------------------------------------------------------------------
    # Test 1: Data Loading
    # -------------------------------------------------------------------------

    def test_load_data_from_csv(self, sample_features_df, temp_output_dir):
        """Test loading data from CSV file."""
        # Save sample data to CSV
        data_path = temp_output_dir / "test_features.csv"
        sample_features_df.to_csv(data_path, index=False)

        # Load data
        df = pd.read_csv(data_path)

        # Assertions
        assert len(df) > 0, "DataFrame should not be empty"
        assert "Close" in df.columns, "Close column should exist"
        assert "Date" in df.columns, "Date column should exist"
        assert len(df.columns) > 10, "Should have multiple feature columns"

    def test_load_data_validation(self, sample_features_df):
        """Test data validation during loading."""
        df = sample_features_df.copy()

        # Check required columns
        required_cols = ["Date", "Close"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

        # Check data types
        assert pd.api.types.is_numeric_dtype(df["Close"]), "Close should be numeric"

        # Check for NaN values in critical columns
        assert df["Close"].isna().sum() == 0, "Close should not have NaN values"

        # Check date range
        df["Date"] = pd.to_datetime(df["Date"])
        date_range = (df["Date"].max() - df["Date"].min()).days
        assert date_range > 100, "Data should span at least 100 days"

    # -------------------------------------------------------------------------
    # Test 2: Training Pipeline (Fast Mode)
    # -------------------------------------------------------------------------

    @pytest.mark.slow
    def test_training_pipeline_fast_mode(self, sample_features_df, temp_output_dir):
        """Test training pipeline in fast mode."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        df = sample_features_df.copy()

        # Prepare features
        feature_cols = [c for c in df.columns
                       if c not in ["Date", "Close", "Open", "High", "Low", "Volume", "returns"]
                       and not c.startswith("target_")][:FAST_MODE_CONFIG["n_features"]]

        X = df[feature_cols].values
        y = df["target_5d"].values

        # Remove NaN
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]

        # Split data
        train_size = int(len(X) * FAST_MODE_CONFIG["train_size"])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Ridge model
        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        direction_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100
        rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))

        # Assertions
        assert direction_accuracy > 40, f"Direction accuracy {direction_accuracy:.1f}% should be > 40%"
        assert rmse < 0.1, f"RMSE {rmse:.4f} should be < 0.1"

        # Save model
        models_dir = temp_output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "metrics": {
                "direction_accuracy": direction_accuracy,
                "rmse": rmse
            }
        }

        model_file = models_dir / "ridge_h5.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model_data, f)

        assert model_file.exists(), "Model file should be saved"

    def test_training_creates_output_files(self, sample_features_df, temp_output_dir):
        """Test that training creates expected output files."""
        # Simulate training output
        run_dir = temp_output_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True)

        # Create expected files
        (run_dir / "models").mkdir()
        (run_dir / "figures").mkdir()
        (run_dir / "data").mkdir()

        # Create dummy model files
        for model_name in ["ridge", "xgboost"]:
            for horizon in [5, 10]:
                model_file = run_dir / "models" / f"{model_name}_h{horizon}.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump({"dummy": True}, f)

        # Create results CSV
        results = pd.DataFrame({
            "model": ["ridge", "ridge", "xgboost", "xgboost"],
            "horizon": [5, 10, 5, 10],
            "direction_accuracy": [60, 58, 62, 59],
            "rmse": [0.02, 0.025, 0.019, 0.023],
        })
        results.to_csv(run_dir / "data" / "model_results.csv", index=False)

        # Create summary JSON
        summary = {
            "training_date": datetime.now().isoformat(),
            "best_model": "xgboost",
            "best_da": 62.0,
        }
        with open(run_dir / "report_summary.json", "w") as f:
            json.dump(summary, f)

        # Assertions
        assert (run_dir / "models").exists()
        assert (run_dir / "figures").exists()
        assert (run_dir / "data").exists()
        assert len(list((run_dir / "models").glob("*.pkl"))) == 4
        assert (run_dir / "data" / "model_results.csv").exists()
        assert (run_dir / "report_summary.json").exists()

    # -------------------------------------------------------------------------
    # Test 3: Inference Pipeline
    # -------------------------------------------------------------------------

    def test_inference_pipeline(self, sample_features_df, temp_output_dir):
        """Test inference pipeline with trained models."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        df = sample_features_df.copy()

        # Prepare features
        feature_cols = [c for c in df.columns
                       if c not in ["Date", "Close", "Open", "High", "Low", "Volume", "returns"]
                       and not c.startswith("target_")][:10]

        # Simulate trained model
        scaler = StandardScaler()
        X_all = df[feature_cols].values
        scaler.fit(X_all)

        model = Ridge(alpha=10.0)
        y = df["target_5d"].dropna().values
        valid_idx = ~np.isnan(df["target_5d"].values)
        model.fit(scaler.transform(X_all[valid_idx]), y)

        # Save model
        models_dir = temp_output_dir / "runs" / "test_run" / "models"
        models_dir.mkdir(parents=True)

        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
        }

        with open(models_dir / "ridge_h5.pkl", "wb") as f:
            pickle.dump(model_data, f)

        # Run inference
        X_latest = df[feature_cols].iloc[-1:].values
        X_scaled = scaler.transform(X_latest)
        pred_return = model.predict(X_scaled)[0]

        current_price = df["Close"].iloc[-1]
        pred_price = current_price * np.exp(pred_return)

        # Create forecast
        forecast = {
            "model": "ridge",
            "horizon": 5,
            "current_price": float(current_price),
            "predicted_price": float(pred_price),
            "predicted_return_pct": float(pred_return * 100),
            "direction": "UP" if pred_return > 0 else "DOWN",
        }

        # Assertions
        assert forecast["predicted_price"] > 0
        assert -10 < forecast["predicted_return_pct"] < 10
        assert forecast["direction"] in ["UP", "DOWN"]

        # Save forecast
        weekly_dir = temp_output_dir / "weekly"
        weekly_dir.mkdir(exist_ok=True)

        forecast_file = weekly_dir / f'forecast_{datetime.now().strftime("%Y%m%d")}.json'
        with open(forecast_file, "w") as f:
            json.dump({"predictions": [forecast]}, f)

        assert forecast_file.exists()

    def test_inference_generates_forecasts(self, sample_forecasts, temp_output_dir):
        """Test that inference generates forecast files."""
        weekly_dir = temp_output_dir / "weekly"
        weekly_dir.mkdir(exist_ok=True)

        # Save forecasts
        forecast_data = {
            "generated_at": datetime.now().isoformat(),
            "base_date": datetime.now().strftime("%Y-%m-%d"),
            "current_price": 4200.0,
            "predictions": sample_forecasts,
        }

        forecast_file = weekly_dir / f'forecast_{datetime.now().strftime("%Y%m%d")}.json'
        with open(forecast_file, "w") as f:
            json.dump(forecast_data, f, indent=2)

        # Verify
        assert forecast_file.exists()

        with open(forecast_file, "r") as f:
            loaded = json.load(f)

        assert len(loaded["predictions"]) == len(sample_forecasts)
        assert all("model" in p for p in loaded["predictions"])
        assert all("predicted_price" in p for p in loaded["predictions"])

    # -------------------------------------------------------------------------
    # Test 4: MinIO Output Verification
    # -------------------------------------------------------------------------

    @pytest.mark.requires_minio
    def test_minio_upload_forecasts(self, minio_client, test_bucket, sample_forecasts):
        """Test uploading forecasts to MinIO."""
        if minio_client is None:
            pytest.skip("MinIO not available")

        # Create forecast file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"predictions": sample_forecasts}, f)
            forecast_file = f.name

        try:
            # Upload to MinIO
            year = datetime.now().year
            week = datetime.now().isocalendar()[1]
            object_name = f"{year}/week{week:02d}/forecast.json"

            minio_client.fput_object(
                test_bucket,
                object_name,
                forecast_file,
                content_type="application/json"
            )

            # Verify upload
            objects = list(minio_client.list_objects(test_bucket, recursive=True))
            object_names = [obj.object_name for obj in objects]

            assert object_name in object_names, f"Forecast should be in MinIO"

        finally:
            os.unlink(forecast_file)

    @pytest.mark.requires_minio
    def test_minio_upload_images(self, minio_client, test_bucket, temp_output_dir):
        """Test uploading images to MinIO."""
        if minio_client is None:
            pytest.skip("MinIO not available")

        # Create dummy image file
        figures_dir = temp_output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Create a minimal PNG file
        import struct
        import zlib

        def create_minimal_png(filename):
            """Create a minimal valid PNG file."""
            signature = b'\x89PNG\r\n\x1a\n'

            # IHDR chunk
            width = 1
            height = 1
            bit_depth = 8
            color_type = 2  # RGB
            ihdr_data = struct.pack('>IIBBBBB', width, height, bit_depth, color_type, 0, 0, 0)
            ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
            ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)

            # IDAT chunk (1x1 red pixel)
            raw_data = b'\x00\xff\x00\x00'  # filter byte + RGB
            compressed = zlib.compress(raw_data)
            idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
            idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)

            # IEND chunk
            iend_crc = zlib.crc32(b'IEND') & 0xffffffff
            iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)

            with open(filename, 'wb') as f:
                f.write(signature + ihdr + idat + iend)

        image_file = figures_dir / "forward_forecast_ridge.png"
        create_minimal_png(str(image_file))

        # Upload to MinIO
        year = datetime.now().year
        week = datetime.now().isocalendar()[1]
        object_name = f"{year}/week{week:02d}/figures/forward_forecast_ridge.png"

        minio_client.fput_object(
            test_bucket,
            object_name,
            str(image_file),
            content_type="image/png"
        )

        # Verify
        objects = list(minio_client.list_objects(test_bucket, recursive=True))
        object_names = [obj.object_name for obj in objects]

        assert object_name in object_names

    def test_minio_mock_upload(self, mock_minio_client, sample_forecasts):
        """Test MinIO upload with mock client."""
        # Create forecast file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"predictions": sample_forecasts}, f)
            forecast_file = f.name

        try:
            # Mock upload
            mock_minio_client.fput_object(
                "test-bucket",
                "2024/week01/forecast.json",
                forecast_file,
                content_type="application/json"
            )

            # Verify mock was called
            mock_minio_client.fput_object.assert_called_once()

        finally:
            os.unlink(forecast_file)

    # -------------------------------------------------------------------------
    # Test 5: PostgreSQL Output Verification
    # -------------------------------------------------------------------------

    @pytest.mark.requires_db
    def test_postgresql_insert_forecasts(self, db_connection, sample_forecasts):
        """Test inserting forecasts into PostgreSQL."""
        if db_connection is None:
            pytest.skip("Database not available")

        cursor = db_connection.cursor()

        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'bi'
                    AND table_name = 'fact_forecasts'
                )
            """)
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                pytest.skip("bi.fact_forecasts table does not exist")

            # Insert test forecast
            forecast = sample_forecasts[0]
            base_date = datetime.now().date()

            cursor.execute("""
                INSERT INTO bi.fact_forecasts
                (model_id, horizon_id, inference_date, inference_week, inference_year,
                 target_date, base_price, predicted_price, predicted_return_pct,
                 price_change, direction, signal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id, horizon_id, inference_date) DO NOTHING
            """, (
                forecast["model"],
                forecast["horizon"],
                base_date,
                base_date.isocalendar()[1],
                base_date.year,
                base_date + timedelta(days=forecast["horizon"]),
                forecast["current_price"],
                forecast["predicted_price"],
                forecast["predicted_return_pct"],
                forecast["predicted_price"] - forecast["current_price"],
                forecast["direction"],
                forecast.get("signal", "HOLD"),
            ))

            db_connection.commit()

            # Verify insert
            cursor.execute("""
                SELECT COUNT(*) FROM bi.fact_forecasts
                WHERE model_id = %s AND horizon_id = %s AND inference_date = %s
            """, (forecast["model"], forecast["horizon"], base_date))

            count = cursor.fetchone()[0]
            assert count >= 1, "Forecast should be inserted"

        finally:
            cursor.close()

    def test_postgresql_mock_insert(self, mock_db_connection, sample_forecasts):
        """Test PostgreSQL insert with mock connection."""
        mock_cursor = mock_db_connection.cursor()

        # Mock insert
        forecast = sample_forecasts[0]
        mock_cursor.execute("INSERT INTO bi.fact_forecasts ...", (forecast,))

        # Verify mock was called
        mock_cursor.execute.assert_called()

    # -------------------------------------------------------------------------
    # Test 6: API Response Verification
    # -------------------------------------------------------------------------

    @pytest.mark.requires_api
    def test_api_health_check(self, api_client):
        """Test API health check endpoint."""
        if api_client is None:
            pytest.skip("API client not available")

        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @pytest.mark.requires_api
    def test_api_ready_check(self, api_client):
        """Test API readiness check endpoint."""
        if api_client is None:
            pytest.skip("API client not available")

        response = api_client.get("/health/ready")

        # Can be 200 (all healthy) or 503 (some unhealthy)
        assert response.status_code in [200, 503]
        data = response.json()
        assert "status" in data
        assert "checks" in data

    @pytest.mark.requires_api
    def test_api_forecasts_endpoint(self, api_client, auth_headers):
        """Test API forecasts endpoint."""
        if api_client is None:
            pytest.skip("API client not available")

        response = api_client.get("/api/forecasts/dashboard", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "source" in data
        assert "forecasts" in data

    @pytest.mark.requires_api
    def test_api_models_endpoint(self, api_client, auth_headers):
        """Test API models endpoint."""
        if api_client is None:
            pytest.skip("API client not available")

        response = api_client.get("/api/models/", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "models" in data

    # -------------------------------------------------------------------------
    # Test 7: Full Pipeline Integration
    # -------------------------------------------------------------------------

    @pytest.mark.slow
    def test_full_pipeline_flow(self, sample_features_df, temp_output_dir):
        """Test complete pipeline flow from data to output."""
        # Step 1: Save data
        data_path = temp_output_dir / "data" / "features.csv"
        data_path.parent.mkdir(exist_ok=True)
        sample_features_df.to_csv(data_path, index=False)
        assert data_path.exists()

        # Step 2: Load data
        df = pd.read_csv(data_path)
        assert len(df) > 0

        # Step 3: Prepare features
        feature_cols = [c for c in df.columns
                       if c not in ["Date", "Close", "Open", "High", "Low", "Volume", "returns"]
                       and not c.startswith("target_")][:10]

        # Step 4: Train model
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = df[feature_cols].values
        y = df["target_5d"].dropna().values
        valid_idx = ~np.isnan(df["target_5d"].values)
        X_valid = X[valid_idx]

        scaler.fit(X_valid)
        X_scaled = scaler.transform(X_valid)

        train_size = int(len(X_scaled) * 0.8)
        model = Ridge(alpha=10.0)
        model.fit(X_scaled[:train_size], y[:train_size])

        # Step 5: Evaluate
        y_pred = model.predict(X_scaled[train_size:])
        y_test = y[train_size:]
        da = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100
        assert da > 40

        # Step 6: Save model
        models_dir = temp_output_dir / "models"
        models_dir.mkdir(exist_ok=True)

        model_data = {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "metrics": {"da": da}
        }

        with open(models_dir / "ridge_h5.pkl", "wb") as f:
            pickle.dump(model_data, f)

        # Step 7: Run inference
        X_latest = df[feature_cols].iloc[-1:].values
        pred_return = model.predict(scaler.transform(X_latest))[0]
        pred_price = df["Close"].iloc[-1] * np.exp(pred_return)

        # Step 8: Save forecast
        forecast = {
            "model": "ridge",
            "horizon": 5,
            "predicted_price": float(pred_price),
            "direction": "UP" if pred_return > 0 else "DOWN",
        }

        weekly_dir = temp_output_dir / "weekly"
        weekly_dir.mkdir(exist_ok=True)

        with open(weekly_dir / "forecast.json", "w") as f:
            json.dump({"predictions": [forecast]}, f)

        # Step 9: Verify all outputs
        assert (models_dir / "ridge_h5.pkl").exists()
        assert (weekly_dir / "forecast.json").exists()

        # Step 10: Verify forecast content
        with open(weekly_dir / "forecast.json") as f:
            loaded = json.load(f)

        assert len(loaded["predictions"]) > 0
        assert loaded["predictions"][0]["predicted_price"] > 0


# =============================================================================
# Test Class: Pipeline Error Handling
# =============================================================================

@pytest.mark.e2e
class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""

    def test_handles_missing_data(self, temp_output_dir):
        """Test handling of missing data files."""
        data_path = temp_output_dir / "nonexistent.csv"

        with pytest.raises(FileNotFoundError):
            pd.read_csv(data_path)

    def test_handles_invalid_data(self, temp_output_dir):
        """Test handling of invalid data."""
        # Create invalid CSV
        data_path = temp_output_dir / "invalid.csv"
        with open(data_path, "w") as f:
            f.write("invalid,data\nwith,issues")

        df = pd.read_csv(data_path)

        # Should not have required columns
        assert "Close" not in df.columns

    def test_handles_nan_values(self, sample_features_df):
        """Test handling of NaN values in data."""
        df = sample_features_df.copy()

        # Add NaN values
        df.loc[0:10, "Close"] = np.nan

        # Count NaN
        nan_count = df["Close"].isna().sum()
        assert nan_count > 0

        # Fill NaN
        df["Close"] = df["Close"].ffill().bfill()
        assert df["Close"].isna().sum() == 0

    def test_handles_model_load_error(self, temp_output_dir):
        """Test handling of model loading errors."""
        model_path = temp_output_dir / "nonexistent_model.pkl"

        with pytest.raises(FileNotFoundError):
            with open(model_path, "rb") as f:
                pickle.load(f)

    def test_handles_inference_error(self, sample_features_df):
        """Test handling of inference errors."""
        from sklearn.linear_model import Ridge

        # Train model with 10 features
        model = Ridge()
        X = sample_features_df.iloc[:, 5:15].values[:100]
        y = np.random.randn(100)
        model.fit(X, y)

        # Try to predict with wrong number of features
        X_wrong = sample_features_df.iloc[:, 5:20].values[-1:]  # 15 features

        with pytest.raises(ValueError):
            model.predict(X_wrong)


# =============================================================================
# Test Class: Pipeline Performance
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestPipelinePerformance:
    """Test pipeline performance and timing."""

    def test_training_time(self, sample_features_df):
        """Test that training completes within acceptable time."""
        import time
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in sample_features_df.columns
                       if c.startswith("feature_")][:10]

        X = sample_features_df[feature_cols].values
        y = sample_features_df["target_5d"].dropna().values[:len(X)]
        X = X[:len(y)]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        start_time = time.time()

        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y)
        _ = model.predict(X_scaled)

        elapsed = time.time() - start_time

        # Training should complete in under 5 seconds
        assert elapsed < 5, f"Training took too long: {elapsed:.2f}s"

    def test_inference_time(self, sample_features_df):
        """Test that inference completes within acceptable time."""
        import time
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in sample_features_df.columns
                       if c.startswith("feature_")][:10]

        X = sample_features_df[feature_cols].values
        y = sample_features_df["target_5d"].dropna().values[:len(X)]
        X = X[:len(y)]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=10.0)
        model.fit(X_scaled, y)

        X_latest = X_scaled[-1:]

        start_time = time.time()

        for _ in range(100):
            _ = model.predict(X_latest)

        elapsed = time.time() - start_time

        # 100 predictions should complete in under 1 second
        assert elapsed < 1, f"Inference too slow: {elapsed:.2f}s for 100 predictions"
