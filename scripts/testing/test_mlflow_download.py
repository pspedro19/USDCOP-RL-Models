#!/usr/bin/env python3
"""
Test script to validate MLflow model download fix for evaluate_production_gates
"""

import os
import sys
import mlflow
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mlflow_download(mlflow_run_id, seed=42):
    """Test downloading model from MLflow with correct artifact path"""
    
    # Set MLflow tracking URI if needed
    mlflow.set_tracking_uri("http://trading-mlflow:5000")
    
    # Create temp directory for test
    eval_temp_dir = tempfile.mkdtemp(prefix=f"test_l5_eval_seed_{seed}_")
    logger.info(f"Created temp directory: {eval_temp_dir}")
    
    try:
        # Test 1: Try the CORRECTED artifact path (models/ with plural)
        model_filename = f"ppo_seed_{seed}_final.zip"
        model_path = os.path.join(eval_temp_dir, model_filename)
        
        logger.info(f"Testing download from artifact path: models/{model_filename}")
        try:
            mlflow.artifacts.download_artifacts(
                run_id=mlflow_run_id,
                artifact_path=f"models/{model_filename}",
                dst_path=eval_temp_dir
            )
            
            downloaded_model = os.path.join(eval_temp_dir, "models", model_filename)
            if os.path.exists(downloaded_model):
                logger.info(f"✅ SUCCESS: Model found at {downloaded_model}")
                file_size = os.path.getsize(downloaded_model) / (1024 * 1024)  # Convert to MB
                logger.info(f"   File size: {file_size:.2f} MB")
                return True
            else:
                logger.error(f"❌ Model not found at expected location: {downloaded_model}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to download with corrected path: {e}")
            
            # Test 2: List all artifacts to debug
            logger.info("Listing all artifacts in the MLflow run:")
            try:
                artifacts = mlflow.artifacts.list_artifacts(run_id=mlflow_run_id)
                for artifact in artifacts:
                    logger.info(f"  - {artifact.path} (is_dir: {artifact.is_dir})")
                    if artifact.is_dir:
                        # List contents of directories
                        sub_artifacts = mlflow.artifacts.list_artifacts(
                            run_id=mlflow_run_id, 
                            artifact_path=artifact.path
                        )
                        for sub in sub_artifacts:
                            logger.info(f"    - {artifact.path}/{sub.path}")
            except Exception as list_error:
                logger.error(f"Failed to list artifacts: {list_error}")
            
            return False
            
    finally:
        # Clean up temp directory
        import shutil
        if os.path.exists(eval_temp_dir):
            shutil.rmtree(eval_temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {eval_temp_dir}")

def main():
    # Use the MLflow run ID from the error log
    mlflow_run_id = "7a388035e9184c46a26a5978a33a7b40"
    seed = 42
    
    logger.info("=" * 80)
    logger.info("TESTING MLFLOW MODEL DOWNLOAD FIX")
    logger.info("=" * 80)
    logger.info(f"MLflow Run ID: {mlflow_run_id}")
    logger.info(f"Seed: {seed}")
    logger.info("")
    
    success = test_mlflow_download(mlflow_run_id, seed)
    
    if success:
        logger.info("")
        logger.info("✅ Test PASSED: The fix correctly downloads the model from MLflow")
        logger.info("The evaluate_production_gates task should now work correctly.")
    else:
        logger.info("")
        logger.info("❌ Test FAILED: Model download issue persists")
        logger.info("Please check the MLflow run and ensure the model was properly uploaded during training.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())