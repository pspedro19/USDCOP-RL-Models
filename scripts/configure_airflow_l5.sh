#!/bin/bash
"""
Configure Airflow Variables for L5 Pipeline
============================================
Run this script to set up all required Airflow Variables
"""

echo "=========================================="
echo "Configuring Airflow Variables for L5"
echo "=========================================="

# Set timeout configuration
airflow variables set L5_TRAIN_TIMEOUT_HOURS 12
echo "✅ Set L5_TRAIN_TIMEOUT_HOURS = 12 hours"

airflow variables set L5_TRAIN_SAFETY_MARGIN_SEC 300
echo "✅ Set L5_TRAIN_SAFETY_MARGIN_SEC = 300 seconds (5 min margin)"

# Set training configuration
airflow variables set L5_FORCE_DUMMY_VEC true
echo "✅ Set L5_FORCE_DUMMY_VEC = true (avoid daemon issues in Airflow)"

airflow variables set L5_TRAIN_CHUNK_STEPS 50000
echo "✅ Set L5_TRAIN_CHUNK_STEPS = 50000 (train in chunks)"

airflow variables set L5_CHECKPOINT_FREQ 50000
echo "✅ Set L5_CHECKPOINT_FREQ = 50000"

# Debug mode (set to true for testing)
airflow variables set L5_DEBUG_MODE false
echo "✅ Set L5_DEBUG_MODE = false (set to true for quick testing)"

# MLflow configuration
airflow variables set MLFLOW_TRACKING_URI http://trading-mlflow:5000
echo "✅ Set MLFLOW_TRACKING_URI = http://trading-mlflow:5000"

echo ""
echo "=========================================="
echo "Configuration Complete!"
echo "=========================================="
echo ""
echo "Current settings:"
airflow variables list | grep L5_

echo ""
echo "To enable debug mode (quick testing):"
echo "  airflow variables set L5_DEBUG_MODE true"
echo ""
echo "To trigger the DAG:"
echo "  airflow dags trigger usdcop_l5_serving_fixed"