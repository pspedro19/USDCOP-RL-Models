#!/bin/bash
# Deploy Optimized L5 Pipeline
# =============================

echo "==========================================="
echo "DEPLOYING OPTIMIZED L5 PIPELINE"
echo "==========================================="

# Set Airflow Variables
echo "Configuring Airflow Variables..."

# Training configuration
docker exec usdcop-airflow-webserver airflow variables set L5_TRAIN_TIMEOUT_HOURS 6
docker exec usdcop-airflow-webserver airflow variables set L5_TRAIN_PPO true
docker exec usdcop-airflow-webserver airflow variables set L5_TRAIN_DQN false

# Evaluation configuration  
docker exec usdcop-airflow-webserver airflow variables set L5_IN_LOOP_EVAL_EPISODES 15
docker exec usdcop-airflow-webserver airflow variables set L5_GATE_EVAL_EPISODES 100

# MLflow configuration
docker exec usdcop-airflow-webserver airflow variables set MLFLOW_TRACKING_URI "http://trading-mlflow:5000"

# MinIO configuration
docker exec usdcop-airflow-webserver airflow variables set MINIO_ENDPOINT "trading-minio:9000"
docker exec usdcop-airflow-webserver airflow variables set MINIO_ACCESS_KEY "minioadmin"
docker exec usdcop-airflow-webserver airflow variables set MINIO_SECRET_KEY "minioadmin123"

echo "✅ Variables configured"

# Copy DAG to container
echo "Deploying DAG..."
docker cp "airflow/dags/usdcop_m5__06_l5_serving.py" usdcop-airflow-webserver:/opt/airflow/dags/

echo "✅ DAG deployed"

# Unpause DAG
echo "Activating DAG..."
docker exec usdcop-airflow-webserver airflow dags unpause usdcop_m5__06_l5_serving

echo ""
echo "==========================================="
echo "DEPLOYMENT COMPLETE"
echo "==========================================="
echo ""
echo "Optimizations Applied:"
echo "✅ PPO with target_kl=0.01 for stability"
echo "✅ n_steps=1024, batch_size=256, n_epochs=10"
echo "✅ DummyVecEnv with parallel seeds [42,123,456]"
echo "✅ In-loop eval: 15 episodes"
echo "✅ Gate eval: 100 episodes with Sortino/MaxDD/Calmar"
echo "✅ DQN optional (disabled by default)"
echo ""
echo "To trigger pipeline:"
echo "docker exec usdcop-airflow-webserver airflow dags trigger usdcop_m5__06_l5_serving"
echo ""
echo "To enable DQN:"
echo "docker exec usdcop-airflow-webserver airflow variables set L5_TRAIN_DQN true"