#!/bin/bash
# Setup MinIO buckets for data pipeline
# ======================================

echo "Setting up MinIO buckets..."

# Install mc (MinIO client) in a temporary container
docker run --rm \
  --network trading-network \
  --entrypoint /bin/sh \
  minio/mc:latest -c "
    # Configure MinIO client
    mc alias set minio http://trading-minio:9000 minioadmin minioadmin123
    
    # Create buckets for each pipeline layer
    mc mb --ignore-existing minio/00-l0-ds-usdcop-raw
    mc mb --ignore-existing minio/01-l1-ds-usdcop-standardized
    mc mb --ignore-existing minio/02-l2-ds-usdcop-prepared
    mc mb --ignore-existing minio/03-l3-ds-usdcop-featured
    mc mb --ignore-existing minio/04-l4-ds-usdcop-rlready
    mc mb --ignore-existing minio/05-l5-ds-usdcop-models
    mc mb --ignore-existing minio/mlflow
    
    # Set public read policy for MLflow bucket
    mc anonymous set download minio/mlflow
    
    # List buckets
    echo '============================='
    echo 'Created MinIO buckets:'
    mc ls minio/
    echo '============================='
"

echo "MinIO setup complete!"