#!/bin/bash
# Clean up duplicate MinIO buckets

echo "Cleaning up duplicate MinIO buckets..."
echo "======================================"

docker run --rm --network trading-network --entrypoint sh minio/mc:latest -c "
mc alias set minio http://trading-minio:9000 minioadmin minioadmin123 > /dev/null 2>&1

echo 'Deleting empty duplicate buckets...'

# Delete empty L0 duplicate (keep 00-raw-usdcop-marketdata with 8 files)
echo 'Removing 00-l0-ds-usdcop-raw (empty)'
mc rb --force minio/00-l0-ds-usdcop-raw

# Delete empty L1 duplicate (keep 01-l1-ds-usdcop-standardize with 30 files)
echo 'Removing 01-l1-ds-usdcop-standardized (empty)'
mc rb --force minio/01-l1-ds-usdcop-standardized

# Delete empty L2 duplicate (keep 02-l2-ds-usdcop-prepare with 24 files)
echo 'Removing 02-l2-ds-usdcop-prepared (empty)'
mc rb --force minio/02-l2-ds-usdcop-prepared

# Delete empty L3 duplicate (keep 03-l3-ds-usdcop-feature with 37 files)
echo 'Removing 03-l3-ds-usdcop-featured (empty)'
mc rb --force minio/03-l3-ds-usdcop-featured

# Delete empty L5 duplicate (keep 05-l5-ds-usdcop-serving with 3 files)
echo 'Removing 05-l5-ds-usdcop-models (empty)'
mc rb --force minio/05-l5-ds-usdcop-models

# Delete other empty buckets
echo 'Removing mlflow-artifacts (empty - using mlflow bucket instead)'
mc rb --force minio/mlflow-artifacts

echo 'Removing trading-data (empty)'
mc rb --force minio/trading-data

echo 'Removing trading-models (empty)'
mc rb --force minio/trading-models

echo 'Removing trading-reports (empty)'
mc rb --force minio/trading-reports

# Delete 99-common buckets (empty)
echo 'Removing 99-common-trading-backups (empty)'
mc rb --force minio/99-common-trading-backups

echo 'Removing 99-common-trading-models (empty)'
mc rb --force minio/99-common-trading-models

echo 'Removing 99-common-trading-reports (empty)'
mc rb --force minio/99-common-trading-reports

echo ''
echo 'Final bucket list:'
echo '------------------'
mc ls minio/
"