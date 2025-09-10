#!/bin/bash
# Check which MinIO buckets have data

echo "Checking MinIO bucket contents..."
echo "================================="

docker run --rm --network trading-network --entrypoint sh minio/mc:latest -c "
mc alias set minio http://trading-minio:9000 minioadmin minioadmin123 > /dev/null 2>&1

echo 'Buckets with data:'
echo '------------------'

# Check L0 buckets
echo -n '00-l0-ds-usdcop-raw: '
mc ls minio/00-l0-ds-usdcop-raw/ --recursive | wc -l

echo -n '00-raw-usdcop-marketdata: '
mc ls minio/00-raw-usdcop-marketdata/ --recursive | wc -l

# Check L1 buckets
echo -n '01-l1-ds-usdcop-standardize: '
mc ls minio/01-l1-ds-usdcop-standardize/ --recursive | wc -l

echo -n '01-l1-ds-usdcop-standardized: '
mc ls minio/01-l1-ds-usdcop-standardized/ --recursive | wc -l

# Check L2 buckets
echo -n '02-l2-ds-usdcop-prepare: '
mc ls minio/02-l2-ds-usdcop-prepare/ --recursive | wc -l

echo -n '02-l2-ds-usdcop-prepared: '
mc ls minio/02-l2-ds-usdcop-prepared/ --recursive | wc -l

# Check L3 buckets
echo -n '03-l3-ds-usdcop-feature: '
mc ls minio/03-l3-ds-usdcop-feature/ --recursive | wc -l

echo -n '03-l3-ds-usdcop-featured: '
mc ls minio/03-l3-ds-usdcop-featured/ --recursive | wc -l

# Check L4 bucket
echo -n '04-l4-ds-usdcop-rlready: '
mc ls minio/04-l4-ds-usdcop-rlready/ --recursive | wc -l

# Check L5 buckets
echo -n '05-l5-ds-usdcop-models: '
mc ls minio/05-l5-ds-usdcop-models/ --recursive | wc -l

echo -n '05-l5-ds-usdcop-serving: '
mc ls minio/05-l5-ds-usdcop-serving/ --recursive | wc -l

# Check other buckets
echo -n 'airflow: '
mc ls minio/airflow/ --recursive | wc -l

echo -n 'mlflow: '
mc ls minio/mlflow/ --recursive | wc -l

echo -n 'mlflow-artifacts: '
mc ls minio/mlflow-artifacts/ --recursive | wc -l

echo -n 'trading-data: '
mc ls minio/trading-data/ --recursive | wc -l

echo -n 'trading-models: '
mc ls minio/trading-models/ --recursive | wc -l
"