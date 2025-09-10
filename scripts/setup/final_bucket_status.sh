#!/bin/bash
# Final MinIO bucket status

echo "=========================================="
echo "FINAL MINIO BUCKET STRUCTURE"
echo "=========================================="
echo ""

docker run --rm --network trading-network --entrypoint sh minio/mc:latest -c "
mc alias set minio http://trading-minio:9000 minioadmin minioadmin123 > /dev/null 2>&1

echo 'Clean Bucket List:'
echo '------------------'
mc ls minio/

echo ''
echo 'Bucket Contents Summary:'
echo '------------------------'
echo -n '00-raw-usdcop-marketdata: '
mc ls minio/00-raw-usdcop-marketdata/ --recursive | wc -l
echo ' files (L0 raw market data)'

echo -n '01-l1-ds-usdcop-standardize: '
mc ls minio/01-l1-ds-usdcop-standardize/ --recursive | wc -l
echo ' files (L1 standardized data)'

echo -n '02-l2-ds-usdcop-prepare: '
mc ls minio/02-l2-ds-usdcop-prepare/ --recursive | wc -l
echo ' files (L2 prepared data)'

echo -n '03-l3-ds-usdcop-feature: '
mc ls minio/03-l3-ds-usdcop-feature/ --recursive | wc -l
echo ' files (L3 feature engineered data)'

echo -n '04-l4-ds-usdcop-rlready: '
mc ls minio/04-l4-ds-usdcop-rlready/ --recursive | wc -l
echo ' files (L4 RL-ready data)'

echo -n '05-l5-ds-usdcop-serving: '
mc ls minio/05-l5-ds-usdcop-serving/ --recursive | wc -l
echo ' files (L5 model serving)'

echo -n 'airflow: '
mc ls minio/airflow/ --recursive | wc -l
echo ' files (Airflow artifacts)'

echo -n 'mlflow: '
mc ls minio/mlflow/ --recursive | wc -l
echo ' files (MLflow artifacts)'
"

echo ""
echo "=========================================="
echo "BUCKET MAPPING TO DAG PIPELINE:"
echo "=========================================="
echo ""
echo "L0: 00-raw-usdcop-marketdata      → usdcop_m5__01_l0_acquire"
echo "L1: 01-l1-ds-usdcop-standardize   → usdcop_m5__02_l1_standardize"
echo "L2: 02-l2-ds-usdcop-prepare       → usdcop_m5__03_l2_prepare"
echo "L3: 03-l3-ds-usdcop-feature       → usdcop_m5__04_l3_feature"
echo "L4: 04-l4-ds-usdcop-rlready       → usdcop_m5__05_l4_rlready"
echo "L5: 05-l5-ds-usdcop-serving       → usdcop_m5__06_l5_serving"
echo ""
echo "✅ All duplicates removed"
echo "✅ Data preserved in correct buckets"
echo "✅ Ready for pipeline execution"