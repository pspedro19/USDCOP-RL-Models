#!/bin/bash

# Script to set L5 Airflow Variables
# Run this ONCE before executing the L5 DAG

echo "Setting L5 Airflow Variables..."

# Use docker exec to set variables in the Airflow container
docker exec -it usdcop-airflow-webserver airflow variables set L5_TOTAL_TIMESTEPS 1000000
docker exec -it usdcop-airflow-webserver airflow variables set L5_SORTINO_THRESHOLD 1.3
docker exec -it usdcop-airflow-webserver airflow variables set L5_MAX_DD_THRESHOLD 0.15
docker exec -it usdcop-airflow-webserver airflow variables set L5_CALMAR_THRESHOLD 0.8
docker exec -it usdcop-airflow-webserver airflow variables set L5_SORTINO_DIFF_THRESHOLD 0.5
docker exec -it usdcop-airflow-webserver airflow variables set L5_COST_STRESS_MULTIPLIER 1.25
docker exec -it usdcop-airflow-webserver airflow variables set L5_GATE_EVAL_EPISODES 100
docker exec -it usdcop-airflow-webserver airflow variables set L5_IN_LOOP_EVAL_EPISODES 15
docker exec -it usdcop-airflow-webserver airflow variables set L5_CAGR_DROP_THRESHOLD 0.20
docker exec -it usdcop-airflow-webserver airflow variables set L5_INFERENCE_P99_MS 20
docker exec -it usdcop-airflow-webserver airflow variables set L5_E2E_P99_MS 100
docker exec -it usdcop-airflow-webserver airflow variables set L5_TRAIN_TIMEOUT_HOURS 6

echo "L5 Variables set successfully!"
echo ""
echo "Verifying variables..."
docker exec -it usdcop-airflow-webserver airflow variables list | grep L5_

echo ""
echo "Variables configured. You can now run the L5 DAG!"
echo ""
echo "Next steps:"
echo "1. Clear L5 tasks in Airflow UI: usdcop_m5__06_l5_serving"
echo "2. Trigger the DAG manually"
echo "3. Monitor training logs for Monitor wrapper confirmation"
echo "4. Check gate evaluation for clean PASS status"