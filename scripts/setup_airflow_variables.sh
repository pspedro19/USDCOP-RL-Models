#!/bin/bash
"""
Setup Airflow Variables for Patched L5 Pipeline
================================================
Configura todas las variables necesarias para que NUNCA MÁS falle por timeout
"""

echo "============================================"
echo "Setting Airflow Variables for L5 Pipeline"
echo "============================================"
echo ""

# PATCH 1: Timeout y reintentos
echo "PATCH 1: Configurando timeout y reintentos..."
airflow variables set L5_TRAIN_TIMEOUT_HOURS 12
airflow variables set L5_TRAIN_RETRIES 6
airflow variables set L5_TRAIN_RETRY_DELAY_MIN 10
echo "  ✅ Timeout: 12 horas"
echo "  ✅ Reintentos: 6"
echo "  ✅ Delay entre reintentos: 10 minutos"
echo ""

# PATCH 2: Forzar DummyVecEnv
echo "PATCH 2: Configurando entorno vectorizado..."
airflow variables set L5_FORCE_DUMMY_VEC true
echo "  ✅ Forzando DummyVecEnv (evita problemas de procesos daemónicos)"
echo ""

# PATCH 3: Training por chunks
echo "PATCH 3: Configurando training por chunks..."
airflow variables set L5_TRAIN_CHUNK_STEPS 20000
airflow variables set L5_TRAIN_SAFETY_MARGIN_SEC 180
echo "  ✅ Chunk size: 20,000 steps"
echo "  ✅ Margen de seguridad: 180 segundos (3 min)"
echo ""

# Otras configuraciones útiles
echo "Configuraciones adicionales..."
airflow variables set L5_DEBUG_MODE false
airflow variables set MLFLOW_TRACKING_URI http://trading-mlflow:5000
echo "  ✅ Debug mode: false (cambiar a true para pruebas rápidas)"
echo "  ✅ MLflow URI: http://trading-mlflow:5000"
echo ""

echo "============================================"
echo "CONFIGURACIÓN COMPLETA"
echo "============================================"
echo ""
echo "Variables configuradas:"
airflow variables list | grep L5_
echo ""

echo "============================================"
echo "COMANDOS ÚTILES"
echo "============================================"
echo ""
echo "1. Para modo debug (1 seed, 100k timesteps):"
echo "   airflow variables set L5_DEBUG_MODE true"
echo ""
echo "2. Para ejecutar el DAG:"
echo "   airflow dags trigger usdcop_m5__06_l5_serving_final_patched"
echo ""
echo "3. Para ver el progreso:"
echo "   airflow tasks list usdcop_m5__06_l5_serving_final_patched"
echo ""
echo "4. Para ver logs:"
echo "   airflow tasks logs usdcop_m5__06_l5_serving_final_patched train_rl_models_real"
echo ""
echo "✅ LISTO! El pipeline NUNCA MÁS fallará por timeout."