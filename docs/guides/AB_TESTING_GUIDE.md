# Guía Completa: A/B Testing de Modelos USD/COP

## Resumen

Esta guía explica cómo ejecutar el pipeline completo desde datos crudos hasta ver resultados de A/B testing en el dashboard.

## Arquitectura de Almacenamiento

```
┌─────────────────────────────────────────────────────────────────┐
│                    ¿DÓNDE SE GUARDA CADA COSA?                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  POSTGRESQL (Tablas transaccionales - datos operativos):        │
│  ├── usdcop_m5_ohlcv           → Velas OHLCV 5min               │
│  ├── macro_indicators_daily    → 37 indicadores macro           │
│  ├── inference_features_5m     → Features calculados (L1)       │
│  ├── config.models             → Registro modelos activos       │
│  ├── trading.model_inferences  → Señales producción             │
│  └── model_registry            → Historial modelos              │
│                                                                 │
│  MINIO/DVC (Artifacts versionados - reproducibilidad):          │
│  ├── dvc-storage/              → Datasets versionados           │
│  │   └── files/md5/{hash}      → Archivos por hash              │
│  └── mlflow/                   → Modelos y artifacts            │
│                                                                 │
│  FILESYSTEM (Archivos locales - desarrollo):                    │
│  ├── data/pipeline/07_output/  → Datasets CSV                   │
│  ├── models/                   → Modelos .zip                   │
│  └── config/                   → Configs, contracts, norm_stats │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Paso 0: Levantar Infraestructura

```bash
# Levantar todos los servicios
docker-compose up -d

# Verificar que todo esté corriendo
docker-compose ps

# Servicios esperados:
# - postgres       (5432)
# - redis          (6379)
# - minio          (9000, 9001)
# - mlflow         (5000)
# - airflow        (8080)
# - dashboard      (3000)
```

**URLs de acceso:**
| Servicio | URL | Credenciales |
|----------|-----|--------------|
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5000 | - |
| MinIO | http://localhost:9001 | minio / minio123 |
| Dashboard | http://localhost:3000 | - |
| PostgreSQL | localhost:5432 | postgres / postgres |

---

## Paso 1: Cargar Datos (L0)

### Opción A: Via Airflow UI

1. Ir a http://localhost:8080
2. Trigger DAG: `v3.l0_ohlcv_backfill`
   - Config: `{"start_date": "2023-01-01", "end_date": "2024-12-31"}`
3. Trigger DAG: `v3.l0_macro_unified`
4. Esperar a que ambos completen (verde)

### Opción B: Via CLI

```bash
# Trigger OHLCV backfill
airflow dags trigger v3.l0_ohlcv_backfill \
  --conf '{"start_date": "2023-01-01", "end_date": "2024-12-31"}'

# Trigger macro indicators
airflow dags trigger v3.l0_macro_unified

# Monitorear progreso
airflow dags list-runs -d v3.l0_ohlcv_backfill --state running
```

### Verificar datos cargados:

```sql
-- Conectar a PostgreSQL
psql -h localhost -U postgres -d usdcop

-- Verificar OHLCV
SELECT COUNT(*), MIN(time), MAX(time) FROM usdcop_m5_ohlcv;
-- Esperado: ~100,000+ rows, 2023-01-01 a 2024-12-31

-- Verificar Macro
SELECT COUNT(*), MIN(fecha), MAX(fecha) FROM macro_indicators_daily;
-- Esperado: ~500 rows (días hábiles)
```

---

## Paso 2: Generar Features (L1)

```bash
# Via Airflow
airflow dags trigger v3.l1_feature_refresh

# Verificar
psql -h localhost -U postgres -d usdcop -c "
  SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
  FROM inference_features_5m;
"
```

---

## Paso 3: Generar Datasets (L2)

### Opción A: Via script (recomendado para A/B testing)

```bash
# Generar ambas variantes de dataset
python scripts/generate_dataset_variants.py

# Output esperado:
# data/pipeline/07_output/datasets_5min/
# ├── RL_DS3_MACRO_FULL.csv  (15 features)
# └── RL_DS3_MACRO_CORE.csv  (12 features)
```

### Opción B: Via Airflow DAG

```bash
airflow dags trigger v3.l2_preprocessing_pipeline \
  --conf '{"generate_variants": true}'
```

### Verificar datasets:

```bash
# Ver archivos generados
ls -la data/pipeline/07_output/datasets_5min/

# Ver estructura de cada dataset
head -5 data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_FULL.csv
head -5 data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv

# Contar filas
wc -l data/pipeline/07_output/datasets_5min/*.csv
```

---

## Paso 4: Entrenar Experimentos (L3)

### Experimento A: Full Macro (baseline)

```bash
airflow dags trigger v3.l3_model_training --conf '{
  "experiment_config_path": "config/experiments/baseline_full_macro.yaml",
  "version": "exp_a_v1",
  "dvc_enabled": true,
  "mlflow_enabled": true
}'
```

### Experimento B: Reduced Macro

```bash
airflow dags trigger v3.l3_model_training --conf '{
  "experiment_config_path": "config/experiments/reduced_core_macro.yaml",
  "version": "exp_b_v1",
  "dvc_enabled": true,
  "mlflow_enabled": true
}'
```

### Monitorear entrenamiento:

```bash
# Ver logs en Airflow
airflow dags list-runs -d v3.l3_model_training

# Ver en MLflow
open http://localhost:5000
# → Experiments → usdcop_ab_testing
```

### Verificar artefactos generados:

```bash
# Modelos
ls -la models/ppo_exp_a_v1_production/
ls -la models/ppo_exp_b_v1_production/

# DVC tags
git tag -l "dataset-exp/*"

# MLflow
curl http://localhost:5000/api/2.0/mlflow/experiments/list | jq
```

---

## Paso 5: Validar y Comparar (L4)

### Ejecutar backtest

```bash
airflow dags trigger v3.l4_backtest_validation
```

### Ejecutar comparación A/B

```bash
airflow dags trigger v3.l4_experiment_runner --conf '{
  "experiment_name": "reduced_core_macro",
  "compare_to": "baseline_full_macro"
}'
```

### Ver resultados en MLflow:

1. Ir a http://localhost:5000
2. Click en "usdcop_ab_testing"
3. Seleccionar ambos runs
4. Click "Compare"

### Métricas a comparar:

| Métrica | Full Macro | Reduced | Mejor |
|---------|------------|---------|-------|
| Sharpe Ratio | ? | ? | Mayor |
| Max Drawdown | ? | ? | Menor |
| Win Rate | ? | ? | Mayor |
| Total Return | ? | ? | Mayor |

---

## Paso 6: Ver en Dashboard

1. Ir a http://localhost:3000
2. Navegar a "Models" → "Backtest Results"
3. Ver comparación visual de equity curves

---

## Experimentos Creados

### Experimento A: `baseline_full_macro.yaml`

```yaml
# 15 features total: 6 técnicos + 7 macro + 2 estado
feature_columns:
  # Técnicos (6)
  - log_ret_5m, log_ret_1h, log_ret_4h
  - rsi_9, atr_pct, adx_14
  # Macro COMPLETO (7)
  - dxy_z           # Dollar Index
  - dxy_change_1d   # Cambio DXY
  - vix_z           # Volatilidad
  - embi_z          # Riesgo país
  - brent_change_1d # Petróleo
  - rate_spread     # Diferencial tasas
  - usdmxn_change_1d # Proxy EM
```

### Experimento B: `reduced_core_macro.yaml`

```yaml
# 12 features total: 6 técnicos + 4 macro + 2 estado
feature_columns:
  # Técnicos (6) - igual
  - log_ret_5m, log_ret_1h, log_ret_4h
  - rsi_9, atr_pct, adx_14
  # Macro CORE (4) - solo los más importantes
  - dxy_z           # Correlación 0.85 con USD/COP
  - vix_z           # Risk-on/risk-off global
  - embi_z          # Riesgo soberano Colombia
  - brent_change_1d # Petróleo (exportador)

# EXCLUIDOS (hipótesis: redundantes):
# - dxy_change_1d   → ya capturado en dxy_z
# - rate_spread     → menos predictivo que EMBI
# - usdmxn_change_1d → correlación indirecta
```

---

## Script Todo-en-Uno

```bash
# Ejecutar pipeline completo
chmod +x scripts/run_ab_experiment.sh
./scripts/run_ab_experiment.sh all

# O paso a paso:
./scripts/run_ab_experiment.sh verify   # Verificar setup
./scripts/run_ab_experiment.sh data     # Cargar datos
./scripts/run_ab_experiment.sh train    # Entrenar modelos
./scripts/run_ab_experiment.sh compare  # Comparar resultados
```

---

## Troubleshooting

### Error: "Dataset not found"
```bash
# Verificar que L2 generó los datasets
ls data/pipeline/07_output/datasets_5min/

# Si no existen, regenerar
python scripts/generate_dataset_variants.py
```

### Error: "Feature order mismatch"
```bash
# Verificar contracts
cat config/contracts/exp_a_v1_contract.json | jq .feature_order

# Comparar con SSOT
python -c "from src.core.contracts import FEATURE_ORDER; print(FEATURE_ORDER)"
```

### Error: "DVC push failed"
```bash
# Verificar MinIO está corriendo
docker-compose ps minio

# Verificar DVC remote
dvc remote list
dvc remote modify minio endpointurl http://localhost:9000
```

### Ver logs de DAG
```bash
# Logs recientes
airflow tasks logs v3.l3_model_training train_model -1

# O en UI
http://localhost:8080/dags/v3.l3_model_training/graph
```

---

## Checklist Final

- [ ] PostgreSQL tiene datos (usdcop_m5_ohlcv, macro_indicators_daily)
- [ ] Datasets CSV generados en data/pipeline/07_output/
- [ ] Experimentos YAML en config/experiments/
- [ ] Modelos entrenados en models/
- [ ] Runs visibles en MLflow
- [ ] DVC tags creados (git tag -l)
- [ ] Backtest completado
- [ ] Comparación A/B disponible en MLflow

---

## Resultados Esperados

Después de ejecutar ambos experimentos, deberías poder:

1. **En MLflow**: Ver side-by-side comparison con métricas
2. **En Dashboard**: Ver equity curves de ambos modelos
3. **En PostgreSQL**: Query resultados de backtest
4. **En MinIO**: Datasets versionados con DVC

```sql
-- Query resultados backtest
SELECT
  model_version,
  sharpe_ratio,
  max_drawdown,
  total_return,
  win_rate
FROM backtest_results
ORDER BY created_at DESC
LIMIT 2;
```
