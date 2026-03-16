# Checklist de Lanzamiento de Experimento

## Experimento: `exp1_curriculum_aggressive_v1`
**Hipótesis**: Transiciones de curriculum 50% más rápidas
**Fecha de Creación**: 2026-01-19
**Estado**: PENDIENTE DE LANZAMIENTO

---

## Resumen Ejecutivo

Este checklist valida todo el sistema antes de lanzar el experimento de entrenamiento RL:
1. ✅ Infraestructura (Docker, PostgreSQL, Redis, MinIO, MLflow)
2. ✅ Datos (OHLCV 5min, Macro Daily, Date Ranges)
3. ✅ Pipelines (L0→L1→L2→L3→L4)
4. ✅ Contratos (Features, XCom, Rewards)
5. ✅ Entrenamiento (Dataset, Config, Engine)
6. ✅ Backtest y Frontend (Replicabilidad)

---

## FASE 0: Pre-requisitos de Ambiente

### 0.1 Variables de Entorno
```bash
# Crear archivo .env si no existe
cp .env.example .env

# Verificar variables críticas
grep -E "^POSTGRES_|^REDIS_|^MINIO_|^MLFLOW_" .env
```

**Variables Requeridas:**
- [ ] `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- [ ] `REDIS_PASSWORD`
- [ ] `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- [ ] `MLFLOW_TRACKING_URI=http://mlflow:5000`

### 0.2 Secretos de Docker
```bash
# Crear directorio de secretos
mkdir -p secrets

# Generar secretos (o usar existentes)
python scripts/generate_secrets.py

# Verificar archivos
ls -la secrets/
# Esperado:
# db_password.txt
# redis_password.txt
# minio_secret_key.txt
# airflow_password.txt
# airflow_fernet_key.txt
# airflow_secret_key.txt
# grafana_password.txt
# pgadmin_password.txt
```
- [ ] Secretos generados/verificados

---

## FASE 1: Infraestructura Docker

### 1.1 Levantar Servicios Base
```bash
# Levantar infraestructura core
docker-compose up -d postgres redis minio

# Esperar a que PostgreSQL esté healthy
docker-compose ps
# Verificar: postgres debe mostrar (healthy)

# Verificar conexión PostgreSQL
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT version();"
```
- [ ] PostgreSQL healthy
- [ ] Redis healthy
- [ ] MinIO healthy

### 1.2 Inicializar MinIO Buckets
```bash
# Ejecutar inicializador de buckets
docker-compose up -d minio-init

# Verificar buckets creados
docker exec usdcop-minio mc ls minio/
# Esperado: experiments/, production/, dvc-storage/
```
- [ ] Bucket `experiments` existe
- [ ] Bucket `production` existe
- [ ] Bucket `dvc-storage` existe

### 1.3 Levantar MLflow
```bash
# Levantar MLflow
docker-compose up -d mlflow

# Verificar acceso
curl -s http://localhost:5000/health || echo "Verificar manualmente en browser"
# URL: http://localhost:5000
```
- [ ] MLflow accesible en http://localhost:5000

### 1.4 Levantar Airflow
```bash
# Levantar Airflow (webserver + scheduler + worker)
docker-compose up -d airflow-webserver airflow-scheduler

# Esperar inicialización (puede tomar 1-2 minutos)
sleep 60

# Verificar estado
docker-compose ps | grep airflow
# URL: http://localhost:8080 (user: airflow, password: ver secrets/airflow_password.txt)
```
- [ ] Airflow webserver accesible en http://localhost:8080
- [ ] Airflow scheduler running

---

## FASE 2: Verificación de Base de Datos

### 2.1 Tablas Requeridas
```bash
# Conectar a PostgreSQL y verificar tablas
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
"
```

**Tablas Core (MUST EXIST):**
- [ ] `usdcop_m5_ohlcv` - Datos OHLCV 5 minutos
- [ ] `macro_indicators_daily` - Indicadores macro diarios (37 columnas)
- [ ] `model_registry` - Registro de modelos entrenados
- [ ] `experiment_runs` - Registro de experimentos
- [ ] `trading_state` - Estado de paper trading
- [ ] `trades_history` - Historial de trades
- [ ] `equity_snapshots` - Curva de equity

### 2.2 Verificar Esquema de Macro
```bash
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'macro_indicators_daily'
ORDER BY ordinal_position;
"
# Esperado: 37+ columnas incluyendo release_date y ffilled_from_date
```
- [ ] `macro_indicators_daily` tiene 37+ columnas
- [ ] Columna `release_date` existe
- [ ] Columna `ffilled_from_date` existe

### 2.3 Verificar Vistas de Inference
```bash
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT viewname FROM pg_views WHERE schemaname = 'public' AND viewname LIKE '%macro%';
"
# Esperado: inference_macro_features, macro_daily_simple, latest_macro
```
- [ ] Vista `inference_macro_features` existe (7 features para RL)
- [ ] Vista `macro_daily_simple` existe
- [ ] Vista `latest_macro` existe

---

## FASE 3: Población de Datos

### 3.1 Verificar Datos OHLCV
```bash
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    MIN(time) as first_bar,
    MAX(time) as last_bar,
    COUNT(*) as total_bars,
    COUNT(DISTINCT DATE(time)) as trading_days
FROM usdcop_m5_ohlcv;
"
```

**Criterios de Suficiencia:**
- [ ] `first_bar` <= 2023-01-01 (para experiment_training.start)
- [ ] `last_bar` >= 2024-12-31 (para experiment_training.end)
- [ ] `total_bars` >= 100,000 (aprox 2 años de datos 5min)

**Si datos insuficientes, ejecutar backfill:**
```bash
# Opción 1: Trigger DAG de backfill desde Airflow UI
# DAG: v3.l0_ohlcv_backfill

# Opción 2: Restaurar desde backup (si existe)
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB < data/backups/ohlcv_backup.sql
```

### 3.2 Verificar Datos Macro
```bash
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    MIN(fecha) as first_date,
    MAX(fecha) as last_date,
    COUNT(*) as total_days,
    COUNT(fxrt_index_dxy_usa_d_dxy) as dxy_non_null,
    COUNT(volt_vix_usa_d_vix) as vix_non_null,
    COUNT(crsk_spread_embi_col_d_embi) as embi_non_null
FROM macro_indicators_daily;
"
```

**Criterios de Suficiencia:**
- [ ] `first_date` <= 2023-01-01
- [ ] `last_date` >= 2024-12-31
- [ ] Variables críticas (DXY, VIX, EMBI) tienen >90% non-null

**Si datos insuficientes, ejecutar L0 Macro:**
```bash
# Trigger DAG L0 Macro desde Airflow UI
# DAG: v3.l0_macro_unified
```

### 3.3 Verificar Alineación de Fechas
```bash
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    (SELECT MIN(time)::DATE FROM usdcop_m5_ohlcv) as ohlcv_start,
    (SELECT MAX(time)::DATE FROM usdcop_m5_ohlcv) as ohlcv_end,
    (SELECT MIN(fecha) FROM macro_indicators_daily) as macro_start,
    (SELECT MAX(fecha) FROM macro_indicators_daily) as macro_end;
"
```
- [ ] Rangos de OHLCV y Macro se solapan con `experiment_training` (2023-01-01 a 2024-12-31)

---

## FASE 4: Validación de Contratos SSOT

### 4.1 Feature Contract
```bash
# Validar feature contract
python -c "
from src.core.contracts.feature_contract import FEATURE_ORDER, OBSERVATION_DIM, FEATURE_ORDER_HASH
print(f'FEATURE_ORDER ({len(FEATURE_ORDER)} features):')
for i, f in enumerate(FEATURE_ORDER):
    print(f'  {i}: {f}')
print(f'OBSERVATION_DIM: {OBSERVATION_DIM}')
print(f'FEATURE_ORDER_HASH: {FEATURE_ORDER_HASH}')
"
```

**Verificar:**
- [ ] `OBSERVATION_DIM` = 15
- [ ] Features 0-5: Technical (log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14)
- [ ] Features 6-12: Macro (dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_change_1d)
- [ ] Features 13-14: State (position, time_normalized)

### 4.2 XCom Contracts
```bash
# Validar XCom contracts
python -c "
from airflow.dags.contracts.xcom_contracts import (
    L0XComKeysEnum, L1XComKeysEnum, L2XComKeysEnum,
    L3XComKeysEnum, L5XComKeysEnum, L6XComKeysEnum
)
print('L0 Keys:', len(L0XComKeysEnum.__members__))
print('L1 Keys:', len(L1XComKeysEnum.__members__))
print('L2 Keys:', len(L2XComKeysEnum.__members__))
print('L3 Keys:', len(L3XComKeysEnum.__members__))
print('L5 Keys:', len(L5XComKeysEnum.__members__))
print('L6 Keys:', len(L6XComKeysEnum.__members__))
"
```
- [ ] Todos los enums tienen keys definidos

### 4.3 CanonicalFeatureBuilder
```bash
# Validar CanonicalFeatureBuilder
python -c "
from src.feature_store.builders.canonical_feature_builder import CanonicalFeatureBuilder
builder = CanonicalFeatureBuilder()
print(f'Builder VERSION: {builder.VERSION}')
print(f'Features: {len(builder.FEATURE_ORDER)}')
print(f'Has compute_features: {hasattr(builder, \"compute_features\")}')
print(f'Has validate_features: {hasattr(builder, \"validate_features\")}')
"
```
- [ ] VERSION = "1.0.0"
- [ ] Features = 15
- [ ] Métodos compute_features y validate_features existen

### 4.4 Date Ranges SSOT
```bash
# Validar date_ranges.yaml
python -c "
import yaml
with open('config/date_ranges.yaml') as f:
    dates = yaml.safe_load(f)
print('experiment_training:')
print(f'  start: {dates[\"experiment_training\"][\"start\"]}')
print(f'  end: {dates[\"experiment_training\"][\"end\"]}')
print('validation:')
print(f'  start: {dates[\"validation\"][\"start\"]}')
print(f'  end: {dates[\"validation\"][\"end\"]}')
"
```
- [ ] experiment_training: 2023-01-01 a 2024-12-31
- [ ] validation: 2025-01-01 a 2025-06-30

---

## FASE 5: Validación de Experiment Config

### 5.1 Validar YAML del Experimento
```bash
# Validar sintaxis YAML
python -c "
import yaml
with open('config/experiments/exp1_curriculum_aggressive_v1.yaml') as f:
    config = yaml.safe_load(f)
print(f'Experiment: {config[\"experiment\"][\"name\"]}')
print(f'Version: {config[\"experiment\"][\"version\"]}')
print(f'Algorithm: {config[\"model\"][\"algorithm\"]}')
print(f'Total Timesteps: {config[\"training\"][\"total_timesteps\"]}')
print(f'Reward Weights: {config[\"reward\"][\"weights\"]}')
# Verificar suma de pesos = 1.0
weights = config['reward']['weights']
total = sum(weights.values())
print(f'Weight Sum: {total} (should be 1.0)')
"
```
- [ ] YAML parsea sin errores
- [ ] Suma de reward weights = 1.0

### 5.2 Validar Alineación con Feature Contract
```bash
# Verificar features en config vs contract
python -c "
import yaml
from src.core.contracts.feature_contract import FEATURE_ORDER

with open('config/experiments/exp1_curriculum_aggressive_v1.yaml') as f:
    config = yaml.safe_load(f)

config_features = config['data']['feature_columns']
contract_features = list(FEATURE_ORDER)[:13]  # Exclude state features

missing = set(contract_features) - set(config_features)
extra = set(config_features) - set(contract_features)

print(f'Config features: {len(config_features)}')
print(f'Contract features (market only): {len(contract_features)}')
print(f'Missing: {missing}')
print(f'Extra: {extra}')

assert not missing, f'Missing features: {missing}'
assert not extra, f'Extra features: {extra}'
print('✓ Feature alignment OK')
"
```
- [ ] No hay features faltantes
- [ ] No hay features extra

### 5.3 Validar Reward Components
```bash
# Verificar que todos los reward components existen
python -c "
from src.training.reward_components import (
    DifferentialSharpeRatio,
    SortinoCalculator,
    StableRegimeDetector,
    OilCorrelationTracker,
    BanrepInterventionDetector,
    HoldingDecay,
    InactivityTracker,
    ChurnTracker,
    BiasDetector,
    ActionCorrelationTracker,
)
print('✓ All reward components importable')
"
```
- [ ] Todos los reward components importan sin error

---

## FASE 6: Generar Dataset (L2)

### 6.1 Opción A: Via Airflow DAG (Recomendado)
```bash
# Trigger L2 desde Airflow UI
# DAG: v3.l2_preprocessing_pipeline
# Parámetros (opcional):
#   experiment_name: exp1_curriculum_aggressive_v1
#   date_range_start: 2023-01-01
#   date_range_end: 2024-12-31

# Verificar estado en Airflow UI
# http://localhost:8080/dags/v3.l2_preprocessing_pipeline/grid
```

### 6.2 Opción B: Script Directo
```bash
# Generar dataset manualmente
python data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --output-dir data/experiments/exp1_curriculum_aggressive_v1
```

### 6.3 Verificar Dataset Generado
```bash
# Listar archivos generados
ls -la data/experiments/exp1_curriculum_aggressive_v1/

# Esperado:
# train.parquet (o .csv)
# norm_stats.json
# manifest.json

# Verificar contenido
python -c "
import pandas as pd
df = pd.read_parquet('data/experiments/exp1_curriculum_aggressive_v1/train.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Date range: {df.index.min()} to {df.index.max()}')
"
```
- [ ] Dataset generado con >100,000 rows
- [ ] 13 columnas de features (sin state)
- [ ] Rango de fechas 2023-01-01 a 2024-12-31

---

## FASE 7: Ejecutar Entrenamiento (L3)

### 7.1 Dry Run (Validación sin Entrenar)
```bash
# Ejecutar dry run para validar todo
python scripts/run_experiment.py \
    --config config/experiments/exp1_curriculum_aggressive_v1.yaml \
    --dry-run \
    -v

# Debe mostrar:
# ✓ Configuration valid
# ✓ Dataset found
# ✓ Feature contract matches
# ✓ Ready to train
```
- [ ] Dry run pasa sin errores

### 7.2 Ejecutar Entrenamiento
```bash
# Opción A: Via script (recomendado para primer experimento)
python scripts/run_experiment.py \
    --config config/experiments/exp1_curriculum_aggressive_v1.yaml \
    -v

# Opción B: Via Airflow DAG
# DAG: v3.l3_model_training
# Parámetros:
#   experiment_name: exp1_curriculum_aggressive_v1
#   dataset_path: data/experiments/exp1_curriculum_aggressive_v1/train.parquet
```

### 7.3 Monitorear Entrenamiento
```bash
# TensorBoard (si habilitado)
tensorboard --logdir models/exp1_curriculum_aggressive_v1/tensorboard

# MLflow UI
# http://localhost:5000 → Experimento: usdcop_ab_testing

# Logs de entrenamiento
tail -f logs/training.log
```
- [ ] Entrenamiento iniciado
- [ ] MLflow run creado
- [ ] Métricas actualizándose

### 7.4 Verificar Resultados
```bash
# Verificar modelo generado
ls -la models/exp1_curriculum_aggressive_v1/

# Esperado:
# final_model.zip
# norm_stats.json
# training_result.json
# tensorboard/

# Verificar métricas finales
python -c "
import json
with open('models/exp1_curriculum_aggressive_v1/training_result.json') as f:
    result = json.load(f)
print(f'Success: {result[\"success\"]}')
print(f'Best Reward: {result[\"best_mean_reward\"]}')
print(f'Duration: {result[\"training_duration_seconds\"]/60:.1f} min')
print(f'MLflow Run: {result.get(\"mlflow_run_id\", \"N/A\")}')
"
```
- [ ] `final_model.zip` existe
- [ ] `norm_stats.json` existe
- [ ] `training_result.json` muestra success=true

---

## FASE 8: Registrar Modelo

### 8.1 Registrar en model_registry
```bash
# El TrainingEngine debe registrar automáticamente
# Verificar en PostgreSQL
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    model_id,
    model_version,
    status,
    observation_dim,
    test_sharpe,
    created_at
FROM model_registry
WHERE model_id LIKE '%exp1_curriculum%'
ORDER BY created_at DESC
LIMIT 1;
"
```
- [ ] Modelo registrado en `model_registry`
- [ ] observation_dim = 15

### 8.2 Verificar Hashes de Integridad
```bash
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    model_id,
    model_hash,
    norm_stats_hash,
    feature_order_hash
FROM model_registry
WHERE model_id LIKE '%exp1_curriculum%'
ORDER BY created_at DESC
LIMIT 1;
"
```
- [ ] model_hash no es NULL
- [ ] norm_stats_hash no es NULL
- [ ] feature_order_hash coincide con FEATURE_ORDER_HASH del contract

---

## FASE 9: Ejecutar Backtest (L4)

### 9.1 Ejecutar Backtest de Validación
```bash
# Via Airflow DAG
# DAG: v3.l4_backtest_validation
# O manual:

python -c "
from src.backtest.engine.unified_backtest_engine import UnifiedBacktestEngine

engine = UnifiedBacktestEngine(
    model_path='models/exp1_curriculum_aggressive_v1/final_model.zip',
    norm_stats_path='models/exp1_curriculum_aggressive_v1/norm_stats.json',
)

# Usar periodo de validación
results = engine.run(
    start_date='2025-01-01',
    end_date='2025-06-30',
)

print(f'Sharpe: {results.sharpe_ratio:.3f}')
print(f'Max DD: {results.max_drawdown:.2%}')
print(f'Win Rate: {results.win_rate:.2%}')
print(f'Total Trades: {results.total_trades}')
"
```
- [ ] Backtest ejecutado sin errores
- [ ] Sharpe > 0.5 (según success_criteria)
- [ ] Max DD < 15%
- [ ] Win Rate > 45%
- [ ] Total Trades > 50

### 9.2 Guardar Resultados de Backtest
```bash
# Los resultados deben guardarse en:
# - MinIO: s3://experiments/exp1_curriculum_aggressive_v1/backtests/
# - PostgreSQL: experiment_runs (actualizar backtest_metrics)

# Verificar en PostgreSQL
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT
    experiment_name,
    status,
    backtest_metrics
FROM experiment_runs
WHERE experiment_name = 'exp1_curriculum_aggressive_v1'
ORDER BY started_at DESC
LIMIT 1;
"
```
- [ ] Backtest metrics guardados

---

## FASE 10: Replicar en Frontend (Dashboard)

### 10.1 Iniciar API de Inference
```bash
# Levantar inference_api
docker-compose up -d inference-api

# Verificar health
curl http://localhost:8000/health
# Esperado: {"status": "healthy"}

# Verificar modelos disponibles
curl http://localhost:8000/models
```
- [ ] API healthy
- [ ] Modelo exp1_curriculum_aggressive_v1 listado

### 10.2 Configurar Modelo en Dashboard
```bash
# Agregar modelo a config del dashboard
# Archivo: usdcop-trading-dashboard/lib/config/models.config.ts

# El modelo debe aparecer en:
# - Dropdown de selección de modelo
# - Panel de backtest
```

### 10.3 Ejecutar Backtest desde Dashboard
```bash
# Iniciar dashboard
cd usdcop-trading-dashboard
npm run dev

# Acceder a: http://localhost:3000
# 1. Ir a sección Backtest
# 2. Seleccionar modelo: exp1_curriculum_aggressive_v1
# 3. Seleccionar rango: 2025-01-01 a 2025-06-30 (validation)
# 4. Ejecutar backtest

# Los resultados deben coincidir con backtest del FASE 9
```
- [ ] Dashboard accesible
- [ ] Modelo seleccionable
- [ ] Backtest ejecutable
- [ ] Resultados coinciden con FASE 9 (±5%)

### 10.4 Verificar Data Lineage en Dashboard
```bash
# El dashboard debe mostrar:
# - model_id
# - feature_order_hash
# - norm_stats_hash
# - training_date_range
# - backtest_date_range
# - experiment_config_link

# Verificar endpoint de lineage
curl http://localhost:8000/lineage/exp1_curriculum_aggressive_v1
```
- [ ] Lineage visible en dashboard
- [ ] Hashes coinciden con model_registry

---

## FASE 11: Validaciones Finales

### 11.1 Ejecutar Scripts de Validación
```bash
# Validar blockers
python scripts/validate_blockers.py

# Validar DVC
python scripts/validate_dvc.py

# Validar hashes
python scripts/validate_hash_reconciliation.py
```
- [ ] Todos los scripts pasan

### 11.2 Checklist de Data Lineage Completo
```
LINEAGE COMPLETO:
┌─────────────────────────────────────────────────────────────────┐
│ SOURCE DATA                                                     │
├─────────────────────────────────────────────────────────────────┤
│ OHLCV: usdcop_m5_ohlcv                                          │
│   Date Range: ______ to ______                                  │
│   Rows: ______                                                  │
│                                                                 │
│ MACRO: macro_indicators_daily                                   │
│   Date Range: ______ to ______                                  │
│   Rows: ______                                                  │
├─────────────────────────────────────────────────────────────────┤
│ DATASET (L2 Output)                                             │
├─────────────────────────────────────────────────────────────────┤
│ Path: data/experiments/exp1_curriculum_aggressive_v1/           │
│ Hash: ________________                                          │
│ Rows: ______                                                    │
│ Feature Order Hash: ________________                            │
├─────────────────────────────────────────────────────────────────┤
│ MODEL (L3 Output)                                               │
├─────────────────────────────────────────────────────────────────┤
│ Path: models/exp1_curriculum_aggressive_v1/final_model.zip      │
│ Model Hash: ________________                                    │
│ Norm Stats Hash: ________________                               │
│ MLflow Run ID: ________________                                 │
├─────────────────────────────────────────────────────────────────┤
│ BACKTEST (L4 Output)                                            │
├─────────────────────────────────────────────────────────────────┤
│ Period: 2025-01-01 to 2025-06-30                                │
│ Sharpe: ______                                                  │
│ Max DD: ______%                                                 │
│ Win Rate: ______%                                               │
│ Trades: ______                                                  │
├─────────────────────────────────────────────────────────────────┤
│ REGISTRATION                                                    │
├─────────────────────────────────────────────────────────────────┤
│ model_registry.model_id: ________________                       │
│ model_registry.status: ________________                         │
│ model_registry.feature_order_hash: ________________             │
│                                                                 │
│ experiment_runs.run_uuid: ________________                      │
│ experiment_runs.status: ________________                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Comandos Rápidos de Referencia

### Infraestructura
```bash
# Levantar todo
docker-compose up -d

# Ver status
docker-compose ps

# Ver logs
docker-compose logs -f postgres
docker-compose logs -f airflow-webserver

# Reiniciar servicio
docker-compose restart inference-api
```

### Base de Datos
```bash
# Conectar a PostgreSQL
docker exec -it usdcop-postgres-timescale psql -U $POSTGRES_USER -d $POSTGRES_DB

# Queries útiles
\dt                          # Listar tablas
\d+ usdcop_m5_ohlcv         # Describir tabla
SELECT COUNT(*) FROM usdcop_m5_ohlcv;
```

### Airflow
```bash
# Trigger DAG via CLI
docker exec -it usdcop-airflow-scheduler airflow dags trigger v3.l2_preprocessing_pipeline

# Ver estado de DAG
docker exec -it usdcop-airflow-scheduler airflow dags state v3.l2_preprocessing_pipeline <run_id>
```

### MLflow
```bash
# Ver experimentos
mlflow experiments list --backend-store-uri http://localhost:5000

# Comparar runs
python scripts/compare_experiments.py --run-a <id1> --run-b <id2>
```

---

## Troubleshooting

### Error: "No data in usdcop_m5_ohlcv"
```bash
# Ejecutar backfill
# 1. Via Airflow: Trigger v3.l0_ohlcv_backfill
# 2. Via script: python scripts/backfill_ohlcv.py --start 2023-01-01 --end 2024-12-31
```

### Error: "Feature order mismatch"
```bash
# Verificar contrato
python -c "from src.core.contracts.feature_contract import FEATURE_ORDER; print(FEATURE_ORDER)"

# Regenerar dataset con builder correcto
python data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py --force
```

### Error: "MLflow connection refused"
```bash
# Verificar que MLflow está corriendo
docker-compose ps mlflow

# Reiniciar si necesario
docker-compose restart mlflow

# Verificar URI
echo $MLFLOW_TRACKING_URI  # Debe ser http://mlflow:5000 (dentro de Docker) o http://localhost:5000 (fuera)
```

### Error: "Backtest results differ from dashboard"
```bash
# Verificar que usan mismo modelo
# CLI: models/exp1.../final_model.zip
# API: Verificar model_path en response

# Verificar norm_stats
# Deben ser idénticos (mismo hash)
```

---

## Siguiente Paso Post-Experimento

Una vez completado exitosamente:

1. **Comparar con otros experimentos:**
   ```bash
   python scripts/compare_experiments.py \
       --exp-a exp1_curriculum_aggressive_v1 \
       --exp-b exp2_macro_emphasis_v1 \
       --output docs/comparison_exp1_vs_exp2.md
   ```

2. **Promover a producción (si métricas son satisfactorias):**
   ```bash
   python scripts/promote_model.py \
       --model-id exp1_curriculum_aggressive_v1 \
       --target production
   ```

3. **Ejecutar shadow mode:**
   - Deploy challenger junto a champion
   - Monitorear en L5 multi-model inference

---

**Documento generado**: 2026-01-19
**Autor**: Trading Team
**Versión**: 1.0.0
