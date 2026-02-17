# 🎯 PLAN: Primer A/B Test Completo L0→L4
## USDCOP RL Trading System

**Fecha:** 2026-01-31
**Objetivo:** Ejecutar el flujo completo y documentar visualmente hasta aprobación humana

---

## 📋 PRE-REQUISITOS

### Fase 0: Verificación de Infraestructura

```
[ ] 0.1 Verificar todos los contenedores Docker healthy
[ ] 0.2 Rebuild dashboard con nuevas páginas (/experiments, /production)
[ ] 0.3 Verificar conexiones DB (PostgreSQL, Redis)
[ ] 0.4 Verificar Airflow accesible y sin errores críticos
[ ] 0.5 Verificar MLflow accesible
[ ] 0.6 Identificar tecnología de inferencia (ONNX vs SB3 nativo)
```

---

## 🎬 PLAN DE EJECUCIÓN

### FASE 1: Estado Inicial (Screenshots ANTES)
**Tiempo estimado:** 5 min

| Step | Acción | Screenshot |
|------|--------|------------|
| 1.1 | Estado de tablas en PostgreSQL | `01_db_state_before.png` |
| 1.2 | Dashboard Hub | `02_hub_before.png` |
| 1.3 | Airflow DAGs list | `03_airflow_before.png` |
| 1.4 | MLflow experiments | `04_mlflow_before.png` |

**Queries a ejecutar:**
```sql
-- Contar registros en tablas críticas
SELECT 'usdcop_m5_ohlcv' as tabla, COUNT(*) FROM usdcop_m5_ohlcv
UNION ALL SELECT 'macro_indicators_daily', COUNT(*) FROM macro_indicators_daily
UNION ALL SELECT 'inference_features_5m', COUNT(*) FROM inference_features_5m
UNION ALL SELECT 'model_registry', COUNT(*) FROM model_registry
UNION ALL SELECT 'promotion_proposals', COUNT(*) FROM promotion_proposals;
```

---

### FASE 2: L0 - Data Acquisition
**Tiempo estimado:** 10-15 min

| Step | Acción | Verificación |
|------|--------|--------------|
| 2.1 | Unpause `l0_macro_update` | DAG activo |
| 2.2 | Trigger manual `l0_macro_update` | Task running |
| 2.3 | Esperar completación | Success |
| 2.4 | Verificar `macro_indicators_daily` | Rows aumentaron |
| 2.5 | Screenshot Airflow task logs | `05_l0_macro_complete.png` |

**Comandos:**
```bash
# Activar DAG
docker exec usdcop-airflow-webserver airflow dags unpause l0_macro_update

# Trigger
docker exec usdcop-airflow-webserver airflow dags trigger l0_macro_update

# Verificar estado
docker exec usdcop-airflow-webserver airflow dags state l0_macro_update
```

---

### FASE 3: L1 - Feature Computation
**Tiempo estimado:** 5-10 min

| Step | Acción | Verificación |
|------|--------|--------------|
| 3.1 | Verificar si existe `l1_feature_refresh` DAG | - |
| 3.2 | Trigger L1 feature computation | Task running |
| 3.3 | Verificar `inference_features_5m` poblada | Rows > 0 |
| 3.4 | Screenshot evidencia | `06_l1_features_complete.png` |

**Queries de verificación:**
```sql
SELECT COUNT(*) as feature_rows,
       MIN(time) as oldest,
       MAX(time) as newest
FROM inference_features_5m;
```

---

### FASE 4: L2 - Dataset Engineering
**Tiempo estimado:** 10-20 min

| Step | Acción | Verificación |
|------|--------|--------------|
| 4.1 | Trigger `rl_l2_01_dataset_build` | Task running |
| 4.2 | Verificar DVC tracking | `.dvc` files updated |
| 4.3 | Verificar MinIO storage | Dataset uploaded |
| 4.4 | Screenshot dataset metadata | `07_l2_dataset_complete.png` |

**Verificaciones:**
```bash
# MinIO check
docker exec usdcop-minio mc ls minio/datasets/

# DVC status
dvc status
```

---

### FASE 5: L3 - Model Training
**Tiempo estimado:** 30-60 min (depende de epochs)

| Step | Acción | Verificación |
|------|--------|--------------|
| 5.1 | Trigger `rl_l3_01_model_training` | Task running |
| 5.2 | Monitorear MLflow | Experiment created |
| 5.3 | Esperar completación | Model registered |
| 5.4 | Screenshot MLflow metrics | `08_l3_training_complete.png` |

**Verificaciones:**
```bash
# MLflow experiments
curl http://localhost:5001/api/2.0/mlflow/experiments/list

# Model artifacts
docker exec usdcop-minio mc ls minio/mlflow-artifacts/
```

---

### FASE 6: L4 - Backtest & Promotion
**Tiempo estimado:** 15-30 min

| Step | Acción | Verificación |
|------|--------|--------------|
| 6.1 | Trigger `rl_l4_04_backtest_promotion` | Task running |
| 6.2 | Verificar backtest metrics | Sharpe, returns |
| 6.3 | Verificar `promotion_proposals` | New row created |
| 6.4 | Screenshot proposal | `09_l4_promotion_proposal.png` |

**Query de verificación:**
```sql
SELECT
    proposal_id,
    model_id,
    recommendation,
    confidence,
    status,
    created_at
FROM promotion_proposals
ORDER BY created_at DESC
LIMIT 1;
```

---

### FASE 7: Frontend - Backtest Visual
**Tiempo estimado:** 10 min

| Step | Acción | Screenshot |
|------|--------|------------|
| 7.1 | Ir a Dashboard | `10_dashboard_models.png` |
| 7.2 | Verificar dropdown actualizado | `11_model_selector.png` |
| 7.3 | Ejecutar backtest visual | `12_backtest_running.png` |
| 7.4 | Ver resultados | `13_backtest_results.png` |
| 7.5 | Ver panel de aprobación | `14_approval_panel.png` |

---

### FASE 8: Punto de Aprobación Humana (FIN)
**Donde termina Claude, empieza el humano**

| Step | Estado | Acción Humana Requerida |
|------|--------|------------------------|
| 8.1 | Promotion proposal creado | ✅ |
| 8.2 | Backtest metrics visibles | ✅ |
| 8.3 | Panel de aprobación listo | ✅ |
| 8.4 | **APROBAR/RECHAZAR** | 👤 Usuario decide |

---

## 🔧 VERIFICACIONES TÉCNICAS PENDIENTES

### Tecnología de Inferencia
```
[ ] ¿Usa ONNX Runtime?
[ ] ¿Usa Stable Baselines 3 nativo?
[ ] ¿Modelo serializado con .zip (SB3) o .onnx?
```

**Archivos a revisar:**
- `services/inference_api/core/observation_builder.py`
- `airflow/dags/tasks/l5_inference_task.py`
- `src/inference/model_loader.py`

### Health Checks Requeridos
```bash
# PostgreSQL
docker exec usdcop-postgres-timescale pg_isready

# Redis
docker exec usdcop-redis redis-cli ping

# Airflow
curl http://localhost:8080/health

# MLflow
curl http://localhost:5001/health

# Dashboard
curl http://localhost:5000/api/health
```

---

## 📊 MÉTRICAS DE ÉXITO

| Fase | Criterio de Éxito |
|------|-------------------|
| L0 | macro_indicators_daily actualizado |
| L1 | inference_features_5m > 0 rows |
| L2 | Dataset en MinIO + DVC tracked |
| L3 | Modelo en MLflow registry |
| L4 | promotion_proposals con PENDING_APPROVAL |
| Frontend | Backtest visual funcionando |
| Final | Panel de aprobación visible |

---

## ⏱️ TIEMPO TOTAL ESTIMADO

| Fase | Tiempo |
|------|--------|
| Fase 0: Verificación | 10 min |
| Fase 1: Screenshots inicial | 5 min |
| Fase 2: L0 Data | 15 min |
| Fase 3: L1 Features | 10 min |
| Fase 4: L2 Dataset | 20 min |
| Fase 5: L3 Training | 30-60 min |
| Fase 6: L4 Backtest | 20 min |
| Fase 7: Frontend | 10 min |
| **TOTAL** | **~2 horas** |

---

## 🚀 SIGUIENTE PASO

Ejecutar **Fase 0: Verificación de Infraestructura**

```bash
# Comando inicial
docker ps --format "table {{.Names}}\t{{.Status}}"
```
