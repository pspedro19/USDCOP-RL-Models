# 🎬 DEBUG VIDEO: Ejecución Completa L0 → L4
## USDCOP RL Trading System - Full Pipeline Execution

**Fecha:** 2026-01-31
**Operador:** Claude Code
**Objetivo:** Ejecutar y documentar el flujo completo desde ingesta hasta producción

---

## 📋 CHECKLIST PRE-EJECUCIÓN

### Estado Inicial de Base de Datos
```
[ ] Tablas L0 (OHLCV, Macro)
[ ] Tablas L1 (inference_features_5m)
[ ] Tablas L2 (datasets)
[ ] Tablas L3 (model_registry)
[ ] Tablas L4 (promotion_proposals)
```

### Estado Inicial de Servicios
```
[ ] PostgreSQL/TimescaleDB
[ ] Redis
[ ] Airflow
[ ] MinIO
[ ] MLflow
[ ] Dashboard
```

---

## 🎬 ESCENA 1: Estado Inicial (ANTES)

### Timestamp: [PENDING]

**Comando:** Verificar estado inicial de todas las tablas

---

## 🎬 ESCENA 2: L0 - Data Acquisition

### 2.1 L0 Macro Update
**DAG:** `l0_macro_update`
**Propósito:** Actualizar datos macroeconómicos (FRED, DANE, Banrep)

### 2.2 L0 OHLCV Backfill
**DAG:** `l0_ohlcv_historical_backfill`
**Propósito:** Llenar datos OHLCV históricos

---

## 🎬 ESCENA 3: L1 - Feature Computation

### 3.1 L1 Feature Refresh
**DAG:** `v3.l1_feature_refresh`
**Propósito:** Calcular features y poblar `inference_features_5m`

---

## 🎬 ESCENA 4: L2 - Dataset Engineering

### 4.1 L2 Dataset Build
**DAG:** `rl_l2_01_dataset_build`
**Propósito:** Preprocesar datos, crear splits, DVC versioning, guardar en MinIO

---

## 🎬 ESCENA 5: L3 - Model Training

### 5.1 L3 Model Training
**DAG:** `rl_l3_01_model_training`
**Propósito:** Entrenar modelo PPO, registrar en MLflow

---

## 🎬 ESCENA 6: L4 - Backtest & Promotion

### 6.1 L4 Backtest Promotion
**DAG:** `rl_l4_04_backtest_promotion`
**Propósito:** Ejecutar backtest, generar proposal, habilitar en frontend

---

## 🎬 ESCENA 7: Frontend Verification

### 7.1 Dashboard Model Selector
**Verificar:** Dropdown actualizado con nuevo modelo

### 7.2 Two-Vote Approval
**Verificar:** Panel de aprobación funcional

---

## 📊 MÉTRICAS FINALES

| Etapa | Duración | Registros Creados | Estado |
|-------|----------|-------------------|--------|
| L0 Macro | - | - | PENDING |
| L0 OHLCV | - | - | PENDING |
| L1 Features | - | - | PENDING |
| L2 Dataset | - | - | PENDING |
| L3 Training | - | - | PENDING |
| L4 Backtest | - | - | PENDING |

---

## 🔧 TROUBLESHOOTING

(Se documentarán errores encontrados aquí)

---
