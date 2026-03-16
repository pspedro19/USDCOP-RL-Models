# Forecasting Pipeline Analysis & Recommendations

## Executive Summary

**Pregunta**: ¿Cómo se genera el forecasting semanal, cómo se entrena, qué consume el frontend, CSV o DB?

**Respuesta Corta**:
- **Actualmente**: Pipeline manual → CSV → Frontend lee CSV directo
- **Recomendación**: Pipeline automatizado → PostgreSQL → API → Frontend

---

## 1. Estado Actual del Pipeline

### 1.1 Flujo de Datos Actual

```
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE ACTUAL (Manual - NewFeature/)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Manual] run_hybrid_improved.py                                    │
│     ├─ 9 modelos × 7 horizontes = 63 .pkl files                    │
│     └─ outputs/runs/YYYYMMDD_HHMMSS/models/*.pkl                   │
│                                   ↓                                 │
│  [Manual] run_forward_forecast.py                                   │
│     ├─ Carga 63 modelos                                            │
│     ├─ Predice con X_latest                                        │
│     ├─ Genera 3 ensembles                                          │
│     └─ Sube imágenes a MinIO                                       │
│                                   ↓                                 │
│  [Manual] regenerate_bi_dashboard.py                                │
│     └─ Consolida en bi_dashboard_unified.csv                       │
│                                   ↓                                 │
│  [Frontend] Lee CSV directamente                                    │
│     └─ fetch('/forecasting/bi_dashboard_unified.csv')              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Modelos Entrenados

| Tipo | Modelos | Características |
|------|---------|-----------------|
| **Linear** | Ridge, Bayesian Ridge, ARD | `requires_scaling=True` |
| **Boosting** | XGBoost, LightGBM, CatBoost | `supports_early_stopping=True` |
| **Hybrid** | XGBoost→Ridge, LightGBM→Ridge, CatBoost→Ridge | Clasificador + Regresor |

**Horizontes**: 1, 5, 10, 15, 20, 25, 30 días

### 1.3 Métricas de Evaluación

- **Direction Accuracy (DA)**: % aciertos en dirección
- **RMSE/MAE**: Error en precio
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Peor caída pico-valle
- **Walk-Forward Validation**: 5 ventanas expansivas

---

## 2. Problemas Identificados

### 2.1 CSV vs Database

| Aspecto | CSV (Actual) | Database (Propuesto) |
|---------|--------------|---------------------|
| **Queries** | ❌ No posibles | ✅ SQL flexible |
| **Histórico** | ❌ Solo 2 semanas | ✅ Ilimitado |
| **Índices** | ❌ Scan completo | ✅ Optimizado |
| **Concurrencia** | ❌ Bloqueos | ✅ MVCC |
| **Triggers** | ❌ Manual | ✅ Auto-consensus |
| **Joins** | ❌ No posible | ✅ Con trades, etc. |
| **Auditoría** | ❌ Sin tracking | ✅ timestamps |

**Veredicto: Usar DATABASE**

### 2.2 Código Duplicado

```
Duplicaciones encontradas:
├─ walk_forward_backtest.py (NewFeature/ vs src/forecasting/evaluation/)
├─ minio_client.py (NewFeature/src/mlops/ vs src/mlops/)
├─ metrics.py (calculaciones manuales vs src/forecasting/evaluation/)
└─ model registry (manual vs src/ml_workflow/model_registry.py)
```

### 2.3 Sin Automatización

- ❌ No hay DAG de Airflow para forecasting
- ❌ Ejecución manual cada semana
- ❌ No integrado con MLflow
- ❌ No usa ModelRegistry del proyecto principal

### 2.4 Frontend Acoplado a CSV

```typescript
// Actual: Acoplado a CSV
const csvUrl = '/forecasting/bi_dashboard_unified.csv';
const response = await fetch(csvUrl);
Papa.parse(response.text(), {...});

// Propuesto: API con fallback
const data = await forecastingService.getDashboard();
// Service maneja: API → DB → CSV fallback
```

---

## 3. Infraestructura Reutilizable del Proyecto RL

### 3.1 Ya Existe y se Puede Reusar

| Componente | Ubicación | Uso para Forecasting |
|------------|-----------|---------------------|
| **TrainingEngine** | `src/training/engine.py` | Pattern para training |
| **ModelRegistry** | `src/ml_workflow/model_registry.py` | Registro de modelos |
| **MinIO Client** | `src/mlops/minio_client.py` | Storage de artefactos |
| **MLflow Integration** | `src/ml_workflow/` | Tracking de experimentos |
| **XCom Contracts** | `airflow/dags/contracts/` | Inter-DAG communication |
| **Feature Contract** | `src/core/contracts/` | SSOT para features |
| **DB Schema** | `init-scripts/15-forecasting-schema.sql` | Ya creado |

### 3.2 DAGs Existentes como Template

```python
# l3_model_training.py - Pattern reutilizable:
- Thin wrapper around Engine
- XCom contracts para lineage
- MLflow tracking
- MinIO-first architecture
- Experiment config YAML

# l5_multi_model_inference.py - Pattern reutilizable:
- Event-driven con Sensor
- Contract validation
- Multi-model inference
- PostgreSQL output
- Redis streaming
```

---

## 4. Arquitectura Propuesta

### 4.1 Nuevo Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────────┐
│ PIPELINE PROPUESTO (Automatizado)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Airflow] L3b: Forecasting Training (Mensual)                      │
│     ├─ Usa src/forecasting/models/ModelFactory                     │
│     ├─ Walk-forward via src/forecasting/evaluation/                │
│     ├─ Registra en MLflow + MinIO                                  │
│     └─ Insert métricas en bi.fact_model_metrics                    │
│                                   ↓                                 │
│  [Airflow] L5b: Forecasting Inference (Semanal)                     │
│     ├─ Carga modelos de ModelRegistry                              │
│     ├─ Predice para 7 horizontes                                   │
│     ├─ Calcula 3 ensembles                                         │
│     ├─ INSERT INTO bi.fact_forecasts                               │
│     └─ Trigger auto-calcula bi.fact_consensus                      │
│                                   ↓                                 │
│  [API] GET /api/v1/forecasting/dashboard                            │
│     └─ Query PostgreSQL bi.v_latest_forecasts                      │
│                                   ↓                                 │
│  [Frontend] forecastingService.getDashboard()                       │
│     └─ Consume API con fallback a CSV                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Database Schema (Ya Existe)

```sql
-- bi.fact_forecasts: Predicciones individuales
-- bi.fact_consensus: Consenso por horizonte (auto-trigger)
-- bi.fact_model_metrics: Métricas de walk-forward
-- bi.dim_models: Catálogo de modelos
-- bi.dim_horizons: Catálogo de horizontes
```

### 4.3 Componentes a Crear

```
airflow/dags/
├── l3b_forecasting_training.py      ← NUEVO: Training mensual
└── l5b_forecasting_inference.py     ← NUEVO: Inference semanal

src/forecasting/
├── models/                           ✅ Ya existe
│   └── factory.py                    ✅ ModelFactory
├── evaluation/                       ✅ Ya existe
│   ├── metrics.py                    ✅ Metrics
│   ├── backtest.py                   ✅ BacktestEngine
│   └── walk_forward.py               ✅ WalkForwardValidator
└── engine.py                         ← NUEVO: ForecastingEngine
```

---

## 5. Plan de Migración

### Fase 1: Consolidar Código (1 semana)

1. **Eliminar duplicados en NewFeature/**
   - Usar `src/mlops/minio_client.py` (SSOT)
   - Usar `src/forecasting/evaluation/` (SSOT)

2. **Crear ForecastingEngine**
   ```python
   # src/forecasting/engine.py
   class ForecastingEngine:
       def __init__(self, model_factory: ModelFactory):
           self.model_factory = model_factory

       def train(self, request: ForecastingTrainingRequest) -> TrainingResult:
           """Train all models for all horizons."""

       def predict(self, features: pd.DataFrame) -> Dict[str, Dict[int, float]]:
           """Generate predictions for all models/horizons."""

       def create_ensembles(self, predictions: Dict) -> Dict[str, Dict[int, float]]:
           """Create best-of-breed, top-3, top-6 ensembles."""
   ```

### Fase 2: Crear DAGs (1 semana)

1. **L3b: Forecasting Training**
   ```python
   # Schedule: 0 0 1 * * (Primer día del mes)
   # Tareas:
   #   1. Load features from PostgreSQL
   #   2. Train 9×7=63 models via ForecastingEngine
   #   3. Walk-forward validation
   #   4. Register in MLflow + MinIO
   #   5. Insert metrics to bi.fact_model_metrics
   ```

2. **L5b: Forecasting Inference**
   ```python
   # Schedule: 0 6 * * 0 (Domingos 6am)
   # Tareas:
   #   1. Load latest features
   #   2. Load models from ModelRegistry
   #   3. Predict via ForecastingEngine
   #   4. INSERT INTO bi.fact_forecasts
   #   5. Upload images to MinIO
   #   6. Trigger consensus calculation
   ```

### Fase 3: Migrar API a Database (3 días)

```python
# services/inference_api/routers/forecasting/forecasts.py

@router.get("/forecasts/latest")
async def get_latest_forecasts():
    # Primero: Intentar PostgreSQL
    try:
        return await query_postgres_forecasts()
    except Exception:
        # Fallback: CSV
        return load_csv_forecasts()
```

### Fase 4: Actualizar Frontend (2 días)

```typescript
// Ya está listo: forecastingService.getDashboard()
// Solo necesita: NEXT_PUBLIC_USE_FORECASTING_BACKEND=true
```

---

## 6. Decisión Técnica: CSV vs Database

### Recomendación Final: **USAR DATABASE + CSV FALLBACK**

```
┌─────────────────────────────────────────────────────────────────────┐
│ ARQUITECTURA HÍBRIDA (Mejor de ambos mundos)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PostgreSQL (Primary)           CSV (Fallback)                      │
│  ─────────────────────          ───────────────                     │
│  • Queries complejos            • Demo/desarrollo                   │
│  • Histórico completo           • Offline backup                    │
│  • Triggers automáticos         • Debug rápido                      │
│  • Joins con otras tablas       • CI/CD testing                     │
│  • Producción                   • Edge cases                        │
│                                                                     │
│  API detecta automáticamente:                                       │
│  if db_available: query_postgres()                                  │
│  else: load_csv()                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Beneficios Esperados

| Métrica | Actual | Propuesto |
|---------|--------|-----------|
| **Tiempo de ejecución** | Manual (~2h) | Automatizado (~30min) |
| **Histórico de predicciones** | 2 semanas | Ilimitado |
| **Queries complejos** | Imposible | SQL nativo |
| **Tracking de experimentos** | Ninguno | MLflow completo |
| **Versionado de modelos** | Manual .pkl | ModelRegistry |
| **Lineage** | Ninguno | Hash tracking |
| **Recuperación ante fallos** | Re-ejecutar todo | Re-ejecutar tarea |

---

## 8. Próximos Pasos Inmediatos

1. [ ] Crear `src/forecasting/engine.py` (ForecastingEngine)
2. [ ] Crear `airflow/dags/l3b_forecasting_training.py`
3. [ ] Crear `airflow/dags/l5b_forecasting_inference.py`
4. [ ] Migrar datos históricos de CSV a PostgreSQL
5. [ ] Actualizar API para usar PostgreSQL como primary
6. [ ] Habilitar `NEXT_PUBLIC_USE_FORECASTING_BACKEND=true`

---

## Appendix A: Archivos Clave

```
NewFeature/consolidated_backend/
├── scripts/
│   ├── run_forward_forecast.py      ← Generación de forecasts
│   ├── update_and_forecast.py       ← Update data + forecast
│   └── regenerate_bi_dashboard.py   ← CSV generation
├── pipelines/
│   └── run_hybrid_improved.py       ← Training 63 models
└── src/
    ├── features/common.py           ← prepare_features(), create_targets()
    └── evaluation/walk_forward_backtest.py  ← WF validation

src/ (Proyecto Principal - REUSAR)
├── forecasting/models/factory.py    ← ModelFactory (ya existe)
├── forecasting/evaluation/          ← Metrics, Backtest (ya existe)
├── ml_workflow/model_registry.py    ← ModelRegistry
├── training/engine.py               ← TrainingEngine (pattern)
└── mlops/minio_client.py            ← MinIO client (SSOT)

airflow/dags/
├── l3_model_training.py             ← Template para L3b
└── l5_multi_model_inference.py      ← Template para L5b
```

---

## Appendix B: Ensemble Strategies

```python
# 1. Best-of-Breed: Mejor modelo por horizonte
best_of_breed = {
    h: predictions[best_model_for_h][h]
    for h in [1, 5, 10, 15, 20, 25, 30]
}

# 2. Top-3: Promedio de 3 mejores modelos (global)
top_3 = {
    h: np.mean([predictions[m][h] for m in top_3_models])
    for h in horizons
}

# 3. Top-6: Promedio de 6 mejores modelos (global)
top_6 = {
    h: np.mean([predictions[m][h] for m in top_6_models])
    for h in horizons
}
```
