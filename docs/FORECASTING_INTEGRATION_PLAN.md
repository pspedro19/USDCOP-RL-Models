# Forecasting Backend Integration Plan

## Executive Summary

Integrar el backend de Forecasting del "New Feature" folder al proyecto principal, aplicando:
- **SSOT** (Single Source of Truth)
- **DRY** (Don't Repeat Yourself)
- **Design Patterns**: Factory, Strategy, Repository
- **Clean Architecture**: Separation of concerns

---

## 1. Current State Analysis

### 1.1 Duplicates Identified

| Component | Location 1 | Location 2 | Resolution |
|-----------|-----------|-----------|------------|
| `database.py` | `services/common/database.py` | `New Feature/api/database.py` + `New Feature/src/database.py` | **Keep** `services/common/database.py` as SSOT |
| `mlflow_client.py` | `New Feature/src/mlops/mlflow_client.py` | `src/training/mlflow_signature.py` | **Merge** to `src/mlops/mlflow_client.py` |
| `minio_client.py` | `New Feature/src/mlops/minio_client.py` | `src/infrastructure/repositories/minio_repository.py` | **Keep** repository pattern, enhance |
| Model classes | `New Feature/src/models/*.py` | None (new) | **Create** `src/forecasting/models/` |

### 1.2 New Feature Components to Integrate

```
New Feature/consolidated_backend/
├── api/routers/               → services/inference_api/routers/forecasting/
│   ├── forecasts.py
│   ├── models.py
│   └── images.py
├── src/models/                → src/forecasting/models/
│   ├── factory.py
│   ├── ridge.py
│   ├── bayesian_ridge.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── catboost_model.py
│   └── hybrids.py
├── src/evaluation/            → src/forecasting/evaluation/
│   ├── backtest.py
│   ├── walk_forward_backtest.py
│   ├── purged_kfold.py
│   └── metrics.py
├── data-engineering/scrapers/ → airflow/dags/scrapers/
│   ├── scraper_banrep_selenium.py
│   └── ... (all scrapers)
└── pipelines/                 → scripts/forecasting/
    ├── run_hybrid_ensemble.py
    └── run_statistical_validation.py
```

---

## 2. Target Architecture

### 2.1 New Directory Structure

```
USDCOP-RL-Models/
├── src/
│   ├── core/
│   │   └── contracts/
│   │       ├── feature_contract.py    # SSOT: FEATURE_ORDER, OBSERVATION_DIM
│   │       └── forecast_contract.py   # NEW: Forecasting types
│   │
│   ├── forecasting/                   # NEW MODULE
│   │   ├── __init__.py
│   │   ├── models/                    # ML Models (Factory Pattern)
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # BaseModel ABC
│   │   │   ├── factory.py            # ModelFactory (SSOT for model creation)
│   │   │   ├── ridge.py
│   │   │   ├── bayesian_ridge.py
│   │   │   ├── ard.py
│   │   │   ├── xgboost.py
│   │   │   ├── lightgbm.py
│   │   │   ├── catboost.py
│   │   │   └── hybrids.py
│   │   ├── evaluation/               # Backtesting & Metrics
│   │   │   ├── __init__.py
│   │   │   ├── backtest.py
│   │   │   ├── walk_forward.py
│   │   │   ├── purged_kfold.py
│   │   │   └── metrics.py
│   │   ├── features/                 # Feature Engineering (delegates to src/features)
│   │   │   ├── __init__.py
│   │   │   └── transformer.py
│   │   └── inference/                # Prediction Service
│   │       ├── __init__.py
│   │       └── predictor.py
│   │
│   ├── mlops/                        # CONSOLIDATED (no duplicates)
│   │   ├── __init__.py
│   │   ├── mlflow_client.py          # MLflow experiment tracking
│   │   ├── minio_client.py           # MinIO artifact storage
│   │   └── model_registry.py         # Unified model registry
│   │
│   └── infrastructure/
│       └── repositories/
│           ├── minio_repository.py   # Repository Pattern for storage
│           └── forecast_repository.py # NEW: Forecast data access
│
├── services/
│   ├── common/
│   │   ├── database.py               # SSOT: Database connections
│   │   └── config.py                 # SSOT: Base settings
│   │
│   └── inference_api/
│       └── routers/
│           ├── forecasting/          # NEW: Forecasting endpoints
│           │   ├── __init__.py
│           │   ├── forecasts.py
│           │   ├── models.py
│           │   └── images.py
│           └── ... (existing routers)
│
├── airflow/dags/
│   ├── scrapers/                     # MOVED: Data scrapers
│   │   ├── __init__.py
│   │   ├── banrep.py
│   │   ├── investing.py
│   │   ├── dane.py
│   │   └── fred.py
│   └── l2_forecasting_pipeline.py    # NEW: Forecasting DAG
│
└── init-scripts/
    └── 15-forecasting-schema.sql     # NEW: BI schema for forecasts
```

### 2.2 Design Patterns Applied

| Pattern | Application | Location |
|---------|-------------|----------|
| **Factory** | Model creation | `src/forecasting/models/factory.py` |
| **Strategy** | Normalizers, Metrics | `src/features/normalizers/`, `src/forecasting/evaluation/metrics.py` |
| **Repository** | Data access | `src/infrastructure/repositories/` |
| **SSOT** | Contracts, Config | `src/core/contracts/`, `services/common/` |
| **Adapter** | Data sources | `src/feature_store/adapters.py` |
| **Singleton** | Database pool | `services/common/database.py` |

---

## 3. Database Schema

### 3.1 New Tables (init-scripts/15-forecasting-schema.sql)

```sql
-- Schema for Forecasting BI
CREATE SCHEMA IF NOT EXISTS bi;

-- Dimension: Models
CREATE TABLE bi.dim_models (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(20) NOT NULL, -- linear, boosting, hybrid
    description TEXT,
    requires_scaling BOOLEAN DEFAULT FALSE,
    supports_early_stopping BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Dimension: Horizons
CREATE TABLE bi.dim_horizons (
    horizon_id INTEGER PRIMARY KEY,
    horizon_label VARCHAR(20) NOT NULL, -- e.g., "5 days"
    horizon_category VARCHAR(20) NOT NULL, -- short, medium, long
    days INTEGER NOT NULL
);

-- Fact: Forecasts
CREATE TABLE bi.fact_forecasts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inference_date DATE NOT NULL,
    inference_week INTEGER NOT NULL,
    inference_year INTEGER NOT NULL,
    target_date DATE NOT NULL,
    model_id VARCHAR(50) REFERENCES bi.dim_models(model_id),
    horizon_id INTEGER REFERENCES bi.dim_horizons(horizon_id),
    base_price DECIMAL(12, 4) NOT NULL,
    predicted_price DECIMAL(12, 4) NOT NULL,
    predicted_return_pct DECIMAL(8, 4),
    price_change DECIMAL(12, 4),
    direction VARCHAR(4), -- UP, DOWN
    signal INTEGER, -- -1, 0, 1
    actual_price DECIMAL(12, 4),
    direction_correct BOOLEAN,
    minio_week_path TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(inference_date, model_id, horizon_id)
);

-- Fact: Consensus (aggregated across models)
CREATE TABLE bi.fact_consensus (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    inference_date DATE NOT NULL,
    horizon_id INTEGER REFERENCES bi.dim_horizons(horizon_id),
    avg_predicted_price DECIMAL(12, 4),
    median_predicted_price DECIMAL(12, 4),
    std_predicted_price DECIMAL(12, 4),
    consensus_direction VARCHAR(4),
    bullish_count INTEGER,
    bearish_count INTEGER,
    total_models INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(inference_date, horizon_id)
);

-- Fact: Model Metrics
CREATE TABLE bi.fact_model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    training_date DATE NOT NULL,
    model_id VARCHAR(50) REFERENCES bi.dim_models(model_id),
    horizon_id INTEGER REFERENCES bi.dim_horizons(horizon_id),
    direction_accuracy DECIMAL(6, 4),
    rmse DECIMAL(12, 4),
    mae DECIMAL(12, 4),
    r2 DECIMAL(6, 4),
    mape DECIMAL(8, 4),
    sample_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(training_date, model_id, horizon_id)
);

-- Indexes for performance
CREATE INDEX idx_forecasts_date ON bi.fact_forecasts(inference_date DESC);
CREATE INDEX idx_forecasts_model ON bi.fact_forecasts(model_id);
CREATE INDEX idx_forecasts_week ON bi.fact_forecasts(inference_year, inference_week);

-- Seed dimension data
INSERT INTO bi.dim_horizons (horizon_id, horizon_label, horizon_category, days) VALUES
(1, '1 day', 'short', 1),
(5, '5 days', 'short', 5),
(10, '10 days', 'medium', 10),
(15, '15 days', 'medium', 15),
(20, '20 days', 'medium', 20),
(25, '25 days', 'long', 25),
(30, '30 days', 'long', 30);

INSERT INTO bi.dim_models (model_id, model_name, model_type, requires_scaling, supports_early_stopping) VALUES
('ridge', 'Ridge Regression', 'linear', TRUE, FALSE),
('bayesian_ridge', 'Bayesian Ridge', 'linear', TRUE, FALSE),
('ard', 'ARD Regression', 'linear', TRUE, FALSE),
('xgboost_pure', 'XGBoost', 'boosting', FALSE, TRUE),
('lightgbm_pure', 'LightGBM', 'boosting', FALSE, TRUE),
('catboost_pure', 'CatBoost', 'boosting', FALSE, TRUE),
('hybrid_xgboost', 'XGBoost Hybrid', 'hybrid', TRUE, TRUE),
('hybrid_lightgbm', 'LightGBM Hybrid', 'hybrid', TRUE, TRUE),
('hybrid_catboost', 'CatBoost Hybrid', 'hybrid', TRUE, TRUE);
```

---

## 4. API Contracts

### 4.1 Backend Pydantic Schemas

```python
# services/inference_api/contracts/forecasting.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from enum import Enum

class ForecastDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"

class HorizonCategory(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

class Forecast(BaseModel):
    inference_date: date
    model_id: str
    horizon_id: int
    base_price: float
    predicted_price: float
    predicted_return_pct: Optional[float]
    direction: ForecastDirection
    signal: int  # -1, 0, 1

class ForecastListResponse(BaseModel):
    source: str
    count: int
    data: List[Forecast]

class ModelMetrics(BaseModel):
    model_id: str
    horizon_id: int
    direction_accuracy: float
    rmse: float
    mae: Optional[float]
    r2: Optional[float]

class ConsensusResponse(BaseModel):
    horizon_id: int
    avg_predicted_price: float
    consensus_direction: ForecastDirection
    bullish_count: int
    bearish_count: int
    total_models: int
```

### 4.2 Frontend Zod Schemas

```typescript
// lib/contracts/forecasting.contract.ts
import { z } from 'zod';

export const ForecastDirectionSchema = z.enum(['UP', 'DOWN']);
export type ForecastDirection = z.infer<typeof ForecastDirectionSchema>;

export const HorizonCategorySchema = z.enum(['short', 'medium', 'long']);
export type HorizonCategory = z.infer<typeof HorizonCategorySchema>;

export const ForecastSchema = z.object({
  inference_date: z.string(), // ISO date
  model_id: z.string(),
  horizon_id: z.number().int().positive(),
  base_price: z.number().positive(),
  predicted_price: z.number().positive(),
  predicted_return_pct: z.number().nullable(),
  direction: ForecastDirectionSchema,
  signal: z.number().int().min(-1).max(1),
});
export type Forecast = z.infer<typeof ForecastSchema>;

export const ForecastListResponseSchema = z.object({
  source: z.string(),
  count: z.number().int().nonnegative(),
  data: z.array(ForecastSchema),
});
export type ForecastListResponse = z.infer<typeof ForecastListResponseSchema>;

export const ModelMetricsSchema = z.object({
  model_id: z.string(),
  horizon_id: z.number().int(),
  direction_accuracy: z.number().min(0).max(100),
  rmse: z.number().nonnegative(),
  mae: z.number().nonnegative().nullable(),
  r2: z.number().nullable(),
});
export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;

// API endpoints mapping
export const FORECASTING_ENDPOINTS = {
  FORECASTS: '/api/v1/forecasting/forecasts',
  FORECASTS_LATEST: '/api/v1/forecasting/forecasts/latest',
  FORECASTS_CONSENSUS: '/api/v1/forecasting/forecasts/consensus',
  FORECASTS_BY_WEEK: (year: number, week: number) =>
    `/api/v1/forecasting/forecasts/by-week/${year}/${week}`,
  FORECASTS_BY_HORIZON: (horizon: number) =>
    `/api/v1/forecasting/forecasts/by-horizon/${horizon}`,
  MODELS: '/api/v1/forecasting/models',
  MODEL_DETAIL: (modelId: string) => `/api/v1/forecasting/models/${modelId}`,
  MODEL_COMPARISON: (modelId: string) => `/api/v1/forecasting/models/${modelId}/comparison`,
  MODEL_RANKING: '/api/v1/forecasting/models/ranking',
  IMAGES_BACKTEST: (model: string, horizon: number) =>
    `/api/v1/forecasting/images/backtest/${model}/${horizon}`,
  IMAGES_FORECAST: (model: string) =>
    `/api/v1/forecasting/images/forecast/${model}`,
  DASHBOARD: '/api/v1/forecasting/dashboard',
} as const;
```

---

## 5. Implementation Steps

### Phase 1: Infrastructure (Today)

1. ✅ Create `init-scripts/15-forecasting-schema.sql`
2. ✅ Create `src/forecasting/` module structure
3. ✅ Move ML models from New Feature
4. ✅ Consolidate database utilities

### Phase 2: Backend API (Today)

5. Create `services/inference_api/routers/forecasting/`
6. Adapt routers from New Feature (remove duplicates)
7. Create `services/inference_api/contracts/forecasting.py`
8. Register routers in `main.py`

### Phase 3: Data Layer (Tomorrow)

9. Create `src/infrastructure/repositories/forecast_repository.py`
10. Move scrapers to `airflow/dags/scrapers/`
11. Create `airflow/dags/l2_forecasting_pipeline.py`

### Phase 4: Frontend Integration (Tomorrow)

12. Create `lib/contracts/forecasting.contract.ts`
13. Create `lib/services/forecasting.service.ts`
14. Update `app/forecasting/page.tsx` to use backend
15. Add loading states, error handling

### Phase 5: Testing & Cleanup

16. Integration tests for API
17. E2E tests with frontend
18. Delete `New Feature/` folder after migration
19. Update documentation

---

## 6. Migration Checklist

- [ ] Database schema created and seeded
- [ ] ML models moved to `src/forecasting/models/`
- [ ] Database utilities consolidated (no duplicates)
- [ ] MLOps code consolidated (MLflow + MinIO)
- [ ] Forecasting routers created in inference_api
- [ ] Contracts defined (Pydantic + Zod)
- [ ] Frontend service created
- [ ] Frontend pages updated
- [ ] Scrapers moved to Airflow
- [ ] Integration tests passing
- [ ] New Feature folder deleted
- [ ] Documentation updated

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Data loss during migration | CSV fallback in routers |
| Breaking existing APIs | Backward compatible routes (`/v1/` + `/api/v1/`) |
| Frontend downtime | Feature flag for new backend |
| Missing dependencies | Requirements.txt audit |

---

## 8. Success Criteria

1. **Zero duplicates** in codebase
2. **All endpoints** return valid data
3. **Frontend** displays forecasts from backend API (not CSV)
4. **Latency** < 500ms for dashboard load
5. **Tests** pass (unit + integration)
