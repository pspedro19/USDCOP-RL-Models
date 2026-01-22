# Contratos Backend ↔ Frontend

Este documento define los contratos de datos entre el Backend (FastAPI + ML Pipeline) y el Frontend. Sigue el principio **SSOT (Single Source of Truth)** donde los schemas Pydantic del backend son la fuente de verdad.

---

## Arquitectura de Contratos

```
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND                                  │
│  api/schemas/*.py (Pydantic)  ←── SSOT ──→  pipelines/results/  │
│         ↓                                                        │
│    JSON Responses                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST + JWT
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│  contracts/*.ts (Zod)  ←── DEBE COINCIDIR ──→  api/schemas/*.py │
│         ↓                                                        │
│    Type-safe API calls                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Autenticación

### Backend (Pydantic)
```python
# api/schemas/auth.py
from pydantic import BaseModel
from datetime import datetime

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime
```

### Frontend (Zod)
```typescript
// contracts/auth.ts
import { z } from 'zod';

export const LoginRequestSchema = z.object({
  username: z.string().min(1),
  password: z.string().min(1),
});

export const TokenResponseSchema = z.object({
  access_token: z.string(),
  token_type: z.literal('bearer').default('bearer'),
  expires_at: z.string().datetime(),
});

export const UserResponseSchema = z.object({
  id: z.number().int(),
  username: z.string(),
  email: z.string().email(),
  is_active: z.boolean(),
  created_at: z.string().datetime(),
});

// Types inferidos
export type LoginRequest = z.infer<typeof LoginRequestSchema>;
export type TokenResponse = z.infer<typeof TokenResponseSchema>;
export type UserResponse = z.infer<typeof UserResponseSchema>;
```

### Endpoints
| Endpoint | Método | Auth | Request | Response |
|----------|--------|------|---------|----------|
| `/auth/login` | POST | No | `LoginRequest` | `TokenResponse` |
| `/auth/logout` | POST | Sí | - | `{message: string}` |
| `/auth/me` | GET | Sí | - | `UserResponse` |
| `/auth/refresh` | POST | Sí | - | `TokenResponse` |

---

## 2. Forecasts (Predicciones)

### Backend (Pydantic)
```python
# api/schemas/forecasts.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime, date

class ForecastResponse(BaseModel):
    """Predicción individual de un modelo."""
    model: str = Field(..., description="Nombre del modelo (ridge, hybrid_xgboost, etc)")
    horizon: int = Field(..., ge=1, le=30, description="Horizonte en días")
    prediction: float = Field(..., description="Predicción de log-return")
    prediction_price: Optional[float] = Field(None, description="Precio predicho (si disponible)")
    direction: Literal["up", "down"] = Field(..., description="Dirección predicha")
    confidence: float = Field(..., ge=0, le=1, description="Confianza [0-1]")
    prediction_date: date = Field(..., description="Fecha objetivo de la predicción")
    generated_at: datetime = Field(..., description="Timestamp de generación")

class ForecastListResponse(BaseModel):
    """Lista de predicciones."""
    forecasts: List[ForecastResponse]
    total: int
    page: int = 1
    page_size: int = 50

class ConsensusResponse(BaseModel):
    """Predicción de consenso (promedio ponderado)."""
    consensus_prediction: float
    consensus_direction: Literal["up", "down"]
    confidence: float
    models_used: List[str]
    weights: dict[str, float]
    horizon: int
    generated_at: datetime

class ForecastDashboard(BaseModel):
    """Datos para dashboard principal."""
    latest_forecasts: List[ForecastResponse]
    consensus: ConsensusResponse
    best_model: str
    best_model_da: float
    avg_da_all_models: float
    last_actual_price: float
    last_actual_date: date
    generated_at: datetime
```

### Frontend (Zod)
```typescript
// contracts/forecasts.ts
import { z } from 'zod';

export const ForecastResponseSchema = z.object({
  model: z.string(),
  horizon: z.number().int().min(1).max(30),
  prediction: z.number(),
  prediction_price: z.number().optional().nullable(),
  direction: z.enum(['up', 'down']),
  confidence: z.number().min(0).max(1),
  prediction_date: z.string(), // ISO date string
  generated_at: z.string().datetime(),
});

export const ForecastListResponseSchema = z.object({
  forecasts: z.array(ForecastResponseSchema),
  total: z.number().int(),
  page: z.number().int().default(1),
  page_size: z.number().int().default(50),
});

export const ConsensusResponseSchema = z.object({
  consensus_prediction: z.number(),
  consensus_direction: z.enum(['up', 'down']),
  confidence: z.number(),
  models_used: z.array(z.string()),
  weights: z.record(z.string(), z.number()),
  horizon: z.number().int(),
  generated_at: z.string().datetime(),
});

export const ForecastDashboardSchema = z.object({
  latest_forecasts: z.array(ForecastResponseSchema),
  consensus: ConsensusResponseSchema,
  best_model: z.string(),
  best_model_da: z.number(),
  avg_da_all_models: z.number(),
  last_actual_price: z.number(),
  last_actual_date: z.string(),
  generated_at: z.string().datetime(),
});

// Types
export type ForecastResponse = z.infer<typeof ForecastResponseSchema>;
export type ForecastListResponse = z.infer<typeof ForecastListResponseSchema>;
export type ConsensusResponse = z.infer<typeof ConsensusResponseSchema>;
export type ForecastDashboard = z.infer<typeof ForecastDashboardSchema>;
```

### Endpoints
| Endpoint | Método | Auth | Query Params | Response |
|----------|--------|------|--------------|----------|
| `/forecasts/` | GET | Sí | `model, horizon, page, page_size` | `ForecastListResponse` |
| `/forecasts/latest` | GET | Sí | `limit` | `List[ForecastResponse]` |
| `/forecasts/consensus` | GET | Sí | `horizon` | `ConsensusResponse` |
| `/forecasts/dashboard` | GET | Sí | - | `ForecastDashboard` |
| `/forecasts/by-horizon/{horizon}` | GET | Sí | - | `List[ForecastResponse]` |

---

## 3. Models (Métricas de Modelos)

### Backend (Pydantic)
```python
# api/schemas/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class ModelMetrics(BaseModel):
    """Métricas de un modelo para un horizonte."""
    model_name: str
    model_type: Literal["linear", "boosting", "hybrid"]
    horizon: int
    da_train: float = Field(..., description="Direction Accuracy train (%)")
    da_test: float = Field(..., description="Direction Accuracy test (%)")
    variance_ratio: float = Field(..., description="pred_std / true_std")
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    overfitting_gap: float = Field(..., description="da_train - da_test")

class ModelSummary(BaseModel):
    """Resumen de un modelo."""
    model_name: str
    model_type: str
    avg_da_test: float
    best_horizon: int
    best_da: float
    requires_scaling: bool
    supports_early_stopping: bool

class ModelComparison(BaseModel):
    """Comparación entre todos los modelos."""
    models: List[ModelSummary]
    metrics_by_horizon: Dict[int, List[ModelMetrics]]
    best_model_overall: str
    best_model_by_horizon: Dict[int, str]
    generated_at: datetime

class FeatureImportance(BaseModel):
    """Importancia de features para un modelo."""
    model_name: str
    horizon: int
    features: Dict[str, float]  # {feature_name: importance}
    top_n: int = 15
```

### Frontend (Zod)
```typescript
// contracts/models.ts
import { z } from 'zod';

export const ModelMetricsSchema = z.object({
  model_name: z.string(),
  model_type: z.enum(['linear', 'boosting', 'hybrid']),
  horizon: z.number().int(),
  da_train: z.number(),
  da_test: z.number(),
  variance_ratio: z.number(),
  rmse: z.number().optional().nullable(),
  mae: z.number().optional().nullable(),
  r2: z.number().optional().nullable(),
  sharpe: z.number().optional().nullable(),
  max_drawdown: z.number().optional().nullable(),
  overfitting_gap: z.number(),
});

export const ModelSummarySchema = z.object({
  model_name: z.string(),
  model_type: z.string(),
  avg_da_test: z.number(),
  best_horizon: z.number().int(),
  best_da: z.number(),
  requires_scaling: z.boolean(),
  supports_early_stopping: z.boolean(),
});

export const ModelComparisonSchema = z.object({
  models: z.array(ModelSummarySchema),
  metrics_by_horizon: z.record(z.string(), z.array(ModelMetricsSchema)),
  best_model_overall: z.string(),
  best_model_by_horizon: z.record(z.string(), z.string()),
  generated_at: z.string().datetime(),
});

export const FeatureImportanceSchema = z.object({
  model_name: z.string(),
  horizon: z.number().int(),
  features: z.record(z.string(), z.number()),
  top_n: z.number().int().default(15),
});

// Types
export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;
export type ModelSummary = z.infer<typeof ModelSummarySchema>;
export type ModelComparison = z.infer<typeof ModelComparisonSchema>;
export type FeatureImportance = z.infer<typeof FeatureImportanceSchema>;
```

### Endpoints
| Endpoint | Método | Auth | Response |
|----------|--------|------|----------|
| `/models/` | GET | Sí | `List[ModelSummary]` |
| `/models/{model_name}` | GET | Sí | `ModelSummary` |
| `/models/{model_name}/metrics` | GET | Sí | `List[ModelMetrics]` |
| `/models/comparison` | GET | Sí | `ModelComparison` |
| `/models/{model_name}/feature-importance` | GET | Sí | `FeatureImportance` |

---

## 4. Images (Visualizaciones)

### Backend (Pydantic)
```python
# api/schemas/images.py
from pydantic import BaseModel
from typing import List, Literal
from datetime import datetime

class ImageMetadata(BaseModel):
    """Metadatos de una imagen."""
    image_id: str
    image_type: Literal[
        "model_ranking", "metrics_heatmap", "backtest_equity",
        "backtest_drawdown", "forecast_fan_chart", "feature_importance"
    ]
    model_name: Optional[str] = None
    horizon: Optional[int] = None
    generated_at: datetime
    file_size_bytes: int
    dimensions: tuple[int, int]  # (width, height)

class ImageListResponse(BaseModel):
    """Lista de imágenes disponibles."""
    images: List[ImageMetadata]
    total: int
```

### Frontend (Zod)
```typescript
// contracts/images.ts
import { z } from 'zod';

export const ImageMetadataSchema = z.object({
  image_id: z.string(),
  image_type: z.enum([
    'model_ranking', 'metrics_heatmap', 'backtest_equity',
    'backtest_drawdown', 'forecast_fan_chart', 'feature_importance'
  ]),
  model_name: z.string().optional().nullable(),
  horizon: z.number().int().optional().nullable(),
  generated_at: z.string().datetime(),
  file_size_bytes: z.number().int(),
  dimensions: z.tuple([z.number().int(), z.number().int()]),
});

export const ImageListResponseSchema = z.object({
  images: z.array(ImageMetadataSchema),
  total: z.number().int(),
});

// Types
export type ImageMetadata = z.infer<typeof ImageMetadataSchema>;
export type ImageListResponse = z.infer<typeof ImageListResponseSchema>;
```

### Endpoints
| Endpoint | Método | Auth | Response |
|----------|--------|------|----------|
| `/images/` | GET | Sí | `ImageListResponse` |
| `/images/{image_id}` | GET | Sí | `image/png` (binary) |
| `/images/models` | GET | Sí | `ImageListResponse` |
| `/images/backtest/{model}/{horizon}` | GET | Sí | `image/png` (binary) |

---

## 5. Health (Estado del Sistema)

### Backend (Pydantic)
```python
# api/schemas/health.py
from pydantic import BaseModel
from typing import Dict, Literal
from datetime import datetime

class ServiceHealth(BaseModel):
    """Estado de un servicio individual."""
    name: str
    status: Literal["healthy", "degraded", "unhealthy"]
    latency_ms: Optional[float] = None
    message: Optional[str] = None

class HealthResponse(BaseModel):
    """Estado general del sistema."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    environment: str
    services: Dict[str, ServiceHealth]
    timestamp: datetime
```

### Frontend (Zod)
```typescript
// contracts/health.ts
import { z } from 'zod';

export const ServiceHealthSchema = z.object({
  name: z.string(),
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  latency_ms: z.number().optional().nullable(),
  message: z.string().optional().nullable(),
});

export const HealthResponseSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  version: z.string(),
  environment: z.string(),
  services: z.record(z.string(), ServiceHealthSchema),
  timestamp: z.string().datetime(),
});

// Types
export type ServiceHealth = z.infer<typeof ServiceHealthSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;
```

### Endpoints (SIN AUTH)
| Endpoint | Método | Auth | Response |
|----------|--------|------|----------|
| `/health` | GET | No | `HealthResponse` |
| `/health/ready` | GET | No | `HealthResponse` |
| `/health/live` | GET | No | `{status: "ok"}` |

---

## 6. Errores (Respuestas de Error)

### Backend (Pydantic)
```python
# api/schemas/errors.py
from pydantic import BaseModel
from typing import Optional, List

class ErrorDetail(BaseModel):
    """Detalle de un error."""
    field: Optional[str] = None
    message: str
    code: str

class ErrorResponse(BaseModel):
    """Respuesta estándar de error."""
    error: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    status_code: int
    timestamp: datetime
```

### Frontend (Zod)
```typescript
// contracts/errors.ts
import { z } from 'zod';

export const ErrorDetailSchema = z.object({
  field: z.string().optional().nullable(),
  message: z.string(),
  code: z.string(),
});

export const ErrorResponseSchema = z.object({
  error: z.string(),
  message: z.string(),
  details: z.array(ErrorDetailSchema).optional().nullable(),
  status_code: z.number().int(),
  timestamp: z.string().datetime(),
});

// Type
export type ErrorResponse = z.infer<typeof ErrorResponseSchema>;
```

### Códigos de Error HTTP
| Código | Significado | Uso |
|--------|-------------|-----|
| 400 | Bad Request | Parámetros inválidos |
| 401 | Unauthorized | Token faltante/expirado |
| 403 | Forbidden | Sin permisos |
| 404 | Not Found | Recurso no existe |
| 422 | Validation Error | Error de validación Pydantic |
| 500 | Internal Server Error | Error del servidor |

---

## 7. Archivos CSV Generados

El pipeline genera archivos CSV que pueden ser consumidos directamente por el frontend para dashboards BI.

### metrics.csv
```csv
model,horizon,da_train,da_test,variance_ratio,rmse,mae
ridge,1,54.85,51.40,0.440,0.0089,0.0072
ridge,5,62.79,58.39,2.903,0.0198,0.0156
hybrid_xgboost,15,71.23,68.54,1.245,0.0312,0.0245
```

### forecasts.csv
```csv
date,model,horizon,prediction,direction,confidence
2026-01-15,ridge,1,0.0023,up,0.65
2026-01-15,ridge,5,0.0089,up,0.72
2026-01-15,hybrid_lightgbm,10,-0.0045,down,0.58
```

### summary.json
```json
{
  "generated_at": "2026-01-06T19:16:25.338231",
  "best_model_overall": "hybrid_lightgbm",
  "avg_da_by_model": {
    "ridge": 60.31,
    "bayesian_ridge": 60.36,
    "hybrid_lightgbm": 60.41
  },
  "horizons": [1, 5, 10, 15, 20, 25, 30],
  "n_models": 9
}
```

---

## Validación de Contratos

### Backend → Pruebas Pydantic
```python
# tests/unit/test_schemas.py
from api.schemas.forecasts import ForecastResponse

def test_forecast_response_valid():
    data = {
        "model": "ridge",
        "horizon": 5,
        "prediction": 0.0023,
        "direction": "up",
        "confidence": 0.65,
        "prediction_date": "2026-01-20",
        "generated_at": "2026-01-15T10:30:00"
    }
    response = ForecastResponse(**data)
    assert response.model == "ridge"
    assert response.direction == "up"
```

### Frontend → Pruebas Zod
```typescript
// tests/contracts/forecasts.test.ts
import { ForecastResponseSchema } from '@/contracts/forecasts';

test('validates forecast response', () => {
  const data = {
    model: 'ridge',
    horizon: 5,
    prediction: 0.0023,
    direction: 'up',
    confidence: 0.65,
    prediction_date: '2026-01-20',
    generated_at: '2026-01-15T10:30:00Z',
  };

  const result = ForecastResponseSchema.safeParse(data);
  expect(result.success).toBe(true);
});
```

---

## Notas de Implementación

1. **Fechas**: Backend usa `datetime`, frontend recibe strings ISO 8601
2. **Nullables**: Usar `Optional[T]` en Python, `z.optional().nullable()` en Zod
3. **Enums**: Usar `Literal` en Python, `z.enum()` en Zod
4. **Números**: Python `float`/`int` → TypeScript `number`
5. **Dict/Record**: Python `Dict[str, T]` → TypeScript `z.record(z.string(), T)`

---

## Versionado de API

```
/api/v1/forecasts/
/api/v1/models/
```

Los contratos deben ser versionados junto con la API.
