# Frontend - USD/COP Forecasting Dashboard

Este documento describe los contratos y estructura que el Frontend debe implementar para consumir el Backend del Pipeline de ML.

---

## Estructura Recomendada del Frontend

```
frontend/
├── src/
│   ├── contracts/                # Schemas Zod (SSOT con Backend)
│   │   ├── index.ts
│   │   ├── auth.ts               # LoginRequest, TokenResponse
│   │   ├── forecasts.ts          # ForecastResponse, Dashboard
│   │   ├── models.ts             # ModelMetrics, Comparison
│   │   ├── images.ts             # ImageMetadata
│   │   ├── health.ts             # HealthResponse
│   │   └── errors.ts             # ErrorResponse
│   │
│   ├── api/                      # Cliente API
│   │   ├── client.ts             # Axios/fetch configurado
│   │   ├── auth.ts               # login(), logout(), refreshToken()
│   │   ├── forecasts.ts          # getForecasts(), getDashboard()
│   │   ├── models.ts             # getModels(), getComparison()
│   │   └── images.ts             # getImage(), getBacktestChart()
│   │
│   ├── hooks/                    # React hooks
│   │   ├── useAuth.ts
│   │   ├── useForecast.ts
│   │   ├── useModels.ts
│   │   └── useHealth.ts
│   │
│   ├── components/               # Componentes UI
│   │   ├── Dashboard/
│   │   ├── ForecastTable/
│   │   ├── ModelHeatmap/
│   │   ├── BacktestChart/
│   │   └── ConsensusWidget/
│   │
│   ├── pages/                    # Páginas/Routes
│   │   ├── index.tsx             # Dashboard principal
│   │   ├── forecasts.tsx
│   │   ├── models.tsx
│   │   └── settings.tsx
│   │
│   └── lib/
│       └── utils.ts
│
├── package.json
├── tsconfig.json
└── .env.local
```

---

## Contratos Zod (Copiar de Backend)

Los contratos Zod deben ser **idénticos** a los schemas Pydantic del backend. Ver `CONTRACTS.md` en el backend para los schemas completos.

### contracts/index.ts
```typescript
// Re-exportar todos los contratos
export * from './auth';
export * from './forecasts';
export * from './models';
export * from './images';
export * from './health';
export * from './errors';
```

### contracts/forecasts.ts
```typescript
import { z } from 'zod';

export const ForecastResponseSchema = z.object({
  model: z.string(),
  horizon: z.number().int().min(1).max(30),
  prediction: z.number(),
  prediction_price: z.number().optional().nullable(),
  direction: z.enum(['up', 'down']),
  confidence: z.number().min(0).max(1),
  prediction_date: z.string(),
  generated_at: z.string().datetime(),
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

export type ForecastResponse = z.infer<typeof ForecastResponseSchema>;
export type ConsensusResponse = z.infer<typeof ConsensusResponseSchema>;
export type ForecastDashboard = z.infer<typeof ForecastDashboardSchema>;
```

### contracts/models.ts
```typescript
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
  overfitting_gap: z.number(),
});

export const ModelComparisonSchema = z.object({
  models: z.array(z.object({
    model_name: z.string(),
    model_type: z.string(),
    avg_da_test: z.number(),
    best_horizon: z.number().int(),
    best_da: z.number(),
  })),
  metrics_by_horizon: z.record(z.string(), z.array(ModelMetricsSchema)),
  best_model_overall: z.string(),
  best_model_by_horizon: z.record(z.string(), z.string()),
  generated_at: z.string().datetime(),
});

export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;
export type ModelComparison = z.infer<typeof ModelComparisonSchema>;
```

---

## Cliente API

### api/client.ts
```typescript
import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor para agregar JWT
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('access_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Interceptor para manejar errores
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expirado, redirigir a login
      localStorage.removeItem('access_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

### api/forecasts.ts
```typescript
import { apiClient } from './client';
import {
  ForecastDashboardSchema,
  ForecastResponseSchema,
  type ForecastDashboard,
  type ForecastResponse
} from '@/contracts/forecasts';
import { z } from 'zod';

export async function getForecastDashboard(): Promise<ForecastDashboard> {
  const response = await apiClient.get('/forecasts/dashboard');
  return ForecastDashboardSchema.parse(response.data);
}

export async function getForecasts(params?: {
  model?: string;
  horizon?: number;
  page?: number;
}): Promise<ForecastResponse[]> {
  const response = await apiClient.get('/forecasts/', { params });
  return z.array(ForecastResponseSchema).parse(response.data.forecasts);
}

export async function getConsensus(horizon: number) {
  const response = await apiClient.get('/forecasts/consensus', {
    params: { horizon }
  });
  return response.data;
}
```

### api/models.ts
```typescript
import { apiClient } from './client';
import {
  ModelComparisonSchema,
  type ModelComparison
} from '@/contracts/models';

export async function getModelComparison(): Promise<ModelComparison> {
  const response = await apiClient.get('/models/comparison');
  return ModelComparisonSchema.parse(response.data);
}

export async function getModelMetrics(modelName: string) {
  const response = await apiClient.get(`/models/${modelName}/metrics`);
  return response.data;
}

export async function getFeatureImportance(modelName: string, horizon: number) {
  const response = await apiClient.get(`/models/${modelName}/feature-importance`, {
    params: { horizon }
  });
  return response.data;
}
```

---

## React Hooks

### hooks/useForecast.ts
```typescript
import { useQuery } from '@tanstack/react-query';
import { getForecastDashboard, getForecasts } from '@/api/forecasts';

export function useForecastDashboard() {
  return useQuery({
    queryKey: ['forecast-dashboard'],
    queryFn: getForecastDashboard,
    refetchInterval: 60000, // Refrescar cada minuto
  });
}

export function useForecasts(params?: {
  model?: string;
  horizon?: number;
}) {
  return useQuery({
    queryKey: ['forecasts', params],
    queryFn: () => getForecasts(params),
  });
}
```

### hooks/useModels.ts
```typescript
import { useQuery } from '@tanstack/react-query';
import { getModelComparison, getModelMetrics } from '@/api/models';

export function useModelComparison() {
  return useQuery({
    queryKey: ['model-comparison'],
    queryFn: getModelComparison,
    staleTime: 5 * 60 * 1000, // 5 minutos
  });
}

export function useModelMetrics(modelName: string) {
  return useQuery({
    queryKey: ['model-metrics', modelName],
    queryFn: () => getModelMetrics(modelName),
    enabled: !!modelName,
  });
}
```

---

## Endpoints que el Frontend Consume

### Sin Autenticación
| Endpoint | Uso |
|----------|-----|
| `GET /health` | Verificar estado del sistema |
| `POST /auth/login` | Iniciar sesión |

### Con Autenticación (JWT Bearer)
| Endpoint | Uso |
|----------|-----|
| `GET /forecasts/dashboard` | Dashboard principal |
| `GET /forecasts/latest` | Últimas predicciones |
| `GET /forecasts/consensus?horizon=N` | Consenso para horizonte |
| `GET /models/comparison` | Comparación de modelos |
| `GET /models/{name}/metrics` | Métricas de un modelo |
| `GET /images/{id}` | Obtener imagen PNG |
| `GET /images/backtest/{model}/{horizon}` | Gráfico de backtest |

---

## Variables de Entorno

```env
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=USD/COP Forecasting
```

---

## Componentes Sugeridos

### 1. Dashboard Principal
- Widget de consenso
- Tabla de últimas predicciones
- Gráfico de precios históricos
- Indicador del mejor modelo

### 2. Página de Modelos
- Heatmap de DA por modelo × horizonte
- Ranking de modelos
- Detalles de cada modelo
- Feature importance

### 3. Página de Forecasts
- Tabla filtrable de predicciones
- Fan chart de predicciones
- Selector de horizonte
- Comparación de modelos

### 4. Visualizaciones
- Equity curves de backtest
- Drawdown charts
- Distribución de retornos

---

## Dependencias Recomendadas

```json
{
  "dependencies": {
    "zod": "^3.22.0",
    "axios": "^1.6.0",
    "@tanstack/react-query": "^5.0.0",
    "recharts": "^2.10.0",
    "date-fns": "^3.0.0"
  }
}
```

---

## Pruebas de Contratos

```typescript
// tests/contracts.test.ts
import { ForecastResponseSchema } from '@/contracts/forecasts';

describe('Contract Validation', () => {
  it('validates forecast response from API', async () => {
    const response = await fetch('/forecasts/latest');
    const data = await response.json();

    const result = ForecastResponseSchema.safeParse(data[0]);
    expect(result.success).toBe(true);
  });
});
```

---

## Notas de Integración

1. **Fechas**: El backend envía ISO 8601 strings, parsear con `new Date()` o `date-fns`
2. **Errores**: Siempre validar con Zod antes de usar los datos
3. **Cache**: Usar React Query para cache inteligente
4. **Auth**: Guardar token en localStorage, interceptor lo agrega automáticamente
5. **Refresh**: El dashboard se actualiza cada 60 segundos

---

## Diagrama de Flujo de Datos

```
Usuario → UI Component → React Hook → API Client → Backend
                                           ↓
Usuario ← UI Component ← React Hook ← Zod Validation ← Response
```
