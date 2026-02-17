# PLAN ROBUSTO V2: Arquitectura Completa RL Pipeline

## RESUMEN EJECUTIVO

Después de analizar **40+ archivos** del proyecto, identifico:
- **UI funciona parcialmente**: dropdown de modelos, backtest streaming, replay
- **Backend tiene estructura**: DAGs L0-L6, ExperimentManager, PromotionService
- **Falta integración**: El frontend no está conectado al backend de promoción
- **Falta rigor**: No hay dual-vote, no hay persistencia de backtest, no hay sensor horario

---

## INVENTARIO ACTUAL (LO QUE EXISTE)

### Frontend (Dashboard)
| Componente | Estado | Archivo | Líneas |
|------------|--------|---------|--------|
| Dropdown modelos | ✅ Funciona | `/api/models/route.ts` | 1-250 |
| Backtest streaming | ✅ Funciona | `/api/backtest/stream/route.ts` | 1-150 |
| Progress bar + trades | ✅ Funciona | `BacktestControlPanel.tsx` | 1-600 |
| PromoteButton modal | ⚠️ Parcial | `PromoteButton.tsx` | 1-300 |
| Date presets SSOT | ✅ Funciona | `ssot.contract.ts` | 1-200 |
| Synthetic fallback | ✅ Funciona | `synthetic-backtest.service.ts` | 1-300 |

### Backend (Airflow DAGs)
| DAG | Estado | Archivo |
|-----|--------|---------|
| L0 Macro Update | ✅ Existe | `l0_macro_update.py` |
| L1 Feature Refresh | ✅ Existe | `l1_feature_refresh.py` |
| L2 Preprocessing | ✅ Existe | `l2_preprocessing_pipeline.py` |
| L3 Training | ⚠️ Parcial | `l3_model_training.py` (no TrainingEngine impl) |
| L4 Experiment Runner | ⚠️ Parcial | `l4_experiment_runner.py` |
| L4 Backtest Validation | ⚠️ Parcial | `l4_backtest_validation.py` |
| L5 Inference | ✅ Existe | `l5_multi_model_inference.py` |
| L6 Monitoring | ⚠️ Parcial | `l6_production_monitoring.py` |

### Database (40+ tablas)
| Tabla | Estado | Schema |
|-------|--------|--------|
| `model_registry` | ✅ Existe | public |
| `experiment_runs` | ✅ Existe | public |
| `experiment_comparisons` | ✅ Existe | public |
| `experiment_deployments` | ✅ Existe | public |
| `ml.lineage_records` | ✅ Existe | ml |
| `ml.model_promotion_audit` | ✅ Existe | ml |
| `audit.trades_audit` | ✅ Existe | audit |
| `promotion_requests` | ❌ NO EXISTE | - |
| `inference_features_production` | ❌ NO EXISTE | - |
| `backtest_results` | ❌ NO EXISTE | - |

### Services (Python)
| Servicio | Estado | Archivo |
|----------|--------|---------|
| ExperimentManager | ✅ Existe | `src/ml_workflow/experiment_manager.py` |
| PromotionService | ⚠️ Parcial | `src/ml_workflow/promotion_service.py` |
| CanonicalFeatureBuilder | ✅ Existe | `src/data/canonical_feature_builder.py` |
| DriftDetector | ❌ NO EXISTE | - |

---

## ACCIONES EXACTAS

### FASE 1: CREAR - Tablas de Base de Datos

#### 1.1 CREAR: `promotion_requests` (Dual-Vote System)

```sql
-- ARCHIVO: database/migrations/030_promotion_requests.sql

CREATE TABLE promotion_requests (
    id SERIAL PRIMARY KEY,
    request_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- Modelo a promover
    model_id VARCHAR(100) NOT NULL REFERENCES model_registry(model_id),
    model_version VARCHAR(100) NOT NULL,
    experiment_name VARCHAR(100),

    -- Backtest asociado
    backtest_id UUID,  -- FK a backtest_results
    backtest_period VARCHAR(20) CHECK (backtest_period IN ('validation', 'test', 'both')),

    -- VOTO AUTOMÁTICO
    auto_vote_status VARCHAR(20) DEFAULT 'pending'
        CHECK (auto_vote_status IN ('pending', 'passed', 'failed')),
    auto_vote_details JSONB,  -- {sharpe: 1.2, sharpe_pass: true, ...}
    auto_vote_at TIMESTAMPTZ,

    -- VOTO MANUAL
    manual_vote_status VARCHAR(20) DEFAULT 'pending'
        CHECK (manual_vote_status IN ('pending', 'approved', 'rejected')),
    manual_vote_by VARCHAR(100),
    manual_vote_at TIMESTAMPTZ,
    manual_vote_comment TEXT,

    -- Checklist (3 items)
    checklist_backtest_reviewed BOOLEAN DEFAULT FALSE,
    checklist_metrics_acceptable BOOLEAN DEFAULT FALSE,
    checklist_team_notified BOOLEAN DEFAULT FALSE,

    -- Resultado final
    final_decision VARCHAR(20) DEFAULT 'pending'
        CHECK (final_decision IN ('pending', 'promoted', 'rejected', 'override_promoted')),
    target_stage VARCHAR(20) CHECK (target_stage IN ('staging', 'production')),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    promoted_at TIMESTAMPTZ
);

CREATE INDEX idx_promo_req_model ON promotion_requests(model_id);
CREATE INDEX idx_promo_req_status ON promotion_requests(final_decision);
CREATE INDEX idx_promo_req_created ON promotion_requests(created_at DESC);

-- Trigger para updated_at
CREATE TRIGGER trg_promotion_requests_updated
    BEFORE UPDATE ON promotion_requests
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

**RAZÓN**: El sistema actual tiene `experiment_deployments` con `approval_required` pero no soporta dual-vote (auto + manual).

---

#### 1.2 CREAR: `backtest_results` (Persistencia de Backtests)

```sql
-- ARCHIVO: database/migrations/031_backtest_results.sql

CREATE TABLE backtest_results (
    id SERIAL PRIMARY KEY,
    backtest_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- Identificación
    model_id VARCHAR(100) NOT NULL REFERENCES model_registry(model_id),
    model_version VARCHAR(100),

    -- Período
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    period_type VARCHAR(20) CHECK (period_type IN ('validation', 'test', 'custom')),
    is_out_of_sample BOOLEAN DEFAULT TRUE,

    -- Métricas
    total_trades INTEGER,
    win_rate DECIMAL(5,4),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown_pct DECIMAL(5,4),
    total_pnl DECIMAL(12,2),
    total_return_pct DECIMAL(8,4),
    profit_factor DECIMAL(8,4),

    -- Detalle
    trades JSONB,  -- Array de trades
    equity_curve JSONB,  -- Array de equity snapshots
    daily_returns JSONB,  -- Array de daily returns

    -- Integridad
    signature VARCHAR(64),  -- SHA256 de trades para audit
    run_source VARCHAR(20) CHECK (run_source IN ('backend', 'frontend', 'synthetic')),

    -- Timestamps
    run_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100)
);

CREATE INDEX idx_backtest_model ON backtest_results(model_id);
CREATE INDEX idx_backtest_period ON backtest_results(period_type);
CREATE INDEX idx_backtest_run ON backtest_results(run_at DESC);
CREATE INDEX idx_backtest_sharpe ON backtest_results(sharpe_ratio DESC);
```

**RAZÓN**: Actualmente los backtests se ejecutan y se descartan. No hay persistencia para audit trail ni para vincular a promotion_requests.

---

#### 1.3 CREAR: `inference_features_production` (Tabla de Inferencia)

```sql
-- ARCHIVO: database/migrations/032_inference_features_production.sql

CREATE TABLE inference_features_production (
    id BIGSERIAL PRIMARY KEY,

    -- Temporal
    timestamp_utc TIMESTAMPTZ NOT NULL,
    bar_number INTEGER NOT NULL,  -- 1-60 dentro del día
    trading_date DATE NOT NULL,

    -- 13 Market Features (del CTR-FEATURE-001)
    log_ret_5m DOUBLE PRECISION,
    log_ret_1h DOUBLE PRECISION,
    log_ret_4h DOUBLE PRECISION,
    rsi_9 DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    dxy_z DOUBLE PRECISION,
    dxy_change_1d DOUBLE PRECISION,
    vix_z DOUBLE PRECISION,
    embi_z DOUBLE PRECISION,
    brent_change_1d DOUBLE PRECISION,
    rate_spread DOUBLE PRECISION,
    usdmxn_change_1d DOUBLE PRECISION,

    -- Feature adicionales raw
    close_price DECIMAL(12,4),
    dxy_raw DECIMAL(10,4),
    vix_raw DECIMAL(10,4),
    embi_raw DECIMAL(10,4),

    -- Metadata del modelo
    model_id VARCHAR(100) REFERENCES model_registry(model_id),
    dataset_hash VARCHAR(16),
    feature_order_hash VARCHAR(16),

    -- Quality flags
    macro_was_ffilled BOOLEAN DEFAULT FALSE,
    ohlcv_was_interpolated BOOLEAN DEFAULT FALSE,
    data_quality_score DECIMAL(3,2),

    UNIQUE(timestamp_utc, model_id)
);

CREATE INDEX idx_inf_feat_ts ON inference_features_production(timestamp_utc DESC);
CREATE INDEX idx_inf_feat_model ON inference_features_production(model_id);
CREATE INDEX idx_inf_feat_date ON inference_features_production(trading_date DESC);

-- Particionado por mes (opcional para performance)
-- CREATE TABLE inference_features_production PARTITION BY RANGE (trading_date);
```

**RAZÓN**: L5 actualmente lee de `usdcop_m5_ohlcv` + `macro_indicators_daily` directamente. No hay tabla consolidada con features preprocesadas.

---

### FASE 2: CREAR - API Endpoints

#### 2.1 CREAR: `/api/models/[modelId]/promote/route.ts`

```typescript
// ARCHIVO: usdcop-trading-dashboard/app/api/models/[modelId]/promote/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

const PromoteRequestSchema = z.object({
  target_stage: z.enum(['staging', 'production']),
  backtest_id: z.string().uuid().optional(),
  reason: z.string().min(10, 'Reason must be at least 10 characters'),
  promoted_by: z.string(),
  checklist: z.object({
    backtest_reviewed: z.boolean(),
    metrics_acceptable: z.boolean(),
    team_notified: z.boolean()
  })
});

export async function POST(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  const body = await request.json();
  const validation = PromoteRequestSchema.safeParse(body);

  if (!validation.success) {
    return NextResponse.json({ error: validation.error }, { status: 400 });
  }

  const { target_stage, backtest_id, reason, promoted_by, checklist } = validation.data;
  const modelId = params.modelId;

  // 1. Verificar que modelo existe
  // 2. Verificar que backtest existe y pertenece al modelo
  // 3. Calcular auto-vote basado en métricas
  // 4. Crear promotion_request
  // 5. Si auto_vote=passed Y checklist completo, marcar manual_vote como pending

  // Llamar al backend Python para crear la solicitud
  const backendUrl = process.env.INFERENCE_SERVICE_URL || 'http://inference-service:8001';
  const response = await fetch(`${backendUrl}/api/v1/models/${modelId}/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target_stage,
      backtest_id,
      reason,
      promoted_by,
      checklist
    })
  });

  if (!response.ok) {
    const error = await response.json();
    return NextResponse.json({ error }, { status: response.status });
  }

  const result = await response.json();
  return NextResponse.json(result);
}
```

---

#### 2.2 CREAR: `/api/models/[modelId]/auto-approve/route.ts`

```typescript
// ARCHIVO: usdcop-trading-dashboard/app/api/models/[modelId]/auto-approve/route.ts

// Evalúa métricas del backtest y otorga auto-vote
// Llama al PromotionService.validate_for_auto_vote()

export async function POST(
  request: NextRequest,
  { params }: { params: { modelId: string } }
) {
  const { backtest_id } = await request.json();

  // Fetch backtest metrics from DB
  // Compare against PROMOTION_THRESHOLDS from ssot
  // Return { auto_vote: 'passed' | 'failed', details: {...} }
}
```

---

#### 2.3 CREAR: `/api/backtest/save/route.ts`

```typescript
// ARCHIVO: usdcop-trading-dashboard/app/api/backtest/save/route.ts

// Persiste resultados del backtest a la tabla backtest_results
// Retorna backtest_uuid para vincular a promotion_request

export async function POST(request: NextRequest) {
  const { model_id, start_date, end_date, period_type, trades, summary } = await request.json();

  // Calcular signature (SHA256 de trades)
  // Insert into backtest_results
  // Return backtest_uuid
}
```

---

### FASE 3: MODIFICAR - Archivos Existentes

#### 3.1 MODIFICAR: `PromoteButton.tsx`

**Archivo**: `usdcop-trading-dashboard/components/models/PromoteButton.tsx`

**Cambios**:
```typescript
// LÍNEA 44: Mover thresholds a ssot.contract.ts
- const PROMOTION_THRESHOLDS = { staging: {...}, production: {...} };
+ import { PROMOTION_THRESHOLDS } from '@/lib/contracts/ssot.contract';

// LÍNEA 150-200: Agregar visualización de votos
+ <div className="vote-status">
+   <div className="auto-vote">
+     Auto-Vote: {promotionRequest?.auto_vote_status || 'pending'}
+     {promotionRequest?.auto_vote_status === 'passed' && <CheckIcon />}
+     {promotionRequest?.auto_vote_status === 'failed' && <XIcon />}
+   </div>
+   <div className="manual-vote">
+     Manual Vote: {promotionRequest?.manual_vote_status || 'pending'}
+   </div>
+ </div>

// LÍNEA 250: Cambiar endpoint
- const response = await fetch(`/api/v1/models/${modelId}/promote`, ...);
+ const response = await fetch(`/api/models/${modelId}/promote`, ...);

// LÍNEA 280: Agregar backtest_id al request
+ backtest_id: currentBacktestId,  // Del contexto o prop
```

---

#### 3.2 MODIFICAR: `BacktestControlPanel.tsx`

**Archivo**: `usdcop-trading-dashboard/components/trading/BacktestControlPanel.tsx`

**Cambios**:
```typescript
// LÍNEA 347-400: Eliminar lógica de promoción duplicada
- // Sección "Pasar a Producción" con botón directo
+ // Integrar con PromoteButton component
+ import { PromoteButton } from '@/components/models/PromoteButton';

// LÍNEA 450: Después de backtest completado, guardar resultados
+ useEffect(() => {
+   if (backtestState.status === 'completed' && backtestState.summary) {
+     saveBacktestResults({
+       model_id: selectedModel,
+       start_date: dateRange.start,
+       end_date: dateRange.end,
+       period_type: selectedPreset,
+       trades: backtestState.trades,
+       summary: backtestState.summary
+     }).then(setCurrentBacktestId);
+   }
+ }, [backtestState.status]);

// LÍNEA 528-571: Reemplazar botón de promoción
- <button onClick={handlePromote}>Pasar a Producción</button>
+ <PromoteButton
+   model={selectedModelData}
+   backtestId={currentBacktestId}
+   metrics={backtestState.summary}
+ />
```

---

#### 3.3 MODIFICAR: `ssot.contract.ts`

**Archivo**: `usdcop-trading-dashboard/lib/contracts/ssot.contract.ts`

**Agregar**:
```typescript
// LÍNEA ~100: Agregar promotion thresholds como SSOT

export const PROMOTION_THRESHOLDS = {
  staging: {
    min_sharpe: 0.5,
    min_win_rate: 0.45,
    max_drawdown: -0.15,
    min_trades: 50,
  },
  production: {
    min_sharpe: 1.0,
    min_win_rate: 0.50,
    max_drawdown: -0.10,
    min_trades: 100,
    min_staging_days: 7,
  },
} as const;

export const VOTE_REQUIREMENTS = {
  staging: { auto_votes: 1, manual_votes: 0 },  // Solo auto-vote
  production: { auto_votes: 1, manual_votes: 1 },  // Ambos
} as const;

// LÍNEA ~150: Agregar status de promoción
export const MODEL_PROMOTION_STATUSES = [
  'registered',   // Inicial
  'staging',      // Promovido a staging
  'deployed',     // En producción
  'retired',      // Retirado
  'rejected'      // Rechazado
] as const;
```

---

#### 3.4 MODIFICAR: `PromotionService` (Python)

**Archivo**: `src/ml_workflow/promotion_service.py`

**Agregar método para dual-vote**:
```python
# LÍNEA ~200: Agregar validate_for_auto_vote()

def validate_for_auto_vote(
    self,
    model_id: str,
    backtest_id: str,
    target_stage: str
) -> Dict[str, Any]:
    """
    Evalúa métricas del backtest contra thresholds.
    Retorna decisión de auto-vote.
    """
    # 1. Cargar métricas del backtest
    backtest = self._db_conn.execute(
        "SELECT * FROM backtest_results WHERE backtest_uuid = %s",
        (backtest_id,)
    ).fetchone()

    if not backtest:
        return {"auto_vote": "failed", "reason": "Backtest not found"}

    # 2. Cargar thresholds según target_stage
    thresholds = PROMOTION_THRESHOLDS[target_stage]

    # 3. Evaluar cada métrica
    checks = {
        "sharpe": backtest.sharpe_ratio >= thresholds["min_sharpe"],
        "win_rate": backtest.win_rate >= thresholds["min_win_rate"],
        "max_drawdown": backtest.max_drawdown_pct >= thresholds["max_drawdown"],
        "min_trades": backtest.total_trades >= thresholds["min_trades"],
    }

    all_passed = all(checks.values())

    # 4. Registrar en promotion_requests
    self._db_conn.execute("""
        UPDATE promotion_requests
        SET auto_vote_status = %s,
            auto_vote_details = %s,
            auto_vote_at = NOW()
        WHERE model_id = %s AND backtest_id = %s
    """, (
        "passed" if all_passed else "failed",
        json.dumps(checks),
        model_id,
        backtest_id
    ))

    return {
        "auto_vote": "passed" if all_passed else "failed",
        "details": checks,
        "metrics": {
            "sharpe": backtest.sharpe_ratio,
            "win_rate": backtest.win_rate,
            "max_drawdown": backtest.max_drawdown_pct,
            "total_trades": backtest.total_trades
        }
    }

# LÍNEA ~300: Agregar create_inference_table()

def create_inference_table(self, model_id: str, snapshot: ModelSnapshot):
    """
    Crea/pobla tabla de inferencia cuando modelo es promovido.

    Solo se ejecuta al aprobar promoción a production.
    """
    # 1. Cargar dataset de training del modelo
    dataset_path = self._get_training_dataset_path(model_id)
    dataset = pd.read_parquet(dataset_path)

    # 2. Aplicar preprocesamiento
    norm_stats = self._load_norm_stats(snapshot.norm_stats_hash)

    # 3. Calcular features con CanonicalFeatureBuilder
    from src.data.canonical_feature_builder import CanonicalFeatureBuilder
    builder = CanonicalFeatureBuilder(norm_stats)
    features_df = builder.build_features(dataset)

    # 4. Insertar en inference_features_production
    features_df['model_id'] = model_id
    features_df['feature_order_hash'] = snapshot.feature_order_hash
    features_df['dataset_hash'] = snapshot.dataset_hash[:16]

    self._bulk_insert_inference_features(features_df)

    logger.info(f"Created inference table for {model_id} with {len(features_df)} rows")
```

---

### FASE 4: CREAR - DAG de Sensor Horario

#### 4.1 CREAR: `l0_macro_hourly_inference.py`

**Archivo**: `airflow/dags/l0_macro_hourly_inference.py`

```python
"""
DAG: l0_macro_hourly_inference
==============================

Ejecuta CADA HORA durante horario de trading (13:00-17:00 UTC).
Consolida las 3 fuentes macro y actualiza tabla de inferencia.

Schedule: 0 13-17 * * 1-5
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

default_args = {
    'owner': 'usdcop',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

dag = DAG(
    'l0_macro_hourly_inference',
    default_args=default_args,
    description='Hourly macro consolidation for inference',
    schedule_interval='0 13-17 * * 1-5',  # Cada hora, 8-12 COT
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['l0', 'macro', 'inference', 'hourly'],
)

def consolidate_macro_sources(**context):
    """
    Consolida las 3 frecuencias de macro:
    - Diaria: DXY, VIX, EMBI, Brent
    - Mensual: CPI, Unemployment, Fed Funds
    - Trimestral: GDP
    """
    from src.data.macro_consolidator import MacroConsolidator

    consolidator = MacroConsolidator()

    # 1. Leer de las 3 tablas
    daily_df = consolidator.read_daily_macro()
    monthly_df = consolidator.read_monthly_macro()
    quarterly_df = consolidator.read_quarterly_macro()

    # 2. Aplicar reglas de publicación (delay)
    daily_df = consolidator.apply_publication_delay(daily_df, delay_days=1)
    monthly_df = consolidator.apply_publication_delay(monthly_df, delay_days=30)
    quarterly_df = consolidator.apply_publication_delay(quarterly_df, delay_days=45)

    # 3. Resamplear a 5-min
    resampled = consolidator.resample_to_5min(daily_df, monthly_df, quarterly_df)

    # 4. Forward-fill con límites
    resampled = consolidator.apply_ffill_with_limits(resampled, {
        'daily': 5,      # máx 5 días
        'monthly': 35,   # máx 35 días
        'quarterly': 95  # máx 95 días
    })

    # 5. Calcular features macro
    from src.data.canonical_feature_builder import CanonicalFeatureBuilder
    builder = CanonicalFeatureBuilder()
    macro_features = builder.calculate_macro_features(resampled)

    # 6. UPSERT en inference_features_production
    rows_updated = consolidator.upsert_inference_features(macro_features)

    context['ti'].xcom_push(key='rows_updated', value=rows_updated)
    return {"status": "success", "rows_updated": rows_updated}


consolidate_task = PythonOperator(
    task_id='consolidate_macro_sources',
    python_callable=consolidate_macro_sources,
    dag=dag,
)

# Validación post-consolidación
def validate_consolidation(**context):
    rows_updated = context['ti'].xcom_pull(key='rows_updated')
    if rows_updated < 1:
        raise ValueError("No rows updated - check data sources")
    return {"validation": "passed"}

validate_task = PythonOperator(
    task_id='validate_consolidation',
    python_callable=validate_consolidation,
    dag=dag,
)

consolidate_task >> validate_task
```

---

### FASE 5: ELIMINAR - Código Redundante

#### 5.1 ELIMINAR: Lógica de promoción duplicada en BacktestControlPanel

**Archivo**: `usdcop-trading-dashboard/components/trading/BacktestControlPanel.tsx`

**Eliminar líneas 528-571**:
```typescript
// ELIMINAR esta sección completa (reemplazada por PromoteButton integrado)
- const handlePromote = async () => {
-   setIsPromoting(true);
-   try {
-     const response = await fetch(`/api/models/${selectedModel}/promote`, {
-       method: 'POST',
-     });
-     ...
-   }
- };
-
- {backtestComplete && (
-   <button onClick={handlePromote}>
-     Pasar a Producción
-   </button>
- )}
```

---

#### 5.2 ELIMINAR: Thresholds hardcodeados

**Archivo**: `usdcop-trading-dashboard/components/models/PromoteButton.tsx`

**Eliminar líneas 44-65**:
```typescript
// ELIMINAR - ahora viene de ssot.contract.ts
- const PROMOTION_THRESHOLDS = {
-   staging: {
-     min_sharpe: 0.5,
-     min_win_rate: 0.45,
-     max_drawdown: -0.15,
-     min_trades: 50,
-   },
-   production: {
-     ...
-   }
- };
```

---

### FASE 6: VALIDACIÓN Y TESTING

#### 6.1 CREAR: Tests de integración

**Archivo**: `usdcop-trading-dashboard/tests/integration/promotion-flow.test.ts`

```typescript
describe('Promotion Flow Integration', () => {
  it('should create promotion request with auto-vote', async () => {
    // 1. Run backtest
    // 2. Save backtest results
    // 3. Create promotion request
    // 4. Verify auto-vote is calculated
  });

  it('should require manual vote for production', async () => {
    // 1. Create staging promotion (auto-vote only)
    // 2. Create production promotion
    // 3. Verify manual_vote_status = 'pending'
  });

  it('should promote to production with both votes', async () => {
    // 1. Create promotion request
    // 2. Auto-vote passes
    // 3. Manual vote approves
    // 4. Verify final_decision = 'promoted'
    // 5. Verify inference_features_production is populated
  });
});
```

---

## DIAGRAMA DE FLUJO FINAL

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLUJO COMPLETO CON DUAL-VOTE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐
│ L3: Training       │
│ (ya existe)        │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐     ┌────────────────────────────────────────────────┐
│ L4: Backtest       │     │ Dashboard (BacktestControlPanel)               │
│ Validation         │────▶│ - Ejecuta backtest streaming                   │
│ (ya existe)        │     │ - Muestra equity curve, trades                 │
└────────┬───────────┘     │ - NUEVO: Guarda en backtest_results            │
         │                 │ - NUEVO: Muestra PromoteButton integrado       │
         │                 └───────────────────────────┬────────────────────┘
         │                                             │
         │                                             ▼
         │                 ┌────────────────────────────────────────────────┐
         │                 │ PromoteButton (MODIFICADO)                     │
         │                 │ - Muestra métricas vs thresholds               │
         │                 │ - Muestra checklist (3 items)                  │
         │                 │ - NUEVO: Muestra estado de votos               │
         │                 │ - NUEVO: Vincula backtest_id                   │
         │                 └───────────────────────────┬────────────────────┘
         │                                             │
         │                                             ▼
         │                 ┌────────────────────────────────────────────────┐
         │                 │ /api/models/{id}/promote (NUEVO)               │
         │                 │ - Valida request (Zod)                         │
         │                 │ - Crea promotion_request                       │
         │                 │ - Llama auto-approve                           │
         │                 └───────────────────────────┬────────────────────┘
         │                                             │
         ▼                                             ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                       promotion_requests (NUEVA TABLA)                      │
│                                                                             │
│  ┌─────────────────────┐        ┌─────────────────────┐                    │
│  │ VOTO AUTOMÁTICO     │        │ VOTO MANUAL         │                    │
│  │                     │        │                     │                    │
│  │ PromotionService    │        │ Dashboard UI        │                    │
│  │ .validate_for_      │        │ - Ver métricas      │                    │
│  │   auto_vote()       │        │ - Completar checklist│                   │
│  │                     │        │ - Click "Aprobar"   │                    │
│  │ Checks:             │        │                     │                    │
│  │ - Sharpe >= 0.5/1.0 │        │ Acciones:           │                    │
│  │ - WinRate >= 45/50% │        │ - APROBAR           │                    │
│  │ - MaxDD >= -15/-10% │        │ - RECHAZAR          │                    │
│  │ - Trades >= 50/100  │        │ - Comentario        │                    │
│  │                     │        │                     │                    │
│  │ Result: PASS/FAIL   │        │ Result: APPROVED/   │                    │
│  │                     │        │         REJECTED    │                    │
│  └─────────┬───────────┘        └─────────┬───────────┘                    │
│            │                              │                                 │
│            └──────────────┬───────────────┘                                 │
│                           ▼                                                 │
│            ┌─────────────────────────┐                                      │
│            │ DECISIÓN FINAL          │                                      │
│            │                         │                                      │
│            │ staging:                │                                      │
│            │   auto=PASS → promoted  │                                      │
│            │                         │                                      │
│            │ production:             │                                      │
│            │   auto=PASS AND         │                                      │
│            │   manual=APPROVED       │                                      │
│            │   → promoted            │                                      │
│            │                         │                                      │
│            │   manual=APPROVED AND   │                                      │
│            │   auto=FAIL             │                                      │
│            │   → override_promoted   │                                      │
│            └─────────────────────────┘                                      │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        │ SI PROMOTED:
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ PromotionService.create_inference_table() (NUEVO)                          │
│                                                                             │
│ 1. Cargar dataset de training del modelo                                    │
│ 2. Aplicar CanonicalFeatureBuilder                                          │
│ 3. INSERT INTO inference_features_production                                │
│ 4. Actualizar model_registry.status = 'deployed'                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ L5: Inference (MODIFICADO)                                                  │
│                                                                             │
│ ANTES: Lee de usdcop_m5_ohlcv + macro_indicators_daily                      │
│ AHORA: Lee de inference_features_production                                 │
│                                                                             │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ l0_macro_hourly_inference (NUEVO DAG)                                  │ │
│ │ Schedule: 0 13-17 * * 1-5 (cada hora)                                  │ │
│ │                                                                        │ │
│ │ 1. Leer macro_daily + macro_monthly + macro_quarterly                  │ │
│ │ 2. Aplicar publication delay                                           │ │
│ │ 3. Resamplear a 5-min                                                  │ │
│ │ 4. Forward-fill con límites                                            │ │
│ │ 5. Calcular features macro (z-scores, changes)                         │ │
│ │ 6. UPSERT en inference_features_production                             │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│ ┌────────────────────────────────────────────────────────────────────────┐ │
│ │ l5_multi_model_inference (ya existe)                                   │ │
│ │ Schedule: */5 13-17 * * 1-5 (cada 5 min)                               │ │
│ │                                                                        │ │
│ │ CAMBIO: build_observation() ahora lee de inference_features_production │ │
│ │         en lugar de calcular features on-the-fly                       │ │
│ └────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## RESUMEN DE ACCIONES

### CREAR (9 archivos nuevos)
| # | Archivo | Tipo |
|---|---------|------|
| 1 | `database/migrations/030_promotion_requests.sql` | SQL |
| 2 | `database/migrations/031_backtest_results.sql` | SQL |
| 3 | `database/migrations/032_inference_features_production.sql` | SQL |
| 4 | `/api/models/[modelId]/promote/route.ts` | TypeScript |
| 5 | `/api/models/[modelId]/auto-approve/route.ts` | TypeScript |
| 6 | `/api/backtest/save/route.ts` | TypeScript |
| 7 | `airflow/dags/l0_macro_hourly_inference.py` | Python |
| 8 | `src/data/macro_consolidator.py` | Python |
| 9 | `tests/integration/promotion-flow.test.ts` | TypeScript |

### MODIFICAR (5 archivos)
| # | Archivo | Cambios |
|---|---------|---------|
| 1 | `PromoteButton.tsx` | +votos, +backtest_id, import thresholds |
| 2 | `BacktestControlPanel.tsx` | +guardar backtest, integrar PromoteButton |
| 3 | `ssot.contract.ts` | +PROMOTION_THRESHOLDS, +VOTE_REQUIREMENTS |
| 4 | `promotion_service.py` | +validate_for_auto_vote, +create_inference_table |
| 5 | `l5_multi_model_inference.py` | Leer de inference_features_production |

### ELIMINAR (2 secciones)
| # | Archivo | Líneas |
|---|---------|--------|
| 1 | `BacktestControlPanel.tsx` | 528-571 (handlePromote duplicado) |
| 2 | `PromoteButton.tsx` | 44-65 (thresholds hardcodeados) |

---

## PRIORIDADES DE IMPLEMENTACIÓN

### P0 (Crítico - Primero)
1. Migración `030_promotion_requests.sql`
2. Migración `031_backtest_results.sql`
3. `/api/backtest/save/route.ts`
4. `/api/models/[modelId]/promote/route.ts`

### P1 (Alto - Segundo)
5. Modificar `PromoteButton.tsx`
6. Modificar `BacktestControlPanel.tsx`
7. `validate_for_auto_vote()` en PromotionService

### P2 (Medio - Tercero)
8. Migración `032_inference_features_production.sql`
9. `create_inference_table()` en PromotionService
10. DAG `l0_macro_hourly_inference.py`

### P3 (Bajo - Cuarto)
11. Tests de integración
12. Modificar L5 para leer de inference_features_production
13. Documentación

---

*Plan v2.0 generado: 2026-01-30*
*Basado en análisis de 40+ archivos del proyecto*
