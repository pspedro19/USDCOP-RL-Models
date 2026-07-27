---
kind: roadmap
status: PLANNED
version: 1.0.0
supersedes: []
last_verified: 2026-07-27
code_anchors:
  - src/contracts/signal_contract.py
  - src/contracts/strategy_schema.py
  - config/assets/pipelines.yaml
  - airflow/dags/asset_pipeline_factory.py
  - usdcop-trading-dashboard/lib/contracts/rbac.contract.ts
---

# PLAN — Muralla contractual: contratos separados, permisos de DB y reglas de CI

> Segunda pieza del plan de superficies (ver `00-superficies-action-vs-diagnostic.md`).
> Objetivo: que la separación ACTION/DIAGNOSTIC **no dependa de que el programador
> "recuerde"** que el forecasting es diagnóstico — la base de datos, los contratos y
> el CI la imponen físicamente.

## 1. Contratos separados

No se usa el mismo contrato para ambos outputs.

### Contrato de acción (`strategy_output`)

```yaml
strategy_output:
  signal_id: uuid
  sleeve_id: usdcop_smart_simple_v11
  strategy_version: 11.0.0
  instrument: USDCOP

  as_of: 2026-07-27T13:15:00Z
  available_at: 2026-07-27T13:15:00Z
  valid_from: 2026-07-27T13:30:00Z
  valid_until: 2026-07-31T17:55:00Z

  target:
    type: nav_weight
    value: 0.80
    currency: COP

  direction: LONG
  decision_fingerprint: sha256:...
  health_snapshot_id: ...
  reason_codes:
    - POSITIVE_MODEL_SCORE
    - HURST_GATE_OPEN
    - VOLATILITY_ACCEPTABLE
```

### Contrato de forecasting (`forecast_output`)

```yaml
forecast_output:
  forecast_id: uuid
  forecast_spec_id: usdcop_forecast_zoo_v3
  asset: usdcop
  model_id: ridge_v2
  horizon: 5d

  as_of: 2026-07-27T00:00:00Z
  target_time: 2026-08-03T00:00:00Z

  prediction:
    type: return
    point: 0.0062
    lower: -0.011
    upper: 0.024

  direction_probability:
    up: 0.58

  model_fingerprint: sha256:...
  data_snapshot_id: ...
```

**Regla dura**: el allocator solo acepta `strategy_output`. Debe **rechazar
físicamente** un `forecast_output` (validación de tipo, no convención).

## 2. Separación en base de datos

Esquemas distintos:

```text
action.strategy_signal      forecast.forecast_output       execution.order
action.paper_order          forecast.forecast_score        execution.order_status_event
action.paper_fill           forecast.model_horizon_result  execution.fill
action.fact_position        forecast.calibration_result
action.fact_pnl
                            control.strategy · control.forecast_spec
                            control.metric_event · control.lineage_node/edge
```

### Permisos (la muralla física)

```text
forecast_writer:  INSERT en forecast.*   · NO INSERT en action.* ni execution.*
strategy_writer:  INSERT en action.strategy_signal · NO INSERT en execution.*
executor:         SELECT solo de action.approved_signal · INSERT en execution.*
```

## 3. Separación de DAGs y datasets

Nomenclatura por superficie:

```text
asset__{id}__data                      (compartido: ingesta canónica)
strat__{id}_{estrategia}               (ACTION)
forecast__{id}__weekly_zoo             (DIAGNOSTIC)
book__allocator_v1                     (ACTION agregado)
exec__broker_{id}                      (ejecución)
```

Datasets con URI tipada:

```text
asset://usdcop/canonical
asset://usdcop/action_features
asset://usdcop/forecast_features
strategy://usdcop_smart_simple_v11/signal
forecast://usdcop/ridge_v2/h5/prediction
portfolio://target
```

**Dependencia permitida**: `strategy://*/signal → book__allocator`.
**Dependencia PROHIBIDA**: `forecast://*/prediction → book__allocator` — ese enlace
**debe fallar en CI**.

## 4. Reglas automáticas de CI (obligatorias)

```text
✓ surface=diagnostic no puede declarar signal
✓ surface=diagnostic no puede declarar allocate
✓ surface=diagnostic no puede declarar execute
✓ ningún endpoint de forecasting puede escribir execution.*
✓ book allocator solo consume strategy://*/signal
✓ toda señal tiene strategy_version y decision_fingerprint
✓ toda predicción tiene model_id, horizon y target_time
✓ frontend forecasting no contiene botones APROBAR
✓ frontend forecasting no llama endpoints de ejecución
✓ predictor interno de una estrategia está congelado en el Passport
✓ fallo de predictor interno bloquea la señal
✓ fallo de forecasting público no cambia posiciones
```

## 5. Estado actual vs objetivo (as-built 2026-07-27)

| Pieza | Hoy | Acción |
|---|---|---|
| `UniversalSignalRecord` (≈strategy_output) | ✅ existe (`signal_contract.py`) | añadir `decision_fingerprint`, `reason_codes`, `valid_from/until` |
| `forecast_output` tipado | ❌ (CSV plano + JSON ad-hoc) | crear contrato Python+TS espejo |
| Esquemas DB por superficie | ❌ (todo `public`) | migración por fases; empezar por `forecast.*` (menor riesgo) |
| Roles/permisos DB por writer | ❌ (un usuario app) | roles `forecast_writer`/`strategy_writer`/`executor` |
| DAG naming por superficie | parcial (`asset_*` factory, `forecast_h5_*` COP) | renombrar con alias de compatibilidad (dag_registry) |
| CI: forecasting sin botones aprobar | ✅ (RBAC + no hay componentes de acción en la vista) | formalizar como test explícito |
| CI: allocator solo consume señales | parcial (book construction lee trades de estrategias) | test que el ledger/book rechace inputs forecast |
| Passport de componentes congelados | parcial (manifiestos con code_hash) | añadir bloque `components:` al manifiesto (ver plan 02) |

**Orden de implementación sugerido** (0 trials, todo ingeniería):
1. Contrato `forecast_output` (Py+TS) + validación en el generador del zoo.
2. Tests CI de muralla (los 12 checks — la mayoría son greps/asserts baratos).
3. Campo `surface:` en manifiestos + registry; normalize_champions lo respeta.
4. Esquema `forecast.*` en DB + rol `forecast_writer`.
5. Renombrado de DAGs con alias (último — es cosmético y toca dag_registry).
