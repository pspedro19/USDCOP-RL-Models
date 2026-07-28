---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - config/assets/pipelines.yaml
  - src/contracts/signal_contract.py
  - scripts/pipeline/train_and_export_smart_simple.py
---

> **Nota de reconciliación (2026-07-27)**: la Parte I de este documento es el cuerpo
> FABRIC-003, ya reemplazado por `04-CTR-QLAB-FABRIC-004.md` — se conserva como
> referencia. **La parte VIGENTE es la PARTE II (§33-57): plan de datos, columnas,
> features y resampleos por estrategia (fases D0-D8)**, que la 004 no contiene.

# CTR-QLAB-FABRIC-003

## Especificación consolidada del Control Plane cuantitativo, linaje, investigación, replay, forecasting, asignación y ejecución

| Campo | Valor |
|---|---|
| Documento | `CTR-QLAB-FABRIC-003` |
| Estado | Propuesta normativa consolidada |
| Versión | `3.0.0` |
| Fecha | 2026-07-27 |
| Alcance | USD/COP, XAU/USD, BTC/USDT, SPX500 y futuras extensiones |
| Fuente de verdad propuesta | Git para declaraciones; PostgreSQL para hechos y eventos; MinIO para artefactos; MLflow para modelos |
| Sustituye conceptualmente | `CTR-PIPELINE-002`, `CTR-CONTROL-PLANE-001` y `CTR-CONTROL-PLANE-002` |

---

## 0. Dictamen ejecutivo

La plataforma debe operar como una **Quant Strategy Fabric**: una sola verdad lógica y auditable, formada por múltiples componentes aislados. No se construye un DAG monolítico ni una estrategia universal para todos los activos.

La unidad de gobierno es el **sleeve de estrategia**; la unidad física de datos es el **activo**; la unidad estadística de investigación es la **familia de hipótesis**; y la unidad de capital es el **libro o portfolio target**.

La regla final es:

> Los datos producen snapshots. Las familias consumen trials. Las estrategias de acción producen señales. Los modelos diagnósticos producen forecasts. El allocator asigna riesgo. El executor toca el mercado. Las tablas de hechos son la única verdad del PnL. El Strategy Passport recuerda todo porque se deriva de los hechos y nunca se escribe manualmente.

La arquitectura queda **aprobada para construcción por etapas**. No queda aprobada para ampliar ejecución live hasta completar identidad determinista, event sourcing, sincronización temporal, pre-trade risk, idempotencia, reconciliación, kill switch y retiro operativo.

---

## 1. Objetivos

1. Escalar de cuatro activos a decenas de activos y múltiples sleeves sin duplicar pipelines.
2. Reconstruir cualquier señal, orden, fill, posición, PnL, métrica o decisión de gobierno desde sus datos originales.
3. Separar estrictamente la ciencia de **decisiones económicas** de la ciencia de **predicción diagnóstica**.
4. Evitar look-ahead, p-hacking, selección múltiple oculta, performance chasing y drift de configuración.
5. Aislar fallas: una estrategia, forecast o activo defectuoso no debe bloquear a los demás.
6. Mantener Airflow como orquestador batch, no como broker ni bus transaccional.
7. Permitir paper, canary y live bajo contratos comparables.
8. Mantener una Control Tower única sin convertirla en fuente de verdad ni permitir cálculos en frontend.
9. Usar infraestructura proporcional a la escala actual y adoptar componentes más complejos solamente cuando exista una necesidad demostrada.

### 1.1 No objetivos

- No construir una estrategia única aplicable a todos los activos.
- No utilizar forecasting diagnóstico como recomendación de compra o venta.
- No permitir que la UI recalcule métricas, gates, PnL o aprobación.
- No permitir que un retry de Airflow genere una segunda orden.
- No promover un allocator complejo porque se vea mejor en un backtest aislado.
- No introducir Kubernetes, Kafka, Feast, Iceberg o microestructura avanzada antes de que su ausencia sea un cuello de botella real.

---

## 2. Las leyes constitucionales

1. **Una sola verdad lógica, no una sola aplicación física.**
2. **Toda fila persistida debe tener identidad, tiempo, procedencia y entorno.**
3. **Los trials se cobran una sola vez en screening; Airflow nunca gasta trials.**
4. **Lo congelado no cambia. Lo dinámico es el capital, bajo reglas pre-registradas y acotadas.**
5. **El backtest permite entrar a paper; el paper permite entrar a canary; solo el forward permite capital completo.**
6. **Toda métrica tiene una definición, un motor y una persistencia única.**
7. **El allocator nunca consume “el último valor”; consume un portfolio snapshot con cutoff explícito.**
8. **Replay, paper y live comparten contratos de órdenes y fills; difieren en el executor.**
9. **El forecasting diagnóstico no puede producir señales, asignaciones ni órdenes.**
10. **Una revisión legítima de un dato no invalida una decisión histórica point-in-time correcta.**
11. **Un retiro no elimina un DAG hasta cerrar posiciones, órdenes y reconciliación.**
12. **El frontend presenta hechos; no decide ni corrige datos.**
13. **Toda excepción, override y voto debe ser auditable.**
14. **Ningún atributo puede tener dos sistemas autorizados para modificarlo.**

---

## 3. Dos superficies científicas, un Control Plane

Las dos superficies comparten datos, calendarios, snapshots, linaje, observabilidad, identidad y gobierno técnico. Se separan en objetivos, targets, contratos, métricas, permisos, DAGs, bases de datos y vistas.

### 3.1 Superficie ACTION: replay, decisión y dinero

**Pregunta:** ¿qué posición debo mantener, cuándo entrar, cuándo salir y cuánto riesgo asumir?

```text
Datos point-in-time
  → features causales
  → regla o modelo congelado
  → señal LONG | SHORT | FLAT
  → target de exposición
  → simulador determinista o broker
  → órdenes y fills
  → posiciones y PnL
  → métricas económicas y de riesgo
  → PAPER → CANARY → CHAMPION
  → allocator → ejecución
```

**Outputs principales:**

- `strategy_output`
- señal y target de exposición
- órdenes y fills simulados o reales
- posiciones
- `fact_pnl`
- bundles `summary`, `trades` y `signals`
- gates y resultados del juez

**Métricas principales:** retorno neto, MaxDD, Calmar, Sharpe, Sortino, DSR, PBO, turnover, exposición, slippage, implementation shortfall, fill rate, tracking error y paridad replay-paper-live.

La exactitud predictiva puede observarse como diagnóstico de un componente, pero no es el criterio final de una política económica completa.

### 3.2 Superficie DIAGNOSTIC: forecasting y transparencia

**Pregunta:** ¿qué precio, retorno o dirección estima un modelo para un horizonte definido?

```text
Datos point-in-time
  → features de forecasting
  → target supervisado
  → walk-forward / OOF
  → modelo × horizonte
  → predicción e intervalo
  → comparación contra baseline
  → CSV, tablas, PNG y panel
  → cero capital y cero órdenes
```

**Outputs principales:**

- `forecast_output`
- predicción puntual
- intervalos o cuantiles
- probabilidad o score declarado
- métricas por modelo y horizonte
- paneles y gráficos

**Métricas principales:** directional accuracy, balanced DA, lift contra baseline, MAE, RMSE, pinball loss, Brier score, calibración, cobertura de intervalos y drift.

### 3.3 Matriz de separación

| Propiedad | ACTION | DIAGNOSTIC |
|---|---|---|
| Produce decisión económica | Sí | No |
| Produce PnL | Sí | No |
| Puede ser `CHAMPION` | Sí | No |
| Puede llegar al allocator | Sí | No |
| Puede ejecutar | Solo con aprobación y capacidad | Nunca |
| Usa Vote 1 / Vote 2 | Sí | No |
| Métrica reina | Desempeño económico y riesgo | Lift y error predictivo |
| Vistas | Replay, dashboard, production, execution | Forecasting, analysis |
| Fallo | Puede bloquear dinero | Degrada panel diagnóstico |

### 3.4 Dependencias permitidas y prohibidas

Permitido:

```text
asset://{asset}/canonical → strategy://{sleeve}/signal
asset://{asset}/canonical → forecast://{asset}/{model}/{horizon}
strategy://{sleeve}/signal → portfolio_snapshot → allocator
```

Prohibido:

```text
forecast://*/prediction → allocator
forecast://*/prediction → execution
frontend forecasting → endpoint de órdenes
```

Estas prohibiciones se implementan mediante contratos, permisos de base de datos y validaciones de CI; no dependen de disciplina humana.

---

## 4. Arquitectura lógica

```text
                             QUANT CONTROL PLANE
┌─────────────────────────────────────────────────────────────────────────────┐
│ Gobierno: assets · families · trials · strategies · forecasts · judges      │
│ Operación: runs · signals · snapshots · allocations · orders · fills        │
│ Hechos: positions · pnl · metrics · incidents · lineage                     │
└─────────────────────────────────────────────────────────────────────────────┘
             ↑                         ↑                         ↑
             │ lineage                 │ telemetry               │ governance
             │                         │                         │
PROVEEDORES → RAW → CANONICAL → FEATURES ─┬→ ACTION → SIGNALS ─┬→ ALLOCATOR
                                           │                    │      │
                                           └→ DIAGNOSTIC        │      ▼
                                              FORECASTS         │ PORTFOLIO TARGET
                                                               │      │
                                                               └──────▼
                                                           EXECUTION SERVICE
                                                               │
                                                        ORDERS / FILLS / PNL
```

### 4.1 Planos de entidades

```text
GOBIERNO
assets · families · research_clusters · trials · strategies · strategy_versions
forecast_specs · judges · votes · decisions · withdrawal_protocols

OPERACIÓN
runs · strategy_signals · forecast_outputs · portfolio_snapshots
allocations · portfolio_targets · orders · order_status_events · fills

HECHOS
fact_position · fact_pnl · metric_event · incident
lineage_node · lineage_edge
```

Los archivos grandes, snapshots y bundles viven en MinIO. PostgreSQL almacena identidad, estado, referencias, eventos y hechos consultables.

---

## 5. Estado inicial de migración por activo

Esta tabla representa el punto de partida funcional y debe confirmarse contra los registries al iniciar la migración.

| Activo | Sleeve ACTION principal | Candidatas / retiradas | Superficie DIAGNOSTIC | Ejecución |
|---|---|---|---|---|
| USD/COP | `usdcop_smart_simple_v11` | `v12`, `v14` en paper; otras según registry | Zoo supervisado por modelo y horizonte | Único activo con ruta live actual |
| XAU/USD | `xauusd_trend_simple_v1` | `dynamic_exit` retirada pero preservada | Forecasting semanal diagnóstico | Paper/research |
| BTC/USDT | `btcusdt_hodl_b1` | Retadores bajo moratoria | Forecasting diagnóstico | Paper/research |
| SPX500 | `daily_ma200` o `regime_gated`, pendiente del juez formal | Ambas preservadas como sleeves independientes | Diagnóstico deshabilitado o separado según registry | Paper/research |

Principio: la mecánica ganadora no se copia automáticamente entre activos. Cada activo conserva reloj, calendario, costos, sesiones, liquidez, frecuencia y criterio de validación propios.

---

## 6. Capas actuales y mapeo objetivo

La migración no redefine contratos existentes por accidente.

| Capa actual | Responsabilidad actual | Destino en Fabric |
|---|---|---|
| L0 | Ingesta, validación, OHLCV, macro, seeds y backfill | `asset__{asset}__data` y snapshots raw/canonical |
| L0b | Exportación de datos para charts | Proyección frontend; nunca fuente de verdad |
| L1 | Features RL de USD/COP, única capa autorizada para ese contrato | Se conserva; publica feature snapshot ACTION |
| L2 | Dataset RL | Dataset versionado con linaje y cutoff |
| L3 | Entrenamiento | Tarea opcional de `strat__*` cuando `retrain != never` |
| L4 | Backtest, gates y publicación inmutable | `strat__{sleeve}` ACTION |
| L4b | Deploy tras Vote 2 | Transición de gobierno; no envío directo de orden |
| L5 | Señal o inferencia | Publicación de `strategy_output` |
| L6 | Forward, paper ledger y verify | Hechos, métricas, jueces y Passport |
| L7 | Ejecución | Servicio aislado fuera de Airflow |
| L8 | News/LLM de contexto | Superficie ANALYSIS, nunca decisión |

Para activos factory sin L1 numerada, las features viven como stages versionados del DAG de datos o del DAG de superficie correspondiente. No se fuerza una numeración artificial.

---

## 7. Matriz de fuente de verdad

| Entidad | Fuente autoritativa | Proyección / réplica |
|---|---|---|
| Asset registry | Git | PostgreSQL |
| Family registry y declaración de celdas | Git | PostgreSQL |
| Strategy y forecast specs | Git | PostgreSQL |
| Trial ledger | PostgreSQL append-only con commit de origen | Dashboard / reportes |
| Votos y transiciones | PostgreSQL append-only + referencia al commit | Passport |
| Runs y señales | PostgreSQL | OpenLineage / Grafana |
| Modelos y pesos entrenados | MLflow + MinIO | Referencias en PostgreSQL |
| Snapshots y bundles | MinIO | Metadatos en PostgreSQL |
| Órdenes y fills | PostgreSQL event-sourced | Broker reconciliation / UI |
| Positions y PnL | Tablas de hechos en PostgreSQL | Passport / dashboard |
| Métricas | `metric_event` en PostgreSQL | Grafana / gates |
| Grafo de linaje | PostgreSQL | Marquez como visor opcional |
| Archivos de frontend | Proyección regenerable | Nunca autoritativos |

Regla: ningún atributo tiene dos escritores autorizados.

---

## 8. Identidad determinista

### 8.1 Conceptos

```text
spec_fingerprint = H(
  data_snapshot_id
  ⊕ feature_snapshot_id
  ⊕ code_hash
  ⊕ config_hash
  ⊕ model_or_policy_hash
  ⊕ calendar_hash
  ⊕ cost_model_hash
  ⊕ dependency_lock_hash
  ⊕ container_image_digest
)

decision_fingerprint = H(
  spec_fingerprint
  ⊕ as_of
  ⊕ decision_inputs
)

execution_fingerprint = H(
  decision_fingerprint
  ⊕ env
  ⊕ account_id
  ⊕ broker_id
  ⊕ order_policy_hash
)

derivation_id = H(inputs ⊕ code ⊕ params)
semantic_hash = H(contenido canónico)
bytes_hash = H(bytes físicos)
```

### 8.2 Regla de paridad

> El mismo `decision_fingerprint` debe producir el mismo `semantic_hash` de la señal en replay, paper, canary y live.

- Para JSON, ledger, señal, target y bundle cuyo escritor está controlado, se usa serialización canónica y `semantic_hash == bytes_hash` por construcción.
- Para Parquet, respuestas de broker y formatos externos, se compara hash semántico; `bytes_hash` se conserva solo para integridad física.

### 8.3 Serialización canónica

- UTF-8 normalizado NFC.
- Claves JSON ordenadas.
- Timestamps ISO-8601 UTC terminados en `Z`.
- Decimales cuantizados por campo según esquema.
- Sin `NaN`, `Infinity` o floats binarios sin normalización.
- Sin whitespace no significativo.
- Orden de filas explícito antes de serializar.

### 8.4 Spine mínimo

Toda entidad operacional o de hechos debe contener, directamente o mediante FK inequívoca:

```text
as_of
available_at
created_at
run_id
spec_fingerprint
decision_fingerprint nullable
execution_fingerprint nullable
derivation_id
artifact_or_event_id
sleeve_id nullable
forecast_spec_id nullable
family_id nullable
trial_id nullable
code_hash
data_snapshot_id
env
schema_version
```

No toda fila necesita todas las columnas físicamente; el grafo debe permitir resolverlas sin ambigüedad.

---

## 9. Registries

### 9.1 Asset registry

```yaml
id: spx500
enabled: true

market:
  calendar: NYSE
  session: "09:30-16:00"
  timezone: America/New_York
  annualization: 252
  continuous: false

clock:
  master_bar: 1D
  decision_point: close
  execution_ref: next_open
  warmup_bars: 252

canary_requirements:
  min_decisions: 60
  min_fills: 30
  min_regimes: 2

costs:
  model: fixed_bps
  value: 3.0
  stress_scenarios: [5.0, 10.0]

data:
  price:
    provider: investing
    symbol: "^GSPC"
    return_type: price_return
  macro_bundle: us_core

schedule:
  data: "30 7 * * 1-5"
```

`annualization`, calendario, timezone, costo, frecuencia y reglas canary se leen exclusivamente desde el asset registry.

### 9.2 Strategy registry ACTION

```yaml
id: spx500_daily_ma200_v1
asset: spx500
family: trend_regime
research_cluster: trend
surface: action
version: 2.0.0

engine:
  type: rule_based
  module: strategies.policies:MA200Policy
  retrain: never
  params:
    ma_window: 200
    exposure_cap: 1.0

research_state: PAPER
capital_tier: SHADOW
operational_state: NOMINAL

governance:
  data_cutoff: "2024-12-31"
  frozen_at: "2026-07-27T00:00:00Z"
  code_hash: "sha256:..."
  ssot_manifest: manifests/spx500_ma200_v1.json
  withdrawal_protocol: docs/withdrawal/spx500_ma200_v1.md

judge:
  anchor: "2026-08-01"
  criterion: "Calmar >= incumbent"
  min_periods: 26
  alpha: 0.05
  paired_with: null
  sequential:
    method: none

publish:
  bundles: [summary, trades, signals]
  immutable_key: [strategy_id, version, partition]

capabilities: [backtest, gate, signal, paper, verify]
```

### 9.3 Forecast registry DIAGNOSTIC

```yaml
id: usdcop_forecast_zoo_v3
asset: usdcop
surface: diagnostic
state: ACTIVE

models:
  - ridge
  - bayesian_ridge
  - random_forest

horizons: [1d, 5d, 10d, 20d, 30d]
target: forward_return
validation: purged_walk_forward
baselines: [majority_class, persistence, zero_return]

capabilities: [fit, predict, evaluate, publish_panel]

forbidden_capabilities: [signal, allocate, execute, approve]
```

### 9.4 Family registry

```yaml
family_id: trend_regime
cluster_id: trend
declared_at: "2026-06-01"
question: "¿Un gate de tendencia mejora el Calmar contra el baseline?"
bar: "batir persistencia y baseline tonto, neto de costos, screening <= 2024"
screening_cutoff: "2024-12-31"

cells:
  - asset: spx500
    variant: ma200_pure
    trial_id: T-0113
    status: SCREENED
    result: pass
  - asset: xauusd
    variant: sma_votes
    trial_id: T-0087
    status: FROZEN
    result: pass

trials_charged: 2
closed: false
```

### 9.5 Trial budget

`N_MAX=989` se conserva únicamente como **cota constitucional de gasto**. No entra en DSR ni en ninguna fórmula estadística.

Se reportan:

```text
N_family
N_cluster
N_global
DSR_family
DSR_cluster
DSR_global
```

El criterio de gobierno puede usar `DSR_family`; los otros valores son divulgación obligatoria para hacer visible la dependencia entre familias relacionadas.

---

## 10. Estados en tres dimensiones

### 10.1 Research state

```text
DECLARED
→ SCREENED
→ DESIGN_RUN
→ FROZEN
→ PAPER
→ CHAMPION
→ RETIRING
→ WITHDRAWN
```

### 10.2 Capital tier

```text
ZERO | SHADOW | CANARY | FULL | REDUCED | EXIT_ONLY
```

### 10.3 Operational state

```text
NOMINAL | QUARANTINED
```

### 10.4 Matriz de legalidad

```text
research_state ∈ {DECLARED, SCREENED, DESIGN_RUN, FROZEN}
  ⇒ capital_tier = ZERO

research_state = PAPER
  ⇒ capital_tier ∈ {ZERO, SHADOW}

research_state = CHAMPION
  ⇒ capital_tier ∈ {ZERO, SHADOW, CANARY, FULL, REDUCED}

research_state = RETIRING
  ⇒ capital_tier = EXIT_ONLY

research_state = WITHDRAWN
  ⇒ capital_tier = ZERO ∧ exit_checklist = PASS

operational_state = QUARANTINED
  ⇒ toda orden de apertura es rechazada
  ⇒ capital_tier se conserva para poder reanudar tras recuperación

DAG de estrategia existe
  ⇔ research_state ∈ {FROZEN, PAPER, CHAMPION, RETIRING}
```

El registry validator rechaza cualquier combinación ilegal.

### 10.5 Canary

Canary no se gobierna por un número universal de semanas. Debe cumplir simultáneamente:

- `min_decisions` definido por el asset registry.
- `min_fills` definido por frecuencia y liquidez.
- Exposición a un mínimo de regímenes declarados.
- Tracking error dentro del umbral.
- Slippage ≤ factor sobre el costo modelado.
- Cero incidentes críticos.
- Paridad semántica verde.
- Reconciliación sin diferencias materiales.

---

## 11. DAGs y servicios

### 11.1 Generador A: datos por activo

```text
asset__{asset}__data
  l0_ingest
  → canonicalize
  → quality
  → feature_snapshots autorizados
  → chart_projection
  → data_verify
  → publica asset://{asset}/...
```

El DAG de datos no conoce las estrategias ni los forecasts que lo consumen.

### 11.2 Generador B: estrategia ACTION

```text
strat__{sleeve_id}
  [train si aplica]
  → backtest/replay
  → gates
  → signal
  → paper simulation
  → verify
  → judge tracking
```

Cada estrategia activa tiene un DAG propio. Una falla no bloquea a la campeona ni a otros sleeves.

### 11.3 Generador D: forecasting DIAGNOSTIC

```text
forecast__{forecast_spec_id}
  feature_view
  → walk_forward_fit
  → predict
  → evaluate
  → publish_panel
```

Este generador no importa librerías de ejecución ni publica assets consumibles por el allocator.

### 11.4 Libro y controles

```text
control__portfolio_snapshot
book__allocator_v1
control__pretrade
control__intraday_risk
control__eod_reconciliation
control__weekly_judges
control__system_health
exec__paper_multiasset
```

### 11.5 Servicio de ejecución

La ejecución live ocurre en un servicio independiente:

```text
portfolio_target inmutable
  → execution service
  → pre-trade risk
  → broker adapter
  → order/fill event ledger
```

Airflow puede publicar targets, ejecutar reconciliaciones y evaluar jueces. No mantiene credenciales del broker ni constituye la única barrera contra duplicación.

---

## 12. Barrera temporal del libro

El allocator no lee “la última señal”. Consume un `portfolio_snapshot` explícito:

```yaml
portfolio_snapshot:
  snapshot_id: uuid
  cutoff_time: "2026-07-27T13:15:00Z"
  required_sleeves:
    - usdcop_smart_simple_v11
    - xauusd_trend_simple_v1
  accepted_signals:
    - signal_id: uuid
      sleeve_id: usdcop_smart_simple_v11
      as_of: "..."
  stale_signals: []
  missing_signals: []
  fallback_applied: []
  max_age_by_sleeve:
    usdcop_smart_simple_v11: 7d
    xauusd_trend_simple_v1: 2d
```

Cada sleeve declara política de faltante:

```text
FLAT
KEEP_POSITION_UNTIL_EXPIRY
EXIT_ONLY
USE_LAST_VALID_WITH_MAX_AGE
```

`USE_LAST_VALID` sin límite de edad queda prohibido.

---

## 13. Contratos de salida

### 13.1 `strategy_output` v2

```yaml
strategy_output:
  signal_id: uuid
  sleeve_id: usdcop_smart_simple_v11
  strategy_version: 11.0.0
  model_snapshot_id: uuid
  instrument: USDCOP

  as_of: "2026-07-27T13:15:00Z"
  available_at: "2026-07-27T13:15:00Z"
  valid_from: "2026-07-27T13:30:00Z"
  valid_until: "2026-07-31T17:55:00Z"

  target:
    type: nav_weight
    value: 0.80
    currency: COP

  direction: LONG

  confidence:
    value: 0.67
    type: calibrated_probability
    calibration_version: cal_v2

  forecast:
    volatility: 0.094
    horizon: 5d

  health_snapshot_id: uuid
  liquidity_snapshot_id: uuid
  decision_fingerprint: "sha256:..."
  reason_codes:
    - POSITIVE_MODEL_SCORE
    - HURST_GATE_OPEN
```

La estrategia no autodeclara salud ni liquidez. Solo referencia snapshots producidos por controles independientes.

### 13.2 `forecast_output` v1

```yaml
forecast_output:
  forecast_id: uuid
  forecast_spec_id: usdcop_forecast_zoo_v3
  asset: usdcop
  model_id: ridge_v2
  horizon: 5d

  as_of: "2026-07-27T00:00:00Z"
  available_at: "2026-07-27T00:05:00Z"
  target_time: "2026-08-03T00:00:00Z"

  prediction:
    type: return
    point: 0.0062
    lower: -0.0110
    upper: 0.0240

  direction_probability:
    up: 0.58

  model_fingerprint: "sha256:..."
  data_snapshot_id: uuid
  diagnostic_only: true
```

El allocator acepta exclusivamente `strategy_output` validado.

### 13.3 Permisos de persistencia y ejecución

La muralla ACTION/DIAGNOSTIC se implementa también en base de datos:

```text
forecast_writer
  INSERT/UPDATE controlado en forecast.*
  sin permisos sobre action.*, portfolio.* o exec.*

strategy_writer
  INSERT en action.strategy_signal y bundles
  sin permisos para crear órdenes live

allocator_service
  SELECT de strategy_output aprobado
  INSERT en portfolio_snapshot, allocation y portfolio_target
  sin credenciales de broker

execution_service
  SELECT de portfolio_target aprobado
  INSERT en exec.*
  sin permisos para modificar estrategia, trials o métricas

frontend_role
  SELECT de vistas y endpoints autorizados
  sin acceso directo de escritura a tablas de gobierno o ejecución
```

Las funciones sensibles se exponen mediante APIs o stored procedures con validación, no mediante escritura libre de tablas.

### 13.4 Predictor interno versus zoo diagnóstico

Un modelo de forecasting puede formar parte de una estrategia ACTION como componente congelado:

```yaml
components:
  - component_id: usdcop_ridge_br_v5
    role: decision_input
    model_snapshot_id: uuid
    code_hash: "sha256:..."
```

Eso no convierte al zoo diagnóstico en señal. La estrategia completa se evalúa por PnL, riesgo, costos y forward, aunque el predictor aislado tenga baja precisión. El zoo público se evalúa por lift y error predictivo.

---

## 14. Ejecución event-sourced

### 14.1 Tablas comunes para replay, paper, canary y live

```sql
CREATE TABLE exec.order_header (
  order_id UUID PRIMARY KEY,
  client_order_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  account_id TEXT,
  env TEXT NOT NULL,
  executor_type TEXT NOT NULL,
  sleeve_id TEXT NOT NULL,
  allocation_id UUID,
  instrument TEXT NOT NULL,
  side TEXT NOT NULL,
  qty NUMERIC NOT NULL,
  order_type TEXT NOT NULL,
  limit_price NUMERIC,
  tif TEXT,
  currency TEXT,
  parent_order_id UUID,
  decision_fingerprint TEXT NOT NULL,
  execution_fingerprint TEXT NOT NULL,
  submitted_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE exec.order_status_event (
  event_id UUID PRIMARY KEY,
  order_id UUID NOT NULL REFERENCES exec.order_header(order_id),
  event_time TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL,
  reason_code TEXT,
  broker_order_id TEXT
);

CREATE TABLE exec.fill_event (
  fill_id UUID PRIMARY KEY,
  order_id UUID NOT NULL REFERENCES exec.order_header(order_id),
  fill_time TIMESTAMPTZ NOT NULL,
  qty NUMERIC NOT NULL,
  price NUMERIC NOT NULL,
  commission NUMERIC,
  venue TEXT
);

CREATE TABLE exec.fill_correction_event (
  correction_id UUID PRIMARY KEY,
  fill_id UUID NOT NULL REFERENCES exec.fill_event(fill_id),
  event_time TIMESTAMPTZ NOT NULL,
  field TEXT NOT NULL,
  old_value TEXT,
  new_value TEXT,
  reason TEXT NOT NULL
);
```

El estado actual de una orden es una proyección de eventos, nunca un `UPDATE` destructivo.

### 14.2 Entornos

| Entorno | Executor | Resultado |
|---|---|---|
| replay | `deterministic_simulator` | órdenes, fills y PnL simulados |
| paper | `deterministic_simulator` | eventos comparables con live |
| canary | `broker` o sandbox broker | capital limitado |
| live | `broker` | hechos reconciliados |

### 14.3 Idempotencia

```text
idempotency_key = SHA256(
  account_id
  ⊕ instrument
  ⊕ target_version
  ⊕ decision_fingerprint
  ⊕ rebalance_cutoff
)
```

Un retry que publica el mismo target no produce una segunda orden.

---

## 15. Hechos de posiciones y PnL

`fact_position` y `fact_pnl` deben coexistir por entorno y derivarse de fills.

```text
primary grain fact_position:
(as_of, sleeve_id, instrument, env)

primary grain fact_pnl:
(as_of, sleeve_id, instrument, env, pnl_component)
```

La identidad contable requerida es:

```text
gross_pnl
= pnl_beta
+ pnl_timing
+ pnl_carry
- commissions
- slippage
- financing
+ pnl_residual
```

Campos obligatorios:

```text
attribution_model_version
benchmark_id
pnl_residual
reconciliation_status
source_fill_set_id
```

`timing_ratio` es un diagnóstico de estilo y atribución, no una demostración de alfa. Se reporta con intervalo de confianza por bootstrap en bloques y sensibilidad al benchmark.

---

## 16. Métricas: definir, computar, persistir

### 16.1 Catálogo

`config/metrics/catalog.yaml` es la única definición normativa:

```yaml
strategy.calmar:
  formula_version: v1
  source: fact_pnl
  annualization: from_asset_registry
  windows: [26w, 52w, since_anchor]
  warning: 0.10
  critical: 0.00

research.dsr:
  formula_version: v2
  n_trials: from_trial_ledger
  variants: [family, cluster, global]
```

### 16.2 Motor

```text
metrics_engine.compute(entity, metric, window, env, as_of)
```

Es el único código autorizado para calcular una métrica gobernada. Dashboard, jueces, gates y reportes consumen el mismo resultado.

### 16.3 Persistencia

```sql
CREATE TABLE control.metric_event (
  metric_event_id UUID PRIMARY KEY,
  event_time TIMESTAMPTZ NOT NULL,
  catalog_version TEXT NOT NULL,
  formula_version TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  strategy_id TEXT,
  asset_id TEXT,
  run_id TEXT,
  environment TEXT,
  metric_namespace TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  metric_value DOUBLE PRECISION,
  metric_unit TEXT,
  status TEXT,
  threshold_warning DOUBLE PRECISION,
  threshold_critical DOUBLE PRECISION,
  dimensions JSONB,
  lineage JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Los thresholds vigentes se copian al evento para preservar auditoría histórica.

### 16.4 Namespaces

```text
data.*
research.*
strategy.*
forecast.*
forward.*
execution.*
portfolio.*
operations.*
governance.*
```

---

## 17. Gobierno estadístico

### 17.1 Trials

Cobra trial:

- probar nuevo modelo, feature, target, horizonte o regla;
- elegir el mejor resultado entre alternativas;
- cambiar cutoff después de ver resultados;
- convertir un forecast en señal económica;
- alterar un gate, sizing, salida o costo para mejorar el OOS observado.

No cobra trial:

- regenerar un modelo congelado;
- publicar nuevos períodos forward;
- calcular métricas previamente declaradas;
- actualizar el dashboard;
- monitorear drift sin modificar la política.

### 17.2 Dos linajes de investigación

```text
forecast_family
  → pregunta predictiva
  → modelos, horizontes y forecast_trial_ids

action_family
  → pregunta económica
  → predictor + gate + sizing + salidas
  → action_trial_ids
```

Cuando un predictor entra en una estrategia ACTION, su historia de trials predictivos se conserva y la política económica cobra su trial correspondiente.

### 17.3 Juez secuencial

Campo obligatorio:

```yaml
sequential:
  method: mSPRT | e_process | none
  alpha: 0.05
```

Con `none`, observar resultados es válido; actuar antes del horizonte firmado constituye una violación de gobierno.

### 17.4 A/B correcto

No se distribuye tráfico aleatorio. Se ejecutan políticas congeladas bajo:

- mismos datos;
- mismo cutoff;
- mismos costos;
- mismo simulador;
- ledgers independientes;
- juez prefirmado;
- test pareado cuando comparten señal base;
- corrección de alpha para candidatas concurrentes.

Cambiar una candidata crea una nueva versión, nuevo freeze y nueva ancla.

---

## 18. Allocator

### 18.1 Baseline obligatorio

La primera versión operativa usa **inverse volatility con caps**. HRP corre en shadow y solo se promueve si supera al baseline neto de costos y turnover bajo su propio juez.

### 18.2 Multiplicadores

```text
b_prov_i
= b_base_i
× m_forward_i
× m_liq_i
× m_div_i
× m_ops_i
× m_dd_i
```

Reglas v1:

- `m_forward ∈ [0, 1]`: solo reduce; no premia una racha.
- `m_ops ∈ {0, 1}`: quarantine implica cero nuevas aperturas.
- `m_dd` usa umbrales por sleeve normalizados por volatilidad esperada.
- Restaurar riesgo es más lento que reducirlo.
- Todo multiplicador es una fórmula pre-registrada, versionada y con histéresis.

### 18.3 Optimización restringida

```text
b* = argmin_b ||b - b_prov||² + λ_to ||b - b_prev||₁

sujeto a:
  sqrt(bᵀΣb) ≤ target_vol
  0 ≤ b_i ≤ cap_sleeve_i
  sum(b_i por asset) ≤ cap_asset
  sum(b_i) ≤ gross_cap
  turnover(b, b_prev) ≤ turnover_budget
  restricciones de liquidez
  restricciones de factores
```

Después:

```text
w_i = b_i × side_i
```

El allocator asigna presupuesto de riesgo; no decide dirección.

### 18.4 Fallback de infactibilidad

1. Relajar únicamente el turnover budget dentro del límite declarado.
2. Encoger `b_prov` hacia cero hasta obtener factibilidad.
3. Usar baseline equal-risk o inverse-vol con caps.
4. Si sigue infactible, target cero y incidente crítico.

No existe `normalize(clip(...))`.

---

## 19. Ejecución y riesgo

### 19.1 Pre-trade risk bloqueante

Antes de cada orden:

- señal vigente y no vencida;
- snapshot de salud nominal;
- posición esperada versus broker;
- ausencia de duplicados;
- notional máximo;
- caps por sleeve y activo;
- gross y net exposure;
- apalancamiento;
- pérdida diaria y drawdown;
- price collar;
- liquidez;
- sesión de mercado;
- cash y moneda;
- límites de cuenta;
- operational state;
- kill switch.

### 19.2 Kill switch

- Independiente de Airflow.
- Consultado antes de cada apertura o modificación de orden.
- Auditado con actor, causa, hora y alcance.
- Debe permitir `BLOCK_NEW`, `CANCEL_OPEN`, `EXIT_ALL` y `ACCOUNT_FREEZE`.

### 19.3 Reconciliación

Se ejecuta:

- antes de operar;
- intradía para activos live;
- al cierre;
- después de una recuperación de servicio.

Compara broker, order ledger, fills, posiciones y hechos. Una discrepancia material produce incidente y quarantine.

### 19.4 Retiro operativo

```text
CHAMPION
→ RETIRING
→ capital_tier = EXIT_ONLY
→ target cero o plan de liquidación
→ cancelar órdenes abiertas
→ reconciliar broker
→ confirmar position = 0
→ confirmar open_orders = 0
→ publicar bundle final
→ WITHDRAWN
```

El DAG desaparece solo después de `exit_checklist = PASS`.

---

## 20. Linaje y point-in-time

### 20.1 Camino dorado

```text
provider
→ raw_snapshot
→ canonical_snapshot
→ feature_snapshot
→ model_or_policy_version
→ strategy_or_forecast_run
→ bundle_or_forecast
→ signal
→ portfolio_snapshot
→ allocation
→ order
→ fill
→ position
→ pnl_attribution
```

Toda fila de `fact_pnl` debe poder recorrer el camino hasta el snapshot raw.

### 20.2 Nodos y aristas

`lineage_node` incluye:

```text
node_id
node_type
semantic_hash
bytes_hash
schema_version
row_count
min_event_time
max_event_time
quality_status
status: VALID | STALE | INVALIDATED
storage_uri
```

`lineage_edge` registra `CONSUMED`, `PRODUCED`, `DERIVED_FROM`, `CORRECTED_BY` o `SUPERSEDES`.

### 20.3 Revisiones

```text
revision_type:
LEGITIMATE_RELEASE
PROVIDER_CORRECTION
PIPELINE_ERROR
SCHEMA_REINTERPRETATION
```

Ramas:

```text
as_released: append-only y point-in-time correcta
latest_revised: proyección con revisiones conocidas
```

- `LEGITIMATE_RELEASE`: crea nuevo snapshot; no invalida decisiones históricas correctas.
- `PROVIDER_CORRECTION` o `PIPELINE_ERROR`: marca descendientes afectados como `STALE`.
- `SCHEMA_REINTERPRETATION`: crea nueva versión de esquema y evita comparaciones silenciosas.

Screening usa `as_released` por defecto.

### 20.4 OpenLineage

OpenLineage se usa como formato y transporte de eventos. PostgreSQL mantiene el único grafo consultable. Marquez puede utilizarse como visor. No se mantienen dos grafos de verdad independientes.

---

## 21. Backend y frontend

### 21.1 Regla

> El frontend no calcula. Lee artefactos, vistas o endpoints versionados que ya contienen hechos y estados validados.

### 21.2 Vistas

| Ruta | Propósito | Fuentes | Puede escribir |
|---|---|---|---|
| `/dashboard` | Vote 2 y gates | bundle inmutable + estado de aprobación | voto mediante API validada |
| `/replay` | Señales, trades, equity y KPIs | bundles ACTION | nada |
| `/production` | Forward, canary y live | facts + métricas | nada |
| `/execution` | Órdenes, fills, riesgo y kill switch | vistas live | acciones RBAC controladas |
| `/forecasting` | Predicciones, intervalos y métricas | `forecast.*` | nada |
| `/analysis` | Contexto macro/news/LLM | artefactos namespaced | nada que afecte señal |
| `/hub` | Navegación y RBAC | control plane | nada |

### 21.3 Forecasting UI

Debe mostrar permanentemente:

> DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN

No contiene botones de aprobación, capital, órdenes ni etiquetas imperativas como “COMPRAR”. Puede mostrar “probabilidad estimada de subida: 58%”.

### 21.4 Passport dividido

- `v_strategy_passport_live`: vista ligera no materializada para órdenes, último fill, quarantine, reconciliación, riesgo y kill switch.
- `mv_strategy_performance_daily`: vista materializada para Sharpe, DD, DSR, atribución y métricas históricas.
- `v_strategy_passport`: composición de ambas superficies.

### 21.5 Transportes

1. File-based BFF para bundles, charts y forecasting estático.
2. PostgreSQL/API para estado live.
3. SSE/WS solo para streams necesarios.
4. Todos los archivos son proyecciones regenerables, no fuente de verdad.

---

## 22. Comportamiento ante fallas

| Falla | DIAGNOSTIC | ACTION | Respuesta |
|---|---|---|---|
| Datos rancias | Panel `STALE` | Bloquea nueva señal | incident + health snapshot |
| PNG ausente | Oculta imagen | Sin efecto | degradación elegante |
| Modelo diagnóstico falla | Resto del zoo continúa | Sin efecto | error visible |
| Componente activo falla | No aplica | Fail-closed | target cero o política declarada |
| Predicción sin lift | Veredicto negativo | No invalida automáticamente PnL positivo | separar ciencia predictiva y decisión |
| Señal inválida | No aplica | No se publica | contract violation |
| Paridad rota | Advertencia | Quarantine | incident crítico |
| Executor caído | Sin efecto | No nuevas órdenes | kill switch / recuperación |
| Broker discrepante | Sin efecto | Quarantine | reconciliación |
| Métrica ausente | `N/A` | Gate no aprueba | no imputar silenciosamente |
| Bundle incompleto | UI parcial | No promoción | verify fail |
| Forecast público caído | Página degradada | Posiciones sin cambio | barrera contractual |

---

## 23. Validaciones obligatorias de CI y registry

```text
✓ exactamente una fuente autoritativa por atributo
✓ toda estrategia ACTION referencia una familia y un cluster válidos
✓ toda celda cobrada aparece una vez en el ledger
✓ trials_charged coincide con el ledger
✓ N_MAX no entra en DSR
✓ surface=diagnostic no declara signal, allocate, execute o approve
✓ allocator solo consume strategy_output
✓ toda señal tiene version, valid_until y decision_fingerprint
✓ todo forecast tiene model_id, horizon y target_time
✓ misma decisión produce el mismo semantic_hash
✓ artefactos canónicos controlados producen bytes idénticos
✓ ninguna fila usa NaN o Infinity en JSON
✓ toda estrategia frozen tiene manifest y withdrawal protocol
✓ toda combinación de estados cumple la matriz de legalidad
✓ PAPER no puede tener FULL
✓ QUARANTINED bloquea aperturas
✓ WITHDRAWN exige exit_checklist PASS
✓ toda orden tiene idempotency_key única
✓ paper y live escriben contratos de order/fill comparables
✓ fact_pnl reconcilia con fills
✓ identidad contable del PnL pasa tolerancia
✓ toda métrica existe en el catálogo y guarda formula_version
✓ annualization proviene del asset registry
✓ comparaciones cross-asset respetan reloj y moneda
✓ candidatas concurrentes respetan alpha total
✓ juez sequential declara método o none
✓ portfolio_snapshot tiene cutoff y política de faltantes
✓ no existe lectura de “latest signal” sin cutoff
✓ frontend forecasting no importa endpoints de ejecución
✓ Vote 2 se emite sobre bundle inmutable
✓ una revisión legítima no invalida vintage histórico
✓ backfill incluye campeonas, candidatas, retiradas y baselines
```

---

## 24. Seguridad y gobierno operativo

- RBAC diferenciado: research, approver, risk, executor y auditor.
- Secretos del broker fuera de Airflow, repositorio y frontend.
- Overrides append-only con actor, motivo, ventana y expiración.
- Doble control para Vote 2 y cambios live sensibles.
- Backups de PostgreSQL y MinIO con pruebas de restauración.
- RPO y RTO definidos para datos, control plane y ejecución.
- Runbooks para broker caído, base caída, red caída, posición huérfana y fill tardío.
- Logs con correlación por `run_id`, `decision_fingerprint` y `order_id`.
- Ningún LLM puede crear señales, aprobar estrategias o enviar órdenes.

---

## 25. Stack: ahora y cuando duela

| Función | Ahora | Cuando exista necesidad demostrada |
|---|---|---|
| Orquestación | Airflow | Mantener |
| Control plane | PostgreSQL | Escalar réplicas/particiones si hace falta |
| Snapshots y bundles | MinIO + Parquet | Iceberg para time travel y administración de tablas |
| Modelos | MLflow | Mantener o escalar tracking |
| Linaje | PostgreSQL + eventos propios | OpenLineage + Marquez como transporte/visor |
| Point-in-time | `available_at` + joins as-of | Feast solo si aparecen múltiples equipos y alto reuse |
| Dashboard | Grafana + aplicación actual | Mantener |
| Infra telemetry | Airflow logs | Prometheus/OTel para contenedores y servicios |
| Optimización | NumPy/SciPy o cvxpy/OSQP | Servicio dedicado si aumenta frecuencia |
| Ejecución | Servicio aislado USD/COP | Adaptadores por broker y activo |

No se adoptan colas, Kubernetes o simuladores de microestructura por imitación institucional.

---

## 26. Roadmap consolidado

### Etapa 0 — Constitución técnica

Entregables:

1. Matriz de fuente de verdad.
2. Convenciones de IDs y tiempos.
3. Semántica de revisiones.
4. Matriz de legalidad de estados.
5. Política de serialización canónica.
6. Política de permisos ACTION/DIAGNOSTIC.

Criterio de terminado: CI puede rechazar una declaración inválida antes de ejecutar un DAG.

### Etapa 0.5 — Diagnóstico rápido no durable

Ejecutar `timing_ratio` y atribución preliminar sobre bundles existentes como script desechable. No crea tablas ni decisiones de promoción.

### Etapa 1 — Trials y familias

- Ledger append-only.
- Familias transversales.
- Clusters controlados.
- `N_family`, `N_cluster`, `N_global`.
- DSR bajo las tres sensibilidades.

Es urgente porque una contabilidad incorrecta de trials no se reconstruye fácilmente después.

### Etapa 2 — Identidad

- Fingerprints.
- Hashes.
- Canonical writer.
- Spine.
- CI de paridad.

Debe preceder tablas de hechos y backfills.

### Etapa 3 — Event sourcing y hechos

- Order/fill común para replay, paper, canary y live.
- `fact_position`.
- `fact_pnl` y residual.
- Reconciliación.
- Idempotencia.

### Etapa 4 — Métricas

- Catálogo versionado.
- Motor único.
- `metric_event`.
- Thresholds históricos.
- Intervalos de confianza.

### Etapa 5 — Backfill anti-sesgo

Incluir:

- campeonas;
- candidatas;
- retiradas;
- baselines;
- años completos disponibles;
- todos los entornos reconstruibles.

### Etapa 6 — Linaje

- Nodes/edges.
- Camino dorado.
- Revisión tipificada.
- Cascada selectiva.
- Emisión OpenLineage.

### Etapa 7 — Factories y Passport

- `asset_data_factory.py`.
- `strategy_factory.py`.
- `forecast_factory.py`.
- Ejecución paralela con factory actual.
- Diff semántico de bundles.
- Passport live y performance.

### Etapa 8 — Portfolio snapshot y allocator shadow

- Barrera temporal.
- Baseline inverse-vol con caps.
- HRP shadow.
- Multiplicadores solo reductores.
- Costos y turnover.
- Juez propio del allocator.

### Etapa 9 — Canary de ejecución

Solo después de:

- paridad semántica verde;
- idempotencia probada;
- pre-trade risk;
- reconciliación pre/intra/EOD;
- kill switch;
- RETIRING probado;
- runbooks y recuperación;
- mínimos canary cumplidos.

USD/COP L7 es lo último en migrarse.

---

## 27. Criterios de aceptación de producción

La Fabric se considera apta para ampliar live cuando:

1. El 100% de señales live tiene `decision_fingerprint` reproducible.
2. Un replay independiente reproduce el `semantic_hash` de la señal.
3. Un retry no genera órdenes duplicadas.
4. Paper y live usan el mismo ledger de eventos y esquema.
5. `fact_position` y `fact_pnl` reconcilian con fills y broker.
6. El portfolio target siempre referencia un portfolio snapshot con cutoff.
7. No existe dependencia entre forecast diagnóstico y ejecución.
8. Todos los estados cumplen la matriz de legalidad.
9. El proceso `RETIRING` fue ensayado de punta a punta.
10. El kill switch funciona aun con Airflow caído.
11. Las vistas live no dependen de un refresh materializado diario.
12. Todos los gates y métricas provienen del motor y catálogo únicos.
13. El backfill no presenta sesgo de supervivencia por excluir retiradas o baselines.
14. El linaje de una fila de PnL llega hasta raw snapshot.
15. Los runbooks de incidentes fueron probados mediante simulacros.

---

## 28. Decisiones explícitamente rechazadas

- Un DAG por activo que contenga todas las estrategias indefinidamente.
- Un único YAML gigante con datos, estrategias, forecasting y ejecución mezclados.
- Forecasting como señal por defecto.
- `normalize(clip(...))` en el allocator.
- Consultar el último valor sin cutoff.
- Paper sin órdenes y fills.
- Byte parity sobre formatos externos no canónicos.
- Semantic hash débil donde sí se controla el escritor.
- Invalidar vintages legítimos.
- Retirar una estrategia eliminando inmediatamente su DAG.
- Airflow como única defensa contra duplicación.
- Canary definido solo por semanas calendario.
- `timing_ratio` como prueba de alfa.
- Incrementar capital por Sharpe rolling ruidoso en allocator v1.
- Backfill exclusivo de campeonas.
- Passport materializado diario como fuente de estado live.

---

## 29. Estructura sugerida del repositorio

```text
config/
  assets/
  strategies/
  forecasts/
  macro/
  metrics/
  book/

registries/
  families/
  ledger.jsonl
  research_clusters.yaml

contracts/
  strategy_output.py
  forecast_output.py
  portfolio_snapshot.py
  portfolio_target.py
  order_events.py

control_plane/
  identity/
  lineage/
  metrics/
  governance/
  reconciliation/

execution/
  service/
  pretrade/
  adapters/
  kill_switch/

airflow/dags/
  asset_data_factory.py
  strategy_factory.py
  forecast_factory.py
  control_portfolio_snapshot.py
  book_allocator.py
  control_weekly_judges.py
  control_system_health.py

scripts/analysis/
  qlab.py
  timing_ratio_oneoff.py

docs/
  architecture/
  withdrawal/
  runbooks/
```

---

## 30. Glosario

- **Activo:** mercado o instrumento con calendario, reloj, datos, costos y capacidades físicas.
- **Sleeve:** política ACTION congelada sobre un activo.
- **Familia:** pregunta económica pre-registrada que contiene celdas y consume trials.
- **Forecast spec:** experimento o producto DIAGNOSTIC por modelo, target y horizonte.
- **Trial:** mirada o variante que consume presupuesto estadístico.
- **Snapshot:** conjunto inmutable de datos identificado y fechado.
- **Bundle:** artefactos inmutables publicados por una estrategia.
- **Passport:** vista derivada que reúne identidad, gobierno, linaje, desempeño, riesgo y ejecución.
- **Portfolio snapshot:** conjunto temporalmente coherente de señales aceptadas para un cutoff.
- **Portfolio target:** exposición resultante del allocator.
- **Quarantine:** bloqueo operacional ortogonal al estado de investigación.
- **Canary:** capital limitado sujeto a criterios mínimos de forward y operación.
- **Semantic hash:** hash del contenido normalizado.
- **Bytes hash:** hash de integridad física.
- **Point-in-time:** uso exclusivo de información disponible en el momento de la decisión.

---

## 31. Regla de cierre

> Un forecast puede ser preciso y no producir una estrategia rentable. Una estrategia puede ser rentable aunque su predictor aislado sea débil. El sistema debe medir ambas cosas honestamente, impedir que se confundan y conservar el linaje que explica cómo una observación terminó —o no terminó— en una orden y en PnL.

---

## 32. Disposición consolidada de la auditoría

| Riesgo auditado | Resolución normativa en este documento | Estado |
|---|---|---|
| Fingerprint incluía entorno y luego se comparaba entre entornos | Separación entre `spec`, `decision` y `execution_fingerprint`; paridad por `decision_fingerprint → semantic_hash` | Cerrado en diseño |
| Byte parity universal era inviable | Bytes idénticos solo para escritores canónicos controlados; hash semántico para formatos externos | Cerrado en diseño |
| Paper no registraba órdenes y fills | Contrato event-sourced común para replay, paper, canary y live | Cerrado en diseño |
| `normalize(clip(...))` rompía caps y vol target | Optimización restringida sin normalización final | Cerrado en diseño |
| Allocator mezclaba señales con edades distintas | `portfolio_snapshot` con cutoff, aceptación, staleness y fallbacks | Cerrado en diseño |
| Airflow podía convertirse en bus de ejecución | Portfolio target inmutable y servicio externo con idempotencia | Cerrado en diseño |
| Retiro podía eliminar el DAG con posiciones abiertas | Estado `RETIRING`, tier `EXIT_ONLY` y checklist obligatorio | Cerrado en diseño |
| Revisiones legítimas podían invalidar decisiones PIT correctas | Taxonomía de revisiones y ramas `as_released` / `latest_revised` | Cerrado en diseño |
| Existían dobles fuentes de verdad | Matriz constitucional de autoridad por entidad | Cerrado en diseño |
| Familias podían dividirse para ocultar múltiples pruebas | Cluster controlado y reporte de N/DSR familiar, cluster y global | Cerrado en diseño |
| Canary de ocho semanas era insuficiente | Mínimos parametrizados por decisiones, fills, regímenes e incidentes | Cerrado en diseño |
| Juez secuencial no estaba definido | Método `mSPRT`, `e_process` o `none`; actuar antes del horizonte se registra como violación | Cerrado en diseño |
| `timing_ratio` se presentaba como prueba de alfa | Reclasificado como atribución diagnóstica con residual e intervalos | Cerrado en diseño |
| `strategy_output` mezclaba scores, salud y liquidez | Contrato tipado; salud y liquidez provienen de servicios independientes | Cerrado en diseño |
| Estados mezclaban investigación, capital y salud | Tres dimensiones más matriz de legalidad | Cerrado en diseño |
| `m_forward` premiaba rachas | En v1 solo puede reducir riesgo; incluye histéresis | Cerrado en diseño |
| HRP podía entrar directamente | Baseline inverse-vol incumbente; HRP únicamente en shadow hasta superar juez | Cerrado en diseño |
| Faltaba riesgo pre-trade y kill switch externo | Gate por orden, reconciliación previa y kill switch independiente de Airflow | Cerrado en diseño |
| OpenLineage podía duplicar el grafo | OpenLineage como transporte; PostgreSQL como único grafo | Cerrado en diseño |
| Passport materializado diario se usaba para estado live | Separación vista live y materialized view histórica | Cerrado en diseño |
| Roadmap comenzaba por hechos antes de identidad | Identidad y constitución preceden hechos y backfill | Cerrado en diseño |
| Replay y forecasting podían confundirse | Superficies, contratos, permisos, DAGs, vistas y trials separados | Cerrado en diseño |

**Nota de auditoría:** “cerrado en diseño” no significa “implementado”. Cada fila debe convertirse en código, migración, test y evidencia antes de considerarse cerrada operacionalmente.

---

# PARTE II — PLAN FINAL DE DATOS, COLUMNAS, FEATURES Y RESAMPLEOS

## 33. Propósito de esta extensión

Esta parte convierte la auditoría de la base `usdcop_trading` y los contratos actuales de estrategias en un plan único de implementación. Define:

- qué tabla o artefacto es fuente de verdad;
- qué columnas consume cada estrategia;
- dónde se calculan sus features;
- qué resampleos están permitidos;
- cómo se propaga `available_at`;
- cómo se separan ACTION, DIAGNOSTIC y RL;
- qué tablas actuales se conservan, transforman o retiran;
- cómo migrar sin alterar señales ni bundles existentes.

El perfil del 27 de julio de 2026 contiene 59 tablas de dominio, 964 columnas, aproximadamente 2,82 millones de filas y 7 hypertables. Cerca de la mitad de las tablas están vacías y tres tablas concentran casi todo el volumen. Por tanto, el problema principal no es capacidad de PostgreSQL/TimescaleDB, sino duplicidad semántica, contratos incompletos y múltiples fuentes de verdad.

## 34. Leyes de datos y features

1. **Raw es append-only.** Ninguna corrección destruye el valor recibido del proveedor.
2. **Canonical es una selección versionada.** Cada fila debe apuntar al raw, política de calidad y corrida que la produjo.
3. **Una estrategia no consulta tablas genéricas directamente.** Consume un `feature_snapshot_id` generado por un contrato versionado.
4. **Cada estrategia declara exactamente sus columnas y features.** No existe una lista implícita reconstruida desde código.
5. **ACTION y DIAGNOSTIC pueden compartir raw/canonical, pero no comparten automáticamente feature sets.**
6. **Las estadísticas de normalización pertenecen a una versión entrenada**, nunca a una definición global de feature.
7. **Todo resampleo es explícito, versionado y trazable.** Se distingue `provider_official` de `resampled`.
8. **Nunca se resamplea una barra ya resampleada** cuando existe acceso al grano nativo.
9. **El point-in-time gobierna.** Una observación entra únicamente cuando `available_at <= decision_cutoff`.
10. **Los retornos se almacenan como decimales:** `0.01 = 1%`.
11. **Las columnas de explicación de una política no invaden el esquema común.** Van en `decision_components JSONB` con versión.
12. **El frontend no calcula features, retornos, métricas ni resampleos.**

## 35. Arquitectura física objetivo

```text
PostgreSQL + TimescaleDB
│
├── reference
│   ├── asset
│   ├── instrument
│   ├── provider
│   ├── provider_symbol
│   ├── calendar
│   ├── bar_interval
│   └── unit
│
├── market
│   ├── raw_bar
│   ├── canonical_bar
│   ├── ingestion_run
│   ├── resample_run
│   └── quality_event
│
├── macro
│   ├── series
│   ├── observation
│   ├── source_document
│   └── bundle_health
│
├── feature
│   ├── definition_projection
│   ├── feature_set_projection
│   ├── snapshot
│   └── snapshot_column
│
├── forecast
│   ├── spec_projection
│   ├── output
│   └── score
│
├── action
│   └── strategy_signal
│
├── execution
│   ├── order
│   ├── order_status_event
│   ├── fill_event
│   ├── fill_correction_event
│   └── reconciliation_event
│
├── fact
│   ├── position
│   └── pnl
│
├── control
│   ├── family
│   ├── trial
│   ├── strategy_projection
│   ├── run
│   ├── portfolio_snapshot
│   ├── metric_event
│   ├── lineage_node
│   ├── lineage_edge
│   ├── incident
│   └── vote_event
│
├── news
├── auth
├── secret
└── readmodel / bi
    └── vistas y materialized views regenerables
```

Fuera de PostgreSQL:

```text
Git       → definiciones y SSOT versionados
MLflow    → runs, parámetros, métricas de entrenamiento y modelos registrados
MinIO     → raw documents, Parquet, datasets, feature snapshots, bundles y modelos
Grafana   → lectura de vistas y metric_event
Secret manager → secretos reales; PostgreSQL guarda referencias
```

## 36. Identidades canónicas

Se prohíbe usar símbolos libres como identidad principal.

```yaml
asset_id: btc
instrument_id: binance_btcusdt_perpetual
provider_id: binance
provider_symbol: BTCUSDT
venue_id: binance_usdt_m
bar_interval: PT5M
calendar_id: utc_24_7
```

Ejemplo SPX:

```yaml
asset_id: spx500
instruments:
  - spx500_index
  - spy_etf
  - es_future
```

`SPX500`, `SPX/500` y `SPY` no son sinónimos de instrumento. `5m`, `5min`, `1d` y `1day` deben migrarse a códigos ISO-8601 o a una dimensión única.

## 37. Contrato de barras

### 37.1 Raw

```text
market.raw_bar
--------------
raw_bar_id
instrument_id
provider_id
provider_symbol
bar_interval
event_time
provider_published_at
retrieved_at
ingested_at
open
high
low
close
volume
source_payload_hash
ingestion_run_id
```

### 37.2 Canonical

```text
market.canonical_bar
--------------------
canonical_bar_id
instrument_id
bar_interval
event_time
open
high
low
close
volume
available_at
bar_method                 -- provider_official | resampled
source_raw_bar_id
quality_policy_version
quality_status
calendar_id
canonical_run_id
```

### 37.3 Regla sobre barras diarias

La barra diaria oficial de un proveedor y una barra diaria derivada de 5 minutos son observaciones distintas. Si ambas se conservan:

```text
bar_method = provider_official
bar_method = resampled
```

Una estrategia debe declarar cuál consume. No se reemplaza silenciosamente una por otra.

## 38. Política normativa de resampleo

### 38.1 Orden de operaciones

```text
raw native bars
→ canonicalización del grano nativo
→ filtro de calidad
→ resampleo al grano objetivo, si aplica
→ propagación de available_at
→ as-of join con macro
→ features causales
→ lags/shifts declarados
→ feature snapshot inmutable
```

### 38.2 Agregación OHLCV

```text
open   = primer open válido
high   = máximo high
low    = mínimo low
close  = último close válido
volume = suma, conservando decimales cuando el proveedor los publica
```

No se publica una barra parcial como final salvo que el contrato declare `partial_bar_allowed=true`.

### 38.3 `available_at` derivado

```text
available_at_bar = max(available_at de inputs) + compute_latency_policy
available_at_feature = max(available_at de todos sus inputs) + feature_compute_latency
```

`ingested_at` no sustituye a `available_at`. Cuando la disponibilidad se reconstruye, se guarda `availability_policy` y `availability_quality`.

### 38.4 Sesiones

- USD/COP: calendario Colombia y sesión declarada; no se rellenan precios fuera de sesión.
- SPX: calendario NYSE; decisión al cierre y ejecución next-open.
- XAU: calendario de metales 23h; el corte diario debe estar congelado en el asset registry.
- BTC: calendario UTC 24/7; no existe “fin de semana” y la anualización es 365.

### 38.5 Continuous aggregates

Se usan como proyecciones de serving y no como nueva fuente autoritativa. Deben poder reconstruirse desde `market.canonical_bar` o desde la barra oficial seleccionada.

## 39. Macro point-in-time

La fuente canónica pasa a ser larga:

```text
macro.observation
-----------------
series_id
observation_date
vintage_id
reference_date
release_date
available_at
value
unit_id
source_document_id
availability_policy
availability_quality
promotion_eligible
```

Valores permitidos de calidad:

```text
OFFICIAL_VINTAGE
OFFICIAL_RELEASE_TIMESTAMP
CONSERVATIVE_RECONSTRUCTION
RESEARCH_ONLY_RECONSTRUCTION
UNKNOWN
```

Las tablas anchas diaria, mensual y trimestral pasan a vistas o materialized views. `is_complete` global se elimina; la salud se evalúa por bundle requerido por una estrategia.

```text
macro.bundle_health
-------------------
bundle_id
as_of
required_series
missing_series
stale_series
status
```

Mientras `publication_date` mensual/trimestral esté incompleto, esas observaciones no pueden alimentar una promoción salvo excepción firmada.

## 40. Catálogo y feature sets

### 40.1 Definición en Git

```yaml
feature_id: realized_vol_21d
unit: decimal_annualized
source_contract: market.canonical_bar
transform:
  code_reference: features/volatility.py:realized_vol
  code_hash: sha256:...
lookback_bars: 21
input_interval: P1D
lag_bars: 1
causality_policy: close_t_available_before_decision
null_policy: fail
```

### 40.2 Feature set por estrategia

```yaml
feature_set_id: usdcop_smart_simple_v11_action_v1
surface: action
ordered_features:
  - feature_id: technical_01
  - feature_id: technical_02
  - feature_id: dxy_t1
  - feature_id: wti_t1
  - feature_id: vix_t1
  - feature_id: embi_t1
resample_policy_id: usdcop_daily_official_v1
macro_bundle_id: usdcop_action_macro_v1
```

La lista completa de las 21 features técnicas de `smart_simple_v11` debe importarse del SSOT congelado actual. No se reconstruye ni se adivina desde nombres de columnas históricas.

### 40.3 Normalización

```text
normalization_snapshot_id
model_snapshot_id
training_cutoff
ordered_feature_hash
mean/std o parámetros del transformer
semantic_hash
```

`normalization_mean` y `normalization_std` dejan de vivir como valores globales en `config.feature_definitions`.

### 40.4 Feature snapshot

```text
feature.snapshot
----------------
feature_snapshot_id
feature_set_id
asset_id
instrument_id
surface
as_of_start
as_of_end
row_count
ordered_feature_hash
data_snapshot_id
semantic_hash
storage_uri
created_by_run_id
quality_status
```

El contenido grande vive en MinIO/Parquet. PostgreSQL conserva identidad, calidad y linaje.

## 41. Matriz resumida por estrategia

| Sleeve / superficie | Grano de decisión | Grano de ejecución | Datos principales | Macro | Resampleo autorizado |
|---|---|---|---|---|---|
| `usdcop_smart_simple_v11` ACTION | diario/semanal | PT5M | OHLCV diario oficial + PT5M para TP/HS | DXY, WTI, VIX, EMBI T-1 | no usar PT5M→P1D como serie oficial sin cambio de versión |
| `usdcop_smart_simple_v12/v14` ACTION paper | mismo snapshot que v11 | simulador PT5M | misma señal base | igual v11 | ninguno adicional; cambian política de riesgo |
| PPO/RL USD/COP | PT5M | PT5M | contrato L1 de 15 dimensiones | macro diaria as-of | 1h/4h derivados desde PT5M o lags equivalentes, según contrato congelado |
| `xauusd_trend_simple_v1` ACTION | P1D; corrida semanal actual | paper | OHLC diario oficial, principalmente `close` | ninguna en ACTION actual | oficial P1D; no derivar autoridad desde PT5M |
| `gold_dynamic_exit` retirada | mismo precio base | paper histórico | igual base + política de salida | ninguna | conservar versión congelada |
| `btcusdt_hodl_b1` ACTION | P1D | paper 24/7 | close/returns/volatilidad realizada | ninguna | diario oficial Binance, boundary UTC |
| Retadores BTC funding/basis | P1D | paper | precio + `crypto_derivatives_daily` | no aplica | activar solo tras calidad/ventana mínima y nuevo family trial |
| `spx500_daily_ma200_v1` ACTION | P1D | next-open simulado | `close`, `open_to_open_return`, calendario | ninguna | índice oficial diario; no resample intradía |
| `spx500_regime_gated_v1` ACTION | P1D | next-open simulado | MA200, TSMOM 12-1, proxies de precio | ninguna externa | igual SPX MA200 |
| Zoo forecasting USD/COP | P1D por horizonte | ninguna | 21–25 técnicas+macro según spec | sí, spec independiente | dataset diario PIT, walk-forward |
| Zoo forecasting XAU | P1D | ninguna | 19 features | DXY y VIX; no WTI/EMBI Colombia | dataset diario PIT |
| Forecasting BTC | P1D | ninguna | feature set diagnóstico versionado | según spec | nunca conectado a allocator |
| Forecasting SPX | deshabilitado actual | ninguna | — | — | no crear hasta nueva familia pre-registrada |

## 42. USD/COP ACTION: `smart_simple_v11`

### 42.1 Fuentes y columnas

**Modelo/señal semanal:**

```text
market.canonical_bar
  instrument_id = usdcop_spot
  bar_interval = P1D
  bar_method = provider_official

columnas base:
  event_time
  open
  high
  low
  close
  volume, si está declarado en el SSOT
  available_at
  quality_status
```

**Macro:**

```text
DXY
WTI
VIX
EMBI Colombia
```

Cada valor se selecciona con as-of join y T-1 según el contrato. Macro stale por encima del límite bloquea la señal.

**Ejecución y control de salidas:**

```text
market.canonical_bar
  instrument_id = usdcop_spot
  bar_interval = PT5M

columnas:
  event_time
  open
  high
  low
  close
  available_at
```

### 42.2 Features

El material disponible confirma:

- 21 features técnicas congeladas;
- 4 macro T-1: DXY, WTI, VIX y EMBI;
- predictor Ridge + Bayesian Ridge;
- gate Hurst;
- volatilidad realizada para sizing y TP/HS;
- confianza por niveles.

El listado exacto de 21 features técnicas no está contenido íntegramente en el perfil de base. Debe copiarse del manifiesto congelado de `smart_simple_v11` y quedar en `feature_set_id`. Cualquier diferencia cambia el `ordered_feature_hash` y obliga nueva versión/freeze.

### 42.3 Columnas de decisión, no features

Las columnas actuales siguientes son outputs o explicaciones de la política y no deben confundirse con inputs del modelo:

```text
ensemble_return
direction
realized_vol_21d
raw_leverage
clipped_leverage
adjusted_leverage
confidence_tier
sizing_multiplier
skip_trade
hard_stop_pct
take_profit_pct
hurst_exponent
regime
regime_leverage_scaler
rolling_wr_8w
dl_leverage_scaler
effective_hs_pct
effective_tp_pct
```

Destino objetivo:

```text
action.strategy_signal
  + decision_components JSONB
  + decision_schema_version
```

### 42.4 Resampleo y temporalidad

- El modelo usa barra diaria oficial; no reconstruye su historia de entrenamiento desde PT5M sin crear una nueva versión de datos.
- El executor consume PT5M directamente.
- La tarea cada 30 minutos no convierte el dato a 30 minutos; consulta el estado producido por barras de 5 minutos.
- El entrenamiento es expansivo y semanal; el scaler se ajusta solo con train.
- La señal lunes no puede leer datos posteriores al cutoff firmado.

### 42.5 v12 y v14

`v12` y `v14` deben referenciar el mismo `decision_input_snapshot_id` y la misma señal base que v11. Solo cambia la política de riesgo. Así la comparación es pareada y no crea divergencia por features o datos.

## 43. USD/COP RL pausado

El esquema actual `inference_features_5m` documenta un contrato de 15 dimensiones:

```text
log_ret_5m
log_ret_1h
log_ret_4h
rsi_9
atr_pct
adx_14
dxy_z
dxy_change_1d
vix_z
embi_z
brent_change_1d
rate_spread
usdmxn_change_1d
position
time_normalized
```

Más metadata:

```text
time
builder_version
updated_at
```

Reglas:

1. L1 sigue siendo la única capa autorizada para producir este contrato.
2. `log_ret_1h` y `log_ret_4h` se calculan desde el grano PT5M o mediante lags equivalentes declarados; no desde tablas manuales distintas.
3. Macro diaria se une as-of y nunca se adelanta.
4. `position` es estado del entorno y debe identificarse como feature endógena.
5. Mientras RL esté pausado, la tabla vacía se mueve a `staging_contract` o se convierte en vista; no se presenta como pipeline operativo.

## 44. Forecasting USD/COP DIAGNOSTIC

```text
forecast__usdcop__weekly_zoo
  canonical daily
  → forecast feature set
  → walk-forward
  → 9 modelos × 7 horizontes
  → forecast.output
  → forecast.score
```

Reglas:

- El feature set diagnóstico es distinto del ACTION aunque comparta algunas columnas.
- El perfil actual de `forecast_h5_predictions` muestra dos modelos para H5; eso no sustituye el registry completo del zoo.
- `predicted_return_pct` debe migrarse a `predicted_return_decimal`.
- Ningún forecast puede insertarse en `action.strategy_signal`.
- Probar un nuevo modelo, feature, target u horizonte cobra trial en la familia predictiva.

## 45. XAU/USD ACTION: `xauusd_trend_simple_v1`

### 45.1 Fuente

```text
instrument_id = xauusd_spot
bar_interval = P1D
bar_method = provider_official
calendar_id = metals_23h
annualization = 252
```

Columnas mínimas:

```text
event_time
close
open, para ejecución/PnL cuando aplique
high/low, solo si el contrato de costos o salida los usa
available_at
```

### 45.2 Features

La estrategia ACTION usa votos de medias móviles simples y exposición continua. Las ventanas exactas no están enumeradas en el perfil suministrado; deben leerse del YAML congelado de la estrategia:

```yaml
params:
  sma_windows: [...]
  vote_rule: ...
  exposure_cap: ...
```

No se deben inventar ventanas durante la migración.

### 45.3 Resampleo

- La barra P1D oficial es autoritativa.
- PT5M/1h pueden usarse para monitor o control de calidad, no para sustituir el histórico congelado.
- El DAG corre semanalmente, pero el replay puede mantener una exposición diaria derivada de barras diarias. La cadencia de decisión exacta debe quedar explícita en el registry.

### 45.4 Forecasting XAU

Feature set diagnóstico separado:

- 19 features según el sistema actual;
- incluye DXY y VIX;
- excluye WTI y EMBI Colombia;
- usa walk-forward y métricas de lift/error;
- no genera señal.

`gold_dynamic_exit` se conserva como estrategia retirada con sus bundles y feature set congelado, no como fila especial en la campeona.

## 46. BTC/USDT ACTION: `btcusdt_hodl_b1`

### 46.1 Fuente

```text
instrument_id = binance_btcusdt_spot_o_perpetual_segun_registry
bar_interval = P1D
calendar_id = utc_24_7
annualization = 365
bar_method = provider_official
```

Columnas mínimas:

```text
event_time
close
open, para el contrato de ejecución
high
low
volume, si el quality/risk contract lo exige
available_at
```

### 46.2 Features

La campeona actual es exposición al beta con vol targeting. Su núcleo es:

```text
return series
realized volatility
vol target
exposure cap/floor
```

No utiliza automáticamente funding, basis, on-chain o flows. Añadirlos constituye una nueva familia/celda y cobra trials.

### 46.3 Derivados y nuevas familias

`crypto_derivatives_daily` conserva funding con historia; basis tiene cobertura parcial y open interest/long-short tienen historia muy corta. Antes de habilitar un retador:

```text
coverage gate
available_at gate
minimum history gate
provider consistency gate
family declaration
screening cutoff
```

`crypto_onchain_daily`, `crypto_flows_daily` y `crypto_event_calendar` permanecen en `staging_contract` mientras no tengan productor/consumidor activos.

### 46.4 Resampleo

- Boundary diario UTC congelado.
- No existe tratamiento weekday.
- Una barra diaria derivada sirve para controles, pero la estrategia congelada conserva su fuente oficial.
- La anualización y ventanas se interpretan en 365 días.

## 47. SPX500 ACTION

### 47.1 Fuente común

```text
instrument_id = spx500_index
bar_interval = P1D
bar_method = provider_official
calendar_id = nyse
annualization = 252
execution_ref = next_open
```

Columnas verificadas del dataset congelado:

```text
timestamp / event_time
available_at
close
open_to_open_return
```

Los demás proxies derivados de precio deben quedar enumerados en el feature manifest; no se reemplazan por macro externa sin nueva familia.

### 47.2 `spx500_daily_ma200_v1`

```text
input: close
feature: SMA(close, 200 sesiones)
decision: cierre t
execution/PnL: open t+1
warmup: al menos 200; registry actual recomienda 252
```

No entrena y no usa macro externa.

### 47.3 `spx500_regime_gated_v1`

```text
inputs:
  close
  MA200
  TSMOM 12-1
  proxy/régimen derivado de precio congelado

output:
  exposición acotada [0, 1.5]
```

No se resamplea desde intradía. La barra oficial del índice y el retorno de ejecución next-open deben conservar su semántica.

### 47.4 Forecasting

Actualmente no existe zoo productivo para SPX. No se crea por simetría arquitectónica. Reabrir forecasting exige familia pre-registrada y contrato DIAGNOSTIC; nunca se conecta directamente a ACTION.

## 48. Reglas comunes para forecasting

1. Cada `forecast_spec` declara target, horizonte, feature set, cutoff, purga, scaler y baseline.
2. El target no se infiere del nombre de una columna.
3. H1, H5, H10, H15, H20, H25 y H30 son specs o particiones explícitas.
4. El walk-forward guarda para cada predicción el `training_cutoff`, `data_snapshot_id`, `feature_snapshot_id` y `model_snapshot_id`.
5. DA se acompaña de balanced-DA y lift contra baseline mayoritario.
6. Una predicción sin lift puede publicarse como diagnóstico, no como señal.
7. El frontend muestra probabilidades/scores con su tipo; no traduce automáticamente a COMPRAR/VENDER.

## 49. Disposición de tablas actuales

### 49.1 Mantener y migrar

| Tabla actual | Acción |
|---|---|
| `market_ingestion_manifest` | migrar a `market.ingestion_run` |
| `market_session_calendar` | migrar/proyectar desde `reference.calendar` |
| `dim_asset` | expandir a asset/instrument/provider y convertir en SSOT proyectada |
| `asset_native_ohlcv` | migrar a `market.raw_bar` |
| `usdcop_m5_ohlcv` | renombrar/migrar a canonical PT5M multi-instrumento |
| `asset_daily_ohlcv` | separar oficial/resampled; vista o canonical P1D |
| `macro_indicators_pit` | núcleo de `macro.observation` |
| `macro_banrep_forwards_monthly` | conservar series válidas; retirar `forward_rate` si no existe productor |
| `macro_remesas_monthly` | migrar a observaciones largas |
| `crypto_derivatives_daily` | conservar con gates de cobertura/calidad |
| núcleo de news | conservar metadata, artículos, ingesta, digests y snapshots |
| `audit_log` | conservar y endurecer como ledger append-only |

### 49.2 Convertir en vistas/proyecciones

```text
macro_indicators_daily
macro_indicators_monthly
macro_indicators_quarterly
bi.fact_forecasts
bi.fact_consensus
bi.fact_model_metrics
bi.fact_inference_runs
trading_state
equity_snapshots
trades_history
```

### 49.3 Reemplazar por contratos comunes

```text
forecast_h5_predictions   → forecast.output
forecast_h5_signals       → action.strategy_signal
forecast_h5_executions    → execution.order/status/fill + fact
forecast_h5_paper_trading → metric_event y hechos paper
forecast_h5_subtrades     → fills/trade readmodel
signals / sb_signals      → action.strategy_signal
sb_executions             → execution event sourcing
```

### 49.4 Eliminar como duplicidad de MLflow/control plane

```text
experiment_runs
experiment_comparisons
experiment_deployments
model_registry
metrics.model_performance
```

Si se requiere consulta SQL, se crea una proyección mínima por IDs, no un registry competidor.

### 49.5 Mover a `staging_contract` o retirar

```text
crypto_onchain_daily
crypto_flows_daily
crypto_event_calendar
crypto_exposure_signals
macro_variable_snapshots
daily_analysis
weekly_analysis
news_cross_references
inference_features_5m, mientras RL siga pausado
```

Una tabla vacía solo permanece si tiene owner, productor, consumidor, fecha de activación y test.

### 49.6 Seguridad

Consolidar:

```text
sb_exchange_credentials
user_exchange_keys
```

hacia:

```text
secret.external_account
  credential_id
  owner_id
  provider
  secret_reference
  key_version
  status
  last_validated_at
```

El secreto vive fuera de PostgreSQL. `sb_users` migra a `auth.*`; todos los timestamps pasan a `timestamptz`.

## 50. Calidad y cuarentena

Anomalías como USD/MXN `175814` o USD/CLP `94890` no se corrigen sobre raw.

```text
raw observation
→ quality FAIL
→ quarantine
→ comparación con fuente alterna
→ correction event
→ nueva canonical version
```

Cada instrumento declara reglas versionadas:

```yaml
instrument_id: usdmxn_spot
valid_range: [5, 100]
max_return_per_day: 0.20
```

Cero no equivale a missing. Sentimiento no calculado, liquidaciones no disponibles o fields sin productor deben publicarse como `UNAVAILABLE`, no como `0`.

## 51. TimescaleDB

### Ahora

- Una instancia puede manejar el volumen actual.
- Hypertables solo para series con crecimiento real.
- Índices por `(instrument_id, bar_interval, event_time DESC)`.
- Revisar chunk interval con tamaños físicos y patrón de consulta.
- Compresión por `instrument_id` y orden temporal.
- Continuous aggregates para serving, no para crear una segunda verdad.

### Antes de capital live material

- Separar `execution_db` o al menos recursos/roles dedicados.
- Probar backup/restore.
- Medir bloat, autovacuum y queries lentas.
- No aplicar retención destructiva hasta garantizar raw frío en MinIO y reconstrucción probada.

## 52. DAGs y assets de datos revisados

```text
asset__{asset}__data
  ingest native
  → raw_bar
  → canonical_bar
  → quality
  → official/resampled projections
  → data snapshots

feature__{feature_set_id}
  consume data snapshot explícito
  → as-of joins
  → compute ordered features
  → feature snapshot

strat__{sleeve_id}
  consume feature_snapshot ACTION
  → model/policy
  → strategy_output

forecast__{forecast_spec_id}
  consume feature_snapshot DIAGNOSTIC
  → walk-forward/predict/score
  → forecast_output
```

Una estrategia no ejecuta SQL ad hoc contra “últimos datos”. El input se resuelve por IDs de snapshot.

## 53. Validaciones CI adicionales

```text
✓ cada strategy declara feature_set_id y resample_policy_id
✓ cada forecast declara feature_set_id distinto o sharing explícito
✓ ordered_feature_hash coincide entre train, replay, paper y live
✓ ninguna normalización global se usa sin model_snapshot_id
✓ provider_official y resampled nunca se mezclan sin bar_method
✓ ningún feature usa input con available_at posterior al decision_cutoff
✓ toda feature tiene unit, lookback, lag, null_policy y code_hash
✓ retornos almacenados como decimal
✓ símbolos/timeframes libres rechazados por FK
✓ v12/v14 usan el mismo decision_input_snapshot que v11 en comparación pareada
✓ BTC funding/on-chain no entra a hodl_b1 sin nueva familia
✓ SPX ACTION no consume macro externa no registrada
✓ forecasting no puede escribir action.* ni execution.*
✓ una tabla vacía sin owner/productor/consumidor falla el inventory check
✓ modelo SYNTHETIC solo existe en environment=demo y execution_eligible=false
```

## 54. Roadmap final de implementación de datos

### Fase D0 — Congelar y descubrir

1. Exportar DDL, PK, FK, índices, tamaños, writers y readers.
2. Etiquetar cada tabla `SSOT | projection | cache | staging | deprecated`.
3. Congelar escrituras nuevas de esquemas paralelos.
4. Crear tests de paridad sobre bundles y señales actuales.

**Terminado:** existe dependency graph y no se borra nada a ciegas.

### Fase D1 — Identidad, tiempo, unidades y seguridad

1. Crear `reference.*`.
2. Migrar símbolos y timeframes a IDs canónicos.
3. Normalizar retornos a decimal.
4. Migrar timestamps a `timestamptz`.
5. Consolidar credenciales y sacar secretos de la base.
6. Aislar modelos demo/sintéticos.

**Terminado:** ningún writer nuevo usa símbolos libres, `_pct` ambiguo o secretos locales.

### Fase D2 — Mercado canonical

1. Crear `market.raw_bar`, `canonical_bar`, `quality_event`.
2. Dual-write desde ingestores actuales.
3. Distinguir official/resampled.
4. Corregir anomalías mediante quarantine/correction events.
5. Comparar resultados con las tres tablas OHLCV actuales.

**Terminado:** paridad semántica por instrumento/intervalo y cero pérdida de filas válidas.

### Fase D3 — Macro PIT

1. Migrar series a `macro.observation`.
2. Completar taxonomía de availability.
3. Resolver publication dates o bloquear promotion.
4. Convertir tablas anchas en vistas.
5. Crear `bundle_health`.

**Terminado:** cada estrategia obtiene macro por as-of join reproducible.

### Fase D4 — Feature contracts

1. Extraer del código/SSOT las 21 features exactas de smart_simple.
2. Extraer ventanas SMA de XAU y parámetros exactos de BTC/SPX.
3. Crear `feature_set` por sleeve y forecast spec.
4. Mover normalización a model artifacts.
5. Crear snapshots inmutables y hashes ordenados.

**Terminado:** una corrida puede reconstruir exactamente columnas, orden, lags y resampleos.

### Fase D5 — Separar superficies

1. Migrar H5 predictions a `forecast.output`.
2. Migrar signals a `action.strategy_signal`.
3. Mover columnas específicas a `decision_components`.
4. Aplicar permisos DB ACTION/DIAGNOSTIC.
5. Convertir BI y frontend en vistas/endpoints.

**Terminado:** forecast no tiene camino técnico hacia una orden.

### Fase D6 — Event sourcing y hechos

1. Migrar paper/live a orden, estados y fills comunes.
2. Derivar posiciones y PnL.
3. Retirar duplicidades de trades/signals.
4. Añadir reconciliación e idempotencia.

**Terminado:** paper y live son comparables a nivel de orden/fill y el PnL deriva de hechos.

### Fase D7 — Retiro de legado

1. Ejecutar dual-read y diff durante al menos dos ciclos completos por activo.
2. Marcar legacy read-only.
3. Publicar vistas de compatibilidad.
4. Eliminar tablas solo después de demostrar cero consumidores.
5. Conservar DDL y mapping de migración en Git.

**Terminado:** aproximadamente 25–30 tablas autoritativas, 10–15 vistas y ningún registry duplicado.

### Fase D8 — Optimización

1. Ajustar chunks/compresión/índices con métricas reales.
2. Crear continuous aggregates necesarios.
3. Mover histórico frío a Parquet/MinIO.
4. Separar físicamente execution cuando capital/usuarios lo justifiquen.

## 55. Criterios finales de aceptación

```text
100% de strategies con feature_set_id
100% de features con code_hash, unit, lag y available_at
100% de resampleos con policy_id
100% de señales con decision_fingerprint y feature_snapshot_id
0 símbolos/timeframes libres en tablas nuevas
0 secretos reales en tablas de dominio
0 forecasts consumidos por allocator
0 tablas autoritativas duplicando MLflow
0 PnL calculado desde tablas agregadas manuales
0 promociones con inputs macro sin disponibilidad aprobada
paridad semántica legacy↔nuevo en todos los sleeves activos
```

## 56. Decisiones que deben confirmarse desde el repositorio

El perfil de base no contiene suficiente información para afirmar sin revisión de código:

1. los nombres exactos y orden de las 21 features técnicas de `smart_simple_v11`;
2. las ventanas exactas de SMA de `xauusd_trend_simple_v1`;
3. el instrumento exacto spot/perpetual usado por `btcusdt_hodl_b1`;
4. los proxies de precio restantes del dataset congelado SPX;
5. la cadencia efectiva de decisión de XAU dentro de la corrida semanal;
6. las features exactas de los zoos BTC y USD/COP por horizonte.

Estas no son lagunas para “resolver” con suposiciones. Son tareas de extracción de SSOT y generación automática de manifests. El criterio correcto es:

```text
código/SSOT actual
→ manifest generado
→ revisión humana
→ hash congelado
→ registry
```

## 57. Regla final de datos

> Una estrategia no usa “una tabla”. Usa una versión explícita de un instrumento, una política temporal, un resampleo, un conjunto ordenado de features y un snapshot point-in-time. Si cualquiera de esos elementos cambia, cambia la identidad de la corrida y posiblemente la versión de la estrategia.

