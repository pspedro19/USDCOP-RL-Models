---
kind: roadmap
status: PLANNED
version: 4.0.0
supersedes:
  - .claude/specs/planes/00-superficies-action-vs-diagnostic.md
  - .claude/specs/planes/01-contratos-permisos-ci.md
  - .claude/specs/planes/02-usdcop-doble-rol-y-trials.md
last_verified: 2026-07-27
code_anchors:
  - config/assets/pipelines.yaml
  - src/contracts/signal_contract.py
  - scripts/pipeline/normalize_champions.py
  - services/common/metrics.py
---

# CTR-QLAB-FABRIC-004

## Constitución consolidada del Quant Control Plane: datos, linaje, investigación, replay, forecasting, asignación y ejecución

| Campo | Valor |
|---|---|
| Documento | `CTR-QLAB-FABRIC-004` |
| Estado | Constitución normativa consolidada — versión final de diseño |
| Versión | `4.0.0` |
| Fecha | 2026-07-27 |
| Alcance | USD/COP, XAU/USD, BTC/USDT, SPX500 y toda extensión futura de activos, sleeves y forecasts |
| Fuente de verdad | Git para declaraciones; PostgreSQL para hechos, eventos y estado; MinIO para artefactos; MLflow para modelos |
| Sustituye | `CTR-QLAB-FABRIC-003`, y por transición `CTR-PIPELINE-002`, `CTR-CONTROL-PLANE-001/002` |
| Integra | Auditoría externa (dictamen 8/10, correcciones P0 cerradas en diseño) y documento de separación de superficies ("La Muralla") |

### Changelog 003 → 004

La 004 conserva íntegro el cuerpo normativo de la 003 y **añade** lo que la consolidación anterior dejó fuera o comprimido:

1. **§12 Gate de novedad** — el motor del breadth (`IR ≈ IC·√breadth`): un sleeve se juzga por lo que agrega al libro, no solo, no principalmente, por su Sharpe individual.
2. **§13.5 CLI `qlab` + cutoff impuesto por la capa de lectura** — el screening literalmente no puede leer datos posteriores al cutoff; la disciplina deja de ser humana.
3. **§13.4 Airflow Assets** — scheduling por datasets con shim de compatibilidad Airflow 2/3, esquema de URIs y pools por API externa.
4. **§14.2 Contrato `portfolio_target`** — el artefacto inmutable que consume el servicio de ejecución.
5. **§17–§18 DDL completos** — `fact_position`, `fact_pnl`, `lineage_node/edge` en SQL ejecutable, no solo granos.
6. **§20.5 Registry del libro** — `config/book/allocator_v1.yaml` con su propia familia, trials y juez.
7. **§23 Monitoreo en tres relojes** — datos / modelo / PnL con frecuencias, umbrales y acciones automáticas.
8. **§29 Migración strangler de USD/COP** — el plan capa-por-capa con paridad exigida antes de cada paso; L7 al final.
9. **§33 Runbooks operativos** — nuevo activo, nueva hipótesis, nueva estrategia, candidato A/B, retiro, incidente de datos.
10. **Anexo A — Protocolo de investigación de referencia** — el estándar metodológico que todo screening debe cumplir (particiones, CPCV purgado, meta-labeling, N efectivo, HPO con objetivo robusto, DSR/PBO, SHAP falsificador).
11. **Anexo B — Particularidades por activo** — la tabla de diferencias estructurales que impide copiar mecánicas entre mercados.

---

## 0. Dictamen ejecutivo

La plataforma opera como una **Quant Strategy Fabric**: una sola verdad lógica y auditable, formada por múltiples componentes aislados. No se construye un DAG monolítico ni una estrategia universal para todos los activos.

La unidad de gobierno es el **sleeve de estrategia**; la unidad física de datos es el **activo**; la unidad estadística de investigación es la **familia de hipótesis**; la unidad de capital es el **libro** materializado en un **portfolio target**.

La regla final:

> Los datos producen snapshots. Las familias consumen trials. Las estrategias de acción producen señales. Los modelos diagnósticos producen forecasts. El allocator asigna riesgo. El executor toca el mercado. Las tablas de hechos son la única verdad del PnL. El Strategy Passport recuerda todo porque se deriva de los hechos y nunca se escribe manualmente.

La arquitectura queda **aprobada para construcción por etapas**. No queda aprobada para ampliar ejecución live hasta completar: identidad determinista, event sourcing, sincronización temporal, pre-trade risk, idempotencia, reconciliación, kill switch y retiro operativo (§30).

---

## 1. Objetivos

1. Escalar de cuatro activos a decenas de activos y múltiples sleeves sin duplicar pipelines.
2. Reconstruir cualquier señal, orden, fill, posición, PnL, métrica o decisión de gobierno desde sus datos originales.
3. Separar estrictamente la ciencia de **decisiones económicas** de la ciencia de **predicción diagnóstica**.
4. Evitar look-ahead, p-hacking, selección múltiple oculta, performance chasing y drift de configuración.
5. Aislar fallas: una estrategia, forecast o activo defectuoso no bloquea a los demás.
6. Mantener Airflow como orquestador batch, no como broker ni bus transaccional.
7. Permitir replay, paper, canary y live bajo contratos comparables.
8. Mantener una Control Tower única sin convertirla en fuente de verdad ni permitir cálculos en frontend.
9. Usar infraestructura proporcional a la escala actual (una persona, cuatro activos) y adoptar componentes complejos solo ante necesidad demostrada.
10. Materializar la tesis de breadth: el retorno objetivo se construye con sleeves descorrelacionados bajo apalancamiento moderado, no exprimiendo un solo activo.

### 1.1 No objetivos

- No construir una estrategia única aplicable a todos los activos (la mecánica ganadora **no viaja** entre mercados: mandan vol, sesión y persistencia).
- No utilizar forecasting diagnóstico como recomendación de compra o venta.
- No permitir que la UI recalcule métricas, gates, PnL o aprobación.
- No permitir que un retry de Airflow genere una segunda orden.
- No promover un allocator complejo porque se vea mejor en un backtest aislado.
- No introducir Kubernetes, Kafka, Feast, Iceberg o microestructura avanzada antes de que su ausencia sea un cuello de botella real.

---

## 2. Las leyes constitucionales

1. **Una sola verdad lógica, no una sola aplicación física.**
2. **Toda fila persistida debe tener identidad, tiempo, procedencia y entorno.** Sin spine completo, no se persiste: falla el contrato, no se escribe.
3. **Los trials se cobran una sola vez, en screening; Airflow nunca gasta trials.** Airflow reintenta y exige idempotencia; un trial no es idempotente.
4. **Lo congelado no cambia. Lo dinámico es el capital**, bajo reglas pre-registradas, versionadas y acotadas.
5. **El backtest permite entrar a paper; el paper permite entrar a canary; solo el forward permite capital completo.**
6. **Toda métrica tiene una definición, un motor y una persistencia únicos.**
7. **El allocator nunca consume "el último valor"; consume un portfolio snapshot con cutoff explícito.**
8. **Replay, paper, canary y live comparten contratos de órdenes y fills; difieren solo en el executor.**
9. **El forecasting diagnóstico no puede producir señales, asignaciones ni órdenes.** La muralla es contractual, de permisos y de CI — no de disciplina.
10. **Una revisión legítima de un dato no invalida una decisión histórica point-in-time correcta.**
11. **Un retiro no elimina un DAG hasta cerrar posiciones, órdenes y reconciliación.**
12. **El frontend presenta hechos; no decide ni corrige datos.**
13. **Toda excepción, override y voto es auditable.**
14. **Ningún atributo tiene dos sistemas autorizados para modificarlo.**

Y la regla operativa transversal: **los DAGs miden y publican; el CLI gasta trials; el registry recuerda; el único juez limpio es el forward posterior al freeze — nunca el período que motivó el cambio.**

---

## 3. Dos superficies científicas, un Control Plane

Las dos superficies comparten datos, calendarios, snapshots, linaje, observabilidad, identidad y gobierno técnico. Se separan en objetivos, targets, contratos, métricas, permisos, DAGs, esquemas de base de datos y vistas.

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

**Outputs:** `strategy_output`, señal y target de exposición, órdenes y fills (simulados o reales), posiciones, `fact_pnl`, bundles `summary`/`trades`/`signals`, gates y resultados del juez.

**Métricas reinas:** retorno neto, MaxDD, Calmar, Sharpe, Sortino, DSR, PBO, turnover, exposición, slippage, implementation shortfall, fill rate, tracking error y paridad replay-paper-live.

La exactitud predictiva puede observarse como diagnóstico de un componente, pero **la unidad de evaluación es la decisión económica completa** — un predictor con R² < 0 puede sostener una estrategia rentable si el gate, el sizing y las salidas agregan el valor.

### 3.2 Superficie DIAGNOSTIC: forecasting y transparencia

**Pregunta:** ¿qué precio, retorno o dirección estima un modelo para un horizonte definido?

```text
Datos point-in-time
  → features de forecasting (subconjunto declarado del mismo feature store)
  → target supervisado
  → walk-forward / OOF
  → modelo × horizonte
  → predicción e intervalo
  → comparación contra baseline
  → CSV, tablas, PNG y panel
  → cero capital y cero órdenes
```

**Outputs:** `forecast_output`, predicción puntual, intervalos o cuantiles, probabilidad o score declarado, métricas por modelo y horizonte, paneles.

**Métricas reinas:** directional accuracy, balanced DA, lift contra baseline, MAE, RMSE, pinball loss, Brier score, calibración, cobertura de intervalos y drift. El veredicto contra baseline se publica **siempre**, aun cuando sea negativo: la honestidad publicada es parte del producto.

### 3.3 Matriz de separación

| Propiedad | ACTION | DIAGNOSTIC |
|---|---|---|
| Produce decisión económica | Sí | No |
| Produce PnL | Sí | No |
| Puede ser `CHAMPION` | Sí | No |
| Puede llegar al allocator | Sí | No |
| Puede ejecutar | Solo con aprobación y capacidad | Nunca |
| Usa Vote 1 / Vote 2 | Sí | No |
| Gasta trials | Sí (AT-) | Sí, cuando busca (FT-, §10.1) |
| Métrica reina | Desempeño económico y riesgo | Lift y error predictivo |
| Vistas | Replay, dashboard, production, execution | Forecasting, analysis |
| Fallo | Puede bloquear dinero | Degrada un panel |

### 3.4 Dependencias permitidas y prohibidas

Permitido:

```text
asset://{asset}/canonical  → strategy://{sleeve}/signal
asset://{asset}/canonical  → forecast://{asset}/{model}/{horizon}
strategy://{sleeve}/signal → portfolio_snapshot → allocator → portfolio_target → execution
```

Prohibido (falla el parseo de DAGs y el CI):

```text
forecast://*/prediction → allocator
forecast://*/prediction → execution
frontend forecasting    → endpoint de órdenes
```

Estas prohibiciones se implementan mediante contratos tipados, permisos de base de datos y validaciones de CI; **no dependen de disciplina humana** (§15.3).

### 3.5 Features: un pipeline, dos vistas

Las features se computan **una sola vez por activo** en el pipeline de datos (feature store versionado con catálogo). Cada superficie declara su **subconjunto por contrato** — el patrón del "contrato de 20" del RL, generalizado. Los URIs `action_features` y `forecast_features` son **vistas lógicas sobre la misma tabla física versionada**, jamás dos cómputos: dos implementaciones de la misma feature son una deriva de paridad garantizada. El drift experimental del zoo (agregar features al catálogo) nunca toca el snapshot pineado que consume una estrategia congelada.

---

## 4. Arquitectura lógica

```text
                             QUANT CONTROL PLANE
┌─────────────────────────────────────────────────────────────────────────────┐
│ Gobierno:  assets · families · clusters · trials · strategies · forecasts   │
│            judges · votes · decisions · withdrawal_protocols                │
│ Operación: runs · signals · forecast_outputs · portfolio_snapshots          │
│            allocations · portfolio_targets · orders · status · fills        │
│ Hechos:    fact_position · fact_pnl · metric_event · incident               │
│            lineage_node · lineage_edge                                      │
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

Los archivos grandes, snapshots y bundles viven en MinIO. PostgreSQL almacena identidad, estado, referencias, eventos y hechos consultables. "Consolidado" significa **una sola verdad lógica, no una sola aplicación física.**

---

## 5. Estado inicial de migración por activo

Punto de partida funcional; confirmar contra los registries al iniciar la migración.

| Activo | Sleeve ACTION principal | Candidatas / retiradas | DIAGNOSTIC | Ejecución | Notas de estado |
|---|---|---|---|---|---|
| USD/COP | `usdcop_smart_simple_v11` (CHAMPION, FROZEN, juez=forward) | `v12` cap-1.5× y `v14` ladder en paper con juez sellado; `v13` (techo QR) aprobó diseño, espera freeze | Zoo 9 modelos × 7 horizontes, DA~72% sin lift vs baseline mayoritario | **Único activo con ruta live** (L7 cada 30 min en sesión) | Reentrena Dom 01:30 ventana expansiva 2020→; 2025 +7.35%, 2026 +3.36% vs B&H −14.48% |
| XAU/USD | `xauusd_trend_simple_v1` (votos SMA, nunca reentrena) | `dynamic_exit` retirada pero preservada con bundles (OOS-2025 +38% pero DSR 0.0077 = beta con ingeniería de riesgo) | Zoo 9 modelos, 19 features, DA≈0.46–0.50 = coin-flip publicado | Paper/research | Reloj √252 weekday |
| BTC/USDT | `btcusdt_hodl_b1` (beta vol-targeted, 24/7) | 10+ retadores derrotados; **moratoria activa**, se reabre solo con datos nuevos (H-BASIS-01, funding acumulándose) | Zoo con DA≈0.46 publicado como techo del precio-solo | Paper/research | Reloj √365; hodl +4.70% held-out vs −1.37% de todas las variantes |
| SPX500 | `daily_ma200` (Calmar 0.212) vs `regime_gated` (Calmar 0.144) — decisión de campeona **pendiente del juez formal** | Ambas congeladas juntas (v2.0.0) y preservadas como sleeves independientes | Zoo deshabilitado por plan (dirección cerrada) | Paper/research | Siguiente puerta: S4 — firmar withdrawal → paper ≥26 semanas |

Principio ratificado por la evidencia del propio sistema: **la mecánica ganadora no se copia entre activos** (COP = semanal TP/HS + gate Hurst; Oro = trend simple; BTC = beta pasivo; SPX = probablemente MA200). Cada activo conserva reloj, calendario, costos, sesión, liquidez, frecuencia y criterio de validación propios.

---

## 6. Capas actuales y mapeo objetivo

La migración no redefine contratos existentes por accidente.

| Capa actual | Responsabilidad actual | Destino en Fabric |
|---|---|---|
| L0 | Ingesta, validación, OHLCV, macro (41+ vars), seeds y backfill | `asset__{asset}__data` y snapshots raw/canonical |
| L0b | Exportación de datos para charts | Proyección frontend; nunca fuente de verdad |
| L1 | Features RL de USD/COP (contrato de 20, única capa autorizada) | Se conserva; publica feature snapshot ACTION |
| L2 | Dataset RL | Dataset versionado con linaje y cutoff |
| L3 | Entrenamiento | Tarea opcional de `strat__*` cuando `retrain != never` |
| L4 | Backtest, gates (Vote 1) y publicación inmutable | `strat__{sleeve}` ACTION |
| L4b | Deploy tras Vote 2 | Transición de gobierno; jamás envío directo de orden |
| L5 | Señal / inferencia con modelo congelado | Publicación de `strategy_output` |
| L6 | Forward, paper ledger y verify | Hechos, métricas, jueces y Passport |
| L7 | Ejecución | **Servicio aislado fuera de Airflow** (§21) |
| L8 | News/LLM de contexto | Superficie ANALYSIS; clasifica y resume, **jamás decide** |

Para activos factory sin L1 numerada, las features viven como stages versionados del DAG de datos. No se fuerza numeración artificial: la capa se define por **función**, no por activo.

---

## 7. Matriz de fuente de verdad

| Entidad | Fuente autoritativa | Proyección / réplica |
|---|---|---|
| Asset registry | Git | PostgreSQL |
| Family registry y celdas | Git | PostgreSQL |
| Strategy y forecast specs | Git | PostgreSQL |
| Trial ledger | PostgreSQL append-only + commit de origen | Dashboard / reportes |
| Votos y transiciones | PostgreSQL append-only + referencia al commit | Passport |
| Runs y señales | PostgreSQL | OpenLineage / Grafana |
| Modelos y pesos entrenados | MLflow + MinIO | Referencias en PostgreSQL |
| Snapshots y bundles | MinIO | Metadatos en PostgreSQL |
| Órdenes y fills | PostgreSQL event-sourced | Broker reconciliation / UI |
| Positions y PnL | Tablas de hechos en PostgreSQL | Passport / dashboard |
| Métricas | `metric_event` en PostgreSQL | Grafana / gates |
| Grafo de linaje | PostgreSQL | Marquez como visor opcional |
| Archivos de frontend | Proyección regenerable | Nunca autoritativos |

Regla dura: **ningún atributo tiene dos escritores autorizados.**

---

## 8. Identidad determinista

### 8.1 Seis conceptos

```text
spec_fingerprint = H(
  data_snapshot_id ⊕ feature_snapshot_id ⊕ code_hash ⊕ config_hash
  ⊕ model_or_policy_hash ⊕ calendar_hash ⊕ cost_model_hash
  ⊕ dependency_lock_hash ⊕ container_image_digest
)

decision_fingerprint  = H(spec_fingerprint ⊕ as_of ⊕ decision_inputs)
execution_fingerprint = H(decision_fingerprint ⊕ env ⊕ account_id ⊕ broker_id ⊕ order_policy_hash)

derivation_id = H(inputs ⊕ code ⊕ params)     -- identifica la derivación, no el contenido
semantic_hash = H(contenido canónico)          -- identidad lógica del resultado
bytes_hash    = H(bytes físicos)               -- integridad de storage
```

### 8.2 Regla de paridad (la única)

> El mismo `decision_fingerprint` debe producir el mismo `semantic_hash` de la señal en replay, paper, canary y live.

- Para JSON, ledger, señal, target y bundle cuyo **escritor controlamos**, se usa serialización canónica y `semantic_hash == bytes_hash` **por construcción** — la paridad byte a byte sigue siendo el estándar de oro del CI para esos artefactos.
- Para Parquet, respuestas de broker y formatos externos, se compara hash semántico normalizado; `bytes_hash` se conserva solo para integridad física.
- Dos corridas con el mismo `derivation_id` y distinto `semantic_hash` = no-determinismo = incidente automático.

El gate de CI contra el paper ledger anclado (ene-2026) se expresa en estos términos: **un replay independiente debe reproducir el `semantic_hash` de cada señal del ledger.** Un solo test valida idempotencia, ausencia de look-ahead, paridad train/serve y reproducibilidad.

### 8.3 Serialización canónica

- UTF-8 normalizado NFC.
- Claves JSON ordenadas.
- Timestamps ISO-8601 UTC terminados en `Z`.
- Decimales cuantizados por campo según esquema (nunca float binario directo).
- Sin `NaN`, `Infinity` ni whitespace no significativo.
- Orden de filas explícito antes de serializar.

### 8.4 Spine mínimo

Toda entidad operacional o de hechos contiene, directamente o mediante FK inequívoca:

```text
as_of · available_at · created_at · run_id
spec_fingerprint · decision_fingerprint? · execution_fingerprint?
derivation_id · artifact_or_event_id
sleeve_id? · forecast_spec_id? · family_id? · trial_id?
code_hash · data_snapshot_id · env · schema_version
```

No toda fila necesita todas las columnas físicamente; el grafo debe permitir resolverlas sin ambigüedad. **Una fila sin spine resoluble no se persiste.**

---

## 9. Registries

### 9.1 Asset registry — `config/assets/{asset}.yaml`

Verdad física del mercado. **No cambia cuando cambian las estrategias.**

```yaml
id: spx500
enabled: true

market:
  calendar: NYSE                    # pandas_market_calendars; nunca pd.date_range
  session: "09:30-16:00"
  timezone: America/New_York
  annualization: 252                # EL RELOJ — prohibido cruzarlo entre activos
  continuous: false

clock:
  master_bar: 1D
  decision_point: close
  execution_ref: next_open          # MOO t+1, consistente con open_to_open_return
  warmup_bars: 252

canary_requirements:                # §11.5 — mínimos por activo, jamás universales
  min_decisions: 60                 # semanal usaría 26; sale de la frecuencia de decisión
  min_fills: 30
  min_regimes: 2

costs:
  model: fixed_bps
  value: 3.0
  stress_scenarios: [5.0, 10.0]     # obligatorio para los gates

data:
  price:
    provider: investing
    symbol: "^GSPC"                   # fuente real as-built: investing id 166 (SPX/500)
    return_type: price_return       # ≠ total_return — declarado, no asumido
    history_floor: "1995-01-03"
  macro_bundle: us_core             # → config/macro/us_core.yaml, nunca lista inline
  vintage_required: [nfci, payrolls, indpro, retail_sales, cfnai]

schedule:
  data: "30 7 * * 1-5"
```

`annualization`, calendario, timezone, costos, frecuencia y requisitos canary se leen **exclusivamente** desde este registry — ninguna comparación cross-asset puede obtenerlos de otro lado.

### 9.2 Strategy registry ACTION — `config/strategies/{sleeve_id}.yaml`

Un archivo por estrategia. Es la unidad que se congela, publica, juzga, promueve y retira.

```yaml
id: spx500_daily_ma200_v1
asset: spx500
family: trend_regime
research_cluster: trend
surface: action
version: 2.0.0

engine:
  type: rule_based                  # rule_based | ml_batch | ml_online | rl
  module: strategies.policies:MA200Policy
  retrain: never                    # never | weekly | daily
  params:
    ma_window: 200
    exposure_cap: 1.0
  params_hash: "sha256:..."

research_state: PAPER
capital_tier: SHADOW
operational_state: NOMINAL

governance:
  data_cutoff: "2024-12-31"         # impuesto por la capa de lectura (§13.5)
  frozen_at: "2026-07-27T00:00:00Z"
  code_hash: "sha256:..."
  ssot_manifest: manifests/spx500_ma200_v1.json
  withdrawal_protocol: docs/withdrawal/spx500_ma200_v1.md   # debe EXISTIR o falla CI

provenance:
  hypothesis_ids: [H-TREND-SPX-01]
  action_trial_id: AT-0113
  forecast_trial_ids: []            # herencia FT si un predictor cruzó la muralla (§10.2)
  screening_report: reports/screening/trend_regime_2026-06.md

judge:                              # FIRMADO ANTES de entrar a paper
  anchor: "2026-08-01"
  criterion: "Calmar >= incumbent"
  min_periods: 26
  alpha: 0.05
  paired_with: null                 # != null ⇒ test pareado sobre la serie de diferencias
  sequential:
    method: none                    # mSPRT | e_process | none

publish:
  bundles: [summary, trades, signals]
  immutable_key: [strategy_id, version, partition]

capabilities: [backtest, gate, signal, paper, verify]   # sin execute ⇒ no se emite L7
```

Un candidato A/B es el mismo archivo con `version` nueva, la variante de riesgo, `judge.paired_with: {incumbente}` y α repartido entre concurrentes (Bonferroni/Holm: v12 y v14 ⇒ 0.025 c/u). **El A/B deja de ser un procedimiento y pasa a ser dos campos.**

### 9.3 Forecast registry DIAGNOSTIC — `config/forecasts/{spec_id}.yaml`

```yaml
id: usdcop_forecast_zoo_v3
asset: usdcop
surface: diagnostic
state: ACTIVE                       # ACTIVE | OFF

models: [ridge, bayesian_ridge, random_forest]
horizons: [1d, 5d, 10d, 20d, 30d]
target: forward_return
validation: purged_walk_forward
baselines: [majority_class, persistence, zero_return]

forecast_family: usdcop_direction   # §9.5 — su propia familia FT
capabilities: [fit, predict, evaluate, publish_panel]
forbidden_capabilities: [signal, allocate, execute, approve]
```

### 9.4 Family registry ACTION — `registries/families/{family_id}.yaml`

Transversal a activos. **El único lugar donde se gastan trials de acción.**

```yaml
family_id: trend_regime
cluster_id: trend                   # vocabulario controlado: trend|carry|meanrev|vol|flow|ml_meta
declared_at: "2026-06-01"
declared_by: pedro
question: "¿Un gate de tendencia mejora el Calmar contra el baseline?"
bar: "batir persistencia y baseline tonto, neto de costos, screening <= 2024, purged K-fold"
screening_cutoff: "2024-12-31"

cells:                              # LA FAMILIA COMPLETA, escrita ANTES de mirar nada
  - {asset: spx500,  variant: ma200_pure,    trial_id: AT-0113, status: SCREENED, result: pass}
  - {asset: spx500,  variant: ma200_x_tsmom, trial_id: AT-0114, status: FROZEN,   result: pass}
  - {asset: spx500,  variant: ma200_x_vix,   trial_id: null,    status: DECLARED, result: null}
  - {asset: xauusd,  variant: sma_votes,     trial_id: AT-0087, status: FROZEN,   result: pass}
  - {asset: btcusdt, variant: ma200_pure,    trial_id: AT-0041, status: REJECTED, result: fail}

trials_charged: 4                   # == count(cells con trial_id) — validado en CI
closed: false
closure_note: null
```

La misma mecánica probada en 3 activos = **una familia, N celdas, N trials** — nunca tres registries "independientes" que subestiman la deflación. Cerrar una familia por escrito ("nada batió el bar") es un resultado, no un fracaso.

### 9.5 Forecast family registry — `registries/families/{forecast_family_id}.yaml`

```yaml
family_id: usdcop_direction
kind: forecast
cluster_id: usdcop_directional_models
question: "¿Algún modelo supervisado predice la dirección de COP con lift sobre el baseline mayoritario?"
bar: "lift de DA balanceada > 0 sobre majority_class, OOF purgado, screening <= 2024"
cells:
  - {model: ridge,          horizon: 5d, trial_id: FT-0041, status: EVALUATED, result: no_lift}
  - {model: bayesian_ridge, horizon: 5d, trial_id: FT-0042, status: EVALUATED, result: no_lift}
  # ... una celda por (modelo × horizonte) MIRADO
trials_charged: 63
closed: true
closure_note: "DA titular ~72% sin lift vs baseline mayoritario; dirección cerrada; el zoo queda como panel del modelo congelado"
```

### 9.6 Ledger global — `registries/ledger.jsonl`

Append-only, hasheado, en git. Una línea por trial (FT- o AT-):

```jsonl
{"trial_id":"AT-0113","kind":"action","family":"trend_regime","cluster":"trend","asset":"spx500","variant":"ma200_pure","charged_at":"2026-06-14T11:02:00Z","cutoff":"2024-12-31","env":"screening","code_hash":"c41d...","data_hash":"9ab2...","result":"pass","N_family":4,"N_cluster":9,"N_global":216}
```

### 9.7 Trial budget

`N_MAX = 989` se conserva únicamente como **cota constitucional de gasto** — un commitment device contra el p-hacking, como un presupuesto. Su valor es arbitrario y su función es forzar priorización. **No entra en el DSR ni en ninguna fórmula estadística**; el DSR usa el N efectivamente cobrado en el ledger.

Se reportan siempre: `N_family`, `N_cluster`, `N_global` y `DSR_family`, `DSR_cluster`, `DSR_global`. El gobierno gatea con `DSR_family`; los otros dos son **divulgación obligatoria** en el Passport y en todo claim — la defensa contra dividir familias para ocultar multiplicidad es que la dilución quede visible, no fingir que no existe.

### 9.8 Nomenclatura canónica

```text
activo            {asset}                                spx500
sleeve            {asset}_{mecanica}_{version}           spx500_daily_ma200_v1
familia ACTION    {mecanica}                             trend_regime
familia FORECAST  {asset}_{pregunta}                     usdcop_direction
cluster           vocabulario controlado                 trend | carry | meanrev | vol | flow | ml_meta
hipótesis         H-{FAMILIA}-{ASSET}-{seq}              H-TREND-SPX-01
trial acción      AT-{seq global}                        AT-0113
trial forecast    FT-{seq global}                        FT-0041
DAG datos         asset__{asset}__data                   asset__spx500__data
DAG estrategia    strat__{sleeve_id}                     strat__spx500_daily_ma200_v1
DAG forecast      forecast__{spec_id}                    forecast__usdcop_forecast_zoo_v3
DAG libro         book__allocator_v1
dataset           asset://{asset}/{layer}                asset://spx500/features
señal             strategy://{sleeve_id}/signal
predicción        forecast://{asset}/{model}/{horizon}
target            portfolio://target/{snapshot_id}
bundle            bundles/{sleeve_id}/{version}/{partition}/{summary|trades|signals}.json
```

---

## 10. Gobierno estadístico

### 10.1 Qué cobra trial y qué no

**Cobra** (en cualquiera de las dos superficies):

- probar un nuevo modelo, feature, target, horizonte o regla;
- elegir el mejor resultado entre alternativas;
- cambiar el cutoff después de ver resultados;
- **convertir un forecast en señal económica** (cruzar la muralla es un AT nuevo);
- alterar un gate, sizing, salida o costo para mejorar el OOS observado;
- re-correr un estudio de HPO "porque no gustó" (es un trial de nivel superior).

**No cobra:**

- regenerar un modelo congelado;
- publicar nuevos períodos forward;
- calcular métricas previamente declaradas;
- actualizar el dashboard;
- monitorear drift sin modificar la política.

La regla que cierra el último agujero de minería: **lo gratis es regenerar lo congelado; la búsqueda cobra, viva donde viva.** Sin ella, la superficie diagnóstica sería la puerta trasera obvia: probar 50 modelos "gratis", elegir el ganador, cablearlo con N=1.

### 10.2 Dos linajes y la herencia

```text
forecast_family (FT-)                    action_family (AT-)
  pregunta predictiva                      pregunta económica
  modelos × horizontes                     predictor + gate + sizing + salidas
        │                                        │
        └────────── research_cluster ────────────┘
```

Cuando un predictor entra a una estrategia ACTION:

1. La política económica cobra **su propio AT** (es una hipótesis nueva).
2. El predictor **no llega limpio**: sus `forecast_trial_ids` viajan en la `provenance` de la estrategia y entran a su `N_cluster`.
3. `DSR_family` de la estrategia usa sus AT; `DSR_cluster` ve además los FT heredados; `N_global` suma todo. La dilución de haber mirado 63 celdas predictivas antes de cablear queda **visible en el Passport**.

Backfill obligatorio: los zoos existentes (COP 9×7, Oro 9 modelos) reciben sus FT históricos; si el número exacto no se reconstruye, se declara una **estimación honesta por escrito** con etiqueta `legacy_estimate` — un N estimado documentado vale infinitamente más que un N=0 falso.

### 10.3 Juez secuencial

Campo obligatorio en todo juez:

```yaml
sequential:
  method: mSPRT | e_process | none
  alpha: 0.05
```

Con `none`, **observar** resultados es válido; **actuar** antes del horizonte firmado constituye una violación de gobierno registrada en `governance.*`. Con método secuencial, se puede mirar y decidir en cualquier momento sin inflar α — a cambio de algo de potencia. Pre-registrado, jamás elegido después.

### 10.4 El A/B correcto

No se distribuye tráfico aleatorio. Se corren **políticas congeladas** bajo: mismos datos, mismo cutoff, mismos costos, mismo simulador determinista, ledgers independientes, juez prefirmado.

- **Test pareado cuando comparten señal base** (patrón v11/v12/v14): se compara la serie de diferencias de PnL diario (bootstrap por bloques o t pareado sobre `pnl_B − pnl_A`), no dos Sharpes independientes — la correlación ~0.9 entre variantes hace que la varianza de la diferencia sea mínima y el efecto se detecte con muchas menos observaciones.
- **Corrección por concurrencia**: candidatas simultáneas de la misma familia reparten α (Bonferroni u Holm); la suma se valida en CI.
- **MDE antes de arrancar**: con la ventana pre-firmada, calcular qué diferencia mínima es detectable. Si la respuesta es absurda ("0.8 de Calmar"), el test no sirve y hay que saberlo antes.
- Cambiar una candidata durante la evaluación crea **una versión nueva, un freeze nuevo y un ancla nueva**. Jamás se re-ancla.
- Jamás iterar sobre el mismo OOS — eso fue lo que quemó el 2025.

---

## 11. Estados en tres dimensiones

### 11.1 Las tres dimensiones

```text
research_state:    DECLARED → SCREENED → DESIGN_RUN → FROZEN → PAPER → CHAMPION → RETIRING → WITHDRAWN
                   (ramas de salida: CLOSED desde DECLARED; REJECTED desde SCREENED/DESIGN_RUN)
capital_tier:      ZERO | SHADOW | CANARY | FULL | REDUCED | EXIT_ONLY
operational_state: NOMINAL | QUARANTINED
```

`CANARY` y `REDUCED` son decisiones de **capital**; `QUARANTINED` es un hecho de **salud** ortogonal — si fuera un estado del ciclo, al salir de cuarentena no se sabría a dónde volver.

### 11.2 Matriz de legalidad

Sin matriz, tres dimensiones son 8×6×2 = 96 combinaciones nominales, la mayoría absurdas — más superficie de bugs, no menos. El registry validator **rechaza cualquier combinación ilegal**:

```text
research_state ∈ {DECLARED, SCREENED, DESIGN_RUN, FROZEN}  ⇒ capital_tier = ZERO
research_state = PAPER                                     ⇒ capital_tier ∈ {ZERO, SHADOW}
research_state = CHAMPION                                  ⇒ capital_tier ∈ {ZERO, SHADOW, CANARY, FULL, REDUCED}
research_state = RETIRING                                  ⇒ capital_tier = EXIT_ONLY (forzado)
research_state = WITHDRAWN                                 ⇒ capital_tier = ZERO ∧ exit_checklist = PASS
operational_state = QUARANTINED                            ⇒ pre-trade rechaza toda orden de apertura;
                                                             capital_tier se conserva para poder reanudar
DAG de estrategia existe                                   ⇔ research_state ∈ {FROZEN, PAPER, CHAMPION, RETIRING}
```

### 11.3 Transiciones, gates y autoridad

| Transición | Gate | Autoridad | Trials |
|---|---|---|---|
| → DECLARED | familia completa escrita, bar pre-firmado | humano, commit | 0 |
| DECLARED → SCREENED | screening ≤ cutoff **impuesto por la capa de lectura**, purged K-fold, bate el bar | CLI `qlab screen` | **+1 (AT/FT)** |
| SCREENED → DESIGN_RUN | una sola variante económica, criterio sellado | Vote 1 | 0 |
| DESIGN_RUN → FROZEN | `spec_fingerprint` + manifiesto SSOT + `withdrawal_protocol` **existe** | humano firma | 0 |
| FROZEN → PAPER | juez declarado (ancla, N, α, `paired_with`, `sequential`) | humano | 0 |
| PAPER → CHAMPION (tier CANARY) | juez de paper cumplido + `DSR_family` > 0.95 + **gate de novedad (§12)** + Vote 2 | humano | 0 |
| tier CANARY → FULL | mínimos canary del asset registry (§11.5) | automático + Vote 2 | 0 |
| CHAMPION: FULL ⇄ REDUCED | criterio de PnL pre-firmado (reversible, con histéresis) | automático | 0 |
| CHAMPION → RETIRING | `withdrawal_protocol` disparado | **automático** | 0 |
| RETIRING → WITHDRAWN | exit checklist PASS (§21.4) | automático | 0 |
| cualquiera → QUARANTINED | `data_health != HEALTHY`, paridad rota, reconciliación fallida | **automático, sin voto** | 0 |

Dos consecuencias: los trials solo se cobran en screening (todo lo posterior es forward y es gratis — la contabilidad correcta da más aire del que se cree tener); y retirar una estrategia hace desaparecer su DAG **después** del checklist, sin borrar un solo bundle.

### 11.4 Semántica de `capital_tier`

- `ZERO`: sin capital ni simulación de libro.
- `SHADOW`: las señales alimentan la simulación del allocator, cero órdenes.
- `CANARY`: fracción acotada (p.ej. 25% del presupuesto que el allocator asignaría, o cap notional fijo — el menor).
- `FULL`: presupuesto completo del allocator.
- `REDUCED`: recorte temporal por criterio pre-firmado; la estrategia sigue siendo campeona aprobada.
- `EXIT_ONLY`: solo órdenes de cierre.

### 11.5 Canary parametrizado

Canary no se gobierna por semanas calendario. Debe cumplir **simultáneamente**, con mínimos que salen del asset registry:

- `min_decisions` (p.ej. 60 para decisión diaria, 26 para semanal);
- `min_fills` según frecuencia y liquidez;
- exposición a ≥ `min_regimes` según el clasificador declarado;
- tracking error live-vs-paper dentro del umbral;
- slippage realizado ≤ 1.5× el costo modelado;
- cero incidentes críticos;
- paridad semántica verde;
- reconciliación sin diferencias materiales.

---

## 12. Gate de novedad — el motor del breadth

La tesis de retorno del sistema es breadth (`IR ≈ IC·√breadth`): sleeves descorrelacionados bajo apalancamiento moderado. Por eso, en la transición PAPER → CHAMPION, **un sleeve no se juzga solo: se juzga por lo que agrega al libro.**

```python
def novelty_gate(candidate_id, live_sleeves, env="paper"):
    R = pnl_matrix(live_sleeves + [candidate_id], env)     # de fact_pnl, misma ventana, mismo reloj
    rho = R.corr()[candidate_id].drop(candidate_id)

    ir_before = book_ir(R.drop(columns=[candidate_id]))
    ir_after  = book_ir(R)                                 # con el allocator baseline re-resuelto

    return {
        "rho_max":               rho.abs().max(),
        "rho_argmax":            rho.abs().idxmax(),
        "marginal_ir":           ir_after - ir_before,
        "diversification_ratio": div_ratio(R),
        "verdict": "ACCEPT" if (rho.abs().max() < 0.60 or ir_after - ir_before > 0.15)
                            else "REJECT_REDUNDANT",
    }
```

Reglas:

- Un sleeve con Sharpe 1.2 y ρ = 0.9 contra la campeona vale **menos** que uno con Sharpe 0.6 y ρ = 0.1.
- La correlación puntual no basta: se reporta con incertidumbre (bootstrap) y se examina la **correlación en crisis** (colas), no solo la media.
- Sleeves del mismo activo (v11/v12/v14) **no son diversificación**: comparten el cap por activo del allocator.
- El gate corre también en continuo como `m_div` del allocator (§20.2), con banda [0.70, 1.10].
- Umbrales (0.60 / 0.15) pre-registrados y versionados; cambiarlos es una decisión de gobierno auditada.

Este gate es lo que convierte "coleccionar estrategias" en "construir un libro" — y la conclusión empírica del propio sistema (*la mecánica ganadora no viaja entre activos*) es una buena noticia bajo esta luz: mecánicas distintas por activo ⇒ correlación baja ⇒ breadth real.

---

## 13. Orquestación: DAGs, servicios y el CLI

### 13.1 Generador A — datos, uno por activo

```text
asset__{asset}__data
  l0_ingest → canonicalize → quality → feature_snapshots → chart_projection → data_verify
  outlets: [ Asset("asset://{asset}/canonical"), Asset("asset://{asset}/features") ]
  schedule: {asset.schedule.data}
```

El DAG de datos **no conoce** a las estrategias ni a los forecasts que lo consumen. Corre aunque no exista ninguna estrategia declarada.

### 13.2 Generador B — estrategia ACTION, uno por sleeve activo

```text
strat__{sleeve_id}
  inlets:  [ Asset("asset://{asset}/features") ]     # Asset/Dataset, NUNCA ExternalTaskSensor
  schedule: None                                     # disparado por el dataset
  [l3_train si engine.retrain != never]
  → l4_backtest_replay  (bundle inmutable por (id, version, partition))
  → l4v_gates           (Vote 1 + DSR con N de familia)
  → l5_signal           (strategy_output validado)
  → paper_simulation    (simulador determinista → exec.* env=paper)
  → l6_verify           (ledger + tracking del juez)
  outlets: [ Asset("strategy://{sleeve_id}/signal") ]
```

Cada estrategia activa tiene DAG propio: una candidata rota **jamás bloquea** a la campeona ni a otros sleeves, y cada una tiene sus propios retries, SLA, timeouts y logs (un backtest RL de 40 minutos no comparte timeout con un MA200 de 8 segundos).

### 13.3 Generador D — forecasting DIAGNOSTIC

```text
forecast__{forecast_spec_id}
  feature_view → walk_forward_fit → predict → evaluate → publish_panel
```

Este generador **no importa** librerías de ejecución ni publica assets consumibles por el allocator. Se emite solo para specs `ACTIVE`.

```python
# strategy/forecast factory — la muralla en el generador
if spec.surface == "diagnostic" and ({"signal","allocate","execute","approve"} & set(spec.capabilities)):
    raise ContractViolation(f"{spec.id}: superficie diagnóstica no puede señalar, asignar ni ejecutar")
```

### 13.4 Libro, controles y Airflow Assets

```text
control__portfolio_snapshot      # la barrera temporal (§14)
book__allocator_v1               # consume el snapshot, publica portfolio_target
control__pretrade                # validación por orden (§21.1)
control__intraday_risk           # límites en sesión
control__eod_reconciliation      # broker vs orders vs fact_position
control__weekly_judges           # evalúa jueces; dispara REDUCED / RETIRING
control__system_health           # freshness, STALE, paridad, cuarentenas
exec__paper_multiasset           # simulador determinista multi-activo
```

Encadenado por **Airflow Assets** (Airflow 3 renombró Dataset → Asset; los productores actualizan assets y esos eventos disparan DAGs consumidores). Shim de compatibilidad:

```python
try:
    from airflow.sdk import Asset            # Airflow 3
except ImportError:
    from airflow import Dataset as Asset     # Airflow 2.x
```

Higiene a escala:

- **Pools por API externa, no por activo** (`pool: fred_api, slots: 2`): FRED tiene rate limit y 8 DAGs simultáneos lo revientan.
- **Backfill en DAG aparte**, con `as_of` explícito — jamás el mismo DAG que la corrida diaria.
- **Contratos `pandera`/`pydantic` en cada frontera de capa**: el DAG falla ruidoso, nunca silencioso.
- Nomenclatura rígida `{tipo}__{id}__{propósito}` (§9.8).

### 13.5 Lo que NO es un DAG: el CLI `qlab` y el cutoff duro

La investigación vive **fuera de Airflow** por diseño, no por costumbre: Airflow reintenta y exige idempotencia; **un trial no es idempotente** — un retry a las 3 a.m. cobraría trials en silencio y rompería la contabilidad del DSR.

```bash
qlab family declare  --file registries/families/trend_regime.yaml
qlab screen          --family trend_regime --cell spx500:ma200_x_vix \
                     --cutoff 2024-12-31 --charge-trial
qlab freeze          --strategy spx500_daily_ma200_v1     # exige withdrawal_protocol existente
qlab promote         --strategy ... --vote2-token ...
qlab family close    --family ... --note "..."
```

Y el control de mayor apalancamiento de todo el sistema — el cutoff **impuesto por la capa de lectura**, no por disciplina:

```python
# toda lectura de datos pasa por acá, sin excepción
def read(table, asset, *, max_available_at, run_ctx):
    assert run_ctx.env in {"research", "screening", "paper", "canary", "live"}
    if run_ctx.env == "screening":
        assert max_available_at <= run_ctx.declared_cutoff   # falla el JOB, no el humano
    return q.where(f"available_at <= '{max_available_at}'")
```

Un job de screening **literalmente no puede** leer datos posteriores al cutoff. El look-ahead nunca entra por la puerta grande — entra por un notebook a las 11 de la noche; este assert es la cerradura de esa puerta.

| Entorno | Datos visibles | ¿Cobra trials? | Publica |
|---|---|---|---|
| `research` | ≤ cutoff, muestra de diseño | no | nada |
| `screening` | ≤ cutoff duro | **sí, +1 por celda** | registry + ledger |
| `paper` | todo, forward post-freeze | no (ya congelado) | ledger de paper |
| `canary` / `live` | todo | no | blotter reconciliado |

---

## 14. Barrera temporal del libro

### 14.1 `portfolio_snapshot`

El allocator no lee "la última señal" — un libro construido con la señal de hoy de SPX, la de ayer de Oro y la de hace seis horas de BTC no es un libro, es una foto movida. Consume un conjunto explícito y coherente:

```yaml
portfolio_snapshot:
  snapshot_id: uuid
  cutoff_time: "2026-07-27T13:15:00Z"
  required_sleeves: [usdcop_smart_simple_v11, xauusd_trend_simple_v1]
  accepted_signals:
    - {signal_id: uuid, sleeve_id: usdcop_smart_simple_v11, as_of: "..."}
  stale_signals: []
  missing_signals: []
  fallback_applied: []
  max_age_by_sleeve:
    usdcop_smart_simple_v11: 7d
    xauusd_trend_simple_v1: 2d
```

Cada sleeve declara su política de faltante: `FLAT` | `KEEP_POSITION_UNTIL_EXPIRY` | `EXIT_ONLY` | `USE_LAST_VALID_WITH_MAX_AGE`. **`USE_LAST_VALID` sin límite de edad está prohibido.** La barrera también le da coherencia temporal al test pareado v11/v12/v14: mismas señales, mismo cutoff.

### 14.2 `portfolio_target`

El artefacto **inmutable y canónico** que el servicio de ejecución consume — Airflow lo publica; nunca envía órdenes:

```yaml
portfolio_target:
  target_id: uuid
  target_version: 184
  snapshot_id: uuid                        # el portfolio_snapshot de origen
  rebalance_cutoff: "2026-07-27T13:15:00Z"
  allocator_version: allocator_v1
  decision_fingerprint: "sha256:..."
  exposures:
    - {sleeve_id: usdcop_smart_simple_v11, instrument: USDCOP, side: LONG,
       risk_budget: 0.21, target_weight: 0.21, currency: COP}
  constraints_snapshot: {target_vol: 0.12, gross_cap: 1.5, caps_applied: [...]}
  infeasibility_fallback: null             # o el peldaño aplicado (§20.4)
  semantic_hash: "sha256:..."              # serialización canónica ⇒ == bytes_hash
```

---

## 15. Contratos de salida

### 15.1 `strategy_output` v2

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

  target: {type: nav_weight, value: 0.80, currency: COP}
  direction: LONG

  confidence:
    value: 0.67
    type: calibrated_probability        # calibrated_probability | raw_score — un score 0.8 NO es 80%
    calibration_version: cal_v2

  forecast: {volatility: 0.094, horizon: 5d}
  expected_holding_period: 5d

  health_snapshot_id: uuid              # de servicios de control INDEPENDIENTES —
  liquidity_snapshot_id: uuid           # la estrategia no se califica a sí misma
  decision_fingerprint: "sha256:..."
  reason_codes: [POSITIVE_MODEL_SCORE, HURST_GATE_OPEN, VOLATILITY_ACCEPTABLE]
```

### 15.2 `forecast_output` v1

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

  prediction: {type: return, point: 0.0062, lower: -0.0110, upper: 0.0240}
  direction_probability: {up: 0.58}

  model_fingerprint: "sha256:..."
  data_snapshot_id: uuid
  diagnostic_only: true
```

El allocator acepta **exclusivamente** `strategy_output` validado por contrato; un `forecast_output` se rechaza por tipo antes de cualquier lógica.

### 15.3 La muralla en permisos de base de datos

> La separación no depende de que el programador "recuerde" que el forecasting es diagnóstico. La base de datos la impone.

```text
forecast_writer
  INSERT/UPDATE controlado en forecast.*
  SIN permisos sobre action.*, portfolio.* o exec.*

strategy_writer
  INSERT en action.strategy_signal y bundles
  SIN permisos para crear órdenes

allocator_service
  SELECT de strategy_output aceptado en portfolio_snapshot
  INSERT en portfolio_snapshot, allocation y portfolio_target
  SIN credenciales de broker

execution_service
  SELECT únicamente de control.portfolio_target      # jamás de señales individuales
  INSERT en exec.*
  SIN permisos sobre estrategia, trials o métricas

frontend_role
  SELECT de vistas y endpoints autorizados
  SIN escritura directa a tablas de gobierno o ejecución
```

Las funciones sensibles se exponen mediante APIs o stored procedures con validación, no mediante escritura libre de tablas. El test de permisos corre en CI contra la base real, no contra el código.

---

## 16. Predictor interno versus zoo diagnóstico

Un modelo de forecasting puede ser **componente congelado** de una estrategia ACTION sin que el zoo diagnóstico se convierta en señal. Son productos distintos aunque ambos "pronostiquen".

```yaml
components:
  - component_id: usdcop_ridge_br
    role: decision_input
    spec_fingerprint: "sha256:..."       # ← ESTO es lo congelado: la RECETA
    retrain_policy: weekly_expanding     # parte del spec (v11 reentrena cada domingo)
    current_model_snapshot: "sha256:..." # rota semanalmente; cada snapshot queda registrado
    forecast_trial_ids: [FT-0041, FT-0042]   # la herencia del §10.2
```

Precisión obligatoria para el caso real de v11: **lo congelado es la receta** (features, hiperparámetros, política de reentrenamiento con ventana expansiva, scaler train-only) — no los pesos. Cada reentrenamiento produce un `model_snapshot_id` nuevo bajo el mismo `spec_fingerprint` y queda en el historial del Passport. El check de CI se redacta: *"la receta del componente está congelada; todo snapshot queda registrado con linaje"* — nunca "los pesos no cambian".

Consecuencias operativas:

- La estrategia completa se evalúa por PnL, riesgo, costos y forward — **aunque el predictor aislado tenga baja precisión** (el alfa de v11 vive en el gate Hurst: saber cuándo NO operar).
- El zoo público se evalúa por lift y error predictivo, y su dirección puede estar **cerrada** (familia FT CLOSED) mientras el panel sigue publicando el modelo congelado.
- Fallo del componente interno ⇒ **fail-closed de la señal** (jamás reusar el último forecast). Fallo del panel público ⇒ solo se degrada una página.

---

## 17. Ejecución event-sourced

### 17.1 Tablas comunes para replay, paper, canary y live

```sql
CREATE TABLE exec.order_header (
  order_id              UUID PRIMARY KEY,
  client_order_id       TEXT NOT NULL,
  idempotency_key       TEXT NOT NULL UNIQUE,
  account_id            TEXT,
  env                   TEXT NOT NULL,      -- replay | paper | canary | live
  executor_type         TEXT NOT NULL,      -- deterministic_simulator | broker
  sleeve_id             TEXT NOT NULL,
  allocation_id         UUID,
  instrument            TEXT NOT NULL,
  side                  TEXT NOT NULL,
  qty                   NUMERIC NOT NULL,
  order_type            TEXT NOT NULL,
  limit_price           NUMERIC,
  tif                   TEXT,
  currency              TEXT,
  parent_order_id       UUID,
  decision_fingerprint  TEXT NOT NULL,
  execution_fingerprint TEXT NOT NULL,
  submitted_at          TIMESTAMPTZ NOT NULL
);

CREATE TABLE exec.order_status_event (
  event_id        UUID PRIMARY KEY,
  order_id        UUID NOT NULL REFERENCES exec.order_header(order_id),
  event_time      TIMESTAMPTZ NOT NULL,
  status          TEXT NOT NULL,            -- SENT|ACK|PARTIAL|FILLED|REJECTED|CANCELLED|EXPIRED
  reason_code     TEXT,
  broker_order_id TEXT
);

CREATE TABLE exec.fill_event (
  fill_id    UUID PRIMARY KEY,
  order_id   UUID NOT NULL REFERENCES exec.order_header(order_id),
  fill_time  TIMESTAMPTZ NOT NULL,
  qty        NUMERIC NOT NULL,
  price      NUMERIC NOT NULL,
  commission NUMERIC,
  venue      TEXT
);

CREATE TABLE exec.fill_correction_event (
  correction_id UUID PRIMARY KEY,
  fill_id       UUID NOT NULL REFERENCES exec.fill_event(fill_id),
  event_time    TIMESTAMPTZ NOT NULL,
  field         TEXT NOT NULL,
  old_value     TEXT,
  new_value     TEXT,
  reason        TEXT NOT NULL
);
```

El estado de una orden es una **proyección de sus eventos, nunca un UPDATE destructivo.** El simulador determinista de paper escribe las mismas tablas con `executor_type = deterministic_simulator` — sin infraestructura ficticia de broker, pero con la granularidad que hace comparables fill rate, slippage e implementation shortfall entre paper y live **a igual grano**.

### 17.2 Entornos

| Entorno | Executor | Resultado |
|---|---|---|
| replay | `deterministic_simulator` | órdenes, fills y PnL simulados |
| paper | `deterministic_simulator` | eventos comparables con live |
| canary | `broker` (o sandbox) | capital limitado, reconciliado |
| live | `broker` | hechos reconciliados |

### 17.3 Idempotencia

```text
idempotency_key = SHA256(account_id ⊕ instrument ⊕ target_version ⊕ decision_fingerprint ⊕ rebalance_cutoff)
```

Un retry de Airflow re-publica el mismo target → misma key → **cero órdenes duplicadas.** El UNIQUE de la base es la última línea de defensa, no la única (§21.1).

---

## 18. Hechos de posiciones y PnL

### 18.1 DDL

```sql
-- Grano: (as_of, sleeve_id, instrument, env)
CREATE TABLE facts.fact_position (
  as_of          TIMESTAMPTZ NOT NULL,
  sleeve_id      TEXT NOT NULL,
  asset          TEXT NOT NULL,
  instrument     TEXT NOT NULL,
  env            TEXT NOT NULL,
  target_weight  NUMERIC,        -- lo que el modelo quería
  actual_weight  NUMERIC,        -- lo que quedó tras banda de no-trade
  delta_weight   NUMERIC,        -- turnover del día
  notional_usd   NUMERIC,
  vol_forecast   NUMERIC,
  conviction     NUMERIC,        -- p calibrada, si aplica
  regime_label   TEXT,           -- para atribución por régimen
  derivation_id  TEXT NOT NULL,
  run_id         TEXT NOT NULL,
  source_fill_set_id TEXT,       -- de qué fills deriva (paper y live por igual)
  PRIMARY KEY (as_of, sleeve_id, instrument, env)
);

-- Grano: (as_of, sleeve_id, instrument, env) — PnL DESCOMPUESTO
CREATE TABLE facts.fact_pnl (
  as_of            TIMESTAMPTZ NOT NULL,
  sleeve_id        TEXT NOT NULL,
  asset            TEXT NOT NULL,
  instrument       TEXT NOT NULL,
  env              TEXT NOT NULL,
  gross_pnl        NUMERIC,
  pnl_beta         NUMERIC,      -- exposición media × retorno del activo
  pnl_timing       NUMERIC,      -- cov(peso_t, retorno_t) — el candidato a alfa de timing
  pnl_carry        NUMERIC,      -- funding, roll, dividendos
  cost_commission  NUMERIC,
  cost_slippage    NUMERIC,      -- realizado vs modelado se compara en §23
  cost_financing   NUMERIC,
  pnl_residual     NUMERIC,      -- cierra la identidad contable
  net_pnl          NUMERIC,
  return_pct       NUMERIC,
  attribution_model_version TEXT NOT NULL,
  benchmark_id     TEXT,
  reconciliation_status TEXT,    -- PASS | FAIL | PENDING
  source_fill_set_id TEXT,
  derivation_id    TEXT NOT NULL,
  PRIMARY KEY (as_of, sleeve_id, instrument, env)
);
```

`env` en la clave primaria permite que paper y live convivan en la misma tabla — comparar v11-live contra v12-paper es **una query**, no una exportación.

### 18.2 Identidad contable (test de CI)

```text
gross_pnl = pnl_beta + pnl_timing + pnl_carry − commissions − slippage − financing + pnl_residual
|pnl_residual| / |gross_pnl| ≤ tolerancia    → si no, incidente
```

### 18.3 `timing_ratio`

```text
timing_ratio = Σ pnl_timing / Σ |gross_pnl|
```

Es un **diagnóstico de estilo y atribución dependiente del modelo elegido, no una demostración de alfa** — el claim de alfa sigue siendo exclusivamente el DSR del forward. Se reporta con intervalo de confianza por bootstrap en bloques y sensibilidad al benchmark. Su valor operativo: automatiza el diagnóstico que se hizo a mano con `gold_dynamic_exit` (+38% OOS con DSR 0.0077 = "ingeniería de riesgo sobre el beta del oro") — un sleeve con `pnl_timing` acumulado ~0 es beta disfrazado, por bonito que sea su equity curve, y el sistema lo muestra todos los días para todos los sleeves.

---

## 19. Métricas: definir, computar, persistir

Tres capas, cero duplicación. El pecado que mata la escalabilidad es un Sharpe distinto en el notebook, el DAG y el dashboard.

### 19.1 El catálogo define — `config/metrics/catalog.yaml`

```yaml
strategy.sharpe:
  formula_version: v1
  formula: "mean(r) / std(r) * sqrt({annualization})"
  annualization: from_asset_registry     # 252/365/52 — jamás hardcodeado
  min_periods: 60
  grain: [sleeve, book]

strategy.calmar:
  formula_version: v1
  source: fact_pnl
  annualization: from_asset_registry
  windows: [26w, 52w, since_anchor]
  warning: 0.10
  critical: 0.00

research.dsr:
  formula_version: v2
  n_trials: from_trial_ledger            # el N cobrado — jamás N_MAX
  variants: [family, cluster, global]
  claim_threshold: 0.95

research.pbo:        {method: cscv, n_splits: 16, threshold: 0.20}
strategy.timing_ratio: {formula: "sum(pnl_timing)/sum(abs(gross_pnl))", ci: block_bootstrap}
portfolio.capacity_usd: {formula: "0.01 * median(adv_usd) / max(abs(delta_weight))"}
```

### 19.2 El motor computa

```text
metrics_engine.compute(entity, metric, window, env, as_of)
```

Es el **único** código autorizado para calcular una métrica gobernada. Dashboard, jueces, gates, Vote 1, reportes y papers consumen el mismo número. `annualization: from_asset_registry` convierte la regla "prohibido rankear activos con relojes distintos" en algo que el motor resuelve solo.

### 19.3 `metric_event` persiste

```sql
CREATE TABLE control.metric_event (
  metric_event_id    UUID PRIMARY KEY,
  event_time         TIMESTAMPTZ NOT NULL,
  catalog_version    TEXT NOT NULL,
  formula_version    TEXT NOT NULL,
  entity_type        TEXT NOT NULL,        -- sleeve | asset | book | family | dag | forecast_spec
  entity_id          TEXT NOT NULL,
  strategy_id        TEXT,
  asset_id           TEXT,
  run_id             TEXT,
  environment        TEXT,
  metric_namespace   TEXT NOT NULL,        -- debe existir en el catálogo (CI)
  metric_name        TEXT NOT NULL,
  metric_value       DOUBLE PRECISION,
  metric_unit        TEXT,
  status             TEXT,                 -- OK | WARNING | CRITICAL
  threshold_warning  DOUBLE PRECISION,     -- copiados del catálogo VIGENTE al evaluar:
  threshold_critical DOUBLE PRECISION,     --   cambiar el umbral mañana no reescribe la historia
  dimensions         JSONB,
  lineage            JSONB,
  created_at         TIMESTAMPTZ DEFAULT NOW()
);
```

Ninguna métrica entra a `metric_event` si no está en el catálogo — el evento es persistencia, no un tercer lugar donde definir.

### 19.4 Namespaces

```text
data.*  research.*  strategy.*  forecast.*  forward.*  execution.*  portfolio.*  operations.*  governance.*
```

Ejemplos irrenunciables: `data.freshness_seconds`, `research.trials_charged`, `research.dsr`, `strategy.live_calmar`, `strategy.timing_ratio`, `forecast.lift_vs_baseline`, `execution.slippage_bps`, `execution.fill_rate`, `portfolio.cvar_95`, `operations.immutable_hit`, `governance.replay_live_parity`, `governance.overrides`.

---

## 20. Allocator: donde vive la adaptación

### 20.1 Baseline obligatorio

La primera versión operativa usa **inverse volatility con caps**. HRP (covarianza Ledoit-Wolf, 252d, min_obs 120) corre en **shadow** y solo se promueve si supera al baseline **neto de costos y turnover**, bajo su propio juez. "Probé 12 esquemas de asignación y elegí el mejor" es el mismo pecado un nivel más arriba.

### 20.2 Multiplicadores (pre-registrados, versionados, acotados, con histéresis)

```text
b_prov_i = b_base_i × m_forward_i × m_liq_i × m_div_i × m_ops_i × m_dd_i
```

| Multiplicador | Rango v1 | Fuente | Regla |
|---|---|---|---|
| `m_forward` | **[0, 1]** | Sharpe forward rolling con shrinkage | **solo reduce por deterioro; jamás premia una racha** — un rolling de 26 semanas es demasiado ruidoso para premiar. Restaurar es más lento que recortar |
| `m_liq` | [0, 1] | `liquidity_snapshot` (servicio independiente) | mecánico |
| `m_div` | [0.70, 1.10] | gate de novedad continuo (§12) | mecánico |
| `m_ops` | {0, 1} | `operational_state` | QUARANTINED ⇒ 0, cero aperturas |
| `m_dd` | [0, 1] | escalera de drawdown **por sleeve, normalizada por su vol esperada** | jamás umbrales universales; con histéresis para no oscilar semanalmente |

Solo con historial forward suficiente se consideraría ampliar `m_forward` a un rango como [0.75, 1.10] — y esa ampliación es una celda de la familia `book_allocation` que cobra su trial.

### 20.3 Optimización restringida (sin `normalize()`)

**Etapa A — presupuestos de riesgo** (cvxpy/OSQP; el problema es convexo — la restricción de vol es un cono de segundo orden):

```text
b* = argmin_b  ‖b − b_prov‖² + λ_to·‖b − b_prev‖₁

sujeto a:
  √(bᵀΣb) ≤ target_vol              ← vol como restricción, jamás como post-escala
  0 ≤ b_i ≤ cap_sleeve_i
  Σ_{i ∈ asset} b_i ≤ cap_asset     ← v11/v12/v14 comparten el riesgo del activo
  Σ b_i ≤ gross_cap
  ‖b − b_prev‖₁ ≤ turnover_budget
  restricciones de liquidez y factores
```

**Etapa B — exposición firmada:** `w_i = b_i × side_i`. El allocator asigna presupuesto de riesgo; **no opina de dirección.**

Está prohibido cualquier `normalize()` genérico al final: renormalizar tras el clip rompe los caps y cancela el vol targeting (fue un bug real de diseño, documentado y cerrado).

### 20.4 Fallback de infactibilidad (pre-declarado; cada peldaño loguea incidente)

1. Relajar únicamente el turnover budget dentro del límite declarado.
2. Encoger `b_prov` hacia cero hasta obtener factibilidad (0 siempre es factible).
3. Baseline equal-risk o inverse-vol con caps.
4. Si sigue infactible: target cero e incidente crítico.

### 20.5 Registry del libro — `config/book/allocator_v1.yaml`

```yaml
id: allocator_v1
method_baseline: inverse_vol_capped
method_shadow: hrp
covariance: {estimator: ledoit_wolf, window: 252, min_obs: 120}
constraints:
  max_weight_per_sleeve: 0.35
  max_weight_per_asset:  0.50
  target_vol_annual:     0.12
  gross_cap:             1.50
risk_scaling_book:                      # de-risking del LIBRO, adicional al m_dd por sleeve
  - {drawdown: 0.08, scale: 0.75}
  - {drawdown: 0.15, scale: 0.50}
  - {drawdown: 0.22, scale: 0.00}      # kill del libro
family_id: book_allocation             # el allocator es un modelo: familia, trials y juez propios
judge:
  anchor: "..."
  min_periods: 26
  criterion: "IR_libro > mejor sleeve individual, neto de costos y turnover"
```

Regla de capital, la que ata todo: **la evidencia de backtest permite entrar a paper; solo la evidencia forward permite recibir capital.** Y una estrategia con historial glorioso recibe **cero** si sus datos están vencidos, su paridad está rota, su executor presenta anomalías, violó sus límites o su retiro se activó — eso es `m_ops = 0` y es automático.

---

## 21. Ejecución y riesgo live

### 21.1 Pre-trade risk bloqueante (por orden)

Señal vigente y no vencida · snapshot de salud NOMINAL · posición esperada vs broker (**reconciliación antes de operar**, no solo EOD) · ausencia de duplicados · notional máximo · caps por sleeve y activo · gross/net exposure · apalancamiento · pérdida diaria y drawdown · price collar · liquidez · sesión de mercado · cash y moneda · límites de cuenta · `operational_state` · kill switch.

### 21.2 Kill switch

- **Independiente de Airflow** — matar Airflow no puede ser el mecanismo de emergencia; el servicio lo consulta antes de cada apertura o modificación.
- Niveles: `BLOCK_NEW`, `CANCEL_OPEN`, `EXIT_ALL`, `ACCOUNT_FREEZE`.
- Auditado con actor, causa, hora y alcance.
- Credenciales del broker fuera de Airflow, del repositorio y del frontend.

### 21.3 Reconciliación

Se ejecuta **antes de operar**, intradía para activos live, al cierre (EOD) y tras cualquier recuperación de servicio. Compara broker ↔ order ledger ↔ fills ↔ posiciones ↔ hechos. Discrepancia material ⇒ `incident` + QUARANTINED.

### 21.4 Retiro operativo

```text
CHAMPION → RETIRING
  → capital_tier := EXIT_ONLY
  → target cero o plan de liquidación
  → cancelar órdenes abiertas
  → reconciliar broker
  → confirmar position = 0 ∧ open_orders = 0
  → publicar bundle final
→ WITHDRAWN   (solo con exit_checklist = PASS)
```

El DAG desaparece **después** del checklist, jamás antes. Los bundles quedan para siempre (como `gold_dynamic_exit` y las variantes de BTC hoy). El retiro se dispara por el criterio del `withdrawal_protocol` firmado antes de operar — nunca por cómo se sienta el mes.

---

## 22. Linaje y point-in-time

### 22.1 Camino dorado (el invariante)

```text
provider → raw_snapshot → canonical_snapshot → feature_snapshot
→ model_or_policy_version → strategy_or_forecast_run → bundle_or_forecast
→ signal → portfolio_snapshot → allocation → order → fill → position → pnl_attribution
```

**Toda fila de `fact_pnl` debe poder recorrer el camino hasta el snapshot raw.** Si no puede, es un incidente de linaje. Este recorrido es lo que resuelve incidentes en minutos y hace defendible un paper (IEEE) o una tesis.

### 22.2 DDL del grafo

```sql
CREATE TABLE control.lineage_node (
  node_id        TEXT PRIMARY KEY,       -- derivation_id
  node_type      TEXT NOT NULL,          -- raw|canonical|feature|dataset|model|signal|bundle|forecast|pnl
  asset          TEXT,
  sleeve_id      TEXT,
  as_of          TIMESTAMPTZ,
  produced_by_run TEXT,
  code_hash      TEXT,
  params_hash    TEXT,
  semantic_hash  TEXT,
  bytes_hash     TEXT,                   -- ≠ semantic_hash en el mismo derivation_id ⇒ no-determinismo
  schema_version TEXT,
  row_count      BIGINT,
  min_event_time TIMESTAMPTZ,
  max_event_time TIMESTAMPTZ,
  quality_status TEXT,
  status         TEXT NOT NULL,          -- VALID | STALE | INVALIDATED
  storage_uri    TEXT
);

CREATE TABLE control.lineage_edge (
  parent_id TEXT NOT NULL REFERENCES control.lineage_node(node_id),
  child_id  TEXT NOT NULL REFERENCES control.lineage_node(node_id),
  role      TEXT NOT NULL,               -- CONSUMED|PRODUCED|DERIVED_FROM|CORRECTED_BY|SUPERSEDES
  PRIMARY KEY (parent_id, child_id, role)
);
```

Consultas que justifican el esfuerzo: **procedencia hacia atrás** ("¿de dónde salió el PnL del 2026-04-13?") y **cascada hacia adelante** (un `UPDATE` recursivo sobre `lineage_edge` marca `STALE` todo lo derivado de un dato corregido).

### 22.3 Revisiones tipificadas — la cascada NO es universal

```text
revision_type: LEGITIMATE_RELEASE | PROVIDER_CORRECTION | PIPELINE_ERROR | SCHEMA_REINTERPRETATION
ramas:         as_released (append-only, intocable) | latest_revised (proyección)
```

- `LEGITIMATE_RELEASE` (p.ej. ALFRED revisa el NFCI hacia atrás, como cada semana): crea nuevo snapshot en `latest_revised`; **nada histórico pasa a STALE**. Una decisión que usó el vintage original es la reconstrucción *correcta* de lo conocible — invalidarla sería destruir el point-in-time que el sistema protege.
- `PROVIDER_CORRECTION` / `PIPELINE_ERROR`: cascada `STALE` sobre todo lo derivado del valor erróneo, con toda la fuerza.
- `SCHEMA_REINTERPRETATION`: nueva `schema_version`; las comparaciones cross-versión se flaggean, jamás son silenciosas.

Todo run declara qué rama consumió. **Screening consume `as_released` por defecto** — es la rama PIT-correcta.

### 22.4 OpenLineage

OpenLineage es **formato y transporte** de eventos; PostgreSQL mantiene el **único** grafo consultable; Marquez es un visor opcional. No se mantienen dos grafos de verdad. Los operadores propios requieren emisión manual (`lineage.emit()` desde los scripts) — la integración no es automática para PythonOperators custom.

---

## 23. Monitoreo en tres relojes

Un solo motor de evaluación (lee `fact_*` + `lineage_node` + `metric_event`), tres latencias, tres clases de acción automática:

| Reloj | Qué vigila | Frecuencia | Señal | Acción automática |
|---|---|---|---|---|
| **Datos** | freshness por serie, nulls, `bytes_hash`/`semantic_hash` inconsistentes, artefactos `STALE`, revisiones de vintage detectadas | minutos | rojo binario | **fail-closed**: el sleeve cae a su política degradada declarada — nunca al último valor; QUARANTINED si toca un componente activo |
| **Modelo** | PSI/KS de features vs train, drift de la distribución de predicción, Brier rolling (si hay calibración), paridad train/serve | diaria | amarillo | alerta + **congelar promociones** |
| **PnL** | tracking error live-vs-paper (test de desviación acumulada, 3σ), `timing_ratio`, Sharpe rolling 12m < 50% del backtest, slippage realizado > 2× modelado, decay | semanal | naranja | disparar `withdrawal_protocol` / `REDUCED` |

Umbrales de referencia: PSI > 0.25 (drift de features), predicción fuera de ±2σ histórico, Brier degradado > 20% en 6m, `operations.immutable_hit != true` en re-runs.

### 23.1 Tabla de fallas diferencial (contrato de `control__system_health`)

| Falla | DIAGNOSTIC | ACTION | Respuesta |
|---|---|---|---|
| Datos rancios | Panel `STALE` | Bloquea nueva señal | incident + health snapshot |
| PNG ausente | Oculta imagen | Sin efecto | degradación elegante |
| Modelo diagnóstico falla | Resto del zoo continúa | Sin efecto | error visible |
| Componente activo falla | No aplica | **Fail-closed** | target cero o política declarada; jamás reusar el último forecast |
| Predicción sin lift | Veredicto negativo publicado | **No invalida automáticamente PnL positivo** | separar ciencia predictiva y decisión |
| Señal inválida | No aplica | No se publica | contract violation |
| Paridad rota | Advertencia | **QUARANTINED** | incidente crítico |
| Executor caído | Sin efecto | No nuevas órdenes | kill switch / recuperación |
| Broker discrepante | Sin efecto | QUARANTINED | reconciliación |
| Métrica ausente | `N/A` | El gate no aprueba | jamás imputar en silencio |
| Bundle incompleto | UI parcial | No hay promoción | verify fail |
| Forecast público caído | Página degradada | Posiciones sin cambio | barrera contractual |

> La caída del forecasting público degrada una página. La caída de un componente de la estrategia bloquea dinero.

---

## 24. Frontend y Passport

### 24.1 Regla

> El frontend no calcula. Lee artefactos, vistas o endpoints versionados que ya contienen hechos y estados validados.

### 24.2 Vistas

| Ruta | Propósito | Fuentes | Puede escribir |
|---|---|---|---|
| `/dashboard` | Vote 2 y gates | bundle inmutable + estado de aprobación | voto mediante API validada |
| `/replay` | Señales, trades, equity y KPIs | bundles ACTION | nada |
| `/production` | Forward, canary y live | facts + métricas | nada |
| `/execution` | Órdenes, fills, riesgo y kill switch | vistas live | acciones RBAC controladas |
| `/forecasting` | Predicciones, intervalos y métricas | `forecast.*` | nada |
| `/analysis` | Contexto macro/news/LLM | artefactos namespaced | nada que afecte señal |
| `/hub` | Navegación y RBAC | control plane | nada |

### 24.3 Forecasting UI — barreras visibles

Muestra permanentemente: **DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN**. Sin botones de aprobación, capital ni órdenes; sin colores ni etiquetas imperativas ("COMPRAR"). Correcto: *"probabilidad estimada de subida: 58%"*. El veredicto contra baseline se muestra siempre, incluso negativo.

### 24.4 Passport dividido

- `v_strategy_passport_live`: vista ligera **no materializada** — órdenes abiertas, último fill, quarantine, reconciliación, riesgo actual, kill switch. (Una MV de Postgres se reemplaza en el refresh y no puede sostener estado operativo en vivo.)
- `mv_strategy_performance_daily`: vista **materializada**, refresco nocturno — Sharpe, DD, DSR×3, atribución, históricos.
- `v_strategy_passport`: composición de ambas. Reúne identidad, gobierno (trials FT/AT, N×3, votos, protocolo de retiro), linaje (snapshots, fingerprints), desempeño por entorno (backtest/held-out/paper/canary/live — **la misma métrica del mismo motor en las cinco columnas**), ejecución y riesgo. Deja de existir el paseo Airflow → MLflow → JSONs → SQL para reconstruir qué pasó: es un `SELECT`.

### 24.5 Control Tower (portada, responde en diez segundos)

```text
LIBRO    capital · PnL d/m/y · vol prevista vs objetivo · DD · gross/net · CVaR
         conteo por estado: CHAMPION | CANARY | PAPER | REDUCED | QUARANTINED
         matriz de correlación entre sleeves · ratio de diversificación
         descomposición del PnL: timing vs beta vs carry
SLEEVES  sleeve | research | tier | ops | env | ρ_max | Sharpe | DSR(N×3) | timing_ratio
         | m_forward·m_dd vigentes | turnover | días al juez | semáforo de retiro
         v11-live vs v12/v14-paper: test PAREADO con su p-value (o e-value) declarado
DATOS    freshness · STALE por revisión · paridad replay · trials por familia
         N_global vs N_MAX=989 · última revisión de vintage detectada
```

### 24.6 Transportes

1. File-based BFF para bundles, charts y forecasting estático. 2. PostgreSQL/API para estado live. 3. SSE/WS solo para streams necesarios. 4. Todos los archivos de frontend son proyecciones regenerables, jamás fuente de verdad.

---

## 25. Validaciones obligatorias de CI y registry

```text
─ Fuente de verdad y declaraciones ─
✓ exactamente una fuente autoritativa por atributo
✓ toda estrategia ACTION referencia una familia y un cluster válidos
✓ toda estrategia FROZEN tiene manifest, code_hash y withdrawal_protocol EXISTENTE
✓ exactamente 1 campeona visible por (asset, surface=action)
✓ cambiar campeona exige evidencia en el mismo commit

─ Trials ─
✓ toda celda cobrada aparece exactamente una vez en el ledger
✓ trials_charged de cada familia coincide con el ledger
✓ N_MAX no entra en el DSR
✓ todo FT-/AT- referencia su familia; herencia de FT en provenance validada
✓ candidatas concurrentes de la misma familia: Σ alpha ≤ 0.05
✓ juez declara sequential.method (mSPRT | e_process | none)

─ Muralla ─
✓ surface=diagnostic no declara signal, allocate, execute ni approve
✓ el allocator solo consume strategy_output (rechazo por tipo)
✓ ninguna arista forecast://*/prediction → allocator/execution (falla el parseo)
✓ rol forecast_writer sin INSERT en action.* ni exec.* (test contra la base real)
✓ frontend forecasting sin verbos de orden, botones de aprobación ni endpoints de ejecución
✓ receta de componente decision_input congelada; todo snapshot registrado con linaje
✓ fallo de componente interno bloquea la señal; fallo de panel público no cambia posiciones

─ Identidad y paridad ─
✓ misma decisión (decision_fingerprint) produce el mismo semantic_hash
✓ artefactos canónicos controlados producen bytes idénticos
✓ mismo derivation_id con distinto semantic_hash ⇒ incidente de no-determinismo
✓ ninguna fila serializa NaN o Infinity
✓ replay independiente reproduce el semantic_hash del paper ledger anclado

─ Estados ─
✓ toda combinación (research, tier, ops) cumple la matriz de legalidad
✓ PAPER no puede tener tier FULL; QUARANTINED bloquea aperturas
✓ WITHDRAWN exige exit_checklist = PASS
✓ DAG de estrategia existe ⇔ research_state ∈ {FROZEN, PAPER, CHAMPION, RETIRING}

─ Ejecución y hechos ─
✓ toda orden tiene idempotency_key única
✓ paper y live escriben los mismos contratos de order/fill (difiere executor_type)
✓ toda señal tiene version, valid_until y decision_fingerprint
✓ todo forecast tiene model_id, horizon y target_time
✓ fact_pnl reconcilia con fills; identidad contable dentro de tolerancia
✓ portfolio_snapshot tiene cutoff y política de faltantes; no existe lectura de "latest" sin cutoff
✓ Vote 2 se emite sobre bundle inmutable

─ Métricas y datos ─
✓ toda métrica existe en el catálogo y persiste formula_version + thresholds vigentes
✓ annualization proviene del asset registry; comparaciones cross-asset respetan reloj y moneda
✓ una revisión LEGITIMATE_RELEASE no invalida vintage histórico
✓ el backfill incluye campeonas, candidatas, retiradas y baselines (anti-supervivencia)
✓ gate de novedad evaluado y registrado en toda promoción PAPER → CHAMPION
```

---

## 26. Seguridad y gobierno operativo

- RBAC diferenciado: research, approver, risk, executor y auditor.
- Secretos del broker fuera de Airflow, del repositorio y del frontend.
- Overrides append-only con actor, motivo, ventana y expiración; doble control para Vote 2 y cambios live sensibles.
- Backups de PostgreSQL y MinIO **con pruebas de restauración**; RPO/RTO definidos para datos, control plane y ejecución.
- Runbooks para: broker caído, base caída, red caída, posición huérfana, fill tardío.
- Logs con correlación por `run_id`, `decision_fingerprint` y `order_id`.
- **Ningún LLM puede crear señales, aprobar estrategias ni enviar órdenes** (L8 clasifica y resume; jamás decide).

---

## 27. Stack: ahora y cuando duela

| Función | Ahora (una persona, 4 activos) | Cuando exista necesidad demostrada |
|---|---|---|
| Orquestación | Airflow | mantener |
| Control plane | PostgreSQL (gobierno + hechos + eventos + linaje) | réplicas/particiones |
| Snapshots y bundles | MinIO + Parquet (`raw/canonical/features/snapshots/bundles`) | Iceberg para time-travel nativo |
| Modelos | MLflow (registry, aliases, run↔experimento) | mantener |
| Linaje | tablas propias, `lineage.emit()` desde scripts | OpenLineage + Marquez como transporte/visor |
| Point-in-time | `available_at` + as-of joins con tolerance | Feast solo con múltiples equipos — probablemente nunca |
| Dashboard | Grafana sobre Postgres + app actual | mantener |
| Telemetría infra | logs de Airflow | Prometheus/OTel para contenedores — no para métricas de negocio |
| Optimización | cvxpy/OSQP | servicio dedicado si sube la frecuencia |
| Ejecución | servicio aislado USD/COP | adaptadores por broker/activo |

No se adoptan colas, Kubernetes, feature stores distribuidos ni simuladores de microestructura por imitación institucional: resuelven problemas de decenas de personas y miles de sleeves; a esta escala son costo puro.

---

## 28. Roadmap consolidado (con criterio de terminado por etapa)

| Etapa | Contenido | Criterio de terminado |
|---|---|---|
| **0 — Constitución técnica** | matriz de fuente de verdad; convenciones de IDs y tiempos; semántica de revisiones; matriz de legalidad (estados × superficie); política de serialización canónica; política de permisos ACTION/DIAGNOSTIC; taxonomía de trials FT/AT con herencia | el CI puede rechazar una declaración inválida **antes** de ejecutar un DAG |
| **0.5 — Diagnóstico rápido no durable** | `timing_ratio` + atribución preliminar de las 4 campeonas como **script desechable** sobre bundles publicados | el número existe esta semana; cero tablas, cero deuda de esquema, cero decisiones de promoción |
| **1 — Trials y familias** | ledger doble (FT-/AT-) append-only; familias transversales; clusters controlados; N×3 y DSR×3; backfill `legacy_estimate` de los zoos | **la etapa irreparable-hacia-atrás**: cada activo sumado con el N fragmentado es deuda estadística sin refinanciación |
| **2 — Identidad** | fingerprints; hashes; canonical writer; spine; CI de paridad | precede tablas de hechos y backfills — evita el retrabajo de agregar spine después |
| **3 — Event sourcing y hechos** | order/fill común (4 entornos); `fact_position`; `fact_pnl` + residual; schemas y roles de Postgres; reconciliación; idempotencia | la identidad contable pasa tolerancia; el test de permisos corre contra la base |
| **4 — Métricas** | catálogo versionado; motor único; `metric_event` con thresholds históricos; namespace `forecast.*`; ICs | ningún Sharpe se calcula fuera del motor |
| **5 — Backfill anti-sesgo** | campeonas + candidatas + retiradas + baselines; todos los años y entornos reconstruibles | la Control Tower no tiene sesgo de supervivencia |
| **6 — Linaje** | nodes/edges; camino dorado; revisión tipificada; cascada selectiva; emisión OL | una fila de PnL llega a raw snapshot |
| **7 — Factories y Passport** | `asset_data_factory` + `strategy_factory` + `forecast_factory` en paralelo al factory actual; **diff semántico** de bundles; Passport live + performance | bundles idénticos (semantic_hash) entre factory viejo y nuevo |
| **8 — Snapshot y allocator shadow** | barrera temporal; baseline inverse-vol con caps; HRP shadow; multiplicadores solo-reductores; gate de novedad; juez del allocator | pesos propuestos vs realidad comparados sin capital |
| **9 — Canary de ejecución** | solo tras: paridad verde, idempotencia probada, pre-trade, reconciliación pre/intra/EOD, kill switch, RETIRING ensayado, runbooks probados, mínimos canary | los 15 criterios de aceptación (§30) |

**USD/COP L7 es lo último en migrarse** (§29).

---

## 29. Migración strangler de USD/COP

COP es la cadena artesanal con dinero real: la que más necesita el código parametrizado y testeado, y la que más riesgo corre al migrar. Se migra **capa por capa, de la menos riesgosa a la más**, corriendo ambas implementaciones en paralelo y exigiendo paridad antes de cada paso:

```text
orden:  ingest → canon → verify → features → dataset → train → gate → signal → execute
```

Reglas:

1. En cada paso, el generador nuevo y el DAG artesanal corren en paralelo ≥ 2 semanas; se comparan **semantic_hash** de los artefactos (bytes para los canónicos).
2. Ninguna capa avanza si la anterior no lleva paridad verde sostenida.
3. Los `ExternalTaskSensor` (L3→L5) se reemplazan por Airflow Assets recién cuando esa capa migra.
4. **L7 migra al final**, solo cuando el resto lleve ≥ 1 mes de paridad, y con el servicio de ejecución externo, idempotencia, pre-trade y kill switch ya probados en canary de otro flujo.
5. El A/B vivo (v11/v12/v14) no se toca durante la migración: sus ledgers son precisamente el patrón de referencia de la paridad.
6. Rollback declarado por capa: apagar el DAG nuevo re-habilita el artesanal sin pérdida (ambos escriben artefactos inmutables).

---

## 30. Criterios de aceptación de producción

La Fabric se considera apta para ampliar live cuando:

1. El 100% de señales live tiene `decision_fingerprint` reproducible.
2. Un replay independiente reproduce el `semantic_hash` de cada señal.
3. Un retry no genera órdenes duplicadas (probado con inyección de fallos).
4. Paper y live usan el mismo ledger de eventos y esquema.
5. `fact_position` y `fact_pnl` reconcilian con fills y broker.
6. El portfolio target siempre referencia un portfolio snapshot con cutoff.
7. No existe dependencia entre forecast diagnóstico y ejecución (verificado en permisos y en grafo de DAGs).
8. Todos los estados cumplen la matriz de legalidad.
9. El proceso `RETIRING` fue ensayado de punta a punta.
10. El kill switch funciona **con Airflow caído**.
11. Las vistas live no dependen de un refresh materializado diario.
12. Todos los gates y métricas provienen del motor y catálogo únicos.
13. El backfill no presenta sesgo de supervivencia.
14. El linaje de una fila de PnL llega hasta raw snapshot.
15. Los runbooks de incidentes fueron probados mediante simulacros.

---

## 31. Decisiones explícitamente rechazadas

- Un DAG por activo que contenga todas las estrategias indefinidamente.
- Un único YAML gigante con datos, estrategias, forecasting y ejecución mezclados.
- Forecasting como señal por defecto.
- `normalize(clip(...))` en el allocator.
- Consultar "el último valor" sin cutoff.
- Paper sin órdenes y fills.
- Byte parity sobre formatos externos no canónicos — y semantic hash débil donde **sí** se controla el escritor.
- Invalidar vintages legítimos.
- Retirar una estrategia eliminando inmediatamente su DAG.
- Airflow como única defensa contra duplicación de órdenes.
- Canary definido solo por semanas calendario.
- `timing_ratio` como prueba de alfa.
- Incrementar capital por Sharpe rolling ruidoso en allocator v1.
- Backfill exclusivo de campeonas.
- Passport materializado diario como fuente de estado live.
- Dos pipelines de features por superficie (dos cómputos = deriva de paridad).
- Familias nuevas para lavar multiplicidad (el cluster y el N_global lo hacen visible).

---

## 32. Estructura del repositorio

```text
config/
  assets/            # {asset}.yaml — verdad física
  strategies/        # {sleeve_id}.yaml — ACTION
  forecasts/         # {spec_id}.yaml — DIAGNOSTIC
  macro/             # bundles de series macro
  metrics/           # catalog.yaml
  book/              # allocator_v1.yaml

registries/
  families/          # ACTION y FORECAST
  research_clusters.yaml
  ledger.jsonl       # FT-/AT- append-only, hasheado

contracts/
  strategy_output.py
  forecast_output.py
  portfolio_snapshot.py
  portfolio_target.py
  order_events.py

control_plane/
  identity/          # fingerprints, canonical writer, semantic_hash
  lineage/           # nodes/edges, cascada, emit()
  metrics/           # motor único
  governance/        # estados, matriz de legalidad, votos, overrides
  reconciliation/

execution/
  service/           # fuera de Airflow
  pretrade/
  adapters/          # broker por activo
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
  qlab.py            # el CLI: declare, screen, freeze, promote, close
  timing_ratio_oneoff.py   # Etapa 0.5, desechable

docs/
  architecture/      # este documento
  withdrawal/        # un protocolo firmado por sleeve
  runbooks/
```

---

## 33. Runbooks

### 33.1 Nuevo activo (~1 hora, cero estrategias)

1. `config/assets/{asset}.yaml` — calendario, reloj, `annualization`, fuente, costos, canary_requirements.
2. `config/macro/{bundle}.yaml` si necesita macro nueva.
3. `qlab asset validate {asset}` → CI verde.
4. `asset__{asset}__data` aparece en el próximo parseo; backfill con `--as-of` en el DAG de backfill.
5. **Cero estrategias todavía** — se deja acumular datos y linaje.

### 33.2 Nueva hipótesis (antes de mirar nada)

1. `registries/families/{family}.yaml` con **todas** las celdas y el bar pre-firmado (ACTION o FORECAST según la pregunta).
2. `qlab family declare` → commit. Sin esto, el screening rechaza.
3. `qlab screen --cell ... --charge-trial` por celda mirada — cada una cobra +1 (AT o FT).
4. Si ninguna bate el bar: `qlab family close --note "..."`. **Cerrar una familia es un resultado, no un fracaso.**

### 33.3 Nueva estrategia (solo desde una celda SCREENED que pasó)

1. `config/strategies/{id}.yaml` con family, cluster, `provenance` (AT + FT heredados), juez completo y protocolo de retiro.
2. `docs/withdrawal/{id}.md` firmado **antes** del freeze.
3. `qlab freeze` → `spec_fingerprint`, manifiesto SSOT.
4. `strat__{id}` aparece; corre en PAPER (tier SHADOW como máximo) hasta que el juez hable.

### 33.4 Nuevo candidato A/B

1. Copiar el YAML de la incumbente; cambiar `version` y la variante; `judge.paired_with: {incumbente}`.
2. Repartir α entre concurrentes (Bonferroni/Holm); calcular el MDE con la ventana firmada.
3. `research_state: PAPER`, ancla = fecha del freeze. **Jamás re-anclar.** Cambiarla en evaluación = versión nueva + freeze nuevo + ancla nueva.

### 33.5 Retiro

1. Se dispara solo por el criterio del `withdrawal_protocol` — nunca por opinión.
2. `RETIRING` + tier `EXIT_ONLY` → checklist de §21.4 → `WITHDRAWN`.
3. El DAG desaparece; los bundles y su historia quedan publicados para siempre.

### 33.6 Incidente de datos (revisión / corrupción)

1. Clasificar `revision_type` (§22.3). LEGITIMATE_RELEASE → snapshot nuevo, nada se invalida.
2. Corrección/error → cascada `STALE` vía `lineage_edge`; los sleeves afectados con componente activo pasan a fail-closed.
3. Regenerar solo lo `STALE`; verificar paridad; cerrar el incidente con causa raíz en el ledger de incidentes.

---

## Anexo A — Protocolo de investigación de referencia (lo que `qlab screen` exige)

Todo screening — la única actividad que cobra trials — cumple este estándar; el bar concreto vive en cada familia.

**A.1 Particiones.** Sandbox de diseño (libre) / purga + embargo / OOS sellado (una sola apertura, con contador) / holdout final (una apertura en la vida) / forward (la verdad). El sandbox produce **un candidato pre-registrado**, no un menú.

**A.2 Validación.** Purged K-Fold + embargo (purga = sacar del train todo sample cuyo `[t0, t1]` solape con test; embargo ≈ 1–2% adicional). **CPCV** (N=6, k=2 → caminos múltiples → distribución de Sharpe, no un punto) contrastado con walk-forward anclado — si discrepan fuerte, no hay estrategia. Prohibidos: `KFold(shuffle=True)`, splits aleatorios, estadísticos ajustados sobre la muestra completa (todo escalado/winsorización es expanding o intra-fold).

**A.3 Labels.** Triple barrier (TP/SL en múltiplos de σ dinámica + barrera vertical) y **meta-labeling** como default cuando existe una primaria: la regla congelada da dirección, el modelo aprende cuándo funciona → tamaño. Salida obligatoria `(X, y, w, t1)` con `w = uniqueness × |retorno| × decay` — sin pesos por solapamiento, el modelo cree tener 20× más información de la que tiene.

**A.4 N efectivo.** Con relaciones macro, el N efectivo son los **ciclos** (≈ 6–8 en 1995–2026), no las filas. Corolarios duros: ≤ 4–6 features finales con **prior de signo declarado antes de mirar** (sin prior, la feature no entra al store), modelos de profundidad ≤ 3 con restricciones monótonas codificando los priors, y desconfianza de toda ganancia < ~0.3 de Sharpe. Redes profundas a esta escala = autoengaño.

**A.5 HPO.** Espacio chico con priors; CV **anidada** (outer evalúa, inner elige); objetivo robusto `mediana(Sharpe_caminos) − λ·IQR − γ·turnover` (maximizar el máximo = maximizar la suerte); `n_trials` **pre-declarado y contado en el ledger** — re-correr el estudio "porque no gustó" es un trial de nivel superior; pruner con cautela o ninguno (los folds tempranos son otro régimen, no una muestra peor); sembrar con la config incumbente (`enqueue_trial`).

**A.6 Deflación y falsificación.** DSR > 0.95 con el N del ledger (skew y kurtosis incluidos); PBO (CSCV) < 0.20; superficie de parámetros = meseta, no pico; sub-períodos con signo consistente; supervivencia a costos ×1.7 y ×3.3 y a un día de retraso; labels aleatorizados → Sharpe ≈ 0; y el candidato debe ganarle al **baseline tonto después de deflatar** — "no cablear nada" es el resultado más frecuente de un proceso honesto y se archiva como éxito.

**A.7 SHAP como falsificador.** Solo sobre folds de test; cortes global / por régimen / temporal (SHAP medio por año); regla de kill: una feature cuyo aporte cambia de signo entre décadas o contradice su prior **se elimina aunque mejore el backtest**; agrupado por clúster si quedan |ρ| > 0.7; force plots de los 10 peores días. SHAP explica el modelo, no el mercado — sirve para rechazar modelos absurdos, no para probar verdades.

---

## Anexo B — Particularidades por activo (por qué la mecánica no viaja)

| | SPX500 | XAU/USD | BTC/USDT | USD/COP |
|---|---|---|---|---|
| Calendario | NYSE, gaps overnight grandes (casi todo el retorno histórico del índice es overnight) | ~24h weekday | 24/7, sin gaps | CO ∩ US, sesión 8:00–12:55 COT |
| Reloj | √252 | √252 weekday | √365 | semanal / M5 |
| Macro relevante | crédito (HY/Baa−10y), vol (VIX/term structure), curva, dólar | DXY, tasas reales, VIX | funding, on-chain, liquidez global, DXY | diferencial de tasas, WTI, EMBI, EMFX |
| N efectivo macro | ~30–60 (6–8 ciclos) | ~20–40 | **~4 ciclos** → macro casi inutilizable estadísticamente | ~20 |
| Barra recomendada | time bars diarias | diarias | **dollar bars** para intradía | M5 |
| Costo típico | 3 bps (stress 5/10) | spread moderado | 5–10 bps + funding | spread variable, sube en estrés |
| Riesgo #1 | look-ahead macro (vintages) | confundir beta con alfa (caso `dynamic_exit`) | historia corta + cambio de régimen estructural | iliquidez y saltos por intervención |
| Mecánica ganadora observada | probablemente MA200 puro | trend simple (votos SMA) | beta pasivo vol-targeted | semanal TP/HS + gate Hurst |

Detalles estructurales de SPX que ninguna feature puede ignorar: rango 459 → 7,414 (todo en ratios/retornos/z-scores, jamás niveles); régimen estructural de vol (normalizar por vol local, no de muestra completa); retornos de feriado/fin de semana escalados por días hábiles; price-return ≠ total-return (~1.8–2.0% anual de dividendos) declarado en señal, PnL y benchmark; warmup honesto (MA200 + z-scores ⇒ la serie usable arranca después del piso).

---

## Anexo C — Glosario

- **Activo:** mercado con calendario, reloj, datos, costos y capacidades físicas propias.
- **Sleeve:** política ACTION congelada sobre un activo (`{asset}_{mecanica}_{version}`).
- **Familia:** pregunta pre-registrada (económica o predictiva) que contiene celdas y consume trials.
- **Cluster:** agrupación controlada de familias relacionadas para el conteo N_cluster.
- **Forecast spec:** producto DIAGNOSTIC por modelos, target y horizontes.
- **Trial:** mirada o variante que consume presupuesto estadístico (AT- económico, FT- predictivo).
- **Snapshot:** conjunto inmutable de datos identificado y fechado.
- **Bundle:** artefactos inmutables publicados por una estrategia (`summary/trades/signals`).
- **Passport:** vista derivada que reúne identidad, gobierno, linaje, desempeño, riesgo y ejecución. Nunca se escribe a mano.
- **Portfolio snapshot:** conjunto temporalmente coherente de señales aceptadas para un cutoff.
- **Portfolio target:** exposición inmutable resultante del allocator; lo único que consume el executor.
- **Quarantine:** bloqueo operacional ortogonal al estado de investigación.
- **Canary:** capital limitado sujeto a mínimos por activo.
- **Semantic hash / bytes hash:** identidad lógica del contenido normalizado / integridad física del archivo.
- **Point-in-time:** uso exclusivo de información disponible (`available_at`) en el momento de la decisión.
- **as_released / latest_revised:** rama vintage intocable / proyección con revisiones conocidas.

---

## Anexo D — Disposición consolidada de la auditoría

| Riesgo auditado | Resolución normativa | Estado |
|---|---|---|
| Fingerprint incluía entorno y se comparaba entre entornos | `spec` / `decision` / `execution_fingerprint`; paridad = `decision_fingerprint → semantic_hash` | Cerrado en diseño |
| Byte parity universal inviable | bytes idénticos solo para escritores canónicos; hash semántico para formatos externos | Cerrado en diseño |
| Paper sin órdenes/fills | contrato event-sourced común (4 entornos, `executor_type`) | Cerrado en diseño |
| `normalize(clip(...))` rompía caps y vol target | optimización restringida sin normalización final | Cerrado en diseño |
| Señales con edades distintas en el libro | `portfolio_snapshot` con cutoff, staleness y fallbacks | Cerrado en diseño |
| Airflow como bus de ejecución | `portfolio_target` inmutable + servicio externo con idempotencia | Cerrado en diseño |
| Retiro borraba el DAG con posiciones abiertas | `RETIRING` + `EXIT_ONLY` + checklist obligatorio | Cerrado en diseño |
| Revisiones legítimas invalidaban decisiones PIT correctas | taxonomía de revisiones + ramas `as_released`/`latest_revised` | Cerrado en diseño |
| Dobles fuentes de verdad | matriz constitucional de autoridad por entidad | Cerrado en diseño |
| Familias divididas para ocultar multiplicidad | clusters controlados + N/DSR familiar, cluster y global | Cerrado en diseño |
| Canary de 8 semanas insuficiente | mínimos parametrizados por decisiones, fills, regímenes e incidentes | Cerrado en diseño |
| Juez secuencial indefinido | `mSPRT` / `e_process` / `none`; actuar antes del horizonte = violación registrada | Cerrado en diseño |
| `timing_ratio` como prueba de alfa | reclasificado como atribución diagnóstica con residual e IC | Cerrado en diseño |
| `strategy_output` mezclaba scores, salud y liquidez | contrato tipado; salud y liquidez de servicios independientes | Cerrado en diseño |
| Estados mezclaban investigación, capital y salud | tres dimensiones + matriz de legalidad | Cerrado en diseño |
| `m_forward` premiaba rachas | v1 solo reduce; histéresis; restaurar más lento que recortar | Cerrado en diseño |
| HRP entraba directo | baseline inverse-vol incumbente; HRP en shadow con juez propio | Cerrado en diseño |
| Faltaba pre-trade y kill switch externo | gate por orden, reconciliación previa, kill switch independiente | Cerrado en diseño |
| OpenLineage duplicaba el grafo | OL como transporte; PostgreSQL como único grafo | Cerrado en diseño |
| Passport materializado como estado live | vista live + MV histórica separadas | Cerrado en diseño |
| Roadmap empezaba por hechos antes de identidad | constitución e identidad preceden hechos y backfill | Cerrado en diseño |
| Replay y forecasting confundibles | superficies, contratos, permisos, DAGs, vistas y trials separados | Cerrado en diseño |
| Forecasting "gratis" como puerta de minería | doble ledger FT/AT + herencia por research_cluster | Cerrado en diseño |
| Un sleeve juzgado en aislamiento | gate de novedad (ρ, IR marginal) obligatorio en promoción | Cerrado en diseño |

**Nota de auditoría:** "cerrado en diseño" no significa "implementado". Cada fila debe convertirse en código, migración, test y evidencia antes de considerarse cerrada operacionalmente (§30).

---

## 34. Regla de cierre

> Un forecast puede ser preciso y no producir una estrategia rentable. Una estrategia puede ser rentable aunque su predictor aislado sea débil. El sistema debe medir ambas cosas honestamente, impedir que se confundan y conservar el linaje que explica cómo una observación terminó —o no terminó— en una orden y en PnL.

Y la síntesis operativa de toda la constitución:

> **Los datos producen snapshots. Las familias consumen trials. Las estrategias producen señales. Los diagnósticos producen transparencia. El allocator asigna riesgo. El executor toca el mercado. Las tablas de hechos son la única verdad. El Passport lo recuerda todo — porque se deriva, no se escribe. Y el único juez limpio es siempre el forward posterior al freeze.**
