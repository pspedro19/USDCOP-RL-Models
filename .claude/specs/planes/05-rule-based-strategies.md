---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - config/assets/pipelines.yaml
  - src/contracts/signal_contract.py
  - airflow/dags/asset_pipeline_factory.py
  - usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx
---

# Estrategias basadas en reglas como componentes de primera clase

> Complemento del *Plan Consolidado Final de `usdcop_trading`*. Define cómo incorporar estrategias que **no** usan ML (reglas y condiciones) sin romper la escalabilidad, cubriendo YAML, Airflow, backend y frontend.
>
> **Invariantes duras promovidas a regla siempre-cargada**: `.claude/rules/strategy-engines.md`.
> **Backlog de implementación**: BL-45 (contrato+registry+factory R1-R3), BL-46 (backend+frontend R4-R5), BL-47 (migración R6-R8).

---

## 0. Principio rector

**Una estrategia es un contrato, no una implementación.** Da igual si el "cerebro" es un PPO, un ridge o un conjunto de reglas `if/then`: todas emiten exactamente el mismo `action.strategy_signal` (dirección, `target_exposure`, TP/HS, `decision_fingerprint`, `health_snapshot_id`). Si eso se cumple, **ejecución, fact, BI y frontend no distinguen** entre ML y reglas — y ahí está la escalabilidad. Lo único que cambia es el bloque que produce el `target_exposure`.

De aquí se deriva la regla final que gobierna todo el documento:

> **Una estrategia de reglas no es un modelo incompleto: es una política determinista completa.** Recibe el mismo gobierno, linaje, métricas, replay, paper, canary, ejecución y Passport que una estrategia ML. Solo cambia el artefacto central.

```text
ML:         modelo entrenado + parámetros
Rule-based: código de política + parámetros + reglas
Composite:  modelo(s) + política de decisión
```

Una estrategia rule-based **no se trata como excepción ni simula tener entrenamiento**. Sigue siendo un *sleeve* completo, gobernado, versionado y auditable. Conserva backtest, gates, señal, paper, juez, promoción, ejecución y retiro. La única diferencia con ML está en **cómo se transforma un snapshot de features en una decisión** — no en el gobierno. La arquitectura ya lo permite:

```yaml
engine:
  type: rule_based
  retrain: never
```

y haciendo que la tarea de entrenamiento (L3) exista **únicamente cuando `retrain != never`**. Añadir la estrategia número 50 debe costar "un YAML + un PR", no un despliegue.

---

## 1. Anatomía de una estrategia basada en reglas

Cinco capas explícitas, nunca mezcladas en una sola tabla:

```text
Datos canónicos → Features deterministas → Política de reglas congelada → Decisión y explicación → Simulación / paper / ejecución
```

Ejemplo MA200:

```text
close oficial diario  →  SMA(close,200)  →  close > MA200  →  target_exposure = 1  →  ejecución next-open
```

- `close` es **dato**.
- `MA200` es **feature determinista**.
- `close > MA200` es una **regla**.
- `target_exposure = 1` es una **decisión**.
- El trade y el PnL son **hechos**.

---

## 2. Taxonomía de motores (un solo registry, con discriminador)

No crear un registry de reglas paralelo al de modelos (ya hay demasiados registries duplicados en la base). Un solo registro canónico que reconoce cuatro tipos de motor:

| Motor | Ejemplo | ¿Entrena? | Cómo decide |
|---|---|---:|---|
| `rule_based` | SPX MA200 | No | Condiciones deterministas |
| `rule_based` | Oro SMA votes | No | Votación de reglas |
| `rule_based` | BTC vol-targeted | No | Fórmulas de riesgo |
| `ml` | Predictor supervisado | Sí | Modelo entrenado |
| `rl` | PPO USD/COP | Sí | Política aprendida |
| `composite` | `smart_simple_v11` | Parcial | Predictor ML + Hurst + sizing + TP/HS |

**USD/COP no es un modelo ML puro**: su decisión económica es una **política compuesta** (`Ridge/BR + gate Hurst + sizing + TP/HS + salida temporal`). El predictor es solo uno de sus componentes.

Registro mínimo:

```text
control.strategy          (strategy_id, kind, owner, status, created_at)
control.strategy_version  (strategy_id, version, spec_yaml, spec_hash, valid_from, valid_until, is_active)
```

Git es la fuente de verdad del spec (extiende el patrón `*_ssot.yaml`: un `strategies/` con un archivo por estrategia). Un job de sync valida con JSON Schema/pydantic en CI y carga a `control.strategy_version`. La versión es **inmutable**: cambiar una regla = nueva versión con nuevo `spec_hash`, que viaja hasta el `strategy_signal` para atribución exacta.

Paridad con ML — tres cosas separadas: **`spec`** (la lógica, versionada), **`config`** (params tuneables por despliegue/usuario, tu `sb_trading_configs`) y **`state`** (runtime). El mismo principio de no meter `normalization_mean/std` en el catálogo de features.

---

## 3. El spec YAML

### 3.1 Ejemplo completo con gobierno (SPX MA200)

```yaml
id: spx500_daily_ma200_v1
asset: spx500
family: trend_regime
research_cluster: trend
surface: action
version: 2.0.0
engine:
  type: rule_based
  implementation:
    mode: coded_policy
    module: strategies.policies.spx:MA200Policy
  retrain: never
inputs:
  feature_set_id: spx500_ma200_action_v1
  resample_policy_id: spx500_daily_official_v1
  required_features: [close, ma_200]
  decision_point: close
  execution_ref: next_open
policy:
  params:
    ma_window: 200
    exposure_when_on: 1.0
    exposure_when_off: 0.0
    exposure_cap: 1.0
  resolution:
    mode: first_match
    default_target_exposure: 0.0
  missing_input_policy: FAIL_CLOSED
  stale_input_policy: FLAT
governance:
  data_cutoff: "2024-12-31"
  frozen_at: "2026-07-27T00:00:00Z"
  code_hash: "sha256:..."
  params_hash: "sha256:..."
  feature_set_hash: "sha256:..."
  policy_hash: "sha256:..."
  ssot_manifest: manifests/spx500_ma200_v1.json
  withdrawal_protocol: docs/withdrawal/spx500_ma200_v1.md
research_state: PAPER
capital_tier: SHADOW
operational_state: NOMINAL
judge:
  anchor: "2026-08-01"
  min_decisions: 26
  criterion: "Calmar >= incumbent"
  alpha: 0.05
  sequential: { method: none }
publish:
  bundles: [summary, trades, signals]
  immutable_key: [strategy_id, version, partition]
capabilities: [backtest, gate, signal, paper, verify]
explainability:
  trace_schema: rule_trace_v1
  reason_codes: [CLOSE_ABOVE_MA200, CLOSE_BELOW_MA200, INPUT_STALE]
```

No incluye `train: true`, `model_id` ni `mlflow_model_version`, porque no existe un modelo entrenado.

### 3.2 Dos formas de implementar la lógica

**A. Política implementada en código** (recomendada para estrategias serias/complejas): TP/HS, estados internos, trailing stops, reglas secuenciales, votaciones, múltiples posiciones, políticas con contexto.

```yaml
engine:
  type: rule_based
  implementation:
    mode: coded_policy
    module: strategies.policies.gold:SMAVotingPolicy
```

La lógica vive en Python versionado (`strategies/policies/gold.py`); el YAML declara params, inputs, feature set, estados, capabilities, juez y fallbacks.

**B. Política declarativa en YAML** (para reglas simples): las condiciones se serializan y las renderiza el frontend.

```yaml
engine:
  type: rule_based
  implementation:
    mode: declarative
    schema: qlab_policy_v1
policy:
  resolution: { mode: first_match, default_target_exposure: 0.0 }
  rules:
    - id: trend_on
      priority: 100
      when: { operator: greater_than, left: feature.close, right: feature.ma_200 }
      output: { direction: LONG, target_exposure: 1.0, reason_code: CLOSE_ABOVE_MA200 }
    - id: trend_off
      priority: 90
      when: { operator: less_or_equal, left: feature.close, right: feature.ma_200 }
      output: { direction: FLAT, target_exposure: 0.0, reason_code: CLOSE_BELOW_MA200 }
```

**Seguridad — no negociable:** las condiciones son un DSL restringido evaluado con un AST allowlist, con operadores conocidos únicamente:

```text
greater_than · less_than · equal · all · any · not · crosses_above · crosses_below · between
```

Prohibido ejecutar código arbitrario procedente del YAML (`when: "eval(close > ma200 and custom_python())"`). Esto da tres cosas que el Python suelto no: es **testeable**, es **seguro** y es **serializable** (se guarda en DB y lo renderiza el frontend). Para indicadores complejos que el DSL no expresa, se usa el mismo *escape hatch* que ya existe en `feature_definitions`: una feature registrada por `code_reference + code_hash`, y la regla solo referencia el `feature_id`. La complejidad vive en la capa de features (compartida), no en cada estrategia.

---

## 4. Contrato común de política (un solo motor de evaluación)

Toda política, sin importar su implementación (código o declarativa), cumple la misma interfaz:

```text
Policy.required_features()
Policy.validate_inputs(snapshot)
Policy.evaluate(snapshot, context) → StrategyDecision
```

El resultado común es el mismo `strategy_decision` para todos los motores:

```yaml
strategy_decision:
  signal_id: uuid
  sleeve_id: spx500_daily_ma200_v1
  strategy_version: 2.0.0
  engine_ref:
    type: rule_based
    policy_version_id: uuid
    policy_hash: sha256:...
  instrument_id: spx500_index
  as_of: "2026-07-27T20:00:00Z"
  valid_from: "2026-07-28T13:30:00Z"
  valid_until: "2026-07-28T20:00:00Z"
  direction: LONG
  target: { type: nav_weight, value: 1.0, currency: USD }
  feature_snapshot_id: uuid
  health_snapshot_id: uuid
  liquidity_snapshot_id: uuid
  decision_fingerprint: sha256:...
  reason_codes: [CLOSE_ABOVE_MA200]
  decision_components: { close: 6412.8, ma_200: 5984.2, distance_to_ma_decimal: 0.0716 }
  rule_trace_uri: "s3://bundles/.../rule_trace.json"
```

El contrato **no** exige `model_snapshot_id` (una política de reglas no tiene pesos entrenados). El `engine_ref` es lo que varía:

```yaml
# rule_based
engine_ref: { type: rule_based, policy_version_id: uuid, policy_hash: sha256:... }
# ml
engine_ref: { type: ml, model_snapshot_id: uuid }
# composite
engine_ref: { type: composite, policy_version_id: uuid, model_snapshot_ids: [uuid] }
```

**Regla de paridad:** el motor que evalúa el spec es **la misma librería** en backtest, paper y live. Airflow (batch) y el loop live importan el mismo `evaluate(snapshot, context)`. Si hubiera dos implementaciones, divergen y el backtest miente. Solo cambia la **fuente** de features (histórica con `available_at` para backtest, streaming para live), nunca la lógica.

---

## 5. Features de estrategias rule-based (deterministas, no aprendidas)

Una estrategia de reglas también usa features; la diferencia es que son deterministas.

| Estrategia | Inputs canónicos | Features |
|---|---|---|
| SPX MA200 | `close` | `ma_200` |
| SPX regime-gated | `close` | MA200, TSMOM 12-1, régimen |
| Oro trend simple | `close` | Varias SMA y votos |
| BTC HODL | `close`, retornos | Volatilidad realizada |
| BTC funding | precio, funding | Funding z-score, basis |
| COP rule gate | score ML, OHLC | Hurst, vol, confidence tier |

**Shared vs policy-local** (para no crear una tabla gigante con todas las features de todas las estrategias):

- **Compartidas** — se calculan una vez y muchas consumen (fan-in), viven en un `feature_set` versionado: `close`, `return_1d`, `realized_vol_21d`, `ma_50`, `ma_200`, `tsmom_12_1`.
- **Locales de una política** — se calculan dentro del DAG de estrategia y no se materializan globalmente: `number_of_sma_votes`, `distance_to_stop`, `current_trailing_state`, `regime_leverage_scaler`. Cuando explican una decisión se guardan en `decision_components JSONB`.

---

## 6. Resampleos: snapshots explícitos, nunca consultas arbitrarias

La política rule-based **no** consulta tablas arbitrarias ni resamplea en silencio. El YAML referencia un feature set y una política de resampleo:

```yaml
inputs:
  feature_set_id: gold_sma_action_v1
  resample_policy_id: gold_daily_official_v1
```

El feature set define origen, grano y disponibilidad point-in-time:

```yaml
source:
  dataset: market.canonical_bar
  instrument_id: xauusd_spot
  interval: P1D
  bar_method: provider_official      # provider_official ≠ resampled
features: [close, sma_fast, sma_medium, sma_slow]
availability:
  decision_cutoff: session_close
  require_available_at: true
```

La estrategia consume el `feature_snapshot_id`; **nunca** hace `SELECT * FROM market_bar ORDER BY time DESC LIMIT 200`. Point-in-time también aplica a reglas: `rsi_9 < 30` en backtest debe leer features con `available_at ≤ as_of`, y el `bundle_health` bloquea o marca `degraded` si falta una serie requerida (no inventa).

> Regla institucional: una estrategia no consume "los últimos datos"; consume un snapshot explícito con cutoff, resampleo y hash conocidos.

---

## 7. Airflow: un factory parametrizado, no un DAG por estrategia

El anti-patrón que mata la escalabilidad es un DAG por estrategia. Lo correcto es un factory que hace fan-out dinámico (dynamic task mapping, `.expand()`), con código O(1) aunque haya 5 o 200 estrategias.

### 7.1 DAG de datos (por activo)

```text
asset__spx500__data
  ingest → canonicalize → quality → official_daily_snapshot → shared_feature_snapshots → data_verify
publica: asset://spx500/features/spx500_ma200_action_v1
```

### 7.2 DAG de estrategia rule-based

```text
strat__spx500_daily_ma200_v1
  validate_strategy_spec → resolve_feature_snapshot → validate_policy_inputs
  → evaluate_policy → publish_strategy_signal → paper_simulation
  → compute_metrics → verify → track_judge
```

No genera `l3_train`, `fit_scaler`, `register_model` ni `promote_model` — el factory omite el entrenamiento cuando `engine.retrain == never`.

### 7.3 El factory no tiene código especial por estrategia

```python
if spec.engine.type == "rule_based":
    add("resolve_feature_snapshot"); add("evaluate_policy")
elif spec.engine.type == "ml":
    if retrain_is_due(spec): add("train"); add("register_model")
    add("predict"); add("apply_decision_policy")
elif spec.engine.type == "rl":
    if retrain_is_due(spec): add("train_rl")
    add("infer_policy")
elif spec.engine.type == "composite":
    if retrain_is_due(spec): add("train_components")
    add("run_components"); add("evaluate_policy")

# todas convergen:
# strategy_decision → simulate/signal → metrics → gates → bundles → verify
```

Se guía por `engine.type`, `engine.retrain`, `capabilities`, `research_state`, `capital_tier`, `operational_state`, `run_mode` — **nunca** por `if strategy_id == "spx500_daily_ma200_v1"`.

### 7.4 Modos de corrida (mismo DAG)

| Modo | Qué hace |
|---|---|
| `DECISION` | Con cada snapshot nuevo: resolve → evaluate → signal → verify |
| `FREEZE` | Al crear versión: full replay → baselines → gates → policy manifest → bundles inmutables |
| `REVALIDATE` | Semanal/por evento: paridad → data quality → forward metrics → juez |
| `BACKFILL` | Reconstruye particiones históricas **sin publicar órdenes** |

Esto evita correr un backtest completo todos los días para una política MA200.

---

## 8. Backend y base de datos

Una estrategia rule-based **no necesita tablas nuevas propias**; usa las comunes:

```text
control.strategy_projection · control.policy_version · control.strategy_run
feature.snapshot · action.strategy_signal · control.metric_event
fact.position · fact.pnl · control.lineage_node · control.lineage_edge
```

`control.policy_version`:

```text
policy_version_id · sleeve_id · strategy_version · engine_type · implementation_mode
module_reference · code_hash · params_hash · policy_hash · feature_set_hash
resample_policy_hash · frozen_at · manifest_uri · schema_version
```

`action.strategy_signal` (campos comunes, estables y normalizados):

```text
signal_id · sleeve_id · strategy_version · policy_version_id · instrument_id
as_of · valid_from · valid_until · direction · target_exposure
feature_snapshot_id · decision_fingerprint · reason_codes · decision_components · rule_trace_uri · created_at
```

La política **no crea columnas nuevas** al añadir una regla. Campos como `hurst_exponent`, `sma_vote_count`, `trend_regime`, `effective_stop` van dentro de `decision_components` (con `decision_schema_version`). Solo lo necesario para riesgo, filtrado o ejecución permanece normalizado.

El backend expone el contrato, no estrategias concretas:

- `GET /strategies`, `/strategies/{id}/versions` → leen `control.strategy*`.
- Librería de evaluación **empaquetada como módulo** (la misma de Airflow), no reescrita.
- `POST /strategies/validate` → el frontend valida el spec antes de guardar.

Añadir estrategias no toca el backend.

---

## 9. Frontend: dirigido por el spec (schema-driven)

El frontend **no** necesita una página por motor ni tarjetas hardcodeadas: se construye desde el spec + su metadata. Presenta hechos producidos por el backend y **no recalcula** decisiones, métricas ni gates.

**Vista común de estrategia** (siempre): Activo · Estrategia · Versión · Engine type · Estado de investigación · Capital tier · Estado operacional · Última señal · Exposición objetivo · PnL · Drawdown · Juez · Frescura de datos · Fingerprint.

**Panel específico cuando `engine.type = rule_based`**: Regla evaluada · Valor observado · Umbral · Resultado · Regla ganadora · Fallback aplicado · Reason codes · Policy hash.

| Condición | Observado | Umbral | Resultado |
|---|---:|---:|---|
| Close > MA200 | 6.412 | 5.984 | PASS |
| Datos vigentes | 4 min | ≤1 día | PASS |
| Volatilidad bajo cap | 13,2% | ≤25% | PASS |

El backend entrega el trace y React solo lo renderiza (no vuelve a evaluar `close > ma200`):

```json
{
  "trace_schema": "rule_trace_v1",
  "rules": [
    { "rule_id": "close_above_ma200", "label": "Precio sobre MA200",
      "observed": { "close": 6412.8, "ma_200": 5984.2 },
      "result": true, "reason_code": "CLOSE_ABOVE_MA200" }
  ]
}
```

Un solo renderer con variantes por motor:

| Motor | Componente explicativo |
|---|---|
| Rule-based | `RuleTracePanel` |
| ML | `MLExplanationPanel` (modelo, calibración, contribuciones) |
| RL | `RLPolicyPanel` (estado, acción, policy version) |
| Composite | `CompositeDecisionPanel` (predictor + gates + sizing) |

Las páginas de replay, producción y ejecución son las mismas para todos. Los params tuneables (`sb_trading_configs`) se renderizan desde el schema de config, no desde campos fijos.

### 9.1 Metadatos de presentación (no ejecutables)

```yaml
presentation:
  engine_label: "Reglas MA200"
  description: "Exposición cuando el cierre supera la media de 200 sesiones"
  components:
    - { key: close, label: "Cierre", format: price }
    - { key: ma_200, label: "Media móvil 200", format: price }
    - { key: distance_to_ma_decimal, label: "Distancia a la media", format: percent }
```

Esta sección no decide, no se ejecuta, no cambia el `policy_hash`, tiene su propio `presentation_hash`, y permite que el frontend sea genérico. Cambiar una etiqueta **no** crea una nueva versión económica de la estrategia.

---

## 10. Gobierno y trials (las reglas también se sobreajustan)

Cada variante observada **cobra trial**:

```text
MA100 · MA150 · MA200 · MA250 · close>MA200 · close>MA200+VIX gate · close>MA200+TSMOM
```

También cobran trial: cambiar un umbral tras ver resultados, cambiar una ventana, añadir un filtro, cambiar la salida, cambiar el sizing, escoger la mejor combinación de reglas.

**No** cobra trial: reejecutar una política congelada, actualizar el forward, publicar un nuevo período, recalcular una métrica predeclarada, o corregir un bug que restaure la semántica firmada (documentándolo).

Toda estrategia rule-based mantiene: `family · trial_id · code_hash · params_hash · feature_set_hash · policy_hash · data cutoff · juez · withdrawal protocol`.

---

## 11. Validaciones CI específicas

```text
✓ rule_based implica retrain=never
✓ rule_based no declara capability=train
✓ rule_based no exige model_snapshot_id
✓ toda policy referencia feature_set y resample_policy
✓ todos los required_features existen en el snapshot
✓ ningún input supera decision_cutoff
✓ todo operador declarativo pertenece al whitelist
✓ YAML no contiene eval, SQL libre o Python arbitrario
✓ toda política tiene default/fallback explícito
✓ conflictos entre reglas tienen prioridad o resolución declarada
✓ toda salida respeta direction y exposure caps
✓ mismos inputs + misma policy producen misma decisión (determinismo)
✓ rule_trace contiene todas las condiciones relevantes
✓ policy_hash coincide con código, parámetros y schema
✓ cambiar ventana, threshold o regla exige nueva versión
✓ frontend no recalcula condiciones
✓ WITHDRAWN conserva bundles y manifiesto
```

---

## 12. Aplicación a tus estrategias

| Estrategia | Tipo | Tarea L3 (train) | Explicación frontend |
|---|---|---|---|
| SPX MA200 | `rule_based` | No | Precio vs MA200 |
| SPX regime-gated | `rule_based` | No | MA200, TSMOM y cap |
| Oro SMA votes | `rule_based` | No | Votos SMA y exposición |
| BTC HODL vol-targeted | `rule_based` | No | Vol realizada, target y cap |
| BTC funding/basis | `rule_based` | No | Funding, z-score, coverage gates |
| USD/COP v11 | `composite` | Sí (Ridge/BR) | Predictor, Hurst, sizing y salidas |
| USD/COP v12/v14 | `composite` | Comparte predictor | Misma señal, distinta política de riesgo |
| PPO USD/COP | `rl` | Sí | Estado, acción y policy artifact |

---

## 13. Plan de implementación

| Fase | Entregable |
|---|---|
| **R1 — Contrato de políticas** | `contracts/policy.py`, `contracts/strategy_decision.py`, `contracts/rule_trace.py` |
| **R2 — Registry** | Extender `config/strategies/*.yaml` con `engine.type`, `implementation.mode`, `feature_set_id`, `resample_policy_id`, `policy`, `fallback`, `explainability` |
| **R3 — Factory** | Tareas genéricas `resolve_feature_snapshot`, `validate_policy_inputs`, `evaluate_policy`, `publish_strategy_signal`; omitir `train` automáticamente |
| **R4 — Backend** | `control.policy_version`, `action.strategy_signal`, `rule_trace_uri`, `decision_components` |
| **R5 — Frontend** | Renderer único `StrategyEngineExplanation` con variantes `RuleTracePanel` / `MLExplanationPanel` / `RLPolicyPanel` / `CompositeDecisionPanel` |
| **R6 — Primera migración** | `spx500_daily_ma200_v1` (la más simple). Comparar señales/trades/PnL/bundles/semantic_hash legacy vs nuevo |
| **R7 — Oro y BTC** | `xauusd_trend_simple_v1`, `btcusdt_hodl_b1` |
| **R8 — USD/COP** | Migrar al final como `composite`, conservando L3 y dejando L7 (ejecución) para la última etapa |

---

## 14. El resumen mental

```text
Git (YAML spec, SSOT) → CI valida (JSON Schema) → control.strategy_version
                                                        │
                  ┌──────────────────────────────────────┤
            feature layer (compartida, feature_id)        │
                  │                                        │
      motor de evaluación único (backtest = paper = live) │
                  │                                        │
      action.strategy_signal ←──────────────────────────── ┘
                  │
      execution.* → fact.* → BI (vistas) → frontend (schema-driven)
```

Lo que mantiene escalable el sistema no es una tecnología concreta: es que **reglas y ML comparten el mismo contrato de features, el mismo registro, el mismo motor de evaluación con paridad backtest/live, el mismo `strategy_signal` y el mismo frontend dirigido por spec**. Cada estrategia declara su engine, consume un feature snapshot, produce el mismo `strategy_decision` y deja una explicación estructurada que todo el sistema sabe procesar. Así se incorporan cien políticas sin crear cien diseños de backend o frontend.

---

## 15. Decisiones abiertas (para afinar)

1. **DSL de predicados propio vs safe-eval de expresiones.** Un DSL propio (operadores whitelist, AST) da más control, seguridad y serialización, con más trabajo inicial; un safe-eval arranca más rápido pero es más difícil de auditar y renderizar. Recomendación: DSL propio para el modo `declarative` y `coded_policy` para todo lo complejo.
2. **¿Reglas con estado entre barras?** Si necesitas condiciones tipo "3 cierres seguidos sobre la MA" o trailing stops, el motor requiere una **capa de estado por estrategia** (`current_trailing_state`, contadores) que persiste entre evaluaciones. Si todas las reglas son sin memoria, el motor puede ser puramente funcional (más simple de paralelizar y testear). Definir esto decide la firma de `Policy.evaluate(snapshot, context)` y si `context` incluye estado previo.

---

## 16. Nota de reconciliación con el as-built (2026-07-27)

Contraste contra el repo real: **§7 propone factory con dynamic task mapping y modos
DECISION/FREEZE/REVALIDATE/BACKFILL** — el factory actual (asset_pipeline_factory) es
1-DAG-por-activo con stages secuenciales; la 004 §13 propone 1-DAG-por-sleeve. Las tres
posturas convergen en BL-28 (factories nuevas) y la decisión fina (per-sleeve vs
task-mapping) se toma allí con el criterio de aislamiento de fallas de la 004.
El caso `gold_dynamic_exit` (stateful, publisher propio porque "no cabe en el loop
direction_fn") es el ejemplo vivo de la decisión abierta §15.2: el motor de políticas
DEBE soportar estado o esa clase de estrategia queda fuera del contrato.
