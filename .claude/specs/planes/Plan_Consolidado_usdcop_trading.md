---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - config/assets/pipelines.yaml
  - services/signalbridge_api
---

# Plan Consolidado Final — Consolidación de `usdcop_trading`

**Base:** PostgreSQL + TimescaleDB · **Perfilado:** 2026-07-27 · 59 tablas de dominio · 964 columnas · 2.82M filas · 7 hypertables

Este documento fusiona las dos auditorías (la descriptiva y el dictamen de arquitectura) en **un solo plan ejecutable**, y añade lo que faltaba: **cómo cada estrategia usa columnas, features y resampleos**, que es lo que decide qué tablas son autoritativas y cuáles son proyecciones.

---

## 0. Principio rector

No se rediseña desde cero ni se cambia de motor. Se hace una **consolidación controlada**: una fuente autoritativa por concepto, identidades canónicas, point-in-time verificable, ejecución event-sourced, proyecciones regenerables y secretos aislados. El volumen físico no es el problema (Timescale aguanta esto de sobra); el problema es la **complejidad semántica** — 27 tablas vacías (45.8%) que son andamiaje y varios subsistemas paralelos del mismo concepto.

Evaluación de partida:

| Dimensión | Estado |
|---|---:|
| Perfilado y observabilidad | 9/10 |
| Estructura de datos de mercado | 7/10 |
| Point-in-time / anti-leakage | 6/10 |
| MLOps | 4/10 |
| Ejecución y OMS | 3/10 |
| Seguridad de datos sensibles | 3/10 |
| Preparación para escalar | 6/10 |

Decisión global: **conservar ~25–30 tablas autoritativas, transformar ~10–15 en vistas/proyecciones, retirar ~15–20 heredadas o especulativas.**

---

## 1. Cómo maneja cada estrategia columnas, features y resampleos

Esto es la parte central y lo que gobierna todo lo demás. Hoy conviven **cuatro familias de estrategia con necesidades de datos muy distintas**, más una capa de distribución a usuarios. El error estructural actual es que todas leen de tablas con nombres/tipos/vocabularios inconsistentes y comparten un catálogo de features (`config.feature_definitions`) que mezcla definición, normalización y código.

### 1.1 Inventario real de estrategias (según los datos del perfil)

| Estrategia | Motor / modelo | Activo | Grano base | Resampleo / lookback | Fuentes de columnas | Cadencia | Estado |
|---|---|---|---|---|---|---|---|
| **PPO intradía USDCOP** (`ppo_primary` V20) | RL — PPO (stable-baselines3), observación de **15 dimensiones** | USD/COP | Barra **5m** (`usdcop_m5_ohlcv`) | Lookbacks intradía sobre 5m (p. ej. `log_ret_4h` = `LAG(close,48)` = 48×5m); macro **diario** unido as-of a la barra | `usdcop_m5_ohlcv.close/high/low` + `macro_indicators_daily` (dxy, vix, embi, brent, treasury_10y, usdmxn) + temporales + estado del entorno | Cada barra 5m (tiempo real) | **Vivo** (`trading_state.model_id = ppo_v1`) |
| **Forecast H5 (consenso)** | Ensemble supervisado — hoy `ridge` + `bayesian_ridge`; catálogo de 9 (`bi.dim_models`: linear/boosting/hybrid) | XAU/USD principal (base_price ≈ 3600–3757 = oro); extensible | **Diario** → horizonte 5 días | Sin resampleo intradía; features diarias + horizontes 1/5/10/15/20/25/30d (`bi.dim_horizons`) | `asset_daily_ohlcv` / `asset_native_ohlcv` + macro | **Semanal** (`inference_week`) | Activo con pocos datos (18 pred / 10 señales) |
| **SPX MA200** (`spx500_ma200`) | Regla de tendencia (media móvil 200) | SPX500 / SPY | **Diario** (`asset_daily_ohlcv`) | MA de 200 barras diarias | Solo precio; **cero series macro** (bundle sin requeridos) | Diaria | Definida por bundle |
| **Exposición BTC (ciclo/derivados)** | Señal de exposición por régimen | BTC/USDT | **Diario** | Z-scores y basis anualizada; on-chain planeado | `crypto_derivatives_daily` (funding_rate, funding_zscore, basis_annualized) + `crypto_onchain_daily`/`crypto_flows_daily` (planeadas) | Diaria | Señales vacías (`crypto_exposure_signals` = 0) |
| **Capa SaaS / copy (`sb_*`)** | No es estrategia: distribución a usuarios | Config por usuario | — | — | `sb_trading_configs` (max_position_size, stop_loss_percent, take_profit_percent, allowed_symbols, max_daily_trades) → órdenes por usuario | Por evento | 25 usuarios/configs |

**Lectura clave:** las estrategias no son intercambiables en datos. La PPO necesita el grano fino 5m + macro as-of; H5/SPX/BTC son diarias. Por eso la capa de features debe ser **por estrategia-versión (feature set versionado)**, no un catálogo global plano.

### 1.2 El catálogo de features hoy (`config.feature_definitions`, 30 features)

Distribución real por grupo: `state` (11, estado del entorno RL), `macro_zscore` (3), `macro_changes` (3), `price_returns` (3), `volatility` (2), `temporal` (2), y singles de `regime`, `trend`, `momentum`, `macro_momentum`, `macro_volatility`, `macro_derived`.

- **Origen mixto:** `source_table` = `usdcop_m5_ohlcv` (9) y `macro_indicators_daily` (9); 12 son *runtime* (estado del entorno, sin tabla).
- **Cómputo partido:** `compute_location` = python (17) / sql (13). Ejemplos SQL: `SIN(2*PI()*EXTRACT(HOUR FROM time)/24)`, `LN(close/LAG(close,48) OVER (ORDER BY time))`. Ejemplos Python: `calc_rolling_std(brent, period=5)`, `encode_regime(volatility_percentile)`, `env.current_position`.
- **Normalización peligrosa:** las features `zscore_fixed` traen media/desv **hardcodeadas** en la propia fórmula (p. ej. `(vix - 21.16) / 7.89`) y en `normalization_mean/std` (solo 10 de 30 las tienen). Esas constantes dependen del período de entrenamiento y **no deben vivir en el catálogo de features**.

### 1.3 Los tres problemas transversales de features

1. **Normalización acoplada al catálogo.** `normalization_mean/std` y las constantes en el SQL pertenecen al **artefacto del modelo**, versionadas con `training_cutoff`. Si reentrenas y cambia la media del VIX, hoy tendrías que editar una fila de config → riesgo de leakage y de servir con constantes viejas.
2. **Código como string no autoritativo.** `python_function = 'calc_rolling_std(brent, period=5)'` y `sql_formula` como texto no pueden ser la fuente ejecutable. El código real va en Git; la tabla guarda **referencia + hash** (`code_reference`, `code_hash`).
3. **Fuentes que vamos a renombrar.** `source_table` apunta a `usdcop_m5_ohlcv` y `macro_indicators_daily`, ambas destinadas a cambiar (renombrado / conversión a vista). Hay que apuntar a los nombres canónicos (`market.canonical_bar`, `macro.observation`).

### 1.4 Modelo objetivo de features y resampleo

**Contrato de feature (catálogo, estable):**

```text
feature_id
feature_name
unit
feature_group
causality_policy        -- point_in_time | same_bar | lagged_1
source_contract         -- market.canonical_bar | macro.observation | runtime_state
transformation          -- log_return | zscore | rolling_std | rsi | cyclical | ...
lookback                 -- p. ej. PT4H, 200D
compute_location         -- python | sql  (una sola por feature; evitar mezcla ambigua)
code_reference + code_hash
is_active
```

**Feature set por estrategia-versión (lo que realmente entra al modelo):**

```text
feature_set(strategy_version, feature_id, order, required)
normalization_snapshot_id  -> artefacto en MinIO/MLflow (mean, std, training_cutoff, semantic_hash)
```

**Resampleo canónico (una sola verdad, varias granularidades):**

```text
market.raw_bar            -- lo recibido de cada proveedor (inmutable)
   └── market.canonical_bar   -- serie validada, grano fino 5m por instrumento
          ├── cagg 1h        -- continuous aggregate
          ├── cagg 4h        -- continuous aggregate
          └── cagg 1d        -- continuous aggregate
```

Regla de oro del resampleo: la barra **diaria oficial del proveedor no siempre equivale** al resampleo del intradía. No se mezclan en silencio; se marca el método y se conservan como observaciones distintas cuando difieren:

```text
bar_method = provider_official | resampled
```

**Unión con macro (as-of, point-in-time):** el macro es diario; para la PPO intradía se une a la barra 5m por *as-of join* con `available_at ≤ bar_time` (no un forward-fill ingenuo que introduciría look-ahead). Cada estrategia declara qué series macro requiere vía su bundle (ver §4).

---

## 2. Arquitectura objetivo (esquemas)

```text
PostgreSQL / TimescaleDB
├── reference   -- asset, instrument, provider, provider_symbol, calendar, bar_interval
├── market      -- raw_bar, canonical_bar, ingestion_run, quality_event  (+ caggs 1h/4h/1d)
├── macro       -- series, observation (PIT), source_document, bundle_health
├── forecast    -- forecast_spec, forecast_output, forecast_score
├── action      -- strategy_signal
├── execution   -- order, order_status_event, fill_event, reconciliation_event
├── fact        -- position, pnl
├── control     -- family, trial, strategy, run, metric_event, lineage_node/edge, incident, vote_event
├── news        -- articles, sources, ingestion_log, daily_digests, feature_snapshots, keywords
├── auth        -- users, configs
└── secret      -- external_account (solo referencia, el secreto vive en Vault)
```

Fuera de PostgreSQL: **Git** (declaraciones/config), **MLflow** (runs y modelos), **MinIO** (snapshots, bundles, artefactos, informes), **Secret manager** (credenciales), **frontend/Grafana** (solo vistas y APIs).

`public` debe quedar casi vacío y con `CREATE` revocado para usuarios de aplicación.

---

## 3. Identidades canónicas (bloquea el escalado si no se resuelve)

Hoy el mismo activo aparece como `usdcop` / `USD/COP`, `btcusdt` / `BTC/USDT`, `spx500` / `SPX500` / `SPX/500` / `SPY`, y los timeframes como `5m` / `5min`, `1d` / `1day`. Además `market_ingestion_manifest.asset_id = 'BTC/USDT'` pero `dim_asset.asset_id = 'btcusdt'`. Eso rompe joins.

Identidad objetivo (extiende el `dim_asset` que ya existe, 4 filas: xauusd/spx500/btcusdt/usdcop):

```text
reference.asset        asset_id=btc
reference.instrument   instrument_id=binance_btcusdt_perpetual   -- SPX≠SPY≠ES son instrumentos distintos
reference.provider     provider_id=binance
reference.provider_symbol  (provider_id, provider_symbol) -> instrument_id
reference.bar_interval bar_interval=PT5M   (enum/FK; elimina '5m' vs '5min')
reference.calendar     usa dim_asset.calendar_kind (nyse | metals_23h | colombia | utc_24_7)
```

Acción: `market_ingestion_manifest.asset_id` → se descompone en `instrument_id + provider_id + provider_symbol`; toda tabla de mercado referencia `instrument_id` por FK, no texto libre; `CHECK`/enum para timeframes.

---

## 4. Point-in-time y anti-leakage

El intento PIT es bueno pero **inconsistente**, y esto afecta directamente qué features puede usar cada estrategia.

- **OHLCV:** `available_at` vacío en 6.4% de `usdcop_m5_ohlcv` y 19.9% de `asset_daily_ohlcv`; en `asset_native_ohlcv` el rango de `available_at` arranca en 2026 aunque el precio arranca en 1971 → `available_at` está representando la ingesta/reconstrucción, no la disponibilidad histórica real. Separar y declarar política:

  ```text
  event_time · provider_published_at · available_at · retrieved_at · ingested_at
  available_at(OHLCV) = cierre_de_barra + latencia_conservadora_declarada
  ```

- **Macro mensual/trimestral:** `publication_date` está **100% NULL** en `macro_indicators_monthly` y `macro_indicators_quarterly`. Hallazgo crítico: **esas tablas no deben alimentar backtests promotion-eligible** hasta completar la semántica de rezago de publicación.

- **`macro_indicators_pit` (la joya):** guarda `observation_date`, `release_date`, `available_at`, fuente, hash, política y metadata — pero solo 19.601/216.501 (9.1%) son `pit_vintage`/`promotion_eligible`; el resto son reconstrucciones research-only. No describir toda la tabla como "vintage real". Clasificar:

  ```text
  availability_quality ∈ {
    OFFICIAL_VINTAGE, OFFICIAL_RELEASE_TIMESTAMP,
    CONSERVATIVE_RECONSTRUCTION, RESEARCH_ONLY_RECONSTRUCTION, UNKNOWN }

  Promotion/champion: solo OFFICIAL_VINTAGE (o política aprobada explícita)
  Research exploratorio: acepta reconstructed con caveat visible
  ```

- **Completitud por bundle, no global.** `is_complete` (true en 41.4% daily / 32.6% monthly) no sirve: una fecha no necesita todas las variables históricas. Se evalúa contra el bundle que cada estrategia requiere:

  ```text
  usdcop_smart_simple_v11 -> requiere DXY, WTI, VIX, EMBI
  spx500_ma200            -> no requiere macro
  bundle_health(bundle_id, as_of, required_series, missing_series, stale_series, status)
  ```

  `ffill_count` está siempre en 0 en las tres macro: o se implementa de verdad o se elimina del modelo de servicio.

---

## 5. Calidad de datos: cuarentena, no parches

Anomalías que deben ir a cuarentena (no corregirse a mano en la tabla canónica):

- `USD/MXN`: mediana 19.65, **máx 175 814**. `USD/CLP`: mediana 870, **máx 94 890**. Órdenes de magnitud imposibles → parsing de miles/decimales o columna equivocada.

Pipeline:

```text
raw value → quality rule FAIL → quarantine → provider comparison → correction event → canonical value
```

```yaml
usdmxn: { valid_range: [5, 100],  max_daily_return: 0.20 }
usdclp: { valid_range: [100, 5000], max_daily_return: 0.20 }   # amplios y versionados, no para recortar movimientos reales
```

Columnas fantasma (declarar `feature_status = UNAVAILABLE` en vez de publicar ceros/nulos que parezcan medición):

- `macro_banrep_forwards_monthly.forward_rate` → 100% NULL
- `crypto_derivatives_daily.liquidations_usd` → 100% NULL; `open_interest`/`long_short_ratio` → 44/2506
- `news_articles.sentiment_score` = 0 y `sentiment_label` = neutral para **todos** (el motor no corre); además `content`, `subcategory`, `gdelt_tone`, `entities`, `image_url`, `author` 100% NULL

---

## 6. Consolidación por dominio (tabla actual → destino)

### Mercado (OHLCV)
| Actual | Destino |
|---|---|
| `usdcop_m5_ohlcv` (nombre engañoso; solo 4.5% es COP) | **Renombrar** → `market.canonical_bar` intervalo 5m |
| `asset_native_ohlcv` | Barras crudas/nativas → `market.raw_bar` |
| `asset_daily_ohlcv` | **Vista / continuous aggregate** diaria |
| `market_ingestion_manifest` | Mantener como ledger de ingesta |
| `dim_asset` | Promover a `reference.asset/instrument` |

### Macro
| Actual | Destino |
|---|---|
| `macro_indicators_pit` | **Fuente canónica** → `macro.observation` (con `availability_quality`) |
| `macro_indicators_daily/monthly/quarterly` (ancho) | **Vistas / materialized views / bundle snapshots** sobre el PIT |
| `macro_variable_snapshots` (0, indicadores técnicos) | Decidir en plazo o retirar |
| `macro_banrep_forwards_monthly`, `macro_remesas_monthly` | Mantener (limpiar `forward_rate`) |

### Forecast / acción / ejecución (hoy todo bajo `forecast_h5_*`)
Separar superficies que hoy están mezcladas:
```text
forecast.forecast_output   <- Ridge/BR predice retorno/precio
action.strategy_signal     <- gate + sizing + TP/HS = decisión
execution.order / order_status_event / fill_event  <- replay|paper|canary|live
fact.position / fact.pnl   <- derivados de fills
```
| Actual | Destino |
|---|---|
| `forecast_h5_predictions` | `forecast.forecast_output` |
| `forecast_h5_signals` | `action.strategy_signal` (normalizar; ver §7) |
| `forecast_h5_executions` | `execution.*` (no es forecasting) |
| `forecast_h5_paper_trading`, `forecast_h5_subtrades` | Fusionar en el ledger event-sourced |

### MLOps (no duplicar MLflow)
| Actual | Decisión |
|---|---|
| `experiment_runs` | Eliminar o proyección mínima de MLflow |
| `model_registry` (0, 38 col) | Eliminar como registry paralelo |
| `experiment_comparisons` | Reemplazar por judges/metric events |
| `experiment_deployments` | Reemplazar por strategy transitions/deployment events |
| `metrics.model_performance` | Reemplazar por `control.metric_event` |
| Registros de modelo (`config.models`, `bi.dim_models`, `model_registry`) | **Un solo canónico**; el resto vistas/FK |

### BI (no materializar prematuramente — todas vacías)
`bi.fact_forecasts/consensus/model_metrics/inference_runs` → **vistas** `bi.v_*` sobre canónicas; materializar solo cuando la query sea lenta. Cadena correcta: `forecast.forecast_output → vista BI → endpoint`. Una verdad, varias proyecciones.

### Ejecución / OMS (mayor redundancia del sistema)
```text
execution.order / order_status_event / fill_event / fill_correction_event / reconciliation_event
fact.position / fact.pnl
control.portfolio_target / pretrade_decision / kill_switch_event
```
| Actual | Destino |
|---|---|
| `signals` | `action.strategy_signal` |
| `sb_signals` | Eliminar tras migrar |
| `sb_executions` | `execution.order` + status + fills |
| `trades_history` | Vista derivada de fills |
| `trading.model_trades` | Eliminar o vista |
| `trading_state` (1 fila, `ppo_v1`) | Proyección/caché live, **no SSOT**; si contradice fills+broker, pierde |
| `equity_snapshots` | Derivar de `fact.position/pnl` |

### Noticias (conservar núcleo, retirar campos fantasma)
Mantener `news_articles/sources/ingestion_log/daily_digests/feature_snapshots/keywords`. Elegir **una** autoridad: PostgreSQL (metadata + features) + MinIO (informes completos) + endpoint → entonces `daily_analysis`/`weekly_analysis` se **eliminan**. Lo que no se puede es mantener a la vez tabla-futura-vacía + archivo + JSON frontend + Parquet sin autoridad definida.

---

## 7. Normalización de `action.strategy_signal` (acumulación de versiones)

`forecast_h5_signals` tiene 32 columnas y muchas con ~80% NULL porque cada versión de estrategia añade columnas (hurst, regime, `rolling_wr_8w`, escaladores solo en las últimas 2 filas). No escala a cientos de estrategias. Modelo:

```text
-- normalizado (estable, filtrable, indexable)
signal_id, sleeve_id, strategy_version, instrument_id, as_of,
valid_from, valid_until, direction, target_exposure,
decision_fingerprint, health_snapshot_id

-- componentes propios de la política, versionados
decision_components JSONB   -- {hurst, regime, confidence_tier, take_profit, hard_stop, ...}
decision_schema_version
```

**Unidades de retorno (inconsistencia real):** `forecast_h5_predictions.predicted_return_pct = 1.6481` (puntos %) vs `forecast_h5_signals.ensemble_return = 0.01606` (decimal) representan el mismo 1.606%. Regla: **en la base todo retorno es decimal** (`0.01 = 1%`); el sufijo `_pct` para un decimal induce error → usar `return_decimal`, `drawdown_decimal`, `leverage_ratio`, `price`, `notional_amount`, y meter la unidad en el catálogo. Formateo a % solo en frontend.

**Modelo sintético (riesgo):** `config.models.investor_demo` (algorithm `SYNTHETIC`, "NOT for live trading") aparece con `status=active`. Corregir: `environment=demo`, `surface=synthetic`, `execution_eligible=false`, moverlo a `demo.synthetic_models`, y CI que **rechace** `algorithm=SYNTHETIC AND environment!=demo`. Una demo sintética nunca debe aparecer en métricas de performance real.

---

## 8. Seguridad (P0)

- **Dos tablas de credenciales** (`sb_exchange_credentials`, `user_exchange_keys`) en `public`, junto a datos/research/news/BI. Consolidar en una y mover a `secret.external_account` que guarda **solo referencia**; el secreto vive en Vault/AWS/Azure. Mientras no haya secret manager: esquema separado, claves de cifrado fuera de PostgreSQL, roles propios, sin acceso desde Airflow genérico ni frontend, rotación, log de uso, backup cifrado aparte.
- **Timestamps sin zona:** 15 columnas del módulo `sb_*` usan `timestamp without time zone` → migrar a `timestamptz`.
- **Aislar dominios:** `auth.* / secret.* / execution.* / control.*` separados de mercado/research. Antes de crecer en capital live, `execution_db` con recursos y usuarios independientes.

---

## 9. Tipos numéricos

| Uso | Tipo objetivo |
|---|---|
| Datos de mercado y features | `double precision` (suficiente, más eficiente en cálculo) |
| Dinero / contabilidad / ejecución (cash, fees, NAV, cantidades, precio oficial de fill) | `numeric(precision, scale)` |
| Identidades / estados | `UUID` / `text` / enum con `CHECK` |

Corregir en particular: `usdcop_m5_ohlcv.volume` es `bigint` (pierde decimales de volumen cripto) y los OHLC mezclan `numeric`/`double` entre las tres tablas. Homogeneizar dentro del modelo canónico. No usar `NUMERIC` indiscriminado para millones de barras sin requisito de precisión decimal exacta.

---

## 10. TimescaleDB — confirmar y aplicar políticas

Con 2.2M barras 5m no hay urgencia física, pero el perfil no permite ver chunk interval, compresión, segmentación, retención, índices, continuous aggregates, tamaño físico ni bloat. Política conceptual:

```text
Mercado:  chunk by event_time · segment/compress by instrument_id · order by event_time
Histórico frío: MinIO/Parquet inmutable
Serving: TimescaleDB
```

No aplicar retención destructiva hasta confirmar que el raw inmutable está respaldado y el snapshot es reconstruible. **No** crear hypertables para tablas pequeñas/estáticas ni hypertables vacías (señales, flows, on-chain) — es prematuro. Añadir al próximo perfil: tamaño físico por tabla/índice, última escritura/lectura, PK/unique/FK, índices, chunk interval, compression/retention policy, duplicados por clave, gaps temporales, dependencias vista↔tabla y clasificación `SSOT | projection | cache | deprecated`.

---

## 11. Regla para las 27 tablas vacías

> Una tabla vacía permanece en producción **solo** si tiene productor, consumidor, owner, contrato, fecha de activación y test automatizado.

- **Mantener como contrato pendiente** (mover a `staging_contract.*`): `crypto_onchain_daily`, `crypto_flows_daily`, `crypto_event_calendar` (fuentes forward-only para reabrir familias BTC).
- **Eliminar / reemplazar:** `experiment_runs/comparisons/deployments`, `model_registry`, `metrics.model_performance`, `bi.fact_*`, `signals`/`sb_signals` duplicadas, `user_exchange_keys` duplicada, `trades_history`, `trading.model_trades`.
- **Decidir en plazo concreto:** `daily_analysis`, `weekly_analysis`, `macro_variable_snapshots`, `news_cross_references`, `inference_features_5m`, `equity_snapshots`. Sin productor+consumidor en el roadmap aprobado → retirar (el DDL queda en Git y se recrea cuando exista el caso).

---

## 12. Orden de ejecución

### P0 — antes de operar en live
1. Consolidar las dos tablas de credenciales y sacarlas de `public` (a `secret.*`).
2. Migrar timestamps `sb_*` sin zona → `timestamptz`.
3. Normalizar identidades: instrumentos, símbolos y timeframes (FK a `reference.*`).
4. Renombrar `usdcop_m5_ohlcv` → `market.canonical_bar` (5m).
5. Cuarentena de anomalías USD/MXN y USD/CLP + reglas de rango versionadas.
6. Impedir promoción cuando falta `available_at`.
7. Resolver `publication_date` mensual/trimestral (o bloquear esas tablas para promotion).
8. Separar forecasting / señales / ejecución.
9. Unificar unidades de retorno a decimal.
10. Aislar el modelo sintético de demostración (CI que lo bloquee en no-demo).

### P1 — consolidación
1. `macro_indicators_pit` como fuente canónica (`macro.observation`) con `availability_quality`.
2. Tablas macro anchas → vistas/materialized views.
3. Migrar H5 a event sourcing común (`execution.*` + `fact.*`).
4. Eliminar registries duplicados de MLflow; un solo registro de modelos canónico.
5. BI → vistas `bi.v_*`.
6. Normalizar `action.strategy_signal` (estable + `decision_components` JSONB).
7. Clasificar las 27 tablas vacías; crear la **matriz de fuente de verdad**.

### P2 — escalabilidad
1. Revisar chunks, índices y compresión; continuous aggregates 1h/4h/1d.
2. Enriquecer el perfil (tamaños físicos, PK/FK, políticas, gaps, clasificación).
3. Separar físicamente `execution` cuando aumente el capital.
4. Incorporar spine temporal y fingerprints en todas las tablas activas.
5. Migrar la normalización de features al artefacto del modelo (`normalization_snapshot_id`, `training_cutoff`, `semantic_hash`).

---

## 13. Conclusión

El núcleo es valioso y bastante avanzado: buen histórico OHLCV, manifiestos de ingesta con hashes, calendarios, macro con PIT serio, derivados BTC, noticias y auditoría. Alrededor hay un **segundo sistema** de tablas futuras, prototipos y duplicidades. La prioridad no es añadir otra base de datos, sino convertir PostgreSQL/TimescaleDB en una plataforma con **una fuente autoritativa por concepto, identidades canónicas, event sourcing en ejecución, point-in-time verificable, proyecciones regenerables y secretos aislados**. Se llega a arquitectura institucional **sin reemplazar el motor**: reduciendo autoridades duplicadas y endureciendo los contratos — y, para las estrategias, moviendo la lógica de features/normalización/resampleo a contratos versionados por estrategia en lugar de un catálogo global acoplado.
