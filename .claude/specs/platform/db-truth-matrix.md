---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - scripts/diagnostics/db_inventory_matrix.py
  - .claude/generated/db-inventory.json
  - data/backups/features/feature_backup_manifest.json
  - database/migrations/073_market_quality.sql
  - init-scripts/27-forward-macro-pit.sql
---

# CTR-DB-TRUTH-MATRIX-001 — Matriz de verdad del inventario DB (BL-36)

> Aplicación de FABRIC §7 (fuente de verdad) y §31 ("ningún atributo con dos escritores")
> al inventario heredado. **Este documento PROPONE y EVIDENCIA. No decide, no ejecuta, no
> emite DDL.** Las decisiones de retiro/consolidación/migración son del operador y están
> aisladas en §10.

---

## 0. Qué es esta matriz y qué NO puede ser

### 0.1 Lo que NO se pudo verificar (declaración de honestidad)

**No hubo acceso a la base de datos.** El engine de Docker no estaba corriendo el
2026-07-28 (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file
specified`; puertos 5432/6379/8080 cerrados) y **el operador ordenó explícitamente no
arrancar nada**. En consecuencia:

| Campo | ¿Verificado? |
|---|---|
| Existencia de la tabla en la DB real | **NO VERIFICADO — requiere inspección de DB** (salvo las 18 del manifiesto) |
| Conteo de filas | Solo 18 tablas (manifiestos commiteados); el resto **NO VERIFICADO** |
| Freshness real (última escritura) | Solo 18 tablas; el resto **NO VERIFICADO** |
| Viva o muerta en producción | **NO VERIFICADO** — lo que aquí se publica es una *hipótesis* derivada de código |
| Tamaño físico, índices, PK/FK, chunk interval, políticas Timescale | **NO VERIFICADO** (es justo el gap que BL-44 declara) |
| % de NULL de `available_at` (19.9% daily / 6.4% m5) | **NO VERIFICADO** — cifra tomada de la prosa de BL-36/BL-38, cuyo artefacto crudo no está en el repo |

Una matriz honesta con huecos declarados vale más que una inventada: **ningún número de
este documento se estimó**. Donde no hay evidencia, dice NO VERIFICADO.

### 0.2 Las tres fuentes de evidencia usadas

| # | Fuente | Qué prueba | Qué NO prueba |
|---|---|---|---|
| E1 | DDL del repo (`database/migrations/**`, `database/schemas/**`, `init-scripts/**`) | Que el esquema está **declarado** | Que exista en la DB, ni que tenga filas |
| E2 | Escaneo estático del código (`airflow/`, `services/`, `scripts/`, `src/`, `config/`, `tests/`, `usdcop-trading-dashboard/{app,lib,components,hooks,types}`) | Que **existe una ruta de código** que escribe (`INSERT/UPDATE/DELETE/COPY/MERGE/to_sql/TRUNCATE`) o lee (`FROM/JOIN/read_sql`) | Que esa ruta se ejecute alguna vez en producción |
| E3 | Manifiestos de backup **commiteados** (`data/backups/features/feature_backup_manifest.json`, `data/backups/seeds/backup_manifest.json`) | Filas y timestamp de la fila más reciente, **volcados de la DB real** | Nada sobre las tablas que el backup no cubre |

**E3 es un snapshot, no la DB.** `created_at = 2026-07-27T20:04:28Z`. Es la única evidencia
empírica disponible sin conexión, y cubre 18 objetos de 168 declarados.

### 0.3 Reproducción

```bash
python scripts/diagnostics/db_inventory_matrix.py --write    # regenera la evidencia
python scripts/diagnostics/db_inventory_matrix.py --check    # falla si está stale
```

Salida completa (una entrada por objeto, con escritores/lectores fichero a fichero):
`.claude/generated/db-inventory.json`. El script es **read-only**: no abre conexión, no
emite DDL. Las cifras agregadas de §1 son de la corrida de 2026-07-28.

---

## 1. Hallazgo de entrada: el DDL del repo NO es el inventario de la DB

| Medida | Valor | Fuente |
|---|---|---|
| Tablas en el perfil del operador (2026-07-27) | 59 (964 cols, 2.82M filas, 7 hypertables) | prosa de BL-36 / Plan Consolidado — **artefacto crudo ausente del repo** |
| Objetos declarados en el DDL del repo | 242 (168 tablas + 74 vistas), 14 esquemas | E1, corrida 2026-07-28 |
| Hypertables declaradas en el DDL | 11 | E1 |
| Objetos con evidencia de filas | 18 | E3 |

**Interpretación (no verificable sin DB):** ~109 tablas declaradas no aparecen en el
perfil. O nunca se aplicaron (`init-scripts/` legacy que solo corre en un bootstrap
limpio), o el perfil cubrió solo algunos esquemas. **Cualquiera de las dos hipótesis
convierte al DDL del repo en una fuente NO autoritativa del inventario real** — que es
exactamente la enfermedad que BL-36 debe curar.

> El perfil habla de 7 hypertables; el DDL declara 11. La discrepancia es una pregunta
> abierta para el operador (§10, D-09), no un error de este documento.

**El inventario también se mueve mientras se escribe**: durante esta sesión aparecieron
sin commitear `database/migrations/070..073` (esquemas `control`, `forecast`, `action`,
`portfolio`, `exec`, `reference`, `market`, `quality`). El conteo pasó de 213 a 242
objetos en minutos. Ver §6.4.

---

## 2. Semántica de la matriz (cómo leer las columnas)

| Columna | Significado | Evidencia |
|---|---|---|
| `filas` | Filas al 2026-07-27T20:04Z | E3, o **NO VERIFICADO** |
| `W` | Nº de ficheros con ruta de ESCRITURA | E2 |
| `R` | Nº de ficheros con ruta de LECTURA | E2 |
| `ORM` | Declarada como modelo ORM sin ruta de query | E2 |
| `hipótesis` | Etiqueta derivada, **nunca un claim sobre producción** | derivada |

Etiquetas de hipótesis emitidas por el generador:

| Etiqueta | Definición operativa |
|---|---|
| `LIVE` | filas > 0 (E3) **y** existe escritor en código |
| `DATA_NO_WRITER` | filas > 0 (E3) **y** ningún escritor: dato congelado, nadie lo refresca |
| `DEAD` | filas = 0 (E3) **y** ningún escritor: candidata limpia a retiro |
| `EMPTY_WITH_WRITER` | filas = 0 (E3) pero hay escritor: el escritor nunca produjo fila |
| `WIRED_UNVERIFIED` | escritor **y** lector en código, filas NO VERIFICADAS |
| `PARTIALLY_WIRED_UNVERIFIED` | solo escritor **o** solo lector |
| `ORPHAN_DDL_UNVERIFIED` | DDL sin escritor ni lector en todo el repo |

**Límite de precisión de E2**: es un regex, no un parser SQL. Nombres genéricos
(`users`, `signals`, `executions`, `models`, `events`) se marcan `ambiguous_name: true`
en el JSON y no deben usarse para decidir un drop sin verificación manual.

---

## 3. Bloque A — Objetos con evidencia empírica de filas (18)

Snapshot E3 2026-07-27T20:04:28Z. `W`/`R` de E2.

| Tabla | Filas | Última fila | W | R | Hipótesis | Clasificación PROPUESTA | Justificación |
|---|---:|---|---:|---:|---|---|---|
| `usdcop_m5_ohlcv` | 2,204,770 | 2026-07-27 19:20 | 9 | **53** | LIVE | **SSOT (mercado 5m) → migrar a `market.canonical_bar`** | Núcleo vivo. Nombre engañoso (multi-símbolo). 53 lectores ⇒ renombre en caliente PROHIBIDO: vistas de compatibilidad primero (BL-38) |
| `asset_native_ohlcv` | 286,247 | 2026-07-27 19:00 | 1 | 2 | LIVE | **SSOT (raw multiframe) → `market.raw_bar`** | Es el raw de facto (1h/4h/1mo con provenance) |
| `asset_daily_ohlcv` | 60,416 | 2026-07-27 00:00 | 2 | 7 | LIVE | **SSOT diaria hoy → futura proyección/cagg** | Hoy es tabla ESCRITA, no derivada: separar `provider_official` de `resampled` antes de convertirla en vista |
| `macro_indicators_daily` | 18,014 | 2026-07-27 | **8** | **29** | LIVE | **SSOT ancho hoy → proyección sobre `macro.observation`** | 8 escritores distintos = riesgo §31 (§6.5) |
| `market_ingestion_manifest` | 3,262 | 2026-07-27 19:20 | 1 | 0 | LIVE | **MANTENER → `market.ingestion_run`** | Ledger de provenance, un único escritor: sano |
| `crypto_derivatives_daily` | 2,506 | 2026-07-20 | 1 | 0 | LIVE | **MANTENER con gates de cobertura** | Funding backtesteable; 0 lectores en código ⇒ consumo file-based (verificar) |
| `macro_indicators_monthly` | 799 | 2026-06-01 | 2 | 3 | LIVE | **→ proyección sobre PIT** | — |
| `news_articles` | 119 | 2026-07-27 17:25 | 3 | 4 | LIVE | **MANTENER (núcleo news)** | — |
| `audit_log` | 54 | 2026-07-21 02:54 | 11 | 4 | LIVE | **MANTENER, endurecer append-only** | 11 escritores es correcto aquí: es un ledger de eventos, no un atributo |
| `macro_indicators_quarterly` | 38 | 2025-06-30 | 0 | 1 | **DATA_NO_WRITER** | **DECISIÓN (D-05)** | Tiene datos pero **nadie los refresca**; última fila 2025-06-30 |
| `forecast_h5_predictions` | 18 | 2026-07-07 04:38 | 2 | 1 | LIVE | **→ `forecast.forecast_output`** | Superficie DIAGNOSTIC |
| `forecast_h5_signals` | 10 | 2026-07-07 | 3 | 7 | LIVE | **→ `action.strategy_signal`** | Superficie ACTION; 32 cols acumuladas ⇒ normalizar (BL-42) |
| `forecast_h5_executions` | 8 | 2026-07-05 22:54 | 4 | 7 | LIVE | **→ `exec.*` event-sourced** | — |
| `forecast_h5_subtrades` | 8 | 2026-07-05 22:54 | 4 | 1 | LIVE | **→ fills/trade readmodel** | — |
| `forecast_h5_paper_trading` | 8 | 2026-07-05 22:54 | 2 | 3 | LIVE | **→ hechos paper + `metric_event`** | — |
| `news_feature_snapshots` | 7 | 2026-07-27 | 1 | 0 | LIVE | **MANTENER** | — |
| `daily_analysis` | **0** | — | **0** | **0** | **DEAD** | **DECISIÓN (D-04)** | 37 cols, cero escritores, cero lectores, cero filas |
| `weekly_analysis` | **0** | — | **0** | **0** | **DEAD** | **DECISIÓN (D-04)** | ídem |

> Las 5 tablas `forecast_h5_*` no se han escrito desde **2026-07-05/07**. Si el ciclo
> semanal corre, eso es un síntoma; si el track vive en ficheros, es la confirmación de
> que la DB no es su SSOT. **NO VERIFICADO** cuál de las dos.

---

## 4. Bloque B — Núcleo declarado sin conteo verificado

Todas las filas de este bloque: **filas y freshness = NO VERIFICADO**.

### 4.1 Identidad y calendario

| Tabla | Cols | W | R | Hipótesis | Clasificación PROPUESTA |
|---|---:|---:|---:|---|---|
| `dim_asset` | 6 | **0** | **0** | ORPHAN_DDL | **DECISIÓN (D-06)** — el "embrión del asset registry" **no lo lee ningún código** (solo aparece en `scripts/diagnostics/column_audit.py` y en el módulo de backup) |
| `market_session_calendar` | 6 | 1 | 2 | WIRED | Migrar/proyectar desde `reference.calendar` |
| `reference.{asset,instrument,provider,provider_symbol,instrument_alias,bar_interval,calendar}` | 3-8 | 0 | 0-2 | ORPHAN_DDL (nuevas) | Destino BL-37 recién creado (§6.4) |

### 4.2 Macro

| Tabla | Cols | W | R | Hipótesis | Clasificación PROPUESTA |
|---|---:|---:|---:|---|---|
| `macro_indicators_pit` | 19 | 1 | 1 | WIRED | **SSOT vintage → núcleo de `macro.observation`**. Único esquema del repo con los 5 timestamps + `availability_policy` + `pit_vintage` (§8) |
| `macro_banrep_forwards_monthly` | 13 | 1 | 2 | WIRED | Mantener; `forward_rate` 100% NULL **por diseño** ⇒ declarar como fantasma (BL-40), no borrar en silencio |
| `macro_remesas_monthly` | 5 | 1 | 0 | PARTIAL | Migrar a observaciones largas |
| `macro_variable_snapshots` | 33 | **0** | **0** | ORPHAN_DDL | **DECISIÓN (D-03)** — huérfana total; el PIT ya es la vintage |
| `macro_extraction_log`, `macro_ffill_metadata`, `macro_readiness_log`, `macro_indicators_daily_backup` | 10-24 | 0 | 0 | ORPHAN_DDL | Candidatas a Bloque C |

### 4.3 MLOps / experimentos — **dos registries competidores**

| Tabla | Cols | W | R | Hipótesis | Clasificación PROPUESTA |
|---|---:|---:|---:|---|---|
| `public.model_registry` | 24 | **9** | **10** | WIRED | **DECISIÓN (D-02)** — muy cableado (Airflow L1/L3/L4, inference_api, dashboard `/api/production/monitor`) |
| `models.model_registry` | 14 | **10** | **10** | WIRED | **DECISIÓN (D-02)** — **segundo registry, mismo nombre, otro esquema** |
| `experiment_runs` | 20 | 1 | 0 | PARTIAL | FABRIC §49.4: eliminar como duplicidad de MLflow |
| `experiment_comparisons` | 18 | 0 | 0 | ORPHAN_DDL | ídem |
| `experiment_deployments` | 22 | 0 | 0 | ORPHAN_DDL | ídem |
| `experiment_contracts` | 10 | 1 | 1 | WIRED | No estaba en la lista de FABRIC §49.4 ⇒ **decidir aparte** |
| `metrics.model_performance` | 36 | **0** | **0** | ORPHAN_DDL | Eliminar; autoridad = `control.metric_event` |
| `config.models` | 16 | 8 | 5 | WIRED | Contiene `investor_demo` (`SYNTHETIC`) ⇒ BL-43 |
| `bi.dim_models`, `ml.lineage_*`, `mlflow.*` | 7-26 | 0-2 | 0 | PARTIAL/ORPHAN | Tercera y cuarta copia del concepto "modelo" |

> **`public.model_registry` + `models.model_registry` + `config.models` + `bi.dim_models`
> + `mlflow.model_deployments` = cinco lugares que describen un modelo.** Es la violación
> de §31 más grande encontrada, y **no aparecía en el perfil del operador**.

### 4.4 BI

| Tabla | Cols | W | R | Hipótesis |
|---|---:|---:|---:|---|
| `bi.fact_forecasts` | 19 | **5** | 2 | WIRED — escrita por `forecast_h1_l3/l5`, `src/forecasting/engine.py`, 2 scripts de migración |
| `bi.fact_model_metrics` | 19 | 2 | 1 | WIRED |
| `bi.fact_consensus` | 14 | 1 | 1 | WIRED |
| `bi.fact_inference_runs` | 14 | 0 | 0 | ORPHAN_DDL |
| `bi.forecast_experiment_{runs,comparisons,deployments}` | 25-58 | 0-1 | 0-1 | PARTIAL/ORPHAN |

**Corrección a la premisa de BL-36**: el BL afirma "el forecasting es file-based; nada las
lee". E2 dice que `bi.fact_forecasts` **tiene 5 escritores y 2 lectores declarados**
(incluido `services/inference_api/routers/forecasting.py`). Que estén a 0 filas
(**NO VERIFICADO**) significaría que esas rutas no se ejecutan — pero **el drop no es
gratis**: rompe compilación/ejecución de 7 ficheros. Ver D-01.

### 4.5 OMS legacy / ejecución

| Tabla | Cols | W | R | ORM | Hipótesis | Nota |
|---|---:|---:|---:|---:|---|---|
| `signals` | 15 | 0 | 0 | 0 | ORPHAN_DDL | nombre ambiguo: verificar a mano |
| `sb_signals` | 14 | 0 | 0 | **1** | ORPHAN_DDL | **solo existe como modelo SQLAlchemy** |
| `sb_executions` | 23 | 0 | 1 | 1 | PARTIAL | leída por `pretrade.py` |
| `trades_history` | 19 | 2 | 4 | 0 | WIRED | escrita por `trade_persister.py` + `paper_trading.py` |
| `equity_snapshots` | 8 | 1 | 1 | 0 | WIRED (hypertable) | leída por `/api/production/monitor` |
| `trading_state` | 17 | 1 | 0 | 0 | PARTIAL | 1 fila `ppo_v1` según perfil (**NO VERIFICADO**) |
| `executions`, `user_executions`, `trading.model_trades`, `models.model_trades` | 10-29 | 0-2 | 0-2 | 1 | PARTIAL/ORPHAN | cuarto y quinto sinónimo de "trade" |
| `backtest_trades`, `forecast_executions`, `forecast_paper_trading`, `forecast_vol_targeting_signals` | 16-27 | 1-4 | 1-4 | 0 | WIRED | superficie H1 paralela a `forecast_h5_*` |

### 4.6 Seguridad (empareja con BL-41)

| Tabla | Cols | W | R | ORM | Hipótesis |
|---|---:|---:|---:|---:|---|
| `user_exchange_keys` | 9 | 3 | 1 | 0 | WIRED (dashboard admin + tenant.py) |
| `sb_exchange_credentials` | 15 | 0 | 0 | **1** | ORPHAN_DDL — solo modelo ORM |
| `exchange_credentials` | 15 | 1 | 0 | 0 | PARTIAL — **tercera** tabla de credenciales, no mencionada por el perfil |
| `sb_credential_audit_logs` | 7 | 0 | 0 | 1 | ORPHAN_DDL |

> El perfil dice "DOS tablas de credenciales". E1 encuentra **tres** declaradas
> (`init-scripts/20-signalbridge-schema.sql` aporta la legacy `exchange_credentials`).
> BL-41 debe consolidar tres, no dos. **Presencia real en DB: NO VERIFICADO.**

### 4.7 Cripto forward-only (moratoria BTC)

`crypto_onchain_daily` (13c, hypertable), `crypto_flows_daily` (9c, hypertable),
`crypto_event_calendar` (15c), `crypto_exposure_signals` (21c, hypertable): **0 escritores,
0 lectores** en todo el repo. Son el caso de uso canónico de `staging_contract` (§7).
**NO BORRAR** (directiva BL-36: se llenan solas cuando se reabra la familia).

### 4.8 RL (PAUSED)

`inference_features_5m` (18c, W3/R8), `inference_ready_nrt` (8c, W2/R1),
`inference_signals_nrt` (12c, W0/R0), `dw.fact_rl_inference` (63c, hypertable, W1/R9),
`trading.model_inferences` (19c, hypertable), `python_features_5m` (6c, huérfana).
**Propuesta: mantener esquema, marcar `PAUSED`** (BL-36 §3). `inference_features_5m` está
fuertemente cableada (8 lectores) — un drop rompe el pipeline RL aunque esté pausado.

---

## 5. Bloque C — Superficie DDL huérfana

**143 objetos declarados con 0 escritores y 0 lectores en todo el repo**
(`ORPHAN_DDL_UNVERIFIED`; lista completa en `.claude/generated/db-inventory.json`).
Incluye los esquemas `audit.*` (6 tablas), `mlflow.*` (4), `ml.*` (parcial),
`events.signals_stream`, `circuit_breaker_state`, `drift_*` (parcial), `trading_configs`,
`trading_sessions`, `news_{keywords,sources}` y los recién creados `reference.*`,
`market.*`, `quality.*`, `forecast.*`, `control.*`.

**Dos causas muy distintas viven en el mismo cubo y no deben confundirse:**

1. **Legacy nunca cableado** (`audit.*`, `mlflow.*`, `events.signals_stream`): DDL que
   sobrevivió a su caso de uso. Candidatas reales a retiro.
2. **Destino recién creado y aún no cableado** (`reference.*`, `market.*`, `quality.*`,
   `forecast.*`, `control.*`, migraciones 070-073): huérfano **por ahora**, por diseño.

Distinguirlas requiere la fecha de creación del DDL, no el conteo de referencias.
`ddl_sources` en el JSON lo permite fichero a fichero.

---

## 6. Hallazgos que el perfil del operador NO reportaba

### 6.1 Dos `model_registry` en dos esquemas
`public.model_registry` (24c, W9/R10) y `models.model_registry` (14c, W10/R10). Ambos
cableados. Un `SELECT` sin esquema calificado resuelve por `search_path` — **el mismo
código puede estar leyendo tablas distintas según el rol**. Ver D-02.

### 6.2 Tablas que solo existen como modelo ORM
`sb_signals`, `sb_exchange_credentials`, `sb_credential_audit_logs`: declaradas en
`services/signalbridge_api/app/models.py` sin ninguna ruta de query. El esquema existe
porque SQLAlchemy lo declara, no porque alguien lo use. El generador las separa en
`orm_declarations` para no contarlas como escritores.

### 6.3 `macro_indicators_quarterly`: datos sin escritor
38 filas, última 2025-06-30, **cero escritores**. Dato congelado hace más de un año que
un lector sigue consultando. Es peor que una tabla vacía: sirve datos viejos en silencio.

### 6.4 La premisa "decidir antes de crear" ya fue superada
Durante esta sesión aparecieron sin commitear `database/migrations/070..073`, creando
`control.{strategy_declaration,incident,artifact_identity,metric_event}`,
`forecast.{forecast_output,forecast_score,model_horizon_result,calibration_result}`,
`reference.*` (7 tablas), `market.{raw_bar,canonical_bar}`, `quality.*` (3) y los esquemas
`action`, `portfolio`, `exec`.

**Consecuencia para BL-36**: su gate ("decide ANTES de ejecutar BL-15/18/19/21/22") ya no
puede cumplirse tal cual. El BL cambia de naturaleza: de *"qué destinos creamos"* a
**"qué hacemos con el lado legacy ahora que el destino existe"** — y aparece un riesgo
nuevo: **convivencia de dos verdades** (`usdcop_m5_ohlcv` ∥ `market.canonical_bar`,
`forecast_h5_predictions` ∥ `forecast.forecast_output`) hasta que el strangler cierre.
Ver D-08.

### 6.5 Concentración de escritores (riesgo §31)
`macro_indicators_daily`: **8 escritores** (`l0_macro_update.py`, `macro_cleanup_service`,
`macro_merge_service`, `macro_scraper_robust`, `scraper_banrep_selenium`, +3 scripts).
`usdcop_m5_ohlcv`: 9 escritores / **53 lectores**. Ninguna de las dos tiene un escritor
único identificable ⇒ son las dos primeras candidatas a "canonical writer" (BL-17).

### 6.6 Tres tablas de credenciales, no dos (§4.6).

---

## 7. `staging_contract`: contrato de permanencia de una tabla vacía

**Regla (FABRIC §49.5 / Plan Consolidado §11):** *una tabla vacía permanece en producción
solo si tiene **owner, productor, consumidor, contrato, fecha de activación y test**.*

**Propuesta (NO ejecutada):** un esquema `staging_contract` que no es un cementerio sino
un **compromiso con vencimiento**. Toda tabla vacía sin los seis campos se mueve allí (o
se retira, D-07), y una tabla en `staging_contract`:

1. no es legible por ningún rol de aplicación (ni Airflow genérico ni frontend);
2. no aparece en ninguna vista de producto;
3. tiene `activation_date`; vencida sin productor ⇒ el CI la marca en rojo;
4. su DDL vive en Git y se recrea cuando exista el caso de uso.

**Candidatas propuestas** (todas 0 escritores / 0 lectores; filas **NO VERIFICADAS** salvo
donde se indica):

| Tabla | Por qué a `staging_contract` y no a retiro |
|---|---|
| `crypto_onchain_daily` | Fuente forward-only declarada para reabrir familia BTC |
| `crypto_flows_daily` | ídem |
| `crypto_event_calendar` | ídem |
| `crypto_exposure_signals` | ídem (hypertable vacía: ver BL-44) |
| `inference_features_5m` | RL PAUSED, no muerto — **pero tiene 8 lectores**: mover el esquema rompería el pipeline. Requiere D-07 |

**Candidatas a retiro directo, NO a `staging_contract`** (sin caso de uso declarado):
`macro_variable_snapshots`, `daily_analysis`, `weekly_analysis` (0 filas VERIFICADAS),
`experiment_comparisons`, `experiment_deployments`, `metrics.model_performance`,
`bi.fact_inference_runs`, `signals`, `sb_signals`.

**Lo que este documento NO hace:** crear el esquema, mover ninguna tabla, ni escribir la
migración. Todo eso es D-07 + trabajo de CODEX en `database/migrations/**`.

---

## 8. Semántica de los 5 timestamps y mapeo del legado

### 8.1 Definición normativa (FABRIC §37/§38.3)

| # | Timestamp | Responde a | Quién lo asigna | Invariante |
|---|---|---|---|---|
| 1 | `event_time` | ¿cuándo ocurrió el hecho en el mercado? | el mercado | clave temporal de la barra/observación |
| 2 | `provider_published_at` | ¿cuándo lo publicó el proveedor? | el proveedor | puede ser NULL si no lo expone |
| 3 | `available_at` | **¿desde cuándo podía yo haberlo sabido?** | política declarada | `available_at >= event_time`; **es el único válido para joins de decisión** |
| 4 | `retrieved_at` | ¿cuándo lo pedí yo? | el extractor | `retrieved_at >= available_at` |
| 5 | `ingested_at` | ¿cuándo aterrizó la fila? | la DB | **jamás sustituye a `available_at`** |

`updated_at` **no es uno de los cinco**: marca mutación, y una barra raw inmutable no
tiene nada que mutar. Su presencia en una tabla de hechos es señal de que la tabla se
edita en vez de corregirse con un evento (BL-40).

Cuando `available_at` se **reconstruye** (no es vintage real), se guardan
`availability_policy` y `availability_quality` — y el dato **no es promocionable** hasta
que se declare esa calidad (`macro_indicators_pit.pit_vintage` ya lo modela).

### 8.2 Mapeo del inventario actual (evidencia E1, generada)

| Tabla | 1 event | 2 published | 3 available | 4 retrieved | 5 ingested | mutación |
|---|---|---|---|---|---|---|
| `market.raw_bar` (nueva) | `event_time` | `provider_published_at` | `available_at` | `retrieved_at` | `ingested_at` | — |
| `macro_indicators_pit` | `observation_date`, `reference_date` | `release_date` | `available_at` | `retrieved_at` | `created_at` | `updated_at` |
| `market.canonical_bar` (nueva) | `event_time` | — | `available_at` | — | `created_at` | — |
| `asset_native_ohlcv` | `time` | — | `available_at` | **falta** | `created_at` | — |
| `usdcop_m5_ohlcv` | `time` | — | `available_at` (ALTER 060) | **falta** | `created_at` | `updated_at` |
| `asset_daily_ohlcv` | `time` | — | `available_at` (ALTER 060) | **falta** | **falta** | — |
| `crypto_derivatives_daily` | `date` | `published_at` | **falta** | **falta** | — | `updated_at` |
| `macro_indicators_daily` | `date` | **falta** | **falta** | **falta** | — | `updated_at` |
| `macro_indicators_{monthly,quarterly}` | — | `publication_date` | **falta** | **falta** | — | `updated_at` |
| `forecast_h5_{predictions,signals}` | — | — | **falta** | — | `created_at` | — |

**Conclusiones factuales del mapeo:**

1. **`macro_indicators_pit` es el único esquema legacy completo** — es el patrón a
   generalizar, y confirma la "victoria de alineación" que BL-36 le atribuye.
2. **Ninguna tabla de mercado legacy tiene `retrieved_at`**: `created_at` (ingesta) hace
   de proxy, lo que colapsa los roles 4 y 5 — precisamente lo que §38.3 prohíbe.
3. **Las macro anchas (`daily/monthly/quarterly`) no tienen `available_at` en absoluto**:
   cualquier join de decisión contra ellas es un `shift(1)` a ciegas, no un as-of real.
4. `macro_indicators_{monthly,quarterly}.publication_date` existe pero, según la prosa del
   perfil, está **100% NULL** (**NO VERIFICADO**) — columna fantasma (BL-40) que además
   **bloquea promotion** por §BL-24.
5. Las tablas `forecast_h5_*` solo tienen `created_at`: no hay forma de reconstruir qué se
   sabía en el momento de decidir. Es un gap de auditabilidad, no de rendimiento.

### 8.3 Backfill etiquetado de `available_at` (BL-36 §2) — propuesta, NO ejecutada

- Regla propuesta: `available_at := cierre_de_barra + latencia_declarada_por_activo`.
- **Obligatorio** marcarlo: `availability_policy = 'reconstructed_close_plus_latency'`,
  `availability_quality = 'RECONSTRUCTED'` (nunca `VINTAGE`).
- El % NULL actual (19.9% daily / 6.4% m5) es **NO VERIFICADO** en esta sesión: la query
  de §11 lo mide.
- **Bloqueado por D-10**: la latencia por activo es un parámetro de negocio (declarado
  ex-ante, constitución §1), no una elección del implementador.

---

## 9. Clasificación propuesta por grupo (síntesis)

| Grupo | Propuesta | Evidencia que la sostiene | Riesgo si se ejecuta mal |
|---|---|---|---|
| Núcleo mercado (`usdcop_m5`, `asset_native`, `asset_daily`) | MANTENER + migrar con vistas de compatibilidad | E3 filas > 0; 53 lectores | Renombre en caliente rompe 53 ficheros |
| Macro PIT | SSOT vintage | E1: único con los 5 timestamps | — |
| Macro ancho | Proyección sobre PIT | E3 vivo; 8 escritores | Convertir en vista antes de tener el PIT completo deja el sistema sin macro |
| `forecast_h5_*` | Migrar a `forecast.*`/`action.*`/`exec.*` | E3: filas > 0 pero congeladas desde 2026-07-05 | Migrar sin strangler = doble verdad |
| BI `fact_*` | DEPRECATED | E1+E2: cableadas pero (según perfil) 0 filas | 7 ficheros referencian `bi.fact_forecasts` |
| Registries de experimentos/modelos | DEPRECATED, autoridad = MLflow + bundles | E2: 5 conceptos "modelo" | `model_registry` tiene 10 lectores incl. dashboard |
| `macro_variable_snapshots` | RETIRO | E2: 0/0 | ninguno detectable |
| `daily/weekly_analysis` | RETIRO o escritor real | **E3: 0 filas VERIFICADAS**, 0/0 | ninguno detectable |
| OMS legacy | ABSORBER en BL-21/22, luego drop | E2 mixto (`trades_history` W2/R4) | Drop antes de migrar rompe `/api/production/monitor` |
| SPY | **QUARANTINE por símbolo, NO delete** | prosa del perfil (8,428+403 filas) — **NO VERIFICADO** | Borrarlo destruye el linaje de los bundles v1.0.0 |
| Cripto forward-only | `staging_contract` | E2: 0/0, moratoria declarada | Borrarlas obliga a rehacer el DDL al reabrir |
| RL | MANTENER esquema, marcar PAUSED | E2: `inference_features_5m` R8 | Mover el esquema rompe 8 lectores |

---

## 10. DECISIONES QUE REQUIEREN AL OPERADOR (pendientes, no ejecutadas)

> Ninguna se ha ejecutado. Ninguna migración escrita. Cero DDL emitido.

| ID | Decisión | Opciones | Qué falta para decidir |
|---|---|---|---|
| **D-01** | `bi.fact_*` | (a) DROP tras BL-15/18 y reescribir los 7 ficheros que las referencian; (b) adoptarlas como proyección **read-only** de `forecast.forecast_output` | Confirmar conteo real de filas (**NO VERIFICADO**) y si `/api/.../forecasting` las sirve en vivo |
| **D-02** | **Dos `model_registry`** (`public` vs `models`) + `config.models` + `bi.dim_models` + `mlflow.*` | (a) uno canónico y el resto vistas; (b) todo a MLflow y proyección mínima por IDs | Cuál de los dos tiene filas; qué resuelve el `search_path` en producción |
| **D-03** | `macro_variable_snapshots` (33c, huérfana) | DROP / `staging_contract` | Si el "snapshot para LLM" sigue en el roadmap |
| **D-04** | `daily_analysis` / `weekly_analysis` (**0 filas verificadas**) | (a) DROP; (b) DEPRECATED con fecha de activación y escritor asignado | Si `/analysis` migrará de ficheros a DB, y cuándo |
| **D-05** | `macro_indicators_quarterly`: 38 filas, sin escritor desde 2025-06-30 | (a) asignar escritor; (b) congelar explícitamente; (c) retirar y servir desde PIT | Si algún feature en producción la consume |
| **D-06** | `dim_asset`: 0 escritores / 0 lectores pero es el "embrión del asset registry" | (a) promover a `reference.asset` y retirarla; (b) mantener como semilla | Si `reference.*` (migración 072) ya la sustituye |
| **D-07** | Crear el esquema `staging_contract` y qué se mueve allí | lista de §7 | Owner + fecha de activación por tabla (los 6 campos del contrato) |
| **D-08** | **Convivencia legacy ∥ nuevo** tras 070-073 | (a) strangler con vistas de compatibilidad y un único escritor; (b) congelar el nuevo hasta migrar | Es la decisión con más riesgo: define si habrá dos verdades |
| **D-09** | 242 objetos declarados vs 59 perfilados; 11 hypertables declaradas vs 7 | (a) `init-scripts/` legacy se retira del bootstrap; (b) el perfil se rehace con todos los esquemas | Requiere el perfil v2 contra la DB real |
| **D-10** | Latencia declarada por activo para el backfill de `available_at` | valor por activo (COP/XAU/BTC/SPX) | Parámetro de negocio: **prior ex-ante**, no ajustable a posteriori |
| **D-11** | SPY: quarantine por `source`/símbolo | (a) filtro en lectores; (b) columna `quarantined` | Confirmar que ningún lector activo lo consume |
| **D-12** | Tres tablas de credenciales (no dos) | consolidación en `secret.external_account` (BL-41) | Cuál de las tres tiene filas |

---

## 11. Cómo completar la matriz cuando haya DB (solo lectura)

Consultas **SELECT-only**; ninguna modifica nada. Rellenan exactamente las columnas
marcadas NO VERIFICADO.

```sql
-- 1. Inventario real: esquema, tabla, filas estimadas y última escritura observada
SELECT n.nspname AS schema, c.relname AS table,
       s.n_live_tup AS est_rows, s.last_autovacuum, s.last_analyze
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.relkind = 'r' AND n.nspname NOT IN ('pg_catalog','information_schema')
ORDER BY 1, 2;

-- 2. Conteo exacto (caro; solo sobre las dudosas)
SELECT 'macro_variable_snapshots' t, count(*) FROM macro_variable_snapshots
UNION ALL SELECT 'bi.fact_forecasts', count(*) FROM bi.fact_forecasts
UNION ALL SELECT 'public.model_registry', count(*) FROM public.model_registry
UNION ALL SELECT 'models.model_registry', count(*) FROM models.model_registry;

-- 3. Hypertables reales
SELECT hypertable_schema, hypertable_name, num_chunks
FROM timescaledb_information.hypertables ORDER BY 1,2;

-- 4. El gap de available_at (BL-36 §2)
SELECT 'usdcop_m5_ohlcv' t,
       count(*) FILTER (WHERE available_at IS NULL)::float / count(*) pct_null
FROM usdcop_m5_ohlcv
UNION ALL
SELECT 'asset_daily_ohlcv',
       count(*) FILTER (WHERE available_at IS NULL)::float / count(*)
FROM asset_daily_ohlcv;

-- 5. Columnas fantasma declaradas (BL-40)
SELECT count(*) total,
       count(publication_date) con_publicacion
FROM macro_indicators_monthly;

-- 6. Linaje SPY antes de cualquier cuarentena
SELECT source, symbol, count(*), min(time), max(time)
FROM asset_daily_ohlcv WHERE symbol ILIKE '%SPY%' GROUP BY 1,2;
```

---

## 12. Verificación de este documento

| Check | Estado |
|---|---|
| `python scripts/diagnostics/db_inventory_matrix.py --check` | verde (JSON determinista y al día) |
| `pytest tests/regression/test_scripts_layout.py -q` | verde |
| `pytest tests/regression/test_knowledge_frontmatter.py -q` | sin fallos nuevos vs `.claude/coordination/BASELINE.md` |
| DDL emitido | **cero** |
| Tablas borradas/migradas | **cero** |
| Ficheros de `database/migrations/**` tocados | **cero** (solo lectura) |
| Trials abiertos | **0** (ingeniería pura) |
