# SPEC-12 — Integración con el Registro Dinámico y la Fábrica de Pipelines

## Propósito
Hacer que el Oro **no sea un silo**. En vez de un repo `gold-rl/` aparte con DAGs `xau_*` escritos a mano, el Oro se onboarda como **un activo más** sobre la columna vertebral multi-activo / multi-estrategia que el sistema USD/COP **ya tiene operativa**: `AssetProfile` por activo, **registro dinámico** (`registry.json` + `manifest.json`), **backtests inmutables versionados**, **fábrica de pipelines** config-driven, y **visibilidad automática en el frontend** (dropdown Activo→Estrategia→Versión→Año + replay). Todo **aditivo**: no se rompe ningún contrato ni consumo front↔back existente.

> Esta spec es el puente entre este paquete (este paquete (specs/assets/xauusd), greenfield) y las reglas ya vigentes del sistema:
> - `.claude/specs/architecture-overview.md` — mapa as-built (infra, contratos, drift).
> - `.claude/specs/assets/_onboarding-playbook.md` — contrato `AssetProfile` + stages + tests A1–F1.
> - `.claude/specs/platform/registry-lifecycle.md` — `StrategyBundleManifest`, `registry.json`, contrato I/O de DAG, inmutabilidad, replay, fábrica, tests R1–R9.
>
> **Regla de oro:** este paquete define *la ciencia* (datos, régimen, entorno, riesgo, validación); esta spec define *cómo se publica y se ve*, reutilizando la maquinaria existente. Nada de string-replace de "USDCOP".

---

## Principio: el frontend es función del Registro (config-driven, no copy-driven)

Hoy hay DAGs escritos a mano por track (H5, H1). Eso **no escala**: 5 DAGs × N estrategias × M activos = explosión. El diseño escalable es:

```
config/assets/<asset>.yaml     ─┐
config/strategies/<strat>.yaml ─┤──►  fábrica  build_forecast_pipeline(asset, strategy)
registry (enabled pairs)        ─┘         │
                                           ▼
airflow/dags/generated_pipelines.py  recorre el registro y emite los DAGs (train→signal→exec→monitor)
```

Agregar Oro = **una entrada de config + datos**, cero archivos DAG nuevos. El frontend se entera solo porque lee `/api/registry`, no rutas fijas.

### Qué CREAS vs qué REUSAS (tabla de decisión)

| Escenario | ¿DAGs nuevos? | ¿Pipeline nuevo? | Qué haces exactamente |
|---|---|---|---|
| Par nuevo, misma estrategia (Oro + Smart Simple/RL) | ❌ No | ❌ No (instanciado por la fábrica) | `config/assets/xauusd.yaml`, backfill de datos, definir drivers |
| Estrategia nueva, misma mecánica (otro ensemble, mismo TP/HS) | ❌ No | ❌ No | Entrada de config + registro |
| Estrategia con lógica de ejecución distinta (Momentum, RL, trailing propio) | ⚠️ Parcial | ❌ No | Escribes un `ExecutionStrategy` nuevo (contrato ya definido); la fábrica lo usa |
| Versión nueva del modelo (hiperparámetros/features) | ❌ No | ❌ No | Config congelada nueva → **bundle inmutable nuevo** en el registro |
| Año nuevo de backtest | ❌ No | ❌ No | Entrada inmutable nueva en `backtests[]` |

Lo único que se escribe como **código nuevo** es un `ExecutionStrategy` cuando la mecánica de trade cambia. Todo lo demás es config + datos + reentrenar.

---

## Contrato A — `AssetProfile` del Oro (reemplaza el hardcode COP)

El Oro se define en `config/assets/xauusd.yaml` (contrato `AssetProfile`, ver `_onboarding-playbook.md` §2). Parametriza TODO lo que hoy está pegado a COP: `symbol`, `chart_symbol`, `price_range`, `session` (timezone metales, `bars_per_day`, `bars_per_year`, cierre), `data_source` (símbolo TwelveData/Dukascopy + seed), `macro_drivers` (DXY −, real-yield −, VIX +, Brent +), y umbrales del `regime_gate` (**re-ajustados**, NO copiados de COP 0.52/0.42).

> **Plantilla concreta en este paquete:** [`config/asset-profile.example.yaml`](../config/asset-profile.example.yaml). Documenta el schema completo del perfil del Oro (sesión metales, drivers macro, `bars_per_day`/`hurst_*` = `null` para forzar la medición/re-fit). La copia REAL vive en el repo raíz `config/assets/xauusd.yaml` — **hoy inexistente** (bloque 0 del onboarding, test A1).

> Los drivers macro de este paquete (SPEC-03) — DXY, tasas reales (DFII10/T10YIE), calendario — son exactamente los `macro_drivers` del `AssetProfile`. NO reusar EMBI/IBR/TPM/WTI-como-export de COP.
> **TwelveData verificado (2026-07-03):** 8/8 keys válidas; `XAU/USD` y `BTC/USD` descargables. La ingesta (SPEC-01) alimenta `data_source.seed_file`.

---

## Contrato B — Publicación al Registro (el DAG "sale" publicando un bundle)

El pipeline de validación (SPEC-09 / DAG 4) NO termina en un HTML: su **tarea final es `register_bundle`** (contrato de salida). Reutiliza `airflow/dags/utils/register_bundle.py` (ya existe y se verificó corriendo dentro del scheduler).

```
… produce artefactos de backtest …
        │
        ▼
register_bundle (EXIT CONTRACT — atómico, validado):
  1. VALIDATE: summary/trades JSON-safe (sin Inf/NaN); gates completos (Vote 1/2);
     signals parquet presente si capabilities.replay; lineage feature_hash==norm_stats_hash.
  2. WRITE inmutable: public/data/strategies/<sid>/backtests/<version>/summary_<year>.json
     + trades_<year>.json + signals_<year>.parquet   (NUNCA sobreescribe)
  3. UPDATE manifest.json (write .tmp → rename atómico)
  4. UPSERT registry.json (entrada de esta estrategia/activo)
  IF cualquier paso falla → NO toca registry.json  (el front nunca ve un bundle roto)
```

Salida canónica (ver `registry-lifecycle.md` §3–§4):
- `public/data/registry.json` — índice que el front lee para armar los selectores.
- `public/data/strategies/smart_simple_xauusd/manifest.json` — auto-descriptivo (symbol, chart_symbol, backtests[], model_versions[], capabilities.replay).

---

## Contrato C — Inmutabilidad + versionado (la keystone de escalabilidad)

Un backtest publicado es **inmutable y direccionado por `(strategy_id, model_version, year)`**. Nunca se sobreescribe.

- Path con versión embebida: `backtests/<model_version>/summary_<year>.json`. Una versión nueva escribe una carpeta nueva; el backtest viejo queda replayable **para siempre**.
- Esto permite **replay de v1 vs v2 sobre el mismo 2025** y que coexistan → dos entradas para el dropdown.
- `production/summary.json` + `approval_state.json` son los ÚNICOS archivos mutables (el puntero "actual"), y solo referencian un backtest inmutable por `immutable_id`.
- **Estado (2026-07-03):** el versionado inmutable YA está implementado en `BundlePublisher` y **probado** (3 versiones coexisten bajo `smart_simple_v11`). El export legacy sigue escribiendo `summary_<year>.json` en paralelo **de forma aditiva** — el bundle versionado se escribe *además*, no *en vez de*. La deuda restante es puramente cosmética: retirar el `summary_<year>.json` mutable de la ruta de producción una vez que el front baked consuma solo `/api/registry`. Una `model_version` = una config congelada (`ssot-versioning.md`).

---

## Contrato D — Replay dinámico en el frontend

`replay(strategy_id, model_version, year)` → resuelve el manifest → carga el `signals` parquet + barras OHLCV del `asset_id`/año. El chart usa `chart_symbol` del manifest (NO literal "USDCOP"). API dinámica:

```
GET  /api/registry                                                   → registry.json (índice)
GET  /api/strategies/{strategy_id}/manifest                          → manifest.json
GET  /api/backtest/replay?strategy_id=…&model_version=…&year=2025    → SSE (BacktestSSEEvent)
```

Flujo de resolución del front (cero hardcode): `/api/registry` → selectores Activo→Estrategia→Versión→Año → resuelve paths del manifest → KPIs/trades/gates + Replay si `capabilities.replay`.

---

## Regla transversal: evolución ADITIVA de contratos (no romper consumos)

Todo lo anterior se agrega **al lado** de lo que el front ya consume. Reglas:

1. **Nunca** modificar `lib/contracts/*.ts` ni las rutas `app/api/**` existentes de forma incompatible; solo se **agregan** rutas (`/api/registry`) y archivos (`registry.json`, `manifest.json`).
2. Los archivos legacy (`summary_2025.json`, `strategies.json`) se **siguen escribiendo en paralelo** durante la migración → el front actual no se entera.
3. Cambio de contrato = actualizar **AMBOS** mirrors (TS + Python) + ventana de compatibilidad (regla ya existente del proyecto).
4. Publicar en `registry.json` SOLO desde la tarea `register_bundle`; un bundle inválido **no entra** (gate duro).
5. Antes de multiplicar estrategias: **reconciliar el drift TS↔Python del feature-contract** (`architecture-overview.md` §5.3). No forkear un baseline drifteado.

Fases seguras: **Fase 0** registro aditivo junto a lo existente (front lo ignora, riesgo 0) → **Fase 1** UI nueva LEE `/api/registry` con los endpoints viejos como fallback → **Fase 2** marcar viejos como "legacy" sin borrarlos.

---

## Criterios de aceptación (bind a los tests A1–F1 y R1–R9 existentes)

Onboarding del activo (de `_onboarding-playbook.md` §7):
- [ ] **A1** `AssetProfile` de Oro carga y valida (`config/assets/xauusd.yaml`).
- [ ] **A2/A3** seed OHLCV de Oro en rango; `session.bars_per_day` == mediana observada (sesión metales ≠ COT).
- [ ] **B1** cada `macro_drivers[].db_col` existe; ningún driver COP-only se coló.
- [ ] **C1** `feature_hash` recomputado == almacenado; mirror TS == Python (sin drift).
- [ ] **D1** thresholds Hurst re-ajustados al Oro (no iguales a 0.52/0.42 de COP).
- [ ] **E1–E5** los 5 gates de backtest OOS (retorno>−15%, Sharpe>0, DD<20%, ≥10 trades, p<0.05) — **son los mismos gates de aprobación** (Vote 1/2).

Registro + frontend (de `registry-lifecycle.md` §10):
- [ ] **R2** `manifest.json` del Oro valida; `symbol`/`chart_symbol` presentes; JSON-safe.
- [ ] **R3** publicar v2 NO modifica los archivos `backtests/1.x/**` de v1 (inmutabilidad).
- [ ] **R5** un bundle que no valida NO se agrega a `registry.json`.
- [ ] **R6** `/api/backtest/replay` para `(strategy, version, year)` resuelve dinámicamente vía manifest.
- [ ] **R7** registro con 2 activos → 2 opciones; estrategias filtradas por `asset_id`.
- [ ] **R8** v1 y v2 del mismo backtest/año coexisten y son replayables.
- [ ] **R9** el DAG del Oro corre `register_bundle` como tarea final, idempotente, con provenance.

---

## Estado real hoy (honesto — qué ya funciona y qué falta cablear)

> **Actualización 2026-07-03.** La columna vertebral del registro pasó de PROPOSED a **IMPLEMENTADA y PROBADA** sobre USD/COP. El Oro ya no diseña esta maquinaria — la **reutiliza**. Lo que sigue pendiente para el Oro es el **onboarding del activo** (AssetProfile + datos + reentrenar), no el registro.

**Ya funciona y está VERIFICADO (contrato + evidencia en repo):**
- **Publisher cableado en el export**: `scripts/pipeline/train_and_export_smart_simple.py` llama a `publish_versioned_bundle()` tras los writes legacy, en ambas fases (backtest `experimental`, production `production`). Es **aditivo** — los `production/*.json` legacy se siguen escribiendo intactos.
- **Versionado inmutable real**: `BundlePublisher` + `RegistryBuilder` (`src/contracts/strategy_manifest.py`) escriben `backtests/<version>/summary_<year>.json` sin sobreescribir; una re-publicación de la misma `(version, year)` es un no-op (`immutable_hit`).
- **Coexistencia probada (A/B testing)**: `public/data/strategies/smart_simple_v11/backtests/` contiene **3 versiones vivas** — `2.0.0` (activa, +25.63%), `3.0.0-A` (stops estrechos/gate laxo/Ridge+BR, +26.58%), `3.0.0-B` (stops anchos/gate estricto/+XGB, +18.73%) — todas replayables lado a lado.
- **Rama de estrategia nueva probada**: `smart_simple_aggr` (1.0.0, +29.47%) publicada como `strategy_id` separado vía `--strategy-id` → aparece como 2ª estrategia en el registro sin código de frontend nuevo.
- **Rutas API aditivas existentes**: `GET /api/registry`, `GET /api/strategies/{id}/manifest`, `POST /api/registry/promote` (flip de `model_versions[].active` + `production.model_version`). El registro (`registry.json`) expone `active_version` + headline (return/sharpe/p_value) para que el selector no haga N+1 fetch.
- **Frontend dinámico (a nivel dev-server, validado con Playwright)**: `ForecastingBacktestSection.tsx` lee `/api/registry`, arma el dropdown de estrategias con métricas, un selector **Versión** que sobre-escribe `summary`+`trades` (replay cliente reacciona solo vía `TradingChartWithSignals`), y un botón **Promover a activa**. `chart_symbol` viene del manifest (no literal "USDCOP").
- **Contrato blindado por tests**: `tests/contracts/test_strategy_registry.py` — **9 tests R** (inmutabilidad, coexistencia, ambas versiones replayables, no toca legacy, JSON-safe, chart symbol derivado, headline de versión activa en el registro). Verdes.
- **`register_bundle` en el scheduler**: `airflow/dags/utils/register_bundle.py` corre dentro del contenedor y publica al `public/data` montado (llega al host).
- Fuentes de datos Oro/BTC disponibles (TwelveData 8/8; XAU/BTC descargables).

**Falta para el Oro (onboarding del activo) y para escalar a "0 DAGs por activo":**
1. **`AssetProfile` del Oro NO existe aún**: `config/assets/` está vacío. Crear `config/assets/xauusd.yaml` (Contrato A) es el bloque 0 del onboarding (test A1). USD/COP debería migrar a `config/assets/usdcop.yaml` en paralelo.
2. **Fábrica de pipelines** (`airflow/dags/factories/…` + `generated_pipelines.py`): aún NO existe; los DAGs H5/H1/`xau_*` son a mano. Es el paso para "0 DAGs nuevos por activo".
3. **Rebuild del dashboard baked (prod)**: la imagen Docker en `:5000` aún no consume `/api/registry` (solo el dev-server lo hace). Rebuild para que producción vea el selector Versión→Año.
4. **Deuda transversal antes de multiplicar estrategias**: (a) columna `asset_id`/`symbol` en las tablas `forecast_h5_*` (hoy implícitamente COP); (b) reconciliar el drift TS↔Python del feature-contract (`architecture-overview.md` §5.3) — no forkear el baseline drifteado al derivar el contrato del Oro.

> **Lección de la prueba A/B (referencia para el Oro):** una nueva **versión** (config congelada distinta) → `--config X --version Y`, cero código. Una nueva **rama de estrategia** (mismo motor, otro id) → `--strategy-id`, cero código. Una estrategia con **mecánica de ejecución nueva** (el caso del RL del Oro) → se escribe **un** generador de señales (`UniversalSignalRecord[]`) + se elige/implementa un `ExecutionStrategy`; el resto (backtest, bundle inmutable, registro, dropdown, replay, promote) se reutiliza. El Oro es exactamente el tercer caso.

> Orden de mayor impacto/menor riesgo para el Oro: (1) AssetProfile + datos → (SPEC-01..09) ciencia → `register_bundle` publica su bundle inmutable (ya funciona) → aparece en el front (tras rebuild) → (2) la fábrica generaliza el ciclo a cualquier activo/estrategia.

---

## Dependencias
SPEC-08 (modelos, `model_version` = config congelada), SPEC-09 (backtest → `register_bundle`), SPEC-10 (DAG del Oro emitido por la fábrica), SPEC-11 (visibilidad + replay en el front). Reglas: `_onboarding-playbook.md`, `registry-lifecycle.md`, `architecture-overview.md`.
