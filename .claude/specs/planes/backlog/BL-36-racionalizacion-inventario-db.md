---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - airflow/dags/l0_macro_update.py
  - services/signalbridge_api
---

# BL-36 — Racionalización del inventario DB (59 tablas → matriz de verdad aplicada)

**Fuente**: perfil de datos del operador (artifact 2026-07-27: 59 tablas, 964 cols,
2.82M filas, 7 hypertables) × FABRIC §7 (matriz de fuente de verdad) y §31 ("ningún
atributo con dos escritores") · **Ola**: 2-3 (decisiones antes de crear esquemas
nuevos) · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado contra el perfil)

**Núcleo VIVO y alineado (mantener tal cual):**
`usdcop_m5_ohlcv` (2.20M, multi-símbolo por diseño) · `asset_daily_ohlcv` (60K, 1979→)
· `asset_native_ohlcv` (286K, raw provenance 1h/4h/1mo) · `macro_indicators_{daily 34c,
monthly, quarterly}` · **`macro_indicators_pit` (216K — ES la rama as_released que
BL-24 formaliza: victoria de alineación)** · `macro_banrep_forwards_monthly`
(forward_rate 100% NULL por diseño: la señal vive en implied_dev) · `macro_remesas` ·
`crypto_derivatives_daily` (funding backtesteable) · `forecast_h5_*` (producción) ·
`dim_asset` (¡ya trae annualization 52/252/365 — embrión del asset registry FABRIC!) ·
`market_session_calendar` (per-asset 2020-2027) · `market_ingestion_manifest`
(provenance) · `news_*` vivas · `audit_log` · `sb_users/configs/trading_state` ·
esquemas cripto vacíos forward-only (event_calendar/flows/onchain/exposure — moratoria
BTC: se llenan solos, NO borrar).

**Redundancias detectadas (dos escritores potenciales / peso muerto):**
1. **BI star schema**: `bi.fact_{consensus,forecasts,inference_runs,model_metrics}` =
   0 filas (dims 9/7 pobladas). El forecasting es file-based; esto duplica el destino
   de BL-15 (`forecast_output`) y BL-18 (`metric_event`).
2. **Registro de experimentos paralelo**: `experiment_{runs,comparisons,deployments}`,
   `model_registry`, `metrics.model_performance` = 0 filas. FABRIC §7 asigna modelos a
   MLflow+MinIO; estas 5 tablas son un segundo registry jamás escrito.
3. **`macro_variable_snapshots`** (0 filas, 22c) vs `macro_indicators_pit`: el
   snapshot-para-LLM quedó huérfano (analysis es file-driven).
4. **`daily_analysis`/`weekly_analysis`** (0): declaradas "DB destino futuro" — dos
   verdades latentes con los archivos.
5. **OMS legacy**: `signals`, `trades_history`, `equity_snapshots`, `sb_signals`,
   `sb_executions` = 0 filas — se solapan con el destino de BL-21 (`exec.*`
   event-sourced) y BL-22 (`fact_position/fact_pnl`).
6. **SPY legado**: 8,428 filas en asset_daily + 403 en native (retirado por directiva
   2026-07-27; el lector ya no lo consume).

**Gaps de spine detectados por el perfil** (alimentan BL-17/24):
`asset_daily_ohlcv.available_at` 19.9% NULL · `usdcop_m5.available_at` 6.4% NULL.

## Qué falta exactamente

1. **Decisión por grupo, escrita en la matriz de verdad** (un escritor por atributo):
   - bi.fact_* → DEPRECATED (drop tras BL-15/18; dims se derivan de config YAMLs).
   - experiment_*/model_registry/metrics.model_performance → DEPRECATED; autoridad =
     MLflow + bundles inmutables (o adopción explícita como proyección — pero UNA cosa).
   - macro_variable_snapshots → DROP (pit es la vintage; analysis es file-driven).
   - daily/weekly_analysis → DEPRECATED hasta que exista escritor real (nota en spec).
   - OMS legacy → ABSORBIDAS por BL-21/22 con migración y drop posterior (jamás
     esquemas paralelos conviviendo).
   - SPY → quarantine por `source`/símbolo (excluir de lectores; conservar como linaje
     de los bundles v1.0.0; NO delete).
2. **Backfill etiquetado de `available_at`** (reconstruido = cierre+1d, marcado, no
   vintage) en daily/m5 — prerrequisito del spine (BL-17) y del camino dorado (BL-24).
3. RL (`inference_features_5m`, `trading.*`) → mantener esquemas, marcar PAUSED.

## Impacto frontend

Ninguno directo (las tablas redundantes tienen 0 filas y nada las lee). Indirecto:
evita que BL-32 (Passport/Control Tower) nazca leyendo dos verdades.

## Dependencias

Decide ANTES de ejecutar BL-15/18/19/21/22 (los destinos nuevos absorben, no conviven).

## Verificación

- Documento de decisión por grupo en la matriz de verdad (§7) + migración de drops.
- `SELECT` de disponibilidad: 0 lectores rotos tras cada drop (grep de cada tabla en
  airflow/ services/ scripts/ dashboard antes de tocar).
- available_at: % NULL → 0 en daily/m5 con etiqueta de reconstrucción.

## Notas constitución

Ningún drop toca datos con historia real irrecuperable (todas las candidatas están en
0 filas salvo SPY, que se conserva). La regla que gobierna: "exactamente una fuente
autoritativa por atributo" (§25) — este BL es su aplicación al inventario heredado.
