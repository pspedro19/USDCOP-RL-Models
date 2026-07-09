# Estado de Implementación (traza SDD)

> Registro honesto de lo que YA está construido y verificado vs lo pendiente. Se actualiza a
> medida que se cierran fases del `IMPLEMENTATION_ROADMAP.md`. Cada entrega trae evidencia.

Última actualización: **2026-07-03**

---

## ✅ Fase 0 — AssetProfile (keystone, SPEC-12 / onboarding A1)

| Artefacto | Estado |
|---|---|
| `src/contracts/asset_profile.py` | ✅ Contrato asset-genérico (`AssetProfile`, `load_asset_profile`, `validate`). Import standalone (yaml only). |
| `config/assets/xauusd.yaml` | ✅ Perfil real del Oro (sesión metales/UTC, drivers macro Gold, `bars_per_day: 288` **medido**, range [250,6000] para historia 2004). |
| `config/assets/usdcop.yaml` | ✅ COP como perfil (deja de ser caso especial → ingesta/DAGs asset-genéricos). |
| Test A1 | ✅ ambos perfiles cargan y validan. |

## ✅ Fase 1 (parcial) — Capa de datos del Oro (SPEC-01 + SPEC-02, intermedios)

| Artefacto | Estado / evidencia |
|---|---|
| `scripts/ingest_asset_ohlcv.py` | ✅ Ingesta **asset-genérica** (driven by AssetProfile). CLI: `--asset xauusd`. |
| **5-min TwelveData** | ✅ 21,558 bars (2026-03-18→2026-07-03). La API sirve solo la ventana intradía reciente (floor ~2026-03); se pagina hacia atrás hasta agotarla. Audit limpio (0 dup/nan/integridad/rango). `bars_per_day` medido = **288** (24h metales). |
| **Daily TwelveData (deep)** | ✅ 5,992 bars **2004→2026** (paginado). Audit limpio. |
| **Daily Investing.com (cross-check)** | ✅ 1,205 bars vía cloudscraper (recipe del extractor, `instrument_id=8830`). Acuerdo TD↔Investing: **mediana 0.61%**, flag OK. Rellenó **65** fechas que TD no tenía → **6,057** daily. Degradación grácil si CF bloquea. |
| **Alineación TZ/DST** | ✅ 5-min en instantes UTC; daily normalizado al **cierre NY 17:00 ET** (21:00 UTC verano / 22:00 UTC invierno). Bug DST corregido (naive-17:00→localize, no Timedelta sobre tz-aware). |
| **Escalabilidad en tablas** | ✅ 5-min del Oro en la tabla multi-par existente `usdcop_m5_ohlcv` (symbol='XAU/USD' → 4 símbolos). Daily en tabla **multi-activo nueva** `asset_daily_ohlcv` (migración **051**, PK (time,symbol), vista `asset_daily_coverage`). Sin silos. |
| **Inserts idempotentes** | ✅ UPSERT `ON CONFLICT (time,symbol)`. Re-ejecución no duplica (verificado: 5-min estable, daily solo crece por fechas nuevas reales). |
| **Seeds escalables** | ✅ `seeds/latest/xauusd_m5_ohlcv.parquet` (976 KB) + `xauusd_daily_ohlcv.parquet` (201 KB), mismo schema `[time,symbol,o,h,l,c,volume]` que los pares FX. |
| **Tests TDD (onboarding A1–B1)** | ✅ `tests/onboarding/test_asset_xauusd.py` — 6/6 verdes (perfil válido, seed en rango, `bars_per_day==medido`, tz-aware + NY-close, sin drivers COP filtrados). |

**Comando reproducible:**
```bash
python scripts/ingest_asset_ohlcv.py --asset xauusd --daily-start 2004-01-01
python -m pytest tests/onboarding/test_asset_xauusd.py -q
```

## ✅ Fase 2-3 — Features + Régimen (SPEC-03 / SPEC-04)

| Artefacto | Estado / evidencia |
|---|---|
| `src/gold_rl/indicators.py` | ✅ Wilder ATR/ADX, Hurst rescaled-range, `build_daily_features` (SMA-100/200, realized_vol_20, ADX, Hurst), `classify_regime` (4 regímenes **COMPRESSION / TREND / STRETCHED / EVENT** con **histéresis** dwell≥4 → baja rotación) + `REGIME_RISK_MULT`. |
| Verificación régimen | ✅ `regime_transitions_per_year` bajo (regímenes del Oro duran semanas, no días). Distribución impresa por el runner. |

## ✅ Fase 2 (baselines) + Fase 6 (backtest honesto) — SPEC-07 / SPEC-09

| Artefacto | Estado / evidencia |
|---|---|
| `src/gold_rl/strategies.py` | ✅ Capa de riesgo determinista `vol_target_size` (target_vol=0.10, floor 6%, cap 1.5x) separada de la dirección. **B1** long-only vol-targeted, **B2** trend-follower (SMA100+ADX>25), **regime-gated** (SMA200 gated flat en EVENT/STRETCHED). Posición **causal** (`shift(1)`). |
| `src/gold_rl/backtest.py` | ✅ Modelo de coste realista (2 bps turnover + 2.5%/yr swap), métricas (CAGR/Sharpe/Sortino/Calmar/MaxDD), **block-bootstrap p-value** (numpy `default_rng` seed=42, block=20, n=5000), extracción de trades, **atribución por régimen**, **5 gates** = Voto 1/2. |
| Resultados OOS (2004→2026, cap $10k) | ✅ B1 ret +?%/p=0.0094 · **B2 +61.1% Sharpe 0.38 p=0.0358 → PROMOTE** · regime-gated p=0.0152. Baselines obligatorios presentes (STRATEGY §6). |

## ✅ Fase 6 (registro) + Visibilidad Web — SPEC-12

| Artefacto | Estado / evidencia |
|---|---|
| `scripts/run_gold_pipeline.py` | ✅ Runner E2E: seed → features → régimen → backtest B1/B2/regime-gated → **publica bundles inmutables** vía `BundlePublisher` (additive, no toca COP). |
| Registro dinámico | ✅ `registry.json` ahora lista **2 activos (usdcop, xauusd) · 5 estrategias**. 3 bundles Gold `(strategy_id, 1.0.0, 2026)` inmutables + manifests. |
| **Dashboard web** | ✅ **VERIFICADO** (Playwright): selector muestra "Gold · Trend-follower Daily (B2)" (badge experimental, rule_based). Chart etiquetado **XAUUSD** (chart_symbol del manifest, NO USDCOP). KPIs, Curva de Equity, Trading Summary (95 ops, +63%), **tabla de 95 trades**, Replay con rango **21/12/2004→18/03/2026**, panel de gates, dropdown de versión + "Promover a activa". |
| **No-regresión COP** | ✅ Vista default (smart_simple_v11) intacta: +25.6% / Sharpe 3.30 / 34 trades / gates 5/5. |

**Comando reproducible:**
```bash
python scripts/run_gold_pipeline.py            # backtest B1/B2/regime-gated + publica bundles
# → abrir /dashboard, seleccionar "Gold · ..." en el selector de estrategia
```

## ⏳ Pendiente (fuera del roadmap "visible en web")

| Fase | Qué falta |
|---|---|
| 4-5 (RL) | `GoldTradingEnv` + LSTM→PPO multi-seed + MLflow (SPEC-05/08). **Sustituido** por estrategias basadas en reglas por la tesis del *honest gate*: RL solo se justifica si BATE ambos baselines (B1/B2). Los baselines ya están publicados como el suelo a superar. |
| 1 (resto) | SPEC-03 macro rodante FRED (drivers ya definidos en el AssetProfile) + DAGs 1-2 vía fábrica. |
| infra | ~~Fábrica de pipelines~~ **✅ ENTREGADO (2026-07-05)**: `airflow/dags/asset_pipeline_factory.py` (SSOT `config/assets/pipelines.yaml`, CTR-ASSET-PIPELINE-001) emite `asset_xauusd_pipeline_weekly` (Dom 01:45 COT): `l0_ingest` → `l4_backtest_publish` (`run_gold_pipeline.py`) → `l6_verify_registry`. Gold ya es **DAG-driven**, no solo script manual. Falta: ejecución live (SPEC-11). Columna `asset_id` no requerida (daily ya es multi-activo por `symbol`). |

## Notas de decisión (esta entrega)

- **5-min del Oro reusa `usdcop_m5_ohlcv`** (multi-par por `symbol`), NO una tabla nueva — evita silo. El filtro de sesión COP (8-12:55) **no** se aplica al Oro: la sesión viene del `AssetProfile` (metales, ~24h weekday).
- **Investing.com 8830 == TwelveData XAU/USD** (mismo instrumento, acuerdo <1% típico). Se usa como **validación + relleno de huecos**; TwelveData es autoritativo (gana en fechas compartidas).
- **DST correcto**: los bars daily cierran a 17:00 ET todos los días, incluidos los 2 días de transición DST/año (el test A4 cazó el bug del Timedelta absoluto).
- **RL sustituido por reglas (honest gate)**: la tesis (STRATEGY §6) es que el alpha vive en la gestión de riesgo/adaptación de régimen, NO en la predicción. RL solo se promueve si supera B1 **y** B2. Entregamos primero los baselines + la estrategia regime-gated como suelo publicado; el env PPO (SPEC-05/08) queda como trabajo posterior que debe batir ese suelo para justificarse.
- **Discrepancia summary vs KPIs en pantalla**: el frontend `dynamicMetrics` **recomputa** métricas desde los trades visibles (base trade-level), que difiere del `summary` publicado (base retorno-diario). Ej. B2: pantalla muestra Sharpe 1.31 / p=0.0754 (4/5 gates) vs summary +61.1% / Sharpe 0.38 / p=0.0358 (PROMOTE). Ambos son honestos pero de distinta base; reconciliar (mostrar summary publicado en vez de recomputar) es un pulido pendiente, no un bloqueo.
