# Estado de Implementación — BTC/USDT (traza SDD)

> Registro **honesto** de lo YA construido y verificado vs lo pendiente. Cada entrega trae evidencia.
> Se actualiza al cerrar fases del `IMPLEMENTATION_ROADMAP.md`.

Última actualización: **2026-07-05**

---

## ✅ Fase 0 — Fundación escalable (AssetProfile + tablas + tests)

| Artefacto | Estado / evidencia |
|---|---|
| `config/assets/btcusdt.yaml` | ✅ AssetProfile real: crypto, `BTC/USDT`, `chart_symbol BTCUSDT`, **sesión 24/7 UTC** (`mode:24x7`, days [0-6], open/close null, `bars_per_day:288`, `bars_per_year:105120`, `weekend_flat:false`, `forced_close:none`), 10 drivers macro (4 FRED existentes + 6 cripto-native), régimen Hurst **null (re-fit)**, `strategy_id btc_exposure_s3`. |
| Test A1 (perfil valida) | ✅ `validate()==[]`. `chart_symbol` derivado = `BTCUSDT` (nunca "USDCOP"). |
| Test B1 (drivers) | ✅ sin drivers COP-only filtrados; presente ≥1 cripto-native (funding/mvrv/nupl/etf) + DFII10 (tasas reales). |
| `database/migrations/052_crypto_native_data.sql` | ✅ 5 tablas cripto-native **aditivas** (`crypto_onchain_daily`, `crypto_derivatives_daily`, `crypto_flows_daily`, `crypto_event_calendar`, `crypto_exposure_signals`) + vista `crypto_data_coverage`. Keyed `(date/time,symbol)` → ETH enchufa por symbol. Hypertables best-effort. `published_at` como guard D+1. No toca objeto existente. |
| `tests/onboarding/test_asset_btcusdt.py` | ✅ mirror cripto de Gold. A1/B1 **verdes** (2 passed); A2/A3/A4/**C1** esperan el seed (skip controlado con comando de reproducción). C1 es el test 24/7 nuevo (is_24x7, weekend NO flat, barras Sáb/Dom en el seed). |
| Paquete de specs `.claude/specs/assets/btcusdt/` | ✅ `README` + `SPEC-13` (integración escalable) + `IMPLEMENTATION_ROADMAP` + este status + `design/` (26 archivos, movidos desde `.claude/future plans/`) + `adr/`. |

**Comando reproducible:**
```bash
python -m pytest tests/onboarding/test_asset_btcusdt.py -q -k "a1 or b1"   # 2 passed
# migración: se aplica con el resto vía scripts/ops/db_migrate.py (aditiva)
```

---

## ✅ Fase 1 — Capa de datos (ENTREGADA con data canónica real)

> Se resolvió el bloqueo "necesitamos Binance pero no hay API key": **el endpoint público de
> Binance klines NO requiere key**. Se ingirió historia canónica real (ADR-0008) con lo que hay hoy.

| Artefacto | Estado / evidencia |
|---|---|
| `scripts/data/ingest_btc_ohlcv.py` | ✅ Extractor **Binance público** (sin key, canónico) + fallback TwelveData BTC/USD. NUEVO módulo paralelo (no toca la ingesta de Gold). |
| **Daily canónico** | ✅ **3,245 barras 2017-08-17 → 2026-07-05** (Binance spot BTC/USDT), audit limpio (0 dup/nan/integridad/rango), cierre **UTC 00:00** (test A4). |
| **5-min** | ✅ 8,640 barras (30d), `bars_per_day` medido = **288**, cobertura de días **[0-6]** (fines de semana presentes → prueba 24/7 real). |
| Seeds `seeds/latest/btcusdt_{daily,m5}_ohlcv.parquet` | ✅ mismo schema `[time,symbol,o,h,l,c,volume]` que FX/Gold. `symbol='BTC/USDT'`. |
| Tests A2/A3/A4/C1 | ✅ **7/7 verdes** con seed real (`tests/onboarding/test_asset_btcusdt.py`). |
| DB UPSERT | ⚙️ best-effort (daily→`asset_daily_ohlcv`, 5m→`usdcop_m5_ohlcv` por symbol); degrada si la DB no está arriba. Los seeds siempre se escriben. |
| Cripto-native (BGeo/funding/Farside/DefiLlama) | ⏳ **pendiente** (requiere `BGEOMETRICS_API_KEY`, scrapers Farside/DefiLlama). Tablas 052 listas; alimentan el HMM de ciclo y los gates de liquidez/eventos. |

## ✅ Fase 2 + Fase 7 (S3-lite) + Fase F (publicación) — ENTREGADAS

| Artefacto | Estado / evidencia |
|---|---|
| `src/btc_strategy/` | ✅ NUEVO módulo (indicators/strategies/backtest). Spot-only `exposure∈[0,1]`, **anualización √365**, cost model taker-fee-on-turnover (sin swap), régimen 4-estados con histéresis (proxy de precio hasta que llegue el HMM on-chain). No importa `src/gold_rl`. |
| `scripts/pipeline/run_btc_pipeline.py` | ✅ Runner E2E: seed → features → régimen → backtest B1/B2/S3 → publica bundles inmutables vía `BundlePublisher` (aditivo). |
| **Backtest OOS (2018→2026, cap $10k)** | ✅ **B2 Trend-follower: +351.2%, Sharpe 1.40, Calmar 1.83, MaxDD −11.0%, PF 6.5, p=0.0 → PROMOTE** (5/5 gates). B1 HODL-vol-target +271%/Sharpe 0.79/p=0.027. S3 regime-gated +124%/Sharpe 0.62 (REVIEW). |
| **⚠️ Honestidad OOS + trial-aware (2026-07, bundle v1.1.0)** | El +351% lo hacen los bull cycles 2017/2021. **OOS-2025 aislado: B2 −1.4% / Sharpe −0.05 / p=0.62** (se pica en el rango lateral). DSR (deflado por 3 trials): **B2 0.999 ✓** full-history pero **frágil OOS**; B1 0.896, S3 0.781. Conclusión: la ventaja de B2 es **regime-dependiente y no se reproduce fuera de tendencia** → el salto de calidad depende de la **Fase 1 cripto-native** (features no-precio). Ver `IMPLEMENTATION_ROADMAP.md` (conclusión empírica) y `../../audit/STRATEGIC-ASSESSMENT-2026-07.md §4`. |
| **Gate honesto (design §6)** | ✅ S3 **NO** le gana a B2 en Sharpe **ni** Calmar → "HODL/trend-follower es el suelo honesto; S3 necesita el HMM on-chain para justificarse". Registrado tal cual — sin inflar. B&H crudo +552% pero con drawdowns ~−77% (los baselines cambian retorno por riesgo: MaxDD −11%). |
| **Registro dinámico** | ✅ `registry.json` ahora lista **3 activos (btcusdt, usdcop, xauusd) · 8 estrategias**. 3 bundles BTC `(sid, 1.0.0, 2026)` inmutables + manifests. Meta del activo: Bitcoin / BTC/USDT / chart **BTCUSDT** / crypto. |
| **Frontend** | ✅ Cadena de resolución verificada: selector "Bitcoin" con 3 estrategias, chart **BTCUSDT** (NO USDCOP), replay 2018→2026 (47 trades), KPIs + gates 5/5. Ruta `/api/registry` lee `registry.json` dinámicamente (sin hardcode de activo). Misma maquinaria que Gold (Playwright-verificada). |
| **No-regresión** | ✅ COP (`smart_simple_v11` production, default) + Gold intactos. 41 tests verdes (9 registry R + 7 BTC + 6 Gold onboarding + 19 scripts layout). |

**Comando reproducible:**
```bash
python scripts/data/ingest_btc_ohlcv.py --no-db          # seeds canónicos Binance (sin key)
python scripts/pipeline/run_btc_pipeline.py              # backtest B1/B2/S3 + publica bundles
# → abrir /dashboard, seleccionar activo "Bitcoin" en el selector
```

## ⏳ Fases 3-6, 8-11 — Refinamiento del motor (según `design/`)

Para que **S3 supere el suelo** hace falta la señal que hoy no tenemos: régimen **HMM 4-estados
fit-congelado sobre on-chain** (MVRV-Z/NUPL/Puell de BGeometrics), `z_funding` (Binance perp),
gate de liquidez (ETF flows/stablecoins) y gate de eventos (LLM). Todo eso entra por los
extractores cripto-native pendientes → tablas 052 → y reemplaza el proxy de régimen basado en
precio. Luego meta-labeling (S4) y RL táctico (S5, opcional). El suelo publicado (B2) es el número
a batir. Detalle y gates en el roadmap.

---

## Notas de decisión (esta entrega)

- **Reutilización sin silos:** 5-min de BTC irá a `usdcop_m5_ohlcv` (multi-par por `symbol`), daily a
  `asset_daily_ohlcv` (migr. 051). Solo lo cripto-native (on-chain/funding/flows/eventos/señales)
  motiva la migración 052 — FX/Gold no necesitan esas tablas.
- **24/7 es el break real:** anualización √365 (`bars_per_year 105120`, jamás reusar COP/Gold), sin
  cierre de viernes, ejecución por **bandas de rebalanceo** (nuevo ExecutionStrategy), no trades TP/HS
  semanales. La sesión sale del AssetProfile; el filtro COP 8-12:55 no aplica.
- **`design/` movido, no reescrito:** la ciencia pre-registrada vino de `.claude/future plans/`
  (untracked) y se preservó íntegra bajo `design/`. SPEC-13 es el único puente nuevo hacia el monorepo.
- **Decisiones abiertas al operador (no bloquean Fase 0):** D5 (schedule ingesta 24/7 vía fábrica) y
  D6 (live vs paper). Ruta recomendada = paper + web primero, como el Oro.
- **DAG-driven ✅ (2026-07-05):** la fábrica `airflow/dags/asset_pipeline_factory.py` (SSOT
  `config/assets/pipelines.yaml`, CTR-ASSET-PIPELINE-001) ya emite `asset_btcusdt_pipeline_weekly`
  (Dom 02:00 COT): `l0_ingest` (Binance klines, graceful) → `l4_backtest_publish` (`run_btc_pipeline.py`)
  → `l6_verify_registry`. D5 resuelto como schedule semanal de backtest+publish (no live). BTC ya no es
  solo script manual.
- **Sin regresión:** todo aditivo en `fix/audit-p0-remediation`. Nada commiteado. COP/Gold intactos.
