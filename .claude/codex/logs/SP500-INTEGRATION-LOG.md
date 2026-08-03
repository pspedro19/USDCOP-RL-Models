---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - config/assets/spx500.yaml
  - config/analysis/analysis_assets.yaml
  - usdcop-trading-dashboard/lib/contracts/analysis-assets.ts
  - usdcop-trading-dashboard/lib/billing/prices.ts
---

# Integración S&P 500

## Fuente auditada

`C:\Users\pedro\Downloads\SP500 v2\SP500 v2\SP500` contiene SDD-000..007,
ingesta point-in-time, features sin leakage, purged K-fold, CPCV/PBO/DSR, costos,
trial registry, lockbox, gates y estrategia runnable. Su suite propia reportó 10 tests verdes.

## Decisiones de adaptación

- Canonical id: `spx500`; símbolo de datos: `SPY` total-return como proxy inicial; display: `SPX500`.
- Sesión: días hábiles EEUU, 09:30–16:00 America/New_York, 250 días/año.
- Features: retornos, volatilidad, MA200/momentum, breadth 200/8, distribution days,
  VIX, condiciones financieras, spreads HY y curva 10Y–2Y.
- Validación obligatoria: PIT, causal lag, purged K-fold + embargo, CPCV, DSR, PBO,
  costos y benchmark SPY/MA200/vol-target/60-40.
- El generador SP500 original produce datos sintéticos y contiene un `shift(-3)` en
  `macro_stress`; no se copia a producción. Se reemplazará por series reales con `available_at`.

## Estado actual

Implementado: perfiles backend/frontend, catálogo de análisis, add-on y contrato de features.
Pendiente: seed real y lineage, namespace de estrategia/manifest, DAG parametrizado, rutas sin
allowlist fija, entitlements/revenue dinámicos, backtest OOS y visualizaciones con datos reales.

Regla: SP500 no pasa a `available`/`production` hasta que los gates P0 estén verdes.

## Actualización 2026-07-20 (implementación)

- Copiado el motor de estrategia SP500 v2 (`engine`, `policies`, `regime`, `metrics`, `costs`,
  `benchmarks`, `deflated_sharpe`, `pbo`, `purged_kfold`, `gates`, `economic_metrics`) a
  `src/strategies/spx500_regime_gated_v1/`.
- Se eliminó el `shift(-3)` forward-looking del generador sintético. El scaffold sigue marcado
  como sintético y no puede producir evidencia de alfa.
- Manifest experimental creado y registry regenerado: 4 activos, 16 estrategias; SPX500 permanece
  `experimental`, sin replay/live y con aprobación `PENDING_REAL_DATA_AND_OOS`.
- Rutas forecasting/data, entitlements internos y revenue reconocen `spx500`.
- Pruebas: integración SPX500 `4 passed`; suite de estrategia adaptada `10 passed`.
- Configuración de experimento/retraining añadida en `config/forecast_experiments/spx500_regime_gated_v1.yaml`:
  ventanas temporales, benchmarks, costos 1x/2x/3x, DSR/PBO, champion-challenger y rollback por drift.
- Validación del `AssetProfile` oficial: `spx500`, símbolo canónico `SPX/500`, sesión
  `exchange_hours` America/New_York, resultado `validate() == []`.
- Gate OOS añadido en `src/validation/sp500_oos_gate.py`; tests de integración/OOS/pipeline:
  `10 passed`; suite propia de estrategia: `10 passed`; `run_spx500_pipeline.py --check`: PASS.

## Actualizacion 2026-07-20 (DAG parametrizado)

- Anadido `spx500` al SSOT `config/assets/pipelines.yaml`; la factoria Airflow genera
  `asset_spx500_pipeline_weekly` sin cambios de codigo por activo.
- Nuevo entrypoint `scripts/pipeline/run_spx500_pipeline.py`: valida coherencia asset/experimento
  antes de ejecutar backtest y gates estadisticos; no permite mismatch silencioso.
- Ingesta diaria y export de chart son tareas graceful; la etapa cientifica es bloqueante y la
  verificacion exige el bundle `spx500_regime_gated_v1`.
- Tests de contrato anadidos en `tests/integration/test_spx500_pipeline_config.py`.

## Bloqueos restantes

Feed PIT real SPY/SPX + FRED, seed y lineage; backtest OOS con benchmarks/costos; parametrización de
DAG y retraining; eliminación de allowlists restantes; tests frontend/E2E y publicación de charts.

## 2026-07-20 Frontend/backend follow-up
- Catalog registry now classifies spx500 as equity_index, preventing fallback to FX.
- Removed hardcoded LIVE_FX_SYMBOLS allowlist; catalog probes symbols published by registry and gracefully handles unsupported quotes.
- Added regression contract test for dynamic symbol discovery and asset class.
- Verification: python -m pytest tests/regression/test_spx500_integration.py -q => 5 passed.
