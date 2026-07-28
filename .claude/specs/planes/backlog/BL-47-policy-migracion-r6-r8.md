---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - scripts/pipeline/run_spx500_pipeline.py
  - scripts/pipeline/run_gold_pipeline.py
  - scripts/pipeline/run_btc_pipeline.py
  - scripts/pipeline/train_and_export_smart_simple.py
---

# BL-47 — Migración de estrategias al motor de políticas (R6-R8)

**Fuente**: planes/05-rule-based-strategies.md §12-§13 R6-R8 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Avance 2026-07-28 (R6 + R7 con paridad medida; R8 SPEC_ONLY)

Specs de política en `config/policies/*.yaml` (loader/factory `src/strategies/policies/`,
validador §11 `scripts/validation/validate_policy_specs.py`, arnés de paridad
`scripts/validation/check_policy_parity.py`, tests `tests/unit/test_policy_specs.py`).

| Política | Motor · modo | Paridad vs productor congelado (dato real) |
|---|---|---|
| `spx500_daily_ma200_v1` | rule_based · declarative (DSL) | EXACTA, toda la ventana |
| `gold_trend_simple` | rule_based · coded_policy | EXACTA fuera del calentamiento; divergencia acotada a `bars<252` (declarada) |
| `btc_hodl_b1` | rule_based · coded_policy | EXACTA, toda la ventana |
| `smart_simple_v11` | **composite** · SPEC_ONLY | no migrada — `build_policy()` falla cerrado |

Ningún camino legacy se apagó ni se modificó: el criterio de corte (≥2 semanas verdes,
calendario BL-28/31) sigue pendiente. 0 trials.

**Divergencias declaradas (decisión del operador, NO resueltas aquí)**: dos productores
distintos para `gold_trend_simple` (con y sin multiplicador de régimen); semántica de
calentamiento (NaN como voto negativo vs fallo cerrado); `regime_risk_mult` nunca se
aplica en el pipeline BTC publicado. Detalle en cada spec.

## Estado actual (as-built verificado 2026-07-27)
Publishers actuales por activo (run_spx500/gold/btc_pipeline + publish_gold_dynexit stateful) producen bundles válidos; v11 corre en su cadena artesanal como composite de facto (Ridge/BR + Hurst + sizing + TP/HS).

## Qué falta exactamente
R6: migrar spx500_daily_ma200_v1 primero (la más simple) — paridad semantic_hash de señales/trades/PnL/bundles legacy vs motor nuevo ANTES de apagar el camino viejo. R7: xauusd_trend_simple_v1 y btcusdt_hodl_b1 (dynexit exige el estado de §15.2 resuelto en BL-45). R8: USD/COP al FINAL como engine.type=composite (conserva L3 del predictor; L7 queda para la última etapa del strangler). Clasificación §12 aplicada en todos los specs (PPO = rl).

## Impacto frontend
Cero cambio visual si la paridad es verde (ese es el criterio).

## Dependencias
BL-45, BL-46; se ejecuta DENTRO del calendario de BL-28/31 (mismo patrón strangler: paridad ≥2 semanas por estrategia, rollback declarado).

## Verificación
Diff semántico verde por estrategia; el A/B vivo v11/v12/v14 intocable durante la migración (sus ledgers son el patrón de paridad).

## Notas constitución
v11 FROZEN: migrar su cáscara a composite NO toca fórmula ni señal (re-freeze consciente de manifiesto, 0 trials, bit-check obligatorio).
