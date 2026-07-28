---
kind: roadmap
status: PLANNED
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
