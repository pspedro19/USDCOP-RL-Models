---
kind: roadmap
status: IMPLEMENTED
version: 1.1.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - scripts/analysis/profitability_adapters.py
  - usdcop-trading-dashboard/public/data/registry.json
---

# BL-07 — Etapa 0.5: timing_ratio one-off de las 4 campeonas

**Fuente**: FABRIC §28 E0.5 + §18.3 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Los sleeves tienen streams diarios reproducibles (adapters/bundles signals). El número timing_ratio no existe en ningún lado.

Corrección as-built 2026-07-27 (evidencia:
`scripts/analysis/profitability_adapters.py:147-164`): el adapter de COP NO contiene
el retorno subyacente USD/COP; publica el PnL semanal neto de v11 como `asset_ret` y
marca sus baselines como inválidos. Volver a multiplicarlo por `position` fabricaría
una atribución. BL-07 debe incluir v11 con estado `UNAVAILABLE` y razón explícita,
nunca con un proxy numérico, hasta que exista el join PIT del retorno subyacente.

## Qué falta exactamente
Script DESECHABLE `scripts/analysis/timing_ratio_oneoff.py`:
pnl_timing=cov(peso,retorno) vs beta por campeona (v11, gold_trend_simple,
btc_hodl_b1, spx ma200/gated), IC bootstrap en bloques. Para v11 imprime
`UNAVAILABLE` con la limitación anterior; no inventa retorno subyacente. Imprime;
NO persiste tablas. Aplica el guardarraíl constitucional de muestra pequeña:
si `n_trades < 20` (BTC actualmente tiene 1), publica solo conteos y PnL con estado
`INSUFFICIENT_TRADES`; suprime `timing_ratio`, covarianza e intervalo.

## Impacto frontend
Ninguno en esta etapa (la persistencia es BL-22).

## Dependencias
—

## Verificación
Corre en <5 min; incluye las cuatro campeonas (v11 honestamente `UNAVAILABLE`) y
replica a mano el diagnóstico gold_dynamic_exit (timing≈0 ⇒ beta disfrazado; el IC
bootstrap predefinido contiene cero). BTC N=1 no publica ratio ni IC.

## Evidencia de implementación (2026-07-27)

- Implementación inmutable:
  `d0427d670ff50f8179a29de2b01032e0324202e7`.
- Paquete de revisión:
  `.claude/coordination/reviews/BL-07.md`, sellado en `ed11c9a`.
- TDD: 7 tests unitarios verdes; el candado N<20 fue demostrado rojo antes del
  guard y verde después.
- Corrida real: 5.000 muestras bootstrap en 17.466 s; v11 `UNAVAILABLE`, BTC
  `INSUFFICIENT_TRADES` con N=1 y sin ratio/IC, control Oro con
  `timing_ratio=0.0206378474` e IC `[-0.0076845889, 0.0486331161]`.
- Monitores: front-matter sin delta frente al baseline; manifests 9 verdes;
  scripts-layout 20 verdes.
- Cross-review independiente: Claude `CLD-118`, `APROBADO` contra el hash
  completo, reproducción exacta e idempotencia byte a byte.

## Notas constitución
§18.3: timing_ratio es atribución diagnóstica, JAMÁS prueba de alfa (el claim sigue siendo DSR forward).
