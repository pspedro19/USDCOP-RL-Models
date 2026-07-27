---
kind: roadmap
status: PLANNED
version: 1.0.0
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

## Qué falta exactamente
Script DESECHABLE `scripts/analysis/timing_ratio_oneoff.py`: pnl_timing=cov(peso,retorno) vs beta por campeona (v11, gold_trend_simple, btc_hodl_b1, spx ma200/gated), IC bootstrap en bloques. Imprime; NO persiste tablas.

## Impacto frontend
Ninguno en esta etapa (la persistencia es BL-22).

## Dependencias
—

## Verificación
Corre en <5 min; replica a mano el diagnóstico gold_dynamic_exit (timing≈0 ⇒ beta disfrazado).

## Notas constitución
§18.3: timing_ratio es atribución diagnóstica, JAMÁS prueba de alfa (el claim sigue siendo DSR forward).
