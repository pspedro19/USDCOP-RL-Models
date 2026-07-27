---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - scripts/pipeline/generate_weekly_forecasts.py
---

# BL-10 — Backfill legacy_estimate de FT históricos (zoos)

**Fuente**: FABRIC §10.2 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Los zoos existentes (COP 9×7≈63 celdas miradas, Oro 9, BTC 9) no tienen FT asignados; los 88 de COP no están etiquetados por familia FT/AT.

## Qué falta exactamente
Asignar FT- retroactivos con etiqueta `legacy_estimate` documentada por escrito; etiquetar los 88 COP por familia SIN cambiar el conteo total.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación
Suma de FT+AT por activo == n_trials_total previo; nota legacy_estimate en cada familia backfilled.

## Notas constitución
'Un N estimado documentado vale infinitamente más que un N=0 falso'.
