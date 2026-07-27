---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md
  - .claude/rules/quant-constitution.md
---

# BL-11 — Familias transversales de hipótesis (registries/families/)

**Fuente**: FABRIC §9.4-§9.5 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
La misma mecánica probada en 3 activos vive hoy en 3 registries que subestiman la deflación cruzada.

## Qué falta exactamente
`registries/families/{family}.yaml` con TODAS las celdas (asset×variant), bar pre-firmado, trials_charged validado vs ledger. Piloto: familia `trend_regime` con celdas SPX/Oro/BTC reales existentes.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación
CI: celdas con trial_id == ledger; cerrar familia por escrito es un estado válido.

## Notas constitución
Dividir familias para lavar multiplicidad queda visible vía cluster + N_global (decisión rechazada §31).
