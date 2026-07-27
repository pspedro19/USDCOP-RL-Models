---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
  - tests/regression/test_knowledge_frontmatter.py
---

# BL-01 — Test de regresión del caveat `da-caveat`

**Fuente**: plan 00 §5 / FABRIC §24.3 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El banner existe (`ForecastingView.tsx` ~:1057-1076, texto 'Superficie de diagnóstico, no de señales…') pero NO hay ningún test que lo proteja (grep en tests/: cero asserts sobre `da-caveat`).

## Qué falta exactamente
Test (grep estático o e2e) que falle si el banner o su `data-testid="da-caveat"` desaparece o cambia a texto sin la frase 'no es una señal/no de señales'.

## Impacto frontend
Ninguno visible — es un candado. Congela el contrato honesto de la vista.

## Dependencias
—

## Verificación
`pytest` nuevo test rojo al comentar el banner, verde con él.

## Notas constitución
La honestidad publicada es parte del producto (FABRIC §3.2); sin test es disciplina humana.
