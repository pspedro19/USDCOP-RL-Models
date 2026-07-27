---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
---

# BL-02 — Banner fuerte en Gold weekly-inference

**Fuente**: plan 00 §5 / FABRIC §24.3 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El caveat solo renderiza si `isModelZoo` (~:1057). Gold en modo weekly_inference muestra badges direccionales con colores (~:781,:879) SIN el banner fuerte — solo la nota suave de metodología (~:811-816).

## Qué falta exactamente
Extender el banner 'DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN' a TODA superficie de forecasting, incluido weekly_inference.

## Impacto frontend
`/forecasting?asset=xauusd` gana el banner ámbar permanente.

## Dependencias
BL-01 (el test debe cubrir ambos modos).

## Verificación
Banner visible con asset=xauusd; test de BL-01 lo exige por modo.

## Notas constitución
La muralla es por superficie, no por asset ni por modo de render.
