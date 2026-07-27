---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
  - usdcop-trading-dashboard/app/api/data/[...path]/route.ts
---

# BL-06 — CI muralla frontend: forecasting sin aprobar/ejecutar

**Fuente**: plan 01 §4 / FABRIC §25 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Verificado limpio HOY: ForecastingView sin botones aprobar ni endpoints de ejecución (único link: /pricing). Sin test que lo candadee.

## Qué falta exactamente
Test estático: en ForecastingView (y components/forecasting/*) prohibidos `\/api\/production\/approve`, `\/api\/execution`, `onApprove`, verbos COMPRAR/VENDER.

## Impacto frontend
Ninguno — candado.

## Dependencias
—

## Verificación
Test rojo al inyectar un fetch de approve en la vista.

## Notas constitución
FABRIC §25 bloque Muralla: 'frontend forecasting sin verbos de orden…'.
