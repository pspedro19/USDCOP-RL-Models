---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/forecasting/ForecastingDashboard.tsx
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
---

# BL-04 — Unificar caveat duplicado en legacy ForecastingDashboard

**Fuente**: hallazgo Explore 2026-07-27 · **Ola**: 1 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
`ForecastingDashboard.tsx:123,:132` duplica el banner (texto casi igual) y sigue alcanzable vía `/legacy/forecasting`. Dos implementaciones = deriva garantizada.

## Qué falta exactamente
Extraer el texto del caveat a una constante compartida (lib/) consumida por ambas vistas, o congelar el legacy con nota de superseded.

## Impacto frontend
`/legacy/forecasting` (admin-only) queda alineado o congelado.

## Dependencias
BL-01.

## Verificación
Una sola fuente del string; test BL-01 cubre ambas rutas o el legacy queda excluido explícitamente.

## Notas constitución
Regla 6 FABRIC: toda métrica/mensaje de gobierno con definición única.
