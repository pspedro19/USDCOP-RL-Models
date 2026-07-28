---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app/dashboard/page.tsx
  - usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx
---

# BL-34 — Ruta /replay (alias de la sección de /dashboard)

**Fuente**: plan 00 §2 / FABRIC §24.2 · **Ola**: T · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El replay completo vive como ForecastingBacktestSection dentro de /dashboard; no existe app/replay/. Funcional hoy; el gap es de navegación/nomenclatura vs la constitución.

## Qué falta exactamente
Ruta /replay que monte la misma sección (o redirect), RBAC en la matriz, nav actualizada. Vote-2 se QUEDA en /dashboard.

## Impacto frontend
Nueva entrada de navegación; cero lógica nueva.

## Dependencias
Al final (cosmético); tras BL-05.

## Verificación
rbac:check verde; /replay renderiza selector estrategia/versión + equity.

## Notas constitución
No mover los botones de aprobación: /replay es lectura.
