---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx
---

# BL-03 — Wording probabilístico y neutralización de verde/rojo en predicciones

**Fuente**: plan 00 §5 / FABRIC §24.3 · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
`DIRECTION_TONE` mapea UP→pos(verde)/DOWN→neg(rojo) (:61-63) y se aplica a PREDICCIONES en :524,:536,:584-589,:709,:781,:879. No hay verbos imperativos (verificado), pero la semántica de color compra/vende está presente en la superficie diagnóstica.

## Qué falta exactamente
En superficies DIAGNOSTIC: mostrar 'probabilidad estimada de subida: NN%' y tonos neutros (o un único acento) en badges de predicción; los tonos pos/neg quedan reservados a superficies ACTION (replay/production), donde LONG/SHORT son decisiones auditables.

## Impacto frontend
Cambia la paleta de los badges/cards de predicción en `/forecasting` (todas las assets). Sin cambio en /dashboard ni /production.

## Dependencias
BL-01/BL-02 (tests actualizados con el nuevo wording).

## Verificación
Grep: `DIRECTION_TONE` sin usos en ramas de forecast; snapshot visual.

## Notas constitución
FABRIC §24.3: 'sin colores ni etiquetas imperativas'. No es cosmética: es la barrera visible.
