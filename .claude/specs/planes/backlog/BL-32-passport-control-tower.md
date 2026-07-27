---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/HubView.tsx
  - usdcop-trading-dashboard/components/gm/views/ProductionView.tsx
---

# BL-32 — Passport (vista live + MV) + Control Tower

**Fuente**: FABRIC §24.4-§24.5 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Hub/Production muestran piezas; no existe el SELECT único (identidad+gobierno+linaje+desempeño 5 entornos+riesgo) ni la portada de 10 segundos.

## Qué falta exactamente
v_strategy_passport_live (NO materializada) + mv_strategy_performance_daily (nocturna) + composición; Control Tower: LIBRO/SLEEVES/DATOS (§24.5) incl. N_global vs N_MAX, test pareado v11 vs v12/v14 con p/e-value, semáforo de retiro.

## Impacto frontend
Página/sección nueva (o evolución de /hub) — la cara del sistema.

## Dependencias
BL-18, BL-22, BL-24; BL-05 es el primer ladrillo.

## Verificación
El paseo Airflow→MLflow→JSONs→SQL se reemplaza por un SELECT (demo).

## Notas constitución
MV de Postgres no puede sostener estado live (se reemplaza en refresh) — por eso la división.
