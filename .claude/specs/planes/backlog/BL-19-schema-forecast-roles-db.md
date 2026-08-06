---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - database/migrations/068_deep_macro_credit_fedfunds.sql
---

# BL-19 — Esquema DB forecast.* + rol forecast_writer

**Fuente**: plan 01 §2 / FABRIC §15.3 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: la migración 071 declara `forecast.*`, la separación action/exec y el rol mínimo `forecast_writer`; los invocadores del migrador ya seleccionan el plan `fabric-v1` explícitamente. El plan no está aplicado en la base viva y ningún test ejecuta `SET ROLE` contra PostgreSQL, así que la muralla sigue sin verificación física.

## Qué falta exactamente
Aplicar `fabric-v1` en un fixture PostgreSQL y demostrar con `SET ROLE forecast_writer` que forecast admite su escritura y action/exec la deniegan. Falta también migrar los writers/readers productivos fuera de las tablas de compatibilidad en `public`.

## Impacto frontend
Ninguno.

## Dependencias
BL-15.

## Verificación
`SET ROLE forecast_writer; INSERT INTO action...` ⇒ denegado (test).

## Notas constitución
Primera muralla física; action/exec schemas llegan con BL-21/22.
