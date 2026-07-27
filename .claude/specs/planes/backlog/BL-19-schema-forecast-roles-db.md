---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - database/migrations/068_deep_macro_credit_fedfunds.sql
---

# BL-19 — Esquema DB forecast.* + rol forecast_writer

**Fuente**: plan 01 §2 / FABRIC §15.3 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Todo vive en schema public con un usuario de app. La muralla no es física.

## Qué falta exactamente
Migración: schema `forecast` (forecast_output, forecast_score, model_horizon_result, calibration_result) + rol forecast_writer SIN INSERT en action/exec. Test de permisos CONTRA LA BASE REAL en CI.

## Impacto frontend
Ninguno.

## Dependencias
BL-15.

## Verificación
`SET ROLE forecast_writer; INSERT INTO action...` ⇒ denegado (test).

## Notas constitución
Primera muralla física; action/exec schemas llegan con BL-21/22.
