---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-06
supersedes: []
code_anchors:
  - database/migrations/071_forecast_schema_roles.sql
  - scripts/ops/db_migrate.py
  - Makefile
---

# BL-19 — Esquema DB forecast.* + rol forecast_writer

**Fuente**: plan 01 §2 / FABRIC §15.3 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-08-06)
Entrega parcial: la migración 071 declara `forecast.*`, la separación action/exec y el rol mínimo `forecast_writer`; los invocadores del migrador seleccionan el plan `fabric-v1` explícitamente. La base viva ya contiene el rol y las tablas `forecast.*`. Una sonda transaccional con `SET LOCAL ROLE forecast_writer` insertó una fila válida en `forecast.forecast_output` y observó `count = 1`; después de `ROLLBACK`, `count = 0`. Con el mismo rol, `INSERT INTO exec.order_header DEFAULT VALUES` falló con `permission denied for schema exec`.

## Qué falta exactamente
La sonda puntual demuestra `forecast`→permitido y `exec`→denegado, pero todavía no es un gate repetible de CI. La negativa literal sobre una tabla `action` tampoco puede ejecutarse honestamente: `action.strategy_signal` no existe en la base viva, por lo que hoy un `INSERT` probaría ausencia de objeto, no denegación de privilegios. Falta además migrar los writers/readers productivos fuera de `public.forecast_h5_*`; la existencia de tablas y grants no sustituye ese cableado.

## Impacto frontend
Ninguno.

## Dependencias
BL-15.

## Verificación
`SET ROLE forecast_writer; INSERT INTO action...` ⇒ denegado (test).

## Notas constitución
Primera muralla física; action/exec schemas llegan con BL-21/22.
