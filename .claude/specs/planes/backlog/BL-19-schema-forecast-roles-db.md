---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-06
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - database/migrations/068_deep_macro_credit_fedfunds.sql
---

# BL-19 — Esquema DB forecast.* + rol forecast_writer

**Fuente**: plan 01 §2 / FABRIC §15.3 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-08-06)
Entrega parcial: la migración 071 declara `forecast.*`, la separación action/exec y el rol mínimo `forecast_writer`; los invocadores del migrador seleccionan el plan `fabric-v1` explícitamente. La base viva ya contiene el rol y las tablas `forecast.*`. Una sonda transaccional con `SET LOCAL ROLE forecast_writer` insertó una fila válida en `forecast.forecast_output` y observó `count = 1`; después de `ROLLBACK`, `count = 0`. Con el mismo rol, `INSERT INTO exec.order_header DEFAULT VALUES` falló con `permission denied for schema exec`.

## Qué falta exactamente
La mitad física `forecast`→permitido / `exec`→denegado ya está demostrada contra PostgreSQL real y sin residuo. Falta convertirla en un test repetible de CI y completar la negativa sobre una tabla `action`: el esquema existe, pero `action.strategy_signal` todavía no existe en la base viva, por lo que un `INSERT` allí probaría ausencia de objeto y no denegación de privilegios. Falta también migrar los writers/readers productivos fuera de las tablas de compatibilidad en `public`; la existencia de tablas y grants no sustituye ese cableado.

## Impacto frontend
Ninguno.

## Dependencias
BL-15.

## Verificación
`SET ROLE forecast_writer; INSERT INTO action...` ⇒ denegado (test).

## Notas constitución
Primera muralla física; action/exec schemas llegan con BL-21/22.
