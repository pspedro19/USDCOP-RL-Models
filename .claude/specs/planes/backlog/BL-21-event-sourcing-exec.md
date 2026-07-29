---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - services/signalbridge_api
  - src/execution
---

# BL-21 — Event sourcing exec.* (4 entornos) + idempotencia

**Fuente**: FABRIC §17 + §28 E3 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: la migración 074 define el ledger event-sourced y `src/execution/events.py` sus contratos e idempotencia. No hay writer productivo a esas tablas ni prueba de fencing sobre la función SQL real. La preimagen actual de `order_idempotency_key` tampoco discrimina sleeve/exposición, por lo que dos sleeves del mismo instrumento pueden colisionar.

## Qué falta exactamente
Incluir el discriminante de sleeve/exposición en la identidad, escribir simulador y broker en las mismas tablas y probar concurrencia contra PostgreSQL real. Dos sleeves sobre el mismo instrumento deben producir dos claves/órdenes; dos claims del mismo dispatch deben producir un ganador y un replay idempotente.

## Impacto frontend
/execution puede migrar a leer proyecciones de eventos.

## Dependencias
BL-17.

## Verificación
Retry inyectado ⇒ cero órdenes duplicadas (test §30.3).

## Notas constitución
Estado de orden = proyección de eventos, nunca UPDATE destructivo.
