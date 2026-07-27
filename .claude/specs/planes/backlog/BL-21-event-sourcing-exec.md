---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - services/signalbridge_api
  - src/execution
---

# BL-21 — Event sourcing exec.* (4 entornos) + idempotencia

**Fuente**: FABRIC §17 + §28 E3 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
SignalBridge tiene órdenes/paper propios; no existe el ledger evento-sourced común (order_header/status_event/fill_event) ni idempotency_key formal.

## Qué falta exactamente
DDL §17.1 + escritura del simulador determinista en las MISMAS tablas (executor_type=deterministic_simulator); idempotency_key=SHA256(account⊕instrument⊕target_version⊕decision_fp⊕cutoff) con UNIQUE.

## Impacto frontend
/execution puede migrar a leer proyecciones de eventos.

## Dependencias
BL-17.

## Verificación
Retry inyectado ⇒ cero órdenes duplicadas (test §30.3).

## Notas constitución
Estado de orden = proyección de eventos, nunca UPDATE destructivo.
