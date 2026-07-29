---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - services/signalbridge_api/app/services/pretrade.py
  - src/risk
---

# BL-30 — Servicio de ejecución fuera de Airflow + pre-trade + kill switch independiente

**Fuente**: FABRIC §21 + §30 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: `src/execution/service.py` y la migración 078 implementan un coordinador independiente, reconciliación y niveles de kill switch sobre `portfolio_target`. No hay wiring en services/Airflow/scripts; `ExecutionService` no ofrece todavía un `exit_all` independiente de un target vigente y `_pretrade` trata controles ausentes como nominales.

## Qué falta exactamente
Hacer fail-closed la ausencia de controles, añadir cierre total por cuenta sin depender de un target vigente y cablear el servicio real fuera de Airflow. Faltan el simulacro con Airflow apagado, la persistencia/reconciliación PostgreSQL y el retiro del camino económico duplicado de SignalBridge.

## Impacto frontend
/execution muestra niveles del kill switch y reconciliación.

## Dependencias
BL-21, BL-26; gate final = 15 criterios §30.

## Verificación
Simulacro: Airflow apagado ⇒ kill switch operativo; discrepancia broker ⇒ QUARANTINED.

## Notas constitución
Airflow publica targets; NUNCA envía órdenes.
