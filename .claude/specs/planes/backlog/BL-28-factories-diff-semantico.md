---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - airflow/dags/asset_pipeline_factory.py
  - config/assets/pipelines.yaml
---

# BL-28 — Factories nuevas (data/strategy/forecast) + diff semántico

**Fuente**: FABRIC §13 + §28 E7 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Factory actual: 1 DAG por ACTIVO con estrategias adentro (decisión RECHAZADA §31 como estado final); COP artesanal con ExternalTaskSensor.

## Qué falta exactamente
Generadores A/B/D (§13.1-13.3) en PARALELO al factory actual: DAG por sleeve (aislamiento de fallas/timeouts), Assets con shim Airflow2/3, pools por API externa, guard de muralla en el generador. Criterio E7: bundles idénticos (semantic_hash) viejo vs nuevo.

## Impacto frontend
Ninguno directo.

## Dependencias
BL-17.

## Verificación
Diff semántico verde ≥2 semanas antes de apagar el camino viejo por activo.

## Notas constitución
Backfill SIEMPRE en DAG aparte con as_of explícito.
