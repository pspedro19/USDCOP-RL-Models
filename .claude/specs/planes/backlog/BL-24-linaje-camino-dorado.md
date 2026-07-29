---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - airflow/dags/l0_macro_update.py
  - database/migrations/067_spx500_regime_macro_vars.sql
---

# BL-24 — Linaje nodes/edges + camino dorado + revisiones tipificadas

**Fuente**: FABRIC §22 + §28 E6 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: la migración 076 y `src/lineage/graph.py` definen nodos, aristas y tipos de revisión. No hay emisores desde DAGs/scripts, resolución de camino dorado ni integración de `revision_event` en la ingesta macro; una corrección de proveedor aún no se distingue end-to-end de un release legítimo.

## Qué falta exactamente
Cablear un caso completo desde una señal real del paper ledger hasta su snapshot/barra L0, empezando por el emisor de revisiones en la ingesta macro. El resolvedor debe fallar ante una arista intermedia ausente y `LEGITIMATE_RELEASE` no debe marcar historia como STALE.

## Impacto frontend
Passport muestra linaje (BL-32).

## Dependencias
BL-17.

## Verificación
Camino dorado resuelve para 1 señal del paper ledger; LEGITIMATE_RELEASE no marca STALE histórico.

## Notas constitución
Screening consume as_released por defecto (PIT-correcta).
