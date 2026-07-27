---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - airflow/dags/l0_macro_update.py
  - database/migrations/067_spx500_regime_macro_vars.sql
---

# BL-24 — Linaje nodes/edges + camino dorado + revisiones tipificadas

**Fuente**: FABRIC §22 + §28 E6 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Sin grafo de linaje. CRÍTICO: NFCI/claims/SAHM (ALFRED-revisadas) ya se ingieren — una corrección de proveedor hoy no distingue de release legítimo.

## Qué falta exactamente
DDL §22.2, lineage.emit() desde scripts, cascada STALE selectiva, ramas as_released/latest_revised, revision_type en macro ingest. Consulta: fila de fact_pnl → raw snapshot.

## Impacto frontend
Passport muestra linaje (BL-32).

## Dependencias
BL-17.

## Verificación
Camino dorado resuelve para 1 señal del paper ledger; LEGITIMATE_RELEASE no marca STALE histórico.

## Notas constitución
Screening consume as_released por defecto (PIT-correcta).
