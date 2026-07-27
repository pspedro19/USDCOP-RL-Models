---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - database/migrations/068_deep_macro_credit_fedfunds.sql
  - scripts/pipeline/candidates_paper_ledger.py
---

# BL-22 — fact_position / fact_pnl + identidad contable + timing_ratio persistido

**Fuente**: FABRIC §18 + §28 E3 · **Ola**: 4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
PnL vive en bundles JSON por estrategia; no hay tablas de hechos ni descomposición beta/timing/carry.

## Qué falta exactamente
DDL §18.1 (env en PK: paper y live conviven), identidad contable como test de CI (|residual|/|gross| ≤ tol), timing_ratio con IC como métrica del catálogo.

## Impacto frontend
Production/Passport leen facts (v11-live vs v12-paper = una query).

## Dependencias
BL-17, BL-18, BL-21.

## Verificación
Identidad contable verde sobre el paper ledger anclado.

## Notas constitución
El PnL operativo NO es el NAV legal (plan 03 §3.4).
