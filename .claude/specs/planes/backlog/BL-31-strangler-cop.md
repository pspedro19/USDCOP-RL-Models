---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - airflow/dags/forecast_h5_l3_weekly_training.py
  - airflow/dags/forecast_h5_l7_multiday_executor.py
---

# BL-31 — Migración strangler de USD/COP (L7 al final)

**Fuente**: FABRIC §29 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
COP artesanal L0→L7 en producción con A/B vivo (v11/v12/v14) — intocable durante la migración; sus ledgers SON el patrón de paridad.

## Qué falta exactamente
Orden ingest→…→execute; paralelo ≥2 semanas por capa con paridad semantic_hash; sensores→Assets al migrar cada capa; L7 al FINAL (resto ≥1 mes verde + ejecución externa probada en canary de otro flujo); rollback por capa declarado.

## Impacto frontend
Ninguno (paridad invisible si sale bien).

## Dependencias
BL-28, BL-30, BL-17.

## Verificación
Tabla de paridad por capa en evidencia; ninguna capa avanza sin la anterior verde sostenida.

## Notas constitución
La cadena con dinero real migra última y con red doble.
