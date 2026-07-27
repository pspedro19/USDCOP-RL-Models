---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - airflow/dags/asset_pipeline_factory.py
---

# BL-35 — URIs de datasets + arista prohibida forecast→allocator en parseo

**Fuente**: plan 01 §3 / FABRIC §3.4, §9.8 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
No existe convención de URIs (asset://, strategy://, forecast://); la prohibición forecast→allocator no es verificable en el grafo.

## Qué falta exactamente
Esquema de URIs en los Assets de Airflow (shim 2/3); test de parseo: cualquier DAG que consuma forecast://*/prediction hacia book/exec ⇒ falla el parse + CI.

## Impacto frontend
Ninguno.

## Dependencias
BL-13; base para BL-28.

## Verificación
DAG sintético violador ⇒ import error visible en list-import-errors.

## Notas constitución
'Falla el parseo de DAGs y el CI' — la muralla en el grafo, no en la disciplina.
