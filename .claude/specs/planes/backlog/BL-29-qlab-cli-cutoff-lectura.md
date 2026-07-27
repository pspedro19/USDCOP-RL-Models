---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - scripts/analysis/profitability_evidence.py
  - .claude/rules/quant-constitution.md
---

# BL-29 — CLI qlab + cutoff impuesto por la capa de lectura

**Fuente**: FABRIC §13.5 · **Ola**: 5 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
La investigación corre como scripts sueltos con disciplina humana del cutoff; los trials se cobran editando registries a mano.

## Qué falta exactamente
`qlab` (family declare/screen --charge-trial/freeze/promote/close) FUERA de Airflow ('un trial no es idempotente'); capa read() con assert available_at<=cutoff cuando env=screening — el look-ahead falla el JOB, no al humano. Tabla de entornos §13.5.

## Impacto frontend
Ninguno.

## Dependencias
BL-09, BL-11, BL-19 (o vista equivalente con available_at).

## Verificación
Screening intentando leer >cutoff ⇒ excepción; retry de qlab no duplica cobro (ledger idempotente por trial_id).

## Notas constitución
El control de mayor apalancamiento de todo el sistema.
