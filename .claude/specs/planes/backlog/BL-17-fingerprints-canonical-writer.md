---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - src/contracts/strategy_manifest.py
  - services/common/metrics.py
---

# BL-17 — Identidad: fingerprints + canonical writer + spine

**Fuente**: FABRIC §8 + §28 E2 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Hoy: code_hash de manifiestos + sha16 de ledgers puntuales. No hay spec/decision/execution_fingerprint ni writer canónico.

## Qué falta exactamente
Módulo identity: los 6 conceptos (§8.1), writer canónico (semantic_hash==bytes_hash por construcción en JSON propios), spine mínimo resoluble. Gate CI: replay independiente reproduce el semantic_hash del paper ledger anclado.

## Impacto frontend
Ninguno directo.

## Dependencias
BL-16.

## Verificación
Mismo decision_fingerprint ⇒ mismo semantic_hash (test); derivation_id igual con semantic distinto ⇒ incidente.

## Notas constitución
E2 PRECEDE hechos y backfills (lección del roadmap auditado — evita re-trabajo de spine).
