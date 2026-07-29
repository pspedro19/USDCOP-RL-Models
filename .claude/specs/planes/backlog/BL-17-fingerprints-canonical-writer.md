---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - src/contracts/strategy_manifest.py
  - services/common/metrics.py
---

# BL-17 — Identidad: fingerprints + canonical writer + spine

**Fuente**: FABRIC §8 + §28 E2 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Entrega parcial: `src/identity/fingerprints.py` implementa fingerprints con separación de dominio y `src/identity/canonical.py` aporta el writer canónico. Las invariantes de representación (incluidas LF/CRLF y números equivalentes) tienen cobertura focal. La spine productiva y el replay independiente del ledger todavía no están cableados.

## Qué falta exactamente
Completar la spine mínima con productores/consumidores reales y un gate que reconstruya desde cero el `semantic_hash` de un paper ledger anclado. Mutar una fila del ledger debe romper la reproducción y nombrar ambos hashes.

## Impacto frontend
Ninguno directo.

## Dependencias
BL-16.

## Verificación
Mismo decision_fingerprint ⇒ mismo semantic_hash (test); derivation_id igual con semantic distinto ⇒ incidente.

## Notas constitución
E2 PRECEDE hechos y backfills (lección del roadmap auditado — evita re-trabajo de spine).
