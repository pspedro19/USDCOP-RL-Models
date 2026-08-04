---
kind: roadmap
status: IMPLEMENTED
version: 1.2.0
last_verified: 2026-08-04
supersedes: []
code_anchors:
  - src/contracts/strategy_manifest.py
  - services/common/metrics.py
  - src/identity/ledger_replay.py
  - src/identity/candidate_ledger.py
  - src/market/publication.py
  - scripts/data/seed_reference_spine.py
  - scripts/pipeline/candidates_paper_ledger.py
  - scripts/validation/check_candidate_ledger_identity.py
---

# BL-17 — Identidad: fingerprints + canonical writer + spine

**Fuente**: FABRIC §8 + §28 E2 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-08-04)

Implementado. La identidad canónica conserva separación de dominio e invariantes de
representación. La spine se deriva de AssetProfiles y hechos observados; el publisher de mercado
la consume para resolver alias antes de persistir raw/canonical/quarantine. El replay DB compromete
toda columna persistida salvo los campos técnicos declarados, y el ledger JSON lleva un envelope
sellado que el comando independiente recompone desde el payload.

## Evidencia de cierre

- Spine R2 `cfba9cb7`: autoridad derivada del SSOT y procedencia medida sin confundirlas; CODEX la
  aprobó con consultas read-only en `CXD-453`.
- Replay DB R2 `c0561ecb`: la cobertura se contrasta con `information_schema`; mutar
  `running_da_pct` cambia el hash. CODEX lo aprobó en `CXD-456`.
- Consumidor real `b432d7e9` + candado causal `566af600`: el publisher resuelve la spine y sólo
  el frame aceptado llega al writer legado. CLAUDE reprodujo los ataques de bypass y aprobó en
  `CLD-452`.
- Envelope JSON `4dea8c9` + R2 `0efee96a`: mutar el payload rompe la reproducción y nombra hash
  esperado/obtenido; `generated_at` es deliberadamente volátil; cambiar el productor rompe sólo
  `derivation_id`; desconectar el sellado o volcar otra variable pone el candado rojo. CLAUDE
  aprobó en `CLD-453` y el validador independiente termina con exit 0.

## Impacto frontend
Ninguno directo.

## Dependencias
BL-16.

## Verificación
Mismo decision_fingerprint ⇒ mismo semantic_hash (test); derivation_id igual con semantic distinto ⇒ incidente.

## Notas constitución
E2 PRECEDE hechos y backfills (lección del roadmap auditado — evita re-trabajo de spine).
