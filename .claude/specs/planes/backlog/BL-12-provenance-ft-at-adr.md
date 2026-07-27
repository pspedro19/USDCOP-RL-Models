---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/rules/quant-constitution.md
---

# BL-12 — Provenance FT→AT + enmienda constitución §2 (ADR)

**Fuente**: plan 02 §3-§4 / FABRIC §10.1-§10.2 · **Ola**: 2 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
La práctica ya se siguió (H-META-01 cobró AT al convertir consenso en sizing) pero la regla no está escrita como dura, ni existe el campo provenance.

## Qué falta exactamente
ADR menor + regla en quant-constitution §2: 'convertir un forecast en señal económica = +1 AT; los FT del predictor viajan en provenance y entran al N_cluster'. Campo `provenance:` en pre-registros nuevos.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación
Regla publicada; próximo pre-registro la usa.

## Notas constitución
Cambiar la constitución requiere ADR (su propia cabecera lo exige).
