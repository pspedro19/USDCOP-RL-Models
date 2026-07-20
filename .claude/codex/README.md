---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - README.md
  - CLAUDE.md
  - .claude/specs/audit/AUDIT-2026-07-remediation.md
  - .claude/specs/audit/STRATEGIC-ASSESSMENT-2026-07.md
  - usdcop-trading-dashboard/package.json
---

# Codex — Hallazgos de specs

Esta carpeta conserva las revisiones independientes de Codex sobre las especificaciones del sistema.
Su objetivo es registrar diferencias entre lo documentado, lo implementado y lo demostrado mediante
pruebas, sin reemplazar los SSOT existentes en `.claude/rules/` y `.claude/specs/`.

## Reglas de trabajo

1. Cada afirmación debe apuntar a una spec, archivo de código, configuración, prueba o resultado medido.
2. Se debe distinguir entre `IMPLEMENTED`, `OPERATIONAL`, `PAPER_ONLY`, `PAUSED`, `EXPERIMENTAL` y
   `DESIGNED`.
3. Un resultado de backtest no se describe como evidencia de rentabilidad futura.
4. Los hallazgos no modifican por sí solos el contrato oficial; su incorporación requiere actualizar el
   SSOT correspondiente y sus pruebas.
5. Todo hallazgo debe indicar severidad, impacto, recomendación y estado.

## Índice

- [`SPEC-FINDINGS-2026-07-20.md`](./SPEC-FINDINGS-2026-07-20.md) — auditoría inicial de plataforma,
  frontend, backend, trading y forecasting.

