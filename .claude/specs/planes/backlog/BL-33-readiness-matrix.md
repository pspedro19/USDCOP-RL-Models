---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/planes/03-institutional-readiness.md
---

# BL-33 — Institutional Readiness Matrix con evidencias

**Fuente**: plan 03 §6-§7 · **Ola**: T · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Plan 03 define las 4 clases de evidencia y el mapeo as-built inicial (kill-switch ✓, PreTradeGate ✓, RBAC ✓, restore frío ✓; deudas: .env histórico, identidad única, NAV, DR).

## Qué falta exactamente
`04b-readiness-matrix.md`: una fila por control (tecnología/riesgo/ejecución/seguridad/compliance/ops/inversionistas) con evidencia esperada, estado, dueño, fecha. Primeras filas: BL-08 y segregación de identidades.

## Impacto frontend
Opcional: sección admin de solo lectura.

## Dependencias
—(vive de evidencias de todos los demás).

## Verificación
Cada fila enlaza evidencia verificable (commit/test/simulacro), no intención.

## Notas constitución
'No basta con marcar etapas como completadas. Debes producir evidencia.'
