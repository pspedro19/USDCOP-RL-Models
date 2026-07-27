---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/components/gm/views/ProductionView.tsx
  - scripts/pipeline/candidates_paper_ledger.py
---

# BL-05 — ProductionView consume el paper ledger (A/B v11/v12/v14)

**Fuente**: FABRIC §24.5 Control Tower / plan 00 §2 · **Ola**: 1 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El ledger existe y se refresca cada viernes vía L6 (`candidates_ledger_2026.json`, anclado a ene-2026, judge_window post-freeze). ProductionView NO lo referencia — el A/B vivo es invisible en la UI.

## Qué falta exactamente
Panel en /production: tabla candidatas (v11 real vs v12/v14 paper), judge_window post-2026-07-21 con N y nota N<20, días al juez. Solo lectura del JSON publicado.

## Impacto frontend
`/production` gana el panel A/B. Cero botones (read-only por invariante).

## Dependencias
—

## Verificación
Panel renderiza el JSON real; N<20 muestra solo conteo/PnL.

## Notas constitución
Vote-2/decisiones siguen sobre bundles; esto es monitoreo del juez sellado — jamás re-anclar.
