---
kind: roadmap
status: PLANNED
version: 1.1.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - registries/ledger.jsonl
  - registries/families/usdcop_direction.yaml
  - scripts/validation/check_trial_ledger.py
---

# BL-10 — Backfill legacy_estimate de FT históricos (zoos)

**Fuente**: FABRIC §10.2 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
El borrador heredado ya separó 237 asientos (53 FT/184 AT), pero el plan conservaba el
conteo obsoleto COP=88. El SSOT sellado por BL-12-r2 demuestra 111: el cuerpo registra
H1 daily shadow 109→110 y H1 LatAm transport 110→111. El ledger quedó en 109 y dos
celdas `legacy_estimate` no declaraban la etiqueta en su propia nota.

## Qué falta exactamente
Reconciliar sin modelado ni trials nuevos: exigir `legacy_estimate` por celda backfilled,
añadir al final FT-0054/55 como trials `documented` ya cobrados, extender
`usdcop_direction` a 50 y preservar la igualdad exacta por activo. No tocar
HYPOTHESIS-REGISTRY ni el EXP-DIR modificado por otra línea.

## Impacto frontend
Ninguno.

## Dependencias
BL-09.

## Verificación
`pytest tests/regression/test_bl10_legacy_estimate_contract.py
tests/regression/test_trial_ledger.py`; `python scripts/validation/check_trial_ledger.py`.
Debe resultar 239 globales (55 FT/184 AT), COP=111, hash-chain íntegra y
`legacy_estimate` en cada celda que cubra líneas estimadas.

## Notas constitución
'Un N estimado documentado vale infinitamente más que un N=0 falso'.
