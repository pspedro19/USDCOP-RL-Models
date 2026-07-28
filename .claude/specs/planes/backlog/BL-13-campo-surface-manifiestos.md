---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - config/strategy_manifests/usdcop.yaml
  - scripts/pipeline/normalize_champions.py
---

# BL-13 — Campo surface en manifiestos/registry + normalize lo respeta

**Fuente**: plan 00 §2-§3 / FABRIC §9.2-§9.3 · **Ola**: 2 · **Esfuerzo**: S · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Ni manifiestos ni registry.json declaran surface; la muralla es implícita (por convención de pipelines).

## Qué falta exactamente
`surface: action|diagnostic` en manifiestos + entradas de registry; normalize_champions rechaza CHAMPION para diagnostic; re-freeze consciente de manifiestos (campo aditivo, señal intacta).

## Impacto frontend
Registry API expone surface (futuros filtros de vistas).

## Dependencias
—

## Verificación
Test: entrada diagnostic con status CHAMPION ⇒ CI rojo.

## Notas constitución
Re-freeze de manifiesto = bump versión + nota (patrón refreeze_note_v8 existente).
