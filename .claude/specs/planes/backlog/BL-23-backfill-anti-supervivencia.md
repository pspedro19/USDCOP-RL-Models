---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/public/data/strategies
  - usdcop-trading-dashboard/public/data/registry.json
---

# BL-23 — Backfill anti-supervivencia (campeonas+candidatas+retiradas+baselines)

**Fuente**: FABRIC §28 E5 · **Ola**: 4 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Paridad 2025/2026 hecha en versiones VIGENTES (2026-07-27); versiones superseded quedaron como freezes históricos 2026-only; retiradas preservadas con bundles.

## Qué falta exactamente
Reconstruir facts/metric_event para TODO el catálogo (incl. archived) para que la Control Tower no tenga sesgo de supervivencia.

## Impacto frontend
Vistas históricas completas (dropdowns sin huecos).

## Dependencias
BL-22.

## Verificación
Query: toda estrategia del registry tiene facts en sus años publicados.

## Notas constitución
'Backfill exclusivo de campeonas' es decisión RECHAZADA (§31).
