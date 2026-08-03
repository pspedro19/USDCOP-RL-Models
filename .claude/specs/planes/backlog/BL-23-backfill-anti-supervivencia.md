---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-03
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

## Estado real verificado 2026-08-03

`scripts/data/backfill_catalog_facts.py` ya recorre el registro completo sin filtrar por estado,
incluye retiradas y baselines, y por defecto solo construye un plan. La ejecución contra el
catálogo real terminó con `missing: []`; el candado unitario exige que cada estrategia produzca
facts para cada año publicado y que la población conserve el orden completo del registry,
incluyendo entradas `archived`.

Permanece `PARTIAL`: no se usó `--apply`, el plan Fabric sigue sin autorización y por tanto no
se ejecutó la query de aceptación contra PostgreSQL vivo. El modo plan no sustituye evidencia de
persistencia.

## Notas constitución
'Backfill exclusivo de campeonas' es decisión RECHAZADA (§31).
