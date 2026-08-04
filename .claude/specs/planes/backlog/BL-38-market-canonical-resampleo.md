---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-08-04
supersedes: []
code_anchors:
  - airflow/dags/asset_pipeline_factory.py
  - scripts/data/ingest_asset_ohlcv.py
  - scripts/data/build_unified_fx_seed.py
---

# BL-38 — Mercado canónico: raw_bar/canonical_bar + caggs 1h/4h/1d + política de resampleo

**Fuente**: Plan Consolidado §1.4/§6 / DATA-STRATEGY §37-38 (D2) · **Ola**: 3-4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
Entrega parcial: las migraciones 073/080 definen `raw_bar`, `canonical_bar`, perfil físico y caggs operator-only; `src/market/resampling.py` ya está trackeado. No existen writers productivos, vistas de compatibilidad ni objetos aplicados en la base viva. `asset_native_ohlcv` y `asset_daily` siguen siendo los caminos reales.

## Qué falta exactamente
Implementar writers y backfill `raw_bar → canonical_bar`, crear primero las vistas de compatibilidad y aplicar los caggs bajo preflight del operador. Falta una regresión de anclaje de sesión/UTC que impida el Sunday pile-up y una comparación provider_official frente a resampleo donde aplique.

## Impacto frontend
Charts pasan a leer caggs; cero cambio visual.

## Dependencias
BL-37 (instrument_id), BL-36. Alimenta BL-17 (spine) y BL-24.

## Verificación
cagg 1d == asset_daily actual para provider_official (diff=0 donde aplica); lectores viejos vivos vía vistas.

## Notas constitución
No retención destructiva hasta que el raw esté respaldado (MinIO/Parquet).

## Bloqueo de cableado medido (2026-08-03)

El camino requiere identidad 072, `market.raw_bar`/`market.canonical_bar` de 073 y el perfil
físico/caggs operator-only de 080. Ninguno está aplicado en la base viva;
`src/market/resampling.py` tiene tests pero cero llamadores productivos. El orden obligatorio es
072 → 073 → writers/backfill y vistas compatibles → preflight 080; DONE exige callers y diff
contra datos reales, no sólo el módulo local.

## Estado de la base y hueco de persistencia (2026-08-04)

El plan revisado `fabric-v1` ya fue aplicado y verificado por ambos agentes. Esto retira el
bloqueo de esquema para `market.raw_bar` y `market.canonical_bar`, pero **no vuelve este BL
IMPLEMENTED**: continúan pendientes sus writers, backfill, vistas de compatibilidad, preflight de
agregados y comparación contra los datos vigentes.

`market.resample_policy` no forma parte del DDL revisado y no tiene lectores SQL en el árbol. La
expectativa que lo trataba como tabla requerida fue retirada del validador en `5ed5cac9` porque no
estaba respaldada por ninguna migración. Esto corrige el gate, **no resuelve el diseño pendiente**:
las policies declaran `resample_policy_id` y el contrato de versión conserva
`resample_policy_hash`, pero esos valores todavía no resuelven contra un registro persistente.
Definir ese registro requiere contrato y migración aditiva futuros; no se inventa su esquema como
parte del arreglo del validador ni se edita una migración ya aplicada.
