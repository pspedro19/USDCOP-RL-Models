---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-29
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
