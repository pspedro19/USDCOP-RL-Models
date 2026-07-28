---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - airflow/dags/asset_pipeline_factory.py
  - scripts/data/ingest_asset_ohlcv.py
  - scripts/data/build_unified_fx_seed.py
---

# BL-38 — Mercado canónico: raw_bar/canonical_bar + caggs 1h/4h/1d + política de resampleo

**Fuente**: Plan Consolidado §1.4/§6 / DATA-STRATEGY §37-38 (D2) · **Ola**: 3-4 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
usdcop_m5_ohlcv (2.2M) tiene nombre engañoso (solo 4.5% es COP); asset_native_ohlcv es el raw de facto; asset_daily convive como tabla escrita (no derivada). available_at 6.4%/19.9% NULL y en native arranca 2026 (= ingesta, no disponibilidad histórica).

## Qué falta exactamente
market.raw_bar (inmutable) → market.canonical_bar 5m → continuous aggregates 1h/4h/1d; bar_method = provider_official | resampled (NUNCA mezclar en silencio: la diaria oficial del proveedor no siempre equivale al resampleo del intradía); semántica de 5 timestamps (event/provider_published/available/retrieved/ingested) con available_at(OHLCV)=cierre+latencia declarada; backfill etiquetado de NULLs. Renombres vía VISTAS de compatibilidad primero — nunca renombrar en caliente.

## Impacto frontend
Charts pasan a leer caggs; cero cambio visual.

## Dependencias
BL-37 (instrument_id), BL-36. Alimenta BL-17 (spine) y BL-24.

## Verificación
cagg 1d == asset_daily actual para provider_official (diff=0 donde aplica); lectores viejos vivos vía vistas.

## Notas constitución
No retención destructiva hasta que el raw esté respaldado (MinIO/Parquet).
