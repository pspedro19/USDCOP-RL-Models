---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - airflow/dags/l0_macro_update.py
  - config/assets/pipelines.yaml
---

# BL-37 — Identidades canónicas (reference.asset/instrument/provider_symbol/bar_interval)

**Fuente**: Plan Consolidado §3 / DATA-STRATEGY §36 (P0.3) · **Ola**: 2-3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
El mismo activo vive como usdcop/USD-COP, btcusdt/BTC-USDT, spx500/SPX500/SPX-500/SPY; timeframes 5m vs 5min, 1d vs 1day; market_ingestion_manifest.asset_id=BTC/USDT pero dim_asset.asset_id=btcusdt (joins rotos). dim_asset (4 filas, con annualization+calendar_kind) es el embrión correcto.

## Qué falta exactamente
Esquema reference.*: asset, instrument (SPX, SPY y ES son instrumentos DISTINTOS), provider, provider_symbol→instrument_id, bar_interval enum (PT5M...), calendar. El manifest se descompone en instrument_id+provider_id+provider_symbol; FKs en tablas de mercado; CHECK de timeframes.

## Impacto frontend
Registry/API exponen instrument_id estable; los selectores dejan de depender de strings ambiguos.

## Dependencias
Antes de BL-38 (canonical_bar referencia instrument_id). Coordina con BL-36.

## Verificación
Join manifest-dim_asset sin pérdidas; query de símbolos huérfanos = 0.

## Notas constitución
Ningún atributo con dos escritores aplica también a los NOMBRES: una identidad, N alias mapeados.
