---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-29
supersedes: []
code_anchors:
  - database/migrations/067_spx500_regime_macro_vars.sql
  - database/migrations/072_reference_identity.sql
  - airflow/dags/l0_macro_update.py
  - config/assets/pipelines.yaml
  - src/market/identity.py
  - tests/unit/test_codex_fabric_contracts.py
---

# BL-37 — Identidades canónicas (reference.asset/instrument/provider_symbol/bar_interval)

**Fuente**: Plan Consolidado §3 / DATA-STRATEGY §36 (P0.3) · **Ola**: 2-3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
El mismo activo vive como usdcop/USD-COP, btcusdt/BTC-USDT, spx500/SPX500/SPX-500/SPY; timeframes 5m vs 5min, 1d vs 1day; market_ingestion_manifest.asset_id=BTC/USDT pero dim_asset.asset_id=btcusdt (joins rotos). dim_asset (4 filas, con annualization+calendar_kind) es el embrión correcto.

## Qué falta exactamente
Esquema reference.*: asset, instrument (SPX, SPY y ES son instrumentos DISTINTOS), provider, provider_symbol→instrument_id, bar_interval enum (PT5M...), calendar. El manifest se descompone en instrument_id+provider_id+provider_symbol; FKs en tablas de mercado; CHECK de timeframes.

## Remediación verificable de esta ola (2026-07-29)

Se cerró la divergencia viva del intervalo de un minuto:
`BarInterval.M1 == "PT1M"` tiene ahora seed DDL
`('PT1M', 60, FALSE)`. Un candado bidireccional extrae el `INSERT` real de
`reference.bar_interval` y exige igualdad exacta con todo el enum Python; no
son dos listas que sólo se revisan por separado.

```bash
python -m pytest -q tests/unit/test_codex_fabric_contracts.py -k bar_interval
# 1 passed, 30 deselected
```

Mutaciones ejecutadas y restauradas:

- retirar `P1W` del seed ⇒ **1 failed**, nombra `P1W` huérfano;
- añadir `PT2M` sólo en DDL ⇒ **1 failed**, nombra el extra;
- cambiar PT1M de 60 a 600 segundos ⇒ **1 failed**, muestra `600 != 60`.

El digest pinneado del plan `fabric-v1` queda deliberadamente cerrado tras
cambiar estos bytes. No se autoautoriza ni se actualiza hasta terminar el lote
de migraciones y recibir revisión independiente. La prueba del gate falla como
se espera: pin `sha256:b83bf454...87852` frente a bytes actuales
`sha256:7cac6af9...0656c`; `plan_is_authorized` registra la divergencia.

## Alcance residual declarado

La verificación histórica «join manifest-dim_asset sin pérdidas; símbolos
huérfanos = 0» no es ejecutable todavía: `reference.asset`,
`reference.instrument` y `reference.provider_symbol` no tienen seed/backfill,
el manifest no está descompuesto y las tablas de mercado no tienen sus FKs.
Este corte acredita únicamente esquema + seed/paridad de intervalos. El join,
backfill y FKs siguen abiertos y deben entrar en una ola propia cuando exista
la población que permita falsarlos; no se cuentan como hechos por este cambio.
Hasta revisión bilateral, BL-37 permanece `PARTIAL`.

## Impacto frontend
Registry/API exponen instrument_id estable; los selectores dejan de depender de strings ambiguos.

## Dependencias
Antes de BL-38 (canonical_bar referencia instrument_id). Coordina con BL-36.

## Verificación
Join manifest-dim_asset sin pérdidas; query de símbolos huérfanos = 0.

## Notas constitución
Ningún atributo con dos escritores aplica también a los NOMBRES: una identidad, N alias mapeados.
