---
kind: rule
status: IMPLEMENTED
contract: CTR-L0-GOV-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - scripts/data/build_unified_fx_seed.py
  - src/data_quality/ohlcv_validators.py
  - scripts/data/ingest_asset_ohlcv.py
  - config/macro_variables_ssot.yaml
---
# Rule: Gobernanza L0 (datos)

> **SSOT de las invariantes de datos L0.** Inventario de DAGs, schemas, extractores, comandos:
> [`l0-data-reference.md`](../specs/pipelines/l0-data-reference.md).

## Regla de oro: todo timestamp COP = `America/Bogota`

Para USD/COP y FX acotado por sesión: **8:00-12:55 COT, Lun-Vie**. Convertir UTC antes de
guardar; usar `tz_convert`, no `tz_localize`, si el sello ya tiene timezone.

XAU/USD y BTC/USDT guardan **TIMESTAMPTZ basado en instante**, no COT localizado. El detalle
por activo está en
[`_asbuilt-implementation.md`](../specs/assets/_asbuilt-implementation.md).

## Invariantes

1. La fecha diaria se ancla con `tz_convert("UTC").normalize()` antes del offset de cierre;
   normalizar en ET desplaza la barra.
2. Todo ingest usa `ohlcv_validators.py`; barra fuera de sesión es error.
3. Siempre UPSERT por `(time, symbol)`, nunca INSERT plano.
4. BRL/TwelveData se solicita en UTC en realtime, backfill y seeds.
5. `usdcop_m5_ohlcv` conserva su nombre; el par vive en `symbol`.
6. Macro usa `FrequencyRoutedUpsertService` y cada corrida actualiza `is_complete`.

## DO NOT

- No editar seeds a mano ni aceptar uno que falle validación.
- No asumir un calendario único entre pares.
- No borrar los MASTER de `04_cleaning/output/`: son el backup macro.
