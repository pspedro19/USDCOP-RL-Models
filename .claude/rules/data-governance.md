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
> `../specs/pipelines/l0-data-reference.md`.

## Regla de oro: todo timestamp COP = `America/Bogota`

**Sin excepciones para USD/COP y los pares FX acotados por sesión.** Sesión = **8:00-12:55 COT,
Lun-Vie**. Si recibes UTC, conviértelo antes de guardar:

```python
ts.dt.tz_localize('UTC').dt.tz_convert('America/Bogota')
```

**Carve-out multi-activo (audit A1-01)**: los activos cuyo mercado no está acotado por la sesión
COP (XAU/USD, BTC/USDT) guardan **TIMESTAMPTZ basado en instante**, no COT-localizado. La regla
COT es autoritativa para COP; el detalle por activo está en
`../specs/assets/_asbuilt-implementation.md`.

## Invariantes

1. **Anclar la fecha de una barra diaria en UTC**, no en la tz de cierre:
   `ts.dt.tz_convert("UTC").dt.normalize()` y *después* aplicar el offset de cierre.
   `tz_convert(→ET).normalize()` sobre un sello 00:00-UTC corre **todas** las barras un día atrás
   (bug "Sunday pile-up" de Gold, CTR-DQ-OHLCV-001).
2. **Todo ingest pasa por `src/data_quality/ohlcv_validators.py`** antes de escribir un seed.
   Barra en día no-sesión = ERROR duro.
3. **UPSERT siempre** `ON CONFLICT (time, symbol) DO UPDATE` — nunca INSERT plano.
4. **BRL desde TwelveData se pide en UTC.** Con `timezone=America/Bogota` devuelve datos
   incompletos. Aplica a realtime, backfill y al constructor de seeds.
5. **`usdcop_m5_ohlcv` no se renombra** — el multi-par se resuelve por la columna `symbol`.
6. **El `FrequencyRoutedUpsertService` es obligatorio** para macro: enruta a la tabla correcta
   según frecuencia.
7. **`is_complete` se actualiza siempre** tras cada corrida: L2 filtra por esa bandera.

## DO NOT

- Do NOT guardar timestamps COP en UTC — convierte a `America/Bogota` antes del insert.
- Do NOT pedir BRL a TwelveData con `timezone=America/Bogota`.
- Do NOT usar `tz_localize` sobre un timestamp que ya tiene timezone — usa `tz_convert`.
- Do NOT `tz_convert(→ET).normalize()` una barra diaria 00:00-UTC para tomar su fecha.
- Do NOT editar seeds a mano — regenera con `scripts/data/build_unified_fx_seed.py`.
- Do NOT asumir el mismo calendario para todos los pares (BRL tiene feriados propios).
- Do NOT escribir un seed que falle el validador OHLCV.
- Do NOT saltarte el `FrequencyRoutedUpsertService` ni la actualización de `is_complete`.
- Do NOT borrar los 9 MASTER de `04_cleaning/output/` — son el backup de macro.
