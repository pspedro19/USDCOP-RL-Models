---
kind: roadmap
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - airflow/dags/forecast_h5_l5_weekly_signal.py
---

# BL-42 — Unidades decimales + action.strategy_signal normalizada (JSONB de política)

**Fuente**: Plan Consolidado §7 / DATA-STRATEGY §42 · **Ola**: 3 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
Inconsistencia real: forecast_h5_predictions.predicted_return_pct=1.6481 (puntos pct) vs forecast_h5_signals.ensemble_return=0.01606 (decimal) = el MISMO 1.606 pct. forecast_h5_signals tiene 32 columnas con ~80 pct NULL (cada versión añade columnas — no escala a cientos de estrategias).

## Qué falta exactamente
Regla: en DB todo retorno DECIMAL (0.01 = 1 pct); nombres return_decimal/drawdown_decimal/leverage_ratio; el formateo a pct solo en frontend. strategy_signal normalizada: núcleo estable (signal_id, sleeve_id, version, instrument, as_of, valid_from/until, direction, target_exposure, decision_fingerprint) + decision_components JSONB versionado (hurst, regime, tp, hs, ...).

## Impacto frontend
Formateo pct exclusivo del frontend; tabla de señales filtrable por columnas estables.

## Dependencias
BL-15 (contratos), BL-13. Migración de las 10 señales existentes es trivial.

## Verificación
Grep: ninguna columna _pct con valores decimales; la señal v11 rinde igual en UI antes/después.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_return_units.py -q
verde:   28 passed, 3 skipped   (los skips son los que exigen Postgres arriba)

muta:    scripts/pipeline/train_and_export_smart_simple.py:1052
         "total_return_pct": round(total_return, 2)  ->  round(total_return / 100.0, 6)
espera:  2 failed — "total_return_pct=0.144616 for a ledger that compounds 10_000 ->
         11446.16: expected 14.46 PERCENTAGE POINTS" y el detector de disfraz decimal
         sobre la salida en memoria
```

**Historial honesto**: hasta el 2026-07-28 esa mutación —**literalmente el bug que este BL
prohíbe**, un decimal bajo un sufijo `_pct`— pasaba verde, porque la suite validaba los JSON
**ya commiteados** en `public/data/production/` y no el código que los produce. El detector
funcionaba, pero apuntaba al artefacto en vez de al productor.

## Notas constitución
El sufijo _pct sobre un decimal es un bug de comunicación esperando capital.
