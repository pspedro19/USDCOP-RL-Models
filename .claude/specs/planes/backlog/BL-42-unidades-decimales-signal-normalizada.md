---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
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

## Notas constitución
El sufijo _pct sobre un decimal es un bug de comunicación esperando capital.
