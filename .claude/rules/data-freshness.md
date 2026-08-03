---
kind: rule
status: IMPLEMENTED
contract: CTR-DQ-OPS-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - airflow/dags/utils/data_quality.py
  - src/data_quality/ohlcv_validators.py
---
# Rule: Frescura de datos (umbrales)

> **SSOT de los umbrales de frescura.** Runbooks de recuperación, checklists y tracking de
> migraciones: [`freshness-recovery.md`](../specs/operations/freshness-recovery.md).
> Schedule de DAGs: [`elite-operations.md`](../specs/operations/elite-operations.md) —
> **no lo re-tabules aquí**.

## Umbrales

| Fuente | Máx. staleness | Consecuencia |
|--------|----------------|--------------|
| OHLCV 5-min (`usdcop_m5_ohlcv`) | **3 días** | Training BLOQUEADO |
| Macro diario (`macro_indicators_daily`) | **7 días** | Training BLOQUEADO |
| Modelos `.pkl` (H1/H5) | **10 días** | WARNING (no bloquea) |
| Seed diario / MACRO_DAILY_CLEAN | 3 / 7 días | Features viejas |
| News articles / features | 24 horas | Análisis sin contexto fresco |

OHLCV se mide en días hábiles colombianos mediante
`data_quality.py::_trading_days_since`; la justificación de cada umbral vive en la spec.

## Invariantes

1. El gate pre-training es la primera tarea de H1-L3/H5-L3 y bloquea; pre-inferencia avisa.
2. Todo ingest OHLCV pasa por `ohlcv_validators.py`; barra fuera de sesión es error duro.
3. El backup de seeds es Lun-Vie (`0 20 * * 1-5`, 15:00 COT); no existe backup de fin de semana.
4. Los umbrales en minutos del watchdog no sustituyen estos umbrales de entrenamiento.

## DO NOT

- No entrenar stale ni correr L5 si L3 no completó; usar `ExternalTaskSensor`.
- No borrar seeds ni `MACRO_DAILY_CLEAN.parquet`: son el fallback de restore.
- `--no-validate` es sólo escape de emergencia auditado.
- Una migración aplicada es inmutable; crear otra.
