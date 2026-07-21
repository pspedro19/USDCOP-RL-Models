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
> migraciones: `../specs/operations/freshness-recovery.md`.
> Schedule de DAGs: `../specs/operations/elite-operations.md` — **no lo re-tabules aquí**.

## Umbrales

| Fuente | Máx. staleness | Consecuencia |
|--------|----------------|--------------|
| OHLCV 5-min (`usdcop_m5_ohlcv`) | **3 días** | Training BLOQUEADO |
| Macro diario (`macro_indicators_daily`) | **7 días** | Training BLOQUEADO |
| Modelos `.pkl` (H1/H5) | **10 días** | WARNING (no bloquea) |
| Seed diario / MACRO_DAILY_CLEAN | 3 / 7 días | Features viejas |
| News articles / features | 24 horas | Análisis sin contexto fresco |

**Por qué**: el mercado cierra viernes 12:55 COT y el training corre domingo → el dato más
reciente tiene 2 días; 3 días absorbe eso. Macro admite 7 porque varias variables son semanales.
**Unidad OHLCV = días HÁBILES del calendario colombiano** (2026-07-21, directiva del operador):
el 20-jul (festivo) demostró que contar días calendario bloquea el martes post-festivo con cero
barras faltantes. El umbral 3 se mantiene — en semanas normales la cuenta hábil es MÁS estricta
(vie→dom = 0). Implementación: `utils/data_quality.py::_trading_days_since` + test
`test_freshness_gate_trading_days.py`. Macro sigue en días calendario.
Modelos a 10 días = dos domingos fallidos seguidos; se avisa pero no se detiene el trading.

## Invariantes

1. **El gate pre-training es BLOQUEANTE** y es la primera tarea de H1-L3 y H5-L3
   (`validate_training_data_freshness`). El pre-inferencia solo avisa.
2. **Todo ingest de OHLCV pasa por `src/data_quality/ohlcv_validators.py`** antes de escribir un
   seed (CTR-DQ-OHLCV-001). Barras en día no-sesión = ERROR duro, no warning.
3. **El backup de seeds corre Lun-Vie** (`0 20 * * 1-5`, 15:00 COT). **No hay backup de fin de
   semana** — no asumas datos frescos un lunes por la mañana sin verificar.
4. El watchdog usa umbrales operativos propios y más estrictos (minutos, no días); son otra cosa
   y no sustituyen a estos.

## DO NOT

- Do NOT entrenar con datos stale (>3d OHLCV, >7d macro) — las predicciones salen desfasadas.
- Do NOT ignorar warnings de frescura de modelos: señalan que L3 falló.
- Do NOT borrar `MACRO_DAILY_CLEAN.parquet` ni los seeds — son el fallback de restore.
- Do NOT correr L5 si L3 no completó esta semana — usa `ExternalTaskSensor`.
- Do NOT escribir un seed que falle el validador; `--no-validate` es escape de emergencia logueado.
- Do NOT modificar una migración ya aplicada — crea una nueva.
