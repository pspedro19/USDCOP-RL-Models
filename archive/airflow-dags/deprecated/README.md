# Deprecated DAGs

**Fecha de deprecación:** 2025-10-22

## DAGs en esta carpeta

Estos DAGs han sido marcados como **DEPRECATED** y **NO deben usarse**.

### 1. usdcop_realtime_sync.py
**Motivo de deprecación:**
- Usa tabla `market_data` que fue eliminada
- Redundante con Servicio RT V2 (`realtime_market_ingestion_v2.py`)

**Reemplazo:**
- Servicio RT V2 maneja ingesta en tiempo real directamente a `usdcop_m5_ohlcv`

### 2. usdcop_realtime_failsafe.py
**Motivo de deprecación:**
- Usa tabla `market_data` que fue eliminada
- Redundante con Pipeline L0 (`usdcop_m5__01_l0_intelligent_acquire.py`)

**Reemplazo:**
- Pipeline L0 tiene detección inteligente de gaps y auto-fill

---

## Migración

Si necesitas funcionalidad de estos DAGs:
1. **Sync en tiempo real** → Ya manejado por `realtime_market_ingestion_v2.py`
2. **Gap detection** → Ya manejado por Pipeline L0 `check_existing_data()`
3. **Data validation** → Implementar en servicios actuales si es necesario

---

## Acción Recomendada

**Eliminar permanentemente** después de 30 días (2025-11-22) si no se necesitan.

```bash
# Comando para eliminar (después del 2025-11-22):
rm -rf /home/azureuser/USDCOP-RL-Models/airflow/dags/deprecated/
```
