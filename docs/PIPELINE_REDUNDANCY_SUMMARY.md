# RESUMEN DE REDUNDANCIAS EN PIPELINES USD/COP

**Fecha**: 2025-12-15
**Estado Actual**: 17 DAGs activos
**Estado Propuesto**: 12 DAGs (reducción del 29%)

---

## RESUMEN EJECUTIVO

| Categoría | Cantidad | Acción |
|-----------|----------|--------|
| Mantener sin cambios | 10 | - |
| Eliminar por redundancia | 1 | `01b_l0_macro_acquire` |
| Consolidar en 1 | 4 | `00b_l0_macro_scraping*` |

---

## REDUNDANCIAS IDENTIFICADAS

### 1. MACRO SCRAPING (4 DAGs duplicados)

**DAGs Actuales**:
```
usdcop_m5__00b_l0_macro_scraping        (Schedule: 55 12 * * 1-5)
usdcop_m5__00b_l0_macro_scraping_0755   (Schedule: 55 12 * * 1-5)
usdcop_m5__00b_l0_macro_scraping_1030   (Schedule: 30 15 * * 1-5)
usdcop_m5__00b_l0_macro_scraping_1200   (Schedule: 0 17 * * 1-5)
```

**Problema**:
- El mismo código se ejecuta en 4 DAGs separados
- Solo cambia el schedule de cron
- El DAG principal y `_0755` tienen el mismo schedule (redundancia total)

**Solución Propuesta**:
Consolidar en **1 solo DAG** con múltiples schedules o usar un solo cron con condiciones internas.

```python
# Opción A: Un DAG con TimetableSchedule
schedule = [
    CronTriggerTimetable("55 12 * * 1-5"),  # 7:55 COT
    CronTriggerTimetable("30 15 * * 1-5"),  # 10:30 COT
    CronTriggerTimetable("0 17 * * 1-5"),   # 12:00 COT
]

# Opción B: Un DAG con schedule más frecuente y skip interno
schedule = "*/30 12-17 * * 1-5"  # Cada 30 min, skip si no es hora exacta
```

---

### 2. MACRO ACQUIRE vs MACRO SCRAPING

**DAGs en conflicto**:
```
usdcop_m5__01b_l0_macro_acquire    → Solo WTI, DXY (intervalo 1h)
usdcop_m5__00b_l0_macro_scraping   → DXY, VIX, EMBI, Brent, WTI, GOLD, tasas
```

**Problema**:
- `01b_l0_macro_acquire` es un **subconjunto** de `00b_l0_macro_scraping`
- Ambos escriben a `macro_ohlcv`
- `01b` tiene catchup desde 2002 (datos históricos) pero `00b` es más completo para datos actuales

**Solución**:
- **ELIMINAR** `01b_l0_macro_acquire`
- Si se necesitan datos históricos macro, extender `00b` con modo backfill

---

## MATRIZ DE DECISIÓN

### ELIMINAR (Redundante)

| DAG | Razón | Funcionalidad Cubierta Por |
|-----|-------|---------------------------|
| `usdcop_m5__01b_l0_macro_acquire` | Subconjunto de datos | `00b_l0_macro_scraping` |

### CONSOLIDAR (4 → 1)

| DAGs Actuales | DAG Propuesto |
|---------------|---------------|
| `usdcop_m5__00b_l0_macro_scraping` | `usdcop_m5__00_l0_macro_unified` |
| `usdcop_m5__00b_l0_macro_scraping_0755` | ↑ |
| `usdcop_m5__00b_l0_macro_scraping_1030` | ↑ |
| `usdcop_m5__00b_l0_macro_scraping_1200` | ↑ |

### MANTENER (Sin cambios)

| DAG | Capa | Justificación |
|-----|------|---------------|
| `usdcop_m5__01_l0_intelligent_acquire` | L0 | Fuente principal OHLCV, único |
| `usdcop_m5__02_l1_standardize` | L1 | Capa crítica calidad, único |
| `usdcop_m5__03_l2_prepare` | L2 | Preparación ML, único |
| `usdcop_m5__04_l3_feature` | L3 | Features RL normalizados |
| `usdcop_m5__04b_l3_llm_features` | L3 | Features LLM (complementario, diferente propósito) |
| `usdcop_m5__05_l4_rlready` | L4 | Dataset final RL, único |
| `usdcop_m5__05a_l5_rl_training` | L5 | Entrenamiento PPO, único |
| `usdcop_m5__05b_l5_ml_training` | L5 | Entrenamiento LightGBM (complementario) |
| `usdcop_m5__05c_l5_llm_setup_corrected` | L5 | Setup LLM (complementario) |
| `usdcop_m5__06_l5_realtime_inference` | L5 | Producción tiempo real, único |
| `usdcop_m5__07_l6_backtest_multi_strategy` | L6 | Backtest, único |
| `usdcop_m5__99_alert_monitor` | Ops | Monitoreo, único |

---

## PLAN DE ACCIÓN

### Fase 1: Consolidación Macro Scraping

1. **Crear nuevo DAG unificado**:
   ```
   Nombre: usdcop_m5__00_l0_macro_unified
   Schedule: Multi-timetable o cron con skip logic
   Horarios: 7:55, 10:30, 12:00 COT (L-V)
   ```

2. **Migrar código**:
   - Copiar lógica de `00b_l0_macro_scraping.py`
   - Eliminar factory function que genera múltiples DAGs
   - Implementar schedule unificado

3. **Pausar DAGs antiguos**:
   ```bash
   airflow dags pause usdcop_m5__00b_l0_macro_scraping
   airflow dags pause usdcop_m5__00b_l0_macro_scraping_0755
   airflow dags pause usdcop_m5__00b_l0_macro_scraping_1030
   airflow dags pause usdcop_m5__00b_l0_macro_scraping_1200
   ```

4. **Activar nuevo DAG y validar** por 1 semana

5. **Eliminar DAGs antiguos**

### Fase 2: Eliminar Macro Acquire

1. **Verificar que no hay dependencias**:
   ```sql
   SELECT * FROM dag_run WHERE dag_id = 'usdcop_m5__01b_l0_macro_acquire';
   ```

2. **Pausar DAG**:
   ```bash
   airflow dags pause usdcop_m5__01b_l0_macro_acquire
   ```

3. **Eliminar archivo** después de 2 semanas sin problemas

---

## CÓDIGO PROPUESTO: DAG UNIFICADO

```python
"""
DAG: usdcop_m5__00_l0_macro_unified
Reemplaza los 4 DAGs de macro scraping con uno solo.
"""

from airflow import DAG
from airflow.timetables.trigger import CronTriggerTimetable
from airflow.timetables.base import DagRunInfo, DataInterval, Timetable
from datetime import datetime, timedelta

# Multi-schedule timetable (Airflow 2.4+)
class MultiCronTimetable(Timetable):
    def __init__(self, cron_expressions):
        self.crons = cron_expressions

    # ... implementación ...

# O usar approach más simple con 3 DAGs mínimos:
SCHEDULES = {
    'pre_open': '55 12 * * 1-5',   # 7:55 COT
    'mid_morning': '30 15 * * 1-5', # 10:30 COT
    'close': '0 17 * * 1-5',        # 12:00 COT
}

# Single DAG approach con schedule más frecuente
with DAG(
    dag_id='usdcop_m5__00_l0_macro_unified',
    schedule_interval='55 12,15,17 * * 1-5',  # Simplificado
    # ... resto de config
) as dag:
    # ... tasks
```

---

## IMPACTO ESPERADO

| Métrica | Antes | Después | Mejora |
|---------|-------|---------|--------|
| Total DAGs | 17 | 12 | -29% |
| DAGs L0 Macro | 5 | 1 | -80% |
| Complejidad | Alta | Media | Significativa |
| Mantenimiento | Difícil | Fácil | Mejor |

---

## RIESGOS Y MITIGACIÓN

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Falla en consolidación | Baja | Alto | Testing en staging primero |
| Pérdida de datos históricos | Baja | Medio | Backup antes de eliminar |
| Schedule incorrecto | Media | Medio | Validar horarios COT→UTC |

---

## CHECKLIST DE IMPLEMENTACIÓN

- [ ] Crear DAG unificado en branch feature
- [ ] Testing local con Airflow standalone
- [ ] Deploy a staging
- [ ] Validar 3 ejecuciones (1 por horario)
- [ ] Comparar outputs con DAGs antiguos
- [ ] Pausar DAGs antiguos
- [ ] Activar DAG nuevo en producción
- [ ] Monitorear 1 semana
- [ ] Eliminar DAGs antiguos
- [ ] Eliminar `01b_l0_macro_acquire`
- [ ] Actualizar documentación

---

## REFERENCIAS

- [PIPELINE_AUDIT_REPORT.md](./PIPELINE_AUDIT_REPORT.md) - Informe completo de cada DAG
- [ARCHITECTURE.md](./ARCHITECTURE.md) - Arquitectura general del sistema
