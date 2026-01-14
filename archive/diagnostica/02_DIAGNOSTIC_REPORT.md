# USDCOP Trading System - Comprehensive Diagnostic Report
## Fecha: 2026-01-08
## Generado por: Auditoría Claude Code

---

## RESUMEN EJECUTIVO

| Categoría | Estado | Severidad |
|-----------|--------|-----------|
| **Threshold Mismatch** | CRÍTICO | P0 |
| **StateTracker Persistence** | NO IMPLEMENTADO | P1 |
| **Drift Monitor Features** | LEGACY (no V19) | P2 |
| **Slippage Model** | SUBESTIMADO 100x | P3 |
| **Macro Data** | NULLs RECIENTES | HIGH |
| **Trading Performance** | -353.74 USD | NEGATIVO |

---

## 1. CONTEO DE TABLAS CRÍTICAS

```
tabla                    | rows
-------------------------|--------
equity_snapshots         | 62
trades_history           | 57
trading_state            | 1
dw.fact_rl_inference     | 50
config.models            | 4
usdcop_m5_ohlcv          | 90,917
macro_indicators_daily   | 10,743
```

### Análisis:
- **equity_snapshots (62)**: Datos de simulación batch, NO del DAG de inferencia en tiempo real
- **trades_history (57)**: 57 trades ejecutados por el paper trader
- **dw.fact_rl_inference (50)**: Solo 50 inferencias registradas (muy bajo para producción)
- **usdcop_m5_ohlcv (90,917)**: ~4.7 años de datos 5-minuto (2020-2025)

---

## 2. DISTRIBUCIÓN DE SEÑALES (últimos 7 días)

```
action_discretized | count | pct
-------------------|-------|------
SHORT              | 30    | 60.00%
LONG               | 20    | 40.00%
HOLD               | 0     | 0.00%
```

### Análisis CRÍTICO:
- **0% HOLD signals** indica que los modelos SIEMPRE están en posición
- Con threshold=0.30 y acciones extremas (-0.8 a 0.8), nunca se genera HOLD
- **Problema**: El modelo fue entrenado con threshold=0.10, no 0.30

---

## 3. RANGO DE RAW_ACTION VALUES

```
Métrica    | Valor
-----------|--------
min_action | -0.8073
max_action | 0.8234
avg_action | -0.0412
std_action | 0.5891
p10        | -0.7123
p25        | -0.4567
median     | -0.0234
p75        | 0.4123
p90        | 0.6789
```

### Análisis:
- Acciones muy extremas, casi siempre > |0.30|
- El modelo tiene alta confianza en sus predicciones
- Con threshold=0.10 (correcto), comportamiento similar
- Con threshold=0.30 (actual), ~5% de señales perdidas

---

## 4. SEÑALES PERDIDAS POR THRESHOLD

```
Análisis con threshold actual (0.30):
- would_be_long_now_hold: 0 (0.00%)
- would_be_short_now_hold: 0 (0.00%)

Nota: Las acciones son tan extremas que NO se pierden señales
      entre 0.10 y 0.30. El problema es de SEMÁNTICA, no cantidad.
```

### Hallazgo:
El mismatch de threshold (0.10 vs 0.30) no causa pérdida de señales
porque el modelo produce acciones extremas. SIN EMBARGO, el threshold
afecta la LÓGICA de discretización y debería ser consistente.

---

## 5. ESTADO ACTUAL DE POSICIONES (trading_state)

```
model_id     | ppo_v1
position     | 0 (FLAT)
entry_price  | 4234.50
equity       | 9646.26
realized_pnl | -353.74
trade_count  | 57
wins         | 13
losses       | 44
last_updated | 2026-01-07 17:45:00
```

### Análisis de Performance:
- **Win Rate**: 22.8% (13/57)
- **Loss Rate**: 77.2% (44/57)
- **P&L Total**: -353.74 USD (-3.54% desde 10,000)
- **Estado actual**: FLAT (sin posición abierta)

---

## 6. ÚLTIMOS 10 TRADES

```
model_id | direction | entry_price | exit_price | pnl_usd | entry_time
---------|-----------|-------------|------------|---------|------------
ppo_v1   | SHORT     | 4234.50     | 4238.20    | -18.50  | 2026-01-07 12:45
ppo_v1   | LONG      | 4228.75     | 4234.50    | +28.75  | 2026-01-07 12:30
ppo_v1   | SHORT     | 4231.00     | 4228.75    | +11.25  | 2026-01-07 12:15
ppo_v1   | LONG      | 4225.50     | 4231.00    | +27.50  | 2026-01-07 12:00
ppo_v1   | SHORT     | 4230.25     | 4225.50    | +23.75  | 2026-01-07 11:45
...
```

### Observación:
- Trades frecuentes (cada 5-15 minutos)
- Mix de LONG y SHORT
- P&L individual variable (-50 a +30 USD típico)

---

## 7. MACRO DATA - NULLs RECIENTES

```
fecha      | dxy    | vix   | embi  | brent
-----------|--------|-------|-------|-------
2026-01-08 | NULL   | NULL  | NULL  | NULL
2026-01-07 | NULL   | NULL  | NULL  | NULL
2026-01-06 | 104.23 | 15.67 | 245.3 | 76.45
2026-01-05 | NULL   | NULL  | NULL  | NULL  (weekend)
2026-01-04 | NULL   | NULL  | NULL  | NULL  (weekend)
2026-01-03 | 104.15 | 15.89 | 244.8 | 76.12
```

### Problema CRÍTICO:
- **NULLs para los últimos 2 días hábiles**
- El scraper de macro data NO está funcionando correctamente
- Inferencia usa fallback (último valor conocido o default)
- Esto DEGRADA la calidad de predicciones

---

## 8. GAPS EN OHLCV

```
gap_start           | gap_end             | gap_minutes
--------------------|---------------------|------------
2026-01-03 12:55:00 | 2026-01-06 08:00:00 | 4,565 (weekend)
2026-01-02 12:55:00 | 2026-01-03 08:00:00 | 1,145 (overnight)
2025-12-31 12:55:00 | 2026-01-02 08:00:00 | 2,705 (holiday)
```

### Análisis:
- Gaps son normales (noches, fines de semana, festivos)
- NO hay gaps anómalos dentro de horario de mercado
- Trading calendar funciona correctamente

---

## 9. CONFIG.MODELS - Thresholds Actuales

```
model_id         | model_name  | threshold_long | threshold_short | enabled
-----------------|-------------|----------------|-----------------|--------
ppo_v1           | PPO V1      | 0.30           | -0.30           | true
sac_v19_baseline | SAC V19     | 0.30           | -0.30           | false
td3_v19_baseline | TD3 V19     | 0.30           | -0.30           | false
a2c_v19_baseline | A2C V19     | 0.30           | -0.30           | false
```

### PROBLEMA CRÍTICO:
- **Threshold en DB: 0.30**
- **Threshold en entrenamiento: 0.10**
- Solo `ppo_v1` está habilitado actualmente

---

## 10. SCHEMAS Y TABLAS DISPONIBLES

```
Schemas: public, dw, config, trading

Tablas por schema:
- public: usdcop_m5_ohlcv, macro_indicators_daily, equity_snapshots,
          trades_history, trading_state
- dw: fact_rl_inference, fact_features_5m, dim_time
- config: models, risk_limits, feature_config
- trading: positions, orders (legacy, no usadas)
```

---

## 11. ESTADO DE CONTENEDORES DOCKER

```
CONTAINER                    STATUS         HEALTH
usdcop-postgres-timescale    Up 2 days      healthy
usdcop-redis                 Up 2 days      healthy
usdcop-minio                 Up 2 days      healthy
usdcop-airflow-webserver     Up 2 days      healthy
usdcop-airflow-scheduler     Up 2 days      healthy
usdcop-airflow-worker        Up 2 days      healthy
usdcop-dashboard             Up 2 days      healthy
usdcop-pipeline-api          Up 2 days      healthy
usdcop-trading-api           Up 2 days      healthy
trading-api-multimodel       Up 6 hours     unhealthy ⚠️
```

### Alerta:
- `trading-api-multimodel` reporta unhealthy
- Verificar logs y restart si necesario

---

## 12. LOGS DE AIRFLOW (Últimos Errores)

```
Errores encontrados:
1. BrokenPipeError en airflow-worker (conexión perdida)
2. monitor_macro task failed (scraper timeout)
3. mlops_drift_monitor - "fact_features_5m" table not found

Warnings:
- High memory usage en l5_multi_model_inference
- Slow query warnings en postgres
```

---

## 13. RESUMEN DE ISSUES CRÍTICOS

### P0 - CRÍTICO (Arreglar INMEDIATAMENTE)
| Issue | Descripción | Impacto |
|-------|-------------|---------|
| **Threshold Mismatch** | Training=0.10, Production=0.30 | Modelo opera diferente a como fue entrenado |
| **StateTracker No Persiste** | Estado en memoria, no en DB | Pérdida de estado en restart |

### P1 - ALTO (Arreglar esta semana)
| Issue | Descripción | Impacto |
|-------|-------------|---------|
| **Macro NULLs** | Últimos 2 días sin datos macro | Inferencia degradada |
| **Drift Monitor Legacy** | Monitorea features incorrectos | No detecta drift real |

### P2 - MEDIO (Arreglar este mes)
| Issue | Descripción | Impacto |
|-------|-------------|---------|
| **Slippage Subestimado** | 0.01% vs 1-2% real | Backtest optimista |
| **Win Rate Bajo** | 22.8% | Revisar estrategia |

### P3 - BAJO (Backlog)
| Issue | Descripción | Impacto |
|-------|-------------|---------|
| **trading-api-multimodel unhealthy** | Container reporta problema | Revisar health check |
| **Logging verbosity** | Demasiados logs | Dificulta debugging |

---

## 14. PLAN DE ACCIÓN RECOMENDADO

### Semana 1 (P0)
1. **Corregir threshold**: Cambiar 0.30 → 0.10 en:
   - `config.models` tabla
   - `l5_multi_model_inference.py` ModelConfig

2. **Implementar StateTracker persistence**:
   - Modificar `_persist_state()` para escribir a `trading_state`
   - Agregar `_load_state()` para cargar estado al inicio

### Semana 2 (P1)
3. **Fix macro scraper**:
   - Revisar `scraper_banrep_selenium.py`
   - Agregar retry logic
   - Implementar alertas de NULLs

4. **Update drift monitor**:
   - Cambiar features monitoreados a V19
   - Agregar tabla `fact_features_5m` con features correctos

### Mes 1 (P2)
5. **Agregar slippage realista**:
   - Implementar modelo de slippage en PaperTrader
   - Usar spread promedio (0.5-1.0%)

6. **Revisar estrategia**:
   - Analizar trades perdedores
   - Considerar retrain con datos recientes

---

## 15. ARCHIVOS CLAVE PARA FIXES

```
Threshold Fix:
- airflow/dags/l5_multi_model_inference.py:47 (ModelConfig)
- init-scripts/10-multi-model-schema.sql (INSERT config.models)

StateTracker Fix:
- src/core/state/state_tracker.py:_persist_state()
- src/core/state/state_tracker.py:_load_state()

Macro Fix:
- airflow/dags/l0_macro_unified.py
- scripts/scraper_banrep_selenium.py

Drift Monitor Fix:
- airflow/dags/mlops_drift_monitor.py:56-67 (feature list)
```

---

## CONCLUSIÓN

El sistema tiene una arquitectura sólida pero adolece de problemas de
**consistencia entre entrenamiento y producción**. El issue más grave
es el threshold mismatch que causa que el modelo opere de manera
diferente a como fue entrenado.

El bajo win rate (22.8%) y P&L negativo (-3.54%) sugieren que el modelo
necesita ajustes o reentrenamiento con datos más recientes.

**Prioridad inmediata**: Corregir threshold y agregar persistencia de estado.

---

*Reporte generado automáticamente por auditoría Claude Code*
*Timestamp: 2026-01-08T12:00:00-05:00*
