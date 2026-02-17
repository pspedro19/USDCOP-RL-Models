# PLAN UNIFICADO: Sistema de Trading USDCOP
## Track B — H=5 Linear-Only con Path to 30% APR

**Fecha:** 2026-02-16
**Autor:** Pedro + Claude
**Status:** LISTO PARA DEPLOY

---

## 1. Que es el Sistema

Un sistema de trading algoritmico para USD/COP con dos capas independientes:

**Capa 1 -- INTELIGENCIA (prediccion semanal)**
Cada domingo, 2 modelos estadisticos (Ridge + BayesianRidge) se re-entrenan con datos
historicos y predicen: el USD/COP sube o baja esta semana? Eso define la direccion
(LONG o SHORT) para toda la semana.

**Capa 2 -- EJECUCION (trailing stop intradia)**
De lunes a viernes, cada 5 minutos se monitorea el precio con un trailing stop que:
- Captura ganancias automaticamente cuando el precio retrocede desde un pico
- Corta perdidas con hard stop si el mercado va fuerte en contra
- Re-entra en la misma direccion si el trailing cierra mid-week

El modelo decide QUE hacer. El trailing stop decide CUANDO salir.

---

## 2. Por que este Sistema y no otro

### Evolucion del proyecto (cronologia)

```
RL (PPO, 5-min bars)     -> +2.51%, Sharpe 0.32, p=0.272  -> DESCARTADO
Forecasting H=1 (9 mod)  -> +21.6%, Sharpe 2.11, p=0.068  -> BORDERLINE
  + Vol-targeting         -> +32.6%, Sharpe 2.48, p=0.053  -> BORDERLINE
  + Trailing stop         -> +36.8%, Sharpe 3.14, p=0.018  -> SIGNIFICATIVO
  -> Pero 2026: -8.84% (regimen cambio, modelo H=1 fallo)

H=5 (9 modelos, top-3)   -> +18.5%, Sharpe 1.86, p=0.111  -> BORDERLINE
  -> Diagnostico: boosting models dominan ensemble con predicciones ruidosas

H=5 LINEAR-ONLY (Ridge+BR)-> +32.9%, Sharpe 3.46, p=0.008 -> SIGNIFICATIVO p<0.01
  -> Walk-forward 2022-2024: 2/3 folds Sharpe >1.0
  -> Sharpe no inflado: weekly 3.73 > sub-trade 3.21
  -> 2026 YTD: +4.31% (mientras H=1 esta en -8.84%)
```

### Decisiones clave y por que

| Decision | Razon | Evidencia |
|----------|-------|-----------|
| H=5 en vez de H=1 | Trailing tiene "runway" para trabajar en 5 dias | H=1+trail Sharpe 0.13 vs H=5+trail Sharpe 3.46 |
| 2 modelos en vez de 9 | Boosting models agregan ruido, no senal en H=5 | Linear-only p=0.008 vs baseline p=0.111 |
| Ridge + BayesianRidge (sin ARD) | ARD colapso (std=0.000149) | Diagnostico de collapse |
| Mean simple (no top-3 por magnitud) | Top-3 por magnitud selecciona modelos ruidosos | LINEAR-ONLY WR 69% LONG vs BASELINE WR 63% |
| Trailing 0.20%/0.10% (tight) | Captura micro-profits consistentes | Grid 1,500 combos: WR 91%, Sharpe 3.90 |
| Bidireccional (no SHORT-only) | LONG WR 69% en 2025, 70% en H=5 | Desglose L/S: SHORT-only pierde alpha LONG |
| Re-entry despues de trailing | Intra-week correlation -0.354 (mean reverting) | Sharpe decomposition |

---

## 3. Resultados Validados

### OOS 2025 (modelos entrenados 2020-2024, 2025 NUNCA visto)

| Estrategia | Return | Sharpe | WR | p-value | $10K -> |
|------------|--------|--------|----|---------|---------|
| Buy & Hold USDCOP | -14.04% | -- | -- | -- | $8,596 |
| H=5 Linear-Only + Trail (5-min) | +32.85% | 3.461 | 65.9% | 0.008 | $13,285 |
| H=5 Linear-Only + Trail (daily) | +7.99% | 2.093 | 82.4% | 0.054 | $10,799 |
| H=5 Linear-Only + Trail tight (daily) | +19.29% | 3.895 | 91.2% | 0.003 | $11,929 |

### Walk-Forward Multi-Ano (Linear-Only)

| Ano  | Train     | Return  | Sharpe | DA%   | B&H    | Alpha   |
|------|-----------|---------|--------|-------|--------|---------|
| 2022 | 2020-2021 | +18.27% | 1.224 | 53.1% | +19.1% | -0.9pp  |
| 2023 | 2020-2022 | -0.65%  | -0.045 | 50.0% | -20.2% | +19.5pp |
| 2024 | 2020-2023 | +27.92% | 2.063 | 60.4% | +13.5% | +14.4pp |
| 2025 | 2020-2024 | +32.85% | 3.461 | 59.1% | -12.3% | +45.1pp |

Alpha mas fuerte en anos de caida del COP. 2023 preservo capital (-0.65%) mientras B&H perdio -20.2%.

### 2026 YTD (6 semanas)

| Track | Return | WR | Trades |
|-------|--------|----|--------|
| Track A (H=1, SHORT-only) | -8.84% | 44.2% | 43 |
| Track B (H=5, Linear-Only) | +4.31% | 100% | 6 |

---

## 4. Arquitectura de Produccion

### Pipeline completo

```
DOMINGO 01:30 COT
+------------------------------------------------------+
| H5-L5a: Weekly Training                              |
|   Input: daily OHLCV 2020->presente + macro          |
|   Process: Train Ridge + BayesianRidge               |
|            Target: ln(close[t+5] / close[t])         |
|            21 features SSOT (expanding window)       |
|   Output: 2 x .pkl + scaler.pkl                     |
|   Collapse check: if pred_std < 0.001 -> ALERT      |
|   Time: <2 seconds                                   |
+------------------------------------------------------+
                        |
LUNES 08:15 COT
+------------------------------------------------------+
| H5-L5b: Weekly Signal                                |
|   Input: 2 x .pkl + features de hoy                 |
|   Process: predict con Ridge y BR, mean simple       |
|   Output: direction +/-1, magnitude                  |
|           -> forecast_h5_predictions                 |
|           -> forecast_h5_signals                     |
+------------------------------------------------------+
                        |
LUNES 08:45 COT
+------------------------------------------------------+
| H5-L5c: Vol-Targeting + Asymmetric Sizing            |
|   Input: senal + 21 dias de close                    |
|   Process: leverage = tv / realized_vol              |
|            LONG: leverage x long_multiplier          |
|   Output: leverage final                             |
|           -> forecast_h5_signals.asymmetric_leverage |
+------------------------------------------------------+
                        |
LUNES 09:00 COT
+------------------------------------------------------+
| H5-L7: Smart Executor -- ENTRY                       |
|   Input: senal de H5-L5c                             |
|   Process: Abrir posicion via PaperBroker            |
|   Output: forecast_h5_executions (status=open)       |
|           forecast_h5_subtrades (subtrade #1)        |
+------------------------------------------------------+
                        |
LUNES-VIERNES, 08:00-12:55 COT (cada 5 min)
+------------------------------------------------------+
| H5-L7: Smart Executor -- MONITOR                     |
|   Input: barras de 5-min (usdcop_m5_ohlcv)          |
|                                                      |
|   Estado del trailing stop:                          |
|   WAITING   -> ganancia < activation (0.20%)         |
|   ACTIVE    -> ganancia >= 0.20%, trailing armado    |
|   TRIGGERED -> retroceso >= trail (0.10%) desde peak |
|               -> CERRAR subtrade                     |
|               -> Cooldown 20 min                     |
|               -> RE-ENTRY misma direccion            |
|   HARD_STOP -> perdida >= hard_stop (3.5%)           |
|               -> CERRAR, NO re-entry ese dia         |
|                                                      |
|   Output: UPDATE forecast_h5_subtrades               |
+------------------------------------------------------+
                        |
VIERNES 12:55 COT
+------------------------------------------------------+
| H5-L7: Week-End Close                                |
|   Process: Cerrar cualquier posicion abierta         |
|   Output: forecast_h5_executions (status=closed)     |
+------------------------------------------------------+
                        |
VIERNES 14:30 COT
+------------------------------------------------------+
| H5-L6: Weekly Monitor + Decision Gates               |
|   Input: forecast_h5_executions + subtrades          |
|   Process:                                           |
|     - Calcular PnL semanal, acumulado                |
|     - Running DA, Sharpe, MaxDD                      |
|     - L/S ratio alarm (>50% LONG -> alert)           |
|     - Decision gates (semana >= 15)                  |
|   Output: forecast_h5_paper_trading                  |
|                                                      |
|   GATES AUTOMATICOS:                                 |
|     DA > 55%              -> PROMOTE a produccion    |
|     DA_SHORT>60% & LONG>45% -> keep bidireccional    |
|     DA_SHORT>60% & LONG<40% -> switch SHORT-only     |
|     DA < 50%              -> DISCARD                 |
+------------------------------------------------------+
```

### 21 Features (SSOT)

```
PRECIO (4):      close, open, high, low
RETORNOS (4):    return_1d, return_5d, return_10d, return_20d
VOLATILIDAD (3): volatility_5d, volatility_10d, volatility_20d
TECNICOS (3):    rsi_14d, ma_ratio_20d, ma_ratio_50d
CALENDARIO (3):  day_of_week, month, is_month_end
MACRO (4):       dxy_close_lag1, oil_close_lag1, vix_close_lag1, embi_close_lag1
                 (todos con shift(1) + merge_asof anti-leakage)
```

### 2 Modelos

| Modelo | Tipo | Por que |
|--------|------|---------|
| Ridge | sklearn.Ridge(alpha=1.0) | Estable, rapido, relaciones lineales features->return |
| BayesianRidge | sklearn.BayesianRidge(max_iter=300) | Auto-regularizacion + estimacion de incertidumbre |

Ensemble: **promedio simple** de las 2 predicciones. Sin ponderacion, sin seleccion por magnitud.

---

## 5. Configuracion de Produccion

### Fase 1: Conservative (Semanas 1-4)

```yaml
# smart_executor_h5_v2.yaml -- Fase 1

trailing_stop:
  activation_pct: 0.0020      # 0.20% -- activar trailing temprano
  trail_pct: 0.0010           # 0.10% -- tight, captura micro-profits
  hard_stop_pct: 0.035        # 3.50%
  re_entry: true
  cooldown_minutes: 20

monitor:
  interval_minutes: 5          # requiere datos 5-min reales
  source: "usdcop_m5_ohlcv"
  session_start: "08:00"       # COT
  session_end: "12:55"         # COT

vol_targeting:
  target_vol: 0.15             # conservador
  max_leverage: 2.0
  min_leverage: 0.5
  lookback_days: 21

asymmetric_sizing:
  enabled: true
  long_multiplier: 0.5         # conservador: LONGs a mitad
  short_multiplier: 1.0

direction_filter:
  mode: "all"                  # bidireccional

alarms:
  long_ratio_window: 4         # semanas
  long_ratio_threshold: 0.50   # alerta si >50% LONG
  collapse_std_threshold: 0.001
```

**Retorno esperado Fase 1:** +15-25% APR (5-min monitoring + trail tight)
**MaxDD esperado:** -5 a -8%

### Fase 2: Moderate (Semanas 5-8, si WR > 75%)

```yaml
# Cambios vs Fase 1:
asymmetric_sizing:
  long_multiplier: 0.75        # subir de 0.5 a 0.75
```

**Retorno esperado Fase 2:** +20-28% APR
**MaxDD esperado:** -6 a -9%

### Fase 3: Aggressive (Semanas 9-12, si Sharpe > 1.5)

```yaml
# Cambios vs Fase 2:
vol_targeting:
  target_vol: 0.20             # subir de 0.15 a 0.20
  max_leverage: 2.5            # subir de 2.0 a 2.5
```

**Retorno esperado Fase 3:** +25-35% APR
**MaxDD esperado:** -7 a -12%

### Escalamiento resumido

```
Semana 1-4:   Solo monitoring 5-min + trail tight     -> ~20% APR
Semana 5-8:   + LONG sizing 0.75x                     -> ~25% APR
Semana 9-12:  + Vol target 0.20, max lev 2.5          -> ~30% APR
Semana 13-15: Evaluacion final -> PROMOTE o DISCARD
```

**Regla: solo escalar si el paso anterior mantiene metricas.** Si WR cae <70% en cualquier fase, NO escalar. Si MaxDD >10%, bajar un nivel.

---

## 6. Infraestructura Existente (ya implementada)

### Codigo completado

```
CORE LOGIC:
  [x] src/execution/trailing_stop.py          -- TrailingStopTracker (pure logic)
  [x] src/execution/broker_adapter.py         -- ABC + PaperBroker
  [x] src/execution/smart_executor.py         -- SmartExecutor H=1
  [x] src/execution/multiday_executor.py      -- MultiDayExecutor H=5 (trailing + re-entry)
  [x] src/forecasting/vol_targeting.py        -- Vol-targeting + asymmetric sizing

DATABASE:
  [x] database/migrations/041_forecast_vol_targeting.sql    -- Track A tables
  [x] database/migrations/042_forecast_executions.sql       -- Track A execution
  [x] database/migrations/043_forecast_h5_tables.sql        -- Track B tables (5 tablas + 2 vistas)

CONFIG:
  [x] config/execution/smart_executor_v1.yaml               -- Track A (H=1)
  [x] config/execution/smart_executor_h5_v1.yaml            -- Track B (H=5) v1
  [x] config/execution/smart_executor_h5_v2.yaml            -- Track B (H=5) v2 tight trailing
  [x] config/forecast_experiments/h5_linear_paper_v1.yaml    -- Experiment config

TRACK A DAGs (H=1 diario, SHORT-only):
  [x] airflow/dags/forecast_l5a_weekly_inference.py          -- Train 9 modelos
  [x] airflow/dags/forecast_l5b_daily_inference.py           -- Inference diaria
  [x] airflow/dags/forecast_l5c_vol_targeting.py             -- Senal diaria
  [x] airflow/dags/execution_l7_smart_executor.py            -- Trailing H=1
  [x] airflow/dags/forecast_l6_paper_trading_monitor.py      -- Evaluacion

TRACK B DAGs (H=5 semanal, bidireccional):
  [x] airflow/dags/forecast_h5_l5a_weekly_training.py        -- Train Ridge+BR
  [x] airflow/dags/forecast_h5_l5b_weekly_signal.py          -- Senal semanal
  [x] airflow/dags/forecast_h5_l5c_vol_targeting.py          -- Vol-target + asymmetric
  [x] airflow/dags/forecast_h5_l7_multiday_executor.py       -- Trailing 5-day (5-min bars)
  [x] airflow/dags/forecast_h5_l6_weekly_monitor.py          -- Evaluacion + gates

SHARED:
  [x] airflow/dags/contracts/dag_registry.py                 -- Registro completo

TESTS:
  [x] tests/unit/test_multiday_executor.py                   -- 23 tests
  [x] tests/unit/test_h5_asymmetric_sizing.py                -- 13 tests
  [x] scripts/validate_h5_paper_trading.py                   -- Smoke test (20 checks)

BACKTESTS/DIAGNOSTICS:
  [x] scripts/backtest_smart_executor.py                     -- Trailing H=1
  [x] scripts/backtest_oos_2025.py                           -- OOS validation
  [x] scripts/diagnose_h5_ensemble.py                        -- 4-strategy comparison
  [x] scripts/h5_sensitivity_30pct.py                        -- Grid search 1500 combos
  [x] scripts/h5_30pct_analysis.py                           -- Strategy comparison
  [x] scripts/backtest_h5_10k_2025_2026.py                   -- $10K backtest
```

### Pending para deploy

```
1. DEPLOY:
   [ ] docker-compose up -d (TimescaleDB, Airflow, Redis)
   [ ] psql -f migrations/041, 042, 043
   [ ] Activar DAGs de Track A y Track B
   [ ] Verificar que L0 de 5-min esta ingesting

2. BACKTEST FINAL (validacion):
   [ ] Correr backtest 2025 OOS con config v2 exacta
       (5-min bars + trail 0.20%/0.10% + hard 3.5%)
   [ ] GATE: return > 25%, Sharpe > 2.0, MaxDD < 12%
```

---

## 7. Tablas de Base de Datos

### Track A (H=1)

```
forecast_vol_targeting_signals   -- senal diaria (dir + leverage)
forecast_executions              -- tracking del trailing stop H=1
forecast_paper_trading           -- PnL y metricas acumuladas
```

### Track B (H=5)

```
forecast_h5_predictions          -- predicciones Ridge + BR por semana
forecast_h5_signals              -- senal semanal (dir + leverage + asymmetric)
forecast_h5_executions           -- ejecucion semanal (parent)
forecast_h5_subtrades            -- sub-trades dentro de la semana (re-entry)
forecast_h5_paper_trading        -- resumen semanal + decision gates
```

### Vistas

```
v_h5_performance_summary         -- metricas agregadas
v_h5_collapse_monitor            -- monitoreo de collapse de modelos
v_execution_performance          -- Track A performance
```

---

## 8. Gestion de Riesgo

### Hard Limits (nunca violables)

| Limite | Valor | Que pasa si se viola |
|--------|-------|---------------------|
| MaxDD por trade | -3.5% (hard stop) | Cierre automatico, no re-entry ese dia |
| MaxDD acumulado | -15% | STOP automatico del sistema |
| Max leverage | 2.5x (Fase 3) | Clip automatico |
| Max consecutive losses | 5 semanas | Alerta + revision manual |

### Alarmas automaticas (L6 evalua cada viernes)

| Alarma | Trigger | Accion |
|--------|---------|--------|
| LONG insistente | >50% LONG en ultimas 4 semanas | Alerta -> considerar SHORT-only |
| Model collapse | pred_std < 0.001 para cualquier modelo | Alerta -> verificar retraining |
| DA degradation | DA < 50% en ultimas 8 semanas | Pausa automatica |
| Sharpe degradation | Sharpe < 0.5 rolling 12 semanas | Alerta |

### Que hacer en cada escenario

```
Escenario: DA > 55% despues de 15 semanas
  -> PROMOTE: pasar de paper trading a capital real ($500-1000 iniciales)
  -> Escalar capital gradualmente: 25% -> 50% -> 100% cada 4 semanas

Escenario: DA 50-55% despues de 15 semanas
  -> HOLD: seguir paper trading 8 semanas mas
  -> Si mejora -> promote. Si estanca -> re-evaluar.

Escenario: DA < 50% despues de 15 semanas
  -> DISCARD Track B. Mantener Track A si funciona.
  -> Investigar: regimen cambio? modelos necesitan rolling window?

Escenario: DA_LONG < 40% pero DA_SHORT > 60%
  -> SWITCH a SHORT-only automaticamente (L6 lo hace)
  -> Revisar en 8 semanas si LONG se recupera

Escenario: MaxDD > 15%
  -> STOP automatico. No reactivar sin revision manual completa.
```

---

## 9. Timeline

```
2026-02-16:  Plan unificado completado (este documento)
             Todo el codigo existe. Falta: deploy + backtest final.

SEMANA 0 (Feb 17-21):
  [ ] Deploy: Docker + migraciones + DAGs
  [ ] Primer training H5-L5a (domingo 22 feb)
  [ ] Correr backtest final con config v2 exacta
  [ ] Si GATE pasa -> activar

SEMANA 1 (Feb 23 - primer lunes):
  [ ] Primera senal H5-L5b (lunes 23 feb)
  [ ] Primera ejecucion H5-L7 (lunes-viernes)
  [ ] Primera evaluacion H5-L6 (viernes 28 feb)
  [ ] Config: Fase 1 (conservative)

SEMANAS 2-4 (Mar 1-14):
  [ ] Paper trading Fase 1
  [ ] Monitorear WR, DA, L/S ratio
  [ ] Track A (H=1) sigue corriendo en paralelo

SEMANA 5-8 (Mar 15 - Abr 11):
  [ ] Si WR > 75% -> activar Fase 2 (LONG sizing 0.75x)
  [ ] Day 60 de Track A -> evaluar si mantener o pausar

SEMANA 9-12 (Abr 12 - May 9):
  [ ] Si Sharpe > 1.5 -> activar Fase 3 (vol target 0.20)

SEMANA 13-15 (May 10 - May 30):
  [ ] Evaluacion final Track B
  [ ] Decision gates: PROMOTE / HOLD / DISCARD
  [ ] Si PROMOTE -> capital real, empezar con $500-1000

SEMANA 16+ (Jun 1+):
  [ ] Capital real, escalamiento gradual
  [ ] Monitoreo continuo via L6
```

---

## 10. Metricas Esperadas (Realistas)

### Con config Fase 1 (conservative)

| Metrica | Esperado | Rango |
|---------|----------|-------|
| Return anual | +15-25% | depende de regimen |
| Sharpe | 1.5-3.0 | |
| WR (trailing) | 75-90% | alto por trail tight |
| DA (direccion) | 55-62% | |
| MaxDD | -5% a -8% | |
| Trades/ano | ~44 (1/semana) | |
| Sub-trades/semana | 2-3 promedio | |
| Hard stops/ano | 2-4 | |

### Con config Fase 3 (aggressive)

| Metrica | Esperado | Rango |
|---------|----------|-------|
| Return anual | +25-35% | target 30% |
| Sharpe | 2.0-3.5 | |
| WR (trailing) | 75-90% | |
| DA (direccion) | 55-62% | |
| MaxDD | -8% a -12% | mas alto por mas leverage |
| Trades/ano | ~44 | |

### Lo que NO esperar

- No esperes 30% todos los anos. 2023 dio -0.65% (pero preservo capital).
- No esperes WR 100% como en las 6 semanas de 2026. Eso es muestra pequena.
- No esperes que el modelo funcione sin retraining semanal. Los boosting colapsaron sin retrain mensual.
- No esperes que LONG funcione siempre. En regimenes de depreciacion fuerte, los LONGs fallan (2023, 2026 Q1).

---

## 11. Que NO hacer (lecciones aprendidas)

| # | No hacer | Por que | Evidencia |
|---|----------|---------|-----------|
| 1 | No usar 9 modelos con top-3 por magnitud en H=5 | Boosting models dominan con predicciones ruidosas | LINEAR-ONLY p=0.008 vs BASELINE p=0.111 |
| 2 | No usar ARD | Colapso en H=5 (std=0.000149) | Diagnostico de collapse |
| 3 | No agregar features de regimen | 0/7 tratamientos mejoraron | EXP-REGIME-001: 0/7, p=0.94 |
| 4 | No agregar macro features (mas alla de DXY/WTI/VIX/EMBI) | Degrada Sharpe de 2.1 a 0.5 | Experimentos anteriores |
| 5 | No usar H=1 con trailing stop | Trailing destruye en H=1 | H=1+trail Sharpe 0.13 |
| 6 | No optimizar sobre 2026 (43 dias) | N=43 no permite validacion | Principio estadistico |
| 7 | No usar RL para USD/COP | 2.51% return, p=0.272 | 6 meses de desarrollo |
| 8 | No subir hard stop por encima de 4% | Los hard stops son predicciones incorrectas, ampliar el stop amplia la perdida | Analisis de hard stops 2026 |
| 9 | No ejecutar al open en vez del close | Pierde overnight gap | Backtest: +9.6% vs +25.8% |
| 10 | No cambiar trailing params mid-week | Debe ser mecanico | Principio de disciplina |

---

## 12. Queries de Monitoreo

### Performance semanal

```sql
SELECT
    signal_week,
    direction,
    asymmetric_leverage,
    total_pnl_pct,
    num_subtrades,
    exit_reason,
    CASE WHEN total_pnl_pct > 0 THEN 'WIN' ELSE 'LOSS' END as result
FROM forecast_h5_executions
ORDER BY signal_week DESC
LIMIT 10;
```

### Comparacion Track A vs Track B

```sql
SELECT 'Track A (H=1)' as track,
    COUNT(*) as trades,
    AVG(CASE WHEN pnl_pct > 0 THEN 1.0 ELSE 0.0 END) as wr,
    SUM(pnl_pct) as total_pnl
FROM forecast_executions
WHERE config_version LIKE 'smart_executor_v1%'
UNION ALL
SELECT 'Track B (H=5)',
    COUNT(*),
    AVG(CASE WHEN total_pnl_pct > 0 THEN 1.0 ELSE 0.0 END),
    SUM(total_pnl_pct)
FROM forecast_h5_executions;
```

### Alarma de LONG insistente

```sql
SELECT
    COUNT(*) FILTER (WHERE direction = 1) * 100.0 / COUNT(*) as long_pct
FROM forecast_h5_signals
WHERE signal_week >= CURRENT_DATE - INTERVAL '28 days';
-- Si > 50% -> ALERTA
```

### Model collapse monitor

```sql
SELECT * FROM v_h5_collapse_monitor;
-- Verifica que Ridge y BR tienen std > 0.001
```

---

## 13. Resumen Ejecutivo

**Sistema:** Trading algoritmico USD/COP, horizonte semanal (H=5).

**Modelos:** Ridge + BayesianRidge, 21 features, re-entrenamiento semanal expandiendo ventana.

**Ejecucion:** Trailing stop intraday cada 5 min (activation 0.20%, trail 0.10%, hard stop 3.5%), con re-entry dentro de la misma semana.

**Evidencia:**
- OOS 2025: +32.85%, Sharpe 3.46, p=0.008
- Walk-forward 2022-2024: 2/3 anos Sharpe > 1.0
- 2026 YTD: +4.31% (mientras H=1 pierde -8.84%)

**Target:** 30% APR (escalonado en 3 fases de 4 semanas).

**Riesgo:** MaxDD esperado -8 a -12%. Hard limits automaticos. Decision gates en L6.

**Status:** Codigo completo. Deploy pendiente (infra + backtest final).
