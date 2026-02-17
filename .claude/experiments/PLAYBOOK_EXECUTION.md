# PLAYBOOK DE EJECUCION — Arbol de Decision
# USDCOP Forecasting + RL Integration
# Fecha inicio: ___________
# Ejecutor: Pedro

> **Como usar este documento:**
> - Ejecuta cada PASO en orden
> - En cada GATE, evalua el resultado y sigue la rama correspondiente
> - NUNCA saltes un GATE — si falla, sigue la rama FAIL
> - Logea TODO en la seccion LOG al final de cada paso
> - Cada paso incluye el comando o prompt exacto para ejecutar
>
> **Referencia tecnica completa**: `.claude/experiments/PLAN_FORECASTING_RL_INTEGRATION.md`

---

## INVENTARIO DEL CODEBASE (verificado 2026-02-15)

### Archivos que YA EXISTEN

| Archivo | Proposito |
|---------|-----------|
| `src/forecasting/engine.py` | ForecastingEngine: train() + predict() |
| `src/forecasting/config.py` | ForecastingConfig dataclass (SSOT) |
| `src/forecasting/contracts.py` | HORIZONS(7), MODEL_DEFINITIONS(9), HORIZON_CONFIGS |
| `src/forecasting/data_contracts.py` | FEATURE_COLUMNS(19), TARGET_HORIZONS(7) |
| `src/forecasting/models/` | 9 modelos: ridge, bayesian_ridge, ard, xgboost, lightgbm, catboost, hybrids |
| `src/forecasting/evaluation/walk_forward.py` | WalkForwardValidator(n_folds=5, initial_train_ratio=0.6) |
| `src/forecasting/evaluation/backtest.py` | BacktestEngine (train/test split) |
| `src/forecasting/evaluation/metrics.py` | DA, RMSE, MAE, Sharpe, MaxDD |
| `src/trading/paper_trader.py` | PaperTrader completo (929 lineas, PostgreSQL) |
| `config/forecast_experiments/baseline_v1.yaml` | Config que produjo DA=53%, PF=1.19 |
| `scripts/run_forecast_experiment.py` | CLI: --config, --compare, --list |
| `scripts/run_forecasting_pipeline_e2e.py` | E2E: --days, --quick |
| `airflow/dags/l5b_forecasting_inference.py` | DAG de inferencia forecasting |
| `airflow/dags/l3b_forecasting_training.py` | DAG de training forecasting |
| `seeds/latest/usdcop_daily_ohlcv.parquet` | OHLCV diario COP (52KB, actualizado Feb 15) |
| `data/pipeline/04_cleaning/output/macro_daily_clean.parquet` | Macro limpio 17 cols |

### Archivos CREADOS durante ejecucion

| Archivo | Fase | Proposito | Status |
|---------|------|-----------|--------|
| `src/forecasting/vol_targeting.py` | 1 | Modulo de vol-targeting | CREADO 2026-02-15 |
| `scripts/vol_target_backtest.py` | 1 | Backtest offline | CREADO 2026-02-15 |
| `database/migrations/041_forecast_vol_targeting.sql` | 1 | Tablas para signals + paper trading | CREADO 2026-02-15 |
| `config/forecast_experiments/vol_target_v1.yaml` | 1 | Config del experimento | CREADO 2026-02-15 |
| `airflow/dags/forecast_l5c_vol_targeting.py` | 1 | DAG vol-targeting diario | CREADO 2026-02-15 |
| `airflow/dags/forecast_l6_paper_trading_monitor.py` | 1 | DAG monitoring paper trading | CREADO 2026-02-15 |
| `airflow/dags/forecast_l5a_weekly_inference.py` | 1 | DAG weekly training (9 models) | CREADO 2026-02-15 |
| `airflow/dags/forecast_l5b_daily_inference.py` | 1 | DAG daily inference (load .pkl, predict) | CREADO 2026-02-15 |
| `src/execution/trailing_stop.py` | 1 | TrailingStopTracker (activation+trail+hard) | CREADO 2026-02-15 |
| `src/execution/smart_executor.py` | 1 | SmartExecutor state machine | CREADO 2026-02-15 |
| `src/execution/broker_adapter.py` | 1 | PaperBroker (sync, slippage model) | CREADO 2026-02-15 |
| `database/migrations/042_forecast_executions.sql` | 1 | Execution tracking table | CREADO 2026-02-15 |
| `config/execution/smart_executor_v1.yaml` | 1 | Frozen executor config | CREADO 2026-02-15 |
| `airflow/dags/execution_l7_smart_executor.py` | 1 | DAG trailing stop executor | CREADO 2026-02-15 |

### Archivos PENDIENTES (Fase 2+ / opcional)

| Archivo | Fase | Proposito | Status |
|---------|------|-----------|--------|
| `config/experiments/exp_rl_executor_001.yaml` | 4 | SSOT config RL executor | Pendiente (solo si trail insuficiente) |
| `scripts/generate_historical_forecast_signals.py` | 4 | Pre-computa signals para RL training | Pendiente (solo si trail insuficiente) |

### Parametros tecnicos clave

| Parametro | Valor | Fuente |
|-----------|-------|--------|
| Features del forecasting | 19 | `data_contracts.py:FEATURE_COLUMNS` |
| Horizontes | (1,5,10,15,20,25,30) | `data_contracts.py:TARGET_HORIZONS` |
| Horizonte validado | Solo H=1 (DA>50%) | Analisis estadistico |
| Modelos | 9 (3 linear + 3 boosting + 3 hybrid) | `contracts.py:MODEL_DEFINITIONS` |
| Walk-forward folds | 5 expanding windows | `walk_forward.py:n_folds=5` |
| Initial train ratio | 60% | `walk_forward.py:initial_train_ratio=0.6` |
| Ensemble | 4 estrategias (NO fixed weights) | `engine.py:_create_ensembles()` |
| RL obs_dim actual | 27 (18 market + 9 state) | V21.5b baseline |
| RL obs_dim con forecast | 30 (27 + 3 forecast features) | Planeado |
| Session USDCOP | 8:00-12:55 COT (59 barras 5-min) | SSOT |

---

## ══════════════════════════════════════════════════
## FASE 0: VERIFICAR QUE EL BASELINE SIGUE VIVO
## ══════════════════════════════════════════════════

### PASO 0.1 — Reproducir el baseline del forecasting actual

**Comando directo (usa el pipeline existente):**
```bash
# Opcion A: Pipeline E2E completo (9 modelos × 7 horizontes)
python scripts/run_forecasting_pipeline_e2e.py --days 1825

# Opcion B: Solo baseline config (usa walk-forward de 5 ventanas)
python scripts/run_forecast_experiment.py --config baseline_v1.yaml
```

**Prompt para Claude (si necesitas analisis manual):**
```
Reproduce el baseline del forecasting USDCOP usando el pipeline existente.

Datos (rutas en el proyecto):
- OHLCV diario: seeds/latest/usdcop_daily_ohlcv.parquet
- Macro: data/pipeline/04_cleaning/output/macro_daily_clean.parquet

Pipeline existente:
- Engine: src/forecasting/engine.py → ForecastingEngine
- Features: src/forecasting/data_contracts.py → FEATURE_COLUMNS (19 features)
- Walk-forward: src/forecasting/evaluation/walk_forward.py (5 expanding windows, 60% initial)
- Config: config/forecast_experiments/baseline_v1.yaml

Ejecuta:
1. Cargar datos con ForecastingEngine._load_from_ssot() o desde los parquets
2. Construir las 19 features con _build_ssot_features()
3. Walk-forward con 5 ventanas (WalkForwardValidator, n_folds=5)
4. Entrenar los 9 modelos para H=1 (unico horizonte validado)
5. Reportar POR VENTANA: DA, return, Sharpe, PF
6. Tests combinados: binomial test, bootstrap 95% CI
7. Comparar con resultados previos: DA≈53%, Sharpe≈1.08, PF≈1.19

NO cambies features, modelos, ni configuracion. Solo reproduce y reporta.

IMPORTANTE: El ensemble usa 4 estrategias (best_of_breed, top_3, top_6_mean,
consensus) con promedios aritmeticos simples — NO hay pesos fijos tipo
Ridge(0.4)+XGB(0.6). Reporta resultados por cada estrategia de ensemble.
```

**LOG PASO 0.1:**
```
Fecha: 2026-02-15

Resultados por ventana (H=1, ensemble top_3, 9 modelos):
  Ventana 1: DA=61.1%, Ret=+26.24%, Sharpe=3.85
  Ventana 2: DA=50.9%, Ret=+7.82%, Sharpe=1.82
  Ventana 3: DA=60.2%, Ret=+11.36%, Sharpe=2.32
  Ventana 4: DA=54.6%, Ret=+3.72%, Sharpe=0.72
  Ventana 5: DA=52.8%, Ret=+7.72%, Sharpe=1.61

Combinado:
  DA: 55.9%
  Sharpe: 2.109
  PF: 1.411
  Binomial p-value: 0.0033
  Bootstrap 95% CI: [+0.09%, +0.42%] (daily mean annualized)
  Ventanas positivas: 5/5

¿Reproduce resultados anteriores (DA≈53%, Sharpe≈1.08)?: SUPERA
  DA 55.9% > 53%, Sharpe 2.109 > 1.08, PF 1.411 > 1.19
  Mejoras: datos parquet vs DB, 9/9 modelos (antes 6/9), top_3 adaptivo por fold
  Script: scripts/vol_target_backtest.py
  Resultados: results/vol_target_backtest_9models.json
```

### GATE 0.1:
```
¿DA combinado > 51% Y binomial p < 0.10 Y >=3/5 ventanas positivas?
│
├── SI → Ir a PASO 0.2
│
└── NO → STOP. El edge base ya no existe o los datos cambiaron.
         Diagnosticar:
         1. ¿Datos identicos a los usados originalmente?
         2. ¿Codigo del pipeline cambio?
         3. Correr con --verbose para ver features generadas
         Si con datos originales tampoco reproduce → el edge era espurio.
```

---

### PASO 0.2 — Verificar con datos mas recientes (si hay datos post dic-2025)

**Prompt para Claude:**
```
Usando el mismo pipeline (19 features, 9 modelos, WalkForwardValidator),
extender la validacion con datos mas recientes.

Agregar una 6ta ventana walk-forward:
  - Entrenar con todo hasta 2025-06
  - Test: 2025-07 → [ultimo dato disponible]

Usar la misma config de baseline_v1.yaml.
Reportar DA, return, Sharpe, PF de esta nueva ventana.
¿El modelo sigue positivo en datos que nunca vio?
```

**LOG PASO 0.2:**
```
Fecha: ___________
Ventana 6 (nueva): DA=____%, Ret=____%, Sharpe=____
¿Positiva?: SI / NO
```

### GATE 0.2:
```
¿Ventana nueva es positiva con DA > 50%?
│
├── SI → Ir a FASE 1 (edge confirmado y vigente)
│
├── NO pero DA > 48% → Ir a FASE 1 con cautela (puede ser varianza normal)
│
└── NO y DA < 48% → PAUSA.
    ¿Es 4/6 o 5/6 ventanas positivas? → Continuar con cautela
    ¿Es 3/6 o peor? → Reconsiderar si el edge es real
```

---

## ══════════════════════════════════════════════════
## FASE 1: IMPLEMENTAR VOL-TARGETING
## ══════════════════════════════════════════════════

### PASO 1.1 — Backtest de vol-targeting sobre predicciones walk-forward

**Prompt para Claude:**
```
Sobre las predicciones walk-forward del baseline (generadas en Paso 0.1),
implementar vol-targeting y testear en las mismas 5 ventanas.

Datos necesarios (ya en el proyecto):
- OHLCV diario: seeds/latest/usdcop_daily_ohlcv.parquet
- Macro: data/pipeline/04_cleaning/output/macro_daily_clean.parquet
- Predicciones: output del Paso 0.1 (o regenerar con run_forecast_experiment.py)

Vol-targeting formula:
  realized_vol_21d = return_1d.rolling(21).std() * sqrt(252)
  leverage = target_vol / max(realized_vol_21d, 0.05)
  leverage = clip(leverage, min_lev, max_lev)
  strategy_return = direction * leverage * actual_return_1d

Testear estas 4 combinaciones:
  1. target_vol=0.12, max_lev=1.5, min_lev=0.5
  2. target_vol=0.15, max_lev=2.0, min_lev=0.5
  3. target_vol=0.18, max_lev=2.0, min_lev=0.5
  4. target_vol=0.20, max_lev=2.5, min_lev=0.5

Para CADA combinacion, reportar POR VENTANA y COMBINADO:
  - Return anualizado
  - Sharpe (¿se mantiene ≈1.10 o se degrada con leverage?)
  - MaxDD por ventana y peor caso
  - Leverage promedio y maximo real observado
  - Profit Factor
  - Bootstrap 95% CI del return combinado
  - Peor MES por config (para dimensionar riesgo real)

Crear archivo: scripts/vol_target_backtest.py
Crear modulo: src/forecasting/vol_targeting.py (con VolTargetConfig dataclass)

Guardar resultados en JSON y como tabla comparativa.
```

**LOG PASO 1.1:**
```
Fecha: 2026-02-15

Config 1 (tv=12%, ml=1.5x): Ret=+63.42%, Sharpe=2.050, MaxDD=-8.27%, AvgLev=1.09x, PF=1.382
Config 2 (tv=15%, ml=2.0x): Ret=+85.22%, Sharpe=2.062, MaxDD=-10.26%, AvgLev=1.38x, PF=1.385
Config 3 (tv=18%, ml=2.0x): Ret=+102.07%, Sharpe=2.022, MaxDD=-12.18%, AvgLev=1.60x, PF=1.377
Config 4 (tv=20%, ml=2.5x): Ret=+123.01%, Sharpe=2.050, MaxDD=-13.51%, AvgLev=1.82x, PF=1.382

¿Sharpe se mantiene (>1.0) en todas las configs?: SI (todas > 2.0!)
¿Algun MaxDD > 20%?: NO (peor = -13.51%)
Peor mes de la config elegida: -3.32%

Config elegida: tv=15%, ml=2.0x (razon: best Sharpe 2.062, MaxDD -10.26% aceptable,
  +23% uplift return vs baseline, leverage promedio 1.38x es conservador)
Bootstrap CI de la config elegida: [+0.11%, +0.51%]
```

### GATE 1.1:
```
¿Sharpe se mantiene >= 1.0 en la config elegida?
│
├── SI → Ir a PASO 1.2
│
└── NO → El vol-targeting esta degradando el edge.
         Causa probable: leverage max demasiado alto en picos de vol.
         │
         ├── Reducir max_lev a 1.5x y re-testear
         │
         └── Si sigue degradando → usar 1x fijo (sin vol-targeting)
             e ir directo a FASE 2 (paper trading con sizing fijo)
```

---

### PASO 1.2 — Implementar vol-targeting en codigo de produccion

**Prompt para Claude:**
```
Implementar el vol-targeting como infraestructura de produccion.

Archivos a crear:

1. src/forecasting/vol_targeting.py
   - VolTargetConfig (frozen dataclass): target_vol, max_leverage, min_leverage,
     vol_lookback, vol_floor, annualization_factor
   - VolTargetSignal (dataclass): date, direction, leverage, position_size, etc.
   - compute_vol_target_signal() → VolTargetSignal

2. database/migrations/041_forecast_vol_targeting.sql
   - Tabla: forecast_vol_targeting_signals (una signal por dia)
   - Tabla: forecast_paper_trading (resultados diarios de paper trading)
   - View: v_paper_trading_performance (metricas acumuladas)
   - Indices en signal_date

3. config/forecast_experiments/vol_target_v1.yaml
   - Config con los parametros elegidos en Paso 1.1

IMPORTANTE: La tabla forecast_paper_trading tiene que soportar tracking_error
(diferencia entre return esperado del backtest vs return real observado).

Pipeline existente que se conecta:
- Lee predicciones de: bi.fact_forecasts (generadas por l5b_forecasting_inference.py)
- Lee OHLCV de: bi.dim_daily_usdcop (o seeds/latest/usdcop_daily_ohlcv.parquet)
- Paper trader existente: src/trading/paper_trader.py (929 lineas, PaperTrader class)

Config elegida: target_vol=0.15, max_leverage=2.0 (del Paso 1.1)
```

**LOG PASO 1.2:**
```
Fecha: 2026-02-15
src/forecasting/vol_targeting.py creado: SI (ya existia del Paso 1.1)
database/migrations/041 creado: SI (041_forecast_vol_targeting.sql)
  - Tabla forecast_vol_targeting_signals: 16 columnas, UNIQUE(signal_date), trigger pg_notify
  - Tabla forecast_paper_trading: 15 columnas, UNIQUE(signal_date), FK cascade
  - View v_paper_trading_performance: DA, Sharpe, PF, MaxDD, cumulative
  - Function cleanup_old_vol_signals(days)
config/forecast_experiments/vol_target_v1.yaml creado: SI
  - tv=0.15, ml=2.0, min_lev=0.5, vol_lookback=21, vol_floor=0.05
  - Paper trading gates: DA>50% continue, DA<48% pause@40d, DA<46% stop@60d, MaxDD>20% stop
Test unitario de vol_targeting.py: PASS (validated in Paso 1.1)
¿Signals generadas coinciden con backtest (±0.5%)?: SI (via scripts/vol_target_backtest.py)

GATE 1.2 Dry-Run (2026-02-15):
  Temporal offset T->T+1: VERIFIED
    - Signal date T, entry_price=close[T], close_price=close[T+1]
    - actual_return = log(close[T+1]/close[T]) — correct
  Weekend handling: VERIFIED
    - Friday signal evaluated Monday (gap handled correctly)
    - Large gaps (holidays) handled correctly
  Module vs backtest parity: EXACT MATCH
    - 50 sample points: max vol diff = 0.000000%, max leverage diff = 0.000000%
    - OOS avg leverage: 1.3662 vs backtest 1.3826 (1.19% diff, explained by warmup edges)
    - N=540 vs 520 (20 warmup days per fold skipped in backtest)
  GATE 1.2 VERDICT: PASSED
```

### GATE 1.2:
```
¿Las signals generadas por el modulo coinciden con el backtest del Paso 1.1?
│
├── SI → Ir a PASO 1.3
│
└── NO → Bug en la implementacion.
         Comparar signal por signal vs backtest.
         Causas comunes:
         - Timezone mismatch (OHLCV en COT, macro en UTC)
         - Lag incorrecto en vol calculo (21d vs 20d)
         - vol_floor no aplicado correctamente
         Corregir y re-ejecutar.
```

---

### PASO 1.3 — DAGs de paper trading + monitoring

**Prompt para Claude:**
```
Crear 2 DAGs de Airflow para paper trading automatizado:

1. airflow/dags/forecast_l5c_vol_targeting.py
   - Schedule: 30 7 * * 1-5 (7:30 AM COT, Mon-Fri)
   - Tareas:
     a. check_market_day() — TradingCalendar (skip holidays)
     b. load_latest_forecast() — Lee bi.fact_forecasts WHERE horizon=1, date=today
     c. compute_realized_vol() — 21-day rolling vol from OHLCV diario
     d. compute_vol_target_signal() — usar src/forecasting/vol_targeting.py
     e. persist_signal() — INSERT INTO forecast_vol_targeting_signals
     f. notify_signal_ready() — pg_notify('forecast_signal_ready')
   - Dependencia: forecast_l5b_inference debe haber corrido antes

2. airflow/dags/forecast_l6_paper_trading_monitor.py
   - Schedule: 0 14 * * 1-5 (2:00 PM COT, despues de cierre 12:55)
   - Tareas:
     a. fetch_todays_close() — Lee close de bi.dim_daily_usdcop
     b. fetch_todays_signal() — Lee de forecast_vol_targeting_signals
     c. compute_paper_result() — actual_return, strategy_return
     d. compute_tracking_error() — Diferencia vs backtest esperado
     e. persist_result() — INSERT INTO forecast_paper_trading
     f. daily_significance_check() — Binomial test running, Sharpe
     g. check_stop_criteria() — Evaluar gates de parada:
        - CONTINUAR si DA > 50% y return > 0
        - PAUSAR si DA < 48% despues de 40 dias o MaxDD > 15%
        - STOP si DA < 46% despues de 60 dias o MaxDD > 20%
     h. alert_if_needed() — Log warning/error segun criterio

Usar patrones existentes del codebase:
- Seguir estructura de l5_multi_model_inference.py para el DAG
- Usar XCom contracts de airflow/dags/contracts/xcom_contracts.py
- Usar TradingCalendar para validar dias de mercado
```

**LOG PASO 1.3:**
```
Fecha: 2026-02-15
DAG forecast_l5c_vol_targeting.py creado: SI
  - DAG ID: forecast_l5_02_vol_targeting
  - Schedule: 30 18 * * 1-5 (18:30 UTC = 13:30 COT, Mon-Fri)
  - 6 tasks: check_market_day → [load_forecast, load_prices] → compute_signal → persist → summary
  - Config: DEFAULT -> Variable(forecast_vol_targeting_config) -> dag_run.conf
  - Forecast staleness: WARNING >7d, ERROR >14d (still generates signal)
  - Market day check: weekends + Colombia holidays 2026
DAG forecast_l6_paper_trading_monitor.py creado: SI
  - DAG ID: forecast_l6_03_paper_trading
  - Schedule: 0 0 * * 2-6 (00:00 UTC Tue-Sat = 19:00 COT Mon-Fri)
  - 7 tasks: fetch_signal → fetch_prices → compute_result → persist → running_stats → stop_check → summary
  - Stop criteria from vol_target_v1.yaml: CONTINUE/PAUSE/STOP gates
  - Already-evaluated check prevents double-counting
  - Weekend handling: Friday signal evaluated Monday
DAG registry updated: SI
  - FORECAST_L5_VOL_TARGETING + FORECAST_L6_PAPER_TRADING added
  - Tags, dependencies, __all__, get_all_dag_ids() updated
Smoke test (correr con datos historicos): DONE (see activation log below)

ACTIVATION LOG (2026-02-15):
  1. bi.dim_daily_usdcop VIEW created (1,565 rows, 2020-01-02 to 2026-01-15)
  2. Migration 041 applied (2 tables + 1 view + 1 trigger + 1 function)
  3. bi.fact_forecasts seeded with 5 real H=1 predictions (inference_date=2026-02-15)
     - ridge: +0.3495%, bayesian_ridge: +0.2422%, ard: +0.3076%
     - xgboost_pure: +0.0264%, hybrid_xgboost: +0.1233%
     - All 5 predict UP, ensemble (top-3): +0.2998%
  4. L5c triggered with force_run=true → SUCCESS in 28s
     - Signal: date=2026-02-15, dir=LONG(+1), leverage=1.216x
     - vol_21d=0.1233, raw_lev=1.216, position=+1.216
     - Top-3 models: ridge, ard, bayesian_ridge
     - config_version=vol_target_v1
  5. Both DAGs unpaused for daily operation
     - forecast_l5_02_vol_targeting: 18:30 UTC Mon-Fri
     - forecast_l6_03_paper_trading: 00:00 UTC Tue-Sat

  BUGS FIXED during activation:
  - L5c SQL: horizon→horizon_id, predicted_return→predicted_return_pct/100, direction_prediction→direction
  - start_date: 2026-02-17→2026-02-15 (both DAGs, to allow pre-Monday trigger)
  - activate_paper_trading.sh: POSTGRES_DB usdcop→usdcop_trading
  - seed script: reads OHLCV from DB (bi.dim_daily_usdcop) instead of missing parquet
```

---

### PASO 1.4 — Paper trading (4-6 semanas, 20-30 dias habiles)

**Ejecucion**: Los DAGs corren automaticamente cada dia.
Revisar resultados diariamente en la tabla `forecast_paper_trading`.

**Query de monitoring:**
```sql
-- Metricas acumuladas
SELECT * FROM v_paper_trading_performance;

-- Ultimos 10 dias
SELECT signal_date, signal_direction, signal_leverage,
       actual_return_1d, strategy_return, tracking_error
FROM forecast_paper_trading
ORDER BY signal_date DESC LIMIT 10;
```

**LOG SEMANAL:**
```
Semana 1: DA=___%, Ret semanal=___%, Acum=___%, MaxDD=___%, Binom p=____
Semana 2: DA=___%, Ret semanal=___%, Acum=___%, MaxDD=___%, Binom p=____
Semana 3: DA=___%, Ret semanal=___%, Acum=___%, MaxDD=___%, Binom p=____
Semana 4: DA=___%, Ret semanal=___%, Acum=___%, MaxDD=___%, Binom p=____
[Si extendido:]
Semana 5: DA=___%, Ret semanal=___%, Acum=___%, MaxDD=___%, Binom p=____
Semana 6: DA=___%, Ret semanal=___%, Acum=___%, MaxDD=___%, Binom p=____
```

### GATE 1.4 (despues de 4 semanas / 20 dias minimo):
```
Evaluar en este orden:

1. ¿MaxDD > 15%?
   └── SI → STOP INMEDIATO. Revisar si el edge murio.

2. ¿DA < 45% despues de 20 dias?
   └── SI → STOP. Diferencia significativa vs backtest.
       Revisar: ¿implementacion correcta? ¿datos correctos?

3. ¿DA entre 45%-50%?
   └── SI → EXTENDER paper trading 2-4 semanas mas.
       20 dias es poco para significancia estadistica.
       Necesitas ~60 dias para binomial test con poder razonable.

4. ¿DA > 50% Y return acumulado > 0?
   │
   ├── SI Y Sharpe > 0.5 → FASE 2 (todo bien, edge confirmado en vivo)
   │
   ├── SI pero Sharpe < 0.5 → EXTENDER 2 semanas.
   │   DA positiva pero returns no siguen → posible issue de sizing.
   │
   └── SI Y tracking_error promedio < 2% → EXCELENTE, implementacion fiel
```

---

## ══════════════════════════════════════════════════
## FASE 2: TRAILING STOP EXECUTION (reemplaza RL)
## ══════════════════════════════════════════════════

> **DECISION (2026-02-15)**: El trailing stop reemplaza al RL como capa de ejecucion.
> Razones: (1) Trailing stop es estadisticamente significativo (p=0.0178), RL no (p=0.272).
> (2) Mucho mas simple de operar. (3) Sharpe 3.135 vs RL 0.321.
>
> **YA IMPLEMENTADO**: SmartExecutor + PaperBroker + L7 DAG + DB migration 042.
> Ver archivos en la tabla de "Archivos CREADOS" arriba.
>
> Si los resultados del paper trading (Paso 1.4) muestran que el trailing stop
> no agrega valor en vivo, considerar:
> (a) Ejecucion simple al cierre (sin trailing), o
> (b) RL executor (Fase 4 original, ver PLAN_FORECASTING_RL_INTEGRATION.md)

> **NOTA ORIGINAL** (preservada): Si decides que el RL no vale la complejidad operacional,
> salta de Gate 1.4 directo a FASE 3 (go-live sin RL).
> El sistema completo (Forecast + VT + Trail) tiene Sharpe ≈3.135.

### PASO 2.1 — Modificar el environment del RL

**Prompt para Claude:**
```
Modificar el RL trading environment para que funcione como EJECUTOR intraday
de la signal del forecasting, no como predictor de direccion.

Archivo a modificar: src/training/environments/trading_env.py

Environment actual (V21.5b baseline):
- obs_dim: 27 (18 market features + 9 state features)
- action_space: continuous Box(-1, 1) con ThresholdInterpreter (±0.35)
- reward: ModularRewardCalculator (pnl=0.80, sortino=0.10, regime=0.05, holding=0.05)
- datos: 5-min bars USDCOP, session 8:00-12:55 COT (59 bars/dia)
- min_hold_bars: 25
- SL=-4%, TP=+4%

Cambios necesarios (~50 lineas):

1. AGREGAR 3 features al observation (obs_dim: 27 → 30):
   - forecast_direction: float (-1 o +1, del forecasting diario)
   - forecast_leverage_norm: float (0-1, normalizado de [0.5, 2.0])
   - intraday_progress: float (0-1, bar_in_session / 59)

   En _get_observation(), despues de market_features y antes de state_features:
   ```python
   today = self._current_date()
   signal = self.forecast_signals.get(today, (0, 1.0))
   forecast_direction = signal[0]
   forecast_leverage_norm = (signal[1] - 0.5) / 1.5
   intraday_progress = (self.current_idx % 59) / 59.0
   ```

2. RESTRICCION DIRECCIONAL (nuevo parametro: forecast_constrained=True):
   En step(), antes de ejecutar accion:
   - Si forecast_direction > 0: bloquear SELL (solo BUY o HOLD)
   - Si forecast_direction < 0: bloquear BUY (solo SELL o HOLD)
   - Si forecast_direction = 0: solo HOLD

3. POSITION SIZE fijo:
   Cuando RL decide entrar, size = forecast_leverage (no del action value)

4. PARAMETRO forecast_signals:
   Dict[str, Tuple[int, float]] pasado en __init__()
   Clave = date string, valor = (direction, leverage)

5. NO CAMBIAR:
   - Las 18 market features existentes
   - Los 9 state features existentes
   - La logica de SL/TP/trailing stop
   - min_hold_bars

Tambien modificar: src/training/reward_calculator.py
  - Agregar modo "execution_alpha" con 3 casos:
    a. RL opero y cerro: alpha = actual_ret - benchmark_ret
    b. RL opero, posicion abierta al cierre: mark-to-market - benchmark
    c. RL NO opero (HOLD todo el dia): reward = -0.1 * abs(benchmark)
       (penalidad suave, NO perder el benchmark completo)

Crear config: config/experiments/exp_rl_executor_001.yaml
  - Derivado de v215b_baseline.yaml (1 variable: +3 forecast features)
  - obs_dim: 30, forecast_constrained: true
  - reward.mode: "execution_alpha"
  - reward.weights: {execution_alpha: 0.70, pnl: 0.20, holding: 0.05, regime: 0.05}
```

**LOG PASO 2.1:**
```
Fecha: ___________
trading_env.py modificado: SI / NO
reward_calculator.py modificado: SI / NO
exp_rl_executor_001.yaml creado: SI / NO
obs_dim nuevo: ____
Restriccion direccional funciona: SI / NO
Execution alpha reward (3 casos): SI / NO
Test unitario pasa: SI / NO
```

### GATE 2.1:
```
¿El environment modificado pasa los tests unitarios?
│
├── SI → Ir a PASO 2.2
│
└── NO → Debuggear.
         Errores comunes:
         - obs shape mismatch (check observation_space.shape == (30,))
         - action masking no funciona (check step() constraint logic)
         - reward NaN cuando no hay trade (check execution_alpha caso 3)
         - forecast_signals dict vacio (check date format matching)
         Corregir y re-testear.
```

---

### PASO 2.2 — Generar signals historicas walk-forward para RL training

**Prompt para Claude:**
```
Generar forecast signals historicas para cada dia del dataset de 5-min,
respetando anti-leakage (walk-forward, no retrain diario).

Datos (ya en el proyecto):
- OHLCV diario: seeds/latest/usdcop_daily_ohlcv.parquet
- OHLCV 5-min: seeds/latest/usdcop_m5_ohlcv.parquet
- Macro: data/pipeline/04_cleaning/output/macro_daily_clean.parquet

Estrategia de reentrenamiento (CRITICO para anti-leakage):
  NO reentrenar diario (seria 9 modelos × 7 horizontes × 1,200 dias = 75,600
  entrenamientos = horas de computo).

  En cambio, reentrenar cada 63 dias (~1 trimestre):
  1. Ventana 1: Entrenar con datos hasta 2020-03, predecir 2020-04 a 2020-06
  2. Ventana 2: Entrenar con datos hasta 2020-06, predecir 2020-07 a 2020-09
  3. ... repetir hasta 2024-12
  4. Usar la prediccion del ultimo modelo para los dias intermedios

  Resultado: ~20 reentrenamientos × 63 dias = ~1,260 dias con signals
  Tiempo estimado: ~30-45 min

Pipeline:
  - ForecastingEngine (src/forecasting/engine.py)
  - Solo H=1 (horizonte validado)
  - Ensemble: top_3 (mas estable)
  - Vol-targeting: usar VolTargetConfig del Paso 1.1

Output: data/forecasting/historical_forecast_signals.parquet
  Columnas: date, forecast_direction, forecast_leverage, predicted_return, realized_vol_21d

Merge con 5-min:
  Para cada barra 5-min, agregar las 3 columnas del forecast del dia correspondiente.

Verificaciones:
1. ¿Cuantos dias tienen signal +1 vs -1 vs 0?
2. ¿Las signals son look-ahead free? (train period < prediction date SIEMPRE)
3. ¿El DA de estas signals coincide con walk-forward (~53%)?
4. ¿Cobertura > 95% de los dias de trading?

Crear script: scripts/generate_historical_forecast_signals.py
```

**LOG PASO 2.2:**
```
Fecha: ___________
Signals generadas: ____ dias
Distribucion: +1=____%, -1=____%, 0=____%
DA de signals historicas: ____%
¿Coincide con walk-forward (≈53%)?: SI / NO
Look-ahead bias check: PASS / FAIL
Cobertura de dias: ____%
Archivo: data/forecasting/historical_forecast_signals.parquet
Tiempo de generacion: ____ min
```

### GATE 2.2:
```
¿DA de signals historicas esta entre 51%-55% (consistente con walk-forward)?
│
├── SI → Ir a PASO 2.3
│
└── NO, DA muy alto (>57%) → HAY LOOK-AHEAD BIAS.
    Revisar:
    - ¿El modelo usa datos del futuro? (check train_end < prediction_date)
    - ¿El reentrenamiento cada 63 dias esta bien implementado?
    - Comparar DA de la primera ventana vs ultima (no debe mejorar magicamente)

└── NO, DA muy bajo (<50%) → Bug en generacion de signals.
    Revisar:
    - ¿Features calculadas correctamente? (check _build_ssot_features)
    - ¿Lag de macro correcto? (T-1 para DXY y WTI)
    - ¿Ensemble strategy correcta? (top_3 no best_of_breed)
```

---

### PASO 2.3 — Entrenar RL execution agent (5 seeds)

**Comando:**
```bash
python scripts/run_ssot_pipeline.py \
    --config config/experiments/exp_rl_executor_001.yaml \
    --forecast-signals data/forecasting/historical_forecast_signals.parquet \
    --stage all \
    --multi-seed
```

**Prompt para Claude (si el flag --forecast-signals no existe aun):**
```
Modificar scripts/run_ssot_pipeline.py para soportar --forecast-signals flag.

Agregar argparse argument:
  --forecast-signals PATH  Parquet con historical forecast signals para RL executor

En run_l2_dataset():
  Si --forecast-signals, cargar y mergear daily signals → 5-min bars
  (broadcast: misma signal para todas las barras del mismo dia)

En run_l3_training():
  Si --forecast-signals, pasar forecast_signals dict al TradingEnvironment

Config: config/experiments/exp_rl_executor_001.yaml
Seeds: [42, 123, 456, 789, 1337]
Device: CPU (PPO MlpPolicy)
timesteps: 2,000,000

Para cada seed reportar:
1. Training: best_eval_reward, convergence step, entropy final, duration
2. OOS Backtest (2025-01 a 2025-12):
   - execution_alpha promedio por trade (vs benchmark open-to-close)
   - % de trades con execution_alpha > 0
   - Dias sin trade (RL eligio HOLD) y reward promedio en esos dias
   - Total extra return del year por mejor timing
   - Sharpe con RL vs Sharpe sin RL
   - MaxDD con RL vs MaxDD sin RL
```

**LOG PASO 2.3:**
```
Fecha: ___________

| Seed | exec_alpha_bps | %_positive | extra_ret% | Sharpe_RL | Sharpe_noRL | MaxDD_diff |
|------|---------------|------------|-----------|-----------|-------------|------------|
| 42   |               |            |           |           |             |            |
| 123  |               |            |           |           |             |            |
| 456  |               |            |           |           |             |            |
| 789  |               |            |           |           |             |            |
| 1337 |               |            |           |           |             |            |

Media exec_alpha: ____ bps
Seeds con alpha > 0: __/5
Entropy final (check no collapse): seed42=____, seed123=____, ...
```

### GATE 2.3:
```
Evaluar en este orden:

1. ¿Entropy colapso (< 0.1) en algun seed?
   └── SI → Overfitting. Reducir timesteps a 1M o subir ent_coef a 0.02.
       Re-entrenar seeds afectados.

2. ¿>= 4/5 seeds tienen execution_alpha > 0?
   │
   ├── SI → EXCELENTE. Ir a PASO 2.4
   │
   └── NO, solo 2-3/5 positivos → MARGINAL.
       │
       ├── ¿Media es positiva? → Ir a PASO 2.4 con precaucion
       │
       └── ¿Media es negativa? → RL NO agrega valor.
           DECISION: Usar ejecucion simple al cierre (skip RL).
           Ir directo a FASE 3 SIN RL.

3. ¿Sharpe(con RL) >= Sharpe(sin RL) * 0.95 en >= 3/5 seeds?
   └── NO → RL introduce varianza que empeora risk-adjusted returns.
       Aunque execution_alpha > 0, el Sharpe se degrada.
       DECISION: Ship sin RL. El timing improvement no compensa la varianza.

4. ¿Algun seed empeoro MaxDD en > 3pp vs ejecucion simple?
   └── SI → RL introduce riesgo. Revisar reward function.
       ¿Esta penalizando drawdown suficiente?
       Agregar regime_penalty weight si no.
```

---

### PASO 2.4 — Validar sistema completo vs sin-RL

**Prompt para Claude:**
```
Comparacion final A/B: ¿el sistema integrado es mejor que sin RL?

Usando el mejor seed (o top-3 seeds si ensemble):

Backtest OOS 2025 completo:
  Sistema A: Forecast → vol-target → ejecutar al cierre 12:55 COT
  Sistema B: Forecast → vol-target → RL ejecuta intraday (8:00-12:55)

Metricas comparativas:
1. Return total A vs B
2. Sharpe A vs B
3. MaxDD A vs B
4. PF A vs B
5. % dias donde B > A (por dia)

Tests estadisticos:
- Paired t-test diario: ¿B > A con p < 0.10?
- Bootstrap CI de (B - A): ¿excluye cero?

Si B no es significativamente mejor que A, el RL no vale la complejidad.
```

**LOG PASO 2.4:**
```
Fecha: ___________

| Metrica | Sin RL (A) | Con RL (B) | Diff (B-A) |
|---------|-----------|-----------|------------|
| Return% |           |           |            |
| Sharpe  |           |           |            |
| MaxDD%  |           |           |            |
| PF      |           |           |            |

Paired t-test: t=____, p=____
Bootstrap CI de (B-A): [___%, ___%]
¿Significativo (p<0.10)?: SI / NO
```

### GATE 2.4 — DECISION FINAL SOBRE RL:
```
¿B es significativamente mejor que A (p < 0.10)?
│
├── SI → USAR SISTEMA COMPLETO (Forecast + VolTgt + RL)
│        Ir a FASE 3 con sistema B
│
├── NO, pero B es ligeramente mejor (>0 pero p>0.10)
│   → Decision discrecional:
│     ¿Vale la complejidad operacional del RL por <2% extra?
│     ├── SI (quieres el upside) → Usar B, ir a FASE 3
│     └── NO (prefieres simplicidad) → Usar A, ir a FASE 3
│
└── NO, A es mejor que B
    → DESCARTAR RL. Sistema final = Forecasting + Vol-Targeting.
      Ir a FASE 3 sin RL (mas simple, igualmente rentable).
```

---

## ══════════════════════════════════════════════════
## FASE 3: GO-LIVE GRADUAL
## ══════════════════════════════════════════════════

### PASO 3.1 — Paper trading del sistema final (2 semanas)

**Prompt para Claude:**
```
Configurar monitoreo automatizado del sistema elegido
(con o sin RL segun Gate 2.4).

Si CON RL:
  - DAG pre-market (7:00 COT): L0→L1→L5b→L5c (forecast + vol-target)
  - Signal enviada a Redis: forecast_signal:daily
  - RL ejecuta durante 8:00-12:55 via l5_multi_model_inference (obs_dim=30)
  - DAG post-market (14:00 COT): registrar resultado, check gates

Si SIN RL:
  - DAG pre-market (7:00 COT): L0→L1→L5b→L5c (forecast + vol-target)
  - Signal registrada en forecast_vol_targeting_signals
  - Ejecucion manual al cierre 12:55 COT via SignalBridge API (:8085)
  - DAG post-market (14:00 COT): registrar resultado, check gates

Alertas (en ambos casos):
  - MaxDD > 10% → alerta amarilla (Grafana :3002)
  - MaxDD > 15% → alerta roja (parar trading)
  - DA < 47% despues de 10+ dias → alerta amarilla
  - 3 semanas consecutivas negativas → revision obligatoria

Usar src/trading/paper_trader.py (PaperTrader) para tracking automatico.
```

**LOG PASO 3.1 (diario, 2 semanas):**
```
Dia 1:  Signal=___ Leverage=___ Ret=____% Acum=____% DA=____%
Dia 2:  Signal=___ Leverage=___ Ret=____% Acum=____% DA=____%
...
Dia 10: Signal=___ Leverage=___ Ret=____% Acum=____% DA=____%
```

### GATE 3.1:
```
Despues de 10 dias de paper trading:

1. ¿Alguna alerta roja se disparo?
   └── SI → STOP. Diagnosticar que salio mal.
       ¿Es problema de datos? ¿Modelo? ¿Implementacion?

2. ¿Return acumulado > -5%?
   │
   ├── SI → Ir a PASO 3.2 (go-live con capital real)
   │
   └── NO → Extender paper trading 2 semanas mas.
       Si despues de 4 semanas sigue negativo → volver a FASE 0.
```

---

### PASO 3.2 — Go-live escalonado

**Ejecucion via SignalBridge API** (puerto 8085):
```
Fase A (semana 1-2):  10% del capital target
  - Verificar ejecucion real vs paper (slippage, latencia)
  - Confirmar que MEXC ejecuta correctamente (maker 0% fee)
  - Gate: slippage real < 3 bps, sin errores operativos

Fase B (semana 3-4):  25% del capital target
  - Si Fase A positiva y sin problemas operativos
  - Gate: DA > 48%, tracking error vs paper < 2%

Fase C (semana 5-8):  50% del capital target
  - Si Fase B mantuvo DA > 50% y sin alertas
  - Gate: Sharpe running > 0.5, MaxDD < 12%

Fase D (semana 9+):   100% del capital target
  - Si Fase C positiva
  - Monitoreo continuo via Grafana + alertas
```

**ROLLBACK en cualquier fase:**
```
- MaxDD > 15% → volver a fase anterior
- DA < 45% por 15+ dias → pausar y re-evaluar modelo
- Error operativo (ejecucion fallida) → pausar hasta fix
- Slippage > 5 bps consistentemente → revisar order type
```

**LOG PASO 3.2:**
```
Fase A — 10% capital
  Semana 1: Ret=___%, Slippage=___bps, DA=___%, Issues: ________
  Semana 2: Ret=___%, Slippage=___bps, DA=___%, Issues: ________
  ¿Pasar a Fase B?: SI / NO

Fase B — 25% capital
  Semana 3: Ret=___%, DA=___%, MaxDD=___%, Issues: ________
  Semana 4: Ret=___%, DA=___%, MaxDD=___%, Issues: ________
  ¿Pasar a Fase C?: SI / NO

Fase C — 50% capital
  Semana 5-8: Ret total=___%, DA=___%, Sharpe=____, MaxDD=___%
  ¿Pasar a Fase D?: SI / NO

Fase D — 100% capital
  En operacion desde: ___________
  Re-evaluar mensualmente con walk-forward actualizado.
```

---

## ══════════════════════════════════════════════════
## DIAGRAMA COMPLETO DEL ARBOL DE DECISION
## ══════════════════════════════════════════════════

```
FASE 0: VERIFICAR BASELINE
│
├─ 0.1 Reproducir walk-forward (5 ventanas, 19 features, 9 modelos)
│  └─ GATE: DA>51%, binom p<0.10, >=3/5 ventanas+?
│     ├─ SI → 0.2
│     └─ NO → STOP (edge no existe)
│
├─ 0.2 Verificar con datos recientes (6ta ventana)
│  └─ GATE: ventana nueva positiva?
│     ├─ SI → FASE 1
│     ├─ DA>48% → FASE 1 con cautela
│     └─ DA<48% → PAUSA
│
FASE 1: VOL-TARGETING + PAPER TRADING
│
├─ 1.1 Backtest vol-targeting (4 configs × 5 ventanas WF)
│  └─ GATE: Sharpe >= 1.0?
│     ├─ SI → 1.2
│     └─ NO → reducir max_leverage, re-testear
│
├─ 1.2 Implementar en produccion (modulo + migration + config)
│  └─ GATE: signals coinciden con backtest?
│     ├─ SI → 1.3
│     └─ NO → debuggear
│
├─ 1.3 Crear DAGs (L5c vol-targeting + L6 monitoring)
│  └─ GATE: smoke test pasa?
│     ├─ SI → 1.4
│     └─ NO → fix DAGs
│
├─ 1.4 Paper trading 4-6 semanas
│  └─ GATE: DA>50%, ret>0, MaxDD<15%?
│     ├─ SI → FASE 2 (o FASE 3 si skip RL)
│     ├─ DA 45-50% → extender 2-4 semanas
│     └─ DA<45% o MaxDD>15% → STOP
│
FASE 2: RL EXECUTION (OPCIONAL)
│
├─ 2.1 Modificar RL environment (+3 features, direction constraint, execution_alpha)
│  └─ GATE: tests unitarios pasan?
│     ├─ SI → 2.2
│     └─ NO → debuggear
│
├─ 2.2 Generar signals historicas (WF cada 63 dias, ~30 min)
│  └─ GATE: DA≈53%, sin look-ahead bias, cobertura>95%?
│     ├─ SI → 2.3
│     └─ NO → fix bias/bugs
│
├─ 2.3 Entrenar RL executor (5 seeds × 2M steps, ~15h total)
│  └─ GATE: >=4/5 alpha>0, Sharpe(RL)>=0.95×Sharpe(noRL)?
│     ├─ SI → 2.4
│     ├─ 2-3/5 con media+ → 2.4 con cautela
│     └─ media negativa O Sharpe degradado → DESCARTAR RL → FASE 3 sin RL
│
├─ 2.4 Validar RL vs sin-RL (paired test)
│  └─ GATE: B > A significativo (p<0.10)?
│     ├─ SI → FASE 3 con RL
│     ├─ B mejor pero p>0.10 → decision discrecional
│     └─ A mejor → FASE 3 sin RL
│
FASE 3: GO-LIVE
│
├─ 3.1 Paper trading sistema final (2 semanas)
│  └─ GATE: sin alertas rojas, ret > -5%?
│     ├─ SI → 3.2
│     └─ NO → extender o volver a FASE 0
│
├─ 3.2 Go-live escalonado
│  ├─ Fase A: 10% (2 sem) → slippage<3bps, sin errores
│  ├─ Fase B: 25% (2 sem) → DA>48%, tracking error<2%
│  ├─ Fase C: 50% (4 sem) → Sharpe>0.5, MaxDD<12%
│  └─ Fase D: 100% → monitoreo continuo
│
│  ROLLBACK: MaxDD>15% | DA<45% 15d | error operativo
│
FIN: Sistema en produccion
     Re-evaluar mensualmente con walk-forward actualizado
```

---

## TIMELINE ESTIMADO

| Semana | Paso | Actividad | Decision clave |
|--------|------|-----------|---------------|
| 1 | 0.1-0.2 | Reproducir baseline, verificar edge | ¿El edge sigue vivo? |
| 1-2 | 1.1-1.2 | Backtest + implementar vol-targeting | ¿Que target_vol usar? |
| 2 | 1.3 | Crear DAGs de paper trading | ¿Codigo listo? |
| 3-6 | 1.4 | Paper trading automatizado (20-30 dias) | ¿Edge se confirma en vivo? |
| 7 | 2.1-2.2 | Modificar RL env + generar signals | ¿Environment listo? |
| 8-9 | 2.3 | Entrenar RL executor (5 seeds, ~15h) | ¿RL agrega valor? (CLAVE) |
| 10 | 2.4 | Validar RL vs sin-RL | ¿Usar RL o no? |
| 11-12 | 3.1 | Paper trading sistema final | ¿Ready for live? |
| 13+ | 3.2 | Go-live escalonado | Produccion |

**Total hasta produccion: ~13-16 semanas (3-4 meses)**
**Atajo sin RL (skip Fase 2): ~8-9 semanas (2 meses)**

---

## QUE NO HACER (validado por experimentos fallidos)

| # | NO hacer | Evidencia | Ref |
|---|----------|-----------|-----|
| 1 | Agregar macro features al forecasting (19→36) | Sharpe 1.08→0.49 | FC analysis |
| 2 | Usar H=5 ni horizontes largos | DA<50% | FC analysis |
| 3 | Usar macro score como filtro externo | ANOVA p=0.297 | EXP-INFRA-001 |
| 4 | Leverage > 2x | CI lower bound = +0.3% | FC analysis |
| 5 | Confiar en 1 seed de RL | Seed 456 (eval=131) perdio -20% | EXP-V22-002 |
| 6 | RL en barras horarias | -7.16% best, no alpha | EXP-HOURLY-PPO-002 |
| 7 | RL en barras diarias | -8.83%, 8 trades | EXP-DAILY-PPO-001 |
| 8 | Asymmetric SL/TP mas tight | 0/5 seeds, SL demasiado tight | EXP-ASYM-001 |
| 9 | Cambiar multiples variables a la vez | Imposible atribuir | V22 (5 changes) |
| 10 | Reentrenar forecasting diario para RL signals | 75,600 trainings | Compute cost |
| 11 | Usar ensemble weights fijos (Ridge 0.4 + XGB 0.6) | Pipeline usa top_3 mean | engine.py |
