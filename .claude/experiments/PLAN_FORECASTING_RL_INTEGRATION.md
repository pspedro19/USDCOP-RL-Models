# PLAN DE INTEGRACION FORECASTING + RL
## v2.0 — Basado en evidencia estadistica + arquitectura real del codebase
## Fecha: 2026-02-15

---

## 1. QUE DICE LA EVIDENCIA

### Lo que funciona (p < 0.05)

Pipeline de forecasting actual con **19 features** (4 price + 4 returns + 3 vol + 3 technical + 3 calendar + 2 macro):

| Metrica | Valor | Soporte estadistico |
|---------|-------|---------------------|
| DA combinado (4 ventanas) | 53.0% | Binomial p=0.045 |
| Return anualizado | +13.4% | Bootstrap CI [+0.3%, +26.3%] |
| Sharpe | 1.08 | Lo-MacKinlay p=0.025 |
| PF | 1.19 | — |
| Pesaran-Timmermann | z=1.795 | p=0.036 |
| Random agent percentil | 97.3th | 10,000 simulaciones |
| Ventanas positivas | 4/4 | — |

### Lo que NO funciona

| Idea | Resultado | Por que |
|------|-----------|---------|
| Agregar 17 macro como inputs (19→36) | Sharpe 1.08 → 0.49, CI incluye 0 | Overfitting: mas features, ~1,300 rows |
| Agregar macro + z-scores (25f) | DA baja a 50.4%, 3/4 ventanas | Ruido > signal |
| Macro score como filtro | Solo funciona en 2025 (2/6 years +) | Data snooping (ANOVA p=0.297) |
| H=5 (semanal) | DA=44% (peor que coin flip) | No hay signal semanal |
| RL on hourly bars | -7.16% best case | 5 bars/day insuficiente |
| RL on daily bars | -8.83%, 8 trades | 1,126 training bars insuficiente |

### Hallazgo clave sobre el RL

El gap intraday (high-low) es 0.92%/dia promedio, pero el movimiento open-close es solo 0.56%.
Esa diferencia de 0.36%/dia es el espacio teorico donde el RL puede agregar valor optimizando
timing de entrada y salida. Si captura 5% de ese gap = +4.5%/year. Si captura 10% = +9%/year.

---

## 2. ARQUITECTURA ACTUAL DEL CODEBASE (referencias exactas)

### Forecasting Pipeline

```
src/forecasting/
├── engine.py                    ← ForecastingEngine: train() + predict()
│   ├── _load_from_ssot()        ← Carga OHLCV diario + macro (DB → parquet fallback)
│   ├── _build_ssot_features()   ← 19 features desde raw OHLCV + macro
│   ├── _create_targets()        ← log(close[t+H]/close[t]) para 7 horizontes
│   └── _create_ensembles()      ← 4 estrategias de ensemble
├── config.py                    ← ForecastingConfig dataclass (SSOT)
│   └── FeatureConfig.feature_columns = 19 features (tuple)
├── contracts.py                 ← HORIZONS=(1,5,10,15,20,25,30), MODEL_DEFINITIONS (9)
├── data_contracts.py            ← FEATURE_COLUMNS (19), TARGET_HORIZONS (7)
│   └── Tables: bi.dim_daily_usdcop, bi.fact_forecasts, bi.fact_model_metrics
├── models/                      ← 9 modelos
│   ├── ridge.py, bayesian_ridge.py, ard.py           ← 3 lineales
│   ├── xgboost.py, lightgbm.py, catboost.py          ← 3 boosting
│   ├── hybrids.py               ← 3 hibridos (alpha=0.3 lineal + 0.7 boosting)
│   └── factory.py               ← ModelFactory.create(model_id, params, horizon)
└── evaluation/
    ├── walk_forward.py          ← WalkForwardValidator(n_folds=5, initial_train_ratio=0.6)
    ├── backtest.py              ← BacktestEngine (train/test split)
    └── metrics.py               ← DA, RMSE, MAE, Sharpe, MaxDD
```

### RL Pipeline

```
src/training/
├── environments/
│   ├── trading_env.py           ← TradingEnvironment (obs_dim=27 en V21.5b)
│   │   ├── 18 market features (Z-score normalized)
│   │   ├── 9 state features (position, unrealized_pnl, sl/tp_proximity, bars_held, temporal)
│   │   ├── Action: Box(-1,1) continuous o Discrete(4)
│   │   └── min_hold_bars=25, SL=-4%, TP=+4%
│   ├── action_interpreters.py   ← ThresholdInterpreter (continuous), ZoneInterpreter
│   └── stop_strategies.py       ← FixedPctStopStrategy, ATRDynamicStopStrategy, TrailingStop
├── reward_calculator.py         ← ModularRewardCalculator v2.0
│   └── Weights: pnl=0.80, sortino=0.10, regime=0.05, holding=0.05
└── config.py                    ← TradingEnvConfig (30+ fields)
```

### Infraestructura Disponible (docker-compose.yml)

| Servicio | Puerto | Uso en integracion |
|----------|--------|-------------------|
| PostgreSQL (TimescaleDB) | 5432 | Almacenar forecast_signals, vol_targeting |
| Redis | 6379 | Cache + Streams (signals WebSocket) |
| MinIO | 9000 | Almacenar modelos (.pkl, .zip) |
| Airflow | 8080 | Orquestar DAGs de forecasting + RL |
| MLflow | 5001 | Tracking de experimentos |
| Prometheus + Grafana | 9090/3002 | Monitoreo de drift, latencia |
| SignalBridge API | 8085 | Ejecucion en MEXC |

### Base de Datos Actual

```sql
-- RL (ya existe)
inference_ready_nrt     → FLOAT[18] market features + price + hashes
inference_signals_nrt   → model_id, signal, confidence, raw_action
backtest_trades         → entry/exit times, prices, PnL

-- Forecasting (ya existe)
bi.dim_daily_usdcop     → OHLCV diario (Investing.com official)
bi.fact_forecasts       → model_id, horizon, predicted_return, direction, signal
bi.fact_consensus       → ensemble predictions por horizonte
bi.fact_model_metrics   → DA, RMSE, Sharpe por modelo/horizonte
```

---

## 3. FEATURE SET EXACTO DEL FORECASTING (SSOT)

**Fuente**: `src/forecasting/data_contracts.py` → `FEATURE_COLUMNS`

```python
FEATURE_COLUMNS = (
    # Price (4)
    "close", "open", "high", "low",

    # Returns/Momentum (4)
    "return_1d", "return_5d", "return_10d", "return_20d",

    # Volatility (3)
    "volatility_5d", "volatility_10d", "volatility_20d",

    # Technical (3)
    "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",

    # Calendar (3)
    "day_of_week", "month", "is_month_end",

    # Macro/Exogenous (2, lagged T-1)
    "dxy_close_lag1", "oil_close_lag1",
)
# Total: 19 features
```

**Horizons**: `TARGET_HORIZONS = (1, 5, 10, 15, 20, 25, 30)`

**Horizons validados con alpha**: Solo H=1 tiene DA>50% consistente.
H=5 y superiores no superan coin flip.

---

## 4. ENSEMBLE ACTUAL (como funciona)

**Fuente**: `src/forecasting/engine.py` → `_create_ensembles()` (lines 470-541)

4 estrategias de ensemble:

| Estrategia | Metodo | Cuando usar |
|------------|--------|-------------|
| **best_of_breed** | Mejor modelo individual por horizonte (max magnitude) | Default para H=1 |
| **top_3** | Promedio de top-3 modelos (ranked by avg DA) | Mas estable |
| **top_6_mean** | Promedio de top-6 modelos | Mas conservador |
| **consensus** | Promedio de todos los 9 modelos | Mas robusto pero diluye signal |

**Output**: `ForecastPrediction` con:
```python
predicted_return_pct: float   # log-return en %
direction: UP | DOWN          # sign(predicted_return)
signal: int                   # -1=SELL, 0=HOLD, 1=BUY (threshold: |ret| > 0.001)
confidence: float             # |predicted_return| / std (opcional)
```

---

## 5. ARQUITECTURA DE INTEGRACION

```
                    PIPELINE DIARIO (pre-market, ~7:00 AM COT)
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

     ┌──────────────────────────────────────────────────────────────────┐
     │  L0-L1: Data + Features (pipeline existente, SIN CAMBIOS)       │
     │                                                                  │
     │  Fuente: bi.dim_daily_usdcop + macro_indicators_daily            │
     │  Builder: ForecastingEngine._build_ssot_features()               │
     │  Features: 19 (FEATURE_COLUMNS en data_contracts.py)             │
     │                                                                  │
     │  NO CAMBIAR — es la config estadisticamente validada.            │
     │  Mas features = peor performance (probado: 19→36 mata Sharpe).  │
     └──────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  L3b-L5b: Forecast Inference (pipeline existente, SIN CAMBIOS)  │
     │                                                                  │
     │  Modelos: 9 (3 linear + 3 boosting + 3 hybrid)                  │
     │  Horizonte: H=1 (unico validado con DA>50%)                     │
     │  Ensemble: top_3 o best_of_breed (configurable)                  │
     │                                                                  │
     │  Output: predicted_return (float), direction (±1), signal (int)  │
     │  Destino: bi.fact_forecasts + bi.fact_consensus                  │
     └──────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  L5c: Vol-Targeting (NUEVO — mecanico, sin ML)                  │
     │                                                                  │
     │  Input: direction (de L5b) + realized_vol_21d (de OHLCV)        │
     │  Logic: leverage = target_vol / realized_vol_21d                 │
     │  Clip: [0.5, 2.0]                                               │
     │  Output: position_size = direction × leverage                    │
     │                                                                  │
     │  Nuevo archivo: src/forecasting/vol_targeting.py                 │
     │  Nuevo script: scripts/vol_target_backtest.py                    │
     │  Nueva tabla: forecast_vol_targeting_signals                     │
     └──────────────────────┬───────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     ┌────────────────┐          ┌────────────────────┐
     │  Ejecucion     │          │  RL Execution       │
     │  Simple        │          │  Agent              │
     │  (Fase 1-2)    │          │  (Fase 3)           │
     │                │          │                      │
     │  Operar al     │          │  Recibe:             │
     │  cierre del    │          │  - direction (±1)    │
     │  dia a las     │          │  - position_size     │
     │  12:55 COT     │          │  Decide:             │
     │                │          │  - cuando entrar     │
     │  Sin cambios   │          │  - cuando salir      │
     │  al RL env     │          │  - en barras de 5min │
     │                │          │                      │
     │  Ruta:         │          │  Ruta:               │
     │  SignalBridge  │          │  Modified            │
     │  API :8085     │          │  TradingEnvironment  │
     └────────────────┘          └────────────────────┘
```

---

## 6. IMPLEMENTACION POR FASES

### FASE 1: Vol-Targeting sobre modelo actual (Semana 1-2)

**Objetivo**: Pasar de ~13% a ~18-20% sin tocar el modelo de forecasting.
Cambio PURO de position sizing. No requiere reentrenamiento.

#### Paso 1.1 — Backtest de vol-targeting historico

**Nuevo archivo**: `scripts/vol_target_backtest.py`

```python
"""
Backtest de vol-targeting sobre predicciones historicas del forecasting pipeline.

Input: Walk-forward predictions de baseline_v1 (4 ventanas, ~5 years de datos)
Output: Equity curve con vol-targeting vs fixed sizing

Validacion: Walk-forward (mismas 4 ventanas), no in-sample.
"""
import numpy as np
import pandas as pd
from scipy import stats
from src.forecasting.engine import ForecastingEngine
from src.forecasting.evaluation.walk_forward import WalkForwardValidator
from src.forecasting.data_contracts import FEATURE_COLUMNS, TARGET_HORIZONS

def compute_position_size(
    forecast_direction: int,      # +1 o -1 (de ensemble)
    realized_vol_21d: float,      # volatilidad realizada anualizada
    target_vol: float = 0.15,     # parametro a optimizar
    max_leverage: float = 2.0,    # limite de seguridad
    min_leverage: float = 0.5,    # piso (siempre algo de exposicion)
) -> float:
    """
    Vol-targeting: escala posicion inversamente a volatilidad.

    Evidencia esperada (por confirmar en backtest):
    - target_vol=0.15 → ~18%/yr, 12% MaxDD, 1.2x avg leverage
    - target_vol=0.18 → ~21%/yr, 14.5% MaxDD, 1.5x avg leverage
    - target_vol=0.20 → ~23%/yr, 16% MaxDD, 1.6x avg leverage
    - Sharpe se mantiene ~1.10 en todos los niveles (leverage lineal)
    """
    if realized_vol_21d < 0.05:
        realized_vol_21d = 0.05  # floor para evitar leverage infinito

    leverage = target_vol / realized_vol_21d
    leverage = np.clip(leverage, min_leverage, max_leverage)
    return forecast_direction * leverage


def run_vol_target_backtest(
    predictions_df: pd.DataFrame,   # Columns: date, predicted_return, actual_return
    target_vols: list = [0.12, 0.15, 0.18, 0.20],
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """
    Ejecuta backtest de vol-targeting sobre predicciones walk-forward.

    Para cada target_vol:
    1. Calcula realized_vol_21d = return_1d.rolling(21).std() * sqrt(252)
    2. Calcula leverage = target_vol / realized_vol_21d
    3. strategy_return = direction * leverage * actual_return_1d
    4. Calcula metricas: Sharpe, MaxDD, PF, avg_leverage

    Returns: DataFrame con metricas por target_vol
    """
    results = []

    for tv in target_vols:
        df = predictions_df.copy()

        # Realized vol (21-day rolling, annualized)
        df['vol_21d'] = df['actual_return'].rolling(21).std() * np.sqrt(252)
        df = df.dropna()

        # Direction from forecast
        df['direction'] = np.sign(df['predicted_return'])

        # Leverage
        df['leverage'] = df.apply(
            lambda r: compute_position_size(
                r['direction'], r['vol_21d'], tv, max_leverage
            ), axis=1
        )

        # Strategy return (daily)
        df['strategy_ret'] = df['direction'] * abs(df['leverage']) * df['actual_return']

        # Metrics
        cum_ret = (1 + df['strategy_ret']).cumprod()
        total_ret = cum_ret.iloc[-1] - 1
        sharpe = df['strategy_ret'].mean() / df['strategy_ret'].std() * np.sqrt(252)
        max_dd = ((cum_ret / cum_ret.cummax()) - 1).min()
        avg_lev = abs(df['leverage']).mean()

        # Win rate (direction accuracy, NOT trade WR)
        correct = (df['direction'] * df['actual_return'] > 0).mean()

        # Profit factor
        wins = df.loc[df['strategy_ret'] > 0, 'strategy_ret'].sum()
        losses = abs(df.loc[df['strategy_ret'] < 0, 'strategy_ret'].sum())
        pf = wins / losses if losses > 0 else float('inf')

        results.append({
            'target_vol': tv,
            'total_return_pct': total_ret * 100,
            'annualized_return_pct': ((1 + total_ret) ** (252 / len(df)) - 1) * 100,
            'sharpe': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'profit_factor': pf,
            'avg_leverage': avg_lev,
            'max_leverage_used': abs(df['leverage']).max(),
            'direction_accuracy': correct * 100,
            'n_days': len(df),
        })

    return pd.DataFrame(results)


def bootstrap_significance(strategy_returns, n_bootstrap=10000):
    """Bootstrap CI del return medio diario."""
    n = len(strategy_returns)
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(strategy_returns, size=n, replace=True)
        means.append(sample.mean())

    ci_lower = np.percentile(means, 2.5) * 252  # Annualized
    ci_upper = np.percentile(means, 97.5) * 252

    return ci_lower, ci_upper
```

**Ejecucion**:
```bash
# Paso 1: Generar predicciones walk-forward del baseline
python scripts/run_forecast_experiment.py --config baseline_v1.yaml --output-predictions

# Paso 2: Correr backtest de vol-targeting
python scripts/vol_target_backtest.py \
    --predictions data/forecasting/baseline_v1_wf_predictions.parquet \
    --target-vols 0.12,0.15,0.18,0.20 \
    --max-leverage 2.0
```

**Gate Fase 1.1**:
- Sharpe con vol-targeting > 1.0 (debe mantenerse similar al baseline)
- Bootstrap CI de return anualizado excluye zero
- MaxDD < 20% para target_vol elegido
- avg_leverage < 1.8x (seguridad)

#### Paso 1.2 — Nuevo modulo: `src/forecasting/vol_targeting.py`

```python
"""
Modulo de vol-targeting para produccion.

Usado por:
- scripts/vol_target_backtest.py (validacion offline)
- Fase 2: DAG de paper trading
- Fase 4: DAG integrado con RL
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True)
class VolTargetConfig:
    """Configuracion de vol-targeting (inmutable)."""
    target_vol: float = 0.15        # Annualized target volatility
    max_leverage: float = 2.0       # Hard ceiling
    min_leverage: float = 0.5       # Hard floor
    vol_lookback: int = 21          # Days for realized vol
    vol_floor: float = 0.05         # Min vol to prevent extreme leverage
    annualization_factor: float = 252.0  # sqrt(252) for daily

@dataclass
class VolTargetSignal:
    """Output del vol-targeting."""
    date: str
    forecast_direction: int         # +1 or -1
    forecast_return: float          # Predicted log-return
    realized_vol_21d: float         # Annualized realized vol
    raw_leverage: float             # Before clipping
    clipped_leverage: float         # After clipping
    position_size: float            # direction * clipped_leverage

    @property
    def is_levered(self) -> bool:
        return abs(self.clipped_leverage) > 1.0


def compute_vol_target_signal(
    forecast_direction: int,
    forecast_return: float,
    realized_vol_21d: float,
    config: VolTargetConfig = VolTargetConfig(),
) -> VolTargetSignal:
    """
    Computa signal de vol-targeting.

    Args:
        forecast_direction: +1 (long) or -1 (short) from ensemble
        forecast_return: predicted log-return from ensemble
        realized_vol_21d: annualized 21-day realized volatility
        config: vol-targeting parameters

    Returns:
        VolTargetSignal with leverage and position size
    """
    safe_vol = max(realized_vol_21d, config.vol_floor)
    raw_leverage = config.target_vol / safe_vol
    clipped_leverage = np.clip(raw_leverage, config.min_leverage, config.max_leverage)
    position_size = forecast_direction * clipped_leverage

    return VolTargetSignal(
        date="",  # Set by caller
        forecast_direction=forecast_direction,
        forecast_return=forecast_return,
        realized_vol_21d=realized_vol_21d,
        raw_leverage=raw_leverage,
        clipped_leverage=clipped_leverage,
        position_size=position_size,
    )
```

#### Paso 1.3 — Migracion de base de datos

**Nuevo archivo**: `database/migrations/041_forecast_vol_targeting.sql`

```sql
-- Tabla para almacenar signals de vol-targeting en produccion
CREATE TABLE IF NOT EXISTS forecast_vol_targeting_signals (
    id              BIGSERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    forecast_direction  SMALLINT NOT NULL,    -- +1 or -1
    forecast_return     DOUBLE PRECISION,     -- Predicted log-return
    ensemble_strategy   VARCHAR(30),          -- 'best_of_breed', 'top_3', etc.
    realized_vol_21d    DOUBLE PRECISION,     -- Annualized 21d vol
    raw_leverage        DOUBLE PRECISION,     -- Before clipping
    clipped_leverage    DOUBLE PRECISION,     -- After clipping
    position_size       DOUBLE PRECISION,     -- direction * leverage
    target_vol          DOUBLE PRECISION DEFAULT 0.15,
    max_leverage        DOUBLE PRECISION DEFAULT 2.0,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(signal_date)  -- One signal per day
);

-- Tabla para tracking de paper trading results
CREATE TABLE IF NOT EXISTS forecast_paper_trading (
    id              BIGSERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL REFERENCES forecast_vol_targeting_signals(signal_date),
    signal_direction    SMALLINT NOT NULL,    -- +1 or -1
    signal_leverage     DOUBLE PRECISION,
    execution_price     DOUBLE PRECISION,     -- Price at execution
    close_price         DOUBLE PRECISION,     -- Price at day close
    next_open_price     DOUBLE PRECISION,     -- Next day open (for tracking error)
    actual_return_1d    DOUBLE PRECISION,     -- Actual next-day return
    strategy_return     DOUBLE PRECISION,     -- direction * leverage * actual_return
    slippage_bps        DOUBLE PRECISION,     -- Actual slippage observed
    tracking_error      DOUBLE PRECISION,     -- Difference vs backtest
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Index para queries rapidas
CREATE INDEX idx_vol_target_date ON forecast_vol_targeting_signals(signal_date DESC);
CREATE INDEX idx_paper_trading_date ON forecast_paper_trading(signal_date DESC);

-- View para monitoring
CREATE OR REPLACE VIEW v_paper_trading_performance AS
SELECT
    COUNT(*) as n_days,
    AVG(CASE WHEN strategy_return > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate_pct,
    SUM(strategy_return) as cumulative_return,
    AVG(strategy_return) * 252 as annualized_return,
    STDDEV(strategy_return) * SQRT(252) as annualized_vol,
    AVG(strategy_return) / NULLIF(STDDEV(strategy_return), 0) * SQRT(252) as sharpe,
    MIN(strategy_return) as worst_day,
    MAX(strategy_return) as best_day,
    AVG(ABS(tracking_error)) as avg_tracking_error
FROM forecast_paper_trading
WHERE signal_date >= CURRENT_DATE - INTERVAL '60 days';
```

#### Paso 1.4 — Config YAML para vol-targeting

**Nuevo archivo**: `config/forecast_experiments/vol_target_v1.yaml`

```yaml
experiment:
  name: "vol_target_v1"
  version: "1.0.0"
  description: "Vol-targeting position sizing on baseline_v1 forecasting model"
  hypothesis: "Vol-targeting maintains Sharpe while scaling returns proportionally"
  baseline_experiment: "baseline_v1"
  variable_changed: "Position sizing: fixed 1x → vol-targeting"

vol_targeting:
  target_vol: 0.15            # Start conservative
  max_leverage: 2.0
  min_leverage: 0.5
  vol_lookback: 21            # Days
  vol_floor: 0.05             # Annualized minimum

forecasting:
  horizon: 1                  # Only H=1 validated
  ensemble_strategy: "top_3"  # Most stable
  model_config: "baseline_v1" # No changes to model

evaluation:
  walk_forward_windows: 4     # Match baseline validation
  primary_metric: "sharpe_ratio"
  secondary_metrics:
    - "total_return"
    - "max_drawdown"
    - "avg_leverage"

gates:
  min_sharpe: 0.8             # Must maintain decent risk-adjusted returns
  max_drawdown_pct: 20.0      # Absolute ceiling
  max_avg_leverage: 1.8       # Safety
  bootstrap_ci_excludes_zero: true
```

---

### FASE 2: Validacion en paper trading (Semana 3-6)

**Objetivo**: Confirmar que el edge existe fuera del backtest en 60 dias de paper trading.

#### Paso 2.1 — Nuevo DAG: `airflow/dags/forecast_l5c_vol_targeting.py`

**Schedule**: `30 7 * * 1-5` (7:30 AM COT, Mon-Fri, despues de que macro data esta disponible)

**Flujo**:
```
1. check_market_day()          → Es dia de trading? (TradingCalendar)
2. load_latest_forecast()      → Lee de bi.fact_forecasts WHERE horizon=1 AND date=today
3. compute_realized_vol()      → 21-day rolling vol from bi.dim_daily_usdcop
4. compute_vol_target_signal() → leverage, position_size
5. persist_signal()            → INSERT INTO forecast_vol_targeting_signals
6. notify_signal_ready()       → pg_notify('forecast_signal_ready', ...)
7. log_to_mlflow()             → Track signal history
```

**Dependencias**:
- Requiere que `forecast_l5b_inference` ya haya corrido (genera bi.fact_forecasts)
- Requiere OHLCV diario actualizado (para vol calculo)

#### Paso 2.2 — Nuevo DAG: `airflow/dags/forecast_l6_paper_trading_monitor.py`

**Schedule**: `0 14 * * 1-5` (2:00 PM COT = despues del cierre de sesion 12:55 COT)

**Flujo**:
```
1. fetch_todays_close()        → Lee close price de hoy de bi.dim_daily_usdcop
2. fetch_todays_signal()       → Lee signal de forecast_vol_targeting_signals
3. compute_paper_result()      → actual_return = ln(close_today / close_yesterday)
                               → strategy_return = direction * leverage * actual_return
4. compute_tracking_error()    → Diferencia vs backtest esperado
5. persist_result()            → INSERT INTO forecast_paper_trading
6. daily_significance_check()  → Binomial test, running Sharpe
7. check_stop_criteria()       → Evaluar gates de parada
8. alert_if_needed()           → Slack/email si gate triggered
```

#### Paso 2.3 — Criterios de parada (hardcoded en DAG)

```python
# En forecast_l6_paper_trading_monitor.py

STOP_CRITERIA = {
    # CONTINUAR si despues de N dias:
    'continue': {
        'min_da': 0.50,       # DA > 50%
        'min_cum_return': 0,  # Return acumulado > 0
    },

    # PAUSAR Y REVISAR si:
    'pause': {
        'da_threshold': 0.48,           # DA < 48% despues de 40+ dias
        'max_drawdown_pct': 15.0,       # MaxDD > 15%
        'consecutive_negative_weeks': 3, # 3 semanas negativas seguidas
        'min_days_for_pause': 40,
    },

    # DETENER si:
    'stop': {
        'da_threshold': 0.46,           # DA < 46% despues de 60+ dias
        'max_drawdown_pct': 20.0,       # MaxDD > 20%
        'min_days_for_stop': 60,
        'p_value_threshold': 0.90,      # p>0.90 = edge murio
    },
}

def daily_significance_check(results_df: pd.DataFrame) -> dict:
    """Check estadistico diario despues de cada dia de paper trading."""
    from scipy import stats

    n = len(results_df)
    n_correct = (results_df['actual_return'] * results_df['signal_direction'] > 0).sum()

    binom = stats.binomtest(n_correct, n, 0.5, alternative='greater')

    daily_rets = results_df['strategy_return']
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

    return {
        'n_days': n,
        'da': n_correct / n if n > 0 else 0,
        'binom_p': binom.pvalue,
        'cum_return': daily_rets.sum(),
        'annualized_return': daily_rets.mean() * 252,
        'sharpe': sharpe,
        'max_drawdown': ((1 + daily_rets).cumprod() / (1 + daily_rets).cumprod().cummax() - 1).min(),
        'significant_at_10pct': binom.pvalue < 0.10,
        'significant_at_5pct': binom.pvalue < 0.05,
    }
```

#### Paso 2.4 — Dashboard integration

Agregar endpoint a `services/inference_api/`:

```python
# Nuevo: services/inference_api/routers/forecast_monitoring.py

@router.get("/api/forecast/paper-trading/summary")
async def get_paper_trading_summary():
    """Retorna metricas acumuladas del paper trading."""
    # Query v_paper_trading_performance view
    ...

@router.get("/api/forecast/paper-trading/daily")
async def get_paper_trading_daily(days: int = 30):
    """Retorna resultados diarios del paper trading."""
    # Query forecast_paper_trading table
    ...

@router.get("/api/forecast/vol-targeting/current")
async def get_current_signal():
    """Retorna la signal de vol-targeting de hoy."""
    # Query forecast_vol_targeting_signals WHERE date = today
    ...
```

**Gate Fase 2**: DA > 51% con p < 0.10 despues de 60 dias de paper trading.

---

### FASE 3: Integrar RL como ejecutor intraday (Semana 7-10)

**Objetivo**: Que el RL mejore el timing intraday de la signal del forecasting.

```
SIN RL:  Forecasting dice "comprar" → compras al cierre 12:55 COT → captas |OC| = 0.56%
CON RL:  Forecasting dice "comprar" → RL busca mejor precio en 5-min bars (8:00-12:55)
         → captas parte del gap (HL - |OC|) = 0.36%/dia adicional potencial
         → Si captura 5%: +4.5%/year
         → Si captura 10%: +9%/year
```

#### Paso 3.1 — Nuevo experiment config: `config/experiments/exp_rl_executor_001.yaml`

Derivado de `v215b_baseline.yaml` con un solo cambio: **3 features nuevas en observation**.

```yaml
_meta:
  version: "5.3.0"
  experiment_id: "EXP-RL-EXECUTOR-001"
  contract_id: "CTR-PIPELINE-SSOT-001"
  based_on: "v215b_baseline.yaml"
  based_on_model: "v21.5b_temporal_features"
  based_on_performance:
    total_return: 2.51
    sharpe_ratio: 0.321
    seeds_positive: "4/5"
  variable_changed: "3 new observation features (forecast_direction, forecast_leverage, intraday_progress)"
  hypothesis: "Forecast signal as input enables RL to optimize intraday execution timing"
  created_at: "2026-02-XX"
  status: "pending"

features:
  market_features:
    # ... los 18 market features de V21.5b (sin cambios) ...

    # NUEVOS: 3 features de forecast signal
    - name: "forecast_direction"
      order: 18
      calculator: "pass_through"      # Viene pre-computado
      normalization: "none"           # Ya es [-1, +1]
      description: "Daily forecast direction from L5b ensemble"

    - name: "forecast_leverage"
      order: 19
      calculator: "pass_through"
      normalization: "minmax"         # [0.5, 2.0] → [0, 1]
      description: "Vol-targeting leverage from L5c"

    - name: "intraday_progress"
      order: 20
      calculator: "pass_through"
      normalization: "none"           # Ya es [0, 1]
      description: "Fraction of trading session elapsed (0=8:00, 1=12:55)"

  state_features:
    # ... los 9 state features de V21.5b (sin cambios) ...

  observation_dim: 30               # Was 27: 18 market + 3 new + 9 state

training:
  environment:
    # Mismos parametros de V21.5b EXCEPTO:
    episode_length: 2400
    decision_interval: 1            # Cada barra de 5 min
    min_hold_bars: 25               # V21.5b optimal
    # NUEVO: constraint de direccion
    forecast_constrained: true      # RL NO puede ir contra la direccion del forecast

  reward:
    # CAMBIO CRITICO: reward = execution alpha, NO log-return puro
    mode: "execution_alpha"         # NEW mode
    weights:
      execution_alpha: 0.70         # Mejora vs benchmark (open-to-close)
      pnl: 0.20                     # Return absoluto (para no ignorar PnL)
      holding_decay: 0.05           # Anti-zombie
      regime: 0.05                  # Vol awareness
```

#### Paso 3.2 — Modificar `TradingEnvironment` para recibir forecast signal

**Archivo a modificar**: `src/training/environments/trading_env.py`

Cambios necesarios (minimos, ~50 lineas):

```python
# En __init__():
# Agregar parametro forecast_signals
self.forecast_signals = forecast_signals  # Dict[date_str, (direction, leverage)]
self.forecast_constrained = config.forecast_constrained  # bool

# En _get_observation():
# Agregar 3 features al observation vector
today = self._current_date()
signal = self.forecast_signals.get(today, (0, 1.0))
forecast_direction = signal[0]  # -1, 0, or +1
forecast_leverage = signal[1]   # 0.5 to 2.0

# Normalize
forecast_leverage_norm = (forecast_leverage - 0.5) / 1.5  # [0.5, 2.0] → [0, 1]

# Intraday progress
session_bars = 59  # 8:00-12:55 COT = 59 bars of 5 min
bar_in_session = self.current_idx % session_bars
intraday_progress = bar_in_session / session_bars  # [0, 1]

# Append to observation
obs = np.concatenate([
    market_features,           # 18
    [forecast_direction],      # 1
    [forecast_leverage_norm],  # 1
    [intraday_progress],       # 1
    state_features,            # 9
])  # Total: 30

# En step():
# Si forecast_constrained=True, bloquear acciones contra la direccion
if self.forecast_constrained:
    today = self._current_date()
    signal = self.forecast_signals.get(today, (0, 1.0))
    direction = signal[0]

    if direction > 0 and target_action == SHORT:
        target_action = HOLD  # No ir short cuando forecast dice long
    elif direction < 0 and target_action == LONG:
        target_action = HOLD  # No ir long cuando forecast dice short
```

#### Paso 3.3 — Nuevo reward mode: execution_alpha

**Archivo a modificar**: `src/training/reward_calculator.py`

```python
# Agregar nuevo modo al ModularRewardCalculator

def _compute_execution_alpha(
    self,
    entry_price: float,
    exit_price: float,
    forecast_direction: int,
    day_open: float,
    day_close: float,
    traded: bool,
    no_trade_penalty_factor: float = 0.1,
) -> float:
    """
    Execution alpha = mejora del RL sobre benchmark (open-to-close).

    3 casos:
    1. RL opero y cerro: alpha = actual_ret - benchmark_ret
    2. RL opero y no cerro (posicion abierta al cierre): alpha = mark-to-market - benchmark
    3. RL NO opero (HOLD todo el dia): alpha = -penalty_factor * abs(benchmark)
       → Penaliza ligeramente por no participar, pero NO tanto como perder el benchmark.
       → Esto evita que el RL sea penalizado 100% del benchmark por dias HOLD.

    Benchmark: comprar al open, vender al close del dia.
    """
    benchmark_ret = forecast_direction * (day_close - day_open) / day_open

    if not traded:
        # Caso 3: RL no opero — penalidad suave proporcional al benchmark
        return -no_trade_penalty_factor * abs(benchmark_ret)

    # Caso 1 y 2: RL opero
    actual_ret = forecast_direction * (exit_price - entry_price) / entry_price
    return actual_ret - benchmark_ret
```

#### Paso 3.4 — Generar forecast signals para entrenamiento

Para entrenar el RL executor, necesitamos forecast signals historicas para cada dia de training.

**Nuevo script**: `scripts/generate_historical_forecast_signals.py`

```python
"""
Genera forecast signals historicas para entrenar el RL executor.

Estrategia: Walk-forward con reentrenamiento cada 63 dias (~1 trimestre).
NO reentrenar diario (seria 9 modelos × 7 horizontes × 1,200 dias = 75,600 trainings = horas).

Para cada ventana de 63 dias en el periodo de training (2020-01 a 2024-12):
1. Entrenar 9 modelos × 7 horizontes con datos HASTA inicio de ventana
2. Generar predicciones para los 63 dias siguientes usando el modelo entrenado
3. Computar vol-targeting signal (direction + leverage) para cada dia
4. Avanzar ventana y repetir

Resultado: ~20 reentrenamientos × 63 predicciones = ~1,260 dias con signals.
Tiempo estimado: ~30-45 min (20 reentrenamientos × 63 modelos × ~1 seg cada uno).

Guardar como parquet con columnas: date, direction, leverage, predicted_return, vol_21d.
"""
```

**Output**: `data/forecasting/historical_forecast_signals.parquet`

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Trading day |
| forecast_direction | int | +1 or -1 |
| forecast_leverage | float | Vol-targeting leverage |
| forecast_return | float | Predicted log-return |
| realized_vol_21d | float | 21-day annualized vol |

#### Paso 3.5 — Modificar pipeline runner para RL executor mode

**Archivo a modificar**: `scripts/run_ssot_pipeline.py`

Agregar `--forecast-signals` flag:

```python
# En run_l2_dataset():
# Cargar forecast signals y mergear con dataset
if args.forecast_signals:
    fc_signals = pd.read_parquet(args.forecast_signals)
    # Merge daily signals → 5-min bars (broadcast same signal to all bars in day)
    dataset = merge_forecast_signals(dataset, fc_signals)

# En run_l3_training():
# Pasar forecast_signals al environment
if args.forecast_signals:
    env_kwargs['forecast_signals'] = load_forecast_signals(args.forecast_signals)
```

#### Paso 3.6 — Validacion multi-seed

```bash
# Entrenar 5 seeds del RL executor
python scripts/run_ssot_pipeline.py \
    --config config/experiments/exp_rl_executor_001.yaml \
    --forecast-signals data/forecasting/historical_forecast_signals.parquet \
    --multi-seed
```

**Gate Fase 3**:
- execution_alpha > 0 en >= 4/5 seeds
- Mejora en fill price > 0.01% promedio (significativa vs 0)
- No empeorar MaxDD vs ejecucion al cierre
- Bootstrap CI de execution_alpha excluye zero
- Total return con RL > total return sin RL (simple close execution)
- **Sharpe(con RL) >= Sharpe(sin RL) * 0.95** — RL no puede empeorar risk-adjusted returns
  (es posible que execution_alpha > 0 pero la varianza adicional degrade el Sharpe)

---

### FASE 4: Sistema integrado completo (Semana 11-12)

**Objetivo**: Orquestar todo en un flujo automatizado de produccion.

#### Paso 4.1 — DAG orquestador: `airflow/dags/integrated_forecast_rl.py`

```python
"""
DAG integrado: Forecasting (daily) + Vol-Targeting + RL Execution (5-min)

Schedule: Dos fases diarias
  Phase A: 7:00 AM COT (pre-market) — Genera signal diaria
  Phase B: 8:00-12:55 COT (intraday) — RL ejecuta signal

Dependencies:
  - core_l0_04_macro_update (macro data fresco)
  - forecast_l5b_inference (prediccion diaria)
"""

# Phase A: Pre-market (7:00 AM COT)
with TaskGroup("phase_a_daily_signal"):
    # 1. Validar que macro data esta actualizado
    check_macro_freshness = PythonOperator(...)

    # 2. Correr forecast inference (si no corrio ya)
    run_forecast = TriggerDagRunOperator(
        trigger_dag_id='forecast_l5b_inference',
        wait_for_completion=True,
    )

    # 3. Computar vol-targeting signal
    compute_vol_target = PythonOperator(
        python_callable=compute_vol_target_signal_task,
        # Lee: bi.fact_forecasts (H=1, today)
        # Escribe: forecast_vol_targeting_signals
    )

    # 4. Validar signal (sanity checks)
    validate_signal = PythonOperator(
        python_callable=validate_daily_signal,
        # Checks: direction != 0, leverage in [0.5, 2.0], vol > 0
    )

    # 5. Feed signal al RL agent
    feed_rl_signal = PythonOperator(
        python_callable=feed_signal_to_rl,
        # Escribe: signal a Redis stream 'forecast_signal:daily'
        # El L5 RL inference lee de aqui
    )

# Phase B: Intraday (cada 5 min, 8:00-12:55 COT)
# Reutiliza el DAG existente rl_l5_01_production_inference
# PERO con obs_dim=30 (3 features extra del forecast)
```

#### Paso 4.2 — Flujo de un dia tipico

```
07:00  core_l0_04_macro_update → macro data fresco en macro_indicators_daily
07:15  forecast_l5b_inference → prediccion H=1: "USDCOP sube, pred_return=+0.3%"
07:20  forecast_l5c_vol_targeting → vol_21d=12%, target=15% → leverage=1.25x
       Signal: direction=+1, size=+1.25, confidence=0.62
07:25  Signal publicada en Redis stream + forecast_vol_targeting_signals
07:30  rl_l5_01_production_inference carga signal del dia (obs[18:21] = [+1, 0.5, 0.0])

08:00  Mercado abre. RL observa barras de 5-min con forecast info.
       obs = [18 market features, +1, 0.5, 0.0, 9 state features]

08:15  RL ve dip de -0.15% (RSI sobrevendido) → COMPRA a 4,150
       obs[20] = 0.05 (3/59 bars transcurridos)

10:30  RL ve +0.4% de ganancia + momentum debilitandose
       obs[20] = 0.51 (30/59 bars transcurridos)
       → Mantiene (forecast dijo +0.3%, aun alineado)

11:45  RL ve reversal iniciando (RSI > 70, trend_z cayendo)
       obs[20] = 0.76 (45/59 bars transcurridos)
       → CIERRA POSICION a 4,167

       Entry: 4,150 (08:15), Exit: 4,167 (11:45) → +0.41%
       Con leverage 1.25x → +0.51% del capital

       Benchmark (open→close): +0.28% × 1.25 = +0.35%
       Execution alpha: +0.51% - 0.35% = +0.16%

12:55  Mercado cierra. Resultado registrado.
13:00  forecast_l6_paper_trading_monitor evalua resultado del dia.
```

#### Paso 4.3 — Modificar L5 inference para obs_dim=30

**Archivo a modificar**: `airflow/dags/l5_multi_model_inference.py`

```python
# En build_observation():
# Despues de leer FLOAT[18] de inference_ready_nrt

# NUEVO: Agregar forecast features si modelo es RL executor
if model_config.get('forecast_constrained', False):
    # Leer signal del dia desde Redis o DB
    signal = redis.get('forecast_signal:daily:latest')
    forecast_direction = signal['direction']
    forecast_leverage_norm = (signal['leverage'] - 0.5) / 1.5

    # Calcular intraday progress
    now_cot = datetime.now(tz=pytz.timezone('America/Bogota'))
    session_start = now_cot.replace(hour=8, minute=0)
    session_end = now_cot.replace(hour=12, minute=55)
    intraday_progress = (now_cot - session_start) / (session_end - session_start)
    intraday_progress = np.clip(intraday_progress, 0, 1)

    # Append to observation
    obs = np.concatenate([
        market_features,                    # 18
        [forecast_direction],               # 1
        [forecast_leverage_norm],           # 1
        [intraday_progress],               # 1
        [position, time_normalized],        # 2 (state subset for RL executor)
        # ... remaining state features
    ])
```

---

### FASE 5: Validacion estadistica completa (Semana 13+)

#### Paso 5.1 — Walk-forward del sistema completo

**Script**: `scripts/validate_integrated_system.py`

```python
"""
Walk-forward validation del sistema integrado:
  Forecasting + Vol-targeting + RL execution

4+ ventanas, cada una:
  1. Entrenar forecasting model hasta ventana N
  2. Generar forecast signals para ventana N+1
  3. Entrenar RL executor con forecast signals (hasta ventana N)
  4. Backtest RL executor en ventana N+1 con forecast signals
  5. Comparar: RL execution vs simple close execution

Gate:
  - Return con RL > Return sin RL en >= 3/4 ventanas
  - Bootstrap CI de execution_alpha excluye zero
  - Full system return > 0% en >= 3/4 ventanas
  - Full system Sharpe > 0.8
"""
```

#### Paso 5.2 — Go-live gradual

```
Semana 13-14: 25% del capital (1 modelo, solo COP)
Semana 15-16: 50% del capital (validar tracking error < 3%)
Semana 17-18: 75% del capital
Semana 19+:   100% del capital

En cualquier momento:
- Si MaxDD > 15% → reducir a 25%
- Si DA < 48% por 3 semanas → pausar
- Si Sharpe < 0.5 por 4 semanas → revisar modelo
```

---

## 7. RESUMEN DE ARCHIVOS A CREAR/MODIFICAR

### Nuevos archivos (Fase 1-2)

| # | Archivo | Proposito | Fase |
|---|---------|-----------|------|
| 1 | `src/forecasting/vol_targeting.py` | Modulo de vol-targeting (compute_position_size) | 1.2 |
| 2 | `scripts/vol_target_backtest.py` | Backtest offline de vol-targeting | 1.1 |
| 3 | `database/migrations/041_forecast_vol_targeting.sql` | Tablas para signals + paper trading | 1.3 |
| 4 | `config/forecast_experiments/vol_target_v1.yaml` | Config del experimento | 1.4 |
| 5 | `airflow/dags/forecast_l5c_vol_targeting.py` | DAG de vol-targeting diario | 2.1 |
| 6 | `airflow/dags/forecast_l6_paper_trading_monitor.py` | DAG de monitoring paper trading | 2.2 |
| 7 | `services/inference_api/routers/forecast_monitoring.py` | API endpoints para dashboard | 2.4 |

### Nuevos archivos (Fase 3-4)

| # | Archivo | Proposito | Fase |
|---|---------|-----------|------|
| 8 | `config/experiments/exp_rl_executor_001.yaml` | SSOT config para RL executor | 3.1 |
| 9 | `scripts/generate_historical_forecast_signals.py` | Pre-computa signals para training | 3.4 |
| 10 | `airflow/dags/integrated_forecast_rl.py` | DAG orquestador integrado | 4.1 |
| 11 | `scripts/validate_integrated_system.py` | Walk-forward validation completa | 5.1 |

### Archivos a modificar (Fase 3)

| # | Archivo | Cambio | Lineas aprox |
|---|---------|--------|--------------|
| 1 | `src/training/environments/trading_env.py` | +forecast_signals, +3 obs features, +direction constraint | ~50 lines |
| 2 | `src/training/reward_calculator.py` | +execution_alpha mode | ~30 lines |
| 3 | `scripts/run_ssot_pipeline.py` | +--forecast-signals flag | ~20 lines |
| 4 | `airflow/dags/l5_multi_model_inference.py` | +forecast features en build_observation | ~30 lines |

**Total**: ~11 archivos nuevos + 4 archivos modificados (~130 lineas cambiadas)

---

## 8. RETORNO ESPERADO (HONESTO)

### Fase 1-2: Vol-targeting sin RL

| Target Vol | Return Est. | Sharpe Est. | MaxDD Est. | Avg Leverage |
|------------|-------------|-------------|------------|-------------|
| 0.12 | ~14% | ~1.08 | ~10% | ~1.0x |
| **0.15** | **~18%** | **~1.08** | **~12%** | **~1.2x** |
| 0.18 | ~21% | ~1.08 | ~15% | ~1.5x |
| 0.20 | ~23% | ~1.08 | ~16% | ~1.6x |

**Nota**: Sharpe se mantiene constante porque vol-targeting escala return Y drawdown linealmente.

### Fase 3-4: Con RL execution

| Escenario | Return Adicional | Total Est. | Probabilidad |
|-----------|-----------------|------------|--------------|
| RL captura 0% del gap | +0%/yr | 18% | 40% (null hypothesis) |
| RL captura 5% del gap | +4.5%/yr | 22.5% | 35% |
| RL captura 10% del gap | +9%/yr | 27% | 20% |
| RL captura 15%+ del gap | +13.5%/yr | 31.5% | 5% |

### Para llegar a 30%:

- Necesitas vol-target ≥ 0.18 (leverage 1.5x promedio) → ~21% base
- MAS RL execution alpha de ≥ 9% (capturar 10%+ del gap intraday)
- MAS que no haya degradacion del edge en los 60 dias de paper trading
- **Probabilidad estimada: ~15-20%**. Es posible pero NO probable en el primer year.
- **Escenario mas probable**: 18-22% con vol-targeting. RL agrega 2-5% si funciona.

---

## 9. QUE NO HACER (validado por experimentos fallidos)

| # | NO hacer | Evidencia | Experiment ID |
|---|----------|-----------|---------------|
| 1 | Agregar macro features al forecasting (19→36) | Sharpe 1.08→0.49, overfitting | FC analysis |
| 2 | Usar H=5 ni horizontes largos | DA<50%, peor que coin flip | FC analysis |
| 3 | Usar macro score como filtro externo | ANOVA p=0.297, data snooping | EXP-INFRA-001 |
| 4 | Leverage > 2x | CI lower bound = +0.3%, amplifica errores | FC analysis |
| 5 | Confiar en 1 seed de RL | Seed 456 (eval=131) perdio -20% | EXP-V22-002 |
| 6 | RL en barras horarias | -7.16% best, no alpha | EXP-HOURLY-PPO-002 |
| 7 | RL en barras diarias | -8.83%, 8 trades, 12.5% WR | EXP-DAILY-PPO-001 |
| 8 | Asymmetric SL/TP tighter | 0/5 seeds, SL too tight | EXP-ASYM-001 |
| 9 | Cambiar multiples variables a la vez | Imposible atribuir mejoras | V22 (5 changes) |
| 10 | Confiar en 1 year de walk-forward | Exigir 4+ ventanas siempre | Statistical rigor |

---

## 10. DECISION TREE ACTUALIZADO

```
FASE 1: Vol-target backtest
  │
  ├── Sharpe > 1.0 con leverage?
  │     YES → Elegir target_vol optimo → FASE 2
  │     NO  → Vol-targeting no mejora, skip a FASE 3 con fixed sizing
  │
  FASE 2: Paper trading 60 dias
  │
  ├── DA > 51% con p < 0.10?
  │     YES → Edge confirmado en vivo → FASE 3
  │     NO  │
  │         ├── DA > 48%? → Extender a 90 dias
  │         └── DA < 46%? → STOP. Edge murio. Reentrenar forecasting.
  │
  FASE 3: RL executor training
  │
  ├── execution_alpha > 0 en >= 4/5 seeds?
  │     YES → RL agrega valor → FASE 4
  │     NO  → Ship sin RL (ejecucion al cierre 12:55 COT)
  │
  FASE 4: Sistema integrado
  │
  ├── Walk-forward 4+ ventanas positivas?
  │     YES → FASE 5 (go-live gradual)
  │     NO  → Revertir a solo forecasting + vol-targeting
  │
  FASE 5: Go-live
  │
  ├── 25% → 50% → 75% → 100%
  └── Monitoreo continuo (L6 drift DAG)
```

---

## 11. TIMELINE DETALLADO

| Semana | Fase | Actividad | Deliverable | Gate |
|--------|------|-----------|-------------|------|
| 1 | 1.1 | Vol-target backtest | `scripts/vol_target_backtest.py` + results | Sharpe > 1.0 |
| 1 | 1.2-1.4 | Modulo vol-targeting + config + migration | 4 archivos nuevos | Code review |
| 2 | 1.1 cont. | Analizar resultados, elegir target_vol | Decision document | Bootstrap CI |
| 3 | 2.1-2.2 | DAGs de paper trading + monitoring | 2 DAGs nuevos | DAGs functional |
| 3-6 | 2.3 | Paper trading diario (60 dias) | Daily results in DB | DA > 51%, p < 0.10 |
| 7 | 3.1-3.3 | RL executor config + env mods + reward | Config + 2 file mods | Unit tests pass |
| 8 | 3.4 | Generate historical forecast signals | Parquet con signals | Coverage > 95% |
| 8-9 | 3.5-3.6 | Train 5 seeds RL executor | 5 models | alpha > 0 in 4/5 |
| 10 | 3.6 | Analyze RL executor results | Experiment log entry | Gate pass/fail |
| 11 | 4.1-4.2 | Integrated DAG + flow | `integrated_forecast_rl.py` | E2E smoke test |
| 12 | 5.1 | Walk-forward complete system | Validation report | 3/4 windows + |
| 13+ | 5.2 | Go-live gradual | Live trading | Tracking error < 3% |
