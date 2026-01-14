# PLAN MAESTRO: 10 EXPERTOS
## USDCOP Trading System - Camino Pragmático (Path B)
## Fecha: 2026-01-08

---

## RESUMEN EJECUTIVO

**Decisión**: Camino B - Fix bugs + Reentrenar en PARALELO (2-3 semanas)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TIMELINE CONSOLIDADO                           │
├─────────────────────────────────────────────────────────────────────┤
│ SEMANA 1 (Días 1-5)                                                 │
│ ├── Track A: FIXES TÉCNICOS (Días 1-2)                              │
│ │   ├── StateTracker persistence                                    │
│ │   ├── Threshold 0.30 → 0.10                                       │
│ │   └── Observation logging                                         │
│ │                                                                   │
│ └── Track B: DISEÑO V20 (Días 1-5) [EN PARALELO]                    │
│     ├── Nuevo reward function                                       │
│     ├── Modificaciones al environment                               │
│     └── Setup de entrenamiento                                      │
│                                                                     │
│ SEMANA 2 (Días 6-12)                                                │
│ ├── Track A: Validación histórica + Paper trading limpio            │
│ └── Track B: Entrenamiento V20 + Hyperparameter tuning              │
│                                                                     │
│ SEMANA 3 (Días 13-21)                                               │
│ ├── Evaluación de V20 en backtest                                   │
│ ├── A/B testing V19 corregido vs V20                                │
│ └── Decisión final: Deploy mejor modelo                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PANEL DE 10 EXPERTOS

| # | Experto | Especialidad | Responsabilidad |
|---|---------|--------------|-----------------|
| 1 | **Dr. Chen** | Quant Finance PhD | Estrategia de trading |
| 2 | **Dr. Petrova** | RL/ML PhD | Arquitectura del modelo |
| 3 | **Ing. Martinez** | Software Architect | Sistema y código |
| 4 | **Rodriguez** | Trading Ops | Ejecución y monitoreo |
| 5 | **Dr. Kumar** | Data Engineer | Pipeline y calidad |
| 6 | **Schwartz** | Risk Manager | Controles de riesgo |
| 7 | **Thompson** | MLOps Engineer | Deployment y CI/CD |
| 8 | **Dr. Nakamura** | Financial Engineer | Costos de transacción |
| 9 | **Dr. Williams** | Behavioral Finance | Microestructura |
| 10 | **Garcia** | Project Manager | Coordinación |

---

## EXPERTO 1: Dr. Chen - Estrategia Cuantitativa

### Diagnóstico del Edge

```
SITUACIÓN ACTUAL:
┌─────────────────────────────────────────┐
│ Win Rate:        22.8%                  │
│ Required R:R:    3.5:1 (para break-even)│
│ Actual R:R:      ~1.2:1 (estimado)      │
│ Expected Value:  NEGATIVO               │
└─────────────────────────────────────────┘

PROBLEMA FUNDAMENTAL:
El modelo no tiene edge. Tradea como un random walker
con costos de transacción.
```

### Recomendaciones Estratégicas

1. **Reducir frecuencia de trading**
   - Target: 2-4 trades/día máximo (actual: ~6)
   - Implementar: Zona de HOLD más amplia

2. **Mejorar selectividad**
   - Solo tradear cuando confidence > threshold
   - Agregar filtro de volatilidad (no tradear en mercado lateral)

3. **Asymmetric Risk:Reward**
   ```python
   # Target para V20:
   min_rr_ratio = 2.0  # Mínimo 2:1
   stop_loss_pct = 0.5  # 0.5% stop
   take_profit_pct = 1.0  # 1.0% target
   ```

### Métricas Target para V20

| Métrica | V19 Actual | V20 Target | Mínimo Viable |
|---------|------------|------------|---------------|
| Win Rate | 22.8% | 40%+ | 35% |
| HOLD % | 0% | 30%+ | 20% |
| Trades/día | 6 | 2-3 | <5 |
| Sharpe Ratio | <0 | 1.5+ | 1.0 |
| Max Drawdown | ~5% | <10% | <15% |

---

## EXPERTO 2: Dr. Petrova - RL/ML Research

### Análisis del Reward Function V19 (Problemático)

```python
# REWARD V19 (ACTUAL) - PROBLEMÁTICO
def reward_v19(pnl_change):
    return pnl_change  # Solo P&L, sin costos ni penalizaciones
```

**Problemas identificados:**
1. No hay costo por tradear → Modelo aprende a over-tradear
2. No hay reward por esperar → 0% HOLD
3. Sin penalización por drawdown → Toma riesgos excesivos
4. Sin bonus por consistencia → Acciones erráticas

### Diseño del Reward Function V20

```python
# REWARD V20 (PROPUESTO) - CON PENALIZACIONES
class RewardCalculatorV20:
    def __init__(self):
        self.transaction_cost = 0.002    # 0.2% por trade (spread + slippage)
        self.hold_bonus = 0.0001         # Pequeño bonus por paciencia
        self.drawdown_penalty = 2.0      # Multiplicador de pérdidas
        self.consistency_bonus = 0.0005  # Bonus por trades ganadores consecutivos

    def calculate(self, pnl, action, prev_action, position_time, consecutive_wins, equity_peak, equity_current):
        reward = pnl

        # 1. COSTO POR CAMBIAR POSICIÓN
        if action != prev_action:
            reward -= self.transaction_cost

        # 2. BONUS POR HOLD (cuando acción débil)
        if abs(action) < 0.15:
            reward += self.hold_bonus * position_time

        # 3. PENALIZACIÓN ASIMÉTRICA POR PÉRDIDAS
        if pnl < 0:
            reward *= self.drawdown_penalty  # Pérdidas duelen más

        # 4. BONUS POR CONSISTENCIA
        if pnl > 0:
            reward += self.consistency_bonus * min(consecutive_wins, 5)

        # 5. PENALIZACIÓN POR DRAWDOWN EXCESIVO
        current_dd = (equity_peak - equity_current) / equity_peak
        if current_dd > 0.05:  # >5% drawdown
            reward -= 0.001 * current_dd

        return reward
```

### Modificaciones al Environment V20

```python
# environment_v20.py
class TradingEnvironmentV20(gym.Env):
    def __init__(self):
        # Observation space: 15 dims (igual que V19)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Action space: Continuo [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # NUEVOS PARÁMETROS V20
        self.action_threshold = 0.15      # Más amplio que 0.10
        self.min_hold_bars = 3            # Mínimo 15 min en posición
        self.max_position_bars = 60       # Máximo 5 horas

        # Tracking para reward
        self.consecutive_wins = 0
        self.equity_peak = 10000
        self.position_time = 0

    def step(self, action):
        # Discretizar acción con zona de HOLD más amplia
        if action > self.action_threshold:
            discrete_action = 1  # LONG
        elif action < -self.action_threshold:
            discrete_action = -1  # SHORT
        else:
            discrete_action = 0  # HOLD

        # Forzar hold mínimo
        if self.position_time < self.min_hold_bars and self.current_position != 0:
            discrete_action = self.current_position  # Mantener posición

        # Ejecutar trade y calcular reward
        pnl = self._execute_trade(discrete_action)
        reward = self.reward_calculator.calculate(
            pnl, discrete_action, self.prev_action,
            self.position_time, self.consecutive_wins,
            self.equity_peak, self.equity
        )

        # Actualizar tracking
        if pnl > 0:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0

        self.equity_peak = max(self.equity_peak, self.equity)
        self.prev_action = discrete_action

        return obs, reward, done, info
```

### Hyperparameters Recomendados para PPO V20

```python
ppo_config_v20 = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,        # Aumentar para más exploración
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.01,       # Limitar cambios de policy
    "normalize_advantage": True,
}
```

---

## EXPERTO 3: Ing. Martinez - Arquitectura de Software

### Fixes Técnicos Prioritarios

#### Fix 1: StateTracker Persistence (CRÍTICO)

```python
# src/core/state/state_tracker.py

import redis
import json
from dataclasses import asdict

class StateTracker:
    def __init__(self, model_id: str, redis_url: str = "redis://redis:6379"):
        self.model_id = model_id
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.state_key = f"trading:state:{model_id}"
        self._load_state()

    def _load_state(self):
        """Load state from Redis on startup."""
        stored = self.redis.get(self.state_key)
        if stored:
            data = json.loads(stored)
            self.position = data.get('position', 0)
            self.entry_price = data.get('entry_price', 0.0)
            self.equity = data.get('equity', 10000.0)
            self.realized_pnl = data.get('realized_pnl', 0.0)
            self.trade_count = data.get('trade_count', 0)
            self.wins = data.get('wins', 0)
            self.losses = data.get('losses', 0)
        else:
            self._reset_to_defaults()

    def _persist_state(self):
        """Persist state to Redis after every update."""
        state_dict = {
            'position': self.position,
            'entry_price': self.entry_price,
            'equity': self.equity,
            'realized_pnl': self.realized_pnl,
            'trade_count': self.trade_count,
            'wins': self.wins,
            'losses': self.losses,
            'last_updated': datetime.utcnow().isoformat()
        }
        self.redis.set(self.state_key, json.dumps(state_dict))

        # Also persist to PostgreSQL for durability
        self._persist_to_postgres(state_dict)

    def _persist_to_postgres(self, state_dict):
        """Backup state to PostgreSQL."""
        import psycopg2
        conn_string = os.getenv('DATABASE_URL')
        with psycopg2.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO trading_state
                    (model_id, position, entry_price, equity, realized_pnl,
                     trade_count, wins, losses, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (model_id) DO UPDATE SET
                        position = EXCLUDED.position,
                        entry_price = EXCLUDED.entry_price,
                        equity = EXCLUDED.equity,
                        realized_pnl = EXCLUDED.realized_pnl,
                        trade_count = EXCLUDED.trade_count,
                        wins = EXCLUDED.wins,
                        losses = EXCLUDED.losses,
                        last_updated = NOW()
                ''', (
                    self.model_id,
                    state_dict['position'],
                    state_dict['entry_price'],
                    state_dict['equity'],
                    state_dict['realized_pnl'],
                    state_dict['trade_count'],
                    state_dict['wins'],
                    state_dict['losses']
                ))
                conn.commit()

    def update_position(self, new_position, entry_price=None):
        """Update position and persist."""
        self.position = new_position
        if entry_price:
            self.entry_price = entry_price
        self._persist_state()

    def record_trade(self, pnl):
        """Record completed trade."""
        self.realized_pnl += pnl
        self.equity += pnl
        self.trade_count += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self._persist_state()
```

#### Fix 2: Observation Logging

```python
# airflow/dags/l5_multi_model_inference.py

def run_inference(model, observation, model_id):
    """Run inference with full logging."""
    import logging
    logger = logging.getLogger(__name__)

    # Log observation vector
    logger.info(f"[{model_id}] Observation shape: {observation.shape}")
    logger.info(f"[{model_id}] Observation values: {observation.tolist()}")

    # Validate observation dimension
    expected_dim = 15
    if observation.shape[0] != expected_dim:
        logger.error(f"[{model_id}] DIMENSION MISMATCH: got {observation.shape[0]}, expected {expected_dim}")
        raise ValueError(f"Observation dimension mismatch")

    # Run inference
    action, _ = model.predict(observation, deterministic=True)

    logger.info(f"[{model_id}] Raw action: {action}")
    logger.info(f"[{model_id}] Discretized: {'LONG' if action > 0.10 else 'SHORT' if action < -0.10 else 'HOLD'}")

    return action
```

#### Fix 3: Threshold Consistency

```sql
-- Ejecutar en PostgreSQL
UPDATE config.models
SET
    threshold_long = 0.10,
    threshold_short = -0.10,
    updated_at = NOW()
WHERE model_id IN ('ppo_v1', 'sac_v19_baseline', 'td3_v19_baseline', 'a2c_v19_baseline');
```

---

## EXPERTO 4: Rodriguez - Trading Operations

### Plan de Validación Post-Fixes

```
DÍA 3: VALIDACIÓN HISTÓRICA
┌─────────────────────────────────────────────────────────────┐
│ 1. Correr simulación histórica (Dec 27 - Jan 6)             │
│ 2. Comparar con resultados grabados                         │
│                                                             │
│    SI P&L ≈ -3.54% → Modelo es el problema (confirmar V20)  │
│    SI P&L ≠ -3.54% → Bugs estaban corrompiendo datos        │
└─────────────────────────────────────────────────────────────┘

DÍA 4-5: PAPER TRADING LIMPIO (V19 corregido)
┌─────────────────────────────────────────────────────────────┐
│ • Ejecutar con todos los fixes aplicados                    │
│ • Monitorear distribución de señales                        │
│ • Verificar que HOLD aparece (aunque sea poco)              │
│ • Logging completo de observations                          │
└─────────────────────────────────────────────────────────────┘
```

### Monitoring Dashboard Requerido

```python
# Métricas a monitorear en tiempo real
monitoring_metrics = {
    # Performance
    "equity": "current",
    "pnl_daily": "rolling_sum_1d",
    "pnl_weekly": "rolling_sum_7d",
    "sharpe_ratio": "rolling_30d",

    # Signal Distribution
    "signal_long_pct": "rolling_100_trades",
    "signal_short_pct": "rolling_100_trades",
    "signal_hold_pct": "rolling_100_trades",  # DEBE ser > 0

    # Risk
    "max_drawdown": "peak_to_trough",
    "consecutive_losses": "current",
    "position_duration_avg": "rolling_50_trades",

    # System Health
    "observation_dim": "last",  # DEBE ser 15
    "macro_null_rate": "rolling_24h",
    "state_persistence_ok": "boolean",
}
```

### Alertas Críticas

| Alerta | Condición | Acción |
|--------|-----------|--------|
| **HALT** | observation_dim != 15 | Pausar trading, investigar |
| **HALT** | consecutive_losses > 10 | Pausar, revisar modelo |
| **WARNING** | signal_hold_pct < 5% | Monitorear, posible overtrading |
| **WARNING** | macro_null_rate > 30% | Verificar scraper |
| **INFO** | max_drawdown > 5% | Notificar, seguir monitoreando |

---

## EXPERTO 5: Dr. Kumar - Data Engineering

### Pipeline de Datos para V20

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE V20                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OHLCV (5min)    Macro (daily)     Training Data                    │
│       │               │                  │                          │
│       ▼               ▼                  ▼                          │
│  ┌─────────────────────────────────────────────┐                    │
│  │         FEATURE BUILDER V20                  │                    │
│  │  ─────────────────────────────────          │                    │
│  │  • 13 features técnicos                     │                    │
│  │  • 2 features de estado                      │                    │
│  │  • Normalización Z-score con stats de train │                    │
│  └─────────────────────────────────────────────┘                    │
│                        │                                            │
│                        ▼                                            │
│  ┌─────────────────────────────────────────────┐                    │
│  │         OBSERVATION VECTOR (15-dim)         │                    │
│  │  ─────────────────────────────────          │                    │
│  │  [log_ret_5m, log_ret_1h, rsi_9, ...]       │                    │
│  └─────────────────────────────────────────────┘                    │
│                        │                                            │
│           ┌────────────┴────────────┐                               │
│           ▼                         ▼                               │
│    ┌──────────────┐          ┌──────────────┐                       │
│    │  TRAINING    │          │  INFERENCE   │                       │
│    │  (offline)   │          │  (realtime)  │                       │
│    └──────────────┘          └──────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dataset V20 Requirements

```python
# Configuración del dataset para entrenamiento V20
dataset_config_v20 = {
    "source": "usdcop_m5_ohlcv",
    "date_range": {
        "train": ("2020-01-01", "2024-12-31"),  # 5 años
        "validation": ("2025-01-01", "2025-06-30"),  # 6 meses
        "test": ("2025-07-01", "2025-12-31"),  # 6 meses OOS
    },
    "features": [
        "log_ret_5m", "log_ret_1h",
        "rsi_9", "macd_hist", "bb_width",
        "vol_ratio", "atr_pct",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "dxy_z", "vix_z",
        "position", "time_normalized"
    ],
    "normalization": {
        "method": "z_score",
        "clip_range": (-5, 5),
        "use_train_stats": True  # CRÍTICO: usar stats de train en inference
    },
    "filters": {
        "market_hours_only": True,  # 8:00-12:55 COT
        "exclude_holidays": True,
        "min_volume": 0,  # No filter for FX
    }
}
```

### Data Quality Checks

```python
def validate_observation(obs, feature_names):
    """Validate observation before inference."""
    checks = {
        "dimension": len(obs) == 15,
        "no_nan": not np.isnan(obs).any(),
        "no_inf": not np.isinf(obs).any(),
        "reasonable_range": np.all(np.abs(obs) < 10),  # Z-scores should be < 10
    }

    for check_name, passed in checks.items():
        if not passed:
            logger.error(f"Observation validation failed: {check_name}")
            logger.error(f"Values: {dict(zip(feature_names, obs))}")
            return False

    return True
```

---

## EXPERTO 6: Schwartz - Risk Management

### Controles de Riesgo para V20

```python
class RiskManagerV20:
    def __init__(self):
        # Límites de riesgo
        self.max_daily_loss = -0.02        # -2% máximo diario
        self.max_drawdown = -0.10          # -10% máximo drawdown
        self.max_consecutive_losses = 8    # Pausa después de 8 pérdidas
        self.max_position_size = 1.0       # 100% del capital (ajustar para real)

        # Cooldown después de pérdidas
        self.cooldown_after_dd = 60        # 60 minutos (12 barras)

        # Estado
        self.daily_pnl = 0.0
        self.drawdown = 0.0
        self.consecutive_losses = 0
        self.in_cooldown = False
        self.cooldown_until = None

    def check_can_trade(self, current_equity, peak_equity, last_trade_pnl=None):
        """Check if trading is allowed based on risk limits."""

        # Update metrics
        self.drawdown = (peak_equity - current_equity) / peak_equity

        if last_trade_pnl is not None:
            self.daily_pnl += last_trade_pnl
            if last_trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        # Check limits
        if self.daily_pnl <= self.max_daily_loss:
            return False, "DAILY_LOSS_LIMIT"

        if self.drawdown >= abs(self.max_drawdown):
            return False, "MAX_DRAWDOWN"

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, "CONSECUTIVE_LOSSES"

        if self.in_cooldown and datetime.now() < self.cooldown_until:
            return False, "IN_COOLDOWN"

        return True, "OK"

    def calculate_position_size(self, confidence, volatility):
        """Dynamic position sizing based on confidence and volatility."""
        # Kelly-inspired sizing
        base_size = 0.5  # 50% base

        # Adjust for confidence (action magnitude)
        confidence_factor = min(abs(confidence), 1.0)

        # Adjust for volatility (reduce size in high vol)
        vol_factor = 1.0 / (1.0 + volatility * 10)

        size = base_size * confidence_factor * vol_factor
        return min(size, self.max_position_size)
```

### Risk Limits Comparison

| Parámetro | V19 (Actual) | V20 (Propuesto) | Razón |
|-----------|--------------|-----------------|-------|
| Max Daily Loss | -5% | -2% | Más conservador |
| Max Drawdown | -15% | -10% | Preservar capital |
| Max Consec. Losses | Sin límite | 8 | Prevenir tilt |
| Position Size | 100% fijo | 20-80% dinámico | Risk-adjusted |
| Cooldown | No | 1 hora post-DD | Recovery time |

---

## EXPERTO 7: Thompson - MLOps Engineer

### Training Pipeline V20

```yaml
# training_pipeline_v20.yaml
name: ppo_v20_training
version: "2.0"

stages:
  - name: data_preparation
    script: scripts/prepare_dataset_v20.py
    outputs:
      - data/processed/train_v20.parquet
      - data/processed/val_v20.parquet
      - data/processed/test_v20.parquet
      - config/norm_stats_v20.json

  - name: training
    script: notebooks/train_ppo_v20.py
    inputs:
      - data/processed/train_v20.parquet
      - config/norm_stats_v20.json
    outputs:
      - models/ppo_v20/model.zip
      - models/ppo_v20/metrics.json
    params:
      total_timesteps: 5_000_000
      learning_rate: 3e-4
      n_steps: 2048
      batch_size: 64

  - name: export_onnx
    script: scripts/export_to_onnx.py
    inputs:
      - models/ppo_v20/model.zip
    outputs:
      - models/ppo_v20/model.onnx

  - name: validation
    script: scripts/validate_model.py
    inputs:
      - models/ppo_v20/model.onnx
      - data/processed/val_v20.parquet
    outputs:
      - models/ppo_v20/validation_report.json

  - name: backtest
    script: scripts/backtest_v20.py
    inputs:
      - models/ppo_v20/model.onnx
      - data/processed/test_v20.parquet
    outputs:
      - models/ppo_v20/backtest_results.json
      - models/ppo_v20/equity_curve.png
```

### Deployment Checklist

```markdown
## Pre-Deployment Checklist V20

### Model Validation
- [ ] Observation space matches training (15 dims)
- [ ] Action space is continuous [-1, 1]
- [ ] ONNX export validates correctly
- [ ] Inference latency < 50ms

### Backtest Results
- [ ] Win rate >= 35%
- [ ] Sharpe ratio >= 1.0
- [ ] Max drawdown <= 15%
- [ ] HOLD signals >= 15%

### Integration Tests
- [ ] Model loads in production environment
- [ ] StateTracker persistence works
- [ ] Risk manager limits enforced
- [ ] Observation builder produces valid outputs

### Monitoring
- [ ] Prometheus metrics configured
- [ ] Grafana dashboards deployed
- [ ] Alerting rules active
- [ ] Logging level appropriate
```

---

## EXPERTO 8: Dr. Nakamura - Financial Engineering

### Modelo de Costos de Transacción

```python
class TransactionCostModel:
    """Realistic transaction cost model for USD/COP."""

    def __init__(self):
        # Spread (bid-ask)
        self.base_spread_bps = 30  # 30 bps = 0.30%

        # Slippage model
        self.slippage_base_bps = 10  # 10 bps base
        self.slippage_vol_factor = 0.5  # Additional slippage in high vol

        # Market impact (for larger orders)
        self.impact_coefficient = 0.1

    def calculate_spread_cost(self, price):
        """Calculate bid-ask spread cost."""
        return price * (self.base_spread_bps / 10000)

    def calculate_slippage(self, price, volatility, order_size_pct=0.01):
        """Calculate expected slippage."""
        base_slip = price * (self.slippage_base_bps / 10000)
        vol_slip = price * volatility * self.slippage_vol_factor
        impact = price * order_size_pct * self.impact_coefficient

        return base_slip + vol_slip + impact

    def total_cost(self, price, side, volatility, order_size_pct=0.01):
        """Calculate total transaction cost."""
        spread = self.calculate_spread_cost(price)
        slippage = self.calculate_slippage(price, volatility, order_size_pct)

        total = spread + slippage

        # Direction adjustment
        if side == 'BUY':
            return price + total
        else:
            return price - total

    def cost_per_roundtrip(self, price, volatility):
        """Cost for a complete round-trip trade."""
        return 2 * (self.calculate_spread_cost(price) +
                    self.calculate_slippage(price, volatility))
```

### Impacto en P&L

```
ESCENARIO: 6 trades/día con modelo actual

Precio promedio:    4,230 COP
Costo por trade:    ~0.5% (spread + slippage)
Costo diario:       6 × 0.5% = 3.0%
Costo semanal:      ~15% (!!)

CONCLUSIÓN:
El overtrading está consumiendo cualquier alpha potencial.
Con 57 trades en 10 días, los costos de transacción
representan ~28.5% del capital.

El -3.54% de P&L es en realidad MEJOR de lo esperado
si consideramos los costos ocultos.
```

---

## EXPERTO 9: Dr. Williams - Behavioral Finance

### Análisis de Microestructura

```
USD/COP - CARACTERÍSTICAS DEL MERCADO
┌─────────────────────────────────────────────────────────────┐
│ • Mercado emergente con baja liquidez                       │
│ • Horario limitado: 8:00 AM - 12:55 PM COT (5 horas)        │
│ • Spread típico: 0.3-0.5%                                   │
│ • Movimientos correlacionados con:                          │
│   - DXY (índice dólar)                                      │
│   - Precio del petróleo                                     │
│   - EMBI Colombia (riesgo país)                             │
│   - Decisiones Banco de la República                        │
│                                                             │
│ IMPLICACIONES PARA TRADING:                                 │
│ • Evitar tradear en apertura (primeros 15 min)              │
│ • Evitar tradear en cierre (últimos 15 min)                 │
│ • Mejor liquidez: 9:30 AM - 11:30 AM COT                    │
│ • Evitar días de anuncios macro                             │
└─────────────────────────────────────────────────────────────┘
```

### Filtros de Microestructura para V20

```python
class MicrostructureFilters:
    """Filters based on market microstructure."""

    def should_trade(self, current_time, volatility, spread):
        """Determine if conditions are favorable for trading."""

        hour = current_time.hour
        minute = current_time.minute

        # Avoid first 15 minutes (high spread)
        if hour == 8 and minute < 15:
            return False, "OPENING_AVOID"

        # Avoid last 15 minutes (low liquidity)
        if hour == 12 and minute > 40:
            return False, "CLOSING_AVOID"

        # Avoid high spread periods
        if spread > 0.005:  # >0.5%
            return False, "HIGH_SPREAD"

        # Avoid extreme volatility
        if volatility > 0.02:  # >2% intraday
            return False, "HIGH_VOLATILITY"

        return True, "OK"
```

---

## EXPERTO 10: Garcia - Project Manager

### PLAN MAESTRO CONSOLIDADO

```
══════════════════════════════════════════════════════════════════════
                        PLAN MAESTRO V20
                    Duración Total: 3 Semanas
══════════════════════════════════════════════════════════════════════

SEMANA 1: ESTABILIZACIÓN + DISEÑO
─────────────────────────────────────────────────────────────────────
│ Día │ Track A (Fixes)           │ Track B (V20 Design)         │
├─────┼───────────────────────────┼──────────────────────────────┤
│  1  │ StateTracker persistence  │ Reward function V20          │
│  2  │ Threshold fix + logging   │ Environment V20              │
│  3  │ Validación histórica      │ Dataset preparation          │
│  4  │ Paper trading limpio      │ Training setup               │
│  5  │ Análisis de resultados    │ Hyperparameter config        │
─────────────────────────────────────────────────────────────────────

SEMANA 2: ENTRENAMIENTO + VALIDACIÓN
─────────────────────────────────────────────────────────────────────
│ Día │ Track A (V19 Monitor)     │ Track B (V20 Training)       │
├─────┼───────────────────────────┼──────────────────────────────┤
│  6  │ Paper trading continuo    │ PPO training (1M steps)      │
│  7  │ Colectar métricas         │ PPO training (2M steps)      │
│  8  │ Análisis distribución     │ PPO training (3M steps)      │
│  9  │ Comparar con V20 backtest │ PPO training (4M steps)      │
│ 10  │ Documentar hallazgos      │ PPO training (5M steps)      │
│ 11  │ Preparar A/B test         │ Export ONNX + Validation     │
│ 12  │ Setup A/B environment     │ Backtest OOS                 │
─────────────────────────────────────────────────────────────────────

SEMANA 3: A/B TESTING + DECISIÓN
─────────────────────────────────────────────────────────────────────
│ Día │ Actividad                                                  │
├─────┼────────────────────────────────────────────────────────────┤
│ 13  │ Deploy V20 en paper trading paralelo                       │
│ 14  │ A/B: V19 corregido vs V20 (día 1)                          │
│ 15  │ A/B: V19 corregido vs V20 (día 2)                          │
│ 16  │ A/B: V19 corregido vs V20 (día 3)                          │
│ 17  │ A/B: V19 corregido vs V20 (día 4)                          │
│ 18  │ A/B: V19 corregido vs V20 (día 5)                          │
│ 19  │ Análisis de resultados A/B                                 │
│ 20  │ DECISIÓN: Deploy ganador                                   │
│ 21  │ Documentación + Retrospectiva                              │
─────────────────────────────────────────────────────────────────────

```

### Criterios de Éxito

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CRITERIOS DE ÉXITO V20                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MÍNIMO VIABLE (para continuar development):                        │
│  ├── Win Rate >= 30%                                                │
│  ├── HOLD signals >= 15%                                            │
│  ├── Sharpe Ratio >= 0.5                                            │
│  └── Max Drawdown <= 20%                                            │
│                                                                     │
│  TARGET (para paper trading extendido):                             │
│  ├── Win Rate >= 40%                                                │
│  ├── HOLD signals >= 25%                                            │
│  ├── Sharpe Ratio >= 1.0                                            │
│  └── Max Drawdown <= 15%                                            │
│                                                                     │
│  EXCELENTE (para considerar live trading):                          │
│  ├── Win Rate >= 50%                                                │
│  ├── HOLD signals >= 30%                                            │
│  ├── Sharpe Ratio >= 1.5                                            │
│  └── Max Drawdown <= 10%                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Decisión Final (Día 20)

```
                    MATRIZ DE DECISIÓN A/B
                    ─────────────────────

                        V20 Win Rate
                    <30%    30-40%   >40%
                  ┌───────┬────────┬────────┐
           <30%  │ STOP  │ V20    │ V20    │
V19        ─────┼───────┼────────┼────────┤
Win       30-40%│ V19   │ TEST+  │ V20    │
Rate       ─────┼───────┼────────┼────────┤
           >40% │ V19   │ V19    │ TEST+  │
                  └───────┴────────┴────────┘

STOP   = Pausar todo, revisar fundamentalmente
V19    = Continuar con V19 corregido
V20    = Deploy V20
TEST+  = Extender testing 2 semanas más
```

---

## RESUMEN EJECUTIVO FINAL

### Acciones Inmediatas (HOY)

```bash
# 1. Ejecutar fixes SQL
docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading \
  -c "UPDATE config.models SET threshold_long=0.10, threshold_short=-0.10"

# 2. Implementar StateTracker persistence (código en sección 3)

# 3. Iniciar diseño de reward V20 (código en sección 2)
```

### Entregables por Semana

| Semana | Entregables |
|--------|-------------|
| 1 | Fixes aplicados, Environment V20 diseñado, Dataset preparado |
| 2 | V19 paper trading limpio, V20 entrenado, Backtest completado |
| 3 | A/B test completado, Decisión tomada, Modelo ganador deployado |

### Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| V20 no mejora | Media | Alto | Tener V21 con más cambios listo |
| Bugs en nuevo código | Alta | Medio | Testing exhaustivo, rollback plan |
| Datos macro siguen NULL | Media | Medio | Implementar scraper alternativo |
| Overfit en training | Media | Alto | Usar validation split, early stopping |

---

## APROBACIÓN DEL PLAN

```
┌─────────────────────────────────────────────────────────────────────┐
│  FIRMA DE EXPERTOS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ✓ Dr. Chen (Quant Finance)      - Aprobado con recomendaciones     │
│  ✓ Dr. Petrova (ML/RL)           - Aprobado                         │
│  ✓ Ing. Martinez (Software)      - Aprobado con fixes prioritarios  │
│  ✓ Rodriguez (Trading Ops)       - Aprobado                         │
│  ✓ Dr. Kumar (Data Engineering)  - Aprobado                         │
│  ✓ Schwartz (Risk)               - Aprobado con límites estrictos   │
│  ✓ Thompson (MLOps)              - Aprobado                         │
│  ✓ Dr. Nakamura (Fin. Eng.)      - Aprobado                         │
│  ✓ Dr. Williams (Behavioral)     - Aprobado con filtros adicionales │
│  ✓ Garcia (PM)                   - Plan ejecutable aprobado         │
└─────────────────────────────────────────────────────────────────────┘

FECHA: 2026-01-08
PRÓXIMA REVISIÓN: 2026-01-15 (fin Semana 1)
```

---

*Plan Maestro generado por Panel de 10 Expertos*
*Claude Code Audit System*
