# V21 PPO Implementation Plan v2.0 - COMPREHENSIVE FIX

**Fecha**: 2026-01-13
**Basado en**: Analisis de 4 agentes exploradores + feedback experto

---

## HALLAZGOS CRITICOS DE LOS AGENTES

### 1. YAML NO SE CARGA AUTOMATICAMENTE
Los archivos `v20_config.yaml` y `v21_config.yaml` **NO son cargados** por el DAG de entrenamiento.
- El DAG usa `DEFAULT_TRAINING_CONFIG` hardcodeado en `l3_model_training.py`
- Para usar V21, hay que configurar Airflow Variable `training_config`

### 2. INCONSISTENCIA DE THRESHOLDS (CRITICO)
```
Archivo                                    | Thresholds
-------------------------------------------|-------------
config/v20_config.yaml                     | 0.30 / -0.30
src/training/environments/trading_env.py   | 0.10 / -0.10  ← USADO EN TRAINING
services/inference_api/inference_engine.py | 0.95 / -0.95  ← USADO EN INFERENCE!
services/inference_api/trade_simulator.py  | 0.95 / -0.95
airflow/dags/services/backtest_factory.py  | 0.95 / -0.95
```
**Esto significa que el modelo entrena con un threshold y se usa en produccion con otro completamente diferente.**

### 3. REWARD CALCULATOR NO INTEGRADO
- `RewardCalculatorV20` existe pero **NO se usa** en el TradingEnvironment
- El environment usa `DefaultRewardStrategy` que es mas simple
- Hold bonus NO es condicional a profit

### 4. REGIMEN DE VOLATILIDAD YA EXISTE
- `src/core/calculators/regime.py` tiene `detect_regime()` listo para usar
- Anti-lookahead validado
- Solo falta integrarlo

### 5. CIRCUIT BREAKER EXISTE EN PRODUCCION
- `src/lib/risk/circuit_breakers.py` tiene `LossCircuitBreaker`
- Consecutive loss tracking existe
- **PERO NO ESTA EN TRAINING ENVIRONMENT**

---

## ARQUITECTURA ACTUAL vs REQUERIDA

```
ACTUAL (V20):
┌─────────────────────────────────────────────────────────────────┐
│ Training DAG                                                    │
│   └─> Hardcoded defaults (gamma=0.99, ent=0.01, cost=25bps)    │
│       └─> TradingEnvironment                                   │
│           └─> DefaultRewardStrategy (simple)                   │
│               └─> Thresholds 0.10/-0.10                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓ ENTRENADO
┌─────────────────────────────────────────────────────────────────┐
│ Inference/Backtest                                              │
│   └─> Services hardcoded (0.95/-0.95) ← DIFERENTE!             │
│       └─> Production risk manager (circuit breakers)           │
└─────────────────────────────────────────────────────────────────┘

REQUERIDO (V21):
┌─────────────────────────────────────────────────────────────────┐
│ Single Source of Truth: v21_config.yaml                         │
│   └─> Cargado por: DAG, Environment, Inference, Backtest       │
│       └─> Thresholds consistentes: 0.33/-0.33                  │
│       └─> Costs consistentes: 75 bps                           │
│       └─> Hyperparams: gamma=0.90, ent=0.05                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## FASES DE IMPLEMENTACION

### FASE P0 - CRITICO (Antes de entrenar)

| # | Cambio | Archivo | Lineas | Prioridad |
|---|--------|---------|--------|-----------|
| P0.1 | Thresholds 0.33/-0.33 | `trading_env.py` | 151-152 | BLOQUEANTE |
| P0.2 | Transaction cost 75 bps | `trading_env.py` | 143 | BLOQUEANTE |
| P0.3 | Slippage 15 bps | `trading_env.py` | 144 | BLOQUEANTE |
| P0.4 | gamma 0.90 | `l3_model_training.py` | 168 | BLOQUEANTE |
| P0.5 | ent_coef 0.05 | `l3_model_training.py` | 171 | BLOQUEANTE |
| P0.6 | loss_penalty_multiplier 2.0 | `reward_calculator_v20.py` | 35 | ALTO |
| P0.7 | Disable hold_bonus (set 0) | `reward_calculator_v20.py` | 38 | ALTO |
| P0.8 | Circuit breaker consecutive losses | `trading_env.py` | NEW | ALTO |

### FASE P0.5 - VOLATILIDAD (Recomendado antes de P1)

| # | Cambio | Archivo | Lineas |
|---|--------|---------|--------|
| P0.5.1 | Hard stop en vol extrema | `trading_env.py` | `_execute_action()` |
| P0.5.2 | Integrar `detect_regime()` | `trading_env.py` | observation |

### FASE P1 - ALTA PRIORIDAD (Despues de validar P0)

| # | Cambio | Archivo |
|---|--------|---------|
| P1.1 | Intratrade DD penalty | `reward_calculator_v20.py` |
| P1.2 | Time decay para posiciones | `reward_calculator_v20.py` |
| P1.3 | Hold bonus condicional a profit | `reward_calculator_v20.py` |
| P1.4 | Volatility regime como feature | `feature_store/core.py` |

### FASE P2 - NICE TO HAVE

| # | Cambio | Archivo |
|---|--------|---------|
| P2.1 | 15-min timeframe | Nuevo dataset builder |
| P2.2 | Walk-forward validation | Training scripts |
| P2.3 | Colombian market features | feature_store |

---

## ARCHIVOS A MODIFICAR - RUTAS EXACTAS

### P0 - CRITICOS

```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\
├── src\training\environments\trading_env.py
│   ├── Linea 143: transaction_cost_bps: 25.0 → 75.0
│   ├── Linea 144: slippage_bps: 2.0 → 15.0
│   ├── Linea 151: threshold_long: 0.10 → 0.33
│   ├── Linea 152: threshold_short: -0.10 → -0.33
│   └── NEW: Agregar consecutive loss tracking + circuit breaker
│
├── src\training\reward_calculator_v20.py
│   ├── Linea 35: loss_penalty_multiplier: 1.5 → 2.0
│   ├── Linea 38: hold_bonus_per_bar: 0.0001 → 0.0 (desactivar V21.0)
│   └── NEW: consecutive_loss_limit, cooldown_bars
│
├── airflow\dags\l3_model_training.py
│   ├── Linea 168: gamma: 0.99 → 0.90
│   ├── Linea 171: ent_coef: 0.01 → 0.05
│   └── Linea 175: transaction_cost_bps: 25.0 → 75.0
│
└── config\v21_config.yaml (YA CREADO - actualizar)
```

### SINCRONIZACION INFERENCE/BACKTEST (CRITICO)

```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\
├── services\inference_api\core\inference_engine.py
│   └── Lineas 50-53: ELIMINAR MODEL_THRESHOLDS hardcoded
│
├── services\inference_api\core\trade_simulator.py
│   └── Lineas 96-99: ELIMINAR MODEL_THRESHOLDS hardcoded
│
├── airflow\dags\services\backtest_factory.py
│   └── Lineas 526-527: Cambiar defaults 0.95 → 0.33
│
├── airflow\dags\l4_backtest_validation.py
│   └── Linea 259: transaction_cost_bps: 5 → 75
│
└── airflow\dags\l5_multi_model_inference.py
    └── Lineas 102-103: Validar que carga de config
```

### P0.5 - VOLATILIDAD

```
C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\
├── src\core\calculators\regime.py (YA EXISTE - usar)
│
└── src\training\environments\trading_env.py
    └── _execute_action(): Agregar hard stop vol extrema
```

---

## CAMBIOS DE CODIGO ESPECIFICOS

### 1. trading_env.py - TradingEnvConfig (COMPLETO)

```python
@dataclass
class TradingEnvConfig:
    # Episode settings
    episode_length: int = 1200
    warmup_bars: int = 14

    # Portfolio settings
    initial_balance: float = 10_000.0
    transaction_cost_bps: float = 75.0  # V21: INCREASED from 25
    slippage_bps: float = 15.0  # V21: INCREASED from 2

    # Risk management
    max_drawdown_pct: float = 15.0
    max_position_duration: int = 288

    # Action thresholds - V21: WIDENED for more HOLD
    threshold_long: float = 0.33  # V21: from 0.10
    threshold_short: float = -0.33  # V21: from -0.10

    # V21 NEW: Circuit breaker
    max_consecutive_losses: int = 5  # Stop after 5 losses
    cooldown_bars_after_losses: int = 12  # 1 hour cooldown

    # V21 NEW: Volatility hard stop
    enable_volatility_filter: bool = True
    max_atr_multiplier: float = 2.0  # Skip trades if ATR > 2x historical

    # Observation settings
    observation_dim: int = 15
    clip_range: Tuple[float, float] = (-5.0, 5.0)
```

### 2. trading_env.py - Circuit Breaker (NUEVO)

Agregar despues de `_update_portfolio()`:

```python
def _update_consecutive_losses(self, pnl: float) -> None:
    """Track consecutive losses for circuit breaker."""
    if not hasattr(self, '_consecutive_losses'):
        self._consecutive_losses = 0
        self._cooldown_until_step = 0

    if pnl < 0:
        self._consecutive_losses += 1
    else:
        self._consecutive_losses = 0

    # Trigger cooldown
    if self._consecutive_losses >= self.config.max_consecutive_losses:
        self._cooldown_until_step = self._step_count + self.config.cooldown_bars_after_losses
        self._consecutive_losses = 0  # Reset counter
        logger.warning(f"Circuit breaker triggered: {self.config.cooldown_bars_after_losses} bar cooldown")

def _is_in_cooldown(self) -> bool:
    """Check if currently in cooldown period."""
    if not hasattr(self, '_cooldown_until_step'):
        return False
    return self._step_count < self._cooldown_until_step
```

Modificar `_execute_action()`:

```python
def _execute_action(self, target_action: TradingAction) -> Tuple[bool, float]:
    # V21: Circuit breaker check
    if self._is_in_cooldown() and target_action != TradingAction.HOLD:
        return False, 0.0  # Force HOLD during cooldown

    # V21: Volatility hard stop
    if self.config.enable_volatility_filter:
        current_atr = self._get_current_atr_pct()
        if current_atr > self.config.max_atr_multiplier * self._historical_atr_mean:
            if target_action != TradingAction.HOLD:
                return False, 0.0  # Force HOLD in extreme volatility

    # ... rest of existing logic
```

### 3. reward_calculator_v20.py - RewardConfig (ACTUALIZADO)

```python
@dataclass
class RewardConfig:
    """Configuration for V21 reward calculator."""
    # Transaction costs
    transaction_cost_pct: float = 0.0002

    # V21: INCREASED asymmetric loss penalty
    loss_penalty_multiplier: float = 2.0  # V21: from 1.5

    # V21: DISABLED hold bonus (was encouraging bad holds)
    hold_bonus_per_bar: float = 0.0  # V21: from 0.0001
    hold_bonus_requires_profit: bool = True  # V21: NEW - only if profitable

    # Consistency bonus
    consecutive_win_bonus: float = 0.001
    max_consecutive_bonus: int = 5

    # Drawdown penalty
    drawdown_penalty_threshold: float = 0.05
    drawdown_penalty_multiplier: float = 2.0

    # V21 NEW: Intratrade drawdown
    intratrade_dd_penalty: float = 0.5
    max_intratrade_dd: float = 0.02

    # V21 NEW: Time decay
    time_decay_start_bars: int = 24
    time_decay_per_bar: float = 0.0001
    time_decay_losing_multiplier: float = 2.0

    # Reward clipping
    min_reward: float = -1.0
    max_reward: float = 1.0
```

### 4. l3_model_training.py - DEFAULT_TRAINING_CONFIG

Modificar lineas 135-190:

```python
DEFAULT_TRAINING_CONFIG = {
    "version": "auto",
    "dataset_name": "RL_DS3_MACRO_CORE.csv",

    # V21 Hyperparameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.90,  # V21: DECREASED from 0.99
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,  # V21: INCREASED from 0.01
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # V21 Trading costs
    "transaction_cost_bps": 75.0,  # V21: from 25
    "slippage_bps": 15.0,  # V21: from 2 (or missing)

    # V21 Thresholds
    "threshold_long": 0.33,  # V21: from 0.10
    "threshold_short": -0.33,  # V21: from -0.10

    # Training settings
    "total_timesteps": 500_000,
    "eval_freq": 25000,
    "n_eval_episodes": 5,
    "checkpoint_freq": 50000,

    # ... rest
}
```

---

## CONFIGURACION AIRFLOW VARIABLE

Para usar V21 sin modificar codigo, crear Airflow Variable:

**Key**: `training_config`
**Value**:
```json
{
  "version": "v21",
  "gamma": 0.90,
  "ent_coef": 0.05,
  "transaction_cost_bps": 75.0,
  "slippage_bps": 15.0,
  "threshold_long": 0.33,
  "threshold_short": -0.33,
  "total_timesteps": 500000
}
```

---

## v21_config.yaml ACTUALIZADO

```yaml
# V21 Configuration - UPDATED based on expert feedback
# Contrato: GTR-008-v2
# NUNCA modificar durante ejecucion.

model:
  name: ppo_v21
  version: "21"
  observation_dim: 15
  action_space: 3

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.05  # INCREASED - more exploration
  clip_range: 0.2
  gamma: 0.90  # DECREASED MORE - very noisy 5-min data
  gae_lambda: 0.95

thresholds:
  long: 0.33  # WIDER HOLD zone
  short: -0.33
  confidence_min: 0.6

features:
  norm_stats_path: "config/v20_norm_stats.json"
  clip_range: [-5.0, 5.0]
  warmup_bars: 14

trading:
  initial_capital: 10000
  transaction_cost_bps: 75  # REALISTIC USDCOP
  slippage_bps: 15
  max_position_size: 1.0

risk:
  max_drawdown_pct: 15.0
  daily_loss_limit_pct: 5.0
  position_limit: 1.0
  volatility_scaling: true
  # V21 NEW: Circuit breaker
  max_consecutive_losses: 5
  cooldown_bars_after_losses: 12

reward:
  # V21: MORE ASYMMETRIC
  loss_penalty_multiplier: 2.0  # from 1.5

  # V21: DISABLED hold bonus (was counterproductive)
  hold_bonus_per_bar: 0.0  # from 0.0001
  hold_bonus_requires_profit: true

  # Consistency
  consecutive_win_bonus: 0.001
  max_consecutive_bonus: 5

  # Drawdown
  drawdown_penalty_threshold: 0.05
  drawdown_penalty_multiplier: 2.0

  # V21 NEW: Intratrade DD
  intratrade_dd_penalty: 0.5
  max_intratrade_dd: 0.02

  # V21 NEW: Time decay
  time_decay_start_bars: 24
  time_decay_per_bar: 0.0001
  time_decay_losing_multiplier: 2.0

  # Clipping
  min_reward: -1.0
  max_reward: 1.0

# V21 NEW: Volatility filter
volatility:
  enable_filter: true
  max_atr_multiplier: 2.0
  force_hold_in_extreme: true

dates:
  training_start: "2020-03-01"
  training_end: "2024-12-31"
  validation_start: "2025-01-01"
  validation_end: "2025-06-30"
  test_start: "2025-07-01"

metadata:
  created: "2026-01-13"
  author: "Trading Team + Expert Review"
  base_version: "v20"
  changes_v2: |
    - gamma: 0.95 -> 0.90 (per expert recommendation)
    - loss_penalty: 1.5 -> 2.0 (more asymmetric)
    - hold_bonus: disabled (was counterproductive)
    - Added circuit breaker (5 consecutive losses)
    - Added volatility hard stop
```

---

## CHECKLIST PRE-ENTRENAMIENTO

### P0 Obligatorios
- [ ] `trading_env.py`: threshold_long = 0.33
- [ ] `trading_env.py`: threshold_short = -0.33
- [ ] `trading_env.py`: transaction_cost_bps = 75.0
- [ ] `trading_env.py`: slippage_bps = 15.0
- [ ] `l3_model_training.py`: gamma = 0.90
- [ ] `l3_model_training.py`: ent_coef = 0.05
- [ ] `reward_calculator_v20.py`: loss_penalty_multiplier = 2.0
- [ ] `reward_calculator_v20.py`: hold_bonus_per_bar = 0.0

### P0.5 Recomendados
- [ ] `trading_env.py`: circuit breaker (5 losses → cooldown)
- [ ] `trading_env.py`: volatility hard stop (ATR > 2x → force HOLD)

### Sincronizacion (CRITICO)
- [ ] `inference_engine.py`: eliminar MODEL_THRESHOLDS hardcoded
- [ ] `trade_simulator.py`: eliminar MODEL_THRESHOLDS hardcoded
- [ ] `backtest_factory.py`: actualizar defaults a 0.33/-0.33
- [ ] `l4_backtest_validation.py`: transaction_cost_bps = 75

---

## CHECKLIST POST-ENTRENAMIENTO

| Metrica | Minimo Aceptable | Objetivo |
|---------|------------------|----------|
| HOLD Rate | > 30% | 35-45% |
| Long/Short Ratio | 0.4 - 0.6 | ~0.5 |
| Avg Trade Duration | < 24 bars | < 18 bars |
| Sharpe OOS | > 0.0 | > 0.3 |
| Max DD OOS | < 30% | < 25% |
| Win Rate | > 35% | > 40% |
| Consecutive Losses Max | < 6 | < 5 |
| **Train vs OOS Sharpe Gap** | **< 1.0** | **< 0.5** |

**IMPORTANTE**: Si Train vs OOS Sharpe Gap > 1.0, hay overfitting severo. Revisar:
- gamma demasiado alto
- features con look-ahead bias
- episodios demasiado cortos

---

## EXPECTATIVAS REALISTAS

| Metrica | V20 Actual | V21 Doc Original | V21 Realista |
|---------|------------|------------------|--------------|
| Sharpe | -5.00 | > 0.5 | 0.2 - 0.5 |
| Max DD | -44.6% | < 25% | 20-30% |
| Win Rate | 13.6% | > 40% | 35-42% |
| HOLD Rate | ~5% | > 30% | 35-45% |

**Nota**: Si Sharpe > 0.3 en OOS es un WIN significativo vs -5.0

---

## ORDEN DE IMPLEMENTACION (OPTIMIZADO)

Siguiendo feedback experto: Sincronizacion PRIMERO para evitar valores diferentes durante transicion.

```
DIA 1 MANANA: Sincronizacion PRIMERO
├── 1. Actualizar v21_config.yaml con valores finales
├── 2. Limpiar MODEL_THRESHOLDS de inference_engine.py
├── 3. Limpiar MODEL_THRESHOLDS de trade_simulator.py
├── 4. Actualizar backtest_factory.py defaults
└── 5. Verificar que todos los servicios leen del mismo lugar

DIA 1 TARDE: P0 Core
├── 6. Actualizar trading_env.py (thresholds, costs, circuit breaker, vol filter)
├── 7. Actualizar l3_model_training.py (gamma, ent_coef)
└── 8. Actualizar reward_calculator_v20.py (loss_penalty, hold_bonus=0)

DIA 2: Validacion
├── 9. Ejecutar test de consistencia: pytest tests/unit/test_v21_config_consistency.py
├── 10. Verificar que NO hay 0.95 hardcoded en ningun archivo
└── 11. Configurar Airflow Variable (backup)

DIA 3: Entrenamiento
├── 12. Ejecutar entrenamiento V21
├── 13. Evaluar en OOS (Jan 2025)
└── 14. Calcular Train vs OOS Sharpe Gap

DIA 4+: Iteracion
├── 15. Si Sharpe < 0.3 → P1 (intratrade DD, time decay)
├── 16. Si Sharpe Gap > 1.0 → Revisar overfitting
└── 17. Si Sharpe < 0.1 → Considerar 15-min timeframe
```

### Test de Consistencia (NUEVO)

Archivo: `tests/unit/test_v21_config_consistency.py`

```bash
pytest tests/unit/test_v21_config_consistency.py -v
```

Este test verifica:
- Todos los servicios usan 0.33/-0.33 thresholds
- No hay 0.95 hardcoded en ningun archivo
- Transaction costs = 75 bps en todos lados
- Circuit breaker configurado
- Volatility filter configurado

---

## ARCHIVOS CREADOS/MODIFICADOS

| Archivo | Accion | Estado |
|---------|--------|--------|
| `V21_IMPLEMENTATION_PLAN_v2.md` | CREAR | Este documento |
| `config/v21_config.yaml` | ACTUALIZAR | Pendiente |
| `src/training/environments/trading_env.py` | MODIFICAR | Pendiente |
| `src/training/reward_calculator_v20.py` | MODIFICAR | Pendiente |
| `airflow/dags/l3_model_training.py` | MODIFICAR | Pendiente |
| `services/inference_api/core/inference_engine.py` | MODIFICAR | Pendiente |
| `services/inference_api/core/trade_simulator.py` | MODIFICAR | Pendiente |
| `airflow/dags/services/backtest_factory.py` | MODIFICAR | Pendiente |
| `airflow/dags/l4_backtest_validation.py` | MODIFICAR | Pendiente |

---

## CONCLUSION

El plan V21 v2.0 incorpora:
1. Hallazgos de 4 agentes exploradores
2. Feedback experto sobre hold_bonus, loss_penalty, circuit breaker
3. Correccion de inconsistencias de thresholds entre training/inference
4. Volatility filter elevado a P0.5
5. Expectativas realistas ajustadas

**Siguiente paso**: Aplicar cambios P0 y sincronizacion, luego entrenar.
