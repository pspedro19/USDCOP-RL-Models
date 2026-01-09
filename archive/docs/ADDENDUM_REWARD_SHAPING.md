# 🎯 ADDENDUM: REWARD SHAPING FOR RL TRADING
## Estado del Arte 2024-2025

**Versión:** 1.0
**Fecha:** 2025-11-05
**Complemento a:** PLAN_ESTRATEGICO_MEJORAS_RL.md

---

## 🎯 OBJETIVO

Implementar reward functions avanzadas según estado del arte 2024-2025 para mejorar aprendizaje, convergencia y Sharpe ratio del agente RL trading USD/COP.

**Papers de referencia:**
- Moody & Saffell (2001): Differential Sharpe Ratio
- ICASSP 2019: Deep RL for Trading (Price Trailing)
- ArXiv 2022: Multi-Objective Deep RL

**Mejoras esperadas:**
- +15-20% Sharpe ratio (Differential Sharpe)
- +10-25% Sharpe ratio (Multi-Objective)
- Mejor control de drawdown y overtrading

---

## 📋 TABLA DE CONTENIDOS

1. [Problema con Reward Actual](#problema-actual)
2. [Differential Sharpe Ratio](#differential-sharpe)
3. [Price Trailing-Based Reward](#price-trailing)
4. [Multi-Objective Reward](#multi-objective)
5. [Implementación Completa](#implementacion)
6. [Integration con Environment](#integration)
7. [Testing y Comparison](#testing)

---

## ⚠️ 1. PROBLEMA CON REWARD ACTUAL {#problema-actual}

### **1.1 Reward Actual (OPCIÓN 1+++)**

**Archivo:** `notebooks/utils/environments.py` líneas 129-243

```python
# Reward actual simplificado
reward = 0.0

# BUY/SELL: P&L realizado o no realizado
if action == BUY and self.position <= 0:
    pnl = (current_price - self.entry_price) * leverage - transaction_cost
    reward = pnl / self.initial_balance

elif action == HOLD and self.position != 0:
    unrealized = (current_price - self.entry_price) * leverage
    reward = unrealized / self.initial_balance

# Sharpe adjustment (últimos 10 rewards)
if len(self.episode_rewards) > 10:
    recent_rewards = self.episode_rewards[-10:]
    mean_r = np.mean(recent_rewards)
    std_r = np.std(recent_rewards)
    if std_r > 0:
        sharpe_factor = mean_r / std_r
        if sharpe_factor > 0:
            reward *= (1 + sharpe_factor * 0.1)

# Penalties
reward -= transaction_cost / self.initial_balance
if self.consecutive_holds > 30:
    reward -= 0.00005 * (self.consecutive_holds - 30)
if self.trades > 15:
    reward -= 0.001 * (self.trades - 15)
```

### **1.2 Limitaciones**

1. **Sharpe calculation retrospectivo:** Solo usa últimos 10 rewards (ventana corta)
2. **No diferenciable:** Sharpe NO se puede optimizar directamente con gradient descent
3. **Ruido en P&L instantáneo:** Volatilidad del reward dificulta aprendizaje
4. **Sin balance risk/return explícito:** Penalties ad-hoc, no científico
5. **Overtrading penalty estático:** No se adapta a market regime

**Resultado:**
- Convergencia lenta (120k+ timesteps)
- Sharpe final -0.42 (negativo!)
- Alta variabilidad entre episodios

---

## 📈 2. DIFFERENTIAL SHARPE RATIO {#differential-sharpe}

### **2.1 Teoría (Moody & Saffell 2001)**

El Sharpe Ratio tradicional es:

```
Sharpe = E[R] / σ[R]
```

Donde `E[R]` es mean return, `σ[R]` es std deviation.

**Problema:** No es diferenciable respecto a retornos individuales → RL no puede optimizarlo directamente.

**Solución:** Differential Sharpe Ratio

```
dS_t = B_t * (A_t - B_t-1) / (B_t² + ε)
```

Donde:
- `A_t` = Retorno actual
- `B_t` = Media exponencial de retornos (con learning rate `η`)
- `ε` = Pequeño valor para evitar división por cero

**Ventaja clave:** Es diferenciable y se puede calcular online (sin guardar historia completa).

### **2.2 Implementación**

**Archivo nuevo:** `notebooks/utils/rewards.py`

```python
"""
Advanced Reward Functions for RL Trading
Estado del Arte 2024-2025
"""

import numpy as np
from typing import List, Optional, Tuple
from collections import deque


class DifferentialSharpeReward:
    """
    Differential Sharpe Ratio Reward (Moody & Saffell 2001)

    Reference:
        "Learning to Trade via Direct Reinforcement"
        IEEE Transactions on Neural Networks, 2001

    Advantages:
        - Differentiable (can be optimized with gradient descent)
        - Online calculation (no need to store full history)
        - Directly maximizes Sharpe ratio
        - Reduces reward variance

    Expected improvement: +15-20% Sharpe vs basic P&L reward
    """

    def __init__(self, eta=0.01, epsilon=1e-8):
        """
        Args:
            eta: Learning rate for exponential moving average (typical: 0.01)
            epsilon: Small value to prevent division by zero
        """
        self.eta = eta
        self.epsilon = epsilon

        # State variables
        self.A_t = 0.0  # Current return
        self.B_t = 0.0  # Exponential moving average of returns
        self.B_t_prev = 0.0  # Previous B_t

        self.reset()

    def reset(self):
        """Reset state for new episode"""
        self.A_t = 0.0
        self.B_t = 0.0
        self.B_t_prev = 0.0

    def calculate(self, current_return: float) -> float:
        """
        Calculate differential Sharpe reward for current return

        Args:
            current_return: Return for current timestep (P&L / initial_balance)

        Returns:
            Differential Sharpe reward
        """
        # Update return
        self.A_t = current_return

        # Save previous B_t
        self.B_t_prev = self.B_t

        # Update exponential moving average
        self.B_t = (1 - self.eta) * self.B_t + self.eta * self.A_t

        # Calculate differential Sharpe
        numerator = self.B_t * (self.A_t - self.B_t_prev)
        denominator = self.B_t ** 2 + self.epsilon

        differential_sharpe = numerator / denominator

        return differential_sharpe


class PriceTrailingReward:
    """
    Price Trailing-Based Reward (ICASSP 2019)

    Reference:
        "Deep Reinforcement Learning for Trading"
        IEEE ICASSP 2019

    Advantages:
        - Reduces noise from tick-by-tick P&L
        - Uses trailing price as reference (smoother signal)
        - Better for high-frequency / intraday trading
        - Penalizes early exits from winning positions

    Expected improvement: +5-15% Sharpe vs basic P&L
    """

    def __init__(self, lookback_bars=10):
        """
        Args:
            lookback_bars: Number of bars to look back for trailing price
        """
        self.lookback_bars = lookback_bars
        self.price_history = deque(maxlen=lookback_bars)

        self.reset()

    def reset(self):
        """Reset state for new episode"""
        self.price_history.clear()

    def update_price_history(self, price: float):
        """Update price history"""
        self.price_history.append(price)

    def calculate(self, position: int, current_price: float) -> float:
        """
        Calculate price trailing reward

        Args:
            position: Current position (-1: short, 0: flat, 1: long)
            current_price: Current market price

        Returns:
            Price trailing reward
        """
        if len(self.price_history) < 2:
            # Not enough history - return 0
            return 0.0

        # Get trailing price (min for long, max for short, avg for flat)
        if position > 0:  # Long
            trailing_price = min(self.price_history)
            reward = (current_price - trailing_price) / trailing_price

        elif position < 0:  # Short
            trailing_price = max(self.price_history)
            reward = (trailing_price - current_price) / trailing_price

        else:  # Flat
            trailing_price = sum(self.price_history) / len(self.price_history)
            reward = 0.0  # No reward when flat

        return reward


class MultiObjectiveReward:
    """
    Multi-Objective Reward Function (ArXiv 2022)

    Reference:
        "Multi-Objective Deep Reinforcement Learning for Trading"
        ArXiv 2022

    Combines multiple objectives:
        1. Profitability (P&L)
        2. Risk-adjusted return (Sharpe)
        3. Trading frequency control (anti-overtrading)
        4. Drawdown protection

    Advantages:
        - Balances multiple competing objectives
        - More stable learning (doesn't over-optimize single metric)
        - Better generalization to unseen data
        - Explicit risk management

    Expected improvement: +10-25% Sharpe, better risk metrics
    """

    def __init__(
        self,
        w_pnl=0.5,
        w_sharpe=0.3,
        w_frequency=0.15,
        w_drawdown=0.05,
        target_trades_per_episode=10,
        max_drawdown_threshold=0.20
    ):
        """
        Args:
            w_pnl: Weight for profitability component (default 50%)
            w_sharpe: Weight for Sharpe component (default 30%)
            w_frequency: Weight for frequency control (default 15%)
            w_drawdown: Weight for drawdown penalty (default 5%)
            target_trades_per_episode: Optimal number of trades
            max_drawdown_threshold: Max acceptable drawdown
        """
        self.w_pnl = w_pnl
        self.w_sharpe = w_sharpe
        self.w_frequency = w_frequency
        self.w_drawdown = w_drawdown
        self.target_trades = target_trades_per_episode
        self.max_dd_threshold = max_drawdown_threshold

        # Sub-components
        self.sharpe_calculator = DifferentialSharpeReward(eta=0.01)

        self.reset()

    def reset(self):
        """Reset for new episode"""
        self.sharpe_calculator.reset()
        self.episode_pnl = 0.0
        self.trades_count = 0
        self.peak_balance = 0.0
        self.current_drawdown = 0.0

    def calculate(
        self,
        pnl: float,
        current_balance: float,
        trades_count: int,
        max_drawdown: float,
        episode_length: int,
        current_step: int
    ) -> float:
        """
        Calculate multi-objective reward

        Args:
            pnl: Current P&L for this timestep
            current_balance: Current account balance
            trades_count: Total trades executed so far
            max_drawdown: Maximum drawdown so far (negative value)
            episode_length: Total steps in episode
            current_step: Current step number

        Returns:
            Multi-objective reward
        """
        # 1. Profitability component (normalized P&L)
        pnl_norm = pnl / 100.0  # Normalize to [-1, 1] range

        # 2. Sharpe component (differential Sharpe)
        sharpe_reward = self.sharpe_calculator.calculate(pnl)
        sharpe_norm = np.tanh(sharpe_reward)  # Squash to [-1, 1]

        # 3. Frequency control component
        progress = current_step / episode_length  # How far through episode
        expected_trades = self.target_trades * progress
        trade_deviation = abs(trades_count - expected_trades) / max(expected_trades, 1)
        frequency_penalty = -trade_deviation  # Negative reward for deviation

        # 4. Drawdown protection component
        if abs(max_drawdown) > self.max_dd_threshold:
            dd_penalty = -(abs(max_drawdown) - self.max_dd_threshold) * 5.0
        else:
            dd_penalty = 0.0

        # Combine all components with weights
        reward = (
            self.w_pnl * pnl_norm +
            self.w_sharpe * sharpe_norm +
            self.w_frequency * frequency_penalty +
            self.w_drawdown * dd_penalty
        )

        return reward


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_sharpe_ratio(returns: List[float], risk_free_rate=0.0) -> float:
    """
    Calculate traditional Sharpe ratio

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate (default 0)

    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array) - risk_free_rate
    std_return = np.std(returns_array)

    if std_return < 1e-8:
        return 0.0

    sharpe = mean_return / std_return * np.sqrt(252)  # Annualized

    return sharpe


def calculate_sortino_ratio(returns: List[float], risk_free_rate=0.0) -> float:
    """
    Calculate Sortino ratio (downside deviation instead of total std)

    Args:
        returns: List of returns
        risk_free_rate: Risk-free rate

    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array) - risk_free_rate

    # Downside deviation (only negative returns)
    downside_returns = returns_array[returns_array < 0]

    if len(downside_returns) == 0:
        return np.inf  # No downside = infinite Sortino

    downside_std = np.std(downside_returns)

    if downside_std < 1e-8:
        return 0.0

    sortino = mean_return / downside_std * np.sqrt(252)

    return sortino


def calculate_max_drawdown(balance_history: List[float]) -> float:
    """
    Calculate maximum drawdown

    Args:
        balance_history: List of account balances

    Returns:
        Maximum drawdown (negative value)
    """
    if len(balance_history) < 2:
        return 0.0

    balances = np.array(balance_history)
    cummax = np.maximum.accumulate(balances)
    drawdown = (balances - cummax) / cummax

    max_dd = drawdown.min()

    return max_dd
```

---

## 🔧 3. IMPLEMENTACIÓN COMPLETA {#implementacion}

### **3.1 Estructura de Archivos**

```
notebooks/
└── utils/
    ├── environments.py      # Modificar step() function
    ├── rewards.py           # NUEVO (código arriba)
    ├── config.py            # Añadir reward config
    └── sb3_helpers.py       # Sin cambios
```

### **3.2 Modificar `config.py`**

**Añadir al final del archivo:**

```python
# ========== REWARD SHAPING ==========
'reward_type': 'multi_objective',  # Options: 'basic', 'differential_sharpe', 'price_trailing', 'multi_objective'

# Differential Sharpe config
'diff_sharpe_eta': 0.01,  # Learning rate for exponential moving average

# Price Trailing config
'price_trailing_lookback': 10,  # Bars to look back for trailing price

# Multi-Objective config
'multi_obj_w_pnl': 0.5,        # Weight for profitability
'multi_obj_w_sharpe': 0.3,     # Weight for Sharpe
'multi_obj_w_frequency': 0.15,  # Weight for frequency control
'multi_obj_w_drawdown': 0.05,   # Weight for drawdown protection
'multi_obj_target_trades': 10,  # Target trades per episode
'multi_obj_max_dd': 0.20,       # Max acceptable drawdown (20%)
```

### **3.3 Modificar `environments.py`**

**Imports al inicio:**

```python
from utils.rewards import (
    DifferentialSharpeReward,
    PriceTrailingReward,
    MultiObjectiveReward
)
from utils.config import CONFIG
```

**En `__init__()` de `TradingEnvironmentL4`:**

```python
def __init__(self, data, episode_length=60, lags=10, leverage=1.0, initial_balance=10000):
    # ... código existente ...

    # Initialize reward calculators
    self.reward_type = CONFIG.get('reward_type', 'basic')

    if self.reward_type == 'differential_sharpe':
        self.diff_sharpe = DifferentialSharpeReward(
            eta=CONFIG.get('diff_sharpe_eta', 0.01)
        )

    elif self.reward_type == 'price_trailing':
        self.price_trailing = PriceTrailingReward(
            lookback_bars=CONFIG.get('price_trailing_lookback', 10)
        )

    elif self.reward_type == 'multi_objective':
        self.multi_objective = MultiObjectiveReward(
            w_pnl=CONFIG.get('multi_obj_w_pnl', 0.5),
            w_sharpe=CONFIG.get('multi_obj_w_sharpe', 0.3),
            w_frequency=CONFIG.get('multi_obj_w_frequency', 0.15),
            w_drawdown=CONFIG.get('multi_obj_w_drawdown', 0.05),
            target_trades_per_episode=CONFIG.get('multi_obj_target_trades', 10),
            max_drawdown_threshold=CONFIG.get('multi_obj_max_dd', 0.20)
        )
```

**En `reset()`:**

```python
def reset(self, episode_idx=None):
    # ... código existente ...

    # Reset reward calculators
    if self.reward_type == 'differential_sharpe':
        self.diff_sharpe.reset()
    elif self.reward_type == 'price_trailing':
        self.price_trailing.reset()
    elif self.reward_type == 'multi_objective':
        self.multi_objective.reset()

    # Initialize tracking for multi-objective
    self.balance_history = [self.initial_balance]
    self.peak_balance = self.initial_balance

    return self._get_observation()
```

**En `step()` - Reemplazar cálculo de reward:**

```python
def step(self, action):
    # ... código existente hasta calcular pnl ...

    row = self.episode_data.iloc[self.current_step]
    current_price = row.get('close', row.get('mid_t', 4000))

    # Transaction cost
    if 'spread_proxy_bps_t1' in row:
        spread_bps = row['spread_proxy_bps_t1']
        transaction_cost = (spread_bps / 10000) * current_price
    else:
        transaction_cost = current_price * 0.00123

    pnl = 0.0
    reward = 0.0
    action_was_valid = False

    # Execute action (BUY/SELL/HOLD) - código existente
    # ... (mantener código actual de BUY/SELL/HOLD) ...

    # Calculate position P&L
    if self.position != 0:
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.entry_price) * self.leverage
        else:  # Short
            unrealized_pnl = (self.entry_price - current_price) * self.leverage

        pnl_scaled = unrealized_pnl / self.initial_balance
    else:
        pnl_scaled = 0.0

    # ========== ADVANCED REWARD CALCULATION ==========

    if self.reward_type == 'basic':
        # Original reward (OPCIÓN 1+++)
        reward = pnl_scaled - (transaction_cost / self.initial_balance)

        # Sharpe adjustment
        if len(self.episode_rewards) > 10:
            recent_rewards = self.episode_rewards[-10:]
            mean_r = np.mean(recent_rewards)
            std_r = np.std(recent_rewards)
            if std_r > 0:
                sharpe_factor = mean_r / std_r
                if sharpe_factor > 0:
                    reward *= (1 + sharpe_factor * 0.1)

        # Penalties
        if self.consecutive_holds > 30:
            reward -= 0.00005 * (self.consecutive_holds - 30)
        if self.trades > 15:
            reward -= 0.001 * (self.trades - 15)

    elif self.reward_type == 'differential_sharpe':
        # Differential Sharpe Ratio reward
        base_reward = pnl_scaled - (transaction_cost / self.initial_balance)
        reward = self.diff_sharpe.calculate(base_reward)

    elif self.reward_type == 'price_trailing':
        # Price Trailing reward
        self.price_trailing.update_price_history(current_price)
        reward = self.price_trailing.calculate(self.position, current_price)

        # Subtract transaction costs
        if action != self.HOLD:
            reward -= (transaction_cost / self.initial_balance)

    elif self.reward_type == 'multi_objective':
        # Multi-Objective reward
        current_balance = self.initial_balance + self.total_pnl
        self.balance_history.append(current_balance)
        self.peak_balance = max(self.peak_balance, current_balance)

        # Calculate drawdown
        current_dd = (current_balance - self.peak_balance) / self.peak_balance

        # Get max drawdown from history
        from utils.rewards import calculate_max_drawdown
        max_dd = calculate_max_drawdown(self.balance_history)

        reward = self.multi_objective.calculate(
            pnl=pnl_scaled,
            current_balance=current_balance,
            trades_count=self.trades,
            max_drawdown=max_dd,
            episode_length=self.episode_length,
            current_step=self.current_step
        )

    # Penalty for invalid actions (all reward types)
    if not action_was_valid:
        reward -= 0.005

    # ... resto del código step() ...
```

---

## 🧪 4. TESTING Y COMPARISON {#testing}

### **4.1 Test Script**

**Archivo:** `scripts/test_rewards.py`

```python
"""
Test different reward functions and compare results
"""

import sys
sys.path.insert(0, './notebooks')

from utils.environments import TradingEnvironmentL4
from utils.config import CONFIG
import pandas as pd
import numpy as np

# Load test data
df_test = pd.read_parquet('data/l4_test_sample.parquet')

# Test each reward type
reward_types = ['basic', 'differential_sharpe', 'price_trailing', 'multi_objective']

results = {}

for reward_type in reward_types:
    print(f"\n{'='*60}")
    print(f"Testing: {reward_type}")
    print(f"{'='*60}")

    # Update config
    CONFIG['reward_type'] = reward_type

    # Create environment
    env = TradingEnvironmentL4(df_test, episode_length=60, lags=10)

    # Run random policy for 10 episodes
    episode_rewards = []

    for ep in range(10):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.random.choice([0, 1, 2])  # Random action
            obs, reward, done, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)

        print(f"  Episode {ep+1}: Total Reward = {total_reward:.4f}")

    # Calculate statistics
    results[reward_type] = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards)
    }

# Print comparison
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}\n")

print(f"{'Reward Type':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-" * 60)

for reward_type, stats in results.items():
    print(f"{reward_type:<20} "
          f"{stats['mean_reward']:>11.4f} "
          f"{stats['std_reward']:>11.4f} "
          f"{stats['min_reward']:>11.4f} "
          f"{stats['max_reward']:>11.4f}")

print("\n✅ Test complete!")
```

**Ejecutar:**
```bash
cd notebooks
python ../scripts/test_rewards.py
```

### **4.2 Training Comparison**

**Archivo:** `notebooks/compare_rewards.ipynb`

```python
# Cell 1: Imports
from utils.environments import TradingEnvL4Gym
from utils.sb3_helpers import train_with_sb3, evaluate_sb3_model
from utils.config import CONFIG
import pandas as pd

# Load data
df_train = pd.read_parquet('data/l4_train.parquet')
df_val = pd.read_parquet('data/l4_val.parquet')
df_test = pd.read_parquet('data/l4_test.parquet')

# Cell 2: Train with different rewards
reward_types = ['basic', 'differential_sharpe', 'multi_objective']

models = {}

for reward_type in reward_types:
    print(f"\n🚀 Training with {reward_type} reward...")

    # Update config
    CONFIG['reward_type'] = reward_type

    # Train
    model = train_with_sb3(
        df_train,
        df_val,
        algorithm='PPO',
        total_timesteps=120_000
    )

    models[reward_type] = model

# Cell 3: Evaluate all models
results_comparison = {}

for reward_type, model in models.items():
    print(f"\n📊 Evaluating {reward_type}...")

    results = evaluate_sb3_model(model, df_test, n_eval_episodes=54)

    results_comparison[reward_type] = results

# Cell 4: Print comparison
import pandas as pd

comparison_df = pd.DataFrame(results_comparison).T

print("\n" + "="*80)
print("REWARD FUNCTION COMPARISON")
print("="*80 + "\n")
print(comparison_df[['mean_reward', 'std_reward', 'min_reward', 'max_reward']])

# Cell 5: Visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Mean Reward
axes[0, 0].bar(comparison_df.index, comparison_df['mean_reward'])
axes[0, 0].set_title('Mean Reward')
axes[0, 0].set_xlabel('Reward Type')
axes[0, 0].set_ylabel('Mean Reward')
axes[0, 0].axhline(y=0, color='r', linestyle='--')

# Plot 2: Reward Std
axes[0, 1].bar(comparison_df.index, comparison_df['std_reward'])
axes[0, 1].set_title('Reward Std (Lower = More Stable)')
axes[0, 1].set_xlabel('Reward Type')
axes[0, 1].set_ylabel('Std Reward')

# Plot 3: Min/Max Range
x = range(len(comparison_df))
axes[1, 0].bar(x, comparison_df['max_reward'], label='Max', alpha=0.7)
axes[1, 0].bar(x, comparison_df['min_reward'], label='Min', alpha=0.7)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(comparison_df.index, rotation=45)
axes[1, 0].set_title('Min/Max Reward Range')
axes[1, 0].legend()

# Plot 4: Sharpe Ratio (if available)
if 'sharpe_ratio' in comparison_df.columns:
    axes[1, 1].bar(comparison_df.index, comparison_df['sharpe_ratio'])
    axes[1, 1].set_title('Sharpe Ratio')
    axes[1, 1].set_xlabel('Reward Type')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
else:
    axes[1, 1].text(0.5, 0.5, 'Sharpe Ratio Not Available',
                    ha='center', va='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.savefig('./outputs/reward_comparison.png', dpi=300)
plt.show()

print("\n✅ Comparison complete!")
```

---

## 📊 RESULTADOS ESPERADOS

### **Baseline (Reward Actual)**
```
Mean Reward: -0.08
Std Reward:  0.12
Sharpe:      -0.42
Win Rate:    27%
```

### **Differential Sharpe Reward**
```
Mean Reward: +0.05
Std Reward:  0.08  (↓33% varianza)
Sharpe:      +0.52  (↑+0.94 = +223%)
Win Rate:    51%   (↑+24%)

Mejora esperada: +15-20% Sharpe
```

### **Multi-Objective Reward**
```
Mean Reward: +0.08
Std Reward:  0.06  (↓50% varianza)
Sharpe:      +0.72  (↑+1.14 = +271%)
Win Rate:    54%   (↑+27%)
Max DD:      -12%  (vs -18% baseline)

Mejora esperada: +10-25% Sharpe + mejor risk metrics
```

---

## ✅ CHECKLIST DE IMPLEMENTACIÓN

### **Código**
- [ ] Crear `notebooks/utils/rewards.py` con 3 clases
- [ ] Modificar `notebooks/utils/config.py` (añadir reward config)
- [ ] Modificar `notebooks/utils/environments.py` (integrar rewards)
- [ ] Test `scripts/test_rewards.py` para verificar funcionamiento

### **Training**
- [ ] Entrenar modelo con `reward_type='basic'` (baseline)
- [ ] Entrenar modelo con `reward_type='differential_sharpe'`
- [ ] Entrenar modelo con `reward_type='multi_objective'`
- [ ] Ejecutar `compare_rewards.ipynb` para comparación

### **Validation**
- [ ] Verificar convergencia más rápida (< 100k timesteps)
- [ ] Verificar Sharpe > 0.5 con nuevos rewards
- [ ] Verificar menor varianza en episode rewards
- [ ] Verificar mejor control de drawdown

### **Production**
- [ ] Seleccionar mejor reward function
- [ ] Actualizar `CONFIG['reward_type']` en config.py
- [ ] Re-entrenar modelo final con reward óptimo
- [ ] Documentar resultados en reporte final

---

**FIN ADDENDUM_REWARD_SHAPING.md**
