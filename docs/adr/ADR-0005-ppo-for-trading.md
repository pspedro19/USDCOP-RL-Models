# ADR-0005: Use PPO Algorithm for Trading Signal Generation

## Status

**Accepted** (2026-01-14)

## Context

The USDCOP RL Trading system requires a reinforcement learning algorithm to generate trading signals (LONG, SHORT, HOLD) based on market observations.

Requirements:
- Stable training on noisy financial data
- Continuous action space (for confidence/position sizing)
- Sample efficiency (limited historical data)
- Interpretable policy behavior
- Production-ready implementation

Options considered:
1. PPO (Proximal Policy Optimization)
2. SAC (Soft Actor-Critic)
3. TD3 (Twin Delayed DDPG)
4. A2C (Advantage Actor-Critic)
5. DQN (Deep Q-Network)

## Decision

We chose **PPO (Proximal Policy Optimization)** for trading signal generation.

## Rationale

### Why PPO over alternatives:

| Criterion | PPO | SAC | TD3 | A2C | DQN |
|-----------|-----|-----|-----|-----|-----|
| Stability | Excellent | Good | Good | Moderate | Moderate |
| Sample Efficiency | Good | Excellent | Good | Poor | Moderate |
| Continuous Actions | Yes | Yes | Yes | Yes | No |
| Hyperparameter Sensitivity | Low | Medium | Medium | High | High |
| Implementation Maturity | Excellent | Good | Good | Good | Excellent |
| Financial Data Suitability | Excellent | Good | Moderate | Moderate | Poor |

### Key advantages:

1. **Stability** - Clipped objective prevents catastrophic updates
2. **Robustness** - Less sensitive to hyperparameters than alternatives
3. **Trust Region** - Natural fit for risk-averse trading
4. **Entropy Bonus** - Encourages exploration, prevents mode collapse
5. **Battle-tested** - Used successfully in many trading applications

### Why not SAC?

- Higher sample efficiency but less stable on noisy data
- More hyperparameters to tune
- Entropy maximization can lead to over-trading

### Why not TD3?

- Requires more tuning for discrete-like trading decisions
- Deterministic policy less suitable for stochastic markets
- No built-in exploration mechanism

### Why not DQN?

- Discrete action space only
- Would require discretizing position sizes
- Less suitable for continuous trading decisions

## Consequences

### Positive

- Stable training across different market regimes
- Excellent Stable-Baselines3 implementation
- Well-documented hyperparameter tuning guides
- Natural handling of action thresholds

### Negative

- Not the most sample-efficient algorithm
- On-policy (can't use historical replay efficiently)
- May require curriculum learning for complex environments

### Mitigations

- Use larger datasets (84K samples) to compensate for sample efficiency
- Implement curriculum learning for reward shaping
- Monitor action entropy for mode collapse detection

## Implementation

### Configuration (config/trading_config.yaml)

```yaml
ppo:
  learning_rate: 0.0001
  n_steps: 2048
  batch_size: 128
  n_epochs: 10
  gamma: 0.90  # Shorter-term focus for noisy 5-min data
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.05  # Higher entropy for exploration
```

### Action Mapping

```python
# Continuous action [-1, 1] mapped to discrete signals
if action > 0.33:     # Top 33%
    signal = "LONG"
elif action < -0.33:  # Bottom 33%
    signal = "SHORT"
else:                 # Middle 33%
    signal = "HOLD"
```

### Key Design Decisions

1. **Gamma = 0.90** (not 0.99)
   - Financial data is noisy at 5-minute frequency
   - Shorter horizon reduces variance in value estimates

2. **Entropy Coefficient = 0.05**
   - Higher than default (0.01) to prevent mode collapse
   - Encourages exploration of LONG/SHORT actions

3. **Wide HOLD Zone (33%)**
   - Reduces transaction costs from over-trading
   - Matches real-world trading where most time is spent waiting

## References

- [PPO Paper (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/1911.10107)
