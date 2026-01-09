"""
PPO Configuration V20 - USDCOP Trading
======================================

V20 Configuration with entropy coefficient for exploration.
This prevents the model from learning extreme actions (100% LONG/SHORT, 0% HOLD).

From: 09_Documento Maestro Completo.md Section 6.7

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

# PPO Hyperparameters V20
PPO_CONFIG_V20 = {
    # Learning rate
    "learning_rate": 3e-4,

    # Steps and batching
    "n_steps": 2048,          # Steps per update
    "batch_size": 64,         # Minibatch size
    "n_epochs": 10,           # Epochs per update

    # GAE (Generalized Advantage Estimation)
    "gamma": 0.99,            # Discount factor
    "gae_lambda": 0.95,       # GAE lambda

    # Policy clipping
    "clip_range": 0.2,        # PPO clip range
    "normalize_advantage": True,

    # V20 CRITICAL FIX: Entropy coefficient for exploration
    # Without this, model learns extreme actions (0% HOLD signals)
    # Value 0.01 encourages exploration while maintaining learning stability
    "ent_coef": 0.01,

    # Value function coefficient
    "vf_coef": 0.5,

    # Gradient clipping
    "max_grad_norm": 0.5,

    # KL divergence target (optional early stopping)
    "target_kl": 0.015,
}

# Policy network architecture
POLICY_KWARGS_V20 = {
    "net_arch": {
        "pi": [256, 256],   # Actor (policy) network
        "vf": [256, 256],   # Critic (value) network
    },
}

# Training configuration
TRAINING_CONFIG_V20 = {
    "total_timesteps": 5_000_000,    # 5M steps
    "eval_freq": 50_000,              # Evaluate every 50K steps
    "n_eval_episodes": 10,            # Episodes per evaluation
    "save_freq": 100_000,             # Save checkpoint every 100K steps
    "log_interval": 10,               # Log every 10 updates
}

# V20 Action thresholds (must match production)
ACTION_CONFIG_V20 = {
    "threshold_long": 0.10,           # Action > 0.10 = LONG
    "threshold_short": -0.10,         # Action < -0.10 = SHORT
    "threshold_hold_zone": 0.10,      # |action| <= 0.10 = HOLD
}


def get_ppo_config():
    """Get PPO V20 configuration."""
    return PPO_CONFIG_V20.copy()


def get_policy_kwargs():
    """Get policy network architecture."""
    return POLICY_KWARGS_V20.copy()


def get_training_config():
    """Get training configuration."""
    return TRAINING_CONFIG_V20.copy()


if __name__ == "__main__":
    print("PPO V20 Configuration:")
    print(f"  Learning rate: {PPO_CONFIG_V20['learning_rate']}")
    print(f"  Entropy coef: {PPO_CONFIG_V20['ent_coef']} (CRITICAL for exploration)")
    print(f"  Total timesteps: {TRAINING_CONFIG_V20['total_timesteps']:,}")
    print(f"  Action thresholds: {ACTION_CONFIG_V20}")
