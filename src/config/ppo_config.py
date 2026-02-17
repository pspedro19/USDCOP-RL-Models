"""
PPO Configuration - USDCOP Trading
==================================

Configuration with entropy coefficient for exploration.
This prevents the model from learning extreme actions (100% LONG/SHORT, 0% HOLD).

From: 09_Documento Maestro Completo.md Section 6.7

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

# PPO Hyperparameters
PPO_CONFIG = {
    # Learning rate
    "learning_rate": 3e-4,

    # Steps and batching
    "n_steps": 2048,          # Steps per update
    "batch_size": 64,         # Minibatch size
    "n_epochs": 10,           # Epochs per update

    # GAE (Generalized Advantage Estimation)
    "gamma": 0.95,            # SSOT: ~20-step horizon for 5-min FX
    "gae_lambda": 0.95,       # GAE lambda

    # Policy clipping
    "clip_range": 0.2,        # PPO clip range
    "normalize_advantage": True,

    # Entropy coefficient for exploration
    # Increased for more exploration, prevents converging too quickly to extreme actions
    "ent_coef": 0.05,

    # Value function coefficient
    "vf_coef": 0.5,

    # Gradient clipping
    "max_grad_norm": 0.5,

    # KL divergence target (optional early stopping)
    "target_kl": 0.015,
}

# Policy network architecture
POLICY_KWARGS = {
    "net_arch": {
        "pi": [256, 256],   # Actor (policy) network
        "vf": [256, 256],   # Critic (value) network
    },
}

# Training configuration
TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,    # 5M steps
    "eval_freq": 50_000,              # Evaluate every 50K steps
    "n_eval_episodes": 10,            # Episodes per evaluation
    "save_freq": 100_000,             # Save checkpoint every 100K steps
    "log_interval": 10,               # Log every 10 updates
}

# Action thresholds (SSOT: experiment_ssot.yaml)
ACTION_CONFIG = {
    "threshold_long": 0.50,           # SSOT: Action > 0.50 = LONG
    "threshold_short": -0.50,         # SSOT: Action < -0.50 = SHORT
    "threshold_hold_zone": 0.50,      # SSOT: |action| <= 0.50 = HOLD
}


def get_ppo_config():
    """Get PPO configuration."""
    return PPO_CONFIG.copy()


def get_policy_kwargs():
    """Get policy network architecture."""
    return POLICY_KWARGS.copy()


def get_training_config():
    """Get training configuration."""
    return TRAINING_CONFIG.copy()


if __name__ == "__main__":
    print("PPO Configuration:")
    print(f"  Learning rate: {PPO_CONFIG['learning_rate']}")
    print(f"  Entropy coef: {PPO_CONFIG['ent_coef']} (CRITICAL for exploration)")
    print(f"  Total timesteps: {TRAINING_CONFIG['total_timesteps']:,}")
    print(f"  Action thresholds: {ACTION_CONFIG}")
