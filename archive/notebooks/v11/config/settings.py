"""
USD/COP RL Trading System V11 - Configuration
==============================================

All configuration parameters for the V11 system.
"""

from pathlib import Path
import torch.nn as nn

# ==============================================================================
# PATHS
# ==============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR.parent.parent / "data" / "PASS" / "OUTPUT_RL" / "RL_DS3_MACRO_CORE.csv"

LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ==============================================================================
# FEATURES
# ==============================================================================

FEATURES_FOR_MODEL = [
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    'rsi_9', 'atr_pct', 'adx_14', 'bb_position',
    'dxy_z', 'dxy_change_1d', 'dxy_mom_5d',
    'vix_z', 'vix_regime', 'embi_z',
    'brent_change_1d', 'brent_vol_5d',
    'rate_spread', 'usdmxn_ret_1h',
    'hour_sin', 'hour_cos'
]

# V11 FIX: Column for raw returns (not normalized)
RAW_RETURN_COL = '_raw_ret_5m'

# ==============================================================================
# TRADING PARAMETERS
# ==============================================================================

COST_PER_TRADE = 0.0003  # 3 bps per trade

CONFIG = {
    'initial_balance': 10_000,
    'episode_length': 288,      # 1 day of 5-min bars
    'embargo_days': 21,         # Gap between train and test
    'timesteps_per_fold': 300_000,
}

# ==============================================================================
# PPO HYPERPARAMETERS
# ==============================================================================

PPO_CONFIG = {
    'learning_rate': 3e-4,
    'n_steps': 2048,
    'batch_size': 256,
    'n_epochs': 5,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.05,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'policy_kwargs': {
        'net_arch': dict(pi=[64, 64], vf=[64, 64]),
        'activation_fn': nn.Tanh,
    },
}

# ==============================================================================
# WALK-FORWARD FOLDS
# ==============================================================================

WALK_FORWARD_FOLDS = [
    {
        'fold_id': 0,
        'train_start': '2020-01-01',
        'train_end': '2021-06-30',
        'test_start': '2021-07-21',
        'test_end': '2021-12-31'
    },
    {
        'fold_id': 1,
        'train_start': '2020-04-01',
        'train_end': '2022-03-31',
        'test_start': '2022-04-21',
        'test_end': '2022-10-31'
    },
    {
        'fold_id': 2,
        'train_start': '2020-07-01',
        'train_end': '2022-10-31',
        'test_start': '2022-11-21',
        'test_end': '2023-05-31'
    },
    {
        'fold_id': 3,
        'train_start': '2021-07-01',
        'train_end': '2023-06-30',
        'test_start': '2023-07-21',
        'test_end': '2023-12-31'
    },
    {
        'fold_id': 4,
        'train_start': '2022-01-01',
        'train_end': '2024-06-30',
        'test_start': '2024-07-21',
        'test_end': '2025-05-31'
    },
]

# ==============================================================================
# DEVICE
# ==============================================================================

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
