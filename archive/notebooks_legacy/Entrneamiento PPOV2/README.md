# USD/COP RL Trading System V19.1

## Status: PRODUCTION READY

| Metric | Factory V3 | V1 Config |
|--------|------------|-----------|
| Sharpe Ratio | **3.01** | 0.15 |
| Max Drawdown | 1.09% | 0.83% |
| Win Rate | 41.9% | 38.4% |
| Model | PPO (ent_coef=0.05) | PPO (ent_coef=0.05) |

## Project Structure

```
├── PRODUCTION_CONFIG.json      # Validated production hyperparameters
├── README.md
├── src/                        # Source code (SOLID architecture)
│   ├── core/                   # Interfaces & Exceptions
│   │   ├── interfaces.py       # Protocols (IRewardFunction, IRiskManager, ICostModel)
│   │   └── exceptions.py       # Custom exception hierarchy
│   ├── config/                 # Configuration
│   │   ├── defaults.py         # Production defaults
│   │   └── training_config.py  # Dataclasses with validation
│   ├── factories/              # Factory Pattern
│   │   └── environment_factory.py  # Presets: default, conservative, aggressive, production
│   ├── rewards/                # Strategy Pattern
│   │   ├── base.py             # AbstractRewardFunction, CompositeReward
│   │   └── symmetric_curriculum.py  # SymmetricCurriculumReward
│   ├── callbacks/              # Training callbacks
│   ├── validation/             # PurgedKFoldCV, StressTester
│   ├── environment_v19.py      # Production environment
│   ├── regime_detector.py      # Market regime detection
│   ├── risk_manager.py         # Position sizing
│   └── train_v19.py            # Training entry point
├── outputs/                    # Validation results (JSON)
└── archive/                    # Archived files
```

## SOLID Principles

- **Single Responsibility**: Separate modules for rewards, config, validation
- **Open/Closed**: AbstractRewardFunction allows extension
- **Liskov Substitution**: All rewards implement IRewardFunction protocol
- **Interface Segregation**: Small focused protocols
- **Dependency Inversion**: Factory pattern + protocol-based injection

## Quick Start

```python
from src.factories.environment_factory import EnvironmentFactory

# Production environment
env = EnvironmentFactory.create_from_preset('production', df=your_data)

# Custom config
from src.config import create_production_config
config = create_production_config()
```

## Validated 2025-12-26

5-Fold CV Mean Sharpe: 2.21 | All criteria passed
