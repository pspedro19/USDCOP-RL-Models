#!/usr/bin/env python3
"""
Model Card Generator (Phase 15.1)
=================================

Generates model cards from training artifacts and metrics.

Usage:
    python scripts/generate_model_card.py --model models/ppo_primary --output docs/model_cards/

Author: Trading Team
Date: 2025-01-14
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Template sections
TEMPLATE = '''# Model Card: {model_name}

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | {model_name} |
| **Version** | {version} |
| **Type** | PPO (Proximal Policy Optimization) |
| **Framework** | Stable-Baselines3 |
| **Created** | {created_date} |
| **Author** | Trading Team |
| **Status** | {status} |

## Intended Use

### Primary Use Case
Automated trading signal generation for USD/COP currency pair using reinforcement learning.

### Out-of-Scope Uses
- Live trading without human oversight
- Trading other currency pairs without retraining
- High-frequency trading (sub-second decisions)

## Training Data

### Dataset
| Field | Value |
|-------|-------|
| **Dataset Name** | {dataset_name} |
| **Date Range** | {date_range} |
| **Rows** | {row_count:,} |
| **Features** | 15 (13 market + 2 state) |
| **Data Version** | {data_version} |

### Feature Set
```
log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d,
rate_spread, usdmxn_change_1d, position, time_normalized
```

### Normalization Stats
| Feature | Mean | Std |
|---------|------|-----|
{norm_stats_table}

**Norm Stats Hash**: `{norm_stats_hash}`

## Model Architecture

### Network
```
Policy Network (pi):
  - Input: 15 dimensions
  - Hidden: [256, 256]
  - Activation: Tanh
  - Output: 3 (HOLD, BUY, SELL)

Value Network (vf):
  - Input: 15 dimensions
  - Hidden: [256, 256]
  - Activation: Tanh
  - Output: 1 (state value)
```

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | {learning_rate} |
| N Steps | {n_steps} |
| Batch Size | {batch_size} |
| N Epochs | {n_epochs} |
| Gamma | {gamma} |
| GAE Lambda | {gae_lambda} |
| Clip Range | {clip_range} |
| Entropy Coef | {ent_coef} |

## Performance Metrics

### Backtest Results
| Metric | Value |
|--------|-------|
| Total Return | {total_return} |
| Sharpe Ratio | {sharpe_ratio} |
| Sortino Ratio | {sortino_ratio} |
| Max Drawdown | {max_drawdown} |
| Win Rate | {win_rate} |
| Profit Factor | {profit_factor} |

### Action Distribution
| Action | Percentage |
|--------|------------|
| HOLD | {hold_pct} |
| BUY | {buy_pct} |
| SELL | {sell_pct} |

## Limitations and Biases

### Known Limitations
- Performance may degrade in extreme market conditions (VIX > 40)
- Requires minimum 14 bars warmup before valid predictions
- Trained on 5-minute bars; not suitable for other timeframes
- USD/COP specific; may not generalize to other EM pairs

### Potential Biases
- Training data from 2020-2025 may overweight COVID recovery period
- Model may have learned patterns specific to Colombian market hours

### Failure Modes
- Circuit breaker activates if >20% features are NaN
- Model outputs HOLD during warmup period
- May exhibit stuck behavior in low volatility regimes

## Deployment

### Requirements
- Python 3.9+
- ONNX Runtime 1.15+
- norm_stats.json (exact version used in training)

### Inference Latency
| Metric | Target |
|--------|--------|
| p50 | < 20ms |
| p95 | < 50ms |
| p99 | < 100ms |

### Model Files
| File | Hash |
|------|------|
| model.onnx | `{onnx_hash}` |
| norm_stats.json | `{norm_stats_hash}` |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| {version} | {created_date} | {version_notes} |

---
*Generated on {generated_date} by generate_model_card.py*
'''


def get_file_hash(path: Path) -> str:
    """Calculate MD5 hash of file."""
    if not path.exists():
        return "N/A"

    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()[:12]


def load_norm_stats(path: Path) -> Dict[str, Any]:
    """Load normalization statistics."""
    if not path.exists():
        return {}

    with open(path) as f:
        return json.load(f)


def format_norm_stats_table(stats: Dict[str, Any]) -> str:
    """Format norm stats as markdown table."""
    if not stats:
        return "| N/A | N/A | N/A |"

    lines = []
    for feature, values in stats.items():
        mean = values.get('mean', 0)
        std = values.get('std', 1)
        lines.append(f"| {feature} | {mean:.6f} | {std:.6f} |")

    return '\n'.join(lines)


def load_training_metrics(model_dir: Path) -> Dict[str, Any]:
    """Load training metrics from model directory."""
    metrics_path = model_dir / 'training_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def generate_model_card(
    model_dir: Path,
    norm_stats_path: Path,
    output_dir: Path,
    version: str = "1.0.0",
    status: str = "Development"
) -> Path:
    """
    Generate model card for a trained model.

    Args:
        model_dir: Path to model directory
        norm_stats_path: Path to norm_stats.json
        output_dir: Output directory for model card
        version: Model version string
        status: Model status (Development/Staging/Production)

    Returns:
        Path to generated model card
    """
    model_name = model_dir.name

    # Load data
    norm_stats = load_norm_stats(norm_stats_path)
    metrics = load_training_metrics(model_dir)

    # Calculate hashes
    onnx_path = model_dir / 'model.onnx'
    onnx_hash = get_file_hash(onnx_path)
    norm_stats_hash = get_file_hash(norm_stats_path)

    # Default values
    defaults = {
        'model_name': model_name,
        'version': version,
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'status': status,
        'dataset_name': 'RL_DS3_MACRO_CORE',
        'date_range': '2020-03-02 to 2025-10-29',
        'row_count': 84671,
        'data_version': 'HEAD',
        'norm_stats_table': format_norm_stats_table(norm_stats),
        'norm_stats_hash': norm_stats_hash,
        'onnx_hash': onnx_hash,
        'learning_rate': '1e-4',
        'n_steps': '2048',
        'batch_size': '128',
        'n_epochs': '10',
        'gamma': '0.90',  # From config/trading_config.yaml SSOT
        'gae_lambda': '0.95',
        'clip_range': '0.2',
        'ent_coef': '0.05',
        'total_return': metrics.get('total_return', 'N/A'),
        'sharpe_ratio': metrics.get('sharpe_ratio', 'N/A'),
        'sortino_ratio': metrics.get('sortino_ratio', 'N/A'),
        'max_drawdown': metrics.get('max_drawdown', 'N/A'),
        'win_rate': metrics.get('win_rate', 'N/A'),
        'profit_factor': metrics.get('profit_factor', 'N/A'),
        'hold_pct': metrics.get('hold_pct', '~70%'),
        'buy_pct': metrics.get('buy_pct', '~15%'),
        'sell_pct': metrics.get('sell_pct', '~15%'),
        'version_notes': 'Initial version with Wilder\'s EMA for RSI/ATR/ADX',
        'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Generate card
    content = TEMPLATE.format(**defaults)

    # Write to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{model_name}_v{version.replace(".", "_")}.md'

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Generated model card: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate model card')
    parser.add_argument('--model', type=str, default='models/ppo_primary',
                       help='Path to model directory')
    parser.add_argument('--norm-stats', type=str, default='config/norm_stats.json',
                       help='Path to norm_stats.json')
    parser.add_argument('--output', type=str, default='docs/model_cards',
                       help='Output directory')
    parser.add_argument('--version', type=str, default='1.0.0',
                       help='Model version')
    parser.add_argument('--status', type=str, default='Development',
                       choices=['Development', 'Staging', 'Production', 'Deprecated'],
                       help='Model status')

    args = parser.parse_args()

    generate_model_card(
        model_dir=Path(args.model),
        norm_stats_path=Path(args.norm_stats),
        output_dir=Path(args.output),
        version=args.version,
        status=args.status
    )


if __name__ == '__main__':
    main()
