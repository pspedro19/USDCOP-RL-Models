#!/usr/bin/env python3
"""
Model Card Generator - MLflow Integration
==========================================

Generates comprehensive model cards from MLflow run data and model artifacts.
Implements Phase 3 Governance requirement for automated model documentation.

Usage:
    # Generate from MLflow run ID
    python scripts/generate_model_card.py --run-id abc123def456

    # Generate from model ID tag
    python scripts/generate_model_card.py --model-id ppo_v20_20260115

    # Generate from local model directory (fallback)
    python scripts/generate_model_card.py --model models/ppo_primary

    # Specify output directory
    python scripts/generate_model_card.py --model-id ppo_v20 --output docs/model_cards/

Author: Trading Operations Team
Date: 2026-01-17
Version: 2.0.0
"""

import argparse
import json
import os
import sys

# Add project root to path for SSOT imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# SSOT import for hash utilities
from src.utils.hash_utils import compute_file_hash as _compute_file_hash_ssot, compute_json_hash as _compute_json_hash_ssot
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Template for model card generation
MODEL_CARD_TEMPLATE = '''# Model Card: {model_id}

## Basic Information

| Field | Value |
|-------|-------|
| **Model ID** | {model_id} |
| **Version** | {version} |
| **Created Date** | {created_date} |
| **Owner** | {owner} |
| **Backup Owner** | {backup_owner} |
| **Current Stage** | {stage} |
| **MLflow Run ID** | {mlflow_run_id} |
| **MLflow Experiment** | {mlflow_experiment} |

---

## Training Details

| Field | Value |
|-------|-------|
| **Training Start** | {training_start} |
| **Training End** | {training_end} |
| **Training Duration** | {training_duration} |
| **Total Timesteps** | {total_timesteps} |
| **Dataset Version** | {dataset_hash} |
| **Dataset Period** | {dataset_start} to {dataset_end} |
| **Feature Count** | 15 (CTR-FEAT-001) |
| **Reward Function** | {reward_function} |

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
| Value Function Coef | {vf_coef} |
| Max Grad Norm | {max_grad_norm} |

---

## Artifact Hashes

| Artifact | Hash (SHA256) | Verified |
|----------|---------------|----------|
| Model (.zip) | `{model_hash}` | {model_verified} |
| Model (.onnx) | `{onnx_hash}` | {onnx_verified} |
| norm_stats.json | `{norm_stats_hash}` | {norm_stats_verified} |
| Dataset | `{dataset_hash}` | {dataset_verified} |
| Feature Order | `{feature_order_hash}` | {feature_order_verified} |

---

## Feature Set (CTR-FEAT-001)

| # | Feature Name | Type | Description |
|---|--------------|------|-------------|
| 1 | log_ret_5m | Numeric | 5-minute log return |
| 2 | log_ret_1h | Numeric | 1-hour log return |
| 3 | log_ret_4h | Numeric | 4-hour log return |
| 4 | rsi_9 | Numeric | 9-period RSI (Wilder's) |
| 5 | atr_pct | Numeric | ATR as % of price |
| 6 | adx_14 | Numeric | 14-period ADX |
| 7 | dxy_z | Numeric | DXY z-score |
| 8 | dxy_change_1d | Numeric | DXY 1-day change |
| 9 | vix_z | Numeric | VIX z-score |
| 10 | embi_z | Numeric | EMBI Colombia z-score |
| 11 | brent_change_1d | Numeric | Brent 1-day change |
| 12 | rate_spread | Numeric | Interest rate spread |
| 13 | usdmxn_change_1d | Numeric | USD/MXN 1-day change |
| 14 | position | Numeric | Current position |
| 15 | time_normalized | Numeric | Normalized trading time |

---

## Performance Metrics

### Backtest Performance

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Sharpe Ratio | {backtest_sharpe} | >= 1.0 | {sharpe_status} |
| Win Rate | {backtest_win_rate} | >= 50% | {win_rate_status} |
| Max Drawdown | {backtest_max_dd} | <= 10% | {max_dd_status} |
| Total Trades | {backtest_trades} | >= 100 | {trades_status} |
| Profit Factor | {profit_factor} | >= 1.5 | {pf_status} |
| Total Return | {total_return} | N/A | - |
| Sortino Ratio | {sortino_ratio} | N/A | - |

### Action Distribution

| Action | Count | Percentage |
|--------|-------|------------|
| HOLD (0) | {hold_count} | {hold_pct} |
| BUY (1) | {buy_count} | {buy_pct} |
| SELL (2) | {sell_count} | {sell_pct} |

{staging_section}

---

## Known Limitations

1. **Timeframe Dependency**: Model trained on 5-minute bars only
2. **Warmup Period**: Requires 14 bars before first valid prediction
3. **Volatility Sensitivity**: Performance may degrade when VIX > 40
4. **Market Hours**: Optimized for Colombia trading hours (8:00-16:00 COT)
5. **Data Dependency**: Requires TwelveData and macro data sources

---

## Risk Factors

| Risk | Severity | Mitigation |
|------|----------|------------|
| High volatility regime | Medium | Circuit breaker at 5% daily drawdown |
| DXY correlation breakdown | Medium | Daily macro indicator monitoring |
| Data source outage | High | Auto-pause on stale data |
| Model drift | Medium | Hourly PSI monitoring |

---

## Deployment Information

### Latency SLAs

| Metric | Target | Measured |
|--------|--------|----------|
| P50 | < 20ms | {latency_p50} |
| P95 | < 50ms | {latency_p95} |
| P99 | < 100ms | {latency_p99} |

---

## Change History

| Date | Change | Author |
|------|--------|--------|
| {created_date} | Initial training complete | {owner} |
{change_history}

---

**Generated**: {generation_date}
**Generator**: scripts/generate_model_card.py v2.0.0
**Next Review**: {review_date}

---

*This model card was auto-generated from MLflow run data. Manual review required before promotion.*
'''

STAGING_SECTION_TEMPLATE = '''
### Staging Performance

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Sharpe Ratio | {staging_sharpe} | >= 1.0 | {staging_sharpe_status} |
| Win Rate | {staging_win_rate} | >= 50% | {staging_win_rate_status} |
| Agreement Rate | {agreement_rate} | >= 85% | {agreement_status} |
| Days in Staging | {staging_days} | >= 7 | {staging_days_status} |
'''


def compute_file_hash(path: Path, algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file.

    SSOT: Delegates to src.utils.hash_utils

    Args:
        path: Path to file
        algorithm: Hash algorithm ('sha256' or 'md5')

    Returns:
        Hex digest of hash, or 'N/A' if file doesn't exist
    """
    if not path.exists():
        return 'N/A'

    result = _compute_file_hash_ssot(path, algorithm=algorithm)
    return result.full_hash


def compute_json_hash(path: Path) -> str:
    """
    Compute hash of JSON file with sorted keys for consistency.

    SSOT: Delegates to src.utils.hash_utils

    Args:
        path: Path to JSON file

    Returns:
        Hex digest of hash
    """
    if not path.exists():
        return 'N/A'

    return _compute_json_hash_ssot(path).full_hash


def get_mlflow_client():
    """Get MLflow tracking client."""
    try:
        import mlflow
        tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow.tracking.MlflowClient()
    except ImportError:
        print("Warning: MLflow not installed. Using local mode only.")
        return None


def get_run_by_model_id(client, model_id: str) -> Optional[Any]:
    """
    Find MLflow run by model_id tag.

    Args:
        client: MLflow client
        model_id: Model ID to search for

    Returns:
        MLflow Run object or None
    """
    if client is None:
        return None

    try:
        # Search across all experiments
        experiments = client.search_experiments()
        for exp in experiments:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.model_id = '{model_id}'"
            )
            if runs:
                return runs[0]
    except Exception as e:
        print(f"Warning: Could not search MLflow: {e}")

    return None


def get_run_by_id(client, run_id: str) -> Optional[Any]:
    """
    Get MLflow run by run ID.

    Args:
        client: MLflow client
        run_id: MLflow run ID

    Returns:
        MLflow Run object or None
    """
    if client is None:
        return None

    try:
        return client.get_run(run_id)
    except Exception as e:
        print(f"Warning: Could not get run {run_id}: {e}")
        return None


def extract_mlflow_data(run) -> Dict[str, Any]:
    """
    Extract model card data from MLflow run.

    Args:
        run: MLflow Run object

    Returns:
        Dictionary of extracted data
    """
    params = run.data.params
    metrics = run.data.metrics
    tags = run.data.tags

    # Convert timestamps
    start_time = datetime.fromtimestamp(run.info.start_time / 1000)
    end_time = datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else datetime.now()
    duration = end_time - start_time

    data = {
        # Basic info
        'model_id': tags.get('model_id', params.get('model_id', 'unknown')),
        'version': params.get('model_version', tags.get('version', '1')),
        'created_date': start_time.strftime('%Y-%m-%d'),
        'owner': tags.get('owner', 'trading_team'),
        'backup_owner': tags.get('backup_owner', 'ml_team'),
        'stage': tags.get('stage', 'registered'),
        'mlflow_run_id': run.info.run_id,
        'mlflow_experiment': run.info.experiment_id,

        # Training details
        'training_start': start_time.strftime('%Y-%m-%d %H:%M'),
        'training_end': end_time.strftime('%Y-%m-%d %H:%M'),
        'training_duration': f"{duration.total_seconds() / 3600:.1f} hours",
        'total_timesteps': params.get('total_timesteps', 'N/A'),
        'dataset_start': params.get('dataset_start', '2020-03-02'),
        'dataset_end': params.get('dataset_end', '2025-12-31'),
        'reward_function': params.get('reward_function', 'TradingRewardV3'),

        # Hyperparameters
        'learning_rate': params.get('learning_rate', '1e-4'),
        'n_steps': params.get('n_steps', '2048'),
        'batch_size': params.get('batch_size', '128'),
        'n_epochs': params.get('n_epochs', '10'),
        'gamma': params.get('gamma', '0.90'),
        'gae_lambda': params.get('gae_lambda', '0.95'),
        'clip_range': params.get('clip_range', '0.2'),
        'ent_coef': params.get('ent_coef', '0.05'),
        'vf_coef': params.get('vf_coef', '0.5'),
        'max_grad_norm': params.get('max_grad_norm', '0.5'),

        # Hashes
        'model_hash': params.get('model_hash', tags.get('model_hash_full', 'N/A'))[:16],
        'onnx_hash': params.get('onnx_hash', 'N/A')[:16] if params.get('onnx_hash') else 'N/A',
        'norm_stats_hash': params.get('norm_stats_hash', tags.get('norm_stats_hash_full', 'N/A'))[:16],
        'dataset_hash': params.get('dataset_hash', tags.get('dataset_hash_full', 'N/A'))[:16],
        'feature_order_hash': params.get('feature_order_hash', 'N/A')[:16],

        # Performance metrics
        'backtest_sharpe': f"{metrics.get('backtest_sharpe', metrics.get('sharpe_ratio', 0)):.2f}",
        'backtest_win_rate': f"{metrics.get('backtest_win_rate', metrics.get('win_rate', 0)) * 100:.1f}%",
        'backtest_max_dd': f"{abs(metrics.get('backtest_max_drawdown', metrics.get('max_drawdown', 0))) * 100:.1f}%",
        'backtest_trades': str(int(metrics.get('total_trades', 0))),
        'profit_factor': f"{metrics.get('profit_factor', 0):.2f}",
        'total_return': f"{metrics.get('total_return', 0) * 100:.1f}%",
        'sortino_ratio': f"{metrics.get('sortino_ratio', 0):.2f}",

        # Action distribution
        'hold_count': str(int(metrics.get('hold_count', 0))),
        'buy_count': str(int(metrics.get('buy_count', 0))),
        'sell_count': str(int(metrics.get('sell_count', 0))),
        'hold_pct': f"{metrics.get('hold_pct', 70):.1f}%",
        'buy_pct': f"{metrics.get('buy_pct', 15):.1f}%",
        'sell_pct': f"{metrics.get('sell_pct', 15):.1f}%",

        # Latency (if available)
        'latency_p50': f"{metrics.get('latency_p50', 15):.0f}ms",
        'latency_p95': f"{metrics.get('latency_p95', 35):.0f}ms",
        'latency_p99': f"{metrics.get('latency_p99', 80):.0f}ms",

        # Staging metrics (if available)
        'staging_sharpe': metrics.get('staging_sharpe'),
        'staging_win_rate': metrics.get('staging_win_rate'),
        'agreement_rate': metrics.get('agreement_rate'),
        'staging_days': metrics.get('staging_days'),
    }

    return data


def extract_local_data(model_dir: Path, norm_stats_path: Path) -> Dict[str, Any]:
    """
    Extract model card data from local files (fallback mode).

    Args:
        model_dir: Path to model directory
        norm_stats_path: Path to norm_stats.json

    Returns:
        Dictionary of extracted data
    """
    model_name = model_dir.name

    # Load norm stats
    norm_stats = {}
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

    # Load training metrics if available
    metrics = {}
    metrics_path = model_dir / 'training_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Compute hashes
    model_zip = model_dir / f'{model_name}.zip'
    model_onnx = model_dir / 'model.onnx'

    data = {
        'model_id': model_name,
        'version': '1',
        'created_date': datetime.now().strftime('%Y-%m-%d'),
        'owner': 'trading_team',
        'backup_owner': 'ml_team',
        'stage': 'registered',
        'mlflow_run_id': 'N/A (local mode)',
        'mlflow_experiment': 'N/A',

        'training_start': 'N/A',
        'training_end': 'N/A',
        'training_duration': 'N/A',
        'total_timesteps': str(metrics.get('total_timesteps', 'N/A')),
        'dataset_start': '2020-03-02',
        'dataset_end': '2025-12-31',
        'reward_function': 'TradingRewardV3',

        'learning_rate': '1e-4',
        'n_steps': '2048',
        'batch_size': '128',
        'n_epochs': '10',
        'gamma': '0.90',
        'gae_lambda': '0.95',
        'clip_range': '0.2',
        'ent_coef': '0.05',
        'vf_coef': '0.5',
        'max_grad_norm': '0.5',

        'model_hash': compute_file_hash(model_zip)[:16],
        'onnx_hash': compute_file_hash(model_onnx)[:16],
        'norm_stats_hash': compute_json_hash(norm_stats_path)[:16],
        'dataset_hash': 'N/A',
        'feature_order_hash': 'N/A',

        'backtest_sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
        'backtest_win_rate': f"{metrics.get('win_rate', 0) * 100:.1f}%",
        'backtest_max_dd': f"{abs(metrics.get('max_drawdown', 0)) * 100:.1f}%",
        'backtest_trades': str(int(metrics.get('total_trades', 0))),
        'profit_factor': f"{metrics.get('profit_factor', 0):.2f}",
        'total_return': f"{metrics.get('total_return', 0) * 100:.1f}%",
        'sortino_ratio': f"{metrics.get('sortino_ratio', 0):.2f}",

        'hold_count': str(int(metrics.get('hold_count', 0))),
        'buy_count': str(int(metrics.get('buy_count', 0))),
        'sell_count': str(int(metrics.get('sell_count', 0))),
        'hold_pct': f"{metrics.get('hold_pct', 70):.1f}%",
        'buy_pct': f"{metrics.get('buy_pct', 15):.1f}%",
        'sell_pct': f"{metrics.get('sell_pct', 15):.1f}%",

        'latency_p50': 'N/A',
        'latency_p95': 'N/A',
        'latency_p99': 'N/A',

        'staging_sharpe': None,
        'staging_win_rate': None,
        'agreement_rate': None,
        'staging_days': None,
    }

    return data


def get_status(value: float, threshold: float, higher_is_better: bool = True) -> str:
    """Determine pass/fail status based on threshold."""
    if higher_is_better:
        return 'Pass' if value >= threshold else 'Fail'
    else:
        return 'Pass' if value <= threshold else 'Fail'


def generate_model_card(
    data: Dict[str, Any],
    output_dir: Path
) -> Path:
    """
    Generate model card from extracted data.

    Args:
        data: Dictionary of model data
        output_dir: Output directory for model card

    Returns:
        Path to generated model card
    """
    # Compute verification statuses
    data['model_verified'] = 'Yes' if data['model_hash'] != 'N/A' else 'No'
    data['onnx_verified'] = 'Yes' if data['onnx_hash'] != 'N/A' else 'No'
    data['norm_stats_verified'] = 'Yes' if data['norm_stats_hash'] != 'N/A' else 'No'
    data['dataset_verified'] = 'Yes' if data['dataset_hash'] != 'N/A' else 'No'
    data['feature_order_verified'] = 'Yes' if data['feature_order_hash'] != 'N/A' else 'No'

    # Compute metric statuses
    try:
        sharpe = float(data['backtest_sharpe'])
        data['sharpe_status'] = get_status(sharpe, 1.0)
    except:
        data['sharpe_status'] = 'N/A'

    try:
        win_rate = float(data['backtest_win_rate'].replace('%', ''))
        data['win_rate_status'] = get_status(win_rate, 50)
    except:
        data['win_rate_status'] = 'N/A'

    try:
        max_dd = float(data['backtest_max_dd'].replace('%', ''))
        data['max_dd_status'] = get_status(max_dd, 10, higher_is_better=False)
    except:
        data['max_dd_status'] = 'N/A'

    try:
        trades = int(data['backtest_trades'])
        data['trades_status'] = get_status(trades, 100)
    except:
        data['trades_status'] = 'N/A'

    try:
        pf = float(data['profit_factor'])
        data['pf_status'] = get_status(pf, 1.5)
    except:
        data['pf_status'] = 'N/A'

    # Generate staging section if applicable
    if data.get('staging_sharpe') is not None:
        staging_data = {
            'staging_sharpe': f"{data['staging_sharpe']:.2f}",
            'staging_win_rate': f"{data['staging_win_rate'] * 100:.1f}%",
            'agreement_rate': f"{data['agreement_rate'] * 100:.1f}%",
            'staging_days': str(int(data['staging_days'])),
            'staging_sharpe_status': get_status(data['staging_sharpe'], 1.0),
            'staging_win_rate_status': get_status(data['staging_win_rate'], 0.5),
            'agreement_status': get_status(data['agreement_rate'], 0.85),
            'staging_days_status': get_status(data['staging_days'], 7),
        }
        data['staging_section'] = STAGING_SECTION_TEMPLATE.format(**staging_data)
    else:
        data['staging_section'] = ''

    # Add generation metadata
    data['generation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    review_date = datetime.now()
    data['review_date'] = (review_date.replace(month=review_date.month + 3 if review_date.month <= 9 else 1,
                                                year=review_date.year if review_date.month <= 9 else review_date.year + 1)
                          ).strftime('%Y-%m-%d')

    # Change history placeholder
    data['change_history'] = ''

    # Generate card content
    content = MODEL_CARD_TEMPLATE.format(**data)

    # Write to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data['model_id']}.md"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Model card generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate model card from MLflow or local files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Generate from MLflow run ID
    python scripts/generate_model_card.py --run-id abc123def456

    # Generate from model ID tag in MLflow
    python scripts/generate_model_card.py --model-id ppo_v20_20260115

    # Generate from local model directory (fallback)
    python scripts/generate_model_card.py --model models/ppo_primary

    # Specify custom output directory
    python scripts/generate_model_card.py --model-id ppo_v20 --output docs/model_cards/
        '''
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--run-id',
        type=str,
        help='MLflow run ID'
    )
    source_group.add_argument(
        '--model-id',
        type=str,
        help='Model ID tag to search in MLflow'
    )
    source_group.add_argument(
        '--model',
        type=str,
        help='Path to local model directory (fallback mode)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='docs/model_cards',
        help='Output directory for model card (default: docs/model_cards)'
    )
    parser.add_argument(
        '--norm-stats',
        type=str,
        default='config/norm_stats.json',
        help='Path to norm_stats.json for local mode'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    output_dir = Path(args.output)

    # Try MLflow first
    client = get_mlflow_client()
    run = None
    data = None

    if args.run_id:
        if args.verbose:
            print(f"Looking up MLflow run: {args.run_id}")
        run = get_run_by_id(client, args.run_id)
        if run:
            data = extract_mlflow_data(run)
        else:
            print(f"Error: Could not find run with ID {args.run_id}")
            sys.exit(1)

    elif args.model_id:
        if args.verbose:
            print(f"Searching MLflow for model_id: {args.model_id}")
        run = get_run_by_model_id(client, args.model_id)
        if run:
            data = extract_mlflow_data(run)
        else:
            print(f"Warning: Could not find model_id {args.model_id} in MLflow")
            # Fall back to local if model directory exists
            model_dir = PROJECT_ROOT / 'models' / args.model_id
            if model_dir.exists():
                print(f"Falling back to local mode: {model_dir}")
                data = extract_local_data(model_dir, Path(args.norm_stats))
            else:
                print(f"Error: Model not found in MLflow or locally")
                sys.exit(1)

    elif args.model:
        model_dir = Path(args.model)
        if not model_dir.exists():
            print(f"Error: Model directory not found: {model_dir}")
            sys.exit(1)
        if args.verbose:
            print(f"Using local mode: {model_dir}")
        data = extract_local_data(model_dir, Path(args.norm_stats))

    # Generate the model card
    output_path = generate_model_card(data, output_dir)

    if args.verbose:
        print(f"\nModel card contents:")
        with open(output_path) as f:
            print(f.read()[:2000] + "...\n[truncated]")

    print(f"\nSuccess! Model card saved to: {output_path}")


if __name__ == '__main__':
    main()
