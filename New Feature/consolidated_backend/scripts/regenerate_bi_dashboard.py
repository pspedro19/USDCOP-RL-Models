#!/usr/bin/env python3
"""
regenerate_bi_dashboard.py
==========================
Master script to regenerate bi_dashboard_unified.csv with all available data.

This script scans local data sources and generates a comprehensive CSV for the BI dashboard:
1. Scans outputs/forecasts/ for the 2 most recent date directories (excludes old data)
2. Reads forecast_data.json from each directory to get predictions
3. Reads the latest backtest metrics from outputs/runs/*/data/backtest_metrics.csv
4. Generates bi_dashboard_unified.csv with:
   - All individual models (9)
   - All ensemble types (best_of_breed, top_3, top_6_mean)
   - 2 most recent weeks only
   - Backtest rows: one per model per horizon (7 horizons)
   - Forward forecast rows: ONE per model per week (images contain ALL horizons)

IMPORTANT:
- Forward forecast images contain ALL 7 horizons in ONE image
- Therefore forward_forecast rows are NOT multiplied by horizon
- Correct structure: 13 models x 2 weeks = 26 forward forecast rows
- Backtest structure: 9 models x 7 horizons = 63 backtest rows

Usage:
    python backend/scripts/regenerate_bi_dashboard.py
    python backend/scripts/regenerate_bi_dashboard.py --verbose
    python backend/scripts/regenerate_bi_dashboard.py --output ./custom_output
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Constants
INDIVIDUAL_MODELS = [
    'ridge', 'bayesian_ridge', 'ard',
    'xgboost_pure', 'lightgbm_pure', 'catboost_pure',
    'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'
]

ENSEMBLE_TYPES = ['best_of_breed', 'top_3', 'top_6_mean', 'ensemble']

HORIZONS = [1, 5, 10, 15, 20, 25, 30]

IMAGE_TYPES = ['simple', 'complete', 'fan_chart']

MODEL_TYPE_MAP = {
    'ridge': 'linear',
    'bayesian_ridge': 'linear',
    'ard': 'linear',
    'xgboost_pure': 'tree_ensemble',
    'lightgbm_pure': 'tree_ensemble',
    'catboost_pure': 'tree_ensemble',
    'hybrid_xgboost': 'hybrid',
    'hybrid_lightgbm': 'hybrid',
    'hybrid_catboost': 'hybrid',
    'best_of_breed': 'ensemble',
    'top_3': 'ensemble',
    'top_6_mean': 'ensemble',
    'ensemble': 'ensemble'
}


def get_model_type(model_id: str) -> str:
    """Return the model type based on model ID."""
    return MODEL_TYPE_MAP.get(model_id, 'unknown')


def get_horizon_category(horizon: int) -> str:
    """Return the horizon category based on the horizon value."""
    if horizon <= 5:
        return 'corto_plazo'
    elif horizon <= 15:
        return 'mediano_plazo'
    return 'largo_plazo'


def get_model_display_name(model_id: str) -> str:
    """Return a display-friendly model name."""
    return model_id.upper().replace('_', ' ')


def scan_forecast_directories(forecasts_dir: Path, max_weeks: int = 3) -> List[Path]:
    """
    Scan the forecasts directory for the most recent date directories with forecast_data.json.

    Only returns the most recent weeks to avoid including old/stale data in the dashboard.

    Args:
        forecasts_dir: Path to outputs/forecasts directory
        max_weeks: Maximum number of recent weeks to include (default: 2)

    Returns:
        List of date directory paths sorted by date (most recent last)
    """
    if not forecasts_dir.exists():
        logger.warning(f"Forecasts directory not found: {forecasts_dir}")
        return []

    date_dirs = []
    for item in forecasts_dir.iterdir():
        if item.is_dir() and (item / 'forecast_data.json').exists():
            date_dirs.append(item)

    # Sort by directory name (date format YYYY-MM-DD)
    date_dirs.sort(key=lambda x: x.name)

    # Only keep the most recent weeks
    if len(date_dirs) > max_weeks:
        excluded = date_dirs[:-max_weeks]
        date_dirs = date_dirs[-max_weeks:]
        logger.info(f"Excluding {len(excluded)} old forecast directories: {[d.name for d in excluded]}")

    logger.info(f"Using {len(date_dirs)} most recent forecast directories: {[d.name for d in date_dirs]}")
    return date_dirs


def find_latest_backtest_metrics(runs_dir: Path) -> Optional[Path]:
    """
    Find the most recent backtest_metrics.csv file from the runs directory.

    Args:
        runs_dir: Path to outputs/runs directory

    Returns:
        Path to the latest backtest_metrics.csv or None
    """
    if not runs_dir.exists():
        logger.warning(f"Runs directory not found: {runs_dir}")
        return None

    metrics_files = list(runs_dir.glob('*/data/backtest_metrics.csv'))
    if not metrics_files:
        logger.warning("No backtest_metrics.csv files found")
        return None

    # Sort by parent directory name (timestamp format YYYYMMDD_HHMMSS)
    metrics_files.sort(key=lambda x: x.parent.parent.name, reverse=True)
    latest = metrics_files[0]
    logger.info(f"Using latest backtest metrics: {latest}")
    return latest


def load_backtest_metrics(metrics_file: Path) -> pd.DataFrame:
    """
    Load and process backtest metrics from CSV.

    Args:
        metrics_file: Path to backtest_metrics.csv

    Returns:
        DataFrame with backtest metrics
    """
    df = pd.read_csv(metrics_file)
    logger.info(f"Loaded {len(df)} rows from backtest metrics")
    return df


def load_forecast_data(json_file: Path) -> Dict[str, Any]:
    """
    Load forecast data from JSON file.

    Args:
        json_file: Path to forecast_data.json

    Returns:
        Dictionary with forecast data
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_backtest_rows(
    df_metrics: pd.DataFrame,
    training_date: str
) -> List[Dict[str, Any]]:
    """
    Generate backtest rows for the BI dashboard CSV.

    Args:
        df_metrics: DataFrame with backtest metrics
        training_date: Date string for training date

    Returns:
        List of dictionaries representing CSV rows
    """
    rows = []

    # Calculate model averages
    model_avg_metrics = {}
    for model_id in df_metrics['model'].unique():
        model_data = df_metrics[df_metrics['model'] == model_id]
        model_avg_metrics[model_id] = {
            'avg_da': model_data['direction_accuracy'].mean() * 100,  # Convert to percentage
            'avg_rmse': model_data['rmse'].mean()
        }

    # Find best overall model
    best_overall_model = max(
        model_avg_metrics.keys(),
        key=lambda x: model_avg_metrics[x]['avg_da']
    ) if model_avg_metrics else None

    # Find best model per horizon
    best_per_horizon = {}
    for h in df_metrics['horizon'].unique():
        h_data = df_metrics[df_metrics['horizon'] == h]
        best_idx = h_data['direction_accuracy'].idxmax()
        best_per_horizon[h] = {
            'model': h_data.loc[best_idx, 'model'],
            'da': h_data.loc[best_idx, 'direction_accuracy'] * 100
        }

    # Generate rows
    for _, row in df_metrics.iterrows():
        model_id = row['model']
        horizon = int(row['horizon'])

        model_avg = model_avg_metrics.get(model_id, {'avg_da': None, 'avg_rmse': None})
        best_h = best_per_horizon.get(horizon, {'model': None, 'da': None})

        da_value = row['direction_accuracy'] * 100 if pd.notna(row['direction_accuracy']) else None

        rows.append({
            'record_id': f"BT_{model_id}_h{horizon}",
            'view_type': 'backtest',
            'model_id': model_id,
            'model_name': get_model_display_name(model_id),
            'model_type': get_model_type(model_id),
            'horizon_days': horizon,
            'horizon_label': f"H={horizon}",
            'horizon_category': get_horizon_category(horizon),
            'inference_week': None,
            'inference_date': None,
            'direction_accuracy': round(da_value, 2) if da_value else None,
            'rmse': round(row['rmse'], 6) if pd.notna(row['rmse']) else None,
            'r2': round(row['r2'], 4) if pd.notna(row['r2']) else None,
            'mae': round(row['mae'], 6) if pd.notna(row.get('mae')) else None,
            'base_price': None,
            'predicted_price': None,
            'price_change': None,
            'price_change_pct': None,
            'predicted_return_pct': None,
            'signal': None,
            'direction': None,
            'consensus_direction': None,
            'consensus_strength_pct': None,
            'model_avg_direction_accuracy': round(model_avg['avg_da'], 2) if model_avg['avg_da'] else None,
            'model_avg_rmse': round(model_avg['avg_rmse'], 6) if model_avg['avg_rmse'] else None,
            'model_avg_r2': None,
            'is_best_overall_model': model_id == best_overall_model,
            'is_best_for_this_horizon': row.get('is_best_for_this_horizon', False),
            'best_da_for_this_horizon': round(best_h['da'], 2) if best_h['da'] else None,
            'best_model_for_this_horizon': best_h['model'],
            'image_path': f"results/ml_pipeline/figures/backtest_{model_id}_h{horizon}.png",
            'image_backtest': f"results/ml_pipeline/figures/backtest_{model_id}_h{horizon}.png",
            'image_forecast': None,
            'image_heatmap': 'results/ml_pipeline/figures/metrics_heatmap_da.png',
            'image_ranking': 'results/ml_pipeline/figures/model_ranking_da.png',
            'training_date': training_date,
            'generated_at': datetime.now().isoformat()
        })

    return rows


def generate_forecast_rows(
    forecast_data: Dict[str, Any],
    date_dir: Path,
    backtest_metrics: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Generate forecast rows for the BI dashboard CSV from a single forecast_data.json.

    IMPORTANT: Forward forecast images contain ALL 7 horizons in ONE image.
    Therefore, we generate ONE row per model per week (not 7 rows per horizon).
    The metrics shown are aggregated across all horizons.

    Args:
        forecast_data: Dictionary with forecast data from JSON
        date_dir: Path to the date directory
        backtest_metrics: Optional DataFrame with backtest metrics for DA values

    Returns:
        List of dictionaries representing CSV rows (one per model, not per horizon)
    """
    rows = []

    base_date = forecast_data.get('base_date')
    last_price = forecast_data.get('last_price')
    predictions = forecast_data.get('predictions', {})
    ensembles = forecast_data.get('ensembles', {})
    best_model_per_horizon = forecast_data.get('best_model_per_horizon', {})

    if not base_date:
        logger.warning(f"No base_date found in {date_dir}")
        return rows

    inference_date = pd.to_datetime(base_date)
    week_num = inference_date.isocalendar()[1]
    year = inference_date.year
    date_str = date_dir.name  # e.g., "2025-12-29"

    # Calculate model averages from backtest if available
    model_avg_metrics = {}

    if backtest_metrics is not None and not backtest_metrics.empty:
        for model_id in backtest_metrics['model'].unique():
            model_data = backtest_metrics[backtest_metrics['model'] == model_id]
            model_avg_metrics[model_id] = {
                'avg_da': model_data['direction_accuracy'].mean() * 100,
                'avg_rmse': model_data['rmse'].mean()
            }

    # Calculate overall consensus (across all horizons)
    all_directions = []
    for model_id, preds in predictions.items():
        for h_str, ret in preds.items():
            direction = 'UP' if ret > 0 else 'DOWN'
            all_directions.append(direction)

    up_count = all_directions.count('UP')
    down_count = all_directions.count('DOWN')
    total = len(all_directions) if all_directions else 1
    if up_count >= down_count:
        overall_consensus = {
            'direction': 'BULLISH',
            'strength': round((up_count / total) * 100, 0)
        }
    else:
        overall_consensus = {
            'direction': 'BEARISH',
            'strength': round((down_count / total) * 100, 0)
        }

    # Process individual model predictions - ONE row per model (not per horizon)
    for model_id, preds in predictions.items():
        if model_id not in INDIVIDUAL_MODELS:
            continue

        # Calculate aggregated metrics across all horizons
        returns = list(preds.values())
        avg_return = np.mean(returns) if returns else None

        # Use horizon 1 for primary display values
        h1_return = preds.get('1') or preds.get(1)

        # Calculate predicted price based on average return
        predicted_price = last_price * (1 + avg_return) if last_price and avg_return else None
        price_change = predicted_price - last_price if last_price and predicted_price else None
        price_change_pct = avg_return * 100 if avg_return else None

        # Determine overall direction and signal based on average
        if avg_return is not None:
            if abs(avg_return) < 0.005:  # Less than 0.5%
                signal = 'NEUTRAL'
            elif avg_return > 0:
                signal = 'BUY'
            else:
                signal = 'SELL'
            direction = 'UP' if avg_return > 0 else 'DOWN'
        else:
            signal = None
            direction = None

        # Get model average metrics
        model_avg = model_avg_metrics.get(model_id, {'avg_da': None, 'avg_rmse': None})

        # Count how many horizons this model is best for
        best_horizon_count = sum(1 for h, m in best_model_per_horizon.items() if m == model_id)

        rows.append({
            'record_id': f"FF_{inference_date.strftime('%Y%m%d')}_{model_id}",
            'view_type': 'forward_forecast',
            'model_id': model_id,
            'model_name': get_model_display_name(model_id),
            'model_type': get_model_type(model_id),
            'horizon_days': None,  # All horizons in one image
            'horizon_label': 'All (1-30)',
            'horizon_category': 'all_horizons',
            'inference_week': week_num,
            'inference_date': str(inference_date.date()),
            'direction_accuracy': round(model_avg['avg_da'], 2) if model_avg['avg_da'] else None,
            'rmse': round(model_avg['avg_rmse'], 6) if model_avg['avg_rmse'] else None,
            'r2': None,
            'mae': None,
            'base_price': round(last_price, 2) if last_price else None,
            'predicted_price': round(predicted_price, 2) if predicted_price else None,
            'price_change': round(price_change, 2) if price_change else None,
            'price_change_pct': round(price_change_pct, 2) if price_change_pct else None,
            'predicted_return_pct': round(avg_return * 100, 2) if avg_return else None,
            'signal': signal,
            'direction': direction,
            'consensus_direction': overall_consensus['direction'],
            'consensus_strength_pct': overall_consensus['strength'],
            'model_avg_direction_accuracy': round(model_avg['avg_da'], 2) if model_avg['avg_da'] else None,
            'model_avg_rmse': round(model_avg['avg_rmse'], 6) if model_avg['avg_rmse'] else None,
            'model_avg_r2': None,
            'is_best_overall_model': False,
            'is_best_for_this_horizon': best_horizon_count > 0,
            'best_horizons_count': best_horizon_count,
            'best_da_for_this_horizon': None,
            'best_model_for_this_horizon': None,
            'image_path': f"forecasts/{date_str}/forward_forecast_{model_id}.png",
            'image_backtest': None,
            'image_forecast': f"forecasts/{date_str}/forward_forecast_{model_id}.png",
            'image_heatmap': 'results/ml_pipeline/figures/metrics_heatmap_da.png',
            'image_ranking': 'results/ml_pipeline/figures/model_ranking_da.png',
            'training_date': None,
            'generated_at': datetime.now().isoformat()
        })

    # Process ensemble predictions - ONE row per ensemble (not per horizon)
    for ensemble_name, preds in ensembles.items():
        # Calculate aggregated metrics across all horizons
        returns = list(preds.values())
        avg_return = np.mean(returns) if returns else None

        # Calculate predicted price based on average return
        predicted_price = last_price * (1 + avg_return) if last_price and avg_return else None
        price_change = predicted_price - last_price if last_price and predicted_price else None
        price_change_pct = avg_return * 100 if avg_return else None

        # Determine overall direction and signal based on average
        if avg_return is not None:
            if abs(avg_return) < 0.005:
                signal = 'NEUTRAL'
            elif avg_return > 0:
                signal = 'BUY'
            else:
                signal = 'SELL'
            direction = 'UP' if avg_return > 0 else 'DOWN'
        else:
            signal = None
            direction = None

        rows.append({
            'record_id': f"FF_{inference_date.strftime('%Y%m%d')}_{ensemble_name}",
            'view_type': 'forward_forecast',
            'model_id': ensemble_name,
            'model_name': get_model_display_name(ensemble_name),
            'model_type': 'ensemble',
            'horizon_days': None,  # All horizons in one image
            'horizon_label': 'All (1-30)',
            'horizon_category': 'all_horizons',
            'inference_week': week_num,
            'inference_date': str(inference_date.date()),
            'direction_accuracy': None,  # Ensembles don't have individual backtest metrics
            'rmse': None,
            'r2': None,
            'mae': None,
            'base_price': round(last_price, 2) if last_price else None,
            'predicted_price': round(predicted_price, 2) if predicted_price else None,
            'price_change': round(price_change, 2) if price_change else None,
            'price_change_pct': round(price_change_pct, 2) if price_change_pct else None,
            'predicted_return_pct': round(avg_return * 100, 2) if avg_return else None,
            'signal': signal,
            'direction': direction,
            'consensus_direction': overall_consensus['direction'],
            'consensus_strength_pct': overall_consensus['strength'],
            'model_avg_direction_accuracy': None,
            'model_avg_rmse': None,
            'model_avg_r2': None,
            'is_best_overall_model': False,
            'is_best_for_this_horizon': False,
            'best_horizons_count': 0,
            'best_da_for_this_horizon': None,
            'best_model_for_this_horizon': None,
            'image_path': f"forecasts/{date_str}/forward_forecast_{ensemble_name}.png",
            'image_backtest': None,
            'image_forecast': f"forecasts/{date_str}/forward_forecast_{ensemble_name}.png",
            'image_heatmap': 'results/ml_pipeline/figures/metrics_heatmap_da.png',
            'image_ranking': 'results/ml_pipeline/figures/model_ranking_da.png',
            'training_date': None,
            'generated_at': datetime.now().isoformat()
        })

    return rows


def regenerate_bi_dashboard(
    output_file: Optional[Path] = None,
    copy_to_frontend: bool = True
) -> Dict[str, Any]:
    """
    Main function to regenerate the bi_dashboard_unified.csv with all data.

    Args:
        output_file: Optional custom output file path
        copy_to_frontend: Whether to copy to frontend/public

    Returns:
        Dictionary with generation results
    """
    logger.info("=" * 70)
    logger.info("REGENERATING BI DASHBOARD CSV")
    logger.info("=" * 70)

    # Setup directories
    forecasts_dir = PROJECT_ROOT / 'outputs' / 'forecasts'
    runs_dir = PROJECT_ROOT / 'outputs' / 'runs'
    frontend_dir = PROJECT_ROOT / 'frontend' / 'public'

    if output_file is None:
        output_file = frontend_dir / 'bi_dashboard_unified.csv'
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # 1. Load latest backtest metrics
    logger.info("\n[1/3] Loading backtest metrics...")
    backtest_metrics_file = find_latest_backtest_metrics(runs_dir)
    backtest_metrics = None
    training_date = datetime.now().strftime('%Y-%m-%d')

    if backtest_metrics_file:
        backtest_metrics = load_backtest_metrics(backtest_metrics_file)
        # Extract training date from directory name
        run_dir_name = backtest_metrics_file.parent.parent.name
        try:
            training_date = datetime.strptime(run_dir_name[:8], '%Y%m%d').strftime('%Y-%m-%d')
        except ValueError:
            pass

        # Generate backtest rows
        backtest_rows = generate_backtest_rows(backtest_metrics, training_date)
        all_rows.extend(backtest_rows)
        logger.info(f"  Generated {len(backtest_rows)} backtest rows")
    else:
        logger.warning("  No backtest metrics found - skipping backtest data")

    # 2. Scan forecast directories
    logger.info("\n[2/3] Scanning forecast directories...")
    date_dirs = scan_forecast_directories(forecasts_dir)

    # 3. Process each forecast directory
    logger.info("\n[3/3] Processing forecast data...")
    forecast_row_count = 0

    for date_dir in date_dirs:
        json_file = date_dir / 'forecast_data.json'
        try:
            forecast_data = load_forecast_data(json_file)
            forecast_rows = generate_forecast_rows(forecast_data, date_dir, backtest_metrics)
            all_rows.extend(forecast_rows)
            forecast_row_count += len(forecast_rows)
            logger.info(f"  Processed {date_dir.name}: {len(forecast_rows)} rows")
        except Exception as e:
            logger.error(f"  Error processing {date_dir.name}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Define column order
    column_order = [
        'record_id', 'view_type', 'model_id', 'model_name', 'model_type',
        'horizon_days', 'horizon_label', 'horizon_category',
        'inference_week', 'inference_date',
        'direction_accuracy', 'rmse', 'r2', 'mae',
        'base_price', 'predicted_price', 'price_change', 'price_change_pct',
        'predicted_return_pct', 'signal', 'direction',
        'consensus_direction', 'consensus_strength_pct',
        'model_avg_direction_accuracy', 'model_avg_rmse', 'model_avg_r2',
        'is_best_overall_model', 'is_best_for_this_horizon', 'best_horizons_count',
        'best_da_for_this_horizon', 'best_model_for_this_horizon',
        'image_path', 'image_backtest', 'image_forecast', 'image_heatmap', 'image_ranking',
        'training_date', 'generated_at'
    ]

    # Reorder columns (only existing ones)
    existing_cols = [c for c in column_order if c in df.columns]
    df = df[existing_cols]

    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"\nCSV saved: {output_file}")

    # Copy to frontend if requested and different from output_file
    frontend_file = frontend_dir / 'bi_dashboard_unified.csv'
    if copy_to_frontend and output_file.resolve() != frontend_file.resolve():
        try:
            frontend_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(frontend_file, index=False)
            logger.info(f"CSV copied to frontend: {frontend_file}")
        except Exception as e:
            logger.warning(f"Could not copy to frontend: {e}")

    # Also copy to outputs/bi for backup
    bi_dir = PROJECT_ROOT / 'outputs' / 'bi'
    bi_dir.mkdir(parents=True, exist_ok=True)
    bi_file = bi_dir / 'bi_dashboard_unified.csv'
    df.to_csv(bi_file, index=False)
    logger.info(f"CSV backup saved: {bi_file}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("REGENERATION COMPLETE")
    logger.info("=" * 70)

    # Count unique values
    unique_models = df['model_id'].nunique()
    unique_weeks = df[df['view_type'] == 'forward_forecast']['inference_date'].nunique()
    backtest_count = len(df[df['view_type'] == 'backtest'])
    forecast_count = len(df[df['view_type'] == 'forward_forecast'])

    logger.info(f"Total rows: {len(df)}")
    logger.info(f"  - Backtest rows: {backtest_count}")
    logger.info(f"  - Forecast rows: {forecast_count}")
    logger.info(f"  - Unique models: {unique_models}")
    logger.info(f"  - Unique forecast dates: {unique_weeks}")
    logger.info("=" * 70)

    return {
        'output_file': str(output_file),
        'total_rows': len(df),
        'backtest_rows': backtest_count,
        'forecast_rows': forecast_count,
        'unique_models': unique_models,
        'unique_forecast_dates': unique_weeks,
        'generated_at': datetime.now().isoformat()
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Regenerate bi_dashboard_unified.csv with all available data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backend/scripts/regenerate_bi_dashboard.py
  python backend/scripts/regenerate_bi_dashboard.py --verbose
  python backend/scripts/regenerate_bi_dashboard.py --output ./my_output.csv
  python backend/scripts/regenerate_bi_dashboard.py --no-frontend
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Custom output file path (default: frontend/public/bi_dashboard_unified.csv)'
    )

    parser.add_argument(
        '--no-frontend',
        action='store_true',
        help='Do not copy to frontend/public directory'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose/debug logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = regenerate_bi_dashboard(
            output_file=args.output,
            copy_to_frontend=not args.no_frontend
        )

        print(f"\nResult Summary:")
        print(f"  Output file: {result['output_file']}")
        print(f"  Total rows: {result['total_rows']}")
        print(f"  Backtest rows: {result['backtest_rows']}")
        print(f"  Forecast rows: {result['forecast_rows']}")
        print(f"  Unique models: {result['unique_models']}")
        print(f"  Unique forecast dates: {result['unique_forecast_dates']}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
