#!/usr/bin/env python
"""
Script para actualizar datos desde la base de datos y generar forward forecasts.
"""

import sys
import os
import json
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, '/opt/airflow/dags')

# Constants
HORIZONS = [1, 5, 10, 15, 20, 25, 30]
ML_MODELS = ['ridge', 'bayesian_ridge', 'ard', 'xgboost_pure', 'lightgbm_pure',
             'catboost_pure', 'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost']

# Feature columns (from training)
FEATURE_COLS = None  # Will be loaded from file


def update_training_data():
    """Update training temp files from database."""
    logger.info("Updating training data from database...")

    from utils.dag_common import get_db_connection

    conn = get_db_connection()
    df = pd.read_sql('SELECT * FROM core.features_ml ORDER BY date', conn)
    conn.close()

    logger.info(f"Loaded {len(df)} rows from database")

    # Set date as index
    df['date'] = pd.to_datetime(df['date'])
    df = df.rename(columns={'date': 'Date', 'close_price': 'Close'})
    df = df.set_index('Date')

    # Save to training temp
    temp_dir = Path('/opt/airflow/outputs/training_temp')
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Get feature columns (exclude Date and Close)
    with open(temp_dir / 'feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Save updated features
    df.to_pickle(temp_dir / 'df_features_updated.pkl')

    # Save prices
    prices = df['Close'] if 'Close' in df.columns else df['close']
    prices.to_pickle(temp_dir / 'prices_updated.pkl')

    logger.info(f"Updated data saved. Date range: {df.index.min()} to {df.index.max()}")

    return df, prices, feature_cols


def load_model_results(runs_dir: Path) -> pd.DataFrame:
    """Load the most recent model results with metrics."""
    for run_dir in sorted(runs_dir.glob('20*'), reverse=True):
        results_file = run_dir / 'data' / 'model_results.csv'
        if results_file.exists():
            logger.info(f"Loading results from: {results_file}")
            return pd.read_csv(results_file)
    raise FileNotFoundError("No model_results.csv found")


def load_trained_models(runs_dir: Path) -> Dict[str, Dict[int, Any]]:
    """Load all trained models from the run with the most models."""
    best_count = 0
    best_dir = None

    for run_dir in sorted(runs_dir.glob('20*'), reverse=True):
        models_dir = run_dir / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            if len(model_files) > best_count:
                best_count = len(model_files)
                best_dir = models_dir

    if best_dir is None:
        raise FileNotFoundError("No trained models found")

    logger.info(f"Loading {best_count} models from: {best_dir}")

    models = {}
    for model_file in best_dir.glob('*.pkl'):
        parts = model_file.stem.rsplit('_h', 1)
        if len(parts) == 2:
            model_name = parts[0]
            try:
                horizon = int(parts[1])
            except ValueError:
                continue

            if model_name not in models:
                models[model_name] = {}

            with open(model_file, 'rb') as f:
                models[model_name][horizon] = pickle.load(f)

    logger.info(f"Loaded {len(models)} model types")
    return models


def get_best_model_per_horizon(df_results: pd.DataFrame) -> Dict[int, str]:
    """Get the best model for each horizon based on direction accuracy."""
    best_models = {}
    for h in HORIZONS:
        h_data = df_results[df_results['horizon'] == h]
        if len(h_data) > 0:
            best_idx = h_data['direction_accuracy'].idxmax()
            best_models[h] = h_data.loc[best_idx, 'model']
    return best_models


def get_top_n_models(df_results: pd.DataFrame, n: int) -> List[str]:
    """Get top N models by average direction accuracy."""
    avg_da = df_results.groupby('model')['direction_accuracy'].mean().sort_values(ascending=False)
    return avg_da.head(n).index.tolist()


def generate_predictions(
    models: Dict[str, Dict[int, Any]],
    X_latest: np.ndarray,
    feature_cols: List[str]
) -> Dict[str, Dict[int, float]]:
    """Generate predictions from all models for all horizons."""
    predictions = {}

    for model_name, model_horizons in models.items():
        predictions[model_name] = {}

        for horizon, model_data in model_horizons.items():
            try:
                if isinstance(model_data, dict):
                    model = model_data.get('model')
                    scaler = model_data.get('scaler')
                else:
                    model = model_data
                    scaler = None

                if model is None:
                    continue

                X = X_latest.copy()
                if scaler is not None:
                    X = scaler.transform(X.reshape(1, -1))
                else:
                    X = X.reshape(1, -1)

                pred = model.predict(X)[0]
                predictions[model_name][horizon] = float(pred)

            except Exception as e:
                logger.warning(f"Error predicting {model_name} H={horizon}: {e}")

    return predictions


def create_ensembles(
    predictions: Dict[str, Dict[int, float]],
    best_model_per_horizon: Dict[int, str],
    top_3_models: List[str],
    top_6_models: List[str]
) -> Dict[str, Dict[int, float]]:
    """Create all ensemble predictions."""
    ensembles = {}

    # Best-of-Breed
    best_of_breed = {}
    for horizon, best_model in best_model_per_horizon.items():
        if best_model in predictions and horizon in predictions[best_model]:
            best_of_breed[horizon] = predictions[best_model][horizon]
    ensembles['best_of_breed'] = best_of_breed

    # Top-3
    top_3 = {}
    for h in HORIZONS:
        values = [predictions[m][h] for m in top_3_models if m in predictions and h in predictions[m]]
        if values:
            top_3[h] = float(np.mean(values))
    ensembles['top_3'] = top_3

    # Top-6 Mean
    top_6 = {}
    for h in HORIZONS:
        values = [predictions[m][h] for m in top_6_models if m in predictions and h in predictions[m]]
        if values:
            top_6[h] = float(np.mean(values))
    ensembles['top_6_mean'] = top_6

    return ensembles


def generate_all_model_comparison_image(
    prices: pd.Series,
    predictions: Dict[str, Dict[int, float]],
    ensembles: Dict[str, Dict[int, float]],
    output_dir: Path,
    base_date: str
):
    """Generate a single comprehensive image with ALL models and ensembles."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig = plt.figure(figsize=(20, 16))

    # Create grid
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.2)

    ax_main = fig.add_subplot(gs[0, :])  # Full width for main comparison
    ax_returns = fig.add_subplot(gs[1, 0])  # Individual models returns
    ax_ensemble = fig.add_subplot(gs[1, 1])  # Ensemble returns
    ax_table = fig.add_subplot(gs[2, :])  # Summary table

    # Get last 60 days and last price
    last_60 = prices.iloc[-60:]
    last_date = prices.index[-1]
    last_price = prices.iloc[-1]

    # Calculate volatility
    returns = np.log(prices / prices.shift(1)).dropna()
    hist_vol = returns.std()

    # Color maps
    model_colors = {
        'ridge': '#1f77b4',
        'bayesian_ridge': '#aec7e8',
        'ard': '#ff7f0e',
        'xgboost_pure': '#2ca02c',
        'lightgbm_pure': '#98df8a',
        'catboost_pure': '#d62728',
        'hybrid_xgboost': '#9467bd',
        'hybrid_lightgbm': '#c5b0d5',
        'hybrid_catboost': '#8c564b',
        'best_of_breed': '#e377c2',
        'top_3': '#7f7f7f',
        'top_6_mean': '#bcbd22'
    }

    # Main plot: Price forecasts
    ax_main.plot(last_60.index, last_60.values, 'b-', linewidth=2.5, label='Historical', zorder=10)
    ax_main.axvline(x=last_date, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Plot each model's forecast
    all_forecasts = {**predictions, **ensembles}

    for model_name, forecasts in all_forecasts.items():
        if not forecasts:
            continue

        sorted_horizons = sorted(forecasts.keys())
        forecast_dates = [last_date + timedelta(days=h) for h in sorted_horizons]

        cumulative_return = 0
        forecast_prices = []
        for h in sorted_horizons:
            cumulative_return += forecasts[h]
            forecast_prices.append(last_price * np.exp(cumulative_return))

        color = model_colors.get(model_name, '#333333')
        linestyle = '-' if model_name in ensembles else '--'
        linewidth = 2.5 if model_name in ensembles else 1.5
        alpha = 1.0 if model_name in ensembles else 0.7

        label = model_name.replace('_', ' ').title()
        ax_main.plot(forecast_dates, forecast_prices, linestyle, color=color,
                    linewidth=linewidth, alpha=alpha, label=label, marker='o', markersize=4)

    ax_main.set_title(f'Forward Forecast Comparison - All Models\nBase Date: {base_date} | Last Price: ${last_price:,.2f}',
                     fontsize=14, fontweight='bold')
    ax_main.set_ylabel('USD/COP Price', fontsize=12)
    ax_main.legend(loc='upper left', ncol=3, fontsize=9)
    ax_main.grid(True, alpha=0.3)
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Returns bar chart - Individual models
    x = np.arange(len(HORIZONS))
    width = 0.08

    for i, (model_name, forecasts) in enumerate(predictions.items()):
        if model_name in ensembles:
            continue
        returns_pct = [forecasts.get(h, 0) * 100 for h in HORIZONS]
        color = model_colors.get(model_name, '#333333')
        ax_returns.bar(x + i * width, returns_pct, width, label=model_name.replace('_', ' ').title(),
                      color=color, alpha=0.8)

    ax_returns.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_returns.set_xlabel('Horizon (days)')
    ax_returns.set_ylabel('Expected Return (%)')
    ax_returns.set_title('Individual Model Returns', fontsize=11)
    ax_returns.set_xticks(x + width * 4)
    ax_returns.set_xticklabels([f'H={h}' for h in HORIZONS])
    ax_returns.legend(loc='upper right', fontsize=7, ncol=2)
    ax_returns.grid(True, alpha=0.3, axis='y')

    # Returns bar chart - Ensembles
    x = np.arange(len(HORIZONS))
    width = 0.25

    for i, (name, forecasts) in enumerate(ensembles.items()):
        returns_pct = [forecasts.get(h, 0) * 100 for h in HORIZONS]
        color = model_colors.get(name, '#333333')
        ax_ensemble.bar(x + i * width, returns_pct, width, label=name.replace('_', ' ').title(),
                       color=color, alpha=0.9)

    ax_ensemble.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_ensemble.set_xlabel('Horizon (days)')
    ax_ensemble.set_ylabel('Expected Return (%)')
    ax_ensemble.set_title('Ensemble Returns Comparison', fontsize=11)
    ax_ensemble.set_xticks(x + width)
    ax_ensemble.set_xticklabels([f'H={h}' for h in HORIZONS])
    ax_ensemble.legend(loc='upper right', fontsize=9)
    ax_ensemble.grid(True, alpha=0.3, axis='y')

    # Summary table
    ax_table.axis('off')

    # Create table data
    table_data = []
    headers = ['Model'] + [f'H={h}' for h in HORIZONS] + ['Avg']

    for model_name, forecasts in all_forecasts.items():
        if not forecasts:
            continue
        row = [model_name.replace('_', ' ').title()]
        for h in HORIZONS:
            val = forecasts.get(h, 0) * 100
            row.append(f'{val:+.2f}%')
        avg = np.mean([forecasts.get(h, 0) * 100 for h in HORIZONS])
        row.append(f'{avg:+.2f}%')
        table_data.append(row)

    # Handle empty table case
    if not table_data:
        ax_table.text(0.5, 0.5, 'No predictions available', ha='center', va='center',
                     fontsize=14, color='red')
        logger.warning("No predictions available for table")
        plt.suptitle(f'Complete Forward Forecast Analysis\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = output_dir / 'all_models_complete_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return save_path

    # Sort by average return
    table_data.sort(key=lambda x: float(x[-1].replace('%', '').replace('+', '')), reverse=True)

    table = ax_table.table(cellText=table_data, colLabels=headers,
                          loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code the cells
    for i, row in enumerate(table_data):
        for j in range(1, len(row)):
            val = float(row[j].replace('%', '').replace('+', ''))
            if val > 0:
                table[(i+1, j)].set_facecolor('#d4edda')
            elif val < -2:
                table[(i+1, j)].set_facecolor('#f8d7da')
            else:
                table[(i+1, j)].set_facecolor('#fff3cd')

    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#343a40')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    plt.suptitle(f'Complete Forward Forecast Analysis\nGenerated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = output_dir / 'all_models_complete_comparison.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logger.info(f"Saved complete comparison: {save_path}")
    return save_path


def run_forecast_for_date(
    base_date: str,
    df_features: pd.DataFrame,
    prices: pd.Series,
    feature_cols: List[str],
    models: Dict[str, Dict[int, Any]],
    df_results: pd.DataFrame,
    best_model_per_horizon: Dict[int, str],
    top_3_models: List[str],
    top_6_models: List[str]
):
    """Run forecast for a specific base date."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FORECAST FOR BASE DATE: {base_date}")
    logger.info(f"{'='*60}")

    # Filter data up to base_date
    base_dt = pd.to_datetime(base_date)
    df_filtered = df_features[df_features.index <= base_dt].copy()
    prices_filtered = prices[prices.index <= base_dt].copy()

    if len(df_filtered) == 0:
        logger.error(f"No data available for {base_date}")
        return None

    last_date = df_filtered.index[-1]
    last_price = prices_filtered.iloc[-1]

    logger.info(f"Last available date: {last_date}")
    logger.info(f"Last price: {last_price:.2f}")

    # Exclude target columns (they start with 'target_') - we're predicting these
    target_cols = [c for c in df_filtered.columns if c.startswith('target_')]
    if target_cols:
        logger.info(f"Excluding {len(target_cols)} target columns from features")
        df_filtered = df_filtered.drop(columns=target_cols)

    # CRITICAL: Models were trained with ALL features (29 columns)
    # We MUST fill NaN values with the last known value from the ENTIRE dataset
    # Not just from the filtered data up to base_date

    # First, get the full dataset for filling (not filtered by date)
    df_full_for_fill = df_features.copy()
    if target_cols:
        df_full_for_fill = df_full_for_fill.drop(columns=[c for c in target_cols if c in df_full_for_fill.columns])

    # Check which columns have NaN in the latest row
    nan_before = df_filtered.iloc[-1].isnull().sum()
    nan_cols = df_filtered.columns[df_filtered.iloc[-1].isnull()].tolist()
    if nan_before > 0:
        logger.info(f"Found {nan_before} NaN values in latest row: {nan_cols}")

        # For each NaN column, get the last known value from the FULL dataset
        for col in nan_cols:
            if col in df_full_for_fill.columns:
                # Get the last non-NaN value from the full series
                col_series = df_full_for_fill[col].dropna()
                if len(col_series) > 0:
                    last_known_value = col_series.iloc[-1]
                    df_filtered[col] = df_filtered[col].fillna(last_known_value)
                    logger.info(f"  Filled {col} with last known value: {last_known_value:.4f}")
                else:
                    # If completely empty, use column mean or 0
                    df_filtered[col] = df_filtered[col].fillna(0)
                    logger.warning(f"  Filled {col} with 0 (no known values)")

        nan_after = df_filtered.iloc[-1].isnull().sum()
        logger.info(f"After filling: {nan_after} NaN values remaining")

    # Get latest features - use ALL feature columns (must match training)
    available_cols = [c for c in feature_cols if c in df_filtered.columns and not c.startswith('target_')]
    logger.info(f"Using {len(available_cols)} features for prediction (models expect {len(feature_cols)} features)")

    if len(available_cols) != len([c for c in feature_cols if not c.startswith('target_')]):
        missing_cols = [c for c in feature_cols if c not in available_cols and not c.startswith('target_')]
        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")

    X_latest = df_filtered[available_cols].iloc[-1].values.astype(np.float64)

    # Final NaN check - if still NaN, replace with 0
    nan_mask = np.isnan(X_latest)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        logger.warning(f"Still {nan_count} NaN values after fill, replacing with 0")
        X_latest = np.nan_to_num(X_latest, nan=0.0)

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = generate_predictions(models, X_latest, available_cols)

    # Create ensembles
    logger.info("Creating ensembles...")
    ensembles = create_ensembles(predictions, best_model_per_horizon, top_3_models, top_6_models)

    # Output directory
    output_dir = Path(f'/opt/airflow/outputs/forecasts/{base_date}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations using existing module
    try:
        from src.visualization.forecast_plots import generate_all_forecast_plots

        # Add ensembles to predictions for standard plots
        all_predictions = {**predictions, **ensembles}

        generate_all_forecast_plots(
            prices=prices_filtered,
            forecasts_by_model=all_predictions,
            ensemble_forecasts=ensembles.get('best_of_breed'),
            output_dir=output_dir
        )
    except Exception as e:
        logger.warning(f"Standard plots error: {e}")

    # Generate combined all-models image
    logger.info("Generating combined all-models comparison...")
    generate_all_model_comparison_image(
        prices=prices_filtered,
        predictions=predictions,
        ensembles=ensembles,
        output_dir=output_dir,
        base_date=base_date
    )

    # Save forecast data
    forecast_data = {
        'base_date': base_date,
        'actual_last_date': str(last_date.date()),
        'last_price': float(last_price),
        'predictions': predictions,
        'ensembles': ensembles,
        'best_model_per_horizon': best_model_per_horizon,
        'top_3_models': top_3_models,
        'top_6_models': top_6_models,
        'generated_at': datetime.now().isoformat()
    }

    with open(output_dir / 'forecast_data.json', 'w') as f:
        json.dump(forecast_data, f, indent=2, default=str)

    # Upload to MinIO
    uploaded = upload_to_minio(output_dir, base_date)

    logger.info(f"Forecast complete for {base_date}")
    logger.info(f"Images uploaded to MinIO: {uploaded}")

    return forecast_data


def upload_to_minio(output_dir: Path, base_date: str) -> int:
    """Upload all images to MinIO."""
    try:
        from src.mlops.minio_client import MinioClient

        minio_client = MinioClient()
        minio_raw = minio_client._get_client()

        dt = datetime.strptime(base_date, '%Y-%m-%d')
        year = dt.year
        week = dt.isocalendar()[1]

        base_path = f"forecasts/{year}/week{week:02d}/{base_date}"
        minio_client.ensure_bucket('forecasts')

        uploaded = 0
        for img_file in output_dir.glob('*.png'):
            try:
                s3_path = f"{base_path}/{img_file.name}"
                minio_client.upload_model(
                    bucket='forecasts',
                    model_path=str(img_file),
                    s3_path=s3_path,
                    content_type='image/png'
                )
                uploaded += 1
            except Exception as e:
                logger.warning(f"Error uploading {img_file.name}: {e}")

        # Also upload JSON
        json_file = output_dir / 'forecast_data.json'
        if json_file.exists():
            s3_path = f"{base_path}/forecast_data.json"
            minio_client.upload_model(
                bucket='forecasts',
                model_path=str(json_file),
                s3_path=s3_path,
                content_type='application/json'
            )

        logger.info(f"Uploaded {uploaded} files to s3://forecasts/{base_path}/")
        return uploaded

    except Exception as e:
        logger.error(f"MinIO error: {e}")
        return 0


def main():
    logger.info("="*60)
    logger.info("UPDATE AND FORECAST PIPELINE")
    logger.info("="*60)

    # Paths
    runs_dir = Path('/opt/airflow/outputs/runs')
    temp_dir = Path('/opt/airflow/outputs/training_temp')

    # Step 1: Update training data from database
    logger.info("\n1. Updating training data from database...")
    df_features, prices, feature_cols = update_training_data()

    # Step 2: Load model results
    logger.info("\n2. Loading model results...")
    df_results = load_model_results(runs_dir)
    if df_results['direction_accuracy'].max() > 1:
        df_results['direction_accuracy'] = df_results['direction_accuracy'] / 100

    # Step 3: Determine best models
    logger.info("\n3. Determining best models...")
    best_model_per_horizon = get_best_model_per_horizon(df_results)
    top_3_models = get_top_n_models(df_results, 3)
    top_6_models = get_top_n_models(df_results, 6)

    logger.info(f"Best models: {best_model_per_horizon}")
    logger.info(f"Top 3: {top_3_models}")
    logger.info(f"Top 6: {top_6_models}")

    # Step 4: Load trained models
    logger.info("\n4. Loading trained models...")
    models = load_trained_models(runs_dir)

    # Step 5: Run forecasts for both dates
    dates_to_forecast = ['2025-12-29', '2026-01-05']

    for base_date in dates_to_forecast:
        run_forecast_for_date(
            base_date=base_date,
            df_features=df_features,
            prices=prices,
            feature_cols=feature_cols,
            models=models,
            df_results=df_results,
            best_model_per_horizon=best_model_per_horizon,
            top_3_models=top_3_models,
            top_6_models=top_6_models
        )

    logger.info("\n" + "="*60)
    logger.info("ALL FORECASTS COMPLETED")
    logger.info("="*60)


if __name__ == "__main__":
    main()
