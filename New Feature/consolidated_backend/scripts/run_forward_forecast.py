#!/usr/bin/env python
"""
Script para generar Forward Forecasts con multiples metodos de ensamble.

Ensembles:
1. Best-of-Breed: Cada horizonte usa el mejor modelo para ese horizonte
2. Top-3: Promedio de los 3 mejores modelos overall
3. Top-6: Promedio de los 6 mejores modelos overall

Uso:
    python run_forward_forecast.py [--base_date YYYY-MM-DD]
"""

import sys
import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import pickle

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants
HORIZONS = [1, 5, 10, 15, 20, 25, 30]
ML_MODELS = ['ridge', 'bayesian_ridge', 'ard', 'xgboost_pure', 'lightgbm_pure',
             'catboost_pure', 'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost']

# Try imports
try:
    from src.visualization.forecast_plots import generate_all_forecast_plots, ForecastPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization not available: {e}")
    VISUALIZATION_AVAILABLE = False

try:
    from src.mlops.minio_client import MinioClient, MODELS_BUCKET
    MINIO_AVAILABLE = True
except ImportError:
    logger.warning("MinIO client not available")
    MINIO_AVAILABLE = False


def load_model_results(runs_dir: Path) -> pd.DataFrame:
    """Load the most recent model results with metrics."""
    # Find most recent run with results
    for run_dir in sorted(runs_dir.glob('20*'), reverse=True):
        results_file = run_dir / 'data' / 'model_results.csv'
        if results_file.exists():
            logger.info(f"Loading results from: {results_file}")
            return pd.read_csv(results_file)

    raise FileNotFoundError("No model_results.csv found in any run directory")


def load_trained_models(runs_dir: Path) -> Dict[str, Dict[int, Any]]:
    """Load all trained models from the run with the most models."""
    best_models = {}
    best_count = 0
    best_dir = None

    # Find run with most models (not just most recent)
    for run_dir in sorted(runs_dir.glob('20*'), reverse=True):
        models_dir = run_dir / 'models'
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            if len(model_files) > best_count:
                best_count = len(model_files)
                best_dir = models_dir

    if best_dir is None:
        raise FileNotFoundError("No trained models found in any run directory")

    logger.info(f"Loading {best_count} models from: {best_dir}")

    models = {}
    for model_file in best_dir.glob('*.pkl'):
        # Parse filename: model_name_h{horizon}.pkl
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


def load_feature_data(temp_dir: Path) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load feature data and prices from training temp directory."""
    df_features = pd.read_pickle(temp_dir / 'df_features.pkl')

    with open(temp_dir / 'feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Set Date as index if available
    if 'Date' in df_features.columns:
        df_features['Date'] = pd.to_datetime(df_features['Date'])
        df_features = df_features.set_index('Date')
    elif 'date' in df_features.columns:
        df_features['date'] = pd.to_datetime(df_features['date'])
        df_features = df_features.set_index('date')

    # Extract prices from Close column
    price_col = 'Close' if 'Close' in df_features.columns else 'close'
    prices = df_features[price_col].copy()

    logger.info(f"Loaded {len(df_features)} rows, {len(feature_cols)} features")
    logger.info(f"Date range: {df_features.index.min()} to {df_features.index.max()}")
    return df_features, prices, feature_cols


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
    """Get top N models by average direction accuracy across all horizons."""
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
                # Handle different model storage formats
                if isinstance(model_data, dict):
                    model = model_data.get('model')
                    scaler = model_data.get('scaler')
                else:
                    model = model_data
                    scaler = None

                if model is None:
                    continue

                # Apply scaler if present
                X = X_latest.copy()
                if scaler is not None:
                    X = scaler.transform(X.reshape(1, -1))
                else:
                    X = X.reshape(1, -1)

                # Predict
                pred = model.predict(X)[0]
                predictions[model_name][horizon] = float(pred)

            except Exception as e:
                logger.warning(f"Error predicting {model_name} H={horizon}: {e}")

    return predictions


def create_best_of_breed_ensemble(
    predictions: Dict[str, Dict[int, float]],
    best_model_per_horizon: Dict[int, str]
) -> Dict[int, float]:
    """
    Create Best-of-Breed ensemble: use best model for each horizon.
    """
    ensemble = {}
    for horizon, best_model in best_model_per_horizon.items():
        if best_model in predictions and horizon in predictions[best_model]:
            ensemble[horizon] = predictions[best_model][horizon]
            logger.info(f"  H={horizon}: {best_model} -> {ensemble[horizon]:.6f}")
    return ensemble


def create_top_n_ensemble(
    predictions: Dict[str, Dict[int, float]],
    top_models: List[str],
    ensemble_name: str
) -> Dict[int, float]:
    """
    Create ensemble as average of top N models.
    """
    ensemble = {}
    for horizon in HORIZONS:
        values = []
        for model in top_models:
            if model in predictions and horizon in predictions[model]:
                values.append(predictions[model][horizon])

        if values:
            ensemble[horizon] = float(np.mean(values))
            logger.info(f"  H={horizon}: mean of {len(values)} models -> {ensemble[horizon]:.6f}")

    return ensemble


def generate_forecast_images(
    prices: pd.Series,
    predictions: Dict[str, Dict[int, float]],
    ensembles: Dict[str, Dict[int, float]],
    output_dir: Path,
    base_date: str
):
    """Generate all forecast visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # Calculate historical volatility
    returns = np.log(prices / prices.shift(1)).dropna()
    historical_volatility = returns.std()

    # Get last price and date
    last_date = prices.index[-1]
    last_price = prices.iloc[-1]

    logger.info(f"Last date: {last_date}, Last price: {last_price:.2f}")
    logger.info(f"Historical volatility: {historical_volatility:.4f}")

    # Combine all forecasts for visualization
    all_forecasts = dict(predictions)
    all_forecasts.update(ensembles)

    if VISUALIZATION_AVAILABLE:
        try:
            # Use the existing forecast_plots module
            for ensemble_name, ensemble_forecasts in ensembles.items():
                # Add to predictions for visualization
                predictions[ensemble_name] = ensemble_forecasts

            files = generate_all_forecast_plots(
                prices=prices,
                forecasts_by_model=predictions,
                ensemble_forecasts=ensembles.get('best_of_breed'),
                all_ensembles=ensembles,  # Pass all 3 ensemble types
                output_dir=output_dir,
                historical_volatility=historical_volatility
            )
            generated_files.extend(files)

        except Exception as e:
            logger.warning(f"Error with standard plots: {e}")
            # Fall through to manual plotting

    # Generate additional ensemble-specific plots manually
    for ensemble_name, ensemble_forecasts in ensembles.items():
        try:
            _plot_ensemble_forecast(
                prices=prices,
                forecasts=ensemble_forecasts,
                ensemble_name=ensemble_name,
                historical_volatility=historical_volatility,
                output_dir=output_dir,
                base_date=base_date
            )
            generated_files.append(output_dir / f'ensemble_{ensemble_name}.png')
        except Exception as e:
            logger.warning(f"Error plotting {ensemble_name}: {e}")

    # Generate summary comparison of all ensembles
    _plot_ensemble_comparison(
        prices=prices,
        ensembles=ensembles,
        historical_volatility=historical_volatility,
        output_dir=output_dir,
        base_date=base_date
    )
    generated_files.append(output_dir / 'ensemble_comparison.png')

    return generated_files


def _plot_ensemble_forecast(
    prices: pd.Series,
    forecasts: Dict[int, float],
    ensemble_name: str,
    historical_volatility: float,
    output_dir: Path,
    base_date: str
):
    """Plot a single ensemble forecast with confidence intervals."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # Get last 60 days of historical prices
    last_60 = prices.iloc[-60:]
    last_date = prices.index[-1]
    last_price = prices.iloc[-1]

    # Plot historical prices
    ax1.plot(last_60.index, last_60.values, 'b-', linewidth=2, label='Historical')
    ax1.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')

    # Generate forecast prices
    sorted_horizons = sorted(forecasts.keys())
    forecast_dates = [last_date + timedelta(days=h) for h in sorted_horizons]
    forecast_prices = []

    cumulative_return = 0
    for h in sorted_horizons:
        cumulative_return += forecasts[h]
        forecast_price = last_price * np.exp(cumulative_return)
        forecast_prices.append(forecast_price)

    # Plot forecast line
    ax1.plot(forecast_dates, forecast_prices, 'g-', linewidth=2.5,
             marker='o', markersize=8, label=f'{ensemble_name.replace("_", " ").title()}')

    # Confidence intervals (90%)
    ci_upper = []
    ci_lower = []
    for i, h in enumerate(sorted_horizons):
        ci = 1.645 * historical_volatility * np.sqrt(h)
        cumulative_return = sum(forecasts[hh] for hh in sorted_horizons[:i+1])
        ci_upper.append(last_price * np.exp(cumulative_return + ci))
        ci_lower.append(last_price * np.exp(cumulative_return - ci))

    ax1.fill_between(forecast_dates, ci_lower, ci_upper, alpha=0.2, color='green', label='90% CI')

    # Formatting
    ax1.set_title(f'Forward Forecast: {ensemble_name.replace("_", " ").title()}\nBase Date: {base_date}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('USD/COP Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Bottom plot: Returns by horizon
    ax2.bar(sorted_horizons, [forecasts[h] * 100 for h in sorted_horizons],
            color=['green' if forecasts[h] > 0 else 'red' for h in sorted_horizons],
            alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Horizon (days)', fontsize=12)
    ax2.set_ylabel('Expected Return (%)', fontsize=12)
    ax2.set_title('Expected Returns by Horizon', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for h in sorted_horizons:
        val = forecasts[h] * 100
        ax2.annotate(f'{val:.2f}%', xy=(h, val), ha='center',
                    va='bottom' if val > 0 else 'top', fontsize=9)

    plt.tight_layout()
    save_path = output_dir / f'ensemble_{ensemble_name}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def _plot_ensemble_comparison(
    prices: pd.Series,
    ensembles: Dict[str, Dict[int, float]],
    historical_volatility: float,
    output_dir: Path,
    base_date: str
):
    """Plot comparison of all ensemble methods."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])

    # Get last 60 days
    last_60 = prices.iloc[-60:]
    last_date = prices.index[-1]
    last_price = prices.iloc[-1]

    # Plot historical
    ax1.plot(last_60.index, last_60.values, 'b-', linewidth=2, label='Historical')
    ax1.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)

    colors = {'best_of_breed': 'green', 'top_3': 'orange', 'top_6_mean': 'purple'}

    for ensemble_name, forecasts in ensembles.items():
        sorted_horizons = sorted(forecasts.keys())
        forecast_dates = [last_date + timedelta(days=h) for h in sorted_horizons]

        cumulative_return = 0
        forecast_prices = []
        for h in sorted_horizons:
            cumulative_return += forecasts[h]
            forecast_prices.append(last_price * np.exp(cumulative_return))

        color = colors.get(ensemble_name, 'gray')
        label = ensemble_name.replace('_', ' ').title()
        ax1.plot(forecast_dates, forecast_prices, '-', linewidth=2,
                marker='o', markersize=6, color=color, label=label)

    ax1.set_title(f'Ensemble Methods Comparison\nBase Date: {base_date}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('USD/COP Price', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Bottom: Returns comparison
    x = np.arange(len(HORIZONS))
    width = 0.25

    for i, (ensemble_name, forecasts) in enumerate(ensembles.items()):
        returns = [forecasts.get(h, 0) * 100 for h in HORIZONS]
        color = colors.get(ensemble_name, 'gray')
        label = ensemble_name.replace('_', ' ').title()
        ax2.bar(x + i * width, returns, width, label=label, color=color, alpha=0.7)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Horizon (days)', fontsize=12)
    ax2.set_ylabel('Expected Return (%)', fontsize=12)
    ax2.set_title('Expected Returns Comparison', fontsize=12)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f'H={h}' for h in HORIZONS])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = output_dir / 'ensemble_comparison.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: {save_path}")


def upload_to_minio(output_dir: Path, base_date: str) -> int:
    """Upload all generated images to MinIO."""
    if not MINIO_AVAILABLE:
        logger.warning("MinIO not available - skipping upload")
        return 0

    try:
        minio_client = MinioClient()
        minio_raw = minio_client._get_client()

        # Parse date for path
        dt = datetime.strptime(base_date, '%Y-%m-%d')
        year = dt.year
        week = dt.isocalendar()[1]

        # Base path in MinIO
        base_path = f"forecasts/{year}/week{week:02d}/{base_date}"

        # Ensure bucket exists
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
                logger.info(f"  Uploaded: {img_file.name}")
            except Exception as e:
                logger.warning(f"  Error uploading {img_file.name}: {e}")

        logger.info(f"Uploaded {uploaded} images to MinIO at forecasts/{year}/week{week:02d}/")
        return uploaded

    except Exception as e:
        logger.error(f"Error connecting to MinIO: {e}")
        return 0


def run_forecast(base_date: str = None):
    """
    Run forward forecast for a specific base date.

    Args:
        base_date: Date to use as last real date (YYYY-MM-DD format)
    """
    logger.info("=" * 60)
    logger.info("FORWARD FORECAST CON ENSEMBLES")
    logger.info("=" * 60)

    # Paths
    runs_dir = Path('/opt/airflow/outputs/runs')
    temp_dir = Path('/opt/airflow/outputs/training_temp')

    # Load model results for determining best models
    logger.info("\n1. Loading model results...")
    df_results = load_model_results(runs_dir)

    # Convert DA to 0-1 if needed
    if df_results['direction_accuracy'].max() > 1:
        df_results['direction_accuracy'] = df_results['direction_accuracy'] / 100

    # Determine best models
    logger.info("\n2. Determining best models...")
    best_model_per_horizon = get_best_model_per_horizon(df_results)
    top_3_models = get_top_n_models(df_results, 3)
    top_6_models = get_top_n_models(df_results, 6)

    logger.info(f"Best model per horizon:")
    for h, model in sorted(best_model_per_horizon.items()):
        da = df_results[(df_results['model'] == model) & (df_results['horizon'] == h)]['direction_accuracy'].values[0]
        logger.info(f"  H={h}: {model} (DA={da:.1%})")

    logger.info(f"\nTop 3 models overall: {top_3_models}")
    logger.info(f"Top 6 models overall: {top_6_models}")

    # Load trained models
    logger.info("\n3. Loading trained models...")
    try:
        models = load_trained_models(runs_dir)
        logger.info(f"Loaded {len(models)} model types")
    except FileNotFoundError:
        logger.error("No trained models found!")
        return False

    # Load feature data
    logger.info("\n4. Loading feature data...")
    df_features, prices, feature_cols = load_feature_data(temp_dir)

    # Handle base_date
    if base_date:
        base_dt = pd.to_datetime(base_date)
        # Filter data up to base_date
        prices = prices[prices.index <= base_dt]
        df_features = df_features[df_features.index <= base_dt]
        logger.info(f"Filtered data up to {base_date}")
    else:
        base_date = prices.index[-1].strftime('%Y-%m-%d')

    logger.info(f"Base date: {base_date}")
    logger.info(f"Last price: {prices.iloc[-1]:.2f}")

    # Get latest features
    X_latest = df_features[feature_cols].iloc[-1].values

    # Generate predictions
    logger.info("\n5. Generating predictions from all models...")
    predictions = generate_predictions(models, X_latest, feature_cols)

    logger.info(f"Generated predictions for {len(predictions)} models")

    # Create ensembles
    logger.info("\n6. Creating ensembles...")

    logger.info("\nBest-of-Breed ensemble (best model per horizon):")
    best_of_breed = create_best_of_breed_ensemble(predictions, best_model_per_horizon)

    logger.info("\nTop-3 ensemble (average of top 3 models):")
    top_3_ensemble = create_top_n_ensemble(predictions, top_3_models, "top_3")

    logger.info("\nTop-6 Mean ensemble (average of top 6 models):")
    top_6_ensemble = create_top_n_ensemble(predictions, top_6_models, "top_6_mean")

    ensembles = {
        'best_of_breed': best_of_breed,
        'top_3': top_3_ensemble,
        'top_6_mean': top_6_ensemble
    }

    # Generate visualizations
    logger.info("\n7. Generating visualizations...")
    output_dir = Path(f'/opt/airflow/outputs/forecasts/{base_date}')

    generated_files = generate_forecast_images(
        prices=prices,
        predictions=predictions,
        ensembles=ensembles,
        output_dir=output_dir,
        base_date=base_date
    )

    logger.info(f"Generated {len(generated_files)} images")

    # Upload to MinIO
    logger.info("\n8. Uploading to MinIO...")
    uploaded = upload_to_minio(output_dir, base_date)

    # Save forecast data as JSON
    forecast_data = {
        'base_date': base_date,
        'last_price': float(prices.iloc[-1]),
        'predictions': predictions,
        'ensembles': ensembles,
        'best_model_per_horizon': best_model_per_horizon,
        'top_3_models': top_3_models,
        'top_6_models': top_6_models,
        'generated_at': datetime.now().isoformat()
    }

    json_path = output_dir / 'forecast_data.json'
    with open(json_path, 'w') as f:
        json.dump(forecast_data, f, indent=2, default=str)
    logger.info(f"Saved forecast data to {json_path}")

    logger.info("\n" + "=" * 60)
    logger.info("FORECAST COMPLETADO")
    logger.info(f"  - Base date: {base_date}")
    logger.info(f"  - Models used: {len(predictions)}")
    logger.info(f"  - Ensembles: {list(ensembles.keys())}")
    logger.info(f"  - Images generated: {len(generated_files)}")
    logger.info(f"  - Images uploaded to MinIO: {uploaded}")
    logger.info("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate forward forecast with ensembles')
    parser.add_argument('--base_date', type=str, help='Base date (YYYY-MM-DD)')
    args = parser.parse_args()

    success = run_forecast(args.base_date)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
