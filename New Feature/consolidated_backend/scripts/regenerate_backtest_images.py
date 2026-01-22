#!/usr/bin/env python
"""
Script para regenerar imagenes de backtest y subirlas a MinIO.
Usa los resultados existentes del entrenamiento sin re-entrenar modelos.

Uso:
    python regenerate_backtest_images.py
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import visualization modules
try:
    from src.visualization.backtest_plotter import BacktestPlotter
    from src.visualization.model_comparison_plotter import ModelComparisonPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization modules not available: {e}")
    VISUALIZATION_AVAILABLE = False

# Try to import MinIO client
try:
    from src.mlops.minio_client import MinioClient, MODELS_BUCKET
    MINIO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MinIO client not available: {e}")
    MINIO_AVAILABLE = False


def find_latest_run_with_results():
    """Find the most recent run directory with model_results.csv."""
    runs_base = Path('/opt/airflow/outputs/runs')

    if not runs_base.exists():
        logger.error(f"Runs directory does not exist: {runs_base}")
        return None

    # Sort directories by name (which includes timestamp)
    runs = sorted(runs_base.glob('20*'), reverse=True)

    for run_dir in runs:
        results_file = run_dir / 'data' / 'model_results.csv'
        if results_file.exists():
            logger.info(f"Found results in: {run_dir}")
            return run_dir

    logger.error("No run directory found with model_results.csv")
    return None


def generate_backtest_images(output_dir: Path, df_results: pd.DataFrame):
    """
    Generate all backtest images from training results.

    Args:
        output_dir: Directory to save images
        df_results: DataFrame with model results

    Returns:
        Tuple of (backtest_count, comparison_count)
    """
    import matplotlib.pyplot as plt

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    backtest_count = 0
    comparison_count = 0

    # Initialize plotters if available
    if VISUALIZATION_AVAILABLE:
        model_plotter = ModelComparisonPlotter()
    else:
        model_plotter = None

    # Ensure DA is in 0-1 range
    if df_results['direction_accuracy'].max() > 1:
        df_results['direction_accuracy'] = df_results['direction_accuracy'] / 100

    # 1. Generate metrics heatmap (DA)
    logger.info("Generating metrics heatmap (DA)...")
    if model_plotter:
        try:
            model_plotter.plot_metrics_heatmap(
                df_results,
                metric='direction_accuracy',
                save_path=figures_dir / 'metrics_heatmap_da.png'
            )
            comparison_count += 1
        except Exception as e:
            logger.warning(f"Error generating DA heatmap with plotter: {e}")
            # Fallback to simple heatmap
            _generate_simple_heatmap(df_results, 'direction_accuracy',
                                     figures_dir / 'metrics_heatmap_da.png')
            comparison_count += 1
    else:
        _generate_simple_heatmap(df_results, 'direction_accuracy',
                                 figures_dir / 'metrics_heatmap_da.png')
        comparison_count += 1

    # 2. Generate model ranking
    logger.info("Generating model ranking...")
    if model_plotter:
        try:
            model_plotter.plot_model_ranking(
                df_results,
                metric='direction_accuracy',
                save_path=figures_dir / 'model_ranking_da.png'
            )
            comparison_count += 1
        except Exception as e:
            logger.warning(f"Error generating ranking with plotter: {e}")
            _generate_simple_ranking(df_results, figures_dir / 'model_ranking_da.png')
            comparison_count += 1
    else:
        _generate_simple_ranking(df_results, figures_dir / 'model_ranking_da.png')
        comparison_count += 1

    # 3. Generate RMSE heatmap if available
    if 'rmse' in df_results.columns:
        logger.info("Generating metrics heatmap (RMSE)...")
        if model_plotter:
            try:
                model_plotter.plot_metrics_heatmap(
                    df_results,
                    metric='rmse',
                    save_path=figures_dir / 'metrics_heatmap_rmse.png'
                )
                comparison_count += 1
            except Exception as e:
                logger.warning(f"Error generating RMSE heatmap: {e}")
                _generate_simple_heatmap(df_results, 'rmse',
                                         figures_dir / 'metrics_heatmap_rmse.png')
                comparison_count += 1
        else:
            _generate_simple_heatmap(df_results, 'rmse',
                                     figures_dir / 'metrics_heatmap_rmse.png')
            comparison_count += 1

    # 4. Generate individual backtest cards for each model/horizon
    logger.info("Generating individual backtest cards...")
    for _, row in df_results.iterrows():
        model_name = row['model']
        horizon = row['horizon']
        da = row.get('direction_accuracy', 0)
        rmse = row.get('rmse', 0)

        save_path = figures_dir / f'backtest_{model_name}_h{horizon}.png'
        _generate_backtest_card(model_name, horizon, da, rmse, save_path)
        backtest_count += 1

    logger.info(f"Generated {backtest_count} backtest images and {comparison_count} comparison images")
    return backtest_count, comparison_count


def _generate_simple_heatmap(df: pd.DataFrame, metric: str, save_path: Path):
    """Generate a simple heatmap using matplotlib."""
    import matplotlib.pyplot as plt

    # Pivot data for heatmap
    pivot = df.pivot_table(index='model', columns='horizon', values=metric, aggfunc='first')

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create heatmap
    cmap = 'RdYlGn' if metric == 'direction_accuracy' else 'RdYlGn_r'
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f'H={h}' for h in pivot.columns])
    ax.set_yticklabels(pivot.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric.replace('_', ' ').title())

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                fmt = '.1%' if metric == 'direction_accuracy' else '.4f'
                text = f'{val:{fmt}}'
                ax.text(j, i, text, ha='center', va='center', fontsize=8,
                       color='white' if val > 0.6 else 'black')

    ax.set_title(f'{metric.replace("_", " ").title()} by Model and Horizon')
    ax.set_xlabel('Horizon (days)')
    ax.set_ylabel('Model')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _generate_simple_ranking(df: pd.DataFrame, save_path: Path):
    """Generate a simple model ranking bar chart."""
    import matplotlib.pyplot as plt

    # Calculate average DA per model
    model_avg = df.groupby('model')['direction_accuracy'].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_avg)))
    bars = ax.barh(model_avg.index, model_avg.values, color=colors)

    # Add value labels
    for bar, val in zip(bars, model_avg.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=10)

    ax.set_xlabel('Average Direction Accuracy')
    ax.set_title('Model Ranking by Average Direction Accuracy')
    ax.set_xlim(0, max(model_avg.values) * 1.15)

    # Add 50% reference line
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
    ax.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _generate_backtest_card(model_name: str, horizon: int, da: float, rmse: float, save_path: Path):
    """Generate a simple backtest result card."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Background color based on DA
    if da > 0.55:
        bg_color = '#d4edda'  # Green
    elif da > 0.50:
        bg_color = '#fff3cd'  # Yellow
    else:
        bg_color = '#f8d7da'  # Red

    ax.set_facecolor(bg_color)

    # Model name
    ax.text(0.5, 0.75, f'{model_name.upper()}', fontsize=28, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Horizon
    ax.text(0.5, 0.55, f'Horizon: {horizon} days', fontsize=18,
            ha='center', transform=ax.transAxes)

    # Direction Accuracy
    da_color = 'green' if da > 0.55 else ('orange' if da > 0.50 else 'red')
    ax.text(0.5, 0.35, f'Direction Accuracy: {da:.1%}', fontsize=20,
            ha='center', transform=ax.transAxes, color=da_color, fontweight='bold')

    # RMSE
    ax.text(0.5, 0.20, f'RMSE: {rmse:.6f}', fontsize=14,
            ha='center', transform=ax.transAxes)

    # Timestamp
    ax.text(0.5, 0.05, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
            fontsize=10, ha='center', transform=ax.transAxes, color='gray')

    ax.axis('off')

    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=bg_color)
    plt.close(fig)


def upload_images_to_minio(figures_dir: Path, run_name: str):
    """
    Upload all images from figures_dir to MinIO.

    Args:
        figures_dir: Directory containing images
        run_name: Name of the run (used for path in MinIO)

    Returns:
        Number of images uploaded
    """
    if not MINIO_AVAILABLE:
        logger.warning("MinIO not available - cannot upload images")
        return 0

    try:
        minio_client = MinioClient()

        # Get year/month from current date
        now = datetime.now()
        year = now.year
        month = now.month
        timestamp = run_name if run_name else now.strftime('%Y%m%d_%H%M%S')

        # Base path in MinIO
        base_minio_path = f"{year}/month{month:02d}/{timestamp}/figures"

        # Ensure bucket exists
        minio_client.ensure_bucket(MODELS_BUCKET)

        uploaded = 0
        for img_file in figures_dir.glob('*.png'):
            try:
                s3_path = f"{base_minio_path}/{img_file.name}"
                minio_client.upload_model(
                    bucket=MODELS_BUCKET,
                    model_path=str(img_file),
                    s3_path=s3_path,
                    content_type='image/png'
                )
                uploaded += 1
                logger.info(f"  Uploaded: {img_file.name}")
            except Exception as e:
                logger.warning(f"  Error uploading {img_file.name}: {e}")

        logger.info(f"Uploaded {uploaded} images to MinIO")
        logger.info(f"MinIO path: s3://{MODELS_BUCKET}/{base_minio_path}/")

        return uploaded

    except Exception as e:
        logger.error(f"Error connecting to MinIO: {e}")
        return 0


def main():
    """Main function to regenerate backtest images."""
    logger.info("=" * 60)
    logger.info("REGENERANDO IMAGENES DE BACKTEST")
    logger.info("=" * 60)

    # Find latest run with results
    run_dir = find_latest_run_with_results()
    if not run_dir:
        logger.error("No training results found")
        return False

    # Load model results
    results_file = run_dir / 'data' / 'model_results.csv'
    logger.info(f"Loading results from: {results_file}")
    df_results = pd.read_csv(results_file)

    # Filter valid results
    df_results = df_results[df_results['direction_accuracy'].notna()].copy()
    logger.info(f"Found {len(df_results)} valid model results")

    if len(df_results) == 0:
        logger.error("No valid results found")
        return False

    # Generate images
    logger.info("\nGenerating backtest images...")
    backtest_count, comparison_count = generate_backtest_images(run_dir, df_results)
    total = backtest_count + comparison_count

    logger.info(f"\nTotal images generated: {total}")
    logger.info(f"  - Backtest cards: {backtest_count}")
    logger.info(f"  - Comparison charts: {comparison_count}")

    # Upload to MinIO
    figures_dir = run_dir / 'figures'
    run_name = run_dir.name

    logger.info("\nUploading images to MinIO...")
    uploaded = upload_images_to_minio(figures_dir, run_name)

    logger.info("=" * 60)
    logger.info("COMPLETADO")
    logger.info(f"  - Imagenes generadas: {total}")
    logger.info(f"  - Imagenes subidas a MinIO: {uploaded}")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
