#!/usr/bin/env python3
"""
generate_bi_csv.py
==================
Script standalone para generar el archivo bi_dashboard_unified.csv

Este script puede ejecutarse manualmente o como parte del pipeline de Airflow.
Lee datos de PostgreSQL (bi schema) y genera un CSV consolidado para el dashboard.

Uso:
    python generate_bi_csv.py                    # Usa valores por defecto
    python generate_bi_csv.py --output ./output  # Especifica directorio de salida
    python generate_bi_csv.py --copy-to-frontend # Copia tambien a frontend/public

Columnas generadas:
    - record_id: Identificador unico del registro
    - view_type: 'backtest' o 'forward_forecast'
    - image_type: Tipo de imagen ('simple', 'complete', 'fan_chart') - solo para forward_forecast
        - simple: forward_forecast_{model}.png (grafico basico)
        - complete: complete_forecast_{model}.png (premium con 4 paneles)
        - fan_chart: fan_chart_{model}.png (intervalos de confianza expandidos)
    - model_id, model_name, model_type
    - horizon_days, horizon_label, horizon_category
    - inference_date, inference_week, inference_year
    - current_price, predicted_price, predicted_return_pct
    - direction, signal, direction_correct
    - model_avg_da, model_avg_rmse
    - image_path (ruta MinIO) - cambia segun image_type
    - consensus_direction, consensus_strength

Nota: Para cada combinacion de model/horizon se generan 3 filas, una por cada image_type.
"""

import os
import sys
import argparse
import logging
import shutil
import json
import re
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Agregar path del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_db_connection():
    """Obtiene conexion a PostgreSQL."""
    import psycopg2

    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'localhost'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_DB', 'pipeline_db'),
        user=os.getenv('POSTGRES_USER', 'pipeline'),
        password=os.getenv('POSTGRES_PASSWORD', 'pipeline_secret')
    )


def load_walk_forward_metrics(runs_dir: Path = None) -> pd.DataFrame:
    """
    Carga métricas de walk-forward del último run de entrenamiento.

    Estas métricas incluyen: Sharpe Ratio, Profit Factor, Max Drawdown, Total Return
    calculadas con validación walk-forward (estándar doctoral).

    Args:
        runs_dir: Directorio de runs de entrenamiento (default: outputs/runs en Airflow)

    Returns:
        DataFrame con columnas: model, horizon, wf_sharpe, wf_max_drawdown,
                                wf_profit_factor, wf_total_return, wf_da_mean
    """
    # Buscar en múltiples ubicaciones posibles
    possible_paths = [
        runs_dir,
        PROJECT_ROOT / 'outputs' / 'runs',
        Path('/opt/airflow/outputs/runs'),  # Ruta en Airflow container
    ]

    for base_path in possible_paths:
        if base_path is None or not base_path.exists():
            continue

        # Buscar el directorio de run más reciente
        run_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()], reverse=True)
        if not run_dirs:
            continue

        # Buscar walk_forward_results.csv en el run más reciente
        for run_dir in run_dirs:
            wf_file = run_dir / 'walk_forward' / 'walk_forward_results.csv'
            if wf_file.exists():
                try:
                    df = pd.read_csv(wf_file)
                    logger.info(f"Loaded {len(df)} walk-forward metrics from {wf_file}")
                    return df
                except Exception as e:
                    logger.warning(f"Error reading {wf_file}: {e}")
                    continue

    logger.warning("Walk-forward results not found in any location")
    return pd.DataFrame()


def load_from_postgresql() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga datos de forecasts y backtest desde PostgreSQL.

    Returns:
        Tuple de (df_forecasts, df_backtest)
    """
    conn = get_db_connection()

    # Query para forecasts con metricas y consensus
    query_forecasts = """
    WITH model_avg_metrics AS (
        SELECT
            model_id,
            AVG(direction_accuracy) as model_avg_da,
            AVG(rmse) as model_avg_rmse
        FROM bi.fact_model_metrics
        GROUP BY model_id
    ),
    best_per_horizon AS (
        SELECT DISTINCT ON (horizon_id)
            horizon_id,
            model_id as best_model,
            direction_accuracy as best_da
        FROM bi.fact_model_metrics
        ORDER BY horizon_id, direction_accuracy DESC
    )
    SELECT
        f.forecast_id as record_id,
        'forward_forecast' as view_type,
        f.model_id,
        UPPER(f.model_id) as model_name,
        CASE
            WHEN f.model_id IN ('xgboost', 'lightgbm', 'catboost') THEN 'tree_ensemble'
            WHEN f.model_id = 'ridge' THEN 'linear'
            WHEN f.model_id = 'ensemble' THEN 'weighted_avg'
            ELSE 'unknown'
        END as model_type,
        f.horizon_id as horizon_days,
        CONCAT('H=', f.horizon_id) as horizon_label,
        CASE
            WHEN f.horizon_id <= 5 THEN 'corto_plazo'
            WHEN f.horizon_id <= 15 THEN 'mediano_plazo'
            ELSE 'largo_plazo'
        END as horizon_category,
        f.inference_week,
        f.inference_date,
        m.direction_accuracy,
        m.rmse,
        m.r2,
        NULL::numeric as mae,
        f.base_price,
        f.predicted_price,
        f.predicted_price - f.base_price as price_change,
        ROUND(((f.predicted_price - f.base_price) / f.base_price * 100)::numeric, 2) as price_change_pct,
        f.predicted_return_pct,
        f.signal,
        f.direction,
        c.consensus_direction,
        c.consensus_strength as consensus_strength_pct,
        mam.model_avg_da as model_avg_direction_accuracy,
        mam.model_avg_rmse,
        NULL::numeric as model_avg_r2,
        CASE WHEN mam.model_avg_da = (SELECT MAX(model_avg_da) FROM model_avg_metrics) THEN true ELSE false END as is_best_overall_model,
        COALESCE(m.is_best_for_horizon, false) as is_best_for_this_horizon,
        bph.best_da as best_da_for_this_horizon,
        bph.best_model as best_model_for_this_horizon,
        CONCAT('forward_forecast_', f.model_id, '.png') as image_path,
        NULL as image_backtest,
        CONCAT('forward_forecast_', f.model_id, '.png') as image_forecast,
        'metrics_heatmap_da.png' as image_heatmap,
        'model_ranking_da.png' as image_ranking,
        dm.training_date,
        NOW() as generated_at
    FROM bi.fact_forecasts f
    LEFT JOIN bi.fact_model_metrics m ON f.model_id = m.model_id AND f.horizon_id = m.horizon_id
    LEFT JOIN bi.fact_consensus c ON f.inference_date = c.inference_date AND f.horizon_id = c.horizon_id
    LEFT JOIN bi.dim_modelos dm ON f.model_id = dm.model_id
    LEFT JOIN model_avg_metrics mam ON f.model_id = mam.model_id
    LEFT JOIN best_per_horizon bph ON f.horizon_id = bph.horizon_id
    ORDER BY f.inference_date DESC, f.model_id, f.horizon_id
    """

    df_forecasts = pd.read_sql(query_forecasts, conn)
    logger.info(f"Cargados {len(df_forecasts)} registros de forecasts desde PostgreSQL")

    # Query para backtest metrics
    query_backtest = """
    WITH model_avg_metrics AS (
        SELECT
            model_id,
            AVG(direction_accuracy) as model_avg_da,
            AVG(rmse) as model_avg_rmse
        FROM bi.fact_model_metrics
        GROUP BY model_id
    ),
    best_per_horizon AS (
        SELECT DISTINCT ON (horizon_id)
            horizon_id,
            model_id as best_model,
            direction_accuracy as best_da
        FROM bi.fact_model_metrics
        ORDER BY horizon_id, direction_accuracy DESC
    )
    SELECT
        CONCAT('BT_', m.model_id, '_h', m.horizon_id) as record_id,
        'backtest' as view_type,
        m.model_id,
        UPPER(m.model_id) as model_name,
        CASE
            WHEN m.model_id IN ('xgboost', 'lightgbm', 'catboost') THEN 'tree_ensemble'
            WHEN m.model_id = 'ridge' THEN 'linear'
            WHEN m.model_id = 'ensemble' THEN 'weighted_avg'
            ELSE 'unknown'
        END as model_type,
        m.horizon_id as horizon_days,
        CONCAT('H=', m.horizon_id) as horizon_label,
        CASE
            WHEN m.horizon_id <= 5 THEN 'corto_plazo'
            WHEN m.horizon_id <= 15 THEN 'mediano_plazo'
            ELSE 'largo_plazo'
        END as horizon_category,
        NULL::integer as inference_week,
        NULL::date as inference_date,
        m.direction_accuracy,
        m.rmse,
        m.r2,
        NULL::numeric as mae,
        NULL::numeric as base_price,
        NULL::numeric as predicted_price,
        NULL::numeric as price_change,
        NULL::numeric as price_change_pct,
        NULL::numeric as predicted_return_pct,
        NULL as signal,
        NULL as direction,
        NULL as consensus_direction,
        NULL::numeric as consensus_strength_pct,
        mam.model_avg_da as model_avg_direction_accuracy,
        mam.model_avg_rmse,
        NULL::numeric as model_avg_r2,
        CASE WHEN mam.model_avg_da = (SELECT MAX(model_avg_da) FROM model_avg_metrics) THEN true ELSE false END as is_best_overall_model,
        m.is_best_for_horizon as is_best_for_this_horizon,
        bph.best_da as best_da_for_this_horizon,
        bph.best_model as best_model_for_this_horizon,
        CONCAT('backtest_', m.model_id, '_h', m.horizon_id, '.png') as image_path,
        CONCAT('backtest_', m.model_id, '_h', m.horizon_id, '.png') as image_backtest,
        NULL as image_forecast,
        'metrics_heatmap_da.png' as image_heatmap,
        'model_ranking_da.png' as image_ranking,
        m.training_date,
        NOW() as generated_at
    FROM bi.fact_model_metrics m
    LEFT JOIN bi.dim_modelos dm ON m.model_id = dm.model_id
    LEFT JOIN model_avg_metrics mam ON m.model_id = mam.model_id
    LEFT JOIN best_per_horizon bph ON m.horizon_id = bph.horizon_id
    ORDER BY m.model_id, m.horizon_id
    """

    df_backtest = pd.read_sql(query_backtest, conn)
    logger.info(f"Cargados {len(df_backtest)} registros de backtest desde PostgreSQL")

    conn.close()

    return df_forecasts, df_backtest


def load_from_csv_files(bi_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga datos desde archivos CSV locales (fallback).

    Args:
        bi_dir: Directorio donde estan los archivos CSV de BI

    Returns:
        Tuple de (df_forecasts, df_backtest)
    """
    logger.info(f"Cargando desde archivos CSV en {bi_dir}")

    # Cargar archivos necesarios
    forecasts_file = bi_dir / 'fact_forecasts.csv'
    metrics_file = bi_dir / 'fact_model_metrics.csv'
    modelos_file = bi_dir / 'dim_modelos.csv'
    consensus_file = bi_dir / 'fact_consensus.csv'

    # Verificar que existen
    if not forecasts_file.exists():
        raise FileNotFoundError(f"No se encontro {forecasts_file}")
    if not metrics_file.exists():
        raise FileNotFoundError(f"No se encontro {metrics_file}")

    df_forecasts_raw = pd.read_csv(forecasts_file)
    df_metrics = pd.read_csv(metrics_file)
    df_modelos = pd.read_csv(modelos_file) if modelos_file.exists() else pd.DataFrame()
    df_consensus = pd.read_csv(consensus_file) if consensus_file.exists() else pd.DataFrame()

    # Calcular metricas agregadas por modelo
    model_avg_metrics = {}
    for model_id in df_metrics['model_id'].unique():
        model_data = df_metrics[df_metrics['model_id'] == model_id]
        model_avg_metrics[model_id] = {
            'avg_da': model_data['direction_accuracy'].mean(),
            'avg_rmse': model_data['rmse'].mean()
        }

    # Encontrar mejor modelo overall
    best_overall_model = max(model_avg_metrics.keys(),
                            key=lambda x: model_avg_metrics[x]['avg_da']) if model_avg_metrics else None

    # Encontrar mejor modelo por horizonte
    best_per_horizon = {}
    for h in df_metrics['horizon_id'].unique():
        h_data = df_metrics[df_metrics['horizon_id'] == h]
        best_idx = h_data['direction_accuracy'].idxmax()
        best_per_horizon[h] = {
            'model': h_data.loc[best_idx, 'model_id'],
            'da': h_data.loc[best_idx, 'direction_accuracy']
        }

    # Construir DataFrame de backtest
    backtest_rows = []
    for _, m in df_metrics.iterrows():
        model_id = m['model_id']
        horizon = m['horizon_id']
        model_avg = model_avg_metrics.get(model_id, {'avg_da': None, 'avg_rmse': None})
        best_h = best_per_horizon.get(horizon, {'model': None, 'da': None})

        backtest_rows.append({
            'record_id': f"BT_{model_id}_h{horizon}",
            'view_type': 'backtest',
            'image_type': None,  # Backtest rows don't have image_type
            'model_id': model_id,
            'model_name': model_id.upper(),
            'model_type': _get_model_type(model_id),
            'horizon_days': horizon,
            'horizon_label': f"H={horizon}",
            'horizon_category': _get_horizon_category(horizon),
            'inference_week': None,
            'inference_date': None,
            'direction_accuracy': m.get('direction_accuracy'),
            'rmse': m.get('rmse'),
            'r2': m.get('r2'),
            'mae': None,
            'base_price': None,
            'predicted_price': None,
            'price_change': None,
            'price_change_pct': None,
            'predicted_return_pct': None,
            'signal': None,
            'direction': None,
            'consensus_direction': None,
            'consensus_strength_pct': None,
            'model_avg_direction_accuracy': model_avg['avg_da'],
            'model_avg_rmse': model_avg['avg_rmse'],
            'model_avg_r2': None,
            'is_best_overall_model': model_id == best_overall_model,
            'is_best_for_this_horizon': m.get('is_best_for_horizon', False),
            'best_da_for_this_horizon': best_h['da'],
            'best_model_for_this_horizon': best_h['model'],
            'image_path': f"backtest_{model_id}_h{horizon}.png",
            'image_backtest': f"backtest_{model_id}_h{horizon}.png",
            'image_forecast': None,
            'image_heatmap': 'metrics_heatmap_da.png',
            'image_ranking': 'model_ranking_da.png',
            'training_date': m.get('training_date'),
            'generated_at': datetime.now().isoformat()
        })

    df_backtest = pd.DataFrame(backtest_rows)

    # Construir DataFrame de forecasts - Generate MULTIPLE rows per forecast (one per image_type)
    forecast_rows = []
    base_figures_path = "figures"  # Base path for images

    for _, f in df_forecasts_raw.iterrows():
        model_id = f['model_id']
        horizon = f['horizon_id']

        # Obtener consensus
        consensus_row = None
        if not df_consensus.empty:
            matches = df_consensus[df_consensus['horizon_id'] == horizon]
            if not matches.empty:
                consensus_row = matches.iloc[0]

        # Obtener metricas del modelo
        model_metrics = None
        metrics_matches = df_metrics[(df_metrics['model_id'] == model_id) & (df_metrics['horizon_id'] == horizon)]
        if not metrics_matches.empty:
            model_metrics = metrics_matches.iloc[0]

        model_avg = model_avg_metrics.get(model_id, {'avg_da': None, 'avg_rmse': None})
        best_h = best_per_horizon.get(horizon, {'model': None, 'da': None})

        inference_date = pd.to_datetime(f['inference_date'])
        week_num = inference_date.isocalendar()[1]
        year = inference_date.year

        base_model = _get_base_model_name(model_id)
        base_record_id = f.get('forecast_id', f"FF_{inference_date.strftime('%Y%m%d')}_{model_id}_h{horizon}")

        # Generate one row for each image type: simple, complete, fan_chart
        for img_type in ['simple', 'complete', 'fan_chart']:
            # Generate unique record_id including image_type
            record_id = f"{base_record_id}_{img_type}"

            # Get image path based on type
            image_forecast_path = _get_image_path_for_type(img_type, model_id, base_figures_path)

            forecast_rows.append({
                'record_id': record_id,
                'view_type': 'forward_forecast',
                'image_type': img_type,  # NEW COLUMN: simple, complete, or fan_chart
                'model_id': model_id,
                'model_name': model_id.upper(),
                'model_type': _get_model_type(model_id),
                'horizon_days': horizon,
                'horizon_label': f"H={horizon}",
                'horizon_category': _get_horizon_category(horizon),
                'inference_week': week_num,
                'inference_date': str(inference_date.date()),
                'direction_accuracy': model_metrics['direction_accuracy'] if model_metrics is not None else None,
                'rmse': model_metrics['rmse'] if model_metrics is not None else None,
                'r2': model_metrics['r2'] if model_metrics is not None else None,
                'mae': None,
                'base_price': f.get('base_price'),
                'predicted_price': f.get('predicted_price'),
                'price_change': f.get('price_change'),
                'price_change_pct': f.get('price_change_pct'),
                'predicted_return_pct': f.get('predicted_return'),
                'signal': f.get('signal'),
                'direction': f.get('direction'),
                'consensus_direction': consensus_row['consensus_direction'] if consensus_row is not None else None,
                'consensus_strength_pct': consensus_row['consensus_strength'] if consensus_row is not None else None,
                'model_avg_direction_accuracy': model_avg['avg_da'],
                'model_avg_rmse': model_avg['avg_rmse'],
                'model_avg_r2': None,
                'is_best_overall_model': model_id == best_overall_model,
                'is_best_for_this_horizon': model_metrics['is_best_for_horizon'] if model_metrics is not None else False,
                'best_da_for_this_horizon': best_h['da'],
                'best_model_for_this_horizon': best_h['model'],
                'image_path': image_forecast_path,  # Updated based on image_type
                'image_backtest': f"backtest_{base_model}_h{horizon}.png" if model_id != 'ensemble' else None,
                'image_forecast': image_forecast_path,  # Updated based on image_type
                'image_heatmap': 'metrics_heatmap_da.png',
                'image_ranking': 'model_ranking_da.png',
                'training_date': df_modelos[df_modelos['model_id'] == model_id]['training_date'].iloc[0] if not df_modelos.empty and model_id in df_modelos['model_id'].values else None,
                'generated_at': datetime.now().isoformat()
            })

    df_forecasts = pd.DataFrame(forecast_rows)
    logger.info(f"Generated {len(forecast_rows)} forecast rows (3 image types per model/horizon)")

    return df_forecasts, df_backtest


def _get_base_model_name(model_id: str) -> str:
    """Extract base model name for image paths (strips _pure, _hybrid suffixes)."""
    base = model_id.lower()
    for suffix in ['_pure', '_hybrid', '_tuned']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return base


def _get_model_type(model_id: str) -> str:
    """Retorna el tipo de modelo basado en su ID."""
    base = _get_base_model_name(model_id)
    if base in ['xgboost', 'lightgbm', 'catboost']:
        return 'tree_ensemble'
    elif base in ['ridge', 'bayesian_ridge', 'ard']:
        return 'linear'
    elif base == 'ensemble':
        return 'weighted_avg'
    return 'unknown'


def _get_horizon_category(horizon: int) -> str:
    """Retorna la categoria del horizonte."""
    if horizon <= 5:
        return 'corto_plazo'
    elif horizon <= 15:
        return 'mediano_plazo'
    return 'largo_plazo'


def get_iso_week_year(date_str: str) -> Tuple[int, int]:
    """
    Calculate the correct ISO week number and year for a given date.

    ISO weeks properly handle year boundaries:
    - 2025-12-29 -> Week 1 of 2026 (Monday of that week is in 2026)
    - 2026-01-05 -> Week 1 of 2026 (same week, Sunday)
    - 2026-01-06 -> Week 2 of 2026 (Monday starts new week)

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Tuple of (iso_week, iso_year)
    """
    dt = pd.to_datetime(date_str)
    iso_cal = dt.isocalendar()
    # isocalendar() returns (year, week, weekday)
    # The year returned is the ISO year, which may differ from calendar year
    return iso_cal[1], iso_cal[0]


def scan_forecast_directories(forecasts_dir: Path) -> List[Tuple[Path, str]]:
    """
    Scan the forecasts directory for all date-named subdirectories.

    Args:
        forecasts_dir: Path to outputs/forecasts directory

    Returns:
        List of tuples (directory_path, date_string) sorted by date
    """
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    date_dirs = []

    if not forecasts_dir.exists():
        logger.warning(f"Forecasts directory not found: {forecasts_dir}")
        return []

    for item in forecasts_dir.iterdir():
        if item.is_dir() and date_pattern.match(item.name):
            date_dirs.append((item, item.name))

    # Sort by date
    date_dirs.sort(key=lambda x: x[1])

    logger.info(f"Found {len(date_dirs)} forecast date directories: {[d[1] for d in date_dirs]}")
    return date_dirs


def load_forecasts_from_all_directories(forecasts_dir: Path) -> pd.DataFrame:
    """
    Load ALL forecasts (individual models and ensembles) from ALL date directories.

    This function scans outputs/forecasts/ for all date folders and loads
    forecast_data.json from each, generating records for:
    - Individual model predictions (ard, bayesian_ridge, catboost_pure, etc.)
    - Ensemble predictions (best_of_breed, top_3, top_6_mean)

    Args:
        forecasts_dir: Path to outputs/forecasts directory

    Returns:
        DataFrame with all forecast rows including correct inference_week and inference_year
    """
    all_forecast_rows = []

    date_dirs = scan_forecast_directories(forecasts_dir)

    if not date_dirs:
        return pd.DataFrame()

    for date_dir, date_str in date_dirs:
        json_file = date_dir / 'forecast_data.json'
        if not json_file.exists():
            logger.warning(f"No forecast_data.json found in {date_dir}")
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
            continue

        base_date = data.get('base_date', date_str)
        last_price = data.get('last_price')
        predictions = data.get('predictions', {})
        ensembles = data.get('ensembles', {})

        if not last_price:
            logger.warning(f"No last_price found in {json_file}")
            continue

        # Get correct ISO week and year
        inference_week, inference_year = get_iso_week_year(base_date)
        inference_date_obj = pd.to_datetime(base_date)

        logger.info(f"Processing {base_date}: ISO Week {inference_week}, Year {inference_year}")

        # Process individual model predictions
        for model_id, model_predictions in predictions.items():
            for horizon_str, predicted_return in model_predictions.items():
                horizon = int(horizon_str)
                predicted_price = last_price * (1 + predicted_return) if predicted_return is not None else None
                price_change = predicted_price - last_price if predicted_price else None
                price_change_pct = predicted_return * 100 if predicted_return is not None else None

                # Determine direction and signal
                direction = 'UP' if predicted_return and predicted_return > 0 else 'DOWN'
                signal = 'BUY' if direction == 'UP' else 'SELL'

                all_forecast_rows.append({
                    'record_id': f"FF_{inference_date_obj.strftime('%Y%m%d')}_{model_id}_h{horizon}",
                    'view_type': 'forward_forecast',
                    'model_id': model_id,
                    'model_name': model_id.upper().replace('_', ' '),
                    'model_type': _get_model_type(model_id),
                    'horizon_days': horizon,
                    'horizon_label': f"H={horizon}",
                    'horizon_category': _get_horizon_category(horizon),
                    'inference_week': inference_week,
                    'inference_year': inference_year,
                    'inference_date': str(inference_date_obj.date()),
                    'direction_accuracy': None,
                    'rmse': None,
                    'r2': None,
                    'mae': None,
                    'base_price': last_price,
                    'predicted_price': round(predicted_price, 2) if predicted_price else None,
                    'price_change': round(price_change, 2) if price_change else None,
                    'price_change_pct': round(price_change_pct, 4) if price_change_pct is not None else None,
                    'predicted_return_pct': round(predicted_return * 100, 4) if predicted_return is not None else None,
                    'signal': signal,
                    'direction': direction,
                    'consensus_direction': None,
                    'consensus_strength_pct': None,
                    'model_avg_direction_accuracy': None,
                    'model_avg_rmse': None,
                    'model_avg_r2': None,
                    'is_best_overall_model': False,
                    'is_best_for_this_horizon': False,
                    'best_da_for_this_horizon': None,
                    'best_model_for_this_horizon': None,
                    'image_path': f"forecasts/{base_date}/forward_forecast_{model_id}.png",
                    'image_backtest': None,
                    'image_forecast': f"forecasts/{base_date}/forward_forecast_{model_id}.png",
                    'image_heatmap': 'metrics_heatmap_da.png',
                    'image_ranking': 'model_ranking_da.png',
                    'training_date': None,
                    'generated_at': datetime.now().isoformat()
                })

        # Process ensemble predictions
        for ensemble_name, ensemble_predictions in ensembles.items():
            for horizon_str, predicted_return in ensemble_predictions.items():
                horizon = int(horizon_str)
                predicted_price = last_price * (1 + predicted_return) if predicted_return is not None else None
                price_change = predicted_price - last_price if predicted_price else None
                price_change_pct = predicted_return * 100 if predicted_return is not None else None

                # Determine direction and signal
                direction = 'UP' if predicted_return and predicted_return > 0 else 'DOWN'
                signal = 'BUY' if direction == 'UP' else 'SELL'

                all_forecast_rows.append({
                    'record_id': f"FF_{inference_date_obj.strftime('%Y%m%d')}_{ensemble_name}_h{horizon}",
                    'view_type': 'forward_forecast',
                    'model_id': ensemble_name,
                    'model_name': ensemble_name.upper().replace('_', ' '),
                    'model_type': 'ensemble',
                    'horizon_days': horizon,
                    'horizon_label': f"H={horizon}",
                    'horizon_category': _get_horizon_category(horizon),
                    'inference_week': inference_week,
                    'inference_year': inference_year,
                    'inference_date': str(inference_date_obj.date()),
                    'direction_accuracy': None,
                    'rmse': None,
                    'r2': None,
                    'mae': None,
                    'base_price': last_price,
                    'predicted_price': round(predicted_price, 2) if predicted_price else None,
                    'price_change': round(price_change, 2) if price_change else None,
                    'price_change_pct': round(price_change_pct, 4) if price_change_pct is not None else None,
                    'predicted_return_pct': round(predicted_return * 100, 4) if predicted_return is not None else None,
                    'signal': signal,
                    'direction': direction,
                    'consensus_direction': None,
                    'consensus_strength_pct': None,
                    'model_avg_direction_accuracy': None,
                    'model_avg_rmse': None,
                    'model_avg_r2': None,
                    'is_best_overall_model': False,
                    'is_best_for_this_horizon': False,
                    'best_da_for_this_horizon': None,
                    'best_model_for_this_horizon': None,
                    'image_path': f"forecasts/{base_date}/forward_forecast_{ensemble_name}.png",
                    'image_backtest': None,
                    'image_forecast': f"forecasts/{base_date}/forward_forecast_{ensemble_name}.png",
                    'image_heatmap': 'metrics_heatmap_da.png',
                    'image_ranking': 'model_ranking_da.png',
                    'training_date': None,
                    'generated_at': datetime.now().isoformat()
                })

    df_forecasts = pd.DataFrame(all_forecast_rows)
    if not df_forecasts.empty:
        # Get unique weeks for logging
        unique_weeks = df_forecasts[['inference_year', 'inference_week']].drop_duplicates()
        weeks_str = ', '.join([f"W{row['inference_week']}/{row['inference_year']}"
                              for _, row in unique_weeks.iterrows()])
        logger.info(f"Loaded {len(df_forecasts)} forecast rows from {len(date_dirs)} directories")
        logger.info(f"  Weeks covered: {weeks_str}")

    return df_forecasts


# Image type configurations for multiple rows per forecast
IMAGE_TYPE_CONFIG = {
    'simple': {
        'image_type': 'simple',
        'filename_prefix': 'forward_forecast',
        'description': 'Simple forward forecast visualization'
    },
    'complete': {
        'image_type': 'complete',
        'filename_prefix': 'complete_forecast',
        'description': 'Complete forecast with 4 panels (premium)'
    },
    'fan_chart': {
        'image_type': 'fan_chart',
        'filename_prefix': 'fan_chart',
        'description': 'Fan chart with expanding confidence intervals'
    }
}


def _get_image_path_for_type(image_type: str, model_id: str, base_path: str) -> str:
    """
    Generate image path based on image type.

    Args:
        image_type: One of 'simple', 'complete', 'fan_chart'
        model_id: The model identifier
        base_path: Base path for images (e.g., 'results/weekly_update/figures')

    Returns:
        Full image path string
    """
    config = IMAGE_TYPE_CONFIG.get(image_type, IMAGE_TYPE_CONFIG['simple'])
    prefix = config['filename_prefix']
    base_model = _get_base_model_name(model_id)
    return f"{base_path}/{prefix}_{base_model}.png"


def load_ensemble_forecasts_from_json(forecasts_dir: Path) -> pd.DataFrame:
    """
    Load ensemble forecasts from forecast_data.json files.

    Args:
        forecasts_dir: Path to outputs/forecasts directory

    Returns:
        DataFrame with ensemble forecast rows
    """
    ensemble_rows = []
    ensemble_types = ['best_of_breed', 'top_3', 'top_6_mean']

    # Find all date directories with forecast_data.json
    if not forecasts_dir.exists():
        logger.warning(f"Forecasts directory not found: {forecasts_dir}")
        return pd.DataFrame()

    for date_dir in sorted(forecasts_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        json_file = date_dir / 'forecast_data.json'
        if not json_file.exists():
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")
            continue

        base_date = data.get('base_date')
        last_price = data.get('last_price')
        ensembles = data.get('ensembles', {})

        if not base_date or not ensembles:
            continue

        inference_date = pd.to_datetime(base_date)
        week_num = inference_date.isocalendar()[1]
        year = inference_date.year

        for ensemble_name in ensemble_types:
            if ensemble_name not in ensembles:
                continue

            predictions = ensembles[ensemble_name]

            for horizon_str, predicted_return in predictions.items():
                horizon = int(horizon_str)
                predicted_price = last_price * (1 + predicted_return) if last_price else None
                price_change = predicted_price - last_price if last_price and predicted_price else None
                price_change_pct = predicted_return * 100 if predicted_return else None

                # Determine direction and signal
                direction = 'UP' if predicted_return and predicted_return > 0 else 'DOWN'
                signal = 'BUY' if direction == 'UP' else 'SELL'

                base_record_id = f"FF_{inference_date.strftime('%Y%m%d')}_{ensemble_name}_h{horizon}"
                base_figures_path = "figures"

                # Generate one row for each image type: simple, complete, fan_chart
                for img_type in ['simple', 'complete', 'fan_chart']:
                    record_id = f"{base_record_id}_{img_type}"
                    image_forecast_path = _get_image_path_for_type(img_type, ensemble_name, base_figures_path)

                    ensemble_rows.append({
                        'record_id': record_id,
                        'view_type': 'forward_forecast',
                        'image_type': img_type,  # NEW COLUMN: simple, complete, or fan_chart
                        'model_id': ensemble_name,
                        'model_name': ensemble_name.upper().replace('_', ' '),
                        'model_type': 'ensemble',
                        'horizon_days': horizon,
                        'horizon_label': f"H={horizon}",
                        'horizon_category': _get_horizon_category(horizon),
                        'inference_week': week_num,
                        'inference_date': str(inference_date.date()),
                        'direction_accuracy': None,  # Ensembles don't have individual metrics
                        'rmse': None,
                        'r2': None,
                        'mae': None,
                        'base_price': last_price,
                        'predicted_price': round(predicted_price, 2) if predicted_price else None,
                        'price_change': round(price_change, 2) if price_change else None,
                        'price_change_pct': round(price_change_pct, 2) if price_change_pct else None,
                        'predicted_return_pct': round(predicted_return * 100, 2) if predicted_return else None,
                        'signal': signal,
                        'direction': direction,
                        'consensus_direction': None,
                        'consensus_strength_pct': None,
                        'model_avg_direction_accuracy': None,
                        'model_avg_rmse': None,
                        'model_avg_r2': None,
                        'is_best_overall_model': False,
                        'is_best_for_this_horizon': False,
                        'best_da_for_this_horizon': None,
                        'best_model_for_this_horizon': None,
                        'image_path': image_forecast_path,  # Updated based on image_type
                        'image_backtest': None,
                        'image_forecast': image_forecast_path,  # Updated based on image_type
                        'image_heatmap': 'metrics_heatmap_da.png',
                        'image_ranking': 'model_ranking_da.png',
                        'training_date': None,
                        'generated_at': datetime.now().isoformat()
                    })

    df_ensembles = pd.DataFrame(ensemble_rows)
    if not df_ensembles.empty:
        logger.info(f"Loaded {len(df_ensembles)} ensemble forecast rows from JSON files")

    return df_ensembles


def generate_bi_dashboard_csv(
    output_dir: Optional[Path] = None,
    copy_to_frontend: bool = True,
    frontend_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Genera el archivo bi_dashboard_unified.csv consolidado.

    Args:
        output_dir: Directorio de salida (default: outputs/bi)
        copy_to_frontend: Si copiar al directorio del frontend
        frontend_dir: Directorio del frontend (default: frontend/public)

    Returns:
        Dict con informacion del resultado
    """
    logger.info("=" * 60)
    logger.info("GENERANDO BI DASHBOARD CSV UNIFICADO")
    logger.info("=" * 60)

    # Configurar directorios
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'outputs' / 'bi'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if frontend_dir is None:
        frontend_dir = PROJECT_ROOT / 'frontend' / 'public'
    frontend_dir = Path(frontend_dir)

    # Load backtest data from PostgreSQL or CSV files
    df_backtest = pd.DataFrame()
    data_source = 'forecast_directories'

    try:
        df_db_forecasts, df_backtest = load_from_postgresql()
        data_source = 'postgresql + forecast_directories'
        logger.info("Backtest data loaded from PostgreSQL")
    except Exception as e:
        logger.warning(f"Error cargando desde PostgreSQL: {e}")
        logger.info("Intentando cargar backtest desde archivos CSV locales...")

        try:
            _, df_backtest = load_from_csv_files(output_dir)
            data_source = 'csv_files + forecast_directories'
            logger.info("Backtest data loaded from CSV files")
        except Exception as e2:
            logger.warning(f"Error cargando backtest desde CSV: {e2}")
            logger.info("Continuing without backtest data...")
            data_source = 'forecast_directories_only'

    # Load ALL forecasts (models and ensembles) from ALL date directories
    # This scans outputs/forecasts/ for all date folders (2025-12-29, 2026-01-05, etc.)
    # and generates records with correct ISO week/year for each
    forecasts_dir = PROJECT_ROOT / 'outputs' / 'forecasts'
    df_all_forecasts = load_forecasts_from_all_directories(forecasts_dir)

    # =========================================================================
    # CARGAR MÉTRICAS DE WALK-FORWARD (Sharpe, PF, MDD, Return)
    # =========================================================================
    df_wf_metrics = load_walk_forward_metrics()

    # Merge walk-forward metrics con backtest data
    if not df_wf_metrics.empty and not df_backtest.empty:
        logger.info(f"Merging {len(df_wf_metrics)} walk-forward metrics with backtest data...")

        # Renombrar columnas de WF para el merge
        wf_cols_to_merge = ['model', 'horizon', 'wf_da_mean', 'wf_sharpe',
                           'wf_max_drawdown', 'wf_profit_factor', 'wf_total_return']
        df_wf_subset = df_wf_metrics[[c for c in wf_cols_to_merge if c in df_wf_metrics.columns]].copy()

        # Hacer merge
        df_backtest = df_backtest.merge(
            df_wf_subset,
            left_on=['model_id', 'horizon_days'],
            right_on=['model', 'horizon'],
            how='left'
        )

        # Renombrar columnas para el dashboard
        rename_map = {
            'wf_sharpe': 'sharpe',
            'wf_max_drawdown': 'max_drawdown',
            'wf_profit_factor': 'profit_factor',
            'wf_total_return': 'total_return',
            'wf_da_mean': 'wf_direction_accuracy'
        }
        df_backtest = df_backtest.rename(columns={k: v for k, v in rename_map.items()
                                                   if k in df_backtest.columns})

        # Eliminar columnas duplicadas del merge
        cols_to_drop = ['model', 'horizon']
        df_backtest = df_backtest.drop(columns=[c for c in cols_to_drop if c in df_backtest.columns],
                                        errors='ignore')

        logger.info(f"Walk-forward metrics merged successfully")

    # Combine DataFrames
    dataframes_to_concat = []
    if not df_backtest.empty:
        dataframes_to_concat.append(df_backtest)
    if not df_all_forecasts.empty:
        dataframes_to_concat.append(df_all_forecasts)

    if not dataframes_to_concat:
        raise RuntimeError("No data found from any source")

    df_combined = pd.concat(dataframes_to_concat, ignore_index=True)

    # Ordenar columnas en orden logico (incluyendo nuevas métricas de trading)
    column_order = [
        'record_id', 'view_type', 'image_type', 'model_id', 'model_name', 'model_type',
        'horizon_days', 'horizon_label', 'horizon_category',
        'inference_week', 'inference_year', 'inference_date',
        'direction_accuracy', 'rmse', 'r2', 'mae',
        # NUEVAS COLUMNAS DE TRADING METRICS (Walk-Forward Validated):
        'sharpe', 'profit_factor', 'max_drawdown', 'total_return', 'wf_direction_accuracy',
        'base_price', 'predicted_price', 'price_change', 'price_change_pct',
        'predicted_return_pct', 'signal', 'direction',
        'consensus_direction', 'consensus_strength_pct',
        'model_avg_direction_accuracy', 'model_avg_rmse', 'model_avg_r2',
        'is_best_overall_model', 'is_best_for_this_horizon',
        'best_da_for_this_horizon', 'best_model_for_this_horizon',
        'image_path', 'image_backtest', 'image_forecast', 'image_heatmap', 'image_ranking',
        'training_date', 'generated_at'
    ]

    # Reordenar columnas (solo las que existen)
    existing_cols = [c for c in column_order if c in df_combined.columns]
    df_combined = df_combined[existing_cols]

    # Guardar CSV principal
    output_file = output_dir / 'bi_dashboard_unified.csv'
    df_combined.to_csv(output_file, index=False)
    logger.info(f"CSV guardado: {output_file} ({len(df_combined)} registros)")

    # Count unique weeks in forecasts
    unique_weeks = []
    if not df_all_forecasts.empty and 'inference_week' in df_all_forecasts.columns:
        unique_weeks = df_all_forecasts[['inference_year', 'inference_week']].drop_duplicates().values.tolist()

    result = {
        'output_file': str(output_file),
        'rows': len(df_combined),
        'backtest_rows': len(df_backtest),
        'forecast_rows': len(df_all_forecasts),
        'unique_weeks': unique_weeks,
        'data_source': data_source,
        'generated_at': datetime.now().isoformat()
    }

    # Copiar a frontend si se solicito
    if copy_to_frontend:
        frontend_dir.mkdir(parents=True, exist_ok=True)
        frontend_file = frontend_dir / 'bi_dashboard_unified.csv'
        try:
            shutil.copy(str(output_file), str(frontend_file))
            logger.info(f"CSV copiado a frontend: {frontend_file}")
            result['frontend_copy'] = str(frontend_file)
        except Exception as e:
            logger.warning(f"No se pudo copiar a frontend: {e}")
            result['frontend_copy_error'] = str(e)

    logger.info("=" * 60)
    logger.info("BI DASHBOARD CSV GENERADO EXITOSAMENTE")
    logger.info(f"Total registros: {len(df_combined)}")
    logger.info(f"  - Backtest: {len(df_backtest)}")
    logger.info(f"  - Forecasts from directories: {len(df_all_forecasts)}")
    if unique_weeks:
        weeks_str = ', '.join([f"W{w[1]}/{w[0]}" for w in unique_weeks])
        logger.info(f"  - Weeks covered: {weeks_str}")
    logger.info(f"Fuente de datos: {data_source}")
    logger.info("=" * 60)
    


    return result






def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description='Genera bi_dashboard_unified.csv para el dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python generate_bi_csv.py                      # Genera con valores por defecto
  python generate_bi_csv.py --output ./output    # Especifica directorio de salida
  python generate_bi_csv.py --no-frontend        # No copia al frontend
  python generate_bi_csv.py --verbose            # Muestra mas informacion
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Directorio de salida para el CSV (default: outputs/bi)'
    )

    parser.add_argument(
        '--frontend-dir', '-f',
        type=Path,
        default=None,
        help='Directorio del frontend (default: frontend/public)'
    )

    parser.add_argument(
        '--no-frontend',
        action='store_true',
        help='No copiar al directorio del frontend'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar mas informacion de debug'
    )

    args = parser.parse_args()

    # Configurar nivel de logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result = generate_bi_dashboard_csv(
            output_dir=args.output,
            copy_to_frontend=not args.no_frontend,
            frontend_dir=args.frontend_dir
        )

        print(f"\nResultado:")
        print(f"  Archivo: {result['output_file']}")
        print(f"  Registros: {result['rows']}")
        print(f"  Fuente: {result['data_source']}")
        if 'frontend_copy' in result:
            print(f"  Frontend: {result['frontend_copy']}")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
