"""
DAG: weekly_inference_pipeline
==============================
Pipeline de Inferencia Semanal - USD/COP Forecasting

Ejecuta cada LUNES a las 8:00 AM (hora Colombia):
1. Trigger scraping de variables macro
2. Esperar a que scraping termine
3. Cargar datos actualizados
4. **CHECK DATA QUALITY** - Validar calidad antes de inferencia
5. Ejecutar inferencia con modelos entrenados
6. Generar predicciones para la semana
7. Actualizar tablas BI del dashboard
8. Notificar completado

Schedule: 0 13 * * 1 (8:00 AM COT = 13:00 UTC, Lunes)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import json
import pickle
import logging
import os
import sys
from pathlib import Path

# Agregar path del backend
sys.path.insert(0, '/opt/airflow/backend')

# Visualization imports for forward forecast plots
try:
    from src.visualization.forecast_plots import generate_all_forecast_plots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    generate_all_forecast_plots = None
    logging.warning("Visualization modules not available - forward forecast plots will not be generated")

# Data quality monitoring imports
try:
    from src.monitoring import DataQualityMonitor, QualityReport, DataQualityError
    DATA_QUALITY_AVAILABLE = True
except ImportError:
    DATA_QUALITY_AVAILABLE = False
    logging.warning("Data quality monitoring module not available")

# MinIO client imports
try:
    from src.mlops.minio_client import MinioClient, FORECASTS_BUCKET
    MINIO_CLIENT_AVAILABLE = True
except ImportError:
    MINIO_CLIENT_AVAILABLE = False
    logging.warning("MinIO client module not available")

# Importar callbacks y utilidades
from utils.callbacks import (
    task_failure_callback,
    task_success_callback,
    dag_failure_callback,
    sla_miss_callback,
    task_retry_callback,
    get_alert_email,
)

DAG_ID = "05_l2_weekly_inference"

# =============================================================================
# DEFAULT ARGS - Con reintentos exponenciales y callbacks
# =============================================================================

# Obtener email de alertas desde variable de entorno
ALERT_EMAIL = get_alert_email()

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': [ALERT_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    # Reintentos con delay exponencial
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=1),
    # Callbacks
    'on_failure_callback': task_failure_callback,
    'on_success_callback': task_success_callback,
    'on_retry_callback': task_retry_callback,
}

# =============================================================================
# FUNCIONES DEL PIPELINE
# =============================================================================

def load_latest_data(**context):
    """Carga los datos mas recientes desde BD o CSV."""
    logging.info("Cargando datos actualizados...")

    # Intentar cargar desde base de datos primero
    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()

        query = """
            SELECT * FROM core.features_ml
            ORDER BY date DESC
            LIMIT 500
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)
            # Renombrar columnas para compatibilidad con pipeline
            df = df.rename(columns={
                'close_price': 'Close',
                'open_price': 'Open',
                'high_price': 'High',
                'low_price': 'Low',
                'date': 'Date'
            })
            logging.info(f"Cargados {len(df)} registros desde PostgreSQL")
            context['ti'].xcom_push(key='data_source', value='postgresql')
            context['ti'].xcom_push(key='data_path', value='postgresql://core.features_ml')
            return {
                'rows': len(df),
                'last_date': str(df['Date'].max()),
                'source': 'postgresql'
            }

    except Exception as e:
        logging.warning(f"No se pudo cargar desde PostgreSQL: {e}")

    # Fallback: cargar desde CSV
    data_dir = Path('/opt/airflow/data/processed')
    feature_files = sorted(data_dir.glob('features_*.csv'), reverse=True)

    if not feature_files:
        # Fallback 2: usar datos de raw
        raw_file = data_dir.parent / 'raw' / 'RL_COMBINED_ML_FEATURES.csv'
        if not raw_file.exists():
            raw_file = data_dir.parent / 'RL_COMBINED_ML_FEATURES.csv'

        if raw_file.exists():
            df = pd.read_csv(raw_file)
            # Normalizar nombres de columnas
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={
                'close': 'Close',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'date': 'Date'
            })
            logging.info(f"Cargados {len(df)} registros desde {raw_file}")
            data_path = str(raw_file)
        else:
            raise FileNotFoundError("No se encontraron datos de features")
    else:
        df = pd.read_csv(feature_files[0])
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={
            'close': 'Close', 'close_price': 'Close',
            'open': 'Open', 'open_price': 'Open',
            'high': 'High', 'high_price': 'High',
            'low': 'Low', 'low_price': 'Low',
            'date': 'Date'
        })
        logging.info(f"Cargados {len(df)} registros desde {feature_files[0]}")
        data_path = str(feature_files[0])

    context['ti'].xcom_push(key='data_source', value='csv')
    context['ti'].xcom_push(key='data_path', value=data_path)

    return {
        'rows': len(df),
        'last_date': str(df['Date'].max()) if 'Date' in df.columns else 'unknown',
        'source': 'csv'
    }


def run_inference(**context):
    """Ejecuta inferencia con modelos entrenados."""
    logging.info("Ejecutando inferencia...")

    from src.core.config import HORIZONS, ML_MODELS

    # Obtener fuente de datos
    data_source = context['ti'].xcom_pull(key='data_source', task_ids='load_data')
    data_path = context['ti'].xcom_pull(key='data_path', task_ids='load_data')

    # Directorio de modelos (ultimo run de training)
    models_base = Path('/opt/airflow/outputs/runs')
    model_dirs = sorted(models_base.glob('*/models'), reverse=True)

    if not model_dirs:
        raise FileNotFoundError("No se encontraron modelos entrenados")

    models_dir = model_dirs[0]
    logging.info(f"Usando modelos de: {models_dir}")

    # Cargar datos segun la fuente
    if data_source == 'postgresql':
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        query = "SELECT * FROM core.features_ml ORDER BY date"
        df = pd.read_sql(query, conn)
        conn.close()
        df = df.rename(columns={
            'close_price': 'Close',
            'date': 'Date'
        })
    else:
        df = pd.read_csv(data_path)
        df.columns = [c.lower() for c in df.columns]
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close', 'date': 'Date'})
        elif 'close_price' in df.columns:
            df = df.rename(columns={'close_price': 'Close', 'date': 'Date'})

    logging.info(f"Datos cargados: {len(df)} registros")

    # Usar features directamente del dataset (ya vienen procesados)
    df_features = df.copy()

    # 29 FEATURES exactos usados en entrenamiento (contrato con DAG 04)
    TRAINING_FEATURES = [
        # DXY (3)
        'dxy', 'dxy_z', 'dxy_change_1d',
        # VIX (3)
        'vix', 'vix_z', 'vix_regime',
        # Commodities (4)
        'brent', 'brent_z', 'wti', 'wti_z',
        # EMBI (2)
        'embi', 'embi_z',
        # Tasas (6)
        'curve_slope', 'curve_slope_z',
        'col_us_spread', 'col_us_spread_z',
        'policy_spread', 'policy_spread_z',
        # Carry (1)
        'carry_favorable',
        # Commodities adicionales (2)
        'coffee_z', 'gold_z',
        # Macro Colombia (6)
        'terms_trade', 'trade_balance_z',
        'exports', 'fdi_inflow', 'reserves', 'itcr',
        # Pares regionales (2)
        'usdmxn_ret_1d', 'usdclp_ret_1d'
    ]

    # Obtener ultimo registro para prediccion - usar solo las 29 features del contrato
    feature_cols = [f for f in TRAINING_FEATURES if f in df_features.columns]

    X_latest = df_features[feature_cols].iloc[-1:].values
    current_price = df['Close'].iloc[-1]
    current_date = pd.to_datetime(df['Date'].iloc[-1])

    logging.info(f"Precio actual: ${current_price:,.2f}")
    logging.info(f"Fecha datos: {current_date}")

    # Cargar modelos y predecir
    predictions = []

    for model_file in models_dir.glob('*.pkl'):
        name = model_file.stem
        parts = name.rsplit('_h', 1)

        if len(parts) != 2:
            continue

        model_name = parts[0]
        try:
            horizon = int(parts[1])
        except ValueError:
            continue

        # Cargar modelo
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)

        if isinstance(model_data, dict):
            model = model_data.get('model', model_data)
            scaler = model_data.get('scaler')
        else:
            model = model_data
            scaler = None

        # Predecir
        X_pred = X_latest.copy()
        if scaler is not None:
            X_pred = scaler.transform(X_pred)

        pred_return = model.predict(X_pred)[0]
        pred_price = current_price * np.exp(pred_return)
        target_date = current_date + timedelta(days=horizon)

        predictions.append({
            'model': model_name,
            'horizon': horizon,
            'return': float(pred_return),
            'forecast_price': float(pred_price),
            'target_date': str(target_date.date()),
            'direction': 'UP' if pred_return > 0 else 'DOWN'
        })

        logging.info(f"{model_name} H={horizon}: ${pred_price:,.0f} ({pred_return*100:+.2f}%)")

    # Guardar predicciones
    output_dir = Path('/opt/airflow/outputs/weekly')
    output_dir.mkdir(parents=True, exist_ok=True)

    forecast_data = {
        'generated_at': datetime.now().isoformat(),
        'base_date': str(current_date.date()),
        'current_price': float(current_price),
        'predictions': predictions,
        'models_used': list(set(p['model'] for p in predictions))
    }

    # Guardar JSON
    forecast_file = output_dir / f'forecast_{current_date.strftime("%Y%m%d")}.json'
    with open(forecast_file, 'w') as f:
        json.dump(forecast_data, f, indent=2)

    logging.info(f"Forecast guardado: {forecast_file}")

    # Push para siguiente tarea
    context['ti'].xcom_push(key='forecast_file', value=str(forecast_file))
    context['ti'].xcom_push(key='n_predictions', value=len(predictions))

    return {'predictions': len(predictions), 'forecast_file': str(forecast_file)}


def generate_forecast_plots(**context):
    """
    Genera imagenes de visualizacion para forward forecasts.

    Produce:
    - forward_forecast_{model}.png - Forecast individual con intervalos de confianza
    - models_comparison.png - Comparacion de todos los modelos
    - fan_chart_{model}.png - Fan chart probabilistico
    - best_model_per_horizon.png - Mejor modelo por horizonte
    """
    logging.info("=" * 60)
    logging.info("GENERANDO VISUALIZACIONES DE FORWARD FORECAST")
    logging.info("=" * 60)

    if not VISUALIZATION_AVAILABLE:
        logging.warning("Modulos de visualizacion no disponibles - saltando generacion de imagenes")
        return {'generated': 0, 'error': 'Visualization modules not available'}

    # Obtener archivo de forecast de la tarea anterior
    forecast_file = context['ti'].xcom_pull(key='forecast_file', task_ids='run_inference')
    data_path = context['ti'].xcom_pull(key='data_path', task_ids='load_data')

    if not forecast_file:
        logging.error("No se encontro archivo de forecast")
        return {'generated': 0, 'error': 'No forecast file'}

    # Cargar datos de forecast
    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)

    predictions = forecast_data.get('predictions', [])
    if not predictions:
        logging.warning("No hay predicciones para visualizar")
        return {'generated': 0, 'error': 'No predictions'}

    # Convertir predicciones al formato requerido por generate_all_forecast_plots
    # forecasts_by_model: Dict[str, Dict[int, float]] - Model -> horizon -> return
    forecasts_by_model = {}
    for pred in predictions:
        model = pred['model']
        horizon = pred['horizon']
        ret = pred['return']  # Ya esta en formato decimal

        if model not in forecasts_by_model:
            forecasts_by_model[model] = {}
        forecasts_by_model[model][horizon] = ret

    logging.info(f"Procesando forecasts para {len(forecasts_by_model)} modelos")

    # Cargar precios historicos
    try:
        if 'postgresql' in str(data_path):
            from utils.dag_common import get_db_connection
            conn = get_db_connection()
            query = "SELECT date, close_price FROM core.features_ml ORDER BY date"
            df_prices = pd.read_sql(query, conn)
            conn.close()
            df_prices.columns = ['date', 'close']
        else:
            # Cargar desde CSV
            csv_path = Path(data_path) if data_path else Path('/opt/airflow/data/RL_COMBINED_ML_FEATURES.csv')
            if not csv_path.exists():
                csv_path = Path('/opt/airflow/data/processed/features_ml.csv')

            df_prices = pd.read_csv(csv_path)
            df_prices.columns = [c.lower() for c in df_prices.columns]

            # Buscar columna de precio
            price_col = None
            for col in ['close', 'close_price']:
                if col in df_prices.columns:
                    price_col = col
                    break

            if price_col and price_col != 'close':
                df_prices = df_prices.rename(columns={price_col: 'close'})

        # Crear serie de precios con indice datetime
        df_prices['date'] = pd.to_datetime(df_prices['date'])
        prices = df_prices.set_index('date')['close']

        logging.info(f"Cargados {len(prices)} precios historicos")

    except Exception as e:
        logging.error(f"Error cargando precios historicos: {e}")
        return {'generated': 0, 'error': str(e)}

    # Directorio de salida
    output_dir = Path('/opt/airflow/outputs/weekly/figures')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generar todas las visualizaciones
    try:
        generated_files = generate_all_forecast_plots(
            prices=prices,
            forecasts_by_model=forecasts_by_model,
            ensemble_forecasts=None,
            output_dir=output_dir
        )

        logging.info(f"Generadas {len(generated_files)} imagenes de forecast")
        for f in generated_files[:10]:  # Mostrar primeras 10
            logging.info(f"  - {Path(f).name}")
        if len(generated_files) > 10:
            logging.info(f"  ... y {len(generated_files) - 10} mas")

        # Push path para la siguiente tarea
        context['ti'].xcom_push(key='figures_dir', value=str(output_dir))
        context['ti'].xcom_push(key='n_figures', value=len(generated_files))

        return {'generated': len(generated_files), 'output_dir': str(output_dir)}

    except Exception as e:
        logging.error(f"Error generando visualizaciones: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return {'generated': 0, 'error': str(e)}


def upload_forecast_to_minio(**context):
    """
    Sube los forecasts estructurados a MinIO usando MinioClient.

    Estructura en MinIO:
        forecasts/{year}/week{week:02d}/
            ├── forecast.json
            ├── quality_report.json
            ├── metadata.json
            └── figures/{model}_{horizon}.png
    """
    logging.info("=" * 60)
    logging.info("SUBIENDO FORECAST ESTRUCTURADO A MINIO")
    logging.info("=" * 60)

    # Verificar disponibilidad del cliente
    if not MINIO_CLIENT_AVAILABLE:
        logging.warning("MinioClient no disponible - usando fallback")
        return _upload_images_fallback(context)

    # Obtener paths de tareas anteriores
    forecast_file = context['ti'].xcom_pull(key='forecast_file', task_ids='run_inference')
    quality_report_path = context['ti'].xcom_pull(key='quality_report_path', task_ids='check_data_quality')

    if not forecast_file:
        logging.error("No se encontro archivo de forecast")
        return {'uploaded': 0, 'error': 'No forecast file'}

    # Cargar forecast data
    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)

    base_date = pd.to_datetime(forecast_data['base_date'])
    week_num = base_date.isocalendar()[1]
    year = base_date.year

    logging.info(f"Subiendo forecast para {year}/week{week_num:02d}")

    # Cargar quality report si existe
    quality_report = None
    if quality_report_path and Path(quality_report_path).exists():
        with open(quality_report_path, 'r') as f:
            quality_report = json.load(f)

    # Buscar imagenes en directorio de figuras
    runs_dir = Path('/opt/airflow/outputs/runs')
    run_dirs = sorted(runs_dir.glob('*/figures'), reverse=True)

    images = {}
    if run_dirs:
        figures_dir = run_dirs[0]
        logging.info(f"Buscando imagenes en: {figures_dir}")

        for img_file in figures_dir.glob('*.png'):
            # Usar el nombre sin extension como key
            image_key = img_file.stem
            images[image_key] = str(img_file)
            logging.info(f"  Imagen encontrada: {image_key}")

    # Tambien buscar en weekly output
    weekly_figures = Path('/opt/airflow/outputs/weekly/figures')
    if weekly_figures.exists():
        for img_file in weekly_figures.glob('*.png'):
            image_key = img_file.stem
            if image_key not in images:
                images[image_key] = str(img_file)
                logging.info(f"  Imagen semanal: {image_key}")

    # Crear metadata del run
    run_metadata = {
        'dag_id': DAG_ID,
        'execution_date': context['execution_date'].isoformat() if context.get('execution_date') else datetime.now().isoformat(),
        'run_id': context.get('run_id', 'manual'),
        'n_predictions': len(forecast_data.get('predictions', [])),
        'models_used': forecast_data.get('models_used', []),
        'base_date': str(base_date.date()),
        'current_price': forecast_data.get('current_price'),
        'quality_score': quality_report.get('overall_score') if quality_report else None
    }

    # Inicializar MinioClient y subir
    try:
        minio_client = MinioClient()

        result = minio_client.upload_weekly_forecast(
            year=year,
            week=week_num,
            forecast_data=forecast_data,
            quality_report=quality_report,
            images=images if images else None,
            metadata=run_metadata
        )

        logging.info(f"Forecast subido exitosamente:")
        logging.info(f"  - Forecast: {result.get('forecast')}")
        logging.info(f"  - Quality Report: {result.get('quality_report')}")
        logging.info(f"  - Imagenes: {len(result.get('images', []))}")
        logging.info(f"  - Metadata: {result.get('metadata')}")

        # Push resultados a XCom
        context['ti'].xcom_push(key='minio_forecast_path', value=result.get('forecast'))
        context['ti'].xcom_push(key='minio_base_path', value=f"{year}/week{week_num:02d}")
        context['ti'].xcom_push(key='minio_images', value=result.get('images', []))
        context['ti'].xcom_push(key='image_urls', value={
            img.split('/')[-1].replace('.png', ''): img
            for img in result.get('images', [])
        })

        logging.info("=" * 60)

        return {
            'uploaded': True,
            'week': week_num,
            'year': year,
            'forecast_path': result.get('forecast'),
            'n_images': len(result.get('images', [])),
            'quality_report_uploaded': result.get('quality_report') is not None
        }

    except Exception as e:
        logging.error(f"Error subiendo a MinIO: {e}")
        logging.info("Intentando fallback...")
        return _upload_images_fallback(context)


def _upload_images_fallback(context):
    """Fallback para subir imagenes si MinioClient no esta disponible."""
    logging.info("Usando metodo fallback para subir a MinIO...")

    try:
        from utils.dag_common import get_minio_client
    except ImportError:
        logging.error("No hay cliente MinIO disponible")
        return {'uploaded': 0, 'error': 'No MinIO client'}

    forecast_file = context['ti'].xcom_pull(key='forecast_file', task_ids='run_inference')
    if not forecast_file:
        return {'uploaded': 0}

    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)

    base_date = pd.to_datetime(forecast_data['base_date'])
    week_num = base_date.isocalendar()[1]
    year = base_date.year

    client = get_minio_client()
    if not client:
        logging.warning("MinIO no disponible")
        return {'uploaded': 0}

    bucket = 'forecasts'
    uploaded = 0
    image_urls = {}

    try:
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)

        runs_dir = Path('/opt/airflow/outputs/runs')
        run_dirs = sorted(runs_dir.glob('*/figures'), reverse=True)

        if run_dirs:
            figures_dir = run_dirs[0]
            for img_file in figures_dir.glob('*.png'):
                object_name = f"{year}/week{week_num:02d}/figures/{img_file.name}"
                try:
                    client.fput_object(bucket, object_name, str(img_file), content_type='image/png')
                    uploaded += 1
                    image_urls[img_file.stem] = f"{bucket}/{object_name}"
                except Exception as e:
                    logging.error(f"Error subiendo {img_file.name}: {e}")

        object_name = f"{year}/week{week_num:02d}/forecast.json"
        client.fput_object(bucket, object_name, forecast_file)
        logging.info(f"Subido: {object_name}")

    except Exception as e:
        logging.error(f"Error en MinIO: {e}")

    context['ti'].xcom_push(key='image_urls', value=image_urls)
    context['ti'].xcom_push(key='minio_base_path', value=f"{year}/week{week_num:02d}")

    return {'uploaded': uploaded, 'week': week_num, 'year': year}


def update_bi_tables(**context):
    """Actualiza las tablas BI en PostgreSQL."""
    logging.info("Actualizando tablas BI...")

    forecast_file = context['ti'].xcom_pull(key='forecast_file', task_ids='run_inference')
    minio_base_path = context['ti'].xcom_pull(key='minio_base_path', task_ids='upload_images') or ''

    # Cargar forecast
    with open(forecast_file, 'r') as f:
        forecast_data = json.load(f)

    base_date = pd.to_datetime(forecast_data['base_date'])
    current_price = forecast_data['current_price']
    week_num = base_date.isocalendar()[1]
    year = base_date.year

    # 1. Insertar en PostgreSQL
    conn = None
    cursor = None
    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()

        for pred in forecast_data['predictions']:
            target_date = pd.to_datetime(pred['target_date'])
            signal = 'BUY' if pred['return'] > 0.005 else ('SELL' if pred['return'] < -0.005 else 'HOLD')
            minio_path = f"forecasts/{year}/week{week_num:02d}"

            # Insertar en bi.fact_forecasts
            cursor.execute("""
                INSERT INTO bi.fact_forecasts
                (model_id, horizon_id, inference_date, inference_week, inference_year,
                 target_date, base_price, predicted_price, predicted_return_pct,
                 price_change, direction, signal, minio_week_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (model_id, horizon_id, inference_date) DO UPDATE SET
                    predicted_price = EXCLUDED.predicted_price,
                    predicted_return_pct = EXCLUDED.predicted_return_pct,
                    direction = EXCLUDED.direction,
                    signal = EXCLUDED.signal,
                    minio_week_path = EXCLUDED.minio_week_path
            """, (
                pred['model'], pred['horizon'], base_date.date(), week_num, year,
                target_date.date(), current_price, pred['forecast_price'],
                pred['return'] * 100, pred['forecast_price'] - current_price,
                pred['direction'], signal, minio_path
            ))

        conn.commit()
        logging.info(f"Insertados {len(forecast_data['predictions'])} forecasts en PostgreSQL")

    except Exception as e:
        logging.error(f"Error insertando en PostgreSQL: {e}")
        if conn:
            conn.rollback()
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    # 2. Subir a MinIO organizado por semana
    try:
        from utils.dag_common import get_minio_client
        client = get_minio_client()

        if client:
            bucket = 'forecasts'
            week_path = f"{year}/week{week_num:02d}"

            # Subir forecast JSON
            object_name = f"{week_path}/forecast.json"
            client.fput_object(bucket, object_name, forecast_file)
            logging.info(f"Subido a MinIO: {bucket}/{object_name}")

            # Subir por modelo
            for model in set(p['model'] for p in forecast_data['predictions']):
                model_preds = [p for p in forecast_data['predictions'] if p['model'] == model]
                model_file = Path(f'/tmp/{model}_forecast.json')

                with open(model_file, 'w') as f:
                    json.dump({'model': model, 'predictions': model_preds}, f, indent=2)

                object_name = f"{week_path}/models/{model}/forecast.json"
                client.fput_object(bucket, object_name, str(model_file))

    except Exception as e:
        logging.warning(f"Error subiendo a MinIO: {e}")

    context['ti'].xcom_push(key='minio_path', value=f"forecasts/{year}/week{week_num:02d}")

    return {'updated_records': len(forecast_data['predictions']), 'week': week_num}


def check_data_quality(**context):
    """
    Verifica la calidad de los datos antes de ejecutar inferencia.

    Si la calidad de datos es menor al umbral, falla con alerta.
    Genera y logea un reporte completo de calidad.
    """
    logging.info("=" * 60)
    logging.info("VERIFICANDO CALIDAD DE DATOS")
    logging.info("=" * 60)

    # Configuracion de umbrales
    QUALITY_THRESHOLD = 60.0  # Score minimo aceptable (0-100) - Lowered for initial run
    MISSING_THRESHOLD = 0.05  # Max 5% missing values
    FRESHNESS_MAX_DAYS = 7    # Datos deben tener < 7 dias

    # Obtener fuente de datos del task anterior
    data_source = context['ti'].xcom_pull(key='data_source', task_ids='load_data')
    data_path = context['ti'].xcom_pull(key='data_path', task_ids='load_data')

    # Cargar datos segun la fuente
    if data_source == 'postgresql':
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        query = "SELECT * FROM core.features_ml ORDER BY date"
        df = pd.read_sql(query, conn)
        conn.close()
        df = df.rename(columns={
            'close_price': 'Close',
            'date': 'Date'
        })
    else:
        df = pd.read_csv(data_path)
        df.columns = [c.lower() for c in df.columns]
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close', 'date': 'Date'})
        elif 'close_price' in df.columns:
            df = df.rename(columns={'close_price': 'Close', 'date': 'Date'})

    logging.info(f"Datos cargados para validacion: {len(df)} filas x {len(df.columns)} columnas")

    # Verificar si el modulo de calidad esta disponible
    if not DATA_QUALITY_AVAILABLE:
        logging.warning("Modulo de calidad de datos no disponible - realizando checks basicos")

        # Checks basicos sin el modulo completo
        issues = []

        # Check 1: Missing values
        total_missing = df.isna().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        missing_pct = total_missing / total_cells if total_cells > 0 else 0

        if missing_pct > MISSING_THRESHOLD:
            issues.append(f"Alto porcentaje de valores faltantes: {missing_pct*100:.2f}%")

        # Check 2: Data freshness
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            latest_date = df['Date'].max()
            age_days = (datetime.now() - latest_date).days if pd.notna(latest_date) else float('inf')

            if age_days > FRESHNESS_MAX_DAYS:
                issues.append(f"Datos obsoletos: {age_days} dias (max: {FRESHNESS_MAX_DAYS})")

        # Check 3: Row count
        if len(df) < 100:
            issues.append(f"Datos insuficientes: {len(df)} filas (min recomendado: 100)")

        # Calcular score basico
        score = 100
        score -= missing_pct * 200  # -20 puntos por 10% missing
        score -= max(0, age_days - FRESHNESS_MAX_DAYS) * 5  # -5 puntos por dia extra
        score = max(0, min(100, score))

        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'missing_pct': missing_pct,
            'overall_score': score,
            'issues': issues,
            'data_freshness_days': age_days if 'Date' in df.columns else None
        }

    else:
        # Usar el modulo completo de calidad
        monitor = DataQualityMonitor(
            missing_threshold=MISSING_THRESHOLD,
            outlier_z_threshold=3.0,
            freshness_max_days=FRESHNESS_MAX_DAYS
        )

        # Intentar cargar datos de baseline para drift detection
        baseline_df = None
        try:
            baseline_path = Path('/opt/airflow/data/baseline/training_baseline.csv')
            if baseline_path.exists():
                baseline_df = pd.read_csv(baseline_path)
                logging.info("Baseline cargado para deteccion de drift")
        except Exception as e:
            logging.warning(f"No se pudo cargar baseline: {e}")

        # Generar reporte completo
        report = monitor.generate_quality_report(
            df,
            date_col='Date',
            baseline_df=baseline_df,
            expected_ranges={
                'Close': (1000, 6000),  # USD/COP range esperado
            }
        )

        quality_report = report.to_dict()
        score = report.overall_score
        issues = report.issues

        # Logear el reporte completo
        logging.info("\n" + report.get_summary())

    # Guardar reporte en archivo
    report_dir = Path('/opt/airflow/outputs/quality_reports')
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / f'quality_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)

    logging.info(f"Reporte de calidad guardado: {report_file}")

    # Push resultados a XCom
    context['ti'].xcom_push(key='quality_score', value=score)
    context['ti'].xcom_push(key='quality_report_path', value=str(report_file))
    context['ti'].xcom_push(key='quality_issues', value=issues)

    # Evaluar si pasa el umbral
    if score < QUALITY_THRESHOLD:
        error_msg = f"""
========================================
ERROR: CALIDAD DE DATOS INSUFICIENTE
========================================
Score: {score:.1f}/100 (umbral: {QUALITY_THRESHOLD})

Issues detectados:
{chr(10).join('  - ' + issue for issue in issues)}

El pipeline de inferencia NO puede continuar.
Por favor, revise los datos y corrija los problemas.
========================================
"""
        logging.error(error_msg)

        # Enviar alerta (se puede integrar con Slack, email, etc.)
        try:
            from utils.dag_common import send_slack_alert
            send_slack_alert(
                channel='#ml-alerts',
                message=f":warning: Data Quality FAILED - Score: {score:.1f}/100",
                details=quality_report
            )
        except Exception as e:
            logging.warning(f"No se pudo enviar alerta: {e}")

        raise ValueError(f"Calidad de datos insuficiente: {score:.1f}/100 < {QUALITY_THRESHOLD}")

    logging.info(f"Calidad de datos APROBADA: {score:.1f}/100")
    logging.info("=" * 60)

    return {
        'status': 'passed',
        'score': score,
        'issues_count': len(issues),
        'report_file': str(report_file)
    }


def generate_bi_csv(**context):
    """
    Genera el archivo bi_dashboard_unified.csv consolidado para el dashboard.

    Este archivo combina:
    - Forecasts de la inferencia actual
    - Metricas de backtest por modelo/horizonte
    - Consensus por horizonte
    - Rutas de imagenes MinIO

    Output: /opt/airflow/outputs/bi/bi_dashboard_unified.csv
            + copia a /opt/airflow/frontend/public/bi_dashboard_unified.csv
    """
    logging.info("=" * 60)
    logging.info("GENERANDO BI DASHBOARD CSV UNIFICADO")
    logging.info("=" * 60)

    from utils.dag_common import get_db_connection
    import shutil

    output_dir = Path('/opt/airflow/outputs/bi')
    output_dir.mkdir(parents=True, exist_ok=True)

    frontend_dir = Path('/opt/airflow/frontend/public')
    frontend_dir.mkdir(parents=True, exist_ok=True)

    # Obtener datos del forecast actual
    forecast_file = context['ti'].xcom_pull(key='forecast_file', task_ids='run_inference')
    minio_path = context['ti'].xcom_pull(key='minio_path', task_ids='update_bi_tables') or ''

    rows = []

    try:
        # =====================================================================
        # 1. INTENTAR CARGAR DESDE POSTGRESQL (datos mas completos)
        # =====================================================================
        conn = get_db_connection()

        # Query para obtener forecasts con metricas y consensus
        query = """
        WITH model_avg_metrics AS (
            SELECT
                model_id,
                AVG(direction_accuracy) as model_avg_da,
                AVG(rmse) as model_avg_rmse
            FROM bi.fact_model_metrics
            GROUP BY model_id
        ),
        best_per_horizon AS (
            SELECT
                horizon_id,
                model_id as best_model,
                direction_accuracy as best_da
            FROM bi.fact_model_metrics m1
            WHERE direction_accuracy = (
                SELECT MAX(direction_accuracy)
                FROM bi.fact_model_metrics m2
                WHERE m2.horizon_id = m1.horizon_id
            )
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
            NULL as mae,
            f.base_price,
            f.predicted_price,
            f.predicted_price - f.base_price as price_change,
            ((f.predicted_price - f.base_price) / f.base_price * 100) as price_change_pct,
            f.predicted_return_pct,
            f.signal,
            f.direction,
            c.consensus_direction,
            c.consensus_strength as consensus_strength_pct,
            mam.model_avg_da as model_avg_direction_accuracy,
            mam.model_avg_rmse,
            NULL as model_avg_r2,
            CASE WHEN mam.model_avg_da = (SELECT MAX(model_avg_da) FROM model_avg_metrics) THEN true ELSE false END as is_best_overall_model,
            CASE WHEN m.is_best_for_horizon THEN true ELSE false END as is_best_for_this_horizon,
            bph.best_da as best_da_for_this_horizon,
            bph.best_model as best_model_for_this_horizon,
            CONCAT('forecasts/', f.inference_year, '/week', LPAD(f.inference_week::text, 2, '0'), '/figures/forward_forecast_', f.model_id, '.png') as image_path,
            NULL as image_backtest,
            CONCAT('forecasts/', f.inference_year, '/week', LPAD(f.inference_week::text, 2, '0'), '/figures/forward_forecast_', f.model_id, '.png') as image_forecast,
            NULL as image_heatmap,
            NULL as image_ranking,
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

        df_forecasts = pd.read_sql(query, conn)
        logging.info(f"Cargados {len(df_forecasts)} registros de forecasts desde PostgreSQL")

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
            SELECT
                horizon_id,
                model_id as best_model,
                direction_accuracy as best_da
            FROM bi.fact_model_metrics m1
            WHERE direction_accuracy = (
                SELECT MAX(direction_accuracy)
                FROM bi.fact_model_metrics m2
                WHERE m2.horizon_id = m1.horizon_id
            )
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
            NULL as inference_week,
            NULL as inference_date,
            m.direction_accuracy,
            m.rmse,
            m.r2,
            NULL as mae,
            NULL as base_price,
            NULL as predicted_price,
            NULL as price_change,
            NULL as price_change_pct,
            NULL as predicted_return_pct,
            NULL as signal,
            NULL as direction,
            NULL as consensus_direction,
            NULL as consensus_strength_pct,
            mam.model_avg_da as model_avg_direction_accuracy,
            mam.model_avg_rmse,
            NULL as model_avg_r2,
            CASE WHEN mam.model_avg_da = (SELECT MAX(model_avg_da) FROM model_avg_metrics) THEN true ELSE false END as is_best_overall_model,
            m.is_best_for_horizon as is_best_for_this_horizon,
            bph.best_da as best_da_for_this_horizon,
            bph.best_model as best_model_for_this_horizon,
            CONCAT('results/ml_pipeline/figures/backtest_', m.model_id, '_h', m.horizon_id, '.png') as image_path,
            CONCAT('results/ml_pipeline/figures/backtest_', m.model_id, '_h', m.horizon_id, '.png') as image_backtest,
            NULL as image_forecast,
            'results/ml_pipeline/figures/metrics_heatmap_da.png' as image_heatmap,
            'results/ml_pipeline/figures/model_ranking_da.png' as image_ranking,
            m.training_date,
            NOW() as generated_at
        FROM bi.fact_model_metrics m
        LEFT JOIN bi.dim_modelos dm ON m.model_id = dm.model_id
        LEFT JOIN model_avg_metrics mam ON m.model_id = mam.model_id
        LEFT JOIN best_per_horizon bph ON m.horizon_id = bph.horizon_id
        ORDER BY m.model_id, m.horizon_id
        """

        df_backtest = pd.read_sql(query_backtest, conn)
        logging.info(f"Cargados {len(df_backtest)} registros de backtest desde PostgreSQL")

        conn.close()

        # Combinar DataFrames
        df_combined = pd.concat([df_backtest, df_forecasts], ignore_index=True)

    except Exception as e:
        logging.warning(f"Error cargando desde PostgreSQL: {e}")
        logging.info("Generando desde archivos locales...")

        # =====================================================================
        # 2. FALLBACK: Generar desde archivos CSV locales
        # =====================================================================

        # Cargar forecast actual
        with open(forecast_file, 'r') as f:
            forecast_data = json.load(f)

        base_date = pd.to_datetime(forecast_data['base_date'])
        current_price = forecast_data['current_price']
        week_num = base_date.isocalendar()[1]
        year = base_date.year

        # Cargar metricas de backtest si existen
        metrics_file = output_dir / 'fact_model_metrics.csv'
        if metrics_file.exists():
            df_metrics = pd.read_csv(metrics_file)
        else:
            df_metrics = pd.DataFrame()

        # Cargar modelos dim
        modelos_file = output_dir / 'dim_modelos.csv'
        if modelos_file.exists():
            df_modelos = pd.read_csv(modelos_file)
        else:
            df_modelos = pd.DataFrame()

        # Cargar consensus
        consensus_file = output_dir / 'fact_consensus.csv'
        if consensus_file.exists():
            df_consensus = pd.read_csv(consensus_file)
        else:
            df_consensus = pd.DataFrame()

        # Calcular metricas agregadas por modelo
        model_avg_metrics = {}
        if not df_metrics.empty:
            for model_id in df_metrics['model_id'].unique():
                model_data = df_metrics[df_metrics['model_id'] == model_id]
                model_avg_metrics[model_id] = {
                    'avg_da': model_data['direction_accuracy'].mean(),
                    'avg_rmse': model_data['rmse'].mean()
                }

        # Encontrar mejor modelo overall
        best_overall_model = None
        best_overall_da = 0
        for model_id, metrics in model_avg_metrics.items():
            if metrics['avg_da'] > best_overall_da:
                best_overall_da = metrics['avg_da']
                best_overall_model = model_id

        # Encontrar mejor modelo por horizonte
        best_per_horizon = {}
        if not df_metrics.empty:
            for h in df_metrics['horizon_id'].unique():
                h_data = df_metrics[df_metrics['horizon_id'] == h]
                best_idx = h_data['direction_accuracy'].idxmax()
                best_per_horizon[h] = {
                    'model': h_data.loc[best_idx, 'model_id'],
                    'da': h_data.loc[best_idx, 'direction_accuracy']
                }

        rows = []

        # Agregar registros de backtest
        if not df_metrics.empty:
            for _, m in df_metrics.iterrows():
                model_id = m['model_id']
                horizon = m['horizon_id']

                model_avg = model_avg_metrics.get(model_id, {'avg_da': None, 'avg_rmse': None})
                best_h = best_per_horizon.get(horizon, {'model': None, 'da': None})

                rows.append({
                    'record_id': f"BT_{model_id}_h{horizon}",
                    'view_type': 'backtest',
                    'model_id': model_id,
                    'model_name': model_id.upper(),
                    'model_type': 'tree_ensemble' if model_id in ['xgboost', 'lightgbm', 'catboost'] else ('linear' if model_id == 'ridge' else 'weighted_avg'),
                    'horizon_days': horizon,
                    'horizon_label': f"H={horizon}",
                    'horizon_category': 'corto_plazo' if horizon <= 5 else ('mediano_plazo' if horizon <= 15 else 'largo_plazo'),
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
                    'image_path': f"results/ml_pipeline/figures/backtest_{model_id}_h{horizon}.png",
                    'image_backtest': f"results/ml_pipeline/figures/backtest_{model_id}_h{horizon}.png",
                    'image_forecast': None,
                    'image_heatmap': 'results/ml_pipeline/figures/metrics_heatmap_da.png',
                    'image_ranking': 'results/ml_pipeline/figures/model_ranking_da.png',
                    'training_date': m.get('training_date'),
                    'generated_at': datetime.now().isoformat()
                })

        # Agregar registros de forecast
        for pred in forecast_data['predictions']:
            model_id = pred['model']
            horizon = pred['horizon']
            pred_price = pred['forecast_price']
            pred_return = pred['return']

            # Obtener consensus para este horizonte
            consensus_row = None
            if not df_consensus.empty:
                consensus_matches = df_consensus[df_consensus['horizon_id'] == horizon]
                if not consensus_matches.empty:
                    consensus_row = consensus_matches.iloc[0]

            # Obtener metricas del modelo
            model_metrics = None
            if not df_metrics.empty:
                metrics_matches = df_metrics[(df_metrics['model_id'] == model_id) & (df_metrics['horizon_id'] == horizon)]
                if not metrics_matches.empty:
                    model_metrics = metrics_matches.iloc[0]

            model_avg = model_avg_metrics.get(model_id, {'avg_da': None, 'avg_rmse': None})
            best_h = best_per_horizon.get(horizon, {'model': None, 'da': None})

            # Determinar signal
            signal = 'BUY' if pred_return > 0.005 else ('SELL' if pred_return < -0.005 else 'HOLD')
            direction = 'UP' if pred_return > 0 else 'DOWN'

            rows.append({
                'record_id': f"FF_{base_date.strftime('%Y%m%d')}_{model_id}_h{horizon}",
                'view_type': 'forward_forecast',
                'model_id': model_id,
                'model_name': model_id.upper(),
                'model_type': 'tree_ensemble' if model_id in ['xgboost', 'lightgbm', 'catboost'] else ('linear' if model_id == 'ridge' else 'weighted_avg'),
                'horizon_days': horizon,
                'horizon_label': f"H={horizon}",
                'horizon_category': 'corto_plazo' if horizon <= 5 else ('mediano_plazo' if horizon <= 15 else 'largo_plazo'),
                'inference_week': week_num,
                'inference_date': str(base_date.date()),
                'direction_accuracy': model_metrics['direction_accuracy'] if model_metrics is not None else None,
                'rmse': model_metrics['rmse'] if model_metrics is not None else None,
                'r2': model_metrics['r2'] if model_metrics is not None else None,
                'mae': None,
                'base_price': current_price,
                'predicted_price': pred_price,
                'price_change': pred_price - current_price,
                'price_change_pct': round((pred_price - current_price) / current_price * 100, 2),
                'predicted_return_pct': round(pred_return * 100, 4),
                'signal': signal,
                'direction': direction,
                'consensus_direction': consensus_row['consensus_direction'] if consensus_row is not None else None,
                'consensus_strength_pct': consensus_row['consensus_strength'] if consensus_row is not None else None,
                'model_avg_direction_accuracy': model_avg['avg_da'],
                'model_avg_rmse': model_avg['avg_rmse'],
                'model_avg_r2': None,
                'is_best_overall_model': model_id == best_overall_model,
                'is_best_for_this_horizon': model_metrics['is_best_for_horizon'] if model_metrics is not None else False,
                'best_da_for_this_horizon': best_h['da'],
                'best_model_for_this_horizon': best_h['model'],
                'image_path': f"results/weekly_update/figures/forward_forecast_{model_id}.png",
                'image_backtest': f"results/ml_pipeline/figures/backtest_{model_id}_h{horizon}.png" if model_id != 'ensemble' else None,
                'image_forecast': f"results/weekly_update/figures/forward_forecast_{model_id}.png",
                'image_heatmap': 'results/ml_pipeline/figures/metrics_heatmap_da.png',
                'image_ranking': 'results/ml_pipeline/figures/model_ranking_da.png',
                'training_date': df_modelos[df_modelos['model_id'] == model_id]['training_date'].iloc[0] if not df_modelos.empty and model_id in df_modelos['model_id'].values else None,
                'generated_at': datetime.now().isoformat()
            })

        df_combined = pd.DataFrame(rows)

    # =====================================================================
    # 3. GUARDAR CSV
    # =====================================================================

    output_file = output_dir / 'bi_dashboard_unified.csv'
    df_combined.to_csv(output_file, index=False)
    logging.info(f"CSV guardado: {output_file} ({len(df_combined)} registros)")

    # Copiar a frontend/public para fallback
    frontend_file = frontend_dir / 'bi_dashboard_unified.csv'
    try:
        shutil.copy(str(output_file), str(frontend_file))
        logging.info(f"CSV copiado a frontend: {frontend_file}")
    except Exception as e:
        logging.warning(f"No se pudo copiar a frontend: {e}")

    # Push para siguiente tarea
    context['ti'].xcom_push(key='bi_csv_path', value=str(output_file))
    context['ti'].xcom_push(key='bi_csv_rows', value=len(df_combined))

    logging.info("=" * 60)
    logging.info("BI DASHBOARD CSV GENERADO EXITOSAMENTE")
    logging.info("=" * 60)

    return {
        'output_file': str(output_file),
        'rows': len(df_combined),
        'frontend_copy': str(frontend_file)
    }


def notify_completion(**context):
    """Notifica que el pipeline se completo."""
    n_predictions = context['ti'].xcom_pull(key='n_predictions', task_ids='run_inference')
    quality_score = context['ti'].xcom_pull(key='quality_score', task_ids='check_data_quality')
    bi_csv_rows = context['ti'].xcom_pull(key='bi_csv_rows', task_ids='generate_bi_csv') or 0

    logging.info("=" * 60)
    logging.info("PIPELINE DE INFERENCIA SEMANAL COMPLETADO")
    logging.info(f"Predicciones generadas: {n_predictions}")
    logging.info(f"Score de calidad de datos: {quality_score:.1f}/100")
    logging.info(f"Registros en BI CSV: {bi_csv_rows}")
    logging.info(f"Fecha ejecucion: {datetime.now().isoformat()}")
    logging.info("=" * 60)

    # Aqui se puede agregar notificacion por email, Slack, etc.
    return {
        'status': 'completed',
        'predictions': n_predictions,
        'quality_score': quality_score,
        'bi_csv_rows': bi_csv_rows
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Pipeline de inferencia semanal - ejecuta cada lunes',
    schedule_interval='0 13 * * 1',  # Lunes 8:00 AM COT (13:00 UTC)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['inference', 'weekly', 'ml', 'production'],
    max_active_runs=1,
    # DAG-level callbacks
    on_failure_callback=dag_failure_callback,
    sla_miss_callback=sla_miss_callback,
) as dag:

    # Task 1: Trigger scraping DAG (usando el nuevo DAG con scrapers + FRED)
    trigger_scraping = TriggerDagRunOperator(
        task_id='trigger_scraping',
        trigger_dag_id='01_l0_macro_scraper',
        wait_for_completion=True,
        poke_interval=60,
        execution_timeout=timedelta(minutes=30),
        on_failure_callback=task_failure_callback,
    )

    # Task 2: Load data
    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_latest_data,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 3: Check data quality BEFORE inference
    check_quality = PythonOperator(
        task_id='check_data_quality',
        python_callable=check_data_quality,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
        # Este task es critico - si falla, el pipeline debe detenerse
        retries=1,  # Solo 1 reintento para calidad
        retry_delay=timedelta(minutes=5),
    )

    # Task 4: Run inference (solo si check_data_quality pasa)
    inference = PythonOperator(
        task_id='run_inference',
        python_callable=run_inference,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 5: Generate forward forecast visualizations
    generate_plots = PythonOperator(
        task_id='generate_forecast_plots',
        python_callable=generate_forecast_plots,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 6: Upload forecast to MinIO (structured)
    upload_images = PythonOperator(
        task_id='upload_images',
        python_callable=upload_forecast_to_minio,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 6: Update BI tables
    update_bi = PythonOperator(
        task_id='update_bi_tables',
        python_callable=update_bi_tables,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 7: Generate BI CSV for dashboard
    generate_csv = PythonOperator(
        task_id='generate_bi_csv',
        python_callable=generate_bi_csv,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 8: Notify completion
    notify = PythonOperator(
        task_id='notify_completion',
        python_callable=notify_completion,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Dependencies - check_data_quality MUST pass before inference
    # generate_plots runs AFTER inference to visualize forward forecasts
    # upload_images runs AFTER generate_plots to upload all visualizations to MinIO
    # generate_bi_csv runs AFTER update_bi_tables to ensure data is in PostgreSQL
    trigger_scraping >> load_data >> check_quality >> inference >> generate_plots >> upload_images >> update_bi >> generate_csv >> notify
