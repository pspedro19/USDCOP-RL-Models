"""
DAG: 04_l2_monthly_training
============================
Pipeline de Reentrenamiento Mensual - USD/COP Forecasting

Ejecuta el PRIMER DOMINGO de cada mes a las 2:00 AM (hora Colombia):
1. Cargar datos históricos con CORTE TEMPORAL (hasta último día del mes anterior)
2. Seleccionar las 29 features definidas en el contrato
3. Train/test split temporal (80/20) dentro del período elegible
4. Entrenar 10 modelos × 7 horizontes = 70 modelos
5. Evaluar y comparar modelos (Direction Accuracy, RMSE, MAE, R²)
6. Registrar en MLflow Model Registry
7. Subir modelos a MinIO con estructura organizada

CORTE TEMPORAL (CRÍTICO):
-------------------------
El training NUNCA ve datos del mes actual. Esto previene data leakage temporal.

Ejemplo ejecutado el 3 de enero 2026:
- Datos disponibles: 2020-01-01 a 2026-01-03
- Cutoff aplicado: 2025-12-31
- Training usa: datos hasta 2025-12-31 (80%)
- Test usa: últimos datos hasta 2025-12-31 (20%)
- Datos de enero 2026: EXCLUIDOS del entrenamiento

CONTRATO DE FEATURES (29):
--------------------------
DXY (3): dxy, dxy_z, dxy_change_1d
VIX (3): vix, vix_z, vix_regime
Commodities (4): brent, brent_z, wti, wti_z
EMBI (2): embi, embi_z
Tasas (6): curve_slope, curve_slope_z, col_us_spread, col_us_spread_z,
           policy_spread, policy_spread_z
Carry (1): carry_favorable
Commodities adicionales (2): coffee_z, gold_z
Macro Colombia (6): terms_trade, trade_balance_z, exports, fdi_inflow, reserves, itcr
Pares regionales (2): usdmxn_ret_1d, usdclp_ret_1d

HORIZONTES (7): [1, 5, 10, 15, 20, 25, 30] días
MODELOS (10): Ridge, Bayesian Ridge, ARD, XGBoost Pure, LightGBM Pure,
              CatBoost Pure, DART XGBoost, Hybrid XGBoost, Hybrid LightGBM,
              Hybrid CatBoost

Schedule: 0 7 1-7 * 0 (2:00 AM COT = 7:00 UTC, primer domingo del mes)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
import pandas as pd
import numpy as np
import json
import pickle
import joblib
import logging
import os
import sys
from pathlib import Path

# Agregar path del backend
sys.path.insert(0, '/opt/airflow/backend')

# MinIO client imports
try:
    from src.mlops.minio_client import MinioClient, MODELS_BUCKET
    MINIO_CLIENT_AVAILABLE = True
except ImportError:
    MINIO_CLIENT_AVAILABLE = False
    logging.warning("MinIO client module not available")

# Optuna tuner imports (FASE 1)
try:
    from src.models.optuna_tuner import OptunaTuner
    from src.core.config import OptunaConfig
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna tuner not available")

# Walk-forward backtest imports (FASE 3)
try:
    from src.evaluation.walk_forward_backtest import walk_forward_backtest, BacktestResult
    WALKFORWARD_AVAILABLE = True
except ImportError:
    WALKFORWARD_AVAILABLE = False
    logging.warning("Walk-forward backtest not available")

# Visualization imports (backtest images)
try:
    from src.visualization.backtest_plots import BacktestPlotter
    from src.visualization.model_plots import ModelComparisonPlotter
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization modules not available")

# Importar callbacks y utilidades
from utils.callbacks import (
    task_failure_callback,
    task_success_callback,
    dag_failure_callback,
    sla_miss_callback,
    task_retry_callback,
    get_alert_email,
)

DAG_ID = "04_l2_monthly_training"

# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = [1, 5, 10, 15, 20, 25, 30]  # 7 horizontes

# 9 MODELOS (consistente con run_hybrid_improved.py):
# - 3 Lineales: ridge, bayesian_ridge, ard
# - 3 Boosting Puros: xgboost_pure, lightgbm_pure, catboost_pure
# - 3 Híbridos: hybrid_xgboost, hybrid_lightgbm, hybrid_catboost
ML_MODELS = [
    'ridge', 'bayesian_ridge', 'ard',                    # Lineales
    'xgboost_pure', 'lightgbm_pure', 'catboost_pure',    # Boosting Puros
    'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'  # Híbridos
]

# =============================================================================
# PIPELINE CONFIGURATION - 3 FASES INTEGRADAS
# =============================================================================
# FASE 1: Optuna Tuning (WalkForwardPurged CV)
# FASE 2: Split Simple (80/20 + GAP 30d)
# FASE 3: Walk-Forward Backtest (Validación)
# =============================================================================

# FASE 1: Optuna Configuration
ENABLE_OPTUNA = True   # HABILITADO para primera ejecución completa
N_OPTUNA_TRIALS = 10   # Trials por modelo (10 para velocidad, 20+ para producción)
USE_CACHED_OPTUNA_PARAMS = True  # Si True, usa params de MinIO si existen
OPTUNA_PARAMS_BUCKET = 'ml-models'  # Bucket para guardar params
OPTUNA_PARAMS_KEY = 'optuna/best_hyperparameters.json'  # Key en MinIO

# FASE 2: Training Configuration
N_FEATURES = 15        # Reducido para evitar overfitting con ~1500 samples
TRAIN_SIZE = 0.8       # 80% train, 20% test
GAP_DAYS = 30          # Gap entre train y test = max(HORIZONS)

# FASE 3: Walk-Forward Backtest Configuration
ENABLE_WALKFORWARD = True    # Validación walk-forward post-training
WF_TRAIN_SIZE = 500          # ~2 años de datos para cada ventana
WF_TEST_SIZE = 60            # ~3 meses de test por ventana
WF_STEP_SIZE = 60            # Avanzar 60 días entre ventanas (non-overlapping)
WF_MIN_TRAIN_SIZE = 252      # Mínimo 1 año de train

# Backtest Images
GENERATE_BACKTEST_IMAGES = True  # Generar imágenes de backtest

# =============================================================================
# CHECKPOINT CONFIGURATION - Permite reanudar desde fases completadas
# =============================================================================
# Si FORCE_FULL_RESTART=True, ignora todos los checkpoints y ejecuta desde cero.
# Si False, cada fase verifica si ya completó para este mes y salta si es así.
FORCE_FULL_RESTART = True  # ACTIVADO para primera ejecución con Walk-Forward completo (9 modelos)

# Obtener email de alertas desde variable de entorno
ALERT_EMAIL = get_alert_email()

# =============================================================================
# DEFAULT ARGS - Con callbacks y reintentos
# =============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': [ALERT_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    # Reintentos con delay exponencial
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=60),
    'execution_timeout': timedelta(hours=4),
    # Callbacks
    'on_failure_callback': task_failure_callback,
    'on_success_callback': task_success_callback,
    'on_retry_callback': task_retry_callback,
}

# =============================================================================
# SISTEMA DE CHECKPOINT POR FASES - Permite reanudar desde fase fallida
# =============================================================================
# Archivo de checkpoint guarda estado de cada fase del pipeline mensual.
# Al reiniciar, el pipeline detecta qué fases ya completaron y las salta.
#
# FASES DEL PIPELINE:
#   1. data_preparation    - Carga y preparación de datos
#   2. optuna_tuning       - Optimización de hiperparámetros
#   3. training_<model>    - Entrenamiento de cada modelo (9 checkpoints)
#   4. aggregation         - Agregación de resultados
#   5. walk_forward        - Validación walk-forward
#   6. backtest_images     - Generación de imágenes
#   7. mlflow_register     - Registro en MLflow
#   8. minio_upload        - Subida a MinIO
#
# USO:
#   - FORCE_FULL_RESTART=True: Ignora checkpoints, ejecuta todo desde cero
#   - FORCE_FULL_RESTART=False: Salta fases completadas, continúa desde fallida

PIPELINE_CHECKPOINT_FILE = Path('/opt/airflow/outputs/training_temp/pipeline_checkpoint.json')


def get_current_month_id() -> str:
    """Retorna identificador único del mes actual: YYYY-MM"""
    return datetime.now().strftime('%Y-%m')


def load_pipeline_checkpoint() -> dict:
    """Carga el checkpoint del pipeline si existe y es del mes actual."""
    if FORCE_FULL_RESTART:
        logging.info("FORCE_FULL_RESTART=True, ignorando checkpoint existente")
        return {'month_id': get_current_month_id(), 'phases': {}, 'force_restart': True}

    if not PIPELINE_CHECKPOINT_FILE.exists():
        return {'month_id': get_current_month_id(), 'phases': {}}

    try:
        with open(PIPELINE_CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)

        # Verificar que es del mes actual
        if checkpoint.get('month_id') != get_current_month_id():
            logging.info(f"Checkpoint de mes anterior ({checkpoint.get('month_id')}), iniciando nuevo ciclo de entrenamiento")
            # Guardar el checkpoint anterior como backup
            backup_file = PIPELINE_CHECKPOINT_FILE.parent / f"checkpoint_backup_{checkpoint.get('month_id', 'unknown')}.json"
            try:
                with open(backup_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2, default=str)
                logging.info(f"Backup del checkpoint anterior: {backup_file}")
            except Exception:
                pass
            return {'month_id': get_current_month_id(), 'phases': {}}

        return checkpoint
    except Exception as e:
        logging.warning(f"Error cargando checkpoint: {e}")
        return {'month_id': get_current_month_id(), 'phases': {}}


def save_pipeline_checkpoint(checkpoint: dict):
    """Guarda el checkpoint del pipeline."""
    checkpoint['last_update'] = datetime.now().isoformat()
    PIPELINE_CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)


def mark_phase_complete(phase_name: str, result: dict = None):
    """Marca una fase como completada en el checkpoint."""
    checkpoint = load_pipeline_checkpoint()
    checkpoint['phases'][phase_name] = {
        'status': 'completed',
        'completed_at': datetime.now().isoformat(),
        'result': result or {}
    }
    save_pipeline_checkpoint(checkpoint)
    logging.info(f"*** CHECKPOINT: Fase '{phase_name}' marcada como COMPLETADA ***")


def mark_phase_failed(phase_name: str, error: str):
    """Marca una fase como fallida en el checkpoint."""
    checkpoint = load_pipeline_checkpoint()
    checkpoint['phases'][phase_name] = {
        'status': 'failed',
        'failed_at': datetime.now().isoformat(),
        'error': str(error)[:500]
    }
    save_pipeline_checkpoint(checkpoint)
    logging.warning(f"*** CHECKPOINT: Fase '{phase_name}' marcada como FALLIDA ***")


def is_phase_complete(phase_name: str) -> bool:
    """Verifica si una fase ya está completada en el checkpoint del mes actual."""
    if FORCE_FULL_RESTART:
        return False
    checkpoint = load_pipeline_checkpoint()
    phase_info = checkpoint.get('phases', {}).get(phase_name, {})
    return phase_info.get('status') == 'completed'


def get_phase_result(phase_name: str) -> dict:
    """Obtiene el resultado guardado de una fase completada."""
    checkpoint = load_pipeline_checkpoint()
    phase_info = checkpoint.get('phases', {}).get(phase_name, {})
    return phase_info.get('result', {})


def get_checkpoint_output_dir() -> Path:
    """Obtiene el output_dir del checkpoint si existe."""
    checkpoint = load_pipeline_checkpoint()
    output_dir = checkpoint.get('output_dir')
    if output_dir:
        return Path(output_dir)
    return None


def set_checkpoint_output_dir(output_dir: Path):
    """Guarda el output_dir en el checkpoint."""
    checkpoint = load_pipeline_checkpoint()
    checkpoint['output_dir'] = str(output_dir)
    save_pipeline_checkpoint(checkpoint)


def get_checkpoint_summary() -> str:
    """Retorna resumen del estado del checkpoint."""
    checkpoint = load_pipeline_checkpoint()
    phases = checkpoint.get('phases', {})
    completed = [p for p, info in phases.items() if info.get('status') == 'completed']
    failed = [p for p, info in phases.items() if info.get('status') == 'failed']
    return f"Mes: {checkpoint.get('month_id')} | Completadas: {len(completed)} | Fallidas: {len(failed)}"


def print_checkpoint_status():
    """Imprime el estado completo del checkpoint en logs."""
    checkpoint = load_pipeline_checkpoint()
    logging.info("=" * 70)
    logging.info("ESTADO DEL CHECKPOINT DE ENTRENAMIENTO MENSUAL")
    logging.info("=" * 70)
    logging.info(f"  Mes ID: {checkpoint.get('month_id')}")
    logging.info(f"  Force Restart: {FORCE_FULL_RESTART}")
    logging.info(f"  Output Dir: {checkpoint.get('output_dir', 'No definido')}")
    logging.info(f"  Última actualización: {checkpoint.get('last_update', 'N/A')}")
    logging.info("-" * 70)

    phases = checkpoint.get('phases', {})
    if not phases:
        logging.info("  No hay fases registradas en el checkpoint")
    else:
        for phase_name, info in sorted(phases.items()):
            status = info.get('status', 'unknown')
            if status == 'completed':
                logging.info(f"  ✓ {phase_name}: COMPLETADA ({info.get('completed_at', 'N/A')})")
            else:
                logging.info(f"  ✗ {phase_name}: FALLIDA ({info.get('error', 'Unknown error')[:50]})")

    logging.info("=" * 70)


# =============================================================================
# TEMPORAL CUTOFF - Training uses data UP TO last complete month only
# =============================================================================

def get_training_cutoff_date(reference_date: datetime = None) -> datetime:
    """
    Calcula la fecha de corte para training: último día del mes anterior.

    Ejemplo:
    - Si reference_date = 2026-01-03, retorna 2025-12-31
    - Si reference_date = 2026-02-15, retorna 2026-01-31
    - Si reference_date = 2026-03-01, retorna 2026-02-28

    Esto asegura que el entrenamiento NUNCA vea datos del mes actual,
    previniendo data leakage temporal.

    Args:
        reference_date: Fecha de referencia (default: hoy)

    Returns:
        datetime: Último día del mes anterior
    """
    if reference_date is None:
        reference_date = datetime.now()

    # Primer día del mes actual
    first_day_current_month = reference_date.replace(day=1)

    # Último día del mes anterior = un día antes del primer día del mes actual
    last_day_previous_month = first_day_current_month - timedelta(days=1)

    return last_day_previous_month


# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

def get_mlflow_client():
    """Inicializa y retorna el cliente de MLflow."""
    import mlflow

    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'usdcop-forecasting')

    # Crear experimento si no existe
    # Use set_experiment which creates if not exists (race-condition safe)
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except Exception:
        # Experiment already created by another parallel task
        pass

    mlflow.set_experiment(experiment_name)

    return mlflow


# =============================================================================
# FUNCIONES DEL PIPELINE
# =============================================================================

def check_data_freshness(**context):
    """Verifica que los datos esten actualizados desde BD o CSV.

    IMPORTANTE: Además de verificar que existan filas, verifica que los features
    tengan datos (no sean NULL). Esto previene el problema donde PostgreSQL tiene
    filas pero los features están vacíos.
    """
    logging.info("Verificando frescura de datos...")

    df = None
    data_source = None
    data_path = None

    # Intentar cargar desde PostgreSQL primero
    try:
        from utils.dag_common import get_db_connection
        conn = get_db_connection()

        # CRÍTICO: Verificar no solo filas, sino también que los features tengan datos
        # Verificamos columnas clave que deben tener valores
        query = """
        SELECT
            COUNT(*) as total_rows,
            MAX(date) as last_date,
            COUNT(dxy) as dxy_count,
            COUNT(vix) as vix_count,
            COUNT(brent) as brent_count,
            COUNT(curve_slope) as curve_count
        FROM core.features_ml
        """
        result = pd.read_sql(query, conn)
        conn.close()

        total_rows = int(result.iloc[0]['total_rows'])
        dxy_count = int(result.iloc[0]['dxy_count'])
        vix_count = int(result.iloc[0]['vix_count'])

        # Calcular cobertura de features (% de filas con datos reales)
        feature_coverage = dxy_count / total_rows if total_rows > 0 else 0

        logging.info(f"PostgreSQL: {total_rows} filas, {dxy_count} con DXY ({feature_coverage:.1%} cobertura)")

        # SOLO usar PostgreSQL si hay >50% de cobertura de features
        # Si la cobertura es baja, significa que los features no fueron poblados
        if total_rows > 0 and feature_coverage > 0.50:
            data_source = 'postgresql'
            data_path = 'postgresql://core.features_ml'
            n_samples = total_rows
            last_date = pd.to_datetime(result.iloc[0]['last_date'])
            days_old = (datetime.now() - last_date).days

            logging.info(f"PostgreSQL APROBADO: {n_samples} registros, cobertura {feature_coverage:.1%}")

            if days_old > 7:
                logging.warning(f"Datos tienen {days_old} dias de antiguedad")

            context['ti'].xcom_push(key='data_source', value=data_source)
            context['ti'].xcom_push(key='data_path', value=data_path)
            context['ti'].xcom_push(key='n_samples', value=n_samples)
            return 'load_and_prepare_data'
        else:
            logging.warning(f"PostgreSQL RECHAZADO: cobertura de features muy baja ({feature_coverage:.1%})")
            logging.warning("Los features no están poblados en la base de datos. Usando CSV como fallback.")

    except Exception as e:
        logging.warning(f"No se pudo conectar a PostgreSQL: {e}")

    # ==========================================================================
    # FALLBACK CSV - DEPRECATED DUE TO DATA LEAKAGE
    # ==========================================================================
    # WARNING: CSV files contain z-scores calculated over the ENTIRE dataset,
    # which causes DATA LEAKAGE. The model "knows" future mean/std values.
    #
    # SOLUTION: Use PostgreSQL data populated by DAG 03_l1_feature_engineering
    # which calculates z-scores using ROLLING windows (50-day) to avoid leakage.
    # ==========================================================================

    logging.error("=" * 70)
    logging.error("CRITICAL: PostgreSQL data not available or incomplete!")
    logging.error("=" * 70)
    logging.error("")
    logging.error("The training pipeline REQUIRES data from PostgreSQL (core.features_ml)")
    logging.error("which is populated by DAG '03_l1_feature_engineering'.")
    logging.error("")
    logging.error("CSV FALLBACK IS DISABLED due to DATA LEAKAGE in z-scores:")
    logging.error("  - CSV z-scores were calculated over the ENTIRE dataset")
    logging.error("  - This means early dates 'know' future mean/std values")
    logging.error("  - Results in artificially inflated accuracy metrics")
    logging.error("")
    logging.error("TO FIX THIS:")
    logging.error("  1. Ensure base data is loaded (usdcop_historical + macro_indicators)")
    logging.error("  2. Run: airflow dags trigger 03_l1_feature_engineering")
    logging.error("  3. Wait for feature engineering to complete")
    logging.error("  4. Re-run this training DAG")
    logging.error("")
    logging.error("=" * 70)

    # Skip training if PostgreSQL is not available
    return 'skip_training'

    # ----- DEPRECATED CODE BELOW (CSV fallback disabled) -----
    # The following code is intentionally unreachable to prevent data leakage
    logging.info("Buscando archivo CSV de features...")

    # Lista de posibles archivos en orden de prioridad
    possible_files = [
        Path('/opt/airflow/data/processed/features_ml.csv'),  # Principal
        Path('/opt/airflow/data/archive/COMBINED_V2.csv'),     # Backup
        Path('/opt/airflow/data/raw/RL_COMBINED_ML_FEATURES.csv'),
        Path('/opt/airflow/data/RL_COMBINED_ML_FEATURES.csv'),
    ]

    # También buscar archivos features_*.csv en processed
    data_dir = Path('/opt/airflow/data/processed')
    feature_files = sorted(data_dir.glob('features_*.csv'), reverse=True)
    possible_files = feature_files + possible_files

    # Buscar el primer archivo que exista
    latest_file = None
    for f in possible_files:
        if f.exists():
            latest_file = f
            break

    if latest_file is None:
        logging.error("No se encontraron archivos de features en ninguna ubicación")
        logging.error("Ubicaciones buscadas:")
        for f in possible_files[:5]:
            logging.error(f"  - {f}")
        return 'skip_training'

    logging.info(f"Usando archivo: {latest_file}")
    df = pd.read_csv(latest_file)

    # Verificar calidad del CSV
    date_col = 'Date' if 'Date' in df.columns else 'date'
    if date_col in df.columns:
        last_date = pd.to_datetime(df[date_col]).max()
        days_old = (datetime.now() - last_date).days
        if days_old > 7:
            logging.warning(f"Datos tienen {days_old} dias de antiguedad")

    # Verificar que tenga suficientes columnas de features
    expected_features = ['dxy', 'vix', 'brent']
    available = [f for f in expected_features if f in df.columns]
    if len(available) < 2:
        logging.warning(f"CSV puede no tener features: solo {available} de {expected_features}")

    context['ti'].xcom_push(key='data_source', value='csv')
    context['ti'].xcom_push(key='data_path', value=str(latest_file))
    context['ti'].xcom_push(key='n_samples', value=len(df))

    logging.info(f"CSV: {len(df)} registros, {len(df.columns)} columnas")
    return 'load_and_prepare_data'


def load_and_prepare_data(**context):
    """
    Carga datos y prepara features desde BD o CSV.

    IMPORTANTE: Aplica corte temporal - entrena SOLO con datos hasta el último
    día del mes anterior. Esto previene data leakage del mes actual.

    Ejemplo (ejecutado el 3 de enero 2026):
    - Datos disponibles: hasta 2026-01-03
    - Cutoff aplicado: 2025-12-31
    - Training usa: datos hasta 2025-12-31
    - Test usa: datos de enero 2026 (si hay)
    """
    PHASE_NAME = 'data_preparation'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    print_checkpoint_status()

    if is_phase_complete(PHASE_NAME):
        logging.info("=" * 60)
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        logging.info("  (Para forzar re-ejecución, cambiar FORCE_FULL_RESTART=True)")
        logging.info("=" * 60)
        # Recuperar resultado anterior del checkpoint
        prev_result = get_phase_result(PHASE_NAME)
        context['ti'].xcom_push(key='train_samples', value=prev_result.get('train', 0))
        context['ti'].xcom_push(key='test_samples', value=prev_result.get('test', 0))
        context['ti'].xcom_push(key='n_features', value=prev_result.get('features', 0))
        context['ti'].xcom_push(key='cutoff_date', value=prev_result.get('cutoff_date', ''))
        return prev_result

    logging.info("=" * 60)
    logging.info("CARGANDO Y PREPARANDO DATOS CON CORTE TEMPORAL")
    logging.info("=" * 60)

    from src.features.transformer import FeatureTransformer

    data_source = context['ti'].xcom_pull(key='data_source', task_ids='check_data')
    data_path = context['ti'].xcom_pull(key='data_path', task_ids='check_data')

    # Cargar datos segun la fuente
    if data_source == 'postgresql':
        from utils.dag_common import get_db_connection
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM core.features_ml ORDER BY date", conn)
        conn.close()
        df = df.rename(columns={
            'close_price': 'Close',
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'date': 'Date'
        })
    else:
        df = pd.read_csv(data_path)
        df.columns = [c.lower() if c not in ['Date', 'Close', 'Open', 'High', 'Low'] else c for c in df.columns]
        if 'close' in df.columns:
            df = df.rename(columns={'close': 'Close', 'date': 'Date'})

    logging.info(f"Datos cargados: {len(df)} registros totales")

    # Asegurar que Date es datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # =========================================================================
    # PREPROCESAMIENTO DE NaN: ESTRATEGIA SIN DATA LEAKAGE
    # =========================================================================
    # ANÁLISIS DE 4 AGENTES ESPECIALISTAS CONCLUYE:
    #
    # ❌ PROHIBIDO: bfill (usa datos futuros = data leakage)
    # ❌ PROHIBIDO: fillna(0) (crea outliers extremos, ej: reserves=0 es -20 std)
    # ❌ PROHIBIDO: interpolate (usa datos futuros)
    #
    # ✅ PERMITIDO: ffill (propaga valores pasados)
    # ✅ PERMITIDO: dropna en warmup period (primeras ~90 filas)
    # ✅ PERMITIDO: Mantener NaN en targets al final (es correcto)
    #
    # ESTRATEGIA IMPLEMENTADA:
    # 1. Ordenar por fecha
    # 2. ffill SOLO para features (no targets)
    # 3. Eliminar filas del warmup period (donde ffill no puede llenar)
    # 4. Mantener NaN en targets (rows para predicción, no training)
    # =========================================================================
    logging.info("  [NaN] Aplicando estrategia sin data leakage...")

    # Ordenar por fecha ANTES de cualquier operación
    df = df.sort_values('Date').reset_index(drop=True)
    n_rows_before = len(df)

    # Identificar columnas por tipo
    all_cols = df.columns.tolist()
    target_cols = [c for c in all_cols if c.startswith('target_')]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in target_cols and c != 'Date']

    # Contar NaN antes
    nan_features_before = df[feature_cols].isna().sum()
    nan_targets_before = df[target_cols].isna().sum() if target_cols else pd.Series()
    total_nan_features = nan_features_before.sum()

    if total_nan_features > 0:
        cols_with_nan = nan_features_before[nan_features_before > 0]
        logging.info(f"  [NaN] {total_nan_features} NaN en features ({len(cols_with_nan)} columnas)")

        # PASO 1: ffill SOLO para features (no targets)
        for col in feature_cols:
            if df[col].isna().any():
                n_before = df[col].isna().sum()
                df[col] = df[col].ffill()  # Solo ffill, NUNCA bfill
                n_after = df[col].isna().sum()
                if n_before > 30:
                    logging.info(f"    ffill {col}: {n_before} → {n_after} NaN")

        # PASO 2: Identificar warmup period (filas donde ffill no pudo llenar)
        # Estas son las primeras filas sin datos previos para propagar
        nan_after_ffill = df[feature_cols].isna().sum()
        cols_still_nan = nan_after_ffill[nan_after_ffill > 0]

        if len(cols_still_nan) > 0:
            # Encontrar la primera fila donde TODOS los features están completos
            all_features_valid = ~df[feature_cols].isna().any(axis=1)
            first_valid_idx = all_features_valid.idxmax() if all_features_valid.any() else 0

            warmup_rows = first_valid_idx
            warmup_pct = 100 * warmup_rows / len(df)

            if warmup_rows > 0 and warmup_pct < 15:  # Max 15% de pérdida
                logging.info(f"  [NaN] Warmup period: {warmup_rows} filas ({warmup_pct:.1f}%)")
                logging.info(f"    Fechas eliminadas: {df.iloc[0]['Date']} a {df.iloc[warmup_rows-1]['Date']}")

                # Eliminar warmup period
                df = df.iloc[warmup_rows:].reset_index(drop=True)
            elif warmup_pct >= 15:
                logging.warning(f"  [NaN] Warmup muy largo ({warmup_rows} filas, {warmup_pct:.1f}%), manteniendo datos")
                # Para features que aún tienen NaN, log warning
                for col, n_nan in cols_still_nan.items():
                    logging.warning(f"    {col}: {n_nan} NaN restantes (considerar excluir)")

        n_rows_after = len(df)
        rows_removed = n_rows_before - n_rows_after

        # Verificación final
        final_nan_features = df[feature_cols].isna().sum().sum()
        final_nan_targets = df[target_cols].isna().sum().sum() if target_cols else 0

        logging.info(f"  [NaN] Resultado: {n_rows_before} → {n_rows_after} filas (-{rows_removed})")
        logging.info(f"    Features NaN: {total_nan_features} → {final_nan_features}")
        logging.info(f"    Targets NaN: {nan_targets_before.sum()} → {final_nan_targets} (esperado al final)")

        if final_nan_features > 0:
            logging.warning(f"  [NaN] ADVERTENCIA: {final_nan_features} NaN en features después de limpieza")
    else:
        logging.info(f"  [NaN] Datos limpios: 0 NaN en features")

    # =========================================================================
    # CORTE TEMPORAL: Solo usar datos hasta el último día del mes anterior
    # =========================================================================
    cutoff_date = get_training_cutoff_date()
    data_max_date = df['Date'].max()
    data_min_date = df['Date'].min()

    logging.info(f"  Rango de datos disponibles: {data_min_date.date()} a {data_max_date.date()}")
    logging.info(f"  CORTE TEMPORAL aplicado: {cutoff_date.date()}")
    logging.info(f"  (Training usará datos hasta {cutoff_date.date()}, NO del mes actual)")

    # Filtrar datos para training: solo hasta el cutoff
    df_train_eligible = df[df['Date'] <= cutoff_date].copy()
    df_test_pool = df[df['Date'] > cutoff_date].copy()

    n_excluded = len(df) - len(df_train_eligible)
    logging.info(f"  Registros elegibles para training: {len(df_train_eligible)}")
    logging.info(f"  Registros excluidos (mes actual): {n_excluded}")

    if len(df_train_eligible) < 100:
        raise ValueError(f"Muy pocos datos para training después del corte temporal: {len(df_train_eligible)}")

    # Los features ya vienen procesados en el CSV/BD
    df_features = df_train_eligible.copy()

    # Obtener precios y retornos (solo del período elegible para training)
    prices = df_features['Close'].copy()
    returns = np.log(prices / prices.shift(1)).dropna()

    # =========================================================================
    # SELECCIÓN DE 29 FEATURES (contrato con feature engineering)
    # =========================================================================
    # Estas son las 29 features definidas en el contrato:
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

    # Verificar qué features están disponibles
    available_features = [f for f in TRAINING_FEATURES if f in df_features.columns]
    missing_features = [f for f in TRAINING_FEATURES if f not in df_features.columns]

    if missing_features:
        logging.warning(f"  Features faltantes (usando las disponibles): {missing_features}")

    feature_cols = available_features
    logging.info(f"  Features para training: {len(feature_cols)} de 29 definidas")

    # =========================================================================
    # TRAIN/TEST SPLIT TEMPORAL CON GAP (dentro del período elegible)
    # =========================================================================
    # El split es 80/20 DENTRO de los datos hasta el cutoff
    # IMPORTANTE: GAP de 30 días entre train y test para evitar data leakage
    #
    # Sin GAP (MALO):
    #   Train termina en día T, usa target = log(close_{T+30} / close_T)
    #   → El target de train CONOCE precios hasta T+30
    #   → Test empieza en T+1: DATA LEAKAGE!
    #
    # Con GAP = 30 (CORRECTO):
    #   Train termina en T-30, Test empieza en T+1
    #   → No hay solapamiento temporal

    GAP = max(HORIZONS)  # 30 días
    n = len(df_features)
    train_end = int(n * TRAIN_SIZE) - GAP  # 80% - gap
    test_start = int(n * TRAIN_SIZE)        # 80%

    X_train = df_features[feature_cols].iloc[:train_end]
    X_test = df_features[feature_cols].iloc[test_start:]

    logging.info(f"  Split temporal CON GAP dentro del período elegible:")
    logging.info(f"    - Train: {len(X_train)} registros (0 a {train_end})")
    logging.info(f"    - GAP: {GAP} días ({train_end} a {test_start}) - NO SE USA")
    logging.info(f"    - Test: {len(X_test)} registros ({test_start} a {n})")

    # Guardar para siguientes tareas
    output_dir = Path('/opt/airflow/outputs/training_temp')
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_pickle(output_dir / 'X_train.pkl')
    X_test.to_pickle(output_dir / 'X_test.pkl')
    prices.to_pickle(output_dir / 'prices.pkl')
    df_features.to_pickle(output_dir / 'df_features.pkl')

    with open(output_dir / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    # Guardar metadata del corte temporal
    cutoff_metadata = {
        'cutoff_date': cutoff_date.isoformat(),
        'data_min_date': data_min_date.isoformat(),
        'data_max_date': data_max_date.isoformat(),
        'records_total': len(df),
        'records_training_eligible': len(df_train_eligible),
        'records_excluded': n_excluded
    }
    with open(output_dir / 'cutoff_metadata.json', 'w') as f:
        json.dump(cutoff_metadata, f, indent=2)

    context['ti'].xcom_push(key='train_samples', value=len(X_train))
    context['ti'].xcom_push(key='test_samples', value=len(X_test))
    context['ti'].xcom_push(key='n_features', value=len(feature_cols))
    context['ti'].xcom_push(key='cutoff_date', value=cutoff_date.isoformat())

    logging.info("=" * 60)
    logging.info(f"RESUMEN: Train={len(X_train)}, Test={len(X_test)}, Features={len(feature_cols)}")
    logging.info(f"CORTE TEMPORAL: {cutoff_date.date()}")
    logging.info("=" * 60)

    result = {
        'train': len(X_train),
        'test': len(X_test),
        'features': len(feature_cols),
        'cutoff_date': cutoff_date.isoformat()
    }

    # =========================================================================
    # CHECKPOINT: Marcar fase como completada
    # =========================================================================
    mark_phase_complete(PHASE_NAME, result)

    return result


# =============================================================================
# FASE 1: OPTUNA HYPERPARAMETER TUNING (Opcional)
# =============================================================================

def load_optuna_params_from_minio() -> dict:
    """
    Intenta cargar hiperparámetros de Optuna desde MinIO.

    Returns:
        dict o None si no existen
    """
    if not MINIO_CLIENT_AVAILABLE:
        return None

    try:
        minio_client = MinioClient()

        # Verificar si existe el archivo de params
        from minio.error import S3Error
        try:
            response = minio_client.client.get_object(OPTUNA_PARAMS_BUCKET, OPTUNA_PARAMS_KEY)
            params_json = response.read().decode('utf-8')
            response.close()
            response.release_conn()

            params = json.loads(params_json)
            logging.info(f"Cargados parámetros de Optuna desde MinIO: {OPTUNA_PARAMS_KEY}")
            logging.info(f"  Fecha de optimización: {params.get('optimization_date', 'N/A')}")
            logging.info(f"  Modelos: {len(params.get('params', {}))} configuraciones")

            return params.get('params', {})

        except S3Error as e:
            if e.code == 'NoSuchKey':
                logging.info("No se encontraron parámetros de Optuna en MinIO (primera ejecución)")
                return None
            raise

    except Exception as e:
        logging.warning(f"Error cargando params de MinIO: {e}")
        return None


def save_optuna_params_to_minio(optimized_params: dict, optimization_stats: dict = None):
    """
    Guarda hiperparámetros optimizados de Optuna en MinIO.

    Args:
        optimized_params: Diccionario modelo -> horizonte -> params
        optimization_stats: Estadísticas de la optimización
    """
    if not MINIO_CLIENT_AVAILABLE:
        logging.warning("MinIO no disponible, params solo guardados localmente")
        return False

    try:
        minio_client = MinioClient()

        # Crear estructura completa con metadata
        params_data = {
            'optimization_date': datetime.now().isoformat(),
            'optuna_trials': N_OPTUNA_TRIALS,
            'models': ML_MODELS,
            'horizons': HORIZONS,
            'params': optimized_params,
            'stats': optimization_stats or {}
        }

        # Serializar a JSON
        params_json = json.dumps(params_data, indent=2, default=str)
        params_bytes = params_json.encode('utf-8')

        # Subir a MinIO
        from io import BytesIO
        minio_client.client.put_object(
            OPTUNA_PARAMS_BUCKET,
            OPTUNA_PARAMS_KEY,
            BytesIO(params_bytes),
            length=len(params_bytes),
            content_type='application/json'
        )

        logging.info(f"Parámetros de Optuna guardados en MinIO: {OPTUNA_PARAMS_BUCKET}/{OPTUNA_PARAMS_KEY}")
        logging.info(f"  Tamaño: {len(params_bytes)} bytes")

        # También guardar versión con timestamp para historial
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_key = f"optuna/history/hyperparameters_{timestamp}.json"
        minio_client.client.put_object(
            OPTUNA_PARAMS_BUCKET,
            history_key,
            BytesIO(params_bytes),
            length=len(params_bytes),
            content_type='application/json'
        )
        logging.info(f"Historial guardado: {history_key}")

        return True

    except Exception as e:
        logging.error(f"Error guardando params en MinIO: {e}")
        return False


def run_optuna_tuning(**context):
    """
    FASE 1: Optuna Hyperparameter Tuning con WalkForwardPurged CV.

    Flujo de decisión:
    1. Si USE_CACHED_OPTUNA_PARAMS=True y existen params en MinIO -> usar cached
    2. Si ENABLE_OPTUNA=True -> ejecutar Optuna y guardar en MinIO
    3. Si ENABLE_OPTUNA=False -> usar params pre-optimizados hardcodeados
    """
    PHASE_NAME = 'optuna_tuning'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info("=" * 60)
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        logging.info("  (Para forzar re-ejecución, cambiar FORCE_FULL_RESTART=True)")
        logging.info("=" * 60)
        # Recuperar parámetros del checkpoint local
        temp_dir = Path('/opt/airflow/outputs/training_temp')
        if (temp_dir / 'optuna_params.json').exists():
            with open(temp_dir / 'optuna_params.json', 'r') as f:
                cached_params = json.load(f)
            context['ti'].xcom_push(key='optuna_params', value=cached_params)
            context['ti'].xcom_push(key='optuna_status', value='cached_from_checkpoint')
        prev_result = get_phase_result(PHASE_NAME)
        return prev_result

    logging.info("=" * 60)
    logging.info("FASE 1: OPTUNA HYPERPARAMETER TUNING")
    logging.info(f"Optuna habilitado: {ENABLE_OPTUNA}")
    logging.info(f"Usar cache de MinIO: {USE_CACHED_OPTUNA_PARAMS}")
    logging.info("=" * 60)

    temp_dir = Path('/opt/airflow/outputs/training_temp')
    temp_dir.mkdir(parents=True, exist_ok=True)

    # =================================================================
    # PASO 1: Verificar si existen params en MinIO (cache)
    # =================================================================
    if USE_CACHED_OPTUNA_PARAMS and not ENABLE_OPTUNA:
        cached_params = load_optuna_params_from_minio()
        if cached_params:
            logging.info("*** USANDO PARÁMETROS CACHEADOS DE MINIO ***")

            # Guardar localmente para el training
            with open(temp_dir / 'optuna_params.json', 'w') as f:
                json.dump(cached_params, f, indent=2)

            context['ti'].xcom_push(key='optuna_params', value=cached_params)
            context['ti'].xcom_push(key='optuna_status', value='cached_from_minio')

            return {
                'status': 'cached_from_minio',
                'n_params': sum(len(h) for h in cached_params.values()),
                'source': 'minio'
            }

    # =================================================================
    # PASO 2: Si Optuna deshabilitado, usar defaults
    # =================================================================
    if not ENABLE_OPTUNA:
        logging.info("Usando parámetros pre-optimizados (Optuna deshabilitado)")
        # Params pre-optimizados por horizonte (de run_hybrid_improved.py)
        optimized_params = {}
        for model_name in ML_MODELS:
            optimized_params[model_name] = {}
            for h in HORIZONS:
                if h <= 5:
                    optimized_params[model_name][h] = {
                        'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05,
                        'subsample': 0.7, 'colsample_bytree': 0.7,
                        'reg_alpha': 0.5, 'reg_lambda': 2.0
                    }
                elif h <= 15:
                    optimized_params[model_name][h] = {
                        'n_estimators': 30, 'max_depth': 2, 'learning_rate': 0.08,
                        'subsample': 0.6, 'colsample_bytree': 0.6,
                        'reg_alpha': 1.0, 'reg_lambda': 3.0
                    }
                else:
                    optimized_params[model_name][h] = {
                        'n_estimators': 20, 'max_depth': 1, 'learning_rate': 0.1,
                        'subsample': 0.5, 'colsample_bytree': 0.5,
                        'reg_alpha': 2.0, 'reg_lambda': 5.0
                    }

        # Guardar params localmente
        with open(temp_dir / 'optuna_params.json', 'w') as f:
            json.dump(optimized_params, f, indent=2)

        context['ti'].xcom_push(key='optuna_params', value=optimized_params)
        context['ti'].xcom_push(key='optuna_status', value='pre_optimized')

        return {'status': 'pre_optimized', 'n_params': len(optimized_params) * len(HORIZONS)}

    # Si Optuna está habilitado, ejecutar tuning
    if not OPTUNA_AVAILABLE:
        logging.warning("Optuna no disponible, usando params pre-optimizados")
        return run_optuna_tuning.__wrapped__(**context)  # Fallback

    from src.models.factory import ModelFactory
    from src.evaluation.purged_kfold import get_cv_for_horizon

    temp_dir = Path('/opt/airflow/outputs/training_temp')
    X_train = pd.read_pickle(temp_dir / 'X_train.pkl')
    df_features = pd.read_pickle(temp_dir / 'df_features.pkl')
    prices = pd.read_pickle(temp_dir / 'prices.pkl')

    # =========================================================================
    # SISTEMA DE CHECKPOINT: Permite reanudar entrenamiento desde modelos fallidos
    # =========================================================================
    checkpoint_file = temp_dir / 'optuna_checkpoint.json'
    completed_models = set()
    failed_models = set()
    optimized_params = {}

    # Cargar checkpoint existente si existe
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            completed_models = set(checkpoint.get('completed', []))
            failed_models = set(checkpoint.get('failed', []))
            optimized_params = checkpoint.get('params', {})
            logging.info("=" * 60)
            logging.info("CHECKPOINT ENCONTRADO - REANUDANDO ENTRENAMIENTO")
            logging.info(f"  Modelos completados: {list(completed_models)}")
            logging.info(f"  Modelos fallidos (se reintentarán): {list(failed_models)}")
            logging.info("=" * 60)
            # Limpiar modelos fallidos del checkpoint para reintentarlos
            for failed_model in failed_models:
                if failed_model in optimized_params:
                    del optimized_params[failed_model]
            failed_models.clear()
        except Exception as e:
            logging.warning(f"Error cargando checkpoint: {e}, iniciando desde cero")

    def save_checkpoint():
        """Guarda el estado actual del entrenamiento."""
        checkpoint = {
            'completed': list(completed_models),
            'failed': list(failed_models),
            'params': optimized_params,
            'last_update': datetime.now().isoformat()
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    tuner = OptunaTuner(config=OptunaConfig(n_trials=N_OPTUNA_TRIALS))

    for model_name in ML_MODELS:
        # Saltar modelos ya completados exitosamente
        if model_name in completed_models:
            logging.info(f"SALTANDO {model_name} - ya completado en checkpoint anterior")
            continue

        logging.info(f"Optimizando {model_name}...")
        optimized_params[model_name] = {}
        model_success = True

        try:
            model_class = ModelFactory.get_model_class(model_name)
            if model_class is None:
                logging.warning(f"No se encontró clase para {model_name}")
                failed_models.add(model_name)
                save_checkpoint()
                continue

            for h in HORIZONS:
                # Crear target
                target_col = f'target_{h}d'
                if target_col not in df_features.columns:
                    df_features[target_col] = np.log(prices.shift(-h) / prices)

                y = df_features[target_col].iloc[:len(X_train)].dropna()
                X = X_train.loc[y.index].copy()

                # CRÍTICO: Eliminar filas con NaN en features para evitar errores en modelos lineales
                valid_mask = ~X.isna().any(axis=1)
                X = X[valid_mask]
                y = y[valid_mask]

                if len(X) < 100:
                    logging.warning(f"  H={h}: Muy pocos datos válidos ({len(X)}), saltando")
                    optimized_params[model_name][h] = {}
                    continue

                logging.info(f"  H={h}: {len(X)} muestras válidas (sin NaN)")

                # CV con WalkForwardPurged
                cv = get_cv_for_horizon(h, n_splits=3)

                best_params, best_score = tuner.optimize(
                    model_class=model_class,
                    X=X.values,
                    y=y.values,
                    horizon=h,
                    cv=cv
                )

                optimized_params[model_name][h] = best_params
                logging.info(f"  H={h}: best_score={best_score:.4f}")

            # Si llegamos aquí, el modelo completó todos los horizontes
            if model_success and len(optimized_params[model_name]) == len(HORIZONS):
                completed_models.add(model_name)
                logging.info(f"*** {model_name} COMPLETADO EXITOSAMENTE ***")
            else:
                failed_models.add(model_name)
                logging.warning(f"*** {model_name} INCOMPLETO (solo {len(optimized_params[model_name])}/{len(HORIZONS)} horizontes) ***")

        except Exception as e:
            logging.error(f"Error optimizando {model_name}: {e}")
            failed_models.add(model_name)
            model_success = False

        # Guardar checkpoint después de cada modelo (éxito o fallo)
        save_checkpoint()
        logging.info(f"Checkpoint guardado: {len(completed_models)} completados, {len(failed_models)} fallidos")

    # Guardar params localmente
    with open(temp_dir / 'optuna_params.json', 'w') as f:
        json.dump(optimized_params, f, indent=2, default=str)

    # =================================================================
    # PASO 4: GUARDAR PARAMS EN MINIO PARA FUTUROS ENTRENAMIENTOS
    # =================================================================
    optimization_stats = {
        'n_models_optimized': len(completed_models),
        'n_models_failed': len(failed_models),
        'completed_models': list(completed_models),
        'failed_models': list(failed_models),
        'n_horizons': len(HORIZONS),
        'total_combinations': len(optimized_params) * len(HORIZONS),
        'n_trials_per_model': N_OPTUNA_TRIALS,
        'optimization_completed': datetime.now().isoformat()
    }

    minio_saved = save_optuna_params_to_minio(optimized_params, optimization_stats)

    if minio_saved:
        logging.info("*** PARÁMETROS GUARDADOS EN MINIO - DISPONIBLES PARA CACHE ***")
        logging.info("Para usar estos params en el futuro:")
        logging.info("  1. Cambiar ENABLE_OPTUNA = False")
        logging.info("  2. Mantener USE_CACHED_OPTUNA_PARAMS = True")
        logging.info("  3. Los params se cargarán automáticamente desde MinIO")

    context['ti'].xcom_push(key='optuna_params', value=optimized_params)
    context['ti'].xcom_push(key='optuna_status', value='optimized')
    context['ti'].xcom_push(key='optuna_saved_to_minio', value=minio_saved)
    context['ti'].xcom_push(key='completed_models', value=list(completed_models))
    context['ti'].xcom_push(key='failed_models', value=list(failed_models))

    logging.info("=" * 60)
    logging.info("FASE 1 COMPLETADA: Parámetros optimizados guardados")
    logging.info(f"  Modelos completados: {len(completed_models)}/{len(ML_MODELS)}")
    logging.info(f"  Modelos fallidos: {list(failed_models) if failed_models else 'Ninguno'}")
    logging.info(f"  Guardado en MinIO: {minio_saved}")
    if failed_models:
        logging.info("  NOTA: Relanza el pipeline para reintentar modelos fallidos")
        logging.info(f"        Checkpoint guardado en: {checkpoint_file}")
    logging.info("=" * 60)

    result = {
        'status': 'optimized' if not failed_models else 'partial',
        'n_params': len(optimized_params) * len(HORIZONS),
        'saved_to_minio': minio_saved,
        'completed_models': list(completed_models),
        'failed_models': list(failed_models)
    }

    # =========================================================================
    # CHECKPOINT: Marcar fase como completada
    # =========================================================================
    mark_phase_complete(PHASE_NAME, result)

    return result


# =============================================================================
# FASE 3: WALK-FORWARD BACKTEST (Validación post-training)
# =============================================================================

def run_walk_forward_validation(**context):
    """
    FASE 3: Walk-Forward Backtest para validación rigurosa.

    Ejecuta backtest con reentrenamiento en cada ventana:
    - Ventana expandida (expanding window)
    - Scaler fit solo en train de cada ventana
    - Calcula DA agregado, Sharpe, MaxDD, Profit Factor
    - Compara con métricas de FASE 2 (detecta leakage)
    """
    PHASE_NAME = 'walk_forward'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info("=" * 60)
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        logging.info("=" * 60)
        prev_result = get_phase_result(PHASE_NAME)
        context['ti'].xcom_push(key='wf_results', value=prev_result.get('wf_results', []))
        context['ti'].xcom_push(key='leakage_detected', value=prev_result.get('leakage_detected', False))
        return prev_result

    logging.info("=" * 60)
    logging.info("FASE 3: WALK-FORWARD BACKTEST VALIDATION")
    logging.info(f"Walk-Forward habilitado: {ENABLE_WALKFORWARD}")
    logging.info("=" * 60)

    if not ENABLE_WALKFORWARD:
        logging.info("Walk-Forward deshabilitado, saltando validación")
        return {'status': 'skipped'}

    if not WALKFORWARD_AVAILABLE:
        logging.warning("Walk-Forward module no disponible")
        return {'status': 'unavailable'}

    temp_dir = Path('/opt/airflow/outputs/training_temp')
    df_features = pd.read_pickle(temp_dir / 'df_features.pkl')
    prices = pd.read_pickle(temp_dir / 'prices.pkl')

    with open(temp_dir / 'feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Cargar métricas de FASE 2 para comparación
    output_dir = context['ti'].xcom_pull(key='output_dir', task_ids='train_ridge')
    if output_dir:
        output_dir = Path(output_dir)
        try:
            df_phase2_metrics = pd.read_csv(output_dir / 'data' / 'model_results.csv')
            phase2_lookup = {
                (r['model'], r['horizon']): r.get('direction_accuracy', 0)
                for _, r in df_phase2_metrics.iterrows()
            }
        except Exception:
            phase2_lookup = {}
    else:
        phase2_lookup = {}

    # Preparar X (features) y crear targets para cada horizonte
    X_all = df_features[feature_cols].copy()

    # Eliminar filas con NaN en features
    valid_mask = ~X_all.isna().any(axis=1)
    X_all = X_all[valid_mask]
    prices_aligned = prices.loc[X_all.index]

    wf_results = []
    comparisons = []

    # ==========================================================================
    # WALK-FORWARD PARA TODOS LOS MODELOS (Enfoque Doctoral)
    # Calcula: Sharpe Ratio, Profit Factor, Max Drawdown, Total Return
    # ==========================================================================
    from src.models.factory import ModelFactory

    for model_name in ML_MODELS:  # TODOS los 9 modelos
        for h in HORIZONS:
            logging.info(f"Walk-Forward: {model_name} H={h}...")

            try:
                # Crear target para este horizonte
                y = np.log(prices_aligned.shift(-h) / prices_aligned).dropna()
                X = X_all.loc[y.index].values
                y = y.values

                if len(X) < WF_MIN_TRAIN_SIZE + 100:
                    logging.warning(f"  Datos insuficientes para WF H={h}: {len(X)} muestras")
                    continue

                # Factory function genérica para cualquier modelo
                def create_model_factory(model_name_inner, horizon_inner):
                    """Crea factory function para walk-forward backtest."""
                    def factory():
                        return ModelFactory.create(model_name_inner, horizon=horizon_inner)
                    return factory

                model_factory = create_model_factory(model_name, h)
                requires_scaling = ModelFactory.requires_scaling(model_name)

                # Calcular n_windows basado en datos disponibles
                n_windows = min(5, (len(X) - WF_MIN_TRAIN_SIZE) // WF_TEST_SIZE)
                if n_windows < 2:
                    logging.warning(f"  Muy pocas ventanas posibles para H={h}")
                    continue

                summary = walk_forward_backtest(
                    model_factory=model_factory,
                    X=X,
                    y=y,
                    horizon=h,
                    n_windows=n_windows,
                    min_train_pct=WF_MIN_TRAIN_SIZE / len(X),
                    gap=h,  # Gap igual al horizonte
                    model_name=model_name,
                    requires_scaling=requires_scaling
                )

                wf_results.append({
                    'model': model_name,
                    'horizon': h,
                    'wf_da_mean': summary.da_mean,
                    'wf_da_std': summary.da_std,
                    'wf_sharpe': summary.sharpe_mean,
                    'wf_max_drawdown': summary.max_drawdown,
                    'wf_profit_factor': summary.profit_factor,
                    'wf_n_windows': summary.n_windows,
                    'wf_total_return': summary.total_return
                })

                # Comparar con FASE 2
                phase2_da = phase2_lookup.get((model_name, h), 0.5)
                diff = summary.da_mean - phase2_da
                potential_leakage = diff < -0.05  # WF DA > 5% menor = posible leakage

                comparisons.append({
                    'model': model_name,
                    'horizon': h,
                    'phase2_da': phase2_da,
                    'wf_da': summary.da_mean,
                    'difference': diff,
                    'potential_leakage': potential_leakage
                })

                logging.info(f"  WF_DA={summary.da_mean:.2%} vs Phase2_DA={phase2_da:.2%} (diff={diff:+.2%})")

            except Exception as e:
                logging.error(f"Error en WF {model_name} H={h}: {e}")

    # Guardar resultados
    if output_dir:
        wf_dir = output_dir / 'walk_forward'
        wf_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(wf_results).to_csv(wf_dir / 'walk_forward_results.csv', index=False)
        pd.DataFrame(comparisons).to_csv(wf_dir / 'phase2_vs_wf_comparison.csv', index=False)

    # Detectar leakage
    leakage_detected = any(c['potential_leakage'] for c in comparisons)
    if leakage_detected:
        logging.warning("*** POSIBLE DATA LEAKAGE DETECTADO ***")
        logging.warning("Walk-Forward DA es significativamente menor que Phase 2 DA")

    context['ti'].xcom_push(key='wf_results', value=wf_results)
    context['ti'].xcom_push(key='leakage_detected', value=leakage_detected)

    logging.info("=" * 60)
    logging.info(f"FASE 3 COMPLETADA: {len(wf_results)} validaciones")
    logging.info(f"Leakage detectado: {leakage_detected}")
    logging.info("=" * 60)

    result = {
        'status': 'completed',
        'n_validations': len(wf_results),
        'leakage_detected': leakage_detected,
        'wf_results': wf_results
    }

    # =========================================================================
    # CHECKPOINT: Marcar fase como completada
    # =========================================================================
    mark_phase_complete(PHASE_NAME, result)

    return result


# =============================================================================
# GENERACIÓN DE IMÁGENES DE BACKTEST
# =============================================================================

def generate_backtest_images(**context):
    """
    Genera imágenes de backtest para cada modelo/horizonte.

    Esperado: 9 modelos × 7 horizontes = 63 imágenes
    """
    PHASE_NAME = 'backtest_images'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info("=" * 60)
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        logging.info("=" * 60)
        prev_result = get_phase_result(PHASE_NAME)
        context['ti'].xcom_push(key='backtest_images', value=prev_result.get('backtest_images', 0))
        context['ti'].xcom_push(key='comparison_images', value=prev_result.get('comparison_images', 0))
        return prev_result

    logging.info("=" * 60)
    logging.info("GENERANDO IMAGENES DE BACKTEST")
    logging.info("=" * 60)

    if not GENERATE_BACKTEST_IMAGES:
        logging.info("Generación de imágenes deshabilitada")
        return {'status': 'skipped', 'images': 0}

    if not VISUALIZATION_AVAILABLE:
        logging.warning("Módulos de visualización no disponibles")
        return {'status': 'unavailable', 'images': 0}

    output_dir = context['ti'].xcom_pull(key='output_dir', task_ids='train_ridge')
    if not output_dir:
        logging.error("No se encontró output_dir")
        return {'status': 'error', 'images': 0}

    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Inicializar plotters
    backtest_plotter = BacktestPlotter(figsize=(14, 10), dpi=150)
    model_plotter = ModelComparisonPlotter()

    # Cargar datos y resultados
    temp_dir = Path('/opt/airflow/outputs/training_temp')
    df_features = pd.read_pickle(temp_dir / 'df_features.pkl')
    prices = pd.read_pickle(temp_dir / 'prices.pkl')

    # Cargar predicciones de cada modelo
    all_results = []
    for model_name in ML_MODELS:
        results = context['ti'].xcom_pull(key=f'results_{model_name}', task_ids=f'train_{model_name}')
        if results:
            all_results.extend(results)

    # Contar imágenes generadas
    backtest_count = 0
    comparison_count = 0

    # Generar imágenes de backtest individuales
    # Nota: Los modelos guardados no incluyen predicciones completas,
    # así que generamos plots basados en las métricas disponibles
    try:
        # Crear DataFrame de resultados
        df_results = pd.DataFrame(all_results)
        df_results = df_results[df_results['direction_accuracy'].notna()].copy()

        if len(df_results) > 0:
            # Convertir DA a 0-1 si está en porcentaje
            if df_results['direction_accuracy'].max() > 1:
                df_results['direction_accuracy'] = df_results['direction_accuracy'] / 100

            # 1. Generar heatmap de DA
            model_plotter.plot_metrics_heatmap(
                df_results,
                metric='direction_accuracy',
                save_path=figures_dir / 'metrics_heatmap_da.png'
            )
            comparison_count += 1

            # 2. Generar ranking de modelos
            model_plotter.plot_model_ranking(
                df_results,
                metric='direction_accuracy',
                save_path=figures_dir / 'model_ranking_da.png'
            )
            comparison_count += 1

            # 3. Generar heatmap de RMSE
            if 'rmse' in df_results.columns:
                model_plotter.plot_metrics_heatmap(
                    df_results,
                    metric='rmse',
                    save_path=figures_dir / 'metrics_heatmap_rmse.png'
                )
                comparison_count += 1

            logging.info(f"Generadas {comparison_count} imágenes de comparación")

        # NOTA: Las imágenes de backtest individuales (2x2 con 4 paneles:
        # Predicho vs Real, Scatter, Error Distribution, DA Rolling) se generan
        # durante el entrenamiento en train_model(). No generar placeholders aquí.
        logging.info("Imágenes de backtest 2x2 generadas durante entrenamiento")

    except Exception as e:
        logging.error(f"Error generando imágenes: {e}")

    total_images = backtest_count + comparison_count

    logging.info("=" * 60)
    logging.info(f"IMÁGENES GENERADAS: {total_images}")
    logging.info(f"  - Backtest: {backtest_count}")
    logging.info(f"  - Comparación: {comparison_count}")
    logging.info("=" * 60)

    # =========================================================================
    # SUBIR IMÁGENES A MINIO ANTES DE CLEANUP
    # =========================================================================
    minio_uploaded = 0
    minio_paths = []

    if MINIO_CLIENT_AVAILABLE and total_images > 0:
        logging.info("Subiendo imágenes de backtest a MinIO...")
        try:
            minio_client = MinioClient()

            # Obtener año y mes del directorio de salida (formato: YYYYMMDD_HHMMSS)
            run_name = output_dir.name  # e.g., "20260105_120937"
            now = datetime.now()
            year = now.year
            month = now.month
            timestamp = run_name if run_name else now.strftime('%Y%m%d_%H%M%S')

            # Ruta base en MinIO: ml-models/{year}/month{month:02d}/{timestamp}/figures/
            base_minio_path = f"{year}/month{month:02d}/{timestamp}/figures"

            # Asegurar que el bucket existe
            minio_client.ensure_bucket(MODELS_BUCKET)

            # Subir cada imagen del directorio figures
            for img_file in figures_dir.glob('*.png'):
                try:
                    s3_path = f"{base_minio_path}/{img_file.name}"
                    minio_client.upload_model(
                        bucket=MODELS_BUCKET,
                        model_path=str(img_file),
                        s3_path=s3_path,
                        content_type='image/png'
                    )
                    minio_uploaded += 1
                    minio_paths.append(f"s3://{MODELS_BUCKET}/{s3_path}")
                    logging.info(f"  Subida: {img_file.name}")
                except Exception as e:
                    logging.warning(f"  Error subiendo {img_file.name}: {e}")

            logging.info(f"Imágenes subidas a MinIO: {minio_uploaded}/{total_images}")
            logging.info(f"Ruta MinIO: s3://{MODELS_BUCKET}/{base_minio_path}/")

        except Exception as e:
            logging.error(f"Error conectando a MinIO: {e}")
    else:
        if not MINIO_CLIENT_AVAILABLE:
            logging.warning("MinIO no disponible - imágenes solo guardadas localmente")
        if total_images == 0:
            logging.info("No hay imágenes para subir a MinIO")

    context['ti'].xcom_push(key='backtest_images', value=backtest_count)
    context['ti'].xcom_push(key='comparison_images', value=comparison_count)
    context['ti'].xcom_push(key='minio_uploaded_images', value=minio_uploaded)

    result = {
        'status': 'completed',
        'backtest_images': backtest_count,
        'comparison_images': comparison_count,
        'total': total_images,
        'minio_uploaded': minio_uploaded,
        'minio_paths': minio_paths[:5] if minio_paths else []  # Solo primeras 5 para no saturar
    }

    # =========================================================================
    # CHECKPOINT: Marcar fase como completada
    # =========================================================================
    mark_phase_complete(PHASE_NAME, result)

    return result


def train_model(model_name: str, **context):
    """
    Entrena un modelo específico para todos los horizontes.

    Proceso por horizonte:
    1. Cargar datos preprocesados
    2. Crear target para el horizonte
    3. Aplicar scaler SI el modelo lo requiere (fit SOLO en train)
    4. Entrenar modelo (con early stopping si es boosting)
    5. Evaluar métricas (DA, RMSE, MAE, R²)
    6. Guardar modelo + scaler + metadata
    7. Registrar en MLflow

    IMPORTANTE:
    - El scaler se ajusta (fit) SOLO en train, NUNCA en test
    - Un scaler por horizonte (porque train set puede variar ligeramente)
    - Los modelos híbridos manejan su propio scaler interno
    """
    PHASE_NAME = f'training_{model_name}'

    # =========================================================================
    # CHECKPOINT: Verificar si este modelo ya fue entrenado este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info("=" * 60)
        logging.info(f"MODELO '{model_name}' YA ENTRENADO - SALTANDO")
        logging.info("=" * 60)
        prev_result = get_phase_result(PHASE_NAME)
        # Recuperar output_dir del checkpoint
        checkpoint_output_dir = get_checkpoint_output_dir()
        if checkpoint_output_dir:
            context['ti'].xcom_push(key='output_dir', value=str(checkpoint_output_dir))
        context['ti'].xcom_push(key=f'results_{model_name}', value=prev_result.get('results', []))
        return prev_result.get('results', [])

    logging.info("=" * 60)
    logging.info(f"ENTRENANDO MODELO: {model_name.upper()}")
    logging.info("=" * 60)

    # Inicializar MLflow
    mlflow = get_mlflow_client()

    from src.models.factory import ModelFactory
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # Cargar datos
    temp_dir = Path('/opt/airflow/outputs/training_temp')
    X_train = pd.read_pickle(temp_dir / 'X_train.pkl')
    X_test = pd.read_pickle(temp_dir / 'X_test.pkl')
    df_features = pd.read_pickle(temp_dir / 'df_features.pkl')
    prices = pd.read_pickle(temp_dir / 'prices.pkl')

    with open(temp_dir / 'feature_cols.json', 'r') as f:
        feature_cols = json.load(f)

    # Usar output_dir del checkpoint si existe, o crear uno nuevo
    checkpoint_output_dir = get_checkpoint_output_dir()
    if checkpoint_output_dir and checkpoint_output_dir.exists():
        output_dir = checkpoint_output_dir
        logging.info(f"  Usando output_dir existente del checkpoint: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f'/opt/airflow/outputs/runs/{timestamp}')
        set_checkpoint_output_dir(output_dir)
        logging.info(f"  Creando nuevo output_dir: {output_dir}")

    models_dir = output_dir / 'models'
    scalers_dir = output_dir / 'scalers'
    models_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Determinar si el modelo requiere scaling
    requires_scaling = ModelFactory.requires_scaling(model_name)
    model_type = ModelFactory.get_model_type(model_name)
    supports_early_stopping = ModelFactory.supports_early_stopping(model_name)

    logging.info(f"  Tipo: {model_type}")
    logging.info(f"  Requiere scaling: {requires_scaling}")
    logging.info(f"  Early stopping: {supports_early_stopping}")

    # Iniciar run padre de MLflow para este modelo
    with mlflow.start_run(run_name=f"{model_name}_{timestamp}", nested=False) as parent_run:
        mlflow.log_param('model_type', model_name)
        mlflow.log_param('model_category', model_type)
        mlflow.log_param('requires_scaling', requires_scaling)
        mlflow.log_param('n_features', len(feature_cols))
        mlflow.log_param('train_samples', len(X_train))
        mlflow.log_param('test_samples', len(X_test))
        mlflow.log_param('training_date', timestamp)
        mlflow.log_param('horizons', str(HORIZONS))

        for horizon in HORIZONS:
            logging.info(f"  Horizonte H={horizon}...")

            # Run hijo para cada horizonte
            with mlflow.start_run(run_name=f"{model_name}_h{horizon}", nested=True) as child_run:
                mlflow.log_param('horizon', horizon)
                mlflow.log_param('model', model_name)

                # =============================================================
                # CREAR TARGET PARA ESTE HORIZONTE
                # =============================================================
                target_col = f'target_{horizon}d'
                if target_col not in df_features.columns:
                    df_features[target_col] = np.log(prices.shift(-horizon) / prices)

                # Usar el split con GAP ya aplicado
                GAP = max(HORIZONS)
                n = len(df_features)
                train_end = int(n * TRAIN_SIZE) - GAP
                test_start = int(n * TRAIN_SIZE)

                y_train = df_features[target_col].iloc[:train_end].dropna()
                y_test = df_features[target_col].iloc[test_start:].dropna()

                # Alinear indices
                common_train = X_train.index.intersection(y_train.index)
                common_test = X_test.index.intersection(y_test.index)

                X_tr = X_train.loc[common_train].copy()
                y_tr = y_train.loc[common_train].copy()
                X_te = X_test.loc[common_test].copy()
                y_te = y_test.loc[common_test].copy()

                # =============================================================
                # MANEJO DE NaN: Crítico para modelos que no soportan NaN
                # XGBoost y LightGBM manejan NaN nativamente
                # Ridge, Bayesian, ARD, y modelos híbridos NO soportan NaN
                # =============================================================
                nan_models = {'ridge', 'bayesian_ridge', 'ard',
                              'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'}

                if model_name.lower() in nan_models:
                    # Eliminar filas con NaN en train
                    train_valid = ~X_tr.isna().any(axis=1) & ~y_tr.isna()
                    n_train_before = len(X_tr)
                    X_tr = X_tr[train_valid]
                    y_tr = y_tr[train_valid]
                    n_train_removed = n_train_before - len(X_tr)

                    # Eliminar filas con NaN en test
                    test_valid = ~X_te.isna().any(axis=1) & ~y_te.isna()
                    n_test_before = len(X_te)
                    X_te = X_te[test_valid]
                    y_te = y_te[test_valid]
                    n_test_removed = n_test_before - len(X_te)

                    if n_train_removed > 0 or n_test_removed > 0:
                        logging.info(f"    NaN removidos: train={n_train_removed}, test={n_test_removed}")

                    if len(X_tr) < 50 or len(X_te) < 10:
                        logging.warning(f"    Muy pocos datos después de limpiar NaN: train={len(X_tr)}, test={len(X_te)}")
                        results.append({
                            'model': model_name,
                            'horizon': horizon,
                            'error': f'Insufficient data after NaN removal: train={len(X_tr)}, test={len(X_te)}'
                        })
                        continue

                # =============================================================
                # SCALING: fit SOLO en train, transform en test
                # =============================================================
                scaler = None
                if requires_scaling:
                    scaler = StandardScaler()
                    X_tr_scaled = scaler.fit_transform(X_tr)  # FIT + TRANSFORM
                    X_te_scaled = scaler.transform(X_te)       # SOLO TRANSFORM
                    logging.info(f"    Scaler ajustado en {len(X_tr)} muestras de train")

                    # Guardar scaler para este horizonte
                    scaler_file = scalers_dir / f'scaler_h{horizon}.pkl'
                    joblib.dump(scaler, scaler_file)
                else:
                    X_tr_scaled = X_tr.values if hasattr(X_tr, 'values') else X_tr
                    X_te_scaled = X_te.values if hasattr(X_te, 'values') else X_te

                # =============================================================
                # CREAR Y ENTRENAR MODELO
                # =============================================================
                try:
                    model = ModelFactory.create(model_name, horizon=horizon)

                    # Entrenar con early stopping si es boosting
                    if supports_early_stopping and len(X_te_scaled) > 0:
                        # Usar parte del test como validación para early stopping
                        model.fit(
                            X_tr_scaled, y_tr.values,
                            X_val=X_te_scaled, y_val=y_te.values,
                            early_stopping_rounds=15
                        )
                    else:
                        model.fit(X_tr_scaled, y_tr.values)

                    # =============================================================
                    # PREDECIR Y EVALUAR
                    # =============================================================
                    y_pred = model.predict(X_te_scaled)

                    # Métricas
                    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                    mae = mean_absolute_error(y_te, y_pred)
                    r2 = r2_score(y_te, y_pred)

                    # Direction Accuracy (métrica principal)
                    actual_dir = np.sign(y_te.values)
                    pred_dir = np.sign(y_pred)
                    da = np.mean(actual_dir == pred_dir)

                    # Variance Ratio (detector de colapso)
                    pred_std = np.std(y_pred)
                    true_std = np.std(y_te.values)
                    variance_ratio = pred_std / (true_std + 1e-8)

                    # Log métricas a MLflow
                    mlflow.log_metric('direction_accuracy', da)
                    mlflow.log_metric('rmse', rmse)
                    mlflow.log_metric('mae', mae)
                    mlflow.log_metric('r2', r2)
                    mlflow.log_metric('variance_ratio', variance_ratio)
                    mlflow.log_metric('test_samples', len(y_te))

                    results.append({
                        'model': model_name,
                        'horizon': horizon,
                        'direction_accuracy': da,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'variance_ratio': variance_ratio,
                        'mlflow_run_id': child_run.info.run_id
                    })

                    # Log status
                    status = "OK" if variance_ratio > 0.02 else "WARNING: Low variance"
                    logging.info(f"    DA={da:.2%}, RMSE={rmse:.6f}, VarRatio={variance_ratio:.3f} [{status}]")

                    # =============================================================
                    # GUARDAR MODELO + SCALER + METADATA
                    # =============================================================
                    model_data = {
                        'model': model,
                        'scaler': scaler,
                        'feature_cols': feature_cols,
                        'horizon': horizon,
                        'model_name': model_name,
                        'model_type': model_type,
                        'requires_scaling': requires_scaling,
                        'metrics': {
                            'da': da,
                            'rmse': rmse,
                            'mae': mae,
                            'r2': r2,
                            'variance_ratio': variance_ratio
                        },
                        'training_date': timestamp,
                        'mlflow_run_id': child_run.info.run_id
                    }

                    model_file = models_dir / f'{model_name}_h{horizon}.pkl'
                    with open(model_file, 'wb') as f:
                        pickle.dump(model_data, f)

                    # Log modelo a MLflow
                    mlflow.log_artifact(str(model_file))

                except Exception as e:
                    logging.error(f"    Error entrenando {model_name} H={horizon}: {e}")
                    mlflow.log_param('error', str(e)[:200])
                    results.append({
                        'model': model_name,
                        'horizon': horizon,
                        'error': str(e)
                    })

        # =============================================================
        # MÉTRICAS AGREGADAS DEL MODELO
        # =============================================================
        valid_results = [r for r in results if 'direction_accuracy' in r]
        if valid_results:
            avg_da = np.mean([r['direction_accuracy'] for r in valid_results])
            avg_rmse = np.mean([r['rmse'] for r in valid_results])
            avg_var_ratio = np.mean([r['variance_ratio'] for r in valid_results])
            mlflow.log_metric('avg_direction_accuracy', avg_da)
            mlflow.log_metric('avg_rmse', avg_rmse)
            mlflow.log_metric('avg_variance_ratio', avg_var_ratio)

            logging.info(f"  RESUMEN {model_name}: Avg DA={avg_da:.2%}, Avg VarRatio={avg_var_ratio:.3f}")

    # Guardar resultados
    context['ti'].xcom_push(key=f'results_{model_name}', value=results)
    context['ti'].xcom_push(key='output_dir', value=str(output_dir))

    # =========================================================================
    # CHECKPOINT: Marcar modelo como entrenado
    # =========================================================================
    phase_result = {
        'model_name': model_name,
        'results': results,
        'output_dir': str(output_dir),
        'n_horizons': len(results)
    }
    mark_phase_complete(PHASE_NAME, phase_result)

    logging.info("=" * 60)
    return results


# =============================================================================
# FUNCIONES DE ENTRENAMIENTO PARA CADA MODELO (9 modelos)
# =============================================================================

# --- LINEALES (3) ---
def train_ridge(**context):
    """Ridge Regression - L2 regularizado."""
    return train_model('ridge', **context)

def train_bayesian_ridge(**context):
    """Bayesian Ridge - Regularización automática."""
    return train_model('bayesian_ridge', **context)

def train_ard(**context):
    """ARD - Automatic Relevance Determination."""
    return train_model('ard', **context)

# --- BOOSTING PUROS (3) ---
def train_xgboost_pure(**context):
    """XGBoost Pure - Gradient boosting."""
    return train_model('xgboost_pure', **context)

def train_lightgbm_pure(**context):
    """LightGBM Pure - Fast gradient boosting."""
    return train_model('lightgbm_pure', **context)

def train_catboost_pure(**context):
    """CatBoost Pure - Ordered gradient boosting."""
    return train_model('catboost_pure', **context)

# --- HÍBRIDOS (3) ---
def train_hybrid_xgboost(**context):
    """Hybrid XGBoost - Ridge (magnitud) + XGBoost clasificador (dirección)."""
    return train_model('hybrid_xgboost', **context)

def train_hybrid_lightgbm(**context):
    """Hybrid LightGBM - Ridge (magnitud) + LightGBM clasificador (dirección)."""
    return train_model('hybrid_lightgbm', **context)

def train_hybrid_catboost(**context):
    """Hybrid CatBoost - Ridge (magnitud) + CatBoost clasificador (dirección)."""
    return train_model('hybrid_catboost', **context)


def aggregate_results(**context):
    """
    Agrega resultados de todos los modelos con MÉTRICAS COMPLETAS.

    Genera archivos:
    - model_results.csv: Métricas por modelo/horizonte
    - model_summary.csv: Promedios por modelo
    - horizon_summary.csv: Promedios por horizonte
    - backtest_metrics.csv: Métricas para dashboard (con flags de mejores)
    - report_summary.json: Resumen JSON completo
    - report_summary.txt: Resumen legible

    Métricas incluidas:
    - direction_accuracy (DA)
    - rmse, mae, r2
    - model_avg_direction_accuracy
    - model_avg_rmse
    - is_best_overall_model
    - is_best_for_this_horizon
    - best_da_for_this_horizon
    """
    PHASE_NAME = 'aggregation'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info("=" * 60)
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        logging.info("=" * 60)
        prev_result = get_phase_result(PHASE_NAME)
        context['ti'].xcom_push(key='best_model', value=prev_result.get('best_model', ''))
        context['ti'].xcom_push(key='best_da', value=prev_result.get('best_da', 0.0))
        return prev_result

    logging.info("=" * 60)
    logging.info("AGREGANDO RESULTADOS CON MÉTRICAS COMPLETAS")
    logging.info("=" * 60)

    mlflow = get_mlflow_client()

    all_results = []
    for model_name in ML_MODELS:
        results = context['ti'].xcom_pull(key=f'results_{model_name}', task_ids=f'train_{model_name}')
        if results:
            all_results.extend(results)

    output_dir = context['ti'].xcom_pull(key='output_dir', task_ids='train_ridge')
    output_dir = Path(output_dir)

    # Crear directorios
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Crear DataFrame de resultados base
    df_results = pd.DataFrame(all_results)

    # Asegurar que tenemos las columnas necesarias
    if 'direction_accuracy' not in df_results.columns:
        logging.error("No hay direction_accuracy en resultados")
        return {'error': 'No direction_accuracy'}

    # Convertir DA a decimal si está en porcentaje
    if df_results['direction_accuracy'].max() > 1:
        df_results['direction_accuracy'] = df_results['direction_accuracy'] / 100

    # =================================================================
    # CALCULAR MÉTRICAS AGREGADAS
    # =================================================================

    # Promedios por modelo
    model_avgs = df_results.groupby('model').agg({
        'direction_accuracy': 'mean',
        'rmse': 'mean',
        'mae': 'mean' if 'mae' in df_results.columns else 'first',
        'r2': 'mean' if 'r2' in df_results.columns else 'first'
    }).add_prefix('model_avg_')

    # Mejor modelo global
    best_overall_model = model_avgs['model_avg_direction_accuracy'].idxmax()
    best_overall_da = model_avgs.loc[best_overall_model, 'model_avg_direction_accuracy']

    # Mejor DA por horizonte
    best_per_horizon = df_results.groupby('horizon')['direction_accuracy'].agg(['max', 'idxmax'])
    best_per_horizon.columns = ['best_da_for_horizon', 'best_idx']

    # Merge promedios de modelo
    df_results = df_results.merge(
        model_avgs[['model_avg_direction_accuracy', 'model_avg_rmse']],
        left_on='model',
        right_index=True
    )

    # Agregar flags de mejor modelo
    df_results['is_best_overall_model'] = df_results['model'] == best_overall_model

    # Agregar mejor DA por horizonte
    df_results = df_results.merge(
        best_per_horizon[['best_da_for_horizon']],
        left_on='horizon',
        right_index=True
    )

    # Flag de mejor para este horizonte
    df_results['is_best_for_this_horizon'] = (
        df_results['direction_accuracy'] == df_results['best_da_for_horizon']
    )

    # =================================================================
    # GUARDAR CSV: model_results.csv (completo)
    # =================================================================
    df_results.to_csv(data_dir / 'model_results.csv', index=False)
    logging.info(f"Guardado: model_results.csv ({len(df_results)} registros)")

    # =================================================================
    # GUARDAR CSV: model_summary.csv (promedios por modelo)
    # =================================================================
    model_summary = df_results.groupby('model').agg({
        'direction_accuracy': ['mean', 'std', 'min', 'max'],
        'rmse': ['mean', 'std'],
        'r2': 'mean' if 'r2' in df_results.columns else 'first',
        'variance_ratio': 'mean' if 'variance_ratio' in df_results.columns else 'first'
    }).round(4)
    model_summary.columns = ['_'.join(col).strip() for col in model_summary.columns]
    model_summary['is_best_overall'] = model_summary.index == best_overall_model
    model_summary.to_csv(data_dir / 'model_summary.csv')
    logging.info(f"Guardado: model_summary.csv ({len(model_summary)} modelos)")

    # =================================================================
    # GUARDAR CSV: horizon_summary.csv (promedios por horizonte)
    # =================================================================
    horizon_summary = df_results.groupby('horizon').agg({
        'direction_accuracy': ['mean', 'std', 'max'],
        'rmse': 'mean',
        'r2': 'mean' if 'r2' in df_results.columns else 'first'
    }).round(4)
    horizon_summary.columns = ['_'.join(col).strip() for col in horizon_summary.columns]
    horizon_summary.to_csv(data_dir / 'horizon_summary.csv')
    logging.info(f"Guardado: horizon_summary.csv ({len(horizon_summary)} horizontes)")

    # =================================================================
    # GUARDAR CSV: backtest_metrics.csv (formato dashboard)
    # =================================================================
    dashboard_cols = [
        'model', 'horizon',
        'direction_accuracy', 'rmse', 'mae', 'r2',
        'model_avg_direction_accuracy', 'model_avg_rmse',
        'is_best_overall_model', 'is_best_for_this_horizon',
        'best_da_for_horizon', 'variance_ratio'
    ]
    available_cols = [c for c in dashboard_cols if c in df_results.columns]
    df_dashboard = df_results[available_cols].copy()
    df_dashboard.to_csv(data_dir / 'backtest_metrics.csv', index=False)
    logging.info(f"Guardado: backtest_metrics.csv ({len(df_dashboard)} registros)")

    # =================================================================
    # GUARDAR JSON: report_summary.json
    # =================================================================
    report = {
        'training_date': datetime.now().isoformat(),
        'n_models': len(ML_MODELS),
        'n_horizons': len(HORIZONS),
        'total_model_horizon_combinations': len(df_results),
        'best_model': {
            'name': best_overall_model,
            'avg_direction_accuracy': float(best_overall_da),
            'is_significant': float(best_overall_da) > 0.55
        },
        'metrics_summary': {
            'avg_da_all_models': float(df_results['direction_accuracy'].mean()),
            'max_da': float(df_results['direction_accuracy'].max()),
            'min_da': float(df_results['direction_accuracy'].min()),
            'avg_rmse': float(df_results['rmse'].mean()) if 'rmse' in df_results.columns else None
        },
        'models': ML_MODELS,
        'horizons': HORIZONS,
        'optuna_enabled': ENABLE_OPTUNA,
        'walkforward_enabled': ENABLE_WALKFORWARD
    }

    with open(output_dir / 'report_summary.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logging.info("Guardado: report_summary.json")

    # =================================================================
    # GUARDAR TXT: report_summary.txt (legible)
    # =================================================================
    txt_lines = [
        "=" * 80,
        "REPORTE DE ENTRENAMIENTO - USD/COP FORECASTING",
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "CONFIGURACIÓN",
        "-" * 40,
        f"  Modelos: {len(ML_MODELS)}",
        f"  Horizontes: {HORIZONS}",
        f"  Optuna habilitado: {ENABLE_OPTUNA}",
        f"  Walk-Forward habilitado: {ENABLE_WALKFORWARD}",
        "",
        "MEJOR MODELO",
        "-" * 40,
        f"  Nombre: {best_overall_model}",
        f"  DA Promedio: {best_overall_da:.2%}",
        "",
        "RANKING DE MODELOS (por DA)",
        "-" * 40,
    ]

    for rank, (model, row) in enumerate(model_summary.sort_values('direction_accuracy_mean', ascending=False).iterrows(), 1):
        txt_lines.append(f"  {rank}. {model}: {row['direction_accuracy_mean']:.2%}")

    txt_lines.extend([
        "",
        "MÉTRICAS POR HORIZONTE",
        "-" * 40,
    ])

    for h in HORIZONS:
        h_data = df_results[df_results['horizon'] == h]
        avg_da = h_data['direction_accuracy'].mean()
        best_h = h_data.loc[h_data['direction_accuracy'].idxmax(), 'model']
        txt_lines.append(f"  H={h:2d}: DA={avg_da:.2%} (mejor: {best_h})")

    txt_lines.extend([
        "",
        "=" * 80,
        "ARCHIVOS GENERADOS",
        "-" * 40,
        "  - model_results.csv: Métricas detalladas",
        "  - model_summary.csv: Resumen por modelo",
        "  - horizon_summary.csv: Resumen por horizonte",
        "  - backtest_metrics.csv: Métricas para dashboard",
        "  - report_summary.json: Resumen estructurado",
        "=" * 80,
    ])

    with open(output_dir / 'report_summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(txt_lines))
    logging.info("Guardado: report_summary.txt")

    # Push a XCom
    context['ti'].xcom_push(key='best_model', value=best_overall_model)
    context['ti'].xcom_push(key='best_da', value=float(best_overall_da))

    # Registrar en MLflow
    with mlflow.start_run(run_name=f"training_summary_{datetime.now().strftime('%Y%m%d')}"):
        mlflow.log_param('n_models', len(ML_MODELS))
        mlflow.log_param('n_horizons', len(HORIZONS))
        mlflow.log_param('best_model', best_overall_model)
        mlflow.log_param('optuna_enabled', ENABLE_OPTUNA)
        mlflow.log_metric('best_direction_accuracy', float(best_overall_da))
        mlflow.log_metric('avg_direction_accuracy', float(df_results['direction_accuracy'].mean()))

        # Log artefactos
        for f in data_dir.glob('*.csv'):
            mlflow.log_artifact(str(f))
        mlflow.log_artifact(str(output_dir / 'report_summary.json'))
        mlflow.log_artifact(str(output_dir / 'report_summary.txt'))

    logging.info("=" * 60)
    logging.info(f"MÉTRICAS COMPLETAS GENERADAS: {len(df_results)} registros")
    logging.info(f"Mejor modelo: {best_overall_model} (DA={best_overall_da:.2%})")
    logging.info("=" * 60)

    result = {
        'total_results': len(df_results),
        'best_model': best_overall_model,
        'best_da': float(best_overall_da)
    }

    # =========================================================================
    # CHECKPOINT: Marcar fase como completada
    # =========================================================================
    mark_phase_complete(PHASE_NAME, result)

    return result


def register_mlflow(**context):
    """Registra modelos finales y metricas en MLflow Model Registry."""
    PHASE_NAME = 'mlflow_register'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        return get_phase_result(PHASE_NAME)

    logging.info("Registrando en MLflow Model Registry...")

    try:
        mlflow = get_mlflow_client()
        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        output_dir = context['ti'].xcom_pull(key='output_dir', task_ids='train_ridge')
        best_model = context['ti'].xcom_pull(key='best_model', task_ids='aggregate_results')
        best_da = context['ti'].xcom_pull(key='best_da', task_ids='aggregate_results')

        if not output_dir or not best_model:
            logging.warning("No hay output_dir o best_model disponible")
            return {'mlflow_registered': False}

        output_dir = Path(output_dir)
        models_dir = output_dir / 'models'

        # Registrar el mejor modelo en Model Registry
        model_name_registry = f"usdcop-{best_model}"

        with mlflow.start_run(run_name=f"register_{best_model}_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.log_param('registered_model', best_model)
            mlflow.log_param('registry_name', model_name_registry)

            if best_da:
                mlflow.log_metric('direction_accuracy', best_da)

            # Subir todos los modelos del mejor tipo
            for model_file in models_dir.glob(f'{best_model}_h*.pkl'):
                horizon = model_file.stem.split('_h')[1]

                # Log modelo como artefacto
                mlflow.log_artifact(str(model_file), artifact_path=f"models/{best_model}")

                logging.info(f"Registrado: {model_file.name} -> {model_name_registry}")

            # Registrar modelo en Model Registry
            try:
                # Crear modelo registrado si no existe
                try:
                    client.create_registered_model(
                        name=model_name_registry,
                        description=f"USD/COP Forecasting Model - {best_model.upper()}"
                    )
                except Exception:
                    pass  # El modelo ya existe

                # Obtener el run actual para registrar la version
                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/models/{best_model}"

                # Crear nueva version
                version = client.create_model_version(
                    name=model_name_registry,
                    source=model_uri,
                    run_id=run_id,
                    description=f"Training {datetime.now().strftime('%Y-%m-%d')} - DA: {best_da:.2%}"
                )

                logging.info(f"Model Registry: {model_name_registry} version {version.version}")

                # Transicionar a Staging
                client.transition_model_version_stage(
                    name=model_name_registry,
                    version=version.version,
                    stage="Staging"
                )

                logging.info(f"Modelo transicionado a Staging")

            except Exception as e:
                logging.warning(f"No se pudo registrar en Model Registry: {e}")

        logging.info("Registrado en MLflow exitosamente")
        result = {'mlflow_registered': True, 'model_name': model_name_registry}
        mark_phase_complete(PHASE_NAME, result)
        return result

    except Exception as e:
        logging.warning(f"No se pudo registrar en MLflow: {e}")
        return {'mlflow_registered': False, 'error': str(e)}


def upload_models_to_minio(**context):
    """
    Sube los modelos entrenados a MinIO con estructura organizada.

    Estructura en MinIO:
        ml-models/{year}/month{month:02d}/{timestamp}/
            ├── {model}_h{horizon}.pkl
            └── metadata.json
    """
    PHASE_NAME = 'minio_upload'

    # =========================================================================
    # CHECKPOINT: Verificar si esta fase ya completó para este mes
    # =========================================================================
    if is_phase_complete(PHASE_NAME):
        logging.info(f"FASE '{PHASE_NAME}' YA COMPLETADA - SALTANDO")
        return get_phase_result(PHASE_NAME)

    logging.info("=" * 60)
    logging.info("SUBIENDO MODELOS A MINIO")
    logging.info("=" * 60)

    if not MINIO_CLIENT_AVAILABLE:
        logging.warning("MinioClient no disponible - saltando upload a MinIO")
        return {'uploaded': False, 'error': 'MinIO client not available'}

    # Obtener directorio de modelos
    output_dir = context['ti'].xcom_pull(key='output_dir', task_ids='train_ridge')
    if not output_dir:
        logging.error("No se encontro output_dir")
        return {'uploaded': False, 'error': 'No output directory'}

    models_dir = Path(output_dir) / 'models'
    if not models_dir.exists():
        logging.error(f"Directorio de modelos no existe: {models_dir}")
        return {'uploaded': False, 'error': 'Models directory not found'}

    # Obtener metricas de cada modelo
    all_results = []
    for model_name in ML_MODELS:
        results = context['ti'].xcom_pull(key=f'results_{model_name}', task_ids=f'train_{model_name}')
        if results:
            all_results.extend(results)

    # Crear diccionario de metricas por modelo/horizonte
    metrics_lookup = {}
    for r in all_results:
        if 'direction_accuracy' in r:
            key = f"{r['model']}_h{r['horizon']}"
            metrics_lookup[key] = {
                'direction_accuracy': r.get('direction_accuracy'),
                'rmse': r.get('rmse'),
                'mae': r.get('mae'),
                'r2': r.get('r2')
            }

    # Fecha actual para estructura
    now = datetime.now()
    year = now.year
    month = now.month

    # Preparar lista de modelos para batch upload
    models_to_upload = []

    for model_file in models_dir.glob('*.pkl'):
        # Parse filename: {model}_h{horizon}.pkl
        name_parts = model_file.stem.rsplit('_h', 1)
        if len(name_parts) != 2:
            continue

        model_name = name_parts[0]
        try:
            horizon = int(name_parts[1])
        except ValueError:
            continue

        # Leer bytes del modelo
        with open(model_file, 'rb') as f:
            model_bytes = f.read()

        # Obtener metricas
        metrics_key = f"{model_name}_h{horizon}"
        metrics = metrics_lookup.get(metrics_key, {})

        models_to_upload.append({
            'model_name': model_name,
            'model_bytes': model_bytes,
            'horizon': horizon,
            'metrics': metrics
        })

        logging.info(f"Preparado para upload: {model_file.name} ({len(model_bytes)} bytes)")

    if not models_to_upload:
        logging.warning("No se encontraron modelos para subir")
        return {'uploaded': False, 'error': 'No models found'}

    # Metadata del run
    best_model = context['ti'].xcom_pull(key='best_model', task_ids='aggregate_results')
    best_da = context['ti'].xcom_pull(key='best_da', task_ids='aggregate_results')

    run_metadata = {
        'dag_id': DAG_ID,
        'execution_date': context['execution_date'].isoformat() if context.get('execution_date') else now.isoformat(),
        'run_id': context.get('run_id', 'manual'),
        'best_model': best_model,
        'best_direction_accuracy': best_da,
        'n_models_trained': len(models_to_upload),
        'horizons': HORIZONS,
        'model_types': ML_MODELS,
        'training_timestamp': now.isoformat()
    }

    try:
        minio_client = MinioClient()

        result = minio_client.upload_monthly_models_batch(
            year=year,
            month=month,
            models=models_to_upload,
            run_metadata=run_metadata
        )

        logging.info(f"Modelos subidos exitosamente:")
        logging.info(f"  - Base path: {result.get('base_path')}")
        logging.info(f"  - Modelos: {len(result.get('uploaded', []))}")
        logging.info(f"  - Metadata: {result.get('run_metadata')}")

        # Push resultados a XCom
        context['ti'].xcom_push(key='minio_models_path', value=result.get('base_path'))
        context['ti'].xcom_push(key='minio_uploaded_models', value=len(result.get('uploaded', [])))

        logging.info("=" * 60)

        upload_result = {
            'uploaded': True,
            'year': year,
            'month': month,
            'base_path': result.get('base_path'),
            'n_models': len(result.get('uploaded', []))
        }

        # =========================================================================
        # CHECKPOINT: Marcar fase como completada
        # =========================================================================
        mark_phase_complete(PHASE_NAME, upload_result)

        return upload_result

    except Exception as e:
        logging.error(f"Error subiendo modelos a MinIO: {e}")
        return {'uploaded': False, 'error': str(e)}


def run_storage_cleanup(**context):
    """
    Ejecuta limpieza programada de forecasts y modelos antiguos.

    - Elimina forecasts > 52 semanas
    - Elimina modelos > 6 meses
    """
    logging.info("=" * 60)
    logging.info("EJECUTANDO LIMPIEZA DE STORAGE")
    logging.info("=" * 60)

    if not MINIO_CLIENT_AVAILABLE:
        logging.warning("MinioClient no disponible - saltando limpieza")
        return {'cleanup': False, 'error': 'MinIO client not available'}

    try:
        minio_client = MinioClient()

        # Ejecutar limpieza programada
        result = minio_client.run_scheduled_cleanup(
            forecast_weeks=52,  # Mantener 1 anio de forecasts
            model_months=6,     # Mantener 6 meses de modelos
            dry_run=False       # Ejecutar eliminacion real
        )

        logging.info(f"Limpieza completada:")
        logging.info(f"  - Forecasts eliminados: {result['forecasts']['deleted_count']}")
        logging.info(f"  - Modelos eliminados: {result['models']['deleted_count']}")
        logging.info(f"  - Espacio liberado: {result['total_bytes_freed'] / (1024*1024):.2f} MB")

        context['ti'].xcom_push(key='cleanup_result', value=result)

        return {
            'cleanup': True,
            'forecasts_deleted': result['forecasts']['deleted_count'],
            'models_deleted': result['models']['deleted_count'],
            'bytes_freed': result['total_bytes_freed']
        }

    except Exception as e:
        logging.error(f"Error en limpieza: {e}")
        return {'cleanup': False, 'error': str(e)}


def cleanup_temp(**context):
    """Limpia archivos temporales."""
    import shutil

    temp_dir = Path('/opt/airflow/outputs/training_temp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        logging.info("Archivos temporales eliminados")

    return {'cleanup': 'completed'}


def notify_training_complete(**context):
    """Notifica que el entrenamiento se completo."""
    best_model = context['ti'].xcom_pull(key='best_model', task_ids='aggregate_results')
    best_da = context['ti'].xcom_pull(key='best_da', task_ids='aggregate_results')

    logging.info("=" * 60)
    logging.info("PIPELINE DE ENTRENAMIENTO MENSUAL COMPLETADO")
    logging.info(f"Mejor modelo: {best_model}")
    logging.info(f"Direction Accuracy: {best_da:.2%}" if best_da else "N/A")
    logging.info(f"Fecha: {datetime.now().isoformat()}")
    logging.info("=" * 60)

    return {'status': 'completed', 'best_model': best_model}


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Pipeline de reentrenamiento mensual de modelos ML',
    schedule_interval='0 7 1-7 * 0',  # Primer domingo del mes, 2:00 AM COT
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['training', 'monthly', 'ml', 'production'],
    max_active_runs=1,
    # DAG-level callbacks
    on_failure_callback=dag_failure_callback,
    sla_miss_callback=sla_miss_callback,
) as dag:

    # Task 1: Check data freshness
    check_data = BranchPythonOperator(
        task_id='check_data',
        python_callable=check_data_freshness,
        provide_context=True,
        on_failure_callback=task_failure_callback,
    )

    # Skip training branch
    skip_training = EmptyOperator(task_id='skip_training')

    # Task 2: Load and prepare data
    load_prepare = PythonOperator(
        task_id='load_and_prepare_data',
        python_callable=load_and_prepare_data,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # =========================================================================
    # Task 3: Train ALL 9 models (parallel execution)
    # =========================================================================
    # 9 modelos × 7 horizontes = 63 modelos totales

    # --- LINEALES (3) ---
    train_ridge_task = PythonOperator(
        task_id='train_ridge',
        python_callable=train_ridge,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    train_bayesian_ridge_task = PythonOperator(
        task_id='train_bayesian_ridge',
        python_callable=train_bayesian_ridge,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    train_ard_task = PythonOperator(
        task_id='train_ard',
        python_callable=train_ard,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # --- BOOSTING PUROS (3) ---
    train_xgboost_pure_task = PythonOperator(
        task_id='train_xgboost_pure',
        python_callable=train_xgboost_pure,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    train_lightgbm_pure_task = PythonOperator(
        task_id='train_lightgbm_pure',
        python_callable=train_lightgbm_pure,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    train_catboost_pure_task = PythonOperator(
        task_id='train_catboost_pure',
        python_callable=train_catboost_pure,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # --- HÍBRIDOS (3) ---
    train_hybrid_xgboost_task = PythonOperator(
        task_id='train_hybrid_xgboost',
        python_callable=train_hybrid_xgboost,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    train_hybrid_lightgbm_task = PythonOperator(
        task_id='train_hybrid_lightgbm',
        python_callable=train_hybrid_lightgbm,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    train_hybrid_catboost_task = PythonOperator(
        task_id='train_hybrid_catboost',
        python_callable=train_hybrid_catboost,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Lista de todas las tareas de entrenamiento para dependencias (9 modelos)
    all_training_tasks = [
        # Lineales (3)
        train_ridge_task,
        train_bayesian_ridge_task,
        train_ard_task,
        # Boosting Puros (3)
        train_xgboost_pure_task,
        train_lightgbm_pure_task,
        train_catboost_pure_task,
        # Híbridos (3)
        train_hybrid_xgboost_task,
        train_hybrid_lightgbm_task,
        train_hybrid_catboost_task,
    ]

    # =========================================================================
    # FASE 1: OPTUNA TUNING (antes del training)
    # =========================================================================
    optuna_tuning = PythonOperator(
        task_id='optuna_tuning',
        python_callable=run_optuna_tuning,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 4: Aggregate results
    aggregate = PythonOperator(
        task_id='aggregate_results',
        python_callable=aggregate_results,
        provide_context=True,
        trigger_rule='none_failed_min_one_success',
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # =========================================================================
    # FASE 3: WALK-FORWARD VALIDATION (después del training)
    # =========================================================================
    walk_forward_validation = PythonOperator(
        task_id='walk_forward_validation',
        python_callable=run_walk_forward_validation,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # =========================================================================
    # GENERACIÓN DE IMÁGENES DE BACKTEST
    # =========================================================================
    backtest_images = PythonOperator(
        task_id='generate_backtest_images',
        python_callable=generate_backtest_images,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 5: Register in MLflow
    mlflow_register = PythonOperator(
        task_id='register_mlflow',
        python_callable=register_mlflow,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 6: Upload models to MinIO (structured)
    upload_minio = PythonOperator(
        task_id='upload_models_minio',
        python_callable=upload_models_to_minio,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 7: Run storage cleanup (forecasts > 52 weeks, models > 6 months)
    storage_cleanup = PythonOperator(
        task_id='storage_cleanup',
        python_callable=run_storage_cleanup,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # Task 8: Cleanup temp files
    cleanup = PythonOperator(
        task_id='cleanup',
        python_callable=cleanup_temp,
        provide_context=True,
        on_failure_callback=task_failure_callback,
    )

    # Task 9: Notify
    notify = PythonOperator(
        task_id='notify_complete',
        python_callable=notify_training_complete,
        provide_context=True,
        on_failure_callback=task_failure_callback,
        on_success_callback=task_success_callback,
    )

    # =========================================================================
    # DEPENDENCIES - Flujo del DAG (3 FASES INTEGRADAS)
    # =========================================================================
    #
    # check_data ─┬─> skip_training (si no hay datos)
    #             │
    #             └─> load_prepare ──> optuna_tuning ──> [9 modelos paralelo] ──> aggregate
    #                   (FASE 2)         (FASE 1)                                    │
    #                                                                                │
    #                                         ┌─────────────────────────────────────┘
    #                                         ▼
    #                               walk_forward_validation ─┐
    #                                    (FASE 3)            │
    #                               backtest_images ─────────┤
    #                                                        ▼
    #                                              mlflow_register
    #                                                        │
    #                                                        v
    #                                        upload_minio ──> storage_cleanup ──> cleanup ──> notify
    #

    check_data >> [skip_training, load_prepare]

    # FASE 1: Optuna tuning después de cargar datos
    load_prepare >> optuna_tuning

    # FASE 2: Todos los modelos se entrenan en paralelo después de Optuna
    optuna_tuning >> all_training_tasks

    # Aggregate espera a que TODOS los modelos terminen
    all_training_tasks >> aggregate

    # FASE 3: Walk-forward validation y generación de imágenes (paralelo)
    aggregate >> [walk_forward_validation, backtest_images]

    # MLflow espera a que terminen ambas validaciones
    [walk_forward_validation, backtest_images] >> mlflow_register

    # Post-processing secuencial
    mlflow_register >> upload_minio >> storage_cleanup >> cleanup >> notify
