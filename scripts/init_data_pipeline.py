#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
USD/COP TRADING SYSTEM - MASTER DATA INITIALIZATION PIPELINE
================================================================================

Script maestro para inicializar y mantener el pipeline de datos del sistema
de trading USD/COP con enfoque backup-first.

FLUJO DE EJECUCION:
    1. RESTORE: Cargar backups existentes (OHLCV + Macro + Features)
    2. DETECT:  Identificar gaps en los datos (fechas faltantes)
    3. UPDATE:  Actualizar con fuentes live (TwelveData, FRED, BanRep)
    4. PREPROCESS: Regenerar datasets de entrenamiento RL
    5. BACKUP:  Crear nuevos backups de las tablas actualizadas

USO:
    # Solo restaurar backups (primera instalacion)
    python scripts/init_data_pipeline.py --restore-only

    # Restaurar + detectar gaps
    python scripts/init_data_pipeline.py --restore-only --detect-gaps

    # Actualizar con datos live (requiere APIs configuradas)
    python scripts/init_data_pipeline.py --update-live

    # Regenerar datasets de preprocesamiento
    python scripts/init_data_pipeline.py --regenerate

    # Pipeline completo
    python scripts/init_data_pipeline.py --full

    # Crear backup de las tablas actuales
    python scripts/init_data_pipeline.py --backup

Autor: Pipeline Automatizado
Fecha: 2025-12-26
Version: 1.0.0
================================================================================
"""

import os
import sys
import gzip
import yaml
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
from io import StringIO

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

# =============================================================================
# CONFIGURACION
# =============================================================================

# Rutas base
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
CONFIG_DIR = PROJECT_ROOT / 'config'
PIPELINE_DIR = DATA_DIR / 'pipeline'

# Rutas de backups
BACKUP_DIR = DATA_DIR / 'backups'
MACRO_OUTPUT_DIR = PIPELINE_DIR / '05_resampling' / 'output'

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# COLORES PARA CONSOLA
# =============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[96m'


def print_header(text: str):
    """Imprime header decorado."""
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}")


def print_step(step_num: int, text: str):
    """Imprime paso numerado."""
    print(f"\n{Colors.BLUE}[PASO {step_num}]{Colors.END} {Colors.BOLD}{text}{Colors.END}")


def print_success(text: str):
    print(f"  {Colors.GREEN}[OK]{Colors.END} {text}")


def print_warning(text: str):
    print(f"  {Colors.YELLOW}[!]{Colors.END} {text}")


def print_error(text: str):
    print(f"  {Colors.RED}[X]{Colors.END} {text}")


def print_info(text: str):
    print(f"  {Colors.CYAN}[i]{Colors.END} {text}")


# =============================================================================
# CONFIGURACION DE BASE DE DATOS
# =============================================================================
def load_db_config() -> Dict[str, Any]:
    """Carga configuracion de la base de datos desde YAML o variables de entorno."""
    config_file = CONFIG_DIR / 'database.yaml'

    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            pg_config = config.get('connections', {}).get('postgres_timescale', {})
            return {
                'host': os.environ.get('POSTGRES_HOST', pg_config.get('host', 'localhost')),
                'port': int(os.environ.get('POSTGRES_PORT', pg_config.get('port', 5432))),
                'database': os.environ.get('POSTGRES_DB', pg_config.get('database', 'usdcop_trading')),
                'user': os.environ.get('POSTGRES_USER', pg_config.get('user', 'postgres')),
                'password': os.environ.get('POSTGRES_PASSWORD', pg_config.get('password', ''))
            }

    # Fallback a variables de entorno
    return {
        'host': os.environ.get('POSTGRES_HOST', 'localhost'),
        'port': int(os.environ.get('POSTGRES_PORT', 5432)),
        'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        'user': os.environ.get('POSTGRES_USER', 'postgres'),
        'password': os.environ.get('POSTGRES_PASSWORD', '')
    }


def get_connection():
    """Obtiene conexion a la base de datos con reintentos."""
    config = load_db_config()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**config)
            conn.autocommit = False
            logger.debug(f"Conectado a {config['database']}@{config['host']}")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Intento {attempt + 1}/{max_retries} fallido: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)
            else:
                raise


# =============================================================================
# FUNCIONES DE BACKUP
# =============================================================================
def find_latest_backup(pattern: str, directory: Path = BACKUP_DIR) -> Optional[Path]:
    """Encuentra el backup mas reciente que coincide con el patron."""
    if not directory.exists():
        return None

    backups = list(directory.glob(pattern))
    if not backups:
        return None

    return max(backups, key=lambda p: p.stat().st_mtime)


def table_row_count(conn, table_name: str) -> int:
    """Cuenta filas en una tabla."""
    with conn.cursor() as cur:
        try:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(
                sql.Identifier(table_name)
            ))
            return cur.fetchone()[0]
        except psycopg2.errors.UndefinedTable:
            return -1


def get_table_date_range(conn, table_name: str, date_col: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Obtiene rango de fechas de una tabla."""
    with conn.cursor() as cur:
        try:
            cur.execute(sql.SQL("SELECT MIN({col}), MAX({col}) FROM {table}").format(
                col=sql.Identifier(date_col),
                table=sql.Identifier(table_name)
            ))
            result = cur.fetchone()
            return result[0], result[1]
        except:
            return None, None


# =============================================================================
# RESTORE FUNCTIONS
# =============================================================================
def restore_ohlcv(conn, force: bool = False) -> int:
    """
    Restaura datos OHLCV desde backup al tabla usdcop_m5_ohlcv.

    Args:
        conn: Conexion a PostgreSQL
        force: Si True, inserta aunque haya datos existentes (usa UPSERT)

    Returns:
        Numero de filas insertadas/actualizadas
    """
    table_name = 'usdcop_m5_ohlcv'
    current_count = table_row_count(conn, table_name)

    if current_count > 0 and not force:
        print_info(f"{table_name} tiene {current_count:,} filas, saltando restore (use --force para forzar)")
        return 0

    # Buscar backup
    backup_file = find_latest_backup('usdcop_m5_ohlcv_*.csv.gz')
    if not backup_file:
        backup_file = find_latest_backup('ohlcv_*.csv.gz')

    if not backup_file:
        print_warning("No se encontro backup OHLCV")
        return 0

    print_info(f"Cargando OHLCV desde: {backup_file.name}")

    # Leer CSV comprimido
    with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)

    print_info(f"Leidas {len(df):,} filas del backup")

    # Preparar datos
    df['time'] = pd.to_datetime(df['time'])
    df['symbol'] = df.get('symbol', 'USD/COP')
    df['volume'] = df.get('volume', 0).fillna(0).astype(int)
    df['source'] = df.get('source', 'backup_restore')

    # Insertar con UPSERT
    inserted = 0
    with conn.cursor() as cur:
        # Preparar datos como tuplas
        cols = ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source']
        data = [tuple(row) for row in df[cols].values]

        # UPSERT query
        insert_query = """
            INSERT INTO usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume, source)
            VALUES %s
            ON CONFLICT (time, symbol) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                source = EXCLUDED.source
        """

        # Insertar en batches de 10000
        batch_size = 10000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            execute_values(cur, insert_query, batch, page_size=batch_size)
            inserted += len(batch)

        conn.commit()

    print_success(f"Insertadas/actualizadas {inserted:,} filas OHLCV")
    return inserted


def restore_macro(conn, force: bool = False) -> int:
    """
    Restaura datos macro desde MACRO_DAILY_CONSOLIDATED.csv.

    Args:
        conn: Conexion a PostgreSQL
        force: Si True, inserta aunque haya datos existentes (usa UPSERT)

    Returns:
        Numero de filas insertadas/actualizadas
    """
    table_name = 'macro_indicators_daily'
    current_count = table_row_count(conn, table_name)

    if current_count > 0 and not force:
        print_info(f"{table_name} tiene {current_count:,} filas, saltando restore (use --force para forzar)")
        return 0

    # Buscar archivo macro
    macro_file = MACRO_OUTPUT_DIR / 'MACRO_DAILY_CONSOLIDATED.csv'
    if not macro_file.exists():
        # Buscar en backups
        macro_file = find_latest_backup('macro_*.csv.gz')
        if macro_file:
            with gzip.open(macro_file, 'rt') as f:
                df = pd.read_csv(f)
        else:
            print_warning("No se encontro archivo macro (MACRO_DAILY_CONSOLIDATED.csv)")
            return 0
    else:
        df = pd.read_csv(macro_file)

    print_info(f"Cargando macro desde: {macro_file.name if hasattr(macro_file, 'name') else macro_file}")
    print_info(f"Leidas {len(df):,} filas del archivo macro")

    # Mapeo de columnas (uppercase a lowercase)
    column_mapping = {col: col.lower() for col in df.columns}
    df = df.rename(columns=column_mapping)

    # Convertir fecha
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha']).dt.date
    else:
        print_error("Columna 'fecha' no encontrada en archivo macro")
        return 0

    # Obtener columnas disponibles (excluyendo fecha)
    available_cols = [col for col in df.columns if col != 'fecha']

    # Insertar con UPSERT row by row (macro tiene muchas columnas opcionales)
    inserted = 0
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            # Preparar valores (convertir NaN a None)
            values = {col: (None if pd.isna(row[col]) else row[col]) for col in df.columns}

            # Construir query dinamico
            cols = list(values.keys())
            placeholders = ', '.join(['%s'] * len(cols))
            update_clause = ', '.join([f"{col} = EXCLUDED.{col}" for col in cols if col != 'fecha'])

            query = f"""
                INSERT INTO {table_name} ({', '.join(cols)})
                VALUES ({placeholders})
                ON CONFLICT (fecha) DO UPDATE SET {update_clause}
            """

            try:
                cur.execute(query, list(values.values()))
                inserted += 1
            except Exception as e:
                logger.debug(f"Error insertando fila macro: {e}")
                continue

        conn.commit()

    print_success(f"Insertadas/actualizadas {inserted:,} filas macro")
    return inserted


def restore_features(conn, force: bool = False) -> int:
    """
    Restaura features pre-calculadas desde backup.

    Args:
        conn: Conexion a PostgreSQL
        force: Si True, inserta aunque haya datos existentes

    Returns:
        Numero de filas insertadas/actualizadas
    """
    table_name = 'inference_features_5m'
    current_count = table_row_count(conn, table_name)

    if current_count == -1:
        print_warning(f"Tabla {table_name} no existe (probablemente no configurada)")
        return 0

    if current_count > 0 and not force:
        print_info(f"{table_name} tiene {current_count:,} filas, saltando restore")
        return 0

    # Buscar backup de features
    backup_file = find_latest_backup('features_*.csv.gz')
    if not backup_file:
        backup_file = find_latest_backup('inference_features_*.csv.gz')

    if not backup_file:
        print_info("No se encontro backup de features (esto es normal si es primera instalacion)")
        return 0

    print_info(f"Cargando features desde: {backup_file.name}")

    with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
        df = pd.read_csv(f)

    print_info(f"Leidas {len(df):,} filas de features")

    # TODO: Implementar logica de insercion segun esquema de inference_features_5m
    print_warning("Restore de features no implementado completamente - requiere esquema JSONB")
    return 0


# =============================================================================
# GAP DETECTION
# =============================================================================
def detect_ohlcv_gaps(conn) -> List[Dict]:
    """
    Detecta gaps en datos OHLCV (dias de mercado faltantes).

    Returns:
        Lista de diccionarios con informacion de gaps
    """
    gaps = []

    # Obtener rango de fechas
    min_date, max_date = get_table_date_range(conn, 'usdcop_m5_ohlcv', 'time')

    if not min_date or not max_date:
        return [{'type': 'no_data', 'message': 'No hay datos OHLCV'}]

    print_info(f"Rango OHLCV: {min_date.date()} a {max_date.date()}")

    # Verificar gap hasta hoy
    today = datetime.now()
    if max_date.date() < (today - timedelta(days=3)).date():
        gap_days = (today - max_date).days
        gaps.append({
            'type': 'recent_gap',
            'start': max_date.date(),
            'end': today.date(),
            'days': gap_days,
            'message': f"Gap de {gap_days} dias hasta hoy"
        })

    # Detectar gaps internos (dias sin datos)
    with conn.cursor() as cur:
        cur.execute("""
            WITH dates AS (
                SELECT DATE(time) as date
                FROM usdcop_m5_ohlcv
                GROUP BY DATE(time)
                ORDER BY date
            ),
            gaps AS (
                SELECT
                    date as start_date,
                    LEAD(date) OVER (ORDER BY date) as next_date,
                    LEAD(date) OVER (ORDER BY date) - date - 1 as gap_days
                FROM dates
            )
            SELECT start_date, next_date, gap_days
            FROM gaps
            WHERE gap_days > 3  -- Mas de 3 dias (considera fines de semana + festivos)
            ORDER BY gap_days DESC
            LIMIT 10
        """)

        for row in cur.fetchall():
            gaps.append({
                'type': 'internal_gap',
                'start': row[0],
                'end': row[1],
                'days': row[2],
                'message': f"Gap interno: {row[0]} a {row[1]} ({row[2]} dias)"
            })

    return gaps


def detect_macro_gaps(conn) -> List[Dict]:
    """Detecta gaps en datos macro."""
    gaps = []

    min_date, max_date = get_table_date_range(conn, 'macro_indicators_daily', 'fecha')

    if not min_date or not max_date:
        return [{'type': 'no_data', 'message': 'No hay datos macro'}]

    print_info(f"Rango Macro: {min_date} a {max_date}")

    # Verificar gap hasta hoy
    today = datetime.now().date()
    if max_date < (today - timedelta(days=7)):
        gap_days = (today - max_date).days
        gaps.append({
            'type': 'recent_gap',
            'start': max_date,
            'end': today,
            'days': gap_days,
            'message': f"Gap de {gap_days} dias hasta hoy"
        })

    return gaps


# =============================================================================
# LIVE UPDATE
# =============================================================================
def trigger_live_update() -> bool:
    """
    Intenta actualizar datos desde fuentes live via Airflow.

    Returns:
        True si se activo exitosamente
    """
    airflow_host = os.environ.get('AIRFLOW_HOST', 'localhost')
    airflow_port = os.environ.get('AIRFLOW_PORT', '8080')
    airflow_user = os.environ.get('AIRFLOW_USER', 'admin')
    airflow_password = os.environ.get('AIRFLOW_PASSWORD', 'admin')

    dags_to_trigger = [
        'v3.l0_ohlcv_backfill',
        'v3.l0_macro_unified',
    ]

    try:
        import requests
    except ImportError:
        print_warning("requests no instalado, no se puede trigger Airflow")
        return False

    success_count = 0
    for dag_id in dags_to_trigger:
        url = f'http://{airflow_host}:{airflow_port}/api/v1/dags/{dag_id}/dagRuns'

        try:
            response = requests.post(
                url,
                json={'conf': {'triggered_by': 'init_data_pipeline'}},
                headers={'Content-Type': 'application/json'},
                auth=(airflow_user, airflow_password),
                timeout=10
            )

            if response.status_code in [200, 201]:
                print_success(f"DAG {dag_id} triggered")
                success_count += 1
            else:
                print_warning(f"No se pudo trigger {dag_id}: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print_warning(f"Airflow no disponible para {dag_id}")
        except Exception as e:
            print_warning(f"Error triggering {dag_id}: {e}")

    return success_count > 0


# =============================================================================
# PREPROCESSING
# =============================================================================
def run_preprocessing_pipeline() -> bool:
    """
    Ejecuta el pipeline de preprocesamiento completo.

    Returns:
        True si se completo exitosamente
    """
    pipeline_script = PIPELINE_DIR / 'run_pipeline.py'

    if not pipeline_script.exists():
        print_error(f"Script de pipeline no encontrado: {pipeline_script}")
        return False

    print_info(f"Ejecutando: python {pipeline_script}")

    try:
        result = subprocess.run(
            [sys.executable, str(pipeline_script)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd=str(PIPELINE_DIR)
        )

        if result.returncode == 0:
            print_success("Pipeline de preprocesamiento completado")
            return True
        else:
            print_error(f"Pipeline fallo: {result.stderr[-500:]}")
            return False

    except subprocess.TimeoutExpired:
        print_error("Pipeline timeout (30 min)")
        return False
    except Exception as e:
        print_error(f"Error ejecutando pipeline: {e}")
        return False


# =============================================================================
# BACKUP CREATION
# =============================================================================
def create_backup(conn, table_name: str, date_col: str) -> Optional[Path]:
    """
    Crea backup comprimido de una tabla.

    Args:
        conn: Conexion a PostgreSQL
        table_name: Nombre de la tabla
        date_col: Columna de fecha para ordenar

    Returns:
        Path al archivo de backup creado
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = BACKUP_DIR / f'{table_name}_{timestamp}.csv.gz'

    # Asegurar que existe el directorio
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print_info(f"Creando backup: {backup_file.name}")

    with conn.cursor() as cur:
        # Exportar a buffer
        buffer = StringIO()
        cur.copy_expert(
            f"COPY (SELECT * FROM {table_name} ORDER BY {date_col}) TO STDOUT WITH CSV HEADER",
            buffer
        )
        buffer.seek(0)

        # Comprimir y escribir
        with gzip.open(backup_file, 'wt', encoding='utf-8') as f:
            f.write(buffer.getvalue())

    size_mb = backup_file.stat().st_size / (1024 * 1024)
    print_success(f"Backup creado: {backup_file.name} ({size_mb:.2f} MB)")

    return backup_file


def cleanup_old_backups(pattern: str, keep: int = 7):
    """
    Elimina backups antiguos, manteniendo los N mas recientes.

    Args:
        pattern: Patron glob para buscar backups
        keep: Numero de backups a mantener
    """
    backups = sorted(BACKUP_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    if len(backups) <= keep:
        return

    for old_backup in backups[keep:]:
        old_backup.unlink()
        print_info(f"Eliminado backup antiguo: {old_backup.name}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='USD/COP Trading System - Master Data Initialization Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  Primera instalacion (solo restaurar backups):
    python scripts/init_data_pipeline.py --restore-only

  Restaurar y detectar gaps:
    python scripts/init_data_pipeline.py --restore-only --detect-gaps

  Actualizar con datos live:
    python scripts/init_data_pipeline.py --update-live

  Regenerar datasets de entrenamiento:
    python scripts/init_data_pipeline.py --regenerate

  Pipeline completo (backup-first + update + regenerate):
    python scripts/init_data_pipeline.py --full

  Crear backup de tablas actuales:
    python scripts/init_data_pipeline.py --backup

  Forzar restore aunque haya datos:
    python scripts/init_data_pipeline.py --restore-only --force
        """
    )

    # Modos de operacion
    parser.add_argument('--restore-only', action='store_true',
                        help='Solo restaurar desde backups existentes')
    parser.add_argument('--update-live', action='store_true',
                        help='Actualizar con fuentes live (TwelveData, FRED, etc.)')
    parser.add_argument('--regenerate', action='store_true',
                        help='Regenerar datasets de preprocesamiento')
    parser.add_argument('--backup', action='store_true',
                        help='Crear backup de las tablas actuales')
    parser.add_argument('--full', action='store_true',
                        help='Ejecutar pipeline completo (restore + update + regenerate)')

    # Opciones adicionales
    parser.add_argument('--detect-gaps', action='store_true',
                        help='Detectar gaps en los datos')
    parser.add_argument('--force', action='store_true',
                        help='Forzar operaciones aunque haya datos existentes')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simular operaciones sin ejecutar')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Mostrar logs detallados')

    args = parser.parse_args()

    # Configurar logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    start_time = datetime.now()

    print_header("USD/COP TRADING SYSTEM - DATA INITIALIZATION PIPELINE")
    print(f"Fecha: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Backups: {BACKUP_DIR}")

    if args.dry_run:
        print(f"\n{Colors.YELLOW}[DRY RUN] Modo simulacion - no se ejecutaran cambios{Colors.END}\n")

    # Determinar operaciones
    do_restore = args.restore_only or args.full
    do_detect_gaps = args.detect_gaps or args.full
    do_update = args.update_live or args.full
    do_regenerate = args.regenerate or args.full
    do_backup = args.backup

    # Si no se especifica ninguna operacion, mostrar ayuda
    if not any([do_restore, do_detect_gaps, do_update, do_regenerate, do_backup]):
        parser.print_help()
        return 1

    results = {
        'restore_ohlcv': None,
        'restore_macro': None,
        'restore_features': None,
        'gaps_detected': [],
        'live_update': None,
        'preprocessing': None,
        'backup_created': []
    }

    try:
        conn = get_connection()
        print_success(f"Conectado a la base de datos")

        # =================================================================
        # PASO 1: RESTORE BACKUPS
        # =================================================================
        if do_restore:
            print_step(1, "RESTAURANDO BACKUPS")

            if not args.dry_run:
                results['restore_ohlcv'] = restore_ohlcv(conn, force=args.force)
                results['restore_macro'] = restore_macro(conn, force=args.force)
                results['restore_features'] = restore_features(conn, force=args.force)
            else:
                print_info("[DRY RUN] Se restaurarian backups de OHLCV, Macro y Features")

        # =================================================================
        # PASO 2: DETECTAR GAPS
        # =================================================================
        if do_detect_gaps:
            print_step(2, "DETECTANDO GAPS EN DATOS")

            ohlcv_gaps = detect_ohlcv_gaps(conn)
            macro_gaps = detect_macro_gaps(conn)

            results['gaps_detected'] = ohlcv_gaps + macro_gaps

            if not results['gaps_detected']:
                print_success("No se detectaron gaps significativos")
            else:
                for gap in results['gaps_detected']:
                    if gap['type'] == 'no_data':
                        print_warning(gap['message'])
                    else:
                        print_warning(gap['message'])

        # =================================================================
        # PASO 3: ACTUALIZAR CON DATOS LIVE
        # =================================================================
        if do_update:
            print_step(3, "ACTUALIZANDO CON FUENTES LIVE")

            if not args.dry_run:
                results['live_update'] = trigger_live_update()
                if not results['live_update']:
                    print_warning("No se pudo activar actualizacion live")
                    print_info("Para actualizar manualmente, ejecute los DAGs desde Airflow UI")
            else:
                print_info("[DRY RUN] Se triggerearian DAGs de actualizacion")

        # =================================================================
        # PASO 4: REGENERAR PREPROCESAMIENTO
        # =================================================================
        if do_regenerate:
            print_step(4, "REGENERANDO DATASETS DE ENTRENAMIENTO")

            if not args.dry_run:
                results['preprocessing'] = run_preprocessing_pipeline()
            else:
                print_info("[DRY RUN] Se ejecutaria pipeline de preprocesamiento")

        # =================================================================
        # PASO 5: CREAR BACKUPS
        # =================================================================
        if do_backup:
            print_step(5, "CREANDO BACKUPS")

            if not args.dry_run:
                # Backup OHLCV
                try:
                    backup_path = create_backup(conn, 'usdcop_m5_ohlcv', 'time')
                    if backup_path:
                        results['backup_created'].append(str(backup_path))
                except Exception as e:
                    print_error(f"Error creando backup OHLCV: {e}")

                # Backup Macro
                try:
                    backup_path = create_backup(conn, 'macro_indicators_daily', 'fecha')
                    if backup_path:
                        results['backup_created'].append(str(backup_path))
                except Exception as e:
                    print_error(f"Error creando backup Macro: {e}")

                # Limpiar backups antiguos
                cleanup_old_backups('usdcop_m5_ohlcv_*.csv.gz', keep=7)
                cleanup_old_backups('macro_indicators_daily_*.csv.gz', keep=7)
            else:
                print_info("[DRY RUN] Se crearian backups de tablas")

        conn.close()

    except psycopg2.OperationalError as e:
        print_error(f"No se pudo conectar a la base de datos: {e}")
        print_info("Verifique que PostgreSQL este corriendo y las credenciales sean correctas")
        return 1
    except Exception as e:
        print_error(f"Error inesperado: {e}")
        logger.exception("Error detallado:")
        return 1

    # =================================================================
    # RESUMEN FINAL
    # =================================================================
    elapsed = (datetime.now() - start_time).total_seconds()

    print_header("RESUMEN DE EJECUCION")
    print(f"Tiempo total: {elapsed:.1f} segundos")

    if do_restore:
        print(f"\nRestauracion:")
        print(f"  - OHLCV: {results['restore_ohlcv'] or 0:,} filas")
        print(f"  - Macro: {results['restore_macro'] or 0:,} filas")
        print(f"  - Features: {results['restore_features'] or 0:,} filas")

    if do_detect_gaps and results['gaps_detected']:
        print(f"\nGaps detectados: {len(results['gaps_detected'])}")
        for gap in results['gaps_detected'][:5]:  # Mostrar max 5
            print(f"  - {gap['message']}")

    if do_update:
        status = "Activado" if results['live_update'] else "No disponible"
        print(f"\nActualizacion live: {status}")

    if do_regenerate:
        status = "Completado" if results['preprocessing'] else "No ejecutado"
        print(f"\nPreprocesamiento: {status}")

    if do_backup and results['backup_created']:
        print(f"\nBackups creados: {len(results['backup_created'])}")
        for bp in results['backup_created']:
            print(f"  - {Path(bp).name}")

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
