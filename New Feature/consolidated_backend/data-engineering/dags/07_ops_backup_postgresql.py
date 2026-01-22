"""
DAG: backup_postgresql
======================
Pipeline de Backup Diario de PostgreSQL

Ejecuta diariamente a las 2:00 AM (hora Colombia):
1. Backup completo de la base de datos con pg_dump
2. Compresion con gzip
3. Subida a MinIO bucket "backups"
4. Limpieza de backups antiguos (> 7 dias)
5. Verificacion del backup

Schedule: 0 2 * * * (2:00 AM diario, hora local)
Timeout: 30 minutos por tarea
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
import subprocess
import logging
import os
import gzip
import hashlib
from pathlib import Path

DAG_ID = "07_ops_backup_postgresql"

# =============================================================================
# CONFIGURATION
# =============================================================================

# PostgreSQL
POSTGRES_HOST = os.getenv("PGHOST", "postgres")
POSTGRES_PORT = os.getenv("PGPORT", "5432")
POSTGRES_DB = os.getenv("PGDATABASE", "pipeline")
POSTGRES_USER = os.getenv("PGUSER", "pipeline")
POSTGRES_PASSWORD = os.getenv("PGPASSWORD", "pipeline123")

# MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BACKUP_BUCKET", "backups")
MINIO_USE_SSL = os.getenv("MINIO_USE_SSL", "false").lower() == "true"

# Directorios
BACKUP_DIR = Path("/opt/airflow/backups")
LOG_DIR = Path("/opt/airflow/logs/backups")
RETENTION_DAYS = 7

# Alertas
ALERT_EMAILS = ["alerts@example.com"]

# =============================================================================
# DEFAULT ARGS
# =============================================================================

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email": ALERT_EMAILS,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=30),
}

# =============================================================================
# FUNCIONES DEL PIPELINE
# =============================================================================


def backup_database(**context):
    """
    Realiza backup completo de PostgreSQL con pg_dump y compresion gzip.

    Returns:
        dict: Informacion del backup creado
    """
    import psycopg2

    logging.info("=" * 60)
    logging.info("INICIANDO BACKUP DE POSTGRESQL")
    logging.info("=" * 60)

    # Crear directorios si no existen
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Generar nombre de archivo con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"backup_{timestamp}.sql.gz"
    backup_path = BACKUP_DIR / backup_filename

    logging.info(f"Database: {POSTGRES_DB}")
    logging.info(f"Host: {POSTGRES_HOST}:{POSTGRES_PORT}")
    logging.info(f"Backup file: {backup_path}")

    # Verificar conexion a PostgreSQL primero
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )

        # Obtener estadisticas de la BD
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                pg_size_pretty(pg_database_size(current_database())) as db_size,
                (SELECT COUNT(*) FROM core.usdcop_historical) as usdcop_count,
                (SELECT COUNT(*) FROM core.macro_indicators) as macro_count,
                (SELECT COUNT(*) FROM ml.forecasts) as forecast_count
        """)
        stats = cursor.fetchone()
        logging.info(f"Database size: {stats[0]}")
        logging.info(f"USDCOP records: {stats[1]}")
        logging.info(f"Macro indicators: {stats[2]}")
        logging.info(f"Forecasts: {stats[3]}")

        conn.close()
        logging.info("Conexion a PostgreSQL verificada")

    except Exception as e:
        logging.error(f"Error conectando a PostgreSQL: {e}")
        raise

    # Ejecutar pg_dump con compresion
    env = os.environ.copy()
    env["PGPASSWORD"] = POSTGRES_PASSWORD

    pg_dump_cmd = [
        "pg_dump",
        "-h", POSTGRES_HOST,
        "-p", str(POSTGRES_PORT),
        "-U", POSTGRES_USER,
        "-d", POSTGRES_DB,
        "--verbose",
        "--no-owner",
        "--no-privileges",
        "--format=plain",
        "--encoding=UTF8"
    ]

    start_time = datetime.now()
    logging.info(f"Ejecutando pg_dump a las {start_time}")

    try:
        # Ejecutar pg_dump y comprimir con gzip
        with open(backup_path, "wb") as f_out:
            pg_dump_proc = subprocess.Popen(
                pg_dump_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )

            gzip_proc = subprocess.Popen(
                ["gzip", "-9"],
                stdin=pg_dump_proc.stdout,
                stdout=f_out,
                stderr=subprocess.PIPE
            )

            pg_dump_proc.stdout.close()
            gzip_output, gzip_error = gzip_proc.communicate()
            pg_dump_proc.wait()

            if pg_dump_proc.returncode != 0:
                stderr = pg_dump_proc.stderr.read().decode()
                raise Exception(f"pg_dump failed: {stderr}")

            if gzip_proc.returncode != 0:
                raise Exception(f"gzip failed: {gzip_error.decode()}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Obtener tamano del backup
        backup_size = backup_path.stat().st_size
        backup_size_mb = backup_size / (1024 * 1024)

        # Calcular checksum
        with open(backup_path, "rb") as f:
            checksum = hashlib.md5(f.read()).hexdigest()

        logging.info(f"Backup completado exitosamente")
        logging.info(f"Tamano: {backup_size_mb:.2f} MB")
        logging.info(f"Duracion: {duration:.2f} segundos")
        logging.info(f"Checksum MD5: {checksum}")

        # Guardar metadata en XCom
        backup_info = {
            "filename": backup_filename,
            "path": str(backup_path),
            "size_bytes": backup_size,
            "size_mb": round(backup_size_mb, 2),
            "checksum_md5": checksum,
            "timestamp": timestamp,
            "duration_seconds": round(duration, 2),
            "database": POSTGRES_DB
        }

        context["ti"].xcom_push(key="backup_info", value=backup_info)

        return backup_info

    except Exception as e:
        logging.error(f"Error durante el backup: {e}")
        # Limpiar archivo parcial si existe
        if backup_path.exists():
            backup_path.unlink()
        raise


def upload_to_minio(**context):
    """
    Sube el backup a MinIO bucket.

    Returns:
        dict: Informacion de la subida
    """
    from minio import Minio
    from minio.error import S3Error

    logging.info("=" * 60)
    logging.info("SUBIENDO BACKUP A MINIO")
    logging.info("=" * 60)

    # Obtener info del backup
    backup_info = context["ti"].xcom_pull(key="backup_info", task_ids="backup_database")

    if not backup_info:
        raise ValueError("No se encontro informacion del backup")

    backup_path = Path(backup_info["path"])

    if not backup_path.exists():
        raise FileNotFoundError(f"Archivo de backup no encontrado: {backup_path}")

    # Configurar cliente MinIO
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL
    )

    # Crear bucket si no existe
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            logging.info(f"Bucket '{MINIO_BUCKET}' creado")
        else:
            logging.info(f"Bucket '{MINIO_BUCKET}' existe")
    except S3Error as e:
        logging.error(f"Error con bucket: {e}")
        raise

    # Definir ruta en MinIO (organizado por anio/mes)
    year = datetime.now().strftime("%Y")
    month = datetime.now().strftime("%m")
    object_name = f"postgresql/{year}/{month}/{backup_info['filename']}"

    logging.info(f"Subiendo a: s3://{MINIO_BUCKET}/{object_name}")

    try:
        # Subir archivo
        result = client.fput_object(
            bucket_name=MINIO_BUCKET,
            object_name=object_name,
            file_path=str(backup_path),
            content_type="application/gzip",
            metadata={
                "database": backup_info["database"],
                "checksum-md5": backup_info["checksum_md5"],
                "backup-timestamp": backup_info["timestamp"]
            }
        )

        logging.info(f"Backup subido exitosamente")
        logging.info(f"Object: {result.object_name}")
        logging.info(f"ETag: {result.etag}")
        logging.info(f"Version ID: {result.version_id}")

        upload_info = {
            "bucket": MINIO_BUCKET,
            "object_name": object_name,
            "etag": result.etag,
            "version_id": result.version_id,
            "minio_path": f"s3://{MINIO_BUCKET}/{object_name}"
        }

        context["ti"].xcom_push(key="upload_info", value=upload_info)

        return upload_info

    except S3Error as e:
        logging.error(f"Error subiendo a MinIO: {e}")
        raise


def cleanup_old_backups(**context):
    """
    Limpia backups locales mayores a RETENTION_DAYS dias.

    Returns:
        dict: Estadisticas de limpieza
    """
    from minio import Minio
    from minio.error import S3Error

    logging.info("=" * 60)
    logging.info("LIMPIANDO BACKUPS ANTIGUOS")
    logging.info("=" * 60)

    deleted_local = 0
    deleted_minio = 0
    freed_space = 0

    # Limpiar backups locales
    logging.info(f"Limpiando backups locales mayores a {RETENTION_DAYS} dias...")

    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)

    if BACKUP_DIR.exists():
        for backup_file in BACKUP_DIR.glob("backup_*.sql.gz"):
            # Obtener fecha del archivo
            file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)

            if file_mtime < cutoff_date:
                file_size = backup_file.stat().st_size
                logging.info(f"Eliminando: {backup_file.name} ({file_size / 1024 / 1024:.2f} MB)")
                backup_file.unlink()
                deleted_local += 1
                freed_space += file_size

    logging.info(f"Backups locales eliminados: {deleted_local}")
    logging.info(f"Espacio liberado: {freed_space / 1024 / 1024:.2f} MB")

    # Limpiar backups antiguos en MinIO (opcional, mantener 30 dias en MinIO)
    minio_retention_days = 30

    try:
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_USE_SSL
        )

        if client.bucket_exists(MINIO_BUCKET):
            cutoff_minio = datetime.now() - timedelta(days=minio_retention_days)

            objects = client.list_objects(
                MINIO_BUCKET,
                prefix="postgresql/",
                recursive=True
            )

            for obj in objects:
                if obj.last_modified.replace(tzinfo=None) < cutoff_minio:
                    logging.info(f"Eliminando de MinIO: {obj.object_name}")
                    client.remove_object(MINIO_BUCKET, obj.object_name)
                    deleted_minio += 1

            logging.info(f"Backups eliminados de MinIO: {deleted_minio}")

    except Exception as e:
        logging.warning(f"Error limpiando MinIO (no critico): {e}")

    cleanup_stats = {
        "deleted_local": deleted_local,
        "deleted_minio": deleted_minio,
        "freed_space_mb": round(freed_space / 1024 / 1024, 2),
        "retention_days_local": RETENTION_DAYS,
        "retention_days_minio": minio_retention_days
    }

    context["ti"].xcom_push(key="cleanup_stats", value=cleanup_stats)

    return cleanup_stats


def verify_backup(**context):
    """
    Verifica la integridad del backup creado.

    Returns:
        dict: Resultado de la verificacion
    """
    from minio import Minio
    from minio.error import S3Error
    import gzip

    logging.info("=" * 60)
    logging.info("VERIFICANDO BACKUP")
    logging.info("=" * 60)

    backup_info = context["ti"].xcom_pull(key="backup_info", task_ids="backup_database")
    upload_info = context["ti"].xcom_pull(key="upload_info", task_ids="upload_to_minio")

    verification_results = {
        "local_file_exists": False,
        "local_file_valid": False,
        "minio_object_exists": False,
        "checksums_match": False,
        "verified": False
    }

    # 1. Verificar archivo local
    backup_path = Path(backup_info["path"])

    if backup_path.exists():
        verification_results["local_file_exists"] = True
        logging.info(f"Archivo local existe: {backup_path}")

        # Verificar que el gzip es valido
        try:
            with gzip.open(backup_path, "rt", encoding="utf-8") as f:
                # Leer primeras lineas para verificar
                first_lines = []
                for i, line in enumerate(f):
                    if i >= 10:
                        break
                    first_lines.append(line)

                # Verificar que parece un dump de PostgreSQL
                content = "".join(first_lines)
                if "PostgreSQL" in content or "pg_dump" in content or "CREATE" in content:
                    verification_results["local_file_valid"] = True
                    logging.info("Archivo gzip valido y contiene dump SQL")
                else:
                    logging.warning("Archivo gzip valido pero contenido no parece dump SQL")

        except Exception as e:
            logging.error(f"Error verificando archivo local: {e}")
    else:
        logging.warning(f"Archivo local no existe: {backup_path}")

    # 2. Verificar objeto en MinIO
    if upload_info:
        try:
            client = Minio(
                MINIO_ENDPOINT,
                access_key=MINIO_ACCESS_KEY,
                secret_key=MINIO_SECRET_KEY,
                secure=MINIO_USE_SSL
            )

            stat = client.stat_object(MINIO_BUCKET, upload_info["object_name"])

            verification_results["minio_object_exists"] = True
            logging.info(f"Objeto MinIO existe: {upload_info['object_name']}")
            logging.info(f"Tamano en MinIO: {stat.size / 1024 / 1024:.2f} MB")

            # Verificar tamano coincide
            if abs(stat.size - backup_info["size_bytes"]) < 1024:  # Tolerancia de 1KB
                verification_results["checksums_match"] = True
                logging.info("Tamanos coinciden")
            else:
                logging.warning(f"Tamanos no coinciden: local={backup_info['size_bytes']}, minio={stat.size}")

        except S3Error as e:
            logging.error(f"Error verificando MinIO: {e}")

    # 3. Determinar si verificacion fue exitosa
    verification_results["verified"] = (
        verification_results["local_file_exists"] and
        verification_results["local_file_valid"] and
        verification_results["minio_object_exists"]
    )

    if verification_results["verified"]:
        logging.info("=" * 60)
        logging.info("VERIFICACION EXITOSA")
        logging.info("=" * 60)
    else:
        logging.error("VERIFICACION FALLIDA")
        for key, value in verification_results.items():
            logging.info(f"  {key}: {value}")
        raise Exception("Backup verification failed")

    # Resumen final
    logging.info("RESUMEN DEL BACKUP:")
    logging.info(f"  Archivo: {backup_info['filename']}")
    logging.info(f"  Tamano: {backup_info['size_mb']} MB")
    logging.info(f"  MinIO: {upload_info['minio_path']}")
    logging.info(f"  Checksum: {backup_info['checksum_md5']}")

    context["ti"].xcom_push(key="verification_results", value=verification_results)

    return verification_results


def notify_backup_complete(**context):
    """
    Notifica que el backup se completo exitosamente.
    """
    backup_info = context["ti"].xcom_pull(key="backup_info", task_ids="backup_database")
    upload_info = context["ti"].xcom_pull(key="upload_info", task_ids="upload_to_minio")
    cleanup_stats = context["ti"].xcom_pull(key="cleanup_stats", task_ids="cleanup_old")

    logging.info("=" * 60)
    logging.info("BACKUP POSTGRESQL COMPLETADO")
    logging.info("=" * 60)
    logging.info(f"Fecha: {datetime.now().isoformat()}")
    logging.info(f"Database: {POSTGRES_DB}")
    logging.info(f"Archivo: {backup_info['filename']}")
    logging.info(f"Tamano: {backup_info['size_mb']} MB")
    logging.info(f"MinIO Path: {upload_info['minio_path']}")
    logging.info(f"Backups limpiados: {cleanup_stats['deleted_local']} local, {cleanup_stats['deleted_minio']} MinIO")
    logging.info("=" * 60)

    return {
        "status": "SUCCESS",
        "timestamp": datetime.now().isoformat(),
        "backup_file": backup_info["filename"],
        "size_mb": backup_info["size_mb"],
        "minio_path": upload_info["minio_path"]
    }


# =============================================================================
# DAG DEFINITION
# =============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Backup diario de PostgreSQL a las 2:00 AM con subida a MinIO",
    schedule_interval="0 2 * * *",  # 2:00 AM diario
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["backup", "postgresql", "minio", "infrastructure"],
    max_active_runs=1,
    doc_md=__doc__,
) as dag:

    # Task 1: Realizar backup de la base de datos
    backup_task = PythonOperator(
        task_id="backup_database",
        python_callable=backup_database,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        Realiza backup completo de PostgreSQL usando pg_dump.
        - Comprime el output con gzip nivel 9
        - Genera checksum MD5 para verificacion
        - Guarda metadata en XCom
        """
    )

    # Task 2: Subir backup a MinIO
    upload_task = PythonOperator(
        task_id="upload_to_minio",
        python_callable=upload_to_minio,
        provide_context=True,
        execution_timeout=timedelta(minutes=30),
        doc_md="""
        Sube el backup a MinIO bucket 'backups'.
        - Organiza por anio/mes
        - Agrega metadata al objeto
        """
    )

    # Task 3: Limpiar backups antiguos
    cleanup_task = PythonOperator(
        task_id="cleanup_old",
        python_callable=cleanup_old_backups,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        Limpia backups antiguos:
        - Local: > 7 dias
        - MinIO: > 30 dias
        """
    )

    # Task 4: Verificar backup
    verify_task = PythonOperator(
        task_id="verify_backup",
        python_callable=verify_backup,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
        doc_md="""
        Verifica integridad del backup:
        - Archivo local existe y es valido
        - Objeto MinIO existe
        - Tamanos coinciden
        """
    )

    # Task 5: Notificar completado
    notify_task = PythonOperator(
        task_id="notify_complete",
        python_callable=notify_backup_complete,
        provide_context=True,
        trigger_rule="all_success",
        doc_md="Notifica que el backup se completo exitosamente"
    )

    # Dependencies
    backup_task >> upload_task >> cleanup_task >> verify_task >> notify_task
