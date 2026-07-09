#!/usr/bin/env python3
"""
Script de Verificación: Fase 0 - Datos Macro

Verifica si la tabla macro_ohlcv existe y tiene datos suficientes.

Uso:
    python scripts/verify_fase0_data.py

Salida:
    - Reporte de existencia de tabla
    - Count de registros por símbolo (WTI, DXY)
    - Rango de fechas
    - Status: OK / WARNING / ERROR
"""

import psycopg2
import os
import sys
from datetime import datetime, timedelta

# Colores para output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(msg):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{msg.center(80)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*80}{Colors.END}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def connect_to_postgres():
    """Conectar a PostgreSQL"""
    try:
        conn = psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'localhost'),
            port=int(os.environ.get('POSTGRES_PORT', 5432)),
            database=os.environ.get('POSTGRES_DB', 'usdcop_db'),
            user=os.environ.get('POSTGRES_USER', 'usdcop'),
            password=os.environ.get('POSTGRES_PASSWORD', 'usdcop_pass')
        )
        print_success(f"Conectado a PostgreSQL: {conn.get_dsn_parameters()['dbname']}")
        return conn
    except Exception as e:
        print_error(f"No se pudo conectar a PostgreSQL: {e}")
        print_info("Verifica variables de entorno: POSTGRES_HOST, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        return None

def check_table_exists(conn):
    """Verificar si tabla macro_ohlcv existe"""
    try:
        cursor = conn.cursor()
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'macro_ohlcv'
        );
        """
        cursor.execute(query)
        exists = cursor.fetchone()[0]
        cursor.close()

        if exists:
            print_success("Tabla 'macro_ohlcv' existe")
            return True
        else:
            print_error("Tabla 'macro_ohlcv' NO existe")
            print_info("Ejecuta: psql -U usdcop -d usdcop_db -f init-scripts/02-macro-data-schema.sql")
            return False
    except Exception as e:
        print_error(f"Error verificando tabla: {e}")
        return False

def get_data_stats(conn):
    """Obtener estadísticas de datos macro"""
    try:
        cursor = conn.cursor()

        # Query estadísticas
        query = """
        SELECT
            symbol,
            COUNT(*) as record_count,
            MIN(time) as min_date,
            MAX(time) as max_date,
            COUNT(DISTINCT DATE(time)) as unique_days
        FROM macro_ohlcv
        GROUP BY symbol
        ORDER BY symbol;
        """

        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()

        if not results:
            print_warning("Tabla existe pero está VACÍA")
            print_info("Ejecuta el DAG: airflow dags trigger usdcop_m5__01b_l0_macro_acquire")
            return None

        return results
    except Exception as e:
        print_error(f"Error obteniendo estadísticas: {e}")
        return None

def validate_data_quality(stats):
    """Validar calidad de datos"""
    print_header("VALIDACIÓN DE CALIDAD DE DATOS")

    issues = []
    warnings = []

    # Verificar que existan WTI y DXY
    symbols = [row[0] for row in stats]

    if 'WTI' not in symbols:
        issues.append("Símbolo WTI NO encontrado")
    if 'DXY' not in symbols:
        issues.append("Símbolo DXY NO encontrado")

    # Verificar cada símbolo
    for symbol, count, min_date, max_date, unique_days in stats:
        print(f"\n{Colors.BOLD}Símbolo: {symbol}{Colors.END}")
        print(f"  Registros totales: {count:,}")
        print(f"  Fecha mínima: {min_date}")
        print(f"  Fecha máxima: {max_date}")
        print(f"  Días únicos: {unique_days:,}")

        # Validaciones
        # 1. Verificar count mínimo (al menos 10k registros = ~1 año de datos 1h)
        if count < 10000:
            warnings.append(f"{symbol}: Solo {count:,} registros (esperado > 10,000)")
        else:
            print_success(f"{symbol}: Count OK ({count:,} registros)")

        # 2. Verificar que datos sean recientes (últimos 7 días)
        days_since_last = (datetime.now().replace(tzinfo=max_date.tzinfo) - max_date).days
        if days_since_last > 7:
            warnings.append(f"{symbol}: Datos desactualizados (último: {max_date.date()})")
        else:
            print_success(f"{symbol}: Datos actualizados (último: {max_date.date()})")

        # 3. Verificar cobertura histórica (al menos 2 años)
        years_coverage = (max_date - min_date).days / 365.25
        if years_coverage < 2:
            warnings.append(f"{symbol}: Cobertura histórica insuficiente ({years_coverage:.1f} años)")
        else:
            print_success(f"{symbol}: Cobertura histórica OK ({years_coverage:.1f} años)")

    return issues, warnings

def check_data_gaps(conn):
    """Verificar si hay gaps en los datos"""
    print_header("VERIFICACIÓN DE GAPS")

    try:
        cursor = conn.cursor()

        # Query para detectar gaps > 2 días
        query = """
        WITH time_diffs AS (
            SELECT
                symbol,
                time,
                LAG(time) OVER (PARTITION BY symbol ORDER BY time) as prev_time,
                time - LAG(time) OVER (PARTITION BY symbol ORDER BY time) as time_diff
            FROM macro_ohlcv
        )
        SELECT
            symbol,
            prev_time,
            time,
            time_diff
        FROM time_diffs
        WHERE time_diff > INTERVAL '2 days'
        ORDER BY symbol, time
        LIMIT 10;
        """

        cursor.execute(query)
        gaps = cursor.fetchall()
        cursor.close()

        if gaps:
            print_warning(f"Encontrados {len(gaps)} gaps > 2 días")
            for symbol, prev_time, curr_time, diff in gaps[:5]:
                print(f"  {symbol}: {prev_time.date()} → {curr_time.date()} (gap: {diff})")
            if len(gaps) > 5:
                print(f"  ... y {len(gaps)-5} más")
        else:
            print_success("No se encontraron gaps significativos")

        return gaps
    except Exception as e:
        print_error(f"Error verificando gaps: {e}")
        return []

def print_final_status(issues, warnings, has_gaps):
    """Imprimir status final"""
    print_header("RESULTADO FINAL")

    if issues:
        print_error("FASE 0: FALLÓ ❌")
        print("\nProblemas críticos encontrados:")
        for issue in issues:
            print(f"  ❌ {issue}")
        print("\n" + "="*80)
        print("ACCIÓN REQUERIDA:")
        print("  1. Ejecutar: psql -U usdcop -d usdcop_db -f init-scripts/02-macro-data-schema.sql")
        print("  2. Trigger DAG: airflow dags trigger usdcop_m5__01b_l0_macro_acquire")
        print("  3. Esperar ~2-3 horas para catchup")
        print("  4. Re-ejecutar este script")
        print("="*80)
        return False

    elif warnings or has_gaps:
        print_warning("FASE 0: PARCIAL ⚠️")
        print("\nAdvertencias:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
        if has_gaps:
            print(f"  ⚠️  Gaps detectados en datos (ver arriba)")
        print("\n" + "="*80)
        print("RECOMENDACIÓN:")
        print("  - Datos presentes pero con advertencias")
        print("  - Puedes continuar con Fase 2, pero considera actualizar datos")
        print("  - Trigger DAG para actualizar: airflow dags trigger usdcop_m5__01b_l0_macro_acquire")
        print("="*80)
        return True

    else:
        print_success("FASE 0: COMPLETA ✅")
        print("\n" + "="*80)
        print("✅ Todos los checks pasaron")
        print("✅ Datos macro listos para Fase 2")
        print("✅ Puedes continuar con feature engineering")
        print("="*80)
        return True

def main():
    """Main execution"""
    print_header("VERIFICACIÓN FASE 0: DATOS MACRO")

    print_info("Verificando pipeline L0 de datos macro (WTI, DXY)")
    print_info(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Conectar a PostgreSQL
    conn = connect_to_postgres()
    if not conn:
        sys.exit(1)

    # 2. Verificar si tabla existe
    if not check_table_exists(conn):
        conn.close()
        print_final_status(['Tabla macro_ohlcv no existe'], [], False)
        sys.exit(1)

    # 3. Obtener estadísticas
    stats = get_data_stats(conn)
    if not stats:
        conn.close()
        print_final_status(['Tabla vacía'], [], False)
        sys.exit(1)

    # 4. Validar calidad
    issues, warnings = validate_data_quality(stats)

    # 5. Verificar gaps
    gaps = check_data_gaps(conn)

    # 6. Cerrar conexión
    conn.close()

    # 7. Status final
    success = print_final_status(issues, warnings, len(gaps) > 0)

    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
