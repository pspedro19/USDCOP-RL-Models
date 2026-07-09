"""
Validador de integridad anti-leakage para datasets macro.
==========================================================

Verifica que los datasets macro no tienen data leakage, es decir,
que los valores usados en cada timestamp corresponden a datos que
ya habían sido publicados en ese momento.

Contract: CTR-L0-CALENDAR-001

Uso:
    python scripts/validate_macro_no_leakage.py
    python scripts/validate_macro_no_leakage.py --file MACRO_MONTHLY_CLEAN.csv
    python scripts/validate_macro_no_leakage.py --verbose
    python scripts/validate_macro_no_leakage.py --sample-rate 50

Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path

# Asegurar que src está en el path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import logging
from src.data.economic_calendar import EconomicCalendar

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
DEFAULT_FILES = [
    "MACRO_DAILY_CLEAN.csv",
    "MACRO_MONTHLY_CLEAN.csv",
    "MACRO_QUARTERLY_CLEAN.csv"
]


def validate_file(
    filepath: Path,
    calendar: EconomicCalendar,
    sample_rate: int = 100,
    verbose: bool = False
) -> dict:
    """
    Validar un archivo CSV de datos macro.

    Args:
        filepath: Ruta al archivo CSV
        calendar: Instancia de EconomicCalendar
        sample_rate: Validar cada N filas
        verbose: Mostrar detalles

    Returns:
        Dict con resultados de validación
    """
    logger.info(f"\nValidating: {filepath.name}")
    logger.info("-" * 60)

    # Cargar dataset
    df = pd.read_csv(filepath)

    # Detectar columna de fecha
    date_col = 'fecha' if 'fecha' in df.columns else df.columns[0]

    # Filter out metadata rows (like FUENTE_URL)
    # These rows have non-date values in the date column
    df = df[~df[date_col].astype(str).str.contains('FUENTE|URL|http|Source', case=False, na=False)]

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)

    logger.info(f"  Rows: {len(df)}")
    logger.info(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    # Identificar variables del calendario
    calendar_vars = [col for col in df.columns if col.lower() in calendar.variables]
    # Try lowercase match
    col_map = {col.lower(): col for col in df.columns}

    vars_to_check = []
    for var_name in calendar.variables.keys():
        if var_name in df.columns:
            vars_to_check.append(var_name)
        elif var_name.upper() in df.columns:
            vars_to_check.append(var_name.upper())

    logger.info(f"  Calendar variables found: {len(vars_to_check)}")

    if not vars_to_check:
        logger.warning("  No calendar variables found in dataset")
        return {'file': filepath.name, 'status': 'SKIP', 'results': {}}

    # Validar cada variable
    results = {}
    passed = 0
    failed = 0

    for var in vars_to_check:
        # Usar nombre lowercase para el calendario
        var_lower = var.lower()

        if var_lower not in calendar.variables:
            continue

        # Crear DataFrame temporal con nombre lowercase para validación
        df_temp = df[[var]].copy()
        df_temp.columns = [var_lower]

        # Validar
        var_result = calendar.validate_dataset_no_leakage(
            df_temp,
            variables=[var_lower],
            sample_rate=sample_rate,
            verbose=verbose
        )

        is_valid = var_result.get(var_lower, None)
        results[var] = is_valid

        if is_valid is True:
            status = "PASS"
            passed += 1
        elif is_valid is False:
            status = "FAIL - LEAKAGE DETECTED"
            failed += 1
        else:
            status = "SKIP"

        risk = calendar.get_leakage_risk(var_lower)
        print(f"  {var}: {status} (risk: {risk})")

    # Resumen
    total = passed + failed
    logger.info(f"\n  Summary: {passed}/{total} passed, {failed}/{total} failed")

    return {
        'file': filepath.name,
        'status': 'PASS' if failed == 0 else 'FAIL',
        'passed': passed,
        'failed': failed,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Validate macro datasets for data leakage'
    )
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Specific CSV file to validate (default: all CLEAN files)'
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory containing CSV files'
    )
    parser.add_argument(
        '--sample-rate', '-s',
        type=int,
        default=100,
        help='Validate every N rows (default: 100)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed validation output'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MACRO DATASET LEAKAGE VALIDATOR")
    print("=" * 70)

    # Cargar calendario
    calendar = EconomicCalendar()
    print(f"Loaded calendar with {len(calendar.variables)} variables")

    # Determinar archivos a validar
    output_dir = Path(args.dir)

    if args.file:
        files = [output_dir / args.file]
    else:
        files = [output_dir / f for f in DEFAULT_FILES if (output_dir / f).exists()]

    if not files:
        logger.error(f"No CSV files found in {output_dir}")
        sys.exit(1)

    print(f"Files to validate: {len(files)}")

    # Validar cada archivo
    all_results = []
    total_passed = 0
    total_failed = 0

    for filepath in files:
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        result = validate_file(
            filepath,
            calendar,
            sample_rate=args.sample_rate,
            verbose=args.verbose
        )
        all_results.append(result)
        total_passed += result.get('passed', 0)
        total_failed += result.get('failed', 0)

    # Resumen final
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for result in all_results:
        status_icon = "[OK]" if result['status'] == 'PASS' else "[X]"
        print(f"  {status_icon} {result['file']}: {result['status']}")

    print("-" * 70)
    print(f"Total: {total_passed} passed, {total_failed} failed")

    if total_failed > 0:
        print("\nWARNING: Data leakage detected in one or more variables!")
        print("Review the validation output above for details.")
        sys.exit(1)
    else:
        print("\nAll validations passed. No data leakage detected.")
        sys.exit(0)


if __name__ == '__main__':
    main()
