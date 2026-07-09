#!/usr/bin/env python3
"""
Regenerate Consolidated Datasets
=================================

Regenerates all consolidated datasets in multiple formats (CSV, Parquet, Excel).
This script runs the full pipeline: Fusion -> Cleaning -> Copy to Consolidated.

Output Structure (9 files = 3 datasets x 3 formats):
    data/pipeline/01_sources/consolidated/
    ├── MACRO_DAILY_MASTER.csv
    ├── MACRO_DAILY_MASTER.parquet
    ├── MACRO_DAILY_MASTER.xlsx
    ├── MACRO_MONTHLY_MASTER.csv
    ├── MACRO_MONTHLY_MASTER.parquet
    ├── MACRO_MONTHLY_MASTER.xlsx
    ├── MACRO_QUARTERLY_MASTER.csv
    ├── MACRO_QUARTERLY_MASTER.parquet
    └── MACRO_QUARTERLY_MASTER.xlsx

Contract: CTR-L0-CONSOLIDATED-001

Usage:
    python scripts/regenerate_consolidated_datasets.py

    # With validation only (no regeneration)
    python scripts/regenerate_consolidated_datasets.py --validate-only

Version: 1.0.0
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_DIR = PROJECT_ROOT / "data" / "pipeline"

# Source directories
FUSION_OUTPUT_DIR = PIPELINE_DIR / "03_fusion" / "output"
CLEANING_OUTPUT_DIR = PIPELINE_DIR / "04_cleaning" / "output"

# Target directory
CONSOLIDATED_DIR = PIPELINE_DIR / "01_sources" / "consolidated"

# Dataset definitions (3 datasets x 3 formats = 9 files)
DATASETS = {
    'MACRO_DAILY_MASTER': {
        'source': CLEANING_OUTPUT_DIR / 'MACRO_DAILY_CLEAN.csv',
        'description': 'Daily macro indicators (workdays only)',
    },
    'MACRO_MONTHLY_MASTER': {
        'source': CLEANING_OUTPUT_DIR / 'MACRO_MONTHLY_CLEAN.csv',
        'description': 'Monthly macro indicators (normalized to 1st of month)',
    },
    'MACRO_QUARTERLY_MASTER': {
        'source': CLEANING_OUTPUT_DIR / 'MACRO_QUARTERLY_CLEAN.csv',
        'description': 'Quarterly macro indicators',
    },
}

# Output formats
FORMATS = ['csv', 'parquet', 'xlsx']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "-" * 50)
    print(title)
    print("-" * 50)


def run_pipeline_step(script_path: Path, description: str) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n[RUNNING] {description}")
    print(f"  Script: {script_path}")

    if not script_path.exists():
        print(f"  [ERROR] Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent),
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout
        )

        if result.returncode == 0:
            print(f"  [OK] {description} completed successfully")
            return True
        else:
            print(f"  [ERROR] {description} failed")
            print(f"  STDERR: {result.stderr[:500] if result.stderr else 'None'}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] {description} timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"  [ERROR] {description} failed: {e}")
        return False


def save_dataframe_multiformat(
    df: pd.DataFrame,
    base_path: Path,
    dataset_name: str
) -> Dict[str, bool]:
    """
    Save DataFrame in multiple formats (CSV, Parquet, Excel).

    Args:
        df: DataFrame to save
        base_path: Directory to save files in
        dataset_name: Base name for the files

    Returns:
        Dict mapping format to success status
    """
    results = {}

    # CSV
    csv_path = base_path / f"{dataset_name}.csv"
    try:
        df.to_csv(csv_path, index=False, encoding='utf-8')
        results['csv'] = True
        print(f"    [OK] {csv_path.name}")
    except Exception as e:
        results['csv'] = False
        print(f"    [ERROR] {csv_path.name}: {e}")

    # Parquet
    parquet_path = base_path / f"{dataset_name}.parquet"
    try:
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        results['parquet'] = True
        print(f"    [OK] {parquet_path.name}")
    except Exception as e:
        results['parquet'] = False
        print(f"    [ERROR] {parquet_path.name}: {e}")

    # Excel
    xlsx_path = base_path / f"{dataset_name}.xlsx"
    try:
        df.to_excel(xlsx_path, index=False, engine='openpyxl')
        results['xlsx'] = True
        print(f"    [OK] {xlsx_path.name}")
    except Exception as e:
        results['xlsx'] = False
        print(f"    [ERROR] {xlsx_path.name}: {e}")

    return results


def validate_dataset(df: pd.DataFrame, name: str) -> Dict:
    """Validate a dataset and return statistics."""
    stats = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns) - 1,  # Exclude 'fecha'
        'date_min': None,
        'date_max': None,
        'nulls_pct': 0,
        'valid': True,
        'issues': []
    }

    if 'fecha' in df.columns:
        stats['date_min'] = str(df['fecha'].min())[:10]
        stats['date_max'] = str(df['fecha'].max())[:10]

    # Calculate null percentage
    data_cols = [c for c in df.columns if c != 'fecha']
    if data_cols:
        total_cells = len(df) * len(data_cols)
        null_cells = df[data_cols].isna().sum().sum()
        stats['nulls_pct'] = round(100 * null_cells / total_cells, 2)

    # Validation checks
    if stats['rows'] == 0:
        stats['valid'] = False
        stats['issues'].append("Empty dataset")

    if stats['nulls_pct'] > 50:
        stats['issues'].append(f"High null percentage: {stats['nulls_pct']}%")

    return stats


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def run_fusion_pipeline() -> bool:
    """Run the fusion pipeline step."""
    print_section("Step 1: Running Fusion Pipeline")
    script_path = PIPELINE_DIR / "03_fusion" / "run_fusion.py"
    return run_pipeline_step(script_path, "Fusion (consolidate raw sources)")


def run_cleaning_pipeline() -> bool:
    """Run the cleaning pipeline step."""
    print_section("Step 2: Running Cleaning Pipeline")
    script_path = PIPELINE_DIR / "04_cleaning" / "run_clean.py"
    return run_pipeline_step(script_path, "Cleaning (normalize + anti-leakage ffill)")


def copy_to_consolidated() -> Tuple[int, int]:
    """
    Copy datasets to consolidated directory in multiple formats.

    Returns:
        Tuple of (success_count, failure_count)
    """
    print_section("Step 3: Copying to Consolidated Directory")

    # Ensure consolidated directory exists
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failure = 0

    for dataset_name, config in DATASETS.items():
        source_path = config['source']
        print(f"\n  Processing {dataset_name}...")

        if not source_path.exists():
            print(f"    [SKIP] Source not found: {source_path}")
            failure += 1
            continue

        try:
            # Read source
            df = pd.read_csv(source_path)

            # Parse dates
            if 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])

            # Save in multiple formats
            results = save_dataframe_multiformat(df, CONSOLIDATED_DIR, dataset_name)

            if all(results.values()):
                success += 1
            else:
                failure += 1

        except Exception as e:
            print(f"    [ERROR] Failed to process: {e}")
            failure += 1

    return success, failure


def validate_consolidated() -> List[Dict]:
    """Validate all consolidated datasets."""
    print_section("Validation: Consolidated Datasets")

    results = []

    for dataset_name, config in DATASETS.items():
        csv_path = CONSOLIDATED_DIR / f"{dataset_name}.csv"

        if not csv_path.exists():
            results.append({
                'name': dataset_name,
                'valid': False,
                'issues': ['File not found']
            })
            continue

        try:
            df = pd.read_csv(csv_path)
            stats = validate_dataset(df, dataset_name)
            results.append(stats)
        except Exception as e:
            results.append({
                'name': dataset_name,
                'valid': False,
                'issues': [str(e)]
            })

    # Print results
    print(f"\n{'Dataset':<30} {'Rows':>8} {'Cols':>6} {'Nulls%':>8} {'Status':<10}")
    print("-" * 70)

    for r in results:
        status = "OK" if r.get('valid', False) else "ISSUES"
        print(
            f"{r['name']:<30} "
            f"{r.get('rows', 0):>8} "
            f"{r.get('columns', 0):>6} "
            f"{r.get('nulls_pct', 0):>7.1f}% "
            f"{status:<10}"
        )
        if r.get('issues'):
            for issue in r['issues']:
                print(f"    ! {issue}")

    return results


def print_summary(fusion_ok: bool, cleaning_ok: bool, success: int, failure: int):
    """Print final summary."""
    print_header("SUMMARY")

    print(f"\nPipeline Steps:")
    print(f"  Fusion:   {'OK' if fusion_ok else 'FAILED'}")
    print(f"  Cleaning: {'OK' if cleaning_ok else 'FAILED'}")

    print(f"\nConsolidated Datasets:")
    print(f"  Success: {success} datasets x 3 formats = {success * 3} files")
    print(f"  Failure: {failure} datasets")

    print(f"\nOutput Directory:")
    print(f"  {CONSOLIDATED_DIR}")

    if CONSOLIDATED_DIR.exists():
        files = list(CONSOLIDATED_DIR.glob("*.*"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"  Total files: {len(files)}")
        print(f"  Total size: {total_size:.2f} MB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate consolidated datasets in CSV, Parquet, and Excel formats"
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing datasets, do not regenerate'
    )
    parser.add_argument(
        '--skip-fusion',
        action='store_true',
        help='Skip fusion step (use existing fusion output)'
    )
    parser.add_argument(
        '--skip-cleaning',
        action='store_true',
        help='Skip cleaning step (use existing cleaning output)'
    )

    args = parser.parse_args()

    print_header("REGENERATE CONSOLIDATED DATASETS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")

    if args.validate_only:
        print("\n[MODE] Validation only (no regeneration)")
        validate_consolidated()
        return

    # Run pipeline steps
    fusion_ok = True
    cleaning_ok = True

    if not args.skip_fusion:
        fusion_ok = run_fusion_pipeline()
    else:
        print("\n[SKIP] Fusion step skipped")

    if not args.skip_cleaning:
        cleaning_ok = run_cleaning_pipeline()
    else:
        print("\n[SKIP] Cleaning step skipped")

    # Copy to consolidated
    success, failure = copy_to_consolidated()

    # Validate
    validate_consolidated()

    # Summary
    print_summary(fusion_ok, cleaning_ok, success, failure)

    print("\n" + "=" * 70)
    print("COMPLETED!")
    print("=" * 70)


if __name__ == '__main__':
    main()
