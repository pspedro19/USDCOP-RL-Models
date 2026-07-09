"""
Dataset Calendar Validation Script
====================================

Standalone script to validate that datasets contain no weekends or holidays.
Can be run before training to ensure data quality.

Usage:
    python scripts/validate_dataset_calendar.py data/pipeline/l4_rl_ready/training_data.csv
    python scripts/validate_dataset_calendar.py --check-all

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-17
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime
import json

# Add airflow dags to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'airflow' / 'dags'))

# Import directly from module to avoid package issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "datetime_handler",
    Path(__file__).parent.parent / 'airflow' / 'dags' / 'utils' / 'datetime_handler.py'
)
datetime_handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(datetime_handler)
UnifiedDatetimeHandler = datetime_handler.UnifiedDatetimeHandler


def validate_dataset(file_path: Path, verbose: bool = True) -> dict:
    """
    Validate a single dataset file for holiday/weekend contamination.

    Args:
        file_path: Path to CSV dataset
        verbose: Print detailed output

    Returns:
        Dictionary with validation results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Validating: {file_path}")
        print(f"{'='*80}")

    # Check file exists
    if not file_path.exists():
        return {
            'file': str(file_path),
            'exists': False,
            'valid': False,
            'error': 'File not found'
        }

    try:
        # Load dataset
        if verbose:
            print(f"Loading dataset...")
        df = pd.read_csv(file_path)

        # Check for timestamp column
        timestamp_col = None
        for col in ['timestamp', 'time', 'datetime', 'date']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            return {
                'file': str(file_path),
                'exists': True,
                'valid': False,
                'error': 'No timestamp column found'
            }

        if verbose:
            print(f"Found timestamp column: {timestamp_col}")
            print(f"Dataset rows: {len(df):,}")

        # Parse timestamps
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Initialize calendar
        calendar = UnifiedDatetimeHandler()
        df = calendar.standardize_dataframe_timestamps(df, [timestamp_col])

        # Check for weekends
        weekdays = df[timestamp_col].dt.dayofweek
        weekend_mask = weekdays >= 5
        weekend_count = weekend_mask.sum()

        # Check for holidays
        business_mask = calendar.is_business_day(df[timestamp_col])
        non_business_count = (~business_mask).sum()

        # Check for premium hours
        premium_mask = calendar.is_premium_hours(df[timestamp_col])
        after_hours_count = (~premium_mask).sum()

        # Get unique dates
        unique_dates = df[timestamp_col].dt.date.unique()
        total_days = len(unique_dates)

        # Results
        is_valid = (weekend_count == 0) and (non_business_count == 0)

        result = {
            'file': str(file_path),
            'exists': True,
            'valid': is_valid,
            'total_rows': len(df),
            'total_unique_days': total_days,
            'weekend_count': int(weekend_count),
            'holiday_count': int(non_business_count),
            'after_hours_count': int(after_hours_count),
            'date_range': {
                'start': str(df[timestamp_col].min()),
                'end': str(df[timestamp_col].max())
            }
        }

        if verbose:
            print(f"\nValidation Results:")
            print(f"  Total Rows: {result['total_rows']:,}")
            print(f"  Unique Days: {result['total_unique_days']}")
            print(f"  Date Range: {result['date_range']['start']} to {result['date_range']['end']}")
            print(f"\n  Weekend Records: {weekend_count:,}")
            print(f"  Holiday Records: {non_business_count:,}")
            print(f"  After-Hours Records: {after_hours_count:,}")

            if is_valid:
                print(f"\n  ✅ VALIDATION PASSED - No weekends or holidays found")
            else:
                print(f"\n  ❌ VALIDATION FAILED - Dataset contains non-trading days")

                # Show contaminated dates
                if non_business_count > 0:
                    contaminated_dates = df[~business_mask][timestamp_col].dt.date.unique()
                    print(f"\n  Contaminated Dates:")
                    for date in sorted(contaminated_dates)[:10]:  # Show first 10
                        day_name = pd.Timestamp(date).day_name()
                        print(f"    - {date} ({day_name})")
                    if len(contaminated_dates) > 10:
                        print(f"    ... and {len(contaminated_dates) - 10} more")

        return result

    except Exception as e:
        return {
            'file': str(file_path),
            'exists': True,
            'valid': False,
            'error': str(e)
        }


def validate_directory(dir_path: Path, verbose: bool = True) -> list:
    """
    Validate all CSV files in a directory.

    Args:
        dir_path: Path to directory
        verbose: Print detailed output

    Returns:
        List of validation results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Scanning directory: {dir_path}")
        print(f"{'='*80}")

    csv_files = list(dir_path.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {dir_path}")
        return []

    if verbose:
        print(f"Found {len(csv_files)} CSV files")

    results = []
    for csv_file in csv_files:
        result = validate_dataset(csv_file, verbose=verbose)
        results.append(result)

    return results


def print_summary(results: list):
    """Print summary of all validation results."""
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*80}")

    total = len(results)
    passed = sum(1 for r in results if r.get('valid', False))
    failed = total - passed

    print(f"\nTotal Files Checked: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")

    if failed > 0:
        print(f"\nFailed Files:")
        for result in results:
            if not result.get('valid', False):
                print(f"  - {result['file']}")
                if 'error' in result:
                    print(f"    Error: {result['error']}")
                else:
                    print(f"    Weekend records: {result.get('weekend_count', 0)}")
                    print(f"    Holiday records: {result.get('holiday_count', 0)}")

    # Overall status
    print(f"\n{'='*80}")
    if failed == 0:
        print(f"✅ ALL DATASETS VALIDATED SUCCESSFULLY")
    else:
        print(f"❌ VALIDATION FAILURES DETECTED - DO NOT TRAIN ON THIS DATA")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate datasets for holiday/weekend contamination'
    )
    parser.add_argument(
        'dataset',
        nargs='?',
        help='Path to CSV dataset or directory to validate'
    )
    parser.add_argument(
        '--check-all',
        action='store_true',
        help='Check all datasets in data/pipeline/l4_rl_ready'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output, show only summary'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    # Determine what to validate
    if args.check_all:
        # Validate all L4 datasets
        l4_path = Path(__file__).parent.parent / 'data' / 'pipeline' / 'l4_rl_ready'
        if not l4_path.exists():
            print(f"Error: L4 directory not found: {l4_path}")
            sys.exit(1)
        results = validate_directory(l4_path, verbose=verbose)

    elif args.dataset:
        dataset_path = Path(args.dataset)

        if dataset_path.is_dir():
            # Validate all CSV files in directory
            results = validate_directory(dataset_path, verbose=verbose)
        elif dataset_path.is_file():
            # Validate single file
            result = validate_dataset(dataset_path, verbose=verbose)
            results = [result]
        else:
            print(f"Error: Invalid path: {dataset_path}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

    # Output results
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

    # Exit with error code if validation failed
    failed_count = sum(1 for r in results if not r.get('valid', False))
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == '__main__':
    main()
