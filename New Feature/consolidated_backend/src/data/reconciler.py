# backend/src/data/reconciler.py
"""
Data Reconciliation Module.

Handles merging historical CSV data with recent database records,
ensuring data consistency and integrity across different data sources.

Follows Single Responsibility Principle - only handles data reconciliation.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
import shutil
import hashlib

from ..core.config import PipelineConfig, HORIZONS
from ..core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue found during reconciliation.
    """
    date: date
    column: str
    historical_value: Any
    recent_value: Any
    difference: float
    severity: str  # 'warning', 'error', 'critical'

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.date} - {self.column}: "
            f"historical={self.historical_value:.6f}, recent={self.recent_value:.6f}, "
            f"diff={self.difference:.6f}"
        )


@dataclass
class ValidationReport:
    """
    Complete validation report for data reconciliation.
    """
    historical_date_range: Tuple[date, date]
    recent_date_range: Tuple[date, date]
    overlap_dates: List[date]
    n_overlap_rows: int
    columns_matched: bool
    missing_in_historical: List[str] = field(default_factory=list)
    missing_in_recent: List[str] = field(default_factory=list)
    issues: List[ValidationIssue] = field(default_factory=list)
    value_mismatches: int = 0
    avg_price_difference: float = 0.0
    max_price_difference: float = 0.0
    validation_passed: bool = True
    error_message: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.severity == 'critical' for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return any(issue.severity == 'warning' for issue in self.issues)

    @property
    def n_issues(self) -> int:
        """Total number of issues."""
        return len(self.issues)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "DATA RECONCILIATION VALIDATION REPORT",
            "=" * 60,
            f"Generated at: {self.generated_at.isoformat()}",
            "",
            "Date Ranges:",
            f"  Historical: {self.historical_date_range[0]} to {self.historical_date_range[1]}",
            f"  Recent:     {self.recent_date_range[0]} to {self.recent_date_range[1]}",
            f"  Overlap:    {len(self.overlap_dates)} days",
            "",
            "Column Validation:",
            f"  Columns matched: {'YES' if self.columns_matched else 'NO'}",
        ]

        if self.missing_in_historical:
            lines.append(f"  Missing in historical: {', '.join(self.missing_in_historical[:5])}...")
        if self.missing_in_recent:
            lines.append(f"  Missing in recent: {', '.join(self.missing_in_recent[:5])}...")

        lines.extend([
            "",
            "Value Validation:",
            f"  Mismatches found: {self.value_mismatches}",
            f"  Avg price diff: {self.avg_price_difference:.6f}",
            f"  Max price diff: {self.max_price_difference:.6f}",
            "",
            f"Total Issues: {self.n_issues}",
            f"  Critical: {sum(1 for i in self.issues if i.severity == 'critical')}",
            f"  Errors: {sum(1 for i in self.issues if i.severity == 'error')}",
            f"  Warnings: {sum(1 for i in self.issues if i.severity == 'warning')}",
            "",
            f"VALIDATION: {'PASSED' if self.validation_passed else 'FAILED'}",
        ])

        if self.error_message:
            lines.append(f"Error: {self.error_message}")

        lines.append("=" * 60)

        return "\n".join(lines)


class DataReconciler:
    """
    Reconciles historical CSV data with recent database records.

    Responsibilities:
    - Load historical data from CSV
    - Load recent data from PostgreSQL
    - Find overlapping date ranges
    - Validate data consistency in overlaps
    - Merge datasets safely
    - Backup old versions before saving

    Usage:
        reconciler = DataReconciler(config)
        historical = reconciler.load_historical_csv("path/to/COMBINED_V2.csv")
        recent = reconciler.load_recent_from_db(connection, days=30)
        overlap = reconciler.find_overlap_dates(historical, recent)
        report = reconciler.validate_overlap_values(historical, recent, overlap)
        if report.validation_passed:
            merged = reconciler.merge_datasets(historical, recent)
            reconciler.save_reconciled(merged, "path/to/output.csv", backup=True)
    """

    # Column name mappings for normalization
    DATE_COLUMNS = ['date', 'Date', 'DATE', 'fecha', 'Fecha', 'timestamp']
    PRICE_COLUMNS = ['close', 'Close', 'CLOSE', 'close_price', 'usdcop_close', 'price']

    # Tolerance for value comparison (relative difference)
    VALUE_TOLERANCE = 0.0001  # 0.01% tolerance for floating point comparison
    PRICE_TOLERANCE = 0.01   # $0.01 absolute tolerance for prices

    def __init__(self, config: PipelineConfig = None):
        """
        Initialize DataReconciler.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._date_col = None
        self._price_col = None

    def load_historical_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """
        Load historical data from CSV file.

        Args:
            path: Path to the historical CSV file (e.g., COMBINED_V2.csv)

        Returns:
            DataFrame with historical data, date as index

        Raises:
            FileNotFoundError: If file doesn't exist
            DataValidationError: If data format is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Historical CSV not found: {path}")

        logger.info(f"Loading historical data from {path}")

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DataValidationError(f"Failed to parse CSV: {e}")

        # Normalize columns
        df = self._normalize_dataframe(df)

        logger.info(
            f"Loaded historical data: {len(df)} rows, "
            f"date range: {df.index.min().date()} to {df.index.max().date()}"
        )

        return df

    def load_recent_from_db(
        self,
        connection,
        days: int = 30,
        table: str = 'core.features_ml'
    ) -> pd.DataFrame:
        """
        Load recent data from PostgreSQL database.

        Args:
            connection: Database connection (psycopg2 or SQLAlchemy)
            days: Number of days to load (default 30)
            table: Table name to query from

        Returns:
            DataFrame with recent data, date as index

        Raises:
            DataValidationError: If query fails or data is invalid
        """
        logger.info(f"Loading recent {days} days from {table}")

        query = f"""
            SELECT *
            FROM {table}
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY date ASC
        """

        try:
            df = pd.read_sql(query, connection)
        except Exception as e:
            raise DataValidationError(f"Failed to query database: {e}")

        if df.empty:
            logger.warning(f"No recent data found in last {days} days")
            return pd.DataFrame()

        # Normalize columns
        df = self._normalize_dataframe(df)

        logger.info(
            f"Loaded recent data: {len(df)} rows, "
            f"date range: {df.index.min().date()} to {df.index.max().date()}"
        )

        return df

    def find_overlap_dates(
        self,
        historical: pd.DataFrame,
        recent: pd.DataFrame
    ) -> List[date]:
        """
        Find dates that exist in both historical and recent datasets.

        Args:
            historical: Historical DataFrame with DatetimeIndex
            recent: Recent DataFrame with DatetimeIndex

        Returns:
            List of dates that appear in both datasets
        """
        if historical.empty or recent.empty:
            logger.warning("One or both DataFrames are empty, no overlap possible")
            return []

        # Get date sets
        historical_dates = set(historical.index.date)
        recent_dates = set(recent.index.date)

        # Find intersection
        overlap = sorted(historical_dates & recent_dates)

        logger.info(
            f"Found {len(overlap)} overlapping dates "
            f"({overlap[0] if overlap else 'N/A'} to {overlap[-1] if overlap else 'N/A'})"
        )

        return overlap

    def validate_overlap_values(
        self,
        historical: pd.DataFrame,
        recent: pd.DataFrame,
        overlap_dates: List[date],
        key_columns: List[str] = None
    ) -> ValidationReport:
        """
        Validate that values in overlapping dates match between datasets.

        Args:
            historical: Historical DataFrame
            recent: Recent DataFrame
            overlap_dates: List of dates to validate
            key_columns: Specific columns to validate (default: all common columns)

        Returns:
            ValidationReport with detailed validation results
        """
        issues = []

        # Get date ranges
        hist_range = (historical.index.min().date(), historical.index.max().date())
        recent_range = (recent.index.min().date(), recent.index.max().date())

        # Check column alignment
        hist_cols = set(historical.columns)
        recent_cols = set(recent.columns)

        missing_in_historical = list(recent_cols - hist_cols)
        missing_in_recent = list(hist_cols - recent_cols)
        common_cols = list(hist_cols & recent_cols)

        columns_matched = len(missing_in_historical) == 0 and len(missing_in_recent) == 0

        if missing_in_historical:
            logger.warning(f"Columns in recent but not historical: {missing_in_historical}")
        if missing_in_recent:
            logger.warning(f"Columns in historical but not recent: {missing_in_recent}")

        # Determine which columns to validate
        if key_columns:
            validate_cols = [c for c in key_columns if c in common_cols]
        else:
            validate_cols = common_cols

        # Identify price-like columns for stricter validation
        price_cols = [c for c in validate_cols if any(
            p in c.lower() for p in ['close', 'open', 'high', 'low', 'price']
        )]

        value_mismatches = 0
        price_differences = []

        # Validate each overlap date
        for dt in overlap_dates:
            # Get rows for this date
            hist_mask = historical.index.date == dt
            recent_mask = recent.index.date == dt

            hist_row = historical[hist_mask]
            recent_row = recent[recent_mask]

            if hist_row.empty or recent_row.empty:
                continue

            # Compare values in each column
            for col in validate_cols:
                try:
                    hist_val = hist_row[col].iloc[0]
                    recent_val = recent_row[col].iloc[0]

                    # Skip if both are NaN
                    if pd.isna(hist_val) and pd.isna(recent_val):
                        continue

                    # Handle NaN mismatch
                    if pd.isna(hist_val) != pd.isna(recent_val):
                        issues.append(ValidationIssue(
                            date=dt,
                            column=col,
                            historical_value=hist_val,
                            recent_value=recent_val,
                            difference=float('nan'),
                            severity='warning'
                        ))
                        value_mismatches += 1
                        continue

                    # Calculate difference
                    diff = abs(float(hist_val) - float(recent_val))
                    rel_diff = diff / abs(float(hist_val)) if hist_val != 0 else diff

                    # Check tolerance
                    is_price_col = col in price_cols
                    tolerance = self.PRICE_TOLERANCE if is_price_col else self.VALUE_TOLERANCE

                    if is_price_col:
                        price_differences.append(diff)

                    if diff > tolerance and rel_diff > self.VALUE_TOLERANCE:
                        severity = 'critical' if is_price_col and diff > 1.0 else 'error'
                        issues.append(ValidationIssue(
                            date=dt,
                            column=col,
                            historical_value=float(hist_val),
                            recent_value=float(recent_val),
                            difference=diff,
                            severity=severity
                        ))
                        value_mismatches += 1

                except (TypeError, ValueError) as e:
                    # Non-numeric comparison, skip
                    continue

        # Calculate summary statistics
        avg_price_diff = np.mean(price_differences) if price_differences else 0.0
        max_price_diff = np.max(price_differences) if price_differences else 0.0

        # Determine if validation passed
        critical_count = sum(1 for i in issues if i.severity == 'critical')
        error_count = sum(1 for i in issues if i.severity == 'error')

        validation_passed = critical_count == 0 and error_count < 10

        error_message = None
        if not validation_passed:
            error_message = f"Found {critical_count} critical and {error_count} error issues"

        report = ValidationReport(
            historical_date_range=hist_range,
            recent_date_range=recent_range,
            overlap_dates=overlap_dates,
            n_overlap_rows=len(overlap_dates),
            columns_matched=columns_matched,
            missing_in_historical=missing_in_historical,
            missing_in_recent=missing_in_recent,
            issues=issues,
            value_mismatches=value_mismatches,
            avg_price_difference=avg_price_diff,
            max_price_difference=max_price_diff,
            validation_passed=validation_passed,
            error_message=error_message
        )

        logger.info(f"Validation complete: {'PASSED' if validation_passed else 'FAILED'}")

        return report

    def merge_datasets(
        self,
        historical: pd.DataFrame,
        recent: pd.DataFrame,
        prefer_recent: bool = True,
        fill_missing: bool = True
    ) -> pd.DataFrame:
        """
        Merge historical and recent datasets into a single DataFrame.

        Args:
            historical: Historical DataFrame
            recent: Recent DataFrame
            prefer_recent: If True, use recent values for overlapping dates
            fill_missing: If True, forward-fill missing values after merge

        Returns:
            Merged DataFrame with all dates and columns
        """
        if historical.empty:
            logger.warning("Historical DataFrame is empty, returning recent only")
            return recent.copy()

        if recent.empty:
            logger.warning("Recent DataFrame is empty, returning historical only")
            return historical.copy()

        logger.info("Merging historical and recent datasets...")

        # Get all dates
        all_dates = sorted(set(historical.index) | set(recent.index))

        # Get all columns
        all_columns = list(dict.fromkeys(
            list(historical.columns) + list(recent.columns)
        ))

        # Create merged DataFrame
        merged = pd.DataFrame(index=all_dates, columns=all_columns)
        merged.index.name = 'date'

        # Fill with historical first
        for col in historical.columns:
            merged.loc[historical.index, col] = historical[col].values

        # Then overlay with recent (if prefer_recent) or fill gaps
        for col in recent.columns:
            if prefer_recent:
                # Overwrite with recent values
                merged.loc[recent.index, col] = recent[col].values
            else:
                # Only fill where historical is NaN
                mask = merged.loc[recent.index, col].isna()
                merged.loc[recent.index[mask], col] = recent.loc[mask, col].values

        # Sort by date
        merged = merged.sort_index()

        # Forward-fill missing values if requested
        if fill_missing:
            merged = merged.ffill()

        # Convert columns to appropriate dtypes
        for col in merged.columns:
            try:
                merged[col] = pd.to_numeric(merged[col], errors='ignore')
            except:
                pass

        logger.info(
            f"Merged dataset: {len(merged)} rows, "
            f"date range: {merged.index.min().date()} to {merged.index.max().date()}"
        )

        return merged

    def save_reconciled(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        backup: bool = True,
        backup_dir: Union[str, Path] = None
    ) -> str:
        """
        Save reconciled DataFrame to CSV with optional backup.

        Args:
            df: DataFrame to save
            output_path: Path to save the reconciled data
            backup: If True, backup existing file before overwriting
            backup_dir: Directory for backups (default: same directory as output)

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)

        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup existing file if requested
        if backup and output_path.exists():
            backup_dir = Path(backup_dir) if backup_dir else output_path.parent / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate backup filename with timestamp and hash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_hash = self._get_file_hash(output_path)[:8]
            backup_name = f"{output_path.stem}_{timestamp}_{file_hash}{output_path.suffix}"
            backup_path = backup_dir / backup_name

            shutil.copy2(output_path, backup_path)
            logger.info(f"Backed up existing file to: {backup_path}")

        # Reset index to include date as column
        df_to_save = df.reset_index()

        # Save to CSV
        df_to_save.to_csv(output_path, index=False)

        logger.info(f"Saved reconciled data to: {output_path} ({len(df)} rows)")

        return str(output_path)

    def regenerate_ml_features(
        self,
        df: pd.DataFrame,
        output_path: Union[str, Path],
        horizons: List[int] = None
    ) -> pd.DataFrame:
        """
        Regenerate ML feature targets (forward returns) for the merged dataset.

        Args:
            df: Merged DataFrame with price data
            output_path: Path to save the regenerated features
            horizons: List of prediction horizons (default: from config)

        Returns:
            DataFrame with regenerated target columns
        """
        horizons = horizons or self.config.horizons

        logger.info(f"Regenerating ML features for horizons: {horizons}")

        df = df.copy()

        # Find price column
        price_col = self._find_price_column(df)
        if price_col is None:
            raise DataValidationError("Could not find price column for target calculation")

        prices = df[price_col]

        # Generate target columns (forward log returns)
        for h in horizons:
            target_col = f'target_{h}d'
            df[target_col] = np.log(prices.shift(-h) / prices)
            logger.debug(f"Generated {target_col}: {df[target_col].notna().sum()} valid values")

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df_to_save = df.reset_index()
        df_to_save.to_csv(output_path, index=False)

        logger.info(f"Saved ML features to: {output_path}")

        return df

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame: identify date column, set as index, sort.
        """
        # Find and normalize date column
        date_col = self._find_date_column(df)

        if date_col is None:
            raise DataValidationError(
                f"Could not find date column. Available: {df.columns.tolist()}"
            )

        # Convert to datetime and set as index
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        df.index.name = 'date'

        # Store price column for later use
        self._price_col = self._find_price_column(df)

        return df

    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the date column in DataFrame."""
        for col in self.DATE_COLUMNS:
            if col in df.columns:
                return col
        # Try first column if it looks like a date
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col].iloc[0])
            return first_col
        except:
            pass
        return None

    def _find_price_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the price column in DataFrame."""
        for col in self.PRICE_COLUMNS:
            if col in df.columns:
                return col
        return None

    def _get_file_hash(self, path: Path) -> str:
        """Calculate MD5 hash of file for backup naming."""
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


def reconcile_data(
    historical_path: str,
    db_connection,
    output_path: str,
    days: int = 30,
    config: PipelineConfig = None
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Convenience function to perform full data reconciliation.

    Args:
        historical_path: Path to historical CSV
        db_connection: Database connection
        output_path: Path to save reconciled data
        days: Days of recent data to load
        config: Pipeline configuration

    Returns:
        Tuple of (merged DataFrame, validation report)
    """
    reconciler = DataReconciler(config)

    # Load data
    historical = reconciler.load_historical_csv(historical_path)
    recent = reconciler.load_recent_from_db(db_connection, days=days)

    # Find overlap and validate
    overlap = reconciler.find_overlap_dates(historical, recent)
    report = reconciler.validate_overlap_values(historical, recent, overlap)

    if not report.validation_passed:
        logger.warning(f"Validation failed: {report.error_message}")
        print(report.summary())

    # Merge and save
    merged = reconciler.merge_datasets(historical, recent)
    reconciler.save_reconciled(merged, output_path, backup=True)

    return merged, report
