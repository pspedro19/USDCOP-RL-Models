# usdcop_forecasting_clean/backend/src/data/loader.py
"""
Data loading with validation.

Follows Single Responsibility Principle - only handles data loading.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import logging

from ..core.config import PipelineConfig
from ..core.exceptions import DataValidationError
from .validator import DataValidator, DataReport

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and prepares data for the regression pipeline.

    Responsibilities:
    - Load CSV files
    - Identify date and price columns
    - Calculate returns
    - Validate data quality
    """

    # Column name candidates
    DATE_COLUMNS = ['date', 'Date', 'DATE', 'fecha', 'Fecha', 'timestamp']
    PRICE_COLUMNS = ['close', 'Close', 'usdcop_close', 'price', 'Price', 'adj_close']

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.validator = DataValidator(self.config)

        # Data containers
        self.df: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.Series] = None
        self.returns: Optional[pd.Series] = None
        self.report: Optional[DataReport] = None

    def load(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            data_path: Path to CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            DataValidationError: If data validation fails
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(f"Loading data from {path}")

        # Load CSV
        df = pd.read_csv(path)

        # Process dates
        date_col = self._find_column(df, self.DATE_COLUMNS)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Find price column
        price_col = self._find_column(df, self.PRICE_COLUMNS)

        # Store data
        self.df = df
        self.prices = df[price_col].copy()
        self.returns = np.log(self.prices / self.prices.shift(1)).dropna()

        # Validate
        self.report = self.validator.validate(self.df, self.prices)

        if self.report.has_critical_issues:
            raise DataValidationError(
                "Data validation failed with critical issues",
                issues=self.report.issues
            )

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> str:
        """Find column from list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col

        # Try first column for dates
        if candidates == self.DATE_COLUMNS:
            return df.columns[0]

        raise ValueError(f"Could not find column. Tried: {candidates}. Available: {df.columns.tolist()}")

    def get_train_test_split(
        self,
        train_size: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test respecting temporal order.

        Args:
            train_size: Proportion for training (0-1)

        Returns:
            Tuple of (train_df, test_df)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        train_size = train_size or self.config.train_size
        n = len(self.df)
        split_idx = int(n * train_size)

        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()

        logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        return train_df, test_df

    def get_prices_split(
        self,
        train_size: float = None
    ) -> Tuple[pd.Series, pd.Series]:
        """Split prices into train/test."""
        train_size = train_size or self.config.train_size
        n = len(self.prices)
        split_idx = int(n * train_size)

        return self.prices.iloc[:split_idx], self.prices.iloc[split_idx:]

    def get_returns_split(
        self,
        train_size: float = None
    ) -> Tuple[pd.Series, pd.Series]:
        """Split returns into train/test."""
        train_size = train_size or self.config.train_size
        n = len(self.returns)
        split_idx = int(n * train_size)

        return self.returns.iloc[:split_idx], self.returns.iloc[split_idx:]

    def print_report(self):
        """Print data quality report."""
        if self.report is None:
            print("No report available. Load data first.")
            return

        self.validator.print_report(self.report)


def load_data(
    data_path: str,
    config: PipelineConfig = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to load data.

    Args:
        data_path: Path to CSV
        config: Pipeline configuration
        verbose: Print report

    Returns:
        Tuple of (df, prices, returns)
    """
    loader = DataLoader(config)
    df = loader.load(data_path)

    if verbose:
        loader.print_report()

    return df, loader.prices, loader.returns
