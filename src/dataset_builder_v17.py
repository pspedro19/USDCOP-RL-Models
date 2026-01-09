"""
USD/COP Trading System - Dataset Builder V17 with Colombia-Specific Features
==============================================================================

Adds 4 critical Colombia-specific features to the training dataset:
1. vix_zscore: Rolling Z-score of VIX (volatility regime detection)
2. oil_above_60_flag: Binary flag for Brent > $60 (fiscal stability threshold)
3. usdclp_ret_1d: Lagged daily return of USD/CLP (contagion effects)
4. banrep_intervention_proximity: BanRep intervention proximity signal

These features capture Colombia-specific market dynamics:
- VIX regime changes affect COP as emerging market currency
- Oil > $60 is critical for Colombia's fiscal balance (50% exports)
- USD/CLP shows contagion effects in LatAm FX markets
- BanRep interventions at MA20 ± 5% deviation levels

Author: Pedro @ Lean Tech Solutions
Version: 17.0.0
Date: 2025-12-19
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class ColombiaFeatureBuilder:
    """
    Builder for Colombia-specific features in USD/COP trading dataset.

    This class computes features that capture Colombia's unique market dynamics:
    - Commodity dependence (oil)
    - LatAm FX contagion (CLP)
    - Central bank intervention patterns (BanRep)
    - Emerging market volatility sensitivity (VIX)

    All features are designed to be stationary, clipped, and RL-ready.

    Attributes:
        data_root: Root directory for pipeline data sources
        vix_path: Path to VIX historical data
        brent_path: Path to Brent oil futures data
        usdclp_path: Path to USD/CLP exchange rate data

    Example:
        builder = ColombiaFeatureBuilder()
        df = builder.add_all_colombia_features(ohlcv_df)

        # Verify features were added
        assert 'vix_zscore' in df.columns
        assert 'oil_above_60_flag' in df.columns
        assert 'usdclp_ret_1d' in df.columns
        assert 'banrep_intervention_proximity' in df.columns
    """

    # Feature configuration constants
    VIX_ZSCORE_WINDOW = 20          # Rolling window for VIX z-score
    VIX_ZSCORE_CLIP = 3.0           # Clip extreme values at ±3 sigma

    OIL_THRESHOLD = 60.0            # Critical oil price for Colombia ($60/barrel)

    USDCLP_LAG_DAYS = 1             # Lag to avoid look-ahead bias
    USDCLP_RETURN_CLIP = 0.10       # Clip extreme returns at ±10%

    BANREP_MA_WINDOW = 20           # BanRep uses 20-day MA for interventions
    BANREP_THRESHOLD = 0.05         # ±5% deviation triggers intervention

    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize Colombia feature builder.

        Args:
            data_root: Root directory for data sources. If None, uses default
                      location relative to this module.

        Raises:
            FileNotFoundError: If data root or required files don't exist
        """
        # Set data root (default: repo root / data / pipeline / 01_sources)
        if data_root is None:
            module_dir = Path(__file__).parent.parent
            data_root = module_dir / 'data' / 'pipeline' / '01_sources'

        self.data_root = Path(data_root)

        # Define data source paths
        self.vix_path = self.data_root / '13_volatility' / 'CBOE Volatility Index Historical Data.csv'
        self.brent_path = self.data_root / '01_commodities' / 'Brent Oil Futures Historical Data.csv'
        self.usdclp_path = self.data_root / '02_exchange_rates' / 'fx_usdclp_CHL_d_USDCLP.csv'

        # Validate paths exist
        self._validate_paths()

    def _validate_paths(self) -> None:
        """
        Validate that all required data files exist.

        Raises:
            FileNotFoundError: If any required file is missing
        """
        missing_files = []

        for path_name, path in [
            ('VIX data', self.vix_path),
            ('Brent oil data', self.brent_path),
            ('USD/CLP data', self.usdclp_path)
        ]:
            if not path.exists():
                missing_files.append(f"{path_name}: {path}")

        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files:\n" +
                "\n".join(f"  - {f}" for f in missing_files)
            )

    def _load_investing_csv(self, path: Path, date_col: str = 'Date') -> pd.DataFrame:
        """
        Load CSV file from Investing.com with standard format.

        Investing.com CSVs have format:
        - Date column (MM/DD/YYYY)
        - Price column (closing price)
        - Other columns: Open, High, Low, Vol., Change %

        Args:
            path: Path to CSV file
            date_col: Name of date column (default: 'Date')

        Returns:
            DataFrame with DatetimeIndex and 'price' column

        Raises:
            ValueError: If file cannot be parsed
        """
        try:
            # Read CSV with flexible encoding (handles BOM)
            df = pd.read_csv(path, encoding='utf-8-sig')

            # Parse dates (Investing.com format: MM/DD/YYYY)
            df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
            df = df.set_index(date_col)

            # Extract price column (main closing price)
            if 'Price' in df.columns:
                df['price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce')
            else:
                raise ValueError(f"No 'Price' column found in {path.name}")

            # Sort by date ascending
            df = df.sort_index()

            return df[['price']].copy()

        except Exception as e:
            raise ValueError(f"Failed to load {path.name}: {e}")

    def add_vix_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add VIX z-score feature to DataFrame.

        Formula:
            rolling_mean = vix.rolling(20).mean()
            rolling_std = vix.rolling(20).std() + 1e-8
            vix_zscore = ((vix - rolling_mean) / rolling_std).clip(-3, 3)

        WHY THIS MATTERS:
        - VIX regime changes affect emerging market currencies
        - COP typically weakens when VIX spikes (risk-off flows)
        - Z-score captures relative volatility regime (low/normal/high)

        Args:
            df: DataFrame with DatetimeIndex (5-min bars)

        Returns:
            DataFrame with 'vix_zscore' column added

        Validation:
            - Range: [-3.0, 3.0] (enforced by clipping)
            - NaN handling: Forward-fill first, then zero-fill
        """
        # Load VIX data (daily)
        vix_df = self._load_investing_csv(self.vix_path)

        # Calculate rolling z-score (daily)
        rolling_mean = vix_df['price'].rolling(window=self.VIX_ZSCORE_WINDOW).mean()
        rolling_std = vix_df['price'].rolling(window=self.VIX_ZSCORE_WINDOW).std() + 1e-8
        vix_zscore_daily = ((vix_df['price'] - rolling_mean) / rolling_std).clip(
            -self.VIX_ZSCORE_CLIP,
            self.VIX_ZSCORE_CLIP
        )

        # Resample to 5-min frequency (forward-fill daily values)
        vix_zscore_5min = vix_zscore_daily.reindex(df.index, method='ffill')

        # Handle NaNs (early dates before 20-day window)
        vix_zscore_5min = vix_zscore_5min.fillna(0.0)

        # Add to DataFrame
        df['vix_zscore'] = vix_zscore_5min

        # Validate range
        assert df['vix_zscore'].min() >= -self.VIX_ZSCORE_CLIP, "VIX z-score below minimum clip"
        assert df['vix_zscore'].max() <= self.VIX_ZSCORE_CLIP, "VIX z-score above maximum clip"
        assert df['vix_zscore'].isna().sum() == 0, "VIX z-score contains NaN values"

        print(f"[OK] vix_zscore added: range [{df['vix_zscore'].min():.3f}, {df['vix_zscore'].max():.3f}]")

        return df

    def add_oil_above_60_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add binary flag for Brent oil price > $60.

        Formula:
            oil_above_60_flag = (brent > 60).astype(int)

        WHY THIS MATTERS:
        - Colombia is 50% dependent on oil exports
        - $60/barrel is critical threshold for fiscal stability
        - Above $60: COP strengthens (positive fiscal outlook)
        - Below $60: COP weakens (fiscal concerns)

        Args:
            df: DataFrame with DatetimeIndex (5-min bars)

        Returns:
            DataFrame with 'oil_above_60_flag' column added

        Validation:
            - Range: {0, 1} (binary)
            - NaN handling: Forward-fill first, then zero-fill
        """
        # Load Brent oil data (daily)
        brent_df = self._load_investing_csv(self.brent_path)

        # Calculate binary flag (daily)
        oil_above_60_daily = (brent_df['price'] > self.OIL_THRESHOLD).astype(int)

        # Resample to 5-min frequency (forward-fill daily values)
        oil_above_60_5min = oil_above_60_daily.reindex(df.index, method='ffill')

        # Handle NaNs (fill with 0 = conservative assumption)
        oil_above_60_5min = oil_above_60_5min.fillna(0)

        # Add to DataFrame
        df['oil_above_60_flag'] = oil_above_60_5min

        # Validate range
        assert df['oil_above_60_flag'].isin([0, 1]).all(), "oil_above_60_flag contains non-binary values"
        assert df['oil_above_60_flag'].isna().sum() == 0, "oil_above_60_flag contains NaN values"

        pct_above = df['oil_above_60_flag'].mean() * 100
        print(f"[OK] oil_above_60_flag added: {pct_above:.1f}% of bars with oil > $60")

        return df

    def add_usdclp_ret_1d(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add lagged daily return of USD/CLP exchange rate.

        Formula:
            ret = usdclp.pct_change(1)
            usdclp_ret_1d = ret.shift(1).clip(-0.10, 0.10).fillna(0)

        WHY THIS MATTERS:
        - USD/CLP (Chilean Peso) shows strong contagion with USD/COP
        - Both are commodity-dependent LatAm currencies
        - CLP often leads COP due to higher liquidity
        - Lag avoids look-ahead bias (use yesterday's CLP move)

        Args:
            df: DataFrame with DatetimeIndex (5-min bars)

        Returns:
            DataFrame with 'usdclp_ret_1d' column added

        Validation:
            - Range: [-0.10, 0.10] (enforced by clipping)
            - NaN handling: Zero-fill (no signal = no contagion)
        """
        # Load USD/CLP data (daily)
        usdclp_df = self._load_investing_csv(self.usdclp_path)

        # Calculate daily returns
        usdclp_ret = usdclp_df['price'].pct_change(1)

        # Apply lag (1 day) to avoid look-ahead bias
        usdclp_ret_lagged = usdclp_ret.shift(self.USDCLP_LAG_DAYS)

        # Clip extreme values
        usdclp_ret_clipped = usdclp_ret_lagged.clip(
            -self.USDCLP_RETURN_CLIP,
            self.USDCLP_RETURN_CLIP
        )

        # Resample to 5-min frequency (forward-fill daily values)
        usdclp_ret_5min = usdclp_ret_clipped.reindex(df.index, method='ffill')

        # Handle NaNs (fill with 0 = no signal)
        usdclp_ret_5min = usdclp_ret_5min.fillna(0.0)

        # Add to DataFrame
        df['usdclp_ret_1d'] = usdclp_ret_5min

        # Validate range
        assert df['usdclp_ret_1d'].min() >= -self.USDCLP_RETURN_CLIP, "usdclp_ret_1d below minimum clip"
        assert df['usdclp_ret_1d'].max() <= self.USDCLP_RETURN_CLIP, "usdclp_ret_1d above maximum clip"
        assert df['usdclp_ret_1d'].isna().sum() == 0, "usdclp_ret_1d contains NaN values"

        print(f"[OK] usdclp_ret_1d added: range [{df['usdclp_ret_1d'].min():.4f}, {df['usdclp_ret_1d'].max():.4f}]")

        return df

    def add_banrep_intervention_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add BanRep intervention proximity signal.

        Formula:
            daily_close = df['close'].resample('D').last().ffill()
            ma20_daily = daily_close.rolling(20).mean()
            ma20_5min = ma20_daily.reindex(df.index, method='ffill')
            deviation = (df['close'] - ma20_5min) / (ma20_5min + 1e-8)
            banrep_intervention_proximity = (deviation / 0.05).clip(-1, 1).fillna(0)

        WHY THIS MATTERS:
        - Banco de la República (BanRep) intervenes at MA20 ± 5% levels
        - Interventions stabilize USD/COP, preventing extreme moves
        - Proximity signal helps agent anticipate mean reversion
        - +1 = near upper intervention (expect COP strength)
        - -1 = near lower intervention (expect COP weakness)

        Args:
            df: DataFrame with DatetimeIndex and 'close' column (5-min bars)

        Returns:
            DataFrame with 'banrep_intervention_proximity' column added

        Raises:
            ValueError: If 'close' column is missing

        Validation:
            - Range: [-1.0, 1.0] (enforced by clipping)
            - NaN handling: Zero-fill (no proximity signal)
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for intervention calculation")

        # Step 1: Resample to daily close (last value of each day)
        daily_close = df['close'].resample('D').last().ffill()

        # Step 2: Calculate 20-day moving average (daily)
        ma20_daily = daily_close.rolling(window=self.BANREP_MA_WINDOW).mean()

        # Step 3: Resample MA20 to 5-min frequency (forward-fill)
        ma20_5min = ma20_daily.reindex(df.index, method='ffill')

        # Step 4: Calculate deviation from MA20 (%)
        deviation = (df['close'] - ma20_5min) / (ma20_5min + 1e-8)

        # Step 5: Normalize by intervention threshold (±5%)
        # Result: -1 (at lower bound) to +1 (at upper bound)
        proximity = (deviation / self.BANREP_THRESHOLD).clip(-1.0, 1.0)

        # Step 6: Handle NaNs (early dates before 20-day window)
        proximity = proximity.fillna(0.0)

        # Add to DataFrame
        df['banrep_intervention_proximity'] = proximity

        # Validate range
        assert df['banrep_intervention_proximity'].min() >= -1.0, "BanRep proximity below -1.0"
        assert df['banrep_intervention_proximity'].max() <= 1.0, "BanRep proximity above 1.0"
        assert df['banrep_intervention_proximity'].isna().sum() == 0, "BanRep proximity contains NaN values"

        print(f"[OK] banrep_intervention_proximity added: range [{df['banrep_intervention_proximity'].min():.3f}, {df['banrep_intervention_proximity'].max():.3f}]")

        return df

    def add_all_colombia_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all 4 Colombia-specific features to DataFrame.

        This is the main method to use. It orchestrates the addition of:
        1. vix_zscore
        2. oil_above_60_flag
        3. usdclp_ret_1d
        4. banrep_intervention_proximity

        Args:
            df: DataFrame with DatetimeIndex and 'close' column

        Returns:
            DataFrame with all 4 Colombia features added

        Raises:
            ValueError: If required columns are missing
            FileNotFoundError: If required data files are missing

        Example:
            # Load OHLCV data (5-min bars)
            ohlcv_df = pd.read_parquet('data/ohlcv_5min.parquet')

            # Add Colombia features
            builder = ColombiaFeatureBuilder()
            ohlcv_df = builder.add_all_colombia_features(ohlcv_df)

            # Verify
            print(ohlcv_df[['vix_zscore', 'oil_above_60_flag',
                           'usdclp_ret_1d', 'banrep_intervention_proximity']].describe())
        """
        print("\n" + "="*70)
        print("Adding 4 Colombia-Specific Features to Dataset")
        print("="*70)

        # Validate input
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        initial_shape = df.shape
        print(f"Input shape: {initial_shape}")

        # Add features (order matters for dependencies)
        df = self.add_vix_zscore(df)
        df = self.add_oil_above_60_flag(df)
        df = self.add_usdclp_ret_1d(df)
        df = self.add_banrep_intervention_proximity(df)

        final_shape = df.shape
        print(f"Output shape: {final_shape}")
        print(f"Features added: {final_shape[1] - initial_shape[1]}")

        # Final validation: check all 4 features exist
        required_features = [
            'vix_zscore',
            'oil_above_60_flag',
            'usdclp_ret_1d',
            'banrep_intervention_proximity'
        ]

        missing = [f for f in required_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features after build: {missing}")

        print("\n[OK] All Colombia features successfully added!")
        print("="*70 + "\n")

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for Colombia features.

        Args:
            df: DataFrame with Colombia features

        Returns:
            DataFrame with summary statistics (mean, std, min, max, null count)
        """
        colombia_features = [
            'vix_zscore',
            'oil_above_60_flag',
            'usdclp_ret_1d',
            'banrep_intervention_proximity'
        ]

        # Filter to only features that exist
        existing_features = [f for f in colombia_features if f in df.columns]

        if not existing_features:
            raise ValueError("No Colombia features found in DataFrame")

        # Compute summary statistics
        summary = df[existing_features].describe().T
        summary['null_count'] = df[existing_features].isna().sum()
        summary['null_pct'] = (summary['null_count'] / len(df)) * 100

        return summary


def main():
    """
    Demo script showing how to use ColombiaFeatureBuilder.

    This example:
    1. Creates a mock OHLCV DataFrame
    2. Adds Colombia features
    3. Displays summary statistics
    """
    print("\n" + "="*70)
    print("ColombiaFeatureBuilder Demo")
    print("="*70 + "\n")

    # Create mock OHLCV data (for demonstration)
    # In production, load from actual data source
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='5min')
    np.random.seed(42)

    mock_df = pd.DataFrame({
        'open': 4000 + np.random.randn(len(dates)) * 50,
        'high': 4050 + np.random.randn(len(dates)) * 50,
        'low': 3950 + np.random.randn(len(dates)) * 50,
        'close': 4000 + np.random.randn(len(dates)) * 50,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    print(f"Mock OHLCV data created: {len(mock_df):,} bars")
    print(f"Date range: {mock_df.index.min()} to {mock_df.index.max()}\n")

    # Build Colombia features
    try:
        builder = ColombiaFeatureBuilder()
        mock_df = builder.add_all_colombia_features(mock_df)

        # Display summary
        print("\nFeature Summary Statistics:")
        print("-" * 70)
        summary = builder.get_feature_summary(mock_df)
        print(summary[['mean', 'std', 'min', 'max', 'null_count']])

        # Sample data
        print("\nSample Data (first 5 rows):")
        print("-" * 70)
        colombia_cols = ['vix_zscore', 'oil_above_60_flag', 'usdclp_ret_1d', 'banrep_intervention_proximity']
        print(mock_df[colombia_cols].head())

        print("\n[OK] Demo completed successfully!")

    except FileNotFoundError as e:
        print(f"\n[WARNING] {e}")
        print("This is expected if running outside the main repo directory.")
        print("In production, ensure data files are in the correct location.")


if __name__ == '__main__':
    main()
