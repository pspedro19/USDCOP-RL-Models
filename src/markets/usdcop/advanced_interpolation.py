"""
Advanced Interpolation Techniques for USDCOP M5 Data
=====================================================
Implements Brownian Bridge, Gaussian Process, and other advanced methods
for high-quality financial time series interpolation.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import norm
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available for Gaussian Process")


class AdvancedInterpolator:
    """Advanced interpolation methods for financial time series"""
    
    def __init__(self, method: str = 'brownian_bridge'):
        """
        Initialize interpolator
        
        Args:
            method: 'brownian_bridge', 'gaussian_process', 'cubic_spline', 'akima'
        """
        self.method = method
        self.volatility_cache = {}
        
    def brownian_bridge_interpolation(self, 
                                     df: pd.DataFrame,
                                     start_idx: int,
                                     end_idx: int,
                                     n_points: int) -> pd.DataFrame:
        """
        Brownian Bridge interpolation for missing price data
        
        Generates realistic price paths between two known points using
        Brownian Bridge process which ensures continuity at boundaries.
        """
        
        # Get boundary prices
        start_price = df.iloc[start_idx]['close']
        end_price = df.iloc[end_idx]['close']
        
        # Calculate historical volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Time points
        t = np.linspace(0, 1, n_points + 2)[1:-1]  # Exclude boundaries
        
        # Generate Brownian Bridge
        interpolated_prices = []
        
        for ti in t:
            # Brownian Bridge formula: B(t) = B(0) + t*(B(1) - B(0)) + W(t) - t*W(1)
            # Where W(t) is standard Brownian motion
            
            # Mean of the bridge at time t
            mean_t = start_price + ti * (end_price - start_price)
            
            # Variance of the bridge at time t
            var_t = volatility * np.sqrt(ti * (1 - ti))
            
            # Sample from the distribution
            price_t = np.random.normal(mean_t, var_t)
            interpolated_prices.append(price_t)
        
        # Create OHLC data
        interpolated_data = []
        
        for i, price in enumerate(interpolated_prices):
            # Add realistic OHLC variation
            variation = volatility * 0.1
            
            open_price = price + np.random.normal(0, variation)
            close_price = price + np.random.normal(0, variation)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, variation))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, variation))
            
            interpolated_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
        
        return pd.DataFrame(interpolated_data)
    
    def gaussian_process_interpolation(self,
                                      df: pd.DataFrame,
                                      gap_start: int,
                                      gap_end: int,
                                      n_points: int) -> pd.DataFrame:
        """
        Gaussian Process interpolation for sophisticated price prediction
        
        Uses GP regression to model the price process and generate
        interpolated values with uncertainty quantification.
        """
        
        if not HAS_SKLEARN:
            print("Falling back to Brownian Bridge (sklearn not available)")
            return self.brownian_bridge_interpolation(df, gap_start, gap_end, n_points)
        
        # Prepare training data (use surrounding points)
        context_size = min(50, len(df) // 10)  # Use 50 points or 10% of data
        
        train_start = max(0, gap_start - context_size)
        train_end = min(len(df), gap_end + context_size)
        
        # Extract features (time index) and targets (prices)
        train_indices = list(range(train_start, gap_start)) + list(range(gap_end, train_end))
        X_train = np.array(train_indices).reshape(-1, 1)
        
        # Use log prices for better GP behavior
        y_train = np.log(df.iloc[train_indices]['close'].values)
        
        # Define GP kernel (Matern for financial data)
        kernel = Matern(length_scale=10, nu=1.5) + WhiteKernel(noise_level=0.01)
        
        # Fit Gaussian Process
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-6)
        gp.fit(X_train, y_train)
        
        # Predict for gap points
        X_pred = np.arange(gap_start, gap_end).reshape(-1, 1)
        y_pred, y_std = gp.predict(X_pred, return_std=True)
        
        # Convert back from log space
        predicted_prices = np.exp(y_pred)
        price_std = np.exp(y_std)
        
        # Generate OHLC with uncertainty
        interpolated_data = []
        
        for i, (price, std) in enumerate(zip(predicted_prices, price_std)):
            # Sample around predicted value
            sampled_price = np.random.normal(price, std * 0.1)
            
            # Generate realistic OHLC
            variation = std * 0.05
            open_price = sampled_price + np.random.normal(0, variation)
            close_price = sampled_price + np.random.normal(0, variation)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, variation))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, variation))
            
            interpolated_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
        
        return pd.DataFrame(interpolated_data)
    
    def cubic_hermite_spline(self,
                            df: pd.DataFrame,
                            gap_start: int,
                            gap_end: int,
                            n_points: int) -> pd.DataFrame:
        """
        Cubic Hermite Spline interpolation preserving monotonicity
        
        Ensures smooth transitions while respecting local trends.
        """
        
        # Get context points
        context_before = max(0, gap_start - 5)
        context_after = min(len(df), gap_end + 5)
        
        # Extract known points
        known_indices = list(range(context_before, gap_start)) + list(range(gap_end, context_after))
        known_prices = df.iloc[known_indices]['close'].values
        
        # Create interpolator
        cs = interpolate.PchipInterpolator(known_indices, known_prices)
        
        # Interpolate gap
        gap_indices = np.linspace(gap_start, gap_end - 1, n_points)
        interpolated_prices = cs(gap_indices)
        
        # Add OHLC variation
        interpolated_data = []
        volatility = df['close'].pct_change().std()
        
        for price in interpolated_prices:
            variation = volatility * price * 0.001
            
            open_price = price + np.random.normal(0, variation)
            close_price = price + np.random.normal(0, variation)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, variation))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, variation))
            
            interpolated_data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
        
        return pd.DataFrame(interpolated_data)
    
    def fill_gaps(self, df: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        Fill all gaps in the dataframe using specified method
        
        Args:
            df: DataFrame with time index
            method: Override default method
            
        Returns:
            DataFrame with filled gaps
        """
        
        if method:
            self.method = method
        
        # Make a copy
        df_filled = df.copy()
        
        # Ensure datetime index
        if 'time' in df_filled.columns:
            df_filled.set_index('time', inplace=True)
        
        # Create complete time index (5-minute intervals)
        full_index = pd.date_range(
            start=df_filled.index.min(),
            end=df_filled.index.max(),
            freq='5T'
        )
        
        # Reindex to identify gaps
        df_reindexed = df_filled.reindex(full_index)
        
        # Find gap segments
        is_gap = df_reindexed['close'].isna()
        gap_groups = (is_gap != is_gap.shift()).cumsum()
        gaps = df_reindexed[is_gap].groupby(gap_groups).apply(
            lambda x: (x.index[0], x.index[-1], len(x))
        )
        
        print(f"Found {len(gaps)} gaps to fill using {self.method}")
        
        # Fill each gap
        for gap_id, (start_time, end_time, gap_size) in enumerate(gaps):
            if gap_size == 0:
                continue
                
            # Find indices in original data
            try:
                start_idx = df_filled.index.get_indexer([start_time], method='ffill')[0]
                end_idx = df_filled.index.get_indexer([end_time], method='bfill')[0]
            except:
                # Fallback for newer pandas versions
                start_idx = df_filled.index.get_indexer([start_time])[0]
                if start_idx == -1:
                    start_idx = 0
                end_idx = df_filled.index.get_indexer([end_time])[0]
                if end_idx == -1:
                    end_idx = len(df_filled) - 1
            
            # Choose interpolation method
            if self.method == 'brownian_bridge':
                interpolated = self.brownian_bridge_interpolation(
                    df_filled, start_idx, end_idx, gap_size
                )
            elif self.method == 'gaussian_process':
                interpolated = self.gaussian_process_interpolation(
                    df_filled, start_idx, end_idx, gap_size
                )
            elif self.method == 'cubic_spline':
                interpolated = self.cubic_hermite_spline(
                    df_filled, start_idx, end_idx, gap_size
                )
            else:
                # Fallback to linear
                interpolated = df_filled.iloc[start_idx:end_idx].interpolate(method='linear')
            
            # Fill the gap
            gap_times = pd.date_range(start_time, end_time, periods=gap_size)
            for i, time in enumerate(gap_times):
                if i < len(interpolated):
                    df_reindexed.loc[time] = interpolated.iloc[i]
        
        # Fill remaining columns
        df_reindexed['tick_volume'] = df_reindexed['tick_volume'].fillna(
            df_reindexed['tick_volume'].mean()
        )
        df_reindexed['spread'] = df_reindexed['spread'].fillna(
            df_reindexed['spread'].mean()
        )
        df_reindexed['real_volume'] = df_reindexed['real_volume'].fillna(0)
        
        return df_reindexed
    
    def validate_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLC relationships are valid"""
        
        df = df.copy()
        
        # Fix high to be maximum
        df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
        
        # Fix low to be minimum
        df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        return df


def create_high_quality_m5_advanced(
    input_file: Optional[str] = None,
    output_dir: str = "data/processed/final_m5"
) -> str:
    """
    Create high-quality M5 data using advanced interpolation
    
    Returns:
        Path to the output file
    """
    
    print("=" * 80)
    print("ADVANCED M5 DATA GENERATION WITH BROWNIAN BRIDGE & GAUSSIAN PROCESS")
    print("=" * 80)
    
    # Load existing M5 data
    if not input_file:
        # Find latest M5 file
        from pathlib import Path
        m5_files = list(Path("data/processed/high_quality_m5").glob("USDCOP_M5_*.csv"))
        if not m5_files:
            m5_files = list(Path("data/raw").rglob("*M5*.csv"))
        
        if not m5_files:
            raise FileNotFoundError("No M5 files found")
        
        input_file = str(max(m5_files, key=lambda x: x.stat().st_mtime))
    
    print(f"\nLoading: {input_file}")
    
    df = pd.read_csv(input_file)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    print(f"Original data: {len(df)} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Apply advanced interpolation
    print("\n" + "=" * 40)
    print("APPLYING ADVANCED INTERPOLATION")
    print("=" * 40)
    
    # Try Gaussian Process first, fallback to Brownian Bridge
    interpolator = AdvancedInterpolator(method='gaussian_process')
    
    print("\n1. Gaussian Process interpolation...")
    try:
        df_gp = interpolator.fill_gaps(df.copy(), method='gaussian_process')
        print(f"   OK: Filled to {len(df_gp)} bars")
    except Exception as e:
        print(f"   X GP failed: {e}, using Brownian Bridge")
        df_gp = None
    
    print("\n2. Brownian Bridge interpolation...")
    interpolator.method = 'brownian_bridge'
    df_bb = interpolator.fill_gaps(df.copy(), method='brownian_bridge')
    print(f"   OK: Filled to {len(df_bb)} bars")
    
    # Use best result
    df_final = df_gp if df_gp is not None and len(df_gp) > len(df_bb) else df_bb
    
    # Validate OHLC consistency
    print("\n3. Validating OHLC consistency...")
    df_final = interpolator.validate_ohlc_consistency(df_final)
    print("   OK: OHLC relationships validated")
    
    # Calculate quality metrics
    print("\n" + "=" * 40)
    print("QUALITY METRICS")
    print("=" * 40)
    
    total_minutes = (df_final.index.max() - df_final.index.min()).total_seconds() / 60
    expected_bars = int(total_minutes / 5)
    completeness = (len(df_final) / expected_bars * 100) if expected_bars > 0 else 0
    
    print(f"Final bars: {len(df_final):,}")
    print(f"Expected bars: {expected_bars:,}")
    print(f"Completeness: {completeness:.2f}%")
    print(f"Missing values: {df_final.isna().sum().sum()}")
    
    # Save final data
    from pathlib import Path
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"USDCOP_M5_FINAL_{timestamp}.csv"
    
    df_final.reset_index().to_csv(output_file, index=False)
    
    print(f"\n[SAVED]: {output_file}")
    print(f"   Size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    
    return str(output_file)


if __name__ == "__main__":
    output = create_high_quality_m5_advanced()