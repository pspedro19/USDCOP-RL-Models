#!/usr/bin/env python3
"""
Backfill Forecasting Dataset
=============================

Complete backfill script for USDCOP daily forecasting data.
Uses multiple sources with fallback strategy.

Sources (in priority order):
1. yfinance (free, reliable for historical)
2. TwelveData API (if API key available)
3. Investing.com scraping (fallback)

Features generated:
- Price: close, open, high, low
- Returns: return_1d, return_5d, return_10d, return_20d
- Volatility: volatility_5d, volatility_10d, volatility_20d
- Technical: rsi_14d, ma_ratio_20d, ma_ratio_50d
- Calendar: day_of_week, month, is_month_end
- Macro: dxy_close_lag1, oil_close_lag1, vix_close_lag1

@version 1.0.0
@contract CTR-BACKFILL-001
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# DATA SOURCES
# =============================================================================

class YFinanceSource:
    """Primary source: yfinance (free, reliable)."""

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("yfinance not installed. Run: pip install yfinance")

    def fetch(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch USDCOP daily data from Yahoo Finance."""
        if not self.available:
            return None

        try:
            logger.info(f"Fetching USDCOP from yfinance: {start_date} to {end_date}")

            ticker = self.yf.Ticker("COP=X")
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if df.empty:
                logger.warning("yfinance returned empty data")
                return None

            # Standardize columns
            df = df.reset_index()
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Ensure date is datetime
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['source'] = 'yfinance'

            # Select required columns
            cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'source']
            df = df[[c for c in cols if c in df.columns]]

            if 'volume' not in df.columns:
                df['volume'] = 0

            logger.info(f"yfinance: fetched {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"yfinance error: {e}")
            return None


class TwelveDataSource:
    """Secondary source: TwelveData API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TWELVEDATA_API_KEY_1")
        self.available = bool(self.api_key)
        self.base_url = "https://api.twelvedata.com"

        if not self.available:
            logger.warning("TwelveData API key not found")

    def fetch(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch USDCOP daily data from TwelveData."""
        if not self.available:
            return None

        try:
            import requests

            logger.info(f"Fetching USDCOP from TwelveData: {start_date} to {end_date}")

            params = {
                "symbol": "USD/COP",
                "interval": "1day",
                "start_date": start_date,
                "end_date": end_date,
                "apikey": self.api_key,
                "format": "JSON",
                "outputsize": 5000
            }

            response = requests.get(f"{self.base_url}/time_series", params=params, timeout=60)
            data = response.json()

            if "values" not in data:
                logger.warning(f"TwelveData error: {data.get('message', 'Unknown error')}")
                return None

            df = pd.DataFrame(data["values"])
            df['date'] = pd.to_datetime(df['datetime']).dt.date
            df = df.rename(columns={
                'datetime': 'datetime_orig'
            })

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df['volume'] = 0
            df['source'] = 'twelvedata'

            cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'source']
            df = df[cols]
            df = df.sort_values('date').reset_index(drop=True)

            logger.info(f"TwelveData: fetched {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"TwelveData error: {e}")
            return None


class InvestingComSource:
    """Fallback source: Investing.com scraping."""

    def __init__(self):
        try:
            import cloudscraper
            self.scraper = cloudscraper.create_scraper()
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("cloudscraper not installed. Run: pip install cloudscraper")

    def fetch(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch USDCOP daily data from Investing.com (scraping)."""
        if not self.available:
            return None

        try:
            from bs4 import BeautifulSoup
            import time

            logger.info(f"Scraping USDCOP from Investing.com: {start_date} to {end_date}")

            url = "https://www.investing.com/currencies/usd-cop-historical-data"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = self.scraper.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Investing.com returned status {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find historical data table
            table = soup.find('table', {'class': 'datatable-v2_table__93S4Y'})
            if not table:
                table = soup.find('table', {'data-test': 'historical-data-table'})

            if not table:
                logger.warning("Could not find data table on Investing.com")
                return None

            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header
                cells = tr.find_all('td')
                if len(cells) >= 5:
                    try:
                        date_str = cells[0].text.strip()
                        # Parse various date formats
                        for fmt in ['%b %d, %Y', '%m/%d/%Y', '%d/%m/%Y']:
                            try:
                                date = datetime.strptime(date_str, fmt).date()
                                break
                            except ValueError:
                                continue
                        else:
                            continue

                        # Parse prices (handle commas)
                        close = float(cells[1].text.strip().replace(',', ''))
                        open_ = float(cells[2].text.strip().replace(',', ''))
                        high = float(cells[3].text.strip().replace(',', ''))
                        low = float(cells[4].text.strip().replace(',', ''))

                        rows.append({
                            'date': date,
                            'open': open_,
                            'high': high,
                            'low': low,
                            'close': close,
                            'volume': 0,
                            'source': 'investing'
                        })
                    except (ValueError, IndexError) as e:
                        continue

            if not rows:
                logger.warning("No data extracted from Investing.com")
                return None

            df = pd.DataFrame(rows)
            df = df.sort_values('date').reset_index(drop=True)

            # Filter date range
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            df = df[(df['date'] >= start) & (df['date'] <= end)]

            logger.info(f"Investing.com: fetched {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Investing.com scraping error: {e}")
            return None


class MacroDataSource:
    """Load macro data from existing consolidated CSV or database."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.macro_csv = project_root / "data" / "pipeline" / "05_resampling" / "output" / "MACRO_DAILY_CONSOLIDATED.csv"

    def fetch(self) -> Optional[pd.DataFrame]:
        """Load macro indicators."""
        try:
            # Try consolidated CSV first
            if self.macro_csv.exists():
                logger.info(f"Loading macro data from: {self.macro_csv}")
                df = pd.read_csv(self.macro_csv)

                if 'fecha' in df.columns:
                    df['date'] = pd.to_datetime(df['fecha']).dt.date
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date

                logger.info(f"Macro data: {len(df)} records, {len(df.columns)} columns")
                return df

            # Try fetching fresh data via yfinance for key indicators
            logger.info("Fetching macro data from yfinance...")
            return self._fetch_macro_yfinance()

        except Exception as e:
            logger.error(f"Macro data error: {e}")
            return None

    def _fetch_macro_yfinance(self) -> pd.DataFrame:
        """Fetch key macro indicators from yfinance."""
        import yfinance as yf

        symbols = {
            "DX-Y.NYB": "dxy",      # Dollar Index
            "CL=F": "wti",          # WTI Crude Oil
            "^VIX": "vix",          # VIX Volatility
            "MXN=X": "usdmxn",      # USD/MXN
            "GC=F": "gold",         # Gold
        }

        dfs = []
        for symbol, name in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="max", interval="1d")
                if not data.empty:
                    data = data.reset_index()
                    data['date'] = pd.to_datetime(data['Date']).dt.date
                    data = data[['date', 'Close']].rename(columns={'Close': name})
                    dfs.append(data)
            except Exception as e:
                logger.warning(f"Could not fetch {symbol}: {e}")

        if not dfs:
            return pd.DataFrame()

        # Merge all
        df = dfs[0]
        for d in dfs[1:]:
            df = df.merge(d, on='date', how='outer')

        df = df.sort_values('date').reset_index(drop=True)
        return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Generate forecasting features from OHLCV + Macro data."""

    def __init__(self):
        pass

    def generate_features(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Generate all forecasting features."""
        df = ohlcv.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Ensure numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # =================================================================
        # RETURNS
        # =================================================================
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)

        # Log returns for targets
        df['log_return_1d'] = np.log(df['close'] / df['close'].shift(1))

        # =================================================================
        # VOLATILITY
        # =================================================================
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()

        # =================================================================
        # TECHNICAL INDICATORS
        # =================================================================
        # RSI (14-day)
        df['rsi_14d'] = self._calculate_rsi(df['close'], 14)

        # Moving Average Ratios
        df['ma_20d'] = df['close'].rolling(20).mean()
        df['ma_50d'] = df['close'].rolling(50).mean()
        df['ma_ratio_20d'] = df['close'] / df['ma_20d']
        df['ma_ratio_50d'] = df['close'] / df['ma_50d']

        # ATR (14-day)
        df['atr_14d'] = self._calculate_atr(df, 14)

        # =================================================================
        # CALENDAR FEATURES
        # =================================================================
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

        # =================================================================
        # MACRO FEATURES (if available)
        # =================================================================
        if macro is not None and not macro.empty:
            macro = macro.copy()
            macro['date'] = pd.to_datetime(macro['date'])

            # Map column names (handle both upper and lowercase)
            macro_cols = {
                # DXY (Dollar Index)
                'FXRT_INDEX_DXY_USA_D_DXY': 'dxy',
                'fxrt_index_dxy_usa_d_dxy': 'dxy',
                'dxy': 'dxy',
                # WTI Oil
                'COMM_OIL_WTI_GLB_D_WTI': 'wti',
                'comm_oil_wti_glb_d_wti': 'wti',
                'wti': 'wti',
                # VIX
                'VOLT_VIX_USA_D_VIX': 'vix',
                'volt_vix_usa_d_vix': 'vix',
                'vix': 'vix',
                # USDMXN
                'FXRT_SPOT_USDMXN_MEX_D_USDMXN': 'usdmxn',
                'fxrt_spot_usdmxn_mex_d_usdmxn': 'usdmxn',
                'usdmxn': 'usdmxn',
                # Gold
                'COMM_METAL_GOLD_GLB_D_GOLD': 'gold',
                'comm_metal_gold_glb_d_gold': 'gold',
                'gold': 'gold',
                # EMBI
                'CRSK_SPREAD_EMBI_COL_D_EMBI': 'embi',
                'crsk_spread_embi_col_d_embi': 'embi',
                'embi': 'embi',
                # US Treasury 10Y
                'FINC_BOND_YIELD10Y_USA_D_UST10Y': 'ust10y',
                'finc_bond_yield10y_usa_d_ust10y': 'ust10y',
                # Fed Funds
                'POLR_FED_FUNDS_USA_M_FEDFUNDS': 'fedfunds',
                'polr_fed_funds_usa_m_fedfunds': 'fedfunds',
            }

            # Rename to standardized names
            rename_map = {}
            for old_name, new_name in macro_cols.items():
                if old_name in macro.columns and new_name not in rename_map.values():
                    rename_map[old_name] = new_name

            macro = macro.rename(columns=rename_map)

            # Merge
            macro_features = ['date']
            for col in ['dxy', 'wti', 'vix', 'usdmxn', 'gold', 'embi', 'ust10y', 'fedfunds']:
                if col in macro.columns:
                    macro_features.append(col)

            macro_subset = macro[macro_features].drop_duplicates('date')
            df = df.merge(macro_subset, on='date', how='left')

            # Create lagged features (to avoid lookahead bias)
            for col in ['dxy', 'wti', 'vix', 'gold', 'embi', 'usdmxn']:
                if col in df.columns:
                    df[f'{col}_lag1'] = df[col].shift(1)
                    df[f'{col}_return_1d'] = df[col].pct_change(1).shift(1)

            # Additional macro features
            if 'ust10y' in df.columns:
                df['ust10y_lag1'] = df['ust10y'].shift(1)
            if 'fedfunds' in df.columns:
                df['fedfunds_lag1'] = df['fedfunds'].shift(1)

        # =================================================================
        # TARGET VARIABLES (future returns)
        # =================================================================
        for h in [1, 5, 10, 15, 20, 25, 30]:
            # Future price
            df[f'target_{h}d'] = df['close'].shift(-h)
            # Future log return
            df[f'target_return_{h}d'] = np.log(df['close'].shift(-h) / df['close'])

        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI using Wilder's smoothing."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        # Wilder's EMA
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, min_periods=period).mean()
        return atr


# =============================================================================
# DATA QUALITY
# =============================================================================

class DataQualityValidator:
    """Validate dataset quality."""

    def __init__(self):
        self.issues = []

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all quality checks."""
        self.issues = []

        results = {
            'total_rows': len(df),
            'date_range': (df['date'].min(), df['date'].max()),
            'trading_days': len(df),
            'checks': {}
        }

        # 1. Missing values
        null_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        high_null = {k: v for k, v in null_pct.items() if v > 5}
        results['checks']['null_values'] = {
            'status': 'PASS' if not high_null else 'WARN',
            'high_null_columns': high_null
        }

        # 2. Price range validation (COP should be 2500-6000)
        close_range = (df['close'].min(), df['close'].max())
        price_valid = 2500 <= close_range[0] and close_range[1] <= 6000
        results['checks']['price_range'] = {
            'status': 'PASS' if price_valid else 'FAIL',
            'range': close_range
        }

        # 3. Outliers (>4 sigma)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            if col.startswith('target_') or col == 'volume':
                continue
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                n_outliers = ((df[col] - mean).abs() > 4 * std).sum()
                if n_outliers > 0:
                    outliers[col] = n_outliers

        results['checks']['outliers'] = {
            'status': 'PASS' if len(outliers) < 5 else 'WARN',
            'columns_with_outliers': outliers
        }

        # 4. Date continuity
        df_sorted = df.sort_values('date')
        date_gaps = []
        prev_date = None
        for d in df_sorted['date']:
            if prev_date is not None:
                diff = (d - prev_date).days
                if diff > 5:  # More than 5 days gap (weekends + holiday)
                    date_gaps.append((prev_date, d, diff))
            prev_date = d

        results['checks']['date_continuity'] = {
            'status': 'PASS' if len(date_gaps) <= 5 else 'WARN',
            'large_gaps': date_gaps[:10]  # Show first 10
        }

        # 5. Target coverage
        target_cols = [c for c in df.columns if c.startswith('target_')]
        target_coverage = {
            col: (1 - df[col].isnull().sum() / len(df)) * 100
            for col in target_cols
        }
        results['checks']['target_coverage'] = {
            'status': 'PASS' if all(v > 90 for v in target_coverage.values()) else 'WARN',
            'coverage': target_coverage
        }

        # Calculate quality score
        score = 100
        for check in results['checks'].values():
            if check['status'] == 'WARN':
                score -= 5
            elif check['status'] == 'FAIL':
                score -= 20

        results['quality_score'] = max(0, score)
        results['quality_grade'] = (
            'EXCELLENT' if score >= 95 else
            'GOOD' if score >= 85 else
            'ACCEPTABLE' if score >= 70 else
            'POOR'
        )

        return results

    def print_report(self, results: Dict[str, Any]):
        """Print quality report."""
        print("\n" + "=" * 70)
        print("                    DATA QUALITY REPORT")
        print("=" * 70)

        print(f"\nTotal Rows: {results['total_rows']:,}")
        print(f"Date Range: {results['date_range'][0]} to {results['date_range'][1]}")
        print(f"Trading Days: {results['trading_days']:,}")

        print(f"\nQuality Score: {results['quality_score']}/100 ({results['quality_grade']})")

        print("\n" + "-" * 70)
        print("CHECKS:")
        print("-" * 70)

        for check_name, check_result in results['checks'].items():
            status_icon = {
                'PASS': '[OK]',
                'WARN': '[WARN]',
                'FAIL': '[FAIL]'
            }.get(check_result['status'], '?')

            print(f"\n{status_icon} {check_name.upper().replace('_', ' ')}: {check_result['status']}")

            if check_name == 'null_values' and check_result.get('high_null_columns'):
                for col, pct in check_result['high_null_columns'].items():
                    print(f"   - {col}: {pct:.1f}% null")

            elif check_name == 'price_range':
                print(f"   Range: {check_result['range'][0]:.2f} - {check_result['range'][1]:.2f}")

            elif check_name == 'outliers' and check_result.get('columns_with_outliers'):
                for col, n in list(check_result['columns_with_outliers'].items())[:5]:
                    print(f"   - {col}: {n} outliers")

            elif check_name == 'date_continuity' and check_result.get('large_gaps'):
                for start, end, days in check_result['large_gaps'][:3]:
                    print(f"   - Gap: {start} to {end} ({days} days)")

            elif check_name == 'target_coverage':
                for col, cov in check_result['coverage'].items():
                    print(f"   - {col}: {cov:.1f}%")

        print("\n" + "=" * 70)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_backfill(
    start_date: str,
    end_date: str,
    output_dir: Path,
    use_cache: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run complete backfill pipeline.

    Returns:
        Tuple of (dataset DataFrame, quality report dict)
    """

    print("\n" + "=" * 70)
    print("       FORECASTING DATASET BACKFILL PIPELINE")
    print("=" * 70)
    print(f"\nDate Range: {start_date} to {end_date}")
    print(f"Output Dir: {output_dir}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STEP 1: Fetch USDCOP OHLCV
    # =========================================================================
    print("\n[STEP 1/5] Fetching USDCOP Daily OHLCV...")

    cache_path = output_dir / "cache_usdcop_ohlcv.csv"

    ohlcv_df = None

    if use_cache and cache_path.exists():
        logger.info("Loading from cache...")
        ohlcv_df = pd.read_csv(cache_path, parse_dates=['date'])
        ohlcv_df['date'] = pd.to_datetime(ohlcv_df['date']).dt.date
        print(f"   Loaded {len(ohlcv_df)} records from cache")

    if ohlcv_df is None or len(ohlcv_df) == 0:
        # Try sources in order
        sources = [
            ("yfinance", YFinanceSource()),
            ("TwelveData", TwelveDataSource()),
            ("Investing.com", InvestingComSource()),
        ]

        for name, source in sources:
            if not source.available:
                print(f"   {name}: Not available")
                continue

            print(f"   Trying {name}...")
            ohlcv_df = source.fetch(start_date, end_date)

            if ohlcv_df is not None and len(ohlcv_df) > 100:
                print(f"   {name}: SUCCESS - {len(ohlcv_df)} records")
                # Cache it
                ohlcv_df.to_csv(cache_path, index=False)
                break
            else:
                print(f"   {name}: Insufficient data")

        if ohlcv_df is None or len(ohlcv_df) < 100:
            raise RuntimeError("Could not fetch USDCOP data from any source")

    print(f"   Total OHLCV records: {len(ohlcv_df)}")

    # =========================================================================
    # STEP 2: Fetch Macro Data
    # =========================================================================
    print("\n[STEP 2/5] Fetching Macro Indicators...")

    macro_source = MacroDataSource(PROJECT_ROOT)
    macro_df = macro_source.fetch()

    if macro_df is not None:
        print(f"   Macro records: {len(macro_df)}")
        print(f"   Columns: {', '.join(macro_df.columns[:10])}...")
    else:
        print("   WARNING: No macro data available")

    # =========================================================================
    # STEP 3: Generate Features
    # =========================================================================
    print("\n[STEP 3/5] Generating Features...")

    engineer = FeatureEngineer()
    df = engineer.generate_features(ohlcv_df, macro_df)

    print(f"   Features generated: {len(df.columns)} columns")

    # =========================================================================
    # STEP 4: Validate Quality
    # =========================================================================
    print("\n[STEP 4/5] Validating Data Quality...")

    validator = DataQualityValidator()
    quality_report = validator.validate(df)

    validator.print_report(quality_report)

    # =========================================================================
    # STEP 5: Export Dataset
    # =========================================================================
    print("\n[STEP 5/5] Exporting Dataset...")

    # Drop rows with too many nulls in features
    feature_cols = [
        'close', 'open', 'high', 'low',
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        'volatility_5d', 'volatility_10d', 'volatility_20d',
        'rsi_14d', 'ma_ratio_20d', 'ma_ratio_50d',
        'day_of_week', 'month', 'is_month_end'
    ]

    # Add macro features if available
    macro_feature_cols = [
        'dxy_lag1', 'dxy_return_1d',
        'wti_lag1', 'wti_return_1d',
        'vix_lag1', 'vix_return_1d',
        'gold_lag1', 'gold_return_1d',
        'embi_lag1', 'embi_return_1d',
        'usdmxn_lag1', 'usdmxn_return_1d',
        'ust10y_lag1', 'fedfunds_lag1'
    ]
    for col in macro_feature_cols:
        if col in df.columns:
            feature_cols.append(col)

    available_features = [c for c in feature_cols if c in df.columns]
    df_clean = df.dropna(subset=available_features[:10])  # Core features

    # Export paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Full dataset
    full_path = output_dir / f"forecasting_dataset_full_{timestamp}.csv"
    df_clean.to_csv(full_path, index=False)
    print(f"   Full dataset: {full_path}")
    print(f"   Rows: {len(df_clean)}, Columns: {len(df_clean.columns)}")

    # Training-ready dataset (features + targets only)
    target_cols = [c for c in df_clean.columns if c.startswith('target_')]
    train_cols = ['date'] + available_features + target_cols
    train_cols = [c for c in train_cols if c in df_clean.columns]

    df_train = df_clean[train_cols].copy()
    train_path = output_dir / f"forecasting_train_{timestamp}.csv"
    df_train.to_csv(train_path, index=False)
    print(f"   Training dataset: {train_path}")
    print(f"   Rows: {len(df_train)}, Features: {len(available_features)}, Targets: {len(target_cols)}")

    # Also save as parquet for faster loading
    parquet_path = output_dir / f"forecasting_train_{timestamp}.parquet"
    df_train.to_parquet(parquet_path, index=False)
    print(f"   Parquet: {parquet_path}")

    # Summary
    print("\n" + "=" * 70)
    print("                      SUMMARY")
    print("=" * 70)
    print(f"   Date Range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"   Total Trading Days: {len(df_clean)}")
    print(f"   Features: {len(available_features)}")
    print(f"   Target Horizons: {len(target_cols)}")
    print(f"   Quality Score: {quality_report['quality_score']}/100 ({quality_report['quality_grade']})")
    print("=" * 70)

    return df_train, quality_report


def main():
    parser = argparse.ArgumentParser(description="Backfill forecasting dataset")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), defaults to today")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--no-cache", action="store_true", help="Don't use cached data")

    args = parser.parse_args()

    # Defaults
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "data" / "forecasting" / "datasets"

    try:
        df, quality = run_backfill(
            start_date=args.start,
            end_date=end_date,
            output_dir=output_dir,
            use_cache=not args.no_cache
        )

        print("\n[SUCCESS] BACKFILL COMPLETED SUCCESSFULLY!")
        return 0

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
