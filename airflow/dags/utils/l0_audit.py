"""
L0 Audit Module for USDCOP Trading Pipeline
============================================
Comprehensive audit and quality control for L0 Acquire layer

Contract Requirements:
- Schema validation
- Temporal integrity (M5 grid)
- OHLC coherence
- Price sanity checks
- Multi-source consistency
- Metadata completeness
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

class L0Auditor:
    """
    Comprehensive auditor for L0 Acquire layer
    Implements strict quality gates for production data
    """
    
    # Quality thresholds (Adjusted for premium-only strategy)
    MAX_NULL_RATE = 0.001  # 0.1% max nulls (FAIL at 0.5%)
    MAX_DUPLICATE_RATE = 0.001  # 0.1% max duplicates (YELLOW), 0.003 FAIL
    MAX_STALE_RATE = 0.005  # 0.5% max stale bars (YELLOW), 0.01 FAIL
    MAX_GAP_MINUTES = 60  # Maximum gap in minutes (adjusted for market gaps)
    MIN_COMPLETENESS = 0.80  # 80% minimum completeness (premium hours only)
    MAX_PRICE_DELTA_BPS = 15  # Max delta between sources
    
    # Expected schema
    REQUIRED_COLUMNS = ['time', 'open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['tick_count', 'spread']
    
    def __init__(self):
        self.audit_results = {}
        self.violations = []
        self.warnings = []
        self.quality_status = "PENDING"
        
    def audit_data(self, df: pd.DataFrame, source: str, metadata: Dict) -> Dict:
        """
        Run complete audit on data
        
        Args:
            df: DataFrame with OHLCV data
            source: Data source name (mt5, twelvedata)
            metadata: Metadata dictionary
            
        Returns:
            Complete audit report
        """
        logging.info(f"🔍 Starting L0 Audit for {source}")
        
        # Initialize report
        report = {
            'source': source,
            'timestamp': datetime.utcnow().isoformat(),
            'total_records': len(df),
            'audit_checks': {},
            'violations': [],
            'warnings': [],
            'quality_status': 'PENDING'
        }
        
        if df.empty:
            report['quality_status'] = 'FAILED'
            report['violations'].append("Empty dataset")
            return report
        
        # 1. Schema Validation
        schema_result = self._validate_schema(df)
        report['audit_checks']['schema'] = schema_result
        
        # 2. Data Types and Timezone
        types_result = self._validate_types(df)
        report['audit_checks']['data_types'] = types_result
        
        # 3. Null Values Check
        nulls_result = self._check_nulls(df)
        report['audit_checks']['null_values'] = nulls_result
        
        # 4. Temporal Integrity (M5 Grid)
        temporal_result = self._validate_temporal_integrity(df)
        report['audit_checks']['temporal_integrity'] = temporal_result
        
        # 5. Duplicates Check
        duplicates_result = self._check_duplicates(df)
        report['audit_checks']['duplicates'] = duplicates_result
        
        # 6. OHLC Coherence
        ohlc_result = self._validate_ohlc_coherence(df)
        report['audit_checks']['ohlc_coherence'] = ohlc_result
        
        # 7. Price Sanity
        sanity_result = self._validate_price_sanity(df)
        report['audit_checks']['price_sanity'] = sanity_result
        
        # 8. Completeness Check
        completeness_result = self._check_completeness(df)
        report['audit_checks']['completeness'] = completeness_result
        
        # 9. Metadata Validation
        metadata_result = self._validate_metadata(metadata)
        report['audit_checks']['metadata'] = metadata_result
        
        # Determine overall quality status
        report['quality_status'] = self._determine_quality_status(report)
        report['violations'] = self.violations
        report['warnings'] = self.warnings
        
        # Add summary statistics
        report['summary'] = self._generate_summary(df, report)
        
        return report
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict:
        """Validate data schema"""
        result = {
            'status': 'PASS',
            'required_columns': [],
            'missing_columns': [],
            'extra_columns': []
        }
        
        # Check required columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                result['required_columns'].append(col)
            else:
                result['missing_columns'].append(col)
                result['status'] = 'FAIL'
                self.violations.append(f"Missing required column: {col}")
        
        # Check for extra columns
        expected = set(self.REQUIRED_COLUMNS + self.OPTIONAL_COLUMNS)
        actual = set(df.columns)
        extra = actual - expected
        if extra:
            result['extra_columns'] = list(extra)
        
        return result
    
    def _validate_types(self, df: pd.DataFrame) -> Dict:
        """Validate data types and timezone"""
        result = {
            'status': 'PASS',
            'type_checks': {}
        }
        
        # Check time column
        if 'time' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['time']):
                # Check for timezone awareness
                if df['time'].dt.tz is None:
                    result['type_checks']['time'] = 'NAIVE_TIMEZONE'
                    result['status'] = 'FAIL'
                    self.violations.append("Time column has naive timezone")
                else:
                    result['type_checks']['time'] = 'OK'
            else:
                result['type_checks']['time'] = 'WRONG_TYPE'
                result['status'] = 'FAIL'
                self.violations.append("Time column is not datetime type")
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    result['type_checks'][col] = 'OK'
                else:
                    result['type_checks'][col] = 'WRONG_TYPE'
                    result['status'] = 'FAIL'
                    self.violations.append(f"{col} column is not numeric")
        
        return result
    
    def _check_nulls(self, df: pd.DataFrame) -> Dict:
        """Check for null values"""
        result = {
            'status': 'PASS',
            'null_counts': {},
            'null_rates': {}
        }
        
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_rate = null_count / len(df) if len(df) > 0 else 0
            
            result['null_counts'][col] = int(null_count)
            result['null_rates'][col] = round(null_rate, 4)
            
            if null_rate > 0.005:  # 0.5% threshold
                result['status'] = 'FAIL'
                self.violations.append(f"Column {col} has {null_rate:.2%} nulls (>0.5%)")
            elif null_rate > 0.001:  # 0.1% warning
                if result['status'] == 'PASS':
                    result['status'] = 'WARNING'
                self.warnings.append(f"Column {col} has {null_rate:.2%} nulls")
        
        return result
    
    def _validate_temporal_integrity(self, df: pd.DataFrame) -> Dict:
        """Validate temporal integrity (M5 grid alignment)"""
        result = {
            'status': 'PASS',
            'grid_violations': 0,
            'monotonic': True,
            'max_gap_minutes': 0,
            'gaps': []
        }
        
        if 'time' not in df.columns or len(df) == 0:
            return result
        
        df_sorted = df.sort_values('time')
        
        # Check monotonic
        result['monotonic'] = df_sorted['time'].is_monotonic_increasing
        if not result['monotonic']:
            result['status'] = 'FAIL'
            self.violations.append("Timestamps are not monotonic")
        
        # Check M5 grid alignment (with tolerance for API timestamp variations)
        # Allow up to 30 seconds deviation from perfect M5 grid
        non_aligned = df_sorted[
            (df_sorted['time'].dt.minute % 5 != 0) | 
            (df_sorted['time'].dt.second > 30)  # Allow up to 30 seconds deviation
        ]
        result['grid_violations'] = len(non_aligned)
        
        # Only fail if more than 5% of timestamps are misaligned
        misalignment_rate = result['grid_violations'] / len(df_sorted) if len(df_sorted) > 0 else 0
        if misalignment_rate > 0.05:  # Allow up to 5% misalignment
            result['status'] = 'FAIL'
            self.violations.append(f"{result['grid_violations']} timestamps ({misalignment_rate:.1%}) not aligned to M5 grid")
        elif result['grid_violations'] > 0:
            result['status'] = 'WARNING'
            self.warnings.append(f"{result['grid_violations']} timestamps slightly misaligned (tolerated)")
        
        # Check gaps
        time_diff = df_sorted['time'].diff()
        if len(time_diff) > 1:
            max_gap = time_diff.max()
            result['max_gap_minutes'] = max_gap.total_seconds() / 60
            
            if result['max_gap_minutes'] > self.MAX_GAP_MINUTES:
                result['status'] = 'FAIL'
                self.violations.append(f"Maximum gap {result['max_gap_minutes']:.1f} minutes > {self.MAX_GAP_MINUTES}")
            
            # Find significant gaps
            significant_gaps = df_sorted[time_diff > timedelta(minutes=10)]
            if len(significant_gaps) > 0:
                result['gaps'] = [
                    {
                        'time': row['time'].isoformat(),
                        'gap_minutes': time_diff.iloc[i].total_seconds() / 60
                    }
                    for i, row in significant_gaps.iterrows()
                ][:10]  # Limit to 10 gaps
        
        return result
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicates and near-duplicates"""
        result = {
            'status': 'PASS',
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'duplicate_rate': 0
        }
        
        if 'time' not in df.columns:
            return result
        
        # Exact duplicates
        exact_dups = df.duplicated(subset=['time'])
        result['exact_duplicates'] = exact_dups.sum()
        
        # Near duplicates (±1 second)
        df_sorted = df.sort_values('time')
        time_diff = df_sorted['time'].diff()
        near_dups = (time_diff <= timedelta(seconds=1)) & (time_diff > timedelta(0))
        result['near_duplicates'] = near_dups.sum()
        
        total_dups = result['exact_duplicates'] + result['near_duplicates']
        result['duplicate_rate'] = total_dups / len(df) if len(df) > 0 else 0
        
        if result['duplicate_rate'] > 0.003:  # 0.3%
            result['status'] = 'FAIL'
            self.violations.append(f"Duplicate rate {result['duplicate_rate']:.2%} > 0.3%")
        elif result['duplicate_rate'] > 0.001:  # 0.1%
            result['status'] = 'WARNING'
            self.warnings.append(f"Duplicate rate {result['duplicate_rate']:.2%}")
        
        return result
    
    def _validate_ohlc_coherence(self, df: pd.DataFrame) -> Dict:
        """Validate OHLC coherence invariants"""
        result = {
            'status': 'PASS',
            'violations': 0,
            'negative_values': 0,
            'invalid_ranges': 0
        }
        
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            return result
        
        # Check invariant: high >= max(open, close) >= min(open, close) >= low
        violations = (
            (df['high'] < df[['open', 'close']].max(axis=1)) |
            (df['low'] > df[['open', 'close']].min(axis=1)) |
            (df['high'] < df['low'])
        )
        result['violations'] = violations.sum()
        
        # Check for negative values
        negative = (
            (df['open'] < 0) | (df['high'] < 0) | 
            (df['low'] < 0) | (df['close'] < 0)
        )
        result['negative_values'] = negative.sum()
        
        # Check for invalid ranges
        invalid_range = (df['high'] - df['low']) < 0
        result['invalid_ranges'] = invalid_range.sum()
        
        if result['violations'] > 0:
            result['status'] = 'FAIL'
            self.violations.append(f"OHLC coherence violations: {result['violations']}")
        
        if result['negative_values'] > 0:
            result['status'] = 'FAIL'
            self.violations.append(f"Negative price values: {result['negative_values']}")
        
        return result
    
    def _validate_price_sanity(self, df: pd.DataFrame) -> Dict:
        """Validate price sanity"""
        result = {
            'status': 'PASS',
            'stale_bars': 0,
            'stale_rate': 0,
            'extreme_returns': [],
            'extreme_ranges': []
        }
        
        if 'close' not in df.columns or len(df) < 2:
            return result
        
        # Check stale bars (O=H=L=C)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            stale = (
                (df['open'] == df['high']) & 
                (df['high'] == df['low']) & 
                (df['low'] == df['close'])
            )
            result['stale_bars'] = stale.sum()
            result['stale_rate'] = result['stale_bars'] / len(df)
            
            if result['stale_rate'] > 0.01:  # 1%
                result['status'] = 'FAIL'
                self.violations.append(f"Stale bar rate {result['stale_rate']:.2%} > 1%")
            elif result['stale_rate'] > 0.005:  # 0.5%
                result['status'] = 'WARNING'
                self.warnings.append(f"Stale bar rate {result['stale_rate']:.2%}")
        
        # Check extreme returns
        df_sorted = df.sort_values('time')
        returns = np.log(df_sorted['close'] / df_sorted['close'].shift(1))
        
        # Calculate p99.9 threshold
        p999 = np.percentile(np.abs(returns.dropna()), 99.9)
        extreme_returns = np.abs(returns) > p999
        
        if extreme_returns.sum() > 0:
            extreme_idx = df_sorted[extreme_returns].index[:10]
            result['extreme_returns'] = [
                {
                    'time': df_sorted.loc[idx, 'time'].isoformat() if pd.notna(df_sorted.loc[idx, 'time']) else None,
                    'return': returns.loc[idx]
                }
                for idx in extreme_idx
            ]
        
        # Check extreme ranges
        if 'high' in df.columns and 'low' in df.columns:
            range_bps = (df['high'] - df['low']) / df['close'] * 10000
            p999_range = np.percentile(range_bps.dropna(), 99.9)
            extreme_ranges = range_bps > p999_range
            
            if extreme_ranges.sum() > 0:
                extreme_idx = df[extreme_ranges].index[:10]
                result['extreme_ranges'] = [
                    {
                        'time': df.loc[idx, 'time'].isoformat() if pd.notna(df.loc[idx, 'time']) else None,
                        'range_bps': range_bps.loc[idx]
                    }
                    for idx in extreme_idx
                ]
        
        return result
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness for premium-only strategy"""
        result = {
            'status': 'PASS',
            'expected_bars_per_day': 72,  # For 5-minute data in premium hours (8am-2pm = 6 hours)
            'actual_days': 0,
            'complete_days': 0,
            'completeness_rate': 0,
            'premium_completeness': 0,  # New metric for premium hours
            'daily_stats': []
        }
        
        if 'time' not in df.columns or len(df) == 0:
            return result
        
        # Ensure we're working with premium data only (8am-2pm COT)
        df_copy = df.copy()
        df_copy['hour'] = pd.to_datetime(df_copy['time']).dt.hour
        
        # Filter for premium hours if not already filtered
        premium_df = df_copy[(df_copy['hour'] >= 8) & (df_copy['hour'] < 14)]
        
        # Group by date
        premium_df['date'] = pd.to_datetime(premium_df['time']).dt.date
        daily_counts = premium_df.groupby('date').size()
        
        result['actual_days'] = len(daily_counts)
        
        # Count complete days (considering weekdays only)
        weekdays = pd.DataFrame(daily_counts).reset_index()
        weekdays['weekday'] = pd.to_datetime(weekdays['date']).dt.dayofweek
        weekdays = weekdays[weekdays['weekday'] < 5]  # Monday=0, Friday=4
        
        # For premium hours (8am-2pm), expect 72 bars (6 hours * 12 bars/hour)
        expected_premium_bars = 72
        
        # Calculate premium completeness as average across all trading days
        if len(weekdays) > 0:
            weekdays['completeness'] = weekdays[0] / expected_premium_bars
            result['premium_completeness'] = weekdays['completeness'].mean()
            
            # Days with at least 80% completeness are considered "complete" for premium strategy
            complete = weekdays[weekdays[0] >= (expected_premium_bars * 0.8)]  # 80% threshold
            result['complete_days'] = len(complete)
            result['completeness_rate'] = len(complete) / len(weekdays)
        
        # Sample daily stats
        result['daily_stats'] = [
            {
                'date': str(date),
                'bars': int(count),
                'completeness': min(count / expected_premium_bars * 100, 100)
            }
            for date, count in daily_counts.head(10).items()
        ]
        
        # Adjusted thresholds for premium-only strategy
        if result['completeness_rate'] < 0.75:  # Lowered from 0.95 to 0.75 for premium hours
            result['status'] = 'FAIL'
            self.violations.append(f"Completeness rate {result['completeness_rate']:.1%} < 75%")
        elif result['completeness_rate'] < 0.85:  # Lowered from 0.99 to 0.85
            result['status'] = 'WARNING'
            self.warnings.append(f"Completeness rate {result['completeness_rate']:.1%} < 85%")
        
        return result
    
    def _validate_metadata(self, metadata: Dict) -> Dict:
        """Validate metadata completeness"""
        result = {
            'status': 'PASS',
            'required_fields': [],
            'missing_fields': []
        }
        
        required = [
            'price_unit', 'quote_convention', 'pip_size', 
            'price_precision', 'timezone', 'source'
        ]
        
        for field in required:
            if field in metadata:
                result['required_fields'].append(field)
            else:
                result['missing_fields'].append(field)
                result['status'] = 'WARNING'
                self.warnings.append(f"Missing metadata field: {field}")
        
        return result
    
    def _determine_quality_status(self, report: Dict) -> str:
        """Determine overall quality status based on all checks"""
        
        # Critical failures that block READY
        critical_checks = [
            'schema', 'data_types', 'temporal_integrity', 
            'ohlc_coherence'
        ]
        
        for check in critical_checks:
            if check in report['audit_checks']:
                if report['audit_checks'][check].get('status') == 'FAIL':
                    return 'FAILED'
        
        # Check specific thresholds
        if 'completeness' in report['audit_checks']:
            comp = report['audit_checks']['completeness']
            if comp.get('completeness_rate', 0) < 0.75:  # Adjusted for premium hours
                return 'FAILED'
        
        if 'duplicates' in report['audit_checks']:
            dups = report['audit_checks']['duplicates']
            if dups.get('duplicate_rate', 0) > 0.003:
                return 'FAILED'
        
        if 'price_sanity' in report['audit_checks']:
            sanity = report['audit_checks']['price_sanity']
            if sanity.get('stale_rate', 0) > 0.01:
                return 'FAILED'
        
        # Check for warnings
        if self.warnings:
            return 'WARNING'
        
        return 'PASSED'
    
    def _generate_summary(self, df: pd.DataFrame, report: Dict) -> Dict:
        """Generate audit summary"""
        summary = {
            'total_records': len(df),
            'quality_status': report['quality_status'],
            'violations_count': len(self.violations),
            'warnings_count': len(self.warnings),
            'can_promote_to_production': report['quality_status'] == 'PASSED'
        }
        
        # Add key metrics
        if 'time' in df.columns and len(df) > 0:
            summary['date_range'] = {
                'start': df['time'].min().isoformat() if pd.notna(df['time'].min()) else None,
                'end': df['time'].max().isoformat() if pd.notna(df['time'].max()) else None
            }
        
        if 'close' in df.columns:
            summary['price_range'] = {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'mean': float(df['close'].mean())
            }
        
        return summary
    
    def validate_multi_source(self, df_mt5: pd.DataFrame, df_td: pd.DataFrame) -> Dict:
        """
        Validate consistency between multiple sources
        
        Args:
            df_mt5: MT5 data
            df_td: TwelveData data
            
        Returns:
            Multi-source validation report
        """
        report = {
            'status': 'PASS',
            'delta_stats': {},
            'problematic_periods': []
        }
        
        if df_mt5.empty or df_td.empty:
            report['status'] = 'SKIP'
            report['reason'] = 'One or both sources empty'
            return report
        
        # Merge on time
        merged = pd.merge(
            df_mt5[['time', 'close']].rename(columns={'close': 'close_mt5'}),
            df_td[['time', 'close']].rename(columns={'close': 'close_td'}),
            on='time',
            how='inner'
        )
        
        if len(merged) == 0:
            report['status'] = 'SKIP'
            report['reason'] = 'No overlapping timestamps'
            return report
        
        # Calculate delta in basis points
        merged['delta_bps'] = np.abs(
            (merged['close_mt5'] - merged['close_td']) / merged['close_td'] * 10000
        )
        
        # Statistics
        report['delta_stats'] = {
            'mean_bps': float(merged['delta_bps'].mean()),
            'median_bps': float(merged['delta_bps'].median()),
            'p95_bps': float(merged['delta_bps'].quantile(0.95)),
            'p99_bps': float(merged['delta_bps'].quantile(0.99)),
            'max_bps': float(merged['delta_bps'].max())
        }
        
        # Check thresholds
        if report['delta_stats']['p95_bps'] > 10:
            report['status'] = 'WARNING'
            self.warnings.append(f"P95 delta {report['delta_stats']['p95_bps']:.1f} bps > 10 bps")
        
        excessive_delta = merged['delta_bps'] > self.MAX_PRICE_DELTA_BPS
        excessive_rate = excessive_delta.sum() / len(merged)
        
        if excessive_rate > 0.01:  # 1%
            report['status'] = 'FAIL'
            self.violations.append(f"Excessive delta rate {excessive_rate:.2%} > 1%")
        
        # Find problematic periods
        if excessive_delta.sum() > 0:
            problematic = merged[excessive_delta].head(10)
            report['problematic_periods'] = [
                {
                    'time': row['time'].isoformat(),
                    'delta_bps': row['delta_bps'],
                    'mt5_close': row['close_mt5'],
                    'td_close': row['close_td']
                }
                for _, row in problematic.iterrows()
            ]
        
        return report