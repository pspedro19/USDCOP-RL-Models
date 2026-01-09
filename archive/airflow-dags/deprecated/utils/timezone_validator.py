"""
Timezone Validator for USD/COP Trading Pipeline
===============================================
Ensures proper timezone handling throughout the data pipeline
"""

import pandas as pd
import pytz
from datetime import datetime
import logging
from typing import Optional, Tuple, Dict

class TimezoneValidator:
    """
    Validates and ensures consistent timezone handling
    
    Critical for ensuring premium hours (8am-2pm COT) are correctly identified
    """
    
    # Timezone definitions
    COT_TZ = 'America/Bogota'  # Colombia Time (UTC-5)
    UTC_TZ = 'UTC'
    MT5_TZ = 'Etc/GMT-2'  # MT5 Server time (UTC+2)
    
    # Premium hours in COT
    PREMIUM_START_HOUR = 8   # 8:00 AM COT
    PREMIUM_END_HOUR = 14    # 2:00 PM COT
    
    @staticmethod
    def validate_dataframe_timezone(df: pd.DataFrame, 
                                   timestamp_col: str = 'timestamp') -> Dict:
        """
        Validate timezone consistency in dataframe
        
        Args:
            df: DataFrame to validate
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': False,
            'has_timezone': False,
            'timezone': None,
            'is_cot': False,
            'needs_conversion': False,
            'errors': []
        }
        
        if timestamp_col not in df.columns:
            result['errors'].append(f"Column '{timestamp_col}' not found")
            return result
        
        # Check if column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            try:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            except:
                result['errors'].append(f"Cannot convert '{timestamp_col}' to datetime")
                return result
        
        # Check timezone
        if df[timestamp_col].dt.tz is not None:
            result['has_timezone'] = True
            tz_name = str(df[timestamp_col].dt.tz)
            result['timezone'] = tz_name
            
            # Check if it's COT
            if 'Bogota' in tz_name or 'America/Bogota' in tz_name:
                result['is_cot'] = True
                result['is_valid'] = True
            else:
                result['needs_conversion'] = True
                result['is_valid'] = True  # Valid but needs conversion
        else:
            result['errors'].append("Timestamp column has no timezone (naive)")
            result['needs_conversion'] = True
        
        return result
    
    @staticmethod
    def ensure_cot_timezone(df: pd.DataFrame, 
                           timestamp_col: str = 'timestamp',
                           assume_tz: str = None) -> pd.DataFrame:
        """
        Ensure dataframe timestamps are in COT timezone
        
        Args:
            df: DataFrame to process
            timestamp_col: Name of timestamp column
            assume_tz: Timezone to assume if data is naive
            
        Returns:
            DataFrame with COT timezone
        """
        df = df.copy()
        
        # Validate first
        validation = TimezoneValidator.validate_dataframe_timezone(df, timestamp_col)
        
        if not validation['has_timezone']:
            # Naive timestamps
            if assume_tz:
                logging.info(f"  🕐 Localizing naive timestamps to {assume_tz}")
                df[timestamp_col] = df[timestamp_col].dt.tz_localize(assume_tz)
                
                if assume_tz != TimezoneValidator.COT_TZ:
                    logging.info(f"  🕐 Converting from {assume_tz} to COT")
                    df[timestamp_col] = df[timestamp_col].dt.tz_convert(TimezoneValidator.COT_TZ)
            else:
                # Assume COT if no timezone specified
                logging.warning(f"  ⚠️ Assuming naive timestamps are in COT")
                df[timestamp_col] = df[timestamp_col].dt.tz_localize(TimezoneValidator.COT_TZ)
        
        elif not validation['is_cot']:
            # Has timezone but not COT
            logging.info(f"  🕐 Converting from {validation['timezone']} to COT")
            df[timestamp_col] = df[timestamp_col].dt.tz_convert(TimezoneValidator.COT_TZ)
        
        # Add helper columns
        df['timestamp_cot'] = df[timestamp_col]
        df['hour_cot'] = df['timestamp_cot'].dt.hour
        df['weekday'] = df['timestamp_cot'].dt.dayofweek
        
        return df
    
    @staticmethod
    def identify_premium_hours(df: pd.DataFrame,
                              timestamp_col: str = 'timestamp_cot') -> pd.DataFrame:
        """
        Identify premium trading hours (8am-2pm COT, Mon-Fri)
        
        Args:
            df: DataFrame with COT timestamps
            timestamp_col: Name of COT timestamp column
            
        Returns:
            DataFrame with premium hours marked
        """
        df = df.copy()
        
        # Ensure COT timezone
        if timestamp_col not in df.columns:
            df = TimezoneValidator.ensure_cot_timezone(df, 'timestamp')
            timestamp_col = 'timestamp_cot'
        
        # Mark premium hours
        df['is_premium'] = (
            (df[timestamp_col].dt.dayofweek < 5) &  # Monday-Friday
            (df[timestamp_col].dt.hour >= TimezoneValidator.PREMIUM_START_HOUR) &
            (df[timestamp_col].dt.hour < TimezoneValidator.PREMIUM_END_HOUR)
        )
        
        return df
    
    @staticmethod
    def get_premium_stats(df: pd.DataFrame) -> Dict:
        """
        Calculate statistics about premium hours coverage
        
        Args:
            df: DataFrame with timezone data
            
        Returns:
            Dictionary with premium hours statistics
        """
        # Ensure premium hours are marked
        if 'is_premium' not in df.columns:
            df = TimezoneValidator.identify_premium_hours(df)
        
        total_records = len(df)
        premium_records = df['is_premium'].sum()
        
        stats = {
            'total_records': total_records,
            'premium_records': premium_records,
            'premium_percentage': (premium_records / total_records * 100) if total_records > 0 else 0,
            'hour_distribution': {}
        }
        
        # Hour distribution
        if 'hour_cot' in df.columns:
            for hour in range(24):
                hour_count = len(df[df['hour_cot'] == hour])
                is_premium = TimezoneValidator.PREMIUM_START_HOUR <= hour < TimezoneValidator.PREMIUM_END_HOUR
                stats['hour_distribution'][hour] = {
                    'count': hour_count,
                    'is_premium': is_premium,
                    'percentage': (hour_count / total_records * 100) if total_records > 0 else 0
                }
        
        return stats
    
    @staticmethod
    def convert_timezone_mapping(hour: int, from_tz: str, to_tz: str) -> int:
        """
        Convert hour from one timezone to another
        
        Args:
            hour: Hour in source timezone (0-23)
            from_tz: Source timezone
            to_tz: Target timezone
            
        Returns:
            Hour in target timezone
        """
        # Create a sample datetime
        sample_dt = datetime(2024, 1, 15, hour, 0, 0)  # Use a Monday
        
        # Localize to source timezone
        source_tz = pytz.timezone(from_tz)
        dt_source = source_tz.localize(sample_dt)
        
        # Convert to target timezone
        target_tz = pytz.timezone(to_tz)
        dt_target = dt_source.astimezone(target_tz)
        
        return dt_target.hour
    
    @staticmethod
    def get_premium_hours_in_timezone(target_tz: str) -> Tuple[int, int]:
        """
        Get premium hours (8am-2pm COT) in another timezone
        
        Args:
            target_tz: Target timezone
            
        Returns:
            Tuple of (start_hour, end_hour) in target timezone
        """
        start_hour = TimezoneValidator.convert_timezone_mapping(
            TimezoneValidator.PREMIUM_START_HOUR,
            TimezoneValidator.COT_TZ,
            target_tz
        )
        
        end_hour = TimezoneValidator.convert_timezone_mapping(
            TimezoneValidator.PREMIUM_END_HOUR,
            TimezoneValidator.COT_TZ,
            target_tz
        )
        
        return start_hour, end_hour


def test_timezone_conversions():
    """Test critical timezone conversions"""
    validator = TimezoneValidator()
    
    # Test COT to UTC conversion
    # 8am COT should be 1pm UTC (13:00)
    utc_start, utc_end = validator.get_premium_hours_in_timezone('UTC')
    assert utc_start == 13, f"8am COT should be 1pm UTC, got {utc_start}"
    assert utc_end == 19, f"2pm COT should be 7pm UTC, got {utc_end}"
    
    # Test COT to MT5 (UTC+2) conversion
    # 8am COT should be 3pm UTC+2 (15:00)
    mt5_start, mt5_end = validator.get_premium_hours_in_timezone('Etc/GMT-2')
    assert mt5_start == 15, f"8am COT should be 3pm UTC+2, got {mt5_start}"
    assert mt5_end == 21, f"2pm COT should be 9pm UTC+2, got {mt5_end}"
    
    print("[PASS] All timezone conversion tests passed!")
    print(f"  COT Premium: 8:00-14:00")
    print(f"  UTC Premium: {utc_start}:00-{utc_end}:00")
    print(f"  MT5 Premium: {mt5_start}:00-{mt5_end}:00")


if __name__ == "__main__":
    test_timezone_conversions()