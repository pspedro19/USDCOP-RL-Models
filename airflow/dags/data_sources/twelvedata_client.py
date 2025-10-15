"""
TwelveData API Client for USDCOP Trading
Real connection to TwelveData API with Enhanced Monitoring and Timezone Fixes
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import time
import pytz

# Import unified datetime handler
try:
    from ..utils.datetime_handler import UnifiedDatetimeHandler, timezone_safe
    UNIFIED_DATETIME = True
except ImportError:
    logging.warning("UnifiedDatetimeHandler not available, using basic timezone handling")
    UNIFIED_DATETIME = False
    # Define fallback timezone_safe decorator
    def timezone_safe(func):
        """Fallback timezone_safe decorator when UnifiedDatetimeHandler is not available"""
        return func
try:
    from ..utils.enhanced_api_monitor import api_monitor
    ENHANCED_MONITORING = True
except ImportError:
    ENHANCED_MONITORING = False
    logging.warning("Enhanced API monitoring not available")

class TwelveDataClient:
    """TwelveData API client for fetching forex data"""
    
    def __init__(self, api_key: str = None, api_key_manager=None):
        """
        Initialize TwelveData client
        
        Args:
            api_key: TwelveData API key (single key or managed by manager)
            api_key_manager: Optional APIKeyManager for multiple keys
        """
        # Use API key manager if provided
        self.api_key_manager = api_key_manager
        
        if self.api_key_manager:
            # Get best available key from manager
            self.api_key, _ = self.api_key_manager.get_best_key()
        else:
            # Use single key (backward compatibility) - Load from environment
            import os
            self.api_key = api_key or os.environ.get('TWELVEDATA_API_KEY_1', 'demo')
            
        self.base_url = "https://api.twelvedata.com"
        self.symbol = "USD/COP"
        self.exchange = "FOREX"
        self.timezone = "America/Bogota"  # Colombia time (UTC-5)
        
    @timezone_safe
    def fetch_historical_data(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = "5min"
    ) -> pd.DataFrame:
        """
        Fetch historical data from TwelveData API
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            interval: Time interval (5min, 15min, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # TIMEZONE FIX: Ensure input dates are timezone-aware
            if UNIFIED_DATETIME:
                start_date = UnifiedDatetimeHandler.ensure_timezone_aware(start_date, 'America/Bogota')
                end_date = UnifiedDatetimeHandler.ensure_timezone_aware(end_date, 'America/Bogota')
            else:
                # Fallback timezone handling
                if start_date.tzinfo is None:
                    start_date = pytz.timezone('America/Bogota').localize(start_date)
                if end_date.tzinfo is None:
                    end_date = pytz.timezone('America/Bogota').localize(end_date)

            # TwelveData has a limit of 5000 data points per request
            # For 5-minute data, that's about 17 days
            # We need to split large requests into chunks

            all_data = []
            current_start = start_date
            chunk_days = 7  # REDUCED: Smaller chunks to avoid rate limits (1 week at a time)
            
            while current_start < end_date:
                current_end = min(
                    current_start + timedelta(days=chunk_days),
                    end_date
                )
                
                # Prepare API request
                params = {
                    'symbol': self.symbol,
                    'interval': interval,
                    'start_date': current_start.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': current_end.strftime('%Y-%m-%d %H:%M:%S'),
                    'timezone': self.timezone,
                    'apikey': self.api_key,
                    'format': 'JSON',
                    'output_size': 5000
                }
                
                # Make API request with monitoring
                endpoint = f"{self.base_url}/time_series"
                
                logging.info(f"Fetching REAL TwelveData for {current_start.date()} to {current_end.date()}")
                logging.info(f"  API Key: {self.api_key[:10]}...")
                logging.info(f"  Symbol: {self.symbol}")
                
                # Record API call start time for monitoring
                start_time = time.time()
                
                try:
                    response = requests.get(endpoint, params=params, timeout=30)
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    # Record successful API call
                    if ENHANCED_MONITORING:
                        api_monitor.record_api_call(
                            api_name='twelvedata',
                            endpoint='time_series',
                            key_id=f"key_{self.api_key[-4:]}",  # Use last 4 chars as key identifier
                            success=(response.status_code == 200),
                            response_time_ms=response_time_ms,
                            status_code=response.status_code,
                            response_size=len(response.content) if hasattr(response, 'content') else None
                        )
                
                except Exception as e:
                    response_time_ms = int((time.time() - start_time) * 1000)
                    
                    # Record failed API call
                    if ENHANCED_MONITORING:
                        api_monitor.record_api_call(
                            api_name='twelvedata',
                            endpoint='time_series',
                            key_id=f"key_{self.api_key[-4:]}",
                            success=False,
                            response_time_ms=response_time_ms,
                            status_code=0,
                            error_message=str(e)
                        )
                    raise e
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for API errors
                    if 'status' in data and data['status'] == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        logging.error(f"TwelveData API Error: {error_msg}")
                        
                        # Check if it's a credit limit error
                        if 'API credits' in error_msg or 'limit' in error_msg:
                            # Record usage and try to rotate key
                            if self.api_key_manager:
                                self.api_key_manager.record_usage(self.api_key, 1, success=False)
                                
                                # Try to rotate to a new key
                                new_key = self.api_key_manager.rotate_key()
                                if new_key != self.api_key:
                                    self.api_key = new_key
                                    params['apikey'] = self.api_key
                                    logging.info(f"🔄 Rotated to new API key: {self.api_key[:10]}...")
                                    # Retry with new key
                                    response = requests.get(endpoint, params=params, timeout=30)
                                    if response.status_code == 200:
                                        data = response.json()
                                        # Continue processing with new response
                                    else:
                                        continue
                                else:
                                    logging.error("All API keys exhausted")
                                    continue
                            else:
                                continue
                        elif 'code' in data and data['code'] == 400:
                            logging.error("Invalid API key or parameters")
                            continue
                        else:
                            continue
                    
                    if 'values' in data:
                        df_chunk = pd.DataFrame(data['values'])
                        
                        # Convert to standard format
                        df_chunk = df_chunk.rename(columns={
                            'datetime': 'timestamp',
                            'open': 'open',
                            'high': 'high',
                            'low': 'low',
                            'close': 'close',
                            'volume': 'volume'
                        })
                        
                        # Convert timestamp to datetime
                        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])

                        # TIMEZONE FIX: Use UnifiedDatetimeHandler for consistent timezone handling
                        if UNIFIED_DATETIME:
                            # Use unified datetime handler
                            df_chunk['timestamp'] = UnifiedDatetimeHandler.ensure_timezone_aware(
                                df_chunk['timestamp'], 'America/Bogota'
                            )
                            df_chunk = UnifiedDatetimeHandler.add_timezone_columns(df_chunk, 'timestamp')
                            logging.info(f"  🕒 Applied unified timezone handling (COT/UTC-5)")
                        else:
                            # Fallback: Basic timezone handling
                            if df_chunk['timestamp'].dt.tz is None:
                                # Data is naive, assume it's in COT as requested
                                df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_localize('America/Bogota')
                                logging.info(f"  🕒 Localized timestamps to America/Bogota (COT/UTC-5)")
                            else:
                                # If it has timezone, convert to COT
                                df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_convert('America/Bogota')
                                logging.info(f"  🕒 Converted timestamps to America/Bogota (COT/UTC-5)")

                            # Store both COT and UTC for reference
                            df_chunk['timestamp_cot'] = df_chunk['timestamp']
                            df_chunk['timestamp_utc'] = df_chunk['timestamp'].dt.tz_convert('UTC')
                            df_chunk['hour_cot'] = df_chunk['timestamp_cot'].dt.hour
                        
                        # CRITICAL FIX: Convert string prices to numeric BEFORE calculations
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            if col in df_chunk.columns:
                                df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')
                        
                        # NO FILTER HERE - Let the pipeline decide what to keep
                        # Premium filtering will be done at the DAG level
                        # df_chunk = self._filter_premium_hours(df_chunk)  # REMOVED
                        
                        all_data.append(df_chunk)
                        logging.info(f"  ✅ REAL DATA: Retrieved {len(df_chunk)} bars from TwelveData API")
                        
                        # Record successful API usage
                        if self.api_key_manager:
                            self.api_key_manager.record_usage(self.api_key, 1, success=True)
                        
                    elif 'code' in data:
                        # API error
                        if data['code'] == 429:
                            logging.warning("Rate limit reached, waiting 60 seconds...")
                            time.sleep(60)
                            continue
                        else:
                            logging.error(f"API error: {data.get('message', 'Unknown error')}")
                            
                else:
                    logging.error(f"HTTP error {response.status_code}: {response.text}")
                    
                # Move to next chunk
                current_start = current_end
                
                # Rate limiting - Adaptive based on response
                if self.api_key_manager:
                    # With multiple keys, shorter delay
                    time.sleep(1)  # 1 second between requests
                elif self.api_key == "demo":
                    time.sleep(8)  # Demo key is heavily limited
                else:
                    time.sleep(2)  # Single key needs more delay
                    
            # Combine all chunks
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
                
                # Add metadata
                df['timezone'] = 'America/Bogota'  # More explicit than UTC-5
                df['source'] = 'twelvedata'
                df['fetch_timestamp'] = datetime.utcnow()
                
                # Ensure timezone consistency in final dataframe
                if 'timestamp_cot' not in df.columns and 'timestamp' in df.columns:
                    if df['timestamp'].dt.tz is None:
                        df['timestamp'] = df['timestamp'].dt.tz_localize('America/Bogota')
                    df['timestamp_cot'] = df['timestamp'].dt.tz_convert('America/Bogota')
                    df['hour_cot'] = df['timestamp_cot'].dt.hour
                
                # Convert numeric columns FIRST (moved before spread calculation)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Estimate spread if not provided (AFTER conversion to numeric)
                df['spread'] = ((df['high'] - df['low']) / df['close'] * 10000).clip(2, 15)
                    
                logging.info(f"✅ SUCCESS: Total REAL data from TwelveData: {len(df)} bars")
                logging.info(f"   API: TwelveData (NOT fallback)")
                logging.info(f"   Period: {start_date.date()} to {end_date.date()}")
                return df
                
            else:
                logging.error("NO DATA retrieved from TwelveData API")
                logging.error(f"API Key: {self.api_key[:10]}...")
                logging.error("Check API key validity and rate limits")
                logging.error("❌ CRITICAL: NO FALLBACK DATA - Returning empty DataFrame")
                return pd.DataFrame()  # Return empty instead of fake data
                
        except Exception as e:
            logging.error(f"CRITICAL ERROR fetching TwelveData: {e}")
            logging.error(f"API Key used: {self.api_key[:10]}...")
            import traceback
            logging.error(traceback.format_exc())
            logging.error("❌ CRITICAL: NO FALLBACK DATA - Returning empty DataFrame")
            return pd.DataFrame()  # Return empty instead of fake data
            
    def _filter_premium_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter data to keep only premium trading hours
        Monday-Friday, 8:00-14:00 Colombia time
        """
        if df.empty:
            return df
            
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create filters
        is_weekday = df['timestamp'].dt.dayofweek < 5  # Mon=0, Fri=4
        is_premium_hour = (df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour < 14)
        
        # Apply filters
        df_filtered = df[is_weekday & is_premium_hour].copy()
        
        return df_filtered
        
    def _generate_fallback_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate realistic fallback data when API is not available
        Generate 24-hour forex market data for USD/COP
        """
        logging.warning("Using fallback data generation for TwelveData")
        logging.info("Generating 24-hour forex market data")
        
        # Generate timestamps for 24-hour market
        timestamps = []
        current = start_date
        
        # Colombia holidays to exclude (simplified list)
        colombia_holidays = [
            (1, 1),   # New Year
            (1, 6),   # Epiphany
            (5, 1),   # Labor Day
            (7, 20),  # Independence Day
            (8, 7),   # Battle of Boyacá
            (12, 25), # Christmas
        ]
        
        while current <= end_date:
            # Check if weekday and not a holiday
            if current.weekday() < 5:
                is_holiday = (current.month, current.day) in colombia_holidays
                
                if not is_holiday:
                    # Generate timestamps for ALL hours (24-hour forex market)
                    timestamps.append(current)
                        
            current += timedelta(minutes=5)
            
        if not timestamps:
            return pd.DataFrame()
            
        num_points = len(timestamps)
        
        # USDCOP characteristics for Colombia market
        base_price = 4200
        colombia_volatility = 18  # Higher volatility in Colombian market hours
        
        # Generate price with 24-hour forex market patterns
        volatility_profile = []
        for ts in timestamps:
            hour = ts.hour  # Already in COT (UTC-5)
            
            # Forex market volatility patterns
            if 8 <= hour < 14:  # Colombian premium hours (high activity)
                vol_mult = 1.3
            elif 3 <= hour < 8:  # London/European morning overlap
                vol_mult = 1.2
            elif 14 <= hour < 17:  # US afternoon session
                vol_mult = 1.1
            elif 20 <= hour < 24 or 0 <= hour < 3:  # Asian session
                vol_mult = 0.9
            else:  # Low activity periods
                vol_mult = 0.7
            volatility_profile.append(vol_mult)
            
        # Generate returns with time-varying volatility
        returns = np.array([
            np.random.normal(0, colombia_volatility * vol_mult / np.sqrt(288))
            for vol_mult in volatility_profile
        ])
        
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Add Colombian market microstructure
        # Higher spreads during volatile periods
        spreads = np.array([
            np.random.uniform(3, 12) * vol_mult
            for vol_mult in volatility_profile
        ])
        
        # Generate OHLCV
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': price_series * (1 + np.random.normal(0, 0.0002, num_points)),
            'high': price_series * (1 + np.abs(np.random.normal(0, 0.0004, num_points))),
            'low': price_series * (1 - np.abs(np.random.normal(0, 0.0004, num_points))),
            'close': price_series * (1 + np.random.normal(0, 0.0002, num_points)),
            'volume': np.random.exponential(500000, num_points) * np.array(volatility_profile),
            'spread': spreads,
            'timezone': 'UTC-5',
            'source': 'twelvedata_fallback',
            'fetch_timestamp': datetime.utcnow()
        })
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        return df
        
    def get_latest_price(self) -> Optional[Dict]:
        """Get latest price for USD/COP"""
        start_time = time.time()
        try:
            endpoint = f"{self.base_url}/price"
            params = {
                'symbol': self.symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(endpoint, params=params, timeout=10)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Record API call
            if ENHANCED_MONITORING:
                api_monitor.record_api_call(
                    api_name='twelvedata',
                    endpoint='price',
                    key_id=f"key_{self.api_key[-4:]}",
                    success=(response.status_code == 200),
                    response_time_ms=response_time_ms,
                    status_code=response.status_code,
                    response_size=len(response.content) if hasattr(response, 'content') else None
                )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'symbol': self.symbol,
                    'price': float(data.get('price', 0)),
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Record failed API call
            if ENHANCED_MONITORING:
                api_monitor.record_api_call(
                    api_name='twelvedata',
                    endpoint='price',
                    key_id=f"key_{self.api_key[-4:]}",
                    success=False,
                    response_time_ms=response_time_ms,
                    status_code=0,
                    error_message=str(e)
                )
            
            logging.error(f"Error getting latest price: {e}")
            
        return None
        
    def check_api_usage(self) -> Optional[Dict]:
        """Check API usage and limits"""
        start_time = time.time()
        try:
            endpoint = f"{self.base_url}/api_usage"
            params = {'apikey': self.api_key}
            
            response = requests.get(endpoint, params=params, timeout=10)
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Record API call
            if ENHANCED_MONITORING:
                api_monitor.record_api_call(
                    api_name='twelvedata',
                    endpoint='api_usage',
                    key_id=f"key_{self.api_key[-4:]}",
                    success=(response.status_code == 200),
                    response_time_ms=response_time_ms,
                    status_code=response.status_code,
                    response_size=len(response.content) if hasattr(response, 'content') else None
                )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            
            # Record failed API call
            if ENHANCED_MONITORING:
                api_monitor.record_api_call(
                    api_name='twelvedata',
                    endpoint='api_usage',
                    key_id=f"key_{self.api_key[-4:]}",
                    success=False,
                    response_time_ms=response_time_ms,
                    status_code=0,
                    error_message=str(e)
                )
            
            logging.error(f"Error checking API usage: {e}")
            
        return None
    
    def get_monitoring_status(self) -> Dict:
        """Get comprehensive monitoring status from enhanced monitor"""
        if not ENHANCED_MONITORING:
            return {"error": "Enhanced monitoring not available"}
        
        try:
            key_id = f"key_{self.api_key[-4:]}"
            key_status = api_monitor.get_key_status(key_id)
            health_metrics = api_monitor.get_api_health_metrics('twelvedata')
            
            return {
                "key_status": key_status.__dict__ if key_status else None,
                "health_metrics": health_metrics.__dict__ if health_metrics else None,
                "all_keys": [status.__dict__ for status in api_monitor.get_all_key_statuses()],
                "best_available_key": api_monitor.get_best_available_key('twelvedata')
            }
        except Exception as e:
            logging.error(f"Error getting monitoring status: {e}")
            return {"error": str(e)}