"""
MT5 Data Client for USDCOP Trading
Real connection to MetaTrader 5
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict
import pytz

# Try to import MetaTrader5 - it might not be available in Linux containers
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logging.warning("MetaTrader5 module not available - MT5 functionality disabled")
    mt5 = None
    MT5_AVAILABLE = False

class MT5DataClient:
    """MetaTrader 5 data client for fetching real market data"""
    
    def __init__(self, login: int = None, password: str = None, server: str = None):
        """
        Initialize MT5 client
        
        Args:
            login: MT5 account number
            password: MT5 account password  
            server: MT5 server name
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        self.symbol = "USDCOP"  # Or "USD_COP" depending on broker
        
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if not MT5_AVAILABLE:
            logging.error("MetaTrader5 module not available - cannot connect")
            return False
            
        try:
            # Initialize MT5
            if not mt5.initialize():
                logging.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
            
            # Login if credentials provided
            if self.login and self.password and self.server:
                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server
                )
                if not authorized:
                    logging.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
                    
            self.connected = True
            logging.info("Successfully connected to MT5")
            
            # Check if symbol exists
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info is None:
                # Try alternative symbol names
                for alt_symbol in ["USD_COP", "USDCOP.", "USDCOPm"]:
                    symbol_info = mt5.symbol_info(alt_symbol)
                    if symbol_info:
                        self.symbol = alt_symbol
                        break
                        
                if symbol_info is None:
                    logging.warning(f"Symbol {self.symbol} not found, will use available forex symbols")
                    
            return True
            
        except Exception as e:
            logging.error(f"MT5 connection error: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logging.info("Disconnected from MT5")
            
    def fetch_historical_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        timeframe: str = "M5"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from MT5
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe (M5 for 5 minutes)
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            if not self.connect():
                logging.error("CRITICAL: Cannot connect to MT5. Check credentials and MetaTrader5 installation.")
                logging.error(f"Login: {self.login}, Server: {self.server}")
                # Only use fallback as LAST RESORT
                logging.warning("Using fallback data - THIS IS NOT ACCEPTABLE FOR PRODUCTION")
                return self._generate_fallback_data(start_date, end_date)
        
        try:
            # Convert timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Set timezone to UTC+2 (typical MT5 server time)
            utc_tz = pytz.timezone("Etc/GMT-2")
            start_date = utc_tz.localize(start_date.replace(tzinfo=None))
            end_date = utc_tz.localize(end_date.replace(tzinfo=None))
            
            # Fetch rates
            rates = mt5.copy_rates_range(
                self.symbol,
                mt5_timeframe,
                start_date,
                end_date
            )
            
            if rates is None or len(rates) == 0:
                logging.error(f"NO DATA received from MT5 for {self.symbol}")
                logging.error(f"Requested period: {start_date} to {end_date}")
                logging.error(f"MT5 Error: {mt5.last_error()}")
                
                # Try alternative symbols
                alternative_symbols = ['USDCOP', 'USD_COP', 'USDCOP.', 'USDCOPm', 'USD/COP']
                for alt_symbol in alternative_symbols:
                    logging.info(f"Trying alternative symbol: {alt_symbol}")
                    rates = mt5.copy_rates_range(
                        alt_symbol,
                        mt5_timeframe,
                        start_date,
                        end_date
                    )
                    if rates is not None and len(rates) > 0:
                        self.symbol = alt_symbol
                        logging.info(f"SUCCESS: Found data with symbol {alt_symbol}")
                        break
                
                if rates is None or len(rates) == 0:
                    logging.error("FAILED to get real data from MT5 after trying all symbols")
                    logging.warning("FALLBACK DATA IS NOT ACCEPTABLE - Check MT5 connection")
                    return self._generate_fallback_data(start_date, end_date)
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Rename columns to standard names
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Etc/GMT-2')
            
            # Calculate quality metrics
            df['timezone'] = 'UTC+2'
            df['source'] = 'mt5'
            df['fetch_timestamp'] = datetime.utcnow()
            
            # Add calculated spread if not available
            if 'spread' not in df.columns or df['spread'].isna().all():
                # Estimate spread from high-low
                df['spread'] = ((df['high'] - df['low']) / df['close'] * 10000).clip(1, 10)
            
            logging.info(f"✅ REAL DATA: Fetched {len(df)} bars from MT5")
            logging.info(f"   Period: {start_date.date()} to {end_date.date()}")
            logging.info(f"   Symbol: {self.symbol}")
            logging.info(f"   Server: {self.server}")
            
            return df
            
        except Exception as e:
            logging.error(f"CRITICAL ERROR fetching MT5 data: {e}")
            logging.error(f"This should NOT happen with valid credentials")
            logging.error(f"Login: {self.login}, Server: {self.server}, Symbol: {self.symbol}")
            import traceback
            logging.error(traceback.format_exc())
            logging.warning("USING FALLBACK - THIS IS NOT ACCEPTABLE")
            return self._generate_fallback_data(start_date, end_date)
            
    def _generate_fallback_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate realistic fallback data when MT5 is not available
        Generate 24-hour forex market data (USD/COP trades 24/5)
        """
        logging.warning("Using fallback data generation for MT5")
        logging.info("Generating 24-hour market data for USD/COP forex pair")
        
        # Generate 5-minute timestamps for 24-hour market
        timestamps = []
        current = start_date
        
        while current <= end_date:
            # Check if it's a weekday (forex is 24/5)
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                # Generate data for ALL hours (24-hour market)
                timestamps.append(current)
            current += timedelta(minutes=5)
        
        if not timestamps:
            return pd.DataFrame()
            
        # Generate realistic USDCOP data
        num_points = len(timestamps)
        
        # USDCOP typically ranges from 3000-5000
        base_price = 4200
        volatility = 15  # Daily volatility in COP
        
        # Generate price series with realistic patterns
        returns = np.random.normal(0, volatility/np.sqrt(288), num_points)  # 288 = 5-min bars per day
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # Add intraday patterns
        intraday_pattern = np.array([
            1.0002, 1.0003, 1.0005, 1.0006, 1.0007, 1.0008,  # 8:00-8:30
            1.0009, 1.0010, 1.0011, 1.0012, 1.0011, 1.0010,  # 8:30-9:00
            1.0009, 1.0008, 1.0007, 1.0008, 1.0009, 1.0010,  # 9:00-9:30
            1.0011, 1.0012, 1.0013, 1.0012, 1.0011, 1.0010,  # 9:30-10:00
            1.0009, 1.0008, 1.0007, 1.0006, 1.0007, 1.0008,  # 10:00-10:30
            1.0009, 1.0010, 1.0011, 1.0010, 1.0009, 1.0008,  # 10:30-11:00
            1.0007, 1.0006, 1.0005, 1.0006, 1.0007, 1.0008,  # 11:00-11:30
            1.0009, 1.0010, 1.0011, 1.0010, 1.0009, 1.0008,  # 11:30-12:00
            1.0007, 1.0006, 1.0005, 1.0004, 1.0003, 1.0002,  # 12:00-12:30
            1.0001, 1.0000, 0.9999, 0.9998, 0.9997, 0.9996,  # 12:30-13:00
            0.9995, 0.9994, 0.9995, 0.9996, 0.9997, 0.9998,  # 13:00-13:30
            0.9999, 1.0000, 1.0001, 1.0000, 0.9999, 0.9998   # 13:30-14:00
        ])
        
        # Apply intraday pattern with 24-hour coverage
        for i, ts in enumerate(timestamps):
            # Map hour to intraday pattern (use modulo for 24-hour coverage)
            hour_cot = (ts.hour - 7) % 24  # Convert UTC+2 to COT (UTC-5)
            
            # Apply volatility pattern based on market sessions
            if 8 <= hour_cot < 14:  # Premium hours (high liquidity)
                volatility_mult = 1.2
            elif 14 <= hour_cot < 17:  # US afternoon
                volatility_mult = 1.1
            elif 3 <= hour_cot < 8:  # London/Europe morning
                volatility_mult = 1.15
            else:  # Low activity hours
                volatility_mult = 0.8
            
            # Apply minor price adjustments
            price_series[i] *= (1 + np.random.normal(0, 0.0001 * volatility_mult))
        
        # Generate OHLCV data
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': price_series * (1 + np.random.normal(0, 0.0001, num_points)),
            'high': price_series * (1 + np.abs(np.random.normal(0, 0.0003, num_points))),
            'low': price_series * (1 - np.abs(np.random.normal(0, 0.0003, num_points))),
            'close': price_series * (1 + np.random.normal(0, 0.0001, num_points)),
            'volume': np.random.exponential(1000000, num_points),
            'spread': np.random.uniform(2, 8, num_points),  # 2-8 pips
            'timezone': 'UTC+2',
            'source': 'mt5_fallback',
            'fetch_timestamp': datetime.utcnow()
        })
        
        # Ensure OHLC consistency
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
        
    def get_symbol_info(self) -> Optional[Dict]:
        """Get symbol information"""
        if not self.connected:
            return None
            
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            return {
                'symbol': symbol_info.name,
                'bid': symbol_info.bid,
                'ask': symbol_info.ask,
                'spread': symbol_info.spread,
                'digits': symbol_info.digits,
                'point': symbol_info.point,
                'trade_contract_size': symbol_info.trade_contract_size,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step
            }
        return None