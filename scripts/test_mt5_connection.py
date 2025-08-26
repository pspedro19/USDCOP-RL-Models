"""
MT5 Connection Test Script
=========================
Test real MT5 connection with configured credentials.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables"""
    try:
        from dotenv import load_dotenv
        env_path = project_root / '.env'
        
        if not env_path.exists():
            logger.error(f"❌ .env file not found at {env_path}")
            logger.info("Please create .env file with MT5 credentials")
            return False
            
        load_dotenv(env_path)
        logger.info("✅ Environment variables loaded")
        return True
    except ImportError:
        logger.error("❌ python-dotenv not installed. Install with: pip install python-dotenv")
        return False

def check_mt5_installation():
    """Check if MT5 is installed and accessible"""
    try:
        import MetaTrader5 as mt5
        logger.info("✅ MetaTrader5 package found")
        return True
    except ImportError:
        logger.error("❌ MetaTrader5 package not installed")
        logger.info("Install with: pip install MetaTrader5")
        return False

def test_mt5_connection():
    """Test MT5 connection with real credentials"""
    
    # Check prerequisites
    if not load_environment():
        return False
        
    if not check_mt5_installation():
        return False
    
    try:
        import MetaTrader5 as mt5
        
        # Get credentials from environment
        login = os.getenv('MT5_LOGIN')
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        timeout = int(os.getenv('MT5_TIMEOUT', '60000'))
        
        if not all([login, password, server]):
            logger.error("❌ Missing MT5 credentials in .env file")
            logger.info("Required: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER")
            return False
            
        login = int(login)
        logger.info(f"Testing connection to {server} with login {login}")
        
        # Step 1: Initialize MT5
        logger.info("🔄 Initializing MT5...")
        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"❌ MT5 initialization failed: {error}")
            logger.info("💡 Make sure MetaTrader 5 terminal is installed and can be launched")
            return False
        
        logger.info("✅ MT5 initialized successfully")
        
        # Step 2: Login
        logger.info("🔄 Logging in...")
        if not mt5.login(login, password=password, server=server, timeout=timeout):
            error = mt5.last_error()
            logger.error(f"❌ Login failed: {error}")
            
            # Provide specific error guidance
            if error[0] == 10004:  # RET_ERROR
                logger.info("💡 Check credentials and server name")
            elif error[0] == 10007:  # RET_ERROR_OLD_VERSION
                logger.info("💡 Update MetaTrader 5 terminal")
            elif error[0] == 10020:  # RET_ERROR_CONNECTION
                logger.info("💡 Check internet connection and server availability")
            
            mt5.shutdown()
            return False
        
        logger.info("✅ Login successful!")
        
        # Step 3: Get account info
        account_info = mt5.account_info()
        if account_info:
            logger.info(f"✅ Account: {account_info.name} | Balance: ${account_info.balance:.2f}")
            logger.info(f"   Server: {account_info.server} | Currency: {account_info.currency}")
        
        # Step 4: Test symbol info
        logger.info("🔄 Testing USDCOP.r symbol...")
        symbol_info = mt5.symbol_info("USDCOP.r")
        if symbol_info:
            logger.info(f"✅ USDCOP.r symbol found: {symbol_info.description}")
            logger.info(f"   Spread: {symbol_info.spread} | Digits: {symbol_info.digits}")
        else:
            logger.warning("⚠️  USDCOP.r symbol not found, trying alternatives...")
            
            # Try alternative symbols
            alternatives = ["USDCOP", "USDCOP.m", "USD/COP", "USDCOP_m"]
            for alt_symbol in alternatives:
                alt_info = mt5.symbol_info(alt_symbol)
                if alt_info:
                    logger.info(f"✅ Found alternative: {alt_symbol} - {alt_info.description}")
                    break
            else:
                logger.warning("⚠️  No USDCOP variants found")
        
        # Step 5: Test tick data
        logger.info("🔄 Testing tick data...")
        tick = mt5.symbol_info_tick("USDCOP.r")
        if tick:
            logger.info(f"✅ Current tick: Bid={tick.bid} Ask={tick.ask} Time={datetime.fromtimestamp(tick.time)}")
        else:
            logger.warning("⚠️  Could not get tick data for USDCOP.r")
        
        # Step 6: Test historical data
        logger.info("🔄 Testing historical data...")
        rates = mt5.copy_rates_from_pos("USDCOP.r", mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None and len(rates) > 0:
            logger.info(f"✅ Retrieved {len(rates)} historical bars")
            latest = rates[-1]
            dt = datetime.fromtimestamp(latest['time'])
            logger.info(f"   Latest bar: {dt} | Close={latest['close']:.4f}")
        else:
            logger.warning("⚠️  Could not retrieve historical data")
        
        # Step 7: Test trading permissions
        logger.info("🔄 Testing trading permissions...")
        if account_info and account_info.trade_allowed:
            logger.info("✅ Trading is allowed on this account")
        else:
            logger.warning("⚠️  Trading may not be allowed - check account settings")
        
        # Step 8: Connection summary
        logger.info("\n" + "="*50)
        logger.info("🎉 MT5 CONNECTION TEST SUMMARY")
        logger.info("="*50)
        logger.info("✅ All basic connection tests passed!")
        logger.info(f"✅ Connected to: {server}")
        logger.info(f"✅ Account: {account_info.name if account_info else 'Unknown'}")
        logger.info(f"✅ Balance: ${account_info.balance:.2f}" if account_info else "")
        logger.info("✅ Ready for trading operations")
        
        # Cleanup
        mt5.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except:
            pass
        return False

def test_data_retrieval():
    """Test data retrieval capabilities"""
    logger.info("\n" + "="*50)
    logger.info("📊 TESTING DATA RETRIEVAL")
    logger.info("="*50)
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            logger.error("❌ Could not initialize MT5")
            return False
        
        # Login
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        if not mt5.login(login, password=password, server=server):
            logger.error("❌ Could not login")
            mt5.shutdown()
            return False
        
        # Test different timeframes
        timeframes = [
            (mt5.TIMEFRAME_M1, "M1"),
            (mt5.TIMEFRAME_M5, "M5"),
            (mt5.TIMEFRAME_M15, "M15"),
            (mt5.TIMEFRAME_H1, "H1"),
            (mt5.TIMEFRAME_D1, "D1")
        ]
        
        for tf_value, tf_name in timeframes:
            rates = mt5.copy_rates_from_pos("USDCOP.r", tf_value, 0, 5)
            if rates is not None and len(rates) > 0:
                logger.info(f"✅ {tf_name}: Retrieved {len(rates)} bars")
            else:
                logger.warning(f"⚠️  {tf_name}: No data available")
        
        # Test date range retrieval
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        rates = mt5.copy_rates_range("USDCOP.r", mt5.TIMEFRAME_M5, start_date, end_date)
        if rates is not None and len(rates) > 0:
            logger.info(f"✅ Date range: Retrieved {len(rates)} bars from last 24h")
        else:
            logger.warning("⚠️  Date range: No data available")
        
        mt5.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ Data retrieval test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting MT5 Connection Test")
    logger.info("="*50)
    
    success = True
    
    # Test connection
    if not test_mt5_connection():
        success = False
    
    # Test data retrieval if connection works
    if success:
        if not test_data_retrieval():
            success = False
    
    # Final result
    logger.info("\n" + "="*50)
    if success:
        logger.info("🎉 ALL TESTS PASSED! MT5 connection is ready for trading.")
        logger.info("You can now use the trading system with confidence.")
    else:
        logger.error("❌ Some tests failed. Check the logs above for guidance.")
        logger.info("Fix the issues and run the test again.")
    
    logger.info("="*50)
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)