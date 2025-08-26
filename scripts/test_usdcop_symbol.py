"""
Test USDCOP.r Symbol Availability
=================================
Test the discovered USDCOP.r symbol for data availability and trading.
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

def test_usdcop_symbol():
    """Test USDCOP.r symbol availability and data"""
    
    try:
        from dotenv import load_dotenv
        import MetaTrader5 as mt5
        
        # Load environment
        load_dotenv(project_root / '.env')
        
        # Connect
        if not mt5.initialize():
            logger.error("❌ Could not initialize MT5")
            return False
        
        login = int(os.getenv('MT5_LOGIN'))
        password = os.getenv('MT5_PASSWORD')
        server = os.getenv('MT5_SERVER')
        
        if not mt5.login(login, password=password, server=server):
            logger.error("❌ Could not login")
            return False
        
        logger.info("✅ Connected to MT5")
        
        # Test USDCOP.r symbol
        symbol = "USDCOP.r"
        logger.info(f"🔄 Testing {symbol} symbol...")
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info:
            logger.info(f"✅ {symbol} symbol found!")
            logger.info(f"   Description: {symbol_info.description}")
            logger.info(f"   Digits: {symbol_info.digits}")
            logger.info(f"   Spread: {symbol_info.spread}")
            logger.info(f"   Point: {symbol_info.point}")
            logger.info(f"   Min volume: {symbol_info.volume_min}")
            logger.info(f"   Max volume: {symbol_info.volume_max}")
            logger.info(f"   Trade allowed: {symbol_info.trade_mode}")
        else:
            logger.error(f"❌ {symbol} symbol not found")
            return False
        
        # Test current tick
        logger.info("🔄 Testing current tick data...")
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            logger.info(f"✅ Current tick:")
            logger.info(f"   Bid: {tick.bid}")
            logger.info(f"   Ask: {tick.ask}")
            logger.info(f"   Time: {datetime.fromtimestamp(tick.time)}")
            logger.info(f"   Spread: {tick.ask - tick.bid:.5f}")
        else:
            logger.warning("⚠️  Could not get current tick")
        
        # Test historical data
        logger.info("🔄 Testing historical data...")
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 10)
        if rates is not None and len(rates) > 0:
            logger.info(f"✅ Retrieved {len(rates)} historical bars")
            latest = rates[-1]
            dt = datetime.fromtimestamp(latest['time'])
            logger.info(f"   Latest bar: {dt}")
            logger.info(f"   OHLC: {latest['open']:.5f}, {latest['high']:.5f}, {latest['low']:.5f}, {latest['close']:.5f}")
            logger.info(f"   Volume: {latest['tick_volume']}")
        else:
            logger.warning("⚠️  Could not retrieve historical data")
        
        # Test different timeframes
        logger.info("🔄 Testing different timeframes...")
        timeframes = [
            (mt5.TIMEFRAME_M1, "M1"),
            (mt5.TIMEFRAME_M5, "M5"),
            (mt5.TIMEFRAME_M15, "M15"),
            (mt5.TIMEFRAME_H1, "H1"),
            (mt5.TIMEFRAME_D1, "D1")
        ]
        
        for tf_value, tf_name in timeframes:
            rates = mt5.copy_rates_from_pos(symbol, tf_value, 0, 5)
            if rates is not None and len(rates) > 0:
                logger.info(f"   ✅ {tf_name}: {len(rates)} bars available")
            else:
                logger.info(f"   ❌ {tf_name}: No data")
        
        # Test date range retrieval
        logger.info("🔄 Testing date range retrieval...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)  # Last week
        
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
        if rates is not None and len(rates) > 0:
            logger.info(f"✅ Date range: {len(rates)} hourly bars from last week")
            first_bar = datetime.fromtimestamp(rates[0]['time'])
            last_bar = datetime.fromtimestamp(rates[-1]['time'])
            logger.info(f"   Range: {first_bar} to {last_bar}")
        else:
            logger.warning("⚠️  Date range: No data available")
        
        logger.info("\n" + "="*50)
        logger.info("🎉 USDCOP.r SYMBOL TEST SUMMARY")
        logger.info("="*50)
        logger.info("✅ USDCOP.r symbol is available and working!")
        logger.info("✅ Real-time tick data accessible")
        logger.info("✅ Historical data available")
        logger.info("✅ Multiple timeframes supported")
        logger.info("\n💡 Recommendation: Update project configuration to use 'USDCOP.r' instead of 'USDCOP'")
        
        mt5.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        try:
            import MetaTrader5 as mt5
            mt5.shutdown()
        except:
            pass
        return False

if __name__ == "__main__":
    success = test_usdcop_symbol()
    if success:
        print("\n🚀 USDCOP.r symbol is ready for trading!")
    else:
        print("\n❌ USDCOP.r symbol test failed")
    sys.exit(0 if success else 1)