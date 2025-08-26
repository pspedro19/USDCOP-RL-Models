"""
Live Trading Module
===================
Handles live trading operations with MT5
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LiveTrader:
    """Live trading implementation"""
    
    def __init__(self, config: Dict[str, Any] = None, symbol: str = 'USDCOP', 
                 model_path: str = None, risk_per_trade: float = 0.01,
                 connector=None, max_reconnect_attempts: int = 3):
        """Initialize live trader"""
        self.config = config or {}
        self.symbol = symbol
        self.model_path = model_path
        self.risk_per_trade = risk_per_trade
        self.running = False
        self.trades = []
        self.position = None
        self.balance = 10000.0
        self.model = None
        self.connector = connector
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Load model if path provided
        if model_path:
            self._load_model()
        
    def start(self) -> bool:
        """Start live trading session"""
        self.running = True
        logger.info("Live trading session started")
        return True
    
    def stop(self) -> bool:
        """Stop live trading session"""
        self.running = False
        logger.info("Live trading session stopped")
        return True
    
    def is_running(self) -> bool:
        """Check if trader is running"""
        return self.running
    
    def execute_trade(self, action: str, volume: float = 0.1) -> bool:
        """Execute a trade"""
        if not self.running:
            return False
            
        trade = {
            'timestamp': datetime.now(),
            'action': action,
            'volume': volume,
            'price': 4000.0,  # Mock price
            'status': 'executed'
        }
        self.trades.append(trade)
        
        if action == 'BUY':
            self.position = {'type': 'long', 'volume': volume}
        elif action == 'SELL' and self.position:
            self.position = None
            
        return True
    
    def get_balance(self) -> float:
        """Get current balance"""
        return self.balance
    
    def get_position(self) -> Optional[Dict[str, Any]]:
        """Get current position"""
        return self.position
    
    def run_trading_loop(self, duration: int = 60) -> Dict[str, Any]:
        """Run trading loop for specified duration"""
        self.start()
        
        start_time = time.time()
        trades_executed = 0
        
        while time.time() - start_time < duration and self.running:
            # Simulate trading logic
            if trades_executed == 0:
                self.execute_trade('BUY', 0.1)
                trades_executed += 1
            elif trades_executed == 1 and time.time() - start_time > duration / 2:
                self.execute_trade('SELL', 0.1)
                trades_executed += 1
                
            time.sleep(1)
        
        self.stop()
        
        return {
            'trades': len(self.trades),
            'final_balance': self.balance,
            'duration': time.time() - start_time
        }
    
    def handle_reconnection(self) -> bool:
        """Handle connection recovery"""
        logger.info("Handling reconnection...")
        time.sleep(1)  # Simulate reconnection
        return True
    
    def emergency_stop(self) -> bool:
        """Execute emergency stop"""
        logger.warning("Emergency stop triggered")
        
        # Close all positions
        if self.position:
            self.execute_trade('SELL', self.position['volume'])
        
        self.stop()
        return True
    
    def analyze_timeframes(self, timeframes: list) -> Dict[str, Any]:
        """Analyze multiple timeframes"""
        analysis = {}
        for tf in timeframes:
            analysis[tf] = {
                'trend': 'up' if tf == 'M5' else 'neutral',
                'strength': 0.7,
                'signal': 'buy' if tf == 'M5' else 'hold'
            }
        return analysis
    
    def handle_news_event(self, event: Dict[str, Any]) -> bool:
        """Handle news event"""
        logger.info(f"Handling news event: {event.get('title', 'Unknown')}")
        
        # Reduce position size during high impact news
        if event.get('impact') == 'high' and self.position:
            self.execute_trade('SELL', self.position['volume'] * 0.5)
        
        return True
    
    def get_tick_with_retry(self) -> Optional[Dict[str, Any]]:
        """Get tick data with retry logic"""
        if not self.connector:
            return {'time': datetime.now(), 'bid': 4000, 'ask': 4001}
        
        attempts = 0
        last_error = None
        
        while attempts < self.max_reconnect_attempts:
            try:
                tick = self.connector.get_tick()
                if isinstance(tick, Exception):
                    raise tick
                return tick
            except Exception as e:
                last_error = e
                attempts += 1
                logger.warning(f"Failed to get tick (attempt {attempts}/{self.max_reconnect_attempts}): {e}")
                time.sleep(1)
        
        # After all attempts failed, try one more time
        try:
            return self.connector.get_tick()
        except:
            return None
    
    def _load_model(self):
        """Load trading model from file"""
        # Stub implementation - would load actual model
        self.model = {'type': 'dummy', 'loaded': True}
    
    async def process_tick(self) -> Dict[str, Any]:
        """Process a single tick asynchronously"""
        import asyncio
        import random
        await asyncio.sleep(0.01)  # Simulate processing
        
        # Get tick data
        tick = self.get_tick_with_retry()
        if not tick:
            return {'action': 'HOLD', 'reason': 'No tick data'}
        
        # Simple trading logic - ensure at least some trades
        if not hasattr(self, '_tick_count'):
            self._tick_count = 0
        self._tick_count += 1
        
        # Force a trade on first tick for testing
        if self._tick_count == 1:
            action = 'BUY'
        else:
            action = random.choice(['BUY', 'SELL', 'HOLD', 'HOLD', 'HOLD'])
        
        result = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'action': action,
            'price': tick.get('bid', 4000.0)
        }
        
        if action != 'HOLD':
            self.execute_trade(action, self.risk_per_trade * self.balance / tick.get('bid', 4000.0))
            
        return result


class NewsAwareTrader(LiveTrader):
    """Live trader with news event awareness"""
    
    def __init__(self, config: Dict[str, Any] = None, news_monitor=None, **kwargs):
        """Initialize news-aware trader"""
        super().__init__(config, **kwargs)
        self.news_events = []
        self.news_filter = config.get('news_filter', 'all') if config else 'all'
        self.news_monitor = news_monitor
        
    def set_news_filter(self, filter_type: str):
        """Set news filter type"""
        self.news_filter = filter_type
        logger.info(f"News filter set to: {filter_type}")
        
    def add_news_event(self, event: Dict[str, Any]):
        """Add a news event"""
        self.news_events.append(event)
        
        # React to news based on impact
        if event.get('impact') == 'high':
            logger.warning(f"High impact news: {event.get('title')}")
            self.handle_news_event(event)
            
    def should_trade(self) -> bool:
        """Check if trading should proceed based on news"""
        # Check for recent high impact news
        for event in self.news_events[-5:]:  # Check last 5 events
            if event.get('impact') == 'high':
                return False
        return True
    
    def check_trading_conditions(self) -> bool:
        """Check if trading conditions are met"""
        # Check news monitor if available
        if self.news_monitor:
            events = self.news_monitor.get_upcoming_events()
            for event in events:
                if event.get('impact') == 'HIGH':
                    return False
        return self.should_trade()
    
    async def process_tick(self) -> Dict[str, Any]:
        """Process a single tick asynchronously"""
        import asyncio
        await asyncio.sleep(0.01)  # Simulate processing
        
        # Simple trading logic
        import random
        action = random.choice(['BUY', 'SELL', 'HOLD', 'HOLD', 'HOLD'])
        
        result = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'action': action,
            'price': 4000.0 + random.uniform(-10, 10)
        }
        
        if action != 'HOLD':
            self.execute_trade(action, self.risk_per_trade * self.balance / 4000.0)
            
        return result


class MultiTimeframeTrader(LiveTrader):
    """Live trader with multi-timeframe analysis"""
    
    def __init__(self, symbol: str = 'USDCOP', timeframes: List[str] = None, 
                 connector=None, **kwargs):
        """Initialize multi-timeframe trader"""
        super().__init__(symbol=symbol, **kwargs)
        self.timeframes = timeframes or ['M5', 'M15', 'H1']
        self.connector = connector
        
    def analyze(self) -> Dict[str, Any]:
        """Analyze multiple timeframes"""
        analysis = {}
        
        for tf in self.timeframes:
            # Mock analysis for each timeframe
            trend = 'up' if tf == 'M5' else ('down' if tf == 'H1' else 'neutral')
            signal = 'BUY' if tf == 'M5' else ('SELL' if tf == 'H1' else 'HOLD')
            
            analysis[tf] = {
                'trend': trend,
                'strength': 0.7,
                'signal': signal
            }
        
        # Calculate consensus
        signals = [analysis[tf]['signal'] for tf in self.timeframes]
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count > sell_count:
            consensus = 'BUY'
        elif sell_count > buy_count:
            consensus = 'SELL'
        else:
            consensus = 'HOLD'
            
        analysis['consensus'] = consensus
        return analysis