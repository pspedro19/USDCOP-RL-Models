"""
Kafka Data Processor and Dashboard Updater
==========================================
Consumes market data from Kafka, processes it, and updates the dashboard.
"""

import json
import logging
import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from collections import deque
import pandas as pd
import numpy as np

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MarketDataProcessor:
    """Processes market data and calculates indicators"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.tick_window = deque(maxlen=window_size)
        self.bar_window = deque(maxlen=window_size)
        
        # Calculated metrics
        self.current_price = None
        self.bid_ask_spread = None
        self.volume_profile = {}
        self.indicators = {}
        
    def process_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Process tick data and calculate metrics"""
        self.tick_window.append(tick)
        
        # Update current price and spread
        self.current_price = tick.get('last', tick.get('bid', 0))
        self.bid_ask_spread = tick.get('ask', 0) - tick.get('bid', 0)
        
        # Calculate tick metrics
        metrics = {
            'price': self.current_price,
            'spread': self.bid_ask_spread,
            'spread_percentage': (self.bid_ask_spread / self.current_price * 100) if self.current_price else 0,
            'timestamp': tick.get('timestamp')
        }
        
        # Calculate rolling statistics
        if len(self.tick_window) > 1:
            prices = [t.get('last', t.get('bid', 0)) for t in self.tick_window]
            metrics['tick_volatility'] = np.std(prices)
            metrics['tick_momentum'] = prices[-1] - prices[-10] if len(prices) >= 10 else 0
        
        return metrics
    
    def process_bar(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """Process bar data and calculate indicators"""
        self.bar_window.append(bar)
        
        # Convert to DataFrame for indicator calculation
        df = pd.DataFrame(list(self.bar_window))
        
        indicators = {}
        
        if len(df) >= 2:
            # Price action
            indicators['price_change'] = df['close'].iloc[-1] - df['close'].iloc[-2]
            indicators['price_change_pct'] = (indicators['price_change'] / df['close'].iloc[-2] * 100)
            
        if len(df) >= 14:
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            indicators['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            
        if len(df) >= 20:
            # Moving averages
            indicators['sma_20'] = df['close'].rolling(20).mean().iloc[-1]
            indicators['ema_20'] = df['close'].ewm(span=20).mean().iloc[-1]
            
            # Bollinger Bands
            sma = df['close'].rolling(20).mean()
            std = df['close'].rolling(20).std()
            indicators['bb_upper'] = (sma + 2 * std).iloc[-1]
            indicators['bb_lower'] = (sma - 2 * std).iloc[-1]
            indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
            
        if len(df) >= 26:
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            indicators['macd'] = (ema_12 - ema_26).iloc[-1]
            indicators['macd_signal'] = (ema_12 - ema_26).ewm(span=9).mean().iloc[-1]
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Volume analysis
        if 'volume' in df.columns:
            indicators['volume'] = df['volume'].iloc[-1]
            if len(df) >= 20:
                indicators['volume_sma'] = df['volume'].rolling(20).mean().iloc[-1]
                indicators['volume_ratio'] = indicators['volume'] / indicators['volume_sma']
        
        self.indicators = indicators
        
        return {
            'symbol': bar.get('symbol'),
            'timeframe': bar.get('timeframe'),
            'ohlc': {
                'open': bar.get('open'),
                'high': bar.get('high'),
                'low': bar.get('low'),
                'close': bar.get('close'),
                'volume': bar.get('volume', 0)
            },
            'indicators': indicators,
            'timestamp': bar.get('timestamp')
        }


class KafkaDataConsumer:
    """Consumes and processes data from Kafka"""
    
    def __init__(self, 
                 kafka_servers: str = "localhost:9092",
                 topics: List[str] = None,
                 group_id: str = "dashboard-consumer"):
        
        self.kafka_servers = kafka_servers
        self.topics = topics or ["market-data-ticks", "market-data-bars"]
        self.group_id = group_id
        
        self.consumer = None
        self.running = False
        self._stop_event = threading.Event()
        
        # Data processor
        self.processor = MarketDataProcessor()
        
        # Callbacks
        self.on_tick_processed: List[Callable] = []
        self.on_bar_processed: List[Callable] = []
        
    def start(self):
        """Start consuming from Kafka"""
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available")
            return False
        
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.kafka_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.running = True
            self._stop_event.clear()
            
            # Start consumer thread
            consumer_thread = threading.Thread(target=self._consume, daemon=True)
            consumer_thread.start()
            
            logger.info(f"Kafka consumer started for topics: {self.topics}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            return False
    
    def stop(self):
        """Stop consuming"""
        self.running = False
        self._stop_event.set()
        
        if self.consumer:
            self.consumer.close()
        
        logger.info("Kafka consumer stopped")
    
    def _consume(self):
        """Main consumer loop"""
        while self.running and not self._stop_event.is_set():
            try:
                # Poll for messages
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    topic = topic_partition.topic
                    
                    for record in records:
                        data = record.value
                        
                        if 'ticks' in topic:
                            # Process tick
                            processed = self.processor.process_tick(data)
                            
                            # Call callbacks
                            for callback in self.on_tick_processed:
                                try:
                                    callback(processed)
                                except Exception as e:
                                    logger.error(f"Tick callback error: {e}")
                                    
                        elif 'bars' in topic:
                            # Process bar
                            processed = self.processor.process_bar(data)
                            
                            # Call callbacks
                            for callback in self.on_bar_processed:
                                try:
                                    callback(processed)
                                except Exception as e:
                                    logger.error(f"Bar callback error: {e}")
                        
                        logger.debug(f"Processed message from {topic}")
                
            except Exception as e:
                logger.error(f"Error consuming from Kafka: {e}")
                self._stop_event.wait(1)
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for processed tick data"""
        self.on_tick_processed.append(callback)
    
    def add_bar_callback(self, callback: Callable):
        """Add callback for processed bar data"""
        self.on_bar_processed.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current market metrics"""
        return {
            'price': self.processor.current_price,
            'spread': self.processor.bid_ask_spread,
            'indicators': self.processor.indicators,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


class DashboardWebSocketServer:
    """WebSocket server for dashboard updates"""
    
    def __init__(self, consumer: KafkaDataConsumer, port: int = 8000):
        self.consumer = consumer
        self.port = port
        self.clients = set()
        
        # Register callbacks
        consumer.add_tick_callback(self.broadcast_tick)
        consumer.add_bar_callback(self.broadcast_bar)
    
    async def start(self):
        """Start WebSocket server"""
        if not WS_AVAILABLE:
            logger.error("WebSockets not available")
            return
        
        async def handler(websocket, path):
            # Add client
            self.clients.add(websocket)
            logger.info(f"Dashboard client connected from {websocket.remote_address}")
            
            # Send initial data
            await websocket.send(json.dumps({
                'type': 'initial',
                'data': self.consumer.get_current_metrics()
            }))
            
            try:
                # Keep connection alive
                await websocket.wait_closed()
            finally:
                # Remove client
                self.clients.remove(websocket)
                logger.info(f"Dashboard client disconnected")
        
        # Start server
        async with websockets.serve(handler, "localhost", self.port):
            logger.info(f"Dashboard WebSocket server started on port {self.port}")
            await asyncio.Future()  # Run forever
    
    def broadcast_tick(self, data: Dict):
        """Broadcast tick data to all clients"""
        asyncio.create_task(self._broadcast({
            'type': 'tick',
            'data': data
        }))
    
    def broadcast_bar(self, data: Dict):
        """Broadcast bar data to all clients"""
        asyncio.create_task(self._broadcast({
            'type': 'bar',
            'data': data
        }))
    
    async def _broadcast(self, message: Dict):
        """Send message to all connected clients"""
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_str)
            except Exception:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Kafka Data Processor and Dashboard Server")
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topics', nargs='+', help='Kafka topics to consume')
    parser.add_argument('--ws-port', type=int, default=8000, help='WebSocket server port')
    parser.add_argument('--group-id', default='dashboard-consumer', help='Kafka consumer group ID')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    )
    
    # Create consumer
    topics = args.topics or ["market-data-ticks", "market-data-bars"]
    consumer = KafkaDataConsumer(
        kafka_servers=args.kafka_servers,
        topics=topics,
        group_id=args.group_id
    )
    
    # Create WebSocket server
    ws_server = DashboardWebSocketServer(consumer, port=args.ws_port)
    
    try:
        # Start consumer
        if consumer.start():
            logger.info("Data processor started")
            
            # Run WebSocket server
            asyncio.run(ws_server.start())
        else:
            logger.error("Failed to start data processor")
            
    except KeyboardInterrupt:
        logger.info("Stopping data processor...")
        consumer.stop()


if __name__ == "__main__":
    main()