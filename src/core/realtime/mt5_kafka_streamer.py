"""
Real-Time MT5 to Kafka Data Streamer
=====================================
Streams real-time market data from MT5 to Kafka for processing and visualization.
"""

import json
import logging
import asyncio
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, asdict
import pandas as pd

try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from src.core.connectors.mt5_connector import RobustMT5Connector, ConnectorConfig

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for real-time streaming"""
    symbol: str = "USDCOP.r"
    timeframe: str = "M1"
    kafka_topic: str = "market-data"
    kafka_bootstrap_servers: str = "localhost:9092"
    
    # Stream settings
    stream_ticks: bool = True
    stream_bars: bool = True
    tick_interval_ms: int = 500
    bar_interval_sec: int = 5
    
    # Data persistence
    persist_to_db: bool = True
    db_connection_string: str = "sqlite:///market_data.db"
    
    # Dashboard updates
    enable_dashboard: bool = True
    dashboard_ws_url: str = "ws://localhost:8000/ws"


class MT5KafkaStreamer:
    """Streams real-time MT5 data to Kafka and other destinations"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.running = False
        self._stop_event = threading.Event()
        
        # Initialize MT5 connector
        self.mt5_connector = RobustMT5Connector()
        
        # Initialize Kafka producer
        self.kafka_producer = None
        if KAFKA_AVAILABLE:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None
                )
                logger.info("Kafka producer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka producer: {e}")
        
        # Data storage
        self.tick_buffer = []
        self.bar_buffer = []
        self.last_tick = None
        self.last_bar = None
        
        # Callbacks
        self.on_tick_callbacks: List[Callable] = []
        self.on_bar_callbacks: List[Callable] = []
        
    def start(self) -> bool:
        """Start streaming real-time data"""
        if self.running:
            logger.warning("Streamer already running")
            return False
        
        # Connect to MT5
        if not self.mt5_connector.connect():
            logger.error("Failed to connect to MT5")
            return False
        
        self.running = True
        self._stop_event.clear()
        
        # Start streaming threads
        if self.config.stream_ticks:
            tick_thread = threading.Thread(target=self._stream_ticks, daemon=True)
            tick_thread.start()
            logger.info("Tick streaming started")
        
        if self.config.stream_bars:
            bar_thread = threading.Thread(target=self._stream_bars, daemon=True)
            bar_thread.start()
            logger.info("Bar streaming started")
        
        # Start data persistence thread
        if self.config.persist_to_db:
            persist_thread = threading.Thread(target=self._persist_data, daemon=True)
            persist_thread.start()
            logger.info("Data persistence started")
        
        logger.info(f"MT5 Kafka Streamer started for {self.config.symbol}")
        return True
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        self._stop_event.set()
        
        # Disconnect from MT5
        self.mt5_connector.disconnect()
        
        # Close Kafka producer
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        
        logger.info("MT5 Kafka Streamer stopped")
    
    def _stream_ticks(self):
        """Stream tick data"""
        while self.running and not self._stop_event.is_set():
            try:
                # Get tick from MT5
                tick = self.mt5_connector.get_tick(self.config.symbol)
                
                if tick and tick != self.last_tick:
                    # Add timestamp
                    tick['timestamp'] = datetime.now(timezone.utc).isoformat()
                    tick['symbol'] = self.config.symbol
                    
                    # Send to Kafka
                    self._send_to_kafka('ticks', tick)
                    
                    # Store in buffer
                    self.tick_buffer.append(tick)
                    if len(self.tick_buffer) > 1000:
                        self.tick_buffer = self.tick_buffer[-500:]
                    
                    # Call callbacks
                    for callback in self.on_tick_callbacks:
                        try:
                            callback(tick)
                        except Exception as e:
                            logger.error(f"Tick callback error: {e}")
                    
                    self.last_tick = tick
                    logger.debug(f"Tick: {tick}")
                
                # Wait for next tick
                self._stop_event.wait(self.config.tick_interval_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error streaming ticks: {e}")
                self._stop_event.wait(1)
    
    def _stream_bars(self):
        """Stream bar data"""
        while self.running and not self._stop_event.is_set():
            try:
                # Get latest bar from MT5
                bars = self.mt5_connector.get_latest_rates(
                    self.config.symbol, 
                    self.config.timeframe, 
                    count=1
                )
                
                if not bars.empty:
                    bar = bars.iloc[-1].to_dict()
                    
                    if bar != self.last_bar:
                        # Add metadata
                        bar['timestamp'] = datetime.now(timezone.utc).isoformat()
                        bar['symbol'] = self.config.symbol
                        bar['timeframe'] = self.config.timeframe
                        
                        # Send to Kafka
                        self._send_to_kafka('bars', bar)
                        
                        # Store in buffer
                        self.bar_buffer.append(bar)
                        if len(self.bar_buffer) > 500:
                            self.bar_buffer = self.bar_buffer[-250:]
                        
                        # Call callbacks
                        for callback in self.on_bar_callbacks:
                            try:
                                callback(bar)
                            except Exception as e:
                                logger.error(f"Bar callback error: {e}")
                        
                        self.last_bar = bar
                        logger.debug(f"Bar: {bar}")
                
                # Wait for next bar
                self._stop_event.wait(self.config.bar_interval_sec)
                
            except Exception as e:
                logger.error(f"Error streaming bars: {e}")
                self._stop_event.wait(5)
    
    def _send_to_kafka(self, data_type: str, data: Dict[str, Any]):
        """Send data to Kafka"""
        if not self.kafka_producer:
            return
        
        try:
            topic = f"{self.config.kafka_topic}-{data_type}"
            key = f"{self.config.symbol}:{data_type}"
            
            self.kafka_producer.send(topic, key=key, value=data)
            logger.debug(f"Sent to Kafka topic {topic}: {data}")
            
        except Exception as e:
            logger.error(f"Failed to send to Kafka: {e}")
    
    def _persist_data(self):
        """Persist data to database"""
        import sqlite3
        
        # Extract database path from connection string
        db_path = self.config.db_connection_string.replace("sqlite:///", "")
        
        while self.running and not self._stop_event.is_set():
            try:
                # Save tick buffer
                if self.tick_buffer:
                    df_ticks = pd.DataFrame(self.tick_buffer)
                    with sqlite3.connect(db_path) as conn:
                        df_ticks.to_sql('ticks', conn, if_exists='append', index=False)
                    logger.info(f"Persisted {len(self.tick_buffer)} ticks")
                    self.tick_buffer.clear()
                
                # Save bar buffer
                if self.bar_buffer:
                    df_bars = pd.DataFrame(self.bar_buffer)
                    with sqlite3.connect(db_path) as conn:
                        df_bars.to_sql('bars', conn, if_exists='append', index=False)
                    logger.info(f"Persisted {len(self.bar_buffer)} bars")
                    self.bar_buffer.clear()
                
                # Wait before next persistence
                self._stop_event.wait(30)
                
            except Exception as e:
                logger.error(f"Error persisting data: {e}")
                self._stop_event.wait(10)
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick updates"""
        self.on_tick_callbacks.append(callback)
    
    def add_bar_callback(self, callback: Callable):
        """Add callback for bar updates"""
        self.on_bar_callbacks.append(callback)
    
    def get_recent_ticks(self, count: int = 100) -> List[Dict]:
        """Get recent ticks from buffer"""
        return self.tick_buffer[-count:]
    
    def get_recent_bars(self, count: int = 50) -> List[Dict]:
        """Get recent bars from buffer"""
        return self.bar_buffer[-count:]


class DashboardConnector:
    """Connects streamer to dashboard via WebSocket"""
    
    def __init__(self, streamer: MT5KafkaStreamer, ws_url: str):
        self.streamer = streamer
        self.ws_url = ws_url
        self.ws_client = None
        
        # Register callbacks
        streamer.add_tick_callback(self.on_tick)
        streamer.add_bar_callback(self.on_bar)
    
    async def connect(self):
        """Connect to dashboard WebSocket"""
        try:
            import websockets
            self.ws_client = await websockets.connect(self.ws_url)
            logger.info(f"Connected to dashboard at {self.ws_url}")
        except Exception as e:
            logger.error(f"Failed to connect to dashboard: {e}")
    
    async def disconnect(self):
        """Disconnect from dashboard"""
        if self.ws_client:
            await self.ws_client.close()
    
    def on_tick(self, tick: Dict):
        """Send tick to dashboard"""
        asyncio.create_task(self._send_to_dashboard('tick', tick))
    
    def on_bar(self, bar: Dict):
        """Send bar to dashboard"""
        asyncio.create_task(self._send_to_dashboard('bar', bar))
    
    async def _send_to_dashboard(self, data_type: str, data: Dict):
        """Send data to dashboard via WebSocket"""
        if not self.ws_client:
            return
        
        try:
            message = {
                'type': data_type,
                'data': data,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            await self.ws_client.send(json.dumps(message))
            logger.debug(f"Sent to dashboard: {data_type}")
        except Exception as e:
            logger.error(f"Failed to send to dashboard: {e}")


def main():
    """Main entry point for real-time streaming"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MT5 to Kafka Real-Time Streamer")
    parser.add_argument('--symbol', default='USDCOP.r', help='Symbol to stream')
    parser.add_argument('--timeframe', default='M1', help='Timeframe for bars')
    parser.add_argument('--kafka-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--kafka-topic', default='market-data', help='Kafka topic prefix')
    parser.add_argument('--no-ticks', action='store_true', help='Disable tick streaming')
    parser.add_argument('--no-bars', action='store_true', help='Disable bar streaming')
    parser.add_argument('--no-persist', action='store_true', help='Disable data persistence')
    parser.add_argument('--dashboard-url', default='ws://localhost:8000/ws', help='Dashboard WebSocket URL')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    )
    
    # Create configuration
    config = StreamConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        kafka_bootstrap_servers=args.kafka_servers,
        kafka_topic=args.kafka_topic,
        stream_ticks=not args.no_ticks,
        stream_bars=not args.no_bars,
        persist_to_db=not args.no_persist,
        dashboard_ws_url=args.dashboard_url
    )
    
    # Create and start streamer
    streamer = MT5KafkaStreamer(config)
    
    # Connect to dashboard if enabled
    if config.enable_dashboard:
        dashboard = DashboardConnector(streamer, config.dashboard_ws_url)
        asyncio.create_task(dashboard.connect())
    
    try:
        # Start streaming
        if streamer.start():
            logger.info("Streaming started. Press Ctrl+C to stop.")
            
            # Keep running
            while True:
                import time
                time.sleep(1)
        else:
            logger.error("Failed to start streaming")
            
    except KeyboardInterrupt:
        logger.info("Stopping streamer...")
        streamer.stop()
        
        if config.enable_dashboard:
            asyncio.create_task(dashboard.disconnect())


if __name__ == "__main__":
    main()