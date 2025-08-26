#!/usr/bin/env python
"""
Start Real-Time Data Pipeline
==============================
Starts the complete real-time data pipeline: MT5 → Kafka → Dashboard
"""

import os
import sys
import time
import logging
import subprocess
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.realtime.mt5_kafka_streamer import MT5KafkaStreamer, StreamConfig
from src.core.realtime.kafka_data_processor import KafkaDataConsumer, DashboardWebSocketServer
import asyncio

logger = logging.getLogger(__name__)


def check_kafka():
    """Check if Kafka is running"""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(bootstrap_servers='localhost:9092')
        producer.close()
        return True
    except:
        return False


def start_kafka():
    """Start Kafka using docker-compose"""
    logger.info("Starting Kafka...")
    
    compose_file = project_root / "docker-compose.yml"
    if not compose_file.exists():
        logger.error("docker-compose.yml not found")
        return False
    
    try:
        subprocess.run(["docker-compose", "up", "-d", "zookeeper", "kafka"], 
                      cwd=project_root, check=True)
        
        # Wait for Kafka to be ready
        for i in range(30):
            if check_kafka():
                logger.info("Kafka is ready")
                return True
            time.sleep(2)
        
        logger.error("Kafka failed to start")
        return False
        
    except Exception as e:
        logger.error(f"Failed to start Kafka: {e}")
        return False


def run_mt5_streamer():
    """Run MT5 to Kafka streamer"""
    config = StreamConfig(
        symbol=os.getenv("TRADING_SYMBOL", "USDCOP.r"),
        timeframe="M1",
        kafka_topic="market-data",
        kafka_bootstrap_servers="localhost:9092",
        stream_ticks=True,
        stream_bars=True,
        persist_to_db=True,
        enable_dashboard=False  # We'll handle dashboard separately
    )
    
    streamer = MT5KafkaStreamer(config)
    
    # Add console output for debugging
    def print_tick(tick):
        logger.info(f"Tick: {tick.get('symbol')} - Bid: {tick.get('bid'):.2f} Ask: {tick.get('ask'):.2f}")
    
    def print_bar(bar):
        logger.info(f"Bar: {bar.get('symbol')} {bar.get('timeframe')} - OHLC: {bar.get('open'):.2f}/{bar.get('high'):.2f}/{bar.get('low'):.2f}/{bar.get('close'):.2f}")
    
    streamer.add_tick_callback(print_tick)
    streamer.add_bar_callback(print_bar)
    
    if streamer.start():
        logger.info("MT5 Streamer started successfully")
        
        # Keep running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            streamer.stop()
    else:
        logger.error("Failed to start MT5 Streamer")


def run_data_processor():
    """Run Kafka consumer and data processor"""
    consumer = KafkaDataConsumer(
        kafka_servers="localhost:9092",
        topics=["market-data-ticks", "market-data-bars"],
        group_id="realtime-processor"
    )
    
    # Create WebSocket server for dashboard
    ws_server = DashboardWebSocketServer(consumer, port=8765)
    
    # Add console output
    def print_processed_tick(data):
        logger.info(f"Processed Tick - Price: {data.get('price'):.2f} Spread: {data.get('spread'):.4f}")
    
    def print_processed_bar(data):
        indicators = data.get('indicators', {})
        logger.info(f"Processed Bar - RSI: {indicators.get('rsi', 0):.2f} MACD: {indicators.get('macd', 0):.4f}")
    
    consumer.add_tick_callback(print_processed_tick)
    consumer.add_bar_callback(print_processed_bar)
    
    if consumer.start():
        logger.info("Data Processor started successfully")
        
        # Run WebSocket server
        try:
            asyncio.run(ws_server.start())
        except KeyboardInterrupt:
            consumer.stop()
    else:
        logger.error("Failed to start Data Processor")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Real-Time Data Pipeline")
    parser.add_argument('--component', choices=['all', 'streamer', 'processor', 'kafka'], 
                       default='all', help='Component to start')
    parser.add_argument('--symbol', default='USDCOP.r', help='Symbol to stream')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
    )
    
    # Set symbol in environment
    os.environ["TRADING_SYMBOL"] = args.symbol
    
    try:
        if args.component in ['all', 'kafka']:
            # Ensure Kafka is running
            if not check_kafka():
                logger.info("Kafka not running, starting it...")
                if not start_kafka():
                    logger.error("Cannot proceed without Kafka")
                    return
        
        if args.component == 'streamer':
            # Run only MT5 streamer
            run_mt5_streamer()
            
        elif args.component == 'processor':
            # Run only data processor
            run_data_processor()
            
        elif args.component == 'all':
            # Run both components in separate threads
            logger.info("Starting complete real-time pipeline...")
            
            # Start MT5 streamer in thread
            streamer_thread = threading.Thread(target=run_mt5_streamer, daemon=True)
            streamer_thread.start()
            
            # Give streamer time to start
            time.sleep(5)
            
            # Run data processor in main thread
            run_data_processor()
            
    except KeyboardInterrupt:
        logger.info("Shutting down real-time pipeline...")
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)


if __name__ == "__main__":
    main()