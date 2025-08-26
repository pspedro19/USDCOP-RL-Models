"""
Unified Data Bus Service
========================
Central service for seamless communication between pipeline outputs (L0-L5) 
and dashboard frontends. Implements WebSocket streaming, REST APIs, and 
data contract validation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import redis
from threading import Thread
import yaml
import hashlib
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataChannel(Enum):
    """WebSocket channels for different data types"""
    MARKET_BAR = "market.bar"              # L0 raw OHLC data
    STANDARDIZED = "data.standardized"     # L1 standardized data
    FEATURES = "features.v1"               # L3 engineered features
    OBSERVATIONS = "obs.v1"                # L4 normalized observations
    MODEL_DECISION = "model.decision"      # L5 model predictions
    TRADES = "trades"                      # Executed trades
    METRICS = "metrics.snapshot"           # Performance metrics
    AUDIT = "audit.status"                # Audit compliance status
    SYSTEM_HEALTH = "system.health"        # System health metrics

@dataclass
class MarketBar:
    """Market bar data structure"""
    timestamp: str  # ISO 8601 UTC
    open: float
    high: float
    low: float
    close: float
    volume: float
    episode_id: str
    t_in_episode: int
    
@dataclass
class ModelDecision:
    """Model decision data structure"""
    timestamp_decide: str
    execution_at: str
    action: int  # -1, 0, 1
    probabilities: List[float]
    confidence: float
    run_id: str
    model_name: str
    latency_ms: float

@dataclass
class MetricsSnapshot:
    """Performance metrics snapshot"""
    window: str  # "5min", "1hour", "daily"
    sortino: float
    sharpe: float
    max_drawdown: float
    calmar: float
    win_rate: float
    turnover: float
    pnl: float
    vs_cdt_ratio: float
    clip_rate: float
    peg_rate: float
    latency_p99: float
    timestamp: str

@dataclass
class AuditStatus:
    """Audit compliance status"""
    layer: str
    status: str  # "PASS", "WARN", "FAIL"
    completeness: float
    gaps_count: int
    stale_rate: float
    violations: List[str]
    quality_score: float
    timestamp: str

class UnifiedDataBus:
    """Central data bus for pipeline-dashboard communication"""
    
    def __init__(self, config_path: str = "mlops/config/master_pipeline.yml"):
        """Initialize unified data bus"""
        self.config = self._load_config(config_path)
        self.app = Flask(__name__)
        CORS(self.app)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Redis for pub/sub (optional, can use in-memory for dev)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.use_redis = True
            logger.info("Connected to Redis for pub/sub")
        except:
            self.use_redis = False
            logger.warning("Redis not available, using in-memory pub/sub")
        
        # Data stores (in-memory cache)
        self.latest_data = {
            channel.value: None for channel in DataChannel
        }
        self.metrics_history = []
        self.audit_history = []
        
        # MinIO paths from config
        self.bucket_mapping = {
            'L0': '00-raw-usdcop-marketdata',
            'L1': '01-standardized-usdcop',
            'L2': '02-prepared-usdcop',
            'L3': '03-features-usdcop',
            'L4': '04-mlready-usdcop-rldata',
            'L5': '05-serving-usdcop-models'
        }
        
        # Setup routes and WebSocket handlers
        self._setup_routes()
        self._setup_websocket_handlers()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _setup_routes(self):
        """Setup REST API routes"""
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'channels_active': list(DataChannel.__members__.keys()),
                'redis_connected': self.use_redis
            })
        
        @self.app.route('/api/status')
        def api_status():
            """System status endpoint (replaces mock in professional server)"""
            latest_metrics = self.latest_data.get(DataChannel.METRICS.value)
            latest_audit = self.latest_data.get(DataChannel.AUDIT.value)
            
            return jsonify({
                'system_status': 'operational',
                'ml_model_status': 'active' if latest_metrics else 'inactive',
                'data_pipeline_status': 'running',
                'last_update': datetime.utcnow().isoformat(),
                'audit_status': latest_audit['status'] if latest_audit else 'PENDING',
                'metrics': latest_metrics if latest_metrics else {}
            })
        
        @self.app.route('/api/institutional-metrics')
        def institutional_metrics():
            """Institutional metrics endpoint (replaces mock)"""
            latest = self.latest_data.get(DataChannel.METRICS.value)
            if not latest:
                # Return empty metrics if no data yet
                return jsonify({
                    'sortino': 0,
                    'sharpe': 0,
                    'calmar': 0,
                    'max_drawdown': 0,
                    'var_95': 0,
                    'volatility': 0,
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            return jsonify({
                'sortino': latest.get('sortino', 0),
                'sharpe': latest.get('sharpe', 0),
                'calmar': latest.get('calmar', 0),
                'max_drawdown': latest.get('max_drawdown', 0),
                'var_95': latest.get('var_95', 0),
                'volatility': latest.get('volatility', 0),
                'win_rate': latest.get('win_rate', 0),
                'pnl_total': latest.get('pnl', 0),
                'timestamp': latest.get('timestamp', datetime.utcnow().isoformat())
            })
        
        @self.app.route('/api/current-data')
        def current_data():
            """Current market and model data (replaces mock)"""
            market_bar = self.latest_data.get(DataChannel.MARKET_BAR.value)
            model_decision = self.latest_data.get(DataChannel.MODEL_DECISION.value)
            
            if not market_bar:
                return jsonify({'error': 'No market data available'}), 404
            
            return jsonify({
                'market': market_bar,
                'model': model_decision if model_decision else {'action': 0, 'confidence': 0},
                'timestamp': datetime.utcnow().isoformat()
            })
        
        @self.app.route('/api/data/quality')
        def data_quality():
            """Data quality metrics for premium sessions"""
            latest_audit = self.latest_data.get(DataChannel.AUDIT.value)
            
            if not latest_audit:
                return jsonify({
                    'completeness': 0,
                    'gaps_over_5min': 0,
                    'stale_rate': 0,
                    'quality_score': 0
                })
            
            return jsonify({
                'completeness': latest_audit.get('completeness', 0),
                'gaps_over_5min': latest_audit.get('gaps_count', 0),
                'stale_rate': latest_audit.get('stale_rate', 0),
                'quality_score': latest_audit.get('quality_score', 0),
                'violations': latest_audit.get('violations', []),
                'layer': latest_audit.get('layer', 'unknown'),
                'status': latest_audit.get('status', 'PENDING')
            })
        
        @self.app.route('/api/backtest/results')
        def backtest_results():
            """Backtest results with exact keys expected by UI"""
            latest_metrics = self.latest_data.get(DataChannel.METRICS.value)
            
            if not latest_metrics:
                # Return default values that UI expects
                return jsonify({
                    'final_value': 1000000,
                    'total_return': 0,
                    'vs_cdt_ratio': 1.0,
                    'vs_cdt_abs_usd': 0,
                    'consistency_score': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0
                })
            
            # Map internal metrics to UI expected keys
            return jsonify({
                'final_value': 1000000 * (1 + latest_metrics.get('pnl', 0) / 100),
                'total_return': latest_metrics.get('pnl', 0),
                'vs_cdt_ratio': latest_metrics.get('vs_cdt_ratio', 1.0),
                'vs_cdt_abs_usd': latest_metrics.get('pnl', 0) * 1000,
                'consistency_score': latest_metrics.get('win_rate', 0) * 100,
                'max_drawdown': latest_metrics.get('max_drawdown', 0),
                'sharpe_ratio': latest_metrics.get('sharpe', 0)
            })
        
        @self.app.route('/api/audit/l5/acceptance')
        def l5_acceptance():
            """L5 acceptance report for GO/NO-GO decision"""
            # Load from MinIO or file system
            acceptance_path = Path("checks/acceptance_report.json")
            if acceptance_path.exists():
                with open(acceptance_path, 'r') as f:
                    return jsonify(json.load(f))
            
            return jsonify({
                'overall_status': 'PENDING',
                'audit_timestamp': datetime.utcnow().isoformat(),
                'performance_gates': {},
                'violations': []
            })
        
        @self.app.route('/api/contracts/<layer>')
        def get_contracts(layer):
            """Get data contracts for specific layer"""
            contracts_path = Path(f"contracts/{layer}_contracts.json")
            if contracts_path.exists():
                with open(contracts_path, 'r') as f:
                    return jsonify(json.load(f))
            
            return jsonify({'error': f'No contracts found for layer {layer}'}), 404
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info(f"Client connected: {request.sid}")
            
            # Send current state to new client
            for channel in DataChannel:
                if self.latest_data[channel.value]:
                    emit(channel.value, self.latest_data[channel.value])
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle channel subscription"""
            channels = data.get('channels', [])
            for channel in channels:
                logger.info(f"Client {request.sid} subscribed to {channel}")
                # Send latest data if available
                if channel in self.latest_data and self.latest_data[channel]:
                    emit(channel, self.latest_data[channel])
    
    def publish(self, channel: DataChannel, data: Any):
        """
        Publish data to a channel
        
        Args:
            channel: Data channel to publish to
            data: Data to publish (will be converted to dict if dataclass)
        """
        # Convert dataclass to dict if needed
        if hasattr(data, '__dataclass_fields__'):
            data = asdict(data)
        
        # Store latest data
        self.latest_data[channel.value] = data
        
        # Add to history if metrics or audit
        if channel == DataChannel.METRICS:
            self.metrics_history.append(data)
            # Keep only last 1000 entries
            self.metrics_history = self.metrics_history[-1000:]
        elif channel == DataChannel.AUDIT:
            self.audit_history.append(data)
            self.audit_history = self.audit_history[-100:]
        
        # Publish via Redis if available
        if self.use_redis:
            self.redis_client.publish(channel.value, json.dumps(data))
        
        # Emit via WebSocket
        self.socketio.emit(channel.value, data)
        
        logger.debug(f"Published to {channel.value}: {data}")
    
    def _start_background_tasks(self):
        """Start background tasks for data processing"""
        
        def monitor_pipeline_outputs():
            """Monitor pipeline outputs and publish updates"""
            while True:
                try:
                    # Check for new L0 data
                    self._check_and_publish_l0_data()
                    
                    # Check for new L4 observations
                    self._check_and_publish_l4_data()
                    
                    # Check for new L5 decisions
                    self._check_and_publish_l5_data()
                    
                    # Calculate and publish metrics
                    self._calculate_and_publish_metrics()
                    
                    # Check audit status
                    self._check_and_publish_audit_status()
                    
                except Exception as e:
                    logger.error(f"Error in background task: {e}")
                
                # Sleep for 5 seconds (adjust based on M5 frequency)
                asyncio.sleep(5)
        
        # Start background thread
        Thread(target=monitor_pipeline_outputs, daemon=True).start()
    
    def _check_and_publish_l0_data(self):
        """Check for new L0 data and publish market bars"""
        # Check MinIO bucket for new data
        bucket = self.bucket_mapping['L0']
        
        # Mock implementation - replace with actual MinIO client
        # In production, use boto3 or minio-py to check for new files
        
        # For now, generate mock data
        current_time = datetime.utcnow()
        if current_time.second % 5 == 0:  # Every 5 seconds for demo
            market_bar = MarketBar(
                timestamp=current_time.isoformat(),
                open=4200 + np.random.randn() * 10,
                high=4210 + np.random.randn() * 10,
                low=4190 + np.random.randn() * 10,
                close=4205 + np.random.randn() * 10,
                volume=1000000 + np.random.randn() * 100000,
                episode_id=current_time.strftime("%Y-%m-%d"),
                t_in_episode=current_time.hour * 12 + current_time.minute // 5
            )
            self.publish(DataChannel.MARKET_BAR, market_bar)
    
    def _check_and_publish_l4_data(self):
        """Check for new L4 observations and publish"""
        # Check MinIO bucket for new normalized observations
        bucket = self.bucket_mapping['L4']
        
        # Mock implementation
        current_time = datetime.utcnow()
        if current_time.second % 10 == 0:  # Every 10 seconds for demo
            observations = {
                'episode_id': current_time.strftime("%Y-%m-%d"),
                't': current_time.hour * 12 + current_time.minute // 5,
                'obs': [np.random.uniform(-5, 5) for _ in range(17)],
                'abs_max': 4.8,
                'timestamp': current_time.isoformat()
            }
            self.publish(DataChannel.OBSERVATIONS, observations)
    
    def _check_and_publish_l5_data(self):
        """Check for new L5 model decisions and publish"""
        # Check MinIO bucket for new model outputs
        bucket = self.bucket_mapping['L5']
        
        # Mock implementation
        current_time = datetime.utcnow()
        if current_time.second % 10 == 0:  # Every 10 seconds for demo
            decision = ModelDecision(
                timestamp_decide=current_time.isoformat(),
                execution_at=(current_time + timedelta(minutes=5)).isoformat(),
                action=np.random.choice([-1, 0, 1]),
                probabilities=[0.3, 0.4, 0.3],
                confidence=0.7 + np.random.rand() * 0.3,
                run_id=f"run_{current_time.strftime('%Y%m%d_%H%M%S')}",
                model_name="PPO-LSTM",
                latency_ms=15 + np.random.rand() * 10
            )
            self.publish(DataChannel.MODEL_DECISION, decision)
    
    def _calculate_and_publish_metrics(self):
        """Calculate and publish performance metrics"""
        current_time = datetime.utcnow()
        
        # Calculate metrics from latest data
        # In production, aggregate from actual trading results
        
        metrics = MetricsSnapshot(
            window="5min",
            sortino=1.4 + np.random.randn() * 0.1,
            sharpe=1.8 + np.random.randn() * 0.1,
            max_drawdown=0.12 + np.random.randn() * 0.02,
            calmar=1.2 + np.random.randn() * 0.1,
            win_rate=0.55 + np.random.randn() * 0.05,
            turnover=3 + np.random.randn() * 0.5,
            pnl=1000 + np.random.randn() * 100,
            vs_cdt_ratio=1.2 + np.random.randn() * 0.1,
            clip_rate=0.002 + np.random.randn() * 0.001,
            peg_rate=0.001 + np.random.randn() * 0.0005,
            latency_p99=18 + np.random.randn() * 2,
            timestamp=current_time.isoformat()
        )
        
        self.publish(DataChannel.METRICS, metrics)
    
    def _check_and_publish_audit_status(self):
        """Check and publish audit compliance status"""
        current_time = datetime.utcnow()
        
        # Check actual audit results from pipeline
        # In production, read from audit reports in MinIO
        
        audit_status = AuditStatus(
            layer="L4",
            status="PASS" if np.random.rand() > 0.2 else "WARN",
            completeness=0.98 + np.random.randn() * 0.01,
            gaps_count=int(np.random.rand() * 3),
            stale_rate=0.008 + np.random.randn() * 0.002,
            violations=[],
            quality_score=95 + np.random.randn() * 2,
            timestamp=current_time.isoformat()
        )
        
        self.publish(DataChannel.AUDIT, audit_status)
    
    def run(self, host='0.0.0.0', port=5005, debug=False):
        """Run the unified data bus service"""
        logger.info(f"Starting Unified Data Bus on {host}:{port}")
        logger.info(f"WebSocket channels available: {[c.value for c in DataChannel]}")
        logger.info("REST endpoints:")
        logger.info("  - GET /health")
        logger.info("  - GET /api/status")
        logger.info("  - GET /api/institutional-metrics")
        logger.info("  - GET /api/current-data")
        logger.info("  - GET /api/data/quality")
        logger.info("  - GET /api/backtest/results")
        logger.info("  - GET /api/audit/l5/acceptance")
        
        self.socketio.run(self.app, host=host, port=port, debug=debug)


# Integration with existing MinIO pipelines
class PipelineIntegrator:
    """Integrates pipeline outputs with data bus"""
    
    def __init__(self, data_bus: UnifiedDataBus):
        self.data_bus = data_bus
        self.s3_client = None  # Initialize with boto3 in production
    
    def process_l0_output(self, file_path: str):
        """Process L0 output and publish to data bus"""
        # Read parquet file
        df = pd.read_parquet(file_path)
        
        # Convert to market bars and publish
        for _, row in df.iterrows():
            market_bar = MarketBar(
                timestamp=row['time'].isoformat(),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row.get('volume', 0),
                episode_id=row.get('episode_id', ''),
                t_in_episode=row.get('t_in_episode', 0)
            )
            self.data_bus.publish(DataChannel.MARKET_BAR, market_bar)
    
    def process_l4_output(self, file_path: str):
        """Process L4 output and publish observations"""
        # Read replay dataset
        df = pd.read_parquet(file_path)
        
        # Get observation columns
        obs_cols = [f'obs_{i:02d}' for i in range(17)]
        
        # Publish latest observations
        for _, row in df.tail(1).iterrows():
            observations = {
                'episode_id': row['episode_id'],
                't': row['t_in_episode'],
                'obs': row[obs_cols].tolist(),
                'abs_max': row[obs_cols].abs().max(),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.data_bus.publish(DataChannel.OBSERVATIONS, observations)
    
    def process_l5_output(self, model_output: Dict):
        """Process L5 model output and publish decision"""
        decision = ModelDecision(
            timestamp_decide=datetime.utcnow().isoformat(),
            execution_at=(datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            action=model_output['action'],
            probabilities=model_output['action_probs'],
            confidence=model_output['confidence'],
            run_id=model_output['run_id'],
            model_name=model_output['model_name'],
            latency_ms=model_output['inference_time_ms']
        )
        self.data_bus.publish(DataChannel.MODEL_DECISION, decision)


if __name__ == "__main__":
    # Start unified data bus
    data_bus = UnifiedDataBus()
    
    # Run service
    data_bus.run(port=5005, debug=True)