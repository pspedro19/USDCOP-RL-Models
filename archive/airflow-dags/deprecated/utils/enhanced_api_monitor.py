"""
Enhanced API Usage Monitor
=========================
Comprehensive monitoring system for TwelveData and other external APIs.

Features:
- Real-time rate limit tracking
- Cost monitoring and projection
- Health status monitoring
- Alert system for approaching limits
- Performance metrics tracking
- Multi-API support with unified interface
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sqlite3
from dataclasses import dataclass, asdict
import time
import requests
import threading
from collections import defaultdict, deque
import statistics

@dataclass
class APICall:
    """Represents a single API call record"""
    timestamp: datetime
    api_name: str
    endpoint: str
    key_id: str
    success: bool
    response_time_ms: int
    status_code: int
    cost: float
    error_message: Optional[str] = None
    response_size: Optional[int] = None

@dataclass
class APIKeyStatus:
    """Current status of an API key"""
    key_id: str
    api_name: str
    daily_calls: int
    monthly_calls: int
    daily_limit: int
    monthly_limit: int
    daily_cost: float
    monthly_cost: float
    last_used: Optional[datetime]
    status: str  # ACTIVE, RATE_LIMITED, EXPIRED, ERROR, WARNING
    error_count: int
    success_rate: float
    avg_response_time: float
    plan_type: str  # free, basic, pro, premium

@dataclass
class APIHealthMetrics:
    """Health metrics for an API"""
    api_name: str
    total_calls_today: int
    success_rate: float
    avg_response_time: float
    error_rate: float
    rate_limit_hits: int
    cost_today: float
    estimated_monthly_cost: float
    peak_usage_hour: int
    current_rps: float  # requests per second
    uptime_percentage: float

class EnhancedAPIMonitor:
    """Enhanced API monitoring system with comprehensive tracking"""
    
    # API configurations
    API_CONFIGS = {
        'twelvedata': {
            'base_url': 'https://api.twelvedata.com',
            'rate_limits': {
                'free': {'daily': 800, 'monthly': 800, 'rps': 8},
                'basic': {'daily': 5000, 'monthly': 150000, 'rps': 8},
                'pro': {'daily': 20000, 'monthly': 600000, 'rps': 8},
                'premium': {'daily': 50000, 'monthly': 1500000, 'rps': 8}
            },
            'costs': {
                'quote': 0.001,
                'time_series': 0.002,
                'real_time': 0.005,
                'technical_indicators': 0.003,
                'fundamentals': 0.010,
                'earnings': 0.015,
                'dividends': 0.010
            }
        },
        'alpha_vantage': {
            'base_url': 'https://www.alphavantage.co',
            'rate_limits': {
                'free': {'daily': 500, 'monthly': 500, 'rps': 5},
                'premium': {'daily': 75000, 'monthly': 75000, 'rps': 30}
            },
            'costs': {
                'quote': 0.002,
                'time_series': 0.003,
                'indicators': 0.004
            }
        }
    }
    
    def __init__(self, db_path: str = "/opt/airflow/data_cache/api_monitor.db"):
        """Initialize the enhanced API monitor"""
        self.db_path = db_path
        self.call_history: deque = deque(maxlen=100000)  # Keep last 100k calls in memory
        self.key_statuses: Dict[str, APIKeyStatus] = {}
        self.recent_calls: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_thresholds = {
            'rate_limit_warning': 0.85,  # Warn at 85% of rate limit
            'cost_warning': 100.0,  # Warn if daily cost exceeds $100
            'error_rate_warning': 0.1,  # Warn if error rate > 10%
            'response_time_warning': 5000  # Warn if avg response time > 5s
        }
        self._lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        self._load_existing_data()
        
        # Start background threads
        self._start_background_tasks()
        
        logging.info("Enhanced API Monitor initialized")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                api_name TEXT,
                endpoint TEXT,
                key_id TEXT,
                success INTEGER,
                response_time_ms INTEGER,
                status_code INTEGER,
                cost REAL,
                error_message TEXT,
                response_size INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_key_status (
                key_id TEXT PRIMARY KEY,
                api_name TEXT,
                daily_calls INTEGER,
                monthly_calls INTEGER,
                daily_limit INTEGER,
                monthly_limit INTEGER,
                daily_cost REAL,
                monthly_cost REAL,
                last_used REAL,
                status TEXT,
                error_count INTEGER,
                success_rate REAL,
                avg_response_time REAL,
                plan_type TEXT,
                updated_at REAL
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_calls_timestamp ON api_calls(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_calls_api_name ON api_calls(api_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_calls_key_id ON api_calls(key_id)')
        
        conn.commit()
        conn.close()
    
    def _load_existing_data(self):
        """Load existing data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load recent API calls (last 24 hours)
            yesterday = time.time() - 86400
            cursor.execute(
                'SELECT * FROM api_calls WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 10000',
                (yesterday,)
            )
            
            for row in cursor.fetchall():
                call = APICall(
                    timestamp=datetime.fromtimestamp(row[1]),
                    api_name=row[2],
                    endpoint=row[3],
                    key_id=row[4],
                    success=bool(row[5]),
                    response_time_ms=row[6],
                    status_code=row[7],
                    cost=row[8],
                    error_message=row[9],
                    response_size=row[10]
                )
                self.call_history.append(call)
                self.recent_calls[row[4]].append(call)
            
            # Load API key statuses
            cursor.execute('SELECT * FROM api_key_status')
            for row in cursor.fetchall():
                status = APIKeyStatus(
                    key_id=row[0],
                    api_name=row[1],
                    daily_calls=row[2],
                    monthly_calls=row[3],
                    daily_limit=row[4],
                    monthly_limit=row[5],
                    daily_cost=row[6],
                    monthly_cost=row[7],
                    last_used=datetime.fromtimestamp(row[8]) if row[8] else None,
                    status=row[9],
                    error_count=row[10],
                    success_rate=row[11],
                    avg_response_time=row[12],
                    plan_type=row[13]
                )
                self.key_statuses[row[0]] = status
            
            conn.close()
            logging.info(f"Loaded {len(self.call_history)} recent API calls and {len(self.key_statuses)} key statuses")
            
        except Exception as e:
            logging.warning(f"Failed to load existing data: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def maintenance_worker():
            while True:
                try:
                    self._cleanup_old_data()
                    self._update_key_statuses()
                    self._check_alerts()
                    time.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logging.error(f"Maintenance worker error: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
    
    def record_api_call(
        self,
        api_name: str,
        endpoint: str,
        key_id: str,
        success: bool,
        response_time_ms: int,
        status_code: int,
        error_message: Optional[str] = None,
        response_size: Optional[int] = None
    ) -> None:
        """Record an API call for monitoring"""
        timestamp = datetime.utcnow()
        
        # Calculate cost
        cost = self._calculate_cost(api_name, endpoint)
        
        # Create call record
        call = APICall(
            timestamp=timestamp,
            api_name=api_name,
            endpoint=endpoint,
            key_id=key_id,
            success=success,
            response_time_ms=response_time_ms,
            status_code=status_code,
            cost=cost,
            error_message=error_message,
            response_size=response_size
        )
        
        with self._lock:
            self.call_history.append(call)
            self.recent_calls[key_id].append(call)
        
        # Persist to database
        self._persist_call(call)
        
        # Update key status
        self._update_key_status(key_id, call)
        
        # Check for immediate alerts
        self._check_immediate_alerts(key_id, call)
        
        logging.debug(f"Recorded API call: {api_name}/{endpoint} - {key_id} - {'SUCCESS' if success else 'FAILED'}")
    
    def _calculate_cost(self, api_name: str, endpoint: str) -> float:
        """Calculate the cost of an API call"""
        if api_name in self.API_CONFIGS:
            costs = self.API_CONFIGS[api_name].get('costs', {})
            # Try exact match first, then partial match
            if endpoint in costs:
                return costs[endpoint]
            
            # Try partial matching for endpoints like 'time_series?symbol=...'
            base_endpoint = endpoint.split('?')[0].split('/')[-1]
            for cost_endpoint, cost in costs.items():
                if cost_endpoint in base_endpoint or base_endpoint in cost_endpoint:
                    return cost
            
            # Default cost if not found
            return 0.002
        return 0.0
    
    def _persist_call(self, call: APICall):
        """Persist API call to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_calls 
                (timestamp, api_name, endpoint, key_id, success, response_time_ms, 
                 status_code, cost, error_message, response_size)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                call.timestamp.timestamp(),
                call.api_name,
                call.endpoint,
                call.key_id,
                int(call.success),
                call.response_time_ms,
                call.status_code,
                call.cost,
                call.error_message,
                call.response_size
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.warning(f"Failed to persist API call: {e}")
    
    def _update_key_status(self, key_id: str, call: APICall):
        """Update the status of an API key based on a new call"""
        with self._lock:
            if key_id not in self.key_statuses:
                # Initialize new key status
                api_config = self.API_CONFIGS.get(call.api_name, {})
                rate_limits = api_config.get('rate_limits', {}).get('basic', {'daily': 5000, 'monthly': 150000})
                
                self.key_statuses[key_id] = APIKeyStatus(
                    key_id=key_id,
                    api_name=call.api_name,
                    daily_calls=0,
                    monthly_calls=0,
                    daily_limit=rate_limits['daily'],
                    monthly_limit=rate_limits['monthly'],
                    daily_cost=0.0,
                    monthly_cost=0.0,
                    last_used=None,
                    status='ACTIVE',
                    error_count=0,
                    success_rate=100.0,
                    avg_response_time=0.0,
                    plan_type='basic'
                )
            
            status = self.key_statuses[key_id]
            
            # Update counters
            today = datetime.utcnow().date()
            if status.last_used is None or status.last_used.date() < today:
                # Reset daily counters for new day
                status.daily_calls = 0
                status.daily_cost = 0.0
            
            status.daily_calls += 1
            status.monthly_calls += 1
            status.daily_cost += call.cost
            status.monthly_cost += call.cost
            status.last_used = call.timestamp
            
            # Update error count
            if not call.success:
                status.error_count += 1
            
            # Calculate success rate and avg response time from recent calls
            recent = list(self.recent_calls[key_id])
            if recent:
                successful_calls = [c for c in recent if c.success]
                status.success_rate = (len(successful_calls) / len(recent)) * 100
                status.avg_response_time = statistics.mean([c.response_time_ms for c in recent])
            
            # Update status based on limits and errors
            if status.daily_calls >= status.daily_limit or status.monthly_calls >= status.monthly_limit:
                status.status = 'RATE_LIMITED'
            elif status.error_count > 10:
                status.status = 'ERROR'
            elif status.daily_calls / status.daily_limit > 0.85:
                status.status = 'WARNING'
            else:
                status.status = 'ACTIVE'
            
            # Persist status
            self._persist_key_status(status)
    
    def _persist_key_status(self, status: APIKeyStatus):
        """Persist key status to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO api_key_status
                (key_id, api_name, daily_calls, monthly_calls, daily_limit, 
                 monthly_limit, daily_cost, monthly_cost, last_used, status,
                 error_count, success_rate, avg_response_time, plan_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                status.key_id,
                status.api_name,
                status.daily_calls,
                status.monthly_calls,
                status.daily_limit,
                status.monthly_limit,
                status.daily_cost,
                status.monthly_cost,
                status.last_used.timestamp() if status.last_used else None,
                status.status,
                status.error_count,
                status.success_rate,
                status.avg_response_time,
                status.plan_type,
                time.time()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.warning(f"Failed to persist key status: {e}")
    
    def get_api_health_metrics(self, api_name: str) -> APIHealthMetrics:
        """Get comprehensive health metrics for an API"""
        with self._lock:
            today_calls = [c for c in self.call_history 
                          if c.api_name == api_name and c.timestamp.date() == datetime.utcnow().date()]
            
            if not today_calls:
                return APIHealthMetrics(
                    api_name=api_name,
                    total_calls_today=0,
                    success_rate=0.0,
                    avg_response_time=0.0,
                    error_rate=0.0,
                    rate_limit_hits=0,
                    cost_today=0.0,
                    estimated_monthly_cost=0.0,
                    peak_usage_hour=0,
                    current_rps=0.0,
                    uptime_percentage=0.0
                )
            
            successful_calls = [c for c in today_calls if c.success]
            rate_limit_hits = len([c for c in today_calls if c.status_code == 429])
            
            # Calculate peak usage hour
            hourly_usage = defaultdict(int)
            for call in today_calls:
                hourly_usage[call.timestamp.hour] += 1
            peak_hour = max(hourly_usage.items(), key=lambda x: x[1])[0] if hourly_usage else 0
            
            # Calculate current RPS (last minute)
            one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
            recent_calls = [c for c in today_calls if c.timestamp > one_minute_ago]
            current_rps = len(recent_calls) / 60.0
            
            # Calculate uptime (non-error responses)
            non_error_calls = [c for c in today_calls if c.status_code not in [500, 502, 503, 504]]
            uptime = (len(non_error_calls) / len(today_calls)) * 100 if today_calls else 0
            
            daily_cost = sum(c.cost for c in today_calls)
            estimated_monthly_cost = daily_cost * 30
            
            return APIHealthMetrics(
                api_name=api_name,
                total_calls_today=len(today_calls),
                success_rate=(len(successful_calls) / len(today_calls)) * 100,
                avg_response_time=statistics.mean([c.response_time_ms for c in successful_calls]) if successful_calls else 0,
                error_rate=(len(today_calls) - len(successful_calls)) / len(today_calls) * 100,
                rate_limit_hits=rate_limit_hits,
                cost_today=daily_cost,
                estimated_monthly_cost=estimated_monthly_cost,
                peak_usage_hour=peak_hour,
                current_rps=current_rps,
                uptime_percentage=uptime
            )
    
    def get_key_status(self, key_id: str) -> Optional[APIKeyStatus]:
        """Get status of a specific API key"""
        with self._lock:
            return self.key_statuses.get(key_id)
    
    def get_all_key_statuses(self) -> List[APIKeyStatus]:
        """Get all API key statuses"""
        with self._lock:
            return list(self.key_statuses.values())
    
    def get_best_available_key(self, api_name: str) -> Optional[str]:
        """Get the best available API key for an API"""
        with self._lock:
            available_keys = [
                status for status in self.key_statuses.values()
                if status.api_name == api_name and status.status == 'ACTIVE'
            ]
            
            if not available_keys:
                # Try WARNING status keys if no ACTIVE keys
                available_keys = [
                    status for status in self.key_statuses.values()
                    if status.api_name == api_name and status.status == 'WARNING'
                ]
            
            if available_keys:
                # Sort by remaining daily limit
                available_keys.sort(key=lambda k: k.daily_limit - k.daily_calls, reverse=True)
                return available_keys[0].key_id
            
            return None
    
    def _check_immediate_alerts(self, key_id: str, call: APICall):
        """Check for immediate alerts after an API call"""
        status = self.key_statuses.get(key_id)
        if not status:
            return
        
        alerts = []
        
        # Rate limit warning
        daily_usage_percent = status.daily_calls / status.daily_limit
        if daily_usage_percent > self.alert_thresholds['rate_limit_warning']:
            alerts.append({
                'type': 'RATE_LIMIT_WARNING',
                'key_id': key_id,
                'message': f"API key {key_id} is at {daily_usage_percent:.1%} of daily rate limit",
                'severity': 'WARNING' if daily_usage_percent < 0.95 else 'CRITICAL'
            })
        
        # Cost warning
        if status.daily_cost > self.alert_thresholds['cost_warning']:
            alerts.append({
                'type': 'COST_WARNING',
                'key_id': key_id,
                'message': f"API key {key_id} daily cost ${status.daily_cost:.2f} exceeds threshold",
                'severity': 'WARNING'
            })
        
        # Error rate warning
        if status.success_rate < (100 - self.alert_thresholds['error_rate_warning'] * 100):
            alerts.append({
                'type': 'ERROR_RATE_WARNING',
                'key_id': key_id,
                'message': f"API key {key_id} success rate {status.success_rate:.1f}% is below threshold",
                'severity': 'WARNING'
            })
        
        # Response time warning
        if status.avg_response_time > self.alert_thresholds['response_time_warning']:
            alerts.append({
                'type': 'RESPONSE_TIME_WARNING',
                'key_id': key_id,
                'message': f"API key {key_id} avg response time {status.avg_response_time:.0f}ms exceeds threshold",
                'severity': 'WARNING'
            })
        
        for alert in alerts:
            self._send_alert(alert)
    
    def _check_alerts(self):
        """Check for system-wide alerts"""
        # This can be extended to check for system-wide conditions
        pass
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification"""
        logging.warning(f"API ALERT [{alert['type']}]: {alert['message']}")
        # Here you could integrate with notification systems like:
        # - Slack webhooks
        # - Email notifications
        # - PagerDuty
        # - Custom webhook endpoints
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent database bloat"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete calls older than 30 days
            thirty_days_ago = time.time() - (30 * 86400)
            cursor.execute('DELETE FROM api_calls WHERE timestamp < ?', (thirty_days_ago,))
            
            deleted = cursor.rowcount
            if deleted > 0:
                logging.info(f"Cleaned up {deleted} old API call records")
            
            conn.commit()
            conn.close()
        except Exception as e:
            logging.warning(f"Failed to cleanup old data: {e}")
    
    def _update_key_statuses(self):
        """Update key statuses periodically"""
        # This method can be extended to fetch real-time quota information
        # from APIs that support it (like TwelveData's quota endpoint)
        pass
    
    def export_metrics_json(self) -> str:
        """Export all metrics as JSON for external consumption"""
        with self._lock:
            data = {
                'timestamp': datetime.utcnow().isoformat(),
                'key_statuses': [asdict(status) for status in self.key_statuses.values()],
                'api_health': {
                    api_name: asdict(self.get_api_health_metrics(api_name))
                    for api_name in set(status.api_name for status in self.key_statuses.values())
                },
                'recent_calls': len(self.call_history),
                'alert_thresholds': self.alert_thresholds
            }
            
            # Convert datetime objects to strings for JSON serialization
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            return json.dumps(data, indent=2, default=serialize_datetime)

# Global instance
api_monitor = EnhancedAPIMonitor()