"""
System Health Monitor Component
===============================
Production-ready health monitoring system for USDCOP trading platform.
Combines Prometheus metrics, system monitoring, and real-time visualization.

Features:
- Prometheus metrics file parsing
- System resource monitoring (CPU, RAM, Disk, Network)
- Component health tracking (MT5, Data Pipeline, Risk Manager, ML Model)
- Real-time WebSocket updates
- REST API endpoints
- Dash dashboard visualization
- Alert system with persistence
- Historical metrics storage
- Automated health checks

Author: USDCOP Trading System
Version: 3.0.0
"""

import os
import re
import json
import time
import math
import asyncio
import logging
import threading
import psutil
import sqlite3
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque, defaultdict

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Response
from pydantic import BaseModel, Field

try:
    import yaml
except ImportError:
    yaml = None

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# =====================================================
# CONFIGURATION & TYPES
# =====================================================

class HealthStatus(Enum):
    """Health status levels"""
    UP = "up"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"
    
    @property
    def priority(self) -> int:
        """Get priority for status comparison"""
        priorities = {
            HealthStatus.DOWN: 3,
            HealthStatus.DEGRADED: 2,
            HealthStatus.UP: 1,
            HealthStatus.UNKNOWN: 0
        }
        return priorities.get(self, 0)


class ComponentType(Enum):
    """Component types in the system"""
    CONNECTION = "connection"
    SERVICE = "service"
    RESOURCE = "resource"
    DATA = "data"
    PROCESS = "process"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: float
    unit: str
    status: HealthStatus
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_error: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    def check_thresholds(self) -> HealthStatus:
        """Check thresholds and update status"""
        if self.threshold_critical and self.value >= self.threshold_critical:
            self.status = HealthStatus.DOWN
        elif self.threshold_error and self.value >= self.threshold_error:
            self.status = HealthStatus.DEGRADED
        elif self.threshold_warning and self.value >= self.threshold_warning:
            self.status = HealthStatus.DEGRADED
        else:
            self.status = HealthStatus.UP
        return self.status


@dataclass
class ComponentHealth:
    """Component health state"""
    name: str
    type: ComponentType
    status: HealthStatus
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    uptime_seconds: float = 0
    
    def get_overall_status(self) -> HealthStatus:
        """Calculate overall status from metrics"""
        if not self.metrics:
            return self.status
        
        worst_status = HealthStatus.UNKNOWN
        for metric in self.metrics.values():
            if metric.status.priority > worst_status.priority:
                worst_status = metric.status
        
        self.status = worst_status
        return self.status


@dataclass
class Alert:
    """System alert"""
    id: str
    level: HealthStatus
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    auto_resolve: bool = True
    ttl_seconds: int = 3600


@dataclass
class HealthMonitorConfig:
    """Health monitor configuration"""
    # Paths
    metrics_file: str = "data/reports/usdcop/mt5_health.prom"
    logs_file: str = "logs/mt5_connector.log"
    db_path: str = "./data/health_monitor.db"
    
    # Data directories
    bronze_dir: Optional[str] = "data/bronze/USDCOP"
    silver_dir: Optional[str] = "data/silver/USDCOP"
    gold_dir: Optional[str] = "data/gold/USDCOP"
    
    # Update intervals
    update_interval: int = 5  # seconds
    heartbeat_sec: int = 15
    staleness_limit_sec: int = 1800
    refresh_sec: int = 5  # WebSocket refresh
    
    # Thresholds
    cpu_warning: float = 70.0
    cpu_error: float = 85.0
    cpu_critical: float = 95.0
    
    memory_warning: float = 70.0
    memory_error: float = 85.0
    memory_critical: float = 95.0
    
    disk_warning: float = 80.0
    disk_error: float = 90.0
    disk_critical: float = 95.0
    
    # Features
    prefer_real: bool = True
    degrade_if_simulator: bool = True
    enable_alerts: bool = True
    enable_persistence: bool = True
    enable_system_stats: bool = True
    
    # History
    metric_history_size: int = 1000
    alert_history_size: int = 500
    alert_cooldown: int = 300
    
    @staticmethod
    def from_yaml(path: str = "configs/mt5_config.yaml") -> "HealthMonitorConfig":
        """Load configuration from YAML file"""
        if not yaml or not os.path.exists(path):
            logger.warning(f"Config file not found: {path}, using defaults")
            return HealthMonitorConfig()
        
        try:
            cfg = _load_yaml_with_env(path)
            
            # Extract relevant sections
            metrics_cfg = cfg.get("metrics", {}) or {}
            logging_cfg = cfg.get("logging", {}) or {}
            paths_cfg = cfg.get("paths", {}) or {}
            fallback_cfg = cfg.get("fallback", {}) or {}
            health_cfg = cfg.get("healthcheck", {}) or {}
            
            return HealthMonitorConfig(
                metrics_file=metrics_cfg.get("file", "data/reports/usdcop/mt5_health.prom"),
                logs_file=logging_cfg.get("file", "logs/mt5_connector.log"),
                bronze_dir=paths_cfg.get("bronze_dir", "data/bronze/USDCOP"),
                silver_dir=paths_cfg.get("silver_dir", "data/silver/USDCOP"),
                gold_dir=paths_cfg.get("gold_dir", "data/gold/USDCOP"),
                prefer_real=bool(fallback_cfg.get("prefer_real", True)),
                staleness_limit_sec=int(fallback_cfg.get("staleness_limit_sec", 1800)),
                heartbeat_sec=int(health_cfg.get("ping_interval_sec", 15)),
                degrade_if_simulator=bool(fallback_cfg.get("prefer_real", True))
            )
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return HealthMonitorConfig()


# =====================================================
# PYDANTIC MODELS FOR API
# =====================================================

class HealthSnapshot(BaseModel):
    """Health snapshot for API responses"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str
    reason: str
    system_running: Optional[bool] = None
    last_metrics_mtime: Optional[datetime] = None
    metrics_age_sec: Optional[float] = None
    feed_stale_seconds: Optional[float] = None
    last_bar_time: Optional[datetime] = None
    last_bar_age_sec: Optional[float] = None
    bars_ingested_total: Optional[float] = None
    bars_ingested_rate_per_min: Optional[float] = None
    mt5_connected: Optional[bool] = None
    using_simulator: Optional[bool] = None
    prefer_real: Optional[bool] = None
    cpu_percent: Optional[float] = None
    mem_percent: Optional[float] = None
    bronze_latest_file_mtime: Optional[datetime] = None
    silver_latest_file_mtime: Optional[datetime] = None
    gold_latest_file_mtime: Optional[datetime] = None
    quality: Optional[Dict[str, Any]] = None
    components: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    active_alerts: int = 0
    alerts: List[Dict[str, Any]] = Field(default_factory=list)


class RawHealth(BaseModel):
    """Raw health data"""
    metrics_file: str
    metrics_mtime: Optional[datetime]
    metrics: Dict[str, float]
    components: Dict[str, Dict[str, Any]]


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def _load_yaml_with_env(path: str) -> Dict[str, Any]:
    """Load YAML with environment variable expansion"""
    if not yaml or not os.path.exists(path):
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    
    data = _deep_expand_env(data)
    
    # Handle active profile
    active = os.getenv("APP_PROFILE", data.get("active_profile", "dev"))
    profiles = data.get("profiles", {})
    defaults = data.get("defaults", {})
    
    if isinstance(profiles, dict) and active in profiles:
        merged = _merge_dicts(defaults, profiles[active])
    else:
        merged = defaults
    
    return merged


_env_pat = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")

def _expand_env_string(s: str) -> str:
    """Expand environment variables in string"""
    def _repl(m: re.Match) -> str:
        var = m.group(1)
        default = m.group(3) if m.group(3) is not None else ""
        return os.getenv(var, default)
    return _env_pat.sub(_repl, s)


def _deep_expand_env(obj: Any) -> Any:
    """Recursively expand environment variables"""
    if isinstance(obj, dict):
        return {k: _deep_expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_expand_env(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_string(obj)
    return obj


def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries"""
    out = dict(a or {})
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _parse_prometheus_metrics(path: str) -> Tuple[Dict[str, float], Optional[float]]:
    """Parse Prometheus text metrics file"""
    metrics: Dict[str, float] = {}
    
    try:
        stat = os.stat(path)
        mtime = stat.st_mtime
    except FileNotFoundError:
        return metrics, None
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                try:
                    # Handle 'metric{labels} value' or 'metric value'
                    parts = line.split(None, 1)
                    if len(parts) < 2:
                        continue
                    
                    metric_part, value_str = parts
                    metric = metric_part.split("{", 1)[0]
                    value = float(value_str.strip().split()[0])
                    
                    metrics[metric] = value
                except Exception:
                    continue
    
    except Exception as e:
        logger.warning(f"Error reading metrics file {path}: {e}")
    
    return metrics, mtime


def _epoch_to_dt(ts: Optional[float]) -> Optional[datetime]:
    """Convert epoch timestamp to datetime"""
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    except Exception:
        return None


def _latest_mtime_under(path_dir: Optional[str]) -> Optional[datetime]:
    """Get latest modification time under directory"""
    if not path_dir or not os.path.isdir(path_dir):
        return None
    
    latest = None
    try:
        for root, _, files in os.walk(path_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                try:
                    mt = os.stat(fp).st_mtime
                    if latest is None or mt > latest:
                        latest = mt
                except Exception:
                    continue
    except Exception:
        return None
    
    return _epoch_to_dt(latest) if latest else None


def _bool_from_metric(metrics: Dict[str, float], key: str) -> Optional[bool]:
    """Convert metric value to boolean"""
    if key not in metrics:
        return None
    v = metrics[key]
    if math.isnan(v):
        return None
    return bool(int(v))


def _tail_file(path: str, n_lines: int) -> str:
    """Efficiently read last n lines of file"""
    try:
        size = os.path.getsize(path)
    except Exception:
        return ""
    
    chunk = 4096
    lines: List[str] = []
    
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        buff = b""
        pos = f.tell()
        
        while pos > 0 and len(lines) <= n_lines:
            read = min(chunk, pos)
            pos -= read
            f.seek(pos)
            data = f.read(read)
            buff = data + buff
            
            try:
                lines = buff.decode("utf-8", errors="replace").splitlines()
            except Exception:
                break
    
    return "\n".join(lines[-n_lines:])


# =====================================================
# HEALTH MONITOR CORE
# =====================================================

class HealthMonitor:
    """Core health monitoring system"""
    
    def __init__(self, config: Optional[HealthMonitorConfig] = None):
        """Initialize health monitor"""
        self.config = config or HealthMonitorConfig.from_yaml()
        
        # Components
        self.components: Dict[str, ComponentHealth] = {}
        
        # Metrics history
        self.metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.metric_history_size)
        )
        
        # Alerts
        self.alerts: deque = deque(maxlen=self.config.alert_history_size)
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # State tracking
        self._prev_metrics: Dict[str, Any] = {}
        self.running = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize database
        if self.config.enable_persistence:
            self._initialize_database()
        
        logger.info("HealthMonitor initialized")
    
    def _initialize_components(self):
        """Initialize system components"""
        # System resources
        self.components['system_resources'] = ComponentHealth(
            name="System Resources",
            type=ComponentType.RESOURCE,
            status=HealthStatus.UNKNOWN,
            metrics={}
        )
        
        # MT5 Connection
        self.components['mt5_connection'] = ComponentHealth(
            name="MT5 Connection",
            type=ComponentType.CONNECTION,
            status=HealthStatus.UNKNOWN,
            metrics={}
        )
        
        # Data Pipeline
        self.components['data_pipeline'] = ComponentHealth(
            name="Data Pipeline",
            type=ComponentType.SERVICE,
            status=HealthStatus.UNKNOWN,
            metrics={}
        )
        
        # Risk Manager
        self.components['risk_manager'] = ComponentHealth(
            name="Risk Manager",
            type=ComponentType.SERVICE,
            status=HealthStatus.UNKNOWN,
            metrics={}
        )
        
        # Trading Strategy
        self.components['trading_strategy'] = ComponentHealth(
            name="Trading Strategy",
            type=ComponentType.SERVICE,
            status=HealthStatus.UNKNOWN,
            metrics={}
        )
        
        # ML Model
        self.components['ml_model'] = ComponentHealth(
            name="ML Model",
            type=ComponentType.SERVICE,
            status=HealthStatus.UNKNOWN,
            metrics={}
        )
    
    def _initialize_database(self):
        """Initialize SQLite database for persistence"""
        try:
            os.makedirs(os.path.dirname(self.config.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT 0,
                    resolved_at DATETIME
                )
            """)
            
            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_component ON metrics(component)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
            
            conn.commit()
            conn.close()
            
            logger.info("Health monitor database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            self.config.enable_persistence = False
    
    def start(self):
        """Start monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitor started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        logger.info("Health monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Update from Prometheus metrics
                self._update_from_prometheus()
                
                # Update system resources
                if self.config.enable_system_stats:
                    self._update_system_resources()
                
                # Update component states
                self._update_component_states()
                
                # Check alerts
                if self.config.enable_alerts:
                    self._check_alerts()
                    self._cleanup_old_alerts()
                
                # Persist metrics
                if self.config.enable_persistence:
                    self._persist_current_metrics()
                
                # Wait for next cycle
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(self.config.update_interval)
    
    def _update_from_prometheus(self):
        """Update metrics from Prometheus file"""
        metrics, mtime = _parse_prometheus_metrics(self.config.metrics_file)
        
        if not metrics:
            return
        
        # Update MT5 connection status
        mt5_comp = self.components['mt5_connection']
        
        # Connection status
        mt5_connected = _bool_from_metric(metrics, "mt5_connected")
        if mt5_connected is not None:
            mt5_comp.metrics['connected'] = HealthMetric(
                name="connected",
                value=1.0 if mt5_connected else 0.0,
                unit="bool",
                status=HealthStatus.UP if mt5_connected else HealthStatus.DOWN,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Using simulator
        using_sim = _bool_from_metric(metrics, "using_simulator")
        if using_sim is not None:
            status = HealthStatus.DEGRADED if (using_sim and self.config.degrade_if_simulator) else HealthStatus.UP
            mt5_comp.metrics['using_simulator'] = HealthMetric(
                name="using_simulator",
                value=1.0 if using_sim else 0.0,
                unit="bool",
                status=status,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Data pipeline metrics
        pipeline_comp = self.components['data_pipeline']
        
        # Feed staleness
        feed_stale = metrics.get("feed_stale_seconds")
        if feed_stale is not None:
            stale_metric = HealthMetric(
                name="feed_stale_seconds",
                value=feed_stale,
                unit="seconds",
                status=HealthStatus.UP,
                timestamp=datetime.now(timezone.utc),
                threshold_warning=300,
                threshold_error=600,
                threshold_critical=self.config.staleness_limit_sec
            )
            stale_metric.check_thresholds()
            pipeline_comp.metrics['feed_staleness'] = stale_metric
        
        # Bars ingested
        bars_total = metrics.get("bars_ingested_total")
        if bars_total is not None:
            # Calculate rate
            rate = self._calculate_rate("bars_ingested_total", bars_total, metrics)
            
            pipeline_comp.metrics['bars_ingested_rate'] = HealthMetric(
                name="bars_ingested_rate",
                value=rate * 60,  # per minute
                unit="bars/min",
                status=HealthStatus.UP if rate > 0 else HealthStatus.DEGRADED,
                timestamp=datetime.now(timezone.utc)
            )
        
        # Update overall statuses
        mt5_comp.get_overall_status()
        pipeline_comp.get_overall_status()
        
        # Store metrics for rate calculations
        self._prev_metrics['metrics'] = metrics
        self._prev_metrics['metrics_time'] = time.time()
    
    def _update_system_resources(self):
        """Update system resource metrics"""
        if not psutil:
            return
        
        try:
            comp = self.components['system_resources']
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status=HealthStatus.UP,
                timestamp=datetime.now(timezone.utc),
                threshold_warning=self.config.cpu_warning,
                threshold_error=self.config.cpu_error,
                threshold_critical=self.config.cpu_critical
            )
            cpu_metric.check_thresholds()
            comp.metrics['cpu_usage'] = cpu_metric
            
            # Memory
            memory = psutil.virtual_memory()
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="%",
                status=HealthStatus.UP,
                timestamp=datetime.now(timezone.utc),
                threshold_warning=self.config.memory_warning,
                threshold_error=self.config.memory_error,
                threshold_critical=self.config.memory_critical
            )
            memory_metric.check_thresholds()
            comp.metrics['memory_usage'] = memory_metric
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_metric = HealthMetric(
                name="disk_usage",
                value=disk.percent,
                unit="%",
                status=HealthStatus.UP,
                timestamp=datetime.now(timezone.utc),
                threshold_warning=self.config.disk_warning,
                threshold_error=self.config.disk_error,
                threshold_critical=self.config.disk_critical
            )
            disk_metric.check_thresholds()
            comp.metrics['disk_usage'] = disk_metric
            
            # Update status
            comp.get_overall_status()
            comp.last_check = datetime.now(timezone.utc)
            
            # Store in history
            with self.lock:
                for metric_name, metric in comp.metrics.items():
                    self.metric_history[f"system_resources.{metric_name}"].append({
                        'timestamp': metric.timestamp,
                        'value': metric.value,
                        'status': metric.status.value
                    })
            
        except Exception as e:
            logger.error(f"Error updating system resources: {e}")
            comp.status = HealthStatus.DOWN
            comp.error_message = str(e)
    
    def _update_component_states(self):
        """Update component states based on metrics"""
        now = time.time()
        
        # Check metrics file freshness
        _, mtime = _parse_prometheus_metrics(self.config.metrics_file)
        
        if mtime:
            metrics_age = now - mtime
            
            # If metrics are too old, mark components as unknown
            if metrics_age > self.config.heartbeat_sec * 3:
                for comp in self.components.values():
                    if comp.type != ComponentType.RESOURCE:
                        comp.status = HealthStatus.UNKNOWN
                        comp.error_message = f"Metrics not updated for {int(metrics_age)}s"
    
    def _calculate_rate(self, metric_name: str, current_value: float, 
                       metrics: Dict[str, float]) -> float:
        """Calculate rate of change for a metric"""
        prev_value = self._prev_metrics.get(f"prev_{metric_name}")
        prev_time = self._prev_metrics.get(f"prev_{metric_name}_time")
        
        if prev_value is not None and prev_time is not None:
            time_diff = time.time() - prev_time
            if time_diff > 0 and current_value >= prev_value:
                rate = (current_value - prev_value) / time_diff
                self._prev_metrics[f"prev_{metric_name}"] = current_value
                self._prev_metrics[f"prev_{metric_name}_time"] = time.time()
                return rate
        
        # First measurement
        self._prev_metrics[f"prev_{metric_name}"] = current_value
        self._prev_metrics[f"prev_{metric_name}_time"] = time.time()
        return 0.0
    
    def _check_alerts(self):
        """Check for alert conditions"""
        for comp_name, comp in self.components.items():
            # Component-level alerts
            if comp.status in [HealthStatus.DOWN, HealthStatus.DEGRADED]:
                self._create_alert(
                    level=comp.status,
                    component=comp_name,
                    message=f"{comp.name} is {comp.status.value}: {comp.error_message or 'Check metrics'}"
                )
            
            # Metric-level alerts
            for metric_name, metric in comp.metrics.items():
                if metric.status in [HealthStatus.DOWN, HealthStatus.DEGRADED]:
                    self._create_alert(
                        level=metric.status,
                        component=comp_name,
                        message=f"{metric.name} = {metric.value}{metric.unit} exceeds threshold"
                    )
    
    def _create_alert(self, level: HealthStatus, component: str, message: str):
        """Create a new alert with cooldown"""
        alert_key = f"{component}:{message}"
        
        # Check cooldown
        if alert_key in self.alert_cooldowns:
            if datetime.now(timezone.utc) < self.alert_cooldowns[alert_key]:
                return
        
        # Create alert
        alert = Alert(
            id=f"{component}_{int(time.time()*1000)}",
            level=level,
            component=component,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )
        
        with self.lock:
            self.alerts.append(alert)
        
        # Set cooldown
        self.alert_cooldowns[alert_key] = datetime.now(timezone.utc) + timedelta(
            seconds=self.config.alert_cooldown
        )
        
        # Persist
        if self.config.enable_persistence:
            self._persist_alert(alert)
        
        logger.warning(f"Alert created: {alert.level.value} - {alert.message}")
    
    def _cleanup_old_alerts(self):
        """Auto-resolve old alerts"""
        now = datetime.now(timezone.utc)
        
        with self.lock:
            for alert in self.alerts:
                if alert.auto_resolve and not alert.resolved:
                    if (now - alert.timestamp).total_seconds() > alert.ttl_seconds:
                        alert.resolved = True
                        alert.resolved_at = now
    
    def _persist_current_metrics(self):
        """Persist current metrics to database"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            for comp_name, comp in self.components.items():
                for metric_name, metric in comp.metrics.items():
                    cursor.execute("""
                        INSERT INTO metrics (component, metric_name, value, unit, status, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        comp_name,
                        metric_name,
                        metric.value,
                        metric.unit,
                        metric.status.value,
                        metric.timestamp
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
    
    def _persist_alert(self, alert: Alert):
        """Persist alert to database"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts (id, level, component, message, timestamp, resolved, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.id,
                alert.level.value,
                alert.component,
                alert.message,
                alert.timestamp,
                alert.resolved,
                alert.resolved_at
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error persisting alert: {e}")
    
    def get_snapshot(self) -> HealthSnapshot:
        """Get current health snapshot"""
        metrics, mtime = _parse_prometheus_metrics(self.config.metrics_file)
        now = time.time()
        
        # Calculate metrics age
        metrics_age = (now - mtime) if mtime is not None else None
        
        # Extract key metrics
        system_running = _bool_from_metric(metrics, "system_running")
        feed_stale = metrics.get("feed_stale_seconds")
        last_bar_unixtime = metrics.get("last_bar_unixtime")
        last_bar_dt = _epoch_to_dt(last_bar_unixtime)
        last_bar_age = (now - last_bar_unixtime) if last_bar_unixtime else None
        
        mt5_connected = _bool_from_metric(metrics, "mt5_connected")
        using_simulator = _bool_from_metric(metrics, "using_simulator")
        
        # Get CPU/Memory if available
        cpu_p = mem_p = None
        if self.config.enable_system_stats and psutil:
            try:
                cpu_p = float(psutil.cpu_percent(interval=None))
                mem_p = float(psutil.virtual_memory().percent)
            except Exception:
                pass
        
        # Get latest file mtimes
        bronze_mt = _latest_mtime_under(self.config.bronze_dir)
        silver_mt = _latest_mtime_under(self.config.silver_dir)
        gold_mt = _latest_mtime_under(self.config.gold_dir)
        
        # Determine overall status
        status, reason = self._compute_overall_status(
            system_running=system_running,
            metrics_age=metrics_age,
            feed_stale=feed_stale,
            last_bar_age=last_bar_age,
            using_simulator=using_simulator
        )
        
        # Build components dict
        components = {}
        with self.lock:
            for comp_name, comp in self.components.items():
                components[comp_name] = {
                    'name': comp.name,
                    'type': comp.type.value,
                    'status': comp.status.value,
                    'last_check': comp.last_check.isoformat(),
                    'uptime_seconds': comp.uptime_seconds,
                    'error_message': comp.error_message,
                    'metrics': {
                        metric_name: {
                            'value': metric.value,
                            'unit': metric.unit,
                            'status': metric.status.value
                        }
                        for metric_name, metric in comp.metrics.items()
                    }
                }
            
            # Get active alerts
            active_alerts = [a for a in self.alerts if not a.resolved]
            alerts = [
                {
                    'id': a.id,
                    'level': a.level.value,
                    'component': a.component,
                    'message': a.message,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in active_alerts[-10:]  # Last 10 alerts
            ]
        
        return HealthSnapshot(
            status=status,
            reason=reason,
            system_running=system_running,
            last_metrics_mtime=_epoch_to_dt(mtime) if mtime else None,
            metrics_age_sec=float(metrics_age) if metrics_age else None,
            feed_stale_seconds=float(feed_stale) if feed_stale else None,
            last_bar_time=last_bar_dt,
            last_bar_age_sec=float(last_bar_age) if last_bar_age else None,
            bars_ingested_total=metrics.get("bars_ingested_total"),
            bars_ingested_rate_per_min=self._get_current_rate("bars_ingested_total"),
            mt5_connected=mt5_connected,
            using_simulator=using_simulator,
            prefer_real=self.config.prefer_real,
            cpu_percent=cpu_p,
            mem_percent=mem_p,
            bronze_latest_file_mtime=bronze_mt,
            silver_latest_file_mtime=silver_mt,
            gold_latest_file_mtime=gold_mt,
            quality=self._extract_quality_metrics(metrics),
            components=components,
            active_alerts=len(active_alerts),
            alerts=alerts
        )
    
    def _compute_overall_status(self, **kwargs) -> Tuple[str, str]:
        """Compute overall system status"""
        system_running = kwargs.get('system_running')
        metrics_age = kwargs.get('metrics_age')
        feed_stale = kwargs.get('feed_stale')
        last_bar_age = kwargs.get('last_bar_age')
        using_simulator = kwargs.get('using_simulator')
        
        # No metrics file
        if metrics_age is None:
            return "DOWN", "metrics_file_missing"
        
        # Metrics not updating
        if metrics_age > max(3 * self.config.heartbeat_sec, 30):
            return "DOWN", f"metrics_not_updated_{int(metrics_age)}s"
        
        # System not running
        if system_running is False:
            return "DOWN", "system_running=0"
        
        # Feed stale
        if feed_stale is not None and feed_stale > self.config.staleness_limit_sec:
            return "DEGRADED", f"feed_stale_{int(feed_stale)}s"
        
        # Last bar too old
        if last_bar_age is not None and last_bar_age > self.config.staleness_limit_sec:
            return "DEGRADED", f"last_bar_old_{int(last_bar_age)}s"
        
        # Using simulator when prefer_real
        if self.config.degrade_if_simulator and using_simulator:
            return "DEGRADED", "using_simulator"
        
        return "UP", "healthy"
    
    def _get_current_rate(self, metric_name: str) -> Optional[float]:
        """Get current rate for a metric"""
        rate_key = f"prev_{metric_name}_rate"
        return self._prev_metrics.get(rate_key)
    
    def _extract_quality_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Extract data quality metrics"""
        quality_keys = [
            "quality_max_na_ratio", "quality_max_duplicate_ratio", 
            "quality_max_gap_factor", "quality_min_rows", 
            "quality_min_daily_coverage", "quality_max_staleness_seconds",
            "data_na_ratio", "data_duplicate_ratio", "data_gap_factor",
            "data_rows", "data_daily_coverage", "data_staleness_seconds"
        ]
        
        return {k: metrics[k] for k in quality_keys if k in metrics}
    
    def get_metric_history(self, metric_path: str, 
                          duration_minutes: int = 60) -> pd.DataFrame:
        """Get metric history"""
        if self.config.enable_persistence:
            return self._load_metric_history_from_db(metric_path, duration_minutes)
        else:
            with self.lock:
                if metric_path in self.metric_history:
                    history = list(self.metric_history[metric_path])
                    df = pd.DataFrame(history)
                    
                    if not df.empty:
                        cutoff = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
                        df = df[df['timestamp'] >= cutoff]
                    
                    return df
                else:
                    return pd.DataFrame()
    
    def _load_metric_history_from_db(self, metric_path: str, 
                                   duration_minutes: int) -> pd.DataFrame:
        """Load metric history from database"""
        try:
            conn = sqlite3.connect(self.config.db_path)
            
            component, metric_name = metric_path.split('.')
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
            
            query = """
                SELECT timestamp, value, status
                FROM metrics
                WHERE component = ? AND metric_name = ? AND timestamp >= ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(component, metric_name, cutoff),
                parse_dates=['timestamp']
            )
            
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error loading metric history: {e}")
            return pd.DataFrame()
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert"""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)
                    
                    if self.config.enable_persistence:
                        try:
                            conn = sqlite3.connect(self.config.db_path)
                            cursor = conn.cursor()
                            
                            cursor.execute("""
                                UPDATE alerts 
                                SET resolved = 1, resolved_at = ?
                                WHERE id = ?
                            """, (alert.resolved_at, alert_id))
                            
                            conn.commit()
                            conn.close()
                        except Exception as e:
                            logger.error(f"Error updating alert: {e}")
                    
                    logger.info(f"Alert {alert_id} resolved")
                    return
        
        logger.warning(f"Alert {alert_id} not found or already resolved")


# =====================================================
# FASTAPI ROUTER
# =====================================================

def build_api_router(monitor: Optional[HealthMonitor] = None) -> APIRouter:
    """Build FastAPI router for health monitoring"""
    if monitor is None:
        monitor = HealthMonitor()
        monitor.start()
    
    router = APIRouter(prefix="/health", tags=["health"])
    
    @router.get("/overview", response_model=HealthSnapshot)
    def get_overview() -> HealthSnapshot:
        """Get health overview snapshot"""
        return monitor.get_snapshot()
    
    @router.get("/raw", response_model=RawHealth)
    def get_raw() -> RawHealth:
        """Get raw health data"""
        metrics, mtime = _parse_prometheus_metrics(monitor.config.metrics_file)
        
        components = {}
        with monitor.lock:
            for comp_name, comp in monitor.components.items():
                components[comp_name] = {
                    'name': comp.name,
                    'type': comp.type.value,
                    'status': comp.status.value,
                    'metrics': {
                        metric_name: {
                            'value': metric.value,
                            'unit': metric.unit,
                            'status': metric.status.value
                        }
                        for metric_name, metric in comp.metrics.items()
                    }
                }
        
        return RawHealth(
            metrics_file=monitor.config.metrics_file,
            metrics_mtime=_epoch_to_dt(mtime) if mtime else None,
            metrics=metrics,
            components=components
        )
    
    @router.get("/config")
    def get_config() -> Dict[str, Any]:
        """Get monitor configuration"""
        return {"config": asdict(monitor.config)}
    
    @router.get("/logs/tail")
    def tail_logs(lines: int = Query(200, ge=1, le=2000)) -> Dict[str, Any]:
        """Tail log file"""
        path = monitor.config.logs_file
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        try:
            return {"file": path, "lines": _tail_file(path, lines)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Tail failed: {e}")
    
    @router.websocket("/ws")
    async def websocket_health(websocket: WebSocket, interval: Optional[float] = None):
        """WebSocket endpoint for real-time updates"""
        await websocket.accept()
        refresh = float(interval or monitor.config.refresh_sec or 5)
        
        try:
            while True:
                snapshot = monitor.get_snapshot()
                await websocket.send_text(snapshot.json())
                await asyncio.sleep(refresh)
        except WebSocketDisconnect:
            return
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
            try:
                await websocket.close()
            except Exception:
                pass
    
    @router.post("/alerts/{alert_id}/resolve")
    def resolve_alert(alert_id: str) -> Dict[str, str]:
        """Resolve an alert"""
        monitor.resolve_alert(alert_id)
        return {"status": "resolved", "alert_id": alert_id}
    
    @router.get("/metrics")
    def get_metrics() -> Response:
        """Get Prometheus metrics in text format"""
        try:
            # Try to get metrics from observability system if available
            try:
                from src.core.observability.metrics import get_metrics
                metrics_registry = get_metrics()
                metrics_text = metrics_registry.get_metrics_text()
                return Response(content=metrics_text, media_type="text/plain")
            except ImportError:
                # Fallback to existing Prometheus textfile parsing
                metrics, mtime = _parse_prometheus_metrics(monitor.config.metrics_file)
                # Convert to Prometheus format
                metrics_text = ""
                for metric_name, metric_data in metrics.items():
                    if 'value' in metric_data:
                        metrics_text += f"{metric_name} {metric_data['value']}\n"
                return Response(content=metrics_text, media_type="text/plain")
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return Response(content="", media_type="text/plain")

    @router.get("/metrics/{metric_path}/history")
    def get_metric_history(
        metric_path: str,
        duration_minutes: int = Query(60, ge=1, le=1440)
    ) -> Dict[str, Any]:
        """Get metric history"""
        df = monitor.get_metric_history(metric_path, duration_minutes)
        
        if df.empty:
            return {"metric": metric_path, "data": []}
        
        # Convert to JSON-serializable format
        data = df.to_dict('records')
        for record in data:
            if 'timestamp' in record and hasattr(record['timestamp'], 'isoformat'):
                record['timestamp'] = record['timestamp'].isoformat()
        
        return {"metric": metric_path, "data": data}
    
    return router


# =====================================================
# DASH DASHBOARD COMPONENT
# =====================================================

class HealthMonitorDashboard:
    """Dash dashboard component for health visualization"""
    
    def __init__(self, monitor: HealthMonitor):
        """Initialize dashboard"""
        self.monitor = monitor
        
        # Status colors
        self.status_colors = {
            HealthStatus.UP: "#28a745",
            HealthStatus.DEGRADED: "#ffc107",
            HealthStatus.DOWN: "#dc3545",
            HealthStatus.UNKNOWN: "#6c757d"
        }
        
        # Component icons
        self.component_icons = {
            ComponentType.RESOURCE: "💻",
            ComponentType.CONNECTION: "🔌",
            ComponentType.SERVICE: "⚙️",
            ComponentType.DATA: "📊",
            ComponentType.PROCESS: "🔄"
        }
    
    def create_layout(self) -> html.Div:
        """Create dashboard layout"""
        return html.Div([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("System Health Monitor", className="text-center mb-4"),
                    html.P(
                        "Real-time monitoring of USDCOP trading system components",
                        className="text-center text-muted"
                    )
                ])
            ]),
            
            # Overall Status Card
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id="overall-status-display")
                        ])
                    ], className="mb-4")
                ])
            ]),
            
            # Component Status Grid
            dbc.Row([
                dbc.Col([
                    html.H3("Component Status", className="mb-3"),
                    html.Div(id="component-status-grid")
                ])
            ], className="mb-4"),
            
            # Metrics Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="system-resources-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="performance-metrics-chart")
                ], width=6)
            ], className="mb-4"),
            
            # Alerts Section
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H4("Active Alerts", className="mb-0")
                        ]),
                        dbc.CardBody([
                            html.Div(id="alerts-display")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Detailed Metrics Tabs
            dbc.Row([
                dbc.Col([
                    dcc.Tabs(id="detail-tabs", value="metrics", children=[
                        dcc.Tab(label="Live Metrics", value="metrics"),
                        dcc.Tab(label="Historical Data", value="history"),
                        dcc.Tab(label="System Logs", value="logs"),
                        dcc.Tab(label="Configuration", value="config")
                    ]),
                    html.Div(id="tab-content", className="mt-3")
                ])
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='health-refresh-interval',
                interval=5000,  # 5 seconds
                n_intervals=0
            ),
            
            # Store for state
            dcc.Store(id='health-state-store')
        ])
    
    def register_callbacks(self, app: dash.Dash):
        """Register Dash callbacks"""
        
        @app.callback(
            [Output('overall-status-display', 'children'),
             Output('component-status-grid', 'children'),
             Output('system-resources-chart', 'figure'),
             Output('performance-metrics-chart', 'figure'),
             Output('alerts-display', 'children'),
             Output('health-state-store', 'data')],
            [Input('health-refresh-interval', 'n_intervals')]
        )
        def update_dashboard(n_intervals):
            """Update all dashboard components"""
            # Get current snapshot
            snapshot = self.monitor.get_snapshot()
            
            # Overall status
            overall_status = self._create_overall_status(snapshot)
            
            # Component grid
            component_grid = self._create_component_grid(snapshot.components)
            
            # Charts
            resources_chart = self._create_resources_chart()
            performance_chart = self._create_performance_chart()
            
            # Alerts
            alerts_display = self._create_alerts_display(snapshot.alerts)
            
            # Store state
            state_data = snapshot.dict()
            
            return (
                overall_status,
                component_grid,
                resources_chart,
                performance_chart,
                alerts_display,
                state_data
            )
        
        @app.callback(
            Output('tab-content', 'children'),
            [Input('detail-tabs', 'value')],
            [State('health-state-store', 'data')]
        )
        def update_tab_content(active_tab, state_data):
            """Update tab content based on selection"""
            if active_tab == 'metrics':
                return self._create_metrics_detail(state_data)
            elif active_tab == 'history':
                return self._create_history_view()
            elif active_tab == 'logs':
                return self._create_logs_view()
            elif active_tab == 'config':
                return self._create_config_view()
            
            return html.Div()
    
    def _create_overall_status(self, snapshot: HealthSnapshot) -> html.Div:
        """Create overall status display"""
        status_map = {
            "UP": ("Healthy", "✓", self.status_colors[HealthStatus.UP]),
            "DEGRADED": ("Degraded", "!", self.status_colors[HealthStatus.DEGRADED]),
            "DOWN": ("Down", "✗", self.status_colors[HealthStatus.DOWN]),
            "UNKNOWN": ("Unknown", "?", self.status_colors[HealthStatus.UNKNOWN])
        }
        
        status_text, icon, color = status_map.get(
            snapshot.status, 
            ("Unknown", "?", self.status_colors[HealthStatus.UNKNOWN])
        )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H1(icon, style={
                        'color': color,
                        'fontSize': '72px',
                        'margin': '0',
                        'textAlign': 'center'
                    }),
                    html.H3(status_text, style={
                        'color': color,
                        'textAlign': 'center'
                    }),
                    html.P(f"Reason: {snapshot.reason}", 
                          className="text-center text-muted"),
                    html.P(f"Last update: {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}", 
                          className="text-center text-muted small")
                ], width=4),
                dbc.Col([
                    # Key metrics summary
                    html.Div([
                        self._create_metric_card("MT5", "Connected" if snapshot.mt5_connected else "Disconnected", 
                                               snapshot.mt5_connected),
                        self._create_metric_card("Mode", "Simulator" if snapshot.using_simulator else "Real", 
                                               not snapshot.using_simulator),
                        self._create_metric_card("CPU", f"{snapshot.cpu_percent:.1f}%" if snapshot.cpu_percent else "N/A",
                                               snapshot.cpu_percent is None or snapshot.cpu_percent < 80),
                        self._create_metric_card("Memory", f"{snapshot.mem_percent:.1f}%" if snapshot.mem_percent else "N/A",
                                               snapshot.mem_percent is None or snapshot.mem_percent < 80),
                    ], className="d-flex justify-content-around")
                ], width=8)
            ])
        ])
    
    def _create_metric_card(self, label: str, value: str, is_good: bool) -> html.Div:
        """Create a small metric card"""
        color = self.status_colors[HealthStatus.UP if is_good else HealthStatus.DEGRADED]
        
        return html.Div([
            html.H6(label, className="text-muted mb-0"),
            html.H4(value, style={'color': color})
        ], className="text-center")
    
    def _create_component_grid(self, components: Dict[str, Any]) -> html.Div:
        """Create component status grid"""
        cards = []
        
        for comp_name, comp_data in components.items():
            status = comp_data['status'].upper()
            color = self.status_colors.get(
                HealthStatus[status], 
                self.status_colors[HealthStatus.UNKNOWN]
            )
            
            # Get component type icon
            comp_type = ComponentType[comp_data['type'].upper()]
            icon = self.component_icons.get(comp_type, "📦")
            
            # Format uptime
            uptime = comp_data.get('uptime_seconds', 0)
            uptime_str = self._format_uptime(uptime)
            
            # Count metrics by status
            metrics = comp_data.get('metrics', {})
            metric_counts = {"UP": 0, "DEGRADED": 0, "DOWN": 0}
            for metric in metrics.values():
                metric_status = metric.get('status', 'UNKNOWN').upper()
                if metric_status in metric_counts:
                    metric_counts[metric_status] += 1
            
            card = dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Span(icon, style={'fontSize': '24px', 'marginRight': '10px'}),
                            html.H5(comp_data['name'], className="d-inline")
                        ]),
                        html.Hr(),
                        html.Div([
                            html.Span("●", style={
                                'color': color,
                                'fontSize': '20px',
                                'marginRight': '10px'
                            }),
                            html.Span(status, style={'fontWeight': 'bold'})
                        ]),
                        html.P(f"Uptime: {uptime_str}", className="text-muted small mb-1"),
                        html.Div([
                            html.Span(f"✓ {metric_counts['UP']}", 
                                     style={'color': self.status_colors[HealthStatus.UP], 'marginRight': '10px'}),
                            html.Span(f"! {metric_counts['DEGRADED']}", 
                                     style={'color': self.status_colors[HealthStatus.DEGRADED], 'marginRight': '10px'}),
                            html.Span(f"✗ {metric_counts['DOWN']}", 
                                     style={'color': self.status_colors[HealthStatus.DOWN]})
                        ], className="small")
                    ])
                ], className="h-100")
            ], width=4, className="mb-3")
            
            cards.append(card)
        
        return dbc.Row(cards)
    
    def _create_resources_chart(self) -> go.Figure:
        """Create system resources chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Network Activity'),
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'scatter'}]]
        )
        
        # Get current metrics
        sys_comp = self.monitor.components.get('system_resources')
        
        if sys_comp and sys_comp.metrics:
            # CPU gauge
            cpu_metric = sys_comp.metrics.get('cpu_usage')
            if cpu_metric:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=cpu_metric.value,
                        title={'text': "CPU %"},
                        delta={'reference': self.monitor.config.cpu_warning},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': self._get_gauge_color(cpu_metric.value, 
                                                                  self.monitor.config.cpu_warning,
                                                                  self.monitor.config.cpu_error,
                                                                  self.monitor.config.cpu_critical)},
                            'steps': [
                                {'range': [0, self.monitor.config.cpu_warning], 'color': "lightgray"},
                                {'range': [self.monitor.config.cpu_warning, self.monitor.config.cpu_error], 'color': "yellow"},
                                {'range': [self.monitor.config.cpu_error, self.monitor.config.cpu_critical], 'color': "orange"},
                                {'range': [self.monitor.config.cpu_critical, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': self.monitor.config.cpu_critical
                            }
                        }
                    ),
                    row=1, col=1
                )
            
            # Memory gauge
            mem_metric = sys_comp.metrics.get('memory_usage')
            if mem_metric:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=mem_metric.value,
                        title={'text': "Memory %"},
                        delta={'reference': self.monitor.config.memory_warning},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': self._get_gauge_color(mem_metric.value,
                                                                  self.monitor.config.memory_warning,
                                                                  self.monitor.config.memory_error,
                                                                  self.monitor.config.memory_critical)}
                        }
                    ),
                    row=1, col=2
                )
            
            # Disk gauge
            disk_metric = sys_comp.metrics.get('disk_usage')
            if disk_metric:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=disk_metric.value,
                        title={'text': "Disk %"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': self._get_gauge_color(disk_metric.value,
                                                                  self.monitor.config.disk_warning,
                                                                  self.monitor.config.disk_error,
                                                                  self.monitor.config.disk_critical)}
                        }
                    ),
                    row=2, col=1
                )
            
            # Network history (if available)
            # For now, show placeholder
            fig.add_trace(
                go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[10, 15, 13, 17, 15],
                    mode='lines',
                    name='Network MB/s'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="System Resources"
        )
        
        return fig
    
    def _create_performance_chart(self) -> go.Figure:
        """Create performance metrics chart"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Feed Status', 'Trading Activity', 
                          'Model Performance', 'Risk Metrics')
        )
        
        # Data feed status
        pipeline_comp = self.monitor.components.get('data_pipeline')
        if pipeline_comp and 'feed_staleness' in pipeline_comp.metrics:
            metric = pipeline_comp.metrics['feed_staleness']
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=metric.value,
                    title={'text': "Feed Lag (seconds)"},
                    delta={'reference': 0, 'increasing': {'color': "red"}},
                    number={'suffix': "s"}
                ),
                row=1, col=1
            )
        
        # Add placeholder charts for other metrics
        # These would be populated with real data in production
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="Performance Metrics"
        )
        
        return fig
    
    def _create_alerts_display(self, alerts: List[Dict[str, Any]]) -> html.Div:
        """Create alerts display"""
        if not alerts:
            return html.Div([
                html.P("No active alerts", className="text-muted text-center")
            ])
        
        alert_rows = []
        for alert in alerts:
            level = alert['level'].upper()
            color = self.status_colors.get(
                HealthStatus[level], 
                self.status_colors[HealthStatus.UNKNOWN]
            )
            
            row = html.Tr([
                html.Td(html.Span("●", style={'color': color, 'fontSize': '16px'})),
                html.Td(alert['component']),
                html.Td(alert['message']),
                html.Td(alert['timestamp']),
                html.Td(
                    dbc.Button(
                        "Resolve",
                        color="primary",
                        size="sm",
                        id={'type': 'resolve-alert-btn', 'index': alert['id']}
                    )
                )
            ])
            alert_rows.append(row)
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Level"),
                    html.Th("Component"),
                    html.Th("Message"),
                    html.Th("Time"),
                    html.Th("Action")
                ])
            ]),
            html.Tbody(alert_rows)
        ], striped=True, hover=True, responsive=True, size="sm")
    
    def _create_metrics_detail(self, state_data: Optional[Dict]) -> html.Div:
        """Create detailed metrics view"""
        if not state_data:
            return html.Div("No data available")
        
        components = state_data.get('components', {})
        
        accordion_items = []
        for comp_name, comp_data in components.items():
            metrics = comp_data.get('metrics', {})
            
            metric_cards = []
            for metric_name, metric_data in metrics.items():
                status = metric_data.get('status', 'UNKNOWN').upper()
                color = self.status_colors.get(
                    HealthStatus[status], 
                    self.status_colors[HealthStatus.UNKNOWN]
                )
                
                card = dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(metric_name.replace('_', ' ').title(), 
                                   className="card-title"),
                            html.H3(f"{metric_data['value']:.2f} {metric_data['unit']}",
                                   style={'color': color}),
                            html.P(f"Status: {status}", className="text-muted small")
                        ])
                    ])
                ], width=3, className="mb-3")
                
                metric_cards.append(card)
            
            accordion_items.append(
                dbc.AccordionItem(
                    dbc.Row(metric_cards),
                    title=f"{comp_data['name']} ({len(metrics)} metrics)"
                )
            )
        
        return dbc.Accordion(accordion_items)
    
    def _create_history_view(self) -> html.Div:
        """Create historical data view"""
        # Get all available metrics
        all_metrics = []
        for comp_name, comp in self.monitor.components.items():
            for metric_name in comp.metrics.keys():
                all_metrics.append({
                    'label': f"{comp.name} - {metric_name}",
                    'value': f"{comp_name}.{metric_name}"
                })
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Select Metric:"),
                    dcc.Dropdown(
                        id='history-metric-selector',
                        options=all_metrics,
                        value=all_metrics[0]['value'] if all_metrics else None
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='history-time-range',
                        options=[
                            {'label': 'Last 15 minutes', 'value': 15},
                            {'label': 'Last 30 minutes', 'value': 30},
                            {'label': 'Last 1 hour', 'value': 60},
                            {'label': 'Last 4 hours', 'value': 240},
                            {'label': 'Last 24 hours', 'value': 1440}
                        ],
                        value=60
                    )
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='history-chart')
                ])
            ])
        ])
    
    def _create_logs_view(self) -> html.Div:
        """Create logs view"""
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Number of lines:"),
                    dcc.Input(
                        id='log-lines-input',
                        type='number',
                        value=200,
                        min=1,
                        max=2000
                    ),
                    dbc.Button(
                        "Refresh Logs",
                        id='refresh-logs-btn',
                        color="primary",
                        className="ms-2"
                    )
                ])
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Pre(
                        id='logs-display',
                        style={
                            'backgroundColor': '#f8f9fa',
                            'padding': '1rem',
                            'maxHeight': '600px',
                            'overflowY': 'auto'
                        }
                    )
                ])
            ])
        ])
    
    def _create_config_view(self) -> html.Div:
        """Create configuration view"""
        config_dict = asdict(self.monitor.config)
        
        config_items = []
        for key, value in config_dict.items():
            config_items.append(
                html.Tr([
                    html.Td(key, style={'fontWeight': 'bold'}),
                    html.Td(str(value))
                ])
            )
        
        return html.Div([
            html.H5("Health Monitor Configuration"),
            dbc.Table([
                html.Tbody(config_items)
            ], striped=True, hover=True, responsive=True)
        ])
    
    def _get_gauge_color(self, value: float, warning: float, 
                        error: float, critical: float) -> str:
        """Get color for gauge based on thresholds"""
        if value >= critical:
            return self.status_colors[HealthStatus.DOWN]
        elif value >= error:
            return "#fd7e14"  # Orange
        elif value >= warning:
            return self.status_colors[HealthStatus.DEGRADED]
        else:
            return self.status_colors[HealthStatus.UP]
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        
        return " ".join(parts) if parts else "< 1m"


# =====================================================
# FACTORY FUNCTIONS
# =====================================================

def create_health_monitor_app(config: Optional[HealthMonitorConfig] = None) -> dash.Dash:
    """Create standalone Dash app for health monitoring"""
    # Initialize monitor
    monitor = HealthMonitor(config)
    monitor.start()
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True
    )
    
    # Create dashboard
    dashboard = HealthMonitorDashboard(monitor)
    
    # Set layout
    app.layout = html.Div([
        dbc.Container([
            dashboard.create_layout()
        ], fluid=True)
    ])
    
    # Register callbacks
    dashboard.register_callbacks(app)
    
    # Additional callbacks for interactivity
    @app.callback(
        Output('history-chart', 'figure'),
        [Input('history-metric-selector', 'value'),
         Input('history-time-range', 'value')]
    )
    def update_history_chart(metric_path, time_range):
        """Update history chart based on selection"""
        if not metric_path:
            return go.Figure()
        
        # Get history
        df = monitor.get_metric_history(metric_path, time_range)
        
        if df.empty:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines+markers',
            name=metric_path,
            line=dict(width=2)
        ))
        
        # Add status coloring if available
        if 'status' in df.columns:
            for status in df['status'].unique():
                status_df = df[df['status'] == status]
                if not status_df.empty:
                    try:
                        color = dashboard.status_colors.get(
                            HealthStatus[status.upper()],
                            "#6c757d"
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=status_df['timestamp'],
                            y=status_df['value'],
                            mode='markers',
                            name=f"{status} points",
                            marker=dict(color=color, size=8),
                            showlegend=False
                        ))
                    except KeyError:
                        pass
        
        fig.update_layout(
            title=f"{metric_path} History",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    @app.callback(
        Output('logs-display', 'children'),
        [Input('refresh-logs-btn', 'n_clicks')],
        [State('log-lines-input', 'value')]
    )
    def refresh_logs(n_clicks, num_lines):
        """Refresh log display"""
        if not n_clicks:
            num_lines = 200
        
        try:
            logs = _tail_file(monitor.config.logs_file, num_lines)
            return logs
        except Exception as e:
            return f"Error reading logs: {e}"
    
    return app


def create_health_monitor_router(config: Optional[HealthMonitorConfig] = None) -> APIRouter:
    """Create FastAPI router for health monitoring"""
    monitor = HealthMonitor(config)
    monitor.start()
    
    return build_api_router(monitor)


# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == "__main__":
    # For testing - create standalone app
    import uvicorn
    from fastapi import FastAPI
    
    # Create FastAPI app with health endpoints
    api_app = FastAPI(title="Health Monitor API")
    api_app.include_router(create_health_monitor_router())
    
    # Create Dash app
    dash_app = create_health_monitor_app()
    
    # Run both (in production, use proper ASGI server)
    print("Starting Health Monitor...")
    print("API available at: http://localhost:8000/health")
    print("Dashboard available at: http://localhost:8050")
    
    # You would typically run these on different ports/processes
    # For demo, just run the Dash app
    dash_app.run_server(debug=True, port=8050)