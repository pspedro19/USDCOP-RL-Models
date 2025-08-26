"""
System Health Checks and Monitoring
===================================
Provides comprehensive health monitoring for all system components.
"""

import time
import logging
import json
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime, timedelta, timezone, timezone
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum

# System resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

# Import internal modules with fallback
try:
    from ..database.db_integration import db_integration
    HAS_DB = True
except ImportError:
    HAS_DB = False
    db_integration = None

try:
    from ..config.unified_config import config_loader
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    config_loader = None

try:
    from ..events.bus import event_bus, Event, EventType
    HAS_EVENT_BUS = True
except ImportError:
    HAS_EVENT_BUS = False
    event_bus = None
    Event = None
    
    # Define EventType enum if not available
    class EventType(Enum):
        HEALTH_CHECK = "HEALTH_CHECK"
        HEALTH_WARNING = "HEALTH_WARNING"
        HEALTH_CRITICAL = "HEALTH_CRITICAL"

logger = logging.getLogger(__name__)


# ===========================
# Health Status Types
# ===========================

class HealthState(Enum):
    """Health state enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthStatus:
    """Health status for a component"""
    component: str
    status: str  # Use string for compatibility
    message: str
    last_check: datetime
    metrics: Optional[Dict[str, Any]] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['last_check'] = self.last_check.isoformat()
        return result
    
    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        return self.status == HealthState.HEALTHY.value
    
    def is_critical(self) -> bool:
        """Check if component is in critical state"""
        return self.status == HealthState.UNHEALTHY.value


# ===========================
# Health Checker Class
# ===========================

class HealthChecker:
    """System-wide health checker with history tracking"""
    
    def __init__(self, history_size: int = 100):
        self.checks: Dict[str, Callable] = {}
        self.status_history: deque = deque(maxlen=history_size)
        self.component_history: Dict[str, deque] = {}
        self.start_time = datetime.now(timezone.utc)
        self.check_count = 0
        self.thresholds = self._default_thresholds()
        
    def _default_thresholds(self) -> Dict[str, Any]:
        """Default health thresholds"""
        return {
            'cpu_percent_warning': 70,
            'cpu_percent_critical': 90,
            'memory_percent_warning': 70,
            'memory_percent_critical': 85,
            'disk_percent_warning': 80,
            'disk_percent_critical': 95,
            'db_latency_ms_warning': 100,
            'db_latency_ms_critical': 500,
            'error_rate_warning': 0.1,  # 10% error rate
            'error_rate_critical': 0.25  # 25% error rate
        }
    
    def register_check(self, name: str, check_func: Callable[[], HealthStatus]):
        """Register a health check function"""
        self.checks[name] = check_func
        self.component_history[name] = deque(maxlen=100)
        logger.info(f"Registered health check: {name}")
    
    def unregister_check(self, name: str):
        """Unregister a health check"""
        if name in self.checks:
            del self.checks[name]
            if name in self.component_history:
                del self.component_history[name]
            logger.info(f"Unregistered health check: {name}")
    
    def run_checks(self, components: List[str] = None) -> Dict[str, HealthStatus]:
        """
        Run health checks for specified components or all.
        
        Args:
            components: List of component names to check, or None for all
            
        Returns:
            Dictionary of component health statuses
        """
        results = {}
        components_to_check = components or list(self.checks.keys())
        
        for name in components_to_check:
            if name not in self.checks:
                results[name] = HealthStatus(
                    component=name,
                    status=HealthState.UNKNOWN.value,
                    message="Health check not registered",
                    last_check=datetime.now(timezone.utc)
                )
                continue
            
            try:
                # Run the check
                status = self.checks[name]()
                results[name] = status
                
                # Store in history
                self.component_history[name].append({
                    'timestamp': status.last_check,
                    'status': status.status,
                    'metrics': status.metrics
                })
                
            except Exception as e:
                # Check failed
                results[name] = HealthStatus(
                    component=name,
                    status=HealthState.UNHEALTHY.value,
                    message=f"Check failed: {e}",
                    last_check=datetime.now(timezone.utc),
                    details={'error': str(e)}
                )
                logger.error(f"Health check failed for {name}: {e}")
        
        # Update check count
        self.check_count += 1
        
        # Store overall results
        self.status_history.append({
            'timestamp': datetime.now(timezone.utc),
            'check_number': self.check_count,
            'results': results
        })
        
        # Publish health event if available
        self._publish_health_event(results)
        
        return results
    
    def _publish_health_event(self, results: Dict[str, HealthStatus]):
        """Publish health check results to event bus"""
        if not HAS_EVENT_BUS or not event_bus:
            return
        
        try:
            # Determine overall health
            statuses = [r.status for r in results.values()]
            
            if any(s == HealthState.UNHEALTHY.value for s in statuses):
                event_type = EventType.HEALTH_CRITICAL
            elif any(s == HealthState.DEGRADED.value for s in statuses):
                event_type = EventType.HEALTH_WARNING
            else:
                event_type = EventType.HEALTH_CHECK
            
            # Publish event
            event = Event(
                event=event_type.value,
                payload={
                    'results': {k: v.to_dict() for k, v in results.items()},
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            
            event_bus.publish(event)
            
        except Exception as e:
            logger.warning(f"Failed to publish health event: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        # Run all checks
        results = self.run_checks()
        
        # Calculate overall status
        statuses = [r.status for r in results.values()]
        
        if all(s == HealthState.HEALTHY.value for s in statuses):
            overall = HealthState.HEALTHY.value
        elif any(s == HealthState.UNHEALTHY.value for s in statuses):
            overall = HealthState.UNHEALTHY.value
        elif any(s == HealthState.DEGRADED.value for s in statuses):
            overall = HealthState.DEGRADED.value
        else:
            overall = HealthState.UNKNOWN.value
        
        # Calculate uptime
        uptime = datetime.now(timezone.utc) - self.start_time
        
        return {
            'overall_status': overall,
            'uptime': str(uptime),
            'uptime_seconds': uptime.total_seconds(),
            'checks_performed': self.check_count,
            'components': {k: v.to_dict() for k, v in results.items()},
            'system_metrics': self._get_system_metrics(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics"""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        if HAS_PSUTIL and psutil:
            try:
                # CPU metrics
                metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
                metrics['cpu_count'] = psutil.cpu_count()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                metrics['memory_percent'] = memory.percent
                metrics['memory_available_gb'] = memory.available / (1024**3)
                metrics['memory_used_gb'] = memory.used / (1024**3)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                metrics['disk_percent'] = disk.percent
                metrics['disk_free_gb'] = disk.free / (1024**3)
                
                # Process metrics
                metrics['process_count'] = len(psutil.pids())
                
                # Network metrics (if available)
                try:
                    net_io = psutil.net_io_counters()
                    metrics['network_bytes_sent'] = net_io.bytes_sent
                    metrics['network_bytes_recv'] = net_io.bytes_recv
                except:
                    pass
                
            except Exception as e:
                logger.warning(f"Failed to get system metrics: {e}")
        
        return metrics
    
    def get_component_history(self, component: str, limit: int = 50) -> List[Dict]:
        """Get health history for a specific component"""
        if component not in self.component_history:
            return []
        
        history = list(self.component_history[component])
        if limit:
            history = history[-limit:]
        
        # Convert timestamps to ISO format
        for entry in history:
            if isinstance(entry.get('timestamp'), datetime):
                entry['timestamp'] = entry['timestamp'].isoformat()
        
        return history
    
    def get_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends"""
        trends = {}
        
        for component, history in self.component_history.items():
            if not history:
                continue
            
            # Count states
            state_counts = {
                HealthState.HEALTHY.value: 0,
                HealthState.DEGRADED.value: 0,
                HealthState.UNHEALTHY.value: 0
            }
            
            for entry in history:
                status = entry.get('status', HealthState.UNKNOWN.value)
                if status in state_counts:
                    state_counts[status] += 1
            
            total = sum(state_counts.values())
            
            trends[component] = {
                'total_checks': total,
                'healthy_percent': (state_counts[HealthState.HEALTHY.value] / total * 100) if total > 0 else 0,
                'degraded_percent': (state_counts[HealthState.DEGRADED.value] / total * 100) if total > 0 else 0,
                'unhealthy_percent': (state_counts[HealthState.UNHEALTHY.value] / total * 100) if total > 0 else 0,
                'state_counts': state_counts
            }
        
        return trends
    
    def export_health_report(self, filepath: str):
        """Export health report to file"""
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'system_health': self.get_system_health(),
            'health_trends': self.get_health_trends(),
            'component_history': {
                component: self.get_component_history(component)
                for component in self.component_history.keys()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported health report to {filepath}")


# ===========================
# Global Health Checker
# ===========================

# Initialize global health checker
health_checker = HealthChecker()


# ===========================
# Component Health Checks
# ===========================

def check_database_health() -> HealthStatus:
    """Check database connectivity and performance"""
    if not HAS_DB or not db_integration:
        return HealthStatus(
            component="database",
            status=HealthState.UNKNOWN.value,
            message="Database integration not available",
            last_check=datetime.now(timezone.utc)
        )
    
    try:
        start = time.time()
        
        # Try to get connection
        if not db_integration.db_manager:
            return HealthStatus(
                component="database",
                status=HealthState.UNHEALTHY.value,
                message="Database manager not initialized",
                last_check=datetime.now(timezone.utc)
            )
        
        # Test query
        conn = db_integration.db_manager.get_connection()
        conn.execute("SELECT 1")
        conn.close()
        
        latency = (time.time() - start) * 1000
        
        # Get statistics
        stats = db_integration.get_statistics()
        
        # Determine health based on latency
        thresholds = health_checker.thresholds
        if latency > thresholds['db_latency_ms_critical']:
            status = HealthState.UNHEALTHY.value
            message = f"Database critically slow: {latency:.2f}ms"
        elif latency > thresholds['db_latency_ms_warning']:
            status = HealthState.DEGRADED.value
            message = f"Database slow: {latency:.2f}ms"
        else:
            status = HealthState.HEALTHY.value
            message = f"Database healthy: {latency:.2f}ms"
        
        return HealthStatus(
            component="database",
            status=status,
            message=message,
            last_check=datetime.now(timezone.utc),
            metrics={
                'latency_ms': latency,
                'total_records': sum(v for k, v in stats.items() if k.endswith('_count'))
            },
            details=stats
        )
        
    except Exception as e:
        return HealthStatus(
            component="database",
            status=HealthState.UNHEALTHY.value,
            message=f"Database error: {e}",
            last_check=datetime.now(timezone.utc),
            details={'error': str(e)}
        )


def check_mt5_connector_health() -> HealthStatus:
    """Check MT5 connector health"""
    try:
        from ..connectors.mt5_connector import RobustMT5Connector
        
        # Get or create connector instance
        connector = RobustMT5Connector()
        
        # Check connection
        is_connected = connector.is_connected()
        mode = connector.get_active_mode() if hasattr(connector, 'get_active_mode') else None
        
        # Get health statistics
        health_data = connector.health_check() if hasattr(connector, 'health_check') else {}
        
        if is_connected:
            status = HealthState.HEALTHY.value
            message = f"Connected via {mode}" if mode else "Connected"
        else:
            status = HealthState.UNHEALTHY.value
            message = "Not connected"
        
        return HealthStatus(
            component="mt5_connector",
            status=status,
            message=message,
            last_check=datetime.now(timezone.utc),
            metrics={
                'mode': str(mode) if mode else 'unknown',
                'connected': is_connected
            },
            details=health_data
        )
        
    except ImportError:
        return HealthStatus(
            component="mt5_connector",
            status=HealthState.UNKNOWN.value,
            message="MT5 connector not available",
            last_check=datetime.now(timezone.utc)
        )
    except Exception as e:
        return HealthStatus(
            component="mt5_connector",
            status=HealthState.UNHEALTHY.value,
            message=f"Connector error: {e}",
            last_check=datetime.now(timezone.utc),
            details={'error': str(e)}
        )


def check_event_bus_health() -> HealthStatus:
    """Check event bus health"""
    if not HAS_EVENT_BUS or not event_bus:
        return HealthStatus(
            component="event_bus",
            status=HealthState.UNKNOWN.value,
            message="Event bus not available",
            last_check=datetime.now(timezone.utc)
        )
    
    try:
        # Test publish/subscribe
        test_received = []
        test_event = f"TEST_HEALTH_{time.time()}"
        
        def test_handler(event):
            test_received.append(event)
        
        # Subscribe and publish
        event_bus.subscribe(test_event, test_handler)
        event_bus.publish(Event(event=test_event, payload={'test': True}))
        
        # Small delay for async processing
        time.sleep(0.1)
        
        # Unsubscribe
        event_bus.unsubscribe(test_event, test_handler)
        
        if test_received:
            status = HealthState.HEALTHY.value
            message = "Event bus operational"
        else:
            status = HealthState.DEGRADED.value
            message = "Event bus slow or not responding"
        
        return HealthStatus(
            component="event_bus",
            status=status,
            message=message,
            last_check=datetime.now(timezone.utc),
            metrics={
                'test_passed': bool(test_received)
            }
        )
        
    except Exception as e:
        return HealthStatus(
            component="event_bus",
            status=HealthState.UNHEALTHY.value,
            message=f"Event bus error: {e}",
            last_check=datetime.now(timezone.utc),
            details={'error': str(e)}
        )


def check_configuration_health() -> HealthStatus:
    """Check configuration system health"""
    if not HAS_CONFIG or not config_loader:
        return HealthStatus(
            component="configuration",
            status=HealthState.UNKNOWN.value,
            message="Configuration system not available",
            last_check=datetime.now(timezone.utc)
        )
    
    try:
        # Check if configs are loaded
        configs = config_loader.get_all_configs()
        config_count = len(configs)
        
        if config_count == 0:
            status = HealthState.UNHEALTHY.value
            message = "No configurations loaded"
        elif config_count < 3:
            status = HealthState.DEGRADED.value
            message = f"Only {config_count} configurations loaded"
        else:
            status = HealthState.HEALTHY.value
            message = f"{config_count} configurations loaded"
        
        return HealthStatus(
            component="configuration",
            status=status,
            message=message,
            last_check=datetime.now(timezone.utc),
            metrics={
                'config_count': config_count,
                'loaded_configs': list(configs.keys())
            }
        )
        
    except Exception as e:
        return HealthStatus(
            component="configuration",
            status=HealthState.UNHEALTHY.value,
            message=f"Configuration error: {e}",
            last_check=datetime.now(timezone.utc),
            details={'error': str(e)}
        )


def check_system_resources() -> HealthStatus:
    """Check system resource usage"""
    if not HAS_PSUTIL or not psutil:
        return HealthStatus(
            component="system_resources",
            status=HealthState.UNKNOWN.value,
            message="psutil not available",
            last_check=datetime.now(timezone.utc)
        )
    
    try:
        # Get resource usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        thresholds = health_checker.thresholds
        
        # Check thresholds
        issues = []
        
        if cpu_percent > thresholds['cpu_percent_critical']:
            issues.append(f"CPU critical: {cpu_percent:.1f}%")
        elif cpu_percent > thresholds['cpu_percent_warning']:
            issues.append(f"CPU high: {cpu_percent:.1f}%")
        
        if memory.percent > thresholds['memory_percent_critical']:
            issues.append(f"Memory critical: {memory.percent:.1f}%")
        elif memory.percent > thresholds['memory_percent_warning']:
            issues.append(f"Memory high: {memory.percent:.1f}%")
        
        if disk.percent > thresholds['disk_percent_critical']:
            issues.append(f"Disk critical: {disk.percent:.1f}%")
        elif disk.percent > thresholds['disk_percent_warning']:
            issues.append(f"Disk high: {disk.percent:.1f}%")
        
        # Determine status
        if any('critical' in issue for issue in issues):
            status = HealthState.UNHEALTHY.value
            message = "; ".join(issues)
        elif issues:
            status = HealthState.DEGRADED.value
            message = "; ".join(issues)
        else:
            status = HealthState.HEALTHY.value
            message = "System resources normal"
        
        return HealthStatus(
            component="system_resources",
            status=status,
            message=message,
            last_check=datetime.now(timezone.utc),
            metrics={
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        )
        
    except Exception as e:
        return HealthStatus(
            component="system_resources",
            status=HealthState.UNHEALTHY.value,
            message=f"Resource check error: {e}",
            last_check=datetime.now(timezone.utc),
            details={'error': str(e)}
        )


# ===========================
# Register Default Checks
# ===========================

# Register all default health checks
health_checker.register_check("database", check_database_health)
health_checker.register_check("mt5_connector", check_mt5_connector_health)
health_checker.register_check("event_bus", check_event_bus_health)
health_checker.register_check("configuration", check_configuration_health)
health_checker.register_check("system_resources", check_system_resources)


# ===========================
# API Functions
# ===========================

def get_health_status() -> Dict[str, Any]:
    """Get current system health status"""
    return health_checker.get_system_health()


def run_health_check(component: str = None) -> Dict[str, HealthStatus]:
    """Run health check for specific component or all"""
    if component:
        return health_checker.run_checks([component])
    return health_checker.run_checks()


def get_health_trends() -> Dict[str, Any]:
    """Get health trend analysis"""
    return health_checker.get_health_trends()