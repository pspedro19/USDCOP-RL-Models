"""
Production Monitoring Dashboard for L5 Models
=============================================
Real-time monitoring, alerting, and visualization
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from enum import Enum
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricStatus(Enum):
    """Metric health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class AlertThresholds:
    """Configurable alert thresholds"""
    # Performance thresholds
    min_sortino: float = 1.0
    max_drawdown: float = 0.20
    min_calmar: float = 0.6
    min_win_rate: float = 0.40
    
    # Operational thresholds
    max_latency_p99_ms: float = 100
    max_error_rate: float = 0.02
    min_requests_per_second: float = 10
    
    # Resource thresholds
    max_cpu_percent: float = 80
    max_memory_gb: float = 4.0
    
    # Drift thresholds
    max_feature_drift: float = 0.3
    max_prediction_drift: float = 0.25

@dataclass
class ModelHealthScore:
    """Overall model health assessment"""
    timestamp: str
    model_id: str
    health_score: float  # 0-100
    status: str
    
    performance_score: float
    operational_score: float
    resource_score: float
    drift_score: float
    
    alerts: List[Dict[str, Any]]
    recommendations: List[str]

class ProductionMonitor:
    """
    Comprehensive production monitoring system
    """
    
    def __init__(self, model_id: str, thresholds: Optional[AlertThresholds] = None):
        self.model_id = model_id
        self.thresholds = thresholds or AlertThresholds()
        self.metrics_buffer = []
        self.alerts_history = []
        self.health_history = []
        
    def calculate_health_score(self, metrics: Dict[str, Any]) -> ModelHealthScore:
        """Calculate comprehensive health score"""
        
        # Performance score (40% weight)
        perf_score = self._calculate_performance_score(metrics)
        
        # Operational score (30% weight)
        ops_score = self._calculate_operational_score(metrics)
        
        # Resource score (20% weight)
        resource_score = self._calculate_resource_score(metrics)
        
        # Drift score (10% weight)
        drift_score = self._calculate_drift_score(metrics)
        
        # Weighted overall score
        overall_score = (
            perf_score * 0.4 +
            ops_score * 0.3 +
            resource_score * 0.2 +
            drift_score * 0.1
        )
        
        # Determine status
        if overall_score >= 80:
            status = MetricStatus.HEALTHY.value
        elif overall_score >= 60:
            status = MetricStatus.DEGRADED.value
        elif overall_score >= 40:
            status = MetricStatus.UNHEALTHY.value
        else:
            status = MetricStatus.CRITICAL.value
        
        # Generate alerts
        alerts = self._generate_alerts(metrics, overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            perf_score, ops_score, resource_score, drift_score
        )
        
        health_score = ModelHealthScore(
            timestamp=datetime.now().isoformat(),
            model_id=self.model_id,
            health_score=float(overall_score),
            status=status,
            performance_score=float(perf_score),
            operational_score=float(ops_score),
            resource_score=float(resource_score),
            drift_score=float(drift_score),
            alerts=alerts,
            recommendations=recommendations
        )
        
        self.health_history.append(health_score)
        return health_score
    
    def _calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate performance subscore"""
        score = 100.0
        
        # Sortino ratio
        sortino = metrics.get("sortino_ratio", 0)
        if sortino < self.thresholds.min_sortino:
            penalty = (self.thresholds.min_sortino - sortino) / self.thresholds.min_sortino
            score -= min(30, penalty * 30)
        
        # Maximum drawdown
        max_dd = metrics.get("max_drawdown", 1.0)
        if max_dd > self.thresholds.max_drawdown:
            penalty = (max_dd - self.thresholds.max_drawdown) / self.thresholds.max_drawdown
            score -= min(30, penalty * 30)
        
        # Calmar ratio
        calmar = metrics.get("calmar_ratio", 0)
        if calmar < self.thresholds.min_calmar:
            penalty = (self.thresholds.min_calmar - calmar) / self.thresholds.min_calmar
            score -= min(20, penalty * 20)
        
        # Win rate
        win_rate = metrics.get("win_rate", 0)
        if win_rate < self.thresholds.min_win_rate:
            penalty = (self.thresholds.min_win_rate - win_rate) / self.thresholds.min_win_rate
            score -= min(20, penalty * 20)
        
        return max(0, score)
    
    def _calculate_operational_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate operational subscore"""
        score = 100.0
        
        # Latency
        latency_p99 = metrics.get("latency_p99_ms", 0)
        if latency_p99 > self.thresholds.max_latency_p99_ms:
            penalty = (latency_p99 - self.thresholds.max_latency_p99_ms) / self.thresholds.max_latency_p99_ms
            score -= min(40, penalty * 40)
        
        # Error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.thresholds.max_error_rate:
            penalty = error_rate / self.thresholds.max_error_rate
            score -= min(40, penalty * 40)
        
        # Throughput
        rps = metrics.get("requests_per_second", 0)
        if rps < self.thresholds.min_requests_per_second:
            penalty = (self.thresholds.min_requests_per_second - rps) / self.thresholds.min_requests_per_second
            score -= min(20, penalty * 20)
        
        return max(0, score)
    
    def _calculate_resource_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate resource utilization subscore"""
        score = 100.0
        
        # CPU usage
        cpu = metrics.get("cpu_usage_percent", 0)
        if cpu > self.thresholds.max_cpu_percent:
            penalty = (cpu - self.thresholds.max_cpu_percent) / 20  # 20% buffer
            score -= min(50, penalty * 50)
        
        # Memory usage
        memory_gb = metrics.get("memory_usage_gb", 0)
        if memory_gb > self.thresholds.max_memory_gb:
            penalty = (memory_gb - self.thresholds.max_memory_gb) / self.thresholds.max_memory_gb
            score -= min(50, penalty * 50)
        
        return max(0, score)
    
    def _calculate_drift_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate drift subscore"""
        score = 100.0
        
        # Feature drift
        feature_drift = metrics.get("feature_drift", 0)
        if feature_drift > self.thresholds.max_feature_drift:
            penalty = feature_drift / self.thresholds.max_feature_drift
            score -= min(50, penalty * 50)
        
        # Prediction drift
        pred_drift = metrics.get("prediction_drift", 0)
        if pred_drift > self.thresholds.max_prediction_drift:
            penalty = pred_drift / self.thresholds.max_prediction_drift
            score -= min(50, penalty * 50)
        
        return max(0, score)
    
    def _generate_alerts(self, metrics: Dict[str, Any], health_score: float) -> List[Dict[str, Any]]:
        """Generate alerts based on metrics"""
        alerts = []
        
        # Critical health score
        if health_score < 40:
            alerts.append({
                "level": AlertLevel.CRITICAL.value,
                "message": f"Model health critically low: {health_score:.1f}/100",
                "metric": "health_score",
                "value": health_score,
                "threshold": 40
            })
        
        # Performance alerts
        if metrics.get("max_drawdown", 0) > self.thresholds.max_drawdown:
            alerts.append({
                "level": AlertLevel.WARNING.value,
                "message": f"Maximum drawdown exceeds threshold: {metrics['max_drawdown']:.1%}",
                "metric": "max_drawdown",
                "value": metrics["max_drawdown"],
                "threshold": self.thresholds.max_drawdown
            })
        
        # Operational alerts
        if metrics.get("error_rate", 0) > self.thresholds.max_error_rate:
            alerts.append({
                "level": AlertLevel.CRITICAL.value,
                "message": f"Error rate too high: {metrics['error_rate']:.1%}",
                "metric": "error_rate",
                "value": metrics["error_rate"],
                "threshold": self.thresholds.max_error_rate
            })
        
        # Resource alerts
        if metrics.get("cpu_usage_percent", 0) > self.thresholds.max_cpu_percent:
            alerts.append({
                "level": AlertLevel.WARNING.value,
                "message": f"High CPU usage: {metrics['cpu_usage_percent']:.1f}%",
                "metric": "cpu_usage_percent",
                "value": metrics["cpu_usage_percent"],
                "threshold": self.thresholds.max_cpu_percent
            })
        
        return alerts
    
    def _generate_recommendations(
        self, 
        perf_score: float, 
        ops_score: float, 
        resource_score: float, 
        drift_score: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if perf_score < 60:
            recommendations.append("Consider retraining model - performance degradation detected")
        
        if ops_score < 60:
            recommendations.append("Review system latency and error handling")
        
        if resource_score < 60:
            recommendations.append("Scale resources or optimize model inference")
        
        if drift_score < 60:
            recommendations.append("Investigate data drift - may need model update")
        
        if perf_score < 40 or ops_score < 40:
            recommendations.append("URGENT: Consider rolling back to previous model version")
        
        return recommendations

class DashboardGenerator:
    """
    Generate interactive monitoring dashboards
    """
    
    def __init__(self, monitor: ProductionMonitor):
        self.monitor = monitor
    
    def generate_health_dashboard(self, metrics_history: List[Dict[str, Any]]) -> str:
        """Generate comprehensive health dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Model Health Score', 'Performance Metrics',
                'Operational Metrics', 'Resource Utilization',
                'Alert Timeline', 'Drift Detection'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # Extract time series
        timestamps = [m['timestamp'] for m in metrics_history]
        
        # 1. Health Score
        health_scores = [m.get('health_score', 0) for m in metrics_history]
        fig.add_trace(
            go.Scatter(
                x=timestamps, 
                y=health_scores,
                mode='lines+markers',
                name='Health Score',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=80, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=60, line_dash="dash", line_color="yellow", row=1, col=1)
        fig.add_hline(y=40, line_dash="dash", line_color="red", row=1, col=1)
        
        # 2. Performance Metrics
        sortino_ratios = [m.get('sortino_ratio', 0) for m in metrics_history]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=sortino_ratios,
                mode='lines',
                name='Sortino Ratio',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        
        # 3. Operational Metrics
        latencies = [m.get('latency_p99_ms', 0) for m in metrics_history]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=latencies,
                mode='lines',
                name='P99 Latency (ms)',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # 4. Resource Utilization
        cpu_usage = [m.get('cpu_usage_percent', 0) for m in metrics_history]
        memory_usage = [m.get('memory_usage_gb', 0) * 25 for m in metrics_history]  # Scale for visibility
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cpu_usage,
                mode='lines',
                name='CPU %',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=memory_usage,
                mode='lines',
                name='Memory (GB*25)',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # 5. Alert Timeline
        alert_counts = self._aggregate_alerts(metrics_history)
        fig.add_trace(
            go.Bar(
                x=list(alert_counts.keys()),
                y=list(alert_counts.values()),
                name='Alerts',
                marker_color=['green', 'yellow', 'orange', 'red']
            ),
            row=3, col=1
        )
        
        # 6. Drift Detection
        feature_drift = [m.get('feature_drift', 0) for m in metrics_history]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=feature_drift,
                mode='lines+markers',
                name='Feature Drift',
                line=dict(color='brown')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Production Monitoring Dashboard - Model {self.monitor.model_id}",
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save to HTML
        dashboard_path = f"/tmp/monitoring_dashboard_{self.monitor.model_id}.html"
        fig.write_html(dashboard_path)
        
        logger.info(f"Dashboard generated: {dashboard_path}")
        return dashboard_path
    
    def _aggregate_alerts(self, metrics_history: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate alerts by level"""
        alert_counts = {
            AlertLevel.INFO.value: 0,
            AlertLevel.WARNING.value: 0,
            AlertLevel.CRITICAL.value: 0,
            AlertLevel.EMERGENCY.value: 0
        }
        
        for metrics in metrics_history:
            alerts = metrics.get('alerts', [])
            for alert in alerts:
                level = alert.get('level', AlertLevel.INFO.value)
                alert_counts[level] = alert_counts.get(level, 0) + 1
        
        return alert_counts
    
    def generate_pnl_dashboard(self, trades: List[Dict[str, Any]]) -> str:
        """Generate PnL analysis dashboard"""
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)
        if df.empty:
            return self._generate_empty_dashboard()
        
        # Calculate cumulative PnL
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        # Calculate rolling metrics
        df['rolling_sharpe'] = df['pnl'].rolling(window=100).apply(
            lambda x: x.mean() / x.std() if x.std() > 0 else 0
        )
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Cumulative PnL', 'Trade Distribution',
                'Rolling Sharpe', 'Drawdown'
            )
        )
        
        # 1. Cumulative PnL
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['cumulative_pnl'],
                mode='lines',
                name='Cumulative PnL',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # 2. Trade Distribution
        fig.add_trace(
            go.Histogram(
                x=df['pnl'],
                name='PnL Distribution',
                marker_color='blue',
                nbinsx=50
            ),
            row=1, col=2
        )
        
        # 3. Rolling Sharpe
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rolling_sharpe'],
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        # 4. Drawdown
        running_max = df['cumulative_pnl'].expanding().max()
        drawdown = (df['cumulative_pnl'] - running_max) / (running_max + 1e-8)
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=drawdown * 100,
                mode='lines',
                fill='tozeroy',
                name='Drawdown %',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="PnL Analysis Dashboard",
            height=700,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save to HTML
        dashboard_path = f"/tmp/pnl_dashboard_{self.monitor.model_id}.html"
        fig.write_html(dashboard_path)
        
        return dashboard_path
    
    def _generate_empty_dashboard(self) -> str:
        """Generate empty dashboard when no data available"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        
        fig.update_layout(
            title="Waiting for Data",
            height=400,
            template='plotly_white'
        )
        
        dashboard_path = f"/tmp/empty_dashboard_{self.monitor.model_id}.html"
        fig.write_html(dashboard_path)
        
        return dashboard_path

class AlertManager:
    """
    Manage alerts and notifications
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.alert_history = []
        self.notification_channels = []
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        alert['timestamp'] = datetime.now().isoformat()
        alert['model_id'] = self.model_id
        
        self.alert_history.append(alert)
        
        # Log alert
        if alert['level'] == AlertLevel.CRITICAL.value:
            logger.critical(f"ALERT: {alert['message']}")
        elif alert['level'] == AlertLevel.WARNING.value:
            logger.warning(f"ALERT: {alert['message']}")
        else:
            logger.info(f"ALERT: {alert['message']}")
        
        # In production, would send to:
        # - Slack/Teams
        # - PagerDuty
        # - Email
        # - SMS for critical alerts
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        return [
            a for a in self.alert_history
            if datetime.fromisoformat(a['timestamp']) > cutoff
        ]
    
    def clear_resolved_alerts(self, resolved_metrics: List[str]):
        """Clear alerts for resolved metrics"""
        self.alert_history = [
            a for a in self.alert_history
            if a.get('metric') not in resolved_metrics
        ]

def create_monitoring_report(
    model_id: str,
    metrics_history: List[Dict[str, Any]],
    health_scores: List[ModelHealthScore]
) -> Dict[str, Any]:
    """Create comprehensive monitoring report"""
    
    if not metrics_history:
        return {
            "model_id": model_id,
            "status": "no_data",
            "message": "No monitoring data available"
        }
    
    # Calculate summary statistics
    recent_metrics = metrics_history[-100:]  # Last 100 data points
    
    # Performance summary
    performance_summary = {
        "avg_sortino": np.mean([m.get('sortino_ratio', 0) for m in recent_metrics]),
        "avg_sharpe": np.mean([m.get('sharpe_ratio', 0) for m in recent_metrics]),
        "max_drawdown": np.max([m.get('max_drawdown', 0) for m in recent_metrics]),
        "win_rate": np.mean([m.get('win_rate', 0) for m in recent_metrics]),
        "total_trades": sum([m.get('num_trades', 0) for m in recent_metrics])
    }
    
    # Operational summary
    operational_summary = {
        "avg_latency_p50_ms": np.mean([m.get('latency_p50_ms', 0) for m in recent_metrics]),
        "avg_latency_p99_ms": np.mean([m.get('latency_p99_ms', 0) for m in recent_metrics]),
        "error_rate": np.mean([m.get('error_rate', 0) for m in recent_metrics]),
        "availability": 1.0 - np.mean([m.get('error_rate', 0) for m in recent_metrics])
    }
    
    # Health summary
    if health_scores:
        recent_health = health_scores[-10:]  # Last 10 health assessments
        health_summary = {
            "current_score": recent_health[-1].health_score,
            "avg_score": np.mean([h.health_score for h in recent_health]),
            "trend": "improving" if recent_health[-1].health_score > recent_health[0].health_score else "degrading",
            "status": recent_health[-1].status
        }
    else:
        health_summary = {"status": "unknown"}
    
    # Alert summary
    alert_counts = {}
    for metrics in recent_metrics:
        for alert in metrics.get('alerts', []):
            level = alert.get('level', 'unknown')
            alert_counts[level] = alert_counts.get(level, 0) + 1
    
    return {
        "model_id": model_id,
        "report_time": datetime.now().isoformat(),
        "monitoring_window_hours": len(recent_metrics) * 0.1,  # Assuming 6-minute intervals
        "performance_summary": performance_summary,
        "operational_summary": operational_summary,
        "health_summary": health_summary,
        "alert_summary": alert_counts,
        "recommendations": health_scores[-1].recommendations if health_scores else []
    }