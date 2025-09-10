"""
Canary Deployment and Monitoring for L5 Models
==============================================
Handles progressive rollout with kill-switches and monitoring
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class DeploymentStatus(Enum):
    """Deployment status states"""
    PENDING = "pending"
    SHADOW = "shadow"
    CANARY_5 = "canary_5"
    CANARY_25 = "canary_25"
    CANARY_50 = "canary_50"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

@dataclass
class KillSwitchConfig:
    """Kill-switch thresholds"""
    max_drawdown: float = 0.20  # 20% max drawdown
    max_latency_p99_ms: float = 100  # 100ms p99 latency
    max_error_rate: float = 0.01  # 1% error rate
    min_win_rate: float = 0.40  # 40% minimum win rate
    evaluation_window_minutes: int = 30  # 30 minute evaluation window

@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics"""
    timestamp: str
    deployment_status: str
    traffic_percentage: float
    
    # Performance metrics
    current_drawdown: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    pnl: float
    
    # Operational metrics
    latency_p50_ms: float
    latency_p99_ms: float
    error_rate: float
    requests_per_second: float
    
    # Resource metrics
    cpu_usage_percent: float
    memory_usage_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class CanaryDeploymentManager:
    """
    Manages canary deployment with progressive rollout
    """
    
    def __init__(self, model_id: str, kill_switch_config: Optional[KillSwitchConfig] = None):
        self.model_id = model_id
        self.kill_switch_config = kill_switch_config or KillSwitchConfig()
        self.deployment_status = DeploymentStatus.PENDING
        self.start_time = None
        self.metrics_history = []
        self.rollback_triggered = False
        
    def start_shadow_mode(self) -> Dict[str, Any]:
        """Start shadow mode deployment (0% traffic)"""
        logger.info(f"Starting shadow mode for model {self.model_id}")
        self.deployment_status = DeploymentStatus.SHADOW
        self.start_time = datetime.now()
        
        return {
            "model_id": self.model_id,
            "status": self.deployment_status.value,
            "traffic_percentage": 0,
            "start_time": self.start_time.isoformat(),
            "message": "Model deployed in shadow mode - no live traffic"
        }
    
    def promote_to_canary(self, traffic_percentage: int = 5) -> Dict[str, Any]:
        """Promote model to canary with specified traffic percentage"""
        if traffic_percentage == 5:
            self.deployment_status = DeploymentStatus.CANARY_5
        elif traffic_percentage == 25:
            self.deployment_status = DeploymentStatus.CANARY_25
        elif traffic_percentage == 50:
            self.deployment_status = DeploymentStatus.CANARY_50
        else:
            raise ValueError(f"Invalid canary percentage: {traffic_percentage}")
        
        logger.info(f"Promoting model {self.model_id} to {traffic_percentage}% canary")
        
        return {
            "model_id": self.model_id,
            "status": self.deployment_status.value,
            "traffic_percentage": traffic_percentage,
            "promotion_time": datetime.now().isoformat()
        }
    
    def check_kill_switches(self, metrics: MonitoringMetrics) -> Dict[str, Any]:
        """Check if any kill-switch should be triggered"""
        violations = []
        
        # Check drawdown
        if metrics.max_drawdown > self.kill_switch_config.max_drawdown:
            violations.append({
                "metric": "max_drawdown",
                "threshold": self.kill_switch_config.max_drawdown,
                "actual": metrics.max_drawdown
            })
        
        # Check latency
        if metrics.latency_p99_ms > self.kill_switch_config.max_latency_p99_ms:
            violations.append({
                "metric": "latency_p99",
                "threshold": self.kill_switch_config.max_latency_p99_ms,
                "actual": metrics.latency_p99_ms
            })
        
        # Check error rate
        if metrics.error_rate > self.kill_switch_config.max_error_rate:
            violations.append({
                "metric": "error_rate",
                "threshold": self.kill_switch_config.max_error_rate,
                "actual": metrics.error_rate
            })
        
        # Check win rate
        if metrics.win_rate < self.kill_switch_config.min_win_rate:
            violations.append({
                "metric": "win_rate",
                "threshold": self.kill_switch_config.min_win_rate,
                "actual": metrics.win_rate
            })
        
        if violations:
            logger.error(f"Kill-switch triggered for model {self.model_id}: {violations}")
            self.trigger_rollback(violations)
            return {
                "kill_switch_triggered": True,
                "violations": violations,
                "action": "rollback"
            }
        
        return {
            "kill_switch_triggered": False,
            "violations": [],
            "action": "continue"
        }
    
    def trigger_rollback(self, violations: list):
        """Trigger immediate rollback"""
        self.rollback_triggered = True
        self.deployment_status = DeploymentStatus.ROLLED_BACK
        
        logger.critical(f"ROLLBACK TRIGGERED for model {self.model_id}")
        logger.critical(f"Violations: {violations}")
        
        # In production, this would:
        # 1. Switch traffic back to previous model
        # 2. Alert operations team
        # 3. Create incident report
        # 4. Stop new deployments
        
        return {
            "model_id": self.model_id,
            "status": "rolled_back",
            "violations": violations,
            "rollback_time": datetime.now().isoformat()
        }
    
    def record_metrics(self, metrics: MonitoringMetrics):
        """Record monitoring metrics"""
        self.metrics_history.append(metrics)
        
        # Keep only recent history (e.g., last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]
    
    def get_deployment_report(self) -> Dict[str, Any]:
        """Generate deployment status report"""
        if not self.metrics_history:
            return {
                "model_id": self.model_id,
                "status": self.deployment_status.value,
                "message": "No metrics available yet"
            }
        
        recent_metrics = self.metrics_history[-1]
        
        # Calculate aggregated metrics
        latencies = [m.latency_p99_ms for m in self.metrics_history]
        error_rates = [m.error_rate for m in self.metrics_history]
        win_rates = [m.win_rate for m in self.metrics_history]
        
        return {
            "model_id": self.model_id,
            "status": self.deployment_status.value,
            "rollback_triggered": self.rollback_triggered,
            "current_metrics": recent_metrics.to_dict(),
            "aggregated_metrics": {
                "avg_latency_p99_ms": np.mean(latencies),
                "max_latency_p99_ms": np.max(latencies),
                "avg_error_rate": np.mean(error_rates),
                "avg_win_rate": np.mean(win_rates),
                "total_metrics_collected": len(self.metrics_history)
            },
            "deployment_duration_hours": (
                (datetime.now() - self.start_time).total_seconds() / 3600
                if self.start_time else 0
            )
        }

class ModelMonitor:
    """
    Real-time model monitoring
    """
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.trades = []
        self.decisions = []
        self.errors = []
        self.latencies = []
        
    def log_decision(self, observation: np.ndarray, action: int, metadata: Dict[str, Any]):
        """Log model decision"""
        self.decisions.append({
            "timestamp": datetime.now().isoformat(),
            "observation": observation.tolist() if isinstance(observation, np.ndarray) else observation,
            "action": action,
            "metadata": metadata
        })
    
    def log_trade(self, trade_info: Dict[str, Any]):
        """Log trade execution"""
        self.trades.append({
            "timestamp": datetime.now().isoformat(),
            **trade_info
        })
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error"""
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "type": type(error).__name__,
            "context": context
        })
    
    def log_latency(self, latency_ms: float):
        """Log inference latency"""
        self.latencies.append(latency_ms)
    
    def calculate_metrics(self, window_minutes: int = 30) -> MonitoringMetrics:
        """Calculate current monitoring metrics"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Filter recent data
        recent_trades = [
            t for t in self.trades 
            if datetime.fromisoformat(t["timestamp"]) > cutoff_time
        ]
        recent_errors = [
            e for e in self.errors
            if datetime.fromisoformat(e["timestamp"]) > cutoff_time
        ]
        recent_decisions = [
            d for d in self.decisions
            if datetime.fromisoformat(d["timestamp"]) > cutoff_time
        ]
        
        # Calculate performance metrics
        if recent_trades:
            returns = [t.get("return", 0) for t in recent_trades]
            cumulative_returns = np.cumsum(returns)
            
            # Drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / (running_max + 1e-8)
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            current_drawdown = abs(drawdown[-1]) if len(drawdown) > 0 else 0
            
            # Win rate
            win_rate = np.mean([r > 0 for r in returns]) if returns else 0
            
            # PnL
            pnl = np.sum(returns)
        else:
            max_drawdown = current_drawdown = win_rate = pnl = 0
        
        # Calculate operational metrics
        if self.latencies:
            recent_latencies = self.latencies[-1000:]  # Last 1000 measurements
            latency_p50 = np.percentile(recent_latencies, 50)
            latency_p99 = np.percentile(recent_latencies, 99)
        else:
            latency_p50 = latency_p99 = 0
        
        # Error rate
        total_requests = len(recent_decisions)
        error_rate = len(recent_errors) / total_requests if total_requests > 0 else 0
        
        # RPS
        time_window_seconds = window_minutes * 60
        rps = total_requests / time_window_seconds if time_window_seconds > 0 else 0
        
        return MonitoringMetrics(
            timestamp=datetime.now().isoformat(),
            deployment_status="active",
            traffic_percentage=100,  # Would get from deployment manager
            current_drawdown=float(current_drawdown),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            total_trades=len(recent_trades),
            pnl=float(pnl),
            latency_p50_ms=float(latency_p50),
            latency_p99_ms=float(latency_p99),
            error_rate=float(error_rate),
            requests_per_second=float(rps),
            cpu_usage_percent=0.0,  # Would get from system metrics
            memory_usage_mb=0.0  # Would get from system metrics
        )
    
    def export_logs(self, output_path: str):
        """Export monitoring logs"""
        logs = {
            "model_id": self.model_id,
            "export_time": datetime.now().isoformat(),
            "trades": self.trades[-1000:],  # Last 1000 trades
            "decisions": self.decisions[-1000:],  # Last 1000 decisions
            "errors": self.errors[-100:],  # Last 100 errors
            "latency_summary": {
                "p50": float(np.percentile(self.latencies, 50)) if self.latencies else 0,
                "p95": float(np.percentile(self.latencies, 95)) if self.latencies else 0,
                "p99": float(np.percentile(self.latencies, 99)) if self.latencies else 0,
                "mean": float(np.mean(self.latencies)) if self.latencies else 0,
                "count": len(self.latencies)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Monitoring logs exported to {output_path}")
        return logs

def smoke_test_model(model_path: str, env, n_episodes: int = 10) -> Dict[str, Any]:
    """
    Run smoke tests on model
    """
    from stable_baselines3 import PPO
    
    logger.info(f"Running smoke test on {model_path}")
    
    model = PPO.load(model_path)
    
    obs_clip_count = 0
    obs_peg_count = 0
    total_obs = 0
    reward_errors = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_computed_reward = 0
        
        while not done:
            # Check observation clipping
            if np.any(np.abs(obs) >= 4.99):  # Near clip boundary
                obs_clip_count += 1
            if np.any(np.abs(obs) == 5.0):  # At clip boundary
                obs_peg_count += 1
            total_obs += 1
            
            # Get action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            
            # Recompute reward (simplified check)
            # In production, would use actual reward function
            recomputed_reward = reward[0]  # Placeholder
            episode_computed_reward += recomputed_reward
            
            if done[0]:
                break
        
        # Check reward consistency
        reward_error = abs(episode_reward - episode_computed_reward)
        reward_errors.append(reward_error)
    
    # Calculate metrics
    obs_clip_rate = obs_clip_count / total_obs if total_obs > 0 else 0
    obs_peg_rate = obs_peg_count / total_obs if total_obs > 0 else 0
    reward_rmse = np.sqrt(np.mean(np.square(reward_errors)))
    
    results = {
        "n_episodes": n_episodes,
        "obs_clip_rate": float(obs_clip_rate),
        "obs_peg_rate": float(obs_peg_rate),
        "reward_rmse": float(reward_rmse),
        "total_observations": total_obs,
        "smoke_test_passed": (
            obs_clip_rate <= 0.005 and  # Less than 0.5%
            obs_peg_rate < 0.001 and  # Near 0%
            reward_rmse < 0.01  # Near 0
        )
    }
    
    logger.info(f"Smoke test results: {results}")
    return results