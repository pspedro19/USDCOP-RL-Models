"""
Chaos Engineering Module
========================
Implements chaos engineering experiments to test system resilience.
"""

import asyncio
import json
import logging
import random
import signal
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of a chaos experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentSeverity(Enum):
    """Severity level of a chaos experiment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    rollback_successful: bool = True


class ChaosExperiment:
    """Base class for chaos experiments."""
    
    def __init__(self, name: str, description: str = "", severity: ExperimentSeverity = ExperimentSeverity.MEDIUM):
        self.name = name
        self.description = description
        self.severity = severity
        self.status = ExperimentStatus.PENDING
        self.result: Optional[ExperimentResult] = None
        self.rollback_actions: List[Callable] = []
        self.metrics_collector: Optional[Callable] = None
    
    def setup(self) -> bool:
        """Setup the experiment environment."""
        try:
            logger.info(f"Setting up chaos experiment: {self.name}")
            return self._setup_impl()
        except Exception as e:
            logger.error(f"Failed to setup experiment {self.name}: {str(e)}")
            return False
    
    def execute(self) -> bool:
        """Execute the chaos experiment."""
        try:
            logger.info(f"Executing chaos experiment: {self.name}")
            self.status = ExperimentStatus.RUNNING
            start_time = datetime.now()
            
            success = self._execute_impl()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.result = ExperimentResult(
                experiment_name=self.name,
                status=ExperimentStatus.COMPLETED if success else ExperimentStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success
            )
            
            if success:
                self.status = ExperimentStatus.COMPLETED
                logger.info(f"Chaos experiment {self.name} completed successfully")
            else:
                self.status = ExperimentStatus.FAILED
                logger.error(f"Chaos experiment {self.name} failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing experiment {self.name}: {str(e)}")
            self.status = ExperimentStatus.FAILED
            self.result = ExperimentResult(
                experiment_name=self.name,
                status=ExperimentStatus.FAILED,
                start_time=datetime.now(),
                success=False,
                error_message=str(e)
            )
            return False
    
    def rollback(self) -> bool:
        """Rollback the experiment effects."""
        try:
            logger.info(f"Rolling back chaos experiment: {self.name}")
            
            rollback_success = True
            for rollback_action in self.rollback_actions:
                try:
                    if not rollback_action():
                        rollback_success = False
                except Exception as e:
                    logger.error(f"Rollback action failed: {str(e)}")
                    rollback_success = False
            
            if self.result:
                self.result.rollback_successful = rollback_success
            
            return rollback_success
            
        except Exception as e:
            logger.error(f"Error rolling back experiment {self.name}: {str(e)}")
            return False
    
    def add_rollback_action(self, action: Callable):
        """Add a rollback action."""
        self.rollback_actions.append(action)
    
    def set_metrics_collector(self, collector: Callable):
        """Set a metrics collector function."""
        self.metrics_collector = collector
    
    def _setup_impl(self) -> bool:
        """Implementation of setup - to be overridden by subclasses."""
        return True
    
    def _execute_impl(self) -> bool:
        """Implementation of execute - to be overridden by subclasses."""
        return True


class KillServiceExperiment(ChaosExperiment):
    """Experiment to kill a service and observe recovery."""
    
    def __init__(self, service_name: str, container_name: str = None):
        super().__init__(
            name=f"kill_service_{service_name}",
            description=f"Kill service {service_name} and observe recovery",
            severity=ExperimentSeverity.HIGH
        )
        self.service_name = service_name
        self.container_name = container_name or service_name
        self.original_container_id = None
    
    def _setup_impl(self) -> bool:
        """Setup: identify the container to kill."""
        try:
            # Use docker ps to find the container
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.ID}}"],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0 and result.stdout.strip():
                self.original_container_id = result.stdout.strip()
                logger.info(f"Found container {self.container_name} with ID: {self.original_container_id}")
                return True
            else:
                logger.warning(f"Container {self.container_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error finding container: {str(e)}")
            return False
    
    def _execute_impl(self) -> bool:
        """Execute: kill the service container."""
        try:
            if not self.original_container_id:
                logger.error("No container ID available for killing")
                return False
            
            # Kill the container
            result = subprocess.run(
                ["docker", "kill", self.original_container_id],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully killed container {self.original_container_id}")
                return True
            else:
                logger.error(f"Failed to kill container: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error killing container: {str(e)}")
            return False
    
    def add_rollback_action(self, action: Callable):
        """Add rollback action to restart the service."""
        super().add_rollback_action(action)


class NetworkDelayExperiment(ChaosExperiment):
    """Experiment to introduce network delays."""
    
    def __init__(self, target_host: str, delay_ms: int = 100, duration: int = 60):
        super().__init__(
            name=f"network_delay_{target_host}_{delay_ms}ms",
            description=f"Introduce {delay_ms}ms network delay to {target_host} for {duration}s",
            severity=ExperimentSeverity.MEDIUM
        )
        self.target_host = target_host
        self.delay_ms = delay_ms
        self.duration = duration
        self.tc_process = None
    
    def _execute_impl(self) -> bool:
        """Execute: use tc (traffic control) to introduce delay."""
        try:
            # Use tc to add delay to network interface
            cmd = [
                "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
                "delay", f"{self.delay_ms}ms"
            ]
            
            self.tc_process = subprocess.Popen(cmd, capture_output=True)
            
            # Wait for the specified duration
            time.sleep(self.duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error introducing network delay: {str(e)}")
            return False
    
    def rollback(self) -> bool:
        """Rollback: remove the network delay."""
        try:
            if self.tc_process:
                self.tc_process.terminate()
            
            # Remove the tc rule
            cleanup_cmd = ["tc", "qdisc", "del", "dev", "eth0", "root"]
            subprocess.run(cleanup_cmd, capture_output=True, timeout=10)
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing network delay: {str(e)}")
            return False


class ResourceLimitsExperiment(ChaosExperiment):
    """Experiment to limit system resources."""
    
    def __init__(self, resource_type: str, limit_value: str, duration: int = 60):
        super().__init__(
            name=f"resource_limit_{resource_type}_{limit_value}",
            description=f"Limit {resource_type} to {limit_value} for {duration}s",
            severity=ExperimentSeverity.MEDIUM
        )
        self.resource_type = resource_type
        self.limit_value = limit_value
        self.duration = duration
        self.cgroup_path = None
    
    def _execute_impl(self) -> bool:
        """Execute: use cgroups to limit resources."""
        try:
            # Create a cgroup for the experiment
            cgroup_name = f"chaos_{self.name}_{int(time.time())}"
            self.cgroup_path = f"/sys/fs/cgroup/{self.resource_type}/{cgroup_name}"
            
            # Create the cgroup directory
            Path(self.cgroup_path).mkdir(parents=True, exist_ok=True)
            
            # Set the resource limit
            limit_file = Path(self.cgroup_path) / f"{self.resource_type}.limit"
            limit_file.write_text(self.limit_value)
            
            # Move current process to the cgroup
            tasks_file = Path(self.cgroup_path) / "tasks"
            tasks_file.write_text(str(subprocess.run(["pgrep", "-f", "python"], capture_output=True, text=True).stdout.strip()))
            
            # Wait for the specified duration
            time.sleep(self.duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting resource limits: {str(e)}")
            return False
    
    def rollback(self) -> bool:
        """Rollback: remove resource limits."""
        try:
            if self.cgroup_path and Path(self.cgroup_path).exists():
                # Move processes back to root cgroup
                root_tasks = Path("/sys/fs/cgroup/tasks")
                tasks_file = Path(self.cgroup_path) / "tasks"
                
                if tasks_file.exists():
                    for task_id in tasks_file.read_text().strip().split('\n'):
                        if task_id:
                            root_tasks.write_text(task_id)
                
                # Remove the cgroup
                import shutil
                shutil.rmtree(self.cgroup_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error removing resource limits: {str(e)}")
            return False


class MarketCrashExperiment(ChaosExperiment):
    """Experiment to simulate market crash scenarios."""
    
    def __init__(self, crash_type: str, duration: int = 300):
        super().__init__(
            name=f"market_crash_{crash_type}",
            description=f"Simulate {crash_type} market crash for {duration}s",
            severity=ExperimentSeverity.CRITICAL
        )
        self.crash_type = crash_type
        self.duration = duration
        self.original_data = None
    
    def _execute_impl(self) -> bool:
        """Execute: simulate market crash by manipulating data."""
        try:
            # Simulate different types of market crashes
            if self.crash_type == "flash_crash":
                # Simulate rapid price decline
                self._simulate_flash_crash()
            elif self.crash_type == "liquidity_crisis":
                # Simulate lack of market liquidity
                self._simulate_liquidity_crisis()
            elif self.crash_type == "volatility_spike":
                # Simulate extreme volatility
                self._simulate_volatility_spike()
            else:
                logger.warning(f"Unknown crash type: {self.crash_type}")
                return False
            
            # Wait for the specified duration
            time.sleep(self.duration)
            
            return True
            
        except Exception as e:
            logger.error(f"Error simulating market crash: {str(e)}")
            return False
    
    def _simulate_flash_crash(self):
        """Simulate a flash crash scenario."""
        logger.info("Simulating flash crash: rapid price decline")
        # In a real implementation, this would manipulate market data
        # or trigger specific trading conditions
    
    def _simulate_liquidity_crisis(self):
        """Simulate a liquidity crisis scenario."""
        logger.info("Simulating liquidity crisis: reduced market depth")
        # In a real implementation, this would reduce available liquidity
    
    def _simulate_volatility_spike(self):
        """Simulate extreme volatility scenario."""
        logger.info("Simulating volatility spike: increased price swings")
        # In a real implementation, this would increase price volatility
    
    def rollback(self) -> bool:
        """Rollback: restore normal market conditions."""
        try:
            logger.info("Restoring normal market conditions")
            # In a real implementation, this would restore normal data feeds
            return True
        except Exception as e:
            logger.error(f"Error restoring market conditions: {str(e)}")
            return False


class ExperimentRegistry:
    """Registry for managing chaos experiments."""
    
    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[ExperimentResult] = []
    
    def register_experiment(self, experiment: ChaosExperiment):
        """Register a chaos experiment."""
        self.experiments[experiment.name] = experiment
        logger.info(f"Registered chaos experiment: {experiment.name}")
    
    def get_experiment(self, name: str) -> Optional[ChaosExperiment]:
        """Get an experiment by name."""
        return self.experiments.get(name)
    
    def list_experiments(self) -> List[str]:
        """List all registered experiments."""
        return list(self.experiments.keys())
    
    def run_experiment(self, name: str) -> Optional[ExperimentResult]:
        """Run a specific experiment."""
        experiment = self.get_experiment(name)
        if not experiment:
            logger.error(f"Experiment {name} not found")
            return None
        
        try:
            # Setup the experiment
            if not experiment.setup():
                logger.error(f"Failed to setup experiment {name}")
                return None
            
            # Execute the experiment
            success = experiment.execute()
            
            # Record the result
            if experiment.result:
                self.experiment_history.append(experiment.result)
            
            return experiment.result
            
        except Exception as e:
            logger.error(f"Error running experiment {name}: {str(e)}")
            return None
    
    def get_experiment_history(self) -> List[ExperimentResult]:
        """Get the history of all experiments."""
        return self.experiment_history.copy()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of experiment results."""
        if not self.experiment_history:
            return {"total_experiments": 0}
        
        total_experiments = len(self.experiment_history)
        successful_experiments = sum(1 for r in self.experiment_history if r.success)
        failed_experiments = total_experiments - successful_experiments
        
        return {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "failed_experiments": failed_experiments,
            "success_rate": (successful_experiments / total_experiments * 100) if total_experiments > 0 else 0,
            "experiments": [asdict(r) for r in self.experiment_history]
        }


class ChaosOrchestrator:
    """Orchestrates chaos engineering experiments."""
    
    def __init__(self):
        self.registry = ExperimentRegistry()
        self.scheduler = None
        self.running_experiments: Dict[str, ChaosExperiment] = {}
        self.stop_event = threading.Event()
        
        # Register default experiments
        self._register_default_experiments()
    
    def _register_default_experiments(self):
        """Register default chaos experiments."""
        # Service failure experiments
        self.registry.register_experiment(
            KillServiceExperiment("trading-app", "trading-app")
        )
        self.registry.register_experiment(
            KillServiceExperiment("redis", "trading-redis")
        )
        
        # Network experiments
        self.registry.register_experiment(
            NetworkDelayExperiment("localhost", 200, 30)
        )
        self.registry.register_experiment(
            NetworkDelayExperiment("localhost", 500, 60)
        )
        
        # Resource experiments
        self.registry.register_experiment(
            ResourceLimitsExperiment("memory", "512M", 120)
        )
        self.registry.register_experiment(
            ResourceLimitsExperiment("cpu", "50%", 180)
        )
        
        # Market experiments
        self.registry.register_experiment(
            MarketCrashExperiment("flash_crash", 300)
        )
        self.registry.register_experiment(
            MarketCrashExperiment("volatility_spike", 600)
        )
    
    def run_experiment(self, name: str) -> Optional[ExperimentResult]:
        """Run a specific experiment."""
        return self.registry.run_experiment(name)
    
    def run_experiment_batch(self, experiment_names: List[str]) -> List[ExperimentResult]:
        """Run multiple experiments in sequence."""
        results = []
        
        for name in experiment_names:
            logger.info(f"Running experiment batch: {name}")
            result = self.run_experiment(name)
            if result:
                results.append(result)
            
            # Wait between experiments
            time.sleep(5)
        
        return results
    
    def run_random_experiment(self) -> Optional[ExperimentResult]:
        """Run a random experiment from the registry."""
        available_experiments = self.registry.list_experiments()
        if not available_experiments:
            return None
        
        experiment_name = random.choice(available_experiments)
        return self.run_experiment(experiment_name)
    
    def schedule_experiments(self, schedule: Dict[str, Any]):
        """Schedule experiments to run at specific times."""
        # This is a simplified scheduler - in production, you might use
        # a proper task scheduler like Celery or APScheduler
        
        for experiment_name, schedule_info in schedule.items():
            if experiment_name in self.registry.experiments:
                # Schedule the experiment
                delay = schedule_info.get('delay', 0)
                repeat = schedule_info.get('repeat', False)
                interval = schedule_info.get('interval', 3600)  # 1 hour default
                
                if delay > 0:
                    threading.Timer(delay, self._run_scheduled_experiment, 
                                  args=[experiment_name, repeat, interval]).start()
    
    def _run_scheduled_experiment(self, experiment_name: str, repeat: bool, interval: int):
        """Run a scheduled experiment."""
        if self.stop_event.is_set():
            return
        
        logger.info(f"Running scheduled experiment: {experiment_name}")
        self.run_experiment(experiment_name)
        
        if repeat:
            # Schedule the next run
            threading.Timer(interval, self._run_scheduled_experiment, 
                          args=[experiment_name, repeat, interval]).start()
    
    def stop(self):
        """Stop the chaos orchestrator."""
        self.stop_event.set()
        logger.info("Chaos orchestrator stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the chaos orchestrator."""
        return {
            "running_experiments": len(self.running_experiments),
            "registered_experiments": len(self.registry.experiments),
            "experiment_history": len(self.registry.experiment_history),
            "summary": self.registry.get_experiment_summary()
        }


def run_chaos_experiment(experiment_name: str, orchestrator: ChaosOrchestrator = None) -> Optional[ExperimentResult]:
    """Convenience function to run a chaos experiment."""
    if orchestrator is None:
        orchestrator = ChaosOrchestrator()
    
    return orchestrator.run_experiment(experiment_name)


def schedule_chaos_tests(schedule_config: Dict[str, Any], orchestrator: ChaosOrchestrator = None):
    """Schedule chaos tests based on configuration."""
    if orchestrator is None:
        orchestrator = ChaosOrchestrator()
    
    orchestrator.schedule_experiments(schedule_config)
    return orchestrator
