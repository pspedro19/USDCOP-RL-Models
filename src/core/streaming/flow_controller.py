"""
Flow Controller
==============
Adaptive flow control for managing data flow rates.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from .backpressure_buffer import BackpressureQueue
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class FlowControlConfig:
    """Configuration for flow control"""
    name: str
    target_rate_per_sec: float = 100.0
    min_rate_per_sec: float = 10.0
    max_rate_per_sec: float = 1000.0
    adaptation_factor: float = 0.1
    measurement_window: float = 10.0
    enable_adaptive: bool = True


class FlowController:
    """Adaptive flow controller for managing data flow"""
    
    def __init__(self, config: FlowControlConfig):
        self.config = config
        
        # Flow control state
        self._current_rate = config.target_rate_per_sec
        self._measurements = []
        self._last_measurement = time.time()
        self._lock = threading.RLock()
        
        # Rate limiter
        self._rate_limiter = RateLimiter()
        self._rate_limiter.add_bucket(config.name, self._current_rate)
        
        # Callbacks
        self._on_rate_change: Optional[Callable] = None
        self._on_flow_control: Optional[Callable] = None
        
        logger.info(f"Flow controller '{config.name}' initialized with rate {self._current_rate}/sec")
    
    @property
    def current_rate(self) -> float:
        """Current flow rate"""
        return self._current_rate
    
    @property
    def target_rate(self) -> float:
        """Target flow rate"""
        return self.config.target_rate_per_sec
    
    def set_rate_change_callback(self, callback: Callable):
        """Set callback for rate changes"""
        self._on_rate_change = callback
    
    def set_flow_control_callback(self, callback: Callable):
        """Set callback for flow control events"""
        self._on_flow_control = callback
    
    def measure_flow(self, count: int = 1):
        """Measure current flow rate"""
        with self._lock:
            now = time.time()
            
            # Add measurement
            self._measurements.append((now, count))
            
            # Remove old measurements outside window
            cutoff_time = now - self.config.measurement_window
            self._measurements = [(t, c) for t, c in self._measurements if t > cutoff_time]
            
            # Calculate current rate
            if self._measurements:
                total_count = sum(c for _, c in self._measurements)
                time_span = self._measurements[-1][0] - self._measurements[0][0]
                if time_span > 0:
                    measured_rate = total_count / time_span
                    
                    # Adaptive rate adjustment
                    if self.config.enable_adaptive:
                        self._adapt_rate(measured_rate)
    
    def _adapt_rate(self, measured_rate: float):
        """Adapt flow rate based on measurements"""
        with self._lock:
            # Calculate rate difference
            rate_diff = measured_rate - self._current_rate
            
            # Apply adaptation
            adaptation = rate_diff * self.config.adaptation_factor
            new_rate = self._current_rate + adaptation
            
            # Clamp to limits
            new_rate = max(self.config.min_rate_per_sec, 
                          min(self.config.max_rate_per_sec, new_rate))
            
            # Update if changed significantly
            if abs(new_rate - self._current_rate) > 0.1:
                old_rate = self._current_rate
                self._current_rate = new_rate
                
                # Update rate limiter
                self._rate_limiter.remove_bucket(self.config.name)
                self._rate_limiter.add_bucket(self.config.name, self._current_rate)
                
                logger.info(f"Flow rate adapted: {old_rate:.2f} -> {self._current_rate:.2f}/sec")
                
                # Notify callback
                if self._on_rate_change:
                    try:
                        self._on_rate_change(old_rate, self._current_rate)
                    except Exception as e:
                        logger.warning(f"Rate change callback failed: {e}")
    
    def can_proceed(self, tokens: int = 1) -> bool:
        """Check if flow can proceed"""
        return self._rate_limiter.consume(self.config.name, tokens)
    
    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for tokens to become available"""
        return self._rate_limiter.wait_for_tokens(self.config.name, tokens, timeout)
    
    def force_rate(self, rate: float):
        """Force a specific flow rate"""
        with self._lock:
            old_rate = self._current_rate
            self._current_rate = max(self.config.min_rate_per_sec, 
                                   min(self.config.max_rate_per_sec, rate))
            
            # Update rate limiter
            self._rate_limiter.remove_bucket(self.config.name)
            self._rate_limiter.add_bucket(self.config.name, self._current_rate)
            
            logger.info(f"Flow rate forced: {old_rate:.2f} -> {self._current_rate:.2f}/sec")
            
            # Notify callback
            if self._on_rate_change:
                try:
                    self._on_rate_change(old_rate, self._current_rate)
                except Exception as e:
                    logger.warning(f"Rate change callback failed: {e}")
    
    def reset_to_target(self):
        """Reset flow rate to target"""
        self.force_rate(self.config.target_rate_per_sec)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get flow control statistics"""
        with self._lock:
            return {
                "name": self.config.name,
                "current_rate": self._current_rate,
                "target_rate": self.config.target_rate_per_sec,
                "min_rate": self.config.min_rate_per_sec,
                "max_rate": self.config.max_rate_per_sec,
                "measurements_count": len(self._measurements),
                "adaptation_enabled": self.config.enable_adaptive,
                "rate_limiter_stats": self._rate_limiter.get_bucket_stats(self.config.name)
            }
    
    def reset(self):
        """Reset flow controller to initial state"""
        with self._lock:
            self._current_rate = self.config.target_rate_per_sec
            self._measurements.clear()
            self._last_measurement = time.time()
            
            # Reset rate limiter
            self._rate_limiter.remove_bucket(self.config.name)
            self._rate_limiter.add_bucket(self.config.name, self._current_rate)
            
            logger.info(f"Flow controller '{self.config.name}' reset")


class FlowControllerManager:
    """Manages multiple flow controllers"""
    
    def __init__(self):
        self._controllers: Dict[str, FlowController] = {}
        self._lock = threading.RLock()
        
        logger.info("Flow controller manager initialized")
    
    def create_controller(self, config: FlowControlConfig) -> FlowController:
        """Create and register a flow controller"""
        with self._lock:
            controller = FlowController(config)
            self._controllers[config.name] = controller
            logger.info(f"Created flow controller: {config.name}")
            return controller
    
    def get_controller(self, name: str) -> Optional[FlowController]:
        """Get a flow controller by name"""
        return self._controllers.get(name)
    
    def remove_controller(self, name: str) -> bool:
        """Remove a flow controller"""
        with self._lock:
            if name in self._controllers:
                del self._controllers[name]
                logger.info(f"Removed flow controller: {name}")
                return True
            return False
    
    def list_controllers(self) -> list:
        """List all controller names"""
        return list(self._controllers.keys())
    
    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all controllers"""
        return {name: controller.get_stats() 
                for name, controller in self._controllers.items()}
    
    def reset_all(self):
        """Reset all controllers"""
        with self._lock:
            for controller in self._controllers.values():
                controller.reset()
        logger.info("All flow controllers reset")


# Global instance
_global_flow_controller_manager: Optional[FlowControllerManager] = None


def get_global_flow_controller_manager() -> FlowControllerManager:
    """Get the global flow controller manager"""
    global _global_flow_controller_manager
    if _global_flow_controller_manager is None:
        _global_flow_controller_manager = FlowControllerManager()
    return _global_flow_controller_manager


def create_flow_controller(config: FlowControlConfig) -> FlowController:
    """Create a flow controller in the global manager"""
    return get_global_flow_controller_manager().create_controller(config)


def get_flow_controller(name: str) -> Optional[FlowController]:
    """Get a flow controller from the global manager"""
    return get_global_flow_controller_manager().get_controller(name)
