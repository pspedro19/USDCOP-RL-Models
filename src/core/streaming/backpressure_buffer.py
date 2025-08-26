"""
Intelligent Backpressure Buffer
==============================
Smart buffer management with adaptive watermarks and priority queuing.
"""

import time
import threading
import queue
import logging
from typing import Callable, Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DropStrategy(str, Enum):
    """Message dropping strategies"""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    DROP_RANDOM = "drop_random"
    SAMPLE = "sample"
    NONE = "none"


class Priority(str, Enum):
    """Message priority levels"""
    CRITICAL = "critical"      # Never drop (e.g., trades)
    HIGH = "high"              # Drop only under extreme pressure
    MEDIUM = "medium"          # Drop when approaching limits
    LOW = "low"                # Drop first under pressure
    BULK = "bulk"              # Batch processing, can be dropped


@dataclass
class BackpressureConfig:
    """Configuration for backpressure buffer"""
    name: str
    max_size: int = 1000
    high_watermark: float = 0.8  # Percentage of max_size
    low_watermark: float = 0.2   # Percentage of max_size
    drop_strategy: DropStrategy = DropStrategy.DROP_OLDEST
    priority: Priority = Priority.MEDIUM
    adaptive: bool = True
    metrics_enabled: bool = True
    ttl_seconds: Optional[int] = None  # Time-to-live for messages


class BackpressureQueue:
    """Intelligent backpressure queue with adaptive watermarks"""
    
    def __init__(self, config: BackpressureConfig,
                 on_pause: Optional[Callable] = None,
                 on_resume: Optional[Callable] = None,
                 metrics_callback: Optional[Callable] = None):
        self.config = config
        self.on_pause = on_pause
        self.on_resume = on_resume
        self.metrics_callback = metrics_callback
        
        # Queue management
        self._queue = queue.Queue(maxsize=config.max_size)
        self._paused = False
        self._lock = threading.RLock()
        
        # Adaptive watermarks
        self._current_high_watermark = int(config.max_size * config.high_watermark)
        self._current_low_watermark = int(config.max_size * config.low_watermark)
        self._load_history = []
        self._pressure_history = []
        
        # Statistics
        self._total_put = 0
        self._total_get = 0
        self._total_dropped = 0
        self._pause_count = 0
        self._resume_count = 0
        self._last_pause_time = 0
        self._last_resume_time = 0
        
        # Priority tracking
        self._priority_counts = {p.value: 0 for p in Priority}
        
        logger.info(f"Backpressure queue '{config.name}' initialized with size {config.max_size}")
    
    @property
    def size(self) -> int:
        """Current queue size"""
        return self._queue.qsize()
    
    @property
    def is_paused(self) -> bool:
        """Check if producer is paused"""
        return self._paused
    
    @property
    def usage_ratio(self) -> float:
        """Current usage ratio (0.0 - 1.0)"""
        return self.size / self.config.max_size
    
    @property
    def is_under_pressure(self) -> bool:
        """Check if queue is under pressure"""
        return self.size >= self._current_high_watermark
    
    def _update_adaptive_watermarks(self):
        """Update watermarks based on load patterns"""
        if not self.config.adaptive:
            return
        
        with self._lock:
            # Calculate load factor
            current_load = self._calculate_load_factor()
            
            # Calculate pressure factor
            pressure_factor = self._calculate_pressure_factor()
            
            # Adjust watermarks
            base_high = int(self.config.max_size * self.config.high_watermark)
            base_low = int(self.config.max_size * self.config.low_watermark)
            
            # High watermark: increase under high load, decrease under low load
            load_adjustment = 1.0 + (current_load - 0.5) * 0.3
            pressure_adjustment = 1.0 + pressure_factor * 0.2
            
            new_high = int(base_high * load_adjustment * pressure_adjustment)
            new_high = max(int(self.config.max_size * 0.6), 
                          min(int(self.config.max_size * 0.95), new_high))
            
            # Low watermark: adjust proportionally
            ratio = base_low / base_high
            new_low = int(new_high * ratio)
            new_low = max(int(self.config.max_size * 0.1), 
                         min(int(self.config.max_size * 0.4), new_low))
            
            if (new_high != self._current_high_watermark or 
                new_low != self._current_low_watermark):
                logger.debug(f"Queue '{self.config.name}' watermarks adjusted: "
                           f"high {self._current_high_watermark}->{new_high}, "
                           f"low {self._current_low_watermark}->{new_low}")
                
                self._current_high_watermark = new_high
                self._current_low_watermark = new_low
    
    def _calculate_load_factor(self) -> float:
        """Calculate current load factor"""
        if not self._load_history:
            return 0.5
        
        # Use recent load history (last 20 measurements)
        recent_loads = self._load_history[-20:]
        return sum(recent_loads) / len(recent_loads)
    
    def _calculate_pressure_factor(self) -> float:
        """Calculate pressure factor based on recent operations"""
        if not self._pressure_history:
            return 0.0
        
        # Use pressure in last 10 seconds
        cutoff_time = time.time() - 10
        recent_pressure = [p for p in self._pressure_history if p > cutoff_time]
        
        if not recent_pressure:
            return 0.0
        
        return len(recent_pressure) / 10.0  # Pressure events per second
    
    def _should_drop_message(self, item: Any, priority: Priority) -> bool:
        """Determine if message should be dropped"""
        # Never drop critical messages
        if priority == Priority.CRITICAL:
            return False
        
        # Check if we're under pressure
        if self.size < self._current_high_watermark:
            return False
        
        # Apply priority-based dropping
        if priority == Priority.HIGH and self.size < int(self.config.max_size * 0.9):
            return False
        
        if priority == Priority.MEDIUM and self.size < int(self.config.max_size * 0.85):
            return False
        
        # Always drop low priority when under pressure
        if priority in [Priority.LOW, Priority.BULK]:
            return True
        
        # Apply drop strategy
        if self.config.drop_strategy == DropStrategy.NONE:
            return False
        
        return True
    
    def _apply_drop_strategy(self, item: Any, priority: Priority) -> bool:
        """Apply the configured drop strategy"""
        if not self._should_drop_message(item, priority):
            return False
        
        try:
            if self.config.drop_strategy == DropStrategy.DROP_OLDEST:
                # Remove oldest message (lowest priority first)
                self._drop_oldest_message()
                return True
            
            elif self.config.drop_strategy == DropStrategy.DROP_NEWEST:
                # Don't add new message
                return True
            
            elif self.config.drop_strategy == DropStrategy.DROP_RANDOM:
                # Randomly decide to drop
                import random
                if random.random() < 0.5:
                    return True
            
            elif self.config.drop_strategy == DropStrategy.SAMPLE:
                # Sample based on priority
                sample_rate = {
                    Priority.HIGH: 0.1,
                    Priority.MEDIUM: 0.3,
                    Priority.LOW: 0.7,
                    Priority.BULK: 0.8
                }.get(priority, 0.5)
                
                import random
                if random.random() < sample_rate:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Drop strategy application failed: {e}")
            return False
    
    def _drop_oldest_message(self):
        """Drop the oldest message from the queue"""
        try:
            # This is a simplified approach - in practice, you might want
            # a more sophisticated priority queue implementation
            if not self._queue.empty():
                self._queue.get_nowait()
                self._total_dropped += 1
        except queue.Empty:
            pass
    
    def _check_pause_resume(self):
        """Check if should pause or resume producer"""
        with self._lock:
            if not self._paused and self.size >= self._current_high_watermark:
                # Pause producer
                self._paused = True
                self._pause_count += 1
                self._last_pause_time = time.time()
                
                logger.info(f"Queue '{self.config.name}' producer paused at size {self.size}")
                
                if self.on_pause:
                    try:
                        self.on_pause()
                    except Exception as e:
                        logger.warning(f"Pause callback failed: {e}")
            
            elif self._paused and self.size <= self._current_low_watermark:
                # Resume producer
                self._paused = False
                self._resume_count += 1
                self._last_resume_time = time.time()
                
                logger.info(f"Queue '{self.config.name}' producer resumed at size {self.size}")
                
                if self.on_resume:
                    try:
                        self.on_resume()
                    except Exception as e:
                        logger.warning(f"Resume callback failed: {e}")
    
    def put(self, item: Any, priority: Priority = Priority.MEDIUM, 
            block: bool = False, timeout: Optional[float] = None) -> bool:
        """Put item in queue with priority and backpressure handling"""
        start_time = time.time()
        
        with self._lock:
            # Update adaptive watermarks
            self._update_adaptive_watermarks()
            
            # Check if should drop message
            if self._apply_drop_strategy(item, priority):
                self._total_dropped += 1
                self._update_metrics("message_dropped", {"priority": priority.value, "reason": "backpressure"})
                return False
            
            # Try to put item
            try:
                # Add priority metadata to item
                item_with_priority = {
                    "item": item,
                    "priority": priority.value,
                    "timestamp": start_time,
                    "ttl": self.config.ttl_seconds
                }
                
                self._queue.put(item_with_priority, block=block, timeout=timeout)
                
                # Update statistics
                self._total_put += 1
                self._priority_counts[priority.value] += 1
                
                # Record load
                duration = time.time() - start_time
                self._load_history.append(duration)
                if len(self._load_history) > 100:
                    self._load_history.pop(0)
                
                # Check pause/resume
                self._check_pause_resume()
                
                # Update metrics
                self._update_metrics("message_put", {
                    "priority": priority.value,
                    "queue_size": self.size,
                    "duration": duration
                })
                
                return True
                
            except queue.Full:
                # Queue is full
                self._total_dropped += 1
                self._update_metrics("message_dropped", {"priority": priority.value, "reason": "queue_full"})
                return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """Get item from queue"""
        start_time = time.time()
        
        try:
            item_with_priority = self._queue.get(block=block, timeout=timeout)
            
            # Check TTL
            if (self.config.ttl_seconds and 
                time.time() - item_with_priority["timestamp"] > self.config.ttl_seconds):
                # Message expired
                self._total_dropped += 1
                self._update_metrics("message_dropped", {"reason": "ttl_expired"})
                return self.get(block, timeout)  # Try next message
            
            # Update statistics
            self._total_get += 1
            priority = item_with_priority["priority"]
            self._priority_counts[priority] = max(0, self._priority_counts[priority] - 1)
            
            # Record pressure
            self._pressure_history.append(time.time())
            if len(self._pressure_history) > 100:
                self._pressure_history.pop(0)
            
            # Check pause/resume
            self._check_pause_resume()
            
            # Update metrics
            duration = time.time() - start_time
            self._update_metrics("message_get", {
                "priority": priority,
                "queue_size": self.size,
                "duration": duration
            })
            
            return item_with_priority["item"]
            
        except queue.Empty:
            return None
    
    def _update_metrics(self, operation: str, labels: Dict[str, Any]):
        """Update metrics via callback"""
        if self.metrics_callback:
            try:
                self.metrics_callback(f"backpressure_{operation}", {
                    "queue": self.config.name,
                    **labels
                })
            except Exception as e:
                logger.warning(f"Metrics callback failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                "name": self.config.name,
                "size": self.size,
                "max_size": self.config.max_size,
                "usage_ratio": self.usage_ratio,
                "paused": self._paused,
                "high_watermark": self._current_high_watermark,
                "low_watermark": self._current_low_watermark,
                "total_put": self._total_put,
                "total_get": self._total_get,
                "total_dropped": self._total_dropped,
                "pause_count": self._pause_count,
                "resume_count": self._resume_count,
                "priority_counts": self._priority_counts.copy(),
                "load_history_length": len(self._load_history),
                "pressure_history_length": len(self._pressure_history)
            }
    
    def clear(self):
        """Clear all items from queue"""
        with self._lock:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reset statistics
            self._total_put = 0
            self._total_get = 0
            self._total_dropped = 0
            self._priority_counts = {p.value: 0 for p in Priority}
            
            logger.info(f"Queue '{self.config.name}' cleared")
    
    def reset_watermarks(self):
        """Reset watermarks to default values"""
        with self._lock:
            self._current_high_watermark = int(self.config.max_size * self.config.high_watermark)
            self._current_low_watermark = int(self.config.max_size * self.config.low_watermark)
            logger.info(f"Queue '{self.config.name}' watermarks reset to defaults")
