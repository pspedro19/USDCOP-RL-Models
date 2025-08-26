"""
Shutdown Manager
================
Coordinates phased shutdown across all services.
"""
import asyncio
import logging
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ShutdownPhase(Enum):
    """Shutdown phases in order of execution."""
    STOP_ACCEPTING = 1      # Stop accepting new work
    DRAIN_INFLIGHT = 2      # Wait for in-flight operations to complete
    PERSIST_STATE = 3       # Save state to persistent storage
    CLOSE_RESOURCES = 4     # Close connections and resources
    FINAL_CLEANUP = 5       # Final cleanup and exit

@dataclass
class ShutdownHandler:
    """Handler for a specific shutdown phase."""
    phase: ShutdownPhase
    handler: Callable
    timeout: float = 30.0
    critical: bool = False
    description: str = ""

class ShutdownManager:
    """Manages graceful shutdown across all services."""
    
    def __init__(self, timeout: float = 30.0):
        """Initialize shutdown manager."""
        self.timeout = timeout
        self.handlers: Dict[ShutdownPhase, List[ShutdownHandler]] = {}
        self.shutdown_requested = False
        self.shutdown_start_time: Optional[float] = None
        self.current_phase: Optional[ShutdownPhase] = None
        
        # Initialize handler lists for each phase
        for phase in ShutdownPhase:
            self.handlers[phase] = []
    
    def register_handler(self, phase: ShutdownPhase, handler: Callable, 
                        timeout: float = 30.0, critical: bool = False, 
                        description: str = ""):
        """
        Register a handler for a specific shutdown phase.
        
        Args:
            phase: Shutdown phase when this handler should run
            handler: Function to execute during shutdown
            timeout: Maximum time to wait for handler completion
            critical: Whether this handler is critical for shutdown
            description: Human-readable description of the handler
        """
        shutdown_handler = ShutdownHandler(
            phase=phase,
            handler=handler,
            timeout=timeout,
            critical=critical,
            description=description
        )
        
        self.handlers[phase].append(shutdown_handler)
        logger.info(f"Registered shutdown handler for phase {phase.name}: {description}")
    
    def request_shutdown(self):
        """Request shutdown to begin."""
        if not self.shutdown_requested:
            self.shutdown_requested = True
            self.shutdown_start_time = time.time()
            logger.info("Shutdown requested")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self.shutdown_requested
    
    async def shutdown(self) -> bool:
        """
        Execute shutdown sequence.
        
        Returns:
            True if shutdown completed successfully
        """
        if self.shutdown_requested:
            logger.warning("Shutdown already in progress")
            return False
        
        self.request_shutdown()
        logger.info("Starting graceful shutdown sequence")
        
        try:
            # Execute each phase in order
            for phase in ShutdownPhase:
                await self._execute_phase(phase)
            
            logger.info("Shutdown sequence completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Shutdown sequence failed: {e}", exc_info=True)
            return False
    
    async def _execute_phase(self, phase: ShutdownPhase):
        """Execute all handlers for a specific phase."""
        handlers = self.handlers.get(phase, [])
        if not handlers:
            logger.debug(f"No handlers registered for phase {phase.name}")
            return
        
        logger.info(f"Executing phase {phase.name} with {len(handlers)} handlers")
        self.current_phase = phase
        
        # Execute handlers concurrently with timeouts
        tasks = []
        for handler_info in handlers:
            task = asyncio.create_task(
                self._execute_handler(handler_info),
                name=f"shutdown_{phase.name}_{handler_info.description}"
            )
            tasks.append(task)
        
        if tasks:
            # Wait for all handlers to complete or timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            for i, result in enumerate(results):
                handler_info = handlers[i]
                if isinstance(result, Exception):
                    if handler_info.critical:
                        logger.error(f"Critical handler failed in phase {phase.name}: {result}")
                        raise result
                    else:
                        logger.warning(f"Non-critical handler failed in phase {phase.name}: {result}")
                else:
                    logger.debug(f"Handler completed successfully: {handler_info.description}")
    
    async def _execute_handler(self, handler_info: ShutdownHandler) -> Any:
        """Execute a single shutdown handler with timeout."""
        try:
            logger.debug(f"Executing handler: {handler_info.description}")
            
            # Execute handler with timeout
            if asyncio.iscoroutinefunction(handler_info.handler):
                result = await asyncio.wait_for(
                    handler_info.handler(),
                    timeout=handler_info.timeout
                )
            else:
                # For synchronous functions, run in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, handler_info.handler),
                    timeout=handler_info.timeout
                )
            
            logger.debug(f"Handler completed: {handler_info.description}")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Handler timed out after {handler_info.timeout}s: {handler_info.description}")
            raise
        except Exception as e:
            logger.error(f"Handler failed: {handler_info.description}, error: {e}")
            raise
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        status = {
            'shutdown_requested': self.shutdown_requested,
            'current_phase': self.current_phase.name if self.current_phase else None,
            'handlers_by_phase': {}
        }
        
        for phase in ShutdownPhase:
            handlers = self.handlers.get(phase, [])
            status['handlers_by_phase'][phase.name] = {
                'count': len(handlers),
                'descriptions': [h.description for h in handlers]
            }
        
        if self.shutdown_start_time:
            status['shutdown_duration'] = time.time() - self.shutdown_start_time
        
        return status
    
    def add_health_check_handler(self, health_check_func: Callable):
        """Add a handler to mark health checks as unhealthy during shutdown."""
        def mark_unhealthy():
            try:
                if hasattr(health_check_func, 'set_unhealthy'):
                    health_check_func.set_unhealthy()
                logger.info("Health checks marked as unhealthy")
            except Exception as e:
                logger.warning(f"Failed to mark health checks as unhealthy: {e}")
        
        self.register_handler(
            ShutdownPhase.STOP_ACCEPTING,
            mark_unhealthy,
            timeout=5.0,
            critical=False,
            description="Mark health checks as unhealthy"
        )
    
    def add_connection_close_handler(self, close_func: Callable):
        """Add a handler to close connections."""
        self.register_handler(
            ShutdownPhase.CLOSE_RESOURCES,
            close_func,
            timeout=10.0,
            critical=False,
            description="Close connections"
        )
    
    def add_state_persistence_handler(self, persist_func: Callable):
        """Add a handler to persist state."""
        self.register_handler(
            ShutdownPhase.PERSIST_STATE,
            persist_func,
            timeout=15.0,
            critical=False,
            description="Persist state"
        )

# Global shutdown manager instance
_global_shutdown_manager: Optional[ShutdownManager] = None

def get_shutdown_manager() -> ShutdownManager:
    """Get the global shutdown manager instance."""
    global _global_shutdown_manager
    if _global_shutdown_manager is None:
        _global_shutdown_manager = ShutdownManager()
    return _global_shutdown_manager

async def request_shutdown():
    """Request shutdown using the global manager."""
    manager = get_shutdown_manager()
    await manager.shutdown()

def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    manager = get_shutdown_manager()
    return manager.is_shutdown_requested()
