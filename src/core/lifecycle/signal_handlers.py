"""
Signal Handlers
===============
Signal handlers for graceful shutdown on SIGTERM/SIGINT.
"""
import signal
import asyncio
import logging
from typing import Optional, Callable

from .shutdown_manager import get_shutdown_manager

logger = logging.getLogger(__name__)

class SignalHandler:
    """Handles system signals for graceful shutdown."""
    
    def __init__(self, shutdown_callback: Optional[Callable] = None):
        """Initialize signal handler."""
        self.shutdown_callback = shutdown_callback
        self.original_handlers = {}
        self.registered = False
    
    def register_handlers(self):
        """Register signal handlers for SIGTERM and SIGINT."""
        if self.registered:
            logger.warning("Signal handlers already registered")
            return
        
        try:
            # Store original handlers
            self.original_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, self._signal_handler)
            self.original_handlers[signal.SIGINT] = signal.signal(signal.SIGINT, self._signal_handler)
            
            self.registered = True
            logger.info("Signal handlers registered for SIGTERM and SIGINT")
            
        except Exception as e:
            logger.error(f"Failed to register signal handlers: {e}")
    
    def unregister_handlers(self):
        """Restore original signal handlers."""
        if not self.registered:
            return
        
        try:
            # Restore original handlers
            for sig, handler in self.original_handlers.items():
                if handler is not None:
                    signal.signal(sig, handler)
            
            self.registered = False
            logger.info("Signal handlers unregistered")
            
        except Exception as e:
            logger.error(f"Failed to unregister signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name} ({signum}), initiating graceful shutdown")
        
        # Get shutdown manager
        shutdown_manager = get_shutdown_manager()
        
        # Request shutdown
        shutdown_manager.request_shutdown()
        
        # Call custom shutdown callback if provided
        if self.shutdown_callback:
            try:
                if asyncio.iscoroutinefunction(self.shutdown_callback):
                    # Schedule async callback
                    asyncio.create_task(self.shutdown_callback())
                else:
                    # Call sync callback
                    self.shutdown_callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
        
        # Schedule shutdown execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule shutdown in the event loop
                loop.create_task(self._execute_shutdown())
            else:
                # Run shutdown directly if no event loop
                asyncio.run(self._execute_shutdown())
        except Exception as e:
            logger.error(f"Failed to schedule shutdown: {e}")
    
    async def _execute_shutdown(self):
        """Execute the shutdown sequence."""
        try:
            shutdown_manager = get_shutdown_manager()
            await shutdown_manager.shutdown()
        except Exception as e:
            logger.error(f"Shutdown execution failed: {e}")

def register_shutdown_signals(shutdown_callback: Optional[Callable] = None) -> SignalHandler:
    """
    Register signal handlers for graceful shutdown.
    
    Args:
        shutdown_callback: Optional callback to execute when shutdown is requested
        
    Returns:
        SignalHandler instance
    """
    handler = SignalHandler(shutdown_callback)
    handler.register_handlers()
    return handler

def unregister_shutdown_signals(handler: SignalHandler):
    """Unregister signal handlers."""
    if handler:
        handler.unregister_handlers()

# Global signal handler instance
_global_signal_handler: Optional[SignalHandler] = None

def setup_signal_handlers(shutdown_callback: Optional[Callable] = None) -> SignalHandler:
    """Setup global signal handlers."""
    global _global_signal_handler
    
    if _global_signal_handler is None:
        _global_signal_handler = register_shutdown_signals(shutdown_callback)
    
    return _global_signal_handler

def cleanup_signal_handlers():
    """Cleanup global signal handlers."""
    global _global_signal_handler
    
    if _global_signal_handler:
        unregister_shutdown_signals(_global_signal_handler)
        _global_signal_handler = None
