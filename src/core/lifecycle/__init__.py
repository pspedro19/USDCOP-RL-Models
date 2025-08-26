"""
Lifecycle Management Package
===========================
Graceful shutdown and lifecycle management for services.
"""
from .shutdown_manager import ShutdownManager, ShutdownPhase, get_shutdown_manager
from .signal_handlers import SignalHandler
from .state_persistence import StatePersistence

__all__ = [
    'ShutdownManager', 'ShutdownPhase', 'get_shutdown_manager',
    'SignalHandler',
    'StatePersistence'
]
