"""
SAGA (Saga Orchestration) Package
=================================
Distributed transaction management with compensation patterns.

This package provides:
- SAGA coordinator for orchestrating multi-step transactions
- State persistence with Redis + SQLite fallback
- Event publishing for audit trails
- Compensation strategies for rollback scenarios
- Resource locking and timeout management
"""

from .types import SagaStatus, StepStatus, SagaEvent, StepSpec
from .store import SagaStore
from .bus import SagaEventBus
from .coordinator import SagaCoordinator
from .transaction_manager import TransactionManager

__all__ = [
    'SagaStatus',
    'StepStatus', 
    'SagaEvent',
    'StepSpec',
    'SagaStore',
    'SagaEventBus',
    'SagaCoordinator',
    'TransactionManager'
]
