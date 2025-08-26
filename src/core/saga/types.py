"""
SAGA Types and Enums
====================
Type definitions for SAGA orchestration system.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from uuid import UUID, uuid4


class SagaStatus(str, Enum):
    """SAGA transaction status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATING = "COMPENSATING"
    COMPENSATED = "COMPENSATED"
    TIMED_OUT = "TIMED_OUT"
    CANCELLED = "CANCELLED"


class StepStatus(str, Enum):
    """Individual step status"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"
    COMP_DONE = "COMP_DONE"
    COMP_FAILED = "COMP_FAILED"
    TIMED_OUT = "TIMED_OUT"


class SagaEvent(BaseModel):
    """SAGA event for audit trail"""
    saga_id: str
    name: str
    step: Optional[str] = None
    correlation_id: str
    ts: datetime = Field(default_factory=datetime.utcnow)
    payload: Dict[str, Any] = Field(default_factory=dict)
    source: str = "saga_system"
    severity: str = "info"


class StepSpec(BaseModel):
    """Step specification with execution details"""
    name: str
    timeout_sec: int = 30
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = Field(default_factory=list)
    compensation: Optional[str] = None
    critical: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SagaTransaction(BaseModel):
    """Complete SAGA transaction"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    correlation_id: str
    status: SagaStatus = SagaStatus.PENDING
    steps: List[StepSpec] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CompensationAction(BaseModel):
    """Compensation action specification"""
    name: str
    step_name: str
    action: Callable
    timeout_sec: int = 30
    critical: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResourceLock(BaseModel):
    """Resource locking for SAGA coordination"""
    resource_id: str
    saga_id: str
    acquired_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
