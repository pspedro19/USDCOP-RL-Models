"""
SAGA Coordinator
================
Core orchestration for distributed transactions with compensation.
"""

import asyncio
import uuid
import logging
from typing import List, Callable, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .types import SagaStatus, StepStatus, SagaEvent, StepSpec, SagaTransaction
from .store import SagaStore
from .bus import SagaEventBus

logger = logging.getLogger(__name__)


@dataclass
class StepExecution:
    """Step execution context"""
    name: str
    executor: Callable
    compensation: Optional[Callable] = None
    timeout_sec: int = 30
    retry_count: int = 0
    max_retries: int = 3
    dependencies: List[str] = None
    critical: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class SagaCoordinator:
    """SAGA coordinator for orchestrating distributed transactions"""
    
    def __init__(self, store: SagaStore, bus: SagaEventBus,
                 default_timeout: int = 30, max_retries: int = 3):
        self.store = store
        self.bus = bus
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        
        # Active transactions
        self._active_sagas: Dict[str, SagaTransaction] = {}
        self._step_executions: Dict[str, List[StepExecution]] = {}
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("SAGA coordinator initialized")
    
    async def start(self):
        """Start the coordinator"""
        # Start event bus
        await self.bus.start()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("SAGA coordinator started")
    
    async def stop(self):
        """Stop the coordinator"""
        # Stop cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop event bus
        await self.bus.stop()
        
        logger.info("SAGA coordinator stopped")
    
    async def run_saga(self, name: str, steps: List[StepExecution],
                      correlation_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a SAGA transaction"""
        saga_id = f"{name}-{uuid.uuid4().hex[:8]}"
        
        # Create SAGA transaction
        saga = SagaTransaction(
            name=name,
            correlation_id=correlation_id,
            context=context,
            steps=[StepSpec(
                name=step.name,
                timeout_sec=step.timeout_sec,
                retry_count=step.retry_count,
                max_retries=step.max_retries,
                dependencies=step.dependencies,
                compensation=step.compensation.__name__ if step.compensation else None,
                critical=step.critical,
                metadata=step.metadata or {}
            ) for step in steps]
        )
        
        # Store initial state
        self.store.put_saga(saga)
        self._active_sagas[saga_id] = saga
        self._step_executions[saga_id] = steps
        
        # Publish start event
        await self.bus.publish(SagaEvent(
            saga_id=saga_id,
            name="saga_started",
            correlation_id=correlation_id,
            payload={"name": name, "steps_count": len(steps)}
        ))
        
        logger.info(f"SAGA '{name}' started with ID {saga_id}")
        
        try:
            # Execute steps
            result = await self._execute_saga(saga_id, steps, context)
            
            # Mark as completed
            saga.status = SagaStatus.COMPLETED
            saga.completed_at = datetime.utcnow()
            self.store.put_saga(saga)
            
            # Publish completion event
            await self.bus.publish(SagaEvent(
                saga_id=saga_id,
                name="saga_completed",
                correlation_id=correlation_id,
                payload={"result": result}
            ))
            
            logger.info(f"SAGA '{name}' completed successfully")
            return {"saga_id": saga_id, "status": "completed", "result": result}
            
        except Exception as e:
            # Mark as failed
            saga.status = SagaStatus.FAILED
            self.store.put_saga(saga)
            
            # Publish failure event
            await self.bus.publish(SagaEvent(
                saga_id=saga_id,
                name="saga_failed",
                correlation_id=correlation_id,
                payload={"error": str(e)}
            ))
            
            logger.error(f"SAGA '{name}' failed: {e}")
            
            # Run compensation
            await self._run_compensation(saga_id, steps, context)
            
            raise
        
        finally:
            # Cleanup
            self._cleanup_saga(saga_id)
    
    async def _execute_saga(self, saga_id: str, steps: List[StepExecution], 
                           context: Dict[str, Any]) -> Any:
        """Execute SAGA steps in sequence"""
        completed_steps = []
        result = None
        
        for i, step in enumerate(steps):
            step_name = step.name
            
            # Check dependencies
            if not await self._check_dependencies(saga_id, step, completed_steps):
                raise RuntimeError(f"Step '{step_name}' dependencies not met")
            
            # Execute step
            try:
                step_result = await self._execute_step(saga_id, step, context)
                completed_steps.append(step_name)
                result = step_result
                
                logger.debug(f"Step '{step_name}' completed successfully")
                
            except Exception as e:
                logger.error(f"Step '{step_name}' failed: {e}")
                
                # Mark step as failed
                self.store.append_step(saga_id, step_name, StepStatus.FAILED, {
                    "error": str(e),
                    "retry_count": step.retry_count
                })
                
                # Publish step failure event
                await self.bus.publish(SagaEvent(
                    saga_id=saga_id,
                    name="step_failed",
                    step=step_name,
                    correlation_id=context.get("correlation_id", ""),
                    payload={"error": str(e), "retry_count": step.retry_count}
                ))
                
                raise
        
        return result
    
    async def _execute_step(self, saga_id: str, step: StepExecution, 
                           context: Dict[str, Any]) -> Any:
        """Execute a single step with retry logic"""
        step_name = step.name
        
        # Mark step as in progress
        self.store.append_step(saga_id, step_name, StepStatus.IN_PROGRESS, {})
        
        # Publish step started event
        await self.bus.publish(SagaEvent(
            saga_id=saga_id,
            name="step_started",
            step=step_name,
            correlation_id=context.get("correlation_id", ""),
            payload={"timeout": step.timeout_sec}
        ))
        
        # Execute with timeout and retries
        for attempt in range(step.max_retries + 1):
            try:
                # Execute step
                if asyncio.iscoroutinefunction(step.executor):
                    result = await asyncio.wait_for(
                        step.executor(context),
                        timeout=step.timeout_sec
                    )
                else:
                    # Handle synchronous functions
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, step.executor, context),
                        timeout=step.timeout_sec
                    )
                
                # Mark step as completed
                self.store.append_step(saga_id, step_name, StepStatus.DONE, {
                    "result": str(result),
                    "attempt": attempt + 1
                })
                
                # Publish step completed event
                await self.bus.publish(SagaEvent(
                    saga_id=saga_id,
                    name="step_completed",
                    step=step_name,
                    correlation_id=context.get("correlation_id", ""),
                    payload={"result": str(result), "attempt": attempt + 1}
                ))
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Step '{step_name}' timed out (attempt {attempt + 1})")
                
                if attempt < step.max_retries:
                    # Wait before retry
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Max retries exceeded
                    raise RuntimeError(f"Step '{step_name}' timed out after {step.max_retries} retries")
                    
            except Exception as e:
                logger.warning(f"Step '{step_name}' failed (attempt {attempt + 1}): {e}")
                
                if attempt < step.max_retries:
                    # Wait before retry
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    # Max retries exceeded
                    raise e
    
    async def _check_dependencies(self, saga_id: str, step: StepExecution, 
                                 completed_steps: List[str]) -> bool:
        """Check if step dependencies are met"""
        if not step.dependencies:
            return True
        
        for dependency in step.dependencies:
            if dependency not in completed_steps:
                logger.warning(f"Step '{step.name}' dependency '{dependency}' not met")
                return False
        
        return True
    
    async def _run_compensation(self, saga_id: str, steps: List[StepExecution], 
                               context: Dict[str, Any]):
        """Run compensation for failed SAGA"""
        logger.info(f"Running compensation for SAGA {saga_id}")
        
        # Mark as compensating
        saga = self._active_sagas.get(saga_id)
        if saga:
            saga.status = SagaStatus.COMPENSATING
            self.store.put_saga(saga)
        
        # Publish compensation started event
        await self.bus.publish(SagaEvent(
            saga_id=saga_id,
            name="saga_compensating",
            correlation_id=context.get("correlation_id", ""),
            payload={"steps_count": len(steps)}
        ))
        
        # Run compensations in reverse order
        compensation_results = []
        for step in reversed(steps):
            if step.compensation:
                try:
                    step_name = step.name
                    
                    # Mark compensation as started
                    self.store.append_step(saga_id, step_name, StepStatus.IN_PROGRESS, {
                        "compensation": True
                    })
                    
                    # Execute compensation
                    if asyncio.iscoroutinefunction(step.compensation):
                        result = await step.compensation(context)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, step.compensation, context)
                    
                    # Mark compensation as completed
                    self.store.append_step(saga_id, step_name, StepStatus.COMP_DONE, {
                        "result": str(result),
                        "compensation": True
                    })
                    
                    compensation_results.append({
                        "step": step_name,
                        "status": "completed",
                        "result": result
                    })
                    
                    logger.info(f"Compensation for step '{step_name}' completed")
                    
                except Exception as e:
                    logger.error(f"Compensation for step '{step.name}' failed: {e}")
                    
                    # Mark compensation as failed
                    self.store.append_step(saga_id, step.name, StepStatus.COMP_FAILED, {
                        "error": str(e),
                        "compensation": True
                    })
                    
                    compensation_results.append({
                        "step": step.name,
                        "status": "failed",
                        "error": str(e)
                    })
        
        # Mark as compensated
        if saga:
            saga.status = SagaStatus.COMPENSATED
            self.store.put_saga(saga)
        
        # Publish compensation completed event
        await self.bus.publish(SagaEvent(
            saga_id=saga_id,
            name="saga_compensated",
            correlation_id=context.get("correlation_id", ""),
            payload={"compensation_results": compensation_results}
        ))
        
        logger.info(f"Compensation for SAGA {saga_id} completed")
    
    def _cleanup_saga(self, saga_id: str):
        """Clean up SAGA from active tracking"""
        if saga_id in self._active_sagas:
            del self._active_sagas[saga_id]
        if saga_id in self._step_executions:
            del self._step_executions[saga_id]
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired locks
                expired_count = self.store.cleanup_expired_locks()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired locks")
                
                # Check for stuck SAGAs
                await self._check_stuck_sagas()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    async def _check_stuck_sagas(self):
        """Check for stuck SAGAs and mark them as timed out"""
        current_time = datetime.utcnow()
        timeout_threshold = timedelta(minutes=30)
        
        stuck_sagas = []
        for saga_id, saga in self._active_sagas.items():
            if (saga.status == SagaStatus.IN_PROGRESS and 
                current_time - saga.started_at > timeout_threshold):
                stuck_sagas.append(saga_id)
        
        for saga_id in stuck_sagas:
            logger.warning(f"Marking stuck SAGA {saga_id} as timed out")
            
            saga = self._active_sagas[saga_id]
            saga.status = SagaStatus.TIMED_OUT
            self.store.put_saga(saga)
            
            # Publish timeout event
            await self.bus.publish(SagaEvent(
                saga_id=saga_id,
                name="saga_timed_out",
                correlation_id=saga.correlation_id,
                payload={"reason": "stuck_detection"}
            ))
            
            self._cleanup_saga(saga_id)
    
    def get_saga_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get SAGA status and details"""
        saga = self.store.get_saga(saga_id)
        if not saga:
            return None
        
        return {
            "id": saga.id,
            "name": saga.name,
            "status": saga.status.value,
            "started_at": saga.started_at.isoformat(),
            "updated_at": saga.updated_at.isoformat(),
            "completed_at": saga.completed_at.isoformat() if saga.completed_at else None,
            "retry_count": saga.retry_count,
            "max_retries": saga.max_retries
        }
    
    def get_active_sagas(self) -> List[Dict[str, Any]]:
        """Get all active SAGAs"""
        active_sagas = self.store.get_active_sagas()
        return [
            {
                "id": saga.id,
                "name": saga.name,
                "status": saga.status.value,
                "started_at": saga.started_at.isoformat(),
                "retry_count": saga.retry_count
            }
            for saga in active_sagas
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get SAGA metrics"""
        return self.bus.get_metrics_summary()
