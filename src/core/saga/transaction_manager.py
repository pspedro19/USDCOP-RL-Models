"""
SAGA Transaction Manager
========================
High-level API for defining and executing SAGA transactions.
"""

import asyncio
import logging
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass

from .coordinator import SagaCoordinator, StepExecution
from .types import SagaStatus

logger = logging.getLogger(__name__)


@dataclass
class TransactionDefinition:
    """Transaction definition with steps and metadata"""
    name: str
    steps: List[StepExecution]
    timeout_sec: int = 300
    max_retries: int = 3
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TransactionManager:
    """High-level transaction manager for common SAGA patterns"""
    
    def __init__(self, coordinator: SagaCoordinator):
        self.coordinator = coordinator
        self._transaction_definitions: Dict[str, TransactionDefinition] = {}
        
        # Register common transaction patterns
        self._register_common_patterns()
        
        logger.info("Transaction manager initialized")
    
    def _register_common_patterns(self):
        """Register common transaction patterns"""
        # Trade execution pattern
        self.register_transaction(
            TransactionDefinition(
                name="trade_execution",
                steps=[
                    StepExecution(
                        name="validate",
                        executor=self._validate_trade,
                        timeout_sec=10,
                        critical=True
                    ),
                    StepExecution(
                        name="reserve",
                        executor=self._reserve_funds,
                        compensation=self._release_funds,
                        timeout_sec=10,
                        critical=True
                    ),
                    StepExecution(
                        name="execute",
                        executor=self._execute_trade,
                        compensation=self._cancel_trade,
                        timeout_sec=15,
                        critical=True
                    ),
                    StepExecution(
                        name="record",
                        executor=self._record_trade,
                        compensation=self._delete_trade_record,
                        timeout_sec=10,
                        critical=False
                    ),
                    StepExecution(
                        name="notify",
                        executor=self._notify_trade,
                        compensation=self._retract_notification,
                        timeout_sec=10,
                        critical=False
                    )
                ],
                timeout_sec=60,
                metadata={"type": "trading", "critical": True}
            )
        )
        
        # Data pipeline pattern
        self.register_transaction(
            TransactionDefinition(
                name="data_pipeline",
                steps=[
                    StepExecution(
                        name="fetch",
                        executor=self._fetch_data,
                        timeout_sec=120,
                        critical=True
                    ),
                    StepExecution(
                        name="validate",
                        executor=self._validate_data,
                        timeout_sec=15,
                        critical=True
                    ),
                    StepExecution(
                        name="enrich",
                        executor=self._enrich_data,
                        timeout_sec=60,
                        critical=False
                    ),
                    StepExecution(
                        name="store",
                        executor=self._store_data,
                        compensation=self._delete_data,
                        timeout_sec=60,
                        critical=True
                    ),
                    StepExecution(
                        name="publish",
                        executor=self._publish_data,
                        timeout_sec=10,
                        critical=False
                    )
                ],
                timeout_sec=300,
                metadata={"type": "data_processing", "critical": False}
            )
        )
        
        # Model deployment pattern
        self.register_transaction(
            TransactionDefinition(
                name="model_deployment",
                steps=[
                    StepExecution(
                        name="validate",
                        executor=self._validate_model,
                        timeout_sec=30,
                        critical=True
                    ),
                    StepExecution(
                        name="backup",
                        executor=self._backup_current_model,
                        timeout_sec=60,
                        critical=True
                    ),
                    StepExecution(
                        name="deploy",
                        executor=self._deploy_model,
                        compensation=self._rollback_model,
                        timeout_sec=120,
                        critical=True
                    ),
                    StepExecution(
                        name="test",
                        executor=self._test_model,
                        timeout_sec=60,
                        critical=True
                    ),
                    StepExecution(
                        name="activate",
                        executor=self._activate_model,
                        timeout_sec=30,
                        critical=True
                    )
                ],
                timeout_sec=300,
                metadata={"type": "ml_ops", "critical": True}
            )
        )
    
    def register_transaction(self, definition: TransactionDefinition):
        """Register a transaction definition"""
        self._transaction_definitions[definition.name] = definition
        logger.info(f"Registered transaction pattern: {definition.name}")
    
    def get_transaction_definition(self, name: str) -> Optional[TransactionDefinition]:
        """Get a transaction definition by name"""
        return self._transaction_definitions.get(name)
    
    def list_transactions(self) -> List[str]:
        """List all registered transaction names"""
        return list(self._transaction_definitions.keys())
    
    async def execute_trade(self, trade_data: Dict[str, Any], 
                           correlation_id: str) -> Dict[str, Any]:
        """Execute a trade transaction"""
        context = {
            "trade_data": trade_data,
            "correlation_id": correlation_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        definition = self.get_transaction_definition("trade_execution")
        if not definition:
            raise ValueError("Trade execution pattern not registered")
        
        return await self.coordinator.run_saga(
            name="trade_execution",
            steps=definition.steps,
            correlation_id=correlation_id,
            context=context
        )
    
    async def execute_data_pipeline(self, pipeline_config: Dict[str, Any],
                                   correlation_id: str) -> Dict[str, Any]:
        """Execute a data pipeline transaction"""
        context = {
            "pipeline_config": pipeline_config,
            "correlation_id": correlation_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        definition = self.get_transaction_definition("data_pipeline")
        if not definition:
            raise ValueError("Data pipeline pattern not registered")
        
        return await self.coordinator.run_saga(
            name="data_pipeline",
            steps=definition.steps,
            correlation_id=correlation_id,
            context=context
        )
    
    async def execute_model_deployment(self, model_config: Dict[str, Any],
                                      correlation_id: str) -> Dict[str, Any]:
        """Execute a model deployment transaction"""
        context = {
            "model_config": model_config,
            "correlation_id": correlation_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        definition = self.get_transaction_definition("model_deployment")
        if not definition:
            raise ValueError("Model deployment pattern not registered")
        
        return await self.coordinator.run_saga(
            name="model_deployment",
            steps=definition.steps,
            correlation_id=correlation_id,
            context=context
        )
    
    # Trade execution step implementations
    async def _validate_trade(self, context: Dict[str, Any]) -> bool:
        """Validate trade parameters"""
        trade_data = context["trade_data"]
        
        # Basic validation
        required_fields = ["symbol", "volume", "type", "price"]
        for field in required_fields:
            if field not in trade_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Risk validation
        if trade_data.get("volume", 0) <= 0:
            raise ValueError("Invalid volume")
        
        logger.info(f"Trade validation passed for {trade_data['symbol']}")
        return True
    
    async def _reserve_funds(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reserve funds for trade"""
        trade_data = context["trade_data"]
        
        # Simulate fund reservation
        reservation_id = f"res_{asyncio.get_event_loop().time():.0f}"
        
        logger.info(f"Funds reserved for trade: {reservation_id}")
        return {"reservation_id": reservation_id, "amount": trade_data.get("volume", 0)}
    
    async def _execute_trade(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the trade"""
        trade_data = context["trade_data"]
        
        # Simulate trade execution
        ticket = int(asyncio.get_event_loop().time() * 1000)
        
        logger.info(f"Trade executed with ticket: {ticket}")
        return {"ticket": ticket, "execution_time": asyncio.get_event_loop().time()}
    
    async def _record_trade(self, context: Dict[str, Any]) -> bool:
        """Record trade in database"""
        trade_data = context["trade_data"]
        
        # Simulate database recording
        logger.info(f"Trade recorded in database: {trade_data['symbol']}")
        return True
    
    async def _notify_trade(self, context: Dict[str, Any]) -> bool:
        """Send trade notification"""
        trade_data = context["trade_data"]
        
        # Simulate notification
        logger.info(f"Trade notification sent for: {trade_data['symbol']}")
        return True
    
    # Trade compensation implementations
    async def _release_funds(self, context: Dict[str, Any]) -> bool:
        """Release reserved funds"""
        logger.info("Funds released from reservation")
        return True
    
    async def _cancel_trade(self, context: Dict[str, Any]) -> bool:
        """Cancel executed trade"""
        logger.info("Trade cancelled")
        return True
    
    async def _delete_trade_record(self, context: Dict[str, Any]) -> bool:
        """Delete trade record"""
        logger.info("Trade record deleted")
        return True
    
    async def _retract_notification(self, context: Dict[str, Any]) -> bool:
        """Retract trade notification"""
        logger.info("Trade notification retracted")
        return True
    
    # Data pipeline step implementations
    async def _fetch_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from source"""
        pipeline_config = context["pipeline_config"]
        
        # Simulate data fetching
        logger.info(f"Data fetched for pipeline: {pipeline_config.get('name', 'unknown')}")
        return {"data_size": 1000, "records": 1000}
    
    async def _validate_data(self, context: Dict[str, Any]) -> bool:
        """Validate fetched data"""
        logger.info("Data validation completed")
        return True
    
    async def _enrich_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich data with features"""
        logger.info("Data enrichment completed")
        return {"features_added": 10, "enriched_records": 1000}
    
    async def _store_data(self, context: Dict[str, Any]) -> bool:
        """Store processed data"""
        logger.info("Data stored successfully")
        return True
    
    async def _publish_data(self, context: Dict[str, Any]) -> bool:
        """Publish data availability"""
        logger.info("Data availability published")
        return True
    
    # Data pipeline compensation
    async def _delete_data(self, context: Dict[str, Any]) -> bool:
        """Delete stored data"""
        logger.info("Stored data deleted")
        return True
    
    # Model deployment step implementations
    async def _validate_model(self, context: Dict[str, Any]) -> bool:
        """Validate model configuration"""
        model_config = context["model_config"]
        
        logger.info(f"Model validation completed: {model_config.get('name', 'unknown')}")
        return True
    
    async def _backup_current_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Backup current model"""
        backup_id = f"backup_{asyncio.get_event_loop().time():.0f}"
        
        logger.info(f"Current model backed up: {backup_id}")
        return {"backup_id": backup_id}
    
    async def _deploy_model(self, context: Dict[str, Any]) -> bool:
        """Deploy new model"""
        logger.info("Model deployment completed")
        return True
    
    async def _test_model(self, context: Dict[str, Any]) -> bool:
        """Test deployed model"""
        logger.info("Model testing completed")
        return True
    
    async def _activate_model(self, context: Dict[str, Any]) -> bool:
        """Activate deployed model"""
        logger.info("Model activation completed")
        return True
    
    # Model deployment compensation
    async def _rollback_model(self, context: Dict[str, Any]) -> bool:
        """Rollback to previous model"""
        logger.info("Model rollback completed")
        return True
    
    def get_transaction_status(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction status"""
        return self.coordinator.get_saga_status(saga_id)
    
    def get_active_transactions(self) -> List[Dict[str, Any]]:
        """Get all active transactions"""
        return self.coordinator.get_active_sagas()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get transaction metrics"""
        return self.coordinator.get_metrics()
