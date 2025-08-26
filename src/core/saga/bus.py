"""
SAGA Event Bus
==============
Event publishing for audit trails with Kafka + file fallback
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from aiokafka import AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    AIOKafkaProducer = None

from .types import SagaEvent

logger = logging.getLogger(__name__)


class SagaEventBus:
    """Event bus for SAGA audit trails"""
    
    def __init__(self, kafka_bootstrap: Optional[str] = None, 
                 metrics_file: str = "data/reports/usdcop/mt5_health.prom",
                 audit_log_dir: str = "data/reports/usdcop"):
        self.kafka_producer = None
        self.metrics_file = Path(metrics_file)
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kafka if available
        if KAFKA_AVAILABLE and kafka_bootstrap:
            try:
                self.kafka_producer = AIOKafkaProducer(
                    bootstrap_servers=kafka_bootstrap,
                    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
                )
                logger.info("Kafka producer initialized for SAGA events")
            except Exception as e:
                logger.warning(f"Kafka initialization failed: {e}, falling back to file logging")
                self.kafka_producer = None
        
        # Initialize metrics file
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self._init_metrics_file()
        
        logger.info(f"SAGA event bus initialized with {'Kafka + file' if self.kafka_producer else 'file only'}")
    
    def _init_metrics_file(self):
        """Initialize Prometheus metrics file"""
        try:
            if not self.metrics_file.exists():
                with open(self.metrics_file, 'w') as f:
                    f.write("# SAGA Metrics\n")
                    f.write("# TYPE saga_transactions_total counter\n")
                    f.write("# TYPE saga_steps_total counter\n")
                    f.write("# TYPE saga_compensations_total counter\n")
                    f.write("# TYPE saga_duration_seconds histogram\n")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics file: {e}")
    
    async def publish(self, event: SagaEvent) -> bool:
        """Publish SAGA event to all available channels"""
        success = True
        
        # Publish to Kafka if available
        if self.kafka_producer:
            try:
                await self.kafka_producer.send_and_wait(
                    topic="saga.audit",
                    value=event.dict()
                )
                logger.debug(f"Published event to Kafka: {event.name}")
            except Exception as e:
                logger.warning(f"Failed to publish to Kafka: {e}")
                success = False
        
        # Always log to file for reliability
        try:
            await self._log_to_file(event)
        except Exception as e:
            logger.error(f"Failed to log event to file: {e}")
            success = False
        
        # Update Prometheus metrics
        try:
            await self._update_metrics(event)
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")
            # Don't fail the whole operation for metrics
        
        return success
    
    async def _log_to_file(self, event: SagaEvent) -> None:
        """Log event to audit log file"""
        audit_file = self.audit_log_dir / "saga_audit.log"
        
        log_entry = {
            "timestamp": event.ts.isoformat(),
            "saga_id": event.saga_id,
            "event": event.name,
            "step": event.step,
            "correlation_id": event.correlation_id,
            "source": event.source,
            "severity": event.severity,
            "payload": event.payload
        }
        
        with open(audit_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    async def _update_metrics(self, event: SagaEvent) -> None:
        """Update Prometheus metrics file"""
        try:
            # Read existing metrics
            metrics = {}
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                metric_name = parts[0]
                                try:
                                    value = float(parts[1])
                                    metrics[metric_name] = value
                                except ValueError:
                                    continue
            
            # Update metrics based on event
            if event.name == "saga_started":
                metrics["saga_transactions_total"] = metrics.get("saga_transactions_total", 0) + 1
                metrics["saga_inflight_total"] = metrics.get("saga_inflight_total", 0) + 1
            
            elif event.name == "saga_completed":
                metrics["saga_inflight_total"] = max(0, metrics.get("saga_inflight_total", 1) - 1)
                metrics["saga_success_total"] = metrics.get("saga_success_total", 0) + 1
            
            elif event.name == "saga_failed":
                metrics["saga_inflight_total"] = max(0, metrics.get("saga_inflight_total", 1) - 1)
                metrics["saga_failed_total"] = metrics.get("saga_failed_total", 0) + 1
            
            elif event.name == "saga_compensated":
                metrics["saga_compensated_total"] = metrics.get("saga_compensated_total", 0) + 1
            
            elif event.name == "step_started":
                metrics["saga_steps_total"] = metrics.get("saga_steps_total", 0) + 1
            
            elif event.name == "step_completed":
                metrics["saga_steps_success_total"] = metrics.get("saga_steps_success_total", 0) + 1
            
            elif event.name == "step_failed":
                metrics["saga_steps_failed_total"] = metrics.get("saga_steps_failed_total", 0) + 1
            
            # Write updated metrics
            with open(self.metrics_file, 'w') as f:
                f.write("# SAGA Metrics\n")
                f.write("# TYPE saga_transactions_total counter\n")
                f.write("# TYPE saga_inflight_total gauge\n")
                f.write("# TYPE saga_success_total counter\n")
                f.write("# TYPE saga_failed_total counter\n")
                f.write("# TYPE saga_compensated_total counter\n")
                f.write("# TYPE saga_steps_total counter\n")
                f.write("# TYPE saga_steps_success_total counter\n")
                f.write("# TYPE saga_steps_failed_total counter\n")
                
                for metric_name, value in metrics.items():
                    if metric_name.startswith("saga_"):
                        f.write(f"{metric_name} {value}\n")
        
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")
    
    async def start(self):
        """Start the event bus"""
        if self.kafka_producer:
            try:
                await self.kafka_producer.start()
                logger.info("SAGA event bus started")
            except Exception as e:
                logger.error(f"Failed to start Kafka producer: {e}")
    
    async def stop(self):
        """Stop the event bus"""
        if self.kafka_producer:
            try:
                await self.kafka_producer.stop()
                logger.info("SAGA event bus stopped")
            except Exception as e:
                logger.error(f"Failed to stop Kafka producer: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        try:
            metrics = {}
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[0].startswith("saga_"):
                                try:
                                    value = float(parts[1])
                                    metrics[parts[0]] = value
                                except ValueError:
                                    continue
            
            return {
                "transactions_total": metrics.get("saga_transactions_total", 0),
                "inflight": metrics.get("saga_inflight_total", 0),
                "success_total": metrics.get("saga_success_total", 0),
                "failed_total": metrics.get("saga_failed_total", 0),
                "compensated_total": metrics.get("saga_compensated_total", 0),
                "steps_total": metrics.get("saga_steps_total", 0),
                "steps_success": metrics.get("saga_steps_success_total", 0),
                "steps_failed": metrics.get("saga_steps_failed_total", 0)
            }
        
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {}
