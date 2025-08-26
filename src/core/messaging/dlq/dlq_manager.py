"""
Dead Letter Queue Manager
=========================
Manages failed messages and moves them to DLQ topics.
"""
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

@dataclass
class DLQMessage:
    """Represents a message in the dead letter queue."""
    original_topic: str
    original_message: Dict[str, Any]
    error: str
    retry_count: int
    first_failed: str
    last_failed: str
    correlation_id: Optional[str] = None
    message_id: Optional[str] = None
    source_service: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)

class DLQManager:
    """Manages dead letter queue operations."""
    
    def __init__(self, kafka_producer=None, redis_client=None):
        """Initialize DLQ manager."""
        self.kafka_producer = kafka_producer
        self.redis_client = redis_client
        self.dlq_topics: Dict[str, str] = {}  # main_topic -> dlq_topic mapping
        
    def register_topic(self, main_topic: str, dlq_topic: str = None):
        """Register a topic pair for DLQ handling."""
        if dlq_topic is None:
            dlq_topic = f"{main_topic}.dlq"
        
        self.dlq_topics[main_topic] = dlq_topic
        logger.info(f"Registered DLQ topic: {main_topic} -> {dlq_topic}")
    
    def get_dlq_topic(self, main_topic: str) -> str:
        """Get the DLQ topic for a main topic."""
        return self.dlq_topics.get(main_topic, f"{main_topic}.dlq")
    
    async def send_to_dlq(self, main_topic: str, original_message: Dict[str, Any], 
                          error: str, retry_count: int = 0, 
                          correlation_id: str = None, message_id: str = None,
                          source_service: str = None, additional_context: Dict[str, Any] = None):
        """
        Send a failed message to the DLQ.
        
        Args:
            main_topic: Original topic where the message failed
            original_message: The original message content
            error: Error description
            retry_count: Number of retry attempts made
            correlation_id: Correlation ID for tracing
            message_id: Unique message identifier
            source_service: Service that processed the message
            additional_context: Extra context information
        """
        try:
            # Create DLQ message
            dlq_message = DLQMessage(
                original_topic=main_topic,
                original_message=original_message,
                error=error,
                retry_count=retry_count,
                first_failed=datetime.now(timezone.utc).isoformat(),
                last_failed=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id,
                message_id=message_id,
                source_service=source_service,
                additional_context=additional_context or {}
            )
            
            # Get DLQ topic
            dlq_topic = self.get_dlq_topic(main_topic)
            
            # Send to Kafka if available
            if self.kafka_producer:
                await self._send_to_kafka(dlq_topic, dlq_message)
            
            # Store in Redis if available
            if self.redis_client:
                await self._store_in_redis(dlq_topic, dlq_message)
            
            logger.warning(f"Message sent to DLQ: {main_topic} -> {dlq_topic}, error: {error}")
            
            # Emit metrics
            await self._emit_dlq_metrics(main_topic, dlq_message)
            
        except Exception as e:
            logger.error(f"Failed to send message to DLQ: {e}", exc_info=True)
    
    async def _send_to_kafka(self, dlq_topic: str, dlq_message: DLQMessage):
        """Send DLQ message to Kafka topic."""
        try:
            # Create message key from correlation ID or message ID
            key = dlq_message.correlation_id or dlq_message.message_id or str(time.time())
            
            # Serialize message
            value = dlq_message.to_json()
            
            # Send to Kafka
            if hasattr(self.kafka_producer, 'send_and_wait'):
                await self.kafka_producer.send_and_wait(dlq_topic, key=key, value=value)
            elif hasattr(self.kafka_producer, 'send'):
                self.kafka_producer.send(dlq_topic, key=key, value=value)
                self.kafka_producer.flush()
            
        except Exception as e:
            logger.error(f"Failed to send DLQ message to Kafka: {e}")
            raise
    
    async def _store_in_redis(self, dlq_topic: str, dlq_message: DLQMessage):
        """Store DLQ message in Redis for inspection."""
        try:
            # Create Redis key
            redis_key = f"dlq:{dlq_topic}:{dlq_message.correlation_id or dlq_message.message_id or time.time()}"
            
            # Store message with TTL (7 days)
            await self.redis_client.setex(
                redis_key,
                7 * 24 * 60 * 60,  # 7 days
                dlq_message.to_json()
            )
            
            # Add to DLQ index
            index_key = f"dlq:index:{dlq_topic}"
            await self.redis_client.zadd(
                index_key,
                {redis_key: time.time()}
            )
            
            # Set index TTL
            await self.redis_client.expire(index_key, 7 * 24 * 60 * 60)
            
        except Exception as e:
            logger.error(f"Failed to store DLQ message in Redis: {e}")
            raise
    
    async def _emit_dlq_metrics(self, main_topic: str, dlq_message: DLQMessage):
        """Emit metrics for DLQ operations."""
        try:
            # Increment DLQ message counter
            if self.redis_client:
                metric_key = f"metrics:dlq:{main_topic}:count"
                await self.redis_client.incr(metric_key)
                
                # Set TTL on metric
                await self.redis_client.expire(metric_key, 24 * 60 * 60)  # 24 hours
                
        except Exception as e:
            logger.error(f"Failed to emit DLQ metrics: {e}")
    
    async def get_dlq_stats(self, topic: str = None) -> Dict[str, Any]:
        """Get DLQ statistics."""
        try:
            stats = {}
            
            if self.redis_client:
                if topic:
                    # Get stats for specific topic
                    dlq_topic = self.get_dlq_topic(topic)
                    index_key = f"dlq:index:{dlq_topic}"
                    
                    # Get message count
                    count = await self.redis_client.zcard(index_key)
                    stats[dlq_topic] = {
                        'message_count': count,
                        'topic': dlq_topic
                    }
                else:
                    # Get stats for all topics
                    for main_topic, dlq_topic in self.dlq_topics.items():
                        index_key = f"dlq:index:{dlq_topic}"
                        count = await self.redis_client.zcard(index_key)
                        stats[dlq_topic] = {
                            'message_count': count,
                            'original_topic': main_topic
                        }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get DLQ stats: {e}")
            return {}
    
    async def cleanup_old_messages(self, max_age_days: int = 7):
        """Clean up old DLQ messages."""
        try:
            if not self.redis_client:
                return
            
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            cleaned_count = 0
            
            for main_topic, dlq_topic in self.dlq_topics.items():
                index_key = f"dlq:index:{dlq_topic}"
                
                # Get old message keys
                old_keys = await self.redis_client.zrangebyscore(
                    index_key, 0, cutoff_time
                )
                
                # Remove old messages
                for key in old_keys:
                    await self.redis_client.delete(key)
                    await self.redis_client.zrem(index_key, key)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old DLQ messages")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old DLQ messages: {e}")
    
    def add_message(self, message: Dict[str, Any], error: str = "Unknown error") -> str:
        """Add a message to the DLQ (sync wrapper for backward compatibility)"""
        import uuid
        
        message_id = str(uuid.uuid4())
        dlq_message = DLQMessage(
            original_topic="default",
            original_message=message,
            error=error,
            retry_count=0,
            first_failed=datetime.now(timezone.utc).isoformat(),
            last_failed=datetime.now(timezone.utc).isoformat(),
            message_id=message_id
        )
        
        # Store in memory for testing
        if not hasattr(self, '_messages'):
            self._messages = {}
        self._messages[message_id] = dlq_message
        
        return message_id
    
    def retry_message(self, message_id: str) -> bool:
        """Retry a message from the DLQ"""
        if not hasattr(self, '_messages'):
            return False
            
        if message_id in self._messages:
            message = self._messages[message_id]
            message.retry_count += 1
            message.last_failed = datetime.now(timezone.utc).isoformat()
            
            # Simulate retry logic
            if message.retry_count <= 3:
                return True
            return False
        return False
