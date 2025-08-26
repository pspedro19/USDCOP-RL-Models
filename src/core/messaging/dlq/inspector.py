"""
DLQ Inspector
=============
Tools for inspecting, replaying, and managing dead letter queue messages.
"""
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DLQInspector:
    """Tool for inspecting and managing DLQ messages."""
    
    def __init__(self, redis_client=None, kafka_producer=None):
        """Initialize DLQ inspector."""
        self.redis_client = redis_client
        self.kafka_producer = kafka_producer
    
    async def list_dlq_messages(self, topic: str, limit: int = 100, 
                               offset: int = 0) -> List[Dict[str, Any]]:
        """
        List messages in a DLQ topic.
        
        Args:
            topic: DLQ topic name
            limit: Maximum number of messages to return
            offset: Offset for pagination
            
        Returns:
            List of DLQ messages
        """
        try:
            if not self.redis_client:
                return []
            
            # Get message keys from index
            index_key = f"dlq:index:{topic}"
            message_keys = await self.redis_client.zrevrange(
                index_key, offset, offset + limit - 1
            )
            
            messages = []
            for key in message_keys:
                message_data = await self.redis_client.get(key)
                if message_data:
                    try:
                        message = json.loads(message_data)
                        messages.append(message)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in DLQ message: {key}")
                        continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to list DLQ messages: {e}")
            return []
    
    async def get_dlq_message(self, topic: str, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific DLQ message by ID."""
        try:
            if not self.redis_client:
                return None
            
            # Try to find the message
            pattern = f"dlq:{topic}:*{message_id}*"
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                message_data = await self.redis_client.get(keys[0])
                if message_data:
                    return json.loads(message_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get DLQ message: {e}")
            return None
    
    async def replay_message(self, topic: str, message_id: str, 
                           target_topic: str = None) -> bool:
        """
        Replay a DLQ message to its original topic or a specified target.
        
        Args:
            topic: DLQ topic containing the message
            message_id: ID of the message to replay
            target_topic: Target topic (defaults to original topic)
            
        Returns:
            True if replay was successful
        """
        try:
            # Get the message
            message = await self.get_dlq_message(topic, message_id)
            if not message:
                logger.error(f"Message not found: {message_id}")
                return False
            
            # Determine target topic
            if target_topic is None:
                target_topic = message.get('original_topic', topic.replace('.dlq', ''))
            
            # Extract original message content
            original_message = message.get('original_message', {})
            
            # Add replay metadata
            replay_message = {
                **original_message,
                'dlq_replay': True,
                'replay_timestamp': datetime.utcnow().isoformat() + "Z",
                'original_dlq_topic': topic,
                'replay_message_id': message_id
            }
            
            # Send to target topic
            if self.kafka_producer:
                key = message.get('correlation_id') or message.get('message_id') or str(time.time())
                value = json.dumps(replay_message)
                
                if hasattr(self.kafka_producer, 'send_and_wait'):
                    await self.kafka_producer.send_and_wait(target_topic, key=key, value=value)
                elif hasattr(self.kafka_producer, 'send'):
                    self.kafka_producer.send(target_topic, key=key, value=value)
                    self.kafka_producer.flush()
                
                logger.info(f"Replayed message {message_id} to {target_topic}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to replay message: {e}")
            return False
    
    async def purge_dlq_messages(self, topic: str, older_than_days: int = 7) -> int:
        """
        Purge old messages from a DLQ topic.
        
        Args:
            topic: DLQ topic to purge
            older_than_days: Remove messages older than this many days
            
        Returns:
            Number of messages purged
        """
        try:
            if not self.redis_client:
                return 0
            
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
            index_key = f"dlq:index:{topic}"
            
            # Get old message keys
            old_keys = await self.redis_client.zrangebyscore(
                index_key, 0, cutoff_time
            )
            
            # Remove old messages
            purged_count = 0
            for key in old_keys:
                await self.redis_client.delete(key)
                await self.redis_client.zrem(index_key, key)
                purged_count += 1
            
            logger.info(f"Purged {purged_count} old messages from {topic}")
            return purged_count
            
        except Exception as e:
            logger.error(f"Failed to purge DLQ messages: {e}")
            return 0
    
    async def export_dlq_messages(self, topic: str, format: str = "json", 
                                 limit: int = 1000) -> str:
        """
        Export DLQ messages for analysis.
        
        Args:
            topic: DLQ topic to export
            format: Export format ("json" or "csv")
            limit: Maximum number of messages to export
            
        Returns:
            Exported data as string
        """
        try:
            # Get messages
            messages = await self.list_dlq_messages(topic, limit=limit)
            
            if format.lower() == "csv":
                return self._export_to_csv(messages)
            else:
                return self._export_to_json(messages)
                
        except Exception as e:
            logger.error(f"Failed to export DLQ messages: {e}")
            return ""
    
    def _export_to_csv(self, messages: List[Dict[str, Any]]) -> str:
        """Export messages to CSV format."""
        if not messages:
            return ""
        
        # Get all possible fields
        all_fields = set()
        for message in messages:
            all_fields.update(message.keys())
        
        # Sort fields for consistent output
        field_order = sorted(list(all_fields))
        
        # Create CSV header
        csv_lines = [",".join(field_order)]
        
        # Add data rows
        for message in messages:
            row = []
            for field in field_order:
                value = message.get(field, "")
                # Escape commas and quotes
                if isinstance(value, str):
                    if "," in value or '"' in value:
                        value = f'"{value.replace('"', '""')}"'
                row.append(str(value))
            csv_lines.append(",".join(row))
        
        return "\n".join(csv_lines)
    
    def _export_to_json(self, messages: List[Dict[str, Any]]) -> str:
        """Export messages to JSON format."""
        return json.dumps(messages, indent=2, default=str)
    
    async def get_dlq_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all DLQ topics."""
        try:
            if not self.redis_client:
                return {}
            
            summary = {}
            
            # Get all DLQ index keys
            index_keys = await self.redis_client.keys("dlq:index:*")
            
            for index_key in index_keys:
                topic = index_key.replace("dlq:index:", "")
                
                # Get message count
                count = await self.redis_client.zcard(index_key)
                
                # Get oldest and newest message timestamps
                if count > 0:
                    oldest_score = await self.redis_client.zrange(index_key, 0, 0, withscores=True)
                    newest_score = await self.redis_client.zrevrange(index_key, 0, 0, withscores=True)
                    
                    oldest_time = oldest_score[0][1] if oldest_score else None
                    newest_time = newest_score[0][1] if newest_score else None
                    
                    summary[topic] = {
                        'message_count': count,
                        'oldest_message_age': time.time() - oldest_time if oldest_time else None,
                        'newest_message_age': time.time() - newest_time if newest_time else None
                    }
                else:
                    summary[topic] = {
                        'message_count': 0,
                        'oldest_message_age': None,
                        'newest_message_age': None
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get DLQ summary: {e}")
            return {}
    
    async def analyze_dlq_patterns(self, topic: str, limit: int = 1000) -> Dict[str, Any]:
        """
        Analyze patterns in DLQ messages.
        
        Args:
            topic: DLQ topic to analyze
            limit: Maximum number of messages to analyze
            
        Returns:
            Analysis results
        """
        try:
            messages = await self.list_dlq_messages(topic, limit=limit)
            
            if not messages:
                return {}
            
            # Analyze error patterns
            error_counts = {}
            service_counts = {}
            retry_count_distribution = {}
            
            for message in messages:
                # Count errors
                error = message.get('error', 'Unknown')
                error_type = error.split(':')[0] if ':' in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                # Count by source service
                service = message.get('source_service', 'Unknown')
                service_counts[service] = service_counts.get(service, 0) + 1
                
                # Retry count distribution
                retry_count = message.get('retry_count', 0)
                retry_count_distribution[retry_count] = retry_count_distribution.get(retry_count, 0) + 1
            
            return {
                'total_messages': len(messages),
                'error_patterns': error_counts,
                'service_distribution': service_counts,
                'retry_distribution': retry_count_distribution,
                'analysis_timestamp': datetime.utcnow().isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze DLQ patterns: {e}")
            return {}
