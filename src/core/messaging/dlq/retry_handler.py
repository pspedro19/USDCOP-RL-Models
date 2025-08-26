"""
Retry Handler
=============
Handles retry logic with exponential backoff for failed message processing.
"""
import asyncio
import random
import logging
from typing import Callable, Dict, Any, Optional, List, Type
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RetryOutcome(Enum):
    """Possible outcomes of a retry attempt."""
    SUCCESS = "success"
    RETRY = "retry"
    FAILED = "failed"
    DLQ = "dlq"

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0   # seconds
    backoff_multiplier: float = 2.0
    jitter: float = 0.1       # Random jitter factor
    retry_on: List[Type[Exception]] = None
    dlq_on: List[Type[Exception]] = None
    
    def __post_init__(self):
        """Set default retry and DLQ exception lists."""
        if self.retry_on is None:
            self.retry_on = [
                ConnectionError,
                TimeoutError,
                OSError,
                asyncio.TimeoutError
            ]
        
        if self.dlq_on is None:
            self.dlq_on = [
                ValueError,
                TypeError,
                KeyError,
                AttributeError
            ]

class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, policy: RetryPolicy = None, dlq_manager=None):
        """Initialize retry handler."""
        self.policy = policy or RetryPolicy()
        self.dlq_manager = dlq_manager
    
    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        exception_type = type(exception)
        
        # Check if exception is in retry list
        for retry_exception in self.policy.retry_on:
            if issubclass(exception_type, retry_exception):
                return True
        
        return False
    
    def should_send_to_dlq(self, exception: Exception) -> bool:
        """Determine if an exception should send message to DLQ."""
        exception_type = type(exception)
        
        # Check if exception is in DLQ list
        for dlq_exception in self.policy.dlq_on:
            if issubclass(exception_type, dlq_exception):
                return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with jitter."""
        # Exponential backoff
        delay = min(
            self.policy.base_delay * (self.policy.backoff_multiplier ** (attempt - 1)),
            self.policy.max_delay
        )
        
        # Add jitter
        jitter = delay * self.policy.jitter * random.uniform(-1, 1)
        delay += jitter
        
        return max(0.1, delay)  # Minimum 100ms delay
    
    async def execute_with_retry(self, func: Callable, *args, 
                                message: Dict[str, Any] = None,
                                topic: str = None,
                                correlation_id: str = None,
                                message_id: str = None,
                                source_service: str = None,
                                **kwargs) -> RetryOutcome:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            message: Message being processed (for DLQ)
            topic: Topic name (for DLQ)
            correlation_id: Correlation ID for tracing
            message_id: Message ID for identification
            source_service: Service processing the message
            **kwargs: Function keyword arguments
            
        Returns:
            RetryOutcome indicating the final result
        """
        last_exception = None
        
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - no retry needed
                logger.info(f"Function executed successfully on attempt {attempt}")
                return RetryOutcome.SUCCESS
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} failed: {type(e).__name__}: {e}")
                
                # Check if we should retry
                if attempt < self.policy.max_attempts and self.should_retry(e):
                    delay = self.calculate_delay(attempt)
                    logger.info(f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{self.policy.max_attempts})")
                    
                    await asyncio.sleep(delay)
                    continue
                
                # Check if we should send to DLQ
                if self.should_send_to_dlq(e):
                    logger.warning(f"Sending message to DLQ after {attempt} attempts")
                    await self._send_to_dlq(
                        topic, message, str(e), attempt,
                        correlation_id, message_id, source_service
                    )
                    return RetryOutcome.DLQ
                
                # Final failure
                logger.error(f"Function failed after {attempt} attempts: {e}")
                return RetryOutcome.FAILED
        
        # If we get here, all attempts failed
        if last_exception:
            logger.error(f"All {self.policy.max_attempts} attempts failed. Last error: {last_exception}")
        
        return RetryOutcome.FAILED
    
    async def _send_to_dlq(self, topic: str, message: Dict[str, Any], error: str, 
                           attempt: int, correlation_id: str = None, message_id: str = None,
                           source_service: str = None):
        """Send failed message to DLQ."""
        if self.dlq_manager and topic and message:
            try:
                await self.dlq_manager.send_to_dlq(
                    main_topic=topic,
                    original_message=message,
                    error=error,
                    retry_count=attempt,
                    correlation_id=correlation_id,
                    message_id=message_id,
                    source_service=source_service
                )
            except Exception as e:
                logger.error(f"Failed to send message to DLQ: {e}")
    
    def create_retry_policy(self, **kwargs) -> RetryPolicy:
        """Create a custom retry policy."""
        return RetryPolicy(**kwargs)
    
    async def process_message_with_retry(self, message: Dict[str, Any], topic: str,
                                       processor_func: Callable, *args, **kwargs) -> RetryOutcome:
        """
        Process a message with retry logic.
        
        Args:
            message: Message to process
            topic: Topic name
            processor_func: Function to process the message
            *args: Additional arguments for processor
            **kwargs: Additional keyword arguments for processor
            
        Returns:
            RetryOutcome indicating the result
        """
        # Extract metadata from message
        correlation_id = message.get('correlation_id')
        message_id = message.get('message_id')
        source_service = message.get('source_service')
        
        return await self.execute_with_retry(
            processor_func,
            message,
            *args,
            message=message,
            topic=topic,
            correlation_id=correlation_id,
            message_id=message_id,
            source_service=source_service,
            **kwargs
        )

# Predefined retry policies
DEFAULT_RETRY_POLICY = RetryPolicy()
AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay=0.5,
    max_delay=30.0,
    backoff_multiplier=1.5
)
CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    max_attempts=2,
    base_delay=2.0,
    max_delay=120.0,
    backoff_multiplier=3.0
)
