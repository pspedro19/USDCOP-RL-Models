"""
Drop Policy
===========
Message dropping strategies for backpressure management.
"""

import logging
from typing import Any, Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DropPolicy(str, Enum):
    """Available drop policies"""
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    DROP_RANDOM = "drop_random"
    SAMPLE = "sample"
    NONE = "none"


class MessageDropper:
    """Manages message dropping strategies"""
    
    def __init__(self):
        self._policies: dict = {}
        self._custom_policies: dict = {}
        
        # Register default policies
        self._register_default_policies()
        
        logger.info("Message dropper initialized")
    
    def _register_default_policies(self):
        """Register default dropping policies"""
        self.register_policy(DropPolicy.DROP_OLDEST, self._drop_oldest)
        self.register_policy(DropPolicy.DROP_NEWEST, self._drop_newest)
        self.register_policy(DropPolicy.DROP_RANDOM, self._drop_random)
        self.register_policy(DropPolicy.SAMPLE, self._sample)
        self.register_policy(DropPolicy.NONE, self._no_drop)
    
    def register_policy(self, name: str, policy: Callable):
        """Register a drop policy"""
        self._policies[name] = policy
        logger.info(f"Registered drop policy: {name}")
    
    def register_custom_policy(self, name: str, policy: Callable):
        """Register a custom drop policy"""
        self._custom_policies[name] = policy
        logger.info(f"Registered custom drop policy: {name}")
    
    def get_policy(self, name: str) -> Optional[Callable]:
        """Get a drop policy by name"""
        # Check built-in policies first
        if name in self._policies:
            return self._policies[name]
        
        # Check custom policies
        if name in self._custom_policies:
            return self._custom_policies[name]
        
        return None
    
    def execute_policy(self, policy_name: str, queue, message: Any, 
                      context: dict = None) -> bool:
        """Execute a drop policy"""
        policy = self.get_policy(policy_name)
        if not policy:
            logger.warning(f"Drop policy '{policy_name}' not found, using no_drop")
            policy = self._no_drop
        
        try:
            return policy(queue, message, context or {})
        except Exception as e:
            logger.error(f"Drop policy '{policy_name}' failed: {e}")
            return False
    
    def _drop_oldest(self, queue, message: Any, context: dict) -> bool:
        """Drop oldest message from queue"""
        try:
            if not queue.empty():
                old_message = queue.get_nowait()
                logger.debug(f"Dropped oldest message: {old_message}")
                return True
        except Exception as e:
            logger.warning(f"Failed to drop oldest message: {e}")
        
        return False
    
    def _drop_newest(self, queue, message: Any, context: dict) -> bool:
        """Drop newest message (don't add it)"""
        logger.debug(f"Dropped newest message: {message}")
        return True
    
    def _drop_random(self, queue, message: Any, context: dict) -> bool:
        """Randomly decide whether to drop message"""
        import random
        
        # 50% chance to drop
        if random.random() < 0.5:
            logger.debug(f"Randomly dropped message: {message}")
            return True
        
        return False
    
    def _sample(self, queue, message: Any, context: dict) -> bool:
        """Sample-based dropping"""
        import random
        
        # Get sampling rate from context
        sample_rate = context.get("sample_rate", 0.5)
        
        if random.random() < sample_rate:
            logger.debug(f"Sampled message (dropped): {message}")
            return True
        
        return False
    
    def _no_drop(self, queue, message: Any, context: dict) -> bool:
        """No dropping policy"""
        return False
    
    def list_policies(self) -> list:
        """List all available policy names"""
        policies = list(self._policies.keys())
        policies.extend(list(self._custom_policies.keys()))
        return policies
    
    def get_policy_info(self, name: str) -> Optional[dict]:
        """Get information about a policy"""
        policy = self.get_policy(name)
        if not policy:
            return None
        
        return {
            "name": name,
            "type": "builtin" if name in self._policies else "custom",
            "description": policy.__doc__ or "No description available"
        }


# Global instance
_global_message_dropper: Optional[MessageDropper] = None


def get_global_message_dropper() -> MessageDropper:
    """Get the global message dropper"""
    global _global_message_dropper
    if _global_message_dropper is None:
        _global_message_dropper = MessageDropper()
    return _global_message_dropper


def register_drop_policy(name: str, policy: Callable) -> None:
    """Register a drop policy in the global dropper"""
    get_global_message_dropper().register_policy(name, policy)


def register_custom_drop_policy(name: str, policy: Callable) -> None:
    """Register a custom drop policy in the global dropper"""
    get_global_message_dropper().register_custom_policy(name, policy)


def execute_drop_policy(policy_name: str, queue, message: Any, context: dict = None) -> bool:
    """Execute a drop policy using the global dropper"""
    return get_global_message_dropper().execute_policy(policy_name, queue, message, context)
