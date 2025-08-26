"""
Rate Limiter
============
Token bucket rate limiting for consumer throttling.
"""

import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter"""
    
    def __init__(self, rate_per_sec: float, burst: int = 100):
        self.rate = rate_per_sec
        self.capacity = burst
        self.tokens = burst
        self.last = time.time()
        self.lock = threading.Lock()
        
        logger.info(f"Token bucket initialized: {rate_per_sec} tokens/sec, burst: {burst}")
    
    def consume(self, n: int = 1) -> bool:
        """Consume n tokens, return True if successful"""
        with self.lock:
            now = time.time()
            
            # Add tokens based on time passed
            time_passed = now - self.last
            new_tokens = time_passed * self.rate
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last = now
            
            # Check if we have enough tokens
            if self.tokens >= n:
                self.tokens -= n
                return True
            
            return False
    
    def wait_for_tokens(self, n: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for n tokens to become available"""
        start_time = time.time()
        
        while True:
            if self.consume(n):
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return False
            
            # Wait a bit before checking again
            time.sleep(0.01)
    
    def get_tokens_available(self) -> float:
        """Get number of tokens currently available"""
        with self.lock:
            now = time.time()
            time_passed = now - self.last
            new_tokens = time_passed * self.rate
            return min(self.capacity, self.tokens + new_tokens)
    
    def reset(self):
        """Reset token bucket to initial state"""
        with self.lock:
            self.tokens = self.capacity
            self.last = time.time()
            logger.info("Token bucket reset")


class RateLimiter:
    """Rate limiter with multiple token buckets"""
    
    def __init__(self):
        self._buckets: dict = {}
        self.lock = threading.Lock()
        
        logger.info("Rate limiter initialized")
    
    def add_bucket(self, name: str, rate_per_sec: float, burst: int = 100):
        """Add a token bucket"""
        with self.lock:
            self._buckets[name] = TokenBucket(rate_per_sec, burst)
            logger.info(f"Added rate limit bucket: {name} ({rate_per_sec}/sec)")
    
    def remove_bucket(self, name: str) -> bool:
        """Remove a token bucket"""
        with self.lock:
            if name in self._buckets:
                del self._buckets[name]
                logger.info(f"Removed rate limit bucket: {name}")
                return True
            return False
    
    def consume(self, bucket_name: str, n: int = 1) -> bool:
        """Consume tokens from a specific bucket"""
        bucket = self._buckets.get(bucket_name)
        if not bucket:
            logger.warning(f"Rate limit bucket '{bucket_name}' not found")
            return True  # Allow if no bucket configured
        
        return bucket.consume(n)
    
    def wait_for_tokens(self, bucket_name: str, n: int = 1, 
                       timeout: Optional[float] = None) -> bool:
        """Wait for tokens from a specific bucket"""
        bucket = self._buckets.get(bucket_name)
        if not bucket:
            logger.warning(f"Rate limit bucket '{bucket_name}' not found")
            return True  # Allow if no bucket configured
        
        return bucket.wait_for_tokens(n, timeout)
    
    def get_bucket_stats(self, bucket_name: str) -> Optional[dict]:
        """Get statistics for a specific bucket"""
        bucket = self._buckets.get(bucket_name)
        if not bucket:
            return None
        
        return {
            "rate_per_sec": bucket.rate,
            "capacity": bucket.capacity,
            "tokens_available": bucket.get_tokens_available(),
            "utilization": (bucket.capacity - bucket.get_tokens_available()) / bucket.capacity
        }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all buckets"""
        stats = {}
        with self.lock:
            for name, bucket in self._buckets.items():
                stats[name] = self.get_bucket_stats(name)
        return stats
    
    def reset_bucket(self, bucket_name: str):
        """Reset a specific bucket"""
        bucket = self._buckets.get(bucket_name)
        if bucket:
            bucket.reset()
    
    def reset_all(self):
        """Reset all buckets"""
        with self.lock:
            for bucket in self._buckets.values():
                bucket.reset()
        logger.info("All rate limit buckets reset")


# Global instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_global_rate_limiter() -> RateLimiter:
    """Get the global rate limiter"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


def add_rate_limit_bucket(name: str, rate_per_sec: float, burst: int = 100) -> None:
    """Add a rate limit bucket to the global limiter"""
    get_global_rate_limiter().add_bucket(name, rate_per_sec, burst)


def consume_tokens(bucket_name: str, n: int = 1) -> bool:
    """Consume tokens from the global rate limiter"""
    return get_global_rate_limiter().consume(bucket_name, n)


def wait_for_tokens(bucket_name: str, n: int = 1, timeout: Optional[float] = None) -> bool:
    """Wait for tokens from the global rate limiter"""
    return get_global_rate_limiter().wait_for_tokens(bucket_name, n, timeout)
