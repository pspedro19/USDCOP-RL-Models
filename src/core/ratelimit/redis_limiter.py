"""
Redis Token Bucket Rate Limiter
===============================
Distributed rate limiting using Redis with token bucket algorithm.
"""
import os
import time
import asyncio
import logging
from typing import Dict, Tuple, Optional
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, resource: str, retry_after: float):
        self.resource = resource
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded for {resource}. Retry after {retry_after:.2f}s")

class RedisTokenBucket:
    """Redis-backed token bucket rate limiter with atomic Lua scripts."""
    
    # Lua script for atomic token bucket operations
    LUA_SCRIPT = """
    local bucket_key = KEYS[1]
    local now = tonumber(ARGV[1])
    local rate = tonumber(ARGV[2])
    local burst = tonumber(ARGV[3])
    local cost = tonumber(ARGV[4])
    
    -- Get current bucket state
    local data = redis.call('HMGET', bucket_key, 'tokens', 'last_refill', 'last_reset')
    local tokens = tonumber(data[1]) or burst
    local last_refill = tonumber(data[2]) or now
    local last_reset = tonumber(data[3]) or now
    
    -- Check if we need to reset (e.g., daily limits)
    local reset_interval = tonumber(ARGV[5]) or 0
    if reset_interval > 0 and (now - last_reset) >= reset_interval then
        tokens = burst
        last_refill = now
        last_reset = now
    end
    
    -- Refill tokens based on elapsed time
    local elapsed = now - last_refill
    local refill = math.min(elapsed * rate, burst - tokens)
    tokens = math.min(burst, tokens + refill)
    
    -- Check if we can consume the cost
    local allowed = 0
    local remaining = tokens
    local retry_after = 0
    
    if tokens >= cost then
        allowed = 1
        remaining = tokens - cost
        redis.call('HMSET', bucket_key, 'tokens', remaining, 'last_refill', now, 'last_reset', last_reset)
        redis.call('EXPIRE', bucket_key, 86400)  -- 24 hour TTL
    else
        -- Calculate when we'll have enough tokens
        local needed = cost - tokens
        retry_after = needed / rate
        redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_refill', now, 'last_reset', last_reset)
        redis.call('EXPIRE', bucket_key, 86400)  -- 24 hour TTL
    end
    
    return {allowed, remaining, retry_after}
    """
    
    def __init__(self, redis_url: str = None, timeout: float = 5.0):
        """Initialize the rate limiter."""
        self.redis_url = redis_url or os.getenv("RATE_LIMIT_REDIS_URL", "redis://redis:6379/3")
        self.timeout = timeout
        self.redis: Optional[aioredis.Redis] = None
        self.script_sha: Optional[str] = None
        
    async def connect(self):
        """Connect to Redis and load the Lua script."""
        if self.redis is None:
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            # Load the Lua script
            self.script_sha = await self.redis.script_load(self.LUA_SCRIPT)
            logger.info("Rate limiter connected to Redis")
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("Rate limiter disconnected from Redis")
    
    async def try_consume(self, scope: str, endpoint: str, cost: int = 1, 
                          rate: float = 1.0, burst: int = 10, 
                          reset_interval: int = 0) -> Tuple[bool, float, float]:
        """
        Try to consume tokens from the bucket.
        
        Args:
            scope: Rate limit scope (e.g., 'mt5_api', 'trade_execution')
            endpoint: Specific endpoint or identifier
            cost: Number of tokens to consume
            rate: Tokens per second
            burst: Maximum burst capacity
            reset_interval: Reset interval in seconds (0 = no reset)
            
        Returns:
            Tuple of (allowed, remaining_tokens, retry_after_seconds)
        """
        if self.redis is None:
            await self.connect()
        
        bucket_key = f"ratelimit:{scope}:{endpoint}"
        now = time.time()
        
        try:
            async with asyncio.timeout(self.timeout):
                result = await self.redis.eval(
                    self.script_sha,
                    1,  # Number of keys
                    bucket_key,
                    now,
                    rate,
                    burst,
                    cost,
                    reset_interval
                )
                
                allowed = bool(result[0])
                remaining = float(result[1])
                retry_after = float(result[2])
                
                return allowed, remaining, retry_after
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout acquiring rate limit for {scope}:{endpoint}")
            return False, 0.0, 0.0
        except Exception as e:
            logger.error(f"Error in rate limiting: {e}")
            return False, 0.0, 0.0
    
    async def get_bucket_info(self, scope: str, endpoint: str) -> Dict:
        """Get current bucket information."""
        if self.redis is None:
            await self.connect()
        
        bucket_key = f"ratelimit:{scope}:{endpoint}"
        try:
            data = await self.redis.hgetall(bucket_key)
            return {
                'tokens': float(data.get('tokens', 0)),
                'last_refill': float(data.get('last_refill', 0)),
                'last_reset': float(data.get('last_reset', 0)),
                'ttl': await self.redis.ttl(bucket_key)
            }
        except Exception as e:
            logger.error(f"Error getting bucket info: {e}")
            return {}
    
    async def reset_bucket(self, scope: str, endpoint: str):
        """Reset a bucket to full capacity."""
        if self.redis is None:
            await self.connect()
        
        bucket_key = f"ratelimit:{scope}:{endpoint}"
        try:
            await self.redis.delete(bucket_key)
            logger.info(f"Reset rate limit bucket: {scope}:{endpoint}")
        except Exception as e:
            logger.error(f"Error resetting bucket: {e}")

# Global rate limiter instance
_global_limiter: Optional[RedisTokenBucket] = None

async def get_rate_limiter() -> RedisTokenBucket:
    """Get the global rate limiter instance."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = RedisTokenBucket()
        await _global_limiter.connect()
    return _global_limiter

async def close_rate_limiter():
    """Close the global rate limiter."""
    global _global_limiter
    if _global_limiter:
        await _global_limiter.disconnect()
        _global_limiter = None

# Wrapper for backward compatibility
class RedisRateLimiter(RedisTokenBucket):
    """Backward compatible rate limiter"""
    
    def __init__(self, max_requests: int = 10, window_seconds: int = 60, 
                 redis_url: str = None, timeout: float = 5.0):
        """Initialize with backward compatible parameters"""
        super().__init__(redis_url, timeout)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.rate = max_requests / window_seconds  # Convert to rate
        self.burst = max_requests
    
    async def acquire(self, key: str = 'default', tokens: int = 1) -> bool:
        """Acquire tokens (backward compatible method)"""
        allowed, _, _ = await self.try_consume(
            scope='default',
            endpoint=key,
            cost=tokens,
            rate=self.rate,
            burst=self.burst
        )
        return allowed
    
    async def is_allowed(self, key: str = 'default') -> bool:
        """Check if request is allowed"""
        return await self.acquire(key, 1)
    
    def allow_request(self, key: str = 'default') -> bool:
        """Synchronous check if request is allowed"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # Can't run async in already running loop
            # Return True for now (permissive)
            return True
        else:
            return loop.run_until_complete(self.acquire(key, 1))
