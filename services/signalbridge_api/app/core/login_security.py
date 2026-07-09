"""
Login security primitives: brute-force lockout + JWT blacklist.

Both are Redis-backed and degrade gracefully: if Redis is unavailable they
fail OPEN for availability (login proceeds, token treated as valid) but log a
warning. This is deliberate for a trading dashboard where locking everyone out
on a Redis hiccup is worse than a brief protection gap. Security tests and load
tests should run with Redis up so the protections are exercised.

Design
------
- LoginThrottle: per-email + per-IP sliding counters. After `max_attempts`
  failures within `window_seconds`, the identity is locked for `lockout_seconds`.
  A successful login clears the counters. Returns retry-after so the API can
  emit HTTP 429 with a Retry-After header.
- TokenBlacklist: stores a token's `jti` with TTL equal to the token's remaining
  lifetime. `get_current_user` checks it so logout actually invalidates a token.
"""

from __future__ import annotations

import redis.asyncio as redis
import structlog

from .config import settings

logger = structlog.get_logger()

# Tunables (kept here rather than in the global Settings to avoid churn; can be
# promoted to env-driven settings later without changing callers).
MAX_LOGIN_ATTEMPTS = 5          # failures before lockout
LOGIN_WINDOW_SECONDS = 900      # 15 min rolling window to accumulate failures
LOCKOUT_SECONDS = 900           # 15 min lockout once tripped
BLACKLIST_PREFIX = "auth:blacklist:jti:"
FAIL_PREFIX = "auth:fail:"
LOCK_PREFIX = "auth:lock:"


_redis: redis.Redis | None = None


async def _get_redis() -> redis.Redis | None:
    """Lazily create a shared Redis client. Returns None if unreachable."""
    global _redis
    if _redis is None:
        try:
            _redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await _redis.ping()
        except Exception as exc:
            logger.warning("login_security: Redis unavailable, protections degraded", error=str(exc))
            _redis = None
    return _redis


class LoginThrottle:
    """Brute-force lockout keyed by identity (email) and client IP."""

    @staticmethod
    def _fail_key(scope: str, identity: str) -> str:
        return f"{FAIL_PREFIX}{scope}:{identity.lower()}"

    @staticmethod
    def _lock_key(scope: str, identity: str) -> str:
        return f"{LOCK_PREFIX}{scope}:{identity.lower()}"

    @classmethod
    async def check_locked(cls, email: str, ip: str) -> int:
        """Return remaining lockout seconds (>0 if locked, 0 otherwise)."""
        client = await _get_redis()
        if client is None:
            return 0
        try:
            for scope, identity in (("email", email), ("ip", ip)):
                ttl = await client.ttl(cls._lock_key(scope, identity))
                if ttl and ttl > 0:
                    return ttl
        except Exception as exc:
            logger.warning("login_security: check_locked failed", error=str(exc))
        return 0

    @classmethod
    async def record_failure(cls, email: str, ip: str) -> None:
        """Increment failure counters; trip a lockout when threshold reached."""
        client = await _get_redis()
        if client is None:
            return
        try:
            for scope, identity in (("email", email), ("ip", ip)):
                fkey = cls._fail_key(scope, identity)
                count = await client.incr(fkey)
                if count == 1:
                    await client.expire(fkey, LOGIN_WINDOW_SECONDS)
                if count >= MAX_LOGIN_ATTEMPTS:
                    await client.setex(cls._lock_key(scope, identity), LOCKOUT_SECONDS, "1")
                    await client.delete(fkey)
                    logger.warning(
                        "login_security: identity locked out",
                        scope=scope,
                        identity=identity.lower(),
                        lockout_seconds=LOCKOUT_SECONDS,
                    )
        except Exception as exc:
            logger.warning("login_security: record_failure failed", error=str(exc))

    @classmethod
    async def clear(cls, email: str, ip: str) -> None:
        """Clear failure counters after a successful login."""
        client = await _get_redis()
        if client is None:
            return
        try:
            keys = [
                cls._fail_key("email", email),
                cls._fail_key("ip", ip),
                cls._lock_key("email", email),
            ]
            await client.delete(*keys)
        except Exception as exc:
            logger.warning("login_security: clear failed", error=str(exc))


class TokenBlacklist:
    """JWT revocation list keyed by token jti."""

    @staticmethod
    async def add(jti: str, ttl_seconds: int) -> None:
        if not jti or ttl_seconds <= 0:
            return
        client = await _get_redis()
        if client is None:
            logger.warning("login_security: cannot blacklist token, Redis down", jti=jti)
            return
        try:
            await client.setex(f"{BLACKLIST_PREFIX}{jti}", ttl_seconds, "1")
        except Exception as exc:
            logger.warning("login_security: blacklist add failed", error=str(exc))

    @staticmethod
    async def is_blacklisted(jti: str) -> bool:
        if not jti:
            return False
        client = await _get_redis()
        if client is None:
            return False  # fail open
        try:
            return bool(await client.exists(f"{BLACKLIST_PREFIX}{jti}"))
        except Exception as exc:
            logger.warning("login_security: blacklist check failed", error=str(exc))
            return False
