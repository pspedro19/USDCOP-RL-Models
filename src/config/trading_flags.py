"""
Trading Flags - Single Source of Truth (SSOT)
==============================================
Centralized trading control flags for the USDCOP RL Trading System.

This module provides a frozen dataclass that serves as the SSOT for all
trading control flags, ensuring consistent behavior across the system.

Key Features:
- Frozen dataclass (immutable after creation)
- Environment-based configuration
- Singleton pattern with caching
- Emergency kill switch support
- Production validation

Usage:
    from src.config.trading_flags import get_trading_flags, TradingMode

    flags = get_trading_flags()
    if flags.can_execute_trade():
        # Execute trade
        pass

    mode = flags.get_trading_mode()
    if mode == TradingMode.PAPER:
        # Paper trading logic
        pass

Emergency Kill Switch:
    from src.config.trading_flags import activate_kill_switch

    activate_kill_switch("Market volatility exceeded threshold")

Author: Trading Team
Version: 2.0.0
Created: 2026-01-17
Updated: 2026-01-17 (Dia 1 Remediation - SSOT Implementation)
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Trading Modes and Environments
# =============================================================================

class TradingMode(Enum):
    """Trading execution modes with priority order."""

    KILLED = "KILLED"       # Emergency stop - highest priority
    DISABLED = "DISABLED"   # Trading disabled
    SHADOW = "SHADOW"       # Signal logging only, no execution
    PAPER = "PAPER"         # Simulated trading
    STAGING = "STAGING"     # Pre-production with real data
    LIVE = "LIVE"           # Full production trading


class Environment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# =============================================================================
# TradingFlags - Frozen Dataclass (SSOT)
# =============================================================================

@dataclass(frozen=True)
class TradingFlags:
    """
    Immutable trading control flags - Single Source of Truth (SSOT).

    This frozen dataclass ensures that trading flags cannot be modified
    after creation, preventing accidental state changes that could lead
    to unintended trading behavior.

    Attributes:
        trading_enabled: Master switch for all trading operations
        paper_trading: If True, simulate trades without real execution
        shadow_mode_enabled: Log signals without executing (audit mode)
        kill_switch_active: Emergency stop - blocks ALL trading immediately
        environment: Current deployment environment
        feature_flags_enabled: Enable/disable experimental features
        require_smoke_test: Model must pass smoke test before promotion
        require_dataset_hash: Training dataset hash must be verified
        min_staging_days: Minimum days a model must run in staging
        created_at: Timestamp when flags were loaded
        kill_switch_reason: Reason for kill switch activation (if any)
    """

    # Core trading control flags
    trading_enabled: bool = False
    paper_trading: bool = True
    shadow_mode_enabled: bool = True
    kill_switch_active: bool = False
    environment: str = "development"
    feature_flags_enabled: bool = False

    # Model promotion flags
    require_smoke_test: bool = True
    require_dataset_hash: bool = True
    min_staging_days: int = 7

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    kill_switch_reason: Optional[str] = None

    @classmethod
    def from_env(cls) -> TradingFlags:
        """
        Create TradingFlags from environment variables.

        This is the primary factory method for creating TradingFlags instances.
        All values are read from environment variables with safe defaults that
        prevent accidental live trading.

        Returns:
            TradingFlags: Immutable flags instance

        Example:
            flags = TradingFlags.from_env()
        """
        def parse_bool(value: str, default: bool = False) -> bool:
            """Parse boolean from environment variable string."""
            if not value:
                return default
            return value.lower() in ("true", "1", "yes", "on")

        def parse_int(value: str, default: int) -> int:
            """Parse integer from environment variable string."""
            if not value:
                return default
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Invalid integer value: {value}, using default: {default}")
                return default

        # Read environment variables with safe defaults
        trading_enabled = parse_bool(os.getenv("TRADING_ENABLED"), default=False)
        paper_trading = parse_bool(os.getenv("PAPER_TRADING"), default=True)
        shadow_mode_enabled = parse_bool(os.getenv("SHADOW_MODE_ENABLED"), default=True)
        kill_switch_active = parse_bool(os.getenv("KILL_SWITCH_ACTIVE"), default=False)
        environment = os.getenv("ENVIRONMENT", "development").lower()
        feature_flags_enabled = parse_bool(os.getenv("FEATURE_FLAGS_ENABLED"), default=False)

        # Model promotion flags
        require_smoke_test = parse_bool(os.getenv("REQUIRE_SMOKE_TEST"), default=True)
        require_dataset_hash = parse_bool(os.getenv("REQUIRE_DATASET_HASH"), default=True)
        min_staging_days = parse_int(os.getenv("MIN_STAGING_DAYS"), default=7)

        # Read kill switch reason if active
        kill_switch_reason = os.getenv("KILL_SWITCH_REASON") if kill_switch_active else None

        flags = cls(
            trading_enabled=trading_enabled,
            paper_trading=paper_trading,
            shadow_mode_enabled=shadow_mode_enabled,
            kill_switch_active=kill_switch_active,
            environment=environment,
            feature_flags_enabled=feature_flags_enabled,
            require_smoke_test=require_smoke_test,
            require_dataset_hash=require_dataset_hash,
            min_staging_days=min_staging_days,
            created_at=datetime.utcnow(),
            kill_switch_reason=kill_switch_reason,
        )

        logger.info(f"TradingFlags loaded: mode={flags.get_trading_mode().value}, env={environment}")
        return flags

    def can_execute_trade(self) -> bool:
        """
        Check if trading execution is permitted.

        This is the primary gate for all trading operations. A trade can only
        be executed if ALL of the following conditions are met:
        1. Kill switch is NOT active
        2. Trading is enabled
        3. Paper trading is disabled (for real trades)
        4. Shadow mode is disabled (shadow mode only logs, doesn't execute)
        5. Environment is production

        Returns:
            bool: True if real trade execution is permitted

        Example:
            if flags.can_execute_trade():
                execute_order(order)
            else:
                logger.warning("Trade blocked by trading flags")
        """
        # Kill switch has highest priority - blocks everything
        if self.kill_switch_active:
            logger.warning(f"Trade blocked: Kill switch active. Reason: {self.kill_switch_reason}")
            return False

        # Trading must be enabled
        if not self.trading_enabled:
            logger.debug("Trade blocked: Trading is disabled")
            return False

        # Shadow mode only logs, doesn't execute
        if self.shadow_mode_enabled:
            logger.debug("Trade blocked: Shadow mode active (logging only)")
            return False

        # Paper trading mode - can execute paper trades but not real ones
        if self.paper_trading:
            logger.info("Paper trading mode - no real execution")
            return False

        # Environment check for real trades
        if self.environment != "production":
            logger.warning(f"Trade blocked: Environment is {self.environment}, not production")
            return False

        return True

    def can_execute_paper_trade(self) -> bool:
        """
        Check if paper trading execution is permitted.

        Paper trades can be executed if:
        1. Kill switch is NOT active
        2. Trading is enabled
        3. Paper trading mode is enabled
        4. Shadow mode is disabled

        Returns:
            bool: True if paper trade execution is permitted
        """
        if self.kill_switch_active:
            return False

        if not self.trading_enabled:
            return False

        if self.shadow_mode_enabled:
            return False

        return self.paper_trading

    def can_execute_live_trade(self) -> bool:
        """
        Check if live (real money) trading is permitted.

        Alias for can_execute_trade() for clarity.

        Returns:
            bool: True if live trade execution is permitted
        """
        return self.can_execute_trade()

    def get_trading_mode(self) -> TradingMode:
        """
        Get the current trading mode based on flag states.

        Modes are evaluated in priority order (highest to lowest):
        1. KILLED - Kill switch is active
        2. DISABLED - Trading is not enabled
        3. SHADOW - Shadow mode is active (logging only)
        4. PAPER - Paper trading mode
        5. STAGING - Trading enabled but not in production environment
        6. LIVE - Full production trading

        Returns:
            TradingMode: Current trading mode enum value

        Example:
            mode = flags.get_trading_mode()
            if mode == TradingMode.KILLED:
                send_alert("Trading halted - kill switch active")
        """
        if self.kill_switch_active:
            return TradingMode.KILLED

        if not self.trading_enabled:
            return TradingMode.DISABLED

        if self.shadow_mode_enabled:
            return TradingMode.SHADOW

        if self.paper_trading:
            return TradingMode.PAPER

        if self.environment != "production":
            return TradingMode.STAGING

        return TradingMode.LIVE

    def validate_for_production(self) -> Tuple[bool, List[str]]:
        """
        Validate flags for production deployment.

        Performs comprehensive validation to ensure the system is properly
        configured for production trading. This should be called during
        deployment and system startup.

        Returns:
            tuple[bool, list[str]]: (is_valid, list_of_errors)
                - is_valid: True if all production requirements are met
                - list_of_errors: List of validation error messages

        Example:
            is_valid, errors = flags.validate_for_production()
            if not is_valid:
                for error in errors:
                    logger.error(f"Production validation failed: {error}")
                raise ConfigurationError("Invalid production configuration")
        """
        errors: List[str] = []

        # Environment checks
        if self.environment != "production":
            errors.append(f"Environment is '{self.environment}', expected 'production'")

        # Kill switch should not be active for normal production
        if self.kill_switch_active:
            errors.append(f"Kill switch is active: {self.kill_switch_reason or 'No reason provided'}")

        # Trading must be enabled for production
        if not self.trading_enabled:
            errors.append("Trading is disabled - set TRADING_ENABLED=true for production")

        # Shadow mode should be disabled for real trading
        if self.shadow_mode_enabled:
            errors.append("Shadow mode is enabled - disable for production trading")

        # Paper trading should be disabled for real trading
        if self.paper_trading:
            errors.append("Paper trading is enabled - disable for live production trading")

        # Model promotion safety checks
        if not self.require_smoke_test:
            errors.append("REQUIRE_SMOKE_TEST should be true in production")

        if not self.require_dataset_hash:
            errors.append("REQUIRE_DATASET_HASH should be true in production")

        if self.min_staging_days < 7:
            errors.append(f"MIN_STAGING_DAYS is {self.min_staging_days}, should be >= 7")

        is_valid = len(errors) == 0
        return is_valid, errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert flags to dictionary for serialization/logging.

        Returns:
            dict: Dictionary representation of flags
        """
        return {
            "trading_enabled": self.trading_enabled,
            "paper_trading": self.paper_trading,
            "shadow_mode_enabled": self.shadow_mode_enabled,
            "kill_switch_active": self.kill_switch_active,
            "environment": self.environment,
            "feature_flags_enabled": self.feature_flags_enabled,
            "require_smoke_test": self.require_smoke_test,
            "require_dataset_hash": self.require_dataset_hash,
            "min_staging_days": self.min_staging_days,
            "trading_mode": self.get_trading_mode().value,
            "created_at": self.created_at.isoformat(),
            "kill_switch_reason": self.kill_switch_reason,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        mode = self.get_trading_mode()
        return (
            f"TradingFlags(mode={mode.value}, env={self.environment}, "
            f"enabled={self.trading_enabled}, paper={self.paper_trading}, "
            f"shadow={self.shadow_mode_enabled}, kill_switch={self.kill_switch_active})"
        )


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Alias for backward compatibility with existing code using TradingFlagsEnv
TradingFlagsEnv = TradingFlags


# =============================================================================
# Singleton Pattern with Thread-Safe Caching
# =============================================================================

_trading_flags_cache: Optional[TradingFlags] = None
_trading_flags_lock = threading.Lock()


def get_trading_flags() -> TradingFlags:
    """
    Get cached TradingFlags singleton instance.

    This function implements a thread-safe singleton pattern for TradingFlags.
    The flags are loaded once from environment variables and cached for
    subsequent calls.

    Returns:
        TradingFlags: Cached flags instance

    Example:
        flags = get_trading_flags()
        if flags.can_execute_trade():
            # Proceed with trade
            pass
    """
    global _trading_flags_cache

    if _trading_flags_cache is None:
        with _trading_flags_lock:
            # Double-check locking pattern
            if _trading_flags_cache is None:
                _trading_flags_cache = TradingFlags.from_env()
                logger.info(f"TradingFlags initialized: {_trading_flags_cache}")

    return _trading_flags_cache


def reload_trading_flags() -> TradingFlags:
    """
    Force reload TradingFlags from environment variables.

    Use this function when environment variables have been updated and
    you need to refresh the cached flags. This is useful for:
    - Runtime configuration updates
    - After modifying .env file
    - Testing different configurations

    Returns:
        TradingFlags: Newly loaded flags instance

    Example:
        # After updating environment variables
        os.environ["TRADING_ENABLED"] = "true"
        flags = reload_trading_flags()
    """
    global _trading_flags_cache

    with _trading_flags_lock:
        old_flags = _trading_flags_cache
        _trading_flags_cache = TradingFlags.from_env()

        if old_flags:
            old_mode = old_flags.get_trading_mode()
            new_mode = _trading_flags_cache.get_trading_mode()
            if old_mode != new_mode:
                logger.warning(
                    f"Trading mode changed: {old_mode.value} -> {new_mode.value}"
                )

        logger.info(f"TradingFlags reloaded: {_trading_flags_cache}")
        return _trading_flags_cache


def activate_kill_switch(reason: str) -> TradingFlags:
    """
    Emergency kill switch activation.

    This function immediately sets the kill switch environment variable
    and reloads the trading flags. Use this in emergency situations to
    halt all trading operations immediately.

    IMPORTANT: This is an emergency function. Use with caution.

    Args:
        reason: Human-readable reason for activation (logged and stored)

    Returns:
        TradingFlags: Updated flags with kill switch active

    Example:
        # Emergency situations
        activate_kill_switch("Market volatility exceeded 5 sigma")
        activate_kill_switch("API connection lost - data integrity risk")
        activate_kill_switch("Manual intervention - suspicious activity detected")
    """
    timestamp = datetime.utcnow().isoformat()
    full_reason = f"[{timestamp}] {reason}"

    # Set environment variables
    os.environ["KILL_SWITCH_ACTIVE"] = "true"
    os.environ["KILL_SWITCH_REASON"] = full_reason

    # Log critical alert
    logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    # Reload flags with new state
    return reload_trading_flags()


def deactivate_kill_switch() -> TradingFlags:
    """
    Deactivate the kill switch.

    This function clears the kill switch and reloads trading flags.
    Should only be called after the emergency situation has been resolved
    and the system has been verified safe to resume trading.

    IMPORTANT: Verify system state before deactivating kill switch.

    Returns:
        TradingFlags: Updated flags with kill switch deactivated

    Example:
        # After resolving emergency
        flags = deactivate_kill_switch()
        if flags.can_execute_trade():
            logger.info("Trading resumed after kill switch deactivation")
    """
    os.environ["KILL_SWITCH_ACTIVE"] = "false"
    if "KILL_SWITCH_REASON" in os.environ:
        del os.environ["KILL_SWITCH_REASON"]

    logger.warning("KILL SWITCH DEACTIVATED - Trading may resume")

    return reload_trading_flags()


def reset_trading_flags_cache() -> None:
    """
    Reset the trading flags cache (primarily for testing).

    This function clears the cached TradingFlags instance, forcing
    the next call to get_trading_flags() to reload from environment.

    Note: This is primarily useful for testing scenarios.
    """
    global _trading_flags_cache

    with _trading_flags_lock:
        _trading_flags_cache = None
        logger.debug("TradingFlags cache cleared")


# =============================================================================
# Convenience Functions
# =============================================================================

def is_live_trading_enabled() -> bool:
    """Check if live trading is currently enabled."""
    return get_trading_flags().can_execute_live_trade()


def is_paper_trading_enabled() -> bool:
    """Check if paper trading is currently enabled."""
    return get_trading_flags().can_execute_paper_trade()


def is_kill_switch_active() -> bool:
    """Check if kill switch is currently active."""
    return get_trading_flags().kill_switch_active


def get_current_environment() -> str:
    """Get the current deployment environment."""
    return get_trading_flags().environment


def get_current_trading_mode() -> TradingMode:
    """Get the current trading mode."""
    return get_trading_flags().get_trading_mode()


# =============================================================================
# Backward Compatibility Functions
# =============================================================================

@lru_cache()
def get_trading_flags_env() -> TradingFlags:
    """
    Get singleton TradingFlags instance (backward compatibility alias).

    Deprecated: Use get_trading_flags() instead.
    """
    return get_trading_flags()


def reload_trading_flags_env() -> TradingFlags:
    """
    Reload flags (backward compatibility alias).

    Deprecated: Use reload_trading_flags() instead.
    """
    get_trading_flags_env.cache_clear()
    return reload_trading_flags()


def reset_trading_flags() -> None:
    """
    Reset the singleton instance (backward compatibility alias).

    Deprecated: Use reset_trading_flags_cache() instead.
    """
    reset_trading_flags_cache()
    get_trading_flags_env.cache_clear()


# =============================================================================
# Database-backed TradingFlagsDB (Optional, for runtime control)
# =============================================================================

try:
    import psycopg2

    def _get_db_connection():
        """Get database connection using environment variables."""
        password = os.environ.get('POSTGRES_PASSWORD')
        if not password:
            raise ValueError("POSTGRES_PASSWORD environment variable is required")

        return psycopg2.connect(
            host=os.environ.get('POSTGRES_HOST', 'timescaledb'),
            port=int(os.environ.get('POSTGRES_PORT', '5432')),
            database=os.environ.get('POSTGRES_DB', 'usdcop'),
            user=os.environ.get('POSTGRES_USER', 'admin'),
            password=password
        )

    @dataclass
    class FlagState:
        """State of a trading flag in the database."""
        enabled: bool
        reason: Optional[str]
        updated_at: datetime
        updated_by: str

    class TradingFlagsDB:
        """
        Database-backed runtime control flags for the trading system.

        This class provides runtime-modifiable flags that are persisted in
        the database. Use this for flags that need to be changed during
        operation without restarting services.

        Note: For static configuration flags, use TradingFlags (frozen dataclass).

        Available Flags:
            - kill_switch: Emergency halt of all trading
            - maintenance_mode: System in maintenance, non-critical ops blocked
            - paper_trading: Execute in paper trading mode only
            - inference_enabled: Whether inference is allowed to run

        Thread Safety:
            This class uses database transactions for flag updates.
            Multiple readers are safe; writers should coordinate.

        Example:
            flags = TradingFlagsDB()

            # Read flags
            if flags.kill_switch:
                return {"error": "Trading halted"}

            # Update with reason
            flags.set_kill_switch(True, reason="High volatility")
        """

        FLAG_KILL_SWITCH = "kill_switch"
        FLAG_MAINTENANCE = "maintenance_mode"
        FLAG_PAPER_TRADING = "paper_trading"
        FLAG_INFERENCE_ENABLED = "inference_enabled"

        def __init__(self, db_connection=None):
            """
            Initialize TradingFlagsDB.

            Args:
                db_connection: Optional database connection. If None, creates new connection.
            """
            self._conn = db_connection
            self._cache: Dict[str, FlagState] = {}
            self._cache_ttl_seconds = 10
            self._last_fetch: Optional[datetime] = None
            self._ensure_table()

        def _get_connection(self):
            """Get database connection, creating if needed."""
            if self._conn is None or self._conn.closed:
                self._conn = _get_db_connection()
            return self._conn

        def _ensure_table(self):
            """Ensure trading_flags table exists."""
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS config.trading_flags (
                        flag_name VARCHAR(50) PRIMARY KEY,
                        enabled BOOLEAN NOT NULL DEFAULT FALSE,
                        reason TEXT,
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_by VARCHAR(100) DEFAULT 'system'
                    );
                    INSERT INTO config.trading_flags (flag_name, enabled, reason)
                    VALUES
                        ('kill_switch', FALSE, 'System initialized'),
                        ('maintenance_mode', FALSE, 'System initialized'),
                        ('paper_trading', TRUE, 'Default to paper trading'),
                        ('inference_enabled', TRUE, 'System initialized')
                    ON CONFLICT (flag_name) DO NOTHING;
                """)
                conn.commit()
                cur.close()
            except Exception as e:
                logger.warning(f"Could not ensure trading_flags table: {e}")

        def _fetch_flags(self, force: bool = False):
            """Fetch flags from database with caching."""
            now = datetime.now()
            if not force and self._last_fetch is not None:
                age = (now - self._last_fetch).total_seconds()
                if age < self._cache_ttl_seconds:
                    return

            try:
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute("""
                    SELECT flag_name, enabled, reason, updated_at, updated_by
                    FROM config.trading_flags
                """)
                for row in cur.fetchall():
                    self._cache[row[0]] = FlagState(
                        enabled=row[1], reason=row[2],
                        updated_at=row[3], updated_by=row[4]
                    )
                cur.close()
                self._last_fetch = now
            except Exception as e:
                logger.error(f"Error fetching trading flags: {e}")

        def _get_flag(self, flag_name: str) -> bool:
            """Get a flag value with caching."""
            self._fetch_flags()
            if flag_name in self._cache:
                return self._cache[flag_name].enabled
            if flag_name == self.FLAG_KILL_SWITCH:
                return False
            elif flag_name == self.FLAG_INFERENCE_ENABLED:
                return True
            return False

        def _set_flag(self, flag_name: str, enabled: bool, reason: str = None, updated_by: str = "system"):
            """Set a flag value."""
            try:
                conn = self._get_connection()
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO config.trading_flags (flag_name, enabled, reason, updated_at, updated_by)
                    VALUES (%s, %s, %s, NOW(), %s)
                    ON CONFLICT (flag_name) DO UPDATE SET
                        enabled = EXCLUDED.enabled,
                        reason = EXCLUDED.reason,
                        updated_at = EXCLUDED.updated_at,
                        updated_by = EXCLUDED.updated_by
                """, (flag_name, enabled, reason, updated_by))
                conn.commit()
                cur.close()
                self._cache[flag_name] = FlagState(
                    enabled=enabled, reason=reason,
                    updated_at=datetime.now(), updated_by=updated_by
                )
                logger.info(f"Trading flag '{flag_name}' set to {enabled}. Reason: {reason}")
            except Exception as e:
                logger.error(f"Error setting trading flag '{flag_name}': {e}")
                raise

        @property
        def kill_switch(self) -> bool:
            """Check if kill switch is active."""
            return self._get_flag(self.FLAG_KILL_SWITCH)

        @property
        def maintenance_mode(self) -> bool:
            """Check if system is in maintenance mode."""
            return self._get_flag(self.FLAG_MAINTENANCE)

        @property
        def paper_trading(self) -> bool:
            """Check if paper trading mode is active."""
            return self._get_flag(self.FLAG_PAPER_TRADING)

        @property
        def inference_enabled(self) -> bool:
            """Check if inference is enabled."""
            return self._get_flag(self.FLAG_INFERENCE_ENABLED)

        def set_kill_switch(self, enabled: bool, reason: str = None, updated_by: str = "system"):
            """Set the kill switch state."""
            if enabled:
                logger.critical(f"KILL SWITCH ACTIVATED by {updated_by}. Reason: {reason}")
            else:
                logger.warning(f"Kill switch DEACTIVATED by {updated_by}. Reason: {reason}")
            self._set_flag(self.FLAG_KILL_SWITCH, enabled, reason, updated_by)

        def set_maintenance_mode(self, enabled: bool, reason: str = None, updated_by: str = "system"):
            """Set maintenance mode state."""
            self._set_flag(self.FLAG_MAINTENANCE, enabled, reason, updated_by)

        def set_paper_trading(self, enabled: bool, reason: str = None, updated_by: str = "system"):
            """Set paper trading mode state."""
            self._set_flag(self.FLAG_PAPER_TRADING, enabled, reason, updated_by)

        def set_inference_enabled(self, enabled: bool, reason: str = None, updated_by: str = "system"):
            """Set inference enabled state."""
            self._set_flag(self.FLAG_INFERENCE_ENABLED, enabled, reason, updated_by)

        def get_all_flags(self, force_refresh: bool = False) -> Dict[str, Any]:
            """Get all trading flags with their states."""
            self._fetch_flags(force=force_refresh)
            return {
                "kill_switch": self._cache.get(self.FLAG_KILL_SWITCH, FlagState(False, None, datetime.now(), "default")).__dict__,
                "maintenance_mode": self._cache.get(self.FLAG_MAINTENANCE, FlagState(False, None, datetime.now(), "default")).__dict__,
                "paper_trading": self._cache.get(self.FLAG_PAPER_TRADING, FlagState(True, None, datetime.now(), "default")).__dict__,
                "inference_enabled": self._cache.get(self.FLAG_INFERENCE_ENABLED, FlagState(True, None, datetime.now(), "default")).__dict__,
                "cached_at": self._last_fetch.isoformat() if self._last_fetch else None,
            }

        def is_trading_allowed(self) -> Tuple[bool, str]:
            """Check if trading operations are allowed."""
            if self.kill_switch:
                return False, "Kill switch is active"
            if self.maintenance_mode:
                return False, "System in maintenance mode"
            return True, "Trading allowed"

        def is_inference_allowed(self) -> Tuple[bool, str]:
            """Check if inference operations are allowed."""
            if self.kill_switch:
                return False, "Kill switch is active"
            if not self.inference_enabled:
                return False, "Inference is disabled"
            return True, "Inference allowed"

    # Singleton for database-backed flags
    _trading_flags_db: Optional[TradingFlagsDB] = None

    def get_trading_flags_db() -> TradingFlagsDB:
        """Get the singleton TradingFlagsDB instance."""
        global _trading_flags_db
        if _trading_flags_db is None:
            _trading_flags_db = TradingFlagsDB()
        return _trading_flags_db

    def reset_trading_flags_db() -> None:
        """Reset the database-backed singleton instance (for testing)."""
        global _trading_flags_db
        _trading_flags_db = None

except ImportError:
    # psycopg2 not available - database features disabled
    logger.debug("psycopg2 not available - TradingFlagsDB disabled")
    TradingFlagsDB = None  # type: ignore
    FlagState = None  # type: ignore
    get_trading_flags_db = None  # type: ignore
    reset_trading_flags_db = None  # type: ignore


# =============================================================================
# Redis-backed Trading Flags (P2 Remediation - Kill Switch to Redis)
# =============================================================================

try:
    import redis
    import json

    class TradingFlagsRedis:
        """
        Redis-backed runtime control flags for cross-service coordination.

        This class stores trading flags in Redis, enabling:
        - Cross-service visibility (all services see the same state)
        - Fast reads/writes (in-memory store)
        - Persistence with AOF/RDB
        - Real-time propagation via Pub/Sub

        Available Flags:
            - kill_switch: Emergency halt of all trading
            - maintenance_mode: System in maintenance
            - paper_trading: Execute in paper trading mode only
            - inference_enabled: Whether inference is allowed

        Thread Safety:
            Redis operations are atomic. Safe for concurrent access.

        Example:
            flags = TradingFlagsRedis()

            # Read flags
            if flags.kill_switch:
                return {"error": "Trading halted"}

            # Update with reason
            flags.set_kill_switch(True, reason="High volatility")

        Author: Trading Team
        Date: 2026-01-18 (P2 Remediation)
        """

        REDIS_KEY_PREFIX = "trading:flags"
        FLAG_KILL_SWITCH = "kill_switch"
        FLAG_MAINTENANCE = "maintenance_mode"
        FLAG_PAPER_TRADING = "paper_trading"
        FLAG_INFERENCE_ENABLED = "inference_enabled"

        # Channel for pub/sub notifications
        CHANNEL_FLAG_CHANGES = "trading:flags:changes"

        def __init__(
            self,
            redis_client: Optional[redis.Redis] = None,
            host: Optional[str] = None,
            port: Optional[int] = None,
            db: int = 0,
            enable_pubsub: bool = True,
        ):
            """
            Initialize TradingFlagsRedis.

            Args:
                redis_client: Optional pre-configured Redis client
                host: Redis host (default from REDIS_HOST env var)
                port: Redis port (default from REDIS_PORT env var)
                db: Redis database number
                enable_pubsub: Enable pub/sub for flag change notifications
            """
            if redis_client:
                self._redis = redis_client
            else:
                self._redis = redis.Redis(
                    host=host or os.environ.get("REDIS_HOST", "redis"),
                    port=port or int(os.environ.get("REDIS_PORT", "6379")),
                    db=db,
                    decode_responses=True,
                )
            self._enable_pubsub = enable_pubsub
            self._ensure_defaults()

        def _key(self, flag_name: str) -> str:
            """Generate Redis key for a flag."""
            return f"{self.REDIS_KEY_PREFIX}:{flag_name}"

        def _ensure_defaults(self):
            """Ensure default flag values exist in Redis."""
            defaults = {
                self.FLAG_KILL_SWITCH: {"enabled": False, "reason": "System initialized"},
                self.FLAG_MAINTENANCE: {"enabled": False, "reason": "System initialized"},
                self.FLAG_PAPER_TRADING: {"enabled": True, "reason": "Default to paper"},
                self.FLAG_INFERENCE_ENABLED: {"enabled": True, "reason": "System initialized"},
            }
            for flag_name, default_state in defaults.items():
                if not self._redis.exists(self._key(flag_name)):
                    self._set_flag_internal(flag_name, **default_state, updated_by="system")

        def _get_flag(self, flag_name: str) -> bool:
            """Get a flag value from Redis."""
            try:
                data = self._redis.get(self._key(flag_name))
                if data:
                    state = json.loads(data)
                    return state.get("enabled", False)
                # Defaults
                if flag_name == self.FLAG_KILL_SWITCH:
                    return False
                elif flag_name == self.FLAG_INFERENCE_ENABLED:
                    return True
                return False
            except Exception as e:
                logger.error(f"Error getting flag {flag_name} from Redis: {e}")
                # Safe defaults on error
                if flag_name == self.FLAG_KILL_SWITCH:
                    return True  # Fail safe: assume kill switch is ON
                return False

        def _get_flag_state(self, flag_name: str) -> Optional[Dict[str, Any]]:
            """Get full flag state from Redis."""
            try:
                data = self._redis.get(self._key(flag_name))
                if data:
                    return json.loads(data)
                return None
            except Exception as e:
                logger.error(f"Error getting flag state {flag_name} from Redis: {e}")
                return None

        def _set_flag_internal(
            self,
            flag_name: str,
            enabled: bool,
            reason: Optional[str] = None,
            updated_by: str = "system",
        ):
            """Internal method to set a flag value in Redis."""
            state = {
                "enabled": enabled,
                "reason": reason,
                "updated_at": datetime.utcnow().isoformat(),
                "updated_by": updated_by,
            }
            self._redis.set(self._key(flag_name), json.dumps(state))

        def _set_flag(
            self,
            flag_name: str,
            enabled: bool,
            reason: Optional[str] = None,
            updated_by: str = "system",
        ):
            """Set a flag value in Redis with pub/sub notification."""
            self._set_flag_internal(flag_name, enabled, reason, updated_by)

            # Publish change notification
            if self._enable_pubsub:
                change_msg = json.dumps({
                    "flag": flag_name,
                    "enabled": enabled,
                    "reason": reason,
                    "updated_by": updated_by,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                self._redis.publish(self.CHANNEL_FLAG_CHANGES, change_msg)

            logger.info(f"Redis flag '{flag_name}' set to {enabled}. Reason: {reason}")

        @property
        def kill_switch(self) -> bool:
            """Check if kill switch is active."""
            return self._get_flag(self.FLAG_KILL_SWITCH)

        @property
        def maintenance_mode(self) -> bool:
            """Check if system is in maintenance mode."""
            return self._get_flag(self.FLAG_MAINTENANCE)

        @property
        def paper_trading(self) -> bool:
            """Check if paper trading mode is active."""
            return self._get_flag(self.FLAG_PAPER_TRADING)

        @property
        def inference_enabled(self) -> bool:
            """Check if inference is enabled."""
            return self._get_flag(self.FLAG_INFERENCE_ENABLED)

        def set_kill_switch(
            self,
            enabled: bool,
            reason: Optional[str] = None,
            updated_by: str = "system",
        ):
            """
            Set the kill switch state.

            When activated, this will notify all services via Redis pub/sub.

            Args:
                enabled: Whether to activate the kill switch
                reason: Reason for the change
                updated_by: Who/what made the change
            """
            if enabled:
                logger.critical(f"KILL SWITCH ACTIVATED by {updated_by}. Reason: {reason}")
            else:
                logger.warning(f"Kill switch DEACTIVATED by {updated_by}. Reason: {reason}")
            self._set_flag(self.FLAG_KILL_SWITCH, enabled, reason, updated_by)

        def set_maintenance_mode(
            self,
            enabled: bool,
            reason: Optional[str] = None,
            updated_by: str = "system",
        ):
            """Set maintenance mode state."""
            self._set_flag(self.FLAG_MAINTENANCE, enabled, reason, updated_by)

        def set_paper_trading(
            self,
            enabled: bool,
            reason: Optional[str] = None,
            updated_by: str = "system",
        ):
            """Set paper trading mode state."""
            self._set_flag(self.FLAG_PAPER_TRADING, enabled, reason, updated_by)

        def set_inference_enabled(
            self,
            enabled: bool,
            reason: Optional[str] = None,
            updated_by: str = "system",
        ):
            """Set inference enabled state."""
            self._set_flag(self.FLAG_INFERENCE_ENABLED, enabled, reason, updated_by)

        def get_all_flags(self) -> Dict[str, Any]:
            """Get all trading flags with their states."""
            result = {}
            for flag in [
                self.FLAG_KILL_SWITCH,
                self.FLAG_MAINTENANCE,
                self.FLAG_PAPER_TRADING,
                self.FLAG_INFERENCE_ENABLED,
            ]:
                state = self._get_flag_state(flag)
                result[flag] = state if state else {"enabled": False, "reason": "Not set"}
            return result

        def is_trading_allowed(self) -> Tuple[bool, str]:
            """Check if trading operations are allowed."""
            if self.kill_switch:
                return False, "Kill switch is active"
            if self.maintenance_mode:
                return False, "System in maintenance mode"
            return True, "Trading allowed"

        def is_inference_allowed(self) -> Tuple[bool, str]:
            """Check if inference operations are allowed."""
            if self.kill_switch:
                return False, "Kill switch is active"
            if not self.inference_enabled:
                return False, "Inference is disabled"
            return True, "Inference allowed"

        def subscribe_to_changes(self, callback):
            """
            Subscribe to flag change notifications.

            Args:
                callback: Function to call when a flag changes.
                         Receives dict with: flag, enabled, reason, timestamp

            Example:
                def on_flag_change(change):
                    if change['flag'] == 'kill_switch' and change['enabled']:
                        shutdown_gracefully()

                flags.subscribe_to_changes(on_flag_change)
            """
            pubsub = self._redis.pubsub()
            pubsub.subscribe(**{self.CHANNEL_FLAG_CHANGES: lambda msg: callback(json.loads(msg["data"]))})
            return pubsub

        def health_check(self) -> bool:
            """Check Redis connection health."""
            try:
                return self._redis.ping()
            except Exception:
                return False

    # Singleton for Redis-backed flags
    _trading_flags_redis: Optional[TradingFlagsRedis] = None

    def get_trading_flags_redis() -> TradingFlagsRedis:
        """Get the singleton TradingFlagsRedis instance."""
        global _trading_flags_redis
        if _trading_flags_redis is None:
            _trading_flags_redis = TradingFlagsRedis()
        return _trading_flags_redis

    def reset_trading_flags_redis() -> None:
        """Reset the Redis-backed singleton instance (for testing)."""
        global _trading_flags_redis
        _trading_flags_redis = None

except ImportError:
    # redis not available - Redis features disabled
    logger.debug("redis not available - TradingFlagsRedis disabled")
    TradingFlagsRedis = None  # type: ignore
    get_trading_flags_redis = None  # type: ignore
    reset_trading_flags_redis = None  # type: ignore
