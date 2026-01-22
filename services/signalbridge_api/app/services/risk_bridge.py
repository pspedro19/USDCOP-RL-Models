"""
Risk Bridge Service
===================

Bridge between SignalBridge and the existing RiskEnforcer from src/trading/.
Adds user-specific risk limits from the database.

Design Pattern: Adapter + Decorator
- Adapts SignalBridge contracts to RiskEnforcer interface
- Decorates RiskEnforcer with user-specific limits from DB

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.contracts.signal_bridge import (
    RiskCheckResult,
    RiskDecision,
    RiskReason,
    InferenceSignalCreate,
    UserRiskLimits,
    BridgeEventType,
)
from app.contracts.execution import OrderSide

# Import from src/trading (add to Python path or use relative import based on setup)
try:
    from src.trading.risk_enforcer import (
        RiskEnforcer,
        RiskLimits,
        RiskCheckResult as CoreRiskCheckResult,
        RiskDecision as CoreRiskDecision,
        RiskReason as CoreRiskReason,
    )
    RISK_ENFORCER_AVAILABLE = True
except ImportError:
    RISK_ENFORCER_AVAILABLE = False
    RiskEnforcer = None
    RiskLimits = None

logger = logging.getLogger(__name__)


class RiskBridgeService:
    """
    Bridge service that connects SignalBridge to RiskEnforcer.

    Responsibilities:
    1. Load user-specific risk limits from database
    2. Convert SignalBridge contracts to RiskEnforcer format
    3. Merge user limits with default limits
    4. Record trade results back to RiskEnforcer

    Usage:
        bridge = RiskBridgeService(db_session)
        result = await bridge.validate_execution(signal, quantity, user_id)

        if result.is_allowed:
            # Execute trade
            pass

        # After execution
        await bridge.record_trade_result(pnl_pct, signal, user_id)
    """

    # Cache for RiskEnforcer instances per user
    _enforcers: Dict[UUID, Any] = {}

    def __init__(
        self,
        db_session: AsyncSession,
        default_limits: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RiskBridgeService.

        Args:
            db_session: Database session for user limits
            default_limits: Default risk limits to use
        """
        self.db_session = db_session
        self.default_limits = default_limits or {}

        if not RISK_ENFORCER_AVAILABLE:
            logger.warning(
                "RiskEnforcer not available - using fallback validation. "
                "Ensure src/trading is in the Python path."
            )

    async def get_user_limits(self, user_id: UUID) -> Optional[UserRiskLimits]:
        """
        Get user-specific risk limits from database.

        Args:
            user_id: User identifier

        Returns:
            UserRiskLimits if found, None otherwise
        """
        from sqlalchemy import text

        try:
            # Query user_risk_limits table using proper parameterized query
            result = await self.db_session.execute(
                text("""
                    SELECT id, user_id, max_daily_loss_pct, max_trades_per_day,
                           max_position_size_usd, cooldown_minutes, enable_short,
                           created_at, updated_at
                    FROM user_risk_limits
                    WHERE user_id = :user_id
                """),
                {"user_id": str(user_id)}
            )
            row = result.first()

            if row:
                return UserRiskLimits(
                    id=row.id,
                    user_id=UUID(row.user_id) if isinstance(row.user_id, str) else row.user_id,
                    max_daily_loss_pct=float(row.max_daily_loss_pct),
                    max_trades_per_day=row.max_trades_per_day,
                    max_position_size_usd=float(row.max_position_size_usd),
                    cooldown_minutes=row.cooldown_minutes,
                    enable_short=row.enable_short,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
            return None

        except Exception as e:
            logger.error(f"Error fetching user limits for {user_id}: {e}")
            return None

    async def create_or_update_user_limits(
        self,
        user_id: UUID,
        max_daily_loss_pct: Optional[float] = None,
        max_trades_per_day: Optional[int] = None,
        max_position_size_usd: Optional[float] = None,
        cooldown_minutes: Optional[int] = None,
        enable_short: Optional[bool] = None,
    ) -> UserRiskLimits:
        """
        Create or update user risk limits in database.

        Args:
            user_id: User identifier
            max_daily_loss_pct: Max daily loss percentage
            max_trades_per_day: Max trades per day
            max_position_size_usd: Max position size in USD
            cooldown_minutes: Cooldown between trades
            enable_short: Whether short trading is enabled

        Returns:
            Updated UserRiskLimits
        """
        from sqlalchemy import text
        from uuid import uuid4
        from app.core.config import settings

        # Get existing limits or use defaults
        existing = await self.get_user_limits(user_id)

        final_values = {
            "max_daily_loss_pct": max_daily_loss_pct if max_daily_loss_pct is not None else (
                existing.max_daily_loss_pct if existing else settings.default_max_daily_loss_pct
            ),
            "max_trades_per_day": max_trades_per_day if max_trades_per_day is not None else (
                existing.max_trades_per_day if existing else settings.default_max_trades_per_day
            ),
            "max_position_size_usd": max_position_size_usd if max_position_size_usd is not None else (
                existing.max_position_size_usd if existing else settings.max_position_size_usd
            ),
            "cooldown_minutes": cooldown_minutes if cooldown_minutes is not None else (
                existing.cooldown_minutes if existing else settings.default_cooldown_minutes
            ),
            "enable_short": enable_short if enable_short is not None else (
                existing.enable_short if existing else False
            ),
        }

        if existing:
            # Update existing
            await self.db_session.execute(
                text("""
                    UPDATE user_risk_limits
                    SET max_daily_loss_pct = :max_daily_loss_pct,
                        max_trades_per_day = :max_trades_per_day,
                        max_position_size_usd = :max_position_size_usd,
                        cooldown_minutes = :cooldown_minutes,
                        enable_short = :enable_short,
                        updated_at = NOW()
                    WHERE user_id = :user_id
                """),
                {"user_id": str(user_id), **final_values}
            )
            await self.db_session.commit()

            # Reset enforcer to pick up new limits
            self.reset_user_enforcer(user_id)

            return UserRiskLimits(
                id=existing.id,
                user_id=user_id,
                **final_values,
            )
        else:
            # Insert new
            new_id = uuid4()
            await self.db_session.execute(
                text("""
                    INSERT INTO user_risk_limits (
                        id, user_id, max_daily_loss_pct, max_trades_per_day,
                        max_position_size_usd, cooldown_minutes, enable_short,
                        created_at, updated_at
                    ) VALUES (
                        :id, :user_id, :max_daily_loss_pct, :max_trades_per_day,
                        :max_position_size_usd, :cooldown_minutes, :enable_short,
                        NOW(), NOW()
                    )
                """),
                {"id": str(new_id), "user_id": str(user_id), **final_values}
            )
            await self.db_session.commit()

            return UserRiskLimits(
                id=new_id,
                user_id=user_id,
                **final_values,
            )

    def _get_or_create_enforcer(
        self,
        user_id: UUID,
        user_limits: Optional[UserRiskLimits] = None,
    ) -> Any:
        """
        Get or create a RiskEnforcer instance for a user.

        Args:
            user_id: User identifier
            user_limits: User-specific limits (optional)

        Returns:
            RiskEnforcer instance
        """
        if not RISK_ENFORCER_AVAILABLE:
            return None

        # Check cache
        if user_id in self._enforcers:
            return self._enforcers[user_id]

        # Build limits from user settings + defaults
        limits_kwargs = dict(self.default_limits)

        if user_limits:
            limits_kwargs.update({
                "max_daily_loss_pct": user_limits.max_daily_loss_pct,
                "max_trades_per_day": user_limits.max_trades_per_day,
                "max_position_size": user_limits.max_position_size_usd,
                "cooldown_minutes": user_limits.cooldown_minutes,
                "enable_short": user_limits.enable_short,
            })

        # Create RiskLimits
        limits = RiskLimits(**limits_kwargs)

        # Create enforcer
        enforcer = RiskEnforcer(limits=limits)
        self._enforcers[user_id] = enforcer

        logger.info(f"Created RiskEnforcer for user {user_id} with limits: {limits_kwargs}")
        return enforcer

    async def validate_execution(
        self,
        signal: InferenceSignalCreate,
        quantity: float,
        user_id: UUID,
        current_price: Optional[float] = None,
    ) -> RiskCheckResult:
        """
        Validate a signal against risk rules.

        Args:
            signal: Inference signal to validate
            quantity: Proposed position size
            user_id: User identifier
            current_price: Current market price (optional)

        Returns:
            RiskCheckResult with decision and details
        """
        start_time = datetime.utcnow()

        # Load user limits
        user_limits = await self.get_user_limits(user_id)

        # Get or create enforcer
        enforcer = self._get_or_create_enforcer(user_id, user_limits)

        # Convert signal action to string
        action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
        signal_str = action_map.get(signal.action, "HOLD")

        # If enforcer not available, use fallback validation
        if enforcer is None:
            return self._fallback_validation(signal, quantity, user_limits)

        # Run risk check
        try:
            core_result = enforcer.check_signal(
                signal=signal_str,
                size=quantity,
                price=current_price or 1.0,
                confidence=signal.confidence,
            )

            # Convert core result to bridge contract
            decision = RiskDecision(core_result.decision.value)
            reason = RiskReason(core_result.reason.value)

            return RiskCheckResult(
                decision=decision,
                reason=reason,
                message=core_result.message,
                adjusted_size=core_result.adjusted_size,
                metadata={
                    "user_id": str(user_id),
                    "signal_id": str(signal.signal_id),
                    "model_id": signal.model_id,
                    "confidence": signal.confidence,
                    "original_size": quantity,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    **core_result.metadata,
                },
            )

        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.KILL_SWITCH,
                message=f"Risk validation error: {str(e)}",
                metadata={"error": str(e)},
            )

    def _fallback_validation(
        self,
        signal: InferenceSignalCreate,
        quantity: float,
        user_limits: Optional[UserRiskLimits],
    ) -> RiskCheckResult:
        """
        Fallback validation when RiskEnforcer is not available.

        Implements basic checks:
        1. Confidence threshold
        2. Position size limit
        3. Short trading allowed
        """
        warnings = []

        # Check confidence
        min_confidence = 0.5
        if signal.confidence < min_confidence:
            return RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=RiskReason.LOW_CONFIDENCE,
                message=f"Confidence {signal.confidence:.2f} below threshold {min_confidence}",
                metadata={"confidence": signal.confidence, "threshold": min_confidence},
            )

        # Check position size
        max_size = 1000.0
        if user_limits:
            max_size = user_limits.max_position_size_usd

        if quantity > max_size:
            return RiskCheckResult(
                decision=RiskDecision.REDUCE,
                reason=RiskReason.MAX_POSITION,
                message=f"Position size {quantity} exceeds limit {max_size}",
                adjusted_size=max_size,
                metadata={"requested": quantity, "limit": max_size},
            )

        # Check short trading
        if signal.action == 0:  # SELL/SHORT
            enable_short = user_limits.enable_short if user_limits else False
            if not enable_short:
                return RiskCheckResult(
                    decision=RiskDecision.BLOCK,
                    reason=RiskReason.SHORT_DISABLED,
                    message="Short trading is disabled for this user",
                )

        # All checks passed
        return RiskCheckResult(
            decision=RiskDecision.ALLOW,
            reason=RiskReason.APPROVED,
            message="All risk checks passed (fallback validation)",
            metadata={"fallback": True},
        )

    async def record_trade_result(
        self,
        pnl_pct: float,
        signal: InferenceSignalCreate,
        user_id: UUID,
    ) -> None:
        """
        Record trade result to update risk state.

        Args:
            pnl_pct: Trade P&L as percentage
            signal: Original signal
            user_id: User identifier
        """
        if user_id not in self._enforcers:
            logger.warning(f"No enforcer for user {user_id} - cannot record trade")
            return

        enforcer = self._enforcers[user_id]
        if enforcer is None:
            return

        action_map = {0: "SHORT", 1: "HOLD", 2: "LONG"}
        signal_str = action_map.get(signal.action, "HOLD")

        try:
            enforcer.record_trade(pnl_pct=pnl_pct, signal=signal_str)
            logger.info(
                f"Recorded trade result for user {user_id}: "
                f"pnl={pnl_pct:.4f}%, signal={signal_str}"
            )
        except Exception as e:
            logger.error(f"Error recording trade result: {e}")

    async def get_user_risk_status(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get current risk status for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with risk status details
        """
        if user_id not in self._enforcers:
            # No enforcer yet - return defaults
            return {
                "user_id": str(user_id),
                "is_trading_allowed": True,
                "kill_switch_active": False,
                "daily_blocked": False,
                "trade_count_today": 0,
                "daily_pnl_pct": 0.0,
                "message": "No trading activity yet",
            }

        enforcer = self._enforcers[user_id]
        if enforcer is None:
            return {"error": "RiskEnforcer not available"}

        try:
            status = enforcer.get_status()
            status["user_id"] = str(user_id)
            return status
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {"error": str(e)}

    def reset_user_enforcer(self, user_id: UUID) -> bool:
        """
        Reset the enforcer for a user (e.g., on new day or manual reset).

        Args:
            user_id: User identifier

        Returns:
            True if reset, False if no enforcer found
        """
        if user_id in self._enforcers:
            del self._enforcers[user_id]
            logger.info(f"Reset RiskEnforcer for user {user_id}")
            return True
        return False

    def clear_all_enforcers(self) -> int:
        """
        Clear all cached enforcers (e.g., on service restart).

        Returns:
            Number of enforcers cleared
        """
        count = len(self._enforcers)
        self._enforcers.clear()
        logger.info(f"Cleared {count} RiskEnforcer instances")
        return count
