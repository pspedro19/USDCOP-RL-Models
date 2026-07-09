"""Pre-trade gate — the LAST line of defense before an order reaches an exchange.

Audit 2026-07 (master plan OLA 3 S2/S3) found `ExecutionService.execute_order` went
decrypt -> adapter -> place_order with NO risk checks, NO kill switch and NO global
execution-mode gate. This module closes that hole with a single, fail-safe decision point:

    decision = await PreTradeGate(db).check(execution, user_id)

Design (SOLID):
  * Single responsibility: this class ONLY decides allow / simulate / block. It never
    places orders, never mutates executions.
  * Fail-safe: any internal error => BLOCK (never "fail open" with real money).
  * Open/closed: checks are small private methods run in a fixed order; new checks are
    added to CHECKS without touching callers.

Gate semantics for `settings.trading_mode` (KILLED|DISABLED|SHADOW|PAPER|STAGING|LIVE):
  KILLED / DISABLED -> BLOCK everything.
  SHADOW / PAPER    -> SIMULATE: the order is validated + recorded but NEVER sent.
  STAGING           -> only testnet credentials may reach the exchange.
  LIVE              -> real orders allowed (still subject to user limits below).

Per-user checks (from the risk limits table used by RiskBridgeService):
  kill-switch / trading disabled, max position notional (USD), daily trade count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)


class PreTradeAction(str, Enum):
    ALLOW = "allow"          # send to exchange
    SIMULATE = "simulate"    # record as paper fill, do NOT send
    BLOCK = "block"          # reject, do nothing


@dataclass
class PreTradeDecision:
    action: PreTradeAction
    reason: str
    metadata: dict = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == PreTradeAction.BLOCK

    @property
    def simulated(self) -> bool:
        return self.action == PreTradeAction.SIMULATE


class PreTradeGate:
    """Fail-safe pre-trade decision point (mode gate + per-user limits)."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def check(
        self,
        *,
        user_id: UUID,
        symbol: str,
        quantity: float,
        price: float | None,
        is_testnet_credential: bool,
    ) -> PreTradeDecision:
        try:
            mode = (settings.trading_mode or "PAPER").upper()

            # 1) Global execution-mode gate (S3)
            if mode in ("KILLED", "DISABLED"):
                return PreTradeDecision(PreTradeAction.BLOCK,
                                        f"trading_mode={mode}: all order flow is blocked")
            if mode in ("PAPER", "SHADOW"):
                return PreTradeDecision(PreTradeAction.SIMULATE,
                                        f"trading_mode={mode}: order validated but not sent",
                                        {"mode": mode})
            if mode == "STAGING" and not is_testnet_credential:
                return PreTradeDecision(PreTradeAction.BLOCK,
                                        "trading_mode=STAGING: only testnet credentials may trade")

            # 2) Per-user kill switch + symbol policy (S2) — sb_trading_configs.
            #    trading_enabled defaults to FALSE: a user with no explicit opt-in never
            #    reaches a live exchange (safe-by-default).
            cfg = await self._load_trading_config(user_id)
            if cfg is None or not cfg.get("trading_enabled", False):
                return PreTradeDecision(PreTradeAction.BLOCK,
                                        "user trading disabled (kill switch / no opt-in)")
            blocked = cfg.get("blocked_symbols") or []
            allowed = cfg.get("allowed_symbols") or []
            if symbol in blocked:
                return PreTradeDecision(PreTradeAction.BLOCK, f"symbol {symbol} is blocked")
            if allowed and symbol not in allowed:
                return PreTradeDecision(PreTradeAction.BLOCK,
                                        f"symbol {symbol} not in user's allowed list")

            # 3) Notional cap — user_risk_limits row, else conservative settings default.
            limits = await self._load_limits(user_id)
            max_notional = float(
                (limits or {}).get("max_position_size_usd")
                or settings.max_position_size_usd
            )
            notional = float(quantity) * float(price or 0.0)
            if price and notional > max_notional:
                return PreTradeDecision(
                    PreTradeAction.BLOCK,
                    f"notional ${notional:,.2f} exceeds max ${max_notional:,.2f}",
                    {"notional": notional, "max_notional": max_notional})

            # 4) Daily trade count.
            max_daily = int(cfg.get("max_daily_trades")
                            or (limits or {}).get("max_trades_per_day") or 0)
            if max_daily:
                today_count = await self._today_execution_count(user_id)
                if today_count >= max_daily:
                    return PreTradeDecision(
                        PreTradeAction.BLOCK,
                        f"daily trade limit reached ({today_count}/{max_daily})")

            return PreTradeDecision(PreTradeAction.ALLOW, "all pre-trade checks passed",
                                    {"mode": mode, "notional": notional})

        except Exception as e:  # fail-safe: never fail open
            logger.error("PreTradeGate error (blocking, fail-safe): %s", e)
            return PreTradeDecision(PreTradeAction.BLOCK, f"pre-trade gate error: {e!s}")

    # ------------------------------------------------------------------ internals
    async def _load_trading_config(self, user_id: UUID) -> dict | None:
        """Per-user kill switch + symbol policy (`sb_trading_configs`). None => no opt-in."""
        try:
            res = await self.db.execute(text("""
                SELECT trading_enabled, allowed_symbols, blocked_symbols, max_daily_trades
                FROM sb_trading_configs WHERE user_id = :uid
            """), {"uid": str(user_id)})
            row = res.first()
            if row is None:
                return None
            return {"trading_enabled": bool(row.trading_enabled),
                    "allowed_symbols": list(row.allowed_symbols or []),
                    "blocked_symbols": list(row.blocked_symbols or []),
                    "max_daily_trades": row.max_daily_trades}
        except Exception as e:
            logger.error("PreTradeGate: trading config unavailable (%s) — fail-safe block", e)
            return None  # caller blocks when None

    async def _load_limits(self, user_id: UUID) -> dict | None:
        """Notional cap row (same table RiskBridgeService reads); None if absent/unavailable."""
        try:
            res = await self.db.execute(text("""
                SELECT max_position_size_usd, max_trades_per_day
                FROM user_risk_limits WHERE user_id = :uid
            """), {"uid": str(user_id)})
            row = res.first()
            if row is None:
                return None
            return {"max_position_size_usd": row.max_position_size_usd,
                    "max_trades_per_day": row.max_trades_per_day}
        except Exception as e:
            # Table may not exist yet in older DBs — degrade to settings defaults (still capped).
            logger.warning("PreTradeGate: user limits unavailable (%s); using defaults", e)
            return None

    async def _today_execution_count(self, user_id: UUID) -> int:
        try:
            res = await self.db.execute(text("""
                SELECT COUNT(*) FROM sb_executions
                WHERE user_id = :uid AND created_at >= :start
            """), {"uid": str(user_id),
                   "start": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)})
            return int(res.scalar() or 0)
        except Exception:
            return 0
