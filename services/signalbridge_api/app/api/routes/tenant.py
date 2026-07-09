"""Multi-tenant SignalBridge routes (CTR-RBAC-001 R5 + OLA-3 S4).

  /tenant/me/keys    POST register own exchange keys (VERIFIES anti-withdraw; encrypted)
                     GET  list own keys (masked)
  /tenant/me/limits  GET/PUT own risk limits (users may only LOWER system ceilings)
  /tenant/me/kill    POST own kill switch on/off
  /tenant/system/kill POST global kill (admin only) — dominates every user
  fan_out_signal()   S4: map a published signal onto every eligible user as a PENDING
                     execution (each then passes PreTradeGate at execute time).

Hard rules (rbac.md §5-7): keys scoped to user_id, never plaintext, withdraw-enabled keys
REJECTED at registration; user kill stops only their flow; global kill dominates.
"""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters import get_exchange_adapter
from app.contracts.exchange import SupportedExchange
from app.core.database import get_db
from app.middleware.auth import get_current_active_user
from app.models import User
from app.services.vault import vault_service

router = APIRouter(prefix="/tenant", tags=["Multi-tenant"])

# System ceilings (entitlement defaults; users may only go DOWN from these).
CEILING_NOTIONAL_USD = 5000.0
CEILING_MAX_OPEN = 2
CEILING_DAILY_LOSS_PCT = 3.0


class KeyCreate(BaseModel):
    exchange: SupportedExchange
    api_key: str = Field(min_length=8)
    api_secret: str = Field(min_length=8)


class LimitsUpdate(BaseModel):
    max_notional_usd: float | None = None
    max_daily_loss_pct: float | None = None
    max_open_positions: int | None = None


class FanOutRequest(BaseModel):
    signal_id: str
    symbol: str
    side: str
    notional_usd: float = Field(gt=0)
    price: float | None = None


async def _audit(db: AsyncSession, user_id, action: str, detail: dict) -> None:
    try:
        await db.execute(text(
            "INSERT INTO audit_log (user_id, action, object_type, detail) "
            "VALUES (:u, :a, 'signalbridge', CAST(:d AS jsonb))"),
            {"u": str(user_id), "a": action, "d": __import__("json").dumps(detail)})
        await db.commit()
    except Exception:
        await db.rollback()


@router.post("/me/keys", status_code=201)
async def register_own_key(data: KeyCreate,
                           user: User = Depends(get_current_active_user),
                           db: AsyncSession = Depends(get_db)):
    """Register the user's OWN exchange keys. Withdraw-enabled keys are rejected."""
    adapter = get_exchange_adapter(exchange=data.exchange, api_key=data.api_key,
                                   api_secret=data.api_secret, testnet=False)
    try:
        if not await adapter.validate_credentials():
            raise HTTPException(400, "credenciales inválidas (el exchange las rechazó)")
        # Anti-withdraw check: attempt to read withdraw permission via ccxt account info.
        # Exchanges expose this differently; fail CLOSED (reject) when we cannot prove
        # the key has NO withdraw permission.
        can_withdraw = None
        try:
            info = await adapter._exchange.fetch_status() if False else None  # placeholder branch
        except Exception:
            info = None
        try:
            acct = await adapter._exchange.private_get_account() if hasattr(
                adapter._exchange, "private_get_account") else None
            if isinstance(acct, dict) and "canWithdraw" in acct:
                can_withdraw = bool(acct["canWithdraw"])
        except Exception:
            can_withdraw = None
        if can_withdraw is True:
            raise HTTPException(400, "la llave tiene permiso de RETIRO — crea una llave "
                                     "solo-trade (jamás custodiamos retiros)")
        if can_withdraw is None:
            # cannot verify -> mark pending; NOT usable until admin/exchange confirms
            status = "pending"
        else:
            status = "verified"
    finally:
        await adapter.close()

    await db.execute(text("""
        INSERT INTO user_exchange_keys (user_id, exchange, api_key_enc, api_secret_enc, status)
        VALUES (:u, :e, :k, :s, :st)
        ON CONFLICT (user_id, exchange) DO UPDATE SET
            api_key_enc=:k, api_secret_enc=:s, status=:st, last_verified_at=NOW()
    """), {"u": str(user.id), "e": data.exchange.value,
           "k": vault_service.encrypt(data.api_key), "s": vault_service.encrypt(data.api_secret),
           "st": status})
    await db.commit()
    await _audit(db, user.id, "key_add", {"exchange": data.exchange.value, "status": status})
    return {"exchange": data.exchange.value, "status": status,
            "note": "pending = no pudimos verificar ausencia de permiso de retiro; "
                    "no se opera hasta verificarla" if status == "pending" else "ok"}


@router.get("/me/keys")
async def list_own_keys(user: User = Depends(get_current_active_user),
                        db: AsyncSession = Depends(get_db)):
    res = await db.execute(text(
        "SELECT exchange, status, last_verified_at, created_at FROM user_exchange_keys "
        "WHERE user_id = :u"), {"u": str(user.id)})
    return [dict(r._mapping) for r in res]


@router.get("/me/limits")
async def get_own_limits(user: User = Depends(get_current_active_user),
                         db: AsyncSession = Depends(get_db)):
    res = await db.execute(text(
        "SELECT * FROM user_risk_limits_v2 WHERE user_id = :u"), {"u": str(user.id)})
    row = res.first()
    return dict(row._mapping) if row else {
        "max_notional_usd": 0, "mode": "paper", "kill_switch": False,
        "note": "sin límites configurados — ejecución deshabilitada (safe-by-default)"}


@router.put("/me/limits")
async def update_own_limits(data: LimitsUpdate,
                            user: User = Depends(get_current_active_user),
                            db: AsyncSession = Depends(get_db)):
    """Users may only LOWER the system ceilings, never raise them (rbac.md §3)."""
    notional = min(data.max_notional_usd or CEILING_NOTIONAL_USD, CEILING_NOTIONAL_USD)
    daily = min(data.max_daily_loss_pct or CEILING_DAILY_LOSS_PCT, CEILING_DAILY_LOSS_PCT)
    open_pos = min(data.max_open_positions or CEILING_MAX_OPEN, CEILING_MAX_OPEN)
    await db.execute(text("""
        INSERT INTO user_risk_limits_v2 (user_id, max_notional_usd, max_daily_loss_pct,
                                         max_open_positions, updated_at)
        VALUES (:u, :n, :d, :o, NOW())
        ON CONFLICT (user_id) DO UPDATE SET max_notional_usd=:n, max_daily_loss_pct=:d,
            max_open_positions=:o, updated_at=NOW()
    """), {"u": str(user.id), "n": notional, "d": daily, "o": open_pos})
    await db.commit()
    await _audit(db, user.id, "limits_update",
                 {"notional": notional, "daily": daily, "open": open_pos})
    return {"max_notional_usd": notional, "max_daily_loss_pct": daily,
            "max_open_positions": open_pos, "ceilings_applied": True}


@router.post("/me/kill")
async def own_kill_switch(enable: bool = True,
                          user: User = Depends(get_current_active_user),
                          db: AsyncSession = Depends(get_db)):
    """The user's OWN kill switch — stops only their executions."""
    await db.execute(text("""
        INSERT INTO user_risk_limits_v2 (user_id, kill_switch, updated_at)
        VALUES (:u, :k, NOW())
        ON CONFLICT (user_id) DO UPDATE SET kill_switch=:k, updated_at=NOW()
    """), {"u": str(user.id), "k": enable})
    await db.commit()
    await _audit(db, user.id, "kill_user", {"enabled": enable})
    return {"kill_switch": enable}


PAPER_WEEKS_REQUIRED = 4  # mirrors PLAN_DEFAULTS.execution.paper_weeks_required (rbac.contract.ts)


@router.post("/me/accept-risk")
async def accept_risk(user: User = Depends(get_current_active_user),
                      db: AsyncSession = Depends(get_db)):
    """Record the user's risk-disclosure acceptance (prerequisite for go-live)."""
    await db.execute(text("""
        INSERT INTO user_risk_limits_v2 (user_id, risk_accepted_at, updated_at)
        VALUES (:u, NOW(), NOW())
        ON CONFLICT (user_id) DO UPDATE SET risk_accepted_at=NOW(), updated_at=NOW()
    """), {"u": str(user.id)})
    await db.commit()
    await _audit(db, user.id, "risk_accepted", {})
    return {"risk_accepted": True}


@router.post("/me/go-live")
async def go_live(user: User = Depends(get_current_active_user),
                  db: AsyncSession = Depends(get_db)):
    """Paper→LIVE transition — HARD-GATED (rbac.md §6, paper-first):

    1. risk disclosure accepted (risk_accepted_at set),
    2. ≥ PAPER_WEEKS_REQUIRED weeks of REAL paper history (oldest user_executions row),
    3. own kill switch off.
    Any failure → 403 with the concrete reason + audit. Success stamps mode='live' +
    live_enabled_at and audits. NOTE: the global SFC legal gate still applies — while
    trading_mode=PAPER globally, live mode only changes the user's record, no real orders.
    """
    res = await db.execute(text(
        "SELECT risk_accepted_at, kill_switch, mode FROM user_risk_limits_v2 WHERE user_id=:u"),
        {"u": str(user.id)})
    row = res.first()
    reasons = []
    if not row or row.risk_accepted_at is None:
        reasons.append("risk disclosure no aceptado (POST /me/accept-risk primero)")
    if row and row.kill_switch:
        reasons.append("kill switch activo")
    age = await db.execute(text(
        "SELECT EXTRACT(EPOCH FROM (NOW() - MIN(created_at)))/604800.0 AS weeks "
        "FROM user_executions WHERE user_id=:u"), {"u": str(user.id)})
    weeks = (age.first() or [None])[0]
    if weeks is None or float(weeks) < PAPER_WEEKS_REQUIRED:
        have = 0.0 if weeks is None else round(float(weeks), 1)
        reasons.append(f"paper insuficiente: {have}/{PAPER_WEEKS_REQUIRED} semanas de historial")
    if reasons:
        await _audit(db, user.id, "go_live_denied", {"reasons": reasons})
        raise HTTPException(403, "; ".join(reasons))
    await db.execute(text(
        "UPDATE user_risk_limits_v2 SET mode='live', live_enabled_at=NOW(), updated_at=NOW() "
        "WHERE user_id=:u"), {"u": str(user.id)})
    await db.commit()
    await _audit(db, user.id, "go_live", {"paper_weeks": round(float(weeks), 1)})
    return {"mode": "live", "paper_weeks": round(float(weeks), 1)}


@router.post("/system/kill")
async def global_kill(enable: bool = True,
                      user: User = Depends(get_current_active_user),
                      db: AsyncSession = Depends(get_db)):
    """GLOBAL kill (admin only) — dominates every user (rbac.md §7)."""
    if getattr(user, "role", "") != "admin":
        await _audit(db, user.id, "kill_global_denied", {})
        raise HTTPException(403, "solo admin")
    await db.execute(text(
        "UPDATE user_risk_limits_v2 SET kill_switch = :k, updated_at = NOW()"), {"k": enable})
    await db.commit()
    await _audit(db, user.id, "kill_global", {"enabled": enable})
    return {"global_kill": enable}


# ── S4: signal fan-out ────────────────────────────────────────────────────────────
async def fan_out_signal(db: AsyncSession, *, signal_id: str, symbol: str, side: str,
                         notional_usd: float, price: float | None) -> dict:
    """Map one published signal onto every ELIGIBLE user as an append-only user_execution
    (mode from their limits row; each real order still passes PreTradeGate at execution).

    Eligible = has verified key + limits row + kill_switch off + notional within their cap.
    """
    res = await db.execute(text("""
        SELECT l.user_id, l.max_notional_usd, l.mode
        FROM user_risk_limits_v2 l
        JOIN user_exchange_keys k ON k.user_id = l.user_id AND k.status = 'verified'
        WHERE l.kill_switch = FALSE AND l.max_notional_usd > 0
    """))
    fanned = 0
    for row in res:
        alloc = min(notional_usd, float(row.max_notional_usd))
        qty = (alloc / price) if price else None
        await db.execute(text("""
            INSERT INTO user_executions (user_id, signal_id, symbol, side, qty, px, status, mode)
            VALUES (:u, :sig, :sym, :sd, :q, :p, 'pending', :m)
        """), {"u": str(row.user_id), "sig": signal_id, "sym": symbol, "sd": side,
               "q": qty, "p": price, "m": row.mode})
        fanned += 1
    await db.commit()
    return {"signal_id": signal_id, "fanned_out_to": fanned}


@router.post("/system/fan-out")
async def system_fan_out(data: FanOutRequest,
                         user: User = Depends(get_current_active_user),
                         db: AsyncSession = Depends(get_db)):
    """Publish one signal to every ELIGIBLE user as a PENDING execution (admin only, S4).

    This is the controlled, auditable entry point — a signal is fanned out by an operator/system
    action, NOT by every user-created signal. Eligibility + per-user cap live in fan_out_signal();
    each PENDING row still passes PreTradeGate before any real order (paper-first, fail-safe).
    """
    if getattr(user, "role", "") != "admin":
        await _audit(db, user.id, "fanout_denied", {"signal_id": data.signal_id})
        raise HTTPException(403, "solo admin puede publicar señales al sistema")
    result = await fan_out_signal(
        db, signal_id=data.signal_id, symbol=data.symbol, side=data.side,
        notional_usd=data.notional_usd, price=data.price)
    await _audit(db, user.id, "fanout", result)
    return result
