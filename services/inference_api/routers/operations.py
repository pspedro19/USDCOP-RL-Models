"""
Operations Router - Kill Switch and Emergency Controls
=======================================================
P0 Critical: Emergency trading controls accessible via API

Endpoints:
- POST /operations/kill-switch - Stop all trading immediately
- POST /operations/resume - Resume trading after kill
- GET /operations/status - Get current operational status
- POST /operations/pause - Soft pause (finish current, no new)
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional
import logging

router = APIRouter(prefix="/operations", tags=["operations"])
logger = logging.getLogger(__name__)

# Global state (should be Redis-backed in production)
_kill_switch_state = {
    "active": False,
    "activated_at": None,
    "activated_by": None,
    "reason": None,
    "mode": "normal"  # normal, paused, killed
}


class KillSwitchRequest(BaseModel):
    """Request model for activating kill switch."""
    reason: str
    activated_by: str = "dashboard"
    close_positions: bool = True
    notify_team: bool = True


class KillSwitchResponse(BaseModel):
    """Response model for kill switch activation."""
    success: bool
    mode: str
    activated_at: Optional[str]
    message: str
    positions_closed: int = 0


class OperationsStatusResponse(BaseModel):
    """Response model for operations status."""
    mode: str
    kill_switch_active: bool
    activated_at: Optional[str]
    activated_by: Optional[str]
    reason: Optional[str]
    timestamp: str


class PauseRequest(BaseModel):
    """Request model for pausing trading."""
    paused_by: str = "dashboard"
    reason: str = "Manual pause"


class ResumeRequest(BaseModel):
    """Request model for resuming trading."""
    resumed_by: str = "dashboard"
    confirmation_code: str


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def activate_kill_switch(
    request: KillSwitchRequest,
    background_tasks: BackgroundTasks
):
    """
    EMERGENCY: Stop all trading immediately.

    Actions:
    1. Set global kill switch flag
    2. Optionally close all open positions
    3. Notify team via configured channels
    4. Log to audit trail
    """
    global _kill_switch_state

    _kill_switch_state = {
        "active": True,
        "activated_at": datetime.now(timezone.utc).isoformat(),
        "activated_by": request.activated_by,
        "reason": request.reason,
        "mode": "killed"
    }

    positions_closed = 0

    # Close positions if requested
    if request.close_positions:
        positions_closed = await _close_all_positions()

    # Send notifications
    if request.notify_team:
        background_tasks.add_task(
            _send_kill_switch_notification,
            request.reason,
            request.activated_by
        )

    # Log to audit
    await _log_kill_switch_event(request)

    logger.critical(
        f"KILL SWITCH ACTIVATED by {request.activated_by}: {request.reason}"
    )

    return KillSwitchResponse(
        success=True,
        mode="killed",
        activated_at=_kill_switch_state["activated_at"],
        message=f"Trading stopped. {positions_closed} positions closed.",
        positions_closed=positions_closed
    )


@router.post("/resume")
async def resume_trading(request: ResumeRequest):
    """Resume trading after kill switch. Requires confirmation."""
    global _kill_switch_state

    if not _kill_switch_state["active"]:
        return {"success": True, "mode": "normal", "message": "Trading already active"}

    # Require confirmation for safety
    if request.confirmation_code != "CONFIRM_RESUME":
        raise HTTPException(
            status_code=400,
            detail="Must provide confirmation_code='CONFIRM_RESUME'"
        )

    previous_reason = _kill_switch_state["reason"]
    previous_activated_by = _kill_switch_state["activated_by"]

    _kill_switch_state = {
        "active": False,
        "activated_at": None,
        "activated_by": None,
        "reason": None,
        "mode": "normal"
    }

    logger.info(
        f"Trading RESUMED by {request.resumed_by}. "
        f"Previously killed by {previous_activated_by}: {previous_reason}"
    )

    return {
        "success": True,
        "mode": "normal",
        "message": "Trading resumed successfully",
        "resumed_by": request.resumed_by,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/status", response_model=OperationsStatusResponse)
async def get_operations_status():
    """Get current operational status."""
    return OperationsStatusResponse(
        mode=_kill_switch_state["mode"],
        kill_switch_active=_kill_switch_state["active"],
        activated_at=_kill_switch_state["activated_at"],
        activated_by=_kill_switch_state["activated_by"],
        reason=_kill_switch_state["reason"],
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.post("/pause")
async def pause_trading(request: PauseRequest):
    """Soft pause - finish current trade, no new trades."""
    global _kill_switch_state

    _kill_switch_state["mode"] = "paused"
    _kill_switch_state["reason"] = request.reason

    logger.warning(f"Trading PAUSED by {request.paused_by}: {request.reason}")

    return {
        "success": True,
        "mode": "paused",
        "message": "Trading paused",
        "paused_by": request.paused_by,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


async def _close_all_positions() -> int:
    """
    Close all open positions.

    Returns:
        Number of positions closed

    Note: Implementation depends on broker integration.
    This is a placeholder that should be connected to the
    actual trading system.
    """
    # TODO: Integrate with broker API to close positions
    # For now, return 0 as placeholder
    logger.info("Close all positions requested (placeholder implementation)")
    return 0


async def _send_kill_switch_notification(reason: str, activated_by: str):
    """
    Send notification to team about kill switch activation.

    This should integrate with Slack/email notification system.
    """
    try:
        # Try to use the Slack client if available
        from src.shared.notifications.slack_client import get_slack_client

        client = get_slack_client()
        await client.notify_kill_switch(reason, activated_by)
        logger.info(f"Kill switch notification sent to Slack")
    except ImportError:
        logger.warning("Slack client not available, skipping notification")
    except Exception as e:
        logger.error(f"Failed to send kill switch notification: {e}")


async def _log_kill_switch_event(request: KillSwitchRequest):
    """
    Log kill switch event to audit trail.

    This should persist to database for compliance/audit purposes.
    """
    # TODO: Implement database logging
    # For now, just log to file/console
    audit_entry = {
        "event": "kill_switch_activated",
        "reason": request.reason,
        "activated_by": request.activated_by,
        "close_positions": request.close_positions,
        "notify_team": request.notify_team,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    logger.info(f"AUDIT: {audit_entry}")


# Helper function to check kill switch state from other modules
def is_kill_switch_active() -> bool:
    """Check if kill switch is currently active."""
    return _kill_switch_state["active"]


def get_current_mode() -> str:
    """Get current operational mode: normal, paused, or killed."""
    return _kill_switch_state["mode"]
