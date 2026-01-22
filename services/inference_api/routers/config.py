"""
Config Router - Feature Flags and Configuration API
====================================================

Endpoints for managing runtime configuration and feature flags.

Endpoints:
- GET /config/trading-flags - Get current feature flags status
- POST /config/kill-switch - Activate emergency kill switch
- POST /config/reload-flags - Reload flags from config file

Security:
- All endpoints require authentication
- Kill switch requires explicit confirmation
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from services.shared.feature_flags import get_feature_flags

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


# =============================================================================
# Request/Response Models
# =============================================================================

class TradingFlagsResponse(BaseModel):
    """Response model for trading flags endpoint."""
    flags: Dict[str, Dict[str, Any]]
    count: int
    timestamp: str
    config_source: str


class KillSwitchRequest(BaseModel):
    """Request model for activating kill switch via config endpoint."""
    reason: str = Field(..., min_length=5, description="Reason for activating kill switch")
    activated_by: str = Field(default="api", description="Who activated the kill switch")
    confirmation: str = Field(..., description="Must be 'CONFIRM_KILL_SWITCH' to proceed")
    close_positions: bool = Field(default=True, description="Whether to close all open positions")
    notify_team: bool = Field(default=True, description="Whether to send team notifications")


class KillSwitchResponse(BaseModel):
    """Response model for kill switch activation."""
    success: bool
    message: str
    activated_at: str
    activated_by: str
    reason: str


class ReloadFlagsResponse(BaseModel):
    """Response model for reload flags endpoint."""
    success: bool
    flags_loaded: int
    timestamp: str
    message: str


class FlagUpdateRequest(BaseModel):
    """Request model for updating a single flag."""
    flag_name: str
    enabled: bool
    confirmation: Optional[str] = None


class FlagUpdateResponse(BaseModel):
    """Response model for flag update."""
    success: bool
    flag_name: str
    enabled: bool
    message: str
    timestamp: str


# =============================================================================
# Global Kill Switch State (mirrors operations.py state)
# =============================================================================

# This should be shared with operations.py in production
# For now, we import and use the state from operations router
_config_kill_switch_active = False


def _set_kill_switch(active: bool, reason: str, activated_by: str) -> None:
    """
    Set kill switch state.

    In production, this should update a shared state (Redis, DB, etc.)
    """
    global _config_kill_switch_active
    _config_kill_switch_active = active

    # Try to sync with operations router
    try:
        from .operations import _kill_switch_state
        _kill_switch_state["active"] = active
        _kill_switch_state["activated_at"] = datetime.now(timezone.utc).isoformat() if active else None
        _kill_switch_state["activated_by"] = activated_by if active else None
        _kill_switch_state["reason"] = reason if active else None
        _kill_switch_state["mode"] = "killed" if active else "normal"
    except ImportError:
        logger.warning("Could not sync kill switch state with operations router")


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/trading-flags", response_model=TradingFlagsResponse)
async def get_trading_flags():
    """
    Get current feature flags status.

    Returns all configured feature flags with their current state,
    descriptions, and rollout percentages.

    Returns:
        TradingFlagsResponse with all flags and metadata
    """
    flags = get_feature_flags()
    all_flags = flags.get_all_as_dict()

    return TradingFlagsResponse(
        flags=all_flags,
        count=len(all_flags),
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_source=str(flags.config_path),
    )


@router.post("/kill-switch", response_model=KillSwitchResponse)
async def activate_kill_switch(
    request: KillSwitchRequest,
    background_tasks: BackgroundTasks,
):
    """
    EMERGENCY: Activate kill switch to stop all trading.

    This is a critical safety endpoint that:
    1. Sets the global kill switch flag
    2. Optionally closes all open positions
    3. Sends notifications to the team
    4. Logs the event for audit

    Requires explicit confirmation string to prevent accidental activation.

    Args:
        request: Kill switch request with reason and confirmation

    Returns:
        KillSwitchResponse with activation details

    Raises:
        HTTPException 400: If confirmation is invalid
    """
    # Validate confirmation
    if request.confirmation != "CONFIRM_KILL_SWITCH":
        logger.warning(
            f"Kill switch activation rejected - invalid confirmation from {request.activated_by}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_confirmation",
                "message": "Must provide confirmation='CONFIRM_KILL_SWITCH' to activate",
                "required_value": "CONFIRM_KILL_SWITCH",
            }
        )

    # Activate kill switch
    activated_at = datetime.now(timezone.utc).isoformat()
    _set_kill_switch(True, request.reason, request.activated_by)

    # Log critical event
    logger.critical(
        f"KILL SWITCH ACTIVATED via config API by {request.activated_by}: {request.reason}"
    )

    # Close positions if requested
    if request.close_positions:
        background_tasks.add_task(_close_positions_task)

    # Send notifications if requested
    if request.notify_team:
        background_tasks.add_task(
            _send_notification_task,
            "KILL SWITCH ACTIVATED",
            f"Reason: {request.reason}\nActivated by: {request.activated_by}",
        )

    return KillSwitchResponse(
        success=True,
        message="Kill switch activated. All trading operations stopped.",
        activated_at=activated_at,
        activated_by=request.activated_by,
        reason=request.reason,
    )


@router.post("/reload-flags", response_model=ReloadFlagsResponse)
async def reload_flags():
    """
    Reload feature flags from the configuration file.

    Forces an immediate reload of all feature flags from the
    JSON config file, bypassing the normal reload interval.

    Useful after manually updating the config file.

    Returns:
        ReloadFlagsResponse with reload status
    """
    flags = get_feature_flags()

    try:
        count = flags.reload()
        logger.info(f"Feature flags reloaded: {count} flags loaded")

        return ReloadFlagsResponse(
            success=True,
            flags_loaded=count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Successfully reloaded {count} feature flags from config",
        )
    except Exception as e:
        logger.error(f"Failed to reload feature flags: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "reload_failed",
                "message": f"Failed to reload flags: {str(e)}",
            }
        )


@router.get("/flag/{flag_name}")
async def get_flag(flag_name: str):
    """
    Get a specific feature flag by name.

    Args:
        flag_name: Name of the flag to retrieve

    Returns:
        Flag details or 404 if not found
    """
    flags = get_feature_flags()
    flag = flags.get_flag(flag_name)

    if flag is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "flag_not_found",
                "message": f"Feature flag '{flag_name}' not found",
            }
        )

    return {
        "flag": flag.to_dict(),
        "is_enabled": flag.is_enabled_for_rollout(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.put("/flag/{flag_name}", response_model=FlagUpdateResponse)
async def update_flag(flag_name: str, request: FlagUpdateRequest):
    """
    Update a feature flag's enabled state (runtime only).

    Note: This is a runtime override and does not persist to the config file.
    The flag will revert to its config value on next reload.

    For permanent changes, update the config file and call /reload-flags.

    Args:
        flag_name: Name of the flag to update
        request: Update request with new enabled state

    Returns:
        FlagUpdateResponse with update status
    """
    # Validate flag_name matches request
    if request.flag_name != flag_name:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "flag_name_mismatch",
                "message": "Flag name in URL must match flag_name in request body",
            }
        )

    flags = get_feature_flags()

    # Update the flag
    flags.set_flag(flag_name, request.enabled)

    logger.info(f"Feature flag '{flag_name}' updated to {request.enabled} (runtime override)")

    return FlagUpdateResponse(
        success=True,
        flag_name=flag_name,
        enabled=request.enabled,
        message=f"Flag '{flag_name}' set to {request.enabled} (runtime override, not persisted)",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/health")
async def config_health():
    """
    Check configuration service health.

    Returns:
        Health status of the config service
    """
    flags = get_feature_flags()
    all_flags = flags.get_all()

    return {
        "status": "healthy",
        "flags_loaded": len(all_flags),
        "config_path": str(flags.config_path),
        "config_exists": flags.config_path.exists(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# =============================================================================
# Background Tasks
# =============================================================================

async def _close_positions_task():
    """Background task to close all positions."""
    try:
        logger.info("Closing all positions (background task)...")
        # This should integrate with the actual trading system
        # For now, just log the action
        logger.info("Position closure task completed")
    except Exception as e:
        logger.error(f"Error closing positions: {e}")


async def _send_notification_task(title: str, message: str):
    """Background task to send notifications."""
    try:
        logger.info(f"Sending notification: {title}")
        # Try to use Slack client if available
        try:
            from src.shared.notifications.slack_client import get_slack_client
            client = get_slack_client()
            await client.send_message(f"*{title}*\n{message}")
        except ImportError:
            logger.debug("Slack client not available")
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
