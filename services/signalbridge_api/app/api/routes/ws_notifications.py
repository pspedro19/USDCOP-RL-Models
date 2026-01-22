"""
WebSocket Notifications Router
==============================

WebSocket server for real-time execution notifications to frontend.

Message Types (Server -> Client):
- execution_created: New execution started
- execution_updated: Execution status changed
- execution_filled: Execution completed successfully
- execution_failed: Execution failed
- risk_alert: Risk rule triggered
- kill_switch: Kill switch status changed
- heartbeat: Connection keep-alive

Message Types (Client -> Server):
- subscribe: Subscribe to specific events
- unsubscribe: Unsubscribe from events
- ping: Request heartbeat response

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from starlette.websockets import WebSocketState

from app.contracts.signal_bridge import (
    ExecutionNotification,
    RiskAlertNotification,
    KillSwitchNotification,
    WebSocketMessage,
    TradingMode,
)
from app.contracts.execution import ExecutionStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["WebSocket"])


# =============================================================================
# Connection Manager
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.

    Features:
    - Connection tracking per user
    - Topic-based subscriptions
    - Broadcast to all or specific users
    - Automatic cleanup on disconnect
    """

    def __init__(self):
        # user_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}

        # WebSocket -> set of subscribed topics
        self._subscriptions: Dict[WebSocket, Set[str]] = {}

        # All active connections
        self._all_connections: Set[WebSocket] = set()

        # Statistics
        self._total_connections = 0
        self._total_messages_sent = 0

    @property
    def active_connections(self) -> int:
        """Number of active connections."""
        return len(self._all_connections)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "active_connections": self.active_connections,
            "total_connections": self._total_connections,
            "total_messages_sent": self._total_messages_sent,
            "users_connected": len(self._connections),
        }

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket instance
            user_id: User identifier
        """
        await websocket.accept()

        # Track connection
        self._all_connections.add(websocket)
        self._subscriptions[websocket] = set(["all"])  # Default subscription
        self._total_connections += 1

        # Track by user
        if user_id not in self._connections:
            self._connections[user_id] = set()
        self._connections[user_id].add(websocket)

        logger.info(f"WebSocket connected: user={user_id}, total={self.active_connections}")

    async def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        """
        Handle WebSocket disconnection.

        Args:
            websocket: WebSocket instance
            user_id: User identifier
        """
        # Remove from tracking
        self._all_connections.discard(websocket)
        self._subscriptions.pop(websocket, None)

        if user_id in self._connections:
            self._connections[user_id].discard(websocket)
            if not self._connections[user_id]:
                del self._connections[user_id]

        logger.info(f"WebSocket disconnected: user={user_id}, total={self.active_connections}")

    async def subscribe(self, websocket: WebSocket, topics: list[str]) -> None:
        """
        Subscribe a connection to specific topics.

        Args:
            websocket: WebSocket instance
            topics: List of topics to subscribe to
        """
        if websocket in self._subscriptions:
            self._subscriptions[websocket].update(topics)
            logger.debug(f"Subscribed to topics: {topics}")

    async def unsubscribe(self, websocket: WebSocket, topics: list[str]) -> None:
        """
        Unsubscribe a connection from specific topics.

        Args:
            websocket: WebSocket instance
            topics: List of topics to unsubscribe from
        """
        if websocket in self._subscriptions:
            for topic in topics:
                self._subscriptions[websocket].discard(topic)
            logger.debug(f"Unsubscribed from topics: {topics}")

    async def send_to_user(self, user_id: str, message: dict) -> int:
        """
        Send a message to all connections of a specific user.

        Args:
            user_id: Target user
            message: Message to send

        Returns:
            Number of messages sent
        """
        count = 0
        if user_id in self._connections:
            for ws in self._connections[user_id].copy():
                if await self._send_safe(ws, message):
                    count += 1
        return count

    async def broadcast(self, message: dict, topic: str = "all") -> int:
        """
        Broadcast a message to all connections subscribed to a topic.

        Args:
            message: Message to broadcast
            topic: Topic filter (default "all" matches everyone)

        Returns:
            Number of messages sent
        """
        count = 0
        for ws in self._all_connections.copy():
            subs = self._subscriptions.get(ws, set())
            if "all" in subs or topic in subs:
                if await self._send_safe(ws, message):
                    count += 1
        return count

    async def _send_safe(self, websocket: WebSocket, message: dict) -> bool:
        """
        Safely send a message to a WebSocket.

        Args:
            websocket: Target WebSocket
            message: Message to send

        Returns:
            True if sent successfully
        """
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
                self._total_messages_sent += 1
                return True
        except Exception as e:
            logger.warning(f"Error sending WebSocket message: {e}")
        return False


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@router.websocket("/executions")
async def websocket_executions(
    websocket: WebSocket,
    token: str = Query(None, description="Authentication token"),
):
    """
    WebSocket endpoint for execution notifications.

    Connection URL: ws://host/ws/executions?token=<auth_token>

    Message format:
    {
        "type": "message_type",
        "timestamp": "ISO8601",
        "data": { ... }
    }
    """
    # TODO: Validate token and get user_id
    # For now, use a placeholder
    user_id = "anonymous"
    if token:
        # Decode token to get user_id
        try:
            from app.core.security import decode_access_token
            payload = decode_access_token(token)
            user_id = payload.get("sub", "anonymous")
        except Exception as e:
            logger.warning(f"Invalid WebSocket token: {e}")

    # Accept connection
    await manager.connect(websocket, user_id)

    # Send welcome message
    await websocket.send_json({
        "type": "connected",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "user_id": user_id,
            "message": "Connected to execution notifications",
        },
    })

    try:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            _send_heartbeats(websocket, interval=30)
        )

        # Process incoming messages
        async for data in websocket.iter_json():
            await _handle_client_message(websocket, data, user_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {user_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        await manager.disconnect(websocket, user_id)


async def _send_heartbeats(websocket: WebSocket, interval: int = 30) -> None:
    """Send periodic heartbeats to keep connection alive."""
    while True:
        try:
            await asyncio.sleep(interval)
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {"status": "ok"},
                })
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Heartbeat error: {e}")
            break


async def _handle_client_message(
    websocket: WebSocket,
    data: dict,
    user_id: str,
) -> None:
    """Handle incoming client message."""
    msg_type = data.get("type", "unknown")

    if msg_type == "subscribe":
        topics = data.get("topics", [])
        await manager.subscribe(websocket, topics)
        await websocket.send_json({
            "type": "subscribed",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"topics": topics},
        })

    elif msg_type == "unsubscribe":
        topics = data.get("topics", [])
        await manager.unsubscribe(websocket, topics)
        await websocket.send_json({
            "type": "unsubscribed",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"topics": topics},
        })

    elif msg_type == "ping":
        await websocket.send_json({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {},
        })

    else:
        logger.debug(f"Unknown message type: {msg_type}")


# =============================================================================
# Notification Functions (for use by other services)
# =============================================================================

async def notify_execution_created(
    user_id: str,
    execution_id: str,
    signal_id: str,
    symbol: str,
    side: str,
) -> int:
    """Notify about a new execution."""
    message = {
        "type": "execution_created",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "execution_id": execution_id,
            "signal_id": signal_id,
            "symbol": symbol,
            "side": side,
            "status": "pending",
        },
    }
    return await manager.send_to_user(user_id, message)


async def notify_execution_updated(
    user_id: str,
    execution_id: str,
    status: ExecutionStatus,
    filled_quantity: float = 0,
    filled_price: float = 0,
    error_message: Optional[str] = None,
) -> int:
    """Notify about execution status update."""
    message = {
        "type": "execution_updated",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "execution_id": execution_id,
            "status": status.value,
            "filled_quantity": filled_quantity,
            "filled_price": filled_price,
            "error_message": error_message,
        },
    }
    return await manager.send_to_user(user_id, message)


async def notify_risk_alert(
    user_id: str,
    alert_type: str,
    message: str,
    severity: str = "warning",
) -> int:
    """Notify about a risk alert."""
    notification = {
        "type": "risk_alert",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
        },
    }
    return await manager.send_to_user(user_id, notification)


async def broadcast_kill_switch(
    active: bool,
    reason: Optional[str] = None,
    activated_by: Optional[str] = None,
) -> int:
    """Broadcast kill switch status change to all users."""
    message = {
        "type": "kill_switch",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "active": active,
            "reason": reason,
            "activated_by": activated_by,
        },
    }
    return await manager.broadcast(message, topic="kill_switch")


async def broadcast_trading_mode_change(mode: TradingMode) -> int:
    """Broadcast trading mode change to all users."""
    message = {
        "type": "trading_mode_changed",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "mode": mode.value,
        },
    }
    return await manager.broadcast(message, topic="all")


# =============================================================================
# REST Endpoint for Connection Status
# =============================================================================

@router.get("/status", tags=["WebSocket"])
async def get_ws_status() -> dict:
    """Get WebSocket connection statistics."""
    return manager.stats
