"""
WebSocket Router - Real-time predictions and updates

Provides WebSocket endpoints for:
- Real-time prediction streaming
- Connection management with ping/pong
- Subscription-based message filtering
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================


class PredictionMessage(BaseModel):
    """Prediction update message"""

    type: str = "prediction"
    model_id: str
    timestamp: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SubscribeMessage(BaseModel):
    """Subscription request message"""

    type: str = "subscribe"
    channels: list[str]  # e.g., ["predictions", "trades", "alerts"]


class PongMessage(BaseModel):
    """Pong response message"""

    type: str = "pong"
    timestamp: str


# =============================================================================
# Connection Manager
# =============================================================================


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection"""

    websocket: WebSocket
    client_id: str
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    subscriptions: Set[str] = field(default_factory=set)
    last_ping: Optional[datetime] = None


class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.

    Features:
    - Connection tracking by client ID
    - Channel-based subscriptions
    - Broadcast to all or filtered connections
    - Ping/pong health checking
    """

    def __init__(self):
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept a new WebSocket connection.

        Args:
            websocket: The WebSocket connection
            client_id: Unique identifier for the client
        """
        await websocket.accept()

        async with self._lock:
            # Close existing connection for same client_id if any
            if client_id in self.active_connections:
                old_ws = self.active_connections[client_id].websocket
                try:
                    await old_ws.close(code=4000, reason="Replaced by new connection")
                except Exception:
                    pass

            self.active_connections[client_id] = ConnectionInfo(
                websocket=websocket,
                client_id=client_id,
                subscriptions={"predictions"},  # Default subscription
            )

        logger.info(f"WebSocket connected: {client_id} (total: {len(self.active_connections)})")

        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connected",
                "client_id": client_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": "Connected to USDCOP prediction stream",
            },
            client_id,
        )

    async def disconnect(self, client_id: str) -> None:
        """
        Remove a WebSocket connection.

        Args:
            client_id: The client ID to disconnect
        """
        async with self._lock:
            if client_id in self.active_connections:
                del self.active_connections[client_id]

        logger.info(f"WebSocket disconnected: {client_id} (total: {len(self.active_connections)})")

    async def subscribe(self, client_id: str, channels: list[str]) -> None:
        """
        Subscribe a client to specific channels.

        Args:
            client_id: The client to subscribe
            channels: List of channel names to subscribe to
        """
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].subscriptions.update(channels)
                logger.debug(f"Client {client_id} subscribed to: {channels}")

    async def unsubscribe(self, client_id: str, channels: list[str]) -> None:
        """
        Unsubscribe a client from specific channels.

        Args:
            client_id: The client to unsubscribe
            channels: List of channel names to unsubscribe from
        """
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].subscriptions.difference_update(channels)
                logger.debug(f"Client {client_id} unsubscribed from: {channels}")

    async def send_personal_message(self, message: dict, client_id: str) -> bool:
        """
        Send a message to a specific client.

        Args:
            message: The message to send (will be JSON serialized)
            client_id: The target client ID

        Returns:
            True if message was sent, False if client not found
        """
        if client_id not in self.active_connections:
            return False

        try:
            websocket = self.active_connections[client_id].websocket
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send message to {client_id}: {e}")
            await self.disconnect(client_id)
            return False

    async def broadcast(self, message: dict, channel: Optional[str] = None) -> int:
        """
        Broadcast a message to all connected clients.

        Args:
            message: The message to broadcast
            channel: If provided, only send to clients subscribed to this channel

        Returns:
            Number of clients the message was sent to
        """
        sent_count = 0
        disconnected = []

        async with self._lock:
            connections = list(self.active_connections.items())

        for client_id, conn_info in connections:
            # Skip if channel specified and client not subscribed
            if channel and channel not in conn_info.subscriptions:
                continue

            try:
                await conn_info.websocket.send_json(message)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to broadcast to {client_id}: {e}")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

        return sent_count

    async def handle_ping(self, client_id: str) -> None:
        """
        Handle ping message from client, respond with pong.

        Args:
            client_id: The client that sent the ping
        """
        async with self._lock:
            if client_id in self.active_connections:
                self.active_connections[client_id].last_ping = datetime.now(timezone.utc)

        await self.send_personal_message(
            {
                "type": "pong",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            client_id,
        )

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_connection_info(self) -> list[dict]:
        """Get info about all active connections."""
        return [
            {
                "client_id": info.client_id,
                "connected_at": info.connected_at.isoformat(),
                "subscriptions": list(info.subscriptions),
                "last_ping": info.last_ping.isoformat() if info.last_ping else None,
            }
            for info in self.active_connections.values()
        ]


# Global connection manager instance
manager = ConnectionManager()


# =============================================================================
# Helper Functions
# =============================================================================


async def broadcast_prediction(
    model_id: str,
    signal: str,
    confidence: float,
    features: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Broadcast a prediction update to all subscribed clients.

    This function is designed to be called from other parts of the application
    (e.g., inference engine, paper trading service) to push real-time updates.

    Args:
        model_id: The model that generated the prediction
        signal: The trading signal (BUY, SELL, HOLD)
        confidence: Confidence score (0-1)
        features: Optional feature values used in prediction
        metadata: Optional additional metadata

    Returns:
        Number of clients the prediction was sent to

    Example:
        from services.inference_api.routers.websocket import broadcast_prediction

        await broadcast_prediction(
            model_id="ppo_primary",
            signal="BUY",
            confidence=0.85,
            features={"usdcop": 4250.5, "dxy": 103.2},
        )
    """
    message = {
        "type": "prediction",
        "model_id": model_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "confidence": confidence,
    }

    if features:
        message["features"] = features
    if metadata:
        message["metadata"] = metadata

    return await manager.broadcast(message, channel="predictions")


async def broadcast_trade(trade_data: dict) -> int:
    """
    Broadcast a trade update to all subscribed clients.

    Args:
        trade_data: Trade information dictionary

    Returns:
        Number of clients the trade was sent to
    """
    message = {
        "type": "trade",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **trade_data,
    }
    return await manager.broadcast(message, channel="trades")


async def broadcast_alert(
    alert_type: str,
    message: str,
    severity: str = "info",
    data: Optional[dict] = None,
) -> int:
    """
    Broadcast an alert to all subscribed clients.

    Args:
        alert_type: Type of alert (e.g., "model_reload", "kill_switch", "drift")
        message: Human-readable alert message
        severity: Alert severity (info, warning, error, critical)
        data: Optional additional data

    Returns:
        Number of clients the alert was sent to
    """
    alert_message = {
        "type": "alert",
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if data:
        alert_message["data"] = data

    return await manager.broadcast(alert_message, channel="alerts")


# =============================================================================
# WebSocket Endpoints
# =============================================================================


@router.websocket("/ws/predictions")
async def websocket_predictions(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Client identifier"),
):
    """
    WebSocket endpoint for real-time predictions.

    Supports the following message types:

    **Client -> Server:**
    - `ping`: Request a pong response for connection health
    - `subscribe`: Subscribe to channels (predictions, trades, alerts)
    - `unsubscribe`: Unsubscribe from channels

    **Server -> Client:**
    - `connected`: Sent on connection with client_id
    - `pong`: Response to ping
    - `prediction`: New prediction from model
    - `trade`: Trade execution notification
    - `alert`: System alerts and notifications

    Example client usage (JavaScript):
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/ws/predictions?client_id=dashboard-1');

    ws.onopen = () => {
        // Subscribe to channels
        ws.send(JSON.stringify({type: 'subscribe', channels: ['predictions', 'trades']}));

        // Start ping interval
        setInterval(() => ws.send(JSON.stringify({type: 'ping'})), 30000);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'prediction') {
            console.log('New prediction:', data.signal, data.confidence);
        }
    };
    ```
    """
    # Generate client ID if not provided
    if not client_id:
        client_id = f"client-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive and parse message
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "message": "Invalid JSON"},
                    client_id,
                )
                continue

            msg_type = message.get("type", "").lower()

            # Handle ping
            if msg_type == "ping":
                await manager.handle_ping(client_id)

            # Handle subscribe
            elif msg_type == "subscribe":
                channels = message.get("channels", [])
                if channels:
                    await manager.subscribe(client_id, channels)
                    await manager.send_personal_message(
                        {
                            "type": "subscribed",
                            "channels": channels,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        client_id,
                    )

            # Handle unsubscribe
            elif msg_type == "unsubscribe":
                channels = message.get("channels", [])
                if channels:
                    await manager.unsubscribe(client_id, channels)
                    await manager.send_personal_message(
                        {
                            "type": "unsubscribed",
                            "channels": channels,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                        client_id,
                    )

            # Unknown message type
            else:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                        "supported_types": ["ping", "subscribe", "unsubscribe"],
                    },
                    client_id,
                )

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        await manager.disconnect(client_id)


@router.get("/ws/connections")
async def get_websocket_connections():
    """
    Get information about active WebSocket connections.

    Returns:
        - connection_count: Number of active connections
        - connections: List of connection details
    """
    return {
        "connection_count": manager.get_connection_count(),
        "connections": manager.get_connection_info(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
