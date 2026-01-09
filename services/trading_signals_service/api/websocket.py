"""
WebSocket Handler
==================
Real-time signal broadcasting via WebSocket.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
import asyncio
import json
from typing import List, Set, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque

from ..models.signal_schema import TradingSignal
from ..config import get_config

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_count = 0

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_count += 1
        logger.info(f"WebSocket client connected. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Active connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return

        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


class SignalBroadcaster:
    """Broadcasts trading signals via WebSocket"""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.config = get_config()
        self.message_queue: deque = deque(maxlen=self.config.ws_message_queue_size)
        self.last_signal_id: Optional[str] = None

    async def broadcast_signal(self, signal: TradingSignal):
        """
        Broadcast a trading signal to all connected clients.

        Args:
            signal: Trading signal to broadcast
        """
        # Avoid duplicate broadcasts
        if signal.signal_id == self.last_signal_id:
            logger.debug(f"Skipping duplicate signal broadcast: {signal.signal_id}")
            return

        self.last_signal_id = signal.signal_id

        # Prepare message
        message = {
            "type": "signal",
            "timestamp": datetime.utcnow().isoformat(),
            "data": signal.to_dict()
        }

        message_json = json.dumps(message)
        self.message_queue.append(message)

        # Broadcast to all clients
        await self.connection_manager.broadcast(message_json)

        logger.info(
            f"Broadcasted signal: {signal.action.value} @ {signal.entry_price} "
            f"to {self.connection_manager.get_connection_count()} clients"
        )

    async def broadcast_market_update(self, market_data: dict):
        """
        Broadcast market data update.

        Args:
            market_data: Market data dictionary
        """
        message = {
            "type": "market_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": market_data
        }

        message_json = json.dumps(message)
        await self.connection_manager.broadcast(message_json)

    async def broadcast_position_update(self, position_data: dict):
        """
        Broadcast position update.

        Args:
            position_data: Position data dictionary
        """
        message = {
            "type": "position_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": position_data
        }

        message_json = json.dumps(message)
        await self.connection_manager.broadcast(message_json)

    async def broadcast_error(self, error_message: str):
        """
        Broadcast error message.

        Args:
            error_message: Error message string
        """
        message = {
            "type": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "message": error_message
        }

        message_json = json.dumps(message)
        await self.connection_manager.broadcast(message_json)

    async def send_heartbeat(self):
        """Send heartbeat to all connected clients"""
        message = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "connections": self.connection_manager.get_connection_count()
        }

        message_json = json.dumps(message)
        await self.connection_manager.broadcast(message_json)


async def websocket_endpoint(
    websocket: WebSocket,
    connection_manager: ConnectionManager,
    signal_broadcaster: SignalBroadcaster
):
    """
    WebSocket endpoint handler.

    Args:
        websocket: WebSocket connection
        connection_manager: Connection manager instance
        signal_broadcaster: Signal broadcaster instance
    """
    await connection_manager.connect(websocket)

    try:
        # Send welcome message
        welcome_message = {
            "type": "connected",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to trading signals WebSocket",
            "version": "1.0.0"
        }
        await websocket.send_text(json.dumps(welcome_message))

        # Send recent signals (last 5)
        if signal_broadcaster.message_queue:
            recent_signals = list(signal_broadcaster.message_queue)[-5:]
            history_message = {
                "type": "history",
                "timestamp": datetime.utcnow().isoformat(),
                "data": recent_signals
            }
            await websocket.send_text(json.dumps(history_message))

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message with timeout
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Handle client messages
                try:
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "ping":
                        # Respond to ping
                        pong_message = {
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await websocket.send_text(json.dumps(pong_message))

                    elif message_type == "subscribe":
                        # Handle subscription (placeholder)
                        channels = message.get("channels", [])
                        ack_message = {
                            "type": "subscribed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "channels": channels
                        }
                        await websocket.send_text(json.dumps(ack_message))

                    elif message_type == "unsubscribe":
                        # Handle unsubscription (placeholder)
                        channels = message.get("channels", [])
                        ack_message = {
                            "type": "unsubscribed",
                            "timestamp": datetime.utcnow().isoformat(),
                            "channels": channels
                        }
                        await websocket.send_text(json.dumps(ack_message))

                except json.JSONDecodeError:
                    # Handle non-JSON messages
                    if data == "ping":
                        await websocket.send_text("pong")

            except asyncio.TimeoutError:
                # Send heartbeat on timeout
                await signal_broadcaster.send_heartbeat()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
        connection_manager.disconnect(websocket)

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        connection_manager.disconnect(websocket)


async def heartbeat_loop(signal_broadcaster: SignalBroadcaster, interval: int = 30):
    """
    Background task to send periodic heartbeats.

    Args:
        signal_broadcaster: Signal broadcaster instance
        interval: Heartbeat interval in seconds
    """
    while True:
        try:
            await asyncio.sleep(interval)
            await signal_broadcaster.send_heartbeat()
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")
