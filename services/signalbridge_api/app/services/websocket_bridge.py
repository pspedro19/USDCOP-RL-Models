"""
WebSocket Bridge Service
========================

Consumes prediction signals from the Inference API WebSocket.
Connects to: ws://inference-api:8000/ws/predictions

Design Pattern: Observer/Consumer
- Subscribes to inference API predictions
- Forwards signals to SignalBridgeOrchestrator

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Any, Dict
from uuid import UUID, uuid4

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    InvalidStatusCode,
)

from app.contracts.signal_bridge import (
    InferenceSignalCreate,
    InferencePredictionMessage,
    BridgeEventType,
)
from app.core.config import settings

logger = logging.getLogger(__name__)


class WebSocketBridge:
    """
    Consumes signals from the Inference API WebSocket.

    Features:
    - Automatic reconnection with exponential backoff
    - Message validation
    - Health monitoring
    - Graceful shutdown

    Usage:
        bridge = WebSocketBridge(
            inference_ws_url="ws://inference-api:8000/ws/predictions",
            on_signal_received=handle_signal,
        )

        # Start consuming in background
        await bridge.start()

        # Stop when done
        await bridge.stop()
    """

    # Connection state
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"

    def __init__(
        self,
        inference_ws_url: Optional[str] = None,
        on_signal_received: Optional[Callable[[InferenceSignalCreate], Any]] = None,
        default_credential_id: Optional[UUID] = None,
        reconnect_delay_base: float = 1.0,
        reconnect_delay_max: float = 60.0,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
    ):
        """
        Initialize WebSocket Bridge.

        Args:
            inference_ws_url: WebSocket URL for inference API
            on_signal_received: Callback when signal is received
            default_credential_id: Default exchange credential to use
            reconnect_delay_base: Base delay for reconnection (seconds)
            reconnect_delay_max: Maximum reconnection delay (seconds)
            ping_interval: Ping interval to keep connection alive
            ping_timeout: Timeout for ping response
        """
        self.inference_ws_url = inference_ws_url or getattr(
            settings, "inference_ws_url", "ws://inference-api:8000/ws/predictions"
        )
        self.on_signal_received = on_signal_received
        self.default_credential_id = default_credential_id

        # Reconnection settings
        self.reconnect_delay_base = reconnect_delay_base
        self.reconnect_delay_max = reconnect_delay_max
        self.reconnect_attempt = 0

        # Ping settings
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        # State
        self._connection_state = self.DISCONNECTED
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._last_message_at: Optional[datetime] = None
        self._messages_received = 0
        self._signals_processed = 0
        self._errors_count = 0

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connection_state == self.CONNECTED

    @property
    def state(self) -> str:
        """Get current connection state."""
        return self._connection_state

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "state": self._connection_state,
            "is_connected": self.is_connected,
            "messages_received": self._messages_received,
            "signals_processed": self._signals_processed,
            "errors_count": self._errors_count,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "reconnect_attempts": self.reconnect_attempt,
            "ws_url": self.inference_ws_url,
        }

    async def start(self) -> None:
        """
        Start the WebSocket bridge.

        Begins consuming messages from the inference API.
        Runs until stop() is called.
        """
        if self._running:
            logger.warning("WebSocket bridge already running")
            return

        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info(f"WebSocket bridge started, connecting to {self.inference_ws_url}")

    async def stop(self) -> None:
        """
        Stop the WebSocket bridge gracefully.
        """
        logger.info("Stopping WebSocket bridge...")
        self._running = False

        # Close WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        # Cancel consumer task
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        self._connection_state = self.DISCONNECTED
        logger.info("WebSocket bridge stopped")

    async def _consume_loop(self) -> None:
        """
        Main consumption loop with automatic reconnection.
        """
        while self._running:
            try:
                await self._connect_and_consume()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket consume loop error: {e}")
                self._errors_count += 1

            if self._running:
                # Reconnect with exponential backoff
                await self._reconnect_delay()

    async def _connect_and_consume(self) -> None:
        """
        Connect to WebSocket and consume messages.
        """
        self._connection_state = self.CONNECTING
        logger.info(f"Connecting to inference API: {self.inference_ws_url}")

        try:
            async with websockets.connect(
                self.inference_ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=5.0,
            ) as websocket:
                self._websocket = websocket
                self._connection_state = self.CONNECTED
                self.reconnect_attempt = 0  # Reset on successful connection
                logger.info("Connected to inference API WebSocket")

                # Consume messages
                async for message in websocket:
                    if not self._running:
                        break
                    await self._handle_message(message)

        except InvalidStatusCode as e:
            logger.error(f"WebSocket connection rejected: {e.status_code}")
            self._connection_state = self.DISCONNECTED
        except ConnectionClosedError as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._connection_state = self.DISCONNECTED
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._connection_state = self.DISCONNECTED
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self._connection_state = self.DISCONNECTED
            raise

    async def _handle_message(self, message: str) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            message: Raw message string
        """
        self._messages_received += 1
        self._last_message_at = datetime.utcnow()

        try:
            data = json.loads(message)
            msg_type = data.get("type", "unknown")

            if msg_type == "prediction":
                await self._handle_prediction(data)
            elif msg_type == "heartbeat":
                logger.debug("Received heartbeat from inference API")
            elif msg_type == "error":
                logger.warning(f"Error from inference API: {data.get('message')}")
            else:
                logger.debug(f"Unknown message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
            self._errors_count += 1
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            self._errors_count += 1

    async def _handle_prediction(self, data: Dict[str, Any]) -> None:
        """
        Handle prediction message from inference API.

        Args:
            data: Prediction message data
        """
        try:
            # Parse prediction message
            signal_id = data.get("signal_id") or str(uuid4())
            model_id = data.get("model_id", "unknown")
            action = int(data.get("action", 1))  # Default to HOLD
            confidence = float(data.get("confidence", 0.0))
            symbol = data.get("symbol", "UNKNOWN")
            timestamp = data.get("timestamp")

            # Validate action (0=SELL, 1=HOLD, 2=BUY)
            if action not in (0, 1, 2):
                logger.warning(f"Invalid action value: {action}, defaulting to HOLD")
                action = 1

            # Skip HOLD signals
            if action == 1:
                logger.debug(f"Skipping HOLD signal from model {model_id}")
                return

            # Build signal
            signal = InferenceSignalCreate(
                signal_id=UUID(signal_id) if isinstance(signal_id, str) else signal_id,
                model_id=model_id,
                action=action,
                confidence=confidence,
                symbol=symbol,
                credential_id=self.default_credential_id or uuid4(),
                timestamp=datetime.fromisoformat(timestamp) if timestamp else datetime.utcnow(),
                metadata={
                    "features": data.get("features"),
                    "source": "websocket_bridge",
                },
            )

            # Invoke callback
            if self.on_signal_received:
                self._signals_processed += 1
                logger.info(
                    f"Signal received: {signal.action_name} {symbol} "
                    f"(confidence={confidence:.3f}, model={model_id})"
                )
                await self.on_signal_received(signal)

        except Exception as e:
            logger.error(f"Error processing prediction: {e}", exc_info=True)
            self._errors_count += 1

    async def _reconnect_delay(self) -> None:
        """
        Wait before reconnecting with exponential backoff.
        """
        self.reconnect_attempt += 1
        self._connection_state = self.RECONNECTING

        # Calculate delay with exponential backoff
        delay = min(
            self.reconnect_delay_base * (2 ** (self.reconnect_attempt - 1)),
            self.reconnect_delay_max,
        )

        logger.info(
            f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempt})"
        )
        await asyncio.sleep(delay)

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to the inference API (if bidirectional communication needed).

        Args:
            message: Message to send

        Returns:
            True if sent successfully
        """
        if not self._websocket or not self.is_connected:
            logger.warning("Cannot send message - not connected")
            return False

        try:
            await self._websocket.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the WebSocket connection.

        Returns:
            Dict with health status
        """
        is_healthy = self.is_connected and self._running

        # Check for stale connection
        if self._last_message_at:
            seconds_since_message = (datetime.utcnow() - self._last_message_at).total_seconds()
            if seconds_since_message > 120:  # 2 minutes without messages
                is_healthy = False

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connected": self.is_connected,
            "state": self._connection_state,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "messages_received": self._messages_received,
            "signals_processed": self._signals_processed,
            "errors_count": self._errors_count,
        }


class WebSocketBridgeManager:
    """
    Manager for WebSocket bridge lifecycle.

    Provides a singleton-like interface for the bridge.
    """

    _instance: Optional[WebSocketBridge] = None

    @classmethod
    def get_instance(cls) -> Optional[WebSocketBridge]:
        """Get the current bridge instance."""
        return cls._instance

    @classmethod
    def create_instance(
        cls,
        on_signal_received: Optional[Callable[[InferenceSignalCreate], Any]] = None,
        **kwargs,
    ) -> WebSocketBridge:
        """
        Create a new bridge instance.

        Args:
            on_signal_received: Callback for received signals
            **kwargs: Additional arguments for WebSocketBridge

        Returns:
            New WebSocketBridge instance
        """
        if cls._instance is not None:
            logger.warning("Bridge instance already exists - stopping old instance")
            asyncio.create_task(cls._instance.stop())

        cls._instance = WebSocketBridge(
            on_signal_received=on_signal_received,
            **kwargs,
        )
        return cls._instance

    @classmethod
    async def start(cls) -> bool:
        """Start the bridge instance."""
        if cls._instance is None:
            logger.error("No bridge instance to start")
            return False

        await cls._instance.start()
        return True

    @classmethod
    async def stop(cls) -> None:
        """Stop the bridge instance."""
        if cls._instance:
            await cls._instance.stop()
            cls._instance = None
