"""
L5 NRT Inference Service - PURE INFERENCE ONLY
==============================================

DEPRECATED (2026-02-12): This standalone async service has been absorbed into
the Airflow DAG: airflow/dags/l5_multi_model_inference.py (v6.0.0+).

The DAG version reads pre-normalized features from inference_ready_nrt
(written by L1 DAG), adds state features, and runs model.predict().
It also handles canary deployment, risk management, and Redis streaming.

This file is kept for reference and backward compatibility only.

Contract ID: CTR-L5-NRT-001
Version: 1.0.0
Created: 2026-02-04
Deprecated: 2026-02-12

Responsibilities:
- Load model on 'model_approved'
- On 'features_ready': read inference_ready_nrt, run model.predict(), store signal

NO PREPROCESSING. NO DATA LOADING. NO NORMALIZATION.
L1 already did all that. L5 just runs the model.

~50 lines of actual logic. KISS.

Architecture:
    model_approved            features_ready
          |                        |
          v                        v
    +-----------+           +-----------+
    | Load PPO  |           | Read DB   |
    +-----------+           +-----------+
                                   |
                                   v
                            +-----------+
                            | predict() |
                            +-----------+
                                   |
                                   v
                            +-----------+
                            | Store +   |
                            | Broadcast |
                            +-----------+
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

# Async PostgreSQL
try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore

# Stable Baselines 3
try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# WEBSOCKET MANAGER PROTOCOL
# =============================================================================

class IWebSocketManager(Protocol):
    """Protocol for WebSocket broadcast."""
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to connected clients."""
        ...


class NoOpWebSocketManager:
    """No-op WebSocket manager for testing or when WS is unavailable."""
    async def broadcast(self, message: Dict[str, Any]) -> None:
        logger.debug(f"L5: WS broadcast (no-op): {message.get('type', 'unknown')}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class L5NRTConfig:
    """Configuration for L5 NRT Inference Service."""
    # Action thresholds for discretization
    long_threshold: float = 0.3
    short_threshold: float = -0.3
    # State features
    observation_dim: int = 20  # 18 market + 2 state
    market_features_count: int = 18


# =============================================================================
# L5 NRT INFERENCE SERVICE
# =============================================================================

class L5NRTInferenceService:
    """
    L5 Inference Layer for NRT (Near Real-Time) Trading.

    PURE INFERENCE ONLY. This service:
    - Loads model on 'model_approved' event
    - Reads pre-computed features from inference_ready_nrt (no preprocessing!)
    - Adds state features (position, unrealized_pnl)
    - Runs model.predict()
    - Stores signal and broadcasts via WebSocket

    ~50 lines of actual inference logic. Everything else is infrastructure.

    Usage:
        service = L5NRTInferenceService(pool, ws_manager)
        await service.on_model_approved(payload)
        await service.on_features_ready(payload)
    """

    def __init__(
        self,
        db_pool: "asyncpg.Pool",
        ws_manager: Optional[IWebSocketManager] = None,
        config: Optional[L5NRTConfig] = None,
    ):
        """
        Initialize L5 NRT Inference Service.

        Args:
            db_pool: asyncpg connection pool
            ws_manager: WebSocket manager for broadcasting (optional)
            config: Optional configuration
        """
        if asyncpg is None:
            raise ImportError("asyncpg is required for L5NRTInferenceService")

        self.pool = db_pool
        self.ws_manager = ws_manager or NoOpWebSocketManager()
        self.config = config or L5NRTConfig()

        # Model state
        self._model: Optional[PPO] = None
        self._model_id: Optional[str] = None
        self._model_path: Optional[str] = None

        # Position tracking (for state features)
        self._position: float = 0.0  # -1 (short), 0 (flat), 1 (long)
        self._entry_price: Optional[float] = None
        self._unrealized_pnl: float = 0.0

        logger.info("L5NRTInferenceService initialized")

    # =========================================================================
    # MODEL APPROVAL HANDLER
    # =========================================================================

    async def on_model_approved(self, payload: Dict[str, Any]) -> None:
        """
        Handle model approval event - ONLY load model.

        Args:
            payload: model_approved notification payload containing:
                - model_id: Model identifier
                - model_path: Path to saved model
        """
        model_path = payload.get("model_path", "")
        self._model_id = payload.get("model_id", "unknown")

        if not model_path:
            logger.error("L5: No model_path in approval payload")
            return

        # Load model
        try:
            if PPO is None:
                raise ImportError("stable_baselines3 is required")

            self._model = PPO.load(model_path)
            self._model_path = model_path

            # Reset position tracking
            self._position = 0.0
            self._entry_price = None
            self._unrealized_pnl = 0.0

            logger.info(f"L5: Model loaded from {model_path}")

        except Exception as e:
            logger.error(f"L5: Failed to load model from {model_path}: {e}")
            self._model = None

    # =========================================================================
    # FEATURES READY HANDLER - THE CORE INFERENCE LOGIC
    # =========================================================================

    async def on_features_ready(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle features_ready event - run inference.

        This is the CORE LOGIC (~30 lines):
        1. Check model is loaded
        2. Read pre-computed features from DB
        3. Add state features (position, unrealized_pnl)
        4. Run model.predict()
        5. Discretize action to signal
        6. Update position tracker
        7. Store signal in DB
        8. Broadcast via WebSocket

        Args:
            payload: features_ready notification payload containing:
                - timestamp: Feature timestamp
                - price: Current price

        Returns:
            Signal dict or None if inference failed
        """
        if not self._model:
            logger.debug("L5: No model loaded, skipping inference")
            return None

        timestamp = payload.get("timestamp")
        if not timestamp:
            return None

        async with self.pool.acquire() as conn:
            # Step 1: Read pre-computed features from L1's table
            row = await conn.fetchrow(
                """
                SELECT features, price
                FROM inference_ready_nrt
                WHERE timestamp = $1
                """,
                timestamp,
            )

            if not row:
                logger.warning(f"L5: No features found for timestamp {timestamp}")
                return None

            # Step 2: Build observation (18 market + 2 state)
            market_features = list(row["features"])  # Already normalized by L1
            current_price = float(row["price"])

            # Update unrealized PnL
            if self._position != 0 and self._entry_price:
                pnl_pct = (current_price - self._entry_price) / self._entry_price * self._position
                self._unrealized_pnl = float(np.clip(pnl_pct * 100, -5.0, 5.0))  # Normalize

            state_features = [float(self._position), self._unrealized_pnl]
            observation = np.array(
                market_features + state_features,
                dtype=np.float32
            )

            # Step 3: Run inference
            t0 = time.perf_counter()
            action, _states = self._model.predict(observation, deterministic=True)
            latency_ms = (time.perf_counter() - t0) * 1000

            # Step 4: Discretize action to signal
            raw_action = float(action[0]) if hasattr(action, "__len__") else float(action)
            signal = self._discretize_action(raw_action)

            # Step 5: Update position tracker
            self._update_position(signal, current_price)

            # Step 6: Store signal in DB
            await conn.execute(
                """
                INSERT INTO inference_signals_nrt
                    (model_id, timestamp, signal, raw_action, confidence, price, latency_ms)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                self._model_id,
                timestamp,
                signal,
                raw_action,
                self._compute_confidence(raw_action),
                current_price,
                latency_ms,
            )

            # Step 7: Build result
            result = {
                "type": "nrt_inference",
                "model_id": self._model_id,
                "timestamp": str(timestamp),
                "signal": signal,
                "raw_action": raw_action,
                "confidence": self._compute_confidence(raw_action),
                "price": current_price,
                "position": self._position,
                "unrealized_pnl": self._unrealized_pnl,
                "latency_ms": latency_ms,
            }

            # Step 8: Broadcast via WebSocket
            await self.ws_manager.broadcast(result)

            logger.info(
                f"L5: {signal} @ {current_price:.2f}, "
                f"action={raw_action:.4f}, latency={latency_ms:.1f}ms"
            )

            return result

    def _discretize_action(self, action: float) -> str:
        """
        Convert continuous action to discrete signal.

        Args:
            action: Continuous action from model [-1, 1]

        Returns:
            'LONG', 'SHORT', or 'HOLD'
        """
        if action > self.config.long_threshold:
            return "LONG"
        elif action < self.config.short_threshold:
            return "SHORT"
        return "HOLD"

    def _compute_confidence(self, action: float) -> float:
        """
        Compute signal confidence from action magnitude.

        Args:
            action: Continuous action from model

        Returns:
            Confidence in [0, 1]
        """
        # Higher magnitude = higher confidence
        return float(min(abs(action), 1.0))

    def _update_position(self, signal: str, price: float) -> None:
        """
        Update position tracker based on signal.

        Args:
            signal: 'LONG', 'SHORT', or 'HOLD'
            price: Current price
        """
        if signal == "LONG" and self._position != 1:
            self._position = 1.0
            self._entry_price = price
            self._unrealized_pnl = 0.0
        elif signal == "SHORT" and self._position != -1:
            self._position = -1.0
            self._entry_price = price
            self._unrealized_pnl = 0.0
        # HOLD keeps current position

    # =========================================================================
    # STATUS & HEALTH
    # =========================================================================

    @property
    def model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def model_id(self) -> Optional[str]:
        """Get current model ID."""
        return self._model_id

    @property
    def position(self) -> float:
        """Get current position."""
        return self._position

    async def get_status(self) -> Dict[str, Any]:
        """Get service status for health checks."""
        async with self.pool.acquire() as conn:
            signal_count = await conn.fetchval(
                "SELECT COUNT(*) FROM inference_signals_nrt WHERE model_id = $1",
                self._model_id or "",
            )
            latest_signal = await conn.fetchrow(
                """
                SELECT signal, price, latency_ms, created_at
                FROM inference_signals_nrt
                WHERE model_id = $1
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                self._model_id or "",
            )

        return {
            "model_loaded": self._model is not None,
            "model_id": self._model_id,
            "model_path": self._model_path,
            "position": self._position,
            "entry_price": self._entry_price,
            "unrealized_pnl": self._unrealized_pnl,
            "signals_generated": signal_count,
            "latest_signal": dict(latest_signal) if latest_signal else None,
        }

    async def reset_position(self) -> None:
        """Reset position tracking (e.g., at market close)."""
        self._position = 0.0
        self._entry_price = None
        self._unrealized_pnl = 0.0
        logger.info("L5: Position reset to flat")
