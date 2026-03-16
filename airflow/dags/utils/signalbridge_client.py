"""
SignalBridge Client — HTTP client for DAG→OMS bridge.

Provides a unified interface for DAGs to place orders through SignalBridge
instead of using PaperBroker directly. The execution mode is controlled by
the EXECUTION_MODE environment variable.

Modes:
  - paper (default): Use PaperBroker locally (no SignalBridge call)
  - testnet: Call SignalBridge with testnet exchange credentials
  - live: Call SignalBridge with production exchange credentials

Contract: CTR-EXEC-001 (sdd-execution-bridge.md)
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXECUTION_MODE = os.getenv("EXECUTION_MODE", "paper")
SIGNALBRIDGE_URL = os.getenv("SIGNALBRIDGE_URL", "http://usdcop-signalbridge:8000")
SIGNALBRIDGE_TOKEN = os.getenv("SIGNALBRIDGE_SERVICE_TOKEN", "")
SIGNALBRIDGE_TIMEOUT = int(os.getenv("SIGNALBRIDGE_TIMEOUT", "30"))

# Default credential_id for the service account (stored in SignalBridge DB)
SERVICE_CREDENTIAL_ID = os.getenv("SIGNALBRIDGE_CREDENTIAL_ID", "")
SERVICE_USER_ID = os.getenv("SIGNALBRIDGE_USER_ID", "airflow-service")


@dataclass
class BridgeOrderResult:
    """Unified order result from either PaperBroker or SignalBridge."""

    success: bool
    order_id: str
    fill_price: float
    status: str  # filled, rejected, failed
    slippage_bps: float
    error: Optional[str] = None
    execution_id: Optional[str] = None  # SignalBridge execution UUID


def get_execution_mode() -> str:
    """Return current execution mode (paper/testnet/live)."""
    mode = EXECUTION_MODE.lower()
    if mode not in ("paper", "testnet", "live"):
        logger.warning(f"[SignalBridge] Unknown EXECUTION_MODE={mode}, defaulting to paper")
        return "paper"
    return mode


def is_paper_mode() -> bool:
    """Check if we're in paper trading mode (no SignalBridge calls)."""
    return get_execution_mode() == "paper"


def place_order_via_bridge(
    symbol: str,
    side: str,  # "buy" or "sell"
    quantity: float,
    price: float,
    signal_id: str,
    confidence: float = 0.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> BridgeOrderResult:
    """
    Place an order through SignalBridge API.

    Only called when EXECUTION_MODE is 'testnet' or 'live'.
    For 'paper' mode, the DAG should use PaperBroker directly.

    Args:
        symbol: Trading pair (e.g., "USD/COP")
        side: Order side ("buy" or "sell")
        quantity: Order quantity
        price: Reference price for the order
        signal_id: Unique signal identifier (e.g., "h1_2026-01-15")
        confidence: Signal confidence [0.0, 1.0]
        stop_loss: Optional stop loss price
        take_profit: Optional take profit price
        metadata: Additional signal metadata

    Returns:
        BridgeOrderResult with fill details or error
    """
    try:
        import httpx
    except ImportError:
        logger.error("[SignalBridge] httpx not installed. Install with: pip install httpx")
        return BridgeOrderResult(
            success=False, order_id="", fill_price=price,
            status="failed", slippage_bps=0, error="httpx not installed",
        )

    mode = get_execution_mode()
    logger.info(
        f"[SignalBridge] Placing {side.upper()} order: "
        f"symbol={symbol}, qty={quantity:.4f}, price={price:.2f}, mode={mode}"
    )

    # Map side to SignalBridge action (0=SELL, 1=HOLD, 2=BUY)
    action = 0 if side.lower() == "sell" else 2

    payload = {
        "signal_id": signal_id,
        "model_id": f"forecast_{mode}",
        "action": action,
        "confidence": confidence,
        "symbol": symbol,
        "credential_id": SERVICE_CREDENTIAL_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "source": "airflow_dag",
            "execution_mode": mode,
            "quantity": quantity,
            "reference_price": price,
            **(metadata or {}),
        },
    }

    if stop_loss is not None:
        payload["stop_loss"] = stop_loss
    if take_profit is not None:
        payload["take_profit"] = take_profit

    headers = {
        "Content-Type": "application/json",
    }
    if SIGNALBRIDGE_TOKEN:
        headers["Authorization"] = f"Bearer {SIGNALBRIDGE_TOKEN}"

    try:
        with httpx.Client(timeout=SIGNALBRIDGE_TIMEOUT) as client:
            response = client.post(
                f"{SIGNALBRIDGE_URL}/api/signal-bridge/process",
                json=payload,
                headers=headers,
            )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return BridgeOrderResult(
                    success=True,
                    order_id=data.get("execution_id", ""),
                    fill_price=data.get("filled_price", price),
                    status=data.get("status", "filled"),
                    slippage_bps=0,  # Computed by exchange
                    execution_id=data.get("execution_id"),
                )
            else:
                reason = data.get("risk_check", {}).get("message", "Unknown rejection")
                logger.warning(f"[SignalBridge] Order rejected: {reason}")
                return BridgeOrderResult(
                    success=False, order_id="", fill_price=price,
                    status="rejected", slippage_bps=0, error=reason,
                )
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            logger.error(f"[SignalBridge] API error: {error_msg}")
            return BridgeOrderResult(
                success=False, order_id="", fill_price=price,
                status="failed", slippage_bps=0, error=error_msg,
            )

    except httpx.TimeoutException:
        logger.error(f"[SignalBridge] Timeout after {SIGNALBRIDGE_TIMEOUT}s")
        return BridgeOrderResult(
            success=False, order_id="", fill_price=price,
            status="failed", slippage_bps=0, error="Request timeout",
        )
    except httpx.ConnectError as e:
        logger.error(f"[SignalBridge] Connection failed: {e}")
        return BridgeOrderResult(
            success=False, order_id="", fill_price=price,
            status="failed", slippage_bps=0,
            error=f"Cannot connect to SignalBridge at {SIGNALBRIDGE_URL}",
        )
    except Exception as e:
        logger.error(f"[SignalBridge] Unexpected error: {e}")
        return BridgeOrderResult(
            success=False, order_id="", fill_price=price,
            status="failed", slippage_bps=0, error=str(e),
        )


def check_bridge_health() -> bool:
    """Check if SignalBridge is reachable and healthy."""
    if is_paper_mode():
        return True  # No bridge needed in paper mode

    try:
        import httpx
        with httpx.Client(timeout=5) as client:
            response = client.get(f"{SIGNALBRIDGE_URL}/health")
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"[SignalBridge] Health check failed: {e}")
        return False
