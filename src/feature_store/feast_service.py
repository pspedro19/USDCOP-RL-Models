"""
Feast Inference Service - Feature Retrieval with Fallback
==========================================================
This module provides a unified interface for feature retrieval using Feast,
with automatic fallback to CanonicalFeatureBuilder when Feast is unavailable.

Architecture:
    Inference Request
           │
           ▼
    FeastInferenceService
           │
           ├── [Feast Available] ──> Redis Online Store ──> Features
           │
           └── [Feast Unavailable] ──> CanonicalFeatureBuilder ──> Features

Design Principles:
- Graceful Degradation: Falls back to SSOT when Feast is unavailable
- Metrics: Tracks latency, hit rate, and fallback usage
- Async Support: Non-blocking feature retrieval
- Health Checks: Monitors Feast connectivity

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
Contract: CTR-FEAST-001
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Import Feast conditionally (may not be installed)
try:
    from feast import FeatureStore
    from feast.errors import FeastError

    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    FeatureStore = None
    FeastError = Exception

# Import CanonicalFeatureBuilder as fallback (SSOT)
from .builders import CanonicalFeatureBuilder, BuilderContext

logger = logging.getLogger(__name__)


# =============================================================================
# MARKET HOURS CONFIGURATION (V7.1 Hybrid Logic)
# =============================================================================

# Colombia timezone for market hours check
COT_TIMEZONE = ZoneInfo("America/Bogota")

# USD/COP trading hours (Colombian stock exchange)
MARKET_OPEN_HOUR = 8   # 08:00 COT
MARKET_CLOSE_HOUR = 17  # 17:00 COT (extended session)

# Days when market is open (Monday=0, Sunday=6)
MARKET_OPEN_DAYS = {0, 1, 2, 3, 4}  # Monday-Friday


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if current time is within Colombian trading hours.

    V7.1 Hybrid Logic:
    - During market hours: Use PostgreSQL (fresh data via NOTIFY)
    - Off-market hours: Use Redis (cached data is acceptable)

    Args:
        dt: Datetime to check (default: current UTC time)

    Returns:
        True if within trading hours, False otherwise
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    # Convert to Colombia timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    cot_time = dt.astimezone(COT_TIMEZONE)

    # Check if weekday
    if cot_time.weekday() not in MARKET_OPEN_DAYS:
        return False

    # Check if within trading hours
    current_hour = cot_time.hour
    return MARKET_OPEN_HOUR <= current_hour < MARKET_CLOSE_HOUR


# =============================================================================
# EXCEPTIONS
# =============================================================================

class FeastServiceError(RuntimeError):
    """Base exception for Feast service errors."""
    pass


class FeastConnectionError(FeastServiceError):
    """Raised when unable to connect to Feast online store."""
    pass


class FeastFeatureNotFoundError(FeastServiceError):
    """Raised when requested features are not found in Feast."""
    pass


# =============================================================================
# METRICS DATACLASS
# =============================================================================

@dataclass
class FeastMetrics:
    """Metrics for Feast service performance monitoring."""
    # Request counts
    total_requests: int = 0
    feast_hits: int = 0
    fallback_hits: int = 0
    errors: int = 0

    # Latency tracking (in milliseconds)
    feast_latencies: List[float] = field(default_factory=list)
    fallback_latencies: List[float] = field(default_factory=list)

    # Max samples to keep for latency calculation
    MAX_LATENCY_SAMPLES: int = 1000

    def record_feast_hit(self, latency_ms: float) -> None:
        """Record a successful Feast retrieval."""
        self.total_requests += 1
        self.feast_hits += 1
        self._add_latency(self.feast_latencies, latency_ms)

    def record_fallback_hit(self, latency_ms: float) -> None:
        """Record a fallback to CanonicalFeatureBuilder."""
        self.total_requests += 1
        self.fallback_hits += 1
        self._add_latency(self.fallback_latencies, latency_ms)

    def record_error(self) -> None:
        """Record an error."""
        self.total_requests += 1
        self.errors += 1

    def _add_latency(self, latencies: List[float], value: float) -> None:
        """Add latency sample, maintaining max size."""
        latencies.append(value)
        if len(latencies) > self.MAX_LATENCY_SAMPLES:
            latencies.pop(0)

    @property
    def feast_hit_rate(self) -> float:
        """Calculate Feast hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.feast_hits / self.total_requests

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate."""
        if self.total_requests == 0:
            return 0.0
        return self.fallback_hits / self.total_requests

    @property
    def avg_feast_latency_ms(self) -> float:
        """Average Feast latency in milliseconds."""
        if not self.feast_latencies:
            return 0.0
        return sum(self.feast_latencies) / len(self.feast_latencies)

    @property
    def avg_fallback_latency_ms(self) -> float:
        """Average fallback latency in milliseconds."""
        if not self.fallback_latencies:
            return 0.0
        return sum(self.fallback_latencies) / len(self.fallback_latencies)

    @property
    def p95_feast_latency_ms(self) -> float:
        """95th percentile Feast latency in milliseconds."""
        if not self.feast_latencies:
            return 0.0
        return float(np.percentile(self.feast_latencies, 95))

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            "total_requests": self.total_requests,
            "feast_hits": self.feast_hits,
            "fallback_hits": self.fallback_hits,
            "errors": self.errors,
            "feast_hit_rate": round(self.feast_hit_rate, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "avg_feast_latency_ms": round(self.avg_feast_latency_ms, 2),
            "avg_fallback_latency_ms": round(self.avg_fallback_latency_ms, 2),
            "p95_feast_latency_ms": round(self.p95_feast_latency_ms, 2),
        }


# =============================================================================
# SSOT Configuration Loading (Module Level)
# =============================================================================
# Load feature order from pipeline_ssot.yaml at module import time
# This avoids circular imports and ensures consistency

def _load_ssot_feature_config():
    """Load feature config from SSOT, with fallback to legacy."""
    try:
        from src.config.pipeline_config import load_pipeline_config
        config = load_pipeline_config()
        return config.get_observation_dim(), config.get_feature_order()
    except Exception as e:
        logger.warning(f"[SSOT] Failed to load pipeline_config: {e}, using legacy")
        return 15, (
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized",
        )

# Load at module import time
_SSOT_DIM, _SSOT_ORDER = _load_ssot_feature_config()


# =============================================================================
# FEAST INFERENCE SERVICE
# =============================================================================

class FeastInferenceService:
    """
    Feature retrieval service using Feast with fallback to CanonicalFeatureBuilder.

    This service provides:
    - Low-latency feature retrieval from Feast online store (Redis)
    - Automatic fallback to CanonicalFeatureBuilder when Feast is unavailable
    - Metrics tracking for monitoring and alerting
    - Async support for non-blocking operations
    - Health check endpoints

    Usage:
        # Initialize
        service = FeastInferenceService(
            feast_repo_path="/path/to/feature_repo",
            fallback_builder=CanonicalFeatureBuilder.for_inference()
        )

        # Get features
        features = service.get_features(
            symbol="USD/COP",
            bar_id="20250117_130500",
            position=0.5
        )

        # Check health
        health = service.health_check()
        print(health["feast_available"])

        # Get metrics
        metrics = service.get_metrics()
        print(metrics["feast_hit_rate"])

    Architecture:
        1. Try Feast online store first (Redis)
        2. If Feast fails, use fallback builder (SSOT)
        3. Track metrics for both paths
        4. Ensure feature order matches SSOT contract
    """

    # Feature service name in Feast
    FEATURE_SERVICE_NAME: Final[str] = "observation_15d"

    # ==========================================================================
    # SSOT Integration (v2.0) - Uses module-level loaded config
    # ==========================================================================
    # Observation dimension and feature order loaded from pipeline_ssot.yaml
    # at module import time (see _load_ssot_feature_config above)
    # ==========================================================================
    OBSERVATION_DIM: Final[int] = _SSOT_DIM
    FEATURE_ORDER: Final[Tuple[str, ...]] = _SSOT_ORDER

    def __init__(
        self,
        feast_repo_path: Optional[str] = None,
        fallback_builder: Optional[CanonicalFeatureBuilder] = None,
        enable_metrics: bool = True,
        enable_fallback: bool = True,
        enable_hybrid_mode: bool = True,
        postgres_conn_string: Optional[str] = None,
    ):
        """
        Initialize FeastInferenceService.

        Args:
            feast_repo_path: Path to Feast feature repository
            fallback_builder: CanonicalFeatureBuilder instance for fallback
            enable_metrics: Enable metrics collection
            enable_fallback: Enable fallback to builder when Feast unavailable
            enable_hybrid_mode: V7.1 Hybrid mode (PostgreSQL during market, Redis off-market)
            postgres_conn_string: PostgreSQL connection string for hybrid mode
        """
        self._feast_repo_path = feast_repo_path or self._find_feast_repo()
        self._enable_metrics = enable_metrics
        self._enable_fallback = enable_fallback
        self._enable_hybrid_mode = enable_hybrid_mode
        self._postgres_conn_string = postgres_conn_string or os.environ.get(
            "DATABASE_URL",
            os.environ.get("TIMESCALE_URL")
        )

        # Initialize Feast store (Redis backend)
        self._feast_store: Optional[FeatureStore] = None
        self._feast_available = False
        self._initialize_feast()

        # Initialize PostgreSQL connection for hybrid mode
        self._postgres_conn = None
        if enable_hybrid_mode and self._postgres_conn_string:
            self._initialize_postgres()

        # Initialize fallback builder
        if fallback_builder:
            self._fallback_builder = fallback_builder
        elif enable_fallback:
            try:
                self._fallback_builder = CanonicalFeatureBuilder.for_inference()
            except Exception as e:
                logger.warning(f"Failed to initialize fallback builder: {e}")
                self._fallback_builder = None
        else:
            self._fallback_builder = None

        # Initialize metrics
        self._metrics = FeastMetrics() if enable_metrics else None

        logger.info(
            f"FeastInferenceService initialized: "
            f"feast_available={self._feast_available}, "
            f"hybrid_mode={self._enable_hybrid_mode}, "
            f"fallback_enabled={self._fallback_builder is not None}"
        )

    def _find_feast_repo(self) -> str:
        """Find Feast repository path relative to project root."""
        # Try common locations
        candidates = [
            "feature_repo",
            "../feature_repo",
            "../../feature_repo",
        ]

        project_root = Path(__file__).parent.parent.parent

        for candidate in candidates:
            path = project_root / candidate
            if path.exists() and (path / "feature_store.yaml").exists():
                return str(path)

        # Default path
        return str(project_root / "feature_repo")

    def _initialize_feast(self) -> None:
        """Initialize Feast FeatureStore connection (Redis backend)."""
        if not FEAST_AVAILABLE:
            logger.warning("Feast is not installed. Fallback mode only.")
            self._feast_available = False
            return

        try:
            self._feast_store = FeatureStore(repo_path=self._feast_repo_path)
            # Test connection by getting feature service
            self._feast_store.get_feature_service(self.FEATURE_SERVICE_NAME)
            self._feast_available = True
            logger.info(f"Feast (Redis) initialized from {self._feast_repo_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize Feast: {e}. Fallback mode enabled.")
            self._feast_available = False

    def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL connection for hybrid mode (V7.1)."""
        try:
            import psycopg2
            self._postgres_conn = psycopg2.connect(self._postgres_conn_string)
            logger.info("PostgreSQL connection established for hybrid mode")
        except ImportError:
            logger.warning("psycopg2 not installed. Hybrid mode disabled.")
            self._enable_hybrid_mode = False
        except Exception as e:
            logger.warning(f"Failed to connect to PostgreSQL: {e}. Hybrid mode disabled.")
            self._enable_hybrid_mode = False

    def _get_from_postgres(
        self,
        symbol: str,
        bar_id: str,
        position: float,
        time_normalized: Optional[float] = None,
    ) -> np.ndarray:
        """
        V7.1 Hybrid: Retrieve features directly from PostgreSQL during market hours.

        This provides fresher data than Redis cache during active trading.
        """
        if self._postgres_conn is None or self._postgres_conn.closed:
            self._initialize_postgres()
            if self._postgres_conn is None:
                raise FeastConnectionError("PostgreSQL connection not available")

        try:
            cur = self._postgres_conn.cursor()

            # Get latest features from inference_features_5m
            cur.execute("""
                SELECT
                    log_ret_5m, log_ret_1h, log_ret_4h,
                    rsi_9, atr_pct, adx_14,
                    dxy_z, dxy_change_1d, vix_z, embi_z,
                    brent_change_1d, rate_spread, usdmxn_change_1d
                FROM inference_features_5m
                WHERE time <= NOW()
                ORDER BY time DESC
                LIMIT 1
            """)

            row = cur.fetchone()
            cur.close()

            if row is None:
                raise FeastFeatureNotFoundError("No features found in PostgreSQL")

            # Build observation array in SSOT order
            observation = np.zeros(self.OBSERVATION_DIM, dtype=np.float32)

            # Features from PostgreSQL (indices 0-12)
            for i, value in enumerate(row):
                observation[i] = 0.0 if value is None else float(value)

            # Position (index 13) - from function argument
            observation[13] = float(np.clip(position, -1.0, 1.0))

            # Time normalized (index 14)
            if time_normalized is not None:
                observation[14] = float(np.clip(time_normalized, 0.0, 1.0))
            else:
                observation[14] = 0.5  # Default to middle of session

            # Validate
            if np.isnan(observation).any():
                observation = np.nan_to_num(observation, nan=0.0)

            return observation

        except Exception as e:
            # Reset connection on error
            if self._postgres_conn:
                try:
                    self._postgres_conn.close()
                except:
                    pass
                self._postgres_conn = None
            raise FeastFeatureNotFoundError(f"PostgreSQL query failed: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_features(
        self,
        symbol: str,
        bar_id: str,
        position: float,
        ohlcv_df: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
        bar_idx: Optional[int] = None,
        time_normalized: Optional[float] = None,
    ) -> np.ndarray:
        """
        Get features for inference.

        This method tries Feast first, then falls back to CanonicalFeatureBuilder.

        Args:
            symbol: Trading symbol (e.g., "USD/COP")
            bar_id: Bar identifier (e.g., "20250117_130500")
            position: Current position [-1, 1]
            ohlcv_df: OHLCV DataFrame (required for fallback)
            macro_df: Macro DataFrame (required for fallback)
            bar_idx: Bar index (required for fallback)
            time_normalized: Normalized trading session time [0, 1]
                            Maps to SSOT feature index 14 'time_normalized'

        Returns:
            np.ndarray of shape (15,) with normalized features

        Raises:
            FeastServiceError: If both Feast and fallback fail
        """
        start_time = time.perf_counter()

        # V7.1 Hybrid Mode: Use PostgreSQL during market hours for fresh data
        if self._enable_hybrid_mode and is_market_hours():
            try:
                features = self._get_from_postgres(symbol, bar_id, position, time_normalized)
                latency_ms = (time.perf_counter() - start_time) * 1000

                if self._metrics:
                    self._metrics.record_feast_hit(latency_ms)  # Count as hit for now

                logger.debug(f"PostgreSQL (hybrid) hit: {symbol}/{bar_id} in {latency_ms:.2f}ms")
                return features

            except Exception as e:
                logger.warning(f"PostgreSQL (hybrid) failed: {e}. Trying Feast (Redis)...")

        # Try Feast (Redis) - primary during off-market or if PostgreSQL fails
        if self._feast_available and self._feast_store is not None:
            try:
                features = self._get_from_feast(symbol, bar_id, position, time_normalized)
                latency_ms = (time.perf_counter() - start_time) * 1000

                if self._metrics:
                    self._metrics.record_feast_hit(latency_ms)

                logger.debug(f"Feast (Redis) hit: {symbol}/{bar_id} in {latency_ms:.2f}ms")
                return features

            except Exception as e:
                logger.warning(f"Feast retrieval failed: {e}. Trying fallback...")

        # Fallback to CanonicalFeatureBuilder
        if self._enable_fallback and self._fallback_builder is not None:
            try:
                features = self._get_from_fallback(
                    ohlcv_df=ohlcv_df,
                    macro_df=macro_df,
                    position=position,
                    bar_idx=bar_idx,
                    time_normalized=time_normalized,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                if self._metrics:
                    self._metrics.record_fallback_hit(latency_ms)

                logger.debug(f"Fallback hit: {symbol}/{bar_id} in {latency_ms:.2f}ms")
                return features

            except Exception as e:
                logger.error(f"Fallback retrieval failed: {e}")
                if self._metrics:
                    self._metrics.record_error()
                raise FeastServiceError(f"Both Feast and fallback failed: {e}")

        # Both failed
        if self._metrics:
            self._metrics.record_error()
        raise FeastServiceError(
            "No feature source available. "
            f"Feast available: {self._feast_available}, "
            f"Fallback enabled: {self._enable_fallback}"
        )

    async def get_features_async(
        self,
        symbol: str,
        bar_id: str,
        position: float,
        ohlcv_df: Optional[pd.DataFrame] = None,
        macro_df: Optional[pd.DataFrame] = None,
        bar_idx: Optional[int] = None,
        time_normalized: Optional[float] = None,
    ) -> np.ndarray:
        """
        Async version of get_features.

        Runs the synchronous get_features in a thread pool to avoid blocking.

        Args:
            symbol: Trading symbol
            bar_id: Bar identifier
            position: Current position
            ohlcv_df: OHLCV DataFrame for fallback
            macro_df: Macro DataFrame for fallback
            bar_idx: Bar index for fallback
            time_normalized: Normalized trading session time [0, 1]

        Returns:
            np.ndarray of shape (15,) with normalized features
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_features(
                symbol=symbol,
                bar_id=bar_id,
                position=position,
                ohlcv_df=ohlcv_df,
                macro_df=macro_df,
                bar_idx=bar_idx,
                time_normalized=time_normalized,
            ),
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Check health of the Feast service.

        Returns:
            Dict with health status and metrics
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "feast_installed": FEAST_AVAILABLE,
            "feast_available": self._feast_available,
            "fallback_enabled": self._enable_fallback,
            "fallback_builder_ready": self._fallback_builder is not None,
            # V7.1 Hybrid Mode
            "hybrid_mode_enabled": self._enable_hybrid_mode,
            "is_market_hours": is_market_hours(),
            "active_backend": "postgresql" if (self._enable_hybrid_mode and is_market_hours()) else "redis",
            "postgres_connected": self._postgres_conn is not None and not self._postgres_conn.closed if self._postgres_conn else False,
        }

        # Test Feast connection if available
        if self._feast_available and self._feast_store is not None:
            try:
                self._feast_store.get_feature_service(self.FEATURE_SERVICE_NAME)
                health["feast_connection"] = "ok"
            except Exception as e:
                health["feast_connection"] = f"error: {e}"
                health["status"] = "degraded"
        else:
            health["feast_connection"] = "unavailable"
            if not self._fallback_builder:
                health["status"] = "unhealthy"

        # Add metrics if enabled
        if self._metrics:
            health["metrics"] = self._metrics.to_dict()

        return health

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.

        Returns:
            Dict with performance metrics
        """
        if not self._metrics:
            return {"metrics_enabled": False}

        return {
            "metrics_enabled": True,
            **self._metrics.to_dict(),
        }

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        if self._metrics:
            self._metrics = FeastMetrics()

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_from_feast(
        self,
        symbol: str,
        bar_id: str,
        position: float,
        time_normalized: Optional[float] = None,
    ) -> np.ndarray:
        """
        Retrieve features from Feast online store.

        Args:
            symbol: Trading symbol
            bar_id: Bar identifier
            position: Current position
            time_normalized: Normalized trading session time [0, 1]

        Returns:
            np.ndarray of shape (15,)

        Raises:
            FeastFeatureNotFoundError: If features not found
            FeastConnectionError: If connection fails
        """
        if self._feast_store is None:
            raise FeastConnectionError("Feast store not initialized")

        # Build entity dict for lookup
        entity_dict = {
            "symbol": [symbol],
            "bar_id": [bar_id],
        }

        try:
            # Get features from online store
            feature_vector = self._feast_store.get_online_features(
                features=self._feast_store.get_feature_service(self.FEATURE_SERVICE_NAME),
                entity_rows=[{"symbol": symbol, "bar_id": bar_id}],
            ).to_dict()

            # Extract values in correct order
            observation = np.zeros(self.OBSERVATION_DIM, dtype=np.float32)

            for idx, name in enumerate(self.FEATURE_ORDER):
                if name == "position":
                    observation[idx] = float(np.clip(position, -1.0, 1.0))
                elif name == "time_normalized":
                    if time_normalized is not None:
                        observation[idx] = float(np.clip(time_normalized, 0.0, 1.0))
                    elif f"{name}" in feature_vector:
                        value = feature_vector[f"{name}"][0]
                        observation[idx] = 0.5 if value is None else float(value)
                    else:
                        observation[idx] = 0.5  # Default to middle of session
                else:
                    # Get from Feast (features are prefixed by view name)
                    feast_key = self._get_feast_key(name)
                    if feast_key in feature_vector:
                        value = feature_vector[feast_key][0]
                        observation[idx] = 0.0 if value is None else float(value)
                    else:
                        logger.warning(f"Feature {name} not found in Feast response")
                        observation[idx] = 0.0

            # Validate observation
            if np.isnan(observation).any():
                nan_indices = np.where(np.isnan(observation))[0]
                logger.warning(f"NaN values at indices {nan_indices}, replacing with 0")
                observation = np.nan_to_num(observation, nan=0.0)

            return observation

        except Exception as e:
            raise FeastFeatureNotFoundError(f"Failed to get features from Feast: {e}")

    def _get_from_fallback(
        self,
        ohlcv_df: Optional[pd.DataFrame],
        macro_df: Optional[pd.DataFrame],
        position: float,
        bar_idx: Optional[int],
        time_normalized: Optional[float] = None,
    ) -> np.ndarray:
        """
        Retrieve features using fallback CanonicalFeatureBuilder.

        Args:
            ohlcv_df: OHLCV DataFrame
            macro_df: Macro DataFrame
            position: Current position
            bar_idx: Bar index
            time_normalized: Normalized trading session time [0, 1]

        Returns:
            np.ndarray of shape (15,)

        Raises:
            ValueError: If required data is missing
        """
        if self._fallback_builder is None:
            raise ValueError("Fallback builder not available")

        if ohlcv_df is None or bar_idx is None:
            raise ValueError("OHLCV DataFrame and bar_idx required for fallback")

        return self._fallback_builder.build_observation(
            ohlcv=ohlcv_df,
            macro=macro_df,
            position=position,
            bar_idx=bar_idx,
            time_normalized=time_normalized,
        )

    def _get_feast_key(self, feature_name: str) -> str:
        """
        Get Feast key for a feature name.

        Feast prefixes features with view name: <view_name>:<feature_name>
        """
        # Map feature to view
        if feature_name in ("log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14"):
            return f"technical_features:{feature_name}"
        elif feature_name in ("dxy_z", "dxy_change_1d", "vix_z", "embi_z", "brent_change_1d", "rate_spread", "usdmxn_change_1d"):
            return f"macro_features:{feature_name}"
        elif feature_name in ("position", "time_normalized"):
            return f"state_features:{feature_name}"
        else:
            return feature_name


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_feast_service(
    feast_repo_path: Optional[str] = None,
    fallback_norm_stats_path: Optional[str] = None,
    enable_metrics: bool = True,
) -> FeastInferenceService:
    """
    Factory function to create FeastInferenceService.

    Args:
        feast_repo_path: Path to Feast repository
        fallback_norm_stats_path: Path to norm_stats.json for fallback
        enable_metrics: Enable metrics collection

    Returns:
        Configured FeastInferenceService
    """
    # Create fallback builder
    try:
        fallback_builder = CanonicalFeatureBuilder.for_inference(
            norm_stats_path=fallback_norm_stats_path
        )
    except Exception as e:
        logger.warning(f"Failed to create fallback builder: {e}")
        fallback_builder = None

    return FeastInferenceService(
        feast_repo_path=feast_repo_path,
        fallback_builder=fallback_builder,
        enable_metrics=enable_metrics,
    )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feast Inference Service")
    parser.add_argument("--health", action="store_true", help="Check service health")
    parser.add_argument("--metrics", action="store_true", help="Show metrics")
    args = parser.parse_args()

    # Initialize service
    service = create_feast_service()

    if args.health:
        health = service.health_check()
        import json
        print(json.dumps(health, indent=2))

    elif args.metrics:
        metrics = service.get_metrics()
        import json
        print(json.dumps(metrics, indent=2))

    else:
        print("Feast Inference Service")
        print("=" * 40)
        health = service.health_check()
        print(f"Status: {health['status']}")
        print(f"Feast Available: {health['feast_available']}")
        print(f"Fallback Ready: {health['fallback_builder_ready']}")
