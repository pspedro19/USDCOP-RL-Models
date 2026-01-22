"""
Observation Builder for PPO Model
==================================
Builds 15-dimensional observation vectors from market data.

ARCHITECTURE (2025-01-14 Update):
This builder now DELEGATES all calculations to InferenceFeatureAdapter,
which uses the SSOT calculators from src/feature_store/core.py.

This ensures PERFECT PARITY between training and inference by using
the EXACT SAME calculators with Wilder's EMA smoothing for RSI, ATR, ADX.

SOLID Principles:
- Single Responsibility: Only builds observation vectors
- Open/Closed: Extensible via inheritance (V1 builder)
- Dependency Inversion: Delegates to SSOT via adapter

CRITICAL: This builder uses FAIL-FAST pattern.
If norm_stats are not found, it raises an error rather than
using hardcoded defaults that would produce WRONG predictions.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, Optional
from pathlib import Path
from ..config import get_settings, FEATURE_ORDER

# Import from SSOT
try:
    from src.core.constants import OBSERVATION_DIM as SSOT_OBSERVATION_DIM
except ImportError:
    SSOT_OBSERVATION_DIM = 15  # Fallback for isolated testing

# Import the SSOT adapter
from .feature_adapter import (
    InferenceFeatureAdapter,
    FeatureCircuitBreakerError,
    FeatureCircuitBreakerConfig,
)

# Import Feast service for online feature serving (P1-3 remediation)
try:
    from src.feature_store.feast_service import (
        FeastInferenceService,
        create_feast_service,
        FeastServiceError,
    )
    FEAST_SERVICE_AVAILABLE = True
except ImportError:
    FEAST_SERVICE_AVAILABLE = False
    FeastInferenceService = None
    create_feast_service = None
    FeastServiceError = Exception

logger = logging.getLogger(__name__)
settings = get_settings()


class NormStatsNotFoundError(RuntimeError):
    """
    Raised when normalization statistics file is not found.

    This is a CRITICAL error - the model CANNOT produce correct
    predictions without the exact norm_stats used during training.
    """
    pass


class ObservationBuilder:
    """
    Builds observation vectors for PPO model inference.

    Observation space: 15 dimensions
    - 13 core market features (normalized)
    - 2 state features (position, time_normalized)

    ARCHITECTURE:
    This class now DELEGATES to InferenceFeatureAdapter, which uses
    the SSOT calculators from src/feature_store/core.py. This ensures
    that RSI, ATR, and ADX use Wilder's EMA (alpha=1/period) to match
    the training data EXACTLY.

    OPTIONAL FEAST INTEGRATION (P1-3 remediation):
    When Feast is available, can use FeastInferenceService for low-latency
    feature retrieval from Redis online store. Falls back to SSOT adapter.

    FAIL-FAST: This builder will raise NormStatsNotFoundError if
    the normalization statistics file is not found. This is intentional -
    using wrong defaults would produce incorrect predictions.
    """

    # Class-level constant from SSOT (src/core/constants.py)
    OBSERVATION_DIM = SSOT_OBSERVATION_DIM

    def __init__(
        self,
        norm_stats_path: Optional[Path] = None,
        enable_feast: bool = True,
        feast_repo_path: Optional[str] = None,
    ):
        """
        Initialize ObservationBuilder.

        Args:
            norm_stats_path: Path to norm_stats JSON file.
                           If None, uses settings.full_norm_stats_path.
            enable_feast: Enable Feast online store for feature retrieval.
                         Falls back to SSOT adapter when Feast unavailable.
            feast_repo_path: Optional path to Feast repository.

        Raises:
            NormStatsNotFoundError: If norm_stats file not found.
                                   NO DEFAULTS - this is intentional.
        """
        # Initialize the SSOT adapter for feature calculations (always available)
        norm_path = str(norm_stats_path) if norm_stats_path else None
        try:
            self._adapter = InferenceFeatureAdapter(norm_stats_path=norm_path)
        except FileNotFoundError as e:
            raise NormStatsNotFoundError(str(e))

        # Initialize Feast service if available and enabled (P1-3 remediation)
        self._feast_service: Optional[FeastInferenceService] = None
        self._feast_enabled = enable_feast and FEAST_SERVICE_AVAILABLE

        if self._feast_enabled:
            try:
                self._feast_service = create_feast_service(
                    feast_repo_path=feast_repo_path,
                    fallback_norm_stats_path=norm_path,
                    enable_metrics=True,
                )
                logger.info("Feast online store integration enabled")
            except Exception as e:
                logger.warning(f"Feast initialization failed, using SSOT adapter only: {e}")
                self._feast_service = None
                self._feast_enabled = False

        # Keep backward-compatible attributes
        self.norm_stats = self._adapter._norm_stats
        self.feature_order = FEATURE_ORDER
        self.clip_range = self._adapter.CLIP_RANGE
        logger.info(
            f"ObservationBuilder initialized with SSOT adapter "
            f"({len(self.norm_stats)} features, Wilder's EMA enabled, "
            f"Feast={self._feast_enabled})"
        )

    def _load_norm_stats_strict(self, path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
        """
        DEPRECATED: Now handled by InferenceFeatureAdapter.

        This method is kept for backward compatibility but delegates to adapter.
        """
        # This method is now a no-op - stats are loaded by adapter
        return self._adapter._norm_stats if hasattr(self, '_adapter') else {}

    def normalize_feature(self, value: float, feature_name: str) -> float:
        """
        Z-score normalize a single feature.

        DELEGATES TO: InferenceFeatureAdapter.normalize_feature()
        """
        return self._adapter.normalize_feature(value, feature_name)

    def calculate_technical_features(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        lookback: int = 50
    ) -> Dict[str, float]:
        """
        Calculate technical indicators for a given bar.

        DELEGATES TO: InferenceFeatureAdapter.calculate_technical_features()

        CRITICAL: Uses SSOT calculators with Wilder's EMA smoothing
        for RSI, ATR, and ADX to ensure PERFECT PARITY with training.

        Args:
            df: DataFrame with OHLCV data
            bar_idx: Current bar index
            lookback: Number of bars to look back for calculations

        Returns:
            Dict of technical indicator values
        """
        return self._adapter.calculate_technical_features(df, bar_idx)

    # ==========================================================================
    # DEPRECATED METHODS - Kept for reference but NO LONGER USED
    # ==========================================================================
    # The following methods used SIMPLE MOVING AVERAGE instead of Wilder's EMA.
    # They have been replaced by SSOT calculators via InferenceFeatureAdapter.
    #
    # OLD (INCORRECT):
    #   avg_gain = np.mean(gains)  # Simple SMA
    #   avg_loss = np.mean(losses)  # Simple SMA
    #
    # NEW (CORRECT via adapter):
    #   avg_gains = ewm(alpha=1/period)  # Wilder's EMA
    #   avg_losses = ewm(alpha=1/period)  # Wilder's EMA
    # ==========================================================================

    def _calculate_rsi(self, prices: np.ndarray, period: int = 9) -> float:
        """
        DEPRECATED: Use InferenceFeatureAdapter instead.

        This method used simple SMA for averaging gains/losses.
        The correct method (Wilder's EMA) is now in feature_adapter.py.
        """
        logger.warning(
            "_calculate_rsi is DEPRECATED. "
            "Using SSOT RSICalculator with Wilder's EMA via adapter."
        )
        # Create temporary DataFrame and delegate to adapter
        df = pd.DataFrame({"close": prices})
        return self._adapter.calculate_technical_features(df, len(prices) - 1).get("rsi_9", 50.0)

    def _calculate_atr_pct(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 10
    ) -> float:
        """
        DEPRECATED: Use InferenceFeatureAdapter instead.

        This method used simple SMA for averaging TR values.
        The correct method (Wilder's EMA) is now in feature_adapter.py.
        """
        logger.warning(
            "_calculate_atr_pct is DEPRECATED. "
            "Using SSOT ATRPercentCalculator with Wilder's EMA via adapter."
        )
        # Create temporary DataFrame and delegate to adapter
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        return self._adapter.calculate_technical_features(df, len(close) - 1).get("atr_pct", 0.0)

    def _calculate_adx(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """
        DEPRECATED: Use InferenceFeatureAdapter instead.

        This method used simple SMA for averaging DM values.
        The correct method (Wilder's EMA) is now in feature_adapter.py.
        """
        logger.warning(
            "_calculate_adx is DEPRECATED. "
            "Using SSOT ADXCalculator with Wilder's EMA via adapter."
        )
        # Create temporary DataFrame and delegate to adapter
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        return self._adapter.calculate_technical_features(df, len(close) - 1).get("adx_14", 25.0)

    def calculate_macro_features(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Calculate macro indicator features.

        NOTE: This method is kept for backward compatibility.
        The adapter uses SSOT calculators when operating on full DataFrames.

        Args:
            row: Current row with macro data
            prev_row: Previous day's row for change calculations

        Returns:
            Dict of macro feature values
        """
        # Z-score macro values (already in DB as raw values)
        dxy = row.get("dxy", 100.0) or 100.0
        vix = row.get("vix", 20.0) or 20.0
        embi = row.get("embi", 300.0) or 300.0
        brent = row.get("brent", 80.0) or 80.0
        treasury_10y = row.get("treasury_10y", 4.0) or 4.0
        usdmxn = row.get("usdmxn", 17.0) or 17.0

        # Calculate daily changes
        if prev_row is not None:
            prev_dxy = prev_row.get("dxy", dxy) or dxy
            prev_brent = prev_row.get("brent", brent) or brent
            prev_usdmxn = prev_row.get("usdmxn", usdmxn) or usdmxn

            dxy_change = (dxy - prev_dxy) / prev_dxy if prev_dxy > 0 else 0.0
            brent_change = (brent - prev_brent) / prev_brent if prev_brent > 0 else 0.0
            usdmxn_change = (usdmxn - prev_usdmxn) / prev_usdmxn if prev_usdmxn > 0 else 0.0
        else:
            dxy_change = 0.0
            brent_change = 0.0
            usdmxn_change = 0.0

        # Rate spread (10Y - assumed base rate)
        rate_spread = 10.0 - treasury_10y

        return {
            "dxy_z": dxy,
            "dxy_change_1d": dxy_change,
            "vix_z": vix,
            "embi_z": embi,
            "brent_change_1d": brent_change,
            "rate_spread": rate_spread,
            "usdmxn_change_1d": usdmxn_change,
        }

    def build_observation(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        position: float,
        time_normalized: float = 0.5,
        symbol: str = "USD/COP",
        bar_id: Optional[str] = None,
        use_feast: bool = True,
    ) -> np.ndarray:
        """
        Build complete 15-dimensional observation vector.

        ARCHITECTURE (P1-3 remediation):
        1. If Feast enabled and bar_id provided, try Feast online store first
        2. If Feast fails or not available, delegate to SSOT adapter

        CRITICAL: Uses SSOT calculators with Wilder's EMA smoothing
        for RSI, ATR, and ADX to ensure PERFECT PARITY with training.

        Args:
            df: DataFrame with OHLCV and macro data
            bar_idx: Current bar index
            position: Current position (-1 to 1)
            time_normalized: Normalized trading session time (0 to 1)
                            Maps to SSOT feature index 14 'time_normalized'
            symbol: Trading symbol (default "USD/COP")
            bar_id: Bar identifier for Feast lookup (e.g., "20250117_130500")
            use_feast: Whether to try Feast first (default True)

        Returns:
            numpy array of shape (15,)

        Raises:
            FeatureCircuitBreakerError: If data quality is too low (>20% NaN)
        """
        # Try Feast first if enabled and bar_id provided
        if use_feast and self._feast_service is not None and bar_id is not None:
            try:
                return self._feast_service.get_features(
                    symbol=symbol,
                    bar_id=bar_id,
                    position=position,
                    ohlcv_df=df,  # For fallback
                    macro_df=df,  # For fallback
                    bar_idx=bar_idx,  # For fallback
                    time_normalized=time_normalized,
                )
            except Exception as e:
                logger.debug(f"Feast retrieval failed, using adapter: {e}")

        # Fall back to SSOT adapter
        return self._adapter.build_observation(
            df=df,
            bar_idx=bar_idx,
            position=position,
            time_normalized=time_normalized,
            check_circuit_breaker=True
        )

    def get_feast_health(self) -> Dict:
        """Get Feast service health status (P1-3 remediation)."""
        if self._feast_service is None:
            return {
                "feast_enabled": False,
                "feast_available": FEAST_SERVICE_AVAILABLE,
                "status": "disabled",
            }
        return self._feast_service.health_check()

    def get_feast_metrics(self) -> Dict:
        """Get Feast service metrics (P1-3 remediation)."""
        if self._feast_service is None:
            return {"feast_enabled": False, "metrics_available": False}
        return self._feast_service.get_metrics()
