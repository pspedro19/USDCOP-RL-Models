"""
Reward Components Module.

Modular reward calculation components for RL training.
Each component implements the RewardComponent base class
for consistent interface and DRY principle.

Contract: CTR-REWARD-MODULE-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Components:
-----------
Risk Metrics:
    - DifferentialSharpeRatio: Incremental Sharpe for RL (Moody & Saffell, 2001)
    - SortinoCalculator: Rolling Sortino ratio focusing on downside risk

Detectors:
    - StableRegimeDetector: Market regime classification (LOW_VOL, NORMAL, HIGH_VOL, CRISIS)
    - BanrepInterventionDetector: Central bank intervention detection proxy
    - OilCorrelationTracker: Oil-COP correlation monitoring

Market Impact:
    - AlmgrenChrissImpactModel: Non-linear slippage model (Almgren-Chriss, 2001)

Penalties:
    - HoldingDecay: Exponential holding time penalty
    - GapRiskPenalty: Overnight/weekend gap risk penalty
    - InactivityTracker: Excessive flatness penalty
    - ChurnTracker: Excessive trading penalty
    - ActionCorrelationTracker: Gaming behavior detection
    - BiasDetector: Directional bias detection

Transforms:
    - LogPnLTransform: Log compression for outliers
    - AsymmetricPnLTransform: Different scaling for wins/losses
    - ClippedPnLTransform: Hard clipping
    - RankPnLTransform: Percentile rank transformation
    - ZScorePnLTransform: Z-score normalization
    - CompositePnLTransform: Chain multiple transforms

Normalizers:
    - RewardNormalizer: FinRL-Meta style running mean/std normalization
    - RunningMeanStd: Helper class for running statistics

Usage:
------
    from src.training.reward_components import (
        DifferentialSharpeRatio,
        SortinoCalculator,
        StableRegimeDetector,
        AlmgrenChrissImpactModel,
        RewardNormalizer,
    )

    # Create components
    dsr = DifferentialSharpeRatio(eta=0.01)
    sortino = SortinoCalculator(window_size=20)
    regime = StableRegimeDetector()
    impact = AlmgrenChrissImpactModel()
    normalizer = RewardNormalizer(warmup_steps=1000)

    # Calculate reward
    dsr_value = dsr.calculate(return_pct=0.001)
    sortino_value = sortino.calculate(return_pct=0.001)
    regime.update(volatility=0.02)
    cost = impact.calculate(hour_utc=15, regime="NORMAL", volatility=0.02)
    normalized = normalizer.calculate(reward=total_reward)
"""

# =============================================================================
# BASE CLASSES AND TYPES
# =============================================================================

from .base import (
    # Enums
    MarketRegime,
    ComponentType,
    # Dataclasses
    ComponentResult,
    ComponentStats,
    # Base class
    RewardComponent,
    # Protocols
    IRewardCalculator,
    IRegimeDetector,
    IRewardNormalizer,
    ICostModel,
    IDecayModel,
    # Helper functions
    clip_reward,
    safe_divide,
    exponential_decay,
    z_score,
)

# =============================================================================
# RISK METRICS
# =============================================================================

from .dsr import DifferentialSharpeRatio
from .sortino import SortinoCalculator

# =============================================================================
# DETECTORS
# =============================================================================

from .regime_detector import StableRegimeDetector
from .banrep_detector import BanrepInterventionDetector, InterventionStatus
from .oil_tracker import (
    OilCorrelationTracker,
    OilCorrelationState,
    OilMomentumSignal,
)

# =============================================================================
# MARKET IMPACT
# =============================================================================

from .market_impact import AlmgrenChrissImpactModel, MarketImpactResult

# =============================================================================
# PENALTIES
# =============================================================================

from .holding_decay import HoldingDecay, GapRiskPenalty
from .anti_gaming import (
    InactivityTracker,
    ChurnTracker,
    ActionCorrelationTracker,
    BiasDetector,
)
from .flat_reward import FlatReward, FlatRewardConfig
from .close_reason_detector import CloseReasonDetector
from .drawdown_penalty import DrawdownPenaltyComponent
from .execution_alpha import ExecutionAlphaComponent

# =============================================================================
# TRANSFORMS
# =============================================================================

from .pnl_transforms import (
    LogPnLTransform,
    AsymmetricPnLTransform,
    ClippedPnLTransform,
    RankPnLTransform,
    ZScorePnLTransform,
    CompositePnLTransform,
    create_default_pnl_transform,
)

# =============================================================================
# NORMALIZERS
# =============================================================================

from .reward_normalizer import RewardNormalizer, RunningMeanStd

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes and types
    "MarketRegime",
    "ComponentType",
    "ComponentResult",
    "ComponentStats",
    "RewardComponent",
    "IRewardCalculator",
    "IRegimeDetector",
    "IRewardNormalizer",
    "ICostModel",
    "IDecayModel",
    "clip_reward",
    "safe_divide",
    "exponential_decay",
    "z_score",
    # Risk metrics
    "DifferentialSharpeRatio",
    "SortinoCalculator",
    # Detectors
    "StableRegimeDetector",
    "BanrepInterventionDetector",
    "InterventionStatus",
    "OilCorrelationTracker",
    "OilCorrelationState",
    "OilMomentumSignal",
    # Market impact
    "AlmgrenChrissImpactModel",
    "MarketImpactResult",
    # Penalties
    "HoldingDecay",
    "GapRiskPenalty",
    "InactivityTracker",
    "ChurnTracker",
    "ActionCorrelationTracker",
    "BiasDetector",
    # Bonuses (Phase 2)
    "FlatReward",
    "FlatRewardConfig",
    # Close reason shaping (V22 P2)
    "CloseReasonDetector",
    # Drawdown penalty (Phase 5)
    "DrawdownPenaltyComponent",
    # Execution alpha (EXP-RL-EXECUTOR)
    "ExecutionAlphaComponent",
    # Transforms
    "LogPnLTransform",
    "AsymmetricPnLTransform",
    "ClippedPnLTransform",
    "RankPnLTransform",
    "ZScorePnLTransform",
    "CompositePnLTransform",
    "create_default_pnl_transform",
    # Normalizers
    "RewardNormalizer",
    "RunningMeanStd",
]

# =============================================================================
# VERSION
# =============================================================================

__version__ = "1.0.0"
