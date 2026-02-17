"""
Modular Reward Calculator v2.0 - Component-Based Architecture.

Complete rewrite using modular components:
- DSR (Differential Sharpe Ratio)
- Sortino Ratio
- Market Impact (Almgren-Chriss)
- Regime Detection
- Holding Decay
- Anti-Gaming Penalties
- PnL Transforms
- Reward Normalization
- Curriculum Learning

Contract: CTR-REWARD-CALCULATOR-002
Author: Trading Team
Version: 2.0.0
Created: 2026-01-19

Design Principles:
- SSOT: All configuration from RewardConfig
- DRY: Reusable components
- Strategy Pattern: Interchangeable reward components
- Dependency Injection: Components injected via config
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

from src.training.config import RewardConfig, REWARD_CONFIG
from src.training.curriculum_scheduler import (
    CurriculumScheduler,
    CurriculumPhase,
    create_curriculum_scheduler,
)
from src.training.reward_components import (
    # Risk metrics
    DifferentialSharpeRatio,
    SortinoCalculator,
    # Detectors
    StableRegimeDetector,
    BanrepInterventionDetector,
    OilCorrelationTracker,
    MarketRegime,
    # Market impact
    AlmgrenChrissImpactModel,
    # Penalties
    HoldingDecay,
    GapRiskPenalty,
    InactivityTracker,
    ChurnTracker,
    BiasDetector,
    # Bonuses (PHASE2)
    FlatReward,
    # V22 P2: Close reason reward shaping
    CloseReasonDetector,
    # Drawdown penalty (Phase 5)
    DrawdownPenaltyComponent,
    # Execution alpha (EXP-RL-EXECUTOR)
    ExecutionAlphaComponent,
    # Transforms
    ZScorePnLTransform,
    AsymmetricPnLTransform,
    ClippedPnLTransform,
    # Normalizers
    RewardNormalizer,
    # Base
    RewardComponent,
    clip_reward,
)

logger = logging.getLogger(__name__)


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward calculation."""
    # Raw inputs
    raw_pnl: float = 0.0
    position: int = 0
    holding_bars: int = 0

    # Transformed PnL
    transformed_pnl: float = 0.0

    # Component contributions
    dsr_contribution: float = 0.0
    sortino_contribution: float = 0.0
    regime_penalty: float = 0.0
    market_impact_cost: float = 0.0
    holding_decay_penalty: float = 0.0
    gap_risk_penalty: float = 0.0
    inactivity_penalty: float = 0.0
    churn_penalty: float = 0.0
    bias_penalty: float = 0.0
    banrep_penalty: float = 0.0
    oil_adjustment: float = 0.0
    flat_reward_bonus: float = 0.0  # PHASE2: Counterfactual HOLD reward
    close_shaping: float = 0.0     # V22 P2: Close reason reward shaping
    drawdown_penalty: float = 0.0  # Phase 5: Drawdown penalty
    execution_alpha: float = 0.0  # EXP-RL-EXECUTOR: Execution alpha

    # Weighted total (before normalization)
    weighted_total: float = 0.0

    # Final outputs
    normalized_reward: float = 0.0
    clipped_reward: float = 0.0

    # Curriculum info
    curriculum_phase: str = ""
    cost_multiplier: float = 1.0

    # Regime info
    current_regime: str = "NORMAL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "raw_pnl": self.raw_pnl,
            "transformed_pnl": self.transformed_pnl,
            "dsr": self.dsr_contribution,
            "sortino": self.sortino_contribution,
            "regime_penalty": self.regime_penalty,
            "market_impact": self.market_impact_cost,
            "holding_decay": self.holding_decay_penalty,
            "gap_risk": self.gap_risk_penalty,
            "anti_gaming": self.inactivity_penalty + self.churn_penalty + self.bias_penalty,
            "banrep": self.banrep_penalty,
            "flat_reward": self.flat_reward_bonus,  # PHASE2
            "weighted_total": self.weighted_total,
            "normalized": self.normalized_reward,
            "final": self.clipped_reward,
            "regime": self.current_regime,
            "curriculum_phase": self.curriculum_phase,
        }


class ModularRewardCalculator:
    """
    Modular reward calculator using component-based architecture.

    Composes multiple reward components with configurable weights.
    Supports curriculum learning for phased training.

    Usage:
        >>> config = RewardConfig()
        >>> calculator = ModularRewardCalculator(config)
        >>> reward, breakdown = calculator.calculate(
        ...     pnl_pct=0.001,
        ...     position=1,
        ...     position_change=1,
        ...     holding_bars=10,
        ...     volatility=0.02,
        ...     hour_utc=15,
        ... )
    """

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        enable_curriculum: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize modular reward calculator.

        Args:
            config: RewardConfig (uses default SSOT if not provided)
            enable_curriculum: Whether to use curriculum learning
            verbose: Enable detailed logging
        """
        self._config = config or REWARD_CONFIG
        self._verbose = verbose

        # Initialize components
        self._init_components()

        # Curriculum scheduler
        self._curriculum: Optional[CurriculumScheduler] = None
        if enable_curriculum and self._config.enable_curriculum:
            self._curriculum = create_curriculum_scheduler(
                self._config,
                adaptive=False,
                callback=self._on_phase_change,
            )

        # Statistics tracking
        self._total_steps = 0
        self._total_trades = 0

    def _init_components(self) -> None:
        """Initialize all reward components from config."""
        cfg = self._config

        # Risk metrics
        self._dsr = DifferentialSharpeRatio(
            eta=cfg.dsr.eta,
            min_samples=cfg.dsr.min_samples,
            scale=cfg.dsr.scale,
        )

        self._sortino = SortinoCalculator(
            window_size=cfg.sortino.window_size,
            target_return=cfg.sortino.target_return,
            min_samples=cfg.sortino.min_samples,
            scale=cfg.sortino.scale,
        )

        # Regime detector
        self._regime_detector = StableRegimeDetector(
            low_vol_percentile=cfg.regime.low_vol_percentile,
            high_vol_percentile=cfg.regime.high_vol_percentile,
            crisis_multiplier=cfg.regime.crisis_multiplier,
            min_stability=cfg.regime.min_stability,
            history_window=cfg.regime.history_window,
            smoothing_window=cfg.regime.smoothing_window,
        )

        # Market impact (Almgren-Chriss)
        self._market_impact = AlmgrenChrissImpactModel(
            permanent_impact_coef=cfg.market_impact.permanent_impact_coef,
            temporary_impact_coef=cfg.market_impact.temporary_impact_coef,
            volatility_impact_coef=cfg.market_impact.volatility_impact_coef,
            adv_base_usd=cfg.market_impact.adv_base_usd,
            typical_order_fraction=cfg.market_impact.typical_order_fraction,
            default_spread_bps=cfg.market_impact.default_spread_bps,
        )

        # Holding decay
        self._holding_decay = HoldingDecay(
            half_life_bars=cfg.holding_decay.half_life_bars,
            max_penalty=cfg.holding_decay.max_penalty,
            flat_threshold=cfg.holding_decay.flat_threshold,
            enable_overnight_boost=cfg.holding_decay.enable_overnight_boost,
            overnight_multiplier=cfg.holding_decay.overnight_multiplier,
        )

        self._gap_risk = GapRiskPenalty()

        # Anti-gaming
        self._inactivity = InactivityTracker(
            grace_period=cfg.anti_gaming.inactivity_grace_period,
            max_penalty=cfg.anti_gaming.inactivity_max_penalty,
            penalty_growth_rate=cfg.anti_gaming.inactivity_growth_rate,
        )

        self._churn = ChurnTracker(
            window_size=cfg.anti_gaming.churn_window_size,
            max_trades_in_window=cfg.anti_gaming.churn_max_trades,
            base_penalty=cfg.anti_gaming.churn_base_penalty,
            excess_trade_penalty=cfg.anti_gaming.churn_excess_penalty,
        )

        self._bias = BiasDetector(
            imbalance_threshold=cfg.anti_gaming.bias_imbalance_threshold,
            bias_penalty=cfg.anti_gaming.bias_penalty,
            min_samples=cfg.anti_gaming.bias_min_samples,
        )

        # Banrep detector (optional)
        self._banrep: Optional[BanrepInterventionDetector] = None
        if cfg.enable_banrep_detection:
            self._banrep = BanrepInterventionDetector(
                volatility_spike_zscore=cfg.banrep.volatility_spike_zscore,
                volatility_baseline_window=cfg.banrep.volatility_baseline_window,
                intervention_penalty=cfg.banrep.intervention_penalty,
                cooldown_bars=cfg.banrep.cooldown_bars,
                reversal_threshold=cfg.banrep.reversal_threshold,
                min_history=cfg.banrep.min_history,
            )

        # Oil correlation (optional)
        self._oil_tracker: Optional[OilCorrelationTracker] = None
        if cfg.enable_oil_tracking:
            self._oil_tracker = OilCorrelationTracker(
                window_size=cfg.oil_correlation.window_size,
                strong_threshold=cfg.oil_correlation.strong_threshold,
                weak_threshold=cfg.oil_correlation.weak_threshold,
                breakdown_penalty=cfg.oil_correlation.breakdown_penalty,
                min_samples=cfg.oil_correlation.min_samples,
            )

        # PnL transforms
        self._pnl_clip = ClippedPnLTransform(
            min_value=cfg.pnl_transform.clip_min,
            max_value=cfg.pnl_transform.clip_max,
        )

        self._pnl_zscore = ZScorePnLTransform(
            window_size=cfg.pnl_transform.zscore_window,
            clip_zscore=cfg.pnl_transform.zscore_clip,
        )

        self._pnl_asymmetric = AsymmetricPnLTransform(
            win_multiplier=cfg.pnl_transform.asymmetric_win_mult,
            loss_multiplier=cfg.pnl_transform.asymmetric_loss_mult,
        )

        # Reward normalizer
        self._normalizer: Optional[RewardNormalizer] = None
        if cfg.enable_normalization:
            self._normalizer = RewardNormalizer(
                decay=cfg.normalizer.decay,
                epsilon=cfg.normalizer.epsilon,
                clip_range=cfg.normalizer.clip_range,
                warmup_steps=cfg.normalizer.warmup_steps,
                per_episode_reset=cfg.normalizer.per_episode_reset,
            )

        # PHASE 3 FIX: Flat reward with decay (anti-reward-hacking)
        self._flat_reward: Optional[FlatReward] = None
        if getattr(cfg, 'enable_flat_reward', False):
            fr_cfg = getattr(cfg, 'flat_reward', None)
            if fr_cfg:
                self._flat_reward = FlatReward(
                    scale=fr_cfg.scale,
                    min_move_threshold=fr_cfg.min_move_threshold,
                    loss_avoidance_mult=fr_cfg.loss_avoidance_mult,
                    decay_enabled=getattr(fr_cfg, 'decay_enabled', True),   # PHASE3: default True
                    decay_half_life=getattr(fr_cfg, 'decay_half_life', 12), # PHASE3: 1 hour
                    decay_max=getattr(fr_cfg, 'decay_max', 0.9),            # PHASE3: 90% max
                )
            else:
                # Use PHASE3 defaults (anti-reward-hacking)
                self._flat_reward = FlatReward(
                    scale=50.0,
                    decay_enabled=True,
                    decay_half_life=12,
                    decay_max=0.9,
                )
            logger.info("[PHASE3] FlatReward component enabled with decay (anti-reward-hacking)")

        # Phase 5: Drawdown penalty
        self._drawdown_penalty: Optional[DrawdownPenaltyComponent] = None
        try:
            from src.config.pipeline_config import load_pipeline_config
            pipeline_cfg = load_pipeline_config()
            dd_cfg = pipeline_cfg._raw.get("reward", {}).get("drawdown_penalty", {})
            if dd_cfg.get("enabled", False):
                self._drawdown_penalty = DrawdownPenaltyComponent(
                    threshold=dd_cfg.get("threshold", 0.05),
                    max_penalty=dd_cfg.get("max_penalty", 0.5),
                )
                logger.info("[Phase5] DrawdownPenaltyComponent enabled")
        except (ImportError, FileNotFoundError, AttributeError):
            pass

        # EXP-RL-EXECUTOR: Execution alpha component
        self._execution_alpha: Optional[ExecutionAlphaComponent] = None
        try:
            from src.config.pipeline_config import load_pipeline_config
            pipeline_cfg = load_pipeline_config()
            ea_cfg = pipeline_cfg._raw.get("reward", {}).get("execution_alpha", {})
            if ea_cfg.get("enabled", False):
                self._execution_alpha = ExecutionAlphaComponent(
                    penalty_no_trade=ea_cfg.get("penalty_no_trade", 0.1),
                    scale=ea_cfg.get("scale", 100.0),
                )
                logger.info("[EXP-RL-EXECUTOR] ExecutionAlphaComponent enabled")
        except (ImportError, FileNotFoundError, AttributeError):
            pass

        # V22 P2: Close reason reward shaping
        self._close_reason_detector: Optional[CloseReasonDetector] = None
        try:
            from src.config.pipeline_config import load_pipeline_config
            pipeline_cfg = load_pipeline_config()
            close_shaping_raw = pipeline_cfg._raw.get("training", {}).get("close_shaping", {})
            if close_shaping_raw.get("enabled", False):
                self._close_reason_detector = CloseReasonDetector(close_shaping_raw)
                logger.info("[V22] CloseReasonDetector enabled with PnL multipliers")
        except (ImportError, FileNotFoundError, AttributeError):
            pass

    def _on_phase_change(self, old_phase: CurriculumPhase, new_phase: CurriculumPhase) -> None:
        """Callback when curriculum phase changes."""
        logger.info(f"Curriculum phase change: {old_phase.value} -> {new_phase.value}")

    def calculate(
        self,
        pnl_pct: float,
        position: int,
        position_change: int = 0,
        holding_bars: int = 0,
        volatility: float = 0.01,
        hour_utc: int = 15,
        is_overnight: bool = False,
        is_weekend: bool = False,
        oil_return: Optional[float] = None,
        price_change: float = 0.0,
        close_reason: str = "",
        cumulative_return: float = 0.0,
        forecast_direction: int = 0,
        session_open_price: float = 0.0,
        current_price: float = 0.0,
        has_position: bool = False,
    ) -> Tuple[float, RewardBreakdown]:
        """
        Calculate reward using all components.

        Args:
            pnl_pct: Raw PnL as decimal (0.01 = 1%)
            position: Current position (-1, 0, 1)
            position_change: Position change (-1=closed, 0=unchanged, 1=opened)
            holding_bars: Bars position has been held
            volatility: Current volatility (e.g., ATR %)
            hour_utc: Current hour in UTC
            is_overnight: Whether approaching overnight
            is_weekend: Whether approaching weekend
            oil_return: Optional WTI oil return
            price_change: Price change this bar

        Returns:
            Tuple of (final_reward, breakdown)
        """
        self._total_steps += 1
        if position_change != 0:
            self._total_trades += 1

        breakdown = RewardBreakdown(
            raw_pnl=pnl_pct,
            position=position,
            holding_bars=holding_bars,
        )

        # =================================================================
        # STEP 1: Update regime detector
        # =================================================================
        regime = self._regime_detector.update(volatility)
        breakdown.current_regime = regime.value

        # Check curriculum filtering
        cost_mult = 1.0
        if self._curriculum:
            self._curriculum.step()
            breakdown.curriculum_phase = self._curriculum.current_phase.value
            cost_mult = self._curriculum.cost_multiplier
            breakdown.cost_multiplier = cost_mult

            # Skip calculation if regime not allowed (return zero reward)
            if not self._curriculum.is_regime_allowed(regime):
                breakdown.clipped_reward = 0.0
                return 0.0, breakdown

        # =================================================================
        # STEP 2: Transform PnL
        # =================================================================
        # FIX 6: Clip -> Scale -> Asymmetric (removed ZScore which broke absolute PnL signal)
        # ZScore converted absolute PnL to relative z-scores, allowing agent to accumulate
        # high shaped reward while losing money. Simple scaling preserves the absolute signal.
        pnl_clipped = self._pnl_clip.calculate(pnl_pct)
        # Scale: typical 5-min return ~0.001, scale to ~1.0 range
        pnl_scaled = pnl_clipped * 100.0  # Gentler: 0.1% move → 0.1 reward
        pnl_transformed = self._pnl_asymmetric.calculate(pnl_scaled)
        breakdown.transformed_pnl = pnl_transformed

        # =================================================================
        # STEP 3: Calculate risk metrics
        # =================================================================
        dsr_value = self._dsr.calculate(return_pct=pnl_pct)
        sortino_value = self._sortino.calculate(return_pct=pnl_pct)

        breakdown.dsr_contribution = dsr_value
        breakdown.sortino_contribution = sortino_value

        # =================================================================
        # STEP 4: Calculate regime penalty
        # =================================================================
        regime_penalty = self._regime_detector.calculate(volatility=volatility)
        breakdown.regime_penalty = regime_penalty

        # =================================================================
        # STEP 5: Market impact - DISABLED to avoid double-counting
        # FIX 2026-02-01: Transaction costs already applied in trading_env.py
        # net_pnl = gross_pnl - trade_cost (line ~830 in trading_env.py)
        # Adding market_impact here would double-count costs.
        # =================================================================
        impact_cost = 0.0  # FIX: Disabled - costs already in PnL
        # Original code (kept for reference):
        # if position_change != 0:
        #     impact_cost = self._market_impact.calculate(...)
        #     impact_cost *= cost_mult
        breakdown.market_impact_cost = impact_cost

        # =================================================================
        # STEP 6: Calculate holding decay
        # =================================================================
        holding_penalty = self._holding_decay.calculate(
            holding_bars=holding_bars,
            is_overnight=is_overnight,
            has_position=(position != 0),
        )
        breakdown.holding_decay_penalty = holding_penalty

        # Gap risk
        gap_penalty = self._gap_risk.calculate(
            has_position=(position != 0),
            is_overnight=is_overnight,
            is_weekend=is_weekend,
        )
        breakdown.gap_risk_penalty = gap_penalty

        # =================================================================
        # STEP 7: Anti-gaming penalties
        # =================================================================
        inactivity = self._inactivity.calculate(position=position)
        churn = self._churn.calculate(action_is_trade=(position_change != 0))
        bias = self._bias.calculate(position=position)

        breakdown.inactivity_penalty = inactivity
        breakdown.churn_penalty = churn
        breakdown.bias_penalty = bias

        # =================================================================
        # STEP 8: Banrep detector (if enabled)
        # =================================================================
        banrep_penalty = 0.0
        if self._banrep:
            banrep_penalty = self._banrep.calculate(
                volatility=volatility,
                price_change=price_change,
            )
        breakdown.banrep_penalty = banrep_penalty

        # =================================================================
        # STEP 9: Oil correlation (if enabled)
        # =================================================================
        oil_adj = 0.0
        if self._oil_tracker and oil_return is not None:
            oil_adj = self._oil_tracker.calculate(
                oil_return=oil_return,
                usdcop_return=pnl_pct,
            )
        breakdown.oil_adjustment = oil_adj

        # =================================================================
        # STEP 9.5: Flat reward - PHASE 2 FIX (if enabled)
        # =================================================================
        # Counterfactual reward: give positive signal for avoiding losses
        # when position is FLAT and market would have moved against a LONG
        flat_reward_bonus = 0.0
        if self._flat_reward is not None:
            # Use price_change as market_return (this is the raw return, not z-scored)
            flat_reward_bonus = self._flat_reward.calculate(
                position=position,
                market_return=price_change,
            )
        breakdown.flat_reward_bonus = flat_reward_bonus

        # =================================================================
        # STEP 9.7: V22 P2 - Close reason reward shaping
        # =================================================================
        close_shaping_adj = 0.0
        if self._close_reason_detector and close_reason:
            close_shaping_adj = self._close_reason_detector.get_delta(
                base_reward=pnl_transformed,
                close_reason=close_reason,
                pnl=pnl_pct,
            )
        breakdown.close_shaping = close_shaping_adj

        # =================================================================
        # STEP 9.8: Phase 5 - Drawdown penalty
        # =================================================================
        dd_penalty = 0.0
        if self._drawdown_penalty:
            dd_penalty = self._drawdown_penalty.calculate(
                cumulative_return=cumulative_return,
            )
        breakdown.drawdown_penalty = dd_penalty

        # =================================================================
        # STEP 9.9: EXP-RL-EXECUTOR - Execution alpha
        # =================================================================
        exec_alpha = 0.0
        if self._execution_alpha:
            exec_alpha = self._execution_alpha.calculate(
                rl_pnl_pct=pnl_pct,
                forecast_direction=forecast_direction,
                session_open_price=session_open_price,
                current_price=current_price,
                has_position=has_position,
            )
        breakdown.execution_alpha = exec_alpha

        # =================================================================
        # STEP 10: Weighted combination
        # =================================================================
        cfg = self._config

        # PnL-based component
        pnl_component = pnl_transformed * cfg.weight_pnl

        # Risk metrics
        dsr_component = dsr_value * cfg.weight_dsr
        sortino_component = sortino_value * cfg.weight_sortino

        # Penalties (already negative)
        regime_component = regime_penalty * cfg.weight_regime_penalty
        holding_component = (holding_penalty + gap_penalty) * cfg.weight_holding_decay
        antigaming_component = (inactivity + churn + bias) * cfg.weight_anti_gaming

        # PHASE2: Flat reward component (counterfactual HOLD reward)
        flat_reward_component = flat_reward_bonus * getattr(cfg, 'weight_flat_reward', 0.3)

        # Phase 5: Drawdown penalty component
        drawdown_component = dd_penalty * getattr(cfg, 'weight_drawdown_penalty', 0.0)

        # EXP-RL-EXECUTOR: Execution alpha component
        exec_alpha_component = exec_alpha * getattr(cfg, 'weight_execution_alpha', 0.0)

        # Combine
        weighted_total = (
            pnl_component +
            dsr_component +
            sortino_component +
            regime_component +
            impact_cost +  # Already negative
            holding_component +
            antigaming_component +
            banrep_penalty +
            oil_adj +
            flat_reward_component +  # PHASE2: Counterfactual HOLD reward
            close_shaping_adj +      # V22 P2: Close reason shaping
            drawdown_component +     # Phase 5: Drawdown penalty
            exec_alpha_component     # EXP-RL-EXECUTOR: Execution alpha
        )
        breakdown.weighted_total = weighted_total

        # =================================================================
        # STEP 11: Normalize (if enabled)
        # =================================================================
        if self._normalizer:
            normalized = self._normalizer.calculate(weighted_total)
        else:
            normalized = weighted_total
        breakdown.normalized_reward = normalized

        # =================================================================
        # STEP 12: Final clip
        # =================================================================
        final_reward = clip_reward(normalized, min_reward=-1.0, max_reward=1.0)
        breakdown.clipped_reward = final_reward

        if self._verbose and self._total_steps % 1000 == 0:
            logger.debug(f"Reward breakdown: {breakdown.to_dict()}")

        return final_reward, breakdown

    def reset(self) -> None:
        """Reset all components for new episode."""
        self._dsr.reset()
        self._sortino.reset()
        self._regime_detector.reset()
        self._market_impact.reset()
        self._holding_decay.reset()
        self._gap_risk.reset()
        self._inactivity.reset()
        self._churn.reset()
        self._bias.reset()
        self._pnl_clip.reset()
        self._pnl_zscore.reset()
        self._pnl_asymmetric.reset()

        if self._banrep:
            self._banrep.reset()
        if self._oil_tracker:
            self._oil_tracker.reset()
        if self._normalizer:
            self._normalizer.reset()
        if self._flat_reward:  # PHASE2
            self._flat_reward.reset()
        if self._close_reason_detector:  # V22 P2
            self._close_reason_detector.reset()
        if self._drawdown_penalty:  # Phase 5
            self._drawdown_penalty.reset()
        if self._execution_alpha:  # EXP-RL-EXECUTOR
            self._execution_alpha.reset()

    def reset_position(self) -> None:
        """Reset position-related state (on position close)."""
        self._holding_decay.reset_position()
        if self._flat_reward:  # PHASE2
            self._flat_reward.reset_position()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        stats = {
            "total_steps": self._total_steps,
            "total_trades": self._total_trades,
            "config_hash": self._config.to_hash(),
        }

        # Component stats
        stats.update(self._dsr.get_stats())
        stats.update(self._sortino.get_stats())
        stats.update(self._regime_detector.get_stats())
        stats.update(self._market_impact.get_stats())
        stats.update(self._holding_decay.get_stats())
        stats.update(self._inactivity.get_stats())
        stats.update(self._churn.get_stats())
        stats.update(self._bias.get_stats())

        if self._banrep:
            stats.update(self._banrep.get_stats())

        if self._normalizer:
            stats.update(self._normalizer.get_stats())

        if self._curriculum:
            stats.update(self._curriculum.get_stats())

        if self._flat_reward:  # PHASE2
            stats.update(self._flat_reward.get_stats())

        if self._close_reason_detector:  # V22 P2
            stats.update(self._close_reason_detector.get_stats())

        if self._drawdown_penalty:  # Phase 5
            stats.update(self._drawdown_penalty.get_stats())

        if self._execution_alpha:  # EXP-RL-EXECUTOR
            stats.update(self._execution_alpha.get_stats())

        return stats

    def get_config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self._config.to_dict()

    @property
    def current_regime(self) -> MarketRegime:
        """Get current detected market regime."""
        return self._regime_detector.current_regime

    @property
    def curriculum_phase(self) -> Optional[CurriculumPhase]:
        """Get current curriculum phase."""
        if self._curriculum:
            return self._curriculum.current_phase
        return None

    def get_curriculum_phase(self) -> Optional[str]:
        """Get current curriculum phase as string (for adapter compatibility)."""
        phase = self.curriculum_phase
        return phase.value if phase else None

    def get_curriculum_stats(self) -> Optional[Dict[str, Any]]:
        """Get curriculum statistics (for adapter compatibility)."""
        if self._curriculum:
            return self._curriculum.get_stats()
        return None

    def step_curriculum(self) -> None:
        """Advance curriculum by one step (for adapter compatibility)."""
        if self._curriculum:
            self._curriculum.step()


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# Keep old class name for backward compatibility
RewardCalculator = ModularRewardCalculator


# Legacy RewardConfig (for backward compat imports)
@dataclass
class LegacyRewardConfig:
    """Legacy config - use RewardConfig from src.training.config instead."""
    transaction_cost_pct: float = 0.0002
    loss_penalty_multiplier: float = 2.0
    hold_bonus_per_bar: float = 0.0
    hold_bonus_requires_profit: bool = True
    consecutive_win_bonus: float = 0.001
    max_consecutive_bonus: int = 5
    drawdown_penalty_threshold: float = 0.05
    drawdown_penalty_multiplier: float = 2.0
    intratrade_dd_penalty: float = 0.5
    max_intratrade_dd: float = 0.02
    time_decay_start_bars: int = 24
    time_decay_per_bar: float = 0.0001
    time_decay_losing_multiplier: float = 2.0
    min_reward: float = -1.0
    max_reward: float = 1.0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_reward_calculator(
    config: Optional[RewardConfig] = None,
    enable_curriculum: bool = True,
    verbose: bool = False,
) -> ModularRewardCalculator:
    """
    Create a reward calculator from configuration.

    Args:
        config: RewardConfig (uses SSOT default if not provided)
        enable_curriculum: Whether to enable curriculum learning
        verbose: Enable verbose logging

    Returns:
        Configured ModularRewardCalculator
    """
    return ModularRewardCalculator(
        config=config,
        enable_curriculum=enable_curriculum,
        verbose=verbose,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ModularRewardCalculator",
    "RewardCalculator",  # Backward compat
    "RewardBreakdown",
    "LegacyRewardConfig",
    "create_reward_calculator",
]
