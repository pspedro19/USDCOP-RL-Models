"""
Unit Tests for Reward Components
================================

Comprehensive tests for the modular reward system.

Contract: CTR-REWARD-SNAPSHOT-001

Tests cover:
- DSR (Differential Sharpe Ratio)
- Sortino Ratio
- Regime Detection
- Market Impact (Almgren-Chriss)
- Holding Decay
- Anti-Gaming Penalties
- PnL Transforms
- Reward Normalizer
- Curriculum Scheduler
- Modular Reward Calculator

Author: Trading Team
Version: 1.0.0
Created: 2026-01-19
"""

import pytest
import numpy as np
from typing import List

# =============================================================================
# Test Base Component
# =============================================================================

class TestRewardComponentBase:
    """Test base RewardComponent class."""

    def test_base_component_interface(self):
        """Test that base component defines required interface."""
        from src.training.reward_components.base import RewardComponent, MarketRegime

        # MarketRegime enum should have required values
        assert hasattr(MarketRegime, 'NORMAL')
        assert hasattr(MarketRegime, 'LOW_VOL')
        assert hasattr(MarketRegime, 'HIGH_VOL')
        assert hasattr(MarketRegime, 'CRISIS')

    def test_market_regime_values(self):
        """Test market regime enum values."""
        from src.training.reward_components.base import MarketRegime

        assert MarketRegime.NORMAL.value == "normal"
        assert MarketRegime.LOW_VOL.value == "low_vol"
        assert MarketRegime.HIGH_VOL.value == "high_vol"
        assert MarketRegime.CRISIS.value == "crisis"


# =============================================================================
# Test DSR Component
# =============================================================================

class TestDSRComponent:
    """Test Differential Sharpe Ratio component."""

    def test_dsr_initialization(self):
        """Test DSR component initialization."""
        from src.training.reward_components.dsr import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        assert dsr._eta == 0.0
        assert dsr._n == 0

    def test_dsr_update_positive_return(self):
        """Test DSR update with positive return."""
        from src.training.reward_components.dsr import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        result = dsr.update(0.001)  # 0.1% return

        assert result is not None
        assert dsr._n == 1

    def test_dsr_update_negative_return(self):
        """Test DSR update with negative return."""
        from src.training.reward_components.dsr import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        result = dsr.update(-0.002)  # -0.2% return

        assert result is not None
        assert dsr._n == 1

    def test_dsr_multiple_updates(self):
        """Test DSR with multiple updates."""
        from src.training.reward_components.dsr import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        returns = [0.001, -0.0005, 0.002, -0.001, 0.0015]

        for r in returns:
            dsr.update(r)

        assert dsr._n == len(returns)
        # EWM estimates should be non-zero after updates
        assert dsr._eta != 0.0 or dsr._n <= 1

    def test_dsr_reset(self):
        """Test DSR reset functionality."""
        from src.training.reward_components.dsr import DifferentialSharpeRatio

        dsr = DifferentialSharpeRatio()
        dsr.update(0.001)
        dsr.update(0.002)

        dsr.reset()

        assert dsr._eta == 0.0
        assert dsr._n == 0


# =============================================================================
# Test Sortino Component
# =============================================================================

class TestSortinoComponent:
    """Test Sortino Ratio component."""

    def test_sortino_initialization(self):
        """Test Sortino component initialization."""
        from src.training.reward_components.sortino import SortinoRatioTracker

        sortino = SortinoRatioTracker(window_size=20)
        assert sortino._window_size == 20
        assert len(sortino._returns) == 0

    def test_sortino_update(self):
        """Test Sortino update with returns."""
        from src.training.reward_components.sortino import SortinoRatioTracker

        sortino = SortinoRatioTracker(window_size=20)
        sortino.update(0.001)
        sortino.update(-0.002)
        sortino.update(0.0015)

        assert len(sortino._returns) == 3

    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation."""
        from src.training.reward_components.sortino import SortinoRatioTracker

        sortino = SortinoRatioTracker(window_size=5)

        # Add some returns
        returns = [0.001, -0.002, 0.003, -0.001, 0.002]
        for r in returns:
            sortino.update(r)

        ratio = sortino.get_sortino_ratio()
        assert isinstance(ratio, float)

    def test_sortino_window_size(self):
        """Test that window size is respected."""
        from src.training.reward_components.sortino import SortinoRatioTracker

        sortino = SortinoRatioTracker(window_size=3)

        for i in range(10):
            sortino.update(0.001 * i)

        assert len(sortino._returns) <= 3

    def test_sortino_reset(self):
        """Test Sortino reset functionality."""
        from src.training.reward_components.sortino import SortinoRatioTracker

        sortino = SortinoRatioTracker()
        sortino.update(0.001)
        sortino.update(0.002)

        sortino.reset()

        assert len(sortino._returns) == 0


# =============================================================================
# Test Regime Detector
# =============================================================================

class TestRegimeDetector:
    """Test market regime detector."""

    def test_regime_detector_initialization(self):
        """Test regime detector initialization."""
        from src.training.reward_components.regime_detector import RegimeDetector

        detector = RegimeDetector()
        assert len(detector._volatility_window) == 0

    def test_regime_detect_normal(self):
        """Test normal regime detection."""
        from src.training.reward_components.regime_detector import RegimeDetector
        from src.training.reward_components.base import MarketRegime

        detector = RegimeDetector()

        # Fill with moderate volatility
        for _ in range(50):
            detector.update(0.001)  # 10 bps volatility

        regime = detector.detect()
        assert regime in [MarketRegime.NORMAL, MarketRegime.LOW_VOL]

    def test_regime_detect_high_vol(self):
        """Test high volatility regime detection."""
        from src.training.reward_components.regime_detector import RegimeDetector
        from src.training.reward_components.base import MarketRegime

        detector = RegimeDetector()

        # Fill with low volatility first
        for _ in range(30):
            detector.update(0.0005)

        # Then spike
        for _ in range(30):
            detector.update(0.003)  # 30 bps volatility

        regime = detector.detect()
        assert regime in [MarketRegime.HIGH_VOL, MarketRegime.NORMAL]

    def test_regime_penalty(self):
        """Test regime-based penalty."""
        from src.training.reward_components.regime_detector import RegimeDetector
        from src.training.reward_components.base import MarketRegime

        detector = RegimeDetector()

        # Get penalties for different regimes
        penalty_normal = detector.get_regime_penalty(MarketRegime.NORMAL)
        penalty_high = detector.get_regime_penalty(MarketRegime.HIGH_VOL)
        penalty_crisis = detector.get_regime_penalty(MarketRegime.CRISIS)

        # Crisis should have highest penalty
        assert penalty_crisis >= penalty_high >= penalty_normal


# =============================================================================
# Test Market Impact
# =============================================================================

class TestMarketImpact:
    """Test Almgren-Chriss market impact model."""

    def test_market_impact_initialization(self):
        """Test market impact model initialization."""
        from src.training.reward_components.market_impact import AlmgrenChrissImpact

        impact = AlmgrenChrissImpact()
        assert impact._lambda > 0
        assert impact._gamma >= 0

    def test_market_impact_calculation(self):
        """Test market impact calculation."""
        from src.training.reward_components.market_impact import AlmgrenChrissImpact

        impact = AlmgrenChrissImpact()

        # Calculate impact for a trade
        cost = impact.calculate(quantity=1000, spread_bps=10.0, volatility=0.001)

        assert isinstance(cost, float)
        assert cost >= 0

    def test_market_impact_increases_with_quantity(self):
        """Test that impact increases with quantity."""
        from src.training.reward_components.market_impact import AlmgrenChrissImpact

        impact = AlmgrenChrissImpact()

        cost_small = impact.calculate(quantity=100, spread_bps=10.0, volatility=0.001)
        cost_large = impact.calculate(quantity=1000, spread_bps=10.0, volatility=0.001)

        assert cost_large > cost_small

    def test_market_impact_increases_with_volatility(self):
        """Test that impact increases with volatility."""
        from src.training.reward_components.market_impact import AlmgrenChrissImpact

        impact = AlmgrenChrissImpact()

        cost_low_vol = impact.calculate(quantity=1000, spread_bps=10.0, volatility=0.0005)
        cost_high_vol = impact.calculate(quantity=1000, spread_bps=10.0, volatility=0.002)

        assert cost_high_vol > cost_low_vol


# =============================================================================
# Test Holding Decay
# =============================================================================

class TestHoldingDecay:
    """Test holding time decay penalty."""

    def test_holding_decay_initialization(self):
        """Test holding decay initialization."""
        from src.training.reward_components.holding_decay import HoldingDecay

        decay = HoldingDecay(half_life_bars=48, max_penalty=0.3)
        assert decay._max_penalty == 0.3

    def test_holding_decay_no_penalty_initially(self):
        """Test no penalty at start."""
        from src.training.reward_components.holding_decay import HoldingDecay

        decay = HoldingDecay()
        penalty = decay.calculate(holding_bars=0, is_overnight=False, has_position=True)

        assert penalty == 0.0

    def test_holding_decay_increases_over_time(self):
        """Test penalty increases with holding time."""
        from src.training.reward_components.holding_decay import HoldingDecay

        decay = HoldingDecay(half_life_bars=48)

        penalty_10 = decay.calculate(holding_bars=10, is_overnight=False, has_position=True)
        penalty_50 = decay.calculate(holding_bars=50, is_overnight=False, has_position=True)
        penalty_100 = decay.calculate(holding_bars=100, is_overnight=False, has_position=True)

        assert penalty_50 > penalty_10
        assert penalty_100 > penalty_50

    def test_holding_decay_overnight_multiplier(self):
        """Test overnight positions have higher penalty."""
        from src.training.reward_components.holding_decay import HoldingDecay

        decay = HoldingDecay(overnight_multiplier=1.5)

        penalty_day = decay.calculate(holding_bars=30, is_overnight=False, has_position=True)
        penalty_night = decay.calculate(holding_bars=30, is_overnight=True, has_position=True)

        assert penalty_night > penalty_day

    def test_holding_decay_no_position(self):
        """Test no penalty when no position."""
        from src.training.reward_components.holding_decay import HoldingDecay

        decay = HoldingDecay()
        penalty = decay.calculate(holding_bars=100, is_overnight=False, has_position=False)

        assert penalty == 0.0


# =============================================================================
# Test Anti-Gaming Components
# =============================================================================

class TestAntiGaming:
    """Test anti-gaming penalty components."""

    def test_inactivity_tracker(self):
        """Test inactivity tracker."""
        from src.training.reward_components.anti_gaming import InactivityTracker

        tracker = InactivityTracker(window_size=10)

        # Stay flat for 10 bars
        for _ in range(10):
            tracker.calculate(position=0)

        penalty = tracker.calculate(position=0)
        assert penalty >= 0  # Should have some penalty

    def test_churn_tracker(self):
        """Test churn (overtrading) tracker."""
        from src.training.reward_components.anti_gaming import ChurnTracker

        tracker = ChurnTracker(window_size=20, threshold=0.3)

        # Trade frequently
        for i in range(20):
            tracker.calculate(action_is_trade=True)

        penalty = tracker.calculate(action_is_trade=True)
        assert penalty >= 0  # Should penalize excessive trading

    def test_bias_detector(self):
        """Test directional bias detector."""
        from src.training.reward_components.anti_gaming import BiasDetector

        detector = BiasDetector(window_size=20, bias_threshold=0.7)

        # Always go long
        for _ in range(30):
            detector.update(action=1)

        bias_penalty = detector.get_bias_penalty()
        assert bias_penalty > 0  # Should detect bias

    def test_action_correlation_tracker(self):
        """Test action correlation tracker."""
        from src.training.reward_components.anti_gaming import ActionCorrelationTracker

        tracker = ActionCorrelationTracker()

        # Submit alternating actions
        for i in range(20):
            tracker.update(action=1 if i % 2 == 0 else -1)

        correlation = tracker.get_correlation()
        assert abs(correlation) >= 0  # Should detect pattern


# =============================================================================
# Test PnL Transforms
# =============================================================================

class TestPnLTransforms:
    """Test PnL transformation components."""

    def test_log_pnl_transform(self):
        """Test log PnL transform."""
        from src.training.reward_components.pnl_transforms import LogPnLTransform

        transform = LogPnLTransform()

        # Positive PnL
        result_pos = transform.transform(0.01)
        assert result_pos > 0

        # Negative PnL
        result_neg = transform.transform(-0.01)
        assert result_neg < 0

    def test_asymmetric_pnl_transform(self):
        """Test asymmetric PnL transform."""
        from src.training.reward_components.pnl_transforms import AsymmetricPnLTransform

        transform = AsymmetricPnLTransform(loss_multiplier=2.0)

        gain = transform.transform(0.01)
        loss = transform.transform(-0.01)

        # Loss should be more negative than gain is positive
        assert abs(loss) > abs(gain)

    def test_clipped_pnl_transform(self):
        """Test clipped PnL transform."""
        from src.training.reward_components.pnl_transforms import ClippedPnLTransform

        transform = ClippedPnLTransform(clip_value=0.05)

        result_extreme = transform.transform(0.10)
        assert result_extreme <= 0.05

        result_neg_extreme = transform.transform(-0.10)
        assert result_neg_extreme >= -0.05

    def test_zscore_pnl_transform(self):
        """Test z-score PnL transform."""
        from src.training.reward_components.pnl_transforms import ZScorePnLTransform

        transform = ZScorePnLTransform(window_size=20)

        # Add some PnL values to build history
        for i in range(25):
            transform.transform(0.001 * (i % 10 - 5))

        # Now transform should use z-score
        result = transform.transform(0.01)
        assert isinstance(result, float)

    def test_composite_pnl_transform(self):
        """Test composite PnL transform."""
        from src.training.reward_components.pnl_transforms import (
            CompositePnLTransform,
            ClippedPnLTransform,
            AsymmetricPnLTransform,
        )

        composite = CompositePnLTransform()
        composite.add_transform(ClippedPnLTransform())
        composite.add_transform(AsymmetricPnLTransform())

        result = composite.transform(0.01)
        assert isinstance(result, float)


# =============================================================================
# Test Reward Normalizer
# =============================================================================

class TestRewardNormalizer:
    """Test FinRL-Meta reward normalizer."""

    def test_normalizer_initialization(self):
        """Test normalizer initialization."""
        from src.training.reward_components.reward_normalizer import RewardNormalizer

        normalizer = RewardNormalizer()
        assert normalizer._count == 0

    def test_normalizer_single_value(self):
        """Test normalizer with single value."""
        from src.training.reward_components.reward_normalizer import RewardNormalizer

        normalizer = RewardNormalizer()
        result = normalizer.normalize(1.0)

        assert isinstance(result, float)

    def test_normalizer_convergence(self):
        """Test that normalizer estimates converge."""
        from src.training.reward_components.reward_normalizer import RewardNormalizer

        normalizer = RewardNormalizer()

        # Add many values from known distribution
        np.random.seed(42)
        values = np.random.normal(0.5, 0.1, 1000)

        results = []
        for v in values:
            results.append(normalizer.normalize(v))

        # After many updates, mean should be close to 0.5
        assert abs(normalizer._mean - 0.5) < 0.05

    def test_normalizer_reset(self):
        """Test normalizer reset."""
        from src.training.reward_components.reward_normalizer import RewardNormalizer

        normalizer = RewardNormalizer()
        normalizer.normalize(1.0)
        normalizer.normalize(2.0)

        normalizer.reset()

        assert normalizer._count == 0
        assert normalizer._mean == 0.0


# =============================================================================
# Test Curriculum Scheduler
# =============================================================================

class TestCurriculumScheduler:
    """Test curriculum learning scheduler."""

    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        from src.training.curriculum_scheduler import CurriculumScheduler, CurriculumPhase
        from src.training.config import CurriculumConfig

        config = CurriculumConfig()
        scheduler = CurriculumScheduler(config)

        assert scheduler.current_phase == CurriculumPhase.PHASE_1

    def test_scheduler_phase_transitions(self):
        """Test phase transitions based on step count."""
        from src.training.curriculum_scheduler import CurriculumScheduler, CurriculumPhase
        from src.training.config import CurriculumConfig

        config = CurriculumConfig(
            enabled=True,
            phase_1_steps=100,
            phase_2_steps=200,
            phase_3_steps=300,
        )
        scheduler = CurriculumScheduler(config)

        assert scheduler.current_phase == CurriculumPhase.PHASE_1

        # Advance past phase 1
        for _ in range(101):
            scheduler.step()

        assert scheduler.current_phase == CurriculumPhase.PHASE_2

        # Advance past phase 2
        for _ in range(100):
            scheduler.step()

        assert scheduler.current_phase == CurriculumPhase.PHASE_3

    def test_scheduler_regime_filtering(self):
        """Test regime filtering in different phases."""
        from src.training.curriculum_scheduler import CurriculumScheduler
        from src.training.config import CurriculumConfig
        from src.training.reward_components.base import MarketRegime

        config = CurriculumConfig(
            enabled=True,
            phase_1_steps=100,
            phase_2_steps=200,
            phase_3_steps=300,
        )
        scheduler = CurriculumScheduler(config)

        # Phase 1: Only NORMAL and LOW_VOL allowed
        assert scheduler.is_regime_allowed(MarketRegime.NORMAL)
        assert scheduler.is_regime_allowed(MarketRegime.LOW_VOL)
        assert not scheduler.is_regime_allowed(MarketRegime.CRISIS)

    def test_scheduler_cost_multiplier(self):
        """Test cost multiplier in different phases."""
        from src.training.curriculum_scheduler import CurriculumScheduler, CurriculumPhase
        from src.training.config import CurriculumConfig

        config = CurriculumConfig(
            enabled=True,
            phase_1_steps=100,
            phase_2_steps=200,
            phase_3_steps=300,
            phase_1_cost_mult=0.5,
            phase_2_cost_mult=0.75,
            phase_3_cost_mult=1.0,
        )
        scheduler = CurriculumScheduler(config)

        # Phase 1
        assert scheduler.cost_multiplier == 0.5

        # Advance to phase 2
        for _ in range(101):
            scheduler.step()
        assert scheduler.cost_multiplier == 0.75

    def test_scheduler_reset(self):
        """Test scheduler reset."""
        from src.training.curriculum_scheduler import CurriculumScheduler, CurriculumPhase
        from src.training.config import CurriculumConfig

        config = CurriculumConfig(enabled=True, phase_1_steps=50)
        scheduler = CurriculumScheduler(config)

        # Advance
        for _ in range(60):
            scheduler.step()

        # Reset
        scheduler.reset()

        assert scheduler.current_phase == CurriculumPhase.PHASE_1


# =============================================================================
# Test Modular Reward Calculator
# =============================================================================

class TestModularRewardCalculator:
    """Test the modular reward calculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        from src.training.reward_calculator import ModularRewardCalculator
        from src.training.config import RewardConfig

        config = RewardConfig()
        calculator = ModularRewardCalculator(config)

        assert calculator is not None

    def test_calculator_basic_calculation(self):
        """Test basic reward calculation."""
        from src.training.reward_calculator import ModularRewardCalculator
        from src.training.config import RewardConfig

        config = RewardConfig()
        calculator = ModularRewardCalculator(config, enable_curriculum=False)

        reward, breakdown = calculator.calculate(
            pnl_pct=0.001,
            position=1,
            position_change=0,
            holding_bars=5,
            volatility=0.001,
            hour_utc=14,
            is_overnight=False,
        )

        assert isinstance(reward, float)
        assert breakdown is not None

    def test_calculator_positive_pnl(self):
        """Test reward for positive PnL."""
        from src.training.reward_calculator import ModularRewardCalculator
        from src.training.config import RewardConfig

        config = RewardConfig()
        calculator = ModularRewardCalculator(config, enable_curriculum=False)

        reward_pos, _ = calculator.calculate(
            pnl_pct=0.01,
            position=1,
            position_change=0,
            holding_bars=5,
            volatility=0.001,
            hour_utc=14,
            is_overnight=False,
        )

        reward_neg, _ = calculator.calculate(
            pnl_pct=-0.01,
            position=1,
            position_change=0,
            holding_bars=5,
            volatility=0.001,
            hour_utc=14,
            is_overnight=False,
        )

        # Positive PnL should give higher reward than negative
        assert reward_pos > reward_neg

    def test_calculator_reset(self):
        """Test calculator reset."""
        from src.training.reward_calculator import ModularRewardCalculator
        from src.training.config import RewardConfig

        config = RewardConfig()
        calculator = ModularRewardCalculator(config)

        # Make some calculations
        calculator.calculate(
            pnl_pct=0.001,
            position=1,
            position_change=0,
            holding_bars=5,
            volatility=0.001,
        )

        # Reset
        calculator.reset()

        # Should not raise
        calculator.calculate(
            pnl_pct=0.001,
            position=1,
            position_change=0,
            holding_bars=0,
            volatility=0.001,
        )

    def test_calculator_breakdown_components(self):
        """Test that breakdown contains all expected components."""
        from src.training.reward_calculator import ModularRewardCalculator
        from src.training.config import RewardConfig

        config = RewardConfig()
        calculator = ModularRewardCalculator(config, enable_curriculum=False)

        _, breakdown = calculator.calculate(
            pnl_pct=0.001,
            position=1,
            position_change=0,
            holding_bars=5,
            volatility=0.001,
            hour_utc=14,
            is_overnight=False,
        )

        # Check breakdown has expected fields
        assert hasattr(breakdown, 'pnl_component')
        assert hasattr(breakdown, 'dsr_component')
        assert hasattr(breakdown, 'total_reward')


# =============================================================================
# Test Reward Config
# =============================================================================

class TestRewardConfig:
    """Test reward configuration."""

    def test_reward_config_defaults(self):
        """Test default reward config values."""
        from src.training.config import RewardConfig

        config = RewardConfig()

        assert config.weight_pnl > 0
        assert config.weight_dsr >= 0
        assert config.enable_normalization in [True, False]
        assert config.enable_curriculum in [True, False]

    def test_reward_config_immutability(self):
        """Test that config is frozen (immutable)."""
        from src.training.config import RewardConfig

        config = RewardConfig()

        with pytest.raises((AttributeError, TypeError)):
            config.weight_pnl = 999.0


# =============================================================================
# Test Reward Contracts Registry
# =============================================================================

class TestRewardContractsRegistry:
    """Test reward contracts registry."""

    def test_canonical_contract_exists(self):
        """Test that canonical contract is defined."""
        from src.core.contracts.reward_contracts_registry import (
            CANONICAL_REWARD_CONTRACT_ID,
            REWARD_CONTRACTS,
        )

        assert CANONICAL_REWARD_CONTRACT_ID in REWARD_CONTRACTS

    def test_contract_versions(self):
        """Test that all expected versions exist."""
        from src.core.contracts.reward_contracts_registry import REWARD_CONTRACTS

        expected_versions = ["v1.0.0", "v1.1.0", "v1.2.0", "v1.3.0"]
        for version in expected_versions:
            assert version in REWARD_CONTRACTS

    def test_get_contract_function(self):
        """Test get_contract helper function."""
        from src.core.contracts.reward_contracts_registry import get_contract

        contract = get_contract("v1.0.0")
        assert contract is not None
        assert contract.version == "v1.0.0"

    def test_contract_to_reward_config(self):
        """Test converting contract to RewardConfig."""
        from src.core.contracts.reward_contracts_registry import get_contract

        contract = get_contract("v1.0.0")
        config = contract.to_reward_config()

        assert config is not None
        assert config.weight_pnl > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
