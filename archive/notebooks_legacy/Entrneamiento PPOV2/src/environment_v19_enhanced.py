"""
USD/COP RL Trading System - Environment V19 Enhanced
=====================================================

Environment V19 con TODAS las mejoras recomendadas integradas:

STATE FEATURES EXPANDIDOS (21 total):
- 12 base features (igual que V19)
- 6 regime features (is_crisis, is_volatile, is_normal, confidence, vix_trend, days)
- 3 feedback features (accuracy, trend, consecutive_wrong)

NUEVAS CAPACIDADES:
1. Régimen como Feature: El modelo VE el régimen, no solo reacciona
2. Feedback Loop: El modelo sabe si sus predicciones recientes fallaron
3. Risk Manager Integration: Kill switches automáticos opcionales

BACKWARD COMPATIBLE:
- Puede usarse como drop-in replacement de TradingEnvironmentV19
- Nuevas features son opcionales (use_regime_features, use_feedback_features)

Author: Claude Code
Version: 1.0.0 (Enhanced)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any, List
from collections import deque
import warnings

# Import base components
try:
    from .environment_v19 import (
        TradingEnvironmentV19,
        SETFXCostModel,
        VolatilityScaler,
    )
    from .feedback_tracker import FeedbackTracker, RegimeFeatureGenerator
    from .risk_manager import RiskManager, RiskLimits, RiskStatus
except ImportError:
    from environment_v19 import (
        TradingEnvironmentV19,
        SETFXCostModel,
        VolatilityScaler,
    )
    from feedback_tracker import FeedbackTracker, RegimeFeatureGenerator
    from risk_manager import RiskManager, RiskLimits, RiskStatus


class TradingEnvironmentV19Enhanced(TradingEnvironmentV19):
    """
    Environment V19 Enhanced con regime features y feedback loop.

    NUEVAS FEATURES (9 adicionales):
    - 6 Regime Features:
      * is_crisis (0/1)
      * is_volatile (0/1)
      * is_normal (0/1)
      * regime_confidence (0-1)
      * vix_trend (-1 a +1)
      * days_in_regime (0-1)
    - 3 Feedback Features:
      * accuracy (0-1)
      * accuracy_trend (-1 a +1)
      * consecutive_wrong (0-1)

    TOTAL: 12 base + 6 regime + 3 feedback = 21 state features

    Args:
        Todos los argumentos de TradingEnvironmentV19, más:
        use_regime_features: Si añadir 6 regime features al observation
        use_feedback_features: Si añadir 3 feedback features al observation
        use_risk_manager: Si usar RiskManager con kill switches
        risk_limits: Configuración de RiskLimits (opcional)
    """

    def __init__(
        self,
        df,
        initial_balance: float = 10_000,
        max_position: float = 1.0,
        episode_length: int = 1200,
        max_drawdown_pct: float = 15.0,
        cost_model: Optional[SETFXCostModel] = None,
        use_curriculum_costs: bool = True,
        reward_function=None,
        feature_columns: Optional[List[str]] = None,
        volatility_column: str = 'volatility_pct',
        return_column: str = 'close_return',
        use_vol_scaling: bool = True,
        vol_scaling_config: Optional[Dict[str, Any]] = None,
        vol_feature_column: str = 'atr_pct',
        use_regime_detection: bool = False,
        regime_detector: Optional[Any] = None,
        vix_column: str = 'vix_z',
        embi_column: str = 'embi_z',
        protection_mode: str = 'min',
        # NEW V19 Enhanced parameters
        use_regime_features: bool = True,
        use_feedback_features: bool = True,
        use_risk_manager: bool = False,
        risk_limits: Optional[RiskLimits] = None,
        verbose: int = 0,
    ):
        # Store enhanced parameters before calling super().__init__
        self.use_regime_features = use_regime_features
        self.use_feedback_features = use_feedback_features
        self.use_risk_manager = use_risk_manager

        # Calculate number of additional features
        self.n_regime_features = 6 if use_regime_features else 0
        self.n_feedback_features = 3 if use_feedback_features else 0
        self.n_enhanced_features = self.n_regime_features + self.n_feedback_features

        # Initialize parent
        super().__init__(
            df=df,
            initial_balance=initial_balance,
            max_position=max_position,
            episode_length=episode_length,
            max_drawdown_pct=max_drawdown_pct,
            cost_model=cost_model,
            use_curriculum_costs=use_curriculum_costs,
            reward_function=reward_function,
            feature_columns=feature_columns,
            volatility_column=volatility_column,
            return_column=return_column,
            use_vol_scaling=use_vol_scaling,
            vol_scaling_config=vol_scaling_config,
            vol_feature_column=vol_feature_column,
            use_regime_detection=use_regime_detection,
            regime_detector=regime_detector,
            vix_column=vix_column,
            embi_column=embi_column,
            protection_mode=protection_mode,
            verbose=verbose,
        )

        # Update state features count
        self.n_state_features = 12 + self.n_enhanced_features  # Base + enhanced

        # Update observation space
        total_features = self.n_state_features + self.n_market_features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_features,),
            dtype=np.float32
        )

        # Initialize enhanced components
        if self.use_regime_features:
            self.regime_feature_gen = RegimeFeatureGenerator(
                vix_lookback=20,
            )

        if self.use_feedback_features:
            self.feedback_tracker = FeedbackTracker(
                window_size=20,
                action_threshold=0.1,
            )

        if self.use_risk_manager:
            self.risk_manager = RiskManager(
                limits=risk_limits or RiskLimits(),
            )
        else:
            self.risk_manager = None

        # Tracking for enhanced features
        self._last_action = 0.0
        self._last_market_return = 0.0

        if verbose > 0:
            print(f"TradingEnvironmentV19Enhanced initialized:")
            print(f"  - Base state features: 12")
            print(f"  - Regime features: {self.n_regime_features}")
            print(f"  - Feedback features: {self.n_feedback_features}")
            print(f"  - Total state features: {self.n_state_features}")
            print(f"  - Market features: {self.n_market_features}")
            print(f"  - Total observation size: {total_features}")

    def _reset_state(self):
        """Reset estado interno incluyendo componentes enhanced."""
        super()._reset_state()

        # Reset enhanced components
        if self.use_regime_features and hasattr(self, 'regime_feature_gen'):
            self.regime_feature_gen.reset()

        if self.use_feedback_features and hasattr(self, 'feedback_tracker'):
            self.feedback_tracker.reset()

        if self.use_risk_manager and hasattr(self, 'risk_manager') and self.risk_manager:
            self.risk_manager.reset(self.initial_balance)

        self._last_action = 0.0
        self._last_market_return = 0.0

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Step con enhanced features y risk management.
        """
        # Execute base step
        obs, reward, terminated, truncated, info = super().step(action)

        # Get market return for feedback tracking
        market_return = info.get('market_return', 0)
        action_value = float(action[0]) if hasattr(action, '__len__') else float(action)

        # Update feedback tracker
        if self.use_feedback_features:
            self.feedback_tracker.update(
                predicted_direction=self._last_action,
                actual_return=self._last_market_return,
            )

        # Store for next step
        self._last_action = action_value
        self._last_market_return = market_return

        # Update risk manager if enabled
        if self.use_risk_manager and self.risk_manager:
            status, risk_mult, alerts = self.risk_manager.update(
                portfolio_value=info.get('portfolio', self.initial_balance),
                step_return=info.get('step_return', 0),
                action=action_value,
            )

            info['risk_status'] = status.value
            info['risk_multiplier'] = risk_mult
            info['risk_alerts'] = [a.message for a in alerts]

            # Override terminated if risk manager says PAUSED
            if status == RiskStatus.PAUSED:
                terminated = True
                info['termination_reason'] = 'risk_manager_stop'

        # Rebuild observation with enhanced features
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Construir observation con enhanced features.

        ORDER:
        1. Base state features (12)
        2. Regime features (6) - if enabled
        3. Feedback features (3) - if enabled
        4. Market features (N)
        """
        idx = self.start_idx + self.current_step
        current_data = self.df.iloc[idx]

        # === BASE STATE FEATURES (12) ===
        state_features = [
            self.position,
            np.clip(self.unrealized_pnl / 0.05, -1, 1),
            np.clip(self.cumulative_return / 0.10, -1, 1),
            -self.current_drawdown / self.max_drawdown_pct,
            -self.max_drawdown_episode / self.max_drawdown_pct,
            self._encode_regime(current_data),
            self._get_session_phase(current_data),
            self._get_volatility_percentile(current_data),
            self.curriculum_cost_multiplier,
            min(self.position_duration / 100, 1.0),
            min(self.trade_count / 50, 1.0),
            1.0 - (self.current_step / self.episode_length),
        ]

        # === REGIME FEATURES (6) ===
        if self.use_regime_features:
            vix_z = current_data.get(self.vix_column, 0.0)
            embi_z = current_data.get(self.embi_column, 0.0)
            vol_pct = self._get_volatility_percentile(current_data) * 100

            regime_features = self.regime_feature_gen.get_features(
                vix_z=vix_z,
                embi_z=embi_z,
                vol_pct=vol_pct,
            )
        else:
            regime_features = []

        # === FEEDBACK FEATURES (3) ===
        if self.use_feedback_features:
            feedback_features = self.feedback_tracker.get_features()
        else:
            feedback_features = []

        # === MARKET FEATURES ===
        market_features = []
        for col in self.feature_columns:
            value = current_data.get(col, 0)
            normalized = (value - self.feature_means[col]) / self.feature_stds[col]
            market_features.append(np.clip(normalized, -5, 5))

        # Combine all
        all_features = state_features + regime_features + feedback_features + market_features
        obs = np.array(all_features, dtype=np.float32)

        # Handle NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)

        return obs

    def get_episode_metrics(self) -> Dict[str, float]:
        """Get metrics including enhanced tracking."""
        metrics = super().get_episode_metrics()

        # Add feedback tracker stats
        if self.use_feedback_features:
            fb_stats = self.feedback_tracker.get_stats()
            metrics['feedback_accuracy'] = fb_stats.get('rolling_accuracy', 0.5)
            metrics['feedback_lifetime_acc'] = fb_stats.get('lifetime_accuracy', 0.5)

        # Add risk manager stats
        if self.use_risk_manager and self.risk_manager:
            rm_report = self.risk_manager.get_report()
            metrics['risk_status'] = rm_report.get('status', 'ACTIVE')
            metrics['risk_alerts_count'] = rm_report.get('total_alerts', 0)

        return metrics


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_enhanced_training_env(
    df,
    config,
    reward_function=None,
    feature_columns=None,
    use_regime_features: bool = True,
    use_feedback_features: bool = True,
    use_risk_manager: bool = False,
) -> TradingEnvironmentV19Enhanced:
    """
    Factory para crear environment enhanced de training.
    """
    if hasattr(config, 'environment'):
        env_config = config.environment
        initial_balance = env_config.initial_balance
        max_position = env_config.max_position
        episode_length = env_config.episode_length
        max_drawdown = env_config.max_drawdown_pct
    else:
        initial_balance = config.get('initial_balance', 10000)
        max_position = config.get('max_position', 1.0)
        episode_length = config.get('episode_length', 1200)
        max_drawdown = config.get('max_drawdown_pct', 15.0)

    env = TradingEnvironmentV19Enhanced(
        df=df,
        initial_balance=initial_balance,
        max_position=max_position,
        episode_length=episode_length,
        max_drawdown_pct=max_drawdown,
        use_curriculum_costs=True,
        reward_function=reward_function,
        feature_columns=feature_columns,
        use_regime_features=use_regime_features,
        use_feedback_features=use_feedback_features,
        use_risk_manager=use_risk_manager,
    )

    return env


def create_enhanced_validation_env(
    df,
    config,
    reward_function=None,
    feature_columns=None,
    use_regime_features: bool = True,
    use_feedback_features: bool = True,
    use_risk_manager: bool = True,  # Enable in validation
) -> TradingEnvironmentV19Enhanced:
    """
    Factory para crear environment enhanced de validación.

    Diferencias vs training:
    - Costos completos (no curriculum)
    - Risk manager habilitado por default
    """
    if hasattr(config, 'environment'):
        env_config = config.environment
        initial_balance = env_config.initial_balance
        max_position = env_config.max_position
        episode_length = env_config.episode_length
        max_drawdown = env_config.max_drawdown_pct
    else:
        initial_balance = config.get('initial_balance', 10000)
        max_position = config.get('max_position', 1.0)
        episode_length = config.get('episode_length', 1200)
        max_drawdown = config.get('max_drawdown_pct', 15.0)

    env = TradingEnvironmentV19Enhanced(
        df=df,
        initial_balance=initial_balance,
        max_position=max_position,
        episode_length=episode_length,
        max_drawdown_pct=max_drawdown,
        use_curriculum_costs=False,  # Full costs
        reward_function=reward_function,
        feature_columns=feature_columns,
        use_regime_features=use_regime_features,
        use_feedback_features=use_feedback_features,
        use_risk_manager=use_risk_manager,
    )

    return env


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ENVIRONMENT V19 ENHANCED - USD/COP RL Trading System")
    print("=" * 70)

    import pandas as pd

    # Create dummy data for testing
    np.random.seed(42)
    n_samples = 2000

    df = pd.DataFrame({
        'close': 4500 + np.cumsum(np.random.normal(0, 10, n_samples)),
        'close_return': np.random.normal(0, 0.001, n_samples),
        'log_ret_5m': np.random.normal(0, 0.001, n_samples),
        'volatility_pct': np.random.uniform(0.5, 2.0, n_samples),
        'atr_pct': np.random.uniform(0.001, 0.01, n_samples),
        'vix_z': np.random.normal(0, 1, n_samples),
        'embi_z': np.random.normal(0, 1, n_samples),
        'rsi_14': np.random.uniform(30, 70, n_samples),
        'adx_14': np.random.uniform(10, 50, n_samples),
    })

    print("\n1. Creating Enhanced Environment:")
    print("-" * 50)

    env = TradingEnvironmentV19Enhanced(
        df=df,
        feature_columns=['log_ret_5m', 'rsi_14', 'adx_14'],
        use_regime_features=True,
        use_feedback_features=True,
        use_risk_manager=False,
        verbose=1,
    )

    print(f"\n  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")

    print("\n2. Running Test Episode:")
    print("-" * 50)

    obs, info = env.reset()
    print(f"  Initial obs shape: {obs.shape}")
    print(f"  First 21 features (state): {obs[:21]}")

    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        if i < 3:
            print(f"\n  Step {i+1}:")
            print(f"    Action: {action[0]:.3f}")
            print(f"    Reward: {reward:.3f}")
            print(f"    Position: {info['position']:.3f}")

    print("\n3. Episode Metrics:")
    print("-" * 50)
    metrics = env.get_episode_metrics()
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 70)
    print("TradingEnvironmentV19Enhanced ready for use")
    print("=" * 70)
