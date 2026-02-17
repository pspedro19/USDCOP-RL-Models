"""
Ensemble Predictor - V22 P1 Multi-Model Voting
================================================
Load N models trained with different seeds, vote on actions,
compute confidence scores for position sizing.

Supports both PPO (stateless) and RecurrentPPO (stateful) models.

Contract: CTR-ENSEMBLE-001
Version: 1.0.0
Date: 2026-02-06
"""

import inspect
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Load N models, vote on actions, compute confidence.

    For continuous action spaces, maps each model's action to
    discrete signals (LONG/SHORT/HOLD) then does majority vote.

    For discrete action spaces (V22 P2+), directly votes on
    the integer action.
    """

    def __init__(
        self,
        model_paths: List[Path],
        min_consensus: int = 3,
        action_type: str = "discrete",
        threshold_long: float = 0.35,
        threshold_short: float = -0.35,
        use_lstm: bool = False,
    ):
        """
        Args:
            model_paths: Paths to N model .zip files
            min_consensus: Minimum models that must agree (e.g., 3/5)
            action_type: "discrete" or "continuous"
            threshold_long: Threshold for LONG (continuous only)
            threshold_short: Threshold for SHORT (continuous only)
            use_lstm: Whether models are RecurrentPPO
        """
        self.min_consensus = min_consensus
        self.action_type = action_type
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.use_lstm = use_lstm
        self.models = []
        self._lstm_states = []
        self._episode_starts = []

        self._load_models(model_paths)

    def _load_models(self, model_paths: List[Path]) -> None:
        """Load all models."""
        for path in model_paths:
            path = Path(path)
            if not path.exists():
                logger.warning(f"Model not found: {path}")
                continue

            try:
                if self.use_lstm:
                    from sb3_contrib import RecurrentPPO
                    model = RecurrentPPO.load(str(path))
                else:
                    from stable_baselines3 import PPO
                    model = PPO.load(str(path))

                self.models.append(model)
                self._lstm_states.append(None)
                self._episode_starts.append(np.array([True]))
                logger.info(f"Loaded model: {path}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")

        if not self.models:
            raise ValueError("No models loaded successfully")

        logger.info(
            f"Ensemble loaded: {len(self.models)} models, "
            f"min_consensus={self.min_consensus}"
        )

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, float, Dict[str, Any]]:
        """
        Get ensemble prediction via majority vote.

        Args:
            obs: Observation array
            deterministic: Use deterministic predictions

        Returns:
            Tuple of (action, confidence, vote_details)
            - action: Majority-voted action (int for discrete, mapped for continuous)
            - confidence: Fraction of models agreeing (0.0-1.0)
            - vote_details: {model_idx: action} for logging
        """
        actions = []

        for i, model in enumerate(self.models):
            if self.use_lstm:
                action, self._lstm_states[i] = model.predict(
                    obs,
                    state=self._lstm_states[i],
                    episode_start=self._episode_starts[i],
                    deterministic=deterministic,
                )
                self._episode_starts[i] = np.array([False])
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            if self.action_type == "continuous":
                mapped = self._map_continuous_action(float(action[0]))
            else:
                mapped = int(action)

            actions.append(mapped)

        # Majority vote
        votes = Counter(actions)
        majority_action, majority_count = votes.most_common(1)[0]
        confidence = majority_count / len(self.models)

        vote_details = {
            "votes": dict(enumerate(actions)),
            "vote_counts": dict(votes),
            "n_models": len(self.models),
        }

        # Require minimum consensus
        if majority_count < self.min_consensus:
            hold_action = 0  # HOLD for both discrete and continuous mapping
            return hold_action, confidence, vote_details

        return majority_action, confidence, vote_details

    def _map_continuous_action(self, action_value: float) -> int:
        """Map continuous [-1, 1] action to discrete signal."""
        if action_value > self.threshold_long:
            return 1  # LONG / BUY
        elif action_value < self.threshold_short:
            return 2  # SHORT / SELL
        else:
            return 0  # HOLD

    def reset_states(self) -> None:
        """Reset LSTM states for all models (call on episode boundaries)."""
        self._lstm_states = [None] * len(self.models)
        self._episode_starts = [np.array([True])] * len(self.models)

    @property
    def n_models(self) -> int:
        return len(self.models)


def load_ensemble_from_multi_seed(
    base_dir: Path,
    seeds: Optional[List[int]] = None,
    use_lstm: bool = False,
    **kwargs,
) -> EnsemblePredictor:
    """
    Load ensemble from multi-seed training output.

    Looks for models at base_dir/*seed*/best_model.zip

    Args:
        base_dir: Directory containing seed subdirectories
        seeds: Specific seeds to load (default: all found)
        use_lstm: Whether models are RecurrentPPO
        **kwargs: Additional EnsemblePredictor kwargs

    Returns:
        EnsemblePredictor with all found models
    """
    base_dir = Path(base_dir)
    model_paths = []

    if seeds:
        for seed in seeds:
            model_path = base_dir / f"seed_{seed}" / "best_model.zip"
            if model_path.exists():
                model_paths.append(model_path)
    else:
        # Find all seed directories
        for seed_dir in sorted(base_dir.glob("seed_*")):
            model_path = seed_dir / "best_model.zip"
            if model_path.exists():
                model_paths.append(model_path)

    if not model_paths:
        # Fallback: look for models in parent_seedN directories
        parent = base_dir.parent
        for seed_dir in sorted(parent.glob(f"{base_dir.name}_seed*")):
            model_path = seed_dir / "best_model.zip"
            if model_path.exists():
                model_paths.append(model_path)

    logger.info(f"Found {len(model_paths)} seed models in {base_dir}")
    return EnsemblePredictor(
        model_paths=model_paths,
        use_lstm=use_lstm,
        **kwargs,
    )
