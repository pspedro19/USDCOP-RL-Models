#!/usr/bin/env python3
"""
SHAP Feature Importance Tracker (Phase 15.2)
=============================================

Tracks and logs feature importance using SHAP values for PPO models.

Usage:
    from src.features.shap_tracker import SHAPTracker

    tracker = SHAPTracker(model_path="models/ppo_primary/model.zip")
    importance = tracker.calculate_importance(sample_observations)
    tracker.save_report("docs/model_cards/shap_report.json")

Author: Trading Team
Date: 2025-01-14
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import SHAP - it's optional for the system to work
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import stable-baselines3 for model loading
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


logger = logging.getLogger(__name__)


# Feature names in order (must match observation space)
FEATURE_NAMES = [
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    "rsi_9",
    "atr_pct",
    "adx_14",
    "dxy_z",
    "dxy_change_1d",
    "vix_z",
    "embi_z",
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d",
    "position",
    "time_normalized",
]


@dataclass
class FeatureImportance:
    """Feature importance result."""
    feature_name: str
    mean_abs_shap: float
    std_shap: float
    rank: int = 0


@dataclass
class SHAPReport:
    """SHAP analysis report."""
    model_path: str
    timestamp: str
    n_samples: int
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    action_breakdown: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "timestamp": self.timestamp,
            "n_samples": self.n_samples,
            "feature_importance": [
                {
                    "rank": fi.rank,
                    "feature": fi.feature_name,
                    "mean_abs_shap": round(fi.mean_abs_shap, 6),
                    "std_shap": round(fi.std_shap, 6),
                }
                for fi in self.feature_importance
            ],
            "action_breakdown": self.action_breakdown,
        }


class SHAPTracker:
    """
    SHAP-based feature importance tracker for PPO models.

    Calculates feature importance using SHAP values to understand
    which features most influence model decisions.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        n_background_samples: int = 100,
    ):
        """
        Initialize SHAP tracker.

        Args:
            model_path: Path to trained model (.zip or .onnx)
            feature_names: Names of features in observation
            n_background_samples: Background samples for SHAP
        """
        self.model_path = model_path
        self.feature_names = feature_names or FEATURE_NAMES
        self.n_background_samples = n_background_samples

        self._model = None
        self._explainer = None
        self._shap_values = None

        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load model for SHAP analysis.

        Args:
            model_path: Path to model file
        """
        path = model_path or self.model_path
        if not path:
            raise ValueError("No model path specified")

        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 required. Install with: pip install stable-baselines3")

        self.model_path = path
        path = Path(path)

        if path.suffix == ".zip" or path.is_dir():
            # Load SB3 model
            self._model = PPO.load(str(path))
            logger.info(f"Loaded PPO model from {path}")
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")

    def _predict_fn(self, observations: np.ndarray) -> np.ndarray:
        """
        Prediction function for SHAP.

        Returns action probabilities for each observation.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Get action distribution from policy
        obs_tensor = self._model.policy.obs_to_tensor(observations)[0]
        distribution = self._model.policy.get_distribution(obs_tensor)

        # Get action probabilities
        probs = distribution.distribution.probs.detach().cpu().numpy()
        return probs

    def calculate_importance(
        self,
        observations: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> List[FeatureImportance]:
        """
        Calculate feature importance using SHAP values.

        Args:
            observations: Sample observations to explain (N x n_features)
            background_data: Background data for SHAP (optional)

        Returns:
            List of FeatureImportance sorted by importance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        if self._model is None:
            if self.model_path:
                self.load_model()
            else:
                raise RuntimeError("No model loaded. Call load_model() first.")

        # Ensure observations are numpy array
        observations = np.asarray(observations)
        if observations.ndim == 1:
            observations = observations.reshape(1, -1)

        # Use subset of data as background if not provided
        if background_data is None:
            n_bg = min(self.n_background_samples, len(observations))
            idx = np.random.choice(len(observations), n_bg, replace=False)
            background_data = observations[idx]

        logger.info(f"Creating SHAP explainer with {len(background_data)} background samples")

        # Create explainer
        self._explainer = shap.KernelExplainer(
            self._predict_fn,
            background_data,
        )

        # Calculate SHAP values (for all actions)
        logger.info(f"Calculating SHAP values for {len(observations)} observations")
        self._shap_values = self._explainer.shap_values(observations)

        # Aggregate importance across all actions
        # shap_values is list of (n_samples, n_features) for each action
        all_shap = np.array(self._shap_values)  # (n_actions, n_samples, n_features)

        # Mean absolute SHAP across actions and samples
        mean_abs_shap = np.abs(all_shap).mean(axis=(0, 1))  # (n_features,)
        std_shap = np.abs(all_shap).std(axis=(0, 1))

        # Create importance list
        importances = []
        for i, name in enumerate(self.feature_names):
            importances.append(FeatureImportance(
                feature_name=name,
                mean_abs_shap=float(mean_abs_shap[i]),
                std_shap=float(std_shap[i]),
            ))

        # Sort by importance and assign ranks
        importances.sort(key=lambda x: x.mean_abs_shap, reverse=True)
        for rank, imp in enumerate(importances, 1):
            imp.rank = rank

        return importances

    def get_action_breakdown(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Get per-action feature importance breakdown.

        Returns:
            Dict mapping action name to list of feature importances
        """
        if self._shap_values is None:
            raise RuntimeError("No SHAP values calculated. Call calculate_importance() first.")

        action_names = ["HOLD", "BUY", "SELL"]
        breakdown = {}

        for action_idx, action_name in enumerate(action_names):
            action_shap = self._shap_values[action_idx]  # (n_samples, n_features)
            mean_abs = np.abs(action_shap).mean(axis=0)

            feature_imp = []
            for i, name in enumerate(self.feature_names):
                feature_imp.append({
                    "feature": name,
                    "mean_abs_shap": round(float(mean_abs[i]), 6),
                })

            # Sort by importance
            feature_imp.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            breakdown[action_name] = feature_imp

        return breakdown

    def generate_report(
        self,
        observations: np.ndarray,
        background_data: Optional[np.ndarray] = None,
    ) -> SHAPReport:
        """
        Generate comprehensive SHAP report.

        Args:
            observations: Sample observations
            background_data: Background data for SHAP

        Returns:
            SHAPReport with all analysis results
        """
        importances = self.calculate_importance(observations, background_data)
        action_breakdown = self.get_action_breakdown()

        return SHAPReport(
            model_path=str(self.model_path),
            timestamp=datetime.now().isoformat(),
            n_samples=len(observations),
            feature_importance=importances,
            action_breakdown=action_breakdown,
        )

    def save_report(
        self,
        output_path: str,
        observations: Optional[np.ndarray] = None,
        background_data: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Generate and save SHAP report to file.

        Args:
            output_path: Path for output JSON
            observations: Sample observations (if not already calculated)
            background_data: Background data for SHAP

        Returns:
            Path to saved report
        """
        if observations is not None:
            report = self.generate_report(observations, background_data)
        elif self._shap_values is not None:
            # Use existing SHAP values
            report = SHAPReport(
                model_path=str(self.model_path),
                timestamp=datetime.now().isoformat(),
                n_samples=len(self._shap_values[0]),
                feature_importance=self._get_importance_from_cached(),
                action_breakdown=self.get_action_breakdown(),
            )
        else:
            raise RuntimeError("No observations provided and no cached SHAP values")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"SHAP report saved to {output_path}")
        return output_path

    def _get_importance_from_cached(self) -> List[FeatureImportance]:
        """Get importance from cached SHAP values."""
        all_shap = np.array(self._shap_values)
        mean_abs_shap = np.abs(all_shap).mean(axis=(0, 1))
        std_shap = np.abs(all_shap).std(axis=(0, 1))

        importances = []
        for i, name in enumerate(self.feature_names):
            importances.append(FeatureImportance(
                feature_name=name,
                mean_abs_shap=float(mean_abs_shap[i]),
                std_shap=float(std_shap[i]),
            ))

        importances.sort(key=lambda x: x.mean_abs_shap, reverse=True)
        for rank, imp in enumerate(importances, 1):
            imp.rank = rank

        return importances

    def format_markdown_table(self, top_n: int = 10) -> str:
        """
        Format feature importance as markdown table.

        Args:
            top_n: Number of top features to include

        Returns:
            Markdown formatted table string
        """
        if self._shap_values is None:
            raise RuntimeError("No SHAP values calculated.")

        importances = self._get_importance_from_cached()

        lines = [
            "| Rank | Feature | Mean |SHAP| |",
            "|------|---------|--------------|",
        ]

        for imp in importances[:top_n]:
            lines.append(f"| {imp.rank} | {imp.feature_name} | {imp.mean_abs_shap:.6f} |")

        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Calculate SHAP feature importance")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to sample observations (numpy .npy file)")
    parser.add_argument("--output", type=str, default="shap_report.json",
                       help="Output path for report")
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples to analyze")
    parser.add_argument("--n-background", type=int, default=100,
                       help="Number of background samples for SHAP")

    args = parser.parse_args()

    # Load observations
    observations = np.load(args.data)
    if len(observations) > args.n_samples:
        idx = np.random.choice(len(observations), args.n_samples, replace=False)
        observations = observations[idx]

    # Calculate and save report
    tracker = SHAPTracker(
        model_path=args.model,
        n_background_samples=args.n_background,
    )
    tracker.save_report(args.output, observations)

    # Print summary
    print("\nTop 5 Most Important Features:")
    print(tracker.format_markdown_table(top_n=5))


if __name__ == "__main__":
    main()
