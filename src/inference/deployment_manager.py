"""
Deployment Manager - Rollback and Canary Deployment
====================================================

Manages model deployment lifecycle including:
- Shadow mode validation
- Canary rollout with traffic splitting
- Automatic rollback on performance degradation
- Blue-green deployment switching

Author: Trading Team
Version: 1.0.0
Date: 2026-01-18
Contract: CTR-DEPLOY-001
"""

import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class DeploymentStage(str, Enum):
    """Model deployment stages."""
    SHADOW = "shadow"
    CANARY_10 = "canary_10"
    CANARY_50 = "canary_50"
    PRODUCTION = "production"
    ROLLED_BACK = "rolled_back"
    DEPRECATED = "deprecated"


class DeploymentAction(str, Enum):
    """Actions that can be taken on a deployment."""
    PROMOTE = "promote"
    ROLLBACK = "rollback"
    PAUSE = "pause"
    RESUME = "resume"
    REJECT = "reject"


@dataclass
class ModelDeployment:
    """Represents a model deployment state."""
    model_id: str
    model_path: str
    model_hash: str
    stage: DeploymentStage
    deployed_at: datetime
    promoted_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    rollback_reason: Optional[str] = None
    traffic_percent: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    config_hash: Optional[str] = None
    mlflow_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "stage": self.stage.value,
            "deployed_at": self.deployed_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "rollback_reason": self.rollback_reason,
            "traffic_percent": self.traffic_percent,
            "metrics": self.metrics,
            "config_hash": self.config_hash,
            "mlflow_run_id": self.mlflow_run_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelDeployment":
        return cls(
            model_id=data["model_id"],
            model_path=data["model_path"],
            model_hash=data["model_hash"],
            stage=DeploymentStage(data["stage"]),
            deployed_at=datetime.fromisoformat(data["deployed_at"]),
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
            rolled_back_at=datetime.fromisoformat(data["rolled_back_at"]) if data.get("rolled_back_at") else None,
            rollback_reason=data.get("rollback_reason"),
            traffic_percent=data.get("traffic_percent", 0.0),
            metrics=data.get("metrics", {}),
            config_hash=data.get("config_hash"),
            mlflow_run_id=data.get("mlflow_run_id"),
        )


@dataclass
class RollbackTrigger:
    """Configuration for a rollback trigger."""
    name: str
    metric: str
    condition: str  # 'less_than', 'greater_than'
    threshold: float
    action: str
    alert_severity: str
    cooldown_hours: int


@dataclass
class CanaryStageConfig:
    """Configuration for a canary deployment stage."""
    name: str
    traffic_percent: int
    duration_hours: int
    success_criteria: Dict[str, float]
    on_failure: str
    on_success: str


@dataclass
class PromotionDecision:
    """Result of a promotion decision."""
    should_promote: bool
    current_stage: DeploymentStage
    next_stage: Optional[DeploymentStage]
    reason: str
    metrics: Dict[str, float]
    criteria_met: Dict[str, bool]


class DeploymentManager:
    """
    Manages model deployment lifecycle.

    Supports:
    - Multi-stage canary deployment
    - Traffic splitting between champion and challenger
    - Automatic rollback on performance degradation
    - Deployment state persistence
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        state_path: Optional[Path] = None,
    ):
        """
        Initialize the deployment manager.

        Args:
            config_path: Path to deployment_safeguards.yaml
            state_path: Path to persist deployment state
        """
        self.config_path = config_path or Path("config/deployment_safeguards.yaml")
        self.state_path = state_path or Path("data/deployment_state.json")

        self._load_config()
        self._load_state()

    def _load_config(self):
        """Load configuration from YAML."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                self.rollback_config = config.get("rollback", {})
                self.canary_config = config.get("canary", {})
                self.monitoring_config = config.get("monitoring", {})
        else:
            logger.warning(f"[Deploy] Config not found: {self.config_path}")
            self.rollback_config = {}
            self.canary_config = {}
            self.monitoring_config = {}

        # Parse rollback triggers
        self.rollback_triggers = []
        for trigger_data in self.rollback_config.get("triggers", []):
            self.rollback_triggers.append(RollbackTrigger(
                name=trigger_data["name"],
                metric=trigger_data["metric"],
                condition=trigger_data["condition"],
                threshold=trigger_data["threshold"],
                action=trigger_data["action"],
                alert_severity=trigger_data["alert_severity"],
                cooldown_hours=trigger_data["cooldown_hours"],
            ))

        # Parse canary stages
        self.canary_stages = []
        for stage_data in self.canary_config.get("stages", []):
            self.canary_stages.append(CanaryStageConfig(
                name=stage_data["name"],
                traffic_percent=stage_data["traffic_percent"],
                duration_hours=stage_data["duration_hours"],
                success_criteria=stage_data.get("success_criteria") or {},
                on_failure=stage_data["on_failure"],
                on_success=stage_data["on_success"],
            ))

    def _load_state(self):
        """Load deployment state from disk."""
        self.champion: Optional[ModelDeployment] = None
        self.challenger: Optional[ModelDeployment] = None
        self.deployment_history: List[ModelDeployment] = []
        self.last_rollback: Optional[datetime] = None

        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    state = json.load(f)

                if state.get("champion"):
                    self.champion = ModelDeployment.from_dict(state["champion"])
                if state.get("challenger"):
                    self.challenger = ModelDeployment.from_dict(state["challenger"])
                if state.get("history"):
                    self.deployment_history = [
                        ModelDeployment.from_dict(d) for d in state["history"]
                    ]
                if state.get("last_rollback"):
                    self.last_rollback = datetime.fromisoformat(state["last_rollback"])

                logger.info(f"[Deploy] Loaded state: champion={self.champion.model_id if self.champion else None}")
            except Exception as e:
                logger.error(f"[Deploy] Failed to load state: {e}")

    def _save_state(self):
        """Persist deployment state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "champion": self.champion.to_dict() if self.champion else None,
            "challenger": self.challenger.to_dict() if self.challenger else None,
            "history": [d.to_dict() for d in self.deployment_history[-50:]],  # Keep last 50
            "last_rollback": self.last_rollback.isoformat() if self.last_rollback else None,
            "updated_at": datetime.now().isoformat(),
        }

        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("[Deploy] State saved")

    # =========================================================================
    # Deployment Operations
    # =========================================================================

    def deploy_to_shadow(
        self,
        model_id: str,
        model_path: str,
        model_hash: str,
        config_hash: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> ModelDeployment:
        """
        Deploy a new model to shadow mode.

        In shadow mode, the model runs in parallel with the champion
        but does not execute real trades.

        Args:
            model_id: Unique model identifier
            model_path: Path to the model file
            model_hash: SHA256 hash of the model
            config_hash: Hash of the training config
            mlflow_run_id: MLflow run ID for tracking

        Returns:
            The new deployment
        """
        deployment = ModelDeployment(
            model_id=model_id,
            model_path=model_path,
            model_hash=model_hash,
            stage=DeploymentStage.SHADOW,
            deployed_at=datetime.now(),
            traffic_percent=0.0,
            config_hash=config_hash,
            mlflow_run_id=mlflow_run_id,
        )

        self.challenger = deployment
        self._save_state()

        logger.info(f"[Deploy] Model {model_id} deployed to SHADOW mode")
        return deployment

    def promote(self, metrics: Dict[str, float]) -> PromotionDecision:
        """
        Attempt to promote the challenger to the next stage.

        Args:
            metrics: Current performance metrics

        Returns:
            PromotionDecision with the result
        """
        if not self.challenger:
            return PromotionDecision(
                should_promote=False,
                current_stage=DeploymentStage.SHADOW,
                next_stage=None,
                reason="No challenger model to promote",
                metrics=metrics,
                criteria_met={},
            )

        current_stage = self.challenger.stage
        stage_config = self._get_stage_config(current_stage)

        if not stage_config:
            return PromotionDecision(
                should_promote=False,
                current_stage=current_stage,
                next_stage=None,
                reason=f"No configuration for stage {current_stage}",
                metrics=metrics,
                criteria_met={},
            )

        # Check if enough time has passed
        stage_duration = timedelta(hours=stage_config.duration_hours)
        time_in_stage = datetime.now() - self.challenger.deployed_at

        if time_in_stage < stage_duration:
            remaining = stage_duration - time_in_stage
            return PromotionDecision(
                should_promote=False,
                current_stage=current_stage,
                next_stage=None,
                reason=f"Insufficient time in stage ({remaining} remaining)",
                metrics=metrics,
                criteria_met={},
            )

        # Check success criteria
        criteria_met = {}
        for criterion, threshold in stage_config.success_criteria.items():
            if criterion.startswith("min_"):
                metric_name = criterion[4:]
                criteria_met[criterion] = metrics.get(metric_name, 0) >= threshold
            elif criterion.startswith("max_"):
                metric_name = criterion[4:]
                criteria_met[criterion] = metrics.get(metric_name, float("inf")) <= threshold
            else:
                criteria_met[criterion] = metrics.get(criterion, 0) >= threshold

        all_criteria_met = all(criteria_met.values())

        if not all_criteria_met:
            failed = [k for k, v in criteria_met.items() if not v]
            return PromotionDecision(
                should_promote=False,
                current_stage=current_stage,
                next_stage=None,
                reason=f"Criteria not met: {failed}",
                metrics=metrics,
                criteria_met=criteria_met,
            )

        # Determine next stage
        next_stage = self._get_next_stage(current_stage)

        if next_stage:
            self._execute_promotion(next_stage, metrics)

        return PromotionDecision(
            should_promote=True,
            current_stage=current_stage,
            next_stage=next_stage,
            reason="All criteria met",
            metrics=metrics,
            criteria_met=criteria_met,
        )

    def _execute_promotion(self, next_stage: DeploymentStage, metrics: Dict[str, float]):
        """Execute the promotion to the next stage."""
        if not self.challenger:
            return

        self.challenger.stage = next_stage
        self.challenger.promoted_at = datetime.now()
        self.challenger.metrics = metrics

        # Update traffic percent
        stage_config = self._get_stage_config(next_stage)
        if stage_config:
            self.challenger.traffic_percent = stage_config.traffic_percent

        # If promoted to production, swap champion
        if next_stage == DeploymentStage.PRODUCTION:
            if self.champion:
                self.champion.stage = DeploymentStage.DEPRECATED
                self.deployment_history.append(self.champion)

            self.champion = self.challenger
            self.challenger = None

        self._save_state()
        logger.info(f"[Deploy] Promoted to {next_stage.value}")

    def _get_stage_config(self, stage: DeploymentStage) -> Optional[CanaryStageConfig]:
        """Get configuration for a deployment stage."""
        stage_name_map = {
            DeploymentStage.SHADOW: "shadow",
            DeploymentStage.CANARY_10: "canary_10",
            DeploymentStage.CANARY_50: "canary_50",
            DeploymentStage.PRODUCTION: "production",
        }

        stage_name = stage_name_map.get(stage)
        for config in self.canary_stages:
            if config.name == stage_name:
                return config
        return None

    def _get_next_stage(self, current: DeploymentStage) -> Optional[DeploymentStage]:
        """Get the next stage in the promotion sequence."""
        sequence = [
            DeploymentStage.SHADOW,
            DeploymentStage.CANARY_10,
            DeploymentStage.CANARY_50,
            DeploymentStage.PRODUCTION,
        ]

        try:
            idx = sequence.index(current)
            if idx < len(sequence) - 1:
                return sequence[idx + 1]
        except ValueError:
            pass

        return None

    # =========================================================================
    # Rollback Operations
    # =========================================================================

    def check_rollback_triggers(self, metrics: Dict[str, float]) -> Optional[RollbackTrigger]:
        """
        Check if any rollback trigger conditions are met.

        Args:
            metrics: Current performance metrics

        Returns:
            The triggered rollback condition, or None
        """
        # Check cooldown
        if self.last_rollback:
            cooldown = timedelta(hours=24)  # Default cooldown
            if datetime.now() - self.last_rollback < cooldown:
                logger.debug("[Deploy] In rollback cooldown period")
                return None

        for trigger in self.rollback_triggers:
            metric_value = metrics.get(trigger.metric)

            if metric_value is None:
                continue

            triggered = False
            if trigger.condition == "less_than":
                triggered = metric_value < trigger.threshold
            elif trigger.condition == "greater_than":
                triggered = metric_value > trigger.threshold

            if triggered:
                logger.warning(
                    f"[Deploy] Rollback trigger fired: {trigger.name} "
                    f"({trigger.metric}={metric_value} {trigger.condition} {trigger.threshold})"
                )
                return trigger

        return None

    def rollback(self, reason: str) -> bool:
        """
        Rollback to the previous known good model.

        Args:
            reason: Reason for rollback

        Returns:
            True if rollback successful
        """
        # Find previous model
        previous = None
        for deployment in reversed(self.deployment_history):
            if deployment.stage == DeploymentStage.DEPRECATED:
                previous = deployment
                break

        if not previous:
            # Use fallback from config
            fallback_config = self.rollback_config.get("fallback", {})
            fallback_path = fallback_config.get("baseline_model_path")

            if fallback_path and Path(fallback_path).exists():
                logger.info(f"[Deploy] Using fallback model: {fallback_path}")

                previous = ModelDeployment(
                    model_id="fallback_baseline",
                    model_path=fallback_path,
                    model_hash=self._compute_file_hash(fallback_path),
                    stage=DeploymentStage.PRODUCTION,
                    deployed_at=datetime.now(),
                )
            else:
                logger.error("[Deploy] No previous model available for rollback")
                return False

        # Execute rollback
        if self.champion:
            self.champion.stage = DeploymentStage.ROLLED_BACK
            self.champion.rolled_back_at = datetime.now()
            self.champion.rollback_reason = reason
            self.deployment_history.append(self.champion)

        previous.stage = DeploymentStage.PRODUCTION
        previous.deployed_at = datetime.now()
        previous.traffic_percent = 100.0

        self.champion = previous
        self.challenger = None
        self.last_rollback = datetime.now()

        self._save_state()

        logger.info(f"[Deploy] Rolled back to {previous.model_id}: {reason}")
        return True

    def _compute_file_hash(self, path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]

    # =========================================================================
    # Traffic Splitting
    # =========================================================================

    def should_use_challenger(self, trade_id: Optional[str] = None) -> bool:
        """
        Determine if a trade should use the challenger model.

        Uses deterministic splitting based on trade_id for reproducibility.

        Args:
            trade_id: Unique trade identifier (for deterministic splitting)

        Returns:
            True if challenger should be used
        """
        if not self.challenger or self.challenger.traffic_percent == 0:
            return False

        if self.challenger.traffic_percent >= 100:
            return True

        # Deterministic splitting
        if trade_id:
            hash_value = int(hashlib.md5(trade_id.encode()).hexdigest(), 16)
            return (hash_value % 100) < self.challenger.traffic_percent

        # Random splitting fallback
        return random.random() * 100 < self.challenger.traffic_percent

    def get_active_model_path(self, trade_id: Optional[str] = None) -> str:
        """
        Get the path of the model to use for a trade.

        Args:
            trade_id: Trade identifier for traffic splitting

        Returns:
            Path to the model file
        """
        if self.should_use_challenger(trade_id):
            return self.challenger.model_path

        if self.champion:
            return self.champion.model_path

        raise RuntimeError("No active model available")

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "champion": self.champion.to_dict() if self.champion else None,
            "challenger": self.challenger.to_dict() if self.challenger else None,
            "last_rollback": self.last_rollback.isoformat() if self.last_rollback else None,
            "history_count": len(self.deployment_history),
            "rollback_enabled": self.rollback_config.get("enabled", False),
            "canary_enabled": self.canary_config.get("enabled", False),
        }


# =============================================================================
# Convenience functions for DAG/API usage
# =============================================================================

_manager_instance: Optional[DeploymentManager] = None


def get_deployment_manager() -> DeploymentManager:
    """Get singleton deployment manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DeploymentManager()
    return _manager_instance


def deploy_model_to_shadow(
    model_id: str,
    model_path: str,
    model_hash: str,
    **kwargs,
) -> Dict[str, Any]:
    """Deploy a model to shadow mode."""
    manager = get_deployment_manager()
    deployment = manager.deploy_to_shadow(model_id, model_path, model_hash, **kwargs)
    return deployment.to_dict()


def check_and_promote(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Check promotion criteria and promote if met."""
    manager = get_deployment_manager()
    decision = manager.promote(metrics)
    return {
        "promoted": decision.should_promote,
        "current_stage": decision.current_stage.value,
        "next_stage": decision.next_stage.value if decision.next_stage else None,
        "reason": decision.reason,
        "criteria_met": decision.criteria_met,
    }


def check_rollback_needed(metrics: Dict[str, float]) -> Dict[str, Any]:
    """Check if rollback is needed based on metrics."""
    manager = get_deployment_manager()
    trigger = manager.check_rollback_triggers(metrics)

    if trigger:
        if trigger.action == "rollback_to_previous":
            success = manager.rollback(f"Trigger: {trigger.name}")
            return {
                "rollback_needed": True,
                "rollback_executed": success,
                "trigger": trigger.name,
                "action": trigger.action,
            }
        else:
            return {
                "rollback_needed": True,
                "rollback_executed": False,
                "trigger": trigger.name,
                "action": trigger.action,
            }

    return {"rollback_needed": False}


def get_active_model() -> str:
    """Get path to the active model."""
    manager = get_deployment_manager()
    return manager.get_active_model_path()
