"""
Run Manifest - Unified lineage tracking for L2->L3->L4 pipeline runs.

Creates a single run_manifest.json per execution that links all pipeline stages
with integrity hashes, delegating hash computation to SSOTLineageIntegration.

Usage:
    manifest = RunManifest(experiment_id="EXP-V215c-001", seed=42)
    manifest.record_l2(l2_result_dict)
    manifest.record_l3(l3_result_dict)
    manifest.record_l4(l4_result_dict, "l4_test")
    manifest.compute_lineage()
    manifest.save(Path("results/runs"))

Contract: CTR-RUN-MANIFEST-001
Version: 1.0.0
"""

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_VERSION = "1.0.0"


def generate_run_id() -> str:
    """Generate a timestamp-based run ID."""
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_git_sha() -> Optional[str]:
    """Get short git commit hash. Returns None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


class RunManifest:
    """
    Unified manifest for a single pipeline run (L2->L3->L4).

    Collects results from each layer and produces a single JSON file
    with cross-references and integrity hashes.
    """

    def __init__(
        self,
        experiment_id: Optional[str] = None,
        seed: int = 42,
        multi_seed: bool = False,
        seeds: Optional[List[int]] = None,
    ):
        self.run_id = generate_run_id()
        self.experiment_id = experiment_id
        self.created_at = datetime.now()

        self._environment = {
            "git_sha": get_git_sha(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }

        self._seeds = {
            "primary": seed,
            "multi_seed": multi_seed,
            "all_seeds": seeds or [seed],
        }

        self._layers: Dict[str, Optional[Dict[str, Any]]] = {
            "l2": None,
            "l3": None,
            "l4_val": None,
            "l4_test": None,
        }

        self._timing: Dict[str, Optional[str]] = {
            "l2_completed_at": None,
            "l3_completed_at": None,
            "l4_val_completed_at": None,
            "l4_test_completed_at": None,
        }

        self._lineage: Optional[Dict[str, Any]] = None

    def record_l2(self, l2_result: Dict[str, Any]) -> None:
        """Record L2 dataset build results."""
        self._layers["l2"] = {
            "train_rows": l2_result.get("train_rows"),
            "val_rows": l2_result.get("val_rows"),
            "test_rows": l2_result.get("test_rows"),
            "feature_columns": l2_result.get("feature_columns"),
            "observation_dim": l2_result.get("observation_dim"),
            "norm_stats_path": l2_result.get("norm_stats_path"),
            "dataset_lineage": l2_result.get("lineage"),
        }
        self._timing["l2_completed_at"] = datetime.now().isoformat()
        logger.info(f"Manifest: L2 recorded ({l2_result.get('train_rows')} train rows)")

    def record_l3(self, l3_result: Dict[str, Any]) -> None:
        """Record L3 training results."""
        self._layers["l3"] = {
            "model_path": l3_result.get("model_path"),
            "model_dir": l3_result.get("model_dir"),
            "norm_stats_path": l3_result.get("norm_stats_path"),
            "timestamp": l3_result.get("timestamp"),
            "use_lstm": l3_result.get("use_lstm", False),
            "training_config": l3_result.get("training_config"),
            "multi_seed_stats": l3_result.get("multi_seed_stats"),
            "best_seed": l3_result.get("best_seed"),
        }
        self._timing["l3_completed_at"] = datetime.now().isoformat()
        logger.info(f"Manifest: L3 recorded (model at {l3_result.get('model_dir')})")

    def record_l4(self, l4_result: Dict[str, Any], stage: str = "l4_test") -> None:
        """Record L4 backtest results.

        Args:
            l4_result: Backtest metrics dict from run_l4_test/run_l4_validation
            stage: "l4_test" or "l4_val"
        """
        key = stage if stage in ("l4_test", "l4_val") else "l4_test"
        self._layers[key] = {
            "total_return_pct": l4_result.get("total_return_pct"),
            "sharpe_ratio": l4_result.get("sharpe_ratio"),
            "max_drawdown_pct": l4_result.get("max_drawdown_pct"),
            "n_trades": l4_result.get("n_trades"),
            "win_rate_pct": l4_result.get("win_rate_pct"),
            "profit_factor": l4_result.get("profit_factor"),
            "sortino_ratio": l4_result.get("sortino_ratio"),
            "gates_passed": l4_result.get("gates_passed"),
            "gate_results": l4_result.get("gate_results"),
            "monthly_returns": l4_result.get("monthly_returns"),
        }
        timing_key = f"{key}_completed_at"
        self._timing[timing_key] = datetime.now().isoformat()
        ret = l4_result.get("total_return_pct", 0)
        logger.info(f"Manifest: {stage.upper()} recorded (return={ret:+.2f}%)")

    def compute_lineage(self) -> None:
        """Compute integrity hashes using SSOTLineageIntegration."""
        try:
            from src.ml_workflow.ssot_lineage_integration import SSOTLineageIntegration

            integration = SSOTLineageIntegration()

            # Resolve artifact paths from recorded layers
            model_path = None
            dataset_path = None
            norm_stats_path = None

            if self._layers["l3"]:
                model_dir = self._layers["l3"].get("model_dir")
                if model_dir:
                    best = Path(model_dir) / "best_model.zip"
                    final = Path(model_dir) / "final_model.zip"
                    model_path = best if best.exists() else (final if final.exists() else None)
                ns = self._layers["l3"].get("norm_stats_path")
                if ns:
                    norm_stats_path = Path(ns)

            if self._layers["l2"]:
                tp = self._layers["l2"].get("norm_stats_path")
                if tp and not norm_stats_path:
                    norm_stats_path = Path(tp)
                # Use train dataset for hash
                lineage_data = self._layers["l2"].get("dataset_lineage") or {}
                train_path = lineage_data.get("train_path")
                if not train_path and self._layers["l2"]:
                    # Fallback: reconstruct from norm_stats_path directory
                    ns_path = self._layers["l2"].get("norm_stats_path")
                    if ns_path:
                        candidate = Path(ns_path).parent / "DS_production_train.parquet"
                        if candidate.exists():
                            train_path = str(candidate)
                if train_path:
                    dataset_path = Path(train_path)

            record = integration.create_lineage_record(
                run_id=self.run_id,
                stage="full_pipeline",
                model_path=model_path,
                dataset_path=dataset_path,
                norm_stats_path=norm_stats_path,
            )

            self._lineage = {
                "ssot": {
                    "version": record.ssot_version,
                    "config_hash": record.ssot_config_hash,
                },
                "features": {
                    "order_hash": record.feature_order_hash,
                    "observation_dim": record.observation_dim,
                },
                "hashes": {
                    "dataset": record.dataset_hash,
                    "norm_stats": record.norm_stats_hash,
                    "model": record.model_hash,
                },
                "parity": {
                    "verified": record.training_backtest_parity_verified,
                    "issues": record.parity_issues,
                },
            }
            logger.info(
                f"Manifest: Lineage computed (ssot_v={record.ssot_version}, "
                f"parity={'OK' if record.training_backtest_parity_verified else 'FAIL'})"
            )

        except Exception as e:
            logger.warning(f"Manifest: Lineage computation failed: {e}")
            self._lineage = {"error": str(e)}

    def _build_summary(self) -> Dict[str, Any]:
        """Build summary section from recorded layers."""
        stages_completed = []
        if self._layers["l2"]:
            stages_completed.append("L2")
        if self._layers["l3"]:
            stages_completed.append("L3")
        if self._layers["l4_val"]:
            stages_completed.append("L4-VAL")
        if self._layers["l4_test"]:
            stages_completed.append("L4-TEST")

        # Use l4_test for summary metrics, fallback to l4_val
        l4 = self._layers.get("l4_test") or self._layers.get("l4_val")

        summary: Dict[str, Any] = {
            "stages_completed": stages_completed,
            "overall_result": None,
            "total_return_pct": None,
            "sharpe_ratio": None,
            "gates_passed": None,
        }

        if l4:
            summary["total_return_pct"] = l4.get("total_return_pct")
            summary["sharpe_ratio"] = l4.get("sharpe_ratio")
            summary["gates_passed"] = l4.get("gates_passed")
            if l4.get("gates_passed") is True:
                summary["overall_result"] = "PASSED"
            elif l4.get("gates_passed") is False:
                summary["overall_result"] = "FAILED"

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        return {
            "manifest_version": MANIFEST_VERSION,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat(),
            "environment": self._environment,
            "seeds": self._seeds,
            "layers": self._layers,
            "lineage": self._lineage,
            "timing": self._timing,
            "summary": self._build_summary(),
        }

    def save(self, runs_dir: Path) -> Path:
        """Save manifest to results/runs/{run_id}/run_manifest.json.

        Args:
            runs_dir: Parent directory (e.g. results/runs)

        Returns:
            Path to saved manifest file
        """
        run_dir = runs_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = run_dir / "run_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Manifest saved: {manifest_path}")
        return manifest_path
