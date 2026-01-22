#!/usr/bin/env python3
"""
Experiment Traceability Tool
============================

Provides complete traceability for MLflow experiments, showing the full
lineage from model to dataset, code, and configuration.

Part of TRACE-16 remediation from Experimentation Audit.

Features:
- Full experiment lineage tree
- Dataset traceability (hash, date range, source)
- Feature contract verification
- Git commit tracing
- Artifact listing
- Export to JSON/markdown

Usage:
    # Trace by run ID
    python scripts/trace_experiment.py --run-id abc123

    # Trace by model version
    python scripts/trace_experiment.py --model-name usdcop-ppo-model --version 3

    # Export to markdown
    python scripts/trace_experiment.py --run-id abc123 -o trace.md

    # Export to JSON
    python scripts/trace_experiment.py --run-id abc123 --format json

Author: Trading Team
Date: 2026-01-17
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatasetLineage:
    """Dataset lineage information."""
    dataset_hash: Optional[str] = None
    dataset_path: Optional[str] = None
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    dataset_rows: Optional[int] = None
    dataset_columns: Optional[int] = None
    feature_order_hash: Optional[str] = None


@dataclass
class GitLineage:
    """Git repository lineage information."""
    commit: Optional[str] = None
    commit_short: Optional[str] = None
    branch: Optional[str] = None
    is_dirty: Optional[str] = None
    author: Optional[str] = None
    commit_date: Optional[str] = None
    remote_url: Optional[str] = None


@dataclass
class ModelLineage:
    """Model registry lineage information."""
    model_name: Optional[str] = None
    model_version: Optional[int] = None
    stage: Optional[str] = None
    creation_time: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentLineage:
    """Complete experiment lineage."""
    run_id: str
    run_name: str
    experiment_id: str
    experiment_name: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]

    # Lineage components
    dataset: DatasetLineage = field(default_factory=DatasetLineage)
    git: GitLineage = field(default_factory=GitLineage)
    model: Optional[ModelLineage] = None

    # Configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    network: Dict[str, Any] = field(default_factory=dict)

    # Results
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "dataset": {
                "hash": self.dataset.dataset_hash,
                "path": self.dataset.dataset_path,
                "train_start": self.dataset.train_start_date,
                "train_end": self.dataset.train_end_date,
                "rows": self.dataset.dataset_rows,
                "columns": self.dataset.dataset_columns,
                "feature_order_hash": self.dataset.feature_order_hash,
            },
            "git": {
                "commit": self.git.commit,
                "branch": self.git.branch,
                "is_dirty": self.git.is_dirty,
                "author": self.git.author,
                "commit_date": self.git.commit_date,
            },
            "model": {
                "name": self.model.model_name if self.model else None,
                "version": self.model.model_version if self.model else None,
                "stage": self.model.stage if self.model else None,
            } if self.model else None,
            "hyperparameters": self.hyperparameters,
            "environment": self.environment,
            "network": self.network,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


# =============================================================================
# Tracer Class
# =============================================================================

class ExperimentTracer:
    """Trace experiment lineage from MLflow."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize the tracer.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def trace_run(self, run_id: str) -> ExperimentLineage:
        """
        Trace full lineage for a run.

        Args:
            run_id: MLflow run ID

        Returns:
            ExperimentLineage with full traceability
        """
        run = self.client.get_run(run_id)
        experiment = self.client.get_experiment(run.info.experiment_id)

        # Build base lineage
        lineage = ExperimentLineage(
            run_id=run.info.run_id,
            run_name=run.data.tags.get("mlflow.runName", run.info.run_id[:8]),
            experiment_id=run.info.experiment_id,
            experiment_name=experiment.name,
            status=run.info.status,
            start_time=datetime.fromtimestamp(run.info.start_time / 1000),
            end_time=datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
            tags=dict(run.data.tags),
        )

        # Extract dataset lineage
        lineage.dataset = self._extract_dataset_lineage(run.data.params)

        # Extract git lineage
        lineage.git = self._extract_git_lineage(run.data.params, run.data.tags)

        # Extract configuration
        lineage.hyperparameters = self._extract_hyperparameters(run.data.params)
        lineage.environment = self._extract_environment(run.data.params)
        lineage.network = self._extract_network(run.data.params)

        # Extract metrics
        lineage.metrics = dict(run.data.metrics)

        # List artifacts
        try:
            lineage.artifacts = [a.path for a in self.client.list_artifacts(run_id)]
        except Exception:
            lineage.artifacts = []

        # Check for registered model
        lineage.model = self._find_registered_model(run_id)

        return lineage

    def trace_model_version(
        self,
        model_name: str,
        version: int
    ) -> ExperimentLineage:
        """
        Trace lineage for a registered model version.

        Args:
            model_name: Name of the registered model
            version: Model version number

        Returns:
            ExperimentLineage for the model's training run
        """
        model_version = self.client.get_model_version(model_name, str(version))
        run_id = model_version.run_id

        if not run_id:
            raise ValueError(f"Model {model_name} v{version} has no associated run")

        lineage = self.trace_run(run_id)

        # Add model info
        lineage.model = ModelLineage(
            model_name=model_name,
            model_version=version,
            stage=model_version.current_stage,
            creation_time=datetime.fromtimestamp(model_version.creation_timestamp / 1000),
            last_updated=datetime.fromtimestamp(model_version.last_updated_timestamp / 1000),
            description=model_version.description,
            tags=dict(model_version.tags) if model_version.tags else {},
        )

        return lineage

    def _extract_dataset_lineage(self, params: Dict[str, Any]) -> DatasetLineage:
        """Extract dataset lineage from run parameters."""
        return DatasetLineage(
            dataset_hash=params.get("dataset_hash"),
            dataset_path=params.get("dataset_path"),
            train_start_date=params.get("train_start_date"),
            train_end_date=params.get("train_end_date"),
            dataset_rows=int(params["dataset_rows"]) if "dataset_rows" in params else None,
            dataset_columns=int(params["dataset_columns"]) if "dataset_columns" in params else None,
            feature_order_hash=params.get("feature_order_hash"),
        )

    def _extract_git_lineage(
        self,
        params: Dict[str, Any],
        tags: Dict[str, str]
    ) -> GitLineage:
        """Extract git lineage from run parameters and tags."""
        return GitLineage(
            commit=params.get("git_commit") or tags.get("git.commit"),
            commit_short=params.get("git_commit_short"),
            branch=params.get("git_branch") or tags.get("git.branch"),
            is_dirty=params.get("git_is_dirty") or tags.get("git.is_dirty"),
            author=params.get("git_author"),
            commit_date=params.get("git_commit_date"),
            remote_url=params.get("git_remote_url"),
        )

    def _extract_hyperparameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hyperparameters from run parameters."""
        hp = {}
        for key, value in params.items():
            if key.startswith("hp_"):
                hp[key[3:]] = self._parse_value(value)
        return hp

    def _extract_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract environment config from run parameters."""
        env = {}
        for key, value in params.items():
            if key.startswith("env_"):
                env[key[4:]] = self._parse_value(value)
        return env

    def _extract_network(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network architecture from run parameters."""
        return {
            "pi": params.get("network_pi"),
            "vf": params.get("network_vf"),
            "activation": params.get("activation"),
        }

    def _parse_value(self, value: str) -> Any:
        """Parse string value to appropriate type."""
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                return value

    def _find_registered_model(self, run_id: str) -> Optional[ModelLineage]:
        """Find registered model for a run."""
        try:
            # Search all registered models
            for model in self.client.search_registered_models():
                for version in self.client.search_model_versions(f"name='{model.name}'"):
                    if version.run_id == run_id:
                        return ModelLineage(
                            model_name=model.name,
                            model_version=int(version.version),
                            stage=version.current_stage,
                            creation_time=datetime.fromtimestamp(version.creation_timestamp / 1000),
                            last_updated=datetime.fromtimestamp(version.last_updated_timestamp / 1000),
                            description=version.description,
                        )
        except Exception:
            pass
        return None


# =============================================================================
# Output Formatters
# =============================================================================

def format_markdown(lineage: ExperimentLineage) -> str:
    """Format lineage as markdown."""
    lines = []

    lines.append("# Experiment Lineage Report\n")
    lines.append(f"**Generated**: {datetime.now().isoformat()}\n")

    # Run info
    lines.append("## Run Information\n")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Run ID** | `{lineage.run_id}` |")
    lines.append(f"| **Run Name** | {lineage.run_name} |")
    lines.append(f"| **Experiment** | {lineage.experiment_name} |")
    lines.append(f"| **Status** | {lineage.status} |")
    lines.append(f"| **Started** | {lineage.start_time} |")
    lines.append(f"| **Ended** | {lineage.end_time or 'N/A'} |")
    lines.append("")

    # Dataset lineage
    lines.append("## Dataset Lineage\n")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Dataset Hash** | `{lineage.dataset.dataset_hash or 'N/A'}` |")
    lines.append(f"| **Dataset Path** | {lineage.dataset.dataset_path or 'N/A'} |")
    lines.append(f"| **Train Start** | {lineage.dataset.train_start_date or 'N/A'} |")
    lines.append(f"| **Train End** | {lineage.dataset.train_end_date or 'N/A'} |")
    lines.append(f"| **Rows** | {lineage.dataset.dataset_rows or 'N/A'} |")
    lines.append(f"| **Columns** | {lineage.dataset.dataset_columns or 'N/A'} |")
    lines.append(f"| **Feature Order Hash** | `{lineage.dataset.feature_order_hash or 'N/A'}` |")
    lines.append("")

    # Git lineage
    lines.append("## Git Lineage\n")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    lines.append(f"| **Commit** | `{lineage.git.commit or 'N/A'}` |")
    lines.append(f"| **Branch** | {lineage.git.branch or 'N/A'} |")
    lines.append(f"| **Dirty** | {lineage.git.is_dirty or 'N/A'} |")
    lines.append(f"| **Author** | {lineage.git.author or 'N/A'} |")
    lines.append(f"| **Commit Date** | {lineage.git.commit_date or 'N/A'} |")
    lines.append("")

    # Model lineage
    if lineage.model:
        lines.append("## Registered Model\n")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| **Model Name** | {lineage.model.model_name} |")
        lines.append(f"| **Version** | {lineage.model.model_version} |")
        lines.append(f"| **Stage** | {lineage.model.stage} |")
        lines.append(f"| **Created** | {lineage.model.creation_time} |")
        lines.append("")

    # Hyperparameters
    lines.append("## Hyperparameters\n")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    for key, value in sorted(lineage.hyperparameters.items()):
        lines.append(f"| {key} | {value} |")
    lines.append("")

    # Metrics
    lines.append("## Metrics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    for key, value in sorted(lineage.metrics.items()):
        lines.append(f"| {key} | {value:.6f} |")
    lines.append("")

    # Artifacts
    if lineage.artifacts:
        lines.append("## Artifacts\n")
        for artifact in lineage.artifacts:
            lines.append(f"- `{artifact}`")
        lines.append("")

    # Lineage diagram
    lines.append("## Lineage Diagram\n")
    lines.append("```")
    lines.append("Code                    Data                    Model")
    lines.append("────                    ────                    ─────")
    commit_short = lineage.git.commit[:8] if lineage.git.commit else "N/A"
    lines.append(f"git:{commit_short}  →  dataset:{lineage.dataset.dataset_hash or 'N/A'}  →  run:{lineage.run_id[:8]}")
    if lineage.model:
        lines.append(f"                                                    ↓")
        lines.append(f"                                              {lineage.model.model_name} v{lineage.model.model_version}")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("*Report generated by trace_experiment.py*")

    return "\n".join(lines)


def format_json(lineage: ExperimentLineage) -> str:
    """Format lineage as JSON."""
    return json.dumps(lineage.to_dict(), indent=2, default=str)


def format_text(lineage: ExperimentLineage) -> str:
    """Format lineage as plain text."""
    lines = []
    lines.append("=" * 60)
    lines.append("EXPERIMENT LINEAGE REPORT")
    lines.append("=" * 60)
    lines.append(f"Run ID: {lineage.run_id}")
    lines.append(f"Run Name: {lineage.run_name}")
    lines.append(f"Experiment: {lineage.experiment_name}")
    lines.append(f"Status: {lineage.status}")
    lines.append("")
    lines.append("DATASET:")
    lines.append(f"  Hash: {lineage.dataset.dataset_hash}")
    lines.append(f"  Date Range: {lineage.dataset.train_start_date} to {lineage.dataset.train_end_date}")
    lines.append(f"  Rows: {lineage.dataset.dataset_rows}")
    lines.append("")
    lines.append("GIT:")
    lines.append(f"  Commit: {lineage.git.commit}")
    lines.append(f"  Branch: {lineage.git.branch}")
    lines.append(f"  Dirty: {lineage.git.is_dirty}")
    lines.append("")
    if lineage.model:
        lines.append("MODEL:")
        lines.append(f"  Name: {lineage.model.model_name}")
        lines.append(f"  Version: {lineage.model.model_version}")
        lines.append(f"  Stage: {lineage.model.stage}")
    lines.append("=" * 60)
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trace experiment lineage from MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trace by run ID
  python trace_experiment.py --run-id abc123

  # Trace by model version
  python trace_experiment.py --model-name usdcop-ppo-model --version 3

  # Export to markdown
  python trace_experiment.py --run-id abc123 -o report.md

  # JSON output
  python trace_experiment.py --run-id abc123 --format json
        """
    )

    # Run selection
    parser.add_argument(
        "--run-id", "--run_id",
        type=str,
        help="MLflow run ID to trace"
    )
    parser.add_argument(
        "--model-name", "--model_name",
        type=str,
        help="Registered model name"
    )
    parser.add_argument(
        "--version",
        type=int,
        help="Model version number"
    )

    # Output options
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "text"],
        default="markdown",
        help="Output format (default: markdown)"
    )

    # MLflow options
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Validate arguments
    if not args.run_id and not (args.model_name and args.version):
        print("Error: Must specify either --run-id or both --model-name and --version")
        return 1

    try:
        tracer = ExperimentTracer(tracking_uri=args.tracking_uri)

        # Get lineage
        if args.run_id:
            lineage = tracer.trace_run(args.run_id)
        else:
            lineage = tracer.trace_model_version(args.model_name, args.version)

        # Format output
        if args.format == "markdown":
            output = format_markdown(lineage)
        elif args.format == "json":
            output = format_json(lineage)
        else:
            output = format_text(lineage)

        # Write output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Lineage report saved to: {output_path}")
        else:
            print(output)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
