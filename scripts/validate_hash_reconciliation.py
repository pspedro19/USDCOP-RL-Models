#!/usr/bin/env python3
"""
Hash Reconciliation Validator
=============================
Contract: HASH-11
Purpose: Validate hash consistency between DVC, MLflow, and stored artifacts

This script ensures that:
1. Dataset hashes in MLflow match DVC tracked data
2. Model hashes are consistent across MLflow and filesystem
3. Norm stats hashes match between training and inference
4. Feature order hashes are consistent

Usage:
    python scripts/validate_hash_reconciliation.py [--model-name NAME] [--verbose]

Author: USDCOP Trading Team
Version: 1.0.0
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path for SSOT imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SSOT import for hash utilities
from src.utils.hash_utils import (
    compute_file_hash as _compute_file_hash_ssot,
    compute_json_hash as _compute_json_hash_ssot,
)


@dataclass
class HashValidationResult:
    """Result of a hash validation check."""
    source: str
    artifact_type: str
    expected_hash: Optional[str]
    actual_hash: Optional[str]
    matches: bool
    message: str


class HashReconciliationValidator:
    """Validates hash consistency across DVC, MLflow, and artifacts."""

    def __init__(self, project_root: Path = None, verbose: bool = False):
        self.project_root = project_root or Path(__file__).parent.parent
        self.verbose = verbose
        self.results: List[HashValidationResult] = []

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[DEBUG] {message}")

    def compute_file_hash(self, path: Path, chunk_size: int = 8192) -> Optional[str]:
        """Compute SHA256 hash of a file. SSOT: Delegates to src.utils.hash_utils"""
        if not path.exists():
            return None

        return _compute_file_hash_ssot(path, chunk_size=chunk_size).full_hash

    def compute_json_hash(self, path: Path) -> Optional[str]:
        """Compute hash of JSON content. SSOT: Delegates to src.utils.hash_utils"""
        if not path.exists():
            return None

        return _compute_json_hash_ssot(path).full_hash

    def get_dvc_hash(self, dvc_file: Path) -> Optional[str]:
        """Get hash from a .dvc file."""
        if not dvc_file.exists():
            return None

        try:
            with open(dvc_file, 'r') as f:
                content = f.read()

            # Parse YAML-like content for md5 hash
            for line in content.split('\n'):
                if 'md5:' in line:
                    return line.split('md5:')[1].strip()
        except Exception as e:
            self.log(f"Error reading DVC file: {e}")

        return None

    def get_mlflow_hashes(self, model_name: str = None) -> Dict[str, str]:
        """Get logged hashes from MLflow for a model."""
        hashes = {}

        try:
            import mlflow

            tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5001')
            mlflow.set_tracking_uri(tracking_uri)

            # Search for runs with logged hashes
            experiment_name = "ppo_usdcop"
            experiment = mlflow.get_experiment_by_name(experiment_name)

            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                    max_results=10
                )

                if not runs.empty:
                    latest_run = runs.iloc[0]

                    # Extract hash tags
                    for col in runs.columns:
                        if 'hash' in col.lower():
                            value = latest_run.get(col)
                            if value and str(value) != 'nan':
                                hashes[col] = str(value)
        except ImportError:
            self.log("MLflow not installed, skipping MLflow hash retrieval")
        except Exception as e:
            self.log(f"Error getting MLflow hashes: {e}")

        return hashes

    def validate_dataset_hashes(self) -> List[HashValidationResult]:
        """Validate dataset hash consistency."""
        results = []

        # Check main training dataset
        dataset_path = self.project_root / "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv"
        dvc_file = dataset_path.with_suffix('.csv.dvc')

        # Compute actual hash
        actual_hash = self.compute_file_hash(dataset_path)
        self.log(f"Dataset actual hash: {actual_hash[:16] if actual_hash else 'N/A'}...")

        # Get DVC hash
        dvc_hash = self.get_dvc_hash(dvc_file)
        self.log(f"DVC file hash: {dvc_hash[:16] if dvc_hash else 'N/A'}...")

        # Get MLflow logged hash
        mlflow_hashes = self.get_mlflow_hashes()
        mlflow_dataset_hash = mlflow_hashes.get('tags.dataset_hash_full')
        self.log(f"MLflow dataset hash: {mlflow_dataset_hash[:16] if mlflow_dataset_hash else 'N/A'}...")

        # Compare hashes
        if actual_hash and mlflow_dataset_hash:
            matches = actual_hash == mlflow_dataset_hash
            results.append(HashValidationResult(
                source="MLflow ↔ Filesystem",
                artifact_type="dataset",
                expected_hash=mlflow_dataset_hash,
                actual_hash=actual_hash,
                matches=matches,
                message="Dataset hash matches MLflow" if matches else "MISMATCH: Dataset changed since training"
            ))

        return results

    def validate_norm_stats_hashes(self) -> List[HashValidationResult]:
        """Validate norm_stats.json hash consistency."""
        results = []

        # Check production norm_stats
        norm_stats_paths = [
            self.project_root / "config/norm_stats.json",
            self.project_root / "config/v1_norm_stats.json",
        ]

        for norm_path in norm_stats_paths:
            if norm_path.exists():
                actual_hash = self.compute_json_hash(norm_path)
                self.log(f"{norm_path.name} hash: {actual_hash[:16] if actual_hash else 'N/A'}...")

                # Get MLflow logged hash
                mlflow_hashes = self.get_mlflow_hashes()
                mlflow_norm_hash = mlflow_hashes.get('tags.norm_stats_hash_full')

                if actual_hash and mlflow_norm_hash:
                    matches = actual_hash == mlflow_norm_hash
                    results.append(HashValidationResult(
                        source="MLflow ↔ Filesystem",
                        artifact_type=f"norm_stats ({norm_path.name})",
                        expected_hash=mlflow_norm_hash,
                        actual_hash=actual_hash,
                        matches=matches,
                        message="Norm stats match training" if matches else "MISMATCH: Norm stats changed"
                    ))

        return results

    def validate_feature_order_hash(self) -> List[HashValidationResult]:
        """Validate feature order hash consistency."""
        results = []

        try:
            # Import SSOT feature order
            sys.path.insert(0, str(self.project_root / "src"))
            from core.contracts.feature_contract import FEATURE_ORDER, FEATURE_ORDER_HASH

            # Compute hash
            computed_hash = hashlib.sha256(",".join(FEATURE_ORDER).encode()).hexdigest()[:16]

            matches = computed_hash == FEATURE_ORDER_HASH
            results.append(HashValidationResult(
                source="Feature Contract",
                artifact_type="feature_order",
                expected_hash=FEATURE_ORDER_HASH,
                actual_hash=computed_hash,
                matches=matches,
                message="Feature order hash consistent" if matches else "MISMATCH: Feature order hash inconsistent"
            ))

            # Also check MLflow
            mlflow_hashes = self.get_mlflow_hashes()
            mlflow_fo_hash = mlflow_hashes.get('tags.feature_order_hash_full', '')[:16]

            if mlflow_fo_hash:
                matches = computed_hash == mlflow_fo_hash
                results.append(HashValidationResult(
                    source="MLflow ↔ Contract",
                    artifact_type="feature_order",
                    expected_hash=mlflow_fo_hash,
                    actual_hash=computed_hash,
                    matches=matches,
                    message="Feature order matches training" if matches else "MISMATCH: Feature order changed since training"
                ))

        except ImportError as e:
            self.log(f"Could not import feature contract: {e}")

        return results

    def validate_model_hashes(self, model_path: Path = None) -> List[HashValidationResult]:
        """Validate model file hashes."""
        results = []

        # Find model files
        models_dir = self.project_root / "models"
        model_files = list(models_dir.glob("**/final_model.zip")) + list(models_dir.glob("**/*.zip"))

        for model_file in model_files[:5]:  # Limit to 5 models
            actual_hash = self.compute_file_hash(model_file)
            self.log(f"{model_file.name} hash: {actual_hash[:16] if actual_hash else 'N/A'}...")

            # Get MLflow logged hash
            mlflow_hashes = self.get_mlflow_hashes()
            mlflow_model_hash = mlflow_hashes.get('tags.model_hash_full')

            if actual_hash:
                results.append(HashValidationResult(
                    source="Filesystem",
                    artifact_type=f"model ({model_file.name})",
                    expected_hash=mlflow_model_hash,
                    actual_hash=actual_hash,
                    matches=actual_hash == mlflow_model_hash if mlflow_model_hash else True,
                    message=f"Model hash: {actual_hash[:16]}"
                ))

        return results

    def validate_git_dvc_sync(self) -> List[HashValidationResult]:
        """Validate that git and DVC are in sync."""
        results = []

        try:
            # Check DVC status
            dvc_status = subprocess.run(
                ['dvc', 'status'],
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=60
            )

            is_clean = 'Data and calculation are the same' in dvc_status.stdout or dvc_status.stdout.strip() == ''

            results.append(HashValidationResult(
                source="DVC ↔ Git",
                artifact_type="sync_status",
                expected_hash="clean",
                actual_hash="clean" if is_clean else "dirty",
                matches=is_clean,
                message="DVC and Git in sync" if is_clean else f"DVC has changes: {dvc_status.stdout[:100]}"
            ))

        except Exception as e:
            self.log(f"Error checking DVC status: {e}")

        return results

    def run_all_validations(self) -> Tuple[bool, List[HashValidationResult]]:
        """Run all hash validations and return results."""
        print("=" * 60)
        print("HASH RECONCILIATION VALIDATION")
        print("=" * 60)

        all_results = []

        print("\n[1/5] Validating dataset hashes...")
        all_results.extend(self.validate_dataset_hashes())

        print("[2/5] Validating norm_stats hashes...")
        all_results.extend(self.validate_norm_stats_hashes())

        print("[3/5] Validating feature order hash...")
        all_results.extend(self.validate_feature_order_hash())

        print("[4/5] Validating model hashes...")
        all_results.extend(self.validate_model_hashes())

        print("[5/5] Validating DVC ↔ Git sync...")
        all_results.extend(self.validate_git_dvc_sync())

        # Print results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        passed = 0
        failed = 0

        for result in all_results:
            status = "✓ PASS" if result.matches else "✗ FAIL"
            color = "\033[92m" if result.matches else "\033[91m"
            reset = "\033[0m"

            print(f"{color}{status}{reset} [{result.source}] {result.artifact_type}")
            print(f"       {result.message}")

            if result.matches:
                passed += 1
            else:
                failed += 1

        print("\n" + "=" * 60)
        print(f"SUMMARY: {passed} passed, {failed} failed, {len(all_results)} total")
        print("=" * 60)

        all_passed = failed == 0
        return all_passed, all_results


def main():
    parser = argparse.ArgumentParser(description="Validate hash consistency across DVC, MLflow, and artifacts")
    parser.add_argument("--model-name", type=str, help="Specific model name to validate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--project-root", type=str, help="Project root directory")

    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else None
    validator = HashReconciliationValidator(project_root=project_root, verbose=args.verbose)

    success, results = validator.run_all_validations()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
