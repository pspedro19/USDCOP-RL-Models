"""
DVC Service
===========

Service for automatic dataset versioning with DVC and MinIO.
Implements GAP 1: Auto dvc add + push with semantic tags.

Design Patterns:
- Strategy Pattern: Different storage backends (MinIO, S3, local)
- Repository Pattern: Abstract DVC operations
- Dependency Injection: Configurable remote storage

SOLID Principles:
- Single Responsibility: Only handles DVC operations
- Open/Closed: Extensible for new storage backends
- Dependency Inversion: Depends on abstractions (DVCRemote protocol)

Author: Trading Team
Date: 2026-01-17
"""

import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# SSOT import for hash utilities
from src.utils.hash_utils import compute_file_hash as _compute_file_hash_ssot
from typing import Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols (Interface Segregation)
# =============================================================================

class DVCRemote(Protocol):
    """Protocol for DVC remote storage backends."""

    def push(self, paths: List[Path]) -> bool:
        """Push files to remote storage."""
        ...

    def pull(self, paths: List[Path]) -> bool:
        """Pull files from remote storage."""
        ...

    def get_url(self, path: Path) -> str:
        """Get URL for a tracked file."""
        ...


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class DVCTag:
    """Immutable semantic tag for DVC versioning."""

    prefix: str  # e.g., "dataset", "model", "norm-stats"
    experiment_id: str
    version: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d"))

    def __str__(self) -> str:
        """Generate semantic tag string."""
        return f"{self.prefix}-{self.experiment_id}-v{self.version}-{self.timestamp}"

    @classmethod
    def for_experiment(cls, experiment_name: str, version: str = "1") -> "DVCTag":
        """Factory method for experiment dataset tags."""
        return cls(
            prefix="dataset-exp",
            experiment_id=experiment_name,
            version=version,
        )

    @classmethod
    def for_model(cls, model_id: str, version: str) -> "DVCTag":
        """Factory method for model tags."""
        return cls(
            prefix="model",
            experiment_id=model_id,
            version=version,
        )

    @classmethod
    def for_norm_stats(cls, experiment_name: str) -> "DVCTag":
        """Factory method for normalization stats tags."""
        return cls(
            prefix="norm-stats",
            experiment_id=experiment_name,
            version="1",
        )


@dataclass
class DVCResult:
    """Result of a DVC operation."""

    success: bool
    operation: str
    paths: List[str]
    tag: Optional[str] = None
    remote: Optional[str] = None
    hash: Optional[str] = None
    error: Optional[str] = None
    stdout: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "success": self.success,
            "operation": self.operation,
            "paths": self.paths,
            "tag": self.tag,
            "remote": self.remote,
            "hash": self.hash,
            "error": self.error,
        }


# =============================================================================
# DVC Service Implementation
# =============================================================================

class DVCService:
    """
    Service for automatic dataset versioning with DVC.

    Provides:
    - Automatic `dvc add` for artifacts
    - Semantic tagging with experiment context
    - Push to MinIO/S3 remote storage
    - Hash computation and verification
    - Git integration for .dvc files

    Example:
        dvc = DVCService(project_root=Path("."), remote="minio")

        # Version a dataset
        result = dvc.add_and_push(
            path=Path("data/processed/training.parquet"),
            tag=DVCTag.for_experiment("my_experiment"),
        )

        # Checkout specific version
        dvc.checkout(tag="dataset-exp-my_experiment-v1-20260117")
    """

    def __init__(
        self,
        project_root: Path,
        remote: str = "minio",
        auto_commit: bool = False,
    ):
        """
        Initialize DVC service.

        Args:
            project_root: Root directory of DVC project
            remote: Name of DVC remote (from .dvc/config)
            auto_commit: Whether to auto-commit .dvc files to git
        """
        self.project_root = Path(project_root)
        self.remote = remote
        self.auto_commit = auto_commit
        self._validate_dvc_init()

    def _validate_dvc_init(self) -> None:
        """Validate DVC is initialized in project."""
        dvc_dir = self.project_root / ".dvc"
        if not dvc_dir.exists():
            raise RuntimeError(
                f"DVC not initialized in {self.project_root}. "
                "Run 'dvc init' first."
            )

    def _run_command(
        self,
        args: List[str],
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run DVC command with proper error handling."""
        cmd = ["dvc"] + args
        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=check,
            )
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"DVC command failed: {e.stderr}")
            raise

    def compute_file_hash(self, path: Path) -> str:
        """Compute SHA256 hash of a file. SSOT: Delegates to src.utils.hash_utils"""
        return _compute_file_hash_ssot(path).full_hash

    def add(self, path: Union[Path, str]) -> DVCResult:
        """
        Add file or directory to DVC tracking.

        Args:
            path: Path to file or directory

        Returns:
            DVCResult with operation status
        """
        path = Path(path)
        if not path.exists():
            return DVCResult(
                success=False,
                operation="add",
                paths=[str(path)],
                error=f"Path does not exist: {path}",
            )

        try:
            result = self._run_command(["add", str(path)])

            # Compute hash for tracking
            file_hash = self.compute_file_hash(path) if path.is_file() else None

            return DVCResult(
                success=True,
                operation="add",
                paths=[str(path)],
                hash=file_hash[:16] if file_hash else None,
                stdout=result.stdout,
            )

        except subprocess.CalledProcessError as e:
            return DVCResult(
                success=False,
                operation="add",
                paths=[str(path)],
                error=e.stderr,
            )

    def push(self, paths: Optional[List[Path]] = None) -> DVCResult:
        """
        Push tracked files to remote storage.

        Args:
            paths: Specific paths to push (None = push all)

        Returns:
            DVCResult with operation status
        """
        try:
            args = ["push", "-r", self.remote]
            if paths:
                args.extend([str(p) for p in paths])

            result = self._run_command(args)

            return DVCResult(
                success=True,
                operation="push",
                paths=[str(p) for p in paths] if paths else ["all"],
                remote=self.remote,
                stdout=result.stdout,
            )

        except subprocess.CalledProcessError as e:
            return DVCResult(
                success=False,
                operation="push",
                paths=[str(p) for p in paths] if paths else ["all"],
                remote=self.remote,
                error=e.stderr,
            )

    def add_and_push(
        self,
        path: Union[Path, str],
        tag: Optional[DVCTag] = None,
        message: Optional[str] = None,
    ) -> DVCResult:
        """
        Add file to DVC and push to remote in one operation.

        This is the main method for experiment dataset versioning.

        Args:
            path: Path to file or directory
            tag: Optional semantic tag
            message: Optional git commit message

        Returns:
            DVCResult with complete operation status

        Example:
            result = dvc.add_and_push(
                path="data/processed/train.parquet",
                tag=DVCTag.for_experiment("baseline_ppo_v1"),
            )
            print(f"Pushed with hash: {result.hash}")
        """
        path = Path(path)

        # Step 1: Add to DVC
        add_result = self.add(path)
        if not add_result.success:
            return add_result

        # Step 2: Git add .dvc file
        dvc_file = path.with_suffix(path.suffix + ".dvc")
        if dvc_file.exists() and self.auto_commit:
            self._git_add_and_commit(dvc_file, tag, message)

        # Step 3: Push to remote
        push_result = self.push([path])
        if not push_result.success:
            return push_result

        # Step 4: Create git tag if provided
        if tag and self.auto_commit:
            self._git_tag(str(tag))

        return DVCResult(
            success=True,
            operation="add_and_push",
            paths=[str(path)],
            tag=str(tag) if tag else None,
            remote=self.remote,
            hash=add_result.hash,
        )

    def _git_add_and_commit(
        self,
        dvc_file: Path,
        tag: Optional[DVCTag],
        message: Optional[str],
    ) -> None:
        """Add .dvc file to git and commit."""
        try:
            # Add .dvc file and .gitignore changes
            subprocess.run(
                ["git", "add", str(dvc_file), ".gitignore"],
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
            )

            # Commit
            commit_msg = message or f"DVC: Track {dvc_file.stem}"
            if tag:
                commit_msg += f" (tag: {tag})"

            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
            )

        except subprocess.CalledProcessError as e:
            logger.warning(f"Git commit failed (may be no changes): {e.stderr}")

    def _git_tag(self, tag_name: str) -> None:
        """Create git tag."""
        try:
            subprocess.run(
                ["git", "tag", tag_name],
                cwd=str(self.project_root),
                check=True,
                capture_output=True,
            )
            logger.info(f"Created git tag: {tag_name}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Git tag failed: {e.stderr}")

    def checkout(self, tag: Optional[str] = None, paths: Optional[List[Path]] = None) -> DVCResult:
        """
        Checkout specific version of tracked files.

        Args:
            tag: Git tag to checkout (None = current HEAD)
            paths: Specific paths to checkout

        Returns:
            DVCResult with operation status
        """
        try:
            # If tag provided, checkout git tag first
            if tag:
                subprocess.run(
                    ["git", "checkout", tag],
                    cwd=str(self.project_root),
                    check=True,
                    capture_output=True,
                )

            # DVC checkout
            args = ["checkout"]
            if paths:
                args.extend([str(p) for p in paths])

            result = self._run_command(args)

            return DVCResult(
                success=True,
                operation="checkout",
                paths=[str(p) for p in paths] if paths else ["all"],
                tag=tag,
                stdout=result.stdout,
            )

        except subprocess.CalledProcessError as e:
            return DVCResult(
                success=False,
                operation="checkout",
                paths=[str(p) for p in paths] if paths else ["all"],
                tag=tag,
                error=str(e),
            )

    def pull(self, paths: Optional[List[Path]] = None) -> DVCResult:
        """
        Pull tracked files from remote storage.

        Args:
            paths: Specific paths to pull (None = pull all)

        Returns:
            DVCResult with operation status
        """
        try:
            args = ["pull", "-r", self.remote]
            if paths:
                args.extend([str(p) for p in paths])

            result = self._run_command(args)

            return DVCResult(
                success=True,
                operation="pull",
                paths=[str(p) for p in paths] if paths else ["all"],
                remote=self.remote,
                stdout=result.stdout,
            )

        except subprocess.CalledProcessError as e:
            return DVCResult(
                success=False,
                operation="pull",
                paths=[str(p) for p in paths] if paths else ["all"],
                remote=self.remote,
                error=e.stderr,
            )

    def get_current_tag(self) -> Optional[str]:
        """Get current DVC-related git tag if any."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def list_tags(self, prefix: str = "dataset-exp") -> List[str]:
        """List all git tags matching prefix."""
        try:
            result = subprocess.run(
                ["git", "tag", "-l", f"{prefix}*"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=True,
            )
            return [t for t in result.stdout.strip().split("\n") if t]
        except Exception:
            return []


# =============================================================================
# Factory Function
# =============================================================================

def create_dvc_service(
    project_root: Optional[Path] = None,
    remote: str = "minio",
    auto_commit: bool = False,
) -> DVCService:
    """
    Factory function to create DVCService.

    Args:
        project_root: Project root (auto-detected if None)
        remote: DVC remote name
        auto_commit: Whether to auto-commit changes

    Returns:
        Configured DVCService instance
    """
    if project_root is None:
        # Auto-detect project root
        project_root = Path(__file__).parent.parent.parent.parent

    return DVCService(
        project_root=project_root,
        remote=remote,
        auto_commit=auto_commit,
    )


__all__ = [
    "DVCService",
    "DVCTag",
    "DVCResult",
    "create_dvc_service",
]
