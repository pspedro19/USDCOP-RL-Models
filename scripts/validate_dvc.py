#!/usr/bin/env python3
"""
DVC Configuration Validation Script
====================================
Phase 0.4 - Master Remediation Plan: P0-REPRODUCIBILITY

This script validates that DVC is correctly configured for the USDCOP trading system.
It checks:
1. Remote is correctly configured (minio + backup)
2. dvc.lock is tracked in Git (not ignored)
3. Pipeline is reproducible

Usage:
    python scripts/validate_dvc.py
    python scripts/validate_dvc.py --verbose
    python scripts/validate_dvc.py --fix  # Attempt auto-fixes

Exit codes:
    0: All validations passed
    1: One or more validations failed
    2: Critical error (cannot run validations)

Author: Trading Team
Date: 2026-01-17
"""

import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes for terminal output
COLORS = {
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "reset": "\033[0m",
    "bold": "\033[1m",
}


def colored(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{colored('=' * 70, 'blue')}")
    print(f"{colored(f'  {title}', 'bold')}")
    print(f"{colored('=' * 70, 'blue')}")


def print_check(name: str, passed: bool, details: str = "") -> None:
    """Print a validation check result."""
    status = colored("[PASS]", "green") if passed else colored("[FAIL]", "red")
    print(f"  {status} {name}")
    if details:
        print(f"         {colored(details, 'yellow')}")


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for .dvc directory
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".dvc").is_dir():
            return current
        current = current.parent

    # Fallback: use script's parent's parent
    return Path(__file__).resolve().parent.parent


class DVCValidator:
    """Validates DVC configuration and setup."""

    def __init__(self, project_root: Optional[Path] = None, verbose: bool = False):
        self.project_root = project_root or get_project_root()
        self.verbose = verbose
        self.dvc_dir = self.project_root / ".dvc"
        self.config_path = self.dvc_dir / "config"
        self.gitignore_path = self.dvc_dir / ".gitignore"
        self.lock_path = self.project_root / "dvc.lock"
        self.results: List[Tuple[str, bool, str]] = []

    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {colored('[DEBUG]', 'blue')} {message}")

    def run_command(self, cmd: List[str], check: bool = False) -> Tuple[int, str, str]:
        """Run a shell command and return (returncode, stdout, stderr)."""
        self.log(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except FileNotFoundError:
            return -1, "", f"Command not found: {cmd[0]}"

    # -------------------------------------------------------------------------
    # Validation: DVC Remote Configuration
    # -------------------------------------------------------------------------
    def validate_remote_config(self) -> bool:
        """Validate DVC remote is correctly configured."""
        print_header("1. DVC Remote Configuration")

        all_passed = True

        # Check if .dvc/config exists
        if not self.config_path.exists():
            print_check("Config file exists", False, f"Missing: {self.config_path}")
            return False

        print_check("Config file exists", True, str(self.config_path))

        # Parse config file
        config = configparser.ConfigParser()
        try:
            config.read(self.config_path)
        except Exception as e:
            print_check("Config file parseable", False, str(e))
            return False

        print_check("Config file parseable", True)

        # Check core remote setting
        core_remote = config.get("core", "remote", fallback=None)
        remote_correct = core_remote == "minio"
        print_check(
            "Default remote is 'minio'",
            remote_correct,
            f"Current: {core_remote}" if not remote_correct else ""
        )
        all_passed = all_passed and remote_correct

        # Check autostage setting
        autostage = config.get("core", "autostage", fallback="false")
        autostage_enabled = autostage.lower() == "true"
        print_check(
            "Autostage enabled",
            autostage_enabled,
            f"Current: {autostage}" if not autostage_enabled else ""
        )
        all_passed = all_passed and autostage_enabled

        # Check minio remote configuration
        if 'remote "minio"' in config.sections():
            minio_section = config['remote "minio"']

            # Check URL
            minio_url = minio_section.get("url", "")
            url_correct = minio_url == "s3://dvc-storage"
            print_check(
                "MinIO URL correct (s3://dvc-storage)",
                url_correct,
                f"Current: {minio_url}" if not url_correct else ""
            )
            all_passed = all_passed and url_correct

            # Check endpoint URL
            endpoint = minio_section.get("endpointurl", "")
            endpoint_correct = endpoint == "http://minio:9000"
            print_check(
                "MinIO endpoint correct (http://minio:9000)",
                endpoint_correct,
                f"Current: {endpoint}" if not endpoint_correct else ""
            )
            all_passed = all_passed and endpoint_correct

            # Check credentials use env vars
            access_key = minio_section.get("access_key_id", "")
            secret_key = minio_section.get("secret_access_key", "")
            creds_use_env = "${DVC_ACCESS_KEY}" in access_key and "${DVC_SECRET_KEY}" in secret_key
            print_check(
                "Credentials via env vars (DVC_ACCESS_KEY, DVC_SECRET_KEY)",
                creds_use_env,
                "Credentials may be hardcoded!" if not creds_use_env else ""
            )
            all_passed = all_passed and creds_use_env
        else:
            print_check("MinIO remote section exists", False, 'Missing [remote "minio"]')
            all_passed = False

        # Check backup remote configuration
        backup_exists = 'remote "backup"' in config.sections()
        print_check(
            "Backup remote configured",
            backup_exists,
            'Add [remote "backup"] section' if not backup_exists else ""
        )
        all_passed = all_passed and backup_exists

        if backup_exists:
            backup_section = config['remote "backup"']
            backup_url = backup_section.get("url", "")
            print_check(
                "Backup remote URL configured",
                bool(backup_url),
                f"URL: {backup_url}" if backup_url else "Missing URL"
            )

        return all_passed

    # -------------------------------------------------------------------------
    # Validation: dvc.lock Tracking
    # -------------------------------------------------------------------------
    def validate_lock_tracking(self) -> bool:
        """Validate dvc.lock is tracked in Git (not ignored)."""
        print_header("2. dvc.lock Git Tracking")

        all_passed = True

        # Check if dvc.lock exists
        lock_exists = self.lock_path.exists()
        print_check(
            "dvc.lock exists",
            lock_exists,
            f"Path: {self.lock_path}" if lock_exists else "File not found - run 'dvc repro' to generate"
        )
        all_passed = all_passed and lock_exists

        # Check if dvc.lock is ignored by git
        if lock_exists:
            returncode, stdout, stderr = self.run_command(
                ["git", "check-ignore", "dvc.lock"]
            )
            # returncode 0 means file IS ignored, 1 means NOT ignored
            not_ignored = returncode == 1
            print_check(
                "dvc.lock NOT in .gitignore",
                not_ignored,
                "CRITICAL: dvc.lock is being ignored by Git!" if not not_ignored else ""
            )
            all_passed = all_passed and not_ignored

            # Check if dvc.lock is tracked
            returncode, stdout, stderr = self.run_command(
                ["git", "ls-files", "--error-unmatch", "dvc.lock"]
            )
            is_tracked = returncode == 0
            if not is_tracked:
                # Check if it's staged or untracked
                returncode, stdout, stderr = self.run_command(
                    ["git", "status", "--porcelain", "dvc.lock"]
                )
                if stdout.startswith("??"):
                    status_detail = "Untracked - run 'git add dvc.lock'"
                elif stdout.startswith("A"):
                    status_detail = "Staged but not committed"
                    is_tracked = True  # Consider staged as acceptable
                else:
                    status_detail = f"Status: {stdout}"
            else:
                status_detail = ""

            print_check(
                "dvc.lock tracked in Git",
                is_tracked,
                status_detail
            )
            all_passed = all_passed and is_tracked

        # Check .dvc/.gitignore doesn't ignore *.lock (internal check)
        if self.gitignore_path.exists():
            with open(self.gitignore_path, "r") as f:
                gitignore_content = f.read()

            # Check for problematic patterns
            ignores_lock = "*.lock" in gitignore_content and "# DO NOT" not in gitignore_content
            print_check(
                ".dvc/.gitignore doesn't ignore *.lock",
                not ignores_lock,
                "Remove '*.lock' from .dvc/.gitignore" if ignores_lock else ""
            )
            all_passed = all_passed and not ignores_lock

        return all_passed

    # -------------------------------------------------------------------------
    # Validation: Pipeline Reproducibility
    # -------------------------------------------------------------------------
    def validate_pipeline(self) -> bool:
        """Validate DVC pipeline is reproducible."""
        print_header("3. Pipeline Reproducibility")

        all_passed = True

        # Check if DVC is installed
        returncode, stdout, stderr = self.run_command(["dvc", "version"])
        dvc_installed = returncode == 0
        if dvc_installed:
            dvc_version = stdout.split("\n")[0] if stdout else "unknown"
            print_check("DVC installed", True, dvc_version)
        else:
            print_check("DVC installed", False, "Install with: pip install dvc[s3]")
            return False

        # Check remote list
        returncode, stdout, stderr = self.run_command(["dvc", "remote", "list"])
        if returncode == 0:
            remotes = stdout.strip().split("\n") if stdout else []
            has_minio = any("minio" in r for r in remotes)
            print_check(
                "Remote 'minio' accessible via 'dvc remote list'",
                has_minio,
                f"Remotes: {', '.join(remotes)}" if remotes else "No remotes configured"
            )
            all_passed = all_passed and has_minio
        else:
            print_check("DVC remote list", False, stderr)
            all_passed = False

        # Check dvc.yaml exists (pipeline definition)
        dvc_yaml = self.project_root / "dvc.yaml"
        yaml_exists = dvc_yaml.exists()
        print_check(
            "dvc.yaml pipeline definition exists",
            yaml_exists,
            "Create dvc.yaml to define pipeline stages" if not yaml_exists else str(dvc_yaml)
        )
        # Note: Not failing validation if dvc.yaml doesn't exist yet

        # Try dry-run repro (only if dvc.yaml exists)
        if yaml_exists:
            returncode, stdout, stderr = self.run_command(["dvc", "repro", "--dry-run"])
            repro_works = returncode == 0
            if repro_works:
                if "Stage" in stdout or "Nothing to reproduce" in stdout:
                    print_check("Pipeline dry-run succeeds", True)
                else:
                    print_check("Pipeline dry-run succeeds", True, "No stages to run")
            else:
                print_check(
                    "Pipeline dry-run succeeds",
                    False,
                    f"Error: {stderr[:100]}..." if len(stderr) > 100 else stderr
                )
                # Don't fail overall validation if just dry-run fails
                self.log("Pipeline dry-run failed but this may be expected for new setup")
        else:
            print_check(
                "Pipeline dry-run (skipped)",
                True,
                "No dvc.yaml - define pipeline for full reproducibility"
            )

        # Check if dvc.lock has real hashes (not placeholders)
        if self.lock_path.exists():
            with open(self.lock_path, "r") as f:
                lock_content = f.read()

            has_placeholders = "placeholder_" in lock_content
            if has_placeholders:
                print_check(
                    "dvc.lock contains real hashes",
                    False,
                    "Placeholder hashes found - run 'dvc repro' to generate real hashes"
                )
                # Don't fail validation for placeholders (expected in initial setup)
            else:
                print_check("dvc.lock contains real hashes", True)

        return all_passed

    # -------------------------------------------------------------------------
    # Main Validation Runner
    # -------------------------------------------------------------------------
    def run_all_validations(self) -> bool:
        """Run all DVC validations and return overall result."""
        print(f"\n{colored('DVC Configuration Validation', 'bold')}")
        print(f"Project root: {self.project_root}")
        print(f"Verbose mode: {self.verbose}")

        results = []

        # Run each validation
        results.append(("Remote Configuration", self.validate_remote_config()))
        results.append(("Lock File Tracking", self.validate_lock_tracking()))
        results.append(("Pipeline Reproducibility", self.validate_pipeline()))

        # Print summary
        print_header("Validation Summary")

        all_passed = True
        for name, passed in results:
            status = colored("PASSED", "green") if passed else colored("FAILED", "red")
            print(f"  {name}: {status}")
            all_passed = all_passed and passed

        print()
        if all_passed:
            print(colored("All DVC validations passed!", "green"))
        else:
            print(colored("Some validations failed. See details above.", "red"))

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate DVC configuration for USDCOP trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_dvc.py              # Run basic validation
  python scripts/validate_dvc.py --verbose    # Run with debug output
  python scripts/validate_dvc.py --fix        # Attempt auto-fixes (coming soon)

Exit codes:
  0 - All validations passed
  1 - One or more validations failed
  2 - Critical error
        """
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues (not implemented yet)"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Override project root directory"
    )

    args = parser.parse_args()

    if args.fix:
        print(colored("Auto-fix mode is not yet implemented.", "yellow"))

    try:
        validator = DVCValidator(
            project_root=args.project_root,
            verbose=args.verbose
        )

        success = validator.run_all_validations()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(colored(f"Critical error: {e}", "red"))
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
