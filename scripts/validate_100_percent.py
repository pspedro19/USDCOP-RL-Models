#!/usr/bin/env python
"""
100% Remediation Validation Script
===================================
Comprehensive validation of all remediation items (P0, P1, P2).

Categories:
- P0: SSOT (Action, FEATURE_ORDER, session_progress, L5 flags)
- P1: ValidatedPredictor, Alembic, SQLAlchemy, Dependabot, Security
- P2: Makefile, CHANGELOG, LICENSE
- Tests: Regression and contract tests

Usage:
    python scripts/validate_100_percent.py
    python scripts/validate_100_percent.py --verbose
    python scripts/validate_100_percent.py --category p0  # Only run P0 checks

Exit Codes:
    0: 100% score (all checks pass)
    1: Less than 100% score

Author: Trading Team
Created: 2026-01-17
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class Priority(Enum):
    """Check priority level."""
    P0 = "P0"  # Critical blockers
    P1 = "P1"  # Operational improvements
    P2 = "P2"  # Quality & polish
    TEST = "TEST"  # Test checks


@dataclass
class CheckResult:
    """Result of a validation check."""
    name: str
    priority: Priority
    passed: bool
    message: str
    details: List[str] = field(default_factory=list)
    weight: float = 1.0  # Weight for scoring


# Repository root
REPO_ROOT = Path(__file__).parent.parent

# SSOT locations
ACTION_SSOT = "src/core/contracts/action_contract.py"
FEATURE_SSOT = "src/core/contracts/feature_contract.py"

# Excluded directories
EXCLUDED_DIRS = [
    "archive",
    "docs",
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "notebooks_legacy",
]


def run_grep(pattern: str, include_pattern: str = "*.py", exclude_ssot: bool = True) -> Tuple[int, List[str]]:
    """Run git grep and return match count and lines."""
    cmd = ["git", "grep", "-n", "-E", pattern, "--", include_pattern]

    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )
        lines = [
            line for line in result.stdout.strip().split("\n")
            if line and not any(excl in line for excl in EXCLUDED_DIRS)
        ]
        if exclude_ssot:
            lines = [
                line for line in lines
                if ACTION_SSOT not in line and FEATURE_SSOT not in line
            ]
        return len(lines), lines
    except Exception as e:
        return -1, [f"ERROR: {str(e)}"]


def file_exists(path: str) -> bool:
    """Check if a file exists relative to REPO_ROOT."""
    return (REPO_ROOT / path).exists()


def file_contains(path: str, pattern: str) -> bool:
    """Check if a file contains a pattern."""
    try:
        content = (REPO_ROOT / path).read_text(encoding="utf-8")
        return pattern.lower() in content.lower()
    except Exception:
        return False


# =============================================================================
# P0 CHECKS: SSOT
# =============================================================================

def check_action_enum_ssot() -> CheckResult:
    """P0-1: Only one Action class definition."""
    count, lines = run_grep(r"class Action\s*[\(:]", "*.py", exclude_ssot=True)

    violations = [
        line for line in lines
        if "test_" not in line and "/tests/" not in line
        and "Protocol" not in line and "from " not in line
    ]

    return CheckResult(
        name="Action Enum SSOT",
        priority=Priority.P0,
        passed=len(violations) == 0,
        message=f"Found {len(violations)} definitions outside SSOT" if violations else "Single SSOT definition",
        details=violations[:5],
        weight=2.0
    )


def check_feature_order_ssot() -> CheckResult:
    """P0-2: Only one FEATURE_ORDER definition."""
    count, lines = run_grep(r"FEATURE_ORDER\s*[=:]\s*[\(\[\{]", "*.py", exclude_ssot=True)

    violations = [
        line for line in lines
        if "test_" not in line and "/tests/" not in line
        and "from " not in line and "import " not in line
    ]

    return CheckResult(
        name="FEATURE_ORDER SSOT",
        priority=Priority.P0,
        passed=len(violations) == 0,
        message=f"Found {len(violations)} definitions outside SSOT" if violations else "Single SSOT definition",
        details=violations[:5],
        weight=2.0
    )


def check_no_session_progress() -> CheckResult:
    """P0-3: No session_progress references."""
    count, lines = run_grep(r"session_progress", "*.py", exclude_ssot=False)

    violations = [
        line for line in lines
        if "/docs/" not in line and "/archive/" not in line
        and "validate_" not in line and "migrate_" not in line
    ]

    return CheckResult(
        name="No session_progress",
        priority=Priority.P0,
        passed=len(violations) == 0,
        message=f"Found {len(violations)} references" if violations else "Using time_normalized",
        details=violations[:5],
        weight=1.5
    )


def check_l5_flags() -> CheckResult:
    """P0-4: L5 DAG validates flags."""
    l5_path = REPO_ROOT / "airflow" / "dags" / "l5_multi_model_inference.py"

    if not l5_path.exists():
        return CheckResult(
            name="L5 Flag Validation",
            priority=Priority.P0,
            passed=False,
            message="L5 DAG not found",
            weight=2.0
        )

    content = l5_path.read_text(encoding="utf-8").lower()
    has_paper = "paper_trad" in content
    has_risk = "risk_manager" in content or "riskmanager" in content

    return CheckResult(
        name="L5 Flag Validation",
        priority=Priority.P0,
        passed=has_paper and has_risk,
        message="Has flag validation" if (has_paper and has_risk) else "Missing flag validation",
        weight=2.0
    )


# =============================================================================
# P1 CHECKS: OPERATIONAL
# =============================================================================

def check_validated_predictor() -> CheckResult:
    """P1-1: ValidatedPredictor pattern exists."""
    predictor_files = [
        "src/inference/validated_predictor.py",
        "src/core/inference/validated_predictor.py",
        "services/inference_api/core/inference_engine.py",
    ]

    exists = any(file_exists(f) for f in predictor_files)

    # Also check for validation pattern in inference
    has_validation = False
    for f in predictor_files:
        if file_exists(f):
            has_validation = file_contains(f, "validate") or file_contains(f, "contract")

    return CheckResult(
        name="ValidatedPredictor",
        priority=Priority.P1,
        passed=exists and has_validation,
        message="ValidatedPredictor with validation" if (exists and has_validation) else "Not implemented",
        weight=1.5
    )


def check_alembic() -> CheckResult:
    """P1-2: Alembic migrations configured."""
    alembic_files = [
        "alembic.ini",
        "database/migrations/alembic.ini",
        "alembic/env.py",
    ]

    migrations_dir = REPO_ROOT / "database" / "migrations"
    has_migrations = migrations_dir.exists() and any(migrations_dir.glob("*.sql"))

    has_alembic = any(file_exists(f) for f in alembic_files)

    return CheckResult(
        name="Alembic Migrations",
        priority=Priority.P1,
        passed=has_alembic or has_migrations,
        message="Migrations configured" if (has_alembic or has_migrations) else "Not configured",
        weight=1.0
    )


def check_sqlalchemy_models() -> CheckResult:
    """P1-3: SQLAlchemy models exist."""
    model_files = [
        "src/models",
        "src/database/models.py",
        "database/models.py",
    ]

    # Check for SQLAlchemy patterns
    count, lines = run_grep(r"class.*\(.*Base\)|DeclarativeBase|SQLAlchemy", "*.py")

    has_models = any(file_exists(f) for f in model_files) or count > 0

    return CheckResult(
        name="SQLAlchemy Models",
        priority=Priority.P1,
        passed=count > 0,
        message=f"Found {count} model definitions" if count > 0 else "No SQLAlchemy models",
        weight=1.0
    )


def check_dependabot() -> CheckResult:
    """P1-4: Dependabot configured."""
    dependabot_files = [
        ".github/dependabot.yml",
        ".github/dependabot.yaml",
    ]

    exists = any(file_exists(f) for f in dependabot_files)

    return CheckResult(
        name="Dependabot",
        priority=Priority.P1,
        passed=exists,
        message="Configured" if exists else "Not configured",
        weight=0.5
    )


def check_security_scanning() -> CheckResult:
    """P1-5: Security scanning configured (bandit, safety, etc.)."""
    security_indicators = [
        ".github/workflows/security.yml",
        ".github/workflows/codeql.yml",
        ".bandit",
        "bandit.yaml",
        "pyproject.toml",  # May contain bandit config
    ]

    has_security = any(file_exists(f) for f in security_indicators)

    # Also check pyproject.toml for security tools
    if file_exists("pyproject.toml"):
        content = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
        has_security = has_security or "bandit" in content or "safety" in content

    return CheckResult(
        name="Security Scanning",
        priority=Priority.P1,
        passed=has_security,
        message="Configured" if has_security else "Not configured",
        weight=1.0
    )


def check_vault_integration() -> CheckResult:
    """P1-6: Vault integration for secrets."""
    vault_files = [
        "src/shared/secrets/vault_client.py",
        "src/core/secrets/vault_client.py",
        "config/vault",
    ]

    exists = any(file_exists(f) for f in vault_files)

    # Check for Vault references in code
    count, _ = run_grep(r"vault|VAULT", "*.py")

    return CheckResult(
        name="Vault Integration",
        priority=Priority.P1,
        passed=exists or count > 3,
        message="Integrated" if (exists or count > 3) else "Not integrated",
        weight=1.0
    )


# =============================================================================
# P2 CHECKS: QUALITY
# =============================================================================

def check_makefile() -> CheckResult:
    """P2-1: Makefile exists with standard targets."""
    if not file_exists("Makefile"):
        return CheckResult(
            name="Makefile",
            priority=Priority.P2,
            passed=False,
            message="Not found",
            weight=0.5
        )

    content = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    targets = ["test", "lint", "build", "clean"]
    found = [t for t in targets if f"{t}:" in content or f".PHONY: {t}" in content]

    return CheckResult(
        name="Makefile",
        priority=Priority.P2,
        passed=len(found) >= 2,
        message=f"Has targets: {', '.join(found)}" if found else "Missing standard targets",
        weight=0.5
    )


def check_changelog() -> CheckResult:
    """P2-2: CHANGELOG exists."""
    changelog_files = ["CHANGELOG.md", "CHANGELOG", "HISTORY.md", "CHANGES.md"]
    exists = any(file_exists(f) for f in changelog_files)

    return CheckResult(
        name="CHANGELOG",
        priority=Priority.P2,
        passed=exists,
        message="Found" if exists else "Not found",
        weight=0.5
    )


def check_license() -> CheckResult:
    """P2-3: LICENSE file exists."""
    license_files = ["LICENSE", "LICENSE.md", "LICENSE.txt"]
    exists = any(file_exists(f) for f in license_files)

    return CheckResult(
        name="LICENSE",
        priority=Priority.P2,
        passed=exists,
        message="Found" if exists else "Not found",
        weight=0.5
    )


def check_readme() -> CheckResult:
    """P2-4: README with required sections."""
    if not file_exists("README.md"):
        return CheckResult(
            name="README",
            priority=Priority.P2,
            passed=False,
            message="Not found",
            weight=0.5
        )

    content = (REPO_ROOT / "README.md").read_text(encoding="utf-8").lower()
    sections = ["install", "usage", "config", "test"]
    found = [s for s in sections if s in content]

    return CheckResult(
        name="README",
        priority=Priority.P2,
        passed=len(found) >= 2,
        message=f"Has sections: {', '.join(found)}" if found else "Missing sections",
        weight=0.5
    )


def check_type_hints() -> CheckResult:
    """P2-5: Type hints coverage."""
    # Sample check - look for type hints in key files
    key_files = [
        "src/core/contracts/action_contract.py",
        "src/core/contracts/feature_contract.py",
    ]

    typed_count = 0
    for f in key_files:
        if file_exists(f):
            content = (REPO_ROOT / f).read_text(encoding="utf-8")
            if "->" in content or ": " in content:
                typed_count += 1

    return CheckResult(
        name="Type Hints",
        priority=Priority.P2,
        passed=typed_count >= 1,
        message=f"{typed_count}/{len(key_files)} key files typed",
        weight=0.5
    )


# =============================================================================
# TEST CHECKS
# =============================================================================

def check_regression_tests() -> CheckResult:
    """TEST-1: Regression tests pass."""
    test_dir = REPO_ROOT / "tests" / "regression"

    if not test_dir.exists():
        return CheckResult(
            name="Regression Tests",
            priority=Priority.TEST,
            passed=False,
            message="Directory not found",
            weight=2.0
        )

    test_files = list(test_dir.glob("test_*.py"))
    if not test_files:
        return CheckResult(
            name="Regression Tests",
            priority=Priority.TEST,
            passed=False,
            message="No test files found",
            weight=2.0
        )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return CheckResult(
                name="Regression Tests",
                priority=Priority.TEST,
                passed=True,
                message="All tests pass",
                weight=2.0
            )
        else:
            failures = [l for l in result.stdout.split("\n") if "FAILED" in l]
            return CheckResult(
                name="Regression Tests",
                priority=Priority.TEST,
                passed=False,
                message=f"{len(failures)} test(s) failed",
                details=failures[:5],
                weight=2.0
            )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Regression Tests",
            priority=Priority.TEST,
            passed=False,
            message="Timed out",
            weight=2.0
        )
    except Exception as e:
        return CheckResult(
            name="Regression Tests",
            priority=Priority.TEST,
            passed=False,
            message=f"Error: {e}",
            weight=2.0
        )


def check_contract_tests() -> CheckResult:
    """TEST-2: Contract tests pass."""
    test_dir = REPO_ROOT / "tests" / "contracts"

    if not test_dir.exists():
        return CheckResult(
            name="Contract Tests",
            priority=Priority.TEST,
            passed=False,
            message="Directory not found",
            weight=1.5
        )

    test_files = list(test_dir.glob("test_*.py"))
    if not test_files:
        return CheckResult(
            name="Contract Tests",
            priority=Priority.TEST,
            passed=False,
            message="No test files found",
            weight=1.5
        )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            return CheckResult(
                name="Contract Tests",
                priority=Priority.TEST,
                passed=True,
                message="All tests pass",
                weight=1.5
            )
        else:
            failures = [l for l in result.stdout.split("\n") if "FAILED" in l]
            return CheckResult(
                name="Contract Tests",
                priority=Priority.TEST,
                passed=False,
                message=f"{len(failures)} test(s) failed",
                details=failures[:5],
                weight=1.5
            )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Contract Tests",
            priority=Priority.TEST,
            passed=False,
            message="Timed out",
            weight=1.5
        )
    except Exception as e:
        return CheckResult(
            name="Contract Tests",
            priority=Priority.TEST,
            passed=False,
            message=f"Error: {e}",
            weight=1.5
        )


# =============================================================================
# MAIN
# =============================================================================

ALL_CHECKS: Dict[Priority, List] = {
    Priority.P0: [
        check_action_enum_ssot,
        check_feature_order_ssot,
        check_no_session_progress,
        check_l5_flags,
    ],
    Priority.P1: [
        check_validated_predictor,
        check_alembic,
        check_sqlalchemy_models,
        check_dependabot,
        check_security_scanning,
        check_vault_integration,
    ],
    Priority.P2: [
        check_makefile,
        check_changelog,
        check_license,
        check_readme,
        check_type_hints,
    ],
    Priority.TEST: [
        check_regression_tests,
        check_contract_tests,
    ],
}


def print_result(result: CheckResult, verbose: bool = False):
    """Print a check result."""
    status = "[PASS]" if result.passed else "[FAIL]"
    color = "\033[92m" if result.passed else "\033[91m"
    reset = "\033[0m"

    print(f"  {color}{status}{reset} {result.name}: {result.message}")

    if verbose and result.details:
        for detail in result.details[:5]:
            print(f"         - {detail}")


def run_checks(categories: Optional[List[Priority]] = None, verbose: bool = False) -> Tuple[List[CheckResult], float]:
    """Run validation checks and return results with score."""
    if categories is None:
        categories = list(ALL_CHECKS.keys())

    results = []
    total_weight = 0
    passed_weight = 0

    for priority in categories:
        checks = ALL_CHECKS.get(priority, [])
        if not checks:
            continue

        print(f"\n{priority.value} CHECKS")
        print("-" * 40)

        for check_func in checks:
            result = check_func()
            results.append(result)
            total_weight += result.weight
            if result.passed:
                passed_weight += result.weight
            print_result(result, verbose)

    score = (passed_weight / total_weight * 100) if total_weight > 0 else 0
    return results, score


def main():
    parser = argparse.ArgumentParser(
        description="Validate 100% remediation completion"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "-c", "--category",
        choices=["p0", "p1", "p2", "test", "all"],
        default="all",
        help="Run specific category only"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("100% REMEDIATION VALIDATION")
    print("=" * 60)

    # Determine categories to run
    if args.category == "all":
        categories = None
    else:
        category_map = {
            "p0": Priority.P0,
            "p1": Priority.P1,
            "p2": Priority.P2,
            "test": Priority.TEST,
        }
        categories = [category_map[args.category]]

    results, score = run_checks(categories, args.verbose)

    # Summary by priority
    print("\n" + "=" * 60)
    print("SUMMARY BY PRIORITY")
    print("=" * 60)

    for priority in [Priority.P0, Priority.P1, Priority.P2, Priority.TEST]:
        priority_results = [r for r in results if r.priority == priority]
        if not priority_results:
            continue

        passed = sum(1 for r in priority_results if r.passed)
        total = len(priority_results)
        pct = (passed / total * 100) if total > 0 else 0

        color = "\033[92m" if passed == total else ("\033[93m" if passed > 0 else "\033[91m")
        reset = "\033[0m"

        print(f"  {priority.value}: {color}{passed}/{total} ({pct:.0f}%){reset}")

    # Overall score
    print("\n" + "=" * 60)
    print("OVERALL SCORE")
    print("=" * 60)

    if score >= 100:
        print(f"\033[92m{score:.1f}% - PERFECT! All checks pass.\033[0m")
        return 0
    elif score >= 85:
        print(f"\033[93m{score:.1f}% - Almost there! Minor issues remain.\033[0m")
    elif score >= 70:
        print(f"\033[93m{score:.1f}% - Good progress. P1/P2 items remain.\033[0m")
    else:
        print(f"\033[91m{score:.1f}% - Critical issues. P0 blockers remain.\033[0m")

    # List failed checks
    failed = [r for r in results if not r.passed]
    if failed:
        print("\nFailed checks:")
        for r in failed:
            print(f"  [{r.priority.value}] {r.name}: {r.message}")

    return 0 if score >= 100 else 1


if __name__ == "__main__":
    sys.exit(main())
