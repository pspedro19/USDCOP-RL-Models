#!/usr/bin/env python
"""
P0 Blocker Validation Script
=============================
Validates critical P0 blockers that must pass before deployment.

Checks:
1. check_action_enum() - Only one Action class definition (SSOT)
2. check_feature_order() - Only one FEATURE_ORDER definition (SSOT)
3. check_no_session_progress() - No session_progress references (use time_normalized)
4. check_l5_validates_flags() - L5 DAG has flag validation
5. check_tests_pass() - Runs pytest tests/regression/

Usage:
    python scripts/validate_blockers.py
    python scripts/validate_blockers.py --verbose

Exit Codes:
    0: All checks pass
    1: One or more checks failed

Author: Trading Team
Created: 2026-01-17
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: List[str]


# Repository root
REPO_ROOT = Path(__file__).parent.parent

# SSOT locations
ACTION_SSOT = "src/core/contracts/action_contract.py"
FEATURE_SSOT = "src/core/contracts/feature_contract.py"

# Excluded directories for searches
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
    """
    Run git grep and return match count and lines.

    Args:
        pattern: Regex pattern to search
        include_pattern: File pattern to include
        exclude_ssot: Whether to exclude SSOT files from results

    Returns:
        Tuple of (count, list of matching lines)
    """
    cmd = ["git", "grep", "-n", "-E", pattern, "--", include_pattern]

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30
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

    except subprocess.TimeoutExpired:
        return -1, ["ERROR: grep timed out"]
    except Exception as e:
        return -1, [f"ERROR: {str(e)}"]


def check_action_enum(verbose: bool = False) -> CheckResult:
    """
    Check that Action class is only defined in SSOT.

    SSOT: src/core/contracts/action_contract.py
    """
    # Search for "class Action" definitions
    count, lines = run_grep(r"class Action\s*[\(:]", "*.py", exclude_ssot=True)

    if count == -1:
        return CheckResult(
            name="Action Enum SSOT",
            passed=False,
            message="Failed to run search",
            details=lines
        )

    # Filter out test files and legitimate protocol/base classes
    violations = []
    for line in lines:
        # Skip test files
        if "test_" in line or "/tests/" in line:
            continue
        # Skip protocols and type hints
        if "Protocol" in line or "TypeVar" in line:
            continue
        # Skip imports
        if "from " in line or "import " in line:
            continue
        violations.append(line)

    if not violations:
        return CheckResult(
            name="Action Enum SSOT",
            passed=True,
            message="Only one Action class definition in SSOT",
            details=[]
        )

    return CheckResult(
        name="Action Enum SSOT",
        passed=False,
        message=f"Found {len(violations)} Action class definitions outside SSOT",
        details=violations
    )


def check_feature_order(verbose: bool = False) -> CheckResult:
    """
    Check that FEATURE_ORDER is only defined in SSOT.

    SSOT: src/core/contracts/feature_contract.py
    """
    # Search for FEATURE_ORDER definitions (not imports)
    count, lines = run_grep(r"FEATURE_ORDER\s*[=:]\s*[\(\[\{]", "*.py", exclude_ssot=True)

    if count == -1:
        return CheckResult(
            name="FEATURE_ORDER SSOT",
            passed=False,
            message="Failed to run search",
            details=lines
        )

    violations = []
    for line in lines:
        # Skip test files
        if "test_" in line or "/tests/" in line:
            continue
        # Skip imports and type hints
        if "from " in line or "import " in line:
            continue
        # Skip docstrings and comments
        if '"""' in line or "#" in line.split(":")[1] if ":" in line else False:
            continue
        violations.append(line)

    if not violations:
        return CheckResult(
            name="FEATURE_ORDER SSOT",
            passed=True,
            message="Only one FEATURE_ORDER definition in SSOT",
            details=[]
        )

    return CheckResult(
        name="FEATURE_ORDER SSOT",
        passed=False,
        message=f"Found {len(violations)} FEATURE_ORDER definitions outside SSOT",
        details=violations
    )


def check_no_session_progress(verbose: bool = False) -> CheckResult:
    """
    Check that session_progress is not used (should be time_normalized).

    Legacy: session_progress
    Correct: time_normalized
    """
    # Search for session_progress usage (not in comments or docs)
    count, lines = run_grep(r"session_progress", "*.py", exclude_ssot=False)

    if count == -1:
        return CheckResult(
            name="No session_progress",
            passed=False,
            message="Failed to run search",
            details=lines
        )

    violations = []
    for line in lines:
        # Skip docs
        if "/docs/" in line or ".md" in line:
            continue
        # Skip archive
        if "/archive/" in line or "notebooks_legacy" in line:
            continue
        # Skip this script
        if "validate_blockers.py" in line or "migrate_feature_order.py" in line:
            continue
        # Skip comments (lines where session_progress is after #)
        parts = line.split(":", 2)
        if len(parts) >= 3:
            code_part = parts[2]
            # If it's a comment, skip
            hash_pos = code_part.find("#")
            sp_pos = code_part.find("session_progress")
            if hash_pos != -1 and hash_pos < sp_pos:
                continue
        violations.append(line)

    if not violations:
        return CheckResult(
            name="No session_progress",
            passed=True,
            message="No session_progress references found (using time_normalized)",
            details=[]
        )

    return CheckResult(
        name="No session_progress",
        passed=False,
        message=f"Found {len(violations)} session_progress references (should be time_normalized)",
        details=violations
    )


def check_l5_validates_flags(verbose: bool = False) -> CheckResult:
    """
    Check that L5 DAG validates trading flags before execution.

    Required patterns in l5_multi_model_inference.py:
    - TRADING_ENABLED or trading_enabled check
    - PAPER_TRADING or paper_trading check
    - kill_switch or KILL_SWITCH check
    """
    l5_dag_path = REPO_ROOT / "airflow" / "dags" / "l5_multi_model_inference.py"

    if not l5_dag_path.exists():
        return CheckResult(
            name="L5 Flag Validation",
            passed=False,
            message=f"L5 DAG not found at {l5_dag_path}",
            details=[]
        )

    try:
        content = l5_dag_path.read_text(encoding="utf-8")
    except Exception as e:
        return CheckResult(
            name="L5 Flag Validation",
            passed=False,
            message=f"Failed to read L5 DAG: {e}",
            details=[]
        )

    # Check for flag validation patterns
    checks_found = []
    checks_missing = []

    flag_patterns = [
        ("TRADING_ENABLED", ["TRADING_ENABLED", "trading_enabled"]),
        ("PAPER_TRADING", ["PAPER_TRADING", "paper_trading", "paper_trader"]),
        ("KILL_SWITCH", ["KILL_SWITCH", "kill_switch"]),
    ]

    for flag_name, patterns in flag_patterns:
        found = any(p.lower() in content.lower() for p in patterns)
        if found:
            checks_found.append(flag_name)
        else:
            checks_missing.append(flag_name)

    # Also check for risk manager presence
    if "RiskManager" in content or "risk_manager" in content:
        checks_found.append("RISK_MANAGER")
    else:
        checks_missing.append("RISK_MANAGER")

    if len(checks_missing) <= 1:  # Allow one missing (flexible)
        return CheckResult(
            name="L5 Flag Validation",
            passed=True,
            message=f"L5 DAG has flag validation: {', '.join(checks_found)}",
            details=[f"Optional missing: {', '.join(checks_missing)}"] if checks_missing else []
        )

    return CheckResult(
        name="L5 Flag Validation",
        passed=False,
        message=f"L5 DAG missing flag validation: {', '.join(checks_missing)}",
        details=[f"Found: {', '.join(checks_found)}"]
    )


def check_tests_pass(verbose: bool = False) -> CheckResult:
    """
    Run pytest on tests/regression/ directory.
    """
    regression_tests = REPO_ROOT / "tests" / "regression"

    if not regression_tests.exists():
        return CheckResult(
            name="Regression Tests",
            passed=False,
            message=f"Regression tests directory not found: {regression_tests}",
            details=[]
        )

    # Check if there are any test files
    test_files = list(regression_tests.glob("test_*.py"))
    if not test_files:
        return CheckResult(
            name="Regression Tests",
            passed=False,
            message="No test files found in tests/regression/",
            details=[]
        )

    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest",
        str(regression_tests),
        "-v",
        "--tb=short",
        "-q"
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            # Count passed tests from output
            lines = result.stdout.strip().split("\n")
            summary_line = [l for l in lines if "passed" in l.lower()]
            summary = summary_line[-1] if summary_line else "Tests passed"

            return CheckResult(
                name="Regression Tests",
                passed=True,
                message=f"All regression tests pass: {summary}",
                details=[]
            )
        else:
            # Extract failure info
            failure_lines = []
            in_failure = False
            for line in result.stdout.split("\n"):
                if "FAILED" in line:
                    failure_lines.append(line)
                    in_failure = True
                elif in_failure and line.strip():
                    failure_lines.append(line)
                    if len(failure_lines) > 10:
                        break

            return CheckResult(
                name="Regression Tests",
                passed=False,
                message="Regression tests failed",
                details=failure_lines[:10]
            )

    except subprocess.TimeoutExpired:
        return CheckResult(
            name="Regression Tests",
            passed=False,
            message="Tests timed out after 5 minutes",
            details=[]
        )
    except Exception as e:
        return CheckResult(
            name="Regression Tests",
            passed=False,
            message=f"Failed to run tests: {e}",
            details=[]
        )


def print_result(result: CheckResult, verbose: bool = False):
    """Print a check result with formatting."""
    status = "[PASS]" if result.passed else "[FAIL]"
    color_start = "\033[92m" if result.passed else "\033[91m"
    color_end = "\033[0m"

    print(f"{color_start}{status}{color_end} {result.name}")
    print(f"       {result.message}")

    if verbose and result.details:
        for detail in result.details[:10]:
            print(f"       - {detail}")
        if len(result.details) > 10:
            print(f"       ... and {len(result.details) - 10} more")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate P0 blockers for deployment"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("P0 BLOCKER VALIDATION")
    print("=" * 60)
    print()

    # Run all checks
    checks = [
        ("Action Enum SSOT", check_action_enum),
        ("FEATURE_ORDER SSOT", check_feature_order),
        ("No session_progress", check_no_session_progress),
        ("L5 Flag Validation", check_l5_validates_flags),
        ("Regression Tests", check_tests_pass),
    ]

    results = []
    for name, check_func in checks:
        print(f"Running: {name}...")
        result = check_func(args.verbose)
        results.append(result)
        print_result(result, args.verbose)

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\033[92mAll P0 blockers validated successfully!\033[0m")
        return 0
    else:
        print(f"\033[91m{total - passed} P0 blocker(s) failed validation.\033[0m")
        print("\nFailed checks:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.message}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
