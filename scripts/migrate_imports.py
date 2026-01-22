#!/usr/bin/env python3
"""
Import Migration Script - Migrate Deprecated Imports to SSOT
=============================================================

This script helps migrate deprecated imports to the Single Source of Truth (SSOT).

Usage:
    python scripts/migrate_imports.py --check        # Check for deprecated imports
    python scripts/migrate_imports.py --fix         # Fix deprecated imports
    python scripts/migrate_imports.py --diff        # Show diff without modifying

Deprecated Import Patterns:
    1. from src.features.contract import FEATURE_ORDER
       -> from src.core.contracts import FEATURE_ORDER

    2. from src.feature_store.core import FEATURE_ORDER
       -> from src.core.contracts import FEATURE_ORDER

    3. from src.shared.schemas.features import FEATURE_ORDER
       -> from src.core.contracts import FEATURE_ORDER

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DeprecatedImport:
    """Represents a deprecated import pattern."""
    pattern: str
    replacement: str
    description: str


@dataclass
class ImportViolation:
    """Represents a found violation."""
    file_path: Path
    line_number: int
    original_line: str
    deprecated_import: DeprecatedImport
    suggested_fix: str


# Define deprecated import patterns
DEPRECATED_IMPORTS: List[DeprecatedImport] = [
    # FEATURE_ORDER imports
    DeprecatedImport(
        pattern=r'from\s+src\.features\.contract\s+import\s+.*FEATURE_ORDER',
        replacement='from src.core.contracts import FEATURE_ORDER',
        description='FEATURE_ORDER should be imported from src.core.contracts'
    ),
    DeprecatedImport(
        pattern=r'from\s+src\.feature_store\.core\s+import\s+.*FEATURE_ORDER',
        replacement='from src.core.contracts import FEATURE_ORDER',
        description='FEATURE_ORDER should be imported from src.core.contracts'
    ),
    DeprecatedImport(
        pattern=r'from\s+src\.shared\.schemas\.features\s+import\s+.*FEATURE_ORDER',
        replacement='from src.core.contracts import FEATURE_ORDER',
        description='FEATURE_ORDER is now re-exported, but direct import is preferred'
    ),

    # OBSERVATION_DIM imports
    DeprecatedImport(
        pattern=r'from\s+src\.features\.contract\s+import\s+.*OBSERVATION_DIM',
        replacement='from src.core.contracts import OBSERVATION_DIM',
        description='OBSERVATION_DIM should be imported from src.core.contracts'
    ),
    DeprecatedImport(
        pattern=r'from\s+src\.feature_store\.core\s+import\s+.*OBSERVATION_DIM',
        replacement='from src.core.contracts import OBSERVATION_DIM',
        description='OBSERVATION_DIM should be imported from src.core.contracts'
    ),

    # FEATURE_CONTRACT imports
    DeprecatedImport(
        pattern=r'from\s+src\.features\.contract\s+import\s+.*FEATURE_CONTRACT',
        replacement='from src.core.contracts import FEATURE_CONTRACT',
        description='FEATURE_CONTRACT should be imported from src.core.contracts'
    ),
    DeprecatedImport(
        pattern=r'from\s+src\.feature_store\.core\s+import\s+.*FEATURE_CONTRACT',
        replacement='from src.core.contracts import FEATURE_CONTRACT',
        description='FEATURE_CONTRACT should be imported from src.core.contracts'
    ),

    # Legacy FeatureBuilder imports
    DeprecatedImport(
        pattern=r'from\s+src\.features\.builder\s+import',
        replacement='from src.feature_store.builders import CanonicalFeatureBuilder',
        description='src.features.builder is deprecated, use CanonicalFeatureBuilder'
    ),
    DeprecatedImport(
        pattern=r'from\s+src\.core\.services\.feature_builder\s+import',
        replacement='from src.feature_store.builders import CanonicalFeatureBuilder',
        description='src.core.services.feature_builder is deprecated, use CanonicalFeatureBuilder'
    ),

    # Legacy calculator imports
    DeprecatedImport(
        pattern=r'from\s+src\.core\.calculators',
        replacement='from src.feature_store.calculators import RSICalculator, ATRPercentCalculator, ADXCalculator',
        description='src.core.calculators has been deleted, use src.feature_store.calculators'
    ),
    DeprecatedImport(
        pattern=r'from\s+src\.features\.calculators',
        replacement='from src.feature_store.calculators import RSICalculator, ATRPercentCalculator, ADXCalculator',
        description='src.features.calculators has been deleted, use src.feature_store.calculators'
    ),

    # Constants imports (must use SSOT)
    DeprecatedImport(
        pattern=r'CLIP_MIN\s*=\s*-5\.0',  # Hardcoded value
        replacement='from src.core.constants import CLIP_MIN',
        description='CLIP_MIN should be imported from src.core.constants, not hardcoded'
    ),
    DeprecatedImport(
        pattern=r'CLIP_MAX\s*=\s*5\.0',  # Hardcoded value
        replacement='from src.core.constants import CLIP_MAX',
        description='CLIP_MAX should be imported from src.core.constants, not hardcoded'
    ),
]

# Files/directories to exclude
EXCLUDED_PATHS = [
    '__pycache__',
    '.git',
    'node_modules',
    'venv',
    '.venv',
    'archive',
    'backups',
    '.claude',
    'scripts/migrate_imports.py',  # This script
]


def should_exclude(path: Path) -> bool:
    """Check if path should be excluded."""
    path_str = str(path)
    for excluded in EXCLUDED_PATHS:
        if excluded in path_str:
            return True
    return False


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []
    for path in root.rglob('*.py'):
        if not should_exclude(path):
            python_files.append(path)
    return sorted(python_files)


def check_file_for_violations(
    file_path: Path,
    deprecated_imports: List[DeprecatedImport]
) -> List[ImportViolation]:
    """Check a single file for deprecated imports."""
    violations = []

    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        for line_num, line in enumerate(lines, 1):
            for dep_import in deprecated_imports:
                if re.search(dep_import.pattern, line):
                    # Don't flag if it's a comment
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        continue

                    violations.append(ImportViolation(
                        file_path=file_path,
                        line_number=line_num,
                        original_line=line,
                        deprecated_import=dep_import,
                        suggested_fix=dep_import.replacement
                    ))
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")

    return violations


def format_violation(violation: ImportViolation) -> str:
    """Format a violation for display."""
    rel_path = violation.file_path.relative_to(PROJECT_ROOT)
    return (
        f"\n{rel_path}:{violation.line_number}\n"
        f"  Current:  {violation.original_line.strip()}\n"
        f"  Issue:    {violation.deprecated_import.description}\n"
        f"  Fix:      {violation.suggested_fix}"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Migrate deprecated imports to SSOT'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check for deprecated imports (default action)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Fix deprecated imports automatically'
    )
    parser.add_argument(
        '--diff',
        action='store_true',
        help='Show diff without modifying files'
    )
    parser.add_argument(
        '--path',
        type=str,
        default=None,
        help='Specific path to check (default: entire project)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Default to check mode
    if not args.check and not args.fix and not args.diff:
        args.check = True

    # Determine search path
    search_path = Path(args.path) if args.path else PROJECT_ROOT

    print(f"Scanning for deprecated imports in: {search_path}")
    print("=" * 60)

    # Find Python files
    python_files = find_python_files(search_path)
    print(f"Found {len(python_files)} Python files to scan\n")

    # Check for violations
    all_violations: List[ImportViolation] = []
    files_with_violations = set()

    for file_path in python_files:
        violations = check_file_for_violations(file_path, DEPRECATED_IMPORTS)
        if violations:
            all_violations.extend(violations)
            files_with_violations.add(file_path)

    # Report results
    if not all_violations:
        print("No deprecated imports found.")
        return 0

    print(f"Found {len(all_violations)} deprecated import(s) in {len(files_with_violations)} file(s):\n")

    for violation in all_violations:
        print(format_violation(violation))

    print("\n" + "=" * 60)
    print(f"Total: {len(all_violations)} violation(s)")

    if args.fix:
        print("\n--fix flag is set but automatic fixing is not yet implemented.")
        print("Please review the violations above and fix them manually.")
        print("\nRecommended import locations:")
        print("  - FEATURE_ORDER, OBSERVATION_DIM, FEATURE_CONTRACT:")
        print("      from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM")
        print("  - Constants (CLIP_MIN, CLIP_MAX, RSI_PERIOD, etc.):")
        print("      from src.core.constants import CLIP_MIN, CLIP_MAX")
        print("  - FeatureBuilder:")
        print("      from src.feature_store.builders import CanonicalFeatureBuilder")

    return 1 if all_violations else 0


if __name__ == '__main__':
    sys.exit(main())
