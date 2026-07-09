#!/usr/bin/env python
"""
FEATURE_ORDER Migration Script
===============================
Migrates all FEATURE_ORDER definitions and session_progress references to SSOT.

SSOT Location: src/core/contracts/feature_contract.py

This script:
1. Finds all files with local FEATURE_ORDER definitions
2. Finds all files using session_progress (should be time_normalized)
3. Shows what would change (--dry-run)
4. Applies changes (--apply)

Usage:
    python scripts/migrate_feature_order.py --dry-run   # Preview changes
    python scripts/migrate_feature_order.py --apply     # Apply changes
    python scripts/migrate_feature_order.py --dry-run -v  # Verbose preview

Author: Trading Team
Created: 2026-01-17
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class Replacement:
    """A single replacement in a file."""
    file: Path
    line_num: int
    old_text: str
    new_text: str
    pattern_name: str


@dataclass
class DefinitionMigration:
    """Migration for removing a local FEATURE_ORDER definition."""
    file_path: Path
    start_line: int
    end_line: int
    definition_code: str
    needs_import: bool


# Repository root
REPO_ROOT = Path(__file__).parent.parent

# SSOT paths and imports
FEATURE_SSOT_PATH = "src/core/contracts/feature_contract.py"
FEATURE_IMPORT = "from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM"
FEATURE_IMPORT_FULL = "from src.core.contracts.feature_contract import FEATURE_ORDER, OBSERVATION_DIM"

# Patterns for replacement
PATTERNS: List[Tuple[str, str, str]] = [
    # session_progress -> time_normalized
    (r'\bsession_progress\b', 'time_normalized', 'Replace session_progress with time_normalized'),
    # Old import patterns
    (r"from src\.features\.contract import FEATURE_ORDER",
     "from src.core.contracts import FEATURE_ORDER", "Update FEATURE_ORDER import"),
    (r"from src\.feature_store import FEATURE_ORDER",
     "from src.core.contracts import FEATURE_ORDER", "Update FEATURE_ORDER import"),
    # OBSERVATION_DIM patterns
    (r"\bOBS_DIM\s*=\s*15\b", "OBSERVATION_DIM", "Replace OBS_DIM with OBSERVATION_DIM"),
    (r"\bOBS_DIM\s*=\s*30\b", "# Legacy OBS_DIM=30 - now OBSERVATION_DIM=15", "Mark legacy OBS_DIM"),
]

# Files and directories to skip
FILES_TO_SKIP: Set[str] = {
    "feature_contract.py",
    "migrate_feature_order.py",
    "validate_blockers.py",
    "validate_100_percent.py",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "archive",
    "notebooks_legacy",
    "docs",
}


def should_process_file(filepath: Path) -> bool:
    """Check if a file should be processed."""
    if any(skip in str(filepath) for skip in FILES_TO_SKIP):
        return False
    return filepath.suffix == ".py"


def find_feature_order_definition(content: str) -> Optional[Tuple[int, int, str]]:
    """
    Find local FEATURE_ORDER definition in file content.

    Returns:
        Tuple of (start_line, end_line, definition_code) or None
    """
    lines = content.split("\n")

    # Pattern for multi-line tuple/list definition
    in_definition = False
    def_start = -1
    def_lines = []
    paren_count = 0

    for i, line in enumerate(lines):
        # Check for FEATURE_ORDER definition start
        if re.match(r'^\s*FEATURE_ORDER\s*[=:]\s*[\(\[\{]', line):
            in_definition = True
            def_start = i
            def_lines = [line]
            # Count parentheses/brackets
            paren_count = line.count('(') + line.count('[') + line.count('{')
            paren_count -= line.count(')') + line.count(']') + line.count('}')
            if paren_count <= 0:
                # Single line definition
                return (def_start, i, "\n".join(def_lines))
            continue

        if in_definition:
            def_lines.append(line)
            paren_count += line.count('(') + line.count('[') + line.count('{')
            paren_count -= line.count(')') + line.count(']') + line.count('}')
            if paren_count <= 0:
                return (def_start, i, "\n".join(def_lines))

    return None


def already_imports_from_ssot(content: str) -> bool:
    """Check if file already imports FEATURE_ORDER from SSOT."""
    patterns = [
        r"from src\.core\.contracts import.*FEATURE_ORDER",
        r"from src\.core\.contracts\.feature_contract import.*FEATURE_ORDER",
    ]
    return any(re.search(p, content) for p in patterns)


def find_definition_migrations() -> List[DefinitionMigration]:
    """Find all files with local FEATURE_ORDER definitions."""
    migrations = []

    # Use git grep to find files
    cmd = ["git", "grep", "-l", "-E", r"FEATURE_ORDER\s*[=:]\s*[\(\[\{]", "--", "*.py"]

    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if any(skip in line for skip in FILES_TO_SKIP):
                continue
            if FEATURE_SSOT_PATH in line:
                continue
            if "test_" in line:
                continue

            filepath = REPO_ROOT / line
            try:
                content = filepath.read_text(encoding="utf-8")
            except Exception:
                continue

            def_info = find_feature_order_definition(content)
            if def_info:
                start, end, code = def_info
                needs_import = not already_imports_from_ssot(content)
                migrations.append(DefinitionMigration(
                    file_path=filepath,
                    start_line=start + 1,
                    end_line=end + 1,
                    definition_code=code,
                    needs_import=needs_import
                ))

    except Exception as e:
        print(f"Error searching for definitions: {e}")

    return migrations


def find_session_progress_files() -> List[Tuple[Path, List[int]]]:
    """Find all files with session_progress references."""
    results = []

    cmd = ["git", "grep", "-n", r"\bsession_progress\b", "--", "*.py"]

    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )

        by_file: Dict[Path, List[int]] = {}

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if any(skip in line for skip in FILES_TO_SKIP):
                continue

            parts = line.split(":", 2)
            if len(parts) >= 2:
                filepath = REPO_ROOT / parts[0]
                line_num = int(parts[1])
                by_file.setdefault(filepath, []).append(line_num)

        results = [(path, lines) for path, lines in by_file.items()]

    except Exception as e:
        print(f"Error searching for session_progress: {e}")

    return results


def find_pattern_replacements(filepath: Path) -> List[Replacement]:
    """Find all pattern replacements needed in a file."""
    replacements = []
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return []

    lines = content.split("\n")
    for line_num, line in enumerate(lines, 1):
        for pattern, replacement, desc in PATTERNS:
            if re.search(pattern, line):
                new_line = re.sub(pattern, replacement, line)
                if new_line != line:
                    replacements.append(Replacement(
                        filepath, line_num, line.strip(), new_line.strip(), desc
                    ))
    return replacements


def apply_definition_migration(migration: DefinitionMigration) -> bool:
    """Apply a migration to remove local FEATURE_ORDER definition."""
    try:
        content = migration.file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Build the replacement
        import_line = FEATURE_IMPORT if migration.needs_import else ""
        comment = "# FEATURE_ORDER removed - using SSOT (src.core.contracts.feature_contract)"

        # Replace the definition lines
        new_lines = []
        skip_until = -1

        for i, line in enumerate(lines):
            if i == migration.start_line - 1:
                if import_line:
                    new_lines.append(import_line)
                new_lines.append(comment)
                skip_until = migration.end_line - 1
            elif i <= skip_until:
                continue
            else:
                new_lines.append(line)

        migration.file_path.write_text("\n".join(new_lines), encoding="utf-8")
        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def apply_pattern_replacements(filepath: Path) -> int:
    """Apply all pattern replacements to a file."""
    try:
        content = filepath.read_text(encoding="utf-8")
        original = content

        for pattern, replacement, _ in PATTERNS:
            content = re.sub(pattern, replacement, content)

        if content != original:
            filepath.write_text(content, encoding="utf-8")
            return 1
        return 0

    except Exception:
        return 0


def print_definition_migration(migration: DefinitionMigration, verbose: bool = False):
    """Print definition migration details."""
    rel_path = migration.file_path.relative_to(REPO_ROOT)
    print(f"\n  File: {rel_path}")
    print(f"  Lines: {migration.start_line}-{migration.end_line}")
    print(f"  Needs import: {'Yes' if migration.needs_import else 'No (already imports from SSOT)'}")

    if verbose:
        print("\n  --- DEFINITION TO REMOVE ---")
        for line in migration.definition_code.split("\n")[:10]:
            print(f"  | {line}")
        if migration.definition_code.count("\n") > 10:
            print(f"  | ... ({migration.definition_code.count(chr(10)) - 10} more lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate FEATURE_ORDER definitions to SSOT"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying"
    )
    group.add_argument(
        "--apply",
        action="store_true",
        help="Apply the migrations"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FEATURE_ORDER MIGRATION TO SSOT")
    print("=" * 60)
    print(f"\nSSOT: {FEATURE_SSOT_PATH}")
    print(f"Import: {FEATURE_IMPORT}")

    # Find FEATURE_ORDER definitions to migrate
    print("\n" + "-" * 40)
    print("FEATURE_ORDER DEFINITIONS TO REMOVE")
    print("-" * 40)

    def_migrations = find_definition_migrations()

    if def_migrations:
        print(f"\nFound {len(def_migrations)} local FEATURE_ORDER definition(s):")
        for migration in def_migrations:
            print_definition_migration(migration, args.verbose)
    else:
        print("\nNo local FEATURE_ORDER definitions found outside SSOT.")

    # Find session_progress references
    print("\n" + "-" * 40)
    print("SESSION_PROGRESS REFERENCES")
    print("-" * 40)

    sp_files = find_session_progress_files()

    if sp_files:
        total_refs = sum(len(lines) for _, lines in sp_files)
        print(f"\nFound {total_refs} session_progress reference(s) in {len(sp_files)} file(s):")
        for filepath, line_nums in sp_files:
            rel_path = filepath.relative_to(REPO_ROOT)
            print(f"  {rel_path}: lines {line_nums}")
    else:
        print("\nNo session_progress references found.")

    # Find other pattern replacements
    print("\n" + "-" * 40)
    print("OTHER PATTERN REPLACEMENTS")
    print("-" * 40)

    all_replacements: List[Replacement] = []
    for filepath in REPO_ROOT.rglob("*.py"):
        if should_process_file(filepath):
            replacements = find_pattern_replacements(filepath)
            # Filter out session_progress (already counted)
            replacements = [r for r in replacements if "session_progress" not in r.pattern_name]
            all_replacements.extend(replacements)

    by_file: Dict[Path, List[Replacement]] = {}
    for r in all_replacements:
        by_file.setdefault(r.file, []).append(r)

    if by_file:
        print(f"\nFound {len(all_replacements)} replacement(s) in {len(by_file)} file(s):")
        for filepath, reps in sorted(by_file.items()):
            relative = filepath.relative_to(REPO_ROOT)
            print(f"  {relative}: {len(reps)} replacements")
            if args.verbose:
                for r in reps[:5]:
                    print(f"    L{r.line_num}: {r.pattern_name}")
    else:
        print("\nNo other pattern replacements needed.")

    # Apply changes
    if args.apply:
        print("\n" + "=" * 60)
        print("APPLYING MIGRATIONS")
        print("=" * 60)

        # Apply FEATURE_ORDER definition migrations
        if def_migrations:
            print("\nRemoving local FEATURE_ORDER definitions...")
            for migration in def_migrations:
                rel_path = migration.file_path.relative_to(REPO_ROOT)
                if apply_definition_migration(migration):
                    print(f"  [OK] {rel_path}")
                else:
                    print(f"  [FAIL] {rel_path}")

        # Apply session_progress -> time_normalized replacements
        if sp_files:
            print("\nReplacing session_progress with time_normalized...")
            for filepath, _ in sp_files:
                try:
                    content = filepath.read_text(encoding="utf-8")
                    new_content = re.sub(r'\bsession_progress\b', 'time_normalized', content)
                    if new_content != content:
                        filepath.write_text(new_content, encoding="utf-8")
                        print(f"  [OK] {filepath.relative_to(REPO_ROOT)}")
                except Exception as e:
                    print(f"  [FAIL] {filepath.relative_to(REPO_ROOT)}: {e}")

        # Apply other pattern replacements
        if by_file:
            print("\nApplying other pattern replacements...")
            for filepath in by_file:
                changes = apply_pattern_replacements(filepath)
                if changes > 0:
                    print(f"  [OK] {filepath.relative_to(REPO_ROOT)}")

        print(f"\nMigration complete!")
        print(f"  FEATURE_ORDER definitions removed: {len(def_migrations)}")
        print(f"  Files with session_progress replaced: {len(sp_files)}")
        print(f"  Files with other pattern replacements: {len(by_file)}")

    else:
        print("\n" + "=" * 60)
        print("DRY RUN - No changes made")
        print("=" * 60)
        total = len(def_migrations) + len(sp_files) + len(by_file)
        if total > 0:
            print(f"\nRun with --apply to apply changes to {total} file(s)")
        else:
            print("\nNo migrations needed. All files are properly configured.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
