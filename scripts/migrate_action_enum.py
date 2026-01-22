#!/usr/bin/env python
"""
Action Enum Migration Script
=============================
Migrates all Action class definitions and usages to SSOT.

SSOT Location: src/core/contracts/action_contract.py

This script:
1. Finds all files with local Action class definitions
2. Finds all files using old Action constants (ACTION_HOLD, etc.)
3. Shows what would change (--dry-run)
4. Applies changes (--apply)

Usage:
    python scripts/migrate_action_enum.py --dry-run   # Preview changes
    python scripts/migrate_action_enum.py --apply     # Apply changes
    python scripts/migrate_action_enum.py --dry-run -v  # Verbose preview

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
class ClassMigration:
    """Migration for removing a local Action class definition."""
    file_path: Path
    start_line: int
    end_line: int
    class_code: str
    needs_import: bool


# Repository root
REPO_ROOT = Path(__file__).parent.parent

# SSOT paths and imports
ACTION_SSOT_PATH = "src/core/contracts/action_contract.py"
ACTION_IMPORT = "from src.core.contracts import Action"
ACTION_IMPORT_FULL = "from src.core.contracts.action_contract import Action"

# Patterns for replacement
PATTERNS: List[Tuple[str, str, str]] = [
    # Import replacements
    (r"from src\.core\.constants import (.*?ACTION_HOLD.*?|.*?ACTION_BUY.*?|.*?ACTION_SELL.*?)",
     "from src.core.contracts import Action", "Replace old constants import"),
    (r"from src\.training\.environments\.\w+ import TradingAction",
     "from src.core.contracts import Action", "Replace TradingAction import"),
    # Constant replacements
    (r"\bACTION_HOLD\b", "Action.HOLD.value", "Replace ACTION_HOLD"),
    (r"\bACTION_BUY\b", "Action.BUY.value", "Replace ACTION_BUY"),
    (r"\bACTION_SELL\b", "Action.SELL.value", "Replace ACTION_SELL"),
    # TradingAction replacements
    (r"\bTradingAction\.LONG\b", "Action.BUY", "Replace TradingAction.LONG"),
    (r"\bTradingAction\.SHORT\b", "Action.SELL", "Replace TradingAction.SHORT"),
    (r"\bTradingAction\.HOLD\b", "Action.HOLD", "Replace TradingAction.HOLD"),
]

# Files and directories to skip
FILES_TO_SKIP: Set[str] = {
    "action_contract.py",
    "migrate_action_enum.py",
    "validate_blockers.py",
    "validate_100_percent.py",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "archive",
    "notebooks_legacy",
}


def should_process_file(filepath: Path) -> bool:
    """Check if a file should be processed."""
    if any(skip in str(filepath) for skip in FILES_TO_SKIP):
        return False
    return filepath.suffix == ".py"


def find_action_class_definition(content: str) -> Optional[Tuple[int, int, str]]:
    """
    Find local Action class definition in file content.

    Returns:
        Tuple of (start_line, end_line, class_code) or None
    """
    lines = content.split("\n")

    in_class = False
    class_start = -1
    class_indent = 0
    class_lines = []

    for i, line in enumerate(lines):
        # Check for class Action definition (IntEnum or Enum based)
        if re.match(r'^\s*class Action\s*\(.*(?:IntEnum|Enum)', line):
            in_class = True
            class_start = i
            class_indent = len(line) - len(line.lstrip())
            class_lines = [line]
            continue

        if in_class:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped) if stripped else float('inf')

            # Empty lines or more indented lines are part of the class
            if not stripped or current_indent > class_indent:
                class_lines.append(line)
            # New class/function/decorator at same/less indent ends the class
            elif stripped.startswith(('@', 'class ', 'def ')) or current_indent <= class_indent:
                return (class_start, i - 1, "\n".join(class_lines))

    # If still in class at end of file
    if in_class:
        return (class_start, len(lines) - 1, "\n".join(class_lines))

    return None


def already_imports_from_ssot(content: str) -> bool:
    """Check if file already imports Action from SSOT."""
    patterns = [
        r"from src\.core\.contracts import.*Action",
        r"from src\.core\.contracts\.action_contract import.*Action",
    ]
    return any(re.search(p, content) for p in patterns)


def find_class_migrations() -> List[ClassMigration]:
    """Find all files with local Action class definitions."""
    migrations = []

    # Use git grep to find files
    cmd = ["git", "grep", "-l", "-E", r"class Action\s*\(.*(?:IntEnum|Enum)", "--", "*.py"]

    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            if any(skip in line for skip in FILES_TO_SKIP):
                continue
            if ACTION_SSOT_PATH in line:
                continue
            if "test_" in line:
                continue

            filepath = REPO_ROOT / line
            try:
                content = filepath.read_text(encoding="utf-8")
            except Exception:
                continue

            class_info = find_action_class_definition(content)
            if class_info:
                start, end, code = class_info
                needs_import = not already_imports_from_ssot(content)
                migrations.append(ClassMigration(
                    file_path=filepath,
                    start_line=start + 1,
                    end_line=end + 1,
                    class_code=code,
                    needs_import=needs_import
                ))

    except Exception as e:
        print(f"Error searching for class definitions: {e}")

    return migrations


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


def apply_class_migration(migration: ClassMigration) -> bool:
    """Apply a class migration to remove local Action definition."""
    try:
        content = migration.file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Build the replacement
        import_line = ACTION_IMPORT if migration.needs_import else ""
        comment = "# Action class removed - using SSOT (src.core.contracts.action_contract)"

        # Replace the class lines
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


def print_class_migration(migration: ClassMigration, verbose: bool = False):
    """Print class migration details."""
    rel_path = migration.file_path.relative_to(REPO_ROOT)
    print(f"\n  File: {rel_path}")
    print(f"  Lines: {migration.start_line}-{migration.end_line}")
    print(f"  Needs import: {'Yes' if migration.needs_import else 'No (already imports from SSOT)'}")

    if verbose:
        print("\n  --- CLASS TO REMOVE ---")
        for line in migration.class_code.split("\n")[:10]:
            print(f"  | {line}")
        if migration.class_code.count("\n") > 10:
            print(f"  | ... ({migration.class_code.count(chr(10)) - 10} more lines)")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Action enum definitions to SSOT"
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
    print("ACTION ENUM MIGRATION TO SSOT")
    print("=" * 60)
    print(f"\nSSOT: {ACTION_SSOT_PATH}")
    print(f"Import: {ACTION_IMPORT}")

    # Find class definitions to migrate
    print("\n" + "-" * 40)
    print("CLASS DEFINITIONS TO REMOVE")
    print("-" * 40)

    class_migrations = find_class_migrations()

    if class_migrations:
        print(f"\nFound {len(class_migrations)} local Action class definition(s):")
        for migration in class_migrations:
            print_class_migration(migration, args.verbose)
    else:
        print("\nNo local Action class definitions found outside SSOT.")

    # Find pattern replacements
    print("\n" + "-" * 40)
    print("PATTERN REPLACEMENTS")
    print("-" * 40)

    all_replacements: List[Replacement] = []
    for filepath in REPO_ROOT.rglob("*.py"):
        if should_process_file(filepath):
            replacements = find_pattern_replacements(filepath)
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
                if len(reps) > 5:
                    print(f"    ... and {len(reps) - 5} more")
    else:
        print("\nNo pattern replacements needed.")

    # Apply changes
    if args.apply:
        print("\n" + "=" * 60)
        print("APPLYING MIGRATIONS")
        print("=" * 60)

        # Apply class migrations
        if class_migrations:
            print("\nRemoving local class definitions...")
            for migration in class_migrations:
                rel_path = migration.file_path.relative_to(REPO_ROOT)
                if apply_class_migration(migration):
                    print(f"  [OK] {rel_path}")
                else:
                    print(f"  [FAIL] {rel_path}")

        # Apply pattern replacements
        if by_file:
            print("\nApplying pattern replacements...")
            total_changes = 0
            for filepath in by_file:
                changes = apply_pattern_replacements(filepath)
                if changes > 0:
                    print(f"  [OK] {filepath.relative_to(REPO_ROOT)}")
                    total_changes += changes

        print(f"\nMigration complete!")
        print(f"  Class definitions removed: {len(class_migrations)}")
        print(f"  Files with pattern replacements: {len(by_file)}")

    else:
        print("\n" + "=" * 60)
        print("DRY RUN - No changes made")
        print("=" * 60)
        total = len(class_migrations) + len(by_file)
        if total > 0:
            print(f"\nRun with --apply to apply changes to {total} file(s)")
        else:
            print("\nNo migrations needed. All files are properly configured.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
