"""G-01 — a clean clone of HEAD must import.

=============================================================================
DO NOT "FIX" THIS TEST BY DELETING IT, SKIPPING IT, OR RELAXING THE ASSERTION.
=============================================================================

This module is EXPECTED TO BE RED while another agent's lot is uncommitted.
The red is the whole point: it is the only signal that distinguishes "my code
works" from "my code works *in this working copy*".

Background (INTEGRATION-CONTRACT.md F-01): ``src/strangler/parity.py`` is
committed and imports ``src.identity.canonical``, which is owned by CODEX
(BL-17) and is **not in git**. Every test that imports the strangler therefore
passes only because the untracked directory happens to exist on this disk.
On a clean ``git clone`` those imports raise.

The correct way to make this test green is for the owner of the missing module
to COMMIT it. It is never to remove the import (the composition is correct —
`semantic_hash` has exactly one implementation and we consume it, which is the
DRY-correct arrangement) and never to wrap the import in a silent fallback (a
default value inside an identity computation is precisely the defect class this
whole audit exists to catch).

Contract: INTEGRATION-CONTRACT.md F-01 · TDD-GAPS.md G-01
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules known to be owned by another agent and pending commit. Used ONLY to
#: make the failure message actionable — an unlisted orphan fails just as hard.
KNOWN_PENDING_OWNERS = {
    "src.identity": "CODEX / BL-17 (canonical identity)",
    "src.metrics": "CODEX / BL-18 (metric engine)",
    "src.lineage": "CODEX / BL-24 (lineage graph)",
    "src.portfolio": "CODEX / BL-26/27 (portfolio + allocator)",
    "src.market": "CODEX / BL-38 (market bars)",
    "src.governance": "CODEX (governance)",
    "src.orchestration": "CODEX (FABRIC factories)",
}


def _git_tracked_python_modules() -> tuple[set[str], set[str]]:
    """Return ``(tracked_paths, importable_module_names)`` under ``src/``.

    ``git ls-files`` is the authority on purpose: the filesystem lies about what
    a clean clone contains, git does not.
    """
    result = subprocess.run(
        ["git", "ls-files", "src"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = {
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().endswith(".py")
    }
    modules: set[str] = set()
    for path in paths:
        parts = path[: -len(".py")].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        modules.add(".".join(parts))
    return paths, modules


def _walk_module_level(body: list[ast.stmt]) -> list[ast.stmt]:
    """Statements that run at import time.

    Descends into module-level ``try`` / ``if`` / ``with`` blocks — otherwise an
    import guard (``try: import X / except ImportError: raise``) would hide the
    dependency from this check, which is exactly the loophole that would let a
    tree claim self-containment it does not have. Does NOT descend into
    functions or classes: those imports are lazy by construction.
    """
    out: list[ast.stmt] = []
    for node in body:
        out.append(node)
        if isinstance(node, ast.Try):
            out.extend(_walk_module_level(node.body))
            for handler in node.handlers:
                out.extend(_walk_module_level(handler.body))
            out.extend(_walk_module_level(node.orelse))
            out.extend(_walk_module_level(node.finalbody))
        elif isinstance(node, ast.If):
            out.extend(_walk_module_level(node.body))
            out.extend(_walk_module_level(node.orelse))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            out.extend(_walk_module_level(node.body))
    return out


def _first_level_src_imports(path: Path) -> list[tuple[int, str]]:
    """Module-level ``src.*`` imports of one file (line number, module name).

    Module-level only: an import inside a function is a deliberately lazy
    dependency, while a module-level one decides whether the file can be
    imported at all — which is exactly what a clean clone exercises.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
        return []
    out: list[tuple[int, str]] = []
    for node in _walk_module_level(tree.body):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.lineno, node.module))
    return [(lineno, name) for lineno, name in out if name.startswith("src.")]


def orphan_src_imports(
    extra_tracked_modules: frozenset[str] = frozenset(),
) -> list[tuple[str, int, str]]:
    """``(file, line, module)`` for every committed ``src.*`` import that git does not have.

    ``extra_tracked_modules`` exists only so the self-test below can simulate
    "the owner committed it" without staging anyone else's files.
    """
    tracked_paths, tracked_modules = _git_tracked_python_modules()
    tracked_modules |= set(extra_tracked_modules)
    orphans: list[tuple[str, int, str]] = []
    for rel_path in sorted(tracked_paths):
        for lineno, module in _first_level_src_imports(REPO_ROOT / rel_path):
            if module in tracked_modules:
                continue
            # ``from src.pkg.mod import Symbol`` resolves to the module itself;
            # ``from src.pkg import mod`` resolves to the package.
            if module.rsplit(".", 1)[0] in tracked_modules:
                continue
            orphans.append((rel_path, lineno, module))
    return orphans


def _owner_hint(module: str) -> str:
    for prefix, owner in KNOWN_PENDING_OWNERS.items():
        if module == prefix or module.startswith(prefix + "."):
            return owner
    return "UNKNOWN OWNER — declare it in INTEGRATION-CONTRACT.md"


def test_committed_tree_has_no_untracked_imports() -> None:
    """Every ``src.*`` module imported by committed code is itself committed.

    Mutation that turns this red (it needs none today): add a module-level
    ``import src.<anything-untracked>`` to any tracked file under ``src/``.
    """
    orphans = orphan_src_imports()
    if not orphans:
        return
    lines = [
        f"  {path}:{lineno} imports {module!r} -> owner: {_owner_hint(module)}"
        for path, lineno, module in orphans
    ]
    pytest.fail(
        "A clean `git clone` of HEAD cannot import this tree: "
        f"{len(orphans)} committed import(s) point at modules that are NOT in git.\n"
        + "\n".join(lines)
        + "\n\nEXPECTED_RED_UNTIL_OWNER_COMMITS. "
        "Do not silence it by deleting the import or adding an ImportError "
        "fallback — see this module's docstring.",
        pytrace=False,
    )


def test_orphan_detector_goes_green_exactly_when_the_owner_commits() -> None:
    """The detector is not a permanent red: it clears the moment git has the module.

    This proves the assertion above is a real gate and not an unconditional
    failure — the only thing standing between it and green is a commit by the
    module's owner. Staging CODEX's files here would be a boundary violation,
    so the "committed" state is simulated instead.
    """
    pending = frozenset(module for _path, _lineno, module in orphan_src_imports())
    if not pending:
        pytest.skip("no orphan imports left — the gate above is already green")
    assert orphan_src_imports(extra_tracked_modules=pending) == []


def test_orphan_detector_catches_a_synthetic_violation(tmp_path: Path) -> None:
    """Mutation proof for the parser: this is the exact defect shape it must not miss.

    A module-level import of an untracked ``src.*`` module is reported; the same
    import inside a function is not (a lazy dependency does not stop a clean
    clone from importing the file).
    """
    sample = tmp_path / "sample.py"
    sample.write_text(
        "from src.definitely_not_tracked.mod import Thing\n", encoding="utf-8"
    )
    assert _first_level_src_imports(sample) == [(1, "src.definitely_not_tracked.mod")]

    lazy = tmp_path / "lazy.py"
    lazy.write_text(
        "def f():\n    from src.definitely_not_tracked.mod import Thing\n",
        encoding="utf-8",
    )
    assert _first_level_src_imports(lazy) == []

    # The loophole that must stay closed: an import guard is still a dependency.
    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        "try:\n"
        "    from src.definitely_not_tracked.mod import Thing\n"
        "except ImportError as exc:\n"
        "    raise ImportError('owned by someone else') from exc\n",
        encoding="utf-8",
    )
    assert _first_level_src_imports(guarded) == [(2, "src.definitely_not_tracked.mod")]
