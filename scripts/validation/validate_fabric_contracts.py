"""Static FABRIC contract validator used by CI and local readiness checks."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_BYPASS_ALLOWLIST = Path("config/metrics/legacy_bypass_allowlist.yaml")
SCAN_ROOTS = (Path("src"), Path("services"), Path("scripts"), Path("airflow/dags"))
SSOT_EXEMPTIONS = {
    Path("services/common/metrics.py"),
    Path("src/metrics/engine.py"),
    Path("src/metrics/formulas.py"),
}
METRIC_NAME_MARKERS = ("sharpe", "calmar")
SSOT_MODULES = {
    "services.common.metrics",
    "src.metrics.engine",
    "src.metrics.formulas",
}


def is_runtime_python_path(relative_path: Path) -> bool:
    """Exclude explicit test directories, never production files by basename."""
    return relative_path.name != "conftest.py" and "tests" not in relative_path.parts[:-1]


class _MetricDefinition(NamedTuple):
    identifier: str
    name: str
    return_calls: tuple[frozenset[str], ...]
    ssot_aliases: frozenset[str]


def _walk_without_nested_definitions(node: ast.AST):
    """Yield descendants while keeping nested functions as separate scopes."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        yield child
        yield from _walk_without_nested_definitions(child)


class _MetricDefinitionVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: Path) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.definitions: list[_MetricDefinition] = []

    def _visit_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified_name = "::".join((*self.scope, node.name))
        if any(marker in node.name.lower() for marker in METRIC_NAME_MARKERS):
            scoped_nodes = tuple(_walk_without_nested_definitions(node))
            aliases = {
                alias.asname or alias.name
                for child in scoped_nodes
                if isinstance(child, ast.ImportFrom) and child.module in SSOT_MODULES
                for alias in child.names
            }
            return_calls = []
            for child in scoped_nodes:
                if not isinstance(child, ast.Return) or child.value is None:
                    continue
                return_calls.append(
                    frozenset(
                        call.func.id
                        for call in ast.walk(child.value)
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    )
                )
            self.definitions.append(
                _MetricDefinition(
                    identifier=f"{self.relative_path.as_posix()}::{qualified_name}",
                    name=node.name,
                    return_calls=tuple(return_calls),
                    ssot_aliases=frozenset(aliases),
                )
            )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


def discover_metric_bypasses(repo_root: Path = REPO_ROOT) -> set[str]:
    found: set[str] = set()
    for relative_root in SCAN_ROOTS:
        root = repo_root / relative_root
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            relative_path = path.relative_to(repo_root)
            if relative_path in SSOT_EXEMPTIONS or not is_runtime_python_path(relative_path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                raise ValueError(f"cannot scan {relative_path.as_posix()}: {exc}") from exc
            visitor = _MetricDefinitionVisitor(relative_path)
            visitor.visit(tree)
            delegated_names: set[str] = set()
            unresolved = list(visitor.definitions)
            while True:
                newly_delegated = {
                    definition.name
                    for definition in unresolved
                    if definition.return_calls
                    and all(
                        calls & (definition.ssot_aliases | delegated_names)
                        for calls in definition.return_calls
                    )
                }
                if not newly_delegated - delegated_names:
                    break
                delegated_names.update(newly_delegated)
            found.update(
                definition.identifier
                for definition in visitor.definitions
                if definition.name not in delegated_names
            )
    return found


def validate(
    repo_root: Path = REPO_ROOT,
    allowlist_path: Path = LEGACY_BYPASS_ALLOWLIST,
) -> list[str]:
    path = repo_root / allowlist_path
    if not path.exists():
        return [f"missing {allowlist_path.as_posix()}"]
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return [f"invalid allowlist: {exc}"]
    try:
        discovered = discover_metric_bypasses(repo_root)
    except ValueError as exc:
        return [str(exc)]
    return validate_metric_bypass_inventory(raw, discovered)


def validate_metric_bypass_inventory(
    raw: object, discovered: set[str]
) -> list[str]:
    if not isinstance(raw, dict):
        return ["allowlist document must be a mapping"]
    entries = raw.get("entries")
    ceiling = raw.get("max_entries")
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        return ["allowlist entries must be a list of identifiers"]
    if len(entries) != len(set(entries)):
        return ["allowlist entries must be unique"]
    if type(ceiling) is not int or ceiling < 0:
        return ["max_entries must be a non-negative integer"]
    errors: list[str] = []
    if len(entries) > ceiling:
        errors.append(
            f"allowlist expanded: {len(entries)} entries exceeds frozen ceiling {ceiling}"
        )
    declared = set(entries)
    for item in sorted(discovered - declared):
        errors.append(f"unallowlisted metric implementation: {item}")
    for item in sorted(declared - discovered):
        errors.append(f"stale allowlist entry must be removed: {item}")
    return errors


if __name__ == "__main__":
    violations = validate()
    if violations:
        raise SystemExit("FABRIC contract violations:\n- " + "\n- ".join(violations))
