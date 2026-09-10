"""Fail-closed readers for governed trial counts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class TrialCountError(RuntimeError):
    """The governed trial count is missing or unusable."""


def read_registry_front_matter(path: str | Path) -> dict[str, Any]:
    """Read YAML front matter from ``path`` or fail closed."""
    registry_path = Path(path)
    try:
        text = registry_path.read_text(encoding="utf-8", errors="strict")
    except OSError as exc:
        raise TrialCountError(f"cannot read trial registry {registry_path}: {exc}") from exc
    if not text.startswith("---"):
        raise TrialCountError(f"trial registry {registry_path} has no YAML front matter")
    try:
        end = text.index("\n---", 3)
    except ValueError as exc:
        raise TrialCountError(
            f"trial registry {registry_path} has unterminated YAML front matter"
        ) from exc
    try:
        front_matter = yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError as exc:
        raise TrialCountError(
            f"trial registry {registry_path} has invalid YAML front matter: {exc}"
        ) from exc
    if not isinstance(front_matter, dict):
        raise TrialCountError(
            f"trial registry {registry_path} front matter must be a mapping"
        )
    return front_matter


def read_n_trials_total(path: str | Path) -> int:
    """Return the positive, non-boolean ``n_trials_total`` declared by a registry."""
    registry_path = Path(path)
    value = read_registry_front_matter(registry_path).get("n_trials_total")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TrialCountError(
            f"trial registry {registry_path} lacks a positive integer n_trials_total"
        )
    return value
