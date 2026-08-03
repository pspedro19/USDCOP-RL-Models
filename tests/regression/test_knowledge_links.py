"""The link checker and graph gate share one governed Markdown contract."""
from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.validation import check_knowledge_links as links

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_balanced_link_labels_and_extensionless_notes_are_checked(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / ".claude" / "README.md",
        "[range [0, 1]](specs/linked)\n",
    )
    _write(tmp_path / ".claude" / "specs" / "linked.md", "# Linked\n")

    broken, self_refs, checked = links.check(tmp_path)

    assert broken == []
    assert self_refs == []
    assert checked == 1


def test_root_readme_is_checked_but_coordination_runtime_is_not(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "README.md", "[missing](missing.md)\n")
    _write(
        tmp_path / ".claude" / "coordination" / "CODEX-STATUS.md",
        "[runtime missing](also-missing.md)\n",
    )

    broken, self_refs, checked = links.check(tmp_path)

    assert broken == ["README.md:1 -> missing.md"]
    assert self_refs == []
    assert checked == 1


def test_self_reference_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "README.md", "[self](README.md)\n")

    broken, self_refs, checked = links.check(tmp_path)

    assert broken == []
    assert self_refs == ["README.md:1 -> itself"]
    assert checked == 1
