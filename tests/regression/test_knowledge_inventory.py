"""Knowledge-system inventory is generated from source, never hand-maintained.

Contract: CTR-KNOWLEDGE-INVENTORY-001

The 2026-07-20 audit found every architectural count in the docs was wrong and that
the docs disagreed with each other (DAGs quoted as 38 / 29 / 40 while 45 existed;
API routes as 49 while 93 existed). These tests make that class of drift a test
failure instead of a discovery.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts" / "diagnostics" / "generate_inventory.py"
INVENTORY = ROOT / ".claude" / "generated" / "inventory.json"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_inventory", GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def inventory() -> dict:
    assert INVENTORY.is_file(), (
        "missing .claude/generated/inventory.json — run "
        "`python scripts/diagnostics/generate_inventory.py --write`"
    )
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_generator_exists_and_is_runnable():
    assert GENERATOR.is_file(), "the inventory generator is load-bearing for the specs gate"


def test_inventory_is_not_stale():
    """`--check` must pass: docs + json agree with the code as it exists right now."""
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "inventory is stale — run `python scripts/diagnostics/generate_inventory.py --write`\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_generator_is_deterministic():
    """Two builds must agree, or `--check` would flap in CI."""
    mod = _load_generator()
    assert json.dumps(mod.build(), sort_keys=True) == json.dumps(mod.build(), sort_keys=True)


def test_knowledge_inventory_counts_definitions_not_local_folders(tmp_path):
    mod = _load_generator()
    skill = tmp_path / ".claude" / "skills" / "real-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: real-skill\ndescription: A real executable skill.\n---\n",
        encoding="utf-8",
    )
    local_config = tmp_path / ".claude" / "skills" / ".claude" / "settings.local.json"
    local_config.parent.mkdir(parents=True)
    local_config.write_text("{}", encoding="utf-8")
    live_spec = tmp_path / ".claude" / "specs" / "live.md"
    live_spec.parent.mkdir(parents=True)
    live_spec.write_text("# Live\n", encoding="utf-8")
    (live_spec.parent / "README.md").write_text("# Index\n", encoding="utf-8")
    archived = tmp_path / ".claude" / "specs" / "archive" / "old.md"
    archived.parent.mkdir(parents=True)
    archived.write_text("# Old\n", encoding="utf-8")

    knowledge = mod.collect_knowledge(tmp_path)

    assert knowledge["skills"] == 1
    assert knowledge["skill_names"] == ["real-skill"]
    assert knowledge["specs"] == 1


def test_dag_registry_matches_disk(inventory):
    """The registry is the executable SSOT; the AST inventory is the observed truth.

    Every DAG on disk must be registered (or be explicitly deprecated), and the
    registry must not advertise DAGs that no module emits.
    """
    sys.path.insert(0, str(ROOT / "airflow" / "dags"))
    try:
        from contracts import dag_registry as reg
    except ImportError:  # pragma: no cover - airflow-less env still checks the rest
        pytest.skip("dag_registry not importable in this environment")

    on_disk = set(inventory["dags"]["ids"])
    active = set(reg.get_active_dag_ids())
    deprecated = set(reg.DEPRECATED_DAGS)
    absent = set(reg.ABSENT_DAGS)

    unregistered = on_disk - active - deprecated
    assert not unregistered, (
        f"DAGs exist on disk but are not in the registry: {sorted(unregistered)}. "
        "Add a constant and include it in get_all_dag_ids()."
    )

    phantom = active - on_disk
    assert not phantom, (
        f"registry advertises DAGs that no module emits: {sorted(phantom)}. "
        "Move them to ABSENT_DAGS or delete them."
    )

    assert not (absent & on_disk), (
        "a DAG marked ABSENT is actually on disk — remove it from ABSENT_DAGS"
    )


def test_no_hardcoded_architecture_counts_in_claude_md():
    """Counts in CLAUDE.md must come from inventory markers, not from memory."""
    import re

    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8", errors="replace")
    # strip generated blocks before looking for hand-written claims
    text = re.sub(r"<!--\s*inv:[a-z0-9_.-]+\s*-->.*?<!--\s*/inv\s*-->", "", text, flags=re.S)
    banned = re.findall(
        r"\b(\d{1,3})\s+(DAGs?|API routes?|route files?|migrations?|GitHub Actions)\b",
        text, flags=re.I,
    )
    assert not banned, (
        f"hand-maintained architecture counts found in CLAUDE.md: {banned}. "
        "Wrap them in <!-- inv:<key> --> ... <!-- /inv --> so the generator owns them."
    )
