"""The vendored quant skill library stays vendored, and quarantined skills stay out.

Contract: CTR-QUANT-LIBRARY-001

119 external skills were imported from a US discretionary-equity library. Only 13 were
promoted to `.claude/skills/`; the rest are read-only reference under `vendor/quant-skills/`.

Two failure modes this guards:

1. **Silent bulk promotion.** Skill front matter is preloaded into EVERY session. The full
   library carries ~60 KB of it (~15k tokens — four times the entire auto-loaded rules
   budget), and nothing else measures that cost.
2. **Promoting a skill that contradicts the constitution.** Six were quarantined for concrete
   reasons: one scores a pure in-sample curve fit as "Deploy", another exists to iterate until
   the OOS improves, another refits weights on realized outcomes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "quant-skills"
SKILLS = ROOT / ".claude" / "skills"
PROVENANCE = VENDOR / "PROVENANCE.md"

# Each of these contradicts .claude/rules/quant-constitution.md. See the router skill for why.
QUARANTINED = {
    "crypto-trading-signals",     # third-party API emitting leverage w/ self-reported confidence
    "backtest-expert",            # evaluate_backtest.py scores an in-sample curve fit as "Deploy"
    "strategy-pivot-designer",    # exists to iterate until the OOS improves
    "edge-signal-aggregator",     # refits weights on realized outcomes
    "signal-postmortem",          # feeds results back into the aggregator's weights
    "asset-allocation",           # mean-variance / Black-Litterman = in-sample optimisation
}

pytestmark = pytest.mark.skipif(not VENDOR.is_dir(), reason="quant skill library not vendored")


def _promoted() -> set[str]:
    return {p.name for p in SKILLS.iterdir() if p.is_dir()}


def test_provenance_manifest_exists():
    assert PROVENANCE.is_file(), (
        "vendor/quant-skills/PROVENANCE.md is the audit trail for what was imported and why"
    )


def test_quarantined_skills_are_not_promoted():
    leaked = QUARANTINED & _promoted()
    assert not leaked, (
        f"quarantined skills promoted to .claude/skills/: {sorted(leaked)}. "
        "Each contradicts quant-constitution.md — see .claude/skills/quant-algo-trading/SKILL.md"
    )


def test_every_promoted_library_skill_is_declared():
    """A skill promoted out of the library must have a row saying it was reviewed."""
    text = PROVENANCE.read_text(encoding="utf-8", errors="replace")
    vendored = {p.parent.name for p in VENDOR.rglob("SKILL.md")}
    for name in sorted(_promoted() & vendored):
        assert f"`{name}`" in text, (
            f"{name} was promoted from the library but has no PROVENANCE row"
        )


def test_vendor_stays_out_of_the_discovered_namespace():
    """`vendor/` must not become a skills directory by accident."""
    assert not (SKILLS / "vendor").exists()
    assert not (VENDOR / "SKILL.md").exists(), (
        "a SKILL.md at the vendor root would make the whole library look like one skill"
    )


def test_no_binaries_or_caches_tracked_in_vendor():
    offenders = [
        p.relative_to(ROOT).as_posix()
        for p in VENDOR.rglob("*")
        if p.is_file() and (
            p.suffix.lower() in {".jpeg", ".jpg", ".png", ".zip", ".pyc", ".parquet"}
        )
    ]
    assert not offenders, f"binary/cache files vendored: {offenders[:10]}"
    assert not list(VENDOR.rglob(".pytest_cache")), "strip .pytest_cache before committing"


def test_promoted_skills_do_not_shadow_the_metrics_ssot():
    """No promoted skill may re-implement a function exported by the constitutional SSOT.

    `quant-constitution.md` names `services/common/metrics.py::deflated_sharpe_ratio` as the
    only source for edge claims, and it gates releases in production pipelines.

    `xasset-alpha-engine` shipped its own `deflated_sharpe_ratio(returns, trial_sharpes) ->
    float` while the repo's takes `(sharpe_per_period, n_obs, n_trials, trials_sharpe_std,
    skew, kurtosis) -> dict`. Same name, different arguments, different return type — and the
    skill triggers on the phrase "deflated sharpe". An agent asked to compute it would route
    to the skill, never see the router's warning, and silently bypass the release gate.

    Wrappers that DELEGATE are fine (and are what `validation.py` now does); a second
    implementation of the maths is not.
    """
    import re

    metrics_src = (ROOT / "services" / "common" / "metrics.py").read_text(
        encoding="utf-8", errors="replace"
    )
    ssot_symbols = set(re.findall(r"^def ([a-z_][a-z0-9_]*)\(", metrics_src, re.M))
    ssot_symbols -= {"main"}

    offenders: list[str] = []
    for py in SKILLS.rglob("*.py"):
        src = py.read_text(encoding="utf-8", errors="replace")
        delegates = "services.common" in src
        for name in re.findall(r"^def ([a-z_][a-z0-9_]*)\(", src, re.M):
            if name in ssot_symbols and not delegates:
                offenders.append(f"{py.relative_to(ROOT).as_posix()}::{name}")

    assert not offenders, (
        "promoted skills re-implement SSOT metrics without delegating: "
        f"{offenders}. Import from services.common.metrics instead — two implementations of "
        "the constitution's own gate is worse than none."
    )


def test_vendor_tests_are_not_collected():
    """`testpaths = ['tests']` keeps them out; this pins that guarantee.

    The library ships 236 tests with colliding basenames (8x test_report_generator.py) and
    almost no __init__.py, so collecting them together raises import-file-mismatch errors.
    """
    import tomllib

    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_cfg = cfg["tool"]["pytest"]["ini_options"]
    assert pytest_cfg.get("testpaths") == ["tests"], (
        "testpaths must stay scoped to tests/ or the vendored library joins the suite"
    )
    assert "vendor" in pytest_cfg.get("norecursedirs", []), (
        "add 'vendor' to norecursedirs — belt and braces against `pytest .`"
    )


def test_promoted_skills_shipping_code_also_ship_tests():
    """Promoted code without tests is unverified code, because `.claude` is excluded.

    `norecursedirs` contains `.claude`, so a default `pytest` run never sees a
    promoted skill's modules. That exclusion is correct — the library's 236 external
    tests collide on basenames and hit the network — but it means any EDIT to
    promoted code lands unverified unless something runs its tests explicitly.

    That is not hypothetical. `xasset-alpha-engine` was edited to delegate Deflated
    Sharpe to the constitutional SSOT; the return type went from float to dict and
    `validate()` kept comparing it against a float, so the skill raised TypeError on
    its main entry point. Its 74 tests caught it in one run — but they had been
    stripped during promotion, so nothing ran them and the gate stayed green.

    specs-gate now runs `.claude/skills/*/scripts/tests` explicitly. This test makes
    sure there is always something there for it to run.

    The library uses two verification patterns and both are accepted:
      - a `scripts/tests/` suite (xasset-alpha-engine)
      - a single PEP-723 module exposing `--verify`, which re-checks the worked
        examples in its own SKILL.md (the finance_skills house style)

    What is NOT accepted is executable code with neither.
    """
    offenders = []
    for skill_dir in sorted((ROOT / ".claude" / "skills").iterdir()):
        scripts = skill_dir / "scripts"
        if not scripts.is_dir():
            continue
        modules = [
            p for p in scripts.rglob("*.py")
            if "tests" not in p.parts and p.name != "__init__.py"
        ]
        if not modules:
            continue
        has_tests = bool(list(scripts.rglob("test_*.py")))
        has_verify = any(
            "--verify" in p.read_text(encoding="utf-8", errors="ignore")
            for p in modules
        )
        if not (has_tests or has_verify):
            offenders.append(f"{skill_dir.name} ({len(modules)} modules)")

    assert not offenders, (
        "promoted skills ship executable code with neither tests nor a --verify "
        f"self-check: {offenders}. `.claude` is in norecursedirs, so this code is "
        "invisible to CI. Copy the tests across when promoting, add --verify, or "
        "vendor the skill instead of promoting it."
    )
