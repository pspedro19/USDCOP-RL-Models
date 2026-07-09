"""PRE-REGISTRATION <-> YAML <-> code consistency gate (SPEC-11 / audit I-5, G5).

The audit found silent drift between design YAMLs, PRE-REGISTRATION and the as-built code
(gold risk.yaml 0.12/3.0 vs code 0.10/1.5; data.yaml dukascopy vs TwelveData; BTC vol
estimator semidev-EWMA30 vs realized_vol_20 std with floor 0.30). SPEC-11 required a
consistency test that never existed. This is it:

  1. As-built anchors: the REAL parameter values in code must match the values declared
     here (and in the superseded notes). If someone changes the code, this fails until the
     declaration (and any ADR) is updated — no more silent drift.
  2. Superseded markers: the divergent design YAMLs must carry a `SUPERSEDED` header. If
     someone deletes the marker without syncing, this fails.
  3. Known divergences are ENUMERATED (visible, tracked) — not silently tolerated.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPECS = REPO / ".claude" / "specs" / "assets"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------- 1. as-built anchors
def test_gold_asbuilt_vol_params_match_declaration():
    """Gold as-built: target_vol=0.10, floor=0.06, cap=1.5 (NOT risk.yaml's 0.12/3.0)."""
    src = _read(REPO / "src" / "gold_rl" / "strategies.py")
    assert "target_vol: float = 0.10" in src, "gold target_vol drifted from declared 0.10"
    assert "max_leverage: float = 1.5" in src, "gold max_leverage drifted from declared 1.5"
    assert re.search(r"clip\(lower=0\.06\)", src), "gold vol floor drifted from declared 0.06"


def test_btc_asbuilt_vol_params_match_declaration():
    """BTC as-built: TARGET_VOL=0.30, VOL_FLOOR=0.30, estimator=realized_vol_20 (simple std).

    KNOWN DIVERGENCE vs btc:PRE-REGISTRATION §2 (downside semidev EWMA-30): implemented is
    a simple 20d std with floor==target -> exposure can only shrink (typical ~0.44). This is
    tracked in HYPOTHESIS-REGISTRY / plan OLA 5; changing either side requires an ADR.
    """
    src = _read(REPO / "src" / "btc_strategy" / "strategies.py")
    assert "TARGET_VOL = 0.30" in src, "BTC TARGET_VOL drifted from declared 0.30"
    assert "VOL_FLOOR = 0.30" in src, "BTC VOL_FLOOR drifted from declared 0.30"
    ind = _read(REPO / "src" / "btc_strategy" / "indicators.py")
    assert "realized_vol_20" in ind, "BTC vol estimator drifted from declared realized_vol_20"


# ---------------------------------------------------------------- 2. superseded markers
SUPERSEDED_YAMLS = [
    SPECS / "xauusd" / "config" / "risk.yaml",
    SPECS / "xauusd" / "config" / "data.yaml",
    SPECS / "xauusd" / "config" / "train.yaml",
]


def test_divergent_design_yamls_are_marked_superseded():
    for p in SUPERSEDED_YAMLS:
        assert p.exists(), f"missing design yaml {p}"
        head = "\n".join(_read(p).splitlines()[:8])
        assert "SUPERSEDED" in head, (
            f"{p.name}: design yaml diverges from as-built but lost its SUPERSEDED marker; "
            "either restore the marker or sync it via ADR"
        )


# ---------------------------------------------------------------- 3. constitution wiring
def test_transversal_constitution_exists_and_indexed():
    rule = REPO / ".claude" / "rules" / "quant-constitution.md"
    assert rule.exists(), "transversal quant constitution missing"
    body = _read(rule)
    for needle in ("grid search", "DSR", "B1′", "retiro"):
        assert needle in body, f"constitution lost its '{needle}' clause"
    idx = _read(REPO / ".claude" / "rules" / "00-INDEX.md")
    assert "quant-constitution.md" in idx, "constitution not registered in rules index"


def test_cop_registry_and_withdrawal_protocol_exist():
    cop = SPECS / "usdcop"
    assert (cop / "HYPOTHESIS-REGISTRY.md").exists(), "COP trial registry missing (G2)"
    assert (cop / "WITHDRAWAL-PROTOCOL.md").exists(), "COP withdrawal protocol missing (G3)"
    reg = _read(cop / "HYPOTHESIS-REGISTRY.md")
    assert "#8 of 42" in reg, "registry lost the grid-evidence citation"
