"""The forecasting surfaces must carry the diagnostic caveat.

Contract: CTR-QUANT-CONSTITUTION-001

The model zoo's directional accuracy is ~0.52 with the best model at p=0.11 unadjusted and
p~0.66 adjusted for the 9 models tried; the best model-by-horizon cell reaches p_adj = 1.0
over the 63 cells examined. The dashboard nonetheless displayed "Direction Accuracy: 52%" with
zero context, which reads as "the models work" to anyone who does not carry the significance
tables in their head.

Worse: the zoo family is NOT purely informational — `train_and_export_smart_simple.py` derives
an executed trade direction from the same model family. The caveat is the line between a
diagnostic surface and an implied recommendation.

This is a source-level check (the dashboard is a standalone build; runtime rendering is not
reachable from this suite). It pins the presence of the caveat markup in both surfaces that
show DA. Removing the banner makes this fail, which is the point: a disclaimer that can be
silently deleted is decoration, not disclosure.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "usdcop-trading-dashboard" / "components"

SURFACES = {
    "forecasting/ForecastingDashboard.tsx": "DiagnosticCaveat",
    "gm/views/ForecastingView.tsx": "da-caveat",
}


@pytest.mark.parametrize("rel,marker", SURFACES.items(), ids=list(SURFACES))
def test_da_surface_carries_caveat(rel: str, marker: str):
    p = DASH / rel
    if not p.is_file():
        pytest.skip(f"{rel} absent")
    src = p.read_text(encoding="utf-8", errors="replace")
    if "direction_accuracy" not in src and "Direction Accuracy" not in src:
        pytest.skip(f"{rel} no longer shows DA")
    assert marker in src, (
        f"{rel} displays Direction Accuracy but the caveat ({marker!r}) is gone. A ~52% DA "
        "shown without context reads as 'the models work'; the statistics say coin flip "
        "(p_adj 0.66 across models, 1.0 across model-by-horizon cells)."
    )


def test_caveat_is_not_hardcoded_to_a_stale_number():
    """The main dashboard banner must compute its DA from the loaded data.

    A hardcoded '52%' would silently become false the day the data changes — in either
    direction. The banner's honesty comes from being derived, not asserted.
    """
    src = (DASH / "forecasting/ForecastingDashboard.tsx").read_text(encoding="utf-8",
                                                                    errors="replace")
    assert "useMemo" in src.split("function DiagnosticCaveat")[1].split("function ")[0], (
        "DiagnosticCaveat must derive its statistics from the data prop, not a literal"
    )
