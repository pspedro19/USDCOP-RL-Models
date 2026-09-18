"""Pin the published retrospective evidence, independently of today's engine.

Never update this SHA to make a changed report green. A missing original is an
explicit skip, not a regenerated 'golden' fixture or a confirmation of profit.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "outputs/thesis-repair/research_grade_20260912_v5"
MANIFEST_SHA = "00fc276e22b62ffac144bca1abb6c468f15e272225c46743304c67fe120ae2da"


def test_published_retrospective_totals_match_original_daily_series_and_figures():
    manifest_file = BUNDLE / "manifest.json"
    if not manifest_file.exists():
        pytest.skip(
            "original published retrospective bundle is not available; never regenerate golden data"
        )
    raw = manifest_file.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
    manifest = json.loads(raw)
    for name, expected in manifest["artifacts_sha256"].items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == expected, name
    daily = json.loads((BUNDLE / "daily_series.json").read_bytes())
    metrics = pd.read_csv(BUNDLE / "metrics.csv").set_index("arm")
    for arm, rows in daily.items():
        if arm not in metrics.index:
            continue
        g, c, n = (
            np.array([r[key] for r in rows])
            for key in ("gross_return", "cost_return", "net_return")
        )
        np.testing.assert_allclose(g - c, n, atol=1e-12, rtol=0)
        assert metrics.loc[arm, "return_compounded_pct"] == pytest.approx(
            100 * (np.prod(1 + n) - 1), abs=1e-9
        )
        assert metrics.loc[arm, "gross_compounded_pct"] == pytest.approx(
            100 * (np.prod(1 + g) - 1), abs=1e-9
        )
        assert metrics.loc[arm, "cost_sum_pct"] == pytest.approx(100 * c.sum(), abs=1e-9)
    assert manifest["scope"] == "retrospective_diagnostic"
    assert manifest["confirmatory_evidence_ready"] is False
