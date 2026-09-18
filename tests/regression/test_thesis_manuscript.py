"""Numerical/reporting invariants; no training or provider credentials."""

import numpy as np
import pytest

from scripts.presentation.build_thesis_manuscript import (
    ROOT,
    blocks,
    break_even,
    compound,
    curve,
    diagnostics,
    plain,
    verify_source,
)


def test_compounding_is_not_arithmetic_sum():
    assert compound([0.1, -0.1]) == pytest.approx(-1.0)
    np.testing.assert_allclose(curve([0.1, -0.1]), [100, 110, 99])


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf], [-1.0], [-1.1], [[0.1]]])
def test_returns_fail_closed(values):
    with pytest.raises(ValueError):
        compound(values)


def test_drawdown_includes_initial_capital():
    eq = curve([-0.1, -0.1])
    assert (eq / np.maximum.accumulate(eq) - 1).min() == pytest.approx(-0.19)


def test_break_even_known_solution():
    assert break_even([0.01] * 30, [0.02] * 30) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "g,c", [([-0.01] * 4, [0.02] * 4), ([0.0] * 4, [0.0] * 4), ([0.1] * 4, [0.01] * 4)]
)
def test_no_root_in_declared_domain(g, c):
    assert break_even(g, c) is None


@pytest.mark.parametrize("g,c", [([0.01], [-0.1]), ([0.01], [np.nan]), ([0.01], [0.1, 0.2])])
def test_costs_fail_closed(g, c):
    with pytest.raises(ValueError):
        break_even(g, c)


def test_markup_not_visible_in_rendered_text():
    assert (
        plain("*Paper* y **hecho** [fuente](https://example.org)")
        == "Paper y hecho fuente (https://example.org)"
    )


def test_markdown_tables_and_figures_parsed():
    parsed = list(
        blocks(
            "# 4. Resultados\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n![Real](figures/real.png)"
        )
    )
    assert [k for k, _ in parsed] == ["heading", "table", "figure"]
    assert parsed[1][1] == [["A", "B"], ["1", "2"]]


def test_manuscript_sources_complete_and_not_synthetic_results():
    a = (ROOT / "docs/thesis/chapters_1_2_5.md").read_text(encoding="utf-8")
    b = (ROOT / "docs/thesis/chapters_3_4.md").read_text(encoding="utf-8")
    assert all(
        h in a for h in ["# Resumen", "# Abstract", "# 1.", "# 2.", "# 5.", "# Bibliografía"]
    )
    assert all(h in b for h in ["# 3.", "# 4.", "{{COMPOUND_CI}}", "{{TABLE_GLOBAL}}"])
    assert len((a + b).split()) > 7500
    assert "escenario sintético coherente" not in a + b


def test_pinned_evidence_and_new_diagnostics_reproduce():
    report, daily, snapshot = verify_source()
    result = diagnostics(report, daily)
    assert len(snapshot.used) == 20
    assert result["gross_compounded_ci95_pct"][0] < 0 < result["gross_compounded_ci95_pct"][1]
    np.testing.assert_allclose(
        result["gross_sum_ci95_pct"], [-3.512219204992286, 24.140047137622542]
    )
    r = daily["ppo_median_weights"]
    k = result["break_even_cost_multiplier"]["ppo_median_weights"]
    assert 0 < k < 1
    assert compound([v["gross_return"] - k * v["cost_return"] for v in r]) == pytest.approx(
        0, abs=1e-9
    )
    assert sum(q["n_sessions"] for q in result["quarters"]) == 226
    assert result["alpha_established"] is False


def test_manifest_mutation_rejected(tmp_path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest pin"):
        verify_source(tmp_path)
