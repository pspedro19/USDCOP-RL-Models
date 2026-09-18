"""C044 extends the diagnostic cohort, not the model or the evidence category."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.diagnostics import audit_hmm_2026_representation as runner

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "outputs/thesis-repair/archive/pre_research_grade_20260912_1934/manifest.json"
SNAPSHOT_SHA = "3caf361570a2ed104bd8ff118ce55989eb9b2b0ccb3105877d056ba987194ea1"
REFERENCE = ROOT / "outputs/thesis-repair/hmm_representation_20260913/diagnostic_v3.json"
REFERENCE_SHA = "c50bf861614c0ba26af29482ab37efe72a5d08cb235ac6e89d8c63077e884930"


def test_selection_reference_requires_external_hash():
    assert runner.load_reference(REFERENCE, REFERENCE_SHA)["cohort"]["n_sessions"] == 226
    with pytest.raises(ValueError, match="reference hash"):
        runner.load_reference(REFERENCE, "0" * 64)


def test_only_all_archived_2026_sessions_are_the_comparison_cohort():
    days = pd.bdate_range("2026-01-02", periods=150)
    specs = [SimpleNamespace(date=d.date()) for d in days]
    blob = {"holdout": [SimpleNamespace(date=pd.Timestamp("2025-12-31").date()), *specs]}
    result = runner.cohort_specs(blob)
    assert result == specs
    with pytest.raises(ValueError, match="150"):
        runner.cohort_specs({"holdout": specs[:-1]})
    with pytest.raises(ValueError):
        runner.cohort_specs({"holdout": list(reversed(specs))})
    with pytest.raises(ValueError):
        runner.cohort_specs({"holdout": [*specs[:-1], specs[0]]})


@pytest.fixture(scope="module")
def report():
    return runner.audit_2026(SNAPSHOT, SNAPSHOT_SHA, REFERENCE, REFERENCE_SHA)


def test_real_baseline_2026_parity_uses_full_pre2026_history(report):
    assert report["baseline_parity"]["passed"]
    assert report["baseline_parity"]["n_expected"] == 150
    assert report["baseline_parity"]["max_abs_error"] <= 1e-6
    assert report["baseline_numerics"]["first_observation"] == "2019-12-24"
    assert report["baseline_numerics"]["n_observations"] > 774
    assert report["cohort"]["source_partition"] == "holdout"
    assert report["cohort"]["year"] == 2026
    assert report["cohort"]["n_bars"] == 9000
    assert report["scope"] == "retrospective_diagnostic_not_confirmatory"
    assert report["strategy_returns_evaluated"] is False
    assert report["current_model_eligible"] is False


def test_counterfactual_is_paired_and_preserves_parameters(report):
    assert report["history_alignment"]["passed"]
    assert report["counterfactual"]["n_sessions"] == 150
    assert len(report["counterfactual"]["rows"]) == 150
    assert report["model"]["k"] == 5
    assert report["baseline_numerics"]["fallback_rows"] == []
    assert report["counterfactual_numerics"]["fallback_rows"] == []
    assert report["counterfactual"]["same_fitted_parameters"] is True


def test_reference_runner_and_helper_bytes_are_bound_before_unpickle(monkeypatch):
    reference = runner.load_reference(REFERENCE, REFERENCE_SHA)
    reference["runner_sha256"] = "0" * 64
    monkeypatch.setattr(runner, "load_reference", lambda *a: reference)
    monkeypatch.setattr(runner.pickle, "loads", lambda *a: pytest.fail("pickle must not be read"))
    with pytest.raises(ValueError, match="C043 source"):
        runner.audit_2026(SNAPSHOT, SNAPSHOT_SHA, REFERENCE, REFERENCE_SHA)


def test_mismatched_reference_snapshot_is_rejected(monkeypatch):
    reference = runner.load_reference(REFERENCE, REFERENCE_SHA)
    reference["snapshot_sha256"] = "0" * 64
    monkeypatch.setattr(runner, "load_reference", lambda *a: reference)
    with pytest.raises(ValueError, match="snapshot"):
        runner.audit_2026(SNAPSHOT, SNAPSHOT_SHA, REFERENCE, REFERENCE_SHA)


def test_bad_baseline_cannot_generate_counterfactual(monkeypatch):
    actual = runner.filtered
    builds = []

    def broken(*args):
        full, source, numeric = actual(*args)
        builds.append(1)
        full.loc[full.index.year == 2026] = np.nan
        return full, source, numeric

    monkeypatch.setattr(runner, "filtered", broken)
    result = runner.audit_2026(SNAPSHOT, SNAPSHOT_SHA, REFERENCE, REFERENCE_SHA)
    assert result["status"] == "BASELINE_PARITY_FAILED"
    assert result["counterfactual"] is None
    assert len(builds) == 1


def test_ecdf_is_exact_and_contains_all_observations():
    x, y = runner.ecdf([0.0, 0.0, 0.5, 1.0])
    np.testing.assert_array_equal(x, [0, 0, 0.5, 1])
    np.testing.assert_array_equal(y, [0.25, 0.5, 0.75, 1.0])
    for bad in ([], [np.nan], [-1.0], [1.01]):
        with pytest.raises(ValueError):
            runner.ecdf(bad)


def test_bundle_refuses_existing_directory_and_has_traceable_figure(tmp_path, report):
    output = tmp_path / "bundle"
    reference = runner.load_reference(REFERENCE, REFERENCE_SHA)
    manifest = runner.write_bundle(report, reference, output)
    assert set(manifest["files"]) == {"diagnostic.json", "posterior_rows.csv", "sensitivity.png",
                                     "sensitivity.svg", "sensitivity.txt"}
    for key, digest in manifest["files"].items():
        assert runner.sha256((output / key).read_bytes()) == digest
    with pytest.raises(FileExistsError):
        runner.write_bundle(report, reference, output)


def test_figure_zoom_preserves_full_distribution(tmp_path, report, monkeypatch):
    from matplotlib.figure import Figure

    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda self, *a, **kw: figures.append(self))
    runner.figure(report, runner.load_reference(REFERENCE, REFERENCE_SHA), tmp_path)
    main = figures[0].axes[0]
    assert main.get_xlim()[0] <= 0 and main.get_xlim()[1] >= 1
    assert len(main.child_axes) == 1
    zoom = main.child_axes[0]
    np.testing.assert_allclose(zoom.get_xlim(), [0, 0.05])
    np.testing.assert_allclose(zoom.get_ylim(), [0.8, 1.005])
    for full, inset in zip(main.lines, zoom.lines, strict=True):
        np.testing.assert_array_equal(full.get_xdata(), inset.get_xdata())
        np.testing.assert_array_equal(full.get_ydata(), inset.get_ydata())


def test_failed_report_has_no_figure(tmp_path, report):
    broken = deepcopy(report)
    broken["baseline_parity"]["passed"] = False
    broken["status"] = "BASELINE_PARITY_FAILED"
    broken["counterfactual"] = None
    output = tmp_path / "failed"
    result = runner.write_bundle(broken, runner.load_reference(REFERENCE, REFERENCE_SHA), output)
    assert set(result["files"]) == {"diagnostic.json"}
    assert not (output / "sensitivity.png").exists()


@pytest.mark.parametrize("mutation", ["tv", "summary", "probability", "state", "date", "missing",
                                     "alignment", "fallback", "reference"])
def test_writer_rejects_inconsistent_successful_evidence(tmp_path, report, mutation):
    broken = deepcopy(report)
    reference = runner.load_reference(REFERENCE, REFERENCE_SHA)
    row = broken["counterfactual"]["rows"][0]
    if mutation == "tv":
        row["total_variation"] += 0.1
    elif mutation == "summary":
        broken["counterfactual"]["changed_argmax"] += 1
    elif mutation == "probability":
        row["original"][0] = 2.0
    elif mutation == "state":
        row["original_state_coordinate"] = 10
    elif mutation == "date":
        row["date"] = broken["counterfactual"]["rows"][1]["date"]
    elif mutation == "missing":
        broken["counterfactual"]["rows"].pop()
    elif mutation == "alignment":
        broken["history_alignment"]["passed"] = False
    elif mutation == "fallback":
        broken["counterfactual_numerics"]["fallback_rows"] = [0]
    else:
        reference["counterfactual"]["rows"][0]["total_variation"] = 0.5
    with pytest.raises(ValueError):
        runner.write_bundle(broken, reference, tmp_path / mutation)
    assert not (tmp_path / mutation / "manifest.json").exists()
