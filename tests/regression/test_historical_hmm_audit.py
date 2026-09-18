"""Historical diagnostics must not bypass current model/data identity gates."""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.historical_hmm_audit import (
    HistoricalArchive,
    compare_stored,
    historical_model,
    posterior_path,
    recover_mask,
    representation_delta,
    shifted_path,
)
from src.research.regime_portable import _parameter_digest


def payload():
    result = {
        "contract": "CTR-RESEARCH-REGIME-PORTABLE-001", "k": 2,
        "dataset_identity": "a" * 64, "startprob": [0.7, 0.3],
        "transmat": [[0.9, 0.1], [0.2, 0.8]], "means": [[-1.0], [1.0]],
        "covars": [[[0.5]], [[0.8]]], "std_means": [0.0], "std_scales": [1.0],
        "vol_order": [1, 0], "labels": ["calmo", "shock"],
        "feature_names": ["ret"], "fit_range": ["2020-01-01", "2020-12-31"],
    }
    result["parameter_sha256"] = _parameter_digest(result)
    return result


def metadata(data):
    return {"identity": data["dataset_identity"],
            "regime_artifact_sha256": hashlib.sha256(json.dumps(data).encode()).hexdigest(),
            "regime_meta": {"covariance_type": "full", **{
                k: deepcopy(data[k]) for k in ("k", "labels", "feature_names", "fit_range")}}}


def model():
    data = payload()
    return checked_model(data, metadata(data))[0]


def checked_model(data, blob):
    return historical_model(data, blob,
                            model_sha256=hashlib.sha256(json.dumps(data).encode()).hexdigest())


@pytest.mark.parametrize("field,value", [("regime_artifact_sha256", "0" * 64),
                                        ("regime_artifact_sha256", None),
                                        ("covariance_type", "diag"),
                                        ("covariance_type", None)])
def test_explicit_archived_model_link_and_full_covariance_required(field, value):
    data = payload()
    blob = metadata(data)
    if field == "covariance_type":
        blob["regime_meta"][field] = value
    else:
        blob[field] = value
    with pytest.raises(ValueError):
        historical_model(data, blob,
                         model_sha256=hashlib.sha256(json.dumps(data).encode()).hexdigest())


def archive(tmp_path, files=None):
    entries = []
    for key, data in (files or {"data/example.json": b'{"v": 1}'}).items():
        digest = hashlib.sha256(data).hexdigest()
        target = tmp_path / "objects" / digest
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(data)
        entries.append({"path": key, "sha256": digest, "bytes": len(data),
                        "object": "objects/" + digest})
    raw = json.dumps({"contract": "THESIS-EVIDENCE-SNAPSHOT-1", "files": entries}).encode()
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_archive_requires_external_digest_before_parsing(tmp_path):
    path, digest = archive(tmp_path)
    assert HistoricalArchive(path, digest).read("data/example.json") == b'{"v": 1}'
    with pytest.raises(ValueError, match="manifest hash"):
        HistoricalArchive(path, "0" * 64)


def test_archive_detects_object_corruption(tmp_path):
    path, digest = archive(tmp_path)
    item = json.loads(path.read_bytes())["files"][0]
    (tmp_path / item["object"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="object integrity"):
        HistoricalArchive(path, digest).read(item["path"])


@pytest.mark.parametrize("key", ["../outside", "/absolute", "C:/absolute", ".env",
                                 "foo/.env.test", "secrets/data.json", "x/key.pem"])
def test_archive_refuses_sensitive_or_escaping_logical_paths(tmp_path, key):
    path, digest = archive(tmp_path, {key: b"not a secret"})
    with pytest.raises(ValueError):
        HistoricalArchive(path, digest)


def test_archive_refuses_duplicate_keys_and_entries(tmp_path):
    path, _ = archive(tmp_path)
    for raw in (b'{"contract":1,"contract":2}',
                json.dumps({"contract": "THESIS-EVIDENCE-SNAPSHOT-1", "files":
                            json.loads(path.read_bytes())["files"] * 2}).encode()):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            HistoricalArchive(path, hashlib.sha256(raw).hexdigest())


def test_parameter_hash_identity_and_metadata_are_separate_checks():
    data = payload()
    blob = metadata(data)
    obj, report = checked_model(data, blob)
    assert obj.k == 2
    assert report["current_model_eligible"] is False
    assert report["identity_manifest_present"] is False
    for field, value, expected in (("dataset_identity", "b" * 64, "identity"),
                                   ("parameter_sha256", "b" * 64, "parameter")):
        broken = {**data, field: value}
        with pytest.raises(ValueError, match=expected):
            checked_model(broken, blob)
    blob["regime_meta"]["labels"] = ["other", "shock"]
    with pytest.raises(ValueError, match="metadata"):
        checked_model(data, blob)


@pytest.mark.parametrize("field,value", [
    ("k", True), ("startprob", [0.2, 0.3]), ("startprob", [-0.1, 1.1]),
    ("transmat", [[0.8, 0.1], [0.2, 0.8]]), ("means", [[1.0]]),
    ("std_scales", [0.0]), ("covars", [[[-0.5]], [[0.8]]]),
    ("vol_order", [1, 1]), ("vol_order", [True, 0]),
    ("feature_names", []), ("std_means", [float("nan")]),
])
def test_model_refuses_invalid_parameters_even_if_rehashed(field, value):
    data = payload()
    data[field] = value
    data["parameter_sha256"] = _parameter_digest(data)
    with pytest.raises(ValueError):
        checked_model(data, metadata(data))


def test_posterior_matches_exhaustive_latent_paths_and_production_prefixes():
    obj = model()
    observations = np.array([[-0.4], [0.7], [1.5], [-0.2]])
    actual, diagnostics = posterior_path(obj, observations)
    expected = []
    for length in range(1, len(observations) + 1):
        mass = np.zeros(2)
        for states in itertools.product(range(2), repeat=length):
            prob = obj.startprob[states[0]]
            for t, state in enumerate(states):
                if t:
                    prob *= obj.transmat[states[t-1], state]
                var = obj.covars[state, 0, 0]
                diff = observations[t, 0] - obj.means[state, 0]
                prob *= np.exp(-0.5 * diff**2 / var) / np.sqrt(2 * np.pi * var)
            mass[states[-1]] += prob
        expected.append((mass / mass.sum())[list(obj.vol_order)])
        np.testing.assert_allclose(actual[length-1], obj.filtered_posterior(
            observations[:length]), atol=1e-14, rtol=0)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
    assert diagnostics["fallback_rows"] == []


def test_future_extension_does_not_rewrite_past():
    obj = model()
    short = np.array([[-0.4], [0.7], [1.5]])
    prefix, _ = posterior_path(obj, short)
    extended, _ = posterior_path(obj, np.vstack([short, [[-10.0], [20.0]]]))
    np.testing.assert_array_equal(prefix, extended[:3])


def test_numerical_fallback_is_reported_not_hidden(monkeypatch):
    obj = model()
    monkeypatch.setattr(type(obj), "_log_emission", lambda self, x: np.full((len(x), 2), -np.inf))
    with np.errstate(invalid="ignore"):
        _, diagnostics = posterior_path(obj, np.array([[0.0], [1.0]]))
    assert diagnostics["fallback_rows"] == [0, 1]


@pytest.mark.parametrize("obs", [np.empty((0, 1)), np.ones((2, 2)), [[float("nan")]]])
def test_posterior_refuses_bad_observations(obs):
    with pytest.raises(ValueError):
        posterior_path(model(), obs)


def mask_hash(dates):
    return hashlib.sha256("\n".join(dates).encode()).hexdigest()


def test_mask_outlier_is_valid_but_not_train_valid_and_hash_is_checked():
    dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
    mask = {"flat_ohlc_pct": dict.fromkeys(dates, 0),
            "excluded": {"holiday": [dates[0]], "outlier_suspect": [dates[1]]},
            "n_valid": 2, "sha256": mask_hash(dates[1:]),
            "n_train_valid": 1, "train_valid_sha256": mask_hash(dates[2:])}
    assert [str(x) for x in recover_mask(mask, dates)[0]] == dates[1:]
    with pytest.raises(ValueError, match="mask"):
        recover_mask(mask, [*dates, "2020-01-04"])
    mask["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mask"):
        recover_mask(mask, dates)


def test_shift_is_on_clean_index_not_calendar_and_keeps_warmup():
    index = pd.to_datetime(["2022-12-28", "2022-12-29", "2023-01-03"])
    frame = pd.DataFrame([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], index=index)
    shifted, sources = shifted_path(frame, min_context=2)
    assert shifted.iloc[:2].isna().all().all()
    np.testing.assert_array_equal(shifted.iloc[2], frame.iloc[1])
    assert sources.iloc[2] == pd.Timestamp("2022-12-29")


def test_parity_names_four_coordinate_scope_and_missing_dates_fail():
    index = pd.to_datetime(["2023-01-03", "2023-01-04"])
    full = pd.DataFrame([[0.1, 0.2, 0.3, 0.1, 0.3]] * 2, index=index)
    stored = full.iloc[:, :4].to_numpy(dtype=np.float32)
    result = compare_stored(full, index, stored)
    assert result["passed"] is True
    assert result["parity_scope"] == "four_stored_coordinates_of_a_five_state_posterior"
    assert compare_stored(full.iloc[:1], index, stored)["passed"] is False
    stored[0, 0] += 1e-4
    assert compare_stored(full, index, stored)["passed"] is False


def test_counterfactual_requires_baseline_and_entire_history_alignment():
    index = pd.date_range("2023-01-01", periods=3)
    original = pd.DataFrame([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], index=index)
    for passed, other in ((False, original), (True, original.iloc[1:])):
        with pytest.raises(ValueError):
            representation_delta(original, other, index[1:], baseline_passed=passed)
    result = representation_delta(original, original, index[1:], baseline_passed=True)
    assert result["n_sessions"] == 2
    assert result["changed_argmax"] == 0
    assert result["total_variation_max"] == 0


def test_source_binding_fails_on_changed_code(tmp_path):
    path, digest = archive(tmp_path, {"src/math.py": b"original"})
    source = tmp_path / "src" / "math.py"
    source.parent.mkdir()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source binding"):
        HistoricalArchive(path, digest).bind_current_code(tmp_path, ("src/math.py",))


def test_same_length_object_mutation_cannot_pass(tmp_path):
    path, digest = archive(tmp_path, {"data/item": b"abcd"})
    snap = HistoricalArchive(path, digest)
    snap.object_path("data/item").write_bytes(b"abce")
    with pytest.raises(ValueError, match="object integrity"):
        snap.read("data/item")


def test_parameter_covariance_jitter_is_reported():
    data = payload()
    data["covars"][0][0][0] = 0.0
    data["parameter_sha256"] = _parameter_digest(data)
    _, report = checked_model(data, metadata(data))
    assert report["covariance_diagonal_jitter"] == [1e-12, 0.0]


REAL_SNAPSHOT = Path(__file__).resolve().parents[2] / (
    "outputs/thesis-repair/archive/pre_research_grade_20260912_1934/manifest.json")
REAL_SNAPSHOT_SHA = "3caf361570a2ed104bd8ff118ce55989eb9b2b0ccb3105877d056ba987194ea1"


@pytest.fixture(scope="module")
def real_report():
    from scripts.diagnostics.audit_hmm_representation import audit
    return audit(REAL_SNAPSHOT, REAL_SNAPSHOT_SHA)


def test_real_archive_reproduces_selection_without_current_loader(real_report):
    assert real_report["baseline_parity"]["passed"]
    assert real_report["baseline_parity"]["n_expected"] == 226
    assert real_report["baseline_parity"]["max_abs_error"] < 3e-8
    assert real_report["baseline_numerics"]["n_observations"] == 774
    assert real_report["baseline_numerics"]["first_observation"] == "2019-12-24"
    assert real_report["baseline_parity"]["rows"][0]["observation_date"] == "2022-12-29"
    assert real_report["baseline_numerics"]["fallback_rows"] == []
    assert real_report["model"]["covariance_diagonal_jitter"] == [0.0] * 5


def test_real_counterfactual_has_no_silent_cohort_reduction(real_report):
    assert real_report["cohort"]["not_evaluated_cohorts"] == ["holdout", "2026", "forward"]
    assert real_report["cohort"]["n_bars_selection"] == 226 * 60
    assert 0 <= real_report["cohort"]["flat_ohlc_fraction_selection"] <= 1
    assert real_report["history_alignment"]["passed"] is True
    assert real_report["counterfactual"]["n_sessions"] == 226
    assert real_report["counterfactual"]["changed_argmax"] == 5
    assert real_report["counterfactual_numerics"]["fallback_rows"] == []
    assert real_report["counterfactual"]["total_variation_max"] > 0.89
    for key in ("rv", "log_rv", "ret", "abs_ret", "intraday_autocorr", "dxy_ret", "brent_ret"):
        assert real_report["counterfactual"]["feature_changes_on_clean_history"][key]["n_rows_different"] == 0


def test_failed_baseline_prevents_counterfactual_and_emits_finite_json(monkeypatch):
    from scripts.diagnostics import audit_hmm_representation as runner
    original_filter = runner.filtered
    original_build = runner.observations
    builds = []

    def broken_filter(*args):
        full, sources, diagnostics = original_filter(*args)
        full.loc[pd.Timestamp("2023-01-03")] = np.nan
        return full, sources, diagnostics

    def tracked_build(*args):
        builds.append(1)
        return original_build(*args)

    monkeypatch.setattr(runner, "filtered", broken_filter)
    monkeypatch.setattr(runner, "observations", tracked_build)
    report = runner.audit(REAL_SNAPSHOT, REAL_SNAPSHOT_SHA)
    assert report["status"] == "BASELINE_PARITY_FAILED"
    assert report["counterfactual"] is None
    assert report["baseline_parity"]["rows"][0]["max_abs_error"] is None
    assert len(builds) == 1
    json.dumps(report, allow_nan=False)


def test_runner_never_calls_current_identity_loader(monkeypatch):
    from scripts.diagnostics.audit_hmm_representation import audit
    from src.research.regime_portable import PortableRegimeModel

    def forbidden(*args, **kwargs):
        raise AssertionError("current loader is not the historical adapter")

    monkeypatch.setattr(PortableRegimeModel, "load", forbidden)
    monkeypatch.setattr(PortableRegimeModel, "from_dict", forbidden)
    assert audit(REAL_SNAPSHOT, REAL_SNAPSHOT_SHA)["current_model_eligible"] is False


def test_cli_refuses_overwrite_before_any_audit(tmp_path, monkeypatch):
    from scripts.diagnostics import audit_hmm_representation as runner
    existing = tmp_path / "result.json"
    existing.write_bytes(b"preserve me")
    monkeypatch.setattr("sys.argv", ["audit", "--snapshot", str(REAL_SNAPSHOT),
                                    "--expected-snapshot-sha", REAL_SNAPSHOT_SHA,
                                    "--output", str(existing)])
    monkeypatch.setattr(runner, "audit", lambda *a, **k: pytest.fail("audit must not run"))
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 2
    assert existing.read_bytes() == b"preserve me"
