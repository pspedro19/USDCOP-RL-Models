"""Boundary, BIC-rule and secret-refusal checks; no training or provider calls."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import regime5_bundle as bundle
from src.research import regime5_hmm as regime_hmm
from src.research.llm_forward.arms.ppo_stream import PPOStreamingRunner
from src.research.observation_contract import LEGACY_VERSION, REGIME5_VERSION, observation_contract
from tests.regression.test_regime5_bundle import frozen


@pytest.mark.parametrize("advantage,expected", [(0, 3), (10, 3), (11, 5)])
def test_existing_bic_hysteresis_is_preserved_not_forced_to_five(monkeypatch, advantage, expected):
    n, d = 250, 9
    scores = {2: 1100, 3: 1000, 4: 1100, 5: 1000 - advantage}
    observed = []

    def fit(X, k, covariance):
        observed.append(k)
        count = k * (k - 1) + (k - 1) + k * d + k * d * (d + 1) / 2
        loglik = (count * np.log(n) - scores[k]) / 2
        model = SimpleNamespace(
            covars_=np.array([np.eye(d)] * k),
            covariance_type=covariance,
            monitor_=SimpleNamespace(converged=True, iter=25),
        )
        return model, loglik

    monkeypatch.setattr(regime_hmm, "_fit_one", fit)
    data = pd.DataFrame(
        np.random.default_rng(42).normal(size=(n, d)),
        index=pd.date_range("2020-01-01", periods=n),
        columns=regime_hmm.FEATURE_NAMES,
    )
    result = regime_hmm.fit_frozen(data)
    assert observed == [2, 3, 4, 5]
    assert result.k == expected
    assert len(result.candidate_evidence) == 4
    assert all(row["converged"] is True for row in result.candidate_evidence)
    assert result.fit_range == ("2020-01-01", str(data.index[-1].date()))


@pytest.mark.parametrize("defect", ["boolean_scale", "boolean_startprob", "masked_array"])
def test_arrays_reject_types_before_coercion(defect):
    payload = frozen(5).to_payload()
    if defect == "boolean_scale":
        payload["scaler"]["scale"][0] = True
    elif defect == "boolean_startprob":
        payload["regime"]["startprob"][0] = False
    else:
        payload["scaler"]["mean"] = np.ma.array(np.zeros(25), mask=[True] + [False] * 24)
    with pytest.raises(ValueError):
        bundle.FrozenRegime5.from_payload(payload)


@pytest.mark.parametrize(
    "name", [".env", ".env.test", "secrets/data.json", "credentials.json", "model.key"]
)
def test_secret_rejected_before_any_bytes_are_read(tmp_path, name, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", lambda *a, **k: pytest.fail("secret read"))
    with pytest.raises(ValueError, match="secret"):
        bundle.load_bundle(tmp_path / name, expected_sha256="a" * 64)


def test_stream_legacy_cannot_silently_consume_new_checkpoint():
    model = SimpleNamespace(observation_space=SimpleNamespace(shape=(38,)))
    with pytest.raises(ValueError, match="shape"):
        PPOStreamingRunner(
            model, arm_id="test", model_id="fixture", preregistration_sha256="a" * 64
        )


@pytest.mark.parametrize("partial_version", [LEGACY_VERSION, "unknown"])
def test_stream_rejects_wrong_version_before_predict(partial_version):
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(38,)),
        research_observation_sha256=observation_contract(REGIME5_VERSION).sha256,
        predict=lambda *a, **k: pytest.fail("prediction must not occur"),
    )
    runner = PPOStreamingRunner(
        model,
        arm_id="test",
        model_id="fixture",
        preregistration_sha256="a" * 64,
        observation_version=REGIME5_VERSION,
    )
    partial = SimpleNamespace(observation_version=partial_version)
    with pytest.raises(ValueError, match="version"):
        runner.decide(
            session_date="2023-06-01",
            bar_index=0,
            partial=partial,
            previous_weight=0,
            bars_in_position=0,
            unrealized=0,
            drawdown=0,
            n_changes=0,
            bar_close_utc=datetime.now(UTC),
            bar_received_at_utc=datetime.now(UTC),
        )


def test_stale_checkpoint_metadata_is_rejected_before_deserialization(tmp_path, monkeypatch):
    metadata = tmp_path / "metadata.json"
    metadata.write_bytes(bundle.canonical_json({"observation_version": LEGACY_VERSION}))
    with pytest.raises(ValueError, match="binding"):
        bundle.load_bound_checkpoint(
            tmp_path / "nonexistent.zip",
            metadata_path=metadata,
            expected_metadata_sha256=bundle.digest(metadata.read_bytes()),
            bundle_path=tmp_path,
            expected_bundle_sha256="a" * 64,
        )


def test_diagonal_fallback_uses_one_variance_per_state(monkeypatch):
    def fit(X, k, covariance):
        if covariance == "full":
            return None, -np.inf
        return SimpleNamespace(
            covars_=np.array([np.eye(9) * scale for scale in range(k, 0, -1)]),
            covariance_type="diag",
            monitor_=SimpleNamespace(converged=True, iter=20),
        ), -100.0

    monkeypatch.setattr(regime_hmm, "_fit_one", fit)
    data = pd.DataFrame(
        np.random.default_rng(42).normal(size=(250, 9)),
        index=pd.date_range("2020-01-01", periods=250),
        columns=regime_hmm.FEATURE_NAMES,
    )
    fitted = regime_hmm.fit_frozen(data)
    assert fitted.covariance_type == "diag"
    assert fitted.vol_order == tuple(reversed(range(fitted.k)))
