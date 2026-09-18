"""Bundle publication is exclusive, transactional and independent of frozen globals."""
import hashlib
import json
import pickle
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.data import export_research_bundle_v3 as module


@pytest.fixture
def bundle_data(monkeypatch):
    monkeypatch.setattr(module, "_current_contract", lambda: {"identity_version": 3, "inputs": "fixture"})
    model = SimpleNamespace(
        k=4, feature_names=["volatility", "rates"], fit_range=["2020-01-02", "2022-12-30"],
        vol_order=[0, 1, 2, 3], means=np.zeros(2), scales=np.ones(2),
        covariance_type="full", bic_by_k={4: 100.0}, state_labels=lambda: ["a", "b", "c", "d"],
        model=SimpleNamespace(startprob_=np.ones(4) / 4, transmat_=np.ones((4, 4)) / 4,
                              means_=np.zeros((4, 2)), covars_=np.stack([np.eye(2)] * 4)),
    )
    def spec(year):
        return SimpleNamespace(date=date(year, 1, 2), close=np.full(60, 4000.),
                               market=np.zeros((60, len(module.MARKET_FEATURES))),
                               context=np.concatenate([np.zeros(3), np.ones(4) / 4]), spread_pips=3.)
    return SimpleNamespace(
        scaler_mean=np.zeros(len(module.MARKET_FEATURES)), scaler_scale=np.ones(len(module.MARKET_FEATURES)),
        macro_scaler_mean=np.zeros(3), macro_scaler_scale=np.ones(3), regime_model=model,
        development=[spec(2022)], selection=[spec(2023)], holdout=[spec(2024)], dropped={},
    )


def _publish(data, path):
    return module.export_bundle(data, path, source_contract=module._current_contract(),
                                source_cache_sha256="a" * 64)


def _manifest_sha(manifest):
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_roundtrip_independent_numeric_artifacts_and_no_overwrite(bundle_data, tmp_path):
    output = tmp_path / "bundle"
    manifest = _publish(bundle_data, output)
    loaded = module.load_bundle(output, expected_manifest_sha256=_manifest_sha(manifest))
    identity = loaded.manifest["dataset_identity"]
    assert loaded.scaler["dataset_identity"] == loaded.regime["dataset_identity"] == identity
    assert loaded.data.regime_model["k"] == 4
    assert loaded.manifest["role"] == "frozen_research_bundle_not_live_promotion"
    with pytest.raises(FileExistsError):
        _publish(bundle_data, output)


def test_partial_publication_never_has_commit_marker(bundle_data, tmp_path, monkeypatch):
    original = module.immutable_write
    def fail_second(path, raw):
        if path.name == "regime.json":
            raise OSError("simulated interruption")
        original(path, raw)
    monkeypatch.setattr(module, "immutable_write", fail_second)
    output = tmp_path / "failed_bundle"
    with pytest.raises(OSError):
        _publish(bundle_data, output)
    assert (output / "scaler.json").exists()
    assert not (output / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        module.load_bundle(output, expected_manifest_sha256="a" * 64)


def test_input_change_mid_export_preserves_uncommitted_components(bundle_data, tmp_path, monkeypatch):
    initial = module._current_contract()
    calls = []
    def contract():
        calls.append(True)
        return initial if len(calls) == 1 else {"changed": True}
    monkeypatch.setattr(module, "_current_contract", contract)
    output = tmp_path / "raced_bundle"
    with pytest.raises(ValueError, match="inputs changed during"):
        module.export_bundle(bundle_data, output, source_contract=initial, source_cache_sha256="a" * 64)
    assert (output / "portable.pkl").exists()
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("target", ["scaler.json", "regime.json", "portable.pkl"])
def test_component_tampering_fails_before_deserialization(bundle_data, tmp_path, monkeypatch, target):
    output = tmp_path / "bundle"
    manifest = _publish(bundle_data, output)
    (output / target).write_bytes(b"tampered")
    monkeypatch.setattr(pickle, "loads", lambda raw: pytest.fail("must hash check before deserialization"))
    with pytest.raises(ValueError, match="artifact hash"):
        module.load_bundle(output, expected_manifest_sha256=_manifest_sha(manifest))


def test_manifest_sha_is_required_before_deserialization(bundle_data, tmp_path, monkeypatch):
    output = tmp_path / "bundle"
    _publish(bundle_data, output)
    monkeypatch.setattr(pickle, "loads", lambda raw: pytest.fail("must check manifest first"))
    with pytest.raises(ValueError, match="manifest hash"):
        module.load_bundle(output, expected_manifest_sha256="b" * 64)


def test_current_input_mismatch_requires_explicit_historical_mode(bundle_data, tmp_path, monkeypatch):
    output = tmp_path / "bundle"
    manifest = _publish(bundle_data, output)
    digest = _manifest_sha(manifest)
    monkeypatch.setattr(module, "_current_contract", lambda: {"new": "inputs"})
    with pytest.raises(ValueError, match="current input"):
        module.load_bundle(output, expected_manifest_sha256=digest)
    assert len(module.load_bundle(output, expected_manifest_sha256=digest,
                                   require_current_inputs=False).data.selection) == 1


@pytest.mark.parametrize("path", [".env", ".env.test", "secrets/cache.pkl", "credentials.json", "key.pem"])
def test_sensitive_paths_refused_before_reads(tmp_path, monkeypatch, path):
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("must not read sensitive paths"))
    with pytest.raises(ValueError, match="sensitive"):
        module.load_bundle(tmp_path / path, expected_manifest_sha256="a" * 64)


def test_invalid_numeric_scaler_never_creates_output(bundle_data, tmp_path):
    bundle_data.scaler_scale[0] = 0
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="invalid scale"):
        _publish(bundle_data, output)
    assert not output.exists()


def test_cli_cache_sha_checked_before_unpickle(tmp_path, monkeypatch):
    path = tmp_path / "cache.pkl"
    path.write_bytes(b"not trusted")
    monkeypatch.setattr(module.sys, "argv", ["export", "--cache", str(path),
                                            "--expected-cache-sha256", "a" * 64,
                                            "--output", str(tmp_path / "output")])
    monkeypatch.setattr(pickle, "loads", lambda raw: pytest.fail("bad SHA cannot reach pickle"))
    with pytest.raises(ValueError, match="cache SHA"):
        module.main()


def test_manifest_references_only_generated_component_names(bundle_data, tmp_path):
    manifest = _publish(bundle_data, tmp_path / "bundle")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert set(payload["artifacts"]) == {"scaler.json", "regime.json", "portable.pkl"}


@pytest.mark.parametrize("mutation", ["empty", "cross_block", "duplicate", "wrong_order"])
def test_temporal_partition_mutations_rejected(bundle_data, tmp_path, mutation):
    if mutation == "empty":
        bundle_data.development = []
    elif mutation == "cross_block":
        bundle_data.selection = bundle_data.development
    elif mutation == "duplicate":
        bundle_data.development *= 2
    else:
        bundle_data.development, bundle_data.holdout = bundle_data.holdout, bundle_data.development
    with pytest.raises(ValueError, match="research blocks"):
        _publish(bundle_data, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("covariance", [np.array([[1., .2], [.1, 1.]]),
                                        np.array([[1., 0.], [0., -1.]]),
                                        np.array([[1., 0.], [0., 0.]])])
def test_invalid_covariance_is_not_silently_repaired(bundle_data, tmp_path, covariance):
    bundle_data.regime_model.model.covars_[0] = covariance
    with pytest.raises(ValueError, match="covariance"):
        _publish(bundle_data, tmp_path / "invalid")


def test_current_exporter_hash_is_required(bundle_data, tmp_path, monkeypatch):
    output = tmp_path / "bundle"
    manifest = _publish(bundle_data, output)
    digest = _manifest_sha(manifest)
    original = Path.read_bytes
    source_path = Path(module.__file__).resolve()
    monkeypatch.setattr(Path, "read_bytes", lambda path: (
        b"changed exporter code" if path.resolve() == source_path else original(path)))
    with pytest.raises(ValueError, match="exporter code"):
        module.load_bundle(output, expected_manifest_sha256=digest)
    assert len(module.load_bundle(output, expected_manifest_sha256=digest,
                                   require_current_inputs=False).data.selection) == 1


def test_k_five_cannot_be_exported_into_four_regime_slots(bundle_data, tmp_path):
    bundle_data.regime_model.k = 5
    with pytest.raises(ValueError, match="exceeds 4 observation slots"):
        _publish(bundle_data, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("posterior", [[.2, .2, .2, .2], [0., 0., 0., 0.],
                                       [-.1, .3, .4, .4], [1.1, -.1, 0., 0.]])
def test_posterior_mass_or_domain_failure_blocks_publication(bundle_data, tmp_path, posterior):
    bundle_data.selection[0].context[-4:] = posterior
    with pytest.raises(ValueError, match="posterior is truncated/invalid"):
        _publish(bundle_data, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_three_regimes_with_zero_padding_are_valid(bundle_data, tmp_path):
    model = bundle_data.regime_model
    model.k = 3
    model.vol_order = [0, 1, 2]
    model.state_labels = lambda: ["a", "b", "c"]
    model.model.startprob_ = np.ones(3) / 3
    model.model.transmat_ = np.ones((3, 3)) / 3
    model.model.means_ = np.zeros((3, 2))
    model.model.covars_ = np.stack([np.eye(2)] * 3)
    for block in (bundle_data.development, bundle_data.selection, bundle_data.holdout):
        block[0].context[-4:] = [1 / 3, 1 / 3, 1 / 3, 0]
    manifest = _publish(bundle_data, tmp_path / "valid")
    assert module.load_bundle(tmp_path / "valid", expected_manifest_sha256=_manifest_sha(manifest)).regime["k"] == 3


def test_loader_rejects_more_states_than_current_contract(bundle_data, tmp_path, monkeypatch):
    # Produce an internally consistent wider test-only bundle, then restore the
    # actual four-slot contract. Historical mode cannot waive numeric integrity.
    monkeypatch.setattr(module, "N_REGIMES", 5)
    model = bundle_data.regime_model
    model.k = 5
    model.vol_order = list(range(5))
    model.state_labels = lambda: ["a", "b", "c", "d", "e"]
    model.model.startprob_ = np.ones(5) / 5
    model.model.transmat_ = np.ones((5, 5)) / 5
    model.model.means_ = np.zeros((5, 2))
    model.model.covars_ = np.stack([np.eye(2)] * 5)
    for block in (bundle_data.development, bundle_data.selection, bundle_data.holdout):
        block[0].context = np.concatenate([np.zeros(3), np.ones(5) / 5])
    manifest = _publish(bundle_data, tmp_path / "wider")
    monkeypatch.setattr(module, "N_REGIMES", 4)
    with pytest.raises(ValueError, match="exceeds 4 observation slots"):
        module.load_bundle(tmp_path / "wider", expected_manifest_sha256=_manifest_sha(manifest),
                           require_current_inputs=False)
