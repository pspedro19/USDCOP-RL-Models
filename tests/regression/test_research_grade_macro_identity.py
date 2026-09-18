"""Dataset identity v3 covers joins/configuration; historical replay is explicit."""
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def identity_environment(monkeypatch):
    from src.research import dataset, evaluation_mask

    monkeypatch.setattr(evaluation_mask, "build_mask", lambda: SimpleNamespace(sha256="mask"))
    monkeypatch.setattr(dataset, "_file_digest", lambda path: hashlib.sha256(str(path).encode()).hexdigest())
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: (
        json.dumps({"dataset_identity": "historic", "mean": [1.0]})
        if p.name in {"feature_scaler_frozen.json", "regime_hmm_frozen.json"}
        else original(p, *a, **k)))
    return dataset


@pytest.mark.parametrize("target", [
    "src/research/macro_asof.py", "config/research/macro_availability.yaml",
    "config/trading_calendar.json", "config/research/cost_contract.yaml",
    "config/research/source_lineage.yaml", "src/research/live_spec.py",
    "src/research/regime_portable.py",
])
def test_versioned_identity_invalidates_on_temporal_or_cost_change(identity_environment, monkeypatch, target):
    module = identity_environment
    old = module.dataset_identity()
    baseline = module._file_digest
    monkeypatch.setattr(module, "_file_digest", lambda path: (
        "changed" if path == module.REPO / target else baseline(path)))
    assert module.dataset_identity() != old
    assert module.dataset_identity_manifest()["identity_version"] == 4


def test_frozen_artifact_identity_key_is_stable_for_relative_and_absolute_paths(
    identity_environment, monkeypatch, tmp_path
):
    module = identity_environment
    scaler = module.REPO / "config/research/feature_scaler_frozen.json"
    regime = module.REPO / "config/research/regime_hmm_frozen.json"
    relative_scaler = Path("config/research/feature_scaler_frozen.json")
    relative_regime = Path("config/research/regime_hmm_frozen.json")
    monkeypatch.setattr(module, "frozen_artifact_paths", lambda: (scaler, regime))
    absolute = module.dataset_identity_manifest()
    monkeypatch.setattr(module, "frozen_artifact_paths", lambda: (relative_scaler, relative_regime))
    relative = module.dataset_identity_manifest()
    assert absolute["frozen_content_excluding_dataset_identity"] == relative[
        "frozen_content_excluding_dataset_identity"
    ]


def test_frozen_artifact_backreference_does_not_create_hash_cycle(identity_environment, monkeypatch):
    module = identity_environment
    old = module.dataset_identity()
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: (
        json.dumps({"dataset_identity": "new-backreference", "mean": [1.0]})
        if p.name in {"feature_scaler_frozen.json", "regime_hmm_frozen.json"}
        else original(p, *a, **k)))
    assert module.dataset_identity() == old
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **k: (
        json.dumps({"dataset_identity": "new-backreference", "mean": [2.0]})
        if p.name in {"feature_scaler_frozen.json", "regime_hmm_frozen.json"}
        else original(p, *a, **k)))
    assert module.dataset_identity() != old


def test_historical_loader_requires_exact_bytes_and_does_not_rebuild(tmp_path, monkeypatch):
    from src.research import dataset

    blob = {"development": [], "selection": [], "holdout": [],
            "scaler_mean": np.array([1]), "scaler_scale": np.array([2]),
            "regime_meta": {"k": 4}, "dropped": {}}
    raw = pickle.dumps(blob)
    path = tmp_path / "archived.pkl"
    path.write_bytes(raw)
    monkeypatch.setattr(dataset, "dataset_identity", lambda: pytest.fail("historic cannot rebuild identity"))
    loaded = dataset.load_historical_portable(path, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert loaded.regime_model == {"k": 4}
    monkeypatch.setattr(pickle, "loads", lambda raw: pytest.fail("hash must be checked before unpickle"))
    with pytest.raises(ValueError, match="file hash mismatch"):
        dataset.load_historical_portable(path, expected_sha256="0" * 64)


def test_cache_invalidates_when_availability_changes(identity_environment, tmp_path, monkeypatch):
    module = identity_environment
    seed, macro, part = [tmp_path / name for name in ("seed", "macro", "part")]
    for path in (seed, macro, part):
        path.write_bytes(b"fixture")
    monkeypatch.setattr(module, "SEED_M5", seed)
    monkeypatch.setattr(module, "PARTITION", part)
    monkeypatch.setitem(module.attach_macro_features.__globals__, "MACRO_CLEAN", macro)
    monkeypatch.setattr(module, "CACHE", tmp_path / "cache.pkl")
    builds = []
    def fake_build(**kwargs):
        builds.append(True)
        return {"build_number": len(builds)}
    monkeypatch.setattr(module, "build_research_data", fake_build)
    first = module.load_or_build(verbose=False)
    assert module.load_or_build(verbose=False) == first
    baseline = module._file_digest
    monkeypatch.setattr(module, "_file_digest", lambda path: (
        "changed-availability" if path.name == "macro_availability.yaml" else baseline(path)))
    assert module.load_or_build(verbose=False) != first
    assert len(builds) == 2


def test_save_rejects_stale_frozen_scaler_before_writing(identity_environment, tmp_path):
    module = identity_environment
    data = SimpleNamespace(regime_model=object(), scaler_mean=np.array([999]),
                           scaler_scale=np.array([1]), macro_scaler_mean=None,
                           macro_scaler_scale=None)
    output = tmp_path / "must_not_exist.pkl"
    with pytest.raises(ValueError, match="frozen scaler mean"):
        module.save_portable(data, output)
    assert not output.exists()


def test_historical_loader_refuses_sensitive_path_before_read(tmp_path, monkeypatch):
    from src.research import dataset

    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("sensitive path must not be read"))
    with pytest.raises(ValueError, match="sensitive"):
        dataset.load_historical_portable(tmp_path / ".env", expected_sha256="0" * 64)


def test_build_refuses_unrepresentable_hmm_before_context_or_features(monkeypatch):
    from src.research import dataset

    day = pd.Timestamp("2022-01-03").date()
    monkeypatch.setattr(dataset, "load_partition", lambda: {"fixture": True})
    monkeypatch.setattr(dataset, "build_mask", lambda: SimpleNamespace(valid=[day], train_valid=[day]))
    monkeypatch.setattr(dataset, "_block_dates", lambda *args: [day])
    observations = pd.DataFrame({"feature": [1.]}, index=pd.to_datetime([day]))
    monkeypatch.setattr(dataset, "build_regime_observations", lambda *args, **kwargs: observations)
    monkeypatch.setattr(dataset, "fit_frozen", lambda *args, **kwargs: SimpleNamespace(k=5))
    def forbidden(*args, **kwargs):
        pytest.fail("unrepresentable HMM must fail before context/features generation")
    monkeypatch.setattr(dataset, "spread_series", forbidden)
    monkeypatch.setattr(dataset, "build_market_features", forbidden)
    monkeypatch.setattr(dataset, "attach_macro_features", forbidden)
    with pytest.raises(ValueError, match="new preregistration/schema decision"):
        dataset.build_research_data(m5=pd.DataFrame(), verbose=False)
