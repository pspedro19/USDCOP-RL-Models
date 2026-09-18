"""Synthetic engineering controls; no fitted signal or financial claim."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import regime5_bundle as lane
from src.research.evaluation_mask import _mask_from_frame
from src.research.features import build_market_features
from src.research.live_spec import FrozenScaler
from src.research.observation_contract import REGIME5_VERSION
from src.research.regime5_hmm import build_regime_observations
from src.research.regime_hmm import FEATURE_NAMES
from src.research.regime_portable import PortableRegimeModel
from src.research.session_gym import SessionTradingEnv


def bars(day):
    t = pd.date_range(f"{day} 08:00", periods=60, freq="5min", tz="America/Bogota")
    i = np.arange(60)
    c = 4000 + pd.Timestamp(day).day + i / 5 + 3 * np.sin(i / 3)
    return pd.DataFrame(
        {"time": t, "symbol": "USDCOP", "open": c - 0.2, "high": c + 1, "low": c - 1, "close": c}
    )


def frozen(k):
    n = len(FEATURE_NAMES)
    r = PortableRegimeModel(
        k,
        np.ones(k) / k,
        np.ones((k, k)) / k,
        np.array([np.full(n, m) for m in np.linspace(-0.2, 0.2, k)]),
        np.array([np.eye(n)] * k),
        np.zeros(n),
        np.ones(n),
        tuple(range(k)),
        tuple(str(j) for j in range(k)),
        FEATURE_NAMES,
        ("2020-01-01", "2022-12-31"),
    )
    s = FrozenScaler(
        np.zeros(25), np.ones(25), tuple(lane.MARKET_FEATURES), np.zeros(3), np.ones(3)
    )
    return lane.FrozenRegime5(s, r, {"role": "synthetic_engineering_only"})


@pytest.fixture(scope="module")
def inputs():
    days = pd.bdate_range(end="2023-05-31", periods=100)
    history = pd.concat([bars(d.date()) for d in days], ignore_index=True)
    day = bars("2023-06-01")
    macro = pd.DataFrame(
        0.001, index=days.append(pd.DatetimeIndex(["2023-06-01"])), columns=lane.MACRO_FEATURES
    )
    return SimpleNamespace(history=history, day=day, macro=macro)


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_all_supported_k_through_real_feature_and_hmm_formulas(inputs, k):
    f = lane.FrozenRegime5.from_payload(frozen(k).to_payload())
    p = f.prefix(
        "2023-06-01", inputs.day.iloc[:11], history=inputs.history, macro_features=inputs.macro
    )
    assert p.context.shape == (8,)
    assert p.observation_version == REGIME5_VERSION
    assert p.context[3:].sum() == pytest.approx(1)
    np.testing.assert_array_equal(p.context[3 + k :], 0)


def test_every_prefix_equals_independent_batch_features_and_gym(inputs, monkeypatch):
    # Global readers must never be needed by this explicitly versioned lane.
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("global parquet read"))
    f = frozen(5)
    full = f.session("2023-06-01", inputs.day, history=inputs.history, macro_features=inputs.macro)
    hist = pd.concat([inputs.history, inputs.day], ignore_index=True)
    valid = _mask_from_frame(hist, source="test_batch").valid
    features = build_market_features(hist, valid_sessions=valid)
    expected = features[features["_session"] == pd.Timestamp("2023-06-01").date()][
        lane.MARKET_FEATURES
    ]
    np.testing.assert_array_equal(
        full.market, np.clip(expected.to_numpy(), -5, 5).astype("float32")
    )
    env = SessionTradingEnv([full], shuffle=False)
    obs, _ = env.reset()
    from src.research.llm_forward.arms.ppo_arm import observation_for_closed_bar

    for b in range(59):
        p = f.prefix(
            "2023-06-01",
            inputs.day.iloc[: b + 1],
            history=inputs.history,
            macro_features=inputs.macro,
        )
        np.testing.assert_array_equal(p.market, full.market[: b + 1])
        np.testing.assert_array_equal(p.context, full.context)
        unreal = (
            env._w_prev * (full.close[b] / env._entry_price - 1)
            if env._entry_price and env._w_prev
            else 0
        )
        live = observation_for_closed_bar(
            p,
            previous_weight=env._w_prev,
            bars_in_position=env._bars_in_pos,
            unrealized=unreal,
            drawdown=env._cum - env._peak,
            n_changes=env._n_changes,
        )
        np.testing.assert_array_equal(live, obs)
        obs, _, _, _, _ = env.step(b % 5)


def test_future_market_and_macro_cannot_rewrite_observation(inputs):
    f = frozen(5)
    p = f.prefix(
        "2023-06-01", inputs.day.iloc[:1], history=inputs.history, macro_features=inputs.macro
    )
    future = bars("2023-06-02")
    future[["open", "high", "low", "close"]] *= 100
    history = pd.concat([inputs.history, future], ignore_index=True)
    macro = inputs.macro.copy()
    macro.loc[pd.Timestamp("2023-06-02")] = 999
    actual = f.prefix("2023-06-01", inputs.day.iloc[:1], history=history, macro_features=macro)
    np.testing.assert_array_equal(p.market, actual.market)
    np.testing.assert_array_equal(p.context, actual.context)


def test_hmm_and_agent_share_same_macro_operands(inputs):
    obs = build_regime_observations(inputs.history, macro_features=inputs.macro)
    np.testing.assert_array_equal(obs.dxy_ret, inputs.macro.dxy_ret_prev.reindex(obs.index))
    np.testing.assert_array_equal(obs.brent_ret, inputs.macro.brent_ret_prev.reindex(obs.index))


def test_versioned_market_hmm_formulas_match_historical_code(inputs, monkeypatch):
    from src.research import regime_hmm as historical

    def attach(daily):
        daily["dxy_ret"] = inputs.macro.dxy_ret_prev.reindex(daily.index)
        daily["brent_ret"] = inputs.macro.brent_ret_prev.reindex(daily.index)
        return daily

    monkeypatch.setattr(historical, "_attach_macro", attach)
    previous = historical.build_regime_observations(inputs.history)
    actual = build_regime_observations(inputs.history, macro_features=inputs.macro)
    pd.testing.assert_frame_equal(actual, previous, check_exact=True)


@pytest.mark.parametrize(
    "defect", ["k_bool", "k6", "covariance", "feature_order", "scale_zero", "posterior"]
)
def test_frozen_parameters_reject_defects(defect):
    p = frozen(5).to_payload()
    if defect == "k_bool":
        p["regime"]["k"] = True
    if defect == "k6":
        p["regime"]["k"] = 6
    if defect == "covariance":
        p["regime"]["covars"][0][0][0] = -1
    if defect == "feature_order":
        p["regime"]["feature_names"].reverse()
    if defect == "scale_zero":
        p["scaler"]["scale"][0] = 0
    if defect == "posterior":
        p["regime"]["startprob"][0] = -0.1
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        lane.FrozenRegime5.from_payload(p)


def test_exclusive_bundle_roundtrip_and_tamper_rejection(inputs, tmp_path):
    f = frozen(5)
    s = f.session("2023-06-01", inputs.day, history=inputs.history, macro_features=inputs.macro)
    blocks = {
        name: [replace(s, date=pd.Timestamp(day).date())]
        for name, day in [
            ("development", "2022-06-01"),
            ("selection", "2023-06-01"),
            ("holdout", "2024-06-03"),
        ]
    }
    # Dates/parameters here are engineering fixtures, explicitly NOT fitted evidence.
    path = lane.export_bundle(tmp_path / "bundle", f, blocks, inputs={"role": "synthetic_test"})
    sha = lane.digest(path.read_bytes())
    _, loaded, manifest = lane.load_bundle(path.parent, expected_sha256=sha)
    assert manifest["role"] == "research_only_not_training_authorization"
    for name in blocks:
        np.testing.assert_array_equal(loaded[name][0].context, blocks[name][0].context)
    with pytest.raises(FileExistsError):
        lane.export_bundle(path.parent, f, blocks, inputs={})
    with pytest.raises(ValueError, match="manifest SHA"):
        lane.load_bundle(path.parent, expected_sha256="a" * 64)
    (path.parent / "frozen.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="component SHA"):
        lane.load_bundle(path.parent, expected_sha256=sha)


def test_backbone_zeroes_all_five_preserving_macro_and_version():
    from scripts.analysis.thesis_train_ppo import strip_regimes
    from tests.regression.test_regime5_contract import spec

    s = spec()
    stripped = strip_regimes([s])[0]
    np.testing.assert_array_equal(stripped.context[:3], s.context[:3])
    np.testing.assert_array_equal(stripped.context[3:], np.zeros(5))
    assert stripped.observation_version == REGIME5_VERSION


def test_legacy_trainer_refuses_new_dataset_before_fit(tmp_path):
    from scripts.analysis.thesis_train_ppo import train_one
    from tests.regression.test_regime5_contract import spec

    data = SimpleNamespace(development=[spec()], selection=[spec()])
    with pytest.raises(ValueError, match="training admission"):
        train_one("ppo_regime", 42, data, output_dir=tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()


def release_fixture():
    policies = {
        name: {"unit": unit, "minimum_observations": count, "max_period_age_calendar_days": 7}
        for name, (unit, count) in lane.SERIES.items()
    }
    rows = [
        {
            "series": name,
            "period_end": day,
            "publication_at": f"{day}T18:00:00Z",
            "first_seen_at": f"{day}T18:01:00Z",
            "unit": unit,
            "value": float(100 + i),
            "source_sha256": "a" * 64,
        }
        for name, (unit, _) in lane.SERIES.items()
        for i, day in enumerate(["2023-05-30", "2023-05-31"])
    ]
    return pd.DataFrame(rows), policies


def test_same_day_macro_and_first_seen_cutoff_are_causal():
    releases, policies = release_fixture()
    base, _ = lane.publication_features(["2023-06-01"], releases, policies)
    added = releases[releases.period_end == "2023-05-31"].copy()
    added["period_end"] = "2023-06-01"
    added["publication_at"], added["first_seen_at"] = "2023-06-01T12:00:00Z", "2023-06-01T12:01:00Z"
    added["value"] = 999.0
    actual, _ = lane.publication_features(["2023-06-01"], pd.concat([releases, added]), policies)
    np.testing.assert_array_equal(actual, base)
    releases.loc[releases.period_end == "2023-05-31", "first_seen_at"] = "2023-06-01T13:00:00Z"
    blocked, _ = lane.publication_features(["2023-06-01"], releases, policies)
    assert blocked.isna().all().all()  # exact cutoff is NOT available


def test_stale_previous_operand_is_not_silent_ffill():
    releases, policies = release_fixture()
    releases.loc[releases.period_end == "2023-05-30", "period_end"] = "2023-05-01"
    data, audit = lane.publication_features(["2023-06-01"], releases, policies)
    assert data.isna().all().all()
    assert "STALE_PREVIOUS_OBSERVATION_PERIOD" in set(audit.status)


def test_known_source_is_not_synonymous_with_same_units():
    releases, policies = release_fixture()
    policies["FRED_DTWEXBGS"] = policies.pop("INVESTING_DXY")
    with pytest.raises(ValueError, match="identities"):
        lane.publication_features(["2023-06-01"], releases, policies)


def test_real_sb3_checkpoint_loads_only_with_exact_new_binding(inputs, tmp_path):
    """No learn() calls: exercise real SB3 serialization/inference on engineering inputs."""
    from stable_baselines3 import PPO

    from src.research.observation_contract import require_model_contract

    f = frozen(5)
    s = f.session("2023-06-01", inputs.day, history=inputs.history, macro_features=inputs.macro)
    blocks = {
        name: [replace(s, date=pd.Timestamp(day).date())]
        for name, day in (
            ("development", "2022-06-01"),
            ("selection", "2023-06-01"),
            ("holdout", "2024-06-03"),
        )
    }
    manifest = lane.export_bundle(
        tmp_path / "data", f, blocks, inputs={"role": "engineering_fixture"}
    )
    manifest_sha = lane.digest(manifest.read_bytes())
    env = SessionTradingEnv([s], shuffle=False)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, seed=42, device="cpu")
    checkpoint = tmp_path / "untrained_test_only.zip"
    model.save(checkpoint)
    metadata = tmp_path / "binding.json"
    metadata.write_bytes(
        lane.canonical_json(
            {
                "observation_version": REGIME5_VERSION,
                "observation_sha256": lane.CONTRACT.sha256,
                "checkpoint_sha256": lane.digest(checkpoint.read_bytes()),
                "dataset_manifest_sha256": manifest_sha,
            }
        )
    )
    restored = lane.load_bound_checkpoint(
        checkpoint,
        metadata_path=metadata,
        expected_metadata_sha256=lane.digest(metadata.read_bytes()),
        bundle_path=manifest.parent,
        expected_bundle_sha256=manifest_sha,
    )
    require_model_contract(restored, REGIME5_VERSION)
    obs, _ = env.reset()
    assert model.predict(obs, deterministic=True)[0] == restored.predict(obs, deterministic=True)[0]
    checkpoint.write_bytes(b"not a valid checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        lane.load_bound_checkpoint(
            checkpoint,
            metadata_path=metadata,
            expected_metadata_sha256=lane.digest(metadata.read_bytes()),
            bundle_path=manifest.parent,
            expected_bundle_sha256=manifest_sha,
        )
