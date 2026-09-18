"""Independent five-slot dataset lane, using observed releases and no global cache.

An export is a reproducible research artifact, NOT permission to train/trade.
Historical source identity does not certify first-seen/publication timestamps.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.research.dataset import MACRO_FEATURES, MARKET_FEATURES
from src.research.evaluation_mask import _mask_from_frame
from src.research.features import build_market_features
from src.research.live_spec import (
    MIN_CONTEXT_SESSIONS,
    FrozenScaler,
    PartialLiveSpec,
    _cot_frame,
    _validated_prefix,
)
from src.research.macro_evidence import canonical_json, immutable_write
from src.research.observation_contract import REGIME5_VERSION, observation_contract
from src.research.publication_join_v2 import join_observed_releases
from src.research.regime5_hmm import build_regime_observations, fit_frozen
from src.research.regime_hmm import _levels_for_k
from src.research.regime_portable import PortableRegimeModel
from src.research.session_gym import SessionSpec

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = observation_contract(REGIME5_VERSION)
SERIES = {
    "INVESTING_DXY": ("index_points", 2),
    "FRED_DCOILBRENTEU": ("usd_per_barrel", 2),
    "FRED_DGS2": ("percent", 1),
    "BANREP_IBR": ("percent", 1),
}
CODE_PATHS = (
    "src/research/dataset.py",
    "src/research/cost_contract.py",
    "src/research/regime5_hmm.py",
    "src/research/regime5_bundle.py",
    "src/research/observation_contract.py",
    "src/research/features.py",
    "src/research/evaluation_mask.py",
    "src/research/live_spec.py",
    "src/research/regime_hmm.py",
    "src/research/regime_portable.py",
    "src/research/publication_join_v2.py",
    "src/research/session_gym.py",
    "src/research/session_env.py",
    "src/research/cost_model.py",
    "config/research/cost_contract.yaml",
    "config/trading_calendar.json",
)


def safe_path(path: Path) -> Path:
    for candidate in (Path(path), Path(path).resolve()):
        for part in candidate.parts:
            name = part.lower()
            if (
                name.startswith((".env", "credentials", "service-account"))
                or name == "secrets"
                or name.endswith((".pem", ".key"))
            ):
                raise ValueError("secret path is not a research artifact")
    return Path(path).resolve()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out

    def invalid(value):
        raise ValueError(f"non-finite JSON: {value}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def code_identity() -> dict:
    return {name: digest((ROOT / name).read_bytes()) for name in CODE_PATHS}


def publication_features(
    sessions, releases: pd.DataFrame, policies: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """T-1 levels AND max(publication, first_seen)<08:00, same operands for PPO/HMM.

    A hash in a release row is supplied provenance, not authentication. Preserve
    the long-form join so absence/staleness and both return operands are auditable.
    """
    if set(policies) != set(SERIES):
        raise ValueError("exact four macro identities required; no DXY/Broad fallback")
    for name, (unit, count) in SERIES.items():
        if (
            policies[name].get("unit") != unit
            or policies[name].get("minimum_observations") != count
        ):
            raise ValueError("macro identity/unit/transform mismatch")
    days = [pd.Timestamp(d).date() for d in sessions]
    if days != sorted(set(days)):
        raise ValueError("unique ordered session dates required")
    decisions = pd.DataFrame(
        {
            "decision_id": [str(d) for d in days],
            "cutoff_utc": [
                pd.Timestamp(f"{d} 08:00", tz="America/Bogota").tz_convert("UTC") for d in days
            ],
        }
    )
    joined = join_observed_releases(decisions, releases, policies, strict_prior_period=True)
    out = pd.DataFrame(index=pd.to_datetime(days), columns=MACRO_FEATURES, dtype=float)
    for d in days:
        rows = joined[joined.decision_id == str(d)].set_index("series")
        if not (rows.status == "AVAILABLE").all():
            continue
        if any(pd.Timestamp(p).date() >= d for p in rows.period_end):
            continue
        out.loc[pd.Timestamp(d)] = [
            rows.loc["FRED_DCOILBRENTEU", "log_return_previous"],
            rows.loc["INVESTING_DXY", "log_return_previous"],
            (rows.loc["BANREP_IBR", "level"] - rows.loc["FRED_DGS2", "level"]) / 100,
        ]
    return out, joined


def _array(value, shape, *, positive=False):
    if np.ma.isMaskedArray(value) and np.ma.getmaskarray(value).any():
        raise ValueError("masked numeric artifact")
    if any(isinstance(v, bool | np.bool_) for v in np.asarray(value, dtype=object).flat):
        raise ValueError("boolean numeric artifact")
    arr = np.asarray(value)
    if (
        arr.shape != shape
        or arr.dtype.kind not in "fiu"
        or not np.isfinite(arr).all()
        or (positive and (arr <= 0).any())
    ):
        raise ValueError(f"invalid numeric artifact, expected shape {shape}")
    return arr.astype(float)


@dataclass(frozen=True)
class FrozenRegime5:
    scaler: FrozenScaler
    regime: PortableRegimeModel
    fit_evidence: dict

    @classmethod
    def from_payload(cls, payload: dict) -> FrozenRegime5:
        if payload.get("observation_contract") != CONTRACT.to_dict():
            raise ValueError("frozen observation contract mismatch")
        s, r = payload["scaler"], payload["regime"]
        k, n = r["k"], len(r["feature_names"])
        if type(k) is not int or k not in (2, 3, 4, 5):
            raise ValueError("invalid frozen K")
        if tuple(s["features"]) != tuple(MARKET_FEATURES):
            raise ValueError("scaler feature order mismatch")
        scaler = FrozenScaler(
            _array(s["mean"], (25,)),
            _array(s["scale"], (25,), positive=True),
            tuple(s["features"]),
            _array(s["macro_mean"], (3,)),
            _array(s["macro_scale"], (3,), positive=True),
        )
        arrays = {
            key: _array(r[key], shape, positive=key == "std_scales")
            for key, shape in {
                "startprob": (k,),
                "transmat": (k, k),
                "means": (k, n),
                "covars": (k, n, n),
                "std_means": (n,),
                "std_scales": (n,),
            }.items()
        }
        for key in ("startprob", "transmat"):
            if (arrays[key] < 0).any() or not np.allclose(
                arrays[key].sum(axis=-1), 1, atol=1e-10, rtol=0
            ):
                raise ValueError("invalid frozen probabilities")
        if (
            any(type(i) is not int for i in r["vol_order"])
            or sorted(r["vol_order"]) != list(range(k))
            or len(r["labels"]) != k
        ):
            raise ValueError("invalid frozen state order")
        for cov in arrays["covars"]:
            if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
                raise ValueError("asymmetric frozen covariance")
            np.linalg.cholesky(cov)  # reject, no silent jitter/repair
        from src.research.regime_hmm import FEATURE_NAMES

        if tuple(r["feature_names"]) != tuple(FEATURE_NAMES):
            raise ValueError("HMM input feature order mismatch")
        model = PortableRegimeModel(
            k=k,
            **arrays,
            vol_order=tuple(r["vol_order"]),
            labels=tuple(r["labels"]),
            feature_names=tuple(r["feature_names"]),
            fit_range=tuple(r["fit_range"]),
        )
        return cls(scaler, model, payload["fit_evidence"])

    def to_payload(self) -> dict:
        r, s = self.regime, self.scaler
        return {
            "observation_contract": CONTRACT.to_dict(),
            "scaler": {
                "features": list(s.features),
                **{
                    k: getattr(s, k).tolist()
                    for k in ("mean", "scale", "macro_mean", "macro_scale")
                },
            },
            "regime": {
                "k": r.k,
                "vol_order": list(r.vol_order),
                "labels": list(r.labels),
                "fit_range": list(r.fit_range),
                "feature_names": list(r.feature_names),
                **{
                    k: getattr(r, k).tolist()
                    for k in ("startprob", "transmat", "means", "covars", "std_means", "std_scales")
                },
            },
            "fit_evidence": self.fit_evidence,
        }

    def prefix(
        self,
        session_date,
        bars: pd.DataFrame,
        *,
        history: pd.DataFrame,
        macro_features: pd.DataFrame,
    ) -> PartialLiveSpec:
        target = pd.Timestamp(session_date).date()
        frame = _validated_prefix(target, bars)
        old = _cot_frame(history)
        old = old[old.time.dt.date < target]
        if "symbol" not in old:
            old = old.assign(symbol="USDCOP")
        valid = set(_mask_from_frame(old, source="regime5_history").valid)
        hist = pd.concat([old, frame], ignore_index=True)
        valid.add(target)
        features = build_market_features(hist, valid_sessions=valid)
        day = features[features["_session"] == target]
        macro = macro_features.loc[pd.Timestamp(target), MACRO_FEATURES].to_numpy()
        _array(macro, (3,))
        obs = build_regime_observations(hist, valid, macro_features=macro_features).dropna()
        prior = obs[obs.index.date < target]
        if len(prior) < MIN_CONTEXT_SESSIONS:
            raise ValueError("insufficient causal HMM context")
        probs = CONTRACT.posterior(self.regime.filtered_posterior(prior.to_numpy()), self.regime.k)
        spread = float(np.dot(probs[: self.regime.k], _levels_for_k(self.regime.k)))
        market = np.clip(self.scaler.transform(day[MARKET_FEATURES].to_numpy()), -5, 5).astype(
            np.float32
        )
        context = np.r_[
            np.clip((macro - self.scaler.macro_mean) / self.scaler.macro_scale, -5, 5), probs
        ]
        context = context.astype(np.float32)
        CONTRACT.validate_arrays(market, context, bars=len(frame))
        return PartialLiveSpec(
            target,
            len(frame),
            market,
            context,
            frame.close.to_numpy(),
            spread,
            observation_version=REGIME5_VERSION,
        )

    def session(self, session_date, bars, *, history, macro_features) -> SessionSpec:
        if len(bars) != 60:
            raise ValueError("a training session requires 60 bars")
        p = self.prefix(session_date, bars, history=history, macro_features=macro_features)
        return SessionSpec(
            p.date,
            p.closes,
            p.market,
            p.context,
            p.spread_pips,
            observation_version=REGIME5_VERSION,
        )


def fit_development(
    m5: pd.DataFrame,
    macro_features: pd.DataFrame,
    *,
    development_start: date,
    development_end: date,
) -> FrozenRegime5:
    """Fit only the named development interval; caller must pass its admission gate.

    No holdout selection, no cache/global writes, no PPO training. Macro failures
    remain failures; this function does not invent a historical publication ledger.
    """
    if development_start > development_end:
        raise ValueError("invalid development interval")
    frame = _cot_frame(m5)
    frame = frame[
        (frame.time.dt.date >= development_start) & (frame.time.dt.date <= development_end)
    ]
    mask = _mask_from_frame(frame, source="regime5_development")
    valid = set(mask.train_valid)
    macro = macro_features.reindex(pd.to_datetime(sorted(valid)))
    if not len(macro) or not np.isfinite(macro[MACRO_FEATURES].to_numpy()).all():
        raise ValueError("development macro availability incomplete; no fit")
    obs = build_regime_observations(frame, valid, macro_features=macro)
    fitted = fit_frozen(obs)
    features = build_market_features(frame, valid_sessions=valid)[MARKET_FEATURES].to_numpy()
    mean, scale = features.mean(axis=0), features.std(axis=0)
    scale[scale == 0] = 1
    mm, ms = macro[MACRO_FEATURES].mean().to_numpy(), macro[MACRO_FEATURES].std(ddof=0).to_numpy()
    ms[ms == 0] = 1
    covars = fitted.model.covars_
    if covars.ndim == 2:
        covars = np.stack([np.diag(c) for c in covars])
    portable = PortableRegimeModel(
        fitted.k,
        fitted.model.startprob_,
        fitted.model.transmat_,
        fitted.model.means_,
        covars,
        fitted.means,
        fitted.scales,
        fitted.vol_order,
        tuple(fitted.state_labels()),
        fitted.feature_names,
        fitted.fit_range,
    )
    result = FrozenRegime5(
        FrozenScaler(mean, scale, tuple(MARKET_FEATURES), mm, ms),
        portable,
        {
            "role": "development_only",
            "interval": [str(development_start), str(development_end)],
            "k_candidates": [2, 3, 4, 5],
            "k_default": 3,
            "bic_hysteresis": 10,
            "bic_by_k": {str(k): v for k, v in fitted.bic_by_k.items()},
            "selected_k": fitted.k,
            "selected_covariance_type": fitted.model.covariance_type,
            "selected_converged": bool(fitted.model.monitor_.converged),
            "selected_iterations": int(fitted.model.monitor_.iter),
            "candidates": list(fitted.candidate_evidence),
            "mask": mask.to_dict(),
            "market_rows": len(features),
            "macro_rows": len(macro),
        },
    )
    return FrozenRegime5.from_payload(result.to_payload())


def export_bundle(
    output: Path, frozen: FrozenRegime5, blocks: dict[str, list[SessionSpec]], *, inputs: dict
) -> Path:
    """Exclusive self-contained export, numeric NPZ without executable pickle."""
    output = safe_path(output)
    if set(blocks) != {"development", "selection", "holdout"}:
        raise ValueError("three explicitly classified blocks required")
    arrays, dates, seen, last = {}, {}, set(), None
    for name in ("development", "selection", "holdout"):
        specs = blocks[name]
        days = [str(s.date) for s in specs]
        if (
            not days
            or days != sorted(set(days))
            or seen.intersection(days)
            or (last and days[0] <= last)
        ):
            raise ValueError("nonempty ordered disjoint blocks required")
        dates[name], last = days, days[-1]
        seen.update(days)
        for s in specs:
            if s.observation_version != REGIME5_VERSION:
                raise ValueError("legacy session cannot enter the new bundle")
            CONTRACT.validate_arrays(s.market, s.context, bars=60)
            _array(s.close, (60,), positive=True)
            posterior = s.context[3:]
            if (
                not np.allclose(posterior.sum(), 1, atol=1e-6, rtol=0)
                or (posterior < 0).any()
                or (posterior[frozen.regime.k :] != 0).any()
            ):
                raise ValueError("invalid/truncated session posterior")
            if s.cost_parameters is not None or not np.isfinite(s.spread_pips) or s.spread_pips < 0:
                raise ValueError("synthetic/invalid costs cannot enter a market bundle")
        for field in ("close", "market", "context", "spread_pips"):
            arrays[f"{name}_{field}"] = np.asarray([getattr(s, field) for s in specs])
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = frozen.to_payload()
    FrozenRegime5.from_payload(payload)
    components = {
        "observations.npz": stream.getvalue(),
        "frozen.json": canonical_json(payload),
        "schema.json": canonical_json(CONTRACT.to_dict()),
    }
    manifest = {
        "version": REGIME5_VERSION,
        "role": "research_only_not_training_authorization",
        "code_identity": code_identity(),
        "inputs": inputs,
        "dates": dates,
        "artifacts": {n: digest(raw) for n, raw in components.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in components.items():
        immutable_write(output / name, raw)
    immutable_write(output / "manifest.json", canonical_json(manifest))
    return output / "manifest.json"


def load_bundle(root: Path, *, expected_sha256: str) -> tuple[FrozenRegime5, dict, dict]:
    root = safe_path(root)
    manifest_path = safe_path(root / "manifest.json")
    if manifest_path.parent != root:
        raise ValueError("bundle manifest path escape")
    raw = manifest_path.read_bytes()
    if digest(raw) != expected_sha256:
        raise ValueError("bundle manifest SHA mismatch")
    manifest = strict_json(raw)
    if manifest["version"] != REGIME5_VERSION or manifest["code_identity"] != code_identity():
        raise ValueError("bundle version/code identity mismatch")
    if set(manifest["artifacts"]) != {"observations.npz", "frozen.json", "schema.json"}:
        raise ValueError("bundle component set mismatch")
    raw_parts = {}
    for name, sha in manifest["artifacts"].items():
        path = safe_path(root / name)
        if path.parent != root:
            raise ValueError("bundle path escape")
        raw_parts[name] = path.read_bytes()
        if digest(raw_parts[name]) != sha:
            raise ValueError("bundle component SHA mismatch")
    if strict_json(raw_parts["schema.json"]) != CONTRACT.to_dict():
        raise ValueError("bundle observation schema mismatch")
    frozen = FrozenRegime5.from_payload(strict_json(raw_parts["frozen.json"]))
    if set(manifest["dates"]) != {"development", "selection", "holdout"}:
        raise ValueError("bundle block set mismatch")
    blocks = {}
    with np.load(io.BytesIO(raw_parts["observations.npz"]), allow_pickle=False) as data:
        expected = {
            f"{b}_{f}"
            for b in ("development", "selection", "holdout")
            for f in ("close", "market", "context", "spread_pips")
        }
        if set(data.files) != expected:
            raise ValueError("bundle array set mismatch")
        last = None
        for block in ("development", "selection", "holdout"):
            days = manifest["dates"][block]
            if not days or days != sorted(set(days)) or (last and days[0] <= last):
                raise ValueError("bundle dates must be strictly ordered and disjoint")
            last = days[-1]
            if any(
                len(data[f"{block}_{f}"]) != len(days)
                for f in ("close", "market", "context", "spread_pips")
            ):
                raise ValueError("bundle date/array length mismatch")
            blocks[block] = [
                SessionSpec(
                    date.fromisoformat(day),
                    data[f"{block}_close"][i],
                    data[f"{block}_market"][i],
                    data[f"{block}_context"][i],
                    float(data[f"{block}_spread_pips"][i]),
                    observation_version=REGIME5_VERSION,
                )
                for i, day in enumerate(days)
            ]
            for spec in blocks[block]:
                _array(spec.close, (60,), positive=True)
                p = spec.context[3:]
                if (
                    not np.isclose(p.sum(), 1, atol=1e-6, rtol=0)
                    or (p < 0).any()
                    or (p[frozen.regime.k :] != 0).any()
                    or not np.isfinite(spec.spread_pips)
                    or spec.spread_pips < 0
                ):
                    raise ValueError("invalid bundle posterior/price/cost")
    return frozen, blocks, manifest


def load_bound_checkpoint(
    path: Path,
    *,
    metadata_path: Path,
    expected_metadata_sha256: str,
    bundle_path: Path,
    expected_bundle_sha256: str,
):
    """Load only a new checkpoint bound to the exact verified bundle/schema.

    This validates inference compatibility, not scientific admission or fills.
    No old checkpoint is padded, truncated, rewritten or automatically migrated.
    """
    path, metadata_path = safe_path(path), safe_path(metadata_path)
    raw = metadata_path.read_bytes()
    if digest(raw) != expected_metadata_sha256:
        raise ValueError("checkpoint metadata SHA mismatch")
    metadata = strict_json(raw)
    if (
        metadata.get("observation_version") != REGIME5_VERSION
        or metadata.get("observation_sha256") != CONTRACT.sha256
        or metadata.get("dataset_manifest_sha256") != expected_bundle_sha256
    ):
        raise ValueError("checkpoint observation/dataset binding mismatch")
    load_bundle(bundle_path, expected_sha256=expected_bundle_sha256)
    model_raw = path.read_bytes()
    if digest(model_raw) != metadata.get("checkpoint_sha256"):
        raise ValueError("checkpoint bytes differ from the frozen binding")
    from stable_baselines3 import PPO

    # Load the same checked bytes (not a second path read susceptible to replacement).
    model = PPO.load(io.BytesIO(model_raw), device="cpu")
    if model.observation_space.shape != (38,) or model.action_space.n != 5:
        raise ValueError("checkpoint spaces differ from the frozen contract")
    model.research_observation_sha256 = CONTRACT.sha256
    return model
