"""C043: archive-bound retrospective HMM diagnostics, NEVER a production loader.

The supplied manifest digest is an external trust anchor for a local project
snapshot. Hashes demonstrate preservation, not original collection provenance.
This module neither fits/exports models nor changes current identity guards.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from src.research.regime_portable import PortableRegimeModel, _parameter_digest, _renorm

PARITY_ATOL = 1e-6  # C043 declared before the reproduction was inspected.


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected a complete lowercase SHA256")
    return value


def _safe(key: str) -> str:
    if not isinstance(key, str) or "\\" in key or ":" in key:
        raise ValueError("unsafe archive path")
    path = PurePosixPath(key)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("unsafe archive path")
    for part in path.parts:
        name = part.lower()
        if (name == "secrets" or name == ".env" or name.startswith(".env.")
                or name.endswith((".pem", ".key"))
                or (name.startswith(("credentials", "service-account")) and name.endswith(".json"))):
            raise ValueError("sensitive path refused")
    if str(path) != key:
        raise ValueError("noncanonical archive path")
    return key


def _json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"non-finite JSON constant {value}")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def safe_path(path: Path) -> Path:
    path = Path(path)
    for candidate in (path, path.resolve()):
        for part in candidate.parts:
            if part != candidate.anchor:
                _safe(part)
    return path.resolve()


class HistoricalArchive:
    """Read verified content-addressed bytes; never resolve an original data path."""

    def __init__(self, manifest: Path, expected_sha256: str):
        path = safe_path(manifest)
        raw = path.read_bytes()
        if sha256(raw) != _digest(expected_sha256):
            raise ValueError("manifest hash mismatch")
        data = _json(raw)
        if data.get("contract") != "THESIS-EVIDENCE-SNAPSHOT-1":
            raise ValueError("snapshot contract mismatch")
        self.root = path.parent.resolve()
        self.manifest_sha256 = expected_sha256
        self.entries = {}
        self.used = {}
        for item in data["files"]:
            key = _safe(item["path"])
            digest = _digest(item["sha256"])
            if key in self.entries or item["object"] != "objects/" + digest:
                raise ValueError("duplicate entry or invalid object path")
            if type(item["bytes"]) is not int or item["bytes"] < 0:
                raise ValueError("invalid object size")
            self.entries[key] = item

    def object_path(self, key: str) -> Path:
        item = self.entries[_safe(key)]
        path = safe_path(self.root / item["object"])
        if not path.is_relative_to(self.root):
            raise ValueError("object escapes snapshot")
        return path

    def read(self, key: str) -> bytes:
        item = self.entries[_safe(key)]
        raw = self.object_path(key).read_bytes()
        if len(raw) != item["bytes"] or sha256(raw) != item["sha256"]:
            raise ValueError("object integrity mismatch: " + key)
        self.used[key] = item["sha256"]
        return raw

    def json(self, key: str):
        return _json(self.read(key))

    def bind_current_code(self, repo: Path, keys: tuple[str, ...]):
        for key in keys:
            archived = self.read(key)
            path = safe_path(repo / key)
            if not path.is_relative_to(repo.resolve()) or path.read_bytes() != archived:
                raise ValueError("historical source binding mismatch: " + key)


def historical_model(data: dict, portable: dict, *, model_sha256: str):
    """Validate an archived payload without pretending it has current identity.

    Caller must supply verified archive JSON and the same-byte verified local
    portable. The returned arithmetic object stays inside diagnostic code.
    """
    if data.get("contract") != "CTR-RESEARCH-REGIME-PORTABLE-001":
        raise ValueError("model contract mismatch")
    if _digest(data.get("dataset_identity")) != portable.get("identity"):
        raise ValueError("historical model/portable identity mismatch")
    if _parameter_digest(data) != _digest(data.get("parameter_sha256")):
        raise ValueError("historical parameter digest mismatch")
    if portable.get("regime_artifact_sha256") != _digest(model_sha256):
        raise ValueError("historical model artifact hash mismatch")
    meta = portable["regime_meta"]
    if meta.get("covariance_type") != "full":
        raise ValueError("historical covariance_type must be full")
    for key in ("k", "labels", "feature_names", "fit_range"):
        if json.dumps(meta[key]) != json.dumps(data[key]):
            raise ValueError("historical metadata mismatch: " + key)
    k = data["k"]
    names = data["feature_names"]
    if type(k) is not int or k < 2 or not names or len(names) != len(set(names)):
        raise ValueError("invalid state/feature dimensions")
    if not all(isinstance(x, str) and x for x in names):
        raise ValueError("invalid feature names")
    d = len(names)
    arrays = {}
    shapes = {"startprob": (k,), "transmat": (k, k), "means": (k, d),
              "covars": (k, d, d), "std_means": (d,), "std_scales": (d,)}
    for key, shape in shapes.items():
        arr = np.asarray(data[key], dtype=float)
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError("invalid model array: " + key)
        arrays[key] = arr
    for key in ("startprob", "transmat"):
        arr = arrays[key]
        if (arr < 0).any() or not np.allclose(arr.sum(axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("invalid probability simplex: " + key)
    if (arrays["std_scales"] <= 0).any():
        raise ValueError("invalid standardization scales")
    order = data["vol_order"]
    if any(type(x) is not int for x in order) or sorted(order) != list(range(k)):
        raise ValueError("invalid vol_order permutation")
    if len(data["labels"]) != k or not all(isinstance(x, str) and x for x in data["labels"]):
        raise ValueError("invalid labels")
    fit = data["fit_range"]
    if len(fit) != 2 or date.fromisoformat(fit[0]) > date.fromisoformat(fit[1]):
        raise ValueError("invalid fit_range")
    jitters = []
    for cov in arrays["covars"]:
        if not np.allclose(cov, cov.T, atol=1e-12, rtol=0):
            raise ValueError("asymmetric covariance")
        scale = max(1.0, float(np.max(np.abs(np.diag(cov)))))
        for multiplier in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
            try:
                np.linalg.cholesky(cov + np.eye(d) * scale * multiplier)
                jitters.append(scale * multiplier)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise ValueError("covariance fails bounded Cholesky")
    obj = PortableRegimeModel(k=k, **arrays, vol_order=tuple(order),
                             labels=tuple(data["labels"]), feature_names=tuple(names),
                             fit_range=tuple(fit))
    return obj, {"current_model_eligible": False, "k": k,
                 "portable_to_model_hash_verified": True,
                 "model_artifact_sha256": model_sha256, "covariance_type": "full",
                 "dataset_identity": data["dataset_identity"],
                 "parameter_sha256": data["parameter_sha256"],
                 "identity_manifest_present": portable.get("identity_manifest") is not None,
                 "covariance_diagonal_jitter": jitters,
                 "state_ids_in_volatility_order": [f"state_{i}" for i in order],
                 "labels_in_volatility_order": data["labels"], "fit_range": fit,
                 "feature_names": names}


def posterior_path(model: PortableRegimeModel, observations):
    """One forward scan equivalent to frozen prefix filtering; count fallbacks."""
    obs = np.asarray(observations, dtype=float)
    if (obs.ndim != 2 or not len(obs) or obs.shape[1] != len(model.feature_names)
            or not np.isfinite(obs).all()):
        raise ValueError("invalid observation matrix")
    emission = model._log_emission((obs - model.std_means) / model.std_scales)
    result = np.empty((len(obs), model.k))
    alpha = model.startprob
    fallback = []
    for i, row in enumerate(emission):
        prior = model.startprob if i == 0 else alpha @ model.transmat
        mass = prior * np.exp(row - row.max())
        if not np.isfinite(mass.sum()) or mass.sum() <= 0:
            fallback.append(i)
        alpha = _renorm(mass, model.startprob)
        result[i] = alpha[list(model.vol_order)]
    return result, {"fallback_rows": fallback, "n_observations": len(obs)}


def recover_mask(mask: dict, seed_dates):
    """Recover exactly the archived cohort, without today's holiday calendar."""
    dates = {str(x) for x in seed_dates}
    if dates != set(mask["flat_ohlc_pct"]):
        raise ValueError("mask/seed date universe mismatch")
    excluded = mask["excluded"]
    for values in excluded.values():
        if not set(values).issubset(dates):
            raise ValueError("mask exclusion outside seed")
    invalid = {d for key, values in excluded.items() if key != "outlier_suspect" for d in values}
    valid = sorted(dates - invalid)
    train = sorted(set(valid) - set(excluded.get("outlier_suspect", [])))
    for values, count_key, hash_key in ((valid, "n_valid", "sha256"),
                                        (train, "n_train_valid", "train_valid_sha256")):
        if len(values) != mask[count_key] or sha256("\n".join(values).encode()) != mask[hash_key]:
            raise ValueError("mask count/hash mismatch: " + hash_key)
    return [date.fromisoformat(x) for x in valid], [date.fromisoformat(x) for x in train]


def shifted_path(frame: pd.DataFrame, *, min_context: int = 60):
    if (type(min_context) is not int or min_context < 1 or frame.empty
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
        raise ValueError("invalid filtered path or warmup")
    result = frame.copy()
    result.iloc[:min_context-1] = np.nan
    return result.shift(1), pd.Series(frame.index, index=frame.index).shift(1)


def compare_stored(full: pd.DataFrame, dates, stored: np.ndarray):
    dates = pd.DatetimeIndex(dates)
    arr = np.asarray(stored)
    if (arr.shape != (len(dates), 4) or arr.dtype != np.float32 or len(dates) == 0
            or not dates.is_unique or full.shape[1] != 5 or not np.isfinite(arr).all()):
        raise ValueError("parity requires four float32 coordinates of K5")
    actual = full.reindex(dates).iloc[:, :4].to_numpy()
    error = np.abs(actual - arr)
    missing = dates[~np.isfinite(actual).all(axis=1)]
    finite = error[np.isfinite(error)]
    return {"parity_scope": "four_stored_coordinates_of_a_five_state_posterior",
            "atol": PARITY_ATOL, "rtol": 0, "n_expected": len(dates),
            "n_finite_sessions": len(dates) - len(missing),
            "missing_dates": [str(d.date()) for d in missing],
            "max_abs_error": float(finite.max()) if finite.size else None,
            "mean_abs_error": float(finite.mean()) if finite.size else None,
            "n_coordinates_over_tolerance": int((error > PARITY_ATOL).sum()),
            "passed": bool(len(missing) == 0 and (error <= PARITY_ATOL).all())}


def representation_delta(original, flattened, dates, *, baseline_passed: bool):
    if baseline_passed is not True:
        raise ValueError("baseline parity required before counterfactual")
    if not original.index.equals(flattened.index) or not original.columns.equals(flattened.columns):
        raise ValueError("complete history alignment required; no silent intersection")
    dates = pd.DatetimeIndex(dates)
    a, b = original.reindex(dates).to_numpy(), flattened.reindex(dates).to_numpy()
    if not len(dates) or not dates.is_unique or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("counterfactual cohort must be complete")
    distance = np.abs(a - b).sum(axis=1) / 2
    first, second = a.argmax(axis=1), b.argmax(axis=1)
    return {"n_sessions": len(dates), "changed_argmax": int((first != second).sum()),
            "total_variation_mean": float(distance.mean()),
            "total_variation_median": float(np.median(distance)),
            "total_variation_p95": float(np.quantile(distance, 0.95)),
            "total_variation_max": float(distance.max()),
            "argmax_tie_rule": "first_coordinate_in_frozen_volatility_order",
            "rows": [{"date": str(day.date()), "original": x.tolist(),
                      "flattened": y.tolist(), "total_variation": float(tv),
                      "original_state_coordinate": int(i), "flattened_state_coordinate": int(j)}
                     for day, x, y, tv, i, j in zip(dates, a, b, distance, first, second, strict=True)]}
