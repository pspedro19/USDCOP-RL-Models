#!/usr/bin/env python
"""Publish a NEW self-contained research bundle; never overwrite frozen globals.

Requires a locally trusted freshly built cache, exact cache SHA and current input
identity. Components are written exclusively; manifest.json is the final commit
marker. Failed directories are retained for inspection, never deleted or promoted.
This exports evidence, not a fit or an experiment, and does not retarget live_spec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.dataset import (  # noqa: E402 - repository bootstrap for direct CLI
    MARKET_FEATURES,
    N_REGIMES,
    ResearchData,
    dataset_identity,
    dataset_identity_manifest,
)
from src.research.macro_evidence import canonical_json, immutable_write  # noqa: E402


def _safe_path(path: Path) -> Path:
    resolved = Path(path).resolve()
    for part in resolved.parts:
        name = part.lower()
        if (name in {"secrets", ".env"} or name.startswith(".env.")
                or name.endswith((".pem", ".key"))
                or (name.startswith(("credentials", "service-account")) and name.endswith(".json"))):
            raise ValueError("sensitive path is not a research artifact")
    return resolved


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise ValueError("expected SHA256 must be explicit")


def _current_contract() -> dict:
    result = dataset_identity_manifest()
    # Bundle freezes its own numeric artifacts, not the previous global exports.
    result.pop("frozen_content_excluding_dataset_identity", None)
    return result


def _array(value, name: str, *, positive=False) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if not result.size or not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError(f"invalid {name}")
    return result


def _frozen_payloads(data: ResearchData) -> tuple[dict, dict]:
    scaler = {"contract": "CTR-RESEARCH-FORWARD-001", "features": MARKET_FEATURES}
    for name, value, size, positive in (
        ("mean", data.scaler_mean, len(MARKET_FEATURES), False),
        ("scale", data.scaler_scale, len(MARKET_FEATURES), True),
        ("macro_mean", data.macro_scaler_mean, 3, False),
        ("macro_scale", data.macro_scaler_scale, 3, True),
    ):
        array = _array(value, name, positive=positive)
        if array.shape != (size,):
            raise ValueError(f"invalid {name} shape")
        scaler[name] = array.tolist()
    model = data.regime_model
    hmm = model.model
    covars = _array(hmm.covars_, "covars")
    if covars.ndim == 2:
        covars = np.stack([np.diag(row) for row in covars])
    regime = {"contract": "CTR-RESEARCH-REGIME-PORTABLE-001", "k": int(model.k),
              "labels": list(model.state_labels()), "vol_order": list(model.vol_order),
              "feature_names": list(model.feature_names), "fit_range": list(model.fit_range)}
    for name, value in (("startprob", hmm.startprob_), ("transmat", hmm.transmat_),
                        ("means", hmm.means_), ("covars", covars),
                        ("std_means", model.means), ("std_scales", model.scales)):
        regime[name] = _array(value, name, positive=name == "std_scales").tolist()
    _validate_regime(regime)
    _validate_sessions(data)
    return scaler, regime


def _validate_sessions(data) -> None:
    k = data.regime_model["k"] if isinstance(data.regime_model, dict) else data.regime_model.k
    seen = set()
    previous_last = None
    for block in (data.development, data.selection, data.holdout):
        dates = [spec.date for spec in block]
        if not dates or dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError("research blocks must be nonempty and strictly ordered")
        if seen.intersection(dates) or (previous_last is not None and dates[0] <= previous_last):
            raise ValueError("research blocks must be disjoint and temporally ordered")
        seen.update(dates)
        previous_last = dates[-1]
        for spec in block:
            if (_array(spec.close, "close", positive=True).shape != (60,)
                    or _array(spec.market, "market").shape != (60, len(MARKET_FEATURES))
                    or _array(spec.context, "context").shape != (3 + N_REGIMES,)
                    or not np.isfinite(spec.spread_pips) or spec.spread_pips < 0):
                raise ValueError("invalid SessionSpec in bundle")
            posterior = np.asarray(spec.context[-N_REGIMES:], dtype=float)
            if ((posterior < 0).any() or (posterior > 1).any()
                    or not np.isclose(posterior.sum(), 1.0, atol=1e-6, rtol=0)
                    or (k < N_REGIMES and np.any(np.abs(posterior[k:]) > 1e-6))):
                raise ValueError("SessionSpec regime posterior is truncated/invalid; must sum to one within 1e-6")


def _validate_regime(regime: dict) -> None:
    k, dimensions = regime["k"], len(regime["feature_names"])
    if not isinstance(k, int) or k <= 0 or dimensions <= 0:
        raise ValueError("invalid regime dimensions")
    if k > N_REGIMES:
        raise ValueError(f"regime k={k} exceeds {N_REGIMES} observation slots; no posterior truncation allowed")
    expected = {"startprob": (k,), "transmat": (k, k), "means": (k, dimensions),
                "covars": (k, dimensions, dimensions), "std_means": (dimensions,),
                "std_scales": (dimensions,)}
    for name, shape in expected.items():
        if _array(regime[name], name, positive=name == "std_scales").shape != shape:
            raise ValueError(f"invalid regime {name} shape")
    for name in ("startprob", "transmat"):
        values = np.asarray(regime[name])
        if (values < 0).any() or not np.allclose(values.sum(axis=-1), 1, atol=1e-10, rtol=0):
            raise ValueError("invalid regime probabilities")
    if sorted(regime["vol_order"]) != list(range(k)) or len(regime["labels"]) != k:
        raise ValueError("invalid regime labels/order")
    for covariance in np.asarray(regime["covars"]):
        # Validation only: no silent jitter or repair of frozen parameters.
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0):
            raise ValueError("regime covariance is not symmetric within 1e-12")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("regime covariance is not positive definite; no jitter applied") from exc


def export_bundle(data: ResearchData, output: Path, *, source_contract: dict,
                  source_cache_sha256: str) -> Path:
    """Only call for data built under source_contract; CLI enforces cache identity."""
    _require_sha(source_cache_sha256)
    output = _safe_path(output)
    if source_contract != _current_contract():
        raise ValueError("input contract changed before bundle export")
    scaler, regime = _frozen_payloads(data)
    identity_payload = {"version": 3, "inputs": source_contract,
                        "scaler_content_sha256": _sha(canonical_json(scaler)),
                        "regime_content_sha256": _sha(canonical_json(regime)),
                        "exporter_sha256": _sha(Path(__file__).read_bytes())}
    identity = _sha(canonical_json(identity_payload))
    scaler["dataset_identity"] = regime["dataset_identity"] = identity
    model = data.regime_model
    blob = {"identity": identity, "identity_manifest": identity_payload,
            "development": data.development, "selection": data.selection,
            "holdout": data.holdout, "scaler_mean": data.scaler_mean,
            "scaler_scale": data.scaler_scale, "macro_scaler_mean": data.macro_scaler_mean,
            "macro_scaler_scale": data.macro_scaler_scale, "dropped": data.dropped,
            "regime_meta": {"k": model.k, "labels": model.state_labels(),
                            "fit_range": model.fit_range, "covariance_type": model.covariance_type,
                            "feature_names": list(model.feature_names),
                            "bic_by_k": {str(k): float(v) for k, v in model.bic_by_k.items()}}}
    components = {"scaler.json": canonical_json(scaler), "regime.json": canonical_json(regime)}
    blob["regime_artifact_sha256"] = _sha(components["regime.json"])
    components["portable.pkl"] = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
    # Exclusive leaf directory; no rename/delete and no mutation of global artifacts.
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in components.items():
        immutable_write(output / name, raw)
    if source_contract != _current_contract():
        raise ValueError("inputs changed during export; incomplete directory retained without manifest")
    manifest = {"schema_version": 3, "dataset_identity": identity,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_cache_sha256": source_cache_sha256,
                "identity_manifest": identity_payload,
                "artifacts": {name: {"sha256": _sha(raw), "bytes": len(raw)}
                              for name, raw in components.items()},
                "publication": "complete", "role": "frozen_research_bundle_not_live_promotion"}
    immutable_write(output / "manifest.json", canonical_json(manifest))
    return output / "manifest.json"


def load_bundle(path: Path, *, expected_manifest_sha256: str,
                require_current_inputs: bool = True) -> SimpleNamespace:
    """Load exact trusted bundle; legacy live globals are deliberately not consulted.

    The caller supplies a trusted manifest SHA before any pickle is read. Disabling
    current-input comparison is for an explicitly historical replay only.
    """
    _require_sha(expected_manifest_sha256)
    root = _safe_path(path)
    raw_manifest = _safe_path(root / "manifest.json").read_bytes()
    if _sha(raw_manifest) != expected_manifest_sha256:
        raise ValueError("bundle manifest hash mismatch")
    manifest = json.loads(raw_manifest)
    if manifest.get("schema_version") != 3 or manifest.get("publication") != "complete":
        raise ValueError("incomplete/version-mismatched bundle")
    expected_names = {"scaler.json", "regime.json", "portable.pkl"}
    if set(manifest["artifacts"]) != expected_names:
        raise ValueError("bundle artifact set mismatch")
    identity_payload = manifest["identity_manifest"]
    identity = _sha(canonical_json(identity_payload))
    if manifest["dataset_identity"] != identity:
        raise ValueError("bundle identity mismatch")
    if require_current_inputs and identity_payload["inputs"] != _current_contract():
        raise ValueError("bundle current input contract mismatch")
    if require_current_inputs and identity_payload["exporter_sha256"] != _sha(Path(__file__).read_bytes()):
        raise ValueError("bundle exporter code mismatch")
    raw = {}
    for name in sorted(expected_names):
        target = _safe_path(root / name)
        if target.parent != root:
            raise ValueError("bundle artifact escapes directory")
        raw[name] = target.read_bytes()
        info = manifest["artifacts"][name]
        if _sha(raw[name]) != info["sha256"] or len(raw[name]) != info["bytes"]:
            raise ValueError(f"bundle artifact hash mismatch: {name}")
    scaler, regime = json.loads(raw["scaler.json"]), json.loads(raw["regime.json"])
    for name, payload in (("scaler", scaler), ("regime", regime)):
        content = dict(payload)
        if content.pop("dataset_identity", None) != identity:
            raise ValueError("component dataset identity mismatch")
        if _sha(canonical_json(content)) != identity_payload[f"{name}_content_sha256"]:
            raise ValueError("component numeric identity mismatch")
    blob = pickle.loads(raw["portable.pkl"])
    if blob.get("identity") != identity or blob.get("identity_manifest") != identity_payload:
        raise ValueError("portable identity mismatch")
    for name, key in (("mean", "scaler_mean"), ("scale", "scaler_scale"),
                      ("macro_mean", "macro_scaler_mean"), ("macro_scale", "macro_scaler_scale")):
        if not np.array_equal(np.asarray(scaler[name]), blob[key]):
            raise ValueError("portable/scaler numeric mismatch")
    _validate_regime(regime)
    data = ResearchData(development=blob["development"], selection=blob["selection"],
                        holdout=blob["holdout"], scaler_mean=blob["scaler_mean"],
                        scaler_scale=blob["scaler_scale"], regime_model=blob["regime_meta"],
                        dropped=blob["dropped"], macro_scaler_mean=blob["macro_scaler_mean"],
                        macro_scaler_scale=blob["macro_scaler_scale"])
    _validate_sessions(data)
    return SimpleNamespace(data=data, scaler=scaler, regime=regime, manifest=manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True, help="trusted research cache, not an arbitrary pickle")
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="new exclusive bundle directory")
    args = parser.parse_args()
    _require_sha(args.expected_cache_sha256)
    cache_path, output = _safe_path(args.cache), _safe_path(args.output)
    if output.exists():
        raise ValueError("bundle output already exists")
    raw = cache_path.read_bytes()
    if _sha(raw) != args.expected_cache_sha256:
        raise ValueError("cache SHA mismatch; refusing deserialization")
    cache = pickle.loads(raw)
    if cache.get("key", {}).get("identity_v3") != dataset_identity():
        raise ValueError("cache was not built under the current v3 input contract")
    manifest = export_bundle(cache["data"], output, source_contract=_current_contract(),
                             source_cache_sha256=args.expected_cache_sha256)
    print(f"published={manifest} manifest_sha256={_sha(manifest.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
