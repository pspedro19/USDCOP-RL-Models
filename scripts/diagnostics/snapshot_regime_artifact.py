#!/usr/bin/env python
"""Write a read-only manifest for the currently frozen diagnostic HMM artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(cache: Path, portable: Path, regime: Path) -> dict:
    with cache.open("rb") as handle:
        cache_blob = pickle.load(handle)
    with portable.open("rb") as handle:
        portable_blob = pickle.load(handle)
    model = cache_blob["data"].regime_model
    portable_meta = portable_blob.get("regime_meta", {})
    result = {
        "contract": "CTR-RESEARCH-HMM-ARTIFACT-MANIFEST-001",
        "cache": str(cache),
        "portable": str(portable),
        "regime_artifact": str(regime),
        "cache_key": cache_blob.get("key"),
        "dataset_identity": portable_blob.get("identity"),
        "cache_k": int(model.k),
        "portable_k": int(portable_meta.get("k", -1)),
        "cache_fit_range": list(model.fit_range),
        "portable_fit_range": list(portable_meta.get("fit_range", [])),
        "cache_bic_by_k": {str(k): float(v) for k, v in model.bic_by_k.items()},
        "portable_bic_by_k": {str(k): float(v) for k, v in portable_meta.get("bic_by_k", {}).items()},
        "regime_artifact_sha256": digest(regime),
        "portable_regime_artifact_sha256": portable_blob.get("regime_artifact_sha256"),
        "cache_portable_k_match": int(model.k) == int(portable_meta.get("k", -1)),
        "cache_portable_fit_range_match": list(model.fit_range) == list(portable_meta.get("fit_range", [])),
        "cache_portable_regime_hash_match": portable_blob.get("regime_artifact_sha256") == digest(regime),
    }
    result["coherent"] = all(result[key] for key in (
        "cache_portable_k_match", "cache_portable_fit_range_match",
        "cache_portable_regime_hash_match"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "outputs" / "thesis" / "research_data.pkl")
    parser.add_argument("--portable", type=Path,
                        default=ROOT / "outputs" / "thesis-repair" /
                        "research_data_portable_diagnostic_v2.pkl")
    parser.add_argument("--regime", type=Path,
                        default=ROOT / "config" / "research" / "regime_hmm_frozen.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.cache, args.portable, args.regime):
        if not path.is_file():
            parser.error(f"missing artifact: {path}")
    manifest = build_manifest(args.cache, args.portable, args.regime)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    print(json.dumps({"output": str(args.output), "coherent": manifest["coherent"],
                      "k": manifest["cache_k"], "dataset_identity": manifest["dataset_identity"]}))
    return 0 if manifest["coherent"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
