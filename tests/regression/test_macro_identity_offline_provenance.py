import hashlib
import json

import pandas as pd


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_offline_macro_provenance_is_evidence_but_not_gate(tmp_path):
    from scripts.diagnostics import verify_macro_declared_identity as module

    clean = tmp_path / "MACRO_RESEARCH_v2.parquet"
    frame = pd.DataFrame({"value": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    frame.to_parquet(clean)
    provenance = clean.with_suffix(".provenance.json")
    provenance.write_text(json.dumps({
        "artifact_sha256": _sha256(clean),
        "series": {
            "brent": {
                "declared_source": "FRED_DCOILBRENTEU",
                "source_verified": True,
                "payload_sha256": "a" * 64,
            }
        },
    }), encoding="utf-8")

    loaded = module._load_local_provenance(clean)
    assert loaded["series"]["brent"]["source_verified"] is True

    clean.write_bytes(clean.read_bytes() + b"tamper")
    assert module._load_local_provenance(clean) == {}
