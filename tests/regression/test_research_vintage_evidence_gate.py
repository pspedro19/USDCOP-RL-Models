"""E2E evidence must be recomputed from raw snapshots, not trusted PASS fields."""

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.diagnostics import audit_research_macro_vintages as source
from src.research import alfred_vintages as vintage
from src.research.vintage_evidence_gate import verify_vintage_capture


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(vintage.json_bytes(value))


def hash_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(source, "ROOT", tmp_path)
    base = tmp_path / "outputs/thesis-repair"
    bundle, archive, capture = base / "bundle", base / "archive", base / "capture"
    dates = ["2023-01-04"]
    frame = pd.DataFrame(
        {
            vintage.SERIES_COLUMNS["DGS2"]: [4.3, 4.4],
            vintage.SERIES_COLUMNS["DCOILBRENTEU"]: [80.0, 81.0],
        },
        index=pd.to_datetime(["2023-01-02", "2023-01-03"]),
    )
    data = io.BytesIO()
    frame.to_parquet(data)
    files = []
    for path, payload in [
        ("data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet", data.getvalue()),
        (
            "config/research/macro_availability.yaml",
            yaml.safe_dump({"max_staleness_business_days": 5}).encode(),
        ),
    ]:
        digest = hashlib.sha256(payload).hexdigest()
        target = archive / "objects" / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        files.append({"path": path, "object": "objects/" + digest, "sha256": digest})
    save(archive / "manifest.json", {"files": files})
    save(bundle / "daily_series.json", {"always_flat": [{"date": dates[0], "net": 0}]})
    save(
        bundle / "manifest.json",
        {
            "artifacts_sha256": {"daily_series.json": hash_of(bundle / "daily_series.json")},
            "snapshot_manifest": str(archive / "manifest.json"),
            "snapshot_sha256": hash_of(archive / "manifest.json"),
        },
    )
    bundle_sha = hash_of(bundle / "manifest.json")
    _, _, identities = source.load_inputs(bundle, bundle_sha)

    class Response:
        status = 200

        def __init__(self, url):
            self.url = url

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self):
            return self.url

        def read(self, count):
            series = "DGS2" if "id=DGS2&" in self.url else "DCOILBRENTEU"
            values = "4.3\n2023-01-03,4.4" if series == "DGS2" else "80\n2023-01-03,81"
            return f"observation_date,{series}_20230103\n2023-01-02,{values}\n".encode()

    class Opener:
        def open(self, request, timeout):
            return Response(request.full_url)

    monkeypatch.setattr(vintage.urllib.request, "build_opener", lambda *a: Opener())
    records = [vintage.capture_snapshot(capture, s, dates[0]) for s in vintage.SERIES_COLUMNS]
    freeze = {
        "version": vintage.VERSION,
        "inputs": identities,
        "dates": dates,
        "series": sorted(vintage.SERIES_COLUMNS),
        "window_calendar_days": 60,
        "source_sha256": hash_of(Path(vintage.__file__)),
        "runner_sha256": hash_of(Path(source.__file__)),
        "coverage_is_entire_selection": True,
    }
    save(capture / "audit_freeze.json", freeze)
    for module in (source, vintage):
        path = capture / "source" / Path(module.__file__).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(module.__file__).read_bytes())
    report = source.summarize(frame, dates, records, capture, 5)
    report.update(
        inputs=identities,
        coverage_is_entire_selection=True,
        network_called=True,
        raw_evidence_root=str(capture),
        freeze_sha256=hash_of(capture / "audit_freeze.json"),
    )
    save(capture / "report.json", report)
    manifest = {
        "report_sha256": hash_of(capture / "report.json"),
        "freeze_sha256": hash_of(capture / "audit_freeze.json"),
        "raw_evidence_root": str(capture),
        "capture_records_sha256": {
            str(p.relative_to(capture)): hash_of(p)
            for p in sorted((capture / "captures").glob("*.json"))
        },
    }
    save(capture / "manifest.json", manifest)
    # Any attempted network access by the verifier must fail this test.
    monkeypatch.setattr(
        vintage.urllib.request, "build_opener", lambda *a: pytest.fail("network forbidden")
    )
    return {
        "capture": capture,
        "expected_capture_sha": hash_of(capture / "manifest.json"),
        "bundle": bundle,
        "expected_bundle_sha": bundle_sha,
    }


def test_verified_matching_vintages_never_approve_opening_availability(evidence):
    result = verify_vintage_capture(**evidence)
    assert result["status"] == "DIAGNOSTIC_REPRODUCED"
    assert result["requested_snapshots"] == 2
    assert result["series"]["DGS2"]["counts"] == {"VALUES_MATCH_PRIOR_DATE_VINTAGE": 1}
    assert result["opening_availability_verified"] is False
    assert result["scientific_closure_ready"] is False
    assert result["network_called"] is False
    assert "saved_report_matches_current_offline_recomputation_from_raw_csv" in result["proves"]
    assert {"hora_08", "recibo_historico", "DXY", "IBR", "linaje_de_entrenamiento"} <= set(
        result["does_not_prove"]
    )


@pytest.mark.parametrize("key", ["expected_capture_sha", "expected_bundle_sha"])
def test_external_manifest_hash_is_mandatory(evidence, key):
    evidence[key] = "0" * 64
    with pytest.raises(ValueError, match="identity|digest"):
        verify_vintage_capture(**evidence)


@pytest.mark.parametrize(
    "target", ["manifest.json", "audit_freeze.json", "report.json", "source/alfred_vintages.py"]
)
def test_mutated_evidence_file_is_rejected(evidence, target):
    path = evidence["capture"] / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        verify_vintage_capture(**evidence)


def test_raw_tampering_is_rejected(evidence):
    raw = next((evidence["capture"] / "raw").glob("*.csv"))
    raw.write_bytes(raw.read_bytes().replace(b"4.4", b"9.9").replace(b",81", b",99"))
    with pytest.raises(ValueError, match="hash"):
        verify_vintage_capture(**evidence)


def test_forged_pass_even_with_rehashed_manifest_is_rejected(evidence):
    report_path = evidence["capture"] / "report.json"
    report = json.loads(report_path.read_text())
    report["opening_availability_verified"] = True
    save(report_path, report)
    manifest_path = evidence["capture"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["report_sha256"] = hash_of(report_path)
    save(manifest_path, manifest)
    evidence["expected_capture_sha"] = hash_of(manifest_path)
    with pytest.raises(ValueError, match="recomputed"):
        verify_vintage_capture(**evidence)


def test_new_figure_bundle_with_identical_source_and_cohort_is_accepted(evidence):
    sibling = evidence["bundle"].parent / "new_figure_version"
    sibling.mkdir()
    for name in ("manifest.json", "daily_series.json"):
        (sibling / name).write_bytes((evidence["bundle"] / name).read_bytes())
    evidence["bundle"] = sibling
    assert verify_vintage_capture(**evidence)["status"] == "DIAGNOSTIC_REPRODUCED"


@pytest.mark.parametrize(
    "name", [".env", ".env.example", "secrets/report.json", "a.key", "credentials-a.json"]
)
def test_secret_path_refused_before_any_read(evidence, name, monkeypatch):
    evidence["capture"] = evidence["capture"].parent / name
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="secret"):
        verify_vintage_capture(**evidence)


def amend_manifest(evidence, name):
    path = evidence["capture"] / "manifest.json"
    manifest = json.loads(path.read_text())
    if name == "audit_freeze.json":
        manifest["freeze_sha256"] = hash_of(evidence["capture"] / name)
    elif name == "report.json":
        manifest["report_sha256"] = hash_of(evidence["capture"] / name)
    else:
        manifest["capture_records_sha256"][str(Path(name))] = hash_of(evidence["capture"] / name)
    save(path, manifest)
    evidence["expected_capture_sha"] = hash_of(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dates", ["2023-01-05"]),
        ("coverage_is_entire_selection", False),
        ("series", ["DGS2"]),
        ("window_calendar_days", 61),
        ("source_sha256", "0" * 64),
    ],
)
def test_rehashed_freeze_semantic_changes_still_rejected(evidence, field, value):
    path = evidence["capture"] / "audit_freeze.json"
    frozen = json.loads(path.read_text())
    frozen[field] = value
    save(path, frozen)
    amend_manifest(evidence, "audit_freeze.json")
    with pytest.raises(ValueError):
        verify_vintage_capture(**evidence)


def test_rehashed_record_wrong_session_is_rejected(evidence):
    name = "captures/DGS2_2023-01-03.json"
    path = evidence["capture"] / name
    record = json.loads(path.read_text())
    record["session_date"] = "2023-01-05"
    save(path, record)
    amend_manifest(evidence, name)
    with pytest.raises(ValueError, match="cohort"):
        verify_vintage_capture(**evidence)


def test_rehashed_figure_bundle_with_different_dates_is_rejected(evidence):
    sibling = evidence["bundle"].parent / "different_cohort"
    sibling.mkdir()
    save(sibling / "daily_series.json", {"always_flat": [{"date": "2023-01-05", "net": 0}]})
    manifest = json.loads((evidence["bundle"] / "manifest.json").read_text())
    manifest["artifacts_sha256"]["daily_series.json"] = hash_of(sibling / "daily_series.json")
    save(sibling / "manifest.json", manifest)
    evidence.update(bundle=sibling, expected_bundle_sha=hash_of(sibling / "manifest.json"))
    with pytest.raises(ValueError, match="cohort"):
        verify_vintage_capture(**evidence)


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}'])
def test_ambiguous_json_manifest_is_rejected_before_interpretation(evidence, payload):
    path = evidence["capture"] / "manifest.json"
    path.write_bytes(payload)
    evidence["expected_capture_sha"] = hash_of(path)
    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        verify_vintage_capture(**evidence)


def test_manifest_cannot_add_unexpected_capture_path(evidence):
    path = evidence["capture"] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["capture_records_sha256"]["../../.env"] = "0" * 64
    save(path, manifest)
    evidence["expected_capture_sha"] = hash_of(path)
    with pytest.raises(ValueError, match="cohort"):
        verify_vintage_capture(**evidence)


def test_rehashed_shorter_window_cannot_masquerade_as_frozen_sixty_days(evidence):
    name = "captures/DGS2_2023-01-03.json"
    path = evidence["capture"] / name
    record = json.loads(path.read_text())
    record["start_date"] = "2023-01-01"
    record["url"] = vintage.snapshot_url("DGS2", "2023-01-03", "2023-01-01")
    save(path, record)
    amend_manifest(evidence, name)
    with pytest.raises(ValueError, match="window"):
        verify_vintage_capture(**evidence)


def test_duplicate_json_keys_in_new_figure_manifest_are_rejected(evidence):
    sibling = evidence["bundle"].parent / "ambiguous_figure_bundle"
    sibling.mkdir()
    (sibling / "daily_series.json").write_bytes(
        (evidence["bundle"] / "daily_series.json").read_bytes()
    )
    raw = (evidence["bundle"] / "manifest.json").read_bytes()
    raw = raw.replace(b'"snapshot_sha256":', b'"snapshot_sha256":"ignored", "snapshot_sha256":')
    (sibling / "manifest.json").write_bytes(raw)
    evidence.update(bundle=sibling, expected_bundle_sha=hash_of(sibling / "manifest.json"))
    with pytest.raises(ValueError, match="duplicate"):
        verify_vintage_capture(**evidence)
