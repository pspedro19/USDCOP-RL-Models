"""Mutation tests: a green flag or intact summary hash is never sufficient evidence."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.diagnostics.verify_macro_declared_identity import build_report
from src.research.macro_evidence import (
    capture_payload,
    compare_series,
    immutable_write,
    parse_payload,
    require_macro_evidence,
)


@pytest.fixture
def evidence_case(tmp_path):
    raw = b"observation_date,DGS2\n2023-01-02,4.40\n2023-01-03,4.41\n2023-01-04,4.42\n"
    clean = tmp_path / "clean.parquet"
    series = parse_payload(raw, "FRED_DGS2")
    pd.DataFrame({"rate": series}).to_parquet(clean)
    availability = tmp_path / "availability.yaml"
    availability.write_text(yaml.safe_dump({"series": {"dgs2": {
        "column": "rate", "source": "FRED_DGS2", "unit": "percent", "fallback": "forbidden",
    }}}), encoding="utf-8")
    capture = capture_payload(raw, source="FRED_DGS2",
                              url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",
                              directory=tmp_path)
    manifest = tmp_path / "capture.json"
    manifest.write_text(json.dumps({"evidence_root": str(tmp_path), "series": {
        "dgs2": {"declared_source": "FRED_DGS2", "reference_payloads": [capture]},
    }}), encoding="utf-8")
    report = build_report(clean_path=clean, availability=availability, evidence_dir=tmp_path,
                          reference_manifest=manifest)
    return {"clean": clean, "availability": availability, "root": tmp_path,
            "report": report, "manifest": manifest, "capture": capture, "raw": raw}


def test_offline_replay_is_numerical_not_independent_or_pit(evidence_case, monkeypatch):
    import urllib.request

    def forbidden(*args, **kwargs):
        raise AssertionError("offline must not access network")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    c = evidence_case
    report = build_report(clean_path=c["clean"], availability=c["availability"],
                          evidence_dir=c["root"], reference_manifest=c["manifest"])
    assert report["all_declared_identities_honoured"] is True
    result = require_macro_evidence(report, clean=c["clean"], availability=c["availability"])
    assert result["numerical_identity_verified"] is True
    assert result["historical_availability_verified"] is False
    assert result["source_independence_verified"] is False
    assert result["recomputed"]["dgs2"]["n_common"] == 3


def test_raw_payload_is_exact_and_immutable(evidence_case):
    c = evidence_case
    target = c["root"] / c["capture"]["path"]
    assert target.read_bytes() == c["raw"]
    immutable_write(target, c["raw"])
    with pytest.raises(ValueError, match="immutable"):
        immutable_write(target, b"modified")


@pytest.mark.parametrize("mutation", ["legacy", "statistics", "missing_raw", "tampered_raw",
                                      "source", "parser", "old_timestamp", "local_only",
                                      "widen_tolerance", "incomplete_series", "clean_hash"])
def test_strict_gate_rejects_mutations(evidence_case, mutation):
    c = evidence_case
    report = copy.deepcopy(c["report"])
    entry = report["series"]["dgs2"]
    if mutation == "legacy":
        report.pop("schema_version")
    elif mutation == "statistics":
        entry["n_common"] = 999
    elif mutation == "missing_raw":
        entry["reference_payloads"] = []
    elif mutation == "tampered_raw":
        (c["root"] / c["capture"]["path"]).write_bytes(b"tampered")
    elif mutation == "source":
        entry["reference_payloads"][0]["source"] = "FRED_DCOILBRENTEU"
    elif mutation == "parser":
        entry["reference_payloads"][0]["parser_sha256"] = "0" * 64
    elif mutation == "old_timestamp":
        report["measured_at_utc"] = "2020-01-01T00:00:00+00:00"
    elif mutation == "local_only":
        entry["local_provenance_only"] = True
    elif mutation == "widen_tolerance":
        report["tolerance"] = 10
    elif mutation == "incomplete_series":
        report["series"] = {}
    elif mutation == "clean_hash":
        report["inputs"]["clean_sha256"] = "0" * 64
    with pytest.raises((ValueError, KeyError)):
        require_macro_evidence(report, clean=c["clean"], availability=c["availability"])


def test_actual_legacy_certificate_cannot_pass():
    root = Path(__file__).resolve().parents[2]
    path = root / "outputs/thesis-repair/macro_identity_research_v2_latest.json"
    if not path.exists():
        pytest.skip("historical certificate is not distributed")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report.get("schema_version") != 2
    with pytest.raises(ValueError, match="legacy"):
        require_macro_evidence(report, clean=root / "unused", availability=root / "unused")


@pytest.mark.parametrize("bad_values", [[4.4, 4.41, 4.8], [4.4, np.nan, 4.42]])
def test_discrepancies_and_missing_local_values_fail(evidence_case, bad_values):
    c = evidence_case
    frame = pd.read_parquet(c["clean"])
    frame["rate"] = bad_values
    frame.to_parquet(c["clean"])
    report = build_report(clean_path=c["clean"], availability=c["availability"],
                          evidence_dir=c["root"], reference_manifest=c["manifest"])
    assert report["all_declared_identities_honoured"] is False
    with pytest.raises(ValueError):
        require_macro_evidence(report, clean=c["clean"], availability=c["availability"])


@pytest.mark.parametrize("raw", [
    b"observation_date,DCOILBRENTEU\n2023-01-02,4\n",
    b"observation_date,DGS2\n2023-01-02,4\n2023-01-02,5\n",
    b"observation_date,DGS2\n2023-01-02,nonsense\n",
    b"observation_date,DGS2\n2023-01-02,inf\n",
])
def test_fred_wrong_identity_duplicates_malformed_nonfinite_fail(raw):
    with pytest.raises(ValueError):
        parse_payload(raw, "FRED_DGS2")


def test_investing_keeps_decimal_and_rejects_ambiguous_locale():
    raw = json.dumps({"data": [{"rowDate": "2023-01-02", "last_closeRaw": "104.32"}]}).encode()
    assert parse_payload(raw, "INVESTING_DXY").iloc[0] == 104.32
    with pytest.raises(ValueError, match="locale"):
        parse_payload(raw.replace(b"104.32", b"104,32"), "INVESTING_DXY")


def test_banrep_selects_only_nominal_overnight():
    raw = b'''<GenericData xmlns:g="http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic">
    <g:Series><g:SeriesKey><g:Value id="SUBJECT" value="IRIBRM00"/>
    <g:Value id="UNIT_MEASURE" value="NR"/></g:SeriesKey><g:Obs>
    <g:ObsDimension value="20230102"/><g:ObsValue value="12.05"/></g:Obs></g:Series></GenericData>'''
    assert parse_payload(raw, "BANREP_IBR").iloc[0] == 12.05
    with pytest.raises(ValueError, match="exactly one"):
        parse_payload(raw.replace(b'IRIBRM00', b'IRIBRM01'), "BANREP_IBR")


def test_secrets_and_wrong_instrument_are_not_archived(tmp_path):
    for url in ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2&apikey=secret",
                "https://username:secret@fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"):
        with pytest.raises(ValueError):
            capture_payload(b"raw", source="FRED_DGS2", url=url, directory=tmp_path)
    assert not list(tmp_path.rglob("*.bin"))


def test_nonfinite_statistics_never_written():
    with pytest.raises(ValueError, match="nonfinite"):
        compare_series(pd.Series([np.inf], index=pd.to_datetime(["2023-01-02"])),
                       pd.Series([4.0], index=pd.to_datetime(["2023-01-02"])), tolerance=.01)
