from pathlib import Path

import pytest

from scripts.diagnostics.audit_thesis_e2e_status import _context_artifact, audit


def test_e2e_status_is_fail_closed_without_secrets(tmp_path: Path):
    report = audit(tmp_path)
    assert report["network_called"] is False
    assert report["secrets_read"] is False
    assert report["confirmatory_ready"] is False
    assert report["controls"]["ppo_10_confirmatory_runs"]["status"] == "NOT_EXECUTED"
    assert report["controls"]["deepseek_ledger"]["status"] == "NOT_EXECUTED"
    assert report["controls"]["ppo_10_diagnostic_runs"]["confirmatory"] is False
    assert report["controls"]["macro_vintage_diagnostic"]["status"] == "NOT_PROVIDED"


def test_diagnostic_contexts_are_not_confirmatory(tmp_path: Path):
    contexts = tmp_path / "contexts.jsonl"
    contexts.write_text(
        '{"dataset_sha256":"abc","retrospective":false}\n'
        '{"dataset_sha256":"abc","retrospective":false}\n',
        encoding="utf-8",
    )
    artifact = _context_artifact(contexts)
    assert artifact["status"] == "DIAGNOSTIC_ONLY"
    assert artifact["contexts"] == 2
    assert artifact["dataset_sha256"] == "abc"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vintage_capture": Path("capture")},
        {"expected_vintage_sha": "a" * 64},
        {"expected_bundle_sha": "b" * 64},
        {
            "vintage_capture": Path("capture"),
            "expected_vintage_sha": "a" * 64,
            "expected_bundle_sha": "b" * 64,
        },
    ],
)
def test_vintage_evidence_arguments_must_be_complete_before_any_reads(tmp_path, kwargs):
    with pytest.raises(ValueError, match="together"):
        audit(tmp_path, **kwargs)


def test_e2e_calls_real_vintage_gate_interface_without_lifting_availability(tmp_path, monkeypatch):
    import src.research.vintage_evidence_gate as gate

    calls = []

    def verified(**kwargs):
        calls.append(kwargs)
        return {"status": "DIAGNOSTIC_REPRODUCED", "opening_availability_verified": False}

    monkeypatch.setattr(gate, "verify_vintage_capture", verified)
    report = audit(
        tmp_path,
        bundle=tmp_path / "bundle",
        vintage_capture=tmp_path / "capture",
        expected_vintage_sha="a" * 64,
        expected_bundle_sha="b" * 64,
    )
    assert calls == [
        {
            "capture": tmp_path / "capture",
            "expected_capture_sha": "a" * 64,
            "bundle": tmp_path / "bundle",
            "expected_bundle_sha": "b" * 64,
        }
    ]
    assert report["controls"]["macro_vintage_diagnostic"]["status"] == "DIAGNOSTIC_REPRODUCED"
    assert report["controls"]["macro_publication_and_vintages"]["status"] == "PENDING"
    assert report["scientific_closure_ready"] is False


def test_bad_vintage_evidence_is_reported_invalid_not_missing_or_pass(tmp_path, monkeypatch):
    import src.research.vintage_evidence_gate as gate

    def tampered(**kwargs):
        raise ValueError("raw digest mismatch")

    monkeypatch.setattr(gate, "verify_vintage_capture", tampered)
    report = audit(
        tmp_path,
        bundle=tmp_path / "bundle",
        vintage_capture=tmp_path / "capture",
        expected_vintage_sha="a" * 64,
        expected_bundle_sha="b" * 64,
    )
    assert report["controls"]["macro_vintage_diagnostic"]["status"] == "INVALID"
    assert report["controls"]["macro_publication_and_vintages"]["status"] == "PENDING"
    assert report["confirmatory_ready"] is False
