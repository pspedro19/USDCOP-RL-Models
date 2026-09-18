import json

from scripts.validation.validate_thesis_llm_ledger import validate


def _row(bar=0):
    digest = "a" * 64
    return {"decision_id": f"2026-01-02::llm::{bar}", "session_date": "2026-01-02", "bar": bar,
            "prompt_hash": digest, "raw_response_sha256": digest, "model_id": "deepseek-chat",
            "prompt_version": "thesis-llm-trader-v1", "max_tokens": 256,
            "temperature": 0.1, "top_p": 0.9}


def test_ledger_validator_accepts_hashes_and_sampling_contract(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    result = validate(path)
    assert result["valid"] and result["rows"] == 1
    assert result["incomplete_sessions"]["2026-01-02"] == list(range(1, 59))


def test_ledger_validator_rejects_secret_fields(tmp_path):
    row = _row()
    row["api_key"] = "must-not-be-recorded"
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        validate(path)
    except ValueError as exc:
        assert "secret" in str(exc)
    else:
        raise AssertionError("secret field should fail closed")


def test_ledger_validator_can_bind_rows_to_dataset_hash(tmp_path):
    row = _row()
    row["dataset_sha256"] = "b" * 64
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert validate(path, expected_dataset_sha256="b" * 64)["valid"] is True
    try:
        validate(path, expected_dataset_sha256="c" * 64)
    except ValueError as exc:
        assert "dataset hash mismatch" in str(exc)
    else:
        raise AssertionError("dataset hash mismatch should fail closed")


def test_forward_ledger_requires_block_and_nonretrospective_marker(tmp_path):
    row = _row()
    row.update({"dataset_sha256": "b" * 64, "dataset_block": "forward", "retrospective": False})
    path = tmp_path / "ledger.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = validate(
        path,
        expected_dataset_sha256="b" * 64,
        expected_dataset_block="forward",
        forbid_retrospective=True,
    )
    assert result["valid"] is True
    row["retrospective"] = True
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    try:
        validate(path, expected_dataset_block="forward", forbid_retrospective=True)
    except ValueError as exc:
        assert "retrospective row forbidden" in str(exc)
    else:
        raise AssertionError("retrospective forward row should fail closed")
