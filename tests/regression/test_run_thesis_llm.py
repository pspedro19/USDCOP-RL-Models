import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SCRIPT = ROOT / "scripts" / "analysis" / "run_thesis_llm.py"


def _context(path: Path, *, bar: int = 0) -> None:
    path.write_text(json.dumps({
        "session_date": "2026-01-05",
        "bar": bar,
        "previous_weight": 0.0,
        "system_prompt": "Responde solo JSON.",
        "user_prompt": "Contexto causal de prueba.",
    }) + "\n", encoding="utf-8")


def test_validation_mode_never_calls_network(tmp_path):
    source = tmp_path / "contexts.jsonl"
    _context(source)
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--input-jsonl", str(source), "--model-id", "test-model"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {
        "ledger_written": False, "network_called": False, "validated_contexts": 1,
    }


def test_validation_rejects_duplicate_decision_context(tmp_path):
    source = tmp_path / "contexts.jsonl"
    _context(source, bar=3)
    with source.open("a", encoding="utf-8") as handle:
        handle.write(source.read_text(encoding="utf-8"))
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--input-jsonl", str(source), "--model-id", "test-model"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "duplicate decision context" in result.stderr


def test_execute_blocks_context_without_matching_portable_hash(tmp_path):
    source = tmp_path / "contexts.jsonl"
    _context(source)
    portable = tmp_path / "portable.pkl"
    portable.write_bytes(b"portable-v2-fixture")
    ledger = tmp_path / "ledger.jsonl"
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--input-jsonl", str(source), "--model-id", "test-model",
         "--portable", str(portable), "--ledger", str(ledger), "--execute"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "dataset_sha256" in result.stderr
    assert not ledger.exists()


def test_execute_blocks_unsigned_preregistration_before_provider_call(tmp_path):
    portable = ROOT / "data" / "thesis" / "research_data_portable_v2.pkl"
    if not portable.is_file():
        return
    context = {
        "session_date": "2026-01-05", "bar": 0, "previous_weight": 0.0,
        "system_prompt": "Responde solo JSON.", "user_prompt": "Contexto causal.",
        "dataset_sha256": hashlib.sha256(portable.read_bytes()).hexdigest(),
    }
    source = tmp_path / "contexts.jsonl"
    source.write_text(json.dumps(context) + "\n", encoding="utf-8")
    prereg = tmp_path / "prereg.md"
    prereg.write_text("status: PARTIAL\n", encoding="utf-8")
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--input-jsonl", str(source), "--model-id", "test-model",
         "--portable", str(portable), "--ledger", str(tmp_path / "ledger.jsonl"),
         "--prereg-path", str(prereg), "--execute"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "pre-registration is not SIGNED" in result.stderr
