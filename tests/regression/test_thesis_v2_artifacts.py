"""Gates mínimos para que el carril v2 no vuelva a consumir artefactos v1."""

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "experiments" / "thesis_ppo_v2.yaml"
SCHEMA = ROOT / "config" / "research" / "feature_schema_v2.json"
PORTABLE = ROOT / "data" / "thesis" / "research_data_portable_v2.pkl"


def test_v2_declares_versioned_inputs_and_output_dir():
    text = CONFIG.read_text(encoding="utf-8")
    assert "dataset_version: v2" in text
    assert "feature_schema_v2.json" in text
    assert "research_data_portable_v2.pkl" in text
    assert "outputs/thesis-repair/ppo_v2" in text


def test_v2_schema_excludes_degenerate_ohlc_features():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["n_features"] == 37
    assert "parkinson_12" not in schema["order"]
    assert "garman_klass_12" not in schema["order"]
    assert PORTABLE.is_file()
