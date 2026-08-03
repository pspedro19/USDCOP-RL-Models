import json
import sys
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location

ROOT = Path(__file__).resolve().parents[2]
spec = spec_from_file_location("harness_engine", ROOT / ".claude/codex/harness/harness_engine.py")
module = module_from_spec(spec)
assert spec.loader
sys.modules["harness_engine"] = module
spec.loader.exec_module(module)


def test_harness_covers_all_assets_and_blocks_without_external_evidence(tmp_path):
    out = tmp_path / "manifest.json"
    result = module.run(out, run_tests=False)
    assert result["assets"] == ["usdcop", "xauusd", "btcusdt", "spx500"]
    assert result["decision"] == "NO-GO"
    assert any(g["name"] == "real-data-oos" and g["status"] == "BLOCKED" for g in result["gates"])
    assert json.loads(out.read_text())["passed"] is False
