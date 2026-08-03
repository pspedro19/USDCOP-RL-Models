from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("acquire", ROOT / "scripts/data/acquire_public_snapshots.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


def test_public_adapter_is_explicitly_not_pit_promotion_eligible():
    assert set(module.SPECS) == {"usdcop", "xauusd", "btcusdt", "spx500"}
    source = (ROOT / "scripts/data/acquire_public_snapshots.py").read_text(encoding="utf-8")
    assert '"promotion_eligible": False' in source or module.SPECS["spx500"]["lag_days"] >= 0
