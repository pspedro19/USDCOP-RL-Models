from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_approved_usdcop_ensemble_matches_live_execution():
    with (ROOT / "config" / "execution" / "smart_simple_v1.yaml").open(encoding="utf-8") as f:
        execution = yaml.safe_load(f)
    with (ROOT / "config" / "forecasting_ssot.yaml").open(encoding="utf-8") as f:
        ssot = yaml.safe_load(f)

    expected = [m for m in ssot["tracks"]["h5"]["models"]]
    actual = [m["name"] for m in execution["models"]["list"]]
    assert actual == expected
    assert execution["models"]["ensemble"]["method"] == ssot["tracks"]["h5"]["ensemble"]
    assert execution["models"]["use_xgboost"] is False
