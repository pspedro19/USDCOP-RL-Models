import pandas as pd

from src.gold_rl.backtest import run_backtest


def test_gold_backtest_artifact_marks_unverified_costs():
    times = pd.date_range("2024-01-01", periods=25, freq="D")
    frame = pd.DataFrame({
        "time": times,
        "close": [100.0 + i for i in range(25)],
        "position": [0.0] + [1.0] * 24,
        "regime": ["trend"] * 25,
    })
    result = run_backtest(frame, "fixture", "fixture", year=2024)
    contract = result["summary"]["cost_contract"]
    assert contract["status"] == "PENDING_VENUE"
    assert result["summary"]["asset"] == "XAU/USD"
