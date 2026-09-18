from __future__ import annotations

import pickle
from types import SimpleNamespace

from scripts.analysis.thesis_supervised_baseline_v2 import main


def _session(day: int, direction: float):
    close = [4000.0]
    for _ in range(59):
        close.append(close[-1] * (1.0 + direction * 0.0001))
    market = [[direction] * 25 for _ in range(60)]
    return SimpleNamespace(date=f"2024-01-{day:02d}", close=close, market=market, spread_pips=3.0)


def test_supervised_arm_trains_only_from_development(tmp_path, monkeypatch):
    portable = tmp_path / "portable.pkl"
    pickle.dump(
        {"development": [_session(1, 1.0), _session(2, -1.0)], "selection": [_session(3, 1.0)]},
        portable.open("wb"),
    )
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "sys.argv",
        ["thesis_supervised_baseline_v2.py", "--portable", str(portable), "--output", str(output)],
    )
    assert main() == 0
    assert output.exists()
