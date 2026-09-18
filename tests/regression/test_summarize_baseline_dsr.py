from __future__ import annotations

import json

from scripts.diagnostics.summarize_baseline_dsr import main


def test_baseline_dsr_suppresses_nontraded_ratios(tmp_path, monkeypatch):
    source = tmp_path / "baselines.json"
    output = tmp_path / "dsr.json"
    source.write_text(
        json.dumps(
            {
                "rows": [
                    {"baseline": "always_flat", "n_traded": 0, "daily_returns": [0.0] * 30},
                    {
                        "baseline": "negative",
                        "n_traded": 30,
                        "daily_returns": [-0.001] * 30,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["summarize_baseline_dsr.py", "--input", str(source), "--output", str(output)],
    )
    assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["reports"]["always_flat"]["status"] == "NOT_COMPUTABLE"
    assert report["reports"]["negative"]["passes"] is False
