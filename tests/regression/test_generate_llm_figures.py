import json
from datetime import date, timedelta

from scripts.presentation.generate_llm_figures import generate


def test_figures_are_derived_from_settlement(tmp_path):
    settlement = tmp_path / "settlement.json"
    sessions = [{"session_date": (date(2026, 1, 1) + timedelta(days=i)).isoformat(),
                 "daily_return": 0.001} for i in range(3)]
    settlement.write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    paths = generate(settlement, tmp_path / "figures")
    assert [path.name for path in paths] == [
        "llm_curva_capital.png", "llm_drawdown.png", "llm_sharpe_movil.png"
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_empty_settlement_does_not_create_figures(tmp_path):
    settlement = tmp_path / "empty.json"
    settlement.write_text(json.dumps({"sessions": []}), encoding="utf-8")
    try:
        generate(settlement, tmp_path / "figures")
    except ValueError as exc:
        assert "no settled sessions" in str(exc)
    else:
        raise AssertionError("empty settlement should fail closed")
