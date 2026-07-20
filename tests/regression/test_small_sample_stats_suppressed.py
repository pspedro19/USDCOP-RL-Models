"""Sharpe and p-values must not be published on tiny samples.

Contract: CTR-QUANT-CONSTITUTION-001 §6

`quant-constitution.md`: "con N < 20 trades se reporta solo conteo y PnL (nada de
'Sharpe 19, p=0.000, 3 trades')."

The 2026 production slice had **5 trades** and was still shipping `sharpe: -0.949` and
`p_value: 0.5944` into `summary.json`, which the dashboard renders as if it carried
statistical meaning. The pipeline was breaking the rule the repo wrote for itself.

Descriptive stats (return, drawdown, win rate, trade count) stay — they are observations.
Inferential stats (Sharpe, p-value) are nulled, with `insufficient_trades: true` so the UI
can say so out loud instead of showing a confident-looking number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "usdcop-trading-dashboard" / "public" / "data" / "production"
MIN_TRADES = 20

SUMMARIES = sorted(PROD.glob("summary*.json")) if PROD.is_dir() else []

pytestmark = pytest.mark.skipif(not SUMMARIES, reason="no published summaries")


def _n_trades(doc: dict) -> int | None:
    if isinstance(doc.get("n_trades"), int):
        return doc["n_trades"]
    for block in (doc.get("strategies") or {}).values():
        if isinstance(block, dict):
            for key in ("n_trades", "trades"):
                if isinstance(block.get(key), int):
                    return block[key]
            longs, shorts = block.get("n_long"), block.get("n_short")
            if isinstance(longs, int) and isinstance(shorts, int):
                return longs + shorts
    return None


@pytest.mark.parametrize("path", SUMMARIES, ids=lambda p: p.name)
def test_no_inferential_stats_below_twenty_trades(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    n = _n_trades(doc)
    if n is None or n >= MIN_TRADES:
        pytest.skip(f"n_trades={n} — rule does not apply")

    tests = doc.get("statistical_tests") or {}
    assert tests.get("p_value") is None, (
        f"{path.name} publishes p_value={tests.get('p_value')} on {n} trades. "
        "quant-constitution §6 forbids it — emit null + insufficient_trades."
    )
    assert tests.get("significant") is not True, (
        f"{path.name} claims statistical significance on {n} trades"
    )

    for sid, block in (doc.get("strategies") or {}).items():
        if sid == "buy_and_hold" or not isinstance(block, dict):
            continue
        assert block.get("sharpe") is None, (
            f"{path.name}::{sid} publishes sharpe={block.get('sharpe')} on {n} trades"
        )


@pytest.mark.parametrize("path", SUMMARIES, ids=lambda p: p.name)
def test_small_samples_are_flagged_for_the_ui(path: Path):
    """The UI cannot warn about a limitation the payload does not declare."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    n = _n_trades(doc)
    if n is None or n >= MIN_TRADES:
        pytest.skip(f"n_trades={n} — rule does not apply")
    flagged = doc.get("insufficient_trades") is True or (
        (doc.get("statistical_tests") or {}).get("insufficient_trades") is True
    )
    assert flagged, (
        f"{path.name} has {n} trades but no `insufficient_trades` flag — the dashboard "
        "has no way to tell the operator the numbers are not inferential."
    )
