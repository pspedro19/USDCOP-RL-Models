import pytest

from src.gold_rl.cost_contract import load_gold_cost_contract, require_executable_gold_contract


def test_gold_contract_is_explicitly_pending_until_venue_is_verified():
    payload = load_gold_cost_contract()
    assert payload["instrument"]["symbol"] == "XAU/USD"
    assert payload["status"] == "PENDING_VENUE"
    with pytest.raises(RuntimeError, match="not verified"):
        require_executable_gold_contract()


def test_gold_contract_rejects_incomplete_verified_contract(tmp_path):
    path = tmp_path / "gold.yaml"
    path.write_text(
        "contract: CTR-RESEARCH-XAUUSD-COST-001\nstatus: VERIFIED\n"
        "instrument: {venue: demo, tick_size: 0.01, contract_multiplier: 1}\n"
        "costs: {spread: 1, commission: 1, slippage: 1, swap_long_annual: 0.1, swap_short_annual: 0.1}\n"
        "observability: {bid_ask_history: false, fills_history: false}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lacks bid/ask"):
        require_executable_gold_contract(path)
