"""Fail-closed loader for the executable XAU/USD cost contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "config" / "research" / "gold_cost_contract.yaml"


def load_gold_cost_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if payload.get("contract") != "CTR-RESEARCH-XAUUSD-COST-001":
        raise ValueError("invalid XAU/USD cost contract id")
    return payload


def require_executable_gold_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    """Return only a complete, venue-backed contract; otherwise fail closed."""
    payload = load_gold_cost_contract(path)
    if payload.get("status") != "VERIFIED":
        raise RuntimeError("XAU/USD cost contract is not verified by a venue")
    instrument = payload.get("instrument", {})
    costs = payload.get("costs", {})
    required = [instrument.get("venue"), instrument.get("tick_size"),
                instrument.get("contract_multiplier"), costs.get("spread"),
                costs.get("commission"), costs.get("slippage"),
                costs.get("swap_long_annual"), costs.get("swap_short_annual")]
    if any(value is None for value in required):
        raise RuntimeError("XAU/USD cost contract has incomplete execution fields")
    obs = payload.get("observability", {})
    if not obs.get("bid_ask_history") or not obs.get("fills_history"):
        raise RuntimeError("XAU/USD contract lacks bid/ask and fill evidence")
    return payload
