"""Single source of truth for research transaction-cost assumptions.

The contract deliberately calls the historical ``spread`` values bounds or
scenarios, never observed quotes: TwelveData's current endpoint does not
provide a bid/ask stream for this study.  Any production venue must replace
these assumptions with a measured quote manifest before a profitability claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO / "config" / "research" / "cost_contract.yaml"


@dataclass(frozen=True)
class CostContract:
    unit: str
    market_tick_cop: float
    commission_per_side: float
    slippage_coef: float
    scenarios: dict[str, tuple[float, float, float]]
    source: str


def load_cost_contract(path: Path = CONTRACT_PATH) -> CostContract:
    if not path.is_file():
        raise FileNotFoundError(f"missing research cost contract: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cost contract must be a mapping")
    unit = raw.get("unit")
    if unit != "cop_per_usd":
        raise ValueError("cost contract unit must be cop_per_usd")
    tick = float(raw.get("market_tick_cop", 0.0))
    if tick <= 0:
        raise ValueError("market_tick_cop must be positive")
    scenarios_raw = raw.get("scenarios")
    if not isinstance(scenarios_raw, dict) or set(scenarios_raw) != {"low", "central", "high"}:
        raise ValueError("cost contract must declare low, central and high scenarios")
    scenarios: dict[str, tuple[float, float, float]] = {}
    for name, values in scenarios_raw.items():
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"scenario {name} must contain three regime spreads")
        parsed = tuple(float(v) for v in values)
        if any(v <= 0 for v in parsed):
            raise ValueError(f"scenario {name} spreads must be positive")
        scenarios[name] = parsed
    return CostContract(
        unit=unit,
        market_tick_cop=tick,
        commission_per_side=float(raw["commission_per_side"]),
        slippage_coef=float(raw["slippage_coef"]),
        scenarios=scenarios,
        source=str(raw.get("source", "unspecified")),
    )


COST_CONTRACT = load_cost_contract()
# Central scenario is the only one used by the frozen HMM path.  Low/high are
# sensitivity contracts and must be selected explicitly by an experiment config.
SPREAD_PIPS_BY_LEVEL: tuple[float, float, float] = COST_CONTRACT.scenarios["central"]
COMMISSION_PIPS_PER_SIDE = COST_CONTRACT.commission_per_side
SLIPPAGE_COEF = COST_CONTRACT.slippage_coef

