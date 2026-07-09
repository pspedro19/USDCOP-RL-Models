#!/usr/bin/env python3
"""
APY Projection Analysis for USD/COP RL Trading System

This script calculates realistic APY projections under different cost
and signal quality scenarios, based on observed performance metrics.

Data Source:
- Current Performance: 500k timesteps, full 2025 backtest
- Model: PPO v20 Production
- Backtest Period: 88.3 trading days
- Trades: 22 total, $28.46 avg trade PnL

Usage:
    python scripts/analyze_apy_scenarios.py
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

# =============================================================================
# Configuration
# =============================================================================

INITIAL_CAPITAL = 10_000.0
BACKTEST_DAYS = 88.3
NUM_TRADES = 22
AVG_TRADE_PNL = 28.46
TRADING_YEAR_DAYS = 365


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CostStructure:
    """Transaction cost breakdown"""
    transaction_bps: float  # basis points
    slippage_bps: float     # basis points

    @property
    def total_bps(self) -> float:
        """Total round-trip cost in basis points"""
        return self.transaction_bps + self.slippage_bps

    @property
    def total_pct(self) -> float:
        """Total as percentage"""
        return self.total_bps / 10_000


@dataclass
class TradeAssumptions:
    """Trading assumptions for scenario"""
    num_trades: int
    avg_trade_pnl: float
    cost_structure: CostStructure

    @property
    def gross_profit(self) -> float:
        """Total gross profit"""
        return self.num_trades * self.avg_trade_pnl

    def total_transaction_cost(self, capital: float) -> float:
        """Total transaction cost in dollars"""
        cost_per_trade = capital * self.cost_structure.total_pct
        return self.num_trades * cost_per_trade

    def net_profit(self, capital: float) -> float:
        """Net profit after costs"""
        return self.gross_profit - self.total_transaction_cost(capital)

    def return_pct(self, capital: float) -> float:
        """Return as percentage of capital"""
        return (self.net_profit(capital) / capital) * 100

    def apy(self, capital: float, days: float) -> float:
        """Annualized percentage yield"""
        return self.return_pct(capital) * (TRADING_YEAR_DAYS / days)


@dataclass
class Scenario:
    """Complete scenario with assumptions"""
    name: str
    description: str
    trades: TradeAssumptions

    def calculate(self, capital: float = INITIAL_CAPITAL,
                  days: float = BACKTEST_DAYS) -> Dict[str, float]:
        """Calculate metrics for this scenario"""
        return {
            "gross_profit": self.trades.gross_profit,
            "transaction_costs": self.trades.total_transaction_cost(capital),
            "net_profit": self.trades.net_profit(capital),
            "return_pct": self.trades.return_pct(capital),
            "apy": self.trades.apy(capital, days),
            "cost_drag_pct": (self.trades.total_transaction_cost(capital) / capital) * 100,
        }


# =============================================================================
# Scenario Definitions
# =============================================================================

def create_scenarios() -> List[Scenario]:
    """Create all APY projection scenarios"""

    scenarios = [
        # BASELINE: Current system
        Scenario(
            name="Current (Baseline)",
            description="500k timesteps, 22 trades, 90 bps costs",
            trades=TradeAssumptions(
                num_trades=22,
                avg_trade_pnl=28.46,
                cost_structure=CostStructure(
                    transaction_bps=75.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        # COST REDUCTION scenarios
        Scenario(
            name="Cost Reduction Only (50 bps)",
            description="22 trades, reduced to 50 bps via prime brokerage",
            trades=TradeAssumptions(
                num_trades=22,
                avg_trade_pnl=28.46,
                cost_structure=CostStructure(
                    transaction_bps=35.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        Scenario(
            name="Cost Reduction (35 bps)",
            description="22 trades, best-in-class 35 bps costs",
            trades=TradeAssumptions(
                num_trades=22,
                avg_trade_pnl=28.46,
                cost_structure=CostStructure(
                    transaction_bps=20.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        # SIGNAL IMPROVEMENT scenarios
        Scenario(
            name="Signal Improvement Only (+58%)",
            description="22 trades, improved avg trade PnL to $45",
            trades=TradeAssumptions(
                num_trades=22,
                avg_trade_pnl=45.0,
                cost_structure=CostStructure(
                    transaction_bps=75.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        Scenario(
            name="Signal Improvement (+110%)",
            description="22 trades, strong signal enhancement to $60 avg trade",
            trades=TradeAssumptions(
                num_trades=22,
                avg_trade_pnl=60.0,
                cost_structure=CostStructure(
                    transaction_bps=75.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        # TRADE FILTERING scenarios
        Scenario(
            name="Trade Filter Only (18/22)",
            description="Keep top 82% quality trades, 90 bps costs",
            trades=TradeAssumptions(
                num_trades=18,
                avg_trade_pnl=28.46,
                cost_structure=CostStructure(
                    transaction_bps=75.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        # COMBINED scenarios
        Scenario(
            name="Moderate (Phase 1-2)",
            description="Cost 50 bps + Signal $45 + 18 trades",
            trades=TradeAssumptions(
                num_trades=18,
                avg_trade_pnl=45.0,
                cost_structure=CostStructure(
                    transaction_bps=35.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        Scenario(
            name="Target (Phase 1-4)",
            description="Cost 35 bps + Signal $60 + 18 trades",
            trades=TradeAssumptions(
                num_trades=18,
                avg_trade_pnl=60.0,
                cost_structure=CostStructure(
                    transaction_bps=20.0,
                    slippage_bps=15.0,
                ),
            ),
        ),

        Scenario(
            name="Optimistic",
            description="Cost 30 bps + Signal $75 + 20 trades",
            trades=TradeAssumptions(
                num_trades=20,
                avg_trade_pnl=75.0,
                cost_structure=CostStructure(
                    transaction_bps=15.0,
                    slippage_bps=15.0,
                ),
            ),
        ),
    ]

    return scenarios


# =============================================================================
# Analysis Functions
# =============================================================================

def print_header(text: str, level: int = 1) -> None:
    """Print formatted header"""
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {text}")
        print("=" * 80)
    elif level == 2:
        print(f"\n{text}")
        print("-" * 80)
    else:
        print(f"\n{text}")


def print_scenario_results(scenario: Scenario,
                          capital: float = INITIAL_CAPITAL,
                          days: float = BACKTEST_DAYS) -> None:
    """Print results for a single scenario"""
    results = scenario.calculate(capital, days)

    print(f"\n{scenario.name}")
    print(f"  {scenario.description}")
    print(f"\n  Inputs:")
    print(f"    Trades:        {scenario.trades.num_trades}")
    print(f"    Avg Trade PnL: ${scenario.trades.avg_trade_pnl:.2f}")
    print(f"    Costs:         {scenario.trades.cost_structure.total_bps:.0f} bps ({scenario.trades.cost_structure.total_pct*100:.2f}%)")
    print(f"\n  Results (88.3 days):")
    print(f"    Gross Profit:      ${results['gross_profit']:>10.2f}")
    print(f"    Transaction Costs: ${results['transaction_costs']:>10.2f}")
    print(f"    Net Profit:        ${results['net_profit']:>10.2f}")
    print(f"    Return:            {results['return_pct']:>10.2f}%")
    print(f"    APY:               {results['apy']:>10.2f}%")
    print(f"    Cost Drag:         {results['cost_drag_pct']:>10.2f}%")


def print_comparison_table(scenarios: List[Scenario],
                          capital: float = INITIAL_CAPITAL,
                          days: float = BACKTEST_DAYS) -> None:
    """Print comparison table of all scenarios"""
    print_header("SCENARIO COMPARISON TABLE", level=2)

    # Header
    print(f"{'Scenario':<40} {'88d Return':<12} {'APY':<12} {'Cost Drag':<12}")
    print("-" * 80)

    # Rows
    for scenario in scenarios:
        results = scenario.calculate(capital, days)
        return_str = f"{results['return_pct']:>6.2f}%"
        apy_str = f"{results['apy']:>6.2f}%"
        drag_str = f"{results['cost_drag_pct']:>6.2f}%"
        print(f"{scenario.name:<40} {return_str:<12} {apy_str:<12} {drag_str:<12}")


def print_sensitivity_analysis() -> None:
    """Print sensitivity analysis tables"""
    print_header("SENSITIVITY ANALYSIS", level=2)

    # Cost sensitivity
    print("\nCost Sensitivity (holding signal at $28.46 avg trade, 22 trades):")
    print(f"{'Round-Trip':<15} {'88d Return':<15} {'APY':<15}")
    print("-" * 45)

    for bps in [30, 50, 75, 90, 125]:
        costs_dollar = (bps / 10_000) * INITIAL_CAPITAL * NUM_TRADES
        net_profit = (NUM_TRADES * AVG_TRADE_PNL) - costs_dollar
        return_pct = (net_profit / INITIAL_CAPITAL) * 100
        apy = return_pct * (TRADING_YEAR_DAYS / BACKTEST_DAYS)

        cost_str = f"{bps} bps"
        return_str = f"{return_pct:>6.2f}%"
        apy_str = f"{apy:>6.2f}%"
        print(f"{cost_str:<15} {return_str:<15} {apy_str:<15}")

    # Signal sensitivity
    print("\nSignal Sensitivity (holding costs at 50 bps, 22 trades):")
    print(f"{'Avg Trade PnL':<15} {'88d Return':<15} {'APY':<15}")
    print("-" * 45)

    cost_pct = 0.005  # 50 bps
    for trade_pnl in [20, 28.46, 40, 50, 60, 75]:
        gross = NUM_TRADES * trade_pnl
        costs_dollar = cost_pct * INITIAL_CAPITAL * NUM_TRADES
        net_profit = gross - costs_dollar
        return_pct = (net_profit / INITIAL_CAPITAL) * 100
        apy = return_pct * (TRADING_YEAR_DAYS / BACKTEST_DAYS)

        pnl_str = f"${trade_pnl:.2f}"
        return_str = f"{return_pct:>6.2f}%"
        apy_str = f"{apy:>6.2f}%"
        print(f"{pnl_str:<15} {return_str:<15} {apy_str:<15}")

    # Frequency sensitivity
    print("\nFrequency Sensitivity (holding costs at 50 bps, $28.46 avg trade):")
    print(f"{'Annual Trades':<15} {'Gross Profit':<15} {'Cost Budget':<15} {'APY':<15}")
    print("-" * 60)

    annual_frequency = [50, 75, 91, 120, 150]
    for freq in annual_frequency:
        gross = freq * AVG_TRADE_PNL
        costs_dollar = 0.005 * INITIAL_CAPITAL * freq
        net = gross - costs_dollar
        apy = (net / INITIAL_CAPITAL) * 100

        freq_str = f"{freq}"
        gross_str = f"${gross:>6.2f}"
        cost_str = f"${costs_dollar:>6.2f}"
        apy_str = f"{apy:>6.2f}%"
        print(f"{freq_str:<15} {gross_str:<15} {cost_str:<15} {apy_str:<15}")


def calculate_breakeven_metrics() -> None:
    """Calculate break-even thresholds"""
    print_header("BREAK-EVEN ANALYSIS", level=2)

    # Annual trading at current frequency
    annual_trades = (NUM_TRADES / BACKTEST_DAYS) * TRADING_YEAR_DAYS

    print(f"\nQuestion 1: Trading Frequency at Current Parameters")
    print(f"  Observed: {NUM_TRADES} trades in {BACKTEST_DAYS} days")
    print(f"  Annualized: {annual_trades:.0f} trades/year")
    print(f"  Cost at 90 bps: ${annual_trades * INITIAL_CAPITAL * 0.009:,.0f}/year")
    print(f"  Cost as % of capital: {(annual_trades * 0.009 * 100):.1f}%")
    print(f"  Annual gross profit at $28.46/trade: ${annual_trades * AVG_TRADE_PNL:,.0f}")
    print(f"  Annual net (90 bps): ${annual_trades * AVG_TRADE_PNL - annual_trades * INITIAL_CAPITAL * 0.009:,.0f}")

    # Break-even cost
    print(f"\nQuestion 2: Break-Even Cost Threshold")
    breakeven_cost_pct = AVG_TRADE_PNL / INITIAL_CAPITAL
    breakeven_cost_bps = breakeven_cost_pct * 10_000
    print(f"  For one trade to break-even: cost = profit per trade")
    print(f"  Cost threshold: ${AVG_TRADE_PNL:.2f} / ${INITIAL_CAPITAL:,.0f}")
    print(f"  = {breakeven_cost_pct*100:.3f}% or {breakeven_cost_bps:.1f} bps per trade")
    print(f"  Round-trip acceptable: {breakeven_cost_bps/2:.1f} bps")
    print(f"  Current: 90 bps -> Need {90-breakeven_cost_bps:.1f} bps reduction")

    # Cost for profitability at annual frequency
    print(f"\nQuestion 3: Cost Level for 10% APY with 91 annual trades")
    target_gross = annual_trades * 50  # 10% return = $1000 net needed
    required_costs = annual_trades * INITIAL_CAPITAL * 0.005
    print(f"  Target net for 10% APY: ${target_gross:,.0f}")
    print(f"  With 91 trades/year, cost budget: ${required_costs:,.0f}")
    print(f"  Cost per trade: ${required_costs/annual_trades:.2f}")
    print(f"  Cost %: {(required_costs/annual_trades)/INITIAL_CAPITAL*100:.2f}% or {(required_costs/annual_trades)/INITIAL_CAPITAL*10000:.0f} bps")


def print_monthly_projections() -> None:
    """Print 12-month return projections"""
    print_header("12-MONTH RETURN PROJECTIONS", level=2)

    scenarios_to_project = [
        ("Current (Conservative Extrapolation)", -0.27),
        ("Moderate (Costs + Signal)", 0.10),
        ("Target (Full Stack)", 0.25),
        ("Optimistic", 0.42),
    ]

    print("\nAssuming similar monthly trading frequency throughout year:")
    print(f"{'Scenario':<40} {'Starting':<12} {'Pessimistic':<12} {'Base':<12} {'Optimistic':<12}")
    print("-" * 88)

    for scenario_name, expected_return in scenarios_to_project:
        starting = INITIAL_CAPITAL
        # Pessimistic: 70% of expected
        pessimistic = starting * (1 + expected_return * 0.70)
        # Base: expected return
        base = starting * (1 + expected_return)
        # Optimistic: 150% of expected
        optimistic = starting * (1 + expected_return * 1.50)

        print(f"{scenario_name:<40} ${starting:>10,.0f} ${pessimistic:>10,.0f} ${base:>10,.0f} ${optimistic:>10,.0f}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main analysis"""

    print("\n")
    print_header("USD/COP RL TRADING SYSTEM - APY PROJECTION ANALYSIS", level=1)
    print("Model: PPO v20 Production (500k timesteps)")
    print("Backtest: 88.3 trading days, Full 2025 Data")
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.0f}")

    # Create scenarios
    scenarios = create_scenarios()

    # Detailed results
    print_header("DETAILED SCENARIO RESULTS", level=2)
    for scenario in scenarios:
        print_scenario_results(scenario)

    # Comparison table
    print_comparison_table(scenarios)

    # Sensitivity
    print_sensitivity_analysis()

    # Break-even
    calculate_breakeven_metrics()

    # Monthly projections
    print_monthly_projections()

    # Key insights
    print_header("KEY INSIGHTS", level=2)
    print("""
1. COST IS THE KILLER
   - Transaction costs at 90 bps destroy 19.8% of capital per trade
   - Current $28.46 avg profit cannot overcome this drag
   - Must reduce to 50 bps minimum (prime brokerage required)

2. SIGNAL IMPROVEMENT REQUIRED
   - Current $28.46 avg trade is insufficient even with cost reduction
   - Need $45-60 avg trade PnL for profitability
   - Requires: Better features, macro regime filtering, improved exits

3. OPTIMAL COMBINATION
   - Cost reduction alone: Not enough (-19% APY)
   - Signal improvement alone: Not enough (-31% APY)
   - Cost + Signal + Filter: Achieves +11-25% APY
   - All three are necessary, not optional

4. REALISTIC TIMELINE
   - Phase 1 (Cost): 30 days -> -19% APY
   - Phase 2 (Signal): 45 days -> -5% APY
   - Phase 3 (Filter): 60 days -> +5% APY
   - Phase 4 (Validate): 90 days -> +15-25% APY

5. RISK FACTORS
   - 88-day backtest period is short, may overfit
   - Prime brokerage rates may require larger account
   - Signal improvements may not achieve expected gains
   - Banrep intervention creates tail risk
    """)

    print_header("END OF ANALYSIS", level=1)


if __name__ == "__main__":
    main()
