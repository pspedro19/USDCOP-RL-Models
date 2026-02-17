#!/usr/bin/env python3
"""
V22 P1: Phase 1 Validation Gates
==================================
Validates ensemble + Kelly + temporal features meet targets.

Gates:
- Return > V21.5 baseline (+1.26%)
- Sharpe > 0.3
- Max drawdown < 15%

Usage:
    python scripts/validate_phase1.py --results-file models/ppo_v22/ensemble_backtest_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

V21_5_BASELINE = {
    "total_return_pct": 1.26,
    "sharpe_ratio": 0.202,
    "max_drawdown_pct": 12.72,
    "win_rate_pct": 56.3,
}

P1_GATES = {
    "min_return_pct": 1.26,     # Must beat V21.5
    "min_sharpe": 0.3,          # Meaningful risk-adjusted return
    "max_drawdown_pct": 15.0,   # Risk ceiling
}


def validate_phase1(results_file: Path) -> bool:
    """
    Validate Phase 1 results against gates.

    Returns:
        True if all gates pass
    """
    with open(results_file) as f:
        results = json.load(f)

    logger.info("=" * 60)
    logger.info("V22 PHASE 1 VALIDATION")
    logger.info("=" * 60)

    logger.info(f"\nBaseline (V21.5):")
    for k, v in V21_5_BASELINE.items():
        logger.info(f"  {k}: {v}")

    logger.info(f"\nCurrent Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    # Check gates
    gates_passed = True
    logger.info(f"\nGate Checks:")

    # Gate 1: Return > baseline
    ret = results.get("total_return_pct", 0)
    gate1 = ret > P1_GATES["min_return_pct"]
    status = "PASS" if gate1 else "FAIL"
    logger.info(f"  [{status}] Return {ret:.2f}% > {P1_GATES['min_return_pct']:.2f}%")
    if not gate1:
        gates_passed = False

    # Gate 2: Sharpe > 0.3
    sharpe = results.get("sharpe_ratio", 0)
    gate2 = sharpe > P1_GATES["min_sharpe"]
    status = "PASS" if gate2 else "FAIL"
    logger.info(f"  [{status}] Sharpe {sharpe:.3f} > {P1_GATES['min_sharpe']:.3f}")
    if not gate2:
        gates_passed = False

    # Gate 3: Max DD < 15%
    dd = results.get("max_drawdown_pct", 100)
    gate3 = dd < P1_GATES["max_drawdown_pct"]
    status = "PASS" if gate3 else "FAIL"
    logger.info(f"  [{status}] Max DD {dd:.2f}% < {P1_GATES['max_drawdown_pct']:.2f}%")
    if not gate3:
        gates_passed = False

    # Improvement over baseline
    improvement = ret - V21_5_BASELINE["total_return_pct"]
    logger.info(f"\n  Improvement over V21.5: {improvement:+.2f}%")

    logger.info(f"\n{'=' * 60}")
    if gates_passed:
        logger.info("PHASE 1: ALL GATES PASSED - Proceed to Phase 2")
    else:
        logger.info("PHASE 1: GATES FAILED - Review and iterate")
    logger.info(f"{'=' * 60}")

    return gates_passed


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="V22 Phase 1 Validation")
    parser.add_argument("--results-file", type=str, required=True)
    args = parser.parse_args()

    passed = validate_phase1(Path(args.results_file))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
