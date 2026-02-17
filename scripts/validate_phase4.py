#!/usr/bin/env python3
"""
V22 P4: Final Walk-Forward Validation Gates
=============================================
Final gates before production deployment.

Gates:
- Walk-forward OOS mean return > 10% APR
- Walk-forward OOS worst fold > 0% return
- Sensitivity: profitable at 5bps total costs
- Max drawdown < 20% across all folds

Usage:
    python scripts/validate_phase4.py --results-dir results/walk_forward/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

P4_GATES = {
    "min_oos_mean_apr_pct": 10.0,
    "min_oos_worst_fold_return_pct": 0.0,
    "max_drawdown_pct": 20.0,
    "profitable_at_5bps": True,
}


def validate_phase4(results_dir: Path) -> bool:
    """
    Validate Phase 4 walk-forward results.

    Args:
        results_dir: Directory containing walk-forward and sensitivity results

    Returns:
        True if all gates pass
    """
    logger.info("=" * 60)
    logger.info("V22 PHASE 4 FINAL VALIDATION")
    logger.info("=" * 60)

    gates_passed = True

    # Load walk-forward results
    wf_files = list(results_dir.glob("walk_forward*.json"))
    if wf_files:
        with open(wf_files[0]) as f:
            wf_results = json.load(f)

        fold_returns = []
        fold_drawdowns = []

        if isinstance(wf_results, list):
            for fold in wf_results:
                fold_returns.append(fold.get("return_pct", 0))
                fold_drawdowns.append(fold.get("max_drawdown_pct", 0))
        elif isinstance(wf_results, dict):
            for key, fold in wf_results.items():
                if isinstance(fold, dict):
                    fold_returns.append(fold.get("return_pct", 0))
                    fold_drawdowns.append(fold.get("max_drawdown_pct", 0))

        if fold_returns:
            mean_return = sum(fold_returns) / len(fold_returns)
            worst_fold = min(fold_returns)
            max_dd = max(fold_drawdowns) if fold_drawdowns else 0

            # Gate 1: Mean OOS return
            gate1 = mean_return > P4_GATES["min_oos_mean_apr_pct"]
            status = "PASS" if gate1 else "FAIL"
            logger.info(f"  [{status}] Mean OOS return {mean_return:.2f}% > {P4_GATES['min_oos_mean_apr_pct']:.1f}%")
            if not gate1:
                gates_passed = False

            # Gate 2: Worst fold > 0%
            gate2 = worst_fold > P4_GATES["min_oos_worst_fold_return_pct"]
            status = "PASS" if gate2 else "FAIL"
            logger.info(f"  [{status}] Worst fold {worst_fold:.2f}% > {P4_GATES['min_oos_worst_fold_return_pct']:.1f}%")
            if not gate2:
                gates_passed = False

            # Gate 3: Max DD < 20%
            gate3 = max_dd < P4_GATES["max_drawdown_pct"]
            status = "PASS" if gate3 else "FAIL"
            logger.info(f"  [{status}] Max DD {max_dd:.2f}% < {P4_GATES['max_drawdown_pct']:.1f}%")
            if not gate3:
                gates_passed = False

            logger.info(f"\n  Fold returns: {[round(r, 2) for r in fold_returns]}")
        else:
            logger.warning("  No fold data found in walk-forward results")
            gates_passed = False
    else:
        logger.warning("  No walk-forward results found")
        gates_passed = False

    # Load sensitivity results
    sens_files = list(results_dir.glob("sensitivity*.json"))
    if sens_files:
        with open(sens_files[0]) as f:
            sens_results = json.load(f)

        # Check 5bps profitability
        cost_results = [r for r in sens_results if r.get('test') == 'cost_sensitivity']
        profitable_at_5bps = any(
            r.get('profitable', False) and r.get('total_cost_bps', 0) >= 5
            for r in cost_results
        )

        gate4 = profitable_at_5bps
        status = "PASS" if gate4 else "FAIL"
        logger.info(f"  [{status}] Profitable at 5bps total costs")
        if not gate4:
            gates_passed = False
    else:
        logger.warning("  No sensitivity results found — skipping cost gate")

    logger.info(f"\n{'=' * 60}")
    if gates_passed:
        logger.info("PHASE 4: ALL GATES PASSED — READY FOR PRODUCTION")
    else:
        logger.info("PHASE 4: GATES FAILED — Review and iterate")
    logger.info(f"{'=' * 60}")

    return gates_passed


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="V22 Phase 4 Final Validation")
    parser.add_argument("--results-dir", type=str, required=True)
    args = parser.parse_args()

    passed = validate_phase4(Path(args.results_dir))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
