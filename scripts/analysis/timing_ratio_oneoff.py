"""One-off timing/beta attribution for the four current champion sleeves.

This is the deliberately non-durable Stage 0.5 diagnostic from
CTR-QLAB-FABRIC-004 sections 18.2, 18.3, and 28. It reads the existing
profitability adapters, prints one JSON document, and writes nothing.

For weight ``w_t`` and asset return ``r_t``:

    gross_t  = w_t * r_t
    beta_t   = mean(w) * r_t
    timing_t = (w_t - mean(w)) * r_t

Thus ``gross_t == beta_t + timing_t``. Also, because the centered weights
sum to zero, ``sum(timing_t) == N * cov_population(w, r)``. The timing ratio
is ``sum(timing_t) / sum(abs(gross_t))``.

The confidence interval is a paired circular-block bootstrap. Its defaults
reuse the repository's published cross-asset convention rather than selecting
new values after seeing this diagnostic: 20 daily observations per block,
5,000 resamples, seed 42 (``.claude/specs/assets/_strategy-science.md`` and
the existing BTC/Gold backtests). They remain configurable only for unit tests
and operational constraints, never by inspecting results.

This attribution is DIAGNOSTIC ONLY. It is model/benchmark dependent and is
not evidence of alpha; the repository's alpha claim remains the forward DSR.

Run:
    python -m scripts.analysis.timing_ratio_oneoff
    python -m scripts.analysis.timing_ratio_oneoff --include-gold-control
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts.strategy_schema import MIN_TRADES_FOR_STATS  # noqa: E402


# Fixed before seeing results. These are computation settings, not strategy/model choices.
DEFAULT_BLOCK_SIZE = 20
DEFAULT_BOOTSTRAP_SAMPLES = 5_000
DEFAULT_SEED = 42
CI_LEVEL = 0.95
STATISTICAL_CONVENTION_SOURCE = (
    ".claude/specs/assets/_strategy-science.md:161; "
    "src/btc_strategy/backtest.py:68-72; src/gold_rl/backtest.py:63-67"
)

# Strategy id -> existing adapter key. Keep this tuple fixed so the optional control can
# never replace or silently remove one of the four champions.
CHAMPION_ADAPTERS = (
    ("smart_simple_v11", "usdcop"),
    ("gold_trend_simple", "gold_trend_simple"),
    ("btc_hodl_b1", "btc_hodl_b1"),
    ("spx500_regime_gated_v1", "spx500_regime_gated_v1"),
)
GOLD_CONTROL = ("gold_dynamic_exit", "xauusd")
UNAVAILABLE_CHAMPIONS: dict[str, dict[str, str]] = {
    "smart_simple_v11": {
        "asset": "usdcop",
        "reason": (
            "The current COP adapter does not expose underlying USD/COP returns; "
            "its asset_ret is already the strategy's net weekly published return. "
            "Multiplying it by position again would fabricate attribution."
        ),
        "evidence": "scripts/analysis/profitability_adapters.py:147-164",
        "required_fix": (
            "Join the point-in-time underlying weekly USD/COP return before computing "
            "a numeric timing_ratio."
        ),
    }
}

DIAGNOSTIC_CAVEAT = (
    "DIAGNOSTIC ONLY: timing_ratio is model/benchmark-dependent attribution, "
    "not evidence of alpha and not a promotion decision; the alpha claim remains "
    "the forward DSR."
)


def _finite_float(value: float, *, name: str) -> float:
    """Return a plain finite float so json.dumps(..., allow_nan=False) cannot leak NaN/Inf."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {result!r}")
    return result


def _validate_settings(block_size: int, bootstrap_samples: int) -> None:
    if block_size <= 0:
        raise ValueError("--block-size must be a positive integer")
    if bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be a positive integer")


def _paired_finite_series(sleeve: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract aligned finite weight/return pairs without inventing missing observations."""
    weights = np.asarray(sleeve.position, dtype=float)
    returns = np.asarray(sleeve.asset_ret, dtype=float)
    index = np.asarray(sleeve.index)

    if weights.ndim != 1 or returns.ndim != 1 or index.ndim != 1:
        raise ValueError(f"{sleeve.strategy_id}: adapter series must be one-dimensional")
    if not (len(weights) == len(returns) == len(index)):
        raise ValueError(
            f"{sleeve.strategy_id}: adapter lengths differ "
            f"(weight={len(weights)}, return={len(returns)}, index={len(index)})"
        )

    finite = np.isfinite(weights) & np.isfinite(returns)
    excluded = int(len(weights) - finite.sum())
    if int(finite.sum()) < 2:
        raise ValueError(f"{sleeve.strategy_id}: fewer than two finite weight/return pairs")
    return weights[finite], returns[finite], index[finite], excluded


def _timing_ratio(weights: np.ndarray, returns: np.ndarray) -> float:
    """Compute the requested statistic, recomputing mean weight for this exact sample."""
    mean_weight = float(np.mean(weights))
    gross = weights * returns
    denominator = float(np.sum(np.abs(gross), dtype=float))
    if denominator <= 0.0:
        raise ValueError("sum(abs(gross_pnl)) is zero; timing_ratio is undefined")
    numerator = float(np.sum((weights - mean_weight) * returns, dtype=float))
    return _finite_float(numerator / denominator, name="timing_ratio")


def block_bootstrap_ci(
    weights: np.ndarray,
    returns: np.ndarray,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, float | int | str]:
    """Deterministic paired circular-block percentile CI for ``timing_ratio``.

    Each replicate samples contiguous blocks of paired (weight, return) observations,
    wraps at the end of the series, truncates to the original length, and recomputes
    the replicate's mean weight. Replicates with zero gross denominator are skipped
    rather than serialized as NaN; failure to obtain any valid replicate is an error.
    """
    _validate_settings(block_size, bootstrap_samples)
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    if weights.ndim != 1 or returns.ndim != 1 or len(weights) != len(returns):
        raise ValueError("weights and returns must be aligned one-dimensional arrays")
    if len(weights) < 2:
        raise ValueError("block bootstrap requires at least two observations")
    if not (np.isfinite(weights).all() and np.isfinite(returns).all()):
        raise ValueError("block bootstrap input must contain only finite observations")

    n_obs = len(weights)
    effective_block_size = min(block_size, n_obs)
    blocks_per_sample = math.ceil(n_obs / effective_block_size)
    within_block = np.arange(effective_block_size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    ratios: list[float] = []

    for _ in range(bootstrap_samples):
        starts = rng.integers(0, n_obs, size=blocks_per_sample)
        sample_index = ((starts[:, None] + within_block) % n_obs).reshape(-1)[:n_obs]
        sampled_weights = weights[sample_index]
        sampled_returns = returns[sample_index]
        gross_denominator = float(np.sum(np.abs(sampled_weights * sampled_returns)))
        if gross_denominator <= 0.0:
            continue
        ratios.append(_timing_ratio(sampled_weights, sampled_returns))

    if not ratios:
        raise ValueError("all bootstrap replicates had zero gross denominator")

    alpha = (1.0 - CI_LEVEL) / 2.0
    low, high = np.quantile(np.asarray(ratios, dtype=float), [alpha, 1.0 - alpha])
    return {
        "method": "paired_circular_block_percentile",
        "confidence_level": CI_LEVEL,
        "lower": _finite_float(low, name="bootstrap_ci.lower"),
        "upper": _finite_float(high, name="bootstrap_ci.upper"),
        "configured_block_size_observations": int(block_size),
        "effective_block_size_observations": int(effective_block_size),
        "requested_samples": int(bootstrap_samples),
        "valid_samples": len(ratios),
        "seed": int(seed),
    }


def analyse_sleeve(
    sleeve: Any,
    *,
    expected_strategy_id: str,
    adapter_key: str,
    role: str,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Build one finite, JSON-safe attribution result."""
    if sleeve.strategy_id != expected_strategy_id:
        raise ValueError(
            f"adapter {adapter_key!r} returned {sleeve.strategy_id!r}; "
            f"expected {expected_strategy_id!r}"
        )

    weights, returns, index, excluded = _paired_finite_series(sleeve)
    invalid_baselines = tuple(getattr(sleeve, "invalid_baselines", ()))
    if invalid_baselines:
        return {
            "strategy_id": sleeve.strategy_id,
            "asset": sleeve.asset,
            "role": role,
            "adapter_key": adapter_key,
            "clock": sleeve.clock_label,
            "attribution_status": "UNAVAILABLE",
            "n_observations": len(weights),
            "excluded_nonfinite_pairs": excluded,
            "start": str(index[0]),
            "end": str(index[-1]),
            "unavailable_reason": (
                "The adapter does not expose underlying market returns; asset_ret is "
                "already the strategy's net published return. Multiplying it by position "
                "again would fabricate timing/beta attribution."
            ),
            "adapter_limitation": {
                "invalid_baselines": list(invalid_baselines),
                "required_fix": (
                    "Join the point-in-time underlying market return before computing "
                    "a numeric timing_ratio."
                ),
            },
            "caveat": DIAGNOSTIC_CAVEAT,
        }

    mean_weight = float(np.mean(weights))
    gross = weights * returns
    pnl_beta = mean_weight * returns
    pnl_timing = (weights - mean_weight) * returns

    sum_gross = float(np.sum(gross, dtype=float))
    sum_beta = float(np.sum(pnl_beta, dtype=float))
    sum_timing = float(np.sum(pnl_timing, dtype=float))
    sum_abs_gross = float(np.sum(np.abs(gross), dtype=float))
    raw_n_trades = getattr(sleeve, "n_trades", None)
    n_trades = int(raw_n_trades) if raw_n_trades is not None else None

    # Constitutional small-sample rule: N<20 permits counts and PnL only.
    # Unknown N is suppressed too; absence of evidence cannot unlock inference.
    if n_trades is None or n_trades < MIN_TRADES_FOR_STATS:
        return {
            "strategy_id": sleeve.strategy_id,
            "asset": sleeve.asset,
            "role": role,
            "adapter_key": adapter_key,
            "clock": sleeve.clock_label,
            "attribution_status": "INSUFFICIENT_TRADES",
            "n_trades": n_trades,
            "min_trades_for_stats": MIN_TRADES_FOR_STATS,
            "n_observations": len(weights),
            "excluded_nonfinite_pairs": excluded,
            "start": str(index[0]),
            "end": str(index[-1]),
            "sum_gross_pnl": _finite_float(sum_gross, name="sum_gross_pnl"),
            "sum_pnl_beta": _finite_float(sum_beta, name="sum_pnl_beta"),
            "sum_pnl_timing": _finite_float(sum_timing, name="sum_pnl_timing"),
            "sum_abs_gross_pnl": _finite_float(
                sum_abs_gross, name="sum_abs_gross_pnl"
            ),
            "statistics_suppressed": [
                "timing_ratio",
                "timing_ratio_ci",
                "population_covariance_weight_return",
            ],
            "small_sample_reason": (
                "N<20 (or unknown) permits counts and PnL only; attribution "
                "ratios, covariance, and intervals are suppressed."
            ),
            "caveat": DIAGNOSTIC_CAVEAT,
        }

    if sum_abs_gross <= 0.0:
        raise ValueError(f"{sleeve.strategy_id}: sum(abs(gross_pnl)) is zero")

    population_covariance = float(np.mean((weights - mean_weight) * (returns - returns.mean())))
    identity_residual = sum_gross - sum_beta - sum_timing
    covariance_residual = sum_timing - len(weights) * population_covariance

    result: dict[str, Any] = {
        "strategy_id": sleeve.strategy_id,
        "asset": sleeve.asset,
        "role": role,
        "adapter_key": adapter_key,
        "clock": sleeve.clock_label,
        "attribution_status": "AVAILABLE",
        "n_trades": n_trades,
        "min_trades_for_stats": MIN_TRADES_FOR_STATS,
        "n_observations": len(weights),
        "excluded_nonfinite_pairs": excluded,
        "start": str(index[0]),
        "end": str(index[-1]),
        "mean_weight": _finite_float(mean_weight, name="mean_weight"),
        "sum_gross_pnl": _finite_float(sum_gross, name="sum_gross_pnl"),
        "sum_pnl_beta": _finite_float(sum_beta, name="sum_pnl_beta"),
        "sum_pnl_timing": _finite_float(sum_timing, name="sum_pnl_timing"),
        "sum_abs_gross_pnl": _finite_float(sum_abs_gross, name="sum_abs_gross_pnl"),
        "timing_ratio": _finite_float(sum_timing / sum_abs_gross, name="timing_ratio"),
        "population_covariance_weight_return": _finite_float(
            population_covariance, name="population_covariance_weight_return"
        ),
        "n_times_population_covariance": _finite_float(
            len(weights) * population_covariance,
            name="n_times_population_covariance",
        ),
        "gross_equals_beta_plus_timing_residual": _finite_float(
            identity_residual, name="gross_equals_beta_plus_timing_residual"
        ),
        "timing_sum_equals_n_covariance_residual": _finite_float(
            covariance_residual, name="timing_sum_equals_n_covariance_residual"
        ),
        "timing_ratio_ci": block_bootstrap_ci(
            weights,
            returns,
            block_size=block_size,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        ),
        "caveat": DIAGNOSTIC_CAVEAT,
    }

    return result


def declared_unavailable_result(
    *,
    strategy_id: str,
    adapter_key: str,
    role: str,
) -> dict[str, Any]:
    """Return an explicit non-numeric result for a documented missing input."""
    limitation = UNAVAILABLE_CHAMPIONS[strategy_id]
    return {
        "strategy_id": strategy_id,
        "asset": limitation["asset"],
        "role": role,
        "adapter_key": adapter_key,
        "attribution_status": "UNAVAILABLE",
        "unavailable_reason": limitation["reason"],
        "evidence": limitation["evidence"],
        "required_fix": limitation["required_fix"],
        "caveat": DIAGNOSTIC_CAVEAT,
    }


def build_report(
    adapters: Mapping[str, Any],
    *,
    include_gold_control: bool,
    block_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Run all four champions and optionally append (never substitute) the manual control."""
    _validate_settings(block_size, bootstrap_samples)
    jobs = [(strategy_id, key, "champion") for strategy_id, key in CHAMPION_ADAPTERS]
    warnings: list[str] = []
    control_available = GOLD_CONTROL[1] in adapters
    if include_gold_control:
        if control_available:
            jobs.append((*GOLD_CONTROL, "manual_control"))
        else:
            warnings.append(
                "gold_dynamic_exit control skipped because adapter key 'xauusd' is unavailable"
            )

    results: list[dict[str, Any]] = []
    for strategy_id, adapter_key, role in jobs:
        if strategy_id in UNAVAILABLE_CHAMPIONS:
            results.append(
                declared_unavailable_result(
                    strategy_id=strategy_id,
                    adapter_key=adapter_key,
                    role=role,
                )
            )
            continue
        if adapter_key not in adapters:
            raise KeyError(f"required adapter key {adapter_key!r} is unavailable")
        sleeve = adapters[adapter_key]()
        results.append(
            analyse_sleeve(
                sleeve,
                expected_strategy_id=strategy_id,
                adapter_key=adapter_key,
                role=role,
                block_size=block_size,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        )

    champion_ids = [row["strategy_id"] for row in results if row["role"] == "champion"]
    expected_ids = [strategy_id for strategy_id, _ in CHAMPION_ADAPTERS]
    if champion_ids != expected_ids:
        raise AssertionError("the report did not preserve the four fixed champion sleeves")

    return {
        "schema_version": "timing-ratio-oneoff/1",
        "stage": "0.5",
        "diagnostic_only": True,
        "promotion_eligible": False,
        "trials_charged": 0,
        "persisted": False,
        "caveat": DIAGNOSTIC_CAVEAT,
        "method": {
            "gross_pnl": "weight * asset_return",
            "pnl_beta": "mean(weight) * asset_return",
            "pnl_timing": "(weight - mean(weight)) * asset_return",
            "timing_ratio": "sum(pnl_timing) / sum(abs(gross_pnl))",
            "covariance_convention": (
                "population covariance (ddof=0); sum(pnl_timing) = "
                "n_observations * covariance(weight, asset_return)"
            ),
            "bootstrap": (
                "paired circular blocks; mean weight and timing_ratio recomputed per sample"
            ),
            "settings_policy": (
                "Repository-published technical defaults, configurable only for tests or "
                "operational constraints; never select settings by observed results."
            ),
            "settings_source": STATISTICAL_CONVENTION_SOURCE,
        },
        "settings": {
            "block_size_observations": int(block_size),
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
            "confidence_level": CI_LEVEL,
        },
        "champions_required": expected_ids,
        "gold_control_requested": bool(include_gold_control),
        "gold_control_available": bool(control_available),
        "warnings": warnings,
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print non-durable timing/beta attribution for the four champions as strict JSON."
        )
    )
    parser.add_argument(
        "--include-gold-control",
        action="store_true",
        help=(
            "append gold_dynamic_exit as a manual beta-disguise control; "
            "the four champions still run"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        metavar="OBS",
        help=(
            f"contiguous observations per circular block (default: {DEFAULT_BLOCK_SIZE}; "
            "fixed technical convention, not result-selected)"
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        metavar="N",
        help=(
            f"bootstrap resamples (default: {DEFAULT_BOOTSTRAP_SAMPLES}; "
            "fixed technical convention, not result-selected)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"deterministic bootstrap seed (default: {DEFAULT_SEED})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_settings(args.block_size, args.bootstrap_samples)

    # Importing here keeps pure attribution/bootstrap functions cheap to unit-test.
    from scripts.analysis.profitability_adapters import ADAPTERS

    report = build_report(
        ADAPTERS,
        include_gold_control=args.include_gold_control,
        block_size=args.block_size,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
