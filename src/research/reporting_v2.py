"""Explicit economic reporting conventions for retrospective thesis evidence.

This module never trains, selects a model, reads a hold-out or certifies a source.
Bootstrap intervals are exploratory, conditional on the specified inputs.
"""
from __future__ import annotations

import numpy as np

ANN = 221.0


def vector(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('expected nonempty finite one-dimensional observations')
    return x


def capital_path(returns, initial: float = 100_000.0) -> np.ndarray:
    r = vector(returns)
    if not np.isfinite(initial) or initial <= 0 or (r <= -1).any():
        raise ValueError('positive initial capital and returns greater than -1 required')
    return initial * np.r_[1.0, np.cumprod(1.0 + r)]


def path_counts(weights) -> dict:
    """A round trip is one nonzero-sign position episode, with mandatory daily close.

    Same-sign resizing is turnover, not a newly completed position episode.
    """
    w = vector(weights)
    if (np.abs(w) > 1).any():
        raise ValueError('exposure outside [-1,1]')
    signs = np.sign(w)
    previous = np.r_[0, signs[:-1]]
    dw = np.diff(np.r_[0., w, 0.])
    return {'round_trips': int(np.count_nonzero((signs != 0) & (signs != previous))),
            'changes_including_terminal': int(np.count_nonzero(np.abs(dw) > 1e-12)),
            'changes_decisions_only': int(np.count_nonzero(np.abs(dw[:-1]) > 1e-12)),
            'turnover': float(np.abs(dw).sum()), 'active_bars': int(np.count_nonzero(w))}


def summarize(gross, costs, *, min_round_trips: int, initial: float = 100_000.0,
              count_is_lower_bound: bool = False) -> dict:
    g, c = vector(gross), vector(costs)
    if g.shape != c.shape or (c < 0).any() or min_round_trips < 0:
        raise ValueError('aligned returns, nonnegative costs and trade count required')
    r = g - c
    equity = capital_path(r, initial)
    dd = equity / np.maximum.accumulate(equity) - 1.0
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else 0.
    can_infer = min_round_trips >= 20 and len(r) >= 2 and sd > 0
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.) ** 2)))
    annual = float((equity[-1] / initial) ** (ANN / len(r)) - 1.)
    maxdd = float(dd.min())
    return {
        'n_sessions': len(r), 'round_trips_minimum': int(min_round_trips),
        'trade_count_is_lower_bound': count_is_lower_bound,
        'return_compounded_pct': float((equity[-1] / initial - 1) * 100),
        'gross_compounded_pct': float((capital_path(g)[-1] / 100_000. - 1) * 100),
        'net_sum_pct': float(r.sum() * 100), 'gross_sum_pct': float(g.sum() * 100),
        'cost_sum_pct': float(c.sum() * 100),
        'initial_capital_account_units': initial,
        'final_capital_account_units': float(equity[-1]),
        'cost_account_units': float(np.sum(equity[:-1] * c)),
        'gross_pnl_account_units': float(np.sum(equity[:-1] * g)),
        'max_drawdown_pct': maxdd * 100,
        'annualization_sessions': ANN,
        'annual_return_session_clock_pct': annual * 100,
        'sharpe': float(r.mean() / sd * np.sqrt(ANN)) if can_infer else None,
        'sortino_daily_target_zero': float(r.mean() / downside * np.sqrt(ANN))
        if can_infer and downside > 0 else None,
        'calmar_session_clock': annual / abs(maxdd) if can_infer and maxdd < 0 else None,
        'positive_sessions_fraction': float(np.mean(r > 0)),
        'trade_profit_factor': None, 'trade_win_rate': None,
        'trade_statistics_note': 'daily returns are not a closed-trade PnL ledger',
        'inference_allowed': can_infer,
        'inference_reason': None if can_infer else 'fewer than 20 trades or zero variance',
    }


def bootstrap_indices(n: int, replications: int = 10_000, seed: int = 20260912,
                      blocks=(5, 10, 15, 20)) -> np.ndarray:
    if n < 2 or replications < 2 or not blocks or min(blocks) <= 0:
        raise ValueError('invalid bootstrap configuration')
    from src.research.inference import stationary_bootstrap_indices
    rng = np.random.default_rng(seed)
    return np.stack([stationary_bootstrap_indices(n, blocks[i % len(blocks)], rng)
                     for i in range(replications)])


def sharpe_rows(x, axis=-1):
    sd = np.std(x, axis=axis, ddof=1)
    return np.divide(np.mean(x, axis=axis), sd, out=np.zeros_like(sd), where=sd > 0) * np.sqrt(ANN)


def percentile_test(samples, point: float) -> dict:
    """Two-sided percentile-bootstrap sign-tail estimate; not a calibrated exact test.

    CI and tail estimate use the SAME uncentered bootstrap distribution.
    Sampling uncertainty, multiplicity and retrospective selection remain explicit.
    """
    s = vector(samples)
    lower, upper = int(np.sum(s <= 0)), int(np.sum(s >= 0))
    return {'estimate': float(point), 'ci95': list(map(float, np.percentile(s, [2.5, 97.5]))),
            'p_bootstrap_percentile': min(1., 2. * (min(lower, upper) + 1) / (len(s) + 1)),
            'tail_count': min(lower, upper), 'replications': len(s),
            'mc_floor': 2. / (len(s) + 1), 'confirmatory': False,
            'method': 'stationary block percentile bootstrap, exploratory sign-tail estimate'}


def paired_difference(a, b, *, min_trades_a: int, min_trades_b: int,
                      metric: str = 'mean_daily', flat_reference: bool = False,
                      replications: int = 10_000) -> dict:
    x, y = vector(a), vector(b)
    if x.shape != y.shape:
        raise ValueError('paired series must align')
    if min_trades_a < 20 or (min_trades_b < 20 and not (flat_reference and np.all(y == 0))):
        return {'status': 'SUPPRESSED', 'reason': 'fewer than 20 completed trades'}
    idx = bootstrap_indices(len(x), replications)
    if metric == 'mean_daily':
        samples, point = (x[idx] - y[idx]).mean(axis=1), float(np.mean(x - y))
    elif metric == 'sharpe' and not flat_reference:
        samples = sharpe_rows(x[idx]) - sharpe_rows(y[idx])
        point = float(sharpe_rows(x) - sharpe_rows(y))
    else:
        raise ValueError('undefined Sharpe of flat: use mean_daily as primary comparison')
    result = {**percentile_test(samples, point), 'metric': metric}
    if metric == 'mean_daily':
        # Resample centered differences under H0 E[a-b]=0; two-sided absolute
        # statistic, plus-one Monte Carlo correction. Still retrospective and
        # dependent on stationary block-bootstrap assumptions, never exact.
        exceedances = int(np.sum(np.abs(samples - point) >= abs(point)))
        result.update(p_centered_stationary=(exceedances + 1) / (replications + 1),
                      null_exceedances=exceedances, null_mc_floor=1 / (replications + 1),
                      null_test='two-sided mean difference, centered stationary bootstrap')
    return result


def hierarchical_sharpe(a, b, *, replications: int = 10_000) -> dict:
    """Paired training seeds and common stationary session blocks, both resampled."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 5 or not np.isfinite([x, y]).all():
        raise ValueError('at least five aligned finite paired seeds required')
    from src.research.inference import stationary_bootstrap_indices
    rng = np.random.default_rng(20260912)
    portfolio, seed_metric = [], []
    for i in range(replications):
        seeds = rng.integers(0, len(x), len(x))
        idx = stationary_bootstrap_indices(x.shape[1], (5, 10, 15, 20)[i % 4], rng)
        xa, ya = x[seeds][:, idx], y[seeds][:, idx]
        portfolio.append(float(sharpe_rows(xa.mean(axis=0)) - sharpe_rows(ya.mean(axis=0))))
        seed_metric.append(float(np.mean(sharpe_rows(xa) - sharpe_rows(ya))))
    return {'portfolio': percentile_test(portfolio, float(sharpe_rows(x.mean(axis=0))
                                                           - sharpe_rows(y.mean(axis=0)))),
            'mean_seed_sharpe': percentile_test(seed_metric, float(np.mean(sharpe_rows(x)
                                                                         - sharpe_rows(y)))),
            'seed_pairing': 'same numeric seed; both seeds and sessions resampled',
            'bootstrap_seed': 20260912, 'blocks': [5, 10, 15, 20]}


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm family-wise adjustment on the explicitly supplied full family."""
    ordered = sorted(pvalues, key=pvalues.get)
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in pvalues.values()):
        raise ValueError('finite probabilities required')
    out, previous = {}, 0.
    for i, name in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - i) * pvalues[name]))
        out[name] = previous
    return out
