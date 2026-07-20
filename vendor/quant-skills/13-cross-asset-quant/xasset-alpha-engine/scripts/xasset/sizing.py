"""Volatility targeting and capped Kelly.

Two findings from the audit of this collection motivate this module:

  1. No portfolio volatility targeting exists anywhere - the only multi-asset
     sizing is inverse-vol, which takes no covariance matrix and therefore
     cannot know the portfolio's actual risk.
  2. No Kelly cap exists anywhere. `continuous_kelly()` returns full Kelly
     uncapped, and the shipped worked example recommends 18.5% risk on a single
     trade inside a file that calls >2% "dangerous".

Volatility targeting is what makes a cross-asset book possible at all. Bitcoin
runs ~60% annualised vol, EURUSD ~7%, gold ~15%. Equal notional means the book
is a bitcoin fund with decorations. Equal *risk* is the only sane default.

On Kelly: full Kelly is optimal only if you know the true edge. You never do.
Estimation error makes full Kelly reliably ruinous, so `fractional_kelly`
requires a fraction <= 0.5 and caps output. This is a deliberate refusal to
reproduce the 18.5% example.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "realised_vol",
    "vol_target_weights",
    "portfolio_vol",
    "risk_parity_weights",
    "fractional_kelly",
    "SizingResult",
]

TRADING_DAYS = 252


def realised_vol(
    returns, periods_per_year: int = TRADING_DAYS, halflife: int | None = None
) -> float:
    """Annualised realised volatility.

    With `halflife`, uses an exponentially weighted estimate, which reacts
    faster to regime shifts. For crypto and EM FX - where vol regimes change
    abruptly - an EWMA with a 20-40 day halflife tracks risk far better than a
    trailing simple window that averages a crisis together with the calm before
    it.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError("need >= 2 finite returns")

    if halflife is None:
        sd = float(np.std(r, ddof=1))
    else:
        if halflife <= 0:
            raise ValueError("halflife must be positive")
        lam = 0.5 ** (1.0 / halflife)
        w = lam ** np.arange(r.size - 1, -1, -1)
        w /= w.sum()
        mean = float(np.sum(w * r))
        sd = math.sqrt(float(np.sum(w * (r - mean) ** 2)))

    if sd == 0:
        raise ValueError("zero volatility - cannot size a position")
    return sd * math.sqrt(periods_per_year)


@dataclass(frozen=True)
class SizingResult:
    weights: dict[str, float]
    gross_leverage: float
    ex_ante_vol: float
    scaled_by: float
    binding_constraint: str | None

    def report(self) -> str:
        lines = [
            f"ex-ante portfolio vol : {self.ex_ante_vol:.2%}",
            f"gross leverage        : {self.gross_leverage:.2f}x",
        ]
        if self.binding_constraint:
            lines.append(f"scaled {self.scaled_by:.3f}x by {self.binding_constraint}")
        lines.append("")
        for k, v in sorted(self.weights.items(), key=lambda x: -abs(x[1])):
            lines.append(f"  {k:<12} {v:+.4f}")
        return "\n".join(lines)


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    """Annualised portfolio volatility from weights and an annualised covariance."""
    w = np.asarray(weights, dtype=float)
    c = np.asarray(cov, dtype=float)
    if c.shape != (w.size, w.size):
        raise ValueError(f"cov shape {c.shape} does not match {w.size} weights")
    v = float(w @ c @ w)
    if v < 0:
        raise ValueError("negative portfolio variance - covariance is not PSD")
    return math.sqrt(v)


def vol_target_weights(
    signals: dict[str, float],
    vols: dict[str, float],
    cov: np.ndarray | None = None,
    target_vol: float = 0.10,
    max_gross_leverage: float = 3.0,
    max_weight: float = 2.0,
) -> SizingResult:
    """Size signals to a portfolio volatility target.

    Each position is first scaled to equal risk (signal / its own vol), then the
    whole book is scaled so ex-ante portfolio vol equals `target_vol`.

    Supplying `cov` uses the true covariance, which accounts for the fact that
    a long-gold / short-USD / long-EM-FX book is one dollar trade wearing three
    hats. Without it, correlations are assumed zero and the reported vol is a
    lower bound - so the covariance path is strongly preferred for exactly the
    cross-asset case this engine targets.

    Args:
        signals: symbol -> signed conviction, nominally in [-1, 1].
        vols: symbol -> annualised volatility.
        target_vol: annualised portfolio vol target.
        max_gross_leverage: cap on sum |w| - the primary concentration control.
        max_weight: cap on any single |w|, applied before scaling. Defaults to
            2.0, NOT to a sub-1.0 equity-style cap: EURUSD at 7% vol needs
            ~1.4x weight to contribute 10% vol, so a 0.25 cap would silently
            make low-volatility FX untradeable in a cross-asset book and
            hand the risk budget to whatever is most volatile.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be > 0")
    if not signals:
        raise ValueError("no signals supplied")
    missing = set(signals) - set(vols)
    if missing:
        raise ValueError(f"no volatility for: {sorted(missing)}")

    syms = sorted(signals)
    raw = {}
    for s in syms:
        if vols[s] <= 0:
            raise ValueError(f"{s}: volatility must be > 0")
        raw[s] = signals[s] * (target_vol / vols[s])

    # Cap single-name concentration before scaling the book.
    capped = {s: float(np.clip(w, -max_weight, max_weight)) for s, w in raw.items()}
    w = np.array([capped[s] for s in syms])

    if np.allclose(w, 0):
        return SizingResult({s: 0.0 for s in syms}, 0.0, 0.0, 1.0, "all signals zero")

    if cov is not None:
        ex_ante = portfolio_vol(w, np.asarray(cov, dtype=float))
    else:
        v = np.array([vols[s] for s in syms])
        ex_ante = float(np.sqrt(np.sum((w * v) ** 2)))  # zero-correlation assumption

    scale = target_vol / ex_ante if ex_ante > 0 else 1.0
    binding = "vol target"

    gross = float(np.sum(np.abs(w))) * scale
    if gross > max_gross_leverage:
        scale *= max_gross_leverage / gross
        binding = "max_gross_leverage"

    w_final = w * scale
    final_vol = (
        portfolio_vol(w_final, np.asarray(cov, dtype=float))
        if cov is not None
        else float(np.sqrt(np.sum((w_final * np.array([vols[s] for s in syms])) ** 2)))
    )

    return SizingResult(
        weights={s: float(x) for s, x in zip(syms, w_final)},
        gross_leverage=float(np.sum(np.abs(w_final))),
        ex_ante_vol=final_vol,
        scaled_by=float(scale),
        binding_constraint=binding,
    )


def risk_parity_weights(
    cov: np.ndarray,
    budget: np.ndarray | None = None,
    max_iter: int = 10_000,
    tol: float = 1e-12,
):
    """Equal risk contribution weights, long-only, summing to 1.

    Solves w_i * (Sigma w)_i = b_i via the multiplicative square-root iteration

        w <- normalise( sqrt( w * b / (Sigma w) ) )

    The obvious iteration `w <- normalise(b / (Sigma w))` is NOT used: it
    oscillates between two points and never converges. On a diagonal covariance
    it flips between equal weights and the true solution forever, so a
    max-iteration fallback returns whichever side it landed on - a plausible
    looking but wrong answer. This version has the ERC solution as a genuine
    fixed point, and non-convergence raises instead of returning silently.

    Args:
        cov: annualised covariance matrix, symmetric PSD.
        budget: target risk contributions, defaults to equal. Must be positive
            and is normalised internally.
    """
    c = np.asarray(cov, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("covariance must be a square matrix")
    n = c.shape[0]
    if not np.allclose(c, c.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.any(np.diag(c) <= 0):
        raise ValueError("covariance has non-positive variance on the diagonal")

    if budget is None:
        b = np.ones(n) / n
    else:
        b = np.asarray(budget, dtype=float)
        if b.size != n:
            raise ValueError(f"budget size {b.size} != {n} assets")
        if np.any(b <= 0):
            raise ValueError("risk budgets must be positive")
        b = b / b.sum()

    w = np.ones(n) / n
    for _ in range(max_iter):
        mrc = c @ w
        if np.any(mrc <= 0):
            raise ValueError("non-positive marginal risk - covariance is not PSD")
        w_new = np.sqrt(w * b / mrc)
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            return w_new
        w = w_new

    raise RuntimeError(
        f"risk parity did not converge in {max_iter} iterations - check that the "
        "covariance is well conditioned"
    )


def fractional_kelly(
    expected_excess_return: float,
    volatility: float,
    fraction: float = 0.25,
    cap: float = 0.20,
) -> float:
    """Fractional Kelly weight, capped. Full Kelly is deliberately unreachable.

    Continuous Kelly is mu / sigma^2. It maximises log wealth only when mu is
    known exactly. In practice mu is estimated with wide error, and Kelly is
    convex in that error, so full Kelly systematically oversizes: a strategy
    whose true Sharpe is half the estimate leads to a position that can lose a
    large fraction of capital in one drawdown.

    `fraction` is therefore hard-limited to 0.5. Quarter Kelly captures most of
    the growth with far less drawdown, and is the standard practitioner choice.

    Raises:
        ValueError: if `fraction` exceeds 0.5.
    """
    if volatility <= 0:
        raise ValueError("volatility must be > 0")
    if not 0 < fraction <= 0.5:
        raise ValueError(
            f"fraction must be in (0, 0.5]; got {fraction}. Full or over-Kelly "
            "is not supported - it is ruinous under estimation error."
        )
    if cap <= 0:
        raise ValueError("cap must be > 0")

    full = expected_excess_return / (volatility**2)
    return float(np.clip(full * fraction, -cap, cap))
