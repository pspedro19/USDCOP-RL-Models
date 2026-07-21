"""Daily cross-asset portfolio — ERC over the sleeves that actually have data.

Contract: CTR-QUANT-PORTFOLIO-001

Why this exists separately from `portfolio_layer.py`: that module mixes WEEKLY sleeves read
from published 2025 bundles, and screening revealed there is no portfolio there at all —
`gold_ens` had 0 active weeks out of 52 and `btc_b2` had 6. Its reported Calmar 3.796 with max
|rho| 0.00 was COP diversified against two series of zeros.

The daily adapters in `profitability_adapters.py` do have real overlapping history
(gold 2004-2026, BTC 2017-2026, SPX500 2020-2026), which is a defensible basis for a
covariance. COP stays out: it runs on a weekly clock, and resampling it to daily to force it
into the book would manufacture observations it does not have (`strategy-contract.md` rule 5 —
never mix clocks).

Diversification is the one improvement the skill library sanctions without selection bias
(`diversification/SKILL.md`: max-diversification "does not require expected return inputs").
Everything else in the constitution is a prohibition.

Method, all of it prescribed rather than chosen:
  - `.align()` on the intersection FIRST. The skill measures that only ~68.5% of dates are
    common across assets and that forward-filling understates annualized vol by 17-19%, which
    would leave every vol-targeted position ~20% too large.
  - Ledoit-Wolf shrunk covariance (sklearn, not the skill's simplified version, which its own
    docstring flags).
  - True ERC via `risk_parity_weights`, on a strictly-past rolling window.
  - Aggregate DD breaker, ex-ante priors, never relaxed.
  - H-PORT-D-01 judged against the BEST single sleeve, not the average — beating the average
    is not evidence of anything.

Run: python -m scripts.analysis.portfolio_daily
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from services.common.metrics import _ann_return_dd_calmar, omega_ratio  # noqa: E402
from scripts.analysis.cop_null_suite import block_bootstrap_delta_calmar  # noqa: E402
from scripts.analysis.portfolio_layer import (  # noqa: E402
    MIN_ACTIVE_PERIODS, _apply_dd_breaker, diversification_ratio,
)

CLOCK = 252          # trading days; the book is quoted on the equity/metals calendar
LOOKBACK = 252       # one year of strictly-past observations to estimate the covariance
DAILY_ASSETS = ("xauusd", "btcusdt", "spx500")


def _sleeve_series() -> dict[str, pd.Series]:
    from scripts.analysis.profitability_adapters import ADAPTERS

    out: dict[str, pd.Series] = {}
    for name in DAILY_ASSETS:
        s = ADAPTERS[name]()
        idx = pd.to_datetime(pd.Index(s.index)).tz_localize(None).normalize()
        out[name] = pd.Series(s.strat_ret, index=idx).groupby(level=0).last()
    return out


def align_sleeves(sleeves: dict[str, pd.Series]) -> tuple[pd.DataFrame, dict]:
    """Intersect on common dates. No ffill, no reindex-with-zero.

    Filling a missing day with 0 asserts the asset was flat that day. It was not — it was
    unobserved. That single substitution is what turned the weekly portfolio into a fiction.
    """
    df = pd.concat(sleeves, axis=1)
    before = {k: int(v.notna().sum()) for k, v in sleeves.items()}
    df = df.dropna(how="any")
    info = {
        "rows_per_sleeve_before_align": before,
        "rows_after_align": int(len(df)),
        "common_fraction": {k: round(len(df) / n, 4) if n else None for k, n in before.items()},
        "window": [str(df.index.min().date()), str(df.index.max().date())] if len(df) else None,
    }
    return df, info


def erc_daily(df: pd.DataFrame, lookback: int = LOOKBACK) -> tuple[pd.Series, pd.DataFrame]:
    from sklearn.covariance import LedoitWolf

    sys.path.insert(0, str(REPO / ".claude" / "skills" / "xasset-alpha-engine" / "scripts"))
    from xasset.sizing import risk_parity_weights

    n = df.shape[1]
    rets, weights = np.zeros(len(df)), np.zeros((len(df), n))
    for i in range(len(df)):
        if i < lookback:
            w = np.full(n, 1.0 / n)
        else:
            win = df.iloc[i - lookback:i].to_numpy()
            try:
                w = np.asarray(risk_parity_weights(LedoitWolf().fit(win).covariance_), float)
                if not np.isfinite(w).all() or w.sum() <= 0:
                    raise ValueError("non-finite ERC weights")
            except Exception:  # noqa: BLE001
                w = np.full(n, 1.0 / n)
        weights[i] = w
        rets[i] = float(np.dot(w, df.iloc[i].to_numpy()))
    return pd.Series(rets, index=df.index), pd.DataFrame(weights, index=df.index,
                                                         columns=df.columns)


def main() -> int:
    print("=" * 74)
    print("CARTERA DIARIA CROSS-ASSET (ERC + Ledoit-Wolf)")
    print("=" * 74)

    sleeves = _sleeve_series()
    df, info = align_sleeves(sleeves)
    print(f"\nAlineación (intersección estricta, sin ffill):")
    for k, nb in info["rows_per_sleeve_before_align"].items():
        print(f"  {k:9s} {nb:>6} filas -> comun {info['common_fraction'][k]:.1%}")
    print(f"  ventana comun: {info['window']}  n={info['rows_after_align']}")

    if len(df) < LOOKBACK + MIN_ACTIVE_PERIODS:
        print(f"\n  Solo {len(df)} observaciones comunes: insuficientes para estimar una "
              f"covarianza con lookback {LOOKBACK}. NO se reporta cartera.")
        return 0

    per_sleeve = {c: _ann_return_dd_calmar(df[c].to_numpy(), CLOCK) for c in df.columns}
    print("\nSleeves en la ventana comun:")
    for k, st in per_sleeve.items():
        print(f"  {k:9s} ann={st['ann_return_pct']:>8}%  MaxDD={st['max_dd_pct']:>7}%  "
              f"Calmar={st['calmar']:>6}")

    corr = df.corr()
    off = corr.abs().where(~np.eye(len(corr), dtype=bool))
    print(f"\n  |rho| max entre sleeves: {off.max().max():.3f}")

    mix_raw, w = erc_daily(df)
    mix = _apply_dd_breaker(mix_raw)
    sm = _ann_return_dd_calmar(mix.to_numpy(), CLOCK)
    dr = diversification_ratio({c: df[c] for c in df.columns}, w.mean().to_numpy())

    best_id = max(per_sleeve, key=lambda k: per_sleeve[k]["calmar"])
    res = block_bootstrap_delta_calmar(mix.to_numpy(), df[best_id].to_numpy(),
                                   block=20, periods_per_year=CLOCK)

    print(f"\n[CARTERA ERC] ann={sm['ann_return_pct']}%  MaxDD={sm['max_dd_pct']}%  "
          f"Calmar={sm['calmar']}  Omega={omega_ratio(mix.to_numpy())}")
    print(f"  pesos medios: {', '.join(f'{c}={w[c].mean():.2f}' for c in w.columns)}")
    print(f"  ratio de diversificacion DR={dr}  (1.0 = ninguna diversificacion)")
    print(f"\n[H-PORT-D-01] DCalmar(cartera, mejor sleeve={best_id}) = "
          f"{res['delta_calmar_mean']}  IC95={res['ci95']}  excluye0={res['excludes_zero']}")
    if not res["excludes_zero"]:
        print("  => El IC incluye cero: la cartera NO mejora de forma defendible sobre el "
              "mejor sleeve individual. Resultado honesto y aceptable.")

    out = REPO / ".claude" / "evidence" / "portfolio" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "portfolio_daily.json").write_text(json.dumps({
        "clock": CLOCK, "lookback": LOOKBACK, "alignment": info,
        "per_sleeve": per_sleeve, "max_abs_corr": round(float(off.max().max()), 4),
        "portfolio": sm, "omega": omega_ratio(mix.to_numpy()),
        "mean_weights": {c: round(float(w[c].mean()), 4) for c in w.columns},
        "diversification_ratio": dr,
        "H_PORT_D_01": {"vs_best_sleeve": best_id, **res},
        "evidence_class": "research_only", "promotion_eligible": False,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nartefacto -> {out / 'portfolio_daily.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
