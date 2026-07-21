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

# Cash. The book is flat most of the time -- mean gross exposure is 0.419, and BTC alone is out
# of the market 69% of days -- and it was crediting 0% on all of it. Real money sitting flat
# earns the front rate. Omitting that is not conservatism, it is a mis-statement: at ~4% it
# understates the book by roughly 2-3%/yr on the uninvested balance.
#
# Fed funds is the right proxy: every leg (gold, BTC, SPX) is USD-denominated.
CASH_RATE_COL = "polr_fed_funds_usa_m_fedfunds"
MACRO_SEED = REPO / "seeds" / "latest" / "macro_indicators_daily.parquet"


def cash_rate_daily(index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free rate aligned to the book, causal.

    `shift(1)` because a day's published rate is not knowable at that day's open, and
    forward-filled because the series is monthly. Returns zeros if the seed is missing -- the
    book must still compute, but then it is reporting the understated number and says so.
    """
    if not MACRO_SEED.is_file():
        return pd.Series(0.0, index=index)
    m = pd.read_parquet(MACRO_SEED)
    dcol = "fecha" if "fecha" in m.columns else m.columns[0]
    if CASH_RATE_COL not in m.columns:
        return pd.Series(0.0, index=index)
    s = (m[[dcol, CASH_RATE_COL]].dropna()
         .assign(**{dcol: lambda d: pd.to_datetime(d[dcol])})
         .set_index(dcol)[CASH_RATE_COL].sort_index())
    return (s.reindex(s.index.union(index)).ffill().reindex(index).shift(1).ffill()
            .fillna(0.0) / 100.0)


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


def correlation_report(df: pd.DataFrame) -> dict:
    """Unconditional AND co-active correlations, plus how often the legs are on together.

    The unconditional number on its own is misleading, and I published it that way: I reported
    "max |rho| 0.081, genuinely uncorrelated". Conditioning on both legs actually holding a
    position roughly DOUBLES it, and all three sleeves are simultaneously active only ~12% of
    days. Most of the apparent diversification is non-overlapping PRESENCE, not offsetting
    risk -- the book is closer to three strategies taking turns than three independent streams.

    `forward-risk-var/SKILL.md:228`: "correlations spike toward 1.0 during market stress,
    precisely when diversification is most needed. Stress tests should use crisis-period
    correlations, not calm-period correlations." Computing rho over the full window is exactly
    the calm-period estimate that warning is about.
    """
    cols = list(df.columns)
    active = {c: (df[c] != 0).to_numpy() for c in cols}
    uncond, coactive, overlap = {}, {}, {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            key = f"{a}~{b}"
            uncond[key] = round(float(df[a].corr(df[b])), 4)
            both = active[a] & active[b]
            overlap[key] = round(float(both.mean()), 4)
            coactive[key] = (round(float(df[a][both].corr(df[b][both])), 4)
                             if both.sum() >= 30 else None)
    all_on = float(np.logical_and.reduce([active[c] for c in cols]).mean())
    return {
        "unconditional": uncond,
        "co_active": coactive,
        "pairwise_overlap_fraction": overlap,
        "fraction_days_all_legs_active": round(all_on, 4),
        "max_abs_unconditional": round(max(abs(v) for v in uncond.values()), 4),
        "max_abs_co_active": (round(max(abs(v) for v in coactive.values() if v is not None), 4)
                              if any(v is not None for v in coactive.values()) else None),
        "warning": (
            "The unconditional rho understates dependence. Read it next to co_active and to "
            "fraction_days_all_legs_active: low correlation achieved by not being in the market "
            "is availability, not diversification, and it disappears the moment both legs are on."
        ),
    }


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

    cr = correlation_report(df)
    print(f"\n  |rho| max INCONDICIONAL : {cr['max_abs_unconditional']:.3f}")
    print(f"  |rho| max CO-ACTIVO     : {cr['max_abs_co_active']}   <- el que decide")
    print(f"  dias con las 3 piernas activas a la vez: {cr['fraction_days_all_legs_active']:.1%}")
    for k in cr["unconditional"]:
        print(f"    {k:22} rho={cr['unconditional'][k]:>7}  co-activo={str(cr['co_active'][k]):>7}"
              f"  solape={cr['pairwise_overlap_fraction'][k]:.1%}")

    mix_raw, w = erc_daily(df)
    mix = _apply_dd_breaker(mix_raw)
    sm = _ann_return_dd_calmar(mix.to_numpy(), CLOCK)

    # Cash yield on the uninvested balance. Reported as a SEPARATE series, never folded into
    # the headline: the no-cash number is what compares against the historical backtests, and
    # the with-cash number is the economically honest one.
    gross = (w.abs() * df.abs().gt(0)).sum(axis=1).clip(upper=1.0)
    rf = cash_rate_daily(df.index)
    mix_cash = mix + rf / CLOCK * (1.0 - gross)
    sc = _ann_return_dd_calmar(mix_cash.to_numpy(), CLOCK)
    dr = diversification_ratio({c: df[c] for c in df.columns}, w.mean().to_numpy())

    best_id = max(per_sleeve, key=lambda k: per_sleeve[k]["calmar"])
    res = block_bootstrap_delta_calmar(mix.to_numpy(), df[best_id].to_numpy(),
                                   block=20, periods_per_year=CLOCK)

    print(f"\n[CARTERA ERC] ann={sm['ann_return_pct']}%  MaxDD={sm['max_dd_pct']}%  "
          f"Calmar={sm['calmar']}  Omega={omega_ratio(mix.to_numpy())}")
    print(f"  pesos medios: {', '.join(f'{c}={w[c].mean():.2f}' for c in w.columns)}")
    print(f"  ratio de diversificacion DR={dr}  (1.0 = ninguna diversificacion)")
    print(f"\n[+ EFECTIVO REMUNERADO] exposicion bruta media={gross.mean():.3f} "
          f"(efectivo medio={1-gross.mean():.1%}), tasa media={rf.mean():.2%}")
    print(f"  ann={sc['ann_return_pct']}%  MaxDD={sc['max_dd_pct']}%  Calmar={sc['calmar']}  "
          f"(sin efectivo: ann={sm['ann_return_pct']}%  Calmar={sm['calmar']})")
    print(f"\n[H-PORT-D-01] DCalmar(cartera, mejor sleeve={best_id}) = "
          f"{res['delta_calmar_mean']}  IC95={res['ci95']}  excluye0={res['excludes_zero']}")
    if not res["excludes_zero"]:
        print("  => El IC incluye cero: la cartera NO mejora de forma defendible sobre el "
              "mejor sleeve individual. Resultado honesto y aceptable.")

    out = REPO / ".claude" / "evidence" / "portfolio" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "portfolio_daily.json").write_text(json.dumps({
        "clock": CLOCK, "lookback": LOOKBACK, "alignment": info,
        "per_sleeve": per_sleeve, "correlations": cr,
        "portfolio": sm, "omega": omega_ratio(mix.to_numpy()),
        "portfolio_with_cash_yield": sc,
        "cash": {"rate_column": CASH_RATE_COL, "source": str(MACRO_SEED.name),
                 "lag": "shift(1), ffill (monthly series)",
                 "mean_gross_exposure": round(float(gross.mean()), 4),
                 "mean_rate": round(float(rf.mean()), 5)},
        "mean_weights": {c: round(float(w[c].mean()), 4) for c in w.columns},
        "diversification_ratio": dr,
        "H_PORT_D_01": {"vs_best_sleeve": best_id, **res},
        # ADR-0020: this is a risk-controlled book, not an alpha claim. The class carries a
        # lower evidence bar (B1' + cost x2 + DD brake + crisis behaviour) but stricter
        # disclosure duties, enforced by test_risk_controlled_book_disclosure.py.
        "product_class": "risk_controlled_book", "adr": "ADR-0020",
        "evidence_class": "research_only", "promotion_eligible": False,
        "not_promoted_because": ("H-PORT-D-01 CI includes zero and no forward evidence has "
                                 "accumulated (ADR-0020 criterion 7)"),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nartefacto -> {out / 'portfolio_daily.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
