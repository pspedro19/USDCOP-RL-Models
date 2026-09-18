"""Versioned HMM input lane; the historical source remains byte-for-byte pinned.

Market formulas and BIC rule mirror regime_hmm, with parity tests. The only input
change is an explicit publication-aware macro frame instead of its global join.
No monkeypatching shared globals, forced K, or retroactive change of source pins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research.regime_hmm import (
    BIC_HYSTERESIS,
    FEATURE_NAMES,
    K_CANDIDATES,
    K_DEFAULT,
    FrozenRegimeModel,
    _fit_one,
)


def build_regime_observations(m5, valid_sessions=None, *, macro_features):
    if not macro_features.index.is_unique:
        raise ValueError("duplicate macro context dates")
    df = m5.copy()
    t = pd.to_datetime(df.time)
    if "symbol" in df:
        keep = df.symbol.astype(str).str.upper().str.replace("/", "", regex=False) == "USDCOP"
        df, t = df[keep], t[keep]
    df = df.assign(_t=t, _d=t.dt.date).sort_values("_t")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    if valid_sessions is not None:
        df = df[df["_d"].isin(set(valid_sessions))]
    df["_bar_ret"] = np.log(df.close / df.groupby("_d").close.shift(1))
    g = df.groupby("_d")
    daily = pd.DataFrame(
        {
            "close": g.close.last(),
            "open": g.open.first(),
            "high": g.high.max(),
            "low": g.low.min(),
            "rv": np.sqrt(g["_bar_ret"].apply(lambda s: np.nansum(s**2))),
            "intraday_autocorr": g["_bar_ret"].apply(
                lambda s: s.autocorr(lag=1) if s.notna().sum() > 5 else np.nan
            ),
        }
    )
    daily.index = pd.to_datetime(list(daily.index))
    daily = daily.sort_index()
    daily["ret"] = np.log(daily.close / daily.close.shift(1))
    daily["abs_ret"] = daily.ret.abs()
    daily["log_rv"] = np.log(daily.rv.replace(0.0, np.nan))
    prev = daily.close.shift(1)
    tr = pd.concat(
        [daily.high - daily.low, (daily.high - prev).abs(), (daily.low - prev).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    daily["atr_norm"] = atr / daily.close
    daily["range_over_atr"] = (daily.high - daily.low) / atr.replace(0.0, np.nan)
    for key, col in (("dxy_ret", "dxy_ret_prev"), ("brent_ret", "brent_ret_prev")):
        daily[key] = macro_features[col].reindex(daily.index)
    return daily[list(FEATURE_NAMES)]


@dataclass(frozen=True)
class AuditedRegimeModel(FrozenRegimeModel):
    candidate_evidence: tuple[dict, ...] = ()


def fit_frozen(dev_obs: pd.DataFrame) -> AuditedRegimeModel:
    obs = dev_obs.dropna()
    if len(obs) < 200 or not np.isfinite(obs.to_numpy()).all():
        raise ValueError("at least 200 finite development observations required")
    raw = obs.to_numpy(dtype=float)
    means, scales = raw.mean(axis=0), raw.std(axis=0)
    scales[scales == 0] = 1
    X = (raw - means) / scales
    n, d = X.shape
    cov_type, bics, fitted, candidates = "full", {}, {}, []
    for k in K_CANDIDATES:
        m, ll = _fit_one(X, k, cov_type)
        if m is None:
            cov_type = "diag"  # same declared carry-forward fallback as the archived recipe
            m, ll = _fit_one(X, k, cov_type)
        if m is None:
            candidates.append({"k": k, "status": "NO_FIT", "covariance_type": cov_type})
            continue
        n_cov = k * d * (d + 1) / 2 if cov_type == "full" else k * d
        n_params = k * (k - 1) + (k - 1) + k * d + n_cov
        bics[k] = float(-2 * ll + n_params * np.log(n))
        fitted[k] = m
        monitor = getattr(m, "monitor_", None)
        candidates.append(
            {
                "k": k,
                "status": "FITTED",
                "bic": bics[k],
                "log_likelihood": float(ll),
                "n_parameters": int(n_params),
                "covariance_type": m.covariance_type,
                "converged": bool(monitor.converged) if monitor else None,
                "iterations": int(monitor.iter) if monitor else None,
            }
        )
    if not fitted:
        raise RuntimeError("no K could be fitted")
    best = min(bics, key=bics.get)
    if K_DEFAULT in bics and best != K_DEFAULT and bics[K_DEFAULT] - bics[best] <= BIC_HYSTERESIS:
        best = K_DEFAULT
    model = fitted[best]
    # hmmlearn exposes expanded (K,D,D) covariances even for its diagonal model.
    # State ranking needs one variance per state, never an argsort of a KxD array.
    covariance = np.asarray(model.covars_)
    var = (
        np.diagonal(covariance, axis1=1, axis2=2).mean(axis=1)
        if covariance.ndim == 3
        else covariance.mean(axis=1)
    )
    if var.shape != (best,) or not np.isfinite(var).all():
        raise ValueError("invalid per-state covariance dimensions")
    return AuditedRegimeModel(
        model=model,
        k=best,
        vol_order=tuple(int(i) for i in np.argsort(var)),
        fit_range=(str(obs.index[0].date()), str(obs.index[-1].date())),
        bic_by_k=bics,
        feature_names=tuple(dev_obs.columns),
        means=means,
        scales=scales,
        covariance_type=model.covariance_type,
        candidate_evidence=tuple(candidates),
    )
