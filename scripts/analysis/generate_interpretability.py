"""BL-20 (parte DATOS) — artefactos de interpretabilidad por (surface, asset, model_id, version).

FABRIC Anexo A.7: "SHAP explica el modelo, no el mercado — sirve para rechazar modelos
absurdos, no para probar verdades". Este generador NO computa métricas de acierto ni
performance: 0 trials (diagnóstico declarado sobre congelados).

Fase 1 (este script):
  (a) Zoo COP — SHAP LINEAL cerrado para ridge / bayesian_ridge sobre el ÚLTIMO fit
      walk-forward (mismo esquema que meta01_zoo_ledger: train = filas < último origen
      con purga de 5 días, scaler train-only). Para un modelo lineal sobre features
      estandarizadas, el valor SHAP exacto (features independientes) es
      phi_j = coef_j * (x_j - mu_j) / sigma_j — forma cerrada, sin instalar `shap`.
      Se exporta top features (mean|phi| global) + agregado por año + kill-flags A.7
      (signo del aporte medio que cambia entre años).
  (b) Rule-based SPX500 — ATRIBUCIÓN DE REGLAS (attribution_not_shap=true): % días
      trend_on (MA200), exposición, y descomposición simple de PnL bruto en
      beta (n*mean(pos)*mean(ret)) + timing (n*cov(pos,ret)), global y por año.
      Reusa scripts/analysis/profitability_adapters.ADAPTERS['spx500'] (mismo código
      que produce los bundles publicados — cero re-derivación).

Salida: usdcop-trading-dashboard/public/data/interpretability/<surface>/<asset>/
        <model_id>/<version>/summary.json  (safe JSON: sin NaN/Inf, via safe_json_dump).

Tree SHAP (xgb/lgbm/catboost) = fase 2, fuera de alcance aquí.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.contracts.strategy_schema import safe_json_dump  # noqa: E402

OUT_ROOT = REPO / "usdcop-trading-dashboard" / "public" / "data" / "interpretability"
HORIZON = 5          # mismo H y purga que meta01_zoo_ledger.py
ZOO_LINEAR_MODELS = ("ridge", "bayesian_ridge")   # fase 1: SOLO lineales (SHAP cerrado)

# Header OBLIGATORIO en cada JSON (BL-20 / A.7).
NOTA = "SHAP explica el modelo, no el mercado; solo test-folds; diagnostico 0 trials"


def _write(surface: str, asset: str, model_id: str, version: str, payload: dict) -> Path:
    out = OUT_ROOT / surface / asset / model_id / version / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        safe_json_dump(payload, f)
    return out


# ---------------------------------------------------------------------------
# (a) Zoo COP — SHAP lineal cerrado (ridge / bayesian_ridge)
# ---------------------------------------------------------------------------

def generate_zoo_linear(model_ids: tuple[str, ...] = ZOO_LINEAR_MODELS) -> list[Path]:
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader

    cfg = ForecastingSSOTConfig.load()
    loader = ForecastingDatasetLoader(cfg, project_root=REPO)
    df, _ = loader.load_dataset()
    feat_cols = [c for c in cfg.get_feature_columns() if c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    df["y5"] = df["close"].shift(-HORIZON) / df["close"] - 1.0

    # ÚLTIMO fit walk-forward (esquema meta01): origen = última fila; purga de 5 días
    # (los targets 5d de las últimas HORIZON filas no están realizados).
    origin = pd.Timestamp(df["date"].iloc[-1])
    train = df.iloc[:-HORIZON]
    m_ok = train[feat_cols].notna().all(axis=1) & train["y5"].notna()
    Xtr = train.loc[m_ok, feat_cols].to_numpy(float)
    ytr = train.loc[m_ok, "y5"].to_numpy(float)
    if len(Xtr) < 200:
        raise RuntimeError(f"zoo: solo {len(Xtr)} filas de train — dataset incompleto")

    scaler = StandardScaler().fit(Xtr)          # train-only, sin fuga
    mu, sigma = scaler.mean_, scaler.scale_

    # Filas donde se evalúan las contribuciones: todo el histórico con features completas.
    full_ok = df[feat_cols].notna().all(axis=1)
    rows = df.loc[full_ok, ["date"] + feat_cols].reset_index(drop=True)
    Z = (rows[feat_cols].to_numpy(float) - mu) / sigma
    years = rows["date"].dt.year

    version = origin.date().isoformat()
    paths: list[Path] = []
    for mid in model_ids:
        mdl = ModelFactory.create(mid)
        mdl.fit(scaler.transform(Xtr), ytr)
        coefs = np.asarray(mdl._model.coef_, dtype=float).ravel()
        intercept = float(np.asarray(mdl._model.intercept_).ravel()[0])
        if len(coefs) != len(feat_cols):
            raise RuntimeError(f"{mid}: {len(coefs)} coefs vs {len(feat_cols)} features")

        phi = Z * coefs                                   # (n_rows, n_feat) SHAP lineal
        mean_abs = np.nanmean(np.abs(phi), axis=0)
        mean_phi = np.nanmean(phi, axis=0)
        order = np.argsort(-mean_abs)
        top_features = [
            {"rank": int(r + 1), "feature": feat_cols[j], "coef": float(coefs[j]),
             "mean_abs_shap": float(mean_abs[j]), "mean_shap": float(mean_phi[j])}
            for r, j in enumerate(order)
        ]

        by_year: dict[str, list[dict]] = {}
        yearly_sign: dict[str, list[float]] = {c: [] for c in feat_cols}
        for yr in sorted(years.unique()):
            sel = (years == yr).to_numpy()
            ma = np.nanmean(np.abs(phi[sel]), axis=0)
            mp = np.nanmean(phi[sel], axis=0)
            oy = np.argsort(-ma)
            by_year[str(int(yr))] = [
                {"feature": feat_cols[j], "mean_abs_shap": float(ma[j]),
                 "mean_shap": float(mp[j])}
                for j in oy
            ]
            for j, c in enumerate(feat_cols):
                yearly_sign[c].append(float(mp[j]))

        # Kill-flag A.7: aporte medio que cambia de signo entre años (ambos lados
        # con magnitud material). Solo FLAG diagnóstico — la decisión es humana.
        eps = 0.1 * float(np.nanmean(mean_abs)) if np.isfinite(np.nanmean(mean_abs)) else 0.0
        kill_flags = sorted(
            c for c, vals in yearly_sign.items()
            if min(vals) < -eps and max(vals) > eps
        )

        payload = {
            "nota": NOTA,
            "surface": "zoo",
            "asset": "usdcop",
            "model_id": mid,
            "model_type": "linear",
            "method": "linear_shap_closed_form",
            "attribution_not_shap": False,
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fit": {
                "scheme": "ultimo fit walk-forward (origen = ultima fila, purga 5d)",
                "origin": version,
                "n_train": int(len(Xtr)),
                "horizon": HORIZON,
                "purge_days": HORIZON,
                "scaler": "StandardScaler train-only",
                "params": {k: v for k, v in (mdl.get_params() or {}).items()
                           if isinstance(v, (int, float, str, bool, type(None)))},
            },
            "scope": ("phi_j = coef_j*(x_j-mu_j)/sigma_j del ULTIMO fit aplicado a todo el "
                      "historico de features — diagnostico del modelo congelado, no evidencia "
                      "OOS ni claim de edge"),
            "base_value": intercept,
            "n_rows": int(len(rows)),
            "n_features": len(feat_cols),
            "top_features": top_features,
            "by_year": by_year,
            "kill_flags_sign_change_by_year": kill_flags,
        }
        p = _write("zoo", "usdcop", mid, version, payload)
        paths.append(p)
        print(f"[zoo] {mid}: {p.relative_to(REPO)}", flush=True)
    return paths


# ---------------------------------------------------------------------------
# (b) Rule-based SPX500 — atribución de reglas (NO SHAP)
# ---------------------------------------------------------------------------

def _decompose(pos: np.ndarray, ret: np.ndarray, cost: np.ndarray, swap: np.ndarray) -> dict:
    """PnL bruto = beta + timing.  timing = n*cov(pos, ret); beta = n*mean(pos)*mean(ret)."""
    n = len(pos)
    gross = float(np.nansum(pos * ret))
    beta = float(n * np.nanmean(pos) * np.nanmean(ret)) if n else 0.0
    timing = gross - beta                                 # identidad exacta: cov muestral sesgada
    costs = float(np.nansum(cost) + np.nansum(swap))
    return {"n_days": int(n), "pnl_gross": gross, "pnl_beta": beta,
            "pnl_timing_cov_pos_ret": timing, "costs": costs, "pnl_net": gross - costs}


def generate_rule_attribution(rule_ids: tuple[str, ...] = ("spx500",)) -> list[Path]:
    from scripts.analysis.profitability_adapters import ADAPTERS

    paths: list[Path] = []
    for rid in rule_ids:
        sleeve = ADAPTERS[rid]()
        idx = pd.DatetimeIndex(pd.to_datetime(sleeve.index))
        pos, ret = sleeve.position, sleeve.asset_ret
        cost, swap = sleeve.cost, sleeve.swap
        trend_on = sleeve.dumb_position                   # spx500: ma200_always_on = la regla

        by_year = {}
        for yr in sorted(set(idx.year)):
            sel = (idx.year == yr)
            d = _decompose(pos[sel], ret[sel], cost[sel], swap[sel])
            d["pct_days_trend_on"] = float(np.nanmean(trend_on[sel] > 0))
            d["pct_days_position_active"] = float(np.nanmean(np.abs(pos[sel]) > 1e-9))
            d["avg_exposure"] = float(np.nanmean(pos[sel]))
            by_year[str(int(yr))] = d

        total = _decompose(pos, ret, cost, swap)
        version = idx[-1].date().isoformat()
        payload = {
            "nota": NOTA,
            "surface": "rule_based",
            "asset": sleeve.asset,
            "model_id": sleeve.strategy_id,
            "model_type": "rule_based",
            "method": "rule_attribution",
            "attribution_not_shap": True,               # ATRIBUCION, no SHAP (BL-20 punto 2)
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scope": ("atribucion de reglas sobre la misma serie del adapter publicado "
                      f"({sleeve.clock_label}); descomposicion pnl_gross = beta + timing, "
                      "timing = n*cov(pos,ret) — diagnostico, no claim de edge"),
            "rules": {
                "gate": "trend_on = close > MA200 (dumb baseline ma200_always_on del sleeve)",
                "pct_days_trend_on": float(np.nanmean(trend_on > 0)),
                "pct_days_position_active": float(np.nanmean(np.abs(pos) > 1e-9)),
                "avg_exposure": float(np.nanmean(pos)),
                "n_trades": sleeve.n_trades,
            },
            "pnl_decomposition": total,
            "by_year": by_year,
        }
        p = _write("rule_based", sleeve.asset, sleeve.strategy_id, version, payload)
        paths.append(p)
        print(f"[rule] {rid}: {p.relative_to(REPO)}", flush=True)
    return paths


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zoo-models", default=",".join(ZOO_LINEAR_MODELS),
                    help="modelos lineales del zoo (fase 1: ridge,bayesian_ridge)")
    ap.add_argument("--rules", default="spx500",
                    help="adapters rule-based (profitability_adapters.ADAPTERS)")
    ap.add_argument("--skip-zoo", action="store_true")
    ap.add_argument("--skip-rules", action="store_true")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    if not args.skip_zoo:
        mids = tuple(m.strip() for m in args.zoo_models.split(",") if m.strip())
        bad = [m for m in mids if m not in ZOO_LINEAR_MODELS]
        if bad:
            raise SystemExit(f"fase 1 solo soporta SHAP lineal cerrado: {bad} no permitido "
                             f"(tree SHAP = fase 2)")
        paths += generate_zoo_linear(mids)
    if not args.skip_rules:
        rids = tuple(r.strip() for r in args.rules.split(",") if r.strip())
        paths += generate_rule_attribution(rids)

    print(f"OK: {len(paths)} artefactos")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
