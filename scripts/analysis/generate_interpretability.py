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

Fase 2 (BL-20 hueco TreeSHAP, 2026-07-28):
  (c) Zoo COP — TREE SHAP EXACTO para xgboost / lightgbm / catboost. Se usa el TreeSHAP
      nativo de cada booster (mismo algoritmo Lundberg et al.; la comprobación
      sum(phi)+bias == pred cruda se ejecuta y se persiste como `additivity_max_abs_err`):
        xgboost  -> Booster.predict(DMatrix, pred_contribs=True)
        lightgbm -> Booster.predict(X, pred_contrib=True)
        catboost -> get_feature_importance(Pool(X), type='ShapValues')
      SOLO TEST-FOLDS (invariante A.7): walk-forward EXPANDING ANUAL — para el año Y se
      hace fit con filas <= 31-dic-(Y-1) menos purga de 5 días y se atribuyen ÚNICAMENTE
      las filas del año Y (jamás una fila que estuvo en su propio train). Cortes:
      global + temporal (por año) + por régimen (gate Hurst CONGELADO de
      config/execution/smart_simple_v1.yaml, evaluado con retornos <= la propia fila).
      Si el backend de un modelo no está disponible se emite un artefacto tipado
      `tree_shap_unavailable` con la razón — NUNCA valores inventados.

Salida: data/interpretability/<surface>/<asset>/ (fuera de public; servido SOLO via API admin:all)
        <model_id>/<version>/summary.json  (safe JSON: sin NaN/Inf, via safe_json_dump).

0 trials en las tres rutas: no se computa ninguna métrica de acierto/performance
(ni DA, ni R², ni Sharpe) y no se selecciona nada — los hiperparámetros son los
defaults congelados del ModelFactory, sin tuning.
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

OUT_ROOT = REPO / "data" / "interpretability"  # FUERA de public/ (CXD-040: public bypassea admin:all)
HORIZON = 5          # mismo H y purga que meta01_zoo_ledger.py
ZOO_LINEAR_MODELS = ("ridge", "bayesian_ridge")   # SHAP lineal cerrado
ZOO_TREE_MODELS = ("xgboost", "lightgbm", "catboost")   # TreeSHAP nativo exacto
MIN_TRAIN = 400      # misma guarda que meta01_zoo_ledger.py (años con menos train se SALTAN)

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
# (c) Zoo COP — TreeSHAP exacto (xgboost / lightgbm / catboost), SOLO test-folds
# ---------------------------------------------------------------------------

def _tree_shap_backend(model_id: str):
    """Devuelve (nombre_backend, fn(booster_wrapper, X) -> (phi, bias)) o lanza ImportError.

    Los tres boosters implementan TreeSHAP EXACTO (Lundberg et al.) de forma nativa; la
    última columna que devuelven es el valor base. No se requiere el paquete `shap`
    (que aquí ni siquiera importa: su `_tree.py` arrastra pyspark, roto en py3.12).
    """
    if model_id == "xgboost":
        import xgboost as xgb  # noqa: F401 — falla ⇒ backend no disponible

        def fn(mdl, X):
            raw = np.asarray(
                mdl._model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True), float)
            return raw[:, :-1], raw[:, -1]

        return "xgboost.Booster.predict(pred_contribs=True)", fn

    if model_id == "lightgbm":
        import lightgbm  # noqa: F401

        def fn(mdl, X):
            raw = np.asarray(mdl._model.predict(X, pred_contrib=True), float)
            return raw[:, :-1], raw[:, -1]

        return "lightgbm.Booster.predict(pred_contrib=True)", fn

    if model_id == "catboost":
        from catboost import Pool

        def fn(mdl, X):
            raw = np.asarray(
                mdl._model.get_feature_importance(Pool(X), type="ShapValues"), float)
            return raw[:, :-1], raw[:, -1]

        return "catboost.get_feature_importance(type='ShapValues')", fn

    raise ImportError(f"{model_id}: sin backend TreeSHAP nativo registrado")


def _regime_labels(df: pd.DataFrame) -> pd.Series:
    """Etiqueta de régimen por fila con el gate Hurst CONGELADO de smart_simple_v1.yaml.

    Se evalúa con los retornos <= la propia fila (misma información que tendría el gate
    en vivo en esa barra): sin look-ahead. Devuelve strings del RegimeState.
    """
    import yaml
    from src.forecasting.regime_gate import RegimeGateConfig, classify_regime

    raw = {}
    cfg_path = REPO / "config" / "execution" / "smart_simple_v1.yaml"
    try:
        raw = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("regime_gate", {})
    except OSError:
        raw = {}
    gcfg = RegimeGateConfig(
        hurst_lookback=int(raw.get("hurst_lookback", RegimeGateConfig.hurst_lookback)),
        hurst_trending=float(raw.get("hurst_trending", RegimeGateConfig.hurst_trending)),
        hurst_mean_rev=float(raw.get("hurst_mean_rev", RegimeGateConfig.hurst_mean_rev)),
    )
    rets = df["close"].pct_change().fillna(0.0).to_numpy(float)
    look = gcfg.hurst_lookback
    out = []
    for i in range(len(rets)):
        window = rets[max(0, i - look + 1): i + 1]      # <= la propia fila
        out.append(classify_regime(list(window), gcfg).state.value)
    return pd.Series(out, index=df.index, name="regime")


def _agg_rows(phi: np.ndarray, feat_cols: list[str], sel: np.ndarray) -> list[dict]:
    """mean|phi| y mean(phi) por feature sobre el subconjunto `sel`, ordenado por magnitud."""
    sub = phi[sel]
    if not len(sub):
        return []
    ma, mp = np.nanmean(np.abs(sub), axis=0), np.nanmean(sub, axis=0)
    return [{"feature": feat_cols[j], "mean_abs_shap": float(ma[j]), "mean_shap": float(mp[j])}
            for j in np.argsort(-ma)]


def _sign_change_flags(groups: dict[str, list[dict]], feat_cols: list[str],
                       scale: float) -> list[str]:
    """Kill-flag A.7: mean(phi) que cambia de signo entre grupos con magnitud material.

    MISMA regla que la ruta lineal (eps = 10% de la magnitud media global). Solo FLAG
    diagnóstico: la decisión de rechazar el modelo es humana.
    """
    eps = 0.1 * scale if np.isfinite(scale) else 0.0
    per_feat: dict[str, list[float]] = {c: [] for c in feat_cols}
    for rows in groups.values():
        for r in rows:
            per_feat[r["feature"]].append(float(r["mean_shap"]))
    return sorted(c for c, v in per_feat.items() if v and min(v) < -eps and max(v) > eps)


def _unavailable_payload(model_id: str, version: str, reason: str, detail: str) -> dict:
    """Estado TIPADO de degradación — jamás valores de atribución fabricados."""
    return {
        "nota": NOTA,
        "surface": "zoo",
        "asset": "usdcop",
        "model_id": model_id,
        "model_type": "tree",
        "method": "tree_shap_unavailable",
        "attribution_not_shap": False,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": ("sin atribuciones: el backend TreeSHAP no estuvo disponible en esta "
                  "corrida. NO se emiten valores — un artefacto inventado es peor que un "
                  "hueco declarado."),
        "status": "tree_shap_unavailable",
        "reason": reason,
        "detail": detail,
    }


def generate_zoo_tree(model_ids: tuple[str, ...] = ZOO_TREE_MODELS) -> list[Path]:
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
    df["regime"] = _regime_labels(df)

    version = pd.Timestamp(df["date"].iloc[-1]).date().isoformat()
    years = sorted({int(y) for y in df["date"].dt.year.unique()})

    # Folds expanding ANUALES: train = filas < 1-ene-Y (purgadas), test = filas de Y.
    # Una fila NUNCA se atribuye con un modelo que la vio en su train (invariante A.7).
    folds: list[dict] = []
    for yr in years:
        cut = pd.Timestamp(year=yr, month=1, day=1)
        prev = df[df["date"] < cut]
        if len(prev) <= HORIZON:
            continue
        train = prev.iloc[:-HORIZON]                       # purga: targets 5d no realizados
        m_ok = train[feat_cols].notna().all(axis=1) & train["y5"].notna()
        Xtr = train.loc[m_ok, feat_cols].to_numpy(float)
        ytr = train.loc[m_ok, "y5"].to_numpy(float)
        test = df[(df["date"] >= cut) & (df["date"] < cut.replace(year=yr + 1))]
        test = test[test[feat_cols].notna().all(axis=1)]
        if len(Xtr) < MIN_TRAIN or test.empty:
            continue
        folds.append({"year": yr, "Xtr": Xtr, "ytr": ytr, "test": test,
                      "train_end": pd.Timestamp(train.loc[m_ok, "date"].iloc[-1])})
    if not folds:
        raise RuntimeError("zoo tree: ningún fold anual cumple la guarda de train mínimo")

    paths: list[Path] = []
    for mid in model_ids:
        try:
            backend_name, shap_fn = _tree_shap_backend(mid)
        except Exception as exc:   # backend/librería ausente ⇒ degradación explícita
            p = _write("zoo", "usdcop", mid, version, _unavailable_payload(
                mid, version, "backend_import_failed", f"{type(exc).__name__}: {exc}"))
            paths.append(p)
            print(f"[tree] {mid}: DEGRADADO (backend_import_failed) -> {p.relative_to(REPO)}",
                  flush=True)
            continue

        try:
            phi_parts, base_parts, dates, fold_meta = [], [], [], []
            add_err = 0.0
            for f in folds:
                # Scaler train-only por fold. Los árboles no lo necesitan, pero es el MISMO
                # preproceso del zoo (meta01) — se mantiene para no cambiar el modelo.
                sc = StandardScaler().fit(f["Xtr"])
                Xte = sc.transform(f["test"][feat_cols].to_numpy(float))
                mdl = ModelFactory.create(mid)              # hiperparámetros CONGELADOS (defaults)
                mdl.fit(sc.transform(f["Xtr"]), f["ytr"])
                phi, bias = shap_fn(mdl, Xte)
                if phi.shape[1] != len(feat_cols):
                    raise RuntimeError(
                        f"{mid}: {phi.shape[1]} contribuciones vs {len(feat_cols)} features")
                # Chequeo de aditividad TreeSHAP: sum(phi) + bias == prediccion CRUDA.
                raw_pred = np.asarray(mdl._model.predict(Xte), float).ravel()
                add_err = max(add_err, float(np.nanmax(np.abs(phi.sum(axis=1) + bias - raw_pred))))
                phi_parts.append(phi)
                base_parts.append(bias)
                dates.append(f["test"][["date", "regime"]])
                fold_meta.append({"year": int(f["year"]), "n_train": int(len(f["Xtr"])),
                                  "n_test": int(len(Xte)),
                                  "train_end": f["train_end"].date().isoformat(),
                                  "base_value": float(np.nanmean(bias))})

            phi = np.vstack(phi_parts)
            meta = pd.concat(dates, ignore_index=True)
            base_value = float(np.nanmean(np.concatenate(base_parts)))
        except Exception as exc:   # fit/predict/SHAP falló ⇒ degradación explícita
            p = _write("zoo", "usdcop", mid, version, _unavailable_payload(
                mid, version, "shap_computation_failed", f"{type(exc).__name__}: {exc}"))
            paths.append(p)
            print(f"[tree] {mid}: DEGRADADO (shap_computation_failed) -> {p.relative_to(REPO)}",
                  flush=True)
            continue

        all_rows = np.ones(len(phi), dtype=bool)
        global_rows = _agg_rows(phi, feat_cols, all_rows)
        top_features = [{"rank": i + 1, **r} for i, r in enumerate(global_rows)]
        scale = float(np.nanmean([r["mean_abs_shap"] for r in global_rows]))

        yr_arr = meta["date"].dt.year.to_numpy()
        by_year = {str(int(y)): _agg_rows(phi, feat_cols, yr_arr == y)
                   for y in sorted(set(yr_arr))}
        reg_arr = meta["regime"].to_numpy()
        by_regime = {str(r): _agg_rows(phi, feat_cols, reg_arr == r)
                     for r in sorted(set(reg_arr))}

        payload = {
            "nota": NOTA,
            "surface": "zoo",
            "asset": "usdcop",
            "model_id": mid,
            "model_type": "tree",
            "method": "tree_shap",
            "attribution_not_shap": False,
            "version": version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "shap_backend": backend_name,
            "shap_package_available": _shap_package_available(),
            "additivity_max_abs_err": add_err,
            "fit": {
                "scheme": ("walk-forward EXPANDING ANUAL: fit con filas < 1-ene-Y menos purga "
                           f"de {HORIZON}d; atribucion SOLO sobre filas del año Y (test-fold)"),
                "origin": version,
                "n_train": int(sum(m["n_train"] for m in fold_meta)),
                "horizon": HORIZON,
                "purge_days": HORIZON,
                "scaler": "StandardScaler train-only por fold",
                "params": {k: v for k, v in (ModelFactory.create(mid).get_params() or {}).items()
                           if isinstance(v, (int, float, str, bool, type(None)))},
            },
            "folds": fold_meta,
            "scope": ("TreeSHAP EXACTO del booster nativo sobre filas OOS (ninguna fila fue "
                      "vista por el modelo que la atribuye). Hiperparametros = defaults "
                      "CONGELADOS del ModelFactory, sin tuning ni seleccion; ninguna metrica "
                      "de acierto computada (0 trials). La atribucion es sobre la salida CRUDA "
                      "del booster: el reescalado de varianza del wrapper predict() es un "
                      "post-proceso afin y NO se atribuye."),
            "base_value": base_value,
            "n_rows": int(len(phi)),
            "n_features": len(feat_cols),
            "n_folds": len(fold_meta),
            "top_features": top_features,
            "by_year": by_year,
            "by_regime": by_regime,
            "regime_gate": ("gate Hurst congelado de config/execution/smart_simple_v1.yaml "
                            "evaluado con retornos <= la propia fila (sin look-ahead)"),
            "kill_flags_sign_change_by_year": _sign_change_flags(by_year, feat_cols, scale),
            "kill_flags_sign_change_by_regime": _sign_change_flags(by_regime, feat_cols, scale),
        }
        p = _write("zoo", "usdcop", mid, version, payload)
        paths.append(p)
        print(f"[tree] {mid}: {p.relative_to(REPO)} (n_rows={len(phi)}, folds={len(fold_meta)}, "
              f"add_err={add_err:.2e})", flush=True)
    return paths


def _shap_package_available() -> bool:
    """¿Importa el paquete `shap`? Informativo: el TreeSHAP usado es el NATIVO del booster."""
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


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
                    help="modelos LINEALES del zoo (SHAP cerrado): ridge,bayesian_ridge")
    ap.add_argument("--tree-models", default=",".join(ZOO_TREE_MODELS),
                    help="modelos de ARBOL del zoo (TreeSHAP nativo): xgboost,lightgbm,catboost")
    ap.add_argument("--rules", default="spx500",
                    help="adapters rule-based (profitability_adapters.ADAPTERS)")
    ap.add_argument("--skip-zoo", action="store_true")
    ap.add_argument("--skip-trees", action="store_true")
    ap.add_argument("--skip-rules", action="store_true")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    if not args.skip_zoo:
        mids = tuple(m.strip() for m in args.zoo_models.split(",") if m.strip())
        bad = [m for m in mids if m not in ZOO_LINEAR_MODELS]
        if bad:
            raise SystemExit(f"--zoo-models solo admite lineales (SHAP cerrado): {bad} "
                             f"no permitido — usa --tree-models para arboles")
        paths += generate_zoo_linear(mids)
    if not args.skip_trees:
        tids = tuple(m.strip() for m in args.tree_models.split(",") if m.strip())
        bad = [m for m in tids if m not in ZOO_TREE_MODELS]
        if bad:
            raise SystemExit(f"--tree-models solo admite arboles del zoo: {bad} no permitido")
        paths += generate_zoo_tree(tids)
    if not args.skip_rules:
        rids = tuple(r.strip() for r in args.rules.split(",") if r.strip())
        paths += generate_rule_attribution(rids)

    print(f"OK: {len(paths)} artefactos")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
