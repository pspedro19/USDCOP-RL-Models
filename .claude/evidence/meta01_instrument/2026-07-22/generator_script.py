"""H-META-01 INSTRUMENTO — ledger walk-forward del zoo con ORIGEN VIERNES-ANTERIOR.

Pre-registro sellado + enmienda #1 (registry 2026-07-22). 0 trials: este script SOLO
genera predicciones crudas; NO computa ninguna métrica de acierto/performance (mirarlas
sería abrir celdas). Reglas del diseño que implementa:
- Origen de la semana t = ÚLTIMO día de trading ESTRICTAMENTE ANTERIOR al lunes de t
  (Codex R1: el generador estándar corta el viernes de la MISMA semana = fuga).
- Zoo fijo de 7 modelos (sin ridge/bayesian_ridge): ard, xgboost_pure, lightgbm_pure,
  catboost_pure, hybrid_xgboost, hybrid_lightgbm, hybrid_catboost.
- Por origen: fit en datos <= origen con purga de 5 días (el target 5d de las últimas
  5 filas no está realizado), scaler train-only, predicción cruda H=5 de la última fila.
- Se persisten SOLO predicciones crudas + metadatos/hash. Prohibido backfill.
Salida: data/pipeline/meta01/zoo_ledger.parquet (gitignored-regenerable) + copia en
.claude/evidence/meta01_instrument/2026-07-22/ con generator y hash.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MODELS = ["ard", "xgboost_pure", "lightgbm_pure", "catboost_pure",
          "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"]
HORIZON = 5
START = pd.Timestamp("2020-01-01")


def main() -> int:
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.ssot_config import ForecastingSSOTConfig
    from src.forecasting.dataset_loader import ForecastingDatasetLoader

    cfg = ForecastingSSOTConfig.load()
    loader = ForecastingDatasetLoader(cfg, project_root=REPO)
    df, _ = loader.load_dataset()
    feat_cols = [c for c in cfg.get_feature_columns() if c in df.columns]
    df = df.sort_values("date").reset_index(drop=True)
    print(f"dataset: {len(df)} filas, {len(feat_cols)} features, "
          f"{df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}", flush=True)

    df["y5"] = df["close"].shift(-HORIZON) / df["close"] - 1.0

    # origen = ultimo dia de trading < lunes, para cada semana ISO desde START
    df["week"] = df["date"].dt.to_period("W-SUN")
    weeks = sorted(w for w in df["week"].unique() if w.start_time >= START)
    rows = []
    for k, wk in enumerate(weeks):
        monday = wk.start_time
        prev = df[df["date"] < monday]
        if len(prev) < 400:
            continue
        origin = prev["date"].iloc[-1]
        train = prev.iloc[:-HORIZON]                    # purga: targets no realizados
        m_ok = train[feat_cols].notna().all(axis=1) & train["y5"].notna()
        Xtr, ytr = train.loc[m_ok, feat_cols].to_numpy(), train.loc[m_ok, "y5"].to_numpy()
        x_last = prev[feat_cols].iloc[-1:].to_numpy()
        if len(Xtr) < 200 or not np.isfinite(x_last).all():
            continue
        sc = StandardScaler().fit(Xtr)
        Xs, xs = sc.transform(Xtr), sc.transform(x_last)
        for mid in MODELS:
            try:
                mdl = ModelFactory.create(mid)
                mdl.fit(Xs, ytr)
                pred = float(np.asarray(mdl.predict(xs)).ravel()[0])
            except Exception as e:
                pred = np.nan
            rows.append({"monday": monday, "origin": origin, "model_id": mid,
                         "pred_h5": pred, "n_train": int(len(Xtr))})
        if k % 25 == 0:
            print(f"  {k}/{len(weeks)} semanas ({monday.date()})", flush=True)

    L = pd.DataFrame(rows)
    out_dir = REPO / "data/pipeline/meta01"
    out_dir.mkdir(parents=True, exist_ok=True)
    L.to_parquet(out_dir / "zoo_ledger.parquet", index=False)
    h = hashlib.sha256(pd.util.hash_pandas_object(L[["monday", "model_id", "pred_h5"]]
                                                  .fillna(0)).values.tobytes()).hexdigest()[:16]
    ev = REPO / ".claude/evidence/meta01_instrument" / date.today().isoformat()
    ev.mkdir(parents=True, exist_ok=True)
    L.to_parquet(ev / "zoo_ledger.parquet", index=False)
    meta = {"n_rows": len(L), "n_weeks": int(L["monday"].nunique()),
            "models": MODELS, "horizon": HORIZON, "ledger_sha16": h,
            "origen": "ultimo trading day < lunes (viernes-anterior)",
            "purga_fit": "5 dias (targets no realizados excluidos del train)",
            "nota": "SOLO predicciones crudas; ninguna métrica de acierto computada (0 trials)"}
    (ev / "instrument_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    shutil.copy(__file__, ev / "generator_script.py")
    print(f"ledger: {len(L)} filas, {L['monday'].nunique()} semanas, sha16={h}")
    print(f"NaN preds: {L['pred_h5'].isna().sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
