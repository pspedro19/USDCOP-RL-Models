"""BL-39 bit-check — reproduce the v11 weekly signal from feature_set + snapshot.

PROTOCOL §6 ("Bit-check para lo congelado"): v11 is FROZEN, so the criterion for
anything touching it is EXACT reproduction. This script computes the H5 weekly
signal of the last artifact week through TWO independent paths and requires them
to be bit-identical (raw float64 bytes, not approximate equality):

  PATH A (as-built pipeline): exactly what forecast_h5_l5_weekly_signal.py's
    generate_signal() does — read h5_l5a_pred_features_temp.parquet, select the
    columns of feature_cols_h5.json, scale with scaler_h5.pkl, predict with
    ridge_h5.pkl + bayesian_ridge_h5.pkl, ensemble mean.

  PATH B (declared contracts): the SAME computation driven ONLY by the BL-39
    declarations — config/features/feature_sets/usdcop_smart_simple_v11.yaml
    (dag_snapshot ordered list) + config/features/normalization_snapshots/
    usdcop_h5_scaler_legacy_v1.yaml (artifact pointer + hashes).

A == B bit-for-bit proves the declared feature_set + normalization snapshot
reproduce the frozen pipeline (legacy_v1 bit-identico). Along the way every
artifact hash is verified against the frozen manifest's as-of registration.

NOTE on scope (declared honestly): weights rotate weekly and only the as-of
2026-07-06 snapshot exists on disk; the persisted DB signals (last row
2026-06-22) were produced by an EARLIER week's weights, which no longer exist —
so cross-checking against that DB row is impossible by construction. The
"señal de la última semana" reproduced here is the one the frozen artifacts
produce (features through 2026-07-02, Monday signal week).

Run:  python scripts/validation/bitcheck_v11_signal.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "outputs" / "forecasting" / "h5_weekly_models" / "latest"
PRED_PATH = ROOT / "outputs" / "forecasting" / "h5_l5a_pred_features_temp.parquet"
FEATURE_SET = ROOT / "config" / "features" / "feature_sets" / "usdcop_smart_simple_v11.yaml"
NORM_SNAP = (ROOT / "config" / "features" / "normalization_snapshots"
             / "usdcop_h5_scaler_legacy_v1.yaml")
MANIFEST = ROOT / "config" / "strategy_manifests" / "usdcop.yaml"

H5_MODEL_IDS = ("ridge", "bayesian_ridge")  # == forecast_h5_l5_weekly_signal.py


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_bitcheck(root: Path = ROOT) -> dict:
    import joblib
    import numpy as np
    import pandas as pd

    # The frozen .pkl artifacts reference classes under `src.*`: the repo root
    # must be importable for joblib to unpickle them (pytest adds it; the CLI
    # must too).
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    models_dir = root / "outputs" / "forecasting" / "h5_weekly_models" / "latest"
    pred_path = root / "outputs" / "forecasting" / "h5_l5a_pred_features_temp.parquet"
    errors: list[str] = []
    report: dict = {"ok": False, "errors": errors}

    for p in (pred_path, models_dir / "feature_cols_h5.json",
              models_dir / "scaler_h5.pkl", FEATURE_SET, NORM_SNAP):
        if not p.is_file():
            errors.append(f"missing required input: {p}")
    if errors:
        return report

    # ── Hash wall: artifacts must be the manifest's frozen as-of bytes ──────
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    frozen = manifest["components"][0]["current_model_snapshot"]["artifacts_sha256_16"]
    for fname, expected in frozen.items():
        current = _sha16(models_dir / fname)
        if current != expected:
            errors.append(
                f"{fname}: disk hash {current} != manifest as-of {expected} — "
                "artifacts rotated; register a NEW snapshot before bit-checking")
    if errors:
        return report

    df_pred = pd.read_parquet(pred_path)

    # ── PATH A: as-built L5b (mirrors generate_signal line by line) ─────────
    with open(models_dir / "feature_cols_h5.json") as f:
        cols_a = list(json.load(f))
    X_a = df_pred[cols_a].iloc[-1:].values.astype(np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn version-skew warning (declared in snapshot)
        scaler_a = joblib.load(models_dir / "scaler_h5.pkl")
        Xs_a = scaler_a.transform(X_a)
        preds_a = {}
        for model_id in H5_MODEL_IDS:
            model = joblib.load(models_dir / f"{model_id}_h5.pkl")
            preds_a[model_id] = float(model.predict(Xs_a)[0])
    ensemble_a = float(np.mean(list(preds_a.values())))

    # ── PATH B: from the DECLARED feature_set + normalization snapshot ──────
    fs = yaml.safe_load(FEATURE_SET.read_text(encoding="utf-8"))
    snap = yaml.safe_load(NORM_SNAP.read_text(encoding="utf-8"))
    cols_b = list(fs["dag_snapshot"]["ordered_features"])

    # The declared list must serialize to the EXACT bytes of the frozen file.
    declared_bytes_hash = hashlib.sha256(json.dumps(cols_b).encode()).hexdigest()[:16]
    if declared_bytes_hash != fs["dag_snapshot"]["file_sha256_16"]:
        errors.append(
            f"declared dag_snapshot list hashes to {declared_bytes_hash}, "
            f"pinned file_sha256_16 is {fs['dag_snapshot']['file_sha256_16']}")
    if cols_b != cols_a:
        errors.append("declared ordered_features != feature_cols_h5.json content")

    scaler_path = root / snap["artifact"]["path"]
    if _sha16(scaler_path) != snap["artifact"]["sha256_16"]:
        errors.append("scaler artifact hash != normalization snapshot registration")
    if errors:
        return report

    X_b = df_pred[cols_b].iloc[-1:].values.astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler_b = joblib.load(scaler_path)
        Xs_b = scaler_b.transform(X_b)
        preds_b = {}
        for model_id in H5_MODEL_IDS:
            model = joblib.load(models_dir / f"{model_id}_h5.pkl")
            preds_b[model_id] = float(model.predict(Xs_b)[0])
    ensemble_b = float(np.mean(list(preds_b.values())))

    # Semantic hash of the normalization artifact (formula declared in snapshot).
    h = hashlib.sha256()
    h.update(json.dumps(cols_b).encode())
    h.update(np.asarray(scaler_b.mean_, dtype=np.float64).tobytes())
    h.update(np.asarray(scaler_b.scale_, dtype=np.float64).tobytes())
    if h.hexdigest()[:16] != snap["semantic_hash_sha256_16"]:
        errors.append(
            f"semantic hash {h.hexdigest()[:16]} != registered "
            f"{snap['semantic_hash_sha256_16']}")

    # ── Bit comparison (bytes, not tolerance) ───────────────────────────────
    report["feature_row_bit_identical"] = X_a.tobytes() == X_b.tobytes()
    report["scaled_row_bit_identical"] = Xs_a.tobytes() == Xs_b.tobytes()
    report["predictions_bit_identical"] = all(
        np.float64(preds_a[m]).tobytes() == np.float64(preds_b[m]).tobytes()
        for m in H5_MODEL_IDS)
    report["ensemble_return_bit_identical"] = (
        np.float64(ensemble_a).tobytes() == np.float64(ensemble_b).tobytes())
    for key in ("feature_row_bit_identical", "scaled_row_bit_identical",
                "predictions_bit_identical", "ensemble_return_bit_identical"):
        if not report[key]:
            errors.append(f"BIT MISMATCH: {key}")

    # Signal fields with the DAG's exact rounding (for the human-readable report).
    latest_close = float(df_pred["close"].iloc[-1])
    report["signal"] = {
        "features_through": str(pd.to_datetime(df_pred["date"].iloc[-1]).date()),
        "base_price": latest_close,
        "per_model": {
            m: {
                "predicted_return_pct": round(preds_a[m] * 100, 4),
                "predicted_price": round(latest_close * float(np.exp(preds_a[m])), 4),
                "direction": "UP" if preds_a[m] > 0 else "DOWN",
            } for m in H5_MODEL_IDS
        },
        "ensemble_return": ensemble_a,
        "direction": 1 if ensemble_a > 0 else -1,
    }
    report["ok"] = not errors
    return report


def main() -> int:
    report = run_bitcheck(ROOT)
    if not report["ok"]:
        print("[FAIL] v11 bit-check:")
        for e in report["errors"]:
            print(f"  - {e}")
        return 1
    sig = report["signal"]
    print("[OK] v11 signal reproduced from feature_set + normalization snapshot "
          "== as-built pipeline (bit-identical)")
    print(f"  features_through={sig['features_through']} base_price={sig['base_price']}")
    for m, p in sig["per_model"].items():
        print(f"  {m}: return={p['predicted_return_pct']}% "
              f"price={p['predicted_price']} dir={p['direction']}")
    print(f"  ensemble_return={sig['ensemble_return']:.6f} direction={sig['direction']}")
    for key in ("feature_row_bit_identical", "scaled_row_bit_identical",
                "predictions_bit_identical", "ensemble_return_bit_identical"):
        print(f"  {key}: {report[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
