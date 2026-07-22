"""H-CHRONOS-01 — chronos-bolt-base zero-shot vs persistencia-252 en q90 del rango 5-sesiones.

EJECUTA EXACTAMENTE el pre-registro sellado 2026-07-22 (15 puntos, HYPOTHESIS-REGISTRY):
modelo único con revisión HF < 2025-01-01, input = historial causal de y (max rango 5
sesiones) hasta y_{t-5}, horizon=5 canal q90 nativo del paso 5, gate = Δ pinball pareado
agregado por semana ISO vs persistencia-252, IC95 block-bootstrap b=4/2000/seed42,
WIN solo si borde SUPERIOR < 0. Smoke ≤2023 funcional (--smoke). TEST 2025 un disparo.
+1 trial al abrir 2025 (N 71→72). Fallos >5% de orígenes ⇒ corrida INVÁLIDA.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from services.common.metrics import circular_block_bootstrap  # noqa: E402

MODEL_ID = "amazon/chronos-bolt-base"
CUTOFF = datetime(2025, 1, 1, tzinfo=timezone.utc)
HORIZON = 5
TAU = 0.90
SEED = 42
N_BOOT = 2000
BLOCK = 4


def pinned_revision():
    from huggingface_hub import HfApi
    commits = HfApi().list_repo_commits(MODEL_ID)
    pre = [c for c in commits if c.created_at < CUTOFF]
    if not pre:
        raise RuntimeError("ningún commit del modelo anterior a 2025-01-01")
    pre.sort(key=lambda c: c.created_at)
    c = pre[-1]
    return c.commit_id, c.created_at.isoformat()


def load_range_series():
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("POSTGRES_HOST", "localhost"),
                            dbname="usdcop_trading", user="admin",
                            password=os.environ.get("POSTGRES_PASSWORD", ""))
    q = pd.read_sql("""
        WITH b AS (
          SELECT (time AT TIME ZONE 'UTC')::date d, time, high, low, close
          FROM usdcop_m5_ohlcv WHERE symbol='USD/COP')
        SELECT d, (max(high)-min(low))/ (array_agg(close ORDER BY time DESC))[1] * 100 rng,
               count(*) nbars
        FROM b GROUP BY d ORDER BY d""", conn)
    q = q[q["nbars"] >= 30].reset_index(drop=True)      # constructor H-VOLT-01
    return q[["d", "rng"]]


def main() -> int:
    smoke = "--smoke" in sys.argv
    import torch
    from chronos import BaseChronosPipeline

    rev, rev_date = pinned_revision()
    print(f"revision fijada: {rev} ({rev_date})", flush=True)

    f = load_range_series()
    # y_t = max rango de las PROXIMAS 5 sesiones elegibles (punto 4)
    f["y"] = f["rng"].shift(-1).rolling(HORIZON).max().shift(-(HORIZON - 1))
    y = f["y"].to_numpy(np.float32)
    dates = pd.to_datetime(f["d"])
    print(f"sesiones: {len(f)} ({f.d.min()} -> {f.d.max()})", flush=True)

    pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, revision=rev,
                                               device_map="cpu",
                                               torch_dtype=torch.float32)
    ctx_max = 2048

    year = 2023 if smoke else 2025
    origins = np.where((dates.dt.year == year))[0]
    # el origen t necesita y_t realizado (etiqueta; dic usa enero del año siguiente solo
    # como etiqueta) y >=252+5 historia observable
    origins = [t for t in origins if t >= 300 and np.isfinite(y[t])]
    if smoke:
        origins = origins[:10]
    print(f"origenes {year}: {len(origins)}", flush=True)

    rows, failures = [], 0
    for k, t in enumerate(origins):
        hist = y[: t - HORIZON + 1]            # observable: hasta y_{t-5} (punto 5)
        hist = hist[np.isfinite(hist)][-ctx_max:]
        base = float(np.quantile(hist[-252:], TAU))   # persistencia-252 mismo cutoff
        try:
            qs, _ = pipe.predict_quantiles(
                torch.tensor(hist, dtype=torch.float32),
                prediction_length=HORIZON, quantile_levels=[TAU])
            q_model = float(qs[0, HORIZON - 1, 0])    # canal q90 nativo, paso 5 = y_t
        except Exception as e:                        # politica de fallo (punto 14)
            failures += 1
            print(f"  fallo origen {dates.iloc[t].date()}: {e}", flush=True)
            continue
        rows.append({"d": dates.iloc[t], "y": float(y[t]),
                     "q_model": q_model, "q_base": base})
        if k % 50 == 0:
            print(f"  {k}/{len(origins)}", flush=True)

    if failures > 0.05 * max(len(origins), 1):
        print("CORRIDA INVALIDA: >5% de origenes fallidos")
        return 1

    r = pd.DataFrame(rows)
    e_m = r["y"] - r["q_model"]
    e_b = r["y"] - r["q_base"]
    r["loss_m"] = np.maximum(TAU * e_m, (TAU - 1) * e_m)
    r["loss_b"] = np.maximum(TAU * e_b, (TAU - 1) * e_b)

    if smoke:
        print(f"SMOKE OK: {len(r)} origenes sin metricas comparativas (punto 13)")
        return 0

    # agregacion por semana ISO del origen (punto 10)
    r["w"] = r["d"].dt.strftime("%G-W%V")
    wk = r.groupby("w")[["loss_m", "loss_b"]].mean()
    delta = (wk["loss_m"] - wk["loss_b"]).to_numpy()
    bb = circular_block_bootstrap(delta, np.mean, n_boot=N_BOOT, block=BLOCK, seed=SEED)
    ci = bb["ci95"]
    win = ci[1] is not None and ci[1] < 0                 # punto 12
    out = {"model": MODEL_ID, "revision": rev, "revision_date": rev_date,
           "n_origins": len(r), "failures": failures, "n_weeks": len(wk),
           "pinball_model_weekly": float(wk["loss_m"].mean()),
           "pinball_persistencia_weekly": float(wk["loss_b"].mean()),
           "delta_mean": float(np.mean(delta)), "delta_ci95_block4": ci,
           "verdict": "WIN" if win else "NO_RECHAZA",
           "trials": "+1 (N 71->72)",
           "versions": {"torch": torch.__version__,
                        "chronos": __import__("chronos").__version__,
                        "pandas": pd.__version__, "numpy": np.__version__}}
    o = REPO / ".claude/evidence/cop_chronos" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "h_chronos_01.json").write_text(json.dumps(out, indent=2, default=str))
    r.to_csv(o / "origins_2025.csv", index=False)
    shutil.copy(__file__, o / "generator_script.py")
    print(json.dumps({k: v for k, v in out.items() if k != "versions"}, indent=2))
    print(f"artefacto -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
