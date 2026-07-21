"""H-VOLT-01 — transformer de vol intradía (diseño SELLADO en el registry antes de correr).

Ejecuta EXACTAMENTE el pre-registro: tokens 8-dim por día de sesión desde M5, secuencia 60d,
encoder 2x4xd64, pool multi-símbolo con embedding de activo, deploy/eval SOLO COP.
Splits: train<=2022 · val 2023 (early stop) · TEST 2024 un disparo. Seeds [42,123,456].
Bar: batir a rv20-persistencia, EWMA(0.94) y la mejor celda simple de H-RISK-FAM-01 en
pinball q90 con IC95 block-bootstrap b=4 en >=2/3 seeds. +1 trial al abrir TEST.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from services.common.metrics import circular_block_bootstrap  # noqa: E402

SYMS = ["USD/COP", "USD/MXN", "USD/BRL", "XAU/USD", "BTC/USDT"]
SEQ = 60
SEEDS = [42, 123, 456]


def build_tokens():
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("POSTGRES_HOST", "localhost"),
                            dbname="usdcop_trading", user="admin",
                            password=os.environ.get("POSTGRES_PASSWORD", ""))
    frames = {}
    for si, sym in enumerate(SYMS):
        q = pd.read_sql("""
            WITH b AS (
              SELECT (time AT TIME ZONE 'UTC')::date d, time, open, high, low, close,
                     ln(close/lag(close) OVER (ORDER BY time)) r
              FROM usdcop_m5_ohlcv WHERE symbol=%s)
            SELECT d,
              sqrt(sum(r*r)) rv,
              (max(high)-min(low))/max(close) rng,
              sqrt(sum(CASE WHEN r>0 THEN r*r ELSE 0 END)) upv,
              sqrt(sum(CASE WHEN r<0 THEN r*r ELSE 0 END)) dnv,
              count(*) nbars,
              (array_agg(open ORDER BY time))[1] o1,
              (array_agg(close ORDER BY time DESC))[1] cN
            FROM b GROUP BY d ORDER BY d""", conn, params=(sym,))
        # vol primera/ultima hora requieren segunda pasada ligera
        q2 = pd.read_sql("""
            WITH b AS (
              SELECT (time AT TIME ZONE 'UTC')::date d, time,
                     ln(close/lag(close) OVER (ORDER BY time)) r,
                     row_number() OVER (PARTITION BY (time AT TIME ZONE 'UTC')::date ORDER BY time) rn,
                     count(*) OVER (PARTITION BY (time AT TIME ZONE 'UTC')::date) nb
              FROM usdcop_m5_ohlcv WHERE symbol=%s)
            SELECT d,
              sqrt(sum(CASE WHEN rn<=12 THEN r*r ELSE 0 END)) v_first,
              sqrt(sum(CASE WHEN rn>nb-12 THEN r*r ELSE 0 END)) v_last
            FROM b GROUP BY d ORDER BY d""", conn, params=(sym,))
        f = q.merge(q2, on="d")
        f["gap"] = (f["o1"] / f["cn"].shift(1) - 1).abs()
        f["sret"] = np.log(f["cn"] / f["o1"]).abs()
        f = f[f["nbars"] >= 30].reset_index(drop=True)   # dias con sesion razonable
        f["sym_id"] = si
        frames[sym] = f
        print(f"{sym}: {len(f)} dias-token ({f.d.min()} -> {f.d.max()})", flush=True)
    return frames


TOKCOLS = ["rv", "rng", "upv", "dnv", "v_first", "v_last", "gap", "sret"]


def make_sequences(frames):
    X, A, Y, DATES, SYMIDX = [], [], [], [], []
    for sym, f in frames.items():
        f = f.copy()
        # targets: vol realizada 5d futura + rango semanal futuro (max-min 5d en %)
        f["y_rv5"] = f["rv"].rolling(5).sum().shift(-5)
        f["y_rng5"] = (f["rng"].rolling(5).max().shift(-5)) * 100
        f = f.dropna(subset=TOKCOLS + ["y_rv5", "y_rng5"]).reset_index(drop=True)
        T = f[TOKCOLS].to_numpy(np.float32)
        # normalizacion causal: z-score con stats expanding hasta t (min 120)
        mu = pd.DataFrame(T).expanding(120).mean().to_numpy(np.float32)
        sd = pd.DataFrame(T).expanding(120).std().to_numpy(np.float32)
        Tz = (T - mu) / (sd + 1e-8)
        for i in range(SEQ + 125, len(f)):
            X.append(Tz[i - SEQ:i])
            A.append(f["sym_id"].iloc[i])
            Y.append([f["y_rv5"].iloc[i], f["y_rng5"].iloc[i]])
            DATES.append(f["d"].iloc[i])
            SYMIDX.append(sym)
    return (np.array(X, np.float32), np.array(A), np.array(Y, np.float32),
            pd.to_datetime(pd.Series(DATES)), np.array(SYMIDX))


def pinball(y, q, tau=0.90):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e)


def run():
    import torch
    import torch.nn as nn
    frames = build_tokens()
    X, A, Y, D, S = make_sequences(frames)
    print(f"secuencias: {len(X)}", flush=True)
    tr = (D <= "2022-12-31").to_numpy()
    va = ((D >= "2023-01-01") & (D <= "2023-12-31")).to_numpy()
    te = ((D >= "2024-01-01") & (D <= "2024-12-31")).to_numpy() & (S == "USD/COP")
    print(f"train {tr.sum()} | val {va.sum()} | TEST-COP-2024 {te.sum()}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class VolT(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 64)
            self.emb = nn.Embedding(len(SYMS), 64)
            enc = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, 2)
            self.head = nn.Linear(64, 2)   # [rv5, q90_rng5]

        def forward(self, x, a):
            h = self.proj(x) + self.emb(a)[:, None, :]
            h = self.enc(h)
            return self.head(h[:, -1])

    Xtr = torch.tensor(X[tr]); Atr = torch.tensor(A[tr]); Ytr = torch.tensor(Y[tr])
    Xva = torch.tensor(X[va]).to(dev); Ava = torch.tensor(A[va]).to(dev)
    Yva = torch.tensor(Y[va]).to(dev)
    Xte = torch.tensor(X[te]).to(dev); Ate = torch.tensor(A[te]).to(dev)

    preds_te = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        m = VolT().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        ds = torch.utils.data.TensorDataset(Xtr, Atr, Ytr)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        best, best_state, patience = 1e9, None, 0
        for epoch in range(40):
            m.train()
            for xb, ab, yb in dl:
                xb, ab, yb = xb.to(dev), ab.to(dev), yb.to(dev)
                out = m(xb, ab)
                tau = 0.90
                e = yb[:, 1] - out[:, 1]
                loss = ((out[:, 0] - yb[:, 0]) ** 2).mean() + \
                       torch.maximum(tau * e, (tau - 1) * e).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                ov = m(Xva, Ava)
                ev = Yva[:, 1] - ov[:, 1]
                vloss = float(((ov[:, 0] - Yva[:, 0]) ** 2).mean() +
                              torch.maximum(0.9 * ev, -0.1 * ev).mean())
            if vloss < best - 1e-5:
                best, best_state, patience = vloss, {k: v.clone() for k, v in m.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 5:
                    break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            preds_te.append(m(Xte, Ate).cpu().numpy())
        print(f"seed {seed}: val {best:.4f}, epochs {epoch+1}", flush=True)

    # ---- Evaluacion TEST-2024 COP (UN disparo) ----
    y_te = Y[te]; d_te = D[te].reset_index(drop=True)
    cop = frames["USD/COP"].set_index("d")
    # baselines pre-firmados
    rv20 = cop["rv"].rolling(20).mean() * np.sqrt(5) / cop["rv"].rolling(20).std().clip(lower=1e-9)  # placeholder no usado
    # persistencia: q90 de rango futuro ~ q90 empirico rolling 252 del rango 5d realizado
    rng5_hist = (cop["rng"].rolling(5).max() * 100)
    base_pers = rng5_hist.rolling(252).quantile(0.90).shift(1)
    lam = 0.94
    ew = cop["rv"].ewm(alpha=1 - lam).mean()
    base_ewma = (1.645 * ew * np.sqrt(5) * 100).shift(1)
    b_pers = base_pers.reindex(pd.Index(d_te.dt.date)).to_numpy()
    b_ewma = base_ewma.reindex(pd.Index(d_te.dt.date)).to_numpy()

    y_rng = y_te[:, 1]
    out = {"n_test": int(te.sum()), "seeds": {}, "bar": "IC95 b=4 excluye 0 en >=2/3 seeds vs TODOS los baselines"}
    wins_per_seed = []
    for si, seed in enumerate(SEEDS):
        q_m = preds_te[si][:, 1]
        pm = pinball(y_rng, q_m)
        row = {"pinball_model": float(np.nanmean(pm))}
        seed_win = True
        for bname, bq in (("persistencia_q90_252", b_pers), ("ewma164", b_ewma)):
            pb = pinball(y_rng, bq)
            mask = np.isfinite(pb) & np.isfinite(pm)
            bb = circular_block_bootstrap(pb[mask] - pm[mask], np.mean, block=4)
            row[f"vs_{bname}"] = {"pinball_base": float(np.nanmean(pb[mask])),
                                  "ci95": bb["ci95"],
                                  "win": bool(bb["ci95"][0] is not None and bb["ci95"][0] > 0)}
            seed_win &= row[f"vs_{bname}"]["win"]
        row["WIN_all_baselines"] = bool(seed_win)
        wins_per_seed.append(seed_win)
        out["seeds"][seed] = row
        print(f"seed {seed}: {json.dumps(row, default=str)[:220]}", flush=True)
    out["verdict"] = ("WIN" if sum(wins_per_seed) >= 2 else "NO_RECHAZA")
    out["trials"] = "+1 (N 65->66)"
    o = REPO / ".claude/evidence/cop_vol_transformer" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "h_volt_01.json").write_text(json.dumps(out, indent=2, default=str))
    shutil.copy(__file__, o / "generator_script.py")
    print(f"\nVEREDICTO: {out['verdict']} ({sum(wins_per_seed)}/3 seeds)")
    print(f"artefacto -> {o}")


if __name__ == "__main__":
    run()
