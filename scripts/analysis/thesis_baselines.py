#!/usr/bin/env python3
"""Piso de baselines de la tesis: el bar que todo brazo debe batir.

Contract: CTR-RESEARCH-BASELINES-001 · Date: 2026-08-24

Produce la **tabla 4.3** de §14 sobre las sesiones efectivas del hold-out, y los baselines
que `quant-constitution.md` §3 exige **antes** de cualquier PROMOTE.

## Por qué los baselines viven en el mismo entorno que el agente

B1 es `w=+1` toda la sesión; NULL-A es `w=-1`; always-flat es `w=0`. Son políticas fijas de
`src/research/session_env.py`, no un cálculo aparte. Eso significa que comparten motor de
costos, cronología de §9.1 y contabilidad de §9.2 con el PPO **por construcción**, no por un
acuerdo entre dos implementaciones que podrían divergir sin que nadie lo note. Una
comparación en la que el agente y su baseline se costean distinto no mide skill: mide la
diferencia entre dos hojas de cálculo.

## Los cinco que pide la constitución

| Baseline | La pregunta que responde |
|---|---|
| **B1** buy & hold 1x | ¿bate a estar largo y no hacer nada? |
| **B1'** exposición emparejada | ¿o solo tiene menos beta? |
| **NULL-A** siempre corto 1x | ¿aporta algo sobre el baseline tonto del track? |
| random | ¿bate al azar? |
| always-flat | ¿bate a NO operar? |
| stress x1/x2/x3 | ¿sobrevive con los costos al doble? Si no: REJECT (§3.4) |

## Anualización derivada, no de manual

El factor **no** es 252. La máscara excluye festivos y sesiones incompletas, así que el
hold-out tiene **220,8 sesiones/año** medidas sobre el propio bloque. Usar 252 inflaría el
Sharpe anualizado un ~7% por pura convención.

## Lo que este script NO hace

No decide nada. No hay parámetros que ajustar: las políticas son constantes, el costo sale
del contrato congelado y la máscara está hasheada. Es aritmética sobre artefactos, que es lo
que §14 exige para que las cifras sean auditables.

Uso:
    python -m scripts.analysis.thesis_baselines
    python -m scripts.analysis.thesis_baselines --block development   # sensibilidad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.common.metrics import (  # noqa: E402
    calculate_max_drawdown, cost_stress, paired_exposure_baseline, sharpe_ratio_stderr)
from src.research.evaluation_mask import build_mask  # noqa: E402
from src.research.regime_hmm import (  # noqa: E402
    build_regime_observations, fit_frozen, spread_series)
from src.research.session_env import (  # noqa: E402
    BARS_PER_SESSION, constant_policy, daily_series, equity_curve, random_policy,
    run_policy, run_session)

PARTITION = ROOT / "config" / "research" / "partition.yaml"
SEED = ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
OUT_DIR = ROOT / "outputs" / "thesis"

# Con menos de 20 sesiones operadas no se reportan Sharpe ni p-value (constitución §6).
MIN_SESSIONS_FOR_RATIOS = 20

# Round-trip de referencia para el B1 pasivo (entrada y salida unicas). Se usa el nivel
# medio declarado en §8.4 porque una operacion unica no tiene un regimen que promediar.
MIN_ROUND_TRIP_REFERENCE_PIPS = 4.0


def load_sessions(valid: set) -> dict:
    """`{fecha: close de 60 barras}` para las sesiones de la máscara."""
    df = pd.read_parquet(SEED)
    t = pd.to_datetime(df["time"])
    keep = df["symbol"].astype(str).str.upper().str.replace("/", "", regex=False) == "USDCOP"
    df, t = df[keep], t[keep]
    df = df.assign(_t=t, _d=t.dt.date).sort_values("_t")
    df = df[df["_d"].isin(valid)]
    out = {}
    for d, g in df.groupby("_d"):
        c = g["close"].astype(float).to_numpy()
        if len(c) == BARS_PER_SESSION:      # la máscara ya lo garantiza; se comprueba igual
            out[d] = c
    return out


def annualization_factor(dates) -> float:
    """Sesiones efectivas por año, medidas sobre el propio bloque."""
    idx = pd.to_datetime(sorted(dates))
    years = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / years if years > 0 else float(len(idx))


def summarize(name: str, results, ann: float, costs_arr=None) -> dict:
    """Métricas de un baseline. Con N<20 se reportan solo conteo y PnL (§6)."""
    r = daily_series(results)
    eq = equity_curve(results)
    n = len(r)
    traded = sum(1 for x in results if x.n_changes > 0)

    row = {
        "baseline": name,
        "n_sessions": n,
        "n_traded": traded,
        "total_return_pct": round(100.0 * (eq[-1] / eq[0] - 1.0), 3),
        "mean_abs_exposure": round(float(np.mean([x.mean_abs_exposure for x in results])), 4),
        "total_cost_pct": round(100.0 * float(np.sum([x.total_cost for x in results])), 3),
    }

    if traded < MIN_SESSIONS_FOR_RATIOS:
        row.update({"sharpe": None, "sharpe_ci95": None, "calmar": None,
                    "max_dd_pct": None,
                    "note": f"N={traded} < {MIN_SESSIONS_FOR_RATIOS}: sin Sharpe ni p-value "
                            "(quant-constitution §6)"})
        return row

    sd = float(np.std(r, ddof=1))
    sharpe_per = float(np.mean(r) / sd) if sd > 0 else 0.0
    sharpe_ann = sharpe_per * np.sqrt(ann)
    se_ann = sharpe_ratio_stderr(r, adjust_autocorr=True) * np.sqrt(ann)
    ann_ret = (eq[-1] / eq[0]) ** (ann / n) - 1.0
    mdd = calculate_max_drawdown(eq)

    row.update({
        "sharpe": round(sharpe_ann, 3),
        "sharpe_ci95": [round(sharpe_ann - 1.96 * se_ann, 3),
                        round(sharpe_ann + 1.96 * se_ann, 3)],
        "ann_return_pct": round(100.0 * ann_ret, 3),
        "max_dd_pct": round(100.0 * abs(mdd), 3),
        "calmar": round(ann_ret / abs(mdd), 3) if mdd else None,
    })
    return row


def momentum_policy(close: np.ndarray) -> np.ndarray:
    """Regla fija de momentum: signo del retorno de las últimas 3 barras."""
    c = np.asarray(close, dtype=float)
    out = np.zeros(59, dtype=float)
    for b in range(59):
        if b >= 3:
            out[b] = np.sign(c[b] / c[b - 3] - 1.0)
    return np.clip(out, -1.0, 1.0)


def mean_reversion_policy(close: np.ndarray) -> np.ndarray:
    """Regla fija MR: posición contraria al z-score de una ventana de 12 barras."""
    c = np.asarray(close, dtype=float)
    out = np.zeros(59, dtype=float)
    for b in range(59):
        if b >= 12:
            window = np.log(c[b - 11:b + 1] / c[b - 12:b])
            sd = float(np.std(window, ddof=1))
            if sd > 0:
                z = float((window[-1] - np.mean(window)) / sd)
                out[b] = -1.0 if z > 1.0 else (1.0 if z < -1.0 else 0.0)
    return out


def opening_range_policy(close: np.ndarray) -> np.ndarray:
    """Opening-range breakout fijo: compara cada cierre con el rango de barras 0–5."""
    c = np.asarray(close, dtype=float)
    out = np.zeros(59, dtype=float)
    hi, lo = float(np.max(c[:6])), float(np.min(c[:6]))
    out[6:] = np.where(c[6:59] > hi, 1.0, np.where(c[6:59] < lo, -1.0, 0.0))
    return out


def regime_rules_policy(state: int) -> np.ndarray:
    """Dos reglas publicadas: shock→corto, intermedio_2→largo, resto flat."""
    level = -1.0 if state == 3 else (1.0 if state == 2 else 0.0)
    return np.full(59, level, dtype=float)



def passive_buy_hold(sessions: dict, ann: float) -> dict:
    """**B1 de la constitucion**: exposicion 1x al activo, manteniendo overnight.

    NO es lo mismo que `B1_buy_hold_1x` del entorno, y la diferencia importa tanto que
    reportar solo uno de los dos falsearia la comparacion:

      * Aqui: se compra una vez, se mantiene, se vende una vez. Dos operaciones en total.
        Es el benchmark pasivo que §3.1 de `quant-constitution.md` llama "buy&hold /
        exposicion 1x del activo".
      * En el entorno: §9.1 fuerza `w=0` al cierre de CADA sesion y cobra ese cierre, asi
        que "estar siempre largo" cuesta 584 round-trips. Eso no es buy&hold: es el techo
        de lo que puede hacer un sistema acotado por sesion que nunca se pone corto.

    Medido sobre el hold-out: el pasivo pierde 21,1% y el de sesion 56,1%, y la diferencia
    es casi toda costo. Usar el segundo como unico "B1" pondria un bar artificialmente
    bajo — cualquier brazo lo bateria sin tener skill, que es exactamente lo que un
    baseline existe para impedir.
    """
    dates = sorted(sessions)
    closes = np.asarray([sessions[d][-1] for d in dates], dtype=float)
    daily = closes[1:] / closes[:-1] - 1.0

    # Dos round-trips en toda la ventana, no 584.
    entry_exit_cost = 2.0 * MIN_ROUND_TRIP_REFERENCE_PIPS / float(closes[0])
    eq = np.cumprod(1.0 + daily)
    total = float(eq[-1]) - 1.0 - entry_exit_cost
    n = len(daily)
    sd = float(np.std(daily, ddof=1))
    sharpe_ann = (float(np.mean(daily)) / sd) * np.sqrt(ann) if sd > 0 else 0.0
    se_ann = sharpe_ratio_stderr(daily, adjust_autocorr=True) * np.sqrt(ann)
    ann_ret = (1.0 + total) ** (ann / n) - 1.0
    mdd = calculate_max_drawdown(np.concatenate([[1.0], eq]))
    return {
        "baseline": "B1_passive_buy_hold_overnight",
        "n_sessions": n + 1, "n_traded": 2,
        "total_return_pct": round(100.0 * total, 3),
        "mean_abs_exposure": 1.0,
        "total_cost_pct": round(100.0 * entry_exit_cost, 4),
        "sharpe": round(sharpe_ann, 3),
        "sharpe_ci95": [round(sharpe_ann - 1.96 * se_ann, 3),
                        round(sharpe_ann + 1.96 * se_ann, 3)],
        "ann_return_pct": round(100.0 * ann_ret, 3),
        "max_dd_pct": round(100.0 * abs(mdd), 3),
        "calmar": round(ann_ret / abs(mdd), 3) if mdd else None,
        "note": "benchmark PASIVO (constitucion §3.1): mantiene overnight, 2 operaciones "
                "en total. `B1_buy_hold_1x` es distinto: acotado por sesion, paga un "
                "round-trip por dia.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", default="holdout", choices=["development", "selection", "holdout"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    part = yaml.safe_load(PARTITION.read_text(encoding="utf-8"))
    blk = part["blocks"][a.block]
    mask = build_mask()

    print(f"=== baselines · bloque {a.block} ({blk['start']} -> {blk['end']}) ===")
    print(f"mascara: {len(mask)} sesiones validas en total, hash {mask.sha256[:16]}")

    # Régimen: fit SOLO sobre desarrollo, siempre — incluso cuando se evalúa otro bloque.
    obs = build_regime_observations(pd.read_parquet(SEED), valid_sessions=mask.valid)
    dev_blk = part["blocks"]["development"]
    dev = obs[(obs.index >= dev_blk["start"]) & (obs.index <= dev_blk["end"])]
    model = fit_frozen(dev)
    print(f"HMM congelado sobre desarrollo: K={model.k} {model.state_labels()}")

    sp = spread_series(model, obs).dropna(subset=["spread_pips"])
    spreads = {d.date(): float(v) for d, v in sp["spread_pips"].items()}

    in_block = set(mask.in_block(blk["start"], blk["end"]))
    sessions = {d: c for d, c in load_sessions(in_block).items() if d in spreads}
    print(f"sesiones evaluables: {len(sessions)} (de {len(in_block)} en la mascara)")

    ann = annualization_factor(list(sessions))
    print(f"anualizacion derivada del bloque: {ann:.1f} sesiones/anio "
          f"(NO 252: la mascara quita festivos e incompletas)\n")

    policies = {
        "B1_buy_hold_1x": constant_policy(1.0),
        "NULL_A_short_1x": constant_policy(-1.0),
        "always_flat": constant_policy(0.0),
        "random_seed42": random_policy(42),
        "momentum_3bar": momentum_policy,
        "mean_reversion_12bar": mean_reversion_policy,
        "opening_range_6bar": opening_range_policy,
    }

    rows, per_baseline = [passive_buy_hold(sessions, ann)], {}
    for name, pol in policies.items():
        res = run_policy(sessions, spreads, pol)
        per_baseline[name] = res
        rows.append(summarize(name, res, ann))

    # Regime rule uses the same frozen posterior embedded in each research SessionSpec. It is
    # evaluated through the same run_session engine, with no parameter fitting.
    regime_results = []
    for d, close in sessions.items():
        state = int(np.argmax(sp.loc[pd.Timestamp(d), [c for c in sp.columns if c.startswith("p_")]]))
        regime_results.append(run_session(close, regime_rules_policy(state), spreads[d], date=d))
    per_baseline["regime_two_rules"] = regime_results
    rows.append(summarize("regime_two_rules", regime_results, ann))

    # B1' — exposición constante igual a la media realizada de B1 (§3.2 de la constitución).
    ref = per_baseline["B1_buy_hold_1x"]
    asset_ret = np.asarray([float(np.sum(r.bar_returns)) for r in ref])
    positions = np.asarray([r.mean_abs_exposure for r in ref])
    b1p = paired_exposure_baseline(positions, asset_ret, ann)
    rows.append({"baseline": "B1_prime_exposure_matched", "n_sessions": len(ref),
                 "n_traded": len(ref), "mean_abs_exposure": b1p["mean_exposure"],
                 "ann_return_pct": round(b1p.get("ann_return_pct", 0.0), 3),
                 "max_dd_pct": round(abs(b1p.get("max_dd", 0.0)) * 100, 3),
                 "calmar": b1p.get("calmar"),
                 "note": "sin costos por construccion (exposicion constante = turnover ~0); "
                         "eso hace el bar MAS duro, honesto en la direccion correcta"})

    # Stress de costos x1/x2/x3 sobre las MISMAS posiciones (§3.4).
    stress = {}
    for name in ("B1_buy_hold_1x", "NULL_A_short_1x"):
        res = per_baseline[name]
        pos = np.asarray([r.mean_abs_exposure * np.sign(r.weights[0]) for r in res])
        aret = np.asarray([float(np.sum(r.bar_returns)) for r in res])
        cst = np.asarray([r.total_cost for r in res])
        stress[name] = cost_stress(pos, aret, cst, None, ann)

    cols = ['baseline', 'n_sessions', 'n_traded', 'total_return_pct',
            'ann_return_pct', 'sharpe', 'sharpe_ci95', 'calmar', 'max_dd_pct',
            'mean_abs_exposure', 'total_cost_pct']
    table = pd.DataFrame(rows).reindex(columns=cols)
    with pd.option_context('display.max_rows', None, 'display.width', 200):
        print(table.to_string(index=False, na_rep='-'))
    for r in rows:
        if r.get('note'):
            print(f"    [{r['baseline']}] {r['note']}")
    print("\n=== stress de costos (constitucion §3.4: si muere al doble, REJECT) ===")
    for name, s in stress.items():
        print(f"  {name}: x1 {s['x1']['ann_return_pct']:+.2f}%  "
              f"x2 {s['x2']['ann_return_pct']:+.2f}%  x3 {s['x3']['ann_return_pct']:+.2f}%  "
              f"| sobrevive x2: {s['survives_2x']}  x3: {s['survives_3x']}")

    payload = {
        "contract": "CTR-RESEARCH-BASELINES-001",
        "block": a.block,
        "range": [blk["start"], blk["end"]],
        "mask_sha256": mask.sha256,
        "n_sessions_evaluated": len(sessions),
        "annualization_sessions_per_year": round(ann, 2),
        "regime": {"k": model.k, "labels": model.state_labels(),
                   "fit_range": list(model.fit_range)},
        "rows": rows,
        "cost_stress": stress,
    }
    out = (a.out.resolve() if a.out else (OUT_DIR / f"baselines_{a.block}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nartefacto: {out.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
