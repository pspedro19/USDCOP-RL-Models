"""Runner end-to-end del patrón B1/B2/S3 sobre datos SINTÉTICOS.

    python run_strategy.py            # demo completa (data sintética)
    python run_strategy.py --seed 7   # otra realización sintética

AVISO: El veredicto que imprime mide el CABLEADO del pipeline y la SEVERIDAD de los
gates, no alfa real. Sobre ruido sintético, lo científicamente correcto es que los
gates RECHACEN a las candidatas (H6). Que el DSR con N=989 tumbe un Sharpe de ~1.0
no es un bug: es exactamente el punto de Bailey-López de Prado. Para evidencia real,
enchufá SPY total-return + FRED (ver README) y volvé a correr.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

# Windows: la consola cp1252 revienta con flechas/acentos. Forzamos UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import datagen
from benchmarks import BENCHMARKS, build_benchmark, vol_target_weights
from engine import BacktestConfig, BacktestEngine
from kernels import cer_gain_bps, gate_g4, gate_g6, validate_report_benchmarks
from metrics import (
    compute_metrics,
    dsr_from_family,
    pbo_from_family,
    regime_robustness,
)
from policies import POLICIES
from regime import label_regimes

N_MAX_STUDY = 989          # SDD-000 §3: presupuesto pre-registrado (input del DSR)
COST_BASE = 2.0
COST_STRESS = 4.0          # x2, obligatorio en el reporte (SDD-006)


def _fmt(x: float, p: int = 3) -> str:
    return f"{x:>+8.{p}f}" if x else f"{0.0:>+8.{p}f}"


def _run(df, w, cost_bps):
    eng = BacktestEngine(BacktestConfig(cost_bps_roundtrip=cost_bps))
    res = eng.run(w, df)
    return res, compute_metrics(res.returns_net, res.turnover)


def _config_family(df) -> tuple[pd.DataFrame, np.ndarray]:
    """Familia de variantes de la tendencia/gated para alimentar PBO y Var(SR).

    Variar knobs y guardar cada stream de retornos es lo que el CSCV necesita para
    medir el PROCESO DE SELECCIÓN (¿la mejor in-sample sigue arriba out-of-sample?).
    """
    close = df["close"].astype(float)
    streams: dict[str, pd.Series] = {}
    sharpes: list[float] = []
    for target in (0.08, 0.10, 0.12, 0.15):
        for ma_win in (150, 200, 250):
            ma = close.rolling(ma_win, min_periods=ma_win).mean()
            trend_on = (close > ma).astype(float)
            base = vol_target_weights(close, target=target)
            w = (trend_on * base).clip(upper=1.5)
            res, m = _run(df, w, COST_BASE)
            key = f"t{int(target*100)}_ma{ma_win}"
            streams[key] = res.returns_net
            sharpes.append(m.sharpe)
    matrix = pd.DataFrame(streams).dropna()
    return matrix, np.array(sharpes, dtype=float)


def main(seed: int = datagen.__dict__.get("_DEFAULT_SEED", 20260709)) -> dict:
    print("=" * 78)
    print("SP500 · patrón B1/B2/S3 · " + datagen.SYNTHETIC_WARNING)
    print("=" * 78)

    df = datagen.generate(seed=seed)
    regimes = label_regimes(df)
    print(f"Días: {len(df)}  ({df.index[0].date()} -> {df.index[-1].date()})")
    print("Días por régimen (PIT):",
          {c: int(regimes[c].sum()) for c in regimes.columns})
    print()

    # ---- backtests: 4 benchmarks + 3 políticas, a costo base y a x2 ----------
    names = list(BENCHMARKS) + list(POLICIES)
    base_metrics: dict[str, object] = {}
    stress_metrics: dict[str, object] = {}
    net_returns: dict[str, pd.Series] = {}

    for name in names:
        w = build_benchmark(name, df) if name in BENCHMARKS else POLICIES[name](df)
        res_b, m_b = _run(df, w, COST_BASE)
        _, m_s = _run(df, w, COST_STRESS)
        base_metrics[name] = m_b
        stress_metrics[name] = m_s
        net_returns[name] = res_b.returns_net

    validate_report_benchmarks(BENCHMARKS.keys())   # falla si falta alguno

    hdr = f"{'estrategia':<22}{'Sharpe':>9}{'Calmar':>9}{'CAGR':>9}{'vol':>8}{'MDD':>9}{'turn/y':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name in names:
        m = base_metrics[name]
        tag = "  <- candidata" if name == "spx_regime_gated_v1" else ""
        print(f"{name:<22}{_fmt(m.sharpe)}{_fmt(m.calmar)}{_fmt(m.cagr)}"
              f"{_fmt(m.ann_vol,2)}{_fmt(m.max_drawdown)}{m.turnover_ann:>9.1f}{tag}")
    print()

    # ---- Gates sobre la candidata S3 ----------------------------------------
    cand = "spx_regime_gated_v1"
    b1, b2 = "spx_hodl_b1", "spx_trend_b2"
    voltgt = net_returns["VOL_TARGET_10"]

    matrix, family_sharpes = _config_family(df)
    dsr = dsr_from_family(base_metrics[cand], family_sharpes, N_MAX_STUDY)
    pbo = pbo_from_family(matrix)

    cer_g = cer_gain_bps(net_returns[cand].to_numpy(), voltgt.to_numpy())

    reg = regime_robustness(net_returns[cand], voltgt, regimes)
    wins = sum(1 for r in reg.values() if r["sharpe_strat"] > r["sharpe_bench"])

    # break-even necesita retorno bruto y turnover de la candidata
    eng = BacktestEngine(BacktestConfig(cost_bps_roundtrip=COST_BASE))
    res_cand = eng.run(POLICIES[cand](df), df)
    from costs import break_even_cost
    be = break_even_cost(res_cand.returns_gross, res_cand.turnover)

    g3 = cer_g > 0
    g4 = gate_g4(dsr, pbo)
    g5 = wins >= 3
    g6 = gate_g6(be, COST_BASE)

    print("GATES (candidata spx_regime_gated_v1):")
    print(f"  G3 utilidad económica   CER_gain vs vol-target = {cer_g:>+8.1f} pb   "
          f"-> {'PASA' if g3 else 'FALLA'}")
    print(f"  G4 significancia         DSR = {dsr:6.3f} (>0.95)   PBO = {pbo:5.3f} (<0.50)   "
          f"-> {'PASA' if g4 else 'FALLA'}  [N={N_MAX_STUDY} trials]")
    print(f"  G5 robustez de régimen   bate a vol-target en {wins}/4 regímenes   "
          f"-> {'PASA' if g5 else 'FALLA'}")
    for rn, r in reg.items():
        print(f"       {rn:<9} n={r['n_days']:>5}  "
              f"Sharpe {r['sharpe_strat']:>+6.2f} vs {r['sharpe_bench']:>+6.2f}")
    print(f"  G6 supervivencia costos  break_even = {be:5.2f} pb (>=6 pb)   "
          f"-> {'PASA' if g6 else 'FALLA'}")
    print()

    # ---- Comparación honesta S3 vs B2 vs B1 bajo costos x2 -------------------
    print("Comparación bajo costos x2 (Sharpe / Calmar):")
    for name in (b1, b2, cand):
        m = stress_metrics[name]
        print(f"  {name:<22} Sharpe {m.sharpe:>+6.2f}   Calmar {m.calmar:>+6.2f}")
    s3_beats_b2 = (stress_metrics[cand].sharpe > stress_metrics[b2].sharpe
                   and stress_metrics[cand].calmar > stress_metrics[b2].calmar)
    print()

    # ---- Veredicto -----------------------------------------------------------
    all_gates = g3 and g4 and g5 and g6
    print("=" * 78)
    if all_gates and s3_beats_b2:
        verdict = "PROMOTE spx_regime_gated_v1 (pasa los 4 gates y bate a B2)."
    elif s3_beats_b2 and not all_gates:
        verdict = ("S3 bate a B2 pero NO pasa todos los gates -> NO PROMOTE. "
                   "Sin DSR/PBO/costos no hay alfa.")
    else:
        verdict = ("S3 no bate a B2 -> la estrategia es spx_trend_b2 (H3). "
                   "Igual que en BTC: la gated no supera al trend follower.")
    if not all_gates:
        verdict += "  Resultado científico esperable sobre ruido sintético: H6 (sin alfa)."
    print("VEREDICTO:", verdict)
    print("=" * 78)

    return {
        "dsr": dsr, "pbo": pbo, "cer_gain_bps": cer_g, "regime_wins": wins,
        "break_even_bps": be, "gates": {"G3": g3, "G4": g4, "G5": g5, "G6": g6},
        "s3_beats_b2": s3_beats_b2, "promote": all_gates and s3_beats_b2,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260709)
    args = ap.parse_args()
    main(seed=args.seed)
