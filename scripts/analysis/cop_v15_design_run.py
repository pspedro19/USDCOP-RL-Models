"""H-V15 design-run pareado EN EL MOTOR — v13 vs v15 (= v13 + dispersión EME), 2022-2024.

Consecuencia pre-firmada del screening H-RISK-FAM-02 (registry 2026-07-22): la celda
ganadora g1 (dispersión de analistas EME, pinball 0.261 vs null 0.314) se añade como
QUINTO regresor de la QR del techo probabilístico. UNA variante, composición 50/50 y
fórmula sellada intactas (compute_v13_leverage_ceilings(eme_dispersion=True)).
ABRE +1 trial (N 81→82). Criterio sellado: aprueba diseño <=> Calmar_v15 >= Calmar_v13
en 2022-2024. Juez real = forward desde su freeze. 2025/2026 NO se corren.
Bit-identity: el brazo v13 debe reproducir EXACTO la evidencia congelada
`.claude/evidence/cop_v13_engine/2026-07-22/v13_engine_design_run.json`.
Advertencia vigente (evento contaminación #3): la feature EME llega pre-contaminada por
el audit externo; el design-run ≤2024 mitiga — el juez de cualquier uso es el forward.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

from train_and_export_smart_simple import (  # noqa: E402
    load_config, load_data, run_walkforward_backtest, compute_v13_leverage_ceilings,
)
from src.forecasting.enhance_v2 import enhance_features_v2  # noqa: E402

DESIGN_YEARS = (2022, 2023, 2024)
DESIGN_CUTOFF = pd.Timestamp("2024-12-31")
V12_CONFIG = REPO / "config" / "execution" / "smart_simple_v12_lev_cap.yaml"
OUT_DIR = REPO / ".claude" / "evidence" / "cop_v15_engine" / "2026-07-22"
V13_FROZEN = REPO / ".claude" / "evidence" / "cop_v13_engine" / "2026-07-22" / "v13_engine_design_run.json"


def year_stats(result):
    m = result["metrics"] or {}
    trades = result["trades"]
    levs = [t["leverage"] for t in trades]
    return {"ret_pct": m.get("total_return_pct"), "max_dd_pct": m.get("max_dd_pct"),
            "n_trades": m.get("n_trades", 0),
            "n_hard_stops": (m.get("exit_reasons") or {}).get("hard_stop", 0),
            "lev_mean": round(float(np.mean(levs)), 3) if levs else None,
            "lev_max": round(float(np.max(levs)), 3) if levs else None}


def composite(yearly_results):
    rets = []
    for res in yearly_results:
        rets.extend(t["pnl_pct"] / 100.0 for t in res["trades"])
    eq = [10000.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    max_dd = abs(float(np.min((eq - peak) / peak))) * 100
    tot = (eq[-1] / eq[0] - 1) * 100
    cagr = ((1 + tot / 100) ** (1 / len(DESIGN_YEARS)) - 1) * 100
    return {"ret_total_pct": round(tot, 2), "cagr_pct": round(cagr, 2),
            "max_dd_pct": round(max_dd, 2),
            "calmar_cagr_over_dd": round(cagr / max_dd, 4) if max_dd > 0 else None,
            "n_trades": len(rets)}


def main():
    print("H-V15 design-run pareado (motor real) — v13 vs v15 (+dispersión EME), 2022-2024")
    cfg13 = load_config(config_path=str(V12_CONFIG), version_override="13.0.0",
                        strategy_id="smart_simple_v13_qrisk")
    assert cfg13["vt_max"] == 1.5
    cfg13["v13_ceiling_enabled"] = True
    cfg15 = copy.deepcopy(cfg13)
    cfg15["strategy_id"] = "smart_simple_v15_qrisk_eme"
    cfg15["version"] = "15.0.0"

    df, feats = load_data()
    df, feats = enhance_features_v2(df, feats)

    c13, m13 = compute_v13_leverage_ceilings(df, cutoff=DESIGN_CUTOFF)
    c15, m15 = compute_v13_leverage_ceilings(df, cutoff=DESIGN_CUTOFF, eme_dispersion=True)
    cfg13["_v13_ceiling_cache"], cfg13["_v13_ceiling_meta"] = c13, m13
    cfg15["_v13_ceiling_cache"], cfg15["_v13_ceiling_meta"] = c15, m15
    print(f"techos v13: mean={m13['ceiling_mean']:.3f} min={m13['ceiling_min']:.3f} "
          f"fallback={m13['n_qr_fallback']}")
    print(f"techos v15: mean={m15['ceiling_mean']:.3f} min={m15['ceiling_min']:.3f} "
          f"fallback={m15['n_qr_fallback']}")

    out = {"meta": {
        "hypothesis": "H-V15 (consecuencia pre-firmada H-RISK-FAM-02)",
        "variante_unica": "QR features = {f2,f3,f4,f5} + g1_eme_disp (PIT available_at<=asof); "
                          "resto de la formula sellada de v13 INTACTO",
        "criterio_sellado": "aprueba diseno <=> Calmar_v15 >= Calmar_v13 en 2022-2024",
        "trials": "+1 (N 81->82)", "ceiling_cutoff": str(DESIGN_CUTOFF.date()),
        "contaminacion": "feature EME pre-contaminada por audit externo (evento #3); "
                         "juez real = forward",
    }, "per_year": {}, "composite": {}, "fallbacks_per_year_v15": {}, "bit_identity_v13": {}}

    modes15 = m15["weeks"]
    for year in DESIGN_YEARS:
        yr = {k: v for k, v in modes15.items() if k.year == year}
        out["fallbacks_per_year_v15"][str(year)] = {
            "n_qr_real": sum(1 for m in yr.values() if m == "qr"),
            "n_persistence_only": sum(1 for m in yr.values() if m == "persistence_only"),
            "n_inactive": sum(1 for m in yr.values() if m.startswith("inactive")),
        }

    frozen13 = {}
    if V13_FROZEN.exists():
        frozen13 = json.loads(V13_FROZEN.read_text()).get("per_year", {})

    yearly = {"v13": [], "v15": []}
    for year in DESIGN_YEARS:
        print(f"\n=== {year} ===")
        r13 = run_walkforward_backtest(df, feats, cfg13, year)
        r15 = run_walkforward_backtest(df, feats, cfg15, year)
        yearly["v13"].append(r13)
        yearly["v15"].append(r15)
        out["per_year"][str(year)] = {"v13": year_stats(r13), "v15": year_stats(r15)}
        for arm in ("v13", "v15"):
            s = out["per_year"][str(year)][arm]
            print(f"  {arm}: ret={s['ret_pct']}% dd={s['max_dd_pct']}% hs={s['n_hard_stops']} "
                  f"lev_mean={s['lev_mean']}")
        fz = frozen13.get(str(year), {}).get("v13")
        if fz:
            got = out["per_year"][str(year)]["v13"]
            match = (round(got["ret_pct"], 2) == round(fz["ret_pct"], 2)
                     and got["n_trades"] == fz["n_trades"])
            out["bit_identity_v13"][str(year)] = {"match": bool(match)}
            print(f"  bit-identity v13 vs evidencia congelada: {'MATCH' if match else 'MISMATCH'}")

    out["composite"]["v13"] = composite(yearly["v13"])
    out["composite"]["v15"] = composite(yearly["v15"])
    ca13 = out["composite"]["v13"]["calmar_cagr_over_dd"]
    ca15 = out["composite"]["v15"]["calmar_cagr_over_dd"]
    out["verdict"] = ("APRUEBA_DISENO" if (ca15 is not None and ca13 is not None
                                           and ca15 >= ca13) else "NO_APRUEBA")
    print(f"\ncompuesto v13: {out['composite']['v13']}")
    print(f"compuesto v15: {out['composite']['v15']}")
    print(f"VEREDICTO: {out['verdict']} (Calmar v15 {ca15} vs v13 {ca13})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "v15_engine_design_run.json").write_text(json.dumps(out, indent=2, default=str))
    shutil.copy(__file__, OUT_DIR / "generator_cop_v15_design_run.py")
    print(f"artefacto -> {OUT_DIR}")


if __name__ == "__main__":
    main()
