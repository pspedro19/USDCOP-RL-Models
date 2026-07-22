"""H-V13-QRISK-01 — design-run pareado v12 vs v13 DENTRO del motor (+1 trial, contabilizado
por el operador en HYPOTHESIS-REGISTRY.md — este script NO edita el registry).

Contract: CTR-QUANT-CONSTITUTION-001 · Pre-registro SELLADO: HYPOTHESIS-REGISTRY.md
"H-V13-QRISK-01 (PRE-REGISTRO 2026-07-21)". El intento anterior con replay externo fue
INVALID_INSTRUMENT (no reproducia al motor); este runner ejecuta el MOTOR REAL
(scripts/pipeline/train_and_export_smart_simple.py::_run_v2_ridge_gate_loop) por la via
oficial run_walkforward_backtest, con el flag de candidata v13_ceiling_enabled.

Protocolo (del pre-registro, sin desviaciones — UNA sola pasada):
- v12 = config congelado smart_simple_v12_lev_cap.yaml (vt_max=1.5), flag OFF.
- v13 = MISMO config, flag v13_ceiling_enabled=True (formula sellada en
  compute_v13_leverage_ceilings; techo = 1.5*clip(mediana_diseno/q90_hat, 0.5, 1.0)).
- Mismas senales (Ridge+BR no dependen del leverage); solo cambia el techo semanal.
- Anios de diseno: 2022, 2023, 2024. NO se corre 2025 ni 2026.
- Aprobacion por diseno = Calmar_v13 >= Calmar_v12 en el compuesto 2022-2024.
  Definicion de Calmar (documentada UNA vez, simetrica entre brazos): CAGR del compuesto
  ((1+ret_total)^(1/3)-1, 3 anios calendario) / |MaxDD| de la curva encadenada trade a
  trade a traves de los 3 anios (cada anio arranca el motor en 10k; se encadenan los
  retornos semanales, igual que el design-run de v12). Se reporta tambien ret/DD.
- Features: load_data + enhance_features_v2 (camino EXACTO de main()).

Evidencia: .claude/evidence/cop_v13_engine/2026-07-22/ (JSON + este generator).

Re-medicion 2026-07-21 (review Codex, 0 trials nuevos — misma celda, misma formula):
(1) fallbacks instrumentados por anio (meta['weeks'] del constructor); (2) invariante
week_cap = min(techo, vt_max) movido AL MOTOR; (3) cutoff=2024-12-31 al constructor
(sin QR ni techos 2025+ durante el design-run); (4) chequeo automatico de bit-identidad
del brazo v12 contra la evidencia congelada v12_paired_PURGED_openaware.json.
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

from scripts.pipeline.train_and_export_smart_simple import (  # noqa: E402
    load_config, load_data, run_walkforward_backtest, compute_v13_leverage_ceilings,
)
from src.forecasting.enhance_v2 import enhance_features_v2  # noqa: E402
from src.contracts.strategy_schema import safe_json_dump  # noqa: E402

DESIGN_YEARS = (2022, 2023, 2024)
DESIGN_CUTOFF = pd.Timestamp("2024-12-31")   # scope: no QR fit / ceiling beyond design end
V12_CONFIG = REPO / "config" / "execution" / "smart_simple_v12_lev_cap.yaml"
OUT_DIR = REPO / ".claude" / "evidence" / "cop_v13_engine" / "2026-07-22"
# Frozen authority for the v12 arm (motor honesto purga+open-aware, 2026-07-21): the flag-OFF
# path must reproduce these EXACTLY, otherwise the instrument is not the motor.
V12_FROZEN = REPO / ".claude" / "evidence" / "cop_v12_design" / "2026-07-21" / "v12_paired_PURGED_openaware.json"


def year_stats(result):
    m = result["metrics"] or {}
    trades = result["trades"]
    levs = [t["leverage"] for t in trades]
    return {
        "ret_pct": m.get("total_return_pct"),
        "max_dd_pct": m.get("max_dd_pct"),
        "n_trades": m.get("n_trades", 0),
        "n_hard_stops": (m.get("exit_reasons") or {}).get("hard_stop", 0),
        "lev_mean": round(float(np.mean(levs)), 3) if levs else None,
        "lev_max": round(float(np.max(levs)), 3) if levs else None,
        "exit_reasons": m.get("exit_reasons") or {},
        "win_rate_pct": m.get("win_rate_pct"),
    }


def composite(yearly_results):
    """Chain weekly trade returns across the design years (each engine year starts at 10k)."""
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
    calmar = cagr / max_dd if max_dd > 0 else None
    calmar_simple = tot / max_dd if max_dd > 0 else None
    return {
        "ret_total_pct": round(tot, 2),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar_cagr_over_dd": round(calmar, 4) if calmar is not None else None,
        "calmar_simple_ret_over_dd": round(calmar_simple, 4) if calmar_simple is not None else None,
        "n_trades": len(rets),
    }


def main():
    print("H-V13-QRISK-01 design-run pareado (motor real) — v12 vs v13, 2022-2024")
    cfg_v12 = load_config(config_path=str(V12_CONFIG), version_override="12.0.0",
                          strategy_id="smart_simple_v12_lev_cap")
    assert cfg_v12["vt_max"] == 1.5, f"v12 config debe tener vt_max=1.5, tiene {cfg_v12['vt_max']}"
    assert not cfg_v12["v13_ceiling_enabled"]
    cfg_v13 = copy.deepcopy(cfg_v12)
    cfg_v13["v13_ceiling_enabled"] = True
    cfg_v13["strategy_id"] = "smart_simple_v13_qrisk"
    cfg_v13["version"] = "13.0.0"

    df, feature_cols = load_data()
    df, feature_cols = enhance_features_v2(df, feature_cols)
    print(f"data: {len(df)} rows, {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}, "
          f"{len(feature_cols)} features (camino exacto de main)")

    # Precompute the v13 ceilings BOUNDED at the design cutoff (Codex issue 3) and inject
    # them as the engine cache: the constructor never fits a QR nor computes a ceiling for
    # 2025+ during this design-run.
    ceilings_bounded, ceil_meta = compute_v13_leverage_ceilings(df, cutoff=DESIGN_CUTOFF)
    cfg_v13["_v13_ceiling_cache"] = ceilings_bounded
    cfg_v13["_v13_ceiling_meta"] = ceil_meta
    print(f"ceilings (cutoff {DESIGN_CUTOFF.date()}): {ceil_meta['n_weeks']} semanas, "
          f"qr_fallback={ceil_meta['n_qr_fallback']}, inactive={ceil_meta['n_ceiling_inactive']}")

    out = {"meta": {
        "hypothesis": "H-V13-QRISK-01",
        "instrument": "motor real via run_walkforward_backtest (flag v13_ceiling_enabled)",
        "engine": "scripts/pipeline/train_and_export_smart_simple.py::_run_v2_ridge_gate_loop",
        "config": str(V12_CONFIG.relative_to(REPO)),
        "design_years": list(DESIGN_YEARS),
        "formula_sellada": "techo_t = 1.5*clip(mediana_diseno(q90_hat)/q90_hat_t, 0.5, 1.0); "
                           "q90_hat = 0.5*persistencia(q90 roll-252 rango-5d) + 0.5*QR(tau=.90, "
                           "{vol-of-vol, gap-cola, EMBI-acel, RESINT-z})",
        "calmar_def": "CAGR((1+ret_tot)^(1/3)-1) / |MaxDD curva encadenada 22-24|; "
                      "se reporta tambien ret_tot/DD (misma definicion para ambos brazos)",
        "criterio_sellado": "aprueba diseno <=> Calmar_v13 >= Calmar_v12 en 2022-2024",
        "ceiling_cutoff": str(DESIGN_CUTOFF.date()),
        "desviacion_declarada": "fail-safes de ingenieria NO presentes en la formula sellada, "
                                "una sola variante, no optimizados: (a) QR sin >=52 semanas de "
                                "train o feature NaN => q90_hat = persistencia sola; (b) mediana "
                                "con <26 valores previos o q90_hat invalido => techo inactivo "
                                "1.5. Instrumentados por anio en fallbacks_per_year.",
        "remedicion": "re-medicion de la MISMA celda tras review Codex (instrumentacion + "
                      "invariante min() en motor + cutoff); 0 trials nuevos",
    }, "per_year": {}, "composite": {}, "fallbacks_per_year": {}, "bit_identity_v12": {}}

    # Per-year fallback instrumentation (Codex issue 1): mode of every ceiling week
    modes = ceil_meta["weeks"]
    for year in DESIGN_YEARS:
        yr = {k: v for k, v in modes.items() if k.year == year}
        out["fallbacks_per_year"][str(year)] = {
            "n_weeks_ceiling": len(yr),
            "n_qr_real": sum(1 for m in yr.values() if m == "qr"),
            "n_persistence_only": sum(1 for m in yr.values() if m == "persistence_only"),
            "n_inactive_warmup": sum(1 for m in yr.values() if m == "inactive_warmup"),
            "n_inactive_invalid": sum(1 for m in yr.values() if m == "inactive_invalid"),
        }
    out["fallbacks_pre_design_2020_2021"] = {
        "n_weeks_ceiling": sum(1 for k in modes if k.year < 2022),
        "n_qr_real": sum(1 for k, m in modes.items() if k.year < 2022 and m == "qr"),
        "n_persistence_only": sum(1 for k, m in modes.items()
                                  if k.year < 2022 and m == "persistence_only"),
        "n_inactive_warmup": sum(1 for k, m in modes.items()
                                 if k.year < 2022 and m == "inactive_warmup"),
        "n_inactive_invalid": sum(1 for k, m in modes.items()
                                  if k.year < 2022 and m == "inactive_invalid"),
    }
    print("fallbacks por anio (design):", json.dumps(out["fallbacks_per_year"]))

    frozen_v12 = json.loads(V12_FROZEN.read_text())["v12_cap1.5"] if V12_FROZEN.exists() else {}

    yearly = {"v12": [], "v13": []}
    for year in DESIGN_YEARS:
        print(f"\n=== {year} ===")
        print("  v12 (vt_max=1.5, flag OFF)...")
        r12 = run_walkforward_backtest(df, feature_cols, cfg_v12, year)
        print("  v13 (techo probabilistico)...")
        r13 = run_walkforward_backtest(df, feature_cols, cfg_v13, year)
        yearly["v12"].append(r12)
        yearly["v13"].append(r13)
        out["per_year"][str(year)] = {"v12": year_stats(r12), "v13": year_stats(r13)}
        for arm in ("v12", "v13"):
            s = out["per_year"][str(year)][arm]
            print(f"    {arm}: ret={s['ret_pct']}% dd={s['max_dd_pct']}% n={s['n_trades']} "
                  f"hs={s['n_hard_stops']} lev_mean={s['lev_mean']} lev_max={s['lev_max']}")
        # Bit-identity of the flag-OFF path vs the frozen v12 authority evidence
        fz = frozen_v12.get(str(year))
        if fz:
            got = out["per_year"][str(year)]["v12"]
            match = (round(got["ret_pct"], 2) == round(fz["ret"], 2)
                     and round(got["max_dd_pct"], 2) == round(fz["dd"], 2)
                     and got["n_trades"] == fz["n"])
            out["bit_identity_v12"][str(year)] = {
                "frozen": {"ret": fz["ret"], "dd": fz["dd"], "n": fz["n"]},
                "got": {"ret": got["ret_pct"], "dd": got["max_dd_pct"], "n": got["n_trades"]},
                "match": bool(match),
            }
            print(f"    bit-identity v12 vs evidencia congelada: {'MATCH' if match else 'MISMATCH'}")

    comp12 = composite(yearly["v12"])
    comp13 = composite(yearly["v13"])
    out["composite"] = {"v12": comp12, "v13": comp13}
    # v13 ceiling diagnostics (cached on cfg by the engine)
    ceilings = cfg_v13.get("_v13_ceiling_cache") or {}
    design_c = {str(k.date()): round(float(v), 4) for k, v in sorted(ceilings.items())
                if k.year in DESIGN_YEARS}
    out["v13_ceilings_design_years"] = design_c
    if design_c:
        vals = list(design_c.values())
        out["meta"]["ceiling_stats_design"] = {
            "mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "n_binding_lt_1.5": int(sum(1 for v in vals if v < 1.4999)),
        }

    c12 = comp12["calmar_cagr_over_dd"]
    c13 = comp13["calmar_cagr_over_dd"]
    approve = (c13 is not None and c12 is not None and c13 >= c12)
    out["verdict"] = {
        "calmar_v12": c12, "calmar_v13": c13,
        "aprueba_diseno": bool(approve),
        "criterio": "Calmar_v13 >= Calmar_v12 (compuesto 2022-2024)",
        "nota": "re-medicion de la misma celda (0 trials nuevos; la mirada ya esta pagada); "
                "juez real = forward",
    }
    print(f"\nCOMPUESTO 22-24: v12 ret={comp12['ret_total_pct']}% dd={comp12['max_dd_pct']}% "
          f"calmar={c12} | v13 ret={comp13['ret_total_pct']}% dd={comp13['max_dd_pct']}% calmar={c13}")
    print(f"VEREDICTO diseno: {'APRUEBA' if approve else 'NO aprueba'} (Calmar_v13 "
          f"{'>=' if approve else '<'} Calmar_v12)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "v13_engine_design_run.json", "w") as f:
        safe_json_dump(out, f)
    shutil.copy(__file__, OUT_DIR / "generator_cop_v13_design_run.py")
    print(f"evidencia -> {OUT_DIR}")
    return out


if __name__ == "__main__":
    main()
