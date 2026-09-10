#!/usr/bin/env python
"""Los cuatro análisis que convierten el rechazo en una conclusión con mecanismo.

Contract: CTR-RESEARCH-DECOMP-001 · Date: 2026-08-25

Consume `outputs/thesis/decomposition_<bloque>.json` (producido por `thesis_decompose.py`,
con cuadre verificado a 1e-17 contra la evaluación original) y responde:

1. **¿Hay señal?** — serie bruta con su IC bootstrap. El bruto es una **cota superior
   contrafactual que exige costo cero**, no un retorno alcanzable ni un claim de edge.
2. **¿Cuánto costo aguanta?** — el spread `s*` de break-even, en forma cerrada, contrastado
   contra el supuesto de la tesis (§8.4) y el del track de producción de este repo.
3. **¿Ata la frecuencia?** — re-scoring de las MISMAS decisiones a `k = 1, 5, 15, 30, 59`.
4. **La paradoja del always-flat** — `w = 0` estaba en el espacio de acción y el agente no lo
   encontró; el plan temía el riesgo inverso.

**0 trials**: es descomposición descriptiva de un resultado ya obtenido, no selección de una
variante nueva. El rechazo no cambia.

Uso:
    python scripts/analysis/thesis_explain.py --block holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.research.dataset import PORTABLE, load_or_build, load_portable  # noqa: E402
from src.research.decomposition import (break_even_spread, flat_paradox,  # noqa: E402
                                        frequency_curve,
                                        production_cost_per_side_pips)
from src.research.inference import bootstrap_sharpe_ci  # noqa: E402

OUT = REPO / "outputs" / "thesis"
CONFIGS = ("ppo_regime", "ppo_backbone")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="holdout",
                    choices=["development", "selection", "holdout"])
    args = ap.parse_args()

    src = OUT / f"decomposition_{args.block}.json"
    if not src.is_file():
        raise SystemExit(f"falta {src.relative_to(REPO)}: corre antes thesis_decompose.py "
                         f"--block {args.block}")
    blob = json.loads(src.read_text(encoding="utf-8"))
    runs = blob["runs"]
    print(f"bloque {args.block}: {blob['n_sessions']} sesiones, {len(runs)} corridas\n")

    data = load_portable() if PORTABLE.is_file() else load_or_build(verbose=False)
    specs_by_date = {str(s.date): s for s in data.block(args.block)}
    mean_close = float(np.mean([s.close.mean() for s in data.block(args.block)]))
    mean_spread = float(np.mean([s.spread_pips for s in data.block(args.block)]))

    report = {"contract": "CTR-RESEARCH-DECOMP-001", "block": args.block,
              "n_sessions": blob["n_sessions"], "trials_charged": 0,
              "mean_close": mean_close, "mean_spread_pips": mean_spread,
              "gross_is_upper_bound_note": (
                  "El bruto exige costo CERO, que no existe. Es una cota superior "
                  "contrafactual, no un retorno alcanzable ni un claim de edge."),
              "per_config": {}}

    # ---------------- 1. ¿Hay señal? -------------------------------------
    print("1. BRUTO (cota superior: exige costo cero, NO alcanzable)")
    print(f"{'configuracion':<16} {'bruto':>9} {'costo':>9} {'neto':>9} "
          f"{'Sharpe bruto':>13} {'IC 95%':>18}")
    for cfg in CONFIGS:
        rows = [r for r in runs.values() if r["config"] == cfg]
        if not rows:
            continue
        # Serie bruta media entre semillas, sesion a sesion.
        stack = np.stack([[s["gross_return"] for s in r["sessions"]] for r in rows])
        gross_series = stack.mean(axis=0)
        ci = bootstrap_sharpe_ci(gross_series)
        g = float(np.mean([r["sum_gross"] for r in rows]))
        c = float(np.mean([r["sum_cost"] for r in rows]))
        decisive = not (ci["ci_low"] <= 0.0 <= ci["ci_high"])
        print(f"{cfg:<16} {g:>+8.2%} {c:>8.2%} {g - c:>+8.2%} "
              f"{ci['sharpe']:>+13.2f} [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]"
              f"{'  DECIDIBLE' if decisive else '  indecidible'}")
        report["per_config"][cfg] = {
            "sum_gross": g, "sum_cost": c, "sum_net": g - c,
            "gross_sharpe": ci["sharpe"], "gross_sharpe_ci95": [ci["ci_low"], ci["ci_high"]],
            "gross_decisive": decisive,
            "seeds_gross_positive": int(sum(r["sum_gross"] > 0 for r in rows)),
            "n_seeds": len(rows),
        }

    # ---------------- 2. Break-even --------------------------------------
    prod = production_cost_per_side_pips(mean_close)
    thesis_side = mean_spread / 2.0 + 0.5
    print(f"\n2. BREAK-EVEN  (spread medio de la tesis {mean_spread:.2f} pips => "
          f"{thesis_side:.2f} pips/lado · produccion {prod:.2f} pips/lado)")
    print(f"{'configuracion':<16} {'s* (pips)':>11} {'alfa/op':>9} {'costo/op':>9} "
          f"{'viable @tesis':>14} {'viable @prod':>13}")
    for cfg in CONFIGS:
        rows = [r for r in runs.values() if r["config"] == cfg]
        if not rows:
            continue
        be = break_even_spread(
            sum_gross=float(np.mean([r["sum_gross"] for r in rows])),
            sum_abs_dw=float(np.mean([r["sum_abs_dw"] for r in rows])),
            sum_abs_dw_sigma=float(np.mean([r["sum_abs_dw_sigma"] for r in rows])),
            mean_close=mean_close, spread_assumed=mean_spread)
        # "Viable con el supuesto de produccion": el costo por lado de produccion
        # equivale a un spread de `2*(prod - 0.5)` en la formula de la tesis.
        prod_equiv_spread = 2.0 * (prod - 0.5)
        viable_prod = be.spread_star_pips is not None and \
            be.spread_star_pips >= prod_equiv_spread
        print(f"{cfg:<16} {be.spread_star_pips:>+11.2f} "
              f"{be.alpha_per_unit_dw_pips:>9.2f} {be.cost_per_unit_dw_pips:>9.2f} "
              f"{'SI' if be.viable_at_assumed else 'no':>14} "
              f"{'SI' if viable_prod else 'no':>13}")
        report["per_config"][cfg]["break_even"] = {
            **be.to_dict(),
            "production_cost_per_side_pips": prod,
            "production_equivalent_spread_pips": prod_equiv_spread,
            "viable_at_production_assumption": viable_prod,
        }

    # ---------------- 3. Frecuencia --------------------------------------
    print("\n3. FRECUENCIA  (mismas decisiones, muestreadas cada k barras)")
    print(f"{'config':<16} {'k':>4} {'dec/ses':>8} {'bruto':>9} {'costo':>9} {'neto':>9}")
    for cfg in CONFIGS:
        rows = [r for r in runs.values() if r["config"] == cfg]
        if not rows:
            continue
        # Media sobre las CINCO semillas, no una sola: una curva de frecuencia construida
        # con una semilla mezclaria el efecto de la frecuencia con el de esa semilla, y las
        # semillas dispersan mucho (bruto de +1,3% a +54,5% en hold-out).
        per_seed = [frequency_curve(r["sessions"], specs_by_date) for r in rows]
        curve = []
        for i in range(len(per_seed[0])):
            pt = {"k": per_seed[0][i]["k"],
                  "decisions_per_session": per_seed[0][i]["decisions_per_session"],
                  "n_seeds": len(per_seed)}
            for key in ("sum_gross", "sum_cost", "sum_net", "n_changes"):
                vals = [c[i][key] for c in per_seed]
                pt[key] = float(np.mean(vals))
                pt[f"{key}_std"] = float(np.std(vals))
            curve.append(pt)
        for pt in curve:
            print(f"{cfg if pt['k'] == 1 else '':<16} {pt['k']:>4} "
                  f"{pt['decisions_per_session']:>8} {pt['sum_gross']:>+8.2%} "
                  f"{pt['sum_cost']:>8.2%} {pt['sum_net']:>+8.2%}")
        report["per_config"][cfg]["frequency_curve"] = curve

    # ---------------- 4. Paradoja del always-flat ------------------------
    print("\n4. ALWAYS-FLAT  (`w=0` estaba disponible y el agente no lo encontro)")
    for cfg in CONFIGS:
        rows = [r for r in runs.values() if r["config"] == cfg]
        if not rows:
            continue
        fp = flat_paradox([s for r in rows for s in r["sessions"]])
        dist = " ".join(f"{k}:{v:.1%}" for k, v in fp["action_distribution"].items())
        print(f"  {cfg:<16} {dist}")
        print(f"  {'':<16} sesiones enteras en flat: {fp['fully_flat_sessions']} de "
              f"{fp['total_sessions']} · cambios/sesion {fp['mean_changes_per_session']:.1f}")
        report["per_config"][cfg]["flat_paradox"] = fp

    dest = OUT / f"explanation_{args.block}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n-> {dest.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
