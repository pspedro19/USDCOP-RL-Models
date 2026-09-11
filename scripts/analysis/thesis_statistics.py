#!/usr/bin/env python
"""Contraste estadístico de la tesis (R5): pareado, PBO, DSR y stress de costos.

Contract: CTR-RESEARCH-INFERENCE-001 · Date: 2026-08-25

## La Regla B, en código

`--block holdout` **se niega a correr** mientras `06-PRE-REGISTRATION.md` no esté firmado
(`status: IMPLEMENTED`). No es una advertencia: sale con código 2 y no lee un solo dato del
hold-out.

El plan lo llama Regla B, y la constitución §1 explica por qué: mirar el hold-out para decidir
algo lo convierte en un segundo bloque de selección, y a partir de ahí ya no queda ningún juez
limpio. Una regla que solo vive en un documento se salta sin querer un martes por la tarde.

## Alineación

Baselines y agentes se evalúan sobre **los mismos `SessionSpec`, en el mismo orden**. El
contraste pareado exige eso: si las series fueran de sesiones distintas, su correlación no
significaría nada y el bootstrap con índices comunes sería inválido. Por eso los baselines se
recomputan aquí y no se leen del JSON de la Fase E — que guarda resúmenes, no series.

## Lo que se reporta y lo que no

- **Bootstrap estacionario pareado** de diferencias de Sharpe (10.000 réplicas, bloques 5-20).
- **PBO** por CSCV sobre la matriz de las 10 corridas.
- **DSR** deflactado con `n_trials` del ACTIVO (111 heredados + los de esta tesis).
- **Stress de costos** ×1/×2/×3 (constitución §3.4: morir al doble ⇒ REJECT).
- **B1′**, exposición constante igual a la exposición media realizada.
- **White RC / SPA: NO se computan.** Universo de dos candidatos; ver `WHITE_SPA_OMISSION`.

Uso:
    python scripts/analysis/thesis_statistics.py --block selection
    python scripts/analysis/thesis_statistics.py --block holdout     # exige firma
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# La consola de Windows usa cp1252 y estos scripts imprimen `Δ`, `·`, `→`. Sin esto un
# UnicodeEncodeError aborta la corrida DESPUES de haber calculado todo, que es la peor
# forma de fallar: el trabajo esta hecho y no se escribe el JSON.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from services.common.metrics import (cost_stress, paired_exposure_baseline,  # noqa: E402
                                     pbo_cscv)
from src.research.dataset import PORTABLE, load_or_build, load_portable  # noqa: E402
from src.research.inference import (ANN_SESSIONS, WHITE_SPA_OMISSION,  # noqa: E402
                                    bootstrap_sharpe_ci, dsr_with_inherited_trials,
                                    paired_sharpe_test)
from src.research.session_env import daily_series, run_session  # noqa: E402

PPO_DIR = Path(os.environ.get("THESIS_PPO_OUT", REPO / "outputs" / "thesis" / "ppo"))
OUT_DIR = REPO / "outputs" / "thesis"
PREREG = REPO / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION.md"
PARTITION = REPO / "config" / "research" / "partition.yaml"

SEEDS = (42, 123, 456, 789, 1337)
CONFIGS = ("ppo_regime", "ppo_backbone")


# ---------------------------------------------------------------------------
# Regla B
# ---------------------------------------------------------------------------

def preregistration_is_signed() -> tuple[bool, str]:
    if not PREREG.is_file():
        return False, f"no existe {PREREG.relative_to(REPO)}"
    head = PREREG.read_text(encoding="utf-8")[:2000]
    for line in head.splitlines():
        if line.strip().startswith("status:"):
            st = line.split(":", 1)[1].strip().strip("\"'")
            return st.upper() == "IMPLEMENTED", st
    return False, "sin campo `status` en el front-matter"


# ---------------------------------------------------------------------------
# Series alineadas
# ---------------------------------------------------------------------------

def baseline_series(specs, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Exposición constante sobre los mismos specs. Devuelve (retornos diarios, exposición)."""
    results = [run_session(s.close, np.full(59, level), s.spread_pips, date=s.date)
               for s in specs]
    return daily_series(results), np.full(len(specs), abs(level))


def passive_series(specs) -> np.ndarray:
    """B1 pasivo: comprar el primer día y no tocar nada. Sin costo por sesión.

    Es el B1 de la constitución, distinto del "1× intradía" que paga 584 round-trips porque
    §9.1 fuerza plano al cierre. Los dos se reportan: presentar solo el segundo pondría un
    listón artificialmente bajo.
    """
    closes = np.array([s.close[-1] for s in specs], dtype=float)
    r = np.diff(closes) / closes[:-1]
    return np.concatenate([[0.0], r])


def ppo_series(config: str, block: str) -> dict[int, dict]:
    """Series diarias de las 5 semillas de una configuración."""
    out = {}
    for seed in SEEDS:
        f = PPO_DIR / f"{config}_seed{seed}.json"
        if not f.is_file():
            continue
        blob = json.loads(f.read_text(encoding="utf-8"))
        if block not in blob:
            continue
        b = blob[block]
        out[seed] = {"returns": np.asarray(b["daily_returns"], dtype=float),
                     "gross_returns": (np.asarray(b["daily_gross_returns"], dtype=float)
                                       if "daily_gross_returns" in b else None),
                     "daily_costs": (np.asarray(b["daily_costs"], dtype=float)
                                     if "daily_costs" in b else None),
                     "dates": b["dates"], "n_ops": b["n_ops"],
                     "mean_abs_exposure": b["mean_abs_exposure"],
                     "total_cost": b["total_cost"]}
    return out


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------

def describe(name: str, r: np.ndarray, extra: dict | None = None) -> dict:
    ci = bootstrap_sharpe_ci(r)
    equity = float(np.prod(1.0 + r))
    curve = np.cumprod(1.0 + r)
    dd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
    row = {
        "name": name, "n": len(r),
        "total_return_pct": round(100.0 * (equity - 1.0), 2),
        "sharpe": round(ci["sharpe"], 3),
        "sharpe_ci95": [round(ci["ci_low"], 3), round(ci["ci_high"], 3)],
        "max_dd_pct": round(100.0 * dd, 2),
        "mean_daily": float(np.mean(r)), "std_daily": float(np.std(r, ddof=1)),
    }
    # Constitucion §6: con N<20 no se reportan Sharpe ni p-value.
    if len(r) < 20:
        row["sharpe"] = None
        row["sharpe_ci95"] = None
        row["note"] = "N<20: se omiten Sharpe y p-value (constitucion §6)"
    if extra:
        row.update(extra)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="selection", choices=["development", "selection",
                                                             "holdout"])
    ap.add_argument("--force-holdout", action="store_true",
                    help="ignora la Regla B (solo para depurar; queda registrado)")
    args = ap.parse_args()

    if args.block == "holdout":
        signed, status = preregistration_is_signed()
        if not signed and not args.force_holdout:
            print(f"BLOQUEADO por la Regla B: el pre-registro esta en `{status}`, no en "
                  f"`IMPLEMENTED`.\n  {PREREG.relative_to(REPO)}\n"
                  "  Firmalo antes de abrir el hold-out. No se ha leido ningun dato.")
            return 2
        if args.force_holdout and not signed:
            print("AVISO: --force-holdout con el pre-registro SIN firmar. Queda registrado "
                  "en el JSON de salida como apertura no valida.")

    data = load_portable() if PORTABLE.is_file() else load_or_build(verbose=False)
    specs = data.block(args.block)
    print(f"bloque {args.block}: {len(specs)} sesiones "
          f"({specs[0].date} -> {specs[-1].date})\n")

    import yaml
    part = yaml.safe_load(PARTITION.read_text(encoding="utf-8"))
    inherited = int(part["trials"]["inherited_n"])
    # La constitucion §2 exige deflactar con el conteo ACTUALIZADO del activo, no con el
    # heredado: los 2 AT que cobro esta tesis tambien deflactan su propio claim. La autoridad
    # es el front-matter del HYPOTHESIS-REGISTRY (lo mismo que lee `check_trial_ledger.py`);
    # `partition.yaml::inherited_n` es solo el punto de partida y queda como fallback.
    n_trials = inherited
    reg = REPO / ".claude" / "specs" / "assets" / "usdcop" / "HYPOTHESIS-REGISTRY.md"
    if reg.is_file():
        import re
        m = re.search(r"^n_trials_total:\s*(\d+)", reg.read_text(encoding="utf-8"),
                      re.MULTILINE)
        if m:
            n_trials = max(inherited, int(m.group(1)))
    print(f"n_trials para el DSR: {n_trials} (heredados {inherited} + los de esta tesis)")

    # --- baselines, sobre los MISMOS specs -------------------------------
    rows, series = [], {}
    b1_sess, exp_b1 = baseline_series(specs, 1.0)
    null_a, _ = baseline_series(specs, -1.0)
    flat, _ = baseline_series(specs, 0.0)
    b1_pass = passive_series(specs)

    for name, r in (("B1_pasivo", b1_pass), ("B1_sesion_1x", b1_sess),
                    ("NULL_A_corto_1x", null_a), ("always_flat", flat)):
        series[name] = r
        rows.append(describe(name, r))

    # --- PPO --------------------------------------------------------------
    agg: dict[str, np.ndarray] = {}
    for config in CONFIGS:
        runs = ppo_series(config, args.block)
        if not runs:
            print(f"AVISO: sin corridas de {config} para el bloque {args.block}")
            continue
        for seed, d in sorted(runs.items()):
            key = f"{config}_seed{seed}"
            series[key] = d["returns"]
            rows.append(describe(key, d["returns"],
                                 {"n_ops": d["n_ops"],
                                  "mean_abs_exposure": round(d["mean_abs_exposure"], 3),
                                  "total_cost_pct": round(100 * d["total_cost"], 2)}))
        # La media entre semillas es la unidad de comparacion: una sola semilla es ruido
        # (`experiment-protocol.md` regla 2 exige las cinco justamente por eso).
        stack = np.stack([runs[s]["returns"] for s in sorted(runs)])
        agg[config] = stack.mean(axis=0)
        series[f"{config}_mean5"] = agg[config]
        rows.append(describe(f"{config}_mean5", agg[config],
                             {"n_seeds": len(runs),
                              "seeds_positive": int(sum(
                                  np.prod(1 + runs[s]["returns"]) > 1 for s in runs))}))

    print(f"{'estrategia':<26} {'n':>4} {'ret%':>8} {'Sharpe':>7} {'IC 95%':>18} {'DD%':>8}")
    for r in rows:
        sh = f"{r['sharpe']:+.2f}" if r["sharpe"] is not None else "  n/a"
        ci = (f"[{r['sharpe_ci95'][0]:+.2f},{r['sharpe_ci95'][1]:+.2f}]"
              if r["sharpe_ci95"] else "n/a")
        print(f"{r['name']:<26} {r['n']:>4} {r['total_return_pct']:>8.2f} {sh:>7} "
              f"{ci:>18} {r['max_dd_pct']:>8.2f}")

    # --- contrastes pareados ---------------------------------------------
    tests = []
    if len(agg) == 2:
        # H2: la ablacion controlada. Unica diferencia entre brazos: los 4 posteriores.
        t = paired_sharpe_test(agg["ppo_regime"], agg["ppo_backbone"],
                               "ppo_regime", "ppo_backbone")
        tests.append(("H2_regimen_vs_backbone", t))
    for config, r in agg.items():
        tests.append((f"{config}_vs_always_flat",
                      paired_sharpe_test(r, flat, config, "always_flat")))
        tests.append((f"{config}_vs_B1_pasivo",
                      paired_sharpe_test(r, b1_pass, config, "B1_pasivo")))

    print("\nContrastes pareados (bootstrap estacionario, 10.000 réplicas, bloques 5-20):")
    for name, t in tests:
        print(f"  {name}\n    {t.verdict()}")

    # --- PBO --------------------------------------------------------------
    pbo = None
    ppo_cols = [k for k in series if "_seed" in k]
    if len(ppo_cols) >= 2:
        matrix = np.stack([series[k] for k in ppo_cols], axis=1)
        pbo = pbo_cscv(matrix)
        print(f"\nPBO (CSCV, {len(ppo_cols)} configuraciones): {pbo['pbo']:.3f} "
              f"sobre {pbo['n_combinations']} particiones")
        if pbo["pbo"] > 0.5:
            print("  PBO > 0.5: el ganador in-sample tiende a perder fuera. REJECT.")
        elif pbo["pbo"] > 0.20:
            print("  PBO > 0.20: por encima del umbral pre-registrado.")

    # --- DSR y stress de costos ------------------------------------------
    dsr, stress = {}, {}
    for config, r in agg.items():
        dsr[config] = dsr_with_inherited_trials(r, n_trials=n_trials)
        print(f"\nDSR {config} (n_trials={n_trials}): "
              f"{dsr[config]['headline_dsr']:.4f}  "
              f"({'pasa' if dsr[config]['passes'] else 'NO pasa'} el bar 0.95)")

        runs = ppo_series(config, args.block)
        if not all(runs[s]["gross_returns"] is not None and runs[s]["daily_costs"] is not None
                   for s in runs):
            stress[config] = {"status": "unavailable",
                              "reason": "artefacto legacy sin costos realizados por sesión"}
            print("  stress de costos: NO DISPONIBLE (faltan costos diarios)")
            continue
        # Re-price the realized gross P&L and realized costs, not an exposure×asset proxy.
        gross = np.mean([runs[s]["gross_returns"] for s in runs], axis=0)
        realized_cost = np.mean([runs[s]["daily_costs"] for s in runs], axis=0)
        stress[config] = cost_stress(np.ones(len(gross)), gross, realized_cost,
                                      None, ANN_SESSIONS)
        print(f"  stress de costos: x2 {'sobrevive' if stress[config]['survives_2x'] else 'MUERE'}"
              f" · x3 {'sobrevive' if stress[config]['survives_3x'] else 'MUERE'}")

    # --- B1' --------------------------------------------------------------
    b1p = {}
    for config, r in agg.items():
        runs = ppo_series(config, args.block)
        mean_exp = float(np.mean([runs[s]["mean_abs_exposure"] for s in runs]))
        asset = passive_series(specs)
        b1p[config] = paired_exposure_baseline(np.full(len(asset), mean_exp), asset,
                                               ANN_SESSIONS)
        print(f"  B1' {config}: exposicion {b1p[config]['mean_exposure']:.2f}x -> "
              f"{b1p[config]['ann_return_pct']:+.2f}%/ano")

    print(f"\n{WHITE_SPA_OMISSION}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"statistics_{args.block}.json"
    out.write_text(json.dumps({
        "contract": "CTR-RESEARCH-INFERENCE-001",
        "block": args.block,
        "n_sessions": len(specs),
        "range": [str(specs[0].date), str(specs[-1].date)],
        "annualization_sessions_per_year": ANN_SESSIONS,
        "inherited_trials": inherited,
        "n_trials_for_dsr": n_trials,
        "rows": rows,
        "paired_tests": {n: t.to_dict() for n, t in tests},
        "pbo": pbo, "dsr": dsr, "cost_stress": stress, "b1_prime": b1p,
        "white_spa": WHITE_SPA_OMISSION,
        "preregistration_signed": preregistration_is_signed()[0],
        "forced_holdout": bool(args.block == "holdout" and args.force_holdout
                               and not preregistration_is_signed()[0]),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
