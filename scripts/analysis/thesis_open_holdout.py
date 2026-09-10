#!/usr/bin/env python
"""Abre el hold-out UNA vez, con el gate de la Regla B en el codigo (F8b).

Contract: CTR-RESEARCH-PREREG-001 - Date: 2026-08-25

Este es el unico script del repositorio que evalua modelos sobre el bloque de hold-out.
Se niega a hacerlo si `06-PRE-REGISTRATION.md` no esta firmado (`status: IMPLEMENTED`),
y sale con codigo 2 sin haber leido un solo dato.

Ademas deja constancia: escribe `outputs/thesis/holdout_opening.json` con la fecha, el hash
de la mascara, los modelos evaluados y el `n_trials` vigente. Si el fichero ya existe, avisa
de que el hold-out YA fue abierto — una segunda apertura no es una repeticion inocente, es
convertir el juez final en un segundo bloque de seleccion.

Uso:
    python scripts/analysis/thesis_open_holdout.py --models refit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import yaml  # noqa: E402

from scripts.analysis.thesis_statistics import preregistration_is_signed  # noqa: E402
from scripts.analysis.thesis_train_ppo import (CONFIGS, SEEDS, evaluate,  # noqa: E402
                                               strip_regimes)
from src.research.dataset import PORTABLE, load_or_build, load_portable  # noqa: E402
from src.research.evaluation_mask import build_mask  # noqa: E402

PPO_DIR = Path(os.environ.get("THESIS_PPO_OUT", REPO / "outputs" / "thesis" / "ppo"))
OUT = REPO / "outputs" / "thesis"
LEDGER = OUT / "holdout_opening.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="refit", choices=["refit", "development"],
                    help="`refit` = entrenados en desarrollo+seleccion (lo que pide el pre-registro)")
    ap.add_argument("--config", choices=list(CONFIGS),
                    help="evalua solo una configuracion (la evaluacion completa son ~40 min)")
    ap.add_argument("--reeval", action="store_true",
                    help="reevalua tambien las corridas que ya tienen `holdout` en su JSON")
    args = ap.parse_args()

    signed, status = preregistration_is_signed()
    if not signed:
        print(f"BLOQUEADO por la Regla B: el pre-registro esta en `{status}`, no en "
              "`IMPLEMENTED`. No se ha leido ningun dato del hold-out.")
        return 2

    if LEDGER.is_file():
        prev = json.loads(LEDGER.read_text(encoding="utf-8"))
        print(f"AVISO: el hold-out YA fue abierto el {prev.get('opened_at')}. "
              "Una segunda apertura lo convierte en un segundo bloque de seleccion.\n"
              "Si es una re-ejecucion tecnica del mismo universo, adelante; si has cambiado "
              "algo del universo, el resultado ya NO es confirmatorio.")

    from stable_baselines3 import PPO

    data = load_portable() if PORTABLE.is_file() else load_or_build(verbose=False)
    specs = data.holdout
    mask = build_mask()
    part = yaml.safe_load((REPO / "config" / "research" / "partition.yaml").read_text(
        encoding="utf-8"))

    print(f"HOLD-OUT: {len(specs)} sesiones ({specs[0].date} -> {specs[-1].date})")
    print(f"mascara {mask.sha256[:16]} - n_trials heredado "
          f"{part['trials']['inherited_n']}\n")

    suffix = "_refit" if args.models == "refit" else ""
    evaluated, skipped = [], []
    # Reanudable a proposito: evaluar 10 modelos x 584 sesiones x 59 pasos son ~345.000
    # llamadas a `predict`, y el proceso ya se corto una vez a mitad. Reevaluar un modelo
    # determinista sobre los mismos datos da el MISMO numero, asi que saltarselo no cambia
    # nada — y no cuenta como una segunda apertura: la apertura es del BLOQUE, no del script.
    for cfg in CONFIGS:
        if args.config and cfg != args.config:
            continue
        use = specs if cfg == "ppo_regime" else strip_regimes(specs)
        for seed in SEEDS:
            tag = f"{cfg}{suffix}_seed{seed}"
            zf = PPO_DIR / f"{tag}.zip"
            jf = PPO_DIR / f"{cfg}_seed{seed}.json"
            if not zf.is_file():
                print(f"  falta {zf.name}")
                continue
            if not args.reeval and jf.is_file():
                prev = json.loads(jf.read_text(encoding="utf-8"))
                if prev.get("holdout") and prev.get("holdout_models") == args.models:
                    skipped.append(tag)
                    evaluated.append(tag)
                    print(f"  {tag:<32} ya evaluado, se reutiliza "
                          f"(ret {prev['holdout']['total_return']:+.2%})")
                    continue
            model = PPO.load(str(zf), device="cpu")
            res = evaluate(model, use)
            blob = json.loads(jf.read_text(encoding="utf-8")) if jf.is_file() else {
                "config": cfg, "seed": seed}
            blob["holdout"] = res
            blob["holdout_models"] = args.models
            jf.write_text(json.dumps(blob, indent=2), encoding="utf-8")
            evaluated.append(tag)
            print(f"  {tag:<32} ret {res['total_return']:+7.2%}  "
                  f"Sharpe {res['sharpe']:+6.2f}  ops {res['n_ops']:>5}  "
                  f"|exp| {res['mean_abs_exposure']:.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    import datetime as dt

    # La constancia ACUMULA. Con `--config` la apertura se hace en varias invocaciones, y
    # sobreescribir dejaba el ledger diciendo que solo se evaluo la ultima configuracion —
    # un registro de apertura incompleto es peor que ninguno, porque parece completo.
    prev = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.is_file() else {}
    evaluated = sorted(set(prev.get("evaluated", [])) | set(evaluated))
    skipped = sorted(set(prev.get("reused_from_previous_run", [])) | set(skipped))
    opened_at = prev.get("opened_at") or dt.datetime.now(
        dt.timezone.utc).isoformat(timespec="seconds")

    LEDGER.write_text(json.dumps({
        "contract": "CTR-RESEARCH-PREREG-001",
        "opened_at": opened_at,
        "last_write": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "block": "holdout",
        "range": [str(specs[0].date), str(specs[-1].date)],
        "n_sessions": len(specs),
        "mask_sha256": mask.sha256,
        "models": args.models,
        "evaluated": evaluated,
        "reused_from_previous_run": skipped,
        "inherited_trials": part["trials"]["inherited_n"],
        "holdout_partially_looked_at": part["trials"]["holdout_partially_looked_at"],
    }, indent=2), encoding="utf-8")
    print(f"\nApertura registrada -> {LEDGER.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
