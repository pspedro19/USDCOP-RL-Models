#!/usr/bin/env python
"""Termina la compuerta de sanidad S1 con la receta `flat_init`, en un solo comando.

Estado al 2026-09-11: siete recetas de hiperparametros fallan S1 sobre ruido puro, y la
octava -- `flat_init`, que sesga el arranque de la politica hacia no operar -- **pasa en la
semilla 123**, justo la que peor iba (0,021 frente a 0,966 del baseline). Faltan 42, 456, 789
y 1337 para el veredicto, que exige 4/5.

Este script las corre. Existe porque el entorno de esta maquina corta los trabajos largos:
    * entrena **por tramos** (60k y luego 100k acumulados) para que cada llamada quepa,
    * **salta** las semillas que ya tengan resultado, asi que un corte no pierde nada,
    * al terminar **agrega** el veredicto 4/5 en el mismo JSON de evidencia.

Uso:
    python scripts/analysis/finish_sanity_gate.py
    python scripts/analysis/finish_sanity_gate.py --seeds 42 456     # solo algunas

Coste medido: ~12 min por semilla (20.000 pasos = 147 s). Las cuatro, ~50 min.
Cero trials de mercado: la fixture es sintetica y no toca ningun dato de USD/COP.

AVISO: si una corrida se interrumpe, el proceso hijo puede sobrevivir al shell. Antes de
relanzar, barrer huerfanos:
    powershell -Command "Get-CimInstance Win32_Process -Filter \\"Name='python.exe'\\" |
      Where-Object {$_.CommandLine -like '*thesis_ppo_sanity*'} | Stop-Process -Force"
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "analysis" / "thesis_ppo_sanity.py"
CKPT = ROOT / "outputs" / "thesis-repair" / "ckpt"
OUT = ROOT / "outputs" / "thesis-repair" / "sanity"
EVIDENCE = ROOT / "outputs" / "thesis-repair" / "sanity_S1_protocol.json"
PROBE = "flat_init"
FLAT_THRESHOLD = 0.1
SEEDS_REQUIRED_FLAT = 4


def _run(seed: int, timesteps: int, output: Path, resume: Path | None) -> int:
    cmd = [sys.executable, str(RUNNER), "--fixture", "S1", "--seed", str(seed),
           "--probe", PROBE, "--timesteps", str(timesteps),
           "--checkpoint-dir", str(CKPT), "--output", str(output)]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[42, 456, 789, 1337])
    parser.add_argument("--timesteps", type=int, default=100_000)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        final = OUT / f"S1fi_seed{seed}.json"
        if final.is_file():
            print(f"  semilla {seed}: ya hecha, se salta")
            continue
        checkpoint = CKPT / f"S1_{PROBE}_seed{seed}_final.zip"
        if not checkpoint.is_file():
            print(f"  semilla {seed}: tramo 1 (60k)")
            _run(seed, 60_000, OUT / f"_partial_{seed}.json", None)
        print(f"  semilla {seed}: tramo 2 ({args.timesteps} acumulados)")
        _run(seed, args.timesteps, final, checkpoint if checkpoint.is_file() else None)
        (OUT / f"_partial_{seed}.json").unlink(missing_ok=True)

    rows = []
    for path in sorted(OUT.glob("S1fi_seed*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8"))["rows"][0])
    rows.sort(key=lambda r: r["seed"])
    flat = [r for r in rows if r["mean_abs_exposure"] < FLAT_THRESHOLD]

    print(f"\n  === {PROBE} sobre S1 ===")
    for row in rows:
        mark = "FLAT" if row["mean_abs_exposure"] < FLAT_THRESHOLD else "OPERA"
        print(f"    semilla {row['seed']:>5}: exposicion {row['mean_abs_exposure']:.3f}  {mark}")
    passed = len(flat) >= SEEDS_REQUIRED_FLAT and len(rows) >= 5
    print(f"\n    planas {len(flat)}/{len(rows)} · la regla exige "
          f"{SEEDS_REQUIRED_FLAT}/5 · COMPUERTA: {'ABIERTA' if passed else 'CERRADA'}")

    if EVIDENCE.is_file():
        evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
        evidence.setdefault("probes", {})[PROBE] = {
            "seeds_run": len(rows), "seeds_flat": len(flat),
            "exposures": {str(r["seed"]): round(r["mean_abs_exposure"], 3) for r in rows},
            "passed": passed,
            "note": "candidata estructural: sesga el arranque de la politica hacia no operar",
        }
        if passed:
            evidence["selected_probe"] = PROBE
            evidence["verdict"] = (f"{PROBE} pasa S1 con {len(flat)}/{len(rows)} semillas planas; "
                                   "quedan S2-S4 antes de congelar la receta para v2")
        EVIDENCE.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"    evidencia actualizada en {EVIDENCE.relative_to(ROOT).as_posix()}")

    if passed:
        print("\n    S1 pasa. Antes de congelar la receta para v2 faltan S2, S3 y S4:")
        print("      python scripts/analysis/thesis_ppo_sanity.py --fixture S2 "
              f"--probe {PROBE} --timesteps 100000")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
