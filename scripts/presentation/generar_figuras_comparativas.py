#!/usr/bin/env python
"""Figuras que comparan TODOS los brazos de la tesis en el mismo eje.

Los dos generadores que ya existian no sirven para esto: `generar_resultados_y_figuras.py` es
mono-PPO (`CONFIGS`/`SEEDS`/`PALETTE` fijos, y una comprobacion de coherencia que exige
exactamente 10 corridas) y `generate_llm_figures.py` dibuja un solo brazo. Un capitulo que
compara PPO, LLM, hibrido y baselines necesitaba un tercero.

## Las dos figuras que importan

1. **Curva de capital contra `always_flat`.** La linea plana en cero es el listón real de la
   tesis: si ninguna curva la supera, el resultado esta contado.
2. **Bruto contra coste por brazo.** Es la figura que explica el resultado entero -- el bruto
   existe y el coste se lo come -- y no habia ninguna que lo mostrara.

## Una trampa deliberada que se evita

Cada brazo se dibuja **solo sobre las sesiones que cubre**, y la leyenda dice cuantas son. Un
brazo liquidado sobre 200 sesiones y otro sobre 226 no comparten eje temporal; superponerlos
como si empezaran el mismo dia produce una figura que miente sin que ningun numero este mal.
Cuando las coberturas difieren, la figura lo dice en el titulo en vez de disimularlo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.plot_determinism import enable_deterministic_png  # noqa: E402

PALETTE = {
    "always_flat": "#2ca02c",
    "B1_pasivo": "#7f7f7f",
    "ppo_regime_mean5": "#1f77b4",
    "ppo_backbone_mean5": "#ff7f0e",
    "deepseek": "#d62728",
    "azure": "#9467bd",
    "hibrido_ds": "#8c564b",
    "hibrido_azure": "#e377c2",
}
FALLBACK = "#17becf"


def _series_from_statistics(stats: dict) -> dict[str, dict]:
    """Filas del informe estadistico, que ya trae todos los brazos alineados y descritos."""
    out = {}
    for row in stats.get("rows", []):
        out[row["name"]] = row
    return out


def _curves(arms: dict[str, list[float]], coverage: dict[str, int], out_path: Path,
            title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, daily in arms.items():
        equity = np.cumprod(1.0 + np.asarray(daily, dtype=float))
        ax.plot(range(len(equity)), 100.0 * (equity - 1.0),
                label=f"{name} (n={coverage[name]})",
                color=PALETTE.get(name, FALLBACK),
                linewidth=2.2 if name == "always_flat" else 1.5,
                linestyle="--" if name == "always_flat" else "-")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("sesión dentro del bloque")
    ax.set_ylabel("retorno compuesto acumulado (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _gross_vs_cost(decomposition: dict[str, tuple[float, float]], out_path: Path,
                   title: str) -> None:
    """`decomposition[brazo] = (bruto %, coste %)`.

    Se arma desde los artefactos y no desde las filas del informe estadistico: alli solo las
    corridas PPO por semilla llevan `total_cost_pct`, asi que la figura habria salido con la
    mitad de los brazos y sin decir cuales faltaban.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(decomposition)
    gross = [decomposition[n][0] for n in names]
    cost = [decomposition[n][1] for n in names]
    if not names:
        return

    order = np.argsort(gross)[::-1]
    names = [names[i] for i in order]
    gross = [gross[i] for i in order]
    cost = [cost[i] for i in order]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(names) + 1.5)))
    ax.barh(y - 0.2, gross, height=0.38, label="bruto (%)", color="#1f77b4")
    ax.barh(y + 0.2, [-c for c in cost], height=0.38, label="coste (%)", color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("% sobre el bloque")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--statistics", type=Path, required=True,
                    help="statistics_{block}.json producido por thesis_statistics.py")
    ap.add_argument("--settlement", action="append", default=[], metavar="NOMBRE=RUTA",
                    help="liquidacion o hibrido, para la curva de capital (repetible)")
    ap.add_argument("--ppo-dir", type=Path,
                    help="directorio de corridas PPO, para su curva media entre semillas")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    enable_deterministic_png()
    stats = json.loads(args.statistics.read_text(encoding="utf-8"))
    block = stats.get("block", "selection")
    rows = _series_from_statistics(stats)

    arms: dict[str, list[float]] = {}
    coverage: dict[str, int] = {}
    decomposition: dict[str, tuple[float, float]] = {}

    for item in args.settlement:
        name, _, path = item.partition("=")
        name = name.strip()
        payload = json.loads(Path(path.strip()).read_text(encoding="utf-8"))
        daily = [float(s["daily_return"]) for s in payload["sessions"]]
        arms[name] = daily
        coverage[name] = len(daily)
        decomposition[name] = (
            100.0 * sum(float(s["gross_return"]) for s in payload["sessions"]),
            100.0 * sum(float(s["total_cost"]) for s in payload["sessions"]),
        )

    if args.ppo_dir:
        for config in ("ppo_regime", "ppo_backbone"):
            runs, gross_runs, cost_runs = [], [], []
            for seed in (42, 123, 456, 789, 1337):
                f = args.ppo_dir / f"{config}_seed{seed}.json"
                if f.is_file():
                    blob = json.loads(f.read_text(encoding="utf-8"))
                    if block in blob:
                        b = blob[block]
                        runs.append(np.asarray(b["daily_returns"], dtype=float))
                        if "daily_gross_returns" in b and "daily_costs" in b:
                            gross_runs.append(np.asarray(b["daily_gross_returns"], dtype=float))
                            cost_runs.append(np.asarray(b["daily_costs"], dtype=float))
            if runs:
                mean = np.stack(runs).mean(axis=0)
                arms[f"{config}_mean5"] = mean.tolist()
                coverage[f"{config}_mean5"] = len(mean)
            if gross_runs and cost_runs:
                decomposition[f"{config}_mean5"] = (
                    100.0 * float(np.mean([g.sum() for g in gross_runs])),
                    100.0 * float(np.mean([c.sum() for c in cost_runs])),
                )

    n_block = stats.get("n_sessions")
    if n_block:
        arms["always_flat"] = [0.0] * int(n_block)
        coverage["always_flat"] = int(n_block)

    if not arms:
        print("sin brazos que dibujar")
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spread = sorted(set(coverage.values()))
    nota = ("" if len(spread) == 1 else
            f"  ·  ATENCION: coberturas distintas {spread}, las curvas no comparten eje temporal")
    _curves(arms, coverage, args.output_dir / f"comparativa_capital_{block}.png",
            f"Capital acumulado por brazo — bloque {block} (diagnóstico retrospectivo){nota}")
    if decomposition:
        ordered = dict(sorted(decomposition.items(), key=lambda kv: kv[1][0], reverse=True))
        _gross_vs_cost(ordered, args.output_dir / f"comparativa_bruto_coste_{block}.png",
                       f"Bruto contra coste por brazo — bloque {block}")

    print(json.dumps({
        "figuras": sorted(p.name for p in args.output_dir.glob("comparativa_*.png")),
        "brazos": sorted(arms),
        "coberturas": coverage,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
