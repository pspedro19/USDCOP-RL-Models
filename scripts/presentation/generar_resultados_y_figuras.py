#!/usr/bin/env python
"""Genera TODAS las tablas y figuras de la tesis desde artefactos (R7, §14).

Contract: CTR-RESEARCH-RESULTS-001 · Date: 2026-08-25

## Ningún número se escribe a mano

Cada cifra de cada tabla y cada punto de cada figura sale de un artefacto en disco:
`statistics_<bloque>.json`, las 10 corridas de PPO, `partition.yaml`, `evaluation_mask.json`
y `feature_schema.json`. §14 lo exige, y la razón es práctica: en cuanto una tabla se edita a
mano deja de reproducirse, y a la tercera revisión nadie sabe qué versión de los datos
produjo qué número.

El **reporte de coherencia** (§19.4) comprueba seis invariantes que un copiar-pegar rompería
sin dejar rastro — por ejemplo, que el Sharpe de la tabla 4.3 sea el mismo que el del
contraste pareado de la 4.9.

## PNG deterministas

`enable_deterministic_png()` quita el chunk `Software` que matplotlib estampa con su versión:
sin eso, regenerar las figuras produce bytes distintos con el mismo contenido y Git registra
cambios falsos en cada corrida.

Uso:
    python scripts/presentation/generar_resultados_y_figuras.py --block selection
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


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from src.research.inference import sharpe  # noqa: E402
from src.utils.plot_determinism import enable_deterministic_png  # noqa: E402

PPO_DIR = Path(os.environ.get("THESIS_PPO_OUT", REPO / "outputs" / "thesis" / "ppo"))
OUT = REPO / "outputs" / "thesis"
FIGS = OUT / "figuras"
TABLES = OUT / "tablas"

SEEDS = (42, 123, 456, 789, 1337)
CONFIGS = ("ppo_regime", "ppo_backbone")
PALETTE = {"ppo_regime": "#1f77b4", "ppo_backbone": "#ff7f0e", "always_flat": "#2ca02c",
           "B1_pasivo": "#7f7f7f", "B1_sesion_1x": "#d62728", "NULL_A_corto_1x": "#9467bd"}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_stats(block: str) -> dict:
    p = OUT / f"statistics_{block}.json"
    if not p.is_file():
        raise SystemExit(f"falta {p.relative_to(REPO)}: corre primero thesis_statistics.py "
                         f"--block {block}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_runs(block: str) -> dict:
    runs = {}
    for cfg in CONFIGS:
        for seed in SEEDS:
            f = PPO_DIR / f"{cfg}_seed{seed}.json"
            if f.is_file():
                blob = json.loads(f.read_text(encoding="utf-8"))
                if block in blob:
                    runs[(cfg, seed)] = blob[block]
    return runs


def series_from_runs(runs: dict) -> dict[str, np.ndarray]:
    out = {}
    for cfg in CONFIGS:
        rows = [np.asarray(v["daily_returns"], float)
                for (c, _), v in sorted(runs.items()) if c == cfg]
        if rows:
            out[cfg] = np.stack(rows).mean(axis=0)
    return out


def rebuild_baselines(block: str) -> dict[str, np.ndarray]:
    """Recalcula los baselines sobre los mismos specs: alineación por construcción."""
    from src.research.dataset import PORTABLE, load_or_build, load_portable
    from src.research.session_env import daily_series, run_session

    data = load_portable() if PORTABLE.is_file() else load_or_build(verbose=False)
    specs = data.block(block)

    def const(level):
        return daily_series([run_session(s.close, np.full(59, level), s.spread_pips,
                                         date=s.date) for s in specs])

    closes = np.array([s.close[-1] for s in specs], dtype=float)
    passive = np.concatenate([[0.0], np.diff(closes) / closes[:-1]])
    return ({"B1_pasivo": passive, "B1_sesion_1x": const(1.0),
             "NULL_A_corto_1x": const(-1.0), "always_flat": const(0.0)}, specs, data)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def equity(r: np.ndarray) -> np.ndarray:
    return 10_000.0 * np.cumprod(1.0 + np.asarray(r, float))


def drawdown(r: np.ndarray) -> np.ndarray:
    c = np.cumprod(1.0 + np.asarray(r, float))
    return c / np.maximum.accumulate(c) - 1.0


def rolling_sharpe(r: np.ndarray, w: int = 60) -> np.ndarray:
    r = np.asarray(r, float)
    out = np.full(len(r), np.nan)
    for i in range(w, len(r) + 1):
        out[i - 1] = sharpe(r[i - w:i])
    return out


def save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    p = FIGS / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def write_table(name: str, header: list[str], rows: list[list]) -> Path:
    """Markdown, que es lo que se pega en el documento sin reformatear.

    El NOMBRE lleva el bloque cuando la tabla depende de el (4.3 en adelante). Sin eso,
    generar el hold-out sobreescribia en silencio las tablas de seleccion con las mismas
    etiquetas y numeros distintos — y el fichero seguia pareciendo valido.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    p = TABLES / f"{name}.md"
    def cell(x):
        # Un `|` sin escapar parte la fila y markdown lo renderiza como columnas de mas,
        # en silencio. Paso obligatorio: la cabecera `|Δw|` ya lo provoco una vez.
        return "" if x is None else str(x).replace("|", r"\|")

    lines = ["| " + " | ".join(cell(h) for h in header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(cell(x) for x in r) + " |")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _disambiguate(labels: list[str]) -> list[str]:
    """`[calmo, intermedio, intermedio, shock]` -> `[calmo, intermedio_1, intermedio_2, shock]`."""
    from collections import Counter
    counts = Counter(labels)
    seen: dict[str, int] = {}
    out = []
    for lab in labels:
        if counts[lab] == 1:
            out.append(lab)
        else:
            seen[lab] = seen.get(lab, 0) + 1
            out.append(f"{lab}_{seen[lab]}")
    return out


def fmt(x, nd: int = 2, pct: bool = False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}%" if pct else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def fig_equity(series: dict, dates, block: str):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, r in series.items():
        ax.plot(dates, equity(r), label=name, lw=1.8,
                color=PALETTE.get(name), alpha=0.9)
    ax.axhline(10_000, color="k", lw=0.8, ls=":")
    ax.set_title(f"Curvas de capital · bloque {block} (inicio $10.000)")
    ax.set_ylabel("USD"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return save(fig, f"fig01_curvas_capital_{block}.png")


def fig_underwater(series: dict, dates, block: str):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, r in series.items():
        ax.fill_between(dates, 100 * drawdown(r), 0, alpha=0.25,
                        color=PALETTE.get(name), label=name)
    ax.set_title(f"Underwater · bloque {block}")
    ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return save(fig, f"fig02_underwater_{block}.png")


def fig_rolling_sharpe(series: dict, dates, block: str, w: int = 60):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, r in series.items():
        ax.plot(dates, rolling_sharpe(r, w), label=name, lw=1.5, color=PALETTE.get(name))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"Sharpe móvil de {w} sesiones · bloque {block}")
    ax.set_ylabel("Sharpe anualizado"); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return save(fig, f"fig03_sharpe_movil_{block}.png")


def fig_cost_sensitivity(stats: dict, block: str):
    """El eje que decide REJECT: la constitución §3.4 mata lo que muere al doble."""
    stress = stats.get("cost_stress") or {}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mults = [1, 2, 3]
    for cfg, s in stress.items():
        ys = [s[f"x{m}"]["ann_return_pct"] for m in mults]
        ax.plot(mults, ys, "o-", label=cfg, color=PALETTE.get(cfg), lw=2)
    ax.axhline(0, color="k", lw=1, ls="--", label="umbral de supervivencia")
    ax.set_xticks(mults); ax.set_xlabel("multiplicador de costos")
    ax.set_ylabel("retorno anualizado (%)")
    ax.set_title(f"Sensibilidad a costos ×1/×2/×3 · bloque {block}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return save(fig, f"fig04_sensibilidad_costos_{block}.png")


def fig_action_distribution(exposures: dict, regimes: np.ndarray, labels: list, block: str):
    """Qué hace el agente en cada régimen. Si es lo mismo en todos, el régimen no le informa."""
    levels = (-1.0, -0.5, 0.0, 0.5, 1.0)
    n = len(exposures)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5.5 * max(n, 1), 4.2), squeeze=False)
    for ax, (cfg, exp) in zip(axes[0], exposures.items()):
        width = 0.8 / max(len(labels), 1)
        for j, lab in enumerate(labels):
            m = regimes == j
            if not m.any():
                continue
            # `exp` es (sesiones x 59 barras): se aplana el bloque del regimen. La primera
            # version usaba la MEDIA de exposicion por sesion y la comparaba con los niveles
            # discretos — una media casi nunca cae exactamente en {-1,-0.5,0,0.5,1}, asi que
            # las frecuencias no sumaban 1 y el grafico infra-representaba todo.
            flat = exp[m].ravel()
            counts = [float(np.mean(np.isclose(flat, L))) for L in levels]
            ax.bar(np.arange(len(levels)) + j * width, counts, width, label=f"{lab}")
        ax.set_xticks(np.arange(len(levels)) + 0.4)
        ax.set_xticklabels([str(L) for L in levels])
        ax.set_title(cfg); ax.set_xlabel("exposición"); ax.set_ylabel("frecuencia")
        ax.legend(fontsize=7); ax.grid(alpha=0.2, axis="y")
    fig.suptitle(f"Distribución de acciones por régimen · bloque {block}")
    return save(fig, f"fig05_acciones_por_regimen_{block}.png")


def fig_seed_profiles(runs: dict, block: str):
    """Dispersión entre semillas: una barra alta con vecinas bajas es ruido, no hallazgo."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.38
    for i, cfg in enumerate(CONFIGS):
        vals = [runs.get((cfg, s), {}).get("sharpe", np.nan) for s in SEEDS]
        ax.bar(np.arange(len(SEEDS)) + i * width, vals, width, label=cfg,
               color=PALETTE.get(cfg))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(np.arange(len(SEEDS)) + width / 2)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("semilla"); ax.set_ylabel("Sharpe anualizado")
    ax.set_title(f"Robustez entre semillas · bloque {block}")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    return save(fig, f"fig06_semillas_{block}.png")


def fig_gross_vs_net(expl: dict, decomp: dict, block: str):
    """Bruto y neto acumulados, juntos. Es la tesis entera en una imagen.

    La distancia entre las dos curvas ES el costo de ejecucion. Que el bruto suba mientras el
    neto se hunde dice, sin una sola palabra, que el problema no es la prediccion.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cfg in CONFIGS:
        rows = [r for r in decomp["runs"].values() if r["config"] == cfg]
        if not rows:
            continue
        gross = np.stack([[x["gross_return"] for x in r["sessions"]] for r in rows]).mean(0)
        net = np.stack([[x["daily_return"] for x in r["sessions"]] for r in rows]).mean(0)
        c = PALETTE.get(cfg)
        ax.plot(np.cumsum(gross) * 100, lw=2.0, color=c,
                label=f"{cfg} · BRUTO (costo cero, inalcanzable)")
        ax.plot(np.cumsum(net) * 100, lw=1.6, ls="--", color=c, alpha=0.8,
                label=f"{cfg} · neto")
    ax.axhline(0, color="k", lw=1.0, ls=":")
    ax.set_title(f"Bruto frente a neto acumulados · bloque {block}"
                 "\nla distancia entre las curvas es el costo de ejecucion")
    ax.set_xlabel("sesion"); ax.set_ylabel("retorno acumulado (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    return save(fig, f"fig11_bruto_vs_neto_{block}.png")


def fig_frequency(expl: dict, block: str):
    """Bruto, costo y neto frente a la frecuencia de decision."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for cfg in CONFIGS:
        curve = (expl["per_config"].get(cfg) or {}).get("frequency_curve")
        if not curve:
            continue
        x = [c["decisions_per_session"] for c in curve]
        col = PALETTE.get(cfg)
        ax.plot(x, [100 * c["sum_gross"] for c in curve], "o-", color=col, lw=2,
                label=f"{cfg} · bruto")
        ax.plot(x, [-100 * c["sum_cost"] for c in curve], "s--", color=col, alpha=0.55,
                label=f"{cfg} · -costo")
        ax.plot(x, [100 * c["sum_net"] for c in curve], "^-", color=col, lw=2.4,
                alpha=0.9, label=f"{cfg} · NETO")
    ax.axhline(0, color="k", lw=1.0)
    ax.set_xscale("log"); ax.set_xlabel("decisiones por sesion (escala log)")
    ax.set_ylabel("acumulado sobre el bloque (%)")
    ax.set_title(f"Sensibilidad a la frecuencia · bloque {block}"
                 "\nmismas decisiones del agente, muestreadas cada k barras")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.25, which="both")
    return save(fig, f"fig10_frecuencia_{block}.png")


def fig_partitions():
    """El esquema de particiones. Documenta que 2019 no existe, que es de dónde salió la Opción C."""
    part = yaml.safe_load((REPO / "config" / "research" / "partition.yaml").read_text(
        encoding="utf-8"))
    eff = part["effective_sessions"]
    fig, ax = plt.subplots(figsize=(11, 2.8))
    colors = {"development": "#4c72b0", "selection": "#dd8452", "holdout": "#55a868"}
    import datetime as dt
    for name in ("development", "selection", "holdout"):
        b = part["blocks"][name]
        lo = dt.date.fromisoformat(str(b["start"]))
        hi = dt.date.fromisoformat(str(b["end"]))
        ax.barh(0, (hi - lo).days, left=lo, height=0.5, color=colors[name],
                label=f"{name}: {eff[name]} efectivas de {b['sessions']}")
    ax.set_yticks([]); ax.legend(fontsize=8, loc="upper center", ncol=3,
                                 bbox_to_anchor=(0.5, -0.15))
    ax.set_title("Partición temporal (Opción C) · sesiones efectivas tras la máscara")
    ax.grid(alpha=0.25, axis="x")
    return save(fig, "fig07_particiones.png")


def fig_sharpe_by_regime(series: dict, regimes: np.ndarray, labels: list, block: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / max(len(series), 1)
    for i, (name, r) in enumerate(series.items()):
        vals = []
        for j in range(len(labels)):
            m = regimes == j
            vals.append(sharpe(r[m]) if m.sum() >= 20 else np.nan)
        ax.bar(np.arange(len(labels)) + i * width, vals, width, label=name,
               color=PALETTE.get(name))
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(np.arange(len(labels)) + 0.4 - width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sharpe anualizado")
    ax.set_title(f"Sharpe por régimen · bloque {block} "
                 f"(régimen con <20 sesiones se omite, constitución §6)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    return save(fig, f"fig08_sharpe_por_regimen_{block}.png")


def fig_example_session(spec, exposure: np.ndarray, block: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    bars = np.arange(len(spec.close))
    ax1.plot(bars, spec.close, color="#333", lw=1.6)
    ax1.set_ylabel("USD/COP"); ax1.grid(alpha=0.25)
    ax1.set_title(f"Sesión {spec.date} · spread esperado {spec.spread_pips:.2f} pips")
    ax2.step(np.arange(len(exposure)), exposure, where="post", color="#1f77b4", lw=1.8)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_ylim(-1.2, 1.2); ax2.set_ylabel("exposición"); ax2.set_xlabel("barra de 5 min")
    ax2.grid(alpha=0.25)
    return save(fig, f"fig09_sesion_ejemplo_{block}.png")


# ---------------------------------------------------------------------------
# Detalle por sesión (requiere los modelos)
# ---------------------------------------------------------------------------

def replay_exposures(cfg: str, seed: int, specs) -> np.ndarray | None:
    """Reproduce la senda de exposición barra a barra con el modelo guardado."""
    model_path = PPO_DIR / f"{cfg}_seed{seed}.zip"
    if not model_path.is_file():
        return None
    try:
        from stable_baselines3 import PPO

        from scripts.analysis.thesis_train_ppo import strip_regimes
        from src.research.session_gym import SessionTradingEnv
    except ImportError:
        return None

    use = specs if cfg == "ppo_regime" else strip_regimes(specs)
    model = PPO.load(str(model_path), device="cpu")
    env = SessionTradingEnv(use, seed=0, shuffle=False)
    paths = []
    for _ in range(len(use)):
        obs, _ = env.reset()
        w, done = [], False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(a))
            w.append(env.last_result.weights[-1] if done else env._w_prev)
        paths.append(np.asarray(w, dtype=float))
    return np.stack(paths)


# ---------------------------------------------------------------------------
# Reporte de coherencia (§19.4)
# ---------------------------------------------------------------------------

def coherence_report(stats: dict, series: dict, runs: dict, specs,
                     decomp: dict | None = None) -> list[dict]:
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # 1. n consistente entre la partición, la tabla y las series
    n_decl = stats["n_sessions"]
    ns = {k: len(v) for k, v in series.items()}
    check("n_sesiones_consistente", all(v == n_decl for v in ns.values()),
          f"declarado {n_decl}, series {sorted(set(ns.values()))}")

    # 2. el Sharpe de la tabla == el del contraste pareado
    rows = {r["name"]: r for r in stats["rows"]}
    ok, detail = True, []
    for tname, t in stats.get("paired_tests", {}).items():
        row = rows.get(t["name_a"])
        if row and row["sharpe"] is not None:
            if abs(row["sharpe"] - t["sharpe_a"]) > 0.005:
                ok = False
                detail.append(f"{t['name_a']}: tabla {row['sharpe']} vs test {t['sharpe_a']}")
    check("sharpe_tabla_igual_al_del_contraste", ok, "; ".join(detail) or "coinciden")

    # 3. always_flat no puede tener retorno ni costo
    flat = rows.get("always_flat")
    check("always_flat_es_exactamente_cero",
          flat is not None and abs(flat["total_return_pct"]) < 1e-6,
          f"retorno {flat['total_return_pct'] if flat else 'ausente'}%")

    # 4. las 10 corridas están
    check("las_10_corridas_existen", len(runs) == 10, f"{len(runs)} de 10")

    # 5. el DSR se deflacta con el N heredado, no con cero
    # El DSR debe deflactarse con el conteo ACTUALIZADO del activo (>= el heredado), nunca
    # con cero: sub-deflactar es la violacion concreta que la constitucion §2 prohibe.
    inh = stats.get("inherited_trials")
    used = [d.get("n_trials", 0) for d in (stats.get("dsr") or {}).values()]
    ok5 = bool(used) and all(n >= inh >= 111 for n in used)
    check("dsr_deflactado_con_N_del_activo", ok5,
          f"heredados={inh}, usados en el DSR={used}")

    # 6. White/SPA declarados como omitidos, no ausentes en silencio
    check("omision_white_spa_declarada",
          "White" in (stats.get("white_spa") or ""), "presente en el JSON")

    # 7. La descomposicion cuadra POR SESION, no solo en la suma (CTR-RESEARCH-DECOMP-001).
    #    Es lo que hace fiable el bruto: si el replay no reprodujera la evaluacion original,
    #    todas las conclusiones sobre "hay senal" saldrian de un modelo distinto del que se
    #    publico, y los numeros seguirian pareciendo razonables.
    if decomp:
        worst, offender = 0.0, None
        for tag, run in decomp["runs"].items():
            for sess in run["sessions"]:
                d = abs((sess["gross_return"] - sess["total_cost"]) - sess["daily_return"])
                if d > worst:
                    worst, offender = d, f"{tag}/{sess['date']}"
        check("descomposicion_cuadra_por_sesion", worst < 1e-12,
              f"peor delta {worst:.2e}" + (f" en {offender}" if offender else ""))

    return checks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="selection",
                    choices=["development", "selection", "holdout"])
    ap.add_argument("--no-replay", action="store_true",
                    help="salta las figuras que exigen recargar los modelos")
    args = ap.parse_args()

    enable_deterministic_png()
    stats = load_stats(args.block)
    runs = load_runs(args.block)
    baselines, specs, data = rebuild_baselines(args.block)
    ppo = series_from_runs(runs)

    series = {**baselines, **ppo}
    import pandas as pd
    dates = pd.to_datetime([s.date for s in specs])

    produced = []
    produced.append(fig_equity(series, dates, args.block))
    produced.append(fig_underwater(series, dates, args.block))
    produced.append(fig_rolling_sharpe(series, dates, args.block))
    produced.append(fig_cost_sensitivity(stats, args.block))
    produced.append(fig_seed_profiles(runs, args.block))
    produced.append(fig_partitions())

    # Descomposicion (CTR-RESEARCH-DECOMP-001), si el artefacto existe. No es obligatorio:
    # las tablas y figuras base tienen que poder generarse sin el.
    expl_path = OUT / f"explanation_{args.block}.json"
    decomp_path = OUT / f"decomposition_{args.block}.json"
    expl = json.loads(expl_path.read_text(encoding="utf-8")) if expl_path.is_file() else None
    decomp = (json.loads(decomp_path.read_text(encoding="utf-8"))
              if decomp_path.is_file() else None)
    if expl and decomp:
        produced.append(fig_frequency(expl, args.block))
        produced.append(fig_gross_vs_net(expl, decomp, args.block))

    # Regimen por sesion: el argmax del posterior que ya vive en el contexto del spec.
    n_reg = len(data.regime_model["labels"]) if isinstance(data.regime_model, dict) else 4
    labels = (list(data.regime_model["labels"]) if isinstance(data.regime_model, dict)
              else [f"r{i}" for i in range(n_reg)])
    # K=4 produce DOS estados `intermedio` (ninguno alcanza la persistencia de 0,8 que
    # §8.3 exige para `tendencial`). Sin desambiguar, la tabla 4.4 tiene dos filas con el
    # mismo nombre y numeros distintos, que se lee como un error de la tabla.
    labels = _disambiguate(labels)
    regimes = np.array([int(np.argmax(s.context[-n_reg:])) for s in specs])
    produced.append(fig_sharpe_by_regime(series, regimes, labels, args.block))

    if not args.no_replay:
        exposures = {}
        for cfg in CONFIGS:
            paths = replay_exposures(cfg, SEEDS[0], specs)
            if paths is not None:
                exposures[cfg] = paths                   # (sesiones x barras), sin promediar
        if exposures:
            produced.append(fig_action_distribution(exposures, regimes, labels, args.block))
            first = replay_exposures(CONFIGS[0], SEEDS[0], specs[:1])
            if first is not None:
                produced.append(fig_example_session(specs[0], first[0], args.block))

    # --- tablas -----------------------------------------------------------
    part = yaml.safe_load((REPO / "config" / "research" / "partition.yaml").read_text(
        encoding="utf-8"))
    mask = json.loads((REPO / "config" / "research" / "evaluation_mask.json").read_text(
        encoding="utf-8"))
    schema = json.loads((REPO / "config" / "research" / "feature_schema.json").read_text(
        encoding="utf-8"))

    tabs = []
    tabs.append(write_table(
        "tabla_4_1_datos", ["concepto", "valor"],
        [["fuente", part["source"]["file"]],
         ["barras", part["source"]["rows"]],
         ["sesiones totales", part["source"]["sessions"]],
         ["rango", " → ".join(map(str, part["source"]["range"]))],
         ["sesiones válidas (máscara)", mask["n_valid"]],
         ["hash de la máscara", mask["sha256"][:16]],
         # Las claves las fija `evaluation_mask.py`; un `.get` con el nombre equivocado
         # devolvia 0 en silencio y la tabla decia que no se habia excluido nada.
         ["festivos excluidos", mask["excluded_counts"]["holiday"]],
         ["incompletas excluidas", mask["excluded_counts"]["incomplete"]],
         ["features", schema["n_features"]],
         ["hash del esquema", schema["sha256"][:16]]]))

    # Tres conteos, no dos. La version anterior titulaba "efectivas" al conteo posterior a la
    # mascara de evaluacion, pero ESE NO ES el `n` con el que se hizo el analisis: construir
    # los `SessionSpec` descarta ademas las sesiones sin posterior de regimen (calentamiento
    # del HMM) y las que no traen 60 barras. En development eso son 59 sesiones mas, y la
    # tabla declaraba 558 mientras todos los IC y bootstraps corrian sobre 499.
    #
    # El hold-out no se ve afectado (584 = 584), asi que el veredicto nunca estuvo en juego;
    # era integridad del reporte. Se publican los tres conteos y el rango REALMENTE usado.
    part_rows = []
    for b in ("development", "selection", "holdout"):
        specs = data.block(b)
        part_rows.append([
            b,
            f"{part['blocks'][b]['start']} → {part['blocks'][b]['end']}",
            part["blocks"][b]["sessions"],
            part["effective_sessions"][b],
            len(specs),
            f"{specs[0].date} → {specs[-1].date}" if specs else "—",
            part["blocks"][b]["purpose"]])

    tabs.append(write_table(
        "tabla_4_2_particion",
        ["bloque", "rango declarado", "existen", "tras máscara", "usables (n del análisis)",
         "rango usable", "propósito"],
        part_rows))

    tabs.append(write_table(
        f"tabla_4_3_desempeno_{args.block}",
        ["estrategia", "n", "retorno %", "Sharpe", "IC 95%", "MaxDD %", "ops"],
        [[r["name"], r["n"], fmt(r["total_return_pct"]), fmt(r["sharpe"]),
          (f"[{r['sharpe_ci95'][0]:+.2f}, {r['sharpe_ci95'][1]:+.2f}]"
           if r.get("sharpe_ci95") else "n/a"),
          fmt(r["max_dd_pct"]), r.get("n_ops", "")] for r in stats["rows"]]))

    reg_rows = []
    for name, r in series.items():
        for j, lab in enumerate(labels):
            m = regimes == j
            if m.sum() < 20:
                reg_rows.append([name, lab, int(m.sum()), "n/a",
                                 "N<20: constitución §6"])
            else:
                reg_rows.append([name, lab, int(m.sum()), fmt(sharpe(r[m])),
                                 fmt(100 * (np.prod(1 + r[m]) - 1)) + "%"])
    tabs.append(write_table(f"tabla_4_4_por_regimen_{args.block}",
                            ["estrategia", "régimen", "n", "Sharpe", "retorno"], reg_rows))

    tabs.append(write_table(
        f"tabla_4_5_ablacion_{args.block}",
        ["contraste", "ΔSharpe", "IC 95%", "p", "n", "ρ", "veredicto"],
        [[k, fmt(t["diff"], 3), f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}]",
          fmt(t["p_value"], 4), t["n"], fmt(t["correlation"], 3),
          "DECIDIBLE" if t["decisive"] else "INDECIDIBLE"]
         for k, t in stats.get("paired_tests", {}).items()]))

    stress = stats.get("cost_stress") or {}
    tabs.append(write_table(
        f"tabla_4_6_costos_{args.block}",
        ["configuración", "×1 anual %", "×2 anual %", "×3 anual %", "sobrevive ×2",
         "sobrevive ×3"],
        [[cfg, fmt(s["x1"]["ann_return_pct"]), fmt(s["x2"]["ann_return_pct"]),
          fmt(s["x3"]["ann_return_pct"]), "sí" if s["survives_2x"] else "NO",
          "sí" if s["survives_3x"] else "NO"] for cfg, s in stress.items()]))

    dsr = stats.get("dsr") or {}
    tabs.append(write_table(
        f"tabla_4_7_dsr_pbo_{args.block}",
        ["configuración", "Sharpe", "n_trials", "DSR", "pasa 0.95", "PBO"],
        [[cfg, fmt(d.get("sharpe_annualized")), d.get("n_trials"),
          fmt(d.get("headline_dsr"), 4), "sí" if d.get("passes") else "NO",
          fmt((stats.get("pbo") or {}).get("pbo"), 3)] for cfg, d in dsr.items()]))

    tabs.append(write_table(
        f"tabla_4_10_semillas_{args.block}",
        ["configuración", "semilla", "retorno %", "Sharpe", "ops", "|exposición|"],
        [[cfg, seed, fmt(100 * runs[(cfg, seed)]["total_return"]),
          fmt(runs[(cfg, seed)]["sharpe"]), runs[(cfg, seed)]["n_ops"],
          fmt(runs[(cfg, seed)]["mean_abs_exposure"], 3)]
         for cfg in CONFIGS for seed in SEEDS if (cfg, seed) in runs]))

    if expl and decomp:
        tabs.append(write_table(
            f"tabla_4_8_descomposicion_{args.block}",
            ["configuración", "semilla", "BRUTO %", "costo %", "neto %",
             "turnover (suma de |dw|)"],
            [[r["config"], r["seed"], fmt(100 * r["sum_gross"]), fmt(100 * r["sum_cost"]),
              fmt(100 * r["sum_net"]), fmt(r["sum_abs_dw"], 1)]
             for r in sorted(decomp["runs"].values(),
                             key=lambda x: (x["config"], x["seed"]))]))

        rows_be = []
        for cfg, blk in expl["per_config"].items():
            be = blk.get("break_even") or {}
            rows_be.append([
                cfg, fmt(be.get("spread_star_pips")), fmt(expl["mean_spread_pips"]),
                fmt(be.get("production_equivalent_spread_pips")),
                fmt(be.get("alpha_per_unit_dw_pips")), fmt(be.get("cost_per_unit_dw_pips")),
                "sí" if be.get("viable_at_assumed") else "NO",
                "sí" if be.get("viable_at_production_assumption") else "NO"])
        tabs.append(write_table(
            f"tabla_4_11_break_even_{args.block}",
            ["configuración", "s* (pips)", "spread tesis", "spread producción equiv.",
             "alfa/op (pips)", "costo/op (pips)", "viable @tesis", "viable @producción"],
            rows_be))

        rows_fr = []
        for cfg, blk in expl["per_config"].items():
            for pt in blk.get("frequency_curve", []):
                rows_fr.append([cfg, pt["k"], pt["decisions_per_session"],
                                fmt(100 * pt["sum_gross"]), fmt(100 * pt["sum_cost"]),
                                fmt(100 * pt["sum_net"])])
        if rows_fr:
            tabs.append(write_table(
                f"tabla_4_12_frecuencia_{args.block}",
                ["configuración", "k", "decisiones/sesión", "BRUTO %", "costo %", "neto %"],
                rows_fr))

    # --- coherencia -------------------------------------------------------
    checks = coherence_report(stats, series, runs, specs, decomp)
    report = OUT / f"coherencia_{args.block}.json"
    report.write_text(json.dumps({"block": args.block, "checks": checks,
                                  "all_ok": all(c["ok"] for c in checks)},
                                 indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nFiguras ({len(produced)}):")
    for p in produced:
        print(f"  {p.relative_to(REPO)}")
    print(f"\nTablas ({len(tabs)}):")
    for p in tabs:
        print(f"  {p.relative_to(REPO)}")
    print(f"\nReporte de coherencia (§19.4) -> {report.relative_to(REPO)}")
    for c in checks:
        print(f"  [{'OK ' if c['ok'] else 'FALLA'}] {c['check']}: {c['detail']}")
    return 0 if all(c["ok"] for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
