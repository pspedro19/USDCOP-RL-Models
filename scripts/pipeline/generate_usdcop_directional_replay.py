#!/usr/bin/env python
"""Generate the causal USD/COP weekly directional replay and dashboard charts.

Usage:
    python scripts/pipeline/generate_usdcop_directional_replay.py
    python scripts/pipeline/generate_usdcop_directional_replay.py --validate-only
"""
from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
# PNG deterministas: omite el chunk `Software` de matplotlib para que subir de
# version no reescriba los 1.492 graficos versionados. Ver
# src/utils/plot_determinism.py (auditoria de limpieza 2026-08-24).
from src.utils.plot_determinism import enable_deterministic_png
enable_deterministic_png()
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DAILY_PARQUET = ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"


@lru_cache(maxsize=1)
def _daily_close_series() -> pd.DataFrame:
    """Cargar la serie de cierres diarios USD/COP (train = diario) una sola vez."""
    frame = pd.read_parquet(DAILY_PARQUET)[["time", "close"]].copy()
    frame["time"] = pd.to_datetime(frame["time"])
    return frame.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)


def _recent_daily(origin_date: str, sessions: int = 15):
    """~3 semanas (15 sesiones) de cierre diario ESTRICTAMENTE <= origen (causal)."""
    frame = _daily_close_series()
    cutoff = pd.to_datetime(origin_date)
    window = frame[frame["time"] <= cutoff].tail(sessions)
    return window["time"].tolist(), window["close"].astype(float).tolist()

from src.forecasting.directional_replay import (
    build_directional_replay,
    flatten_ledger,
    load_replay_config,
    resolve_paths,
    summary_frame,
    validate_replay_document,
)
from src.forecasting.directional_replay_excel import export_directional_replay_workbook


DEFAULT_CONFIG = ROOT / "config/forecast_experiments/usdcop_directional_macro_replay_v1.yaml"

COLORS = {
    "background": "#08111f",
    "panel": "#0f1d2e",
    "grid": "#25364a",
    "text": "#e8eef7",
    "muted": "#8fa5bd",
    "up": "#2dd4a8",
    "down": "#fb7185",
    "selected": "#fbbf24",
    "execution": "#60a5fa",
    "ineligible": "#52647a",
    "threshold": "#c084fc",
}


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _direction_label(direction: str) -> str:
    return {
        "UP": "USD ↑ / COP ↓",
        "DOWN": "USD ↓ / COP ↑",
        "FLAT": "SIN POSICIÓN",
    }.get(direction, direction)


def render_week_image(week: dict, output_path: Path) -> None:
    """Render an as-of chart; no future realized label is drawn into the snapshot."""
    horizons = week["horizons"]
    labels = [f"H{item['horizon_days']}" for item in horizons]
    x = np.arange(len(horizons))
    probabilities = np.array([float(item["probability_up"]) for item in horizons])
    thresholds = np.array([float(item["threshold"]) for item in horizons])
    scores = np.array([
        float(item["evidence"]["shrunk_score"])
        if item["evidence"]["shrunk_score"] is not None else 0.5
        for item in horizons
    ])
    evidence_n = [int(item["evidence"]["n"]) for item in horizons]
    forecast_prices = np.array([float(item["forecast_price"]) for item in horizons])
    interval_lower = np.array([float(item["forecast_interval_lower"]) for item in horizons])
    interval_upper = np.array([float(item["forecast_interval_upper"]) for item in horizons])
    base_price = float(week["base_price"])

    bar_colors = []
    for item in horizons:
        if item["selected"]:
            bar_colors.append(COLORS["selected"])
        elif item["role"] == "execution":
            bar_colors.append(COLORS["execution"])
        elif not item["eligible_for_direction"]:
            bar_colors.append(COLORS["ineligible"])
        else:
            bar_colors.append(COLORS["up"] if item["prediction"] == "UP" else COLORS["down"])

    figure = plt.figure(figsize=(13.2, 12.6), facecolor=COLORS["background"])
    grid = figure.add_gridspec(4, 1, height_ratios=[1.25, 1.0, 1.08, 0.8], hspace=0.42)
    context_ax = figure.add_subplot(grid[0])
    probability_ax = figure.add_subplot(grid[1])
    price_ax = figure.add_subplot(grid[2])
    evidence_ax = figure.add_subplot(grid[3])
    for axis in (context_ax, probability_ax, price_ax, evidence_ax):
        axis.set_facecolor(COLORS["panel"])
        axis.tick_params(colors=COLORS["muted"], labelsize=9)
        axis.grid(axis="y", color=COLORS["grid"], alpha=0.55, linewidth=0.8)
        for spine in axis.spines.values():
            spine.set_color(COLORS["grid"])

    # --- Panel héroe: 3 semanas de cierre diario (eje fecha) + trayectoria forward ---
    hist_dates, hist_close = _recent_daily(week["origin_date"], sessions=15)
    target_dates = [pd.to_datetime(item["target_date"]) for item in horizons]
    if hist_dates:
        context_ax.plot(
            hist_dates, hist_close, color=COLORS["execution"], linewidth=1.7,
            marker="o", markersize=3.2, label="Cierre diario (≈3 semanas)", zorder=3,
        )
        forward_x = [hist_dates[-1]] + target_dates
        forward_y = [base_price] + list(forecast_prices)
    else:
        forward_x, forward_y = target_dates, list(forecast_prices)
    context_ax.fill_between(
        target_dates, interval_lower, interval_upper,
        color=COLORS["selected"], alpha=0.10, zorder=1,
    )
    context_ax.plot(
        forward_x, forward_y, color=COLORS["selected"], linewidth=1.8, linestyle="--",
        marker="o", markersize=4.5, label="Forward por horizonte", zorder=4,
    )
    for item, target_date, forecast_price in zip(horizons, target_dates, forecast_prices):
        point_color = COLORS["up"] if item["point_forecast_direction"] == "UP" else COLORS["down"]
        context_ax.scatter(
            target_date, forecast_price,
            s=90 if item["selected"] else 46,
            marker="o" if item["direction_price_agree"] else "X", color=point_color,
            edgecolors=COLORS["selected"] if item["selected"] else COLORS["background"],
            linewidths=1.4, zorder=5,
        )
        context_ax.annotate(
            f"H{item['horizon_days']}", (target_date, forecast_price),
            xytext=(0, 8), textcoords="offset points", ha="center",
            fontsize=7.6, color=COLORS["muted"], fontweight="bold",
        )
    context_ax.axhline(
        base_price, color=COLORS["threshold"], linewidth=1.1, linestyle=":",
        alpha=0.8, label=f"Spot origen {base_price:,.2f}",
    )
    if hist_dates:
        context_ax.axvline(hist_dates[-1], color=COLORS["muted"], linewidth=1.0,
                           linestyle="--", alpha=0.45)
    context_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    context_ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    context_ax.set_ylabel("USD/COP", color=COLORS["muted"], fontsize=10)
    context_ax.set_title(
        "Contexto diario (≈3 semanas) + trayectoria forward por horizonte — línea punteada = pronóstico",
        loc="left", color=COLORS["text"], fontsize=11, fontweight="bold", pad=10,
    )
    context_ax.legend(loc="upper left", frameon=False, labelcolor=COLORS["muted"], fontsize=8.5)

    bars = probability_ax.bar(x, probabilities * 100, color=bar_colors, width=0.66, alpha=0.92)
    probability_ax.scatter(
        x, thresholds * 100, marker="D", s=38, color=COLORS["threshold"],
        edgecolors=COLORS["background"], linewidths=0.7, zorder=4, label="Umbral fijado 2022–2024",
    )
    probability_ax.axhline(50, color=COLORS["muted"], linewidth=1, linestyle="--", alpha=0.55)
    probability_ax.set_ylim(0, 100)
    probability_ax.set_xticks(x, labels)
    probability_ax.set_ylabel("Probabilidad USD/COP ↑", color=COLORS["muted"], fontsize=10)
    probability_ax.set_title(
        "Inferencia por horizonte — amarillo = mejor candidato por sleeve · azul = timing H1",
        loc="left", color=COLORS["text"], fontsize=11, fontweight="bold", pad=10,
    )
    probability_ax.legend(loc="upper right", frameon=False, labelcolor=COLORS["muted"], fontsize=8.5)
    for bar, item, probability in zip(bars, horizons, probabilities):
        marker = "✓" if item["selected"] else ""
        probability_ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(96, probability * 100 + 3.0),
            f"{probability:.1%} {marker}",
            ha="center", va="bottom", fontsize=9, color=COLORS["text"], fontweight="bold",
        )
        probability_ax.text(
            bar.get_x() + bar.get_width() / 2,
            3.0,
            item["prediction"],
            ha="center", va="bottom", fontsize=8, color=COLORS["text"], fontweight="bold",
        )

    price_ax.plot(x, forecast_prices, color=COLORS["muted"], linewidth=1.2, alpha=0.65, zorder=2)
    price_ax.errorbar(
        x,
        forecast_prices,
        yerr=np.vstack([forecast_prices - interval_lower, interval_upper - forecast_prices]),
        fmt="none",
        ecolor=COLORS["grid"],
        elinewidth=4.5,
        capsize=4,
        alpha=0.9,
        zorder=1,
    )
    price_ax.axhline(
        base_price, color=COLORS["threshold"], linewidth=1.2, linestyle="--",
        alpha=0.85, label=f"Spot {base_price:,.2f}",
    )
    for index, item in enumerate(horizons):
        point_color = COLORS["up"] if item["point_forecast_direction"] == "UP" else COLORS["down"]
        marker = "o" if item["direction_price_agree"] else "X"
        price_ax.scatter(
            index, forecast_prices[index], s=82 if item["selected"] else 58,
            marker=marker, color=point_color,
            edgecolors=COLORS["selected"] if item["selected"] else COLORS["background"],
            linewidths=1.5, zorder=4,
        )
        above = forecast_prices[index] >= base_price
        price_ax.annotate(
            f"{forecast_prices[index]:,.2f}\n{item['forecast_return_pct']:+.2f}%",
            (index, forecast_prices[index]),
            xytext=(0, 9 if above else -10), textcoords="offset points",
            ha="center", va="bottom" if above else "top",
            fontsize=8.4, color=COLORS["text"], fontweight="bold",
        )
    price_padding = max(15.0, float((interval_upper.max() - interval_lower.min()) * 0.09))
    price_ax.set_ylim(float(interval_lower.min() - price_padding), float(interval_upper.max() + price_padding))
    price_ax.set_xticks(
        x,
        [f"{label}\n{item['target_date'][5:]}" for label, item in zip(labels, horizons)],
    )
    price_ax.set_ylabel("USD/COP proyectado", color=COLORS["muted"], fontsize=10)
    price_ax.set_title(
        "Forward price ladder — punto Ridge + intervalo residual 80% · × = signo discrepa del clasificador",
        loc="left", color=COLORS["text"], fontsize=11, fontweight="bold", pad=10,
    )
    price_ax.legend(loc="upper right", frameon=False, labelcolor=COLORS["muted"], fontsize=8.5)

    evidence_colors = [
        COLORS["selected"] if item["selected"]
        else COLORS["up"] if item["eligible_for_direction"]
        else COLORS["ineligible"]
        for item in horizons
    ]
    evidence_bars = evidence_ax.bar(x, scores * 100, color=evidence_colors, width=0.66, alpha=0.88)
    evidence_ax.axhline(50, color=COLORS["threshold"], linewidth=1.2, linestyle="--", alpha=0.8)
    evidence_ax.set_ylim(35, max(65, float(scores.max() * 100 + 5)))
    evidence_ax.set_xticks(x, labels)
    evidence_ax.set_ylabel("Score causal contraído", color=COLORS["muted"], fontsize=10)
    evidence_ax.set_title(
        "Evidencia disponible antes del origen (DA + Balanced DA + recall mínimo)",
        loc="left", color=COLORS["text"], fontsize=11, fontweight="bold", pad=10,
    )
    for bar, score, count in zip(evidence_bars, scores, evidence_n):
        evidence_ax.text(
            bar.get_x() + bar.get_width() / 2,
            score * 100 + 0.8,
            f"{score:.1%} · n={count}",
            ha="center", va="bottom", fontsize=8.5, color=COLORS["text"],
        )

    decision = week["decision"]
    title_suffix = " · SEMANA PARCIAL" if week["origin_is_partial_week"] else ""
    figure.suptitle(
        f"USD/COP · Replay direccional causal · {week['iso_week']}{title_suffix}",
        x=0.07, y=0.985, ha="left", color=COLORS["text"], fontsize=17, fontweight="bold",
    )
    selected = ", ".join(f"H{value}" for value in decision["selected_horizons"]) or "ninguno"
    mode = "modelo congelado ≤2024" if week["year"] == 2025 else "reentrenamiento semanal con etiquetas maduras"
    figure.text(
        0.07, 0.942,
        f"Origen {week['origin_date']} · spot {week['base_price']:,.2f} · {mode}",
        color=COLORS["muted"], fontsize=10,
    )
    decision_color = (
        COLORS["up"] if decision["direction"] == "UP"
        else COLORS["down"] if decision["direction"] == "DOWN"
        else COLORS["muted"]
    )
    figure.text(
        0.07, 0.030,
        f"DECISIÓN SHADOW · GATE {'PASS' if decision['promotion_gate_passed'] else 'FAIL'}: "
        f"{_direction_label(decision['direction'])}  ·  horizontes: {selected}  ·  "
        f"confianza proxy: {decision['confidence_proxy']:.1%}",
        color=decision_color, fontsize=10.8, fontweight="bold",
    )
    figure.text(
        0.99, 0.008,
        "No autorizado para ejecución · labels purgados · macro PIT as-of",
        color=COLORS["muted"], fontsize=8.5, ha="right",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=135, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close(figure)
    os.replace(temporary, output_path)


def validate_published(config_path: Path) -> None:
    cfg = load_replay_config(config_path)
    paths = resolve_paths(ROOT, cfg)
    if not paths.index_file.exists():
        raise FileNotFoundError(paths.index_file)
    document = json.loads(paths.index_file.read_text(encoding="utf-8"))
    errors = validate_replay_document(document, ROOT, require_images=True)
    if not paths.report_workbook.exists():
        errors.append(f"workbook missing: {paths.report_workbook}")
    if errors:
        raise RuntimeError("Directional replay validation failed:\n- " + "\n- ".join(errors))
    print(
        f"[directional-replay] contract OK · {len(document['weeks'])} weeks · "
        f"{document['weeks'][0]['iso_week']}..{document['weeks'][-1]['iso_week']} · "
        f"hash={document['contract_hash']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--workbook-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if args.validate_only:
        validate_published(config_path)
        return

    cfg = load_replay_config(config_path)
    paths = resolve_paths(ROOT, cfg)
    if args.workbook_only:
        document = json.loads(paths.index_file.read_text(encoding="utf-8"))
        ledger = flatten_ledger(document)
        summary = summary_frame(document)
        export_directional_replay_workbook(paths.report_workbook, document, ledger, summary)
        validate_published(config_path)
        print(f"[directional-replay] workbook -> {paths.report_workbook.relative_to(ROOT)}")
        return
    if args.render_only:
        document = json.loads(paths.index_file.read_text(encoding="utf-8"))
        for week in document["weeks"]:
            output_path = ROOT / "usdcop-trading-dashboard/public/forecasting" / week["image_path"]
            render_week_image(week, output_path)
        validate_published(config_path)
        print(f"[directional-replay] rendered {len(document['weeks'])} images")
        return
    print("[directional-replay] building frame, frozen models and weekly expanding replay...")
    document, _ = build_directional_replay(ROOT, cfg)
    initial_errors = validate_replay_document(document, ROOT, require_images=False)
    if initial_errors:
        raise RuntimeError("Pre-publication contract failed:\n- " + "\n- ".join(initial_errors))

    if not args.skip_images:
        for index, week in enumerate(document["weeks"], start=1):
            output_path = ROOT / "usdcop-trading-dashboard/public/forecasting" / week["image_path"]
            render_week_image(week, output_path)
            if index == 1 or index % 10 == 0 or index == len(document["weeks"]):
                print(f"[directional-replay] images {index}/{len(document['weeks'])}: {week['iso_week']}")

    ledger = flatten_ledger(document)
    summary = summary_frame(document)
    _atomic_json(paths.index_file, document)
    _atomic_csv(paths.ledger_file, ledger)
    _atomic_csv(paths.report_ledger, ledger)
    _atomic_csv(paths.report_summary, summary)
    export_directional_replay_workbook(paths.report_workbook, document, ledger, summary)

    errors = validate_replay_document(document, ROOT, require_images=not args.skip_images)
    if errors:
        raise RuntimeError("Published contract failed:\n- " + "\n- ".join(errors))
    print(summary.to_string(index=False))
    print(
        f"[directional-replay] published {len(document['weeks'])} weeks, "
        f"{len(ledger)} horizon rows -> {paths.index_file.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
