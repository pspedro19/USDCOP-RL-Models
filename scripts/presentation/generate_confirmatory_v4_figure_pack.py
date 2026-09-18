"""Generate the six thesis figures from the frozen v4 artifacts only."""

# ruff: noqa: RUF001 -- typography in Spanish figure labels is intentional.

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compound(values: np.ndarray) -> np.ndarray:
    return 100_000.0 * np.r_[1.0, np.cumprod(1.0 + values)]


def _drawdown(equity: np.ndarray) -> np.ndarray:
    return equity / np.maximum.accumulate(equity) - 1.0


def _rolling_sharpe(values: np.ndarray, window: int = 60) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        sd = np.std(sample, ddof=1)
        if sd > 0:
            out[i] = np.mean(sample) / sd * np.sqrt(221.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--forward-specs", type=Path, required=True)
    parser.add_argument("--forward-actions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    rows = holdout["rows"]
    dates = np.asarray(rows[0]["metrics"]["dates"], dtype="datetime64[D]")
    by_config: dict[str, list[dict]] = {"ppo_regime": [], "ppo_backbone": []}
    for row in rows:
        by_config[row["config"]].append(row["metrics"])
    median_returns = {
        config: np.median(np.asarray([m["daily_returns"] for m in metrics]), axis=0)
        for config, metrics in by_config.items()
    }
    labels = {"ppo_regime": "PPO régimen (mediana semillas)",
              "ppo_backbone": "PPO backbone (mediana semillas)"}
    colors = {"ppo_regime": "#0072B2", "ppo_backbone": "#D55E00"}
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    def save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=260, bbox_inches="tight")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # 1. Capital curve.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.plot(dates, np.full(len(dates), 100_000.0), "#555555", label="Always flat", linewidth=2)
    for config, values in median_returns.items():
        ax.plot(dates, _compound(values)[1:], color=colors[config], label=labels[config], linewidth=2)
    ax.set_title("Capital acumulado — hold-out 2024–2025 (420 sesiones)")
    ax.set_ylabel("Capital (USD)")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "01_capital_holdout_v4")

    # 2. Drawdown.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for config, values in median_returns.items():
        ax.plot(dates, 100 * _drawdown(_compound(values))[1:], color=colors[config],
                label=labels[config], linewidth=2)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Drawdown comparativo — hold-out 2024–2025")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "02_drawdown_holdout_v4")

    # 3. Rolling Sharpe.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for config, values in median_returns.items():
        ax.plot(dates, _rolling_sharpe(values), color=colors[config], label=labels[config], linewidth=2)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Sharpe móvil (60 sesiones) — hold-out 2024–2025")
    ax.set_ylabel("Sharpe anualizado (√221)")
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "03_sharpe_movil_holdout_v4")

    # 4. Stress on realized gross/cost paths; multipliers, not invented pips.
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for config, metrics in by_config.items():
        gross = np.median(np.asarray([m["daily_gross_returns"] for m in metrics]), axis=0)
        cost = np.median(np.asarray([m["daily_costs"] for m in metrics]), axis=0)
        values = []
        for multiplier in (1.0, 2.0, 3.0):
            net = gross - multiplier * cost
            sd = np.std(net, ddof=1)
            values.append(np.mean(net) / sd * np.sqrt(221.0) if sd > 0 else 0.0)
        ax.plot([1, 2, 3], values, marker="o", color=colors[config], label=labels[config])
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Sensibilidad a costes realizados — hold-out 2024–2025")
    ax.set_xlabel("Multiplicador del contrato de costes")
    ax.set_ylabel("Sharpe anualizado (√221)")
    ax.set_xticks([1, 2, 3], ["×1", "×2", "×3"])
    ax.legend()
    ax.grid(alpha=0.2)
    save(fig, "04_cost_stress_holdout_v4")

    # 5. Forward actions by inferred posterior regime.
    bundle = pickle.loads(args.forward_specs.read_bytes())
    actions = json.loads(args.forward_actions.read_text(encoding="utf-8"))["median"]["ppo_regime"]
    counts = {f"R{i + 1}": {"SHORT": 0, "FLAT": 0, "LONG": 0} for i in range(4)}
    for session in bundle["sessions"]:
        date = session.date.isoformat()
        regime = int(np.argmax(np.asarray(session.context[-4:], dtype=float)))
        key = f"R{regime + 1}"
        for weight in actions[date]:
            counts[key]["SHORT" if weight < 0 else "LONG" if weight > 0 else "FLAT"] += 1
    keys = list(counts)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bottom = np.zeros(len(keys))
    for action, color in (("SHORT", "#D55E00"), ("FLAT", "#999999"), ("LONG", "#009E73")):
        values = np.asarray([counts[key][action] for key in keys], dtype=float)
        totals = np.asarray([sum(counts[key].values()) for key in keys], dtype=float)
        ax.bar(keys, 100 * values / totals, bottom=bottom, label=action, color=color)
        bottom += 100 * values / totals
    ax.set_title("Acciones PPO régimen por estado posterior — forward 2026 parcial")
    ax.set_ylabel("Frecuencia (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save(fig, "05_acciones_regimen_forward_v4")

    # 6. Seed variability.
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    values = [[float(m["sharpe"]) for m in by_config[config]] for config in ("ppo_regime", "ppo_backbone")]
    ax.boxplot(values, labels=["PPO régimen", "PPO backbone"], showmeans=True)
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_title("Variabilidad entre cinco semillas — hold-out 2024–2025")
    ax.set_ylabel("Sharpe anualizado (√221)")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "06_semillas_holdout_v4")

    manifest = {
        "schema_version": "confirmatory-v4-figure-pack-v1",
        "scope": "holdout_2024_2025_and_forward_2026_partial",
        "inputs_sha256": {str(path): _sha(path) for path in (args.holdout, args.forward_specs, args.forward_actions)},
        "figures": sorted(path.name for path in out.glob("*.png")),
        "no_synthetic_returns": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "figures": len(manifest["figures"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
