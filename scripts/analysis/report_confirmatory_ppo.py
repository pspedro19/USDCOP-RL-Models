#!/usr/bin/env python
"""Create auditable statistics and figures from the frozen PPO hold-out report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.research.inference import dsr_with_inherited_trials
from src.research.reporting_v2 import hierarchical_sharpe, paired_difference, summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=115)
    parser.add_argument("--label", default="hold-out confirmatorio 2024-2025",
                        help="period label used in figure titles")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload["rows"]
    by_config = {
        config: [r for r in rows if r["config"] == config]
        for config in ("ppo_regime", "ppo_backbone")
    }
    report = {
        "schema_version": "confirmatory-ppo-report-v4",
        "source": str(args.input.resolve()),
        "portable_sha256": payload["portable_sha256"],
        "block": payload["partition"],
        "n_trials_used_for_dsr": args.n_trials,
        "trial_count_note": "must be reconciled with the registry before confirmatory publication",
        "models": {},
    }
    for config, config_rows in by_config.items():
        daily = np.asarray([r["metrics"]["daily_returns"] for r in config_rows], dtype=float)
        gross = np.asarray([r["metrics"]["daily_gross_returns"] for r in config_rows], dtype=float)
        costs = np.asarray([r["metrics"]["daily_costs"] for r in config_rows], dtype=float)
        # Use the median seed only for the descriptive capital curve; all seeds remain
        # in the model-level inference below and are never silently averaged away.
        med = int(np.argsort([r["metrics"]["total_return"] for r in config_rows])[len(config_rows) // 2])
        m = config_rows[med]["metrics"]
        report["models"][config] = {
            "seed_metrics": [
                {"seed": r["seed"], "total_return": r["metrics"]["total_return"],
                 "sharpe": r["metrics"]["sharpe"], "n_ops": r["metrics"]["n_ops"]}
                for r in config_rows
            ],
            "median_seed_summary": summarize(
                np.asarray(m["daily_gross_returns"]), np.asarray(m["daily_costs"]),
                min_round_trips=int(m["n_ops"]),
            ),
            "dsr_by_seed": [
                dsr_with_inherited_trials(r["metrics"]["daily_returns"], args.n_trials)
                for r in config_rows
            ],
            "daily_returns_by_seed": daily.tolist(),
            "daily_gross_by_seed": gross.tolist(),
            "daily_costs_by_seed": costs.tolist(),
        }
    regime = np.asarray(report["models"]["ppo_regime"]["daily_returns_by_seed"])
    backbone = np.asarray(report["models"]["ppo_backbone"]["daily_returns_by_seed"])
    regime_ops = min(r["metrics"]["n_ops"] for r in by_config["ppo_regime"])
    backbone_ops = min(r["metrics"]["n_ops"] for r in by_config["ppo_backbone"])
    report["paired_regime_vs_backbone"] = hierarchical_sharpe(regime, backbone)
    report["paired_regime_vs_flat"] = paired_difference(
        regime[0], np.zeros(regime.shape[1]), min_trades_a=regime_ops,
        min_trades_b=0, flat_reference=True, metric="mean_daily",
    )
    report["paired_backbone_vs_flat"] = paired_difference(
        backbone[0], np.zeros(backbone.shape[1]), min_trades_a=backbone_ops,
        min_trades_b=0, flat_reference=True, metric="mean_daily",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _figures(report, args.figures, args.label)
    print(json.dumps({k: report[k] for k in report if k.startswith("paired_")}, indent=2))
    return 0


def _figures(report: dict, directory: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    models = report["models"]
    colors = {"ppo_regime": "#1f77b4", "ppo_backbone": "#ff7f0e"}
    for _config, model in models.items():
        model["median_capital"] = (100_000 * np.r_[1, np.cumprod(
            1 + np.asarray(model["daily_returns_by_seed"][2], dtype=float))]).tolist()
    plt.figure(figsize=(12, 6))
    for config, model in models.items():
        plt.plot(model["median_capital"], label=config, color=colors[config])
    plt.axhline(100_000, color="black", linestyle="--", label="always_flat")
    plt.title(f"PPO v4 - capital {label}")
    plt.ylabel("Capital")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "01_capital_holdout_v4.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 6))
    for config, model in models.items():
        capital = np.asarray(model["median_capital"])
        dd = capital / np.maximum.accumulate(capital) - 1
        plt.plot(dd * 100, label=config, color=colors[config])
    plt.title(f"PPO v4 - drawdown {label}")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "02_drawdown_holdout_v4.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    values, labels = [], []
    for config, model in models.items():
        values.append([x["sharpe"] for x in model["seed_metrics"]])
        labels.append(config)
    plt.boxplot(values, labels=labels, showmeans=True)
    plt.axhline(0, color="black", linestyle="--")
    plt.title(f"PPO v4 - variabilidad de Sharpe ({label})")
    plt.ylabel("Sharpe anualizado (√221)")
    plt.tight_layout()
    plt.savefig(directory / "03_seed_variability_v4.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 5))
    multipliers = [1, 2, 3]
    for config, model in models.items():
        cost = np.asarray(model["daily_costs_by_seed"][2])
        gross = np.asarray(model["daily_gross_by_seed"][2])
        sharpes = []
        for k in multipliers:
            r = gross - k * cost
            sharpes.append(float(r.mean() / r.std(ddof=1) * np.sqrt(221)))
        plt.plot(multipliers, sharpes, marker="o", label=config, color=colors[config])
    plt.axhline(0, color="black", linestyle="--")
    plt.xticks(multipliers, ["x1", "x2", "x3"])
    plt.title(f"PPO v4 - sensibilidad a costos ({label})")
    plt.ylabel("Sharpe anualizado")
    plt.legend()
    plt.tight_layout()
    plt.savefig(directory / "04_cost_stress_v4.png", dpi=220)
    plt.close()


if __name__ == "__main__":
    raise SystemExit(main())
