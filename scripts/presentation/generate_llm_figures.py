#!/usr/bin/env python
"""Generate thesis LLM figures from a settlement artifact only.

No strategy is rerun here.  This module is deliberately a pure presentation step so a
plot cannot silently use a different price series or cost contract than settlement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate(settlement_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    report = json.loads(settlement_path.read_text(encoding="utf-8"))
    sessions = report.get("sessions", [])
    if not sessions:
        raise ValueError("settlement has no settled sessions; no figures generated")
    returns = np.asarray([float(row["daily_return"]) for row in sessions], dtype=float)
    labels = [row["session_date"] for row in sessions]
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, equity, color="#1565c0", linewidth=1.4)
    ax.set_title("LLM — curva de capital (liquidación neta)")
    ax.set_ylabel("Capital relativo")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    path = output_dir / "llm_curva_capital.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(range(len(drawdown)), drawdown, 0, color="#c62828", alpha=0.35)
    ax.plot(drawdown, color="#8e0000", linewidth=1.2)
    ax.set_title("LLM — drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    path = output_dir / "llm_drawdown.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    window = min(60, len(returns))
    rolling = np.full(len(returns), np.nan)
    if window > 1:
        for i in range(window - 1, len(returns)):
            sample = returns[i - window + 1:i + 1]
            sd = float(np.std(sample, ddof=1))
            if sd > 0:
                rolling[i] = float(np.mean(sample) / sd * np.sqrt(221))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, rolling, color="#2e7d32", linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"LLM — Sharpe móvil ({window} sesiones)")
    ax.set_ylabel("Sharpe anualizado (√221)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    path = output_dir / "llm_sharpe_movil.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settlement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = generate(args.settlement, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"llm_figures_error: {exc}")
        return 2
    print(json.dumps({"figures": [str(path) for path in paths]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
