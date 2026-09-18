"""Generate figures from a settled hybrid diagnostic artifact only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def generate(path: Path, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    report = json.loads(path.read_text(encoding="utf-8"))
    sessions = report.get("sessions", [])
    if not sessions:
        raise ValueError("hybrid artifact has no settled sessions")
    returns = np.asarray([float(row["daily_return"]) for row in sessions])
    labels = [row["session_date"] for row in sessions]
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, equity, color="#6a1b9a", linewidth=1.4)
    ax.set_title("Híbrido PPO + LLM - curva de capital (diagnóstico)")
    ax.set_ylabel("Capital relativo")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    out = output_dir / "hybrid_curva_capital.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(range(len(drawdown)), drawdown, 0, color="#c62828", alpha=0.35)
    ax.plot(drawdown, color="#8e0000", linewidth=1.2)
    ax.set_title("Híbrido PPO + LLM - drawdown (diagnóstico)")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    out = output_dir / "hybrid_drawdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(out)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settlement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"figures": [str(p) for p in generate(args.settlement, args.output_dir)]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
