"""Build a source-bound results table for the v4 thesis chapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repo.resolve()
    stable = root / "outputs/thesis-repair/confirmatory_v4_stable"
    holdout = _read(stable / "ppo_holdout_2024_2025_stable.json")
    forward = _read(stable / "ppo_forward_2026_partial.json")
    baselines = _read(stable / "forward_baselines_2026.json")

    rows: list[dict] = []
    for config in ("ppo_regime", "ppo_backbone"):
        metrics = [row["metrics"] for row in holdout["rows"] if row["config"] == config]
        rows.append({
            "scope": "holdout_2024_2025",
            "status": "post_freeze_diagnostic_confirmatory_protocol",
            "model": config,
            "n_sessions": metrics[0]["n_sessions"],
            "return_mean": sum(m["total_return"] for m in metrics) / len(metrics),
            "return_median": sorted(m["total_return"] for m in metrics)[len(metrics) // 2],
            "sharpe_mean": sum(m["sharpe"] for m in metrics) / len(metrics),
            "positive_seeds": sum(m["total_return"] > 0 for m in metrics),
        })

    for config, summary in forward["aggregate"].items():
        rows.append({
            "scope": "forward_2026_partial",
            "status": "exploratory_post_freeze",
            "model": config,
            "n_sessions": forward["n_sessions"],
            "return_mean": summary["mean_total_return"],
            "return_median": summary["median_total_return"],
            "sharpe_mean": summary["mean_sharpe"],
            "positive_seeds": summary["positive_return_seeds"],
        })

    for row in baselines["rows"]:
        metrics = row["metrics"]
        rows.append({
            "scope": "forward_2026_partial",
            "status": "exploratory_post_freeze_baseline",
            "model": row["policy"],
            "n_sessions": baselines["n_sessions"],
            "return_mean": metrics["total_return"],
            "return_median": metrics["total_return"],
            "sharpe_mean": metrics["sharpe"],
            "positive_seeds": None,
        })

    diagnostic_files = {
        "llm_deepseek": stable.parent / "llm_selection_deepseek_diagnostic_settlement.json",
        "llm_azure": stable.parent / "llm_selection_azure_diagnostic_settlement.json",
        "hybrid_deepseek": stable.parent / "hybrid_deepseek_selection_diagnostic.json",
        "hybrid_azure": stable.parent / "hybrid_azure_selection_diagnostic.json",
    }
    for model, path in diagnostic_files.items():
        if path.is_file():
            item = _read(path)
            rows.append({
                "scope": "selection_2023",
                "status": "retrospective_diagnostic",
                "model": model,
                "n_sessions": item.get("n_sessions_settled", item.get("n_sessions_common")),
                "return_mean": item.get("compounded_return", item.get("hybrid", {}).get("compounded_return")),
                "return_median": None,
                "sharpe_mean": item.get("sharpe_annualized_sqrt221", item.get("hybrid", {}).get("sharpe_annualized_sqrt221")),
                "positive_seeds": None,
            })

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "thesis-results-table-v4",
        "scope_note": "All rows are bound to their stated scope; retrospective LLM rows are not confirmatory.",
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = [
        "# Tabla consolidada de resultados v4",
        "",
        "Las filas no son intercambiables: `status` y `scope` delimitan la inferencia.",
        "",
        "| Alcance | Estado | Modelo | N | Retorno medio/total | Retorno mediano | Sharpe | Semillas positivas |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        return_mean = "—" if row["return_mean"] is None else f"{row['return_mean'] * 100:.2f}%"
        return_median = "—" if row["return_median"] is None else f"{row['return_median'] * 100:.2f}%"
        sharpe = "—" if row["sharpe_mean"] is None else f"{row['sharpe_mean']:.2f}"
        positive = "—" if row["positive_seeds"] is None else str(row["positive_seeds"])
        markdown.append(f"| {row['scope']} | {row['status']} | {row['model']} | {row['n_sessions']} | {return_mean} | {return_median} | {sharpe} | {positive} |")
    (output.with_suffix(".md")).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
