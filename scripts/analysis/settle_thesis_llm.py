#!/usr/bin/env python
"""Settle the thesis LLM ledger with the canonical session accounting engine.

This is intentionally separate from inference: it consumes only sealed records and the
frozen portable dataset, requires exactly 59 decisions per session, and writes no zero
for missing/incomplete sessions.  The resulting JSON is descriptive until the signed
pre-registration authorizes a confirmatory block.
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

from src.research.dataset import PORTABLE, load_portable
from src.research.session_env import run_session


def _read(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_no}: ledger row must be an object")
            rows.append(value)
    return rows


def settle(ledger: Path, block: str, portable: Path = PORTABLE,
           *, strict_ledger: bool = False) -> dict:
    if strict_ledger:
        from scripts.validation.validate_thesis_llm_ledger import validate
        import hashlib
        digest = hashlib.sha256(portable.read_bytes()).hexdigest()
        validate(ledger, require_complete_sessions=False, expected_dataset_sha256=digest)
    data = load_portable(portable)
    by_date = {s.date.isoformat(): s for s in data.block(block)}
    rows = _read(ledger)
    grouped: dict[str, dict[int, dict]] = {}
    for row in rows:
        session = str(row.get("session_date", ""))
        bar = row.get("bar")
        if session not in by_date or not isinstance(bar, int) or not 0 <= bar <= 58:
            continue
        grouped.setdefault(session, {})[bar] = row

    results: list[dict] = []
    excluded: dict[str, str] = {}
    for session_date, session in by_date.items():
        bars = grouped.get(session_date, {})
        if set(bars) != set(range(59)):
            excluded[session_date] = "incomplete_59_decisions"
            continue
        weights = np.asarray([float(bars[i]["weight"]) for i in range(59)], dtype=float)
        if not np.isfinite(weights).all() or (np.abs(weights) > 1.0).any():
            excluded[session_date] = "invalid_weight"
            continue
        scored = run_session(session.close, weights, session.spread_pips, date=session.date)
        results.append({
            "session_date": session_date,
            "gross_return": float(scored.gross_return),
            "total_cost": float(scored.total_cost),
            "daily_return": float(scored.daily_return),
            "terminal_cost": float(scored.terminal_cost),
            "n_changes": int(scored.n_changes),
            "invalid_json_count": int(sum(not bool(bars[i].get("valid_json")) for i in range(59))),
        })
    daily = np.asarray([r["daily_return"] for r in results], dtype=float)
    if len(daily):
        equity = np.cumprod(1.0 + daily)
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        sharpe = float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(221)) if len(daily) > 1 and np.std(daily, ddof=1) > 0 else 0.0
        compounded = float(equity[-1] - 1.0)
        max_dd = float(np.min(drawdown))
    else:
        sharpe = compounded = max_dd = 0.0
    return {
        "contract": "CTR-RESEARCH-LLM-SETTLEMENT-001",
        "block": block,
        "n_sessions_settled": len(results),
        "n_sessions_excluded": len(excluded),
        "excluded": excluded,
        "compounded_return": compounded,
        "sharpe_annualized_sqrt221": sharpe,
        "max_drawdown": max_dd,
        "sessions": results,
        "confirmatory": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--block", choices=("development", "selection", "holdout"), required=True)
    parser.add_argument("--portable", type=Path,
                        help=("portable explicito. Si se omite, hay que dar --dataset-version: "
                              "el default historico se resolvia en import-time al portable v1 "
                              "y liquidaba un ledger v2 contra el dataset equivocado SIN AVISAR."))
    parser.add_argument("--dataset-version", choices=("v1", "v2"),
                        help="elige el portable congelado de esa version (alternativa a --portable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strict-ledger", action="store_true",
                        help="validate hashes, sampling contract and portable dataset binding")
    args = parser.parse_args()
    if (args.portable is None) == (args.dataset_version is None):
        parser.error("da exactamente uno: --portable o --dataset-version. Un default silencioso "
                     "es justo lo que hace falta evitar aqui.")
    portable = args.portable or (
        ROOT / "data" / "thesis" / ("research_data_portable_v2.pkl" if args.dataset_version == "v2"
                                    else "research_data_portable.pkl"))
    try:
        report = settle(args.ledger, args.block, portable,
                        strict_ledger=args.strict_ledger)
    except (OSError, ValueError, KeyError) as exc:
        print(f"llm_settlement_error: {exc}", file=sys.stderr)
        return 2
    import hashlib
    # Sin esto, dos liquidaciones del mismo ledger contra portables distintos producen JSON
    # indistinguibles. El hash es lo que ata el numero al dato.
    report["portable_path"] = str(portable)
    report["portable_sha256"] = hashlib.sha256(portable.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("block", "n_sessions_settled", "n_sessions_excluded", "confirmatory")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
