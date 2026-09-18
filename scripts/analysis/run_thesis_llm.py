#!/usr/bin/env python
"""Run the preregistered thesis LLM arm over sealed decision contexts.

The input is deliberately a transport format, not an implicit data loader.  Each JSONL
row must contain a causal, already-frozen context (session_date, bar, previous_weight,
system_prompt and user_prompt).  Without ``--execute`` this command only validates the
file; with it, one explicitly selected provider is called and decisions are appended to
the ledger by :class:`ThesisLLMRunner`.

This command never loads a dotenv file unless the operator passes ``--load-dotenv``;
it never prints credentials or raw responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.llm_trader import ThesisLLMRunner, provider_from_environment  # noqa: E402

LLM_CONFIG = ROOT / "config" / "research" / "llm_thesis.yaml"
DEFAULT_PORTABLE = ROOT / "data" / "thesis" / "research_data_portable_v2.pkl"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_contexts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: context must be an object")
            required = {"session_date", "bar", "previous_weight", "system_prompt", "user_prompt"}
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(f"line {line_no}: missing fields {missing}")
            try:
                session_date = str(row["session_date"])
                date.fromisoformat(session_date)
                bar = int(row["bar"])
                previous_weight = float(row["previous_weight"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line_no}: invalid date/bar/previous_weight") from exc
            if not 0 <= bar <= 58:
                raise ValueError(f"line {line_no}: bar must be in [0, 58]")
            if not -1.0 <= previous_weight <= 1.0:
                raise ValueError(f"line {line_no}: previous_weight outside [-1, 1]")
            if not isinstance(row["system_prompt"], str) or not row["system_prompt"].strip():
                raise ValueError(f"line {line_no}: system_prompt is empty")
            if not isinstance(row["user_prompt"], str) or not row["user_prompt"].strip():
                raise ValueError(f"line {line_no}: user_prompt is empty")
            key = (session_date, bar)
            if key in seen:
                raise ValueError(f"line {line_no}: duplicate decision context {session_date} bar {bar}")
            seen.add(key)
            rows.append({**row, "session_date": session_date, "bar": bar,
                         "previous_weight": previous_weight})
    if not rows:
        raise ValueError("input JSONL contains no contexts")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--provider", choices=("deepseek", "azure_openai"), default="deepseek")
    parser.add_argument("--ledger", type=Path, default=Path("data/thesis/llm/decisions.jsonl"))
    parser.add_argument("--model-id", required=True,
                        help="frozen model/deployment identifier recorded in every ledger row")
    parser.add_argument("--execute", action="store_true",
                        help="perform network calls; omitted means validation-only")
    parser.add_argument("--portable", type=Path, default=DEFAULT_PORTABLE,
                        help="portable dataset whose hash must match every executable context")
    parser.add_argument("--allow-retrospective", action="store_true",
                        help="allow contexts explicitly labelled retrospective (never confirmatory)")
    parser.add_argument("--load-dotenv", action="store_true",
                        help="operator opt-in: load workspace .env without printing secrets")
    parser.add_argument("--dotenv-path", type=Path, default=ROOT / ".env",
                        help="operator-supplied dotenv path (used only with --load-dotenv)")
    parser.add_argument("--prereg-path", type=Path,
                        default=ROOT / ".claude" / "specs" / "planes" / "06-PRE-REGISTRATION-v3.md",
                        help="preregistration path; tests may supply an isolated unsigned fixture")
    parser.add_argument("--limit", type=int, default=0,
                        help="optional validation/execution limit (0 means all)")
    parser.add_argument("--resume", action="store_true",
                        help=("reanuda: salta los contextos que ya tienen decision sellada en "
                              "el ledger y arrastra el peso previo desde la ultima fila de cada "
                              "sesion. Sin esto, una corrida de 13.334 llamadas que se corta "
                              "no se puede continuar: el ledger rechaza decision_id repetidos "
                              "y reanudar desde cero volveria a pagar la API entera."))
    args = parser.parse_args()
    if args.load_dotenv:
        from dotenv import load_dotenv
        try:
            load_dotenv(args.dotenv_path, override=False)
        except OSError as exc:
            print(json.dumps({"execution_blocked": "dotenv_unreadable",
                              "error": type(exc).__name__,
                              "network_called": False,
                              "ledger_written": False}), file=sys.stderr)
            return 2
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    try:
        contexts = _read_contexts(args.input_jsonl)
    except (OSError, ValueError) as exc:
        print(f"context_validation_error: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        contexts = contexts[:args.limit]
    if not args.execute:
        print(json.dumps({"validated_contexts": len(contexts), "network_called": False,
                          "ledger_written": False}, sort_keys=True))
        return 0
    if not args.portable.is_file():
        print(f"execution_blocked: missing portable dataset {args.portable}", file=sys.stderr)
        return 2
    expected_hash = _sha256(args.portable.resolve())
    context_hashes = {row.get("dataset_sha256") for row in contexts}
    if context_hashes != {expected_hash}:
        print("execution_blocked: context dataset_sha256 does not match portable v2", file=sys.stderr)
        return 2
    if any(bool(row.get("retrospective")) for row in contexts) and not args.allow_retrospective:
        print("execution_blocked: retrospective contexts require --allow-retrospective", file=sys.stderr)
        return 2
    prereg = args.prereg_path
    prereg_text = prereg.read_text(encoding="utf-8") if prereg.is_file() else ""
    if "operator_signature: SIGNED" not in prereg_text:
        print("execution_blocked: pre-registration is not SIGNED", file=sys.stderr)
        return 2
    provider = provider_from_environment(args.provider)
    runner = ThesisLLMRunner(provider, args.provider, args.model_id, args.ledger)
    config = yaml.safe_load(LLM_CONFIG.read_text(encoding="utf-8"))
    protocol = config["protocol"]
    previous_by_session: dict[str, float] = {}
    sealed: set[str] = set()
    if args.resume and args.ledger.is_file():
        last_bar: dict[str, int] = {}
        with args.ledger.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sealed.add(row["decision_id"])
                session, bar = row["session_date"], int(row["bar"])
                # El peso que arrastra la reanudacion es el de la barra MAS ALTA ya sellada de
                # esa sesion, no el de la ultima linea del fichero: el ledger es append-only
                # pero nada garantiza que este ordenado.
                if bar >= last_bar.get(session, -1):
                    last_bar[session] = bar
                    previous_by_session[session] = float(row["weight"])
        skipped = sum(1 for row in contexts
                      if f"{row['session_date']}::llm::{row['bar']}" in sealed)
        print(json.dumps({"resumed_from_ledger": len(sealed), "contexts_skipped": skipped,
                          "contexts_pending": len(contexts) - skipped}, sort_keys=True))
    processed = 0
    for row in contexts:
        if f"{row['session_date']}::llm::{row['bar']}" in sealed:
            continue
        # The first row supplies the sealed initial exposure. Subsequent bars use the
        # actually validated prior decision, so a timeout/invalid JSON is handled by
        # the runner's retain-previous policy instead of resetting to zero.
        previous_weight = previous_by_session.setdefault(row["session_date"], row["previous_weight"])
        decision = runner.decide(
            session_date=row["session_date"], bar=row["bar"],
            system_prompt=row["system_prompt"], user_prompt=row["user_prompt"],
            previous_weight=previous_weight,
            dataset_block=row.get("dataset_block"),
            max_tokens=int(protocol["max_tokens"]),
            temperature=float(protocol["temperature"]),
            top_p=float(protocol["top_p"]),
            max_retries_invalid_json=int(protocol["max_retries_invalid_json"]),
            dataset_sha256=row.get("dataset_sha256"),
            retrospective=bool(row.get("retrospective", False)),
        )
        previous_by_session[row["session_date"]] = decision.weight
        processed += 1
    print(json.dumps({"processed_contexts": processed, "provider": args.provider,
                      "ledger": str(args.ledger), "network_called": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
