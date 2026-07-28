#!/usr/bin/env python
"""Strangler migration gate for USD/COP — BL-31 (FABRIC §29 + §30).

Answers one question per layer: *may it switch from the artisanal path to the candidate
path?* Fail-closed — with no candidate generator (BL-28) and no execution-readiness
attestation (BL-30), every layer is red, which is the honest state of the migration.

Usage
-----
    # per-layer parity table (human + machine evidence)
    python scripts/validation/check_strangler_parity.py status
    python scripts/validation/check_strangler_parity.py status --write-evidence

    # record one comparison of a legacy artifact against its candidate twin
    python scripts/validation/check_strangler_parity.py observe \
        --layer signal --artifact-id week=2026-W30 \
        --legacy path/to/legacy.json --candidate path/to/candidate.json

    # ask whether a layer may advance (exit 0 = yes, 1 = blocked, 2 = error)
    python scripts/validation/check_strangler_parity.py gate --layer signal

    # declare the switch (or the rollback) — an event, never an edit
    python scripts/validation/check_strangler_parity.py transition \
        --layer signal --state MIGRATED --reason "parity green 21d, sensor->Asset swapped"

Exit codes: 0 ok / 1 blocked or red / 2 usage or contract error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.strangler import (  # noqa: E402
    LayerState,
    LayerTransition,
    ParityLedger,
    ParityVerdict,
    StranglerContractError,
    evaluate_advance,
    load_plan,
    observe_parity,
    summarize,
)
from src.strangler.contracts import parse_layer  # noqa: E402

DEFAULT_LEDGER = PROJECT_ROOT / ".claude" / "evidence" / "strangler_cop" / "parity_ledger.jsonl"
DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / ".claude" / "evidence" / "strangler_cop"

_EVIDENCE_FRONTMATTER = """---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: {today}
supersedes: []
code_anchors:
  - src/strangler/gates.py
  - config/migration/strangler_usdcop.yaml
  - airflow/dags/forecast_h5_l7_multiday_executor.py
---
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _display_path(path: str | Path) -> str:
    """Repo-relative when possible (evidence must be readable from any clone)."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _load(args) -> tuple:
    plan = load_plan(args.plan)
    ledger = ParityLedger(args.ledger)
    return plan, ledger


def _render_table(rows: list[dict]) -> str:
    header = (
        "| layer | state | generator | obs | green | days/req | may advance | blockers |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        blockers = "; ".join(row["blockers"]) if row["blockers"] else "—"
        lines.append(
            "| {layer} | {state} | {gen} | {obs} | {green} | {days:.2f}/{req} | {adv} | {bl} |".format(
                layer=row["layer"],
                state=row["state"],
                gen=row["candidate_generator"] or "**null (BL-28)**",
                obs=row["observations"],
                green=row["green_streak"],
                days=row["green_days"],
                req=row["min_parallel_days"],
                adv="YES" if row["may_advance"] else "NO",
                bl=blockers.replace("|", "/"),
            )
        )
    return header + "\n".join(lines) + "\n"


def cmd_status(args) -> int:
    plan, ledger = _load(args)
    payload = summarize(plan, ledger, now=_now())
    rows = payload["layers"]
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(f"Strangler migration — {plan.asset} (plan {plan.version}, {plan.contract_id})")
        print(f"ledger: {args.ledger}")
        print(_render_table(rows))
        pending = payload["acceptance_pending"]
        print(f"§30 acceptance criteria not PASS: {len(pending)}/15")
        readiness = payload["execution_readiness"]
        print(
            "execution readiness (BL-30): "
            + ("NOT ATTESTED" if not readiness["attested"] else f"missing={readiness['missing']}")
        )
        print(f"A/B cohort untouched (§29.5): {', '.join(payload['ab_cohort_untouched'])}")

    if args.write_evidence:
        out_dir = Path(args.evidence_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "parity_table.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        md = _EVIDENCE_FRONTMATTER.format(today=_now().date().isoformat())
        md += (
            f"\n# Tabla de paridad por capa — {plan.asset} (BL-31)\n\n"
            f"Generada por `scripts/validation/check_strangler_parity.py status "
            f"--write-evidence` el {payload['generated_at']}.\n"
            f"Plan: `config/migration/strangler_usdcop.yaml` v{plan.version}. "
            f"Ledger: `{_display_path(args.ledger)}`.\n\n"
            + _render_table(rows)
            + "\n## §30 — criterios de aceptación pendientes\n\n"
            + "\n".join(
                f"- #{c['id']} [{c['status']}] — owner {c['owner']}"
                for c in payload["acceptance_pending"]
            )
            + "\n\n## Readiness de ejecución (BL-30)\n\n"
            + (
                "- NO ATESTIGUADA — el servicio de ejecución externo es entregable de BL-30.\n"
                if not payload["execution_readiness"]["attested"]
                else f"- faltantes: {payload['execution_readiness']['missing']}\n"
            )
        )
        (out_dir / "parity_table.md").write_text(md, encoding="utf-8")
        print(f"evidence written to {out_dir}")

    return 0 if all(r["may_advance"] for r in rows) else 1


def cmd_observe(args) -> int:
    plan, ledger = _load(args)
    layer = parse_layer(args.layer)
    layer_plan = plan.layer_plan(layer)
    observation = observe_parity(
        layer=layer,
        artifact_id=args.artifact_id,
        legacy_path=args.legacy,
        candidate_path=args.candidate,
        required_hash_kind=layer_plan.required_hash_kind,
        observed_at=_now(),
    )
    ledger.append(observation)
    print(
        f"{observation.verdict} {layer}/{observation.artifact_id} "
        f"[{observation.hash_kind}] {observation.note}"
    )
    return 0 if observation.verdict is ParityVerdict.MATCH else 1


def cmd_gate(args) -> int:
    plan, ledger = _load(args)
    decision = evaluate_advance(plan, ledger, args.layer, now=_now())
    if decision.allowed:
        print(f"ALLOWED — layer '{decision.layer}' may switch to the candidate path")
        return 0
    print(f"BLOCKED — layer '{decision.layer}' may NOT advance:")
    for reason in decision.reasons:
        print(f"  - {reason}")
    return 1


def cmd_transition(args) -> int:
    plan, ledger = _load(args)
    layer = parse_layer(args.layer)
    state = LayerState(args.state)
    if state is LayerState.MIGRATED:
        decision = evaluate_advance(plan, ledger, layer, now=_now())
        if not decision.allowed and not args.force_rollback_only:
            print(f"REFUSED — the gate blocks '{layer}':")
            for reason in decision.reasons:
                print(f"  - {reason}")
            return 1
    ledger.append(
        LayerTransition(layer=layer, state=state, at=_now(), reason=args.reason)
    )
    print(f"recorded: {layer} -> {state}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", default=None, help="path to the strangler plan YAML")
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER), help="append-only parity ledger")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="per-layer parity table")
    p_status.add_argument("--json", action="store_true")
    p_status.add_argument("--write-evidence", action="store_true")
    p_status.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    p_status.set_defaults(func=cmd_status)

    p_obs = sub.add_parser("observe", help="record one legacy-vs-candidate comparison")
    p_obs.add_argument("--layer", required=True)
    p_obs.add_argument("--artifact-id", required=True)
    p_obs.add_argument("--legacy", required=True)
    p_obs.add_argument("--candidate", required=True)
    p_obs.set_defaults(func=cmd_observe)

    p_gate = sub.add_parser("gate", help="may this layer advance?")
    p_gate.add_argument("--layer", required=True)
    p_gate.set_defaults(func=cmd_gate)

    p_tr = sub.add_parser("transition", help="record a layer state change (event, not edit)")
    p_tr.add_argument("--layer", required=True)
    p_tr.add_argument("--state", required=True, choices=[str(s) for s in LayerState])
    p_tr.add_argument("--reason", required=True)
    p_tr.add_argument(
        "--force-rollback-only",
        action="store_true",
        help="allow recording a non-MIGRATED state without consulting the gate",
    )
    p_tr.set_defaults(func=cmd_transition)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StranglerContractError as exc:
        print(f"CONTRACT ERROR: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"NOT FOUND: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
