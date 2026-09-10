"""Job 1 — emit and seal one pre-session decision.

Runs before the open. Reads the corpus, calls the model, appends to the decision
ledger, exits. It has no access to price data and no code path that could reach
an outcome, which is enforced by the simple fact that it never imports one.

    python -m llmfwd.decide --session-date 2026-08-26

Idempotent by construction: a second run for the same session hits the ledger's
duplicate-key guard and refuses.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .canonical import sha256_text
from .corpus import (
    RawDoc,
    fetch_rss,
    filter_by_cutoff,
    session_cutoff_utc,
    store_and_reference,
)
from .ledger import Ledger, LedgerError
from .llm import LlmClient
from .prompt import build_system_prompt, build_user_prompt, prompt_hash
from .schema import DecisionRecord, utc_now_iso

from .paths import CORPUS_STORE, DECISIONS_PATH, PREREG_PATH  # rutas centralizadas


def load_preregistration(path: Path) -> tuple[dict, str]:
    """Load the frozen spec and hash its raw bytes.

    Hashing the file as read, not the parsed dict, is deliberate: it catches a
    comment or a reordering too. The spec is a commitment document, so any edit
    at all should be detectable.
    """
    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw), sha256_text(raw)


def arm_spec(spec: dict, arm_id: str) -> dict:
    """Resuelve un brazo del pre-registro por su `arm_id`.

    El arnes venia con un `arm_id` unico en la raiz. Aqui el mismo pre-registro gobierna
    CUATRO brazos —LLM, dos variantes de RL congelado y el baseline— porque la comparacion
    solo vale si los cuatro comparten cutoff, contrato de costos y motor de liquidacion.
    Tener un fichero por brazo garantizaria que en un mes discrepen en algo.
    """
    for arm in spec.get("arms", []):
        if arm["arm_id"] == arm_id:
            return arm
    known = [a["arm_id"] for a in spec.get("arms", [])]
    raise KeyError(f"arm_id {arm_id!r} no esta en el pre-registro; declarados: {known}")


def gather_documents(spec: dict, cutoff: datetime) -> list[RawDoc]:
    """Pull every configured source and apply the cutoff once, centrally."""
    collected: list[RawDoc] = []
    for source in spec["corpus"]["sources"]:
        if not source.get("enabled", True):
            continue
        try:
            collected.extend(fetch_rss(source["url"]))
        except Exception as exc:  # a dead feed must not abort the session
            print(f"  [warn] source {source['name']} failed: {exc}", file=sys.stderr)

    return filter_by_cutoff(
        collected, cutoff, lookback_days=spec["corpus"]["lookback_days"]
    )


def run(
    session_date: str,
    dry_run: bool = False,
    now_override: datetime | None = None,
    arm_id: str = "llm_direct_fwd_v1",
) -> int:
    """Emit one decision.

    Args:
        now_override: injectable clock. Production never passes it; tests and
            replays do. Without it the seal check is untestable, and an
            untestable guarantee is not a guarantee.
    """
    spec, prereg_hash = load_preregistration(PREREG_PATH)
    arm = arm_spec(spec, arm_id)
    if arm["kind"] != "llm":
        raise ValueError(
            f"{arm_id} es de tipo {arm['kind']!r}; este job sella el brazo LLM. "
            "Los brazos RL los sella `arms/ppo_arm.py`, que corre DESPUES de la barra 0."
        )

    cutoff = session_cutoff_utc(session_date, spec["session"]["cutoff_hour_cot"])
    open_utc = session_cutoff_utc(session_date, spec["session"]["open_hour_cot"])
    now = now_override or datetime.now(timezone.utc)

    # The seal condition. A late run is recorded and marked, never dropped and
    # never quietly kept: an excluded-but-visible record is auditable, a missing
    # one looks like the day never happened.
    sealed_before_open = now < open_utc
    if not sealed_before_open:
        print(
            f"  [warn] running at {now.isoformat()} but the session opened at "
            f"{open_utc.isoformat()}. Record will be flagged and excluded.",
            file=sys.stderr,
        )

    docs = gather_documents(spec, cutoff)
    references = store_and_reference(docs, CORPUS_STORE)
    print(f"  {len(references)} document(s) inside the cutoff window")

    system_prompt = build_system_prompt()
    bodies = {ref.doc_id: doc.text for ref, doc in zip(references, docs)}
    user_prompt = build_user_prompt(session_date, cutoff.isoformat(), references, bodies)

    if dry_run:
        print("\n--- SYSTEM ---\n" + system_prompt)
        print("--- USER ---\n" + user_prompt)
        return 0

    # The client is built even when we will not call it, so provider and model
    # are recorded identically on abstention days. A gap in those fields would
    # make the abstained rows look like a different experiment during analysis.
    client = LlmClient(
        model=spec["model"].get("name"),
        temperature=spec["model"]["temperature"],
        seed=spec["model"]["seed"],
    )

    # Empty corpus is an abstention, not a zero. They mean different things and
    # collapsing them would make "the model saw nothing" indistinguishable from
    # "the model read the news and had no view" in the final dataset.
    abstained = len(references) < spec["corpus"]["min_docs"]
    result = None if abstained else client.decide(system_prompt, user_prompt)

    record = DecisionRecord(
        seq=-1,  # assigned by the ledger on append
        decision_id=f"{session_date}::{arm_id}",
        session_date=session_date,
        emitted_at_utc=utc_now_iso(),
        cutoff_utc=cutoff.isoformat(timespec="seconds"),
        session_open_utc=open_utc.isoformat(timespec="seconds"),
        sealed_before_open=sealed_before_open,
        preregistration_sha256=prereg_hash,
        prompt_sha256=prompt_hash(),
        model=client.model,
        provider=client.provider,
        temperature=client.temperature,
        seed=client.seed,
        corpus=references,
        decision=result.decision if result else None,
        abstained=abstained,
        abstain_reason="corpus below min_docs" if abstained else None,
        usage=result.usage if result else None,
    )

    ledger = Ledger(DECISIONS_PATH, key_field="decision_id")
    try:
        written = ledger.append(record)
    except LedgerError as exc:
        print(f"  [refused] {exc}", file=sys.stderr)
        return 1

    verdict = "ABSTAINED" if abstained else f"score={record.decision.score:+.3f}"
    print(f"  sealed seq={written['seq']} {verdict} hash={written['record_hash'][:12]}...")
    return 0


def main() -> int:
    # Colombian press carries accents, and the whole point of --dry-run is to read the
    # prompt before the first real call. On a cp1252 console that print raises
    # UnicodeEncodeError and the review step is unusable exactly when it matters.
    # The prompt sent to the model was always fine; only this console path was not.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Emit one sealed pre-session decision.")
    parser.add_argument("--session-date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--arm",
        default="llm_direct_fwd_v1",
        help="arm_id declared in the preregistration; refuses non-LLM arms.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the prompt without calling the model or writing.",
    )
    args = parser.parse_args()
    return run(args.session_date, args.dry_run, arm_id=args.arm)


if __name__ == "__main__":
    raise SystemExit(main())
