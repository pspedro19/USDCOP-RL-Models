"""Audit command — chain integrity plus the operational summary.

    python -m llmfwd.verify

Run it in CI on every commit. A ledger you verify only when you suspect a problem
is a ledger that tells you about the problem months late.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .ledger import Ledger

from .paths import DECISIONS_PATH, SETTLEMENTS_PATH  # rutas centralizadas


def main() -> int:
    decisions = Ledger(DECISIONS_PATH, "decision_id")
    settlements = Ledger(SETTLEMENTS_PATH, "decision_id")

    exit_code = 0
    for name, ledger in (("decisions", decisions), ("settlements", settlements)):
        ok, message = ledger.verify()
        print(f"{'OK  ' if ok else 'FAIL'} {name}: {message}")
        if not ok:
            exit_code = 1

    rows = list(decisions)
    if not rows:
        return exit_code

    # Operational summary. The counts that matter for the thesis are not the
    # scores but the denominators: how many sessions are actually usable.
    late = sum(1 for r in rows if not r["sealed_before_open"])
    abstained = sum(1 for r in rows if r["abstained"])
    # Count the property we want, not len minus exclusions: a record can be
    # both late and abstained, and subtracting both double-counts it.
    usable = sum(1 for r in rows if r["sealed_before_open"] and not r["abstained"])
    cost = sum((r["usage"] or {}).get("cost_usd", 0.0) for r in rows)
    docs = sum(len(r["corpus"]) for r in rows)

    print(
        f"\nsessions logged   {len(rows)}"
        f"\n  usable          {usable}"
        f"\n  abstained       {abstained}  (corpus below threshold)"
        f"\n  late / excluded {late}"
        f"\nsettled           {sum(1 for _ in settlements)}"
        f"\ndocs per session  {docs / len(rows):.1f} mean"
        f"\ninference cost    ${cost:.4f} total, ${cost / len(rows):.5f} per session"
    )

    prompts = Counter(r["prompt_sha256"][:12] for r in rows)
    fingerprints = Counter(
        (r["usage"] or {}).get("system_fingerprint") for r in rows if r["usage"]
    )
    if len(prompts) > 1:
        print(f"\n[!] {len(prompts)} distinct prompt hashes: {dict(prompts)}")
        print("    The treatment changed mid-run. These are not one arm.")
        exit_code = 1
    if len(fingerprints) > 1:
        print(f"\n[!] {len(fingerprints)} distinct system_fingerprints: {dict(fingerprints)}")
        print("    The serving backend changed. Note the boundary in the write-up.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
