"""Regression lock: the HYPOTHESIS-REGISTRY header must never lag its own body ledger.

Red-team finding #2 on BL-12 (commit e0a09aa): the USDCOP registry front-matter said
`n_trials_total: 109` while the body's running ledger had already reached
"Contabilidad final: 50 trials direccionales / 111 globales" (H1 LATAM TRANSPORT V1).
`scripts/analysis/profitability_evidence.py::trial_count` reads `n_trials_total` from
this front-matter to deflate the DSR, so a stale header UNDER-deflates the Sharpe —
a violation of quant-constitution §2 (every claim must use the updated trial count).

This test parses the registry the same way profitability_evidence does (YAML
front-matter delimited by `---`), extracts every "N globales" ledger mention from the
body (the file's own append-only accounting: "109 globales", "110 globales",
"111 globales", including line-wrapped forms), and fails if the header is below the
maximum count the body itself records.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / ".claude" / "specs" / "assets" / "usdcop" / "HYPOTHESIS-REGISTRY.md"


def _front_matter_and_body(text: str) -> tuple[dict, str]:
    """Split YAML front-matter from body, mirroring
    scripts/analysis/profitability_evidence.py::read_front_matter."""
    assert text.startswith("---"), f"{REGISTRY} has no YAML front-matter"
    end = text.index("\n---", 3)
    fm = yaml.safe_load(text[3:end]) or {}
    return fm, text[end:]


def _body_ledger_counts(body: str) -> list[int]:
    """Every 'N globales' mention in the body (\\s+ absorbs line wraps like '111\\nglobales')."""
    return [int(m) for m in re.findall(r"(\d+)\s+globales", body)]


def test_header_n_trials_total_not_below_body_ledger() -> None:
    text = REGISTRY.read_text(encoding="utf-8", errors="replace")
    fm, body = _front_matter_and_body(text)

    n_header = fm.get("n_trials_total")
    assert isinstance(n_header, int) and n_header >= 1, (
        f"{REGISTRY}: front-matter lacks a usable integer `n_trials_total` "
        f"(got {n_header!r}) — profitability_evidence.py would refuse to run."
    )

    counts = _body_ledger_counts(body)
    assert counts, (
        f"{REGISTRY}: body no longer mentions any 'N globales' ledger entry — "
        "if the accounting format changed, update this test's regex in the same commit."
    )

    ledger_max = max(counts)
    assert n_header >= ledger_max, (
        f"{REGISTRY}: front-matter declares n_trials_total={n_header} but the body's own "
        f"ledger reaches {ledger_max} globales (mentions found: {sorted(set(counts))}). "
        "A stale header UNDER-deflates the DSR (quant-constitution §2) because "
        "scripts/analysis/profitability_evidence.py reads this field. Reconcile the header "
        "to the ledger maximum — never the other way around (the body is append-only)."
    )


def test_final_accounting_matches_header() -> None:
    """If the body declares a 'Contabilidad final' the header must equal (not merely
    exceed by accident) the latest final count — the header IS the machine-readable
    mirror of the newest ledger line."""
    text = REGISTRY.read_text(encoding="utf-8", errors="replace")
    fm, body = _front_matter_and_body(text)

    finals = [
        int(m)
        for m in re.findall(
            r"Contabilidad\s+final:.*?(\d+)\s+globales", body, flags=re.DOTALL
        )
    ]
    if not finals:  # no explicit final accounting yet — the >= lock above still applies
        return

    latest_final = max(finals)
    n_header = fm.get("n_trials_total")
    assert n_header == latest_final, (
        f"{REGISTRY}: header n_trials_total={n_header} != latest 'Contabilidad final' "
        f"({latest_final} globales). The header must mirror the newest ledger entry exactly."
    )
