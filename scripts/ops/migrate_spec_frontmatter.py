"""One-shot migration: add typed YAML front matter to every `.claude/**/*.md`.

Contract: CTR-KNOWLEDGE-FRONTMATTER-001

Why this exists: `.claude/README.md` claimed every document carried a
`Contract · Version · Status` header. Only 7 of 97 did. Without machine-readable
metadata there is no way to ask "which specs describe code that no longer exists?"
or "which specs haven't been verified in 90 days?" — so specs rot silently.

Inference is deliberately conservative:
  * `kind` and `status` come from location + textual signals, and are meant to be
    reviewed by a human afterwards (status is the one field a machine cannot honestly
    infer).
  * `code_anchors` are extracted from backticked paths **that actually exist**. This is
    the load-bearing field: it ties a document to real files, so the gate can detect
    when a spec starts describing code that was deleted.

Usage:
    python scripts/ops/migrate_spec_frontmatter.py --dry-run
    python scripts/ops/migrate_spec_frontmatter.py --apply
"""
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/ops/<this> -> repo root
CLAUDE = ROOT / ".claude"
TODAY = date.today().isoformat()

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.S)
CONTRACT_RE = re.compile(r"\bContract(?:\s*ID)?[:\s]+`?(CTR-[A-Z0-9-]+)`?", re.I)
VERSION_RE = re.compile(r"\bVersion[:\s]+`?(\d+\.\d+\.\d+)`?", re.I)

# Backticked things that look like repo paths
PATH_RE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|ts|tsx|yaml|yml|sql|json|mjs|sh))`")

STALE_SIGNALS = ("pendiente", "aún inexistente", "aun inexistente", "todo:", "🆕 new", "status: draft")


def infer_kind(rel: Path, text: str) -> str:
    parts = rel.parts
    if parts and parts[0] == "rules":
        return "rule"
    if "archive" in parts:
        return "historical"
    if "audit" in parts:
        return "audit"
    if "adr" in parts or rel.name.startswith("ADR"):
        return "adr"
    lowered = rel.name.lower()
    if lowered.startswith(("plan-", "plan_")) or "roadmap" in lowered:
        return "roadmap"
    return "as-built"


def infer_status(kind: str, text: str) -> str:
    if kind == "historical":
        return "ARCHIVED"
    if kind == "audit":
        return "HISTORICAL"
    if kind == "roadmap":
        return "PLANNED"
    low = text.lower()
    if any(sig in low for sig in STALE_SIGNALS):
        return "PARTIAL"
    return "IMPLEMENTED"


def extract_anchors(text: str, limit: int = 8) -> list[str]:
    seen: list[str] = []
    for m in PATH_RE.finditer(text):
        p = m.group(1).lstrip("./")
        if p in seen:
            continue
        if (ROOT / p).exists():
            seen.append(p)
        if len(seen) >= limit:
            break
    return seen


def build_frontmatter(rel: Path, text: str) -> str:
    kind = infer_kind(rel, text)
    status = infer_status(kind, text)
    contract = CONTRACT_RE.search(text)
    version = VERSION_RE.search(text)
    anchors = extract_anchors(text)

    lines = ["---", f"kind: {kind}", f"status: {status}"]
    if contract:
        lines.append(f"contract: {contract.group(1)}")
    lines.append(f"version: {version.group(1) if version else '1.0.0'}")
    lines.append(f"last_verified: {TODAY}")
    lines.append("supersedes: []")
    if anchors:
        lines.append("code_anchors:")
        lines += [f"  - {a}" for a in anchors]
    else:
        lines.append("code_anchors: []")
    lines += ["---", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    targets = sorted(CLAUDE.rglob("*.md"))
    added = skipped = 0
    by_kind: dict[str, int] = {}
    by_status: dict[str, int] = {}

    for path in targets:
        rel = path.relative_to(CLAUDE)
        text = path.read_text(encoding="utf-8", errors="replace")
        if FRONTMATTER_RE.match(text):
            skipped += 1
            continue
        fm = build_frontmatter(rel, text)
        kind = re.search(r"kind: (\S+)", fm).group(1)
        status = re.search(r"status: (\S+)", fm).group(1)
        by_kind[kind] = by_kind.get(kind, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        if args.apply:
            path.write_text(fm + text, encoding="utf-8")
        added += 1

    verb = "añadiría" if args.dry_run else "añadido"
    print(f"{len(targets)} documentos · {verb} front matter a {added} · ya tenían: {skipped}")
    print("\npor kind:")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:12s} {v}")
    print("\npor status (inferido — revisar a mano):")
    for k, v in sorted(by_status.items()):
        print(f"  {k:12s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
