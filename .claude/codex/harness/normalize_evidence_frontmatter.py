"""Add repository-compatible front matter to Playwright-generated Markdown evidence."""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


HEADER = """---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: {today}
supersedes: []
code_anchors: []
---

"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    changed = 0
    for path in args.root.rglob("*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.startswith("---\n") or text.startswith("---\r\n"):
            continue
        path.write_text(HEADER.format(today=date.today().isoformat()) + text, encoding="utf-8")
        changed += 1
    print(f"Normalized {changed} generated Markdown evidence files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

