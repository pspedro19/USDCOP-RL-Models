"""Content-addressed local preservation of thesis evidence; never reads secrets.

Copies only the explicit research allowlist. Does not certify the copied claims.
The manifest is created exclusively; a changed input during copying aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = (
    "src/research/**/*.py",
    "scripts/analysis/*thesis*.py",
    "scripts/analysis/*sanity*.py",
    "scripts/diagnostics/*thesis*.py",
    "scripts/diagnostics/verify_macro_declared_identity.py",
    "scripts/data/build_research_macro.py",
    "src/analysis/llm_client.py",
    "config/research/*",
    "config/experiments/thesis_ppo_v2.yaml",
    "data/thesis/research_data_portable_v2.pkl",
    "data/thesis/llm/*",
    "outputs/thesis-repair/ppo_v2_recipe_flat/*",
    "outputs/thesis-repair/ppo_v2_diagnostic_full/*",
    "outputs/thesis-repair/*.json",
    "outputs/thesis-repair/*.jsonl",
    "outputs/thesis-repair/sanity/*.json",
    "outputs/thesis-repair/results_v2/*",
    "outputs/thesis/figuras/*",
    "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2*",
    "seeds/latest/usdcop_m5_ohlcv.parquet",
    "seeds/latest/usdcop_daily_ohlcv.parquet",
    "docs/analysis/exp-tesis-*.md",
    ".claude/specs/planes/06-*.md",
    ".claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md",
    "registries/ledger.jsonl",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(root: Path, output: Path, patterns=PATTERNS) -> dict:
    root, output = root.resolve(), output.resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("snapshot output already exists; use a new directory")
    paths = sorted(
        {
            p.resolve()
            for pattern in patterns
            for p in root.glob(pattern)
            if p.is_file() and not p.is_symlink()
        }
    )
    for p in paths:
        parts = p.relative_to(root).parts
        if any(x.lower() == "secrets" or x.lower().startswith(".env") for x in parts):
            raise ValueError("secret path refused")
        if p.suffix.lower() in {".pem", ".key"} or p.name.lower().startswith(
            ("credentials", "service-account")
        ):
            raise ValueError("credential path refused")
    (output / "objects").mkdir(parents=True)
    rows = []
    for p in paths:
        sha = digest(p)
        target = output / "objects" / sha
        if not target.exists():
            shutil.copyfile(p, target)
        if digest(target) != sha or digest(p) != sha:
            raise RuntimeError("source changed while snapshotting: " + str(p.relative_to(root)))
        rows.append(
            {
                "path": p.relative_to(root).as_posix(),
                "sha256": sha,
                "bytes": p.stat().st_size,
                "object": "objects/" + sha,
            }
        )
    report = {
        "contract": "THESIS-EVIDENCE-SNAPSHOT-1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "preservation_only_not_source_certification",
        "files": rows,
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = freeze(ROOT, args.output)
    print(
        json.dumps(
            {
                "files": len(report["files"]),
                "manifest": str(args.output / "manifest.json"),
                "sha256": digest(args.output / "manifest.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
