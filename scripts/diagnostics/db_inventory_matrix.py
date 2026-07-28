#!/usr/bin/env python
"""Static DB inventory: which tables are DECLARED, who WRITES them, who READS them.

Contract: CTR-DB-TRUTH-MATRIX-001 (BL-36)

Why this exists
---------------
BL-36 asks for a "matriz de verdad" over the inherited DB inventory: exactly one
authoritative writer per attribute (FABRIC §7/§31). Deciding what to retire needs
*evidence*, and the two honest sources of evidence are:

  1. **The DDL in the repo** — every ``CREATE TABLE`` under ``database/migrations``,
     ``database/schemas`` and ``init-scripts``. This says what *may* exist.
  2. **The code** — who issues ``INSERT``/``UPDATE``/``COPY``/``to_sql`` against a
     table (writers) and who issues ``FROM``/``JOIN``/``read_sql`` (readers).

Neither of them can tell you how many rows a table has, or whether it is alive in
production. This script therefore **never guesses**: row counts come only from
committed backup manifests (``data/backups/**/*manifest*.json``, dumped from the real
DB by the backup module), and every table not covered by one is emitted with
``row_evidence: null`` and ``verification: "NOT_VERIFIED"``.

Read-only. Emits no DDL, opens no database connection.

Usage
-----
    python scripts/diagnostics/db_inventory_matrix.py --write
    python scripts/diagnostics/db_inventory_matrix.py --check      # CI: not stale
    python scripts/diagnostics/db_inventory_matrix.py --summary    # human tail
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".claude" / "generated" / "db-inventory.json"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

DDL_DIRS = ("database/migrations", "database/schemas", "init-scripts", "database/seed")

CODE_DIRS = (
    "airflow",
    "services",
    "scripts",
    "src",
    "config",
    "tests",
    "usdcop-trading-dashboard/app",
    "usdcop-trading-dashboard/lib",
    "usdcop-trading-dashboard/components",
    "usdcop-trading-dashboard/hooks",
    "usdcop-trading-dashboard/types",
)

CODE_EXT = {".py", ".ts", ".tsx", ".js", ".sql", ".yaml", ".yml", ".sh"}

SKIP_PARTS = {"node_modules", ".next", ".git", "__pycache__", ".venv", "venv", "dist", "build"}

MANIFESTS = (
    "data/backups/features/feature_backup_manifest.json",
    "data/backups/seeds/backup_manifest.json",
)

# Bare names so generic that a static grep cannot separate SQL from prose/imports.
AMBIGUOUS = {"users", "signals", "executions", "for", "if", "config", "models", "events"}

# FABRIC §37/§38 five-timestamp semantics. A legacy table is mapped by column NAME,
# which is the only thing the DDL can prove; `ingested_at` and `created_at` are the same
# role (row landed) and `updated_at` is a MUTATION marker with no FABRIC slot at all —
# an immutable raw bar has nothing to update.
TS_ROLES = {
    "event_time": ("time", "event_time", "bar_start_utc", "observation_date",
                   "reference_date", "ts", "timestamp", "date", "trade_date"),
    "provider_published_at": ("provider_published_at", "published_at", "release_date",
                              "publication_date"),
    "available_at": ("available_at",),
    "retrieved_at": ("retrieved_at", "fetched_at", "scraped_at"),
    "ingested_at": ("ingested_at", "created_at", "inserted_at", "loaded_at"),
    "_mutation": ("updated_at", "modified_at", "last_updated"),
}

CREATE_RE = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(TABLE|MATERIALIZED\s+VIEW|VIEW)"
    r"(?:\s+IF\s+NOT\s+EXISTS)?\s+([A-Za-z0-9_.\"]+)",
    re.I,
)
ALTER_ADD_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?([A-Za-z0-9_.\"]+)\s+"
    r"ADD\s+COLUMN(?:\s+IF\s+NOT\s+EXISTS)?\s+([A-Za-z0-9_\"]+)",
    re.I,
)
HYPERTABLE_RE = re.compile(r"create_hypertable\s*\(\s*'([^']+)'", re.I)
SCHEMA_RE = re.compile(r"CREATE\s+SCHEMA(?:\s+IF\s+NOT\s+EXISTS)?\s+([A-Za-z0-9_\"]+)", re.I)


def _iter_files(dirs: tuple[str, ...], exts: set[str] | None) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if SKIP_PARTS & set(p.parts):
                continue
            if exts is not None and p.suffix.lower() not in exts:
                continue
            out.append(p)
    return sorted(out)


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def _norm(raw: str) -> tuple[str, str]:
    """`bi.fact_forecasts` -> ('bi', 'fact_forecasts'); bare -> ('public', name)."""
    name = raw.replace('"', "").strip().lower()
    if "." in name:
        schema, _, bare = name.partition(".")
        return schema, bare
    return "public", name


# ---------------------------------------------------------------------------
# 1. Declared objects (DDL)
# ---------------------------------------------------------------------------

def collect_ddl() -> tuple[dict[str, dict], list[str]]:
    objects: dict[str, dict] = {}
    schemas: set[str] = {"public"}

    # Filename order is not DDL order: migration 060 ALTERs a table that init-scripts/01
    # creates. CREATEs are therefore collected in a first pass and ALTERs in a second.
    sources = [
        (p, p.read_text(encoding="utf-8", errors="replace"))
        for p in _iter_files(DDL_DIRS, {".sql"})
    ]

    for path, text in sources:
        for m in SCHEMA_RE.finditer(text):
            schemas.add(m.group(1).replace('"', "").lower())
        for m in CREATE_RE.finditer(text):
            kind = "table" if m.group(1).lower() == "table" else (
                "materialized_view" if "materialized" in m.group(1).lower() else "view"
            )
            schema, bare = _norm(m.group(2))
            if bare in {"if", "for", "not", "exists"}:  # regex noise from odd DDL
                continue
            key = f"{schema}.{bare}"
            entry = objects.setdefault(
                key,
                {
                    "schema": schema,
                    "table": bare,
                    "object_kind": kind,
                    "ddl_sources": [],
                    "declared_columns": None,
                    "column_names": [],
                    "timestamps": {},
                    "hypertable": False,
                },
            )
            # A later CREATE TABLE wins over an earlier CREATE VIEW of the same name.
            if kind == "table":
                entry["object_kind"] = "table"
            src = _rel(path)
            if src not in entry["ddl_sources"]:
                entry["ddl_sources"].append(src)
            if kind == "table":
                cols = _column_names(text, m.end())
                if cols:
                    known = set(entry.get("column_names") or [])
                    entry["column_names"] = sorted(known | set(cols))
                    entry["declared_columns"] = len(entry["column_names"])
                    entry["timestamps"] = _timestamp_roles(entry["column_names"])
        for m in HYPERTABLE_RE.finditer(text):
            schema, bare = _norm(m.group(1))
            key = f"{schema}.{bare}"
            if key in objects:
                objects[key]["hypertable"] = True

    # Second pass: columns added later by ALTER TABLE are part of the contract too —
    # `available_at` only exists on the OHLCV tables because migration 060 bolted it on.
    for _path, text in sources:
        for m in ALTER_ADD_RE.finditer(text):
            schema, bare = _norm(m.group(1))
            entry = objects.get(f"{schema}.{bare}")
            if not entry:
                continue
            col = m.group(2).strip('"').lower()
            if col not in entry["column_names"]:
                entry["column_names"] = sorted(set(entry["column_names"]) | {col})
                entry["declared_columns"] = len(entry["column_names"])
                entry["timestamps"] = _timestamp_roles(entry["column_names"])

    return objects, sorted(schemas)


def _timestamp_roles(columns: list[str]) -> dict[str, list[str]]:
    """Map declared columns onto the five FABRIC timestamps. Absence is the finding."""
    out: dict[str, list[str]] = {}
    for role, names in TS_ROLES.items():
        hit = sorted(c for c in columns if c in names)
        out[role] = hit
    return out


def _column_names(text: str, start: int) -> list[str] | None:
    """Best-effort column names of the CREATE TABLE body that starts at `start`."""
    open_idx = text.find("(", start)
    if open_idx == -1:
        return None
    depth, i = 0, open_idx
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[open_idx + 1 : i]
    # split top-level commas
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    cols: list[str] = []
    for p in parts:
        head = p.strip().split()[:1]
        if not head:
            continue
        raw = head[0].strip('"').strip(",")
        if raw.upper() in {
            "PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT",
            "EXCLUDE", "LIKE", "INHERITS",
        }:
            continue
        if raw:
            cols.append(raw.lower())
    return cols or None


# ---------------------------------------------------------------------------
# 2. Writers / readers (static code scan)
# ---------------------------------------------------------------------------

def _patterns(schema: str, bare: str) -> tuple[list[re.Pattern], list[re.Pattern]]:
    q = rf"(?:{re.escape(schema)}\.)?{re.escape(bare)}" if schema != "public" else \
        rf"(?:public\.)?{re.escape(bare)}"
    writers = [
        re.compile(rf"\binsert\s+into\s+{q}\b", re.I),
        re.compile(rf"\bupdate\s+{q}\s+set\b", re.I),
        re.compile(rf"\bdelete\s+from\s+{q}\b", re.I),
        re.compile(rf"\bcopy\s+{q}\s*[(\s]", re.I),
        re.compile(rf"\bmerge\s+into\s+{q}\b", re.I),
        re.compile(rf"to_sql\(\s*(?:name\s*=\s*)?['\"]{re.escape(bare)}['\"]", re.I),
        re.compile(rf"\btruncate\s+(?:table\s+)?{q}\b", re.I),
    ]
    readers = [
        re.compile(rf"\bfrom\s+{q}\b", re.I),
        re.compile(rf"\bjoin\s+{q}\b", re.I),
        re.compile(rf"read_sql[^\n]{{0,200}}\b{re.escape(bare)}\b", re.I),
    ]
    return writers, readers


def _orm_pattern(bare: str) -> re.Pattern:
    """An ORM class declares a table without necessarily querying it.

    `sb_signals` and `sb_exchange_credentials` exist only as SQLAlchemy models: the
    schema is declared, no code path ever writes a row. That distinction is exactly
    what BL-36 needs, so it gets its own bucket instead of being counted as a writer.
    """
    return re.compile(
        rf"(?:__tablename__|tableName|table_name)\s*[:=]\s*['\"]{re.escape(bare)}['\"]",
        re.I,
    )


def scan_code(objects: dict[str, dict]) -> None:
    files = _iter_files(CODE_DIRS, CODE_EXT)
    blobs = []
    for p in files:
        try:
            blobs.append((_rel(p), p.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue

    ddl_prefixes = tuple(DDL_DIRS)

    for key, entry in objects.items():
        writers, readers = _patterns(entry["schema"], entry["table"])
        orm = _orm_pattern(entry["table"])
        w_hits: list[str] = []
        r_hits: list[str] = []
        o_hits: list[str] = []
        for rel, text in blobs:
            if entry["table"] not in text.lower():
                continue
            is_ddl_dir = rel.startswith(ddl_prefixes)
            if is_ddl_dir:
                continue
            if orm.search(text):
                o_hits.append(rel)
            if any(rx.search(text) for rx in writers):
                w_hits.append(rel)
            elif any(rx.search(text) for rx in readers):
                r_hits.append(rel)
        entry["writers"] = sorted(set(w_hits))
        entry["readers"] = sorted(set(r_hits))
        entry["orm_declarations"] = sorted(set(o_hits))
        entry["ambiguous_name"] = entry["table"] in AMBIGUOUS


# ---------------------------------------------------------------------------
# 3. Row evidence (committed backup manifests only)
# ---------------------------------------------------------------------------

def collect_row_evidence() -> dict[str, dict]:
    ev: dict[str, dict] = {}
    for rel in MANIFESTS:
        p = ROOT / rel
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        as_of = data.get("created_at") or data.get("backup_timestamp")
        for name, meta in (data.get("tables") or {}).items():
            if not isinstance(meta, dict):
                continue
            ev[f"public.{name.lower()}"] = {
                "rows": meta.get("rows"),
                "latest_row_ts": meta.get("latest"),
                "evidence_source": rel,
                "evidence_as_of": as_of,
            }
        # seeds manifest has a flat shape
        for block in ("ohlcv", "macro"):
            meta = data.get(block)
            if isinstance(meta, dict) and "rows" in meta:
                name = {"ohlcv": "public.usdcop_m5_ohlcv",
                        "macro": "public.macro_indicators_daily"}[block]
                ev.setdefault(name, {
                    "rows": meta.get("rows"),
                    "latest_row_ts": (meta.get("date_range") or [None, None])[-1],
                    "evidence_source": rel,
                    "evidence_as_of": as_of,
                })
    return ev


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _liveness(entry: dict) -> str:
    """Static liveness *hypothesis*. Never a claim about production rows."""
    rows = (entry.get("row_evidence") or {}).get("rows")
    has_w, has_r = bool(entry["writers"]), bool(entry["readers"])
    if rows is not None:
        if rows > 0 and has_w:
            return "LIVE"
        if rows > 0 and not has_w:
            return "DATA_NO_WRITER"      # historical data, nobody refreshes it
        if rows == 0 and not has_w:
            return "DEAD"                # empty and unwritten: the drop candidates
        return "EMPTY_WITH_WRITER"       # writer exists but never produced a row
    if has_w and has_r:
        return "WIRED_UNVERIFIED"
    if has_w or has_r:
        return "PARTIALLY_WIRED_UNVERIFIED"
    return "ORPHAN_DDL_UNVERIFIED"


def build() -> dict:
    objects, schemas = collect_ddl()
    scan_code(objects)
    evidence = collect_row_evidence()

    tables = []
    for key in sorted(objects):
        e = dict(objects[key])
        e["key"] = key
        e["row_evidence"] = evidence.get(key)
        e["verification"] = "MANIFEST_SNAPSHOT" if e["row_evidence"] else "NOT_VERIFIED"
        e["liveness_hypothesis"] = _liveness(e)
        tables.append(e)

    counts: dict[str, int] = {}
    for t in tables:
        counts[t["liveness_hypothesis"]] = counts.get(t["liveness_hypothesis"], 0) + 1

    return {
        "contract": "CTR-DB-TRUTH-MATRIX-001",
        "generator": "scripts/diagnostics/db_inventory_matrix.py",
        "evidence_model": {
            "declared_objects": "parsed from repo DDL (may exist in DB, may not)",
            "writers_readers": "static regex over repo code; precision caveat on ambiguous names",
            "rows_and_freshness": (
                "ONLY from committed backup manifests; every other table is NOT_VERIFIED "
                "and requires a live DB inspection"
            ),
            "no_db_connection": True,
        },
        "schemas_declared": schemas,
        "totals": {
            "declared_objects": len(tables),
            "tables": sum(1 for t in tables if t["object_kind"] == "table"),
            "views": sum(1 for t in tables if t["object_kind"] != "table"),
            "with_row_evidence": sum(1 for t in tables if t["row_evidence"]),
            "not_verified": sum(1 for t in tables if not t["row_evidence"]),
            "liveness_hypothesis": dict(sorted(counts.items())),
        },
        "objects": tables,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the JSON inventory")
    ap.add_argument("--check", action="store_true", help="fail if the stored JSON is stale")
    ap.add_argument("--summary", action="store_true", help="print a human summary")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    data = build()
    payload = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    out = Path(args.out)

    if args.check:
        if not out.is_file():
            print(f"missing {out} — run with --write", file=sys.stderr)
            return 1
        if out.read_text(encoding="utf-8") != payload:
            print(f"{out} is stale — run with --write", file=sys.stderr)
            return 1
        print(f"{out} up to date")
        return 0

    if args.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"wrote {out} ({data['totals']['declared_objects']} declared objects)")

    if args.summary or not (args.write or args.check):
        t = data["totals"]
        print(f"declared objects : {t['declared_objects']} "
              f"(tables={t['tables']}, views={t['views']})")
        print(f"row evidence     : {t['with_row_evidence']} from manifests, "
              f"{t['not_verified']} NOT_VERIFIED")
        for k, v in t["liveness_hypothesis"].items():
            print(f"  {k:<32} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
