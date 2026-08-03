"""Generate the repo's architectural inventory from SOURCE, never from prose.

Contract: CTR-KNOWLEDGE-INVENTORY-001

The knowledge system (`CLAUDE.md`, `.claude/**`) repeatedly drifted because every
architectural count was hand-maintained: DAG counts disagreed across three documents
(38 / 29 / 40), API routes were stated as 49 while 93 existed, and the spec tree in
`.claude/README.md` still described an `assets/` folder holding only `xauusd/`.

This script is the antidote. It derives every count and tree from the code, writes them
to ``.claude/generated/inventory.json``, and substitutes them into marker blocks:

    <!-- inv:dags -->  ...generated...  <!-- /inv -->

Usage:
    python scripts/diagnostics/generate_inventory.py --write   # refresh docs + json
    python scripts/diagnostics/generate_inventory.py --check   # CI: fail on drift

``--check`` never mutates the tree: it regenerates in memory and diffs. That matters
because a knowledge-CI job that writes is a knowledge-CI job that can launder drift.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # scripts/diagnostics/<this> -> repo root
GENERATED = ROOT / ".claude" / "generated"
INVENTORY_JSON = GENERATED / "inventory.json"

MARKER_RE = re.compile(
    r"(?P<open><!--\s*inv:(?P<key>[a-z0-9_.-]+)\s*-->)(?P<body>.*?)(?P<close><!--\s*/inv\s*-->)",
    re.S,
)


# --------------------------------------------------------------------------- DAGs
def _registry_constants(dags_dir: Path) -> dict[str, str]:
    """Map CONSTANT_NAME -> "dag_id" from the DAG registry contract."""
    reg = dags_dir / "contracts" / "dag_registry.py"
    if not reg.is_file():
        return {}
    out: dict[str, str] = {}
    aliases: dict[str, str] = {}
    for node in ast.parse(reg.read_text(encoding="utf-8")).body:
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        t = node.targets[0]
        if not isinstance(t, ast.Name) or not t.id.isupper():
            continue
        v = node.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            out[t.id] = v.value
        elif isinstance(v, ast.Name):
            # Backwards-compat aliases live at the tail of the registry
            # (e.g. `L6_ALERT_MONITOR = CORE_L6_ALERT_MONITOR`) and DAG modules
            # import the SHORT name. Miss these and their DAGs vanish from the count.
            aliases[t.id] = v.id
    for short, canonical in aliases.items():
        if canonical in out:
            out.setdefault(short, out[canonical])
    return out


def _resolve_dag_ids(path: Path, consts: dict[str, str]) -> list[str]:
    """Extract dag_ids from one DAG module.

    DAG modules here do ``DAG_ID = FORECAST_H5_L3_WEEKLY_TRAINING`` then
    ``with DAG(dag_id=DAG_ID, ...)``. A naive regex on ``dag_id=`` misses that
    indirection entirely (it reports the local name, not the value), so resolve
    module-level string bindings first, then look at the DAG(...) call sites.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []

    # name -> literal string (following one level of constant aliasing).
    # Walk the WHOLE tree, not just tree.body: several DAGs bind their id inside a
    # try/except or an `if` block (e.g. l0_macro_update.py), so a module-level-only
    # scan silently drops them — and a silently-low DAG count is exactly the drift
    # this generator exists to prevent.
    # Registry constants are frequently imported under an alias
    # (`CORE_L6_ALERT_MONITOR as L6_ALERT_MONITOR`), so fold the alias into the
    # constant map before resolving names.
    consts = dict(consts)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.asname and alias.name in consts:
                    consts[alias.asname] = consts[alias.name]

    local: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and node.targets:
            t = node.targets[0]
            if not isinstance(t, ast.Name):
                continue
            v = node.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                local.setdefault(t.id, v.value)
            elif isinstance(v, ast.Name) and v.id in consts:
                local.setdefault(t.id, consts[v.id])

    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name not in {"DAG", "dag"}:
            continue
        # dag_id may be keyword or first positional
        cand: ast.expr | None = None
        for kw in node.keywords:
            if kw.arg == "dag_id":
                cand = kw.value
        if cand is None and node.args:
            cand = node.args[0]
        if cand is None:
            continue
        if isinstance(cand, ast.Constant) and isinstance(cand.value, str):
            found.append(cand.value)
        elif isinstance(cand, ast.Name):
            resolved = local.get(cand.id) or consts.get(cand.id)
            if resolved:
                found.append(resolved)
    return found


def collect_dags() -> dict[str, Any]:
    dags_dir = ROOT / "airflow" / "dags"
    consts = _registry_constants(dags_dir)
    static: dict[str, str] = {}
    for f in sorted(dags_dir.glob("*.py")):
        for dag_id in _resolve_dag_ids(f, consts):
            static[dag_id] = f.name

    # Factory-generated (config-driven, no literal in any module)
    generated: dict[str, str] = {}
    pipelines = ROOT / "config" / "assets" / "pipelines.yaml"
    if pipelines.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(pipelines.read_text(encoding="utf-8")) or {}
            pattern = cfg.get("dag_id_pattern", "asset_{asset}_pipeline_weekly")
            for aid, a in (cfg.get("assets") or {}).items():
                if a.get("enabled"):
                    generated[pattern.format(asset=aid)] = "asset_pipeline_factory.py"
        except Exception:  # pragma: no cover - yaml optional at generate time
            pass

    all_ids = {**static, **generated}
    return {
        "modules": len(list(dags_dir.glob("*.py"))),
        "static_ids": len(static),
        "generated_ids": len(generated),
        "total_ids": len(all_ids),
        "ids": dict(sorted(all_ids.items())),
    }


# ----------------------------------------------------------------------- frontend
def collect_frontend() -> dict[str, Any]:
    app = ROOT / "usdcop-trading-dashboard" / "app"
    if not app.is_dir():
        return {"pages": 0, "pages_active": 0, "pages_legacy": 0, "api_routes": 0}
    pages = sorted(p.relative_to(app).as_posix() for p in app.rglob("page.tsx"))
    legacy = [p for p in pages if p.startswith("legacy/")]
    routes = sorted(
        p.relative_to(app).as_posix() for p in (app / "api").rglob("route.ts")
    ) if (app / "api").is_dir() else []
    return {
        "pages": len(pages),
        "pages_active": len(pages) - len(legacy),
        "pages_legacy": len(legacy),
        "api_routes": len(routes),
    }


# ---------------------------------------------------------------------- contracts
def collect_contracts() -> dict[str, Any]:
    py = ROOT / "src" / "contracts"
    ts = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts"
    py_files = sorted(f.name for f in py.glob("*.py") if f.name != "__init__.py") if py.is_dir() else []
    ts_files = sorted(f.name for f in ts.glob("*.ts")) if ts.is_dir() else []
    return {"python": len(py_files), "typescript": len(ts_files),
            "python_files": py_files, "typescript_files": ts_files}


# --------------------------------------------------------------------- migrations
def collect_migrations() -> dict[str, Any]:
    d = ROOT / "database" / "migrations"
    files = sorted(f.name for f in d.glob("*.sql")) if d.is_dir() else []
    return {"count": len(files), "latest": files[-1] if files else None}


# ---------------------------------------------------------------------- knowledge
KNOWLEDGE_INDEX_NAMES = {"readme.md", "index.md", "00-index.md", "00_index.md"}
DEFINITION_FM_RE = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", re.S)


def _frontmatter_value(path: Path, key: str) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = DEFINITION_FM_RE.match(text)
    if not match:
        return ""
    value_match = re.search(
        rf"^{re.escape(key)}:\s*(.*?)\s*$",
        match.group(1),
        re.M,
    )
    if not value_match:
        return ""
    raw = value_match.group(1).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        try:
            return str(json.loads(raw))
        except json.JSONDecodeError:
            return raw[1:-1]
    if len(raw) >= 2 and raw[0] == raw[-1] == "'":
        return raw[1:-1].replace("''", "'")
    return raw


def _live_spec_files(specs: Path) -> list[Path]:
    if not specs.is_dir():
        return []
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(specs, topdown=True):
        parent = Path(directory)
        rel = parent.relative_to(specs)
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".")
            and not (not rel.parts and name == "archive")
        ]
        files.extend(
            parent / name
            for name in filenames
            if name.lower().endswith(".md")
            and name.lower() not in KNOWLEDGE_INDEX_NAMES
        )
    return sorted(files)


def collect_knowledge(root: Path = ROOT) -> dict[str, Any]:
    c = root / ".claude"

    def _md(sub: str) -> list[str]:
        p = c / sub
        return sorted(f.relative_to(c).as_posix() for f in p.rglob("*.md")) if p.is_dir() else []

    rules = _md("rules")
    specs = _live_spec_files(c / "specs")
    skill_files = (
        sorted((c / "skills").glob("*/SKILL.md"))
        if (c / "skills").is_dir()
        else []
    )
    agent_files = (
        sorted((c / "agents").glob("*.md"))
        if (c / "agents").is_dir()
        else []
    )
    skill_definitions = [
        {
            "name": _frontmatter_value(path, "name") or path.parent.name,
            "path": path.relative_to(c).as_posix(),
            "description": _frontmatter_value(path, "description"),
        }
        for path in skill_files
    ]
    agent_definitions = [
        {
            "name": _frontmatter_value(path, "name") or path.stem,
            "path": path.relative_to(c).as_posix(),
            "description": _frontmatter_value(path, "description"),
        }
        for path in agent_files
    ]
    rules_bytes = sum((c / r).stat().st_size for r in rules)
    rules_words = sum(len((c / r).read_text(encoding="utf-8", errors="replace").split()) for r in rules)
    return {
        "rules": len(rules), "specs": len(specs),
        "skills": len(skill_definitions), "agents": len(agent_definitions),
        "rules_bytes": rules_bytes, "rules_words": rules_words,
        "skill_names": [item["name"] for item in skill_definitions],
        "agent_names": [item["name"] for item in agent_definitions],
        "skill_definitions": skill_definitions,
        "agent_definitions": agent_definitions,
    }


def collect_workflows() -> dict[str, Any]:
    d = ROOT / ".github" / "workflows"
    files = sorted(f.name for f in d.glob("*.yml")) + sorted(f.name for f in d.glob("*.yaml")) if d.is_dir() else []
    return {"count": len(files), "files": files}


def collect_specs_tree(root: Path = ROOT) -> list[str]:
    """One line per live spec directory, excluding indexes and archive."""
    specs = root / ".claude" / "specs"
    if not specs.is_dir():
        return []
    counts: dict[Path, int] = {}
    for path in _live_spec_files(specs):
        counts[path.parent] = counts.get(path.parent, 0) + 1
    return [
        (
            f"{'.' if directory == specs else directory.relative_to(specs).as_posix() + '/'} "
            f"({count})"
        )
        for directory, count in sorted(
            counts.items(),
            key=lambda item: item[0].relative_to(specs).as_posix(),
        )
    ]


# ------------------------------------------------------------------------ render
def build() -> dict[str, Any]:
    return {
        "_note": "GENERATED by scripts/diagnostics/generate_inventory.py — do not hand-edit.",
        "dags": collect_dags(),
        "frontend": collect_frontend(),
        "contracts": collect_contracts(),
        "migrations": collect_migrations(),
        "knowledge": collect_knowledge(),
        "workflows": collect_workflows(),
        "specs_tree": collect_specs_tree(),
    }


def _brief_description(value: str, limit: int = 180) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    shortened = value[: limit - 1].rsplit(" ", 1)[0]
    return shortened.rstrip(".,;:") + "…"


def _table_cell(value: str) -> str:
    return value.replace("|", r"\|")


def _render_capabilities(knowledge: dict[str, Any]) -> str:
    lines = [
        "### Agentes especializados (solo lectura)",
        "",
        "| Agente | Responsabilidad |",
        "|---|---|",
    ]
    for item in knowledge["agent_definitions"]:
        lines.append(
            f"| [{_table_cell(item['name'])}]({item['path']}) | "
            f"{_table_cell(_brief_description(item['description']))} |"
        )
    lines.extend(
        [
            "",
            "<details>",
            "<summary><strong>Skills operativas y de dominio</strong></summary>",
            "",
            "| Skill | Cuándo usarla |",
            "|---|---|",
        ]
    )
    for item in knowledge["skill_definitions"]:
        lines.append(
            f"| [{_table_cell(item['name'])}]({item['path']}) | "
            f"{_table_cell(_brief_description(item['description']))} |"
        )
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def render(key: str, inv: dict[str, Any]) -> str:
    d, f = inv["dags"], inv["frontend"]
    c, k, w = inv["contracts"], inv["knowledge"], inv["workflows"]
    if key == "dags":
        return (f"**{d['total_ids']} DAGs** ({d['static_ids']} declarados en "
                f"{d['modules']} módulos + {d['generated_ids']} generados por factory)")
    if key == "frontend":
        return (f"**{f['pages_active']} páginas activas** ({f['pages_legacy']} en `/legacy`) · "
                f"**{f['api_routes']} rutas API**")
    if key == "contracts":
        return f"**{c['python']} contratos Python** · **{c['typescript']} TypeScript**"
    if key == "migrations":
        return f"**{inv['migrations']['count']} migraciones** (última: `{inv['migrations']['latest']}`)"
    if key == "workflows":
        return f"**{w['count']} GitHub Actions**"
    if key == "knowledge":
        return (f"**{k['rules']} rules** (~{k['rules_words']:,} palabras auto-cargadas) · "
                f"**{k['specs']} specs** · **{k['skills']} skills** · **{k['agents']} agents**")
    if key == "capabilities":
        return _render_capabilities(k)
    if key == "specs_tree":
        return "```\n" + "\n".join(inv["specs_tree"]) + "\n```"
    raise KeyError(f"unknown inventory key: {key}")


FENCE_RE = re.compile(r"^```.*?^```", re.S | re.M)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _code_spans(text: str) -> list[tuple[int, int]]:
    """Byte ranges of code — fenced blocks AND inline backticks.

    Documentation that *explains* the marker convention writes it literally, and a naive
    regex treats that example as a real marker. This actually happened: `.claude/README.md`
    documenting the convention crashed the generator with `unknown inventory key: key`.
    Inline backticks matter as much as fences — the example that broke it was inline.
    """
    return [(m.start(), m.end()) for m in FENCE_RE.finditer(text)] + [
        (m.start(), m.end()) for m in INLINE_CODE_RE.finditer(text)
    ]


def apply_markers(text: str, inv: dict[str, Any]) -> tuple[str, list[str]]:
    keys: list[str] = []
    spans = _code_spans(text)

    def _sub(m: re.Match[str]) -> str:
        if any(start <= m.start() < end for start, end in spans):
            return m.group(0)  # documentation example, not a live marker
        key = m.group("key")
        keys.append(key)
        body = render(key, inv)
        return f"{m.group('open')}\n{body}\n{m.group('close')}"

    return MARKER_RE.sub(_sub, text), keys


# Estado runtime gitignorado: `coordination/tmp/` guarda worktrees desechables de los
# carriles de review en paralelo, cada uno con su propia COPIA de `CLAUDE.md` y
# `.claude/README.md`. Reescribir los bloques generados dentro de esos clones no arregla
# nada — se borran — pero sí hacía fallar `--check` en local con "stale inventory block"
# de ficheros que ni siquiera son de este árbol. CI nunca los ve porque hace checkout limpio.
EPHEMERAL_PARTS = {"tmp", "node_modules", ".next", "__pycache__", ".pytest_cache", "_runtime"}
EPHEMERAL_PREFIXES = (".pytest-",)
MARKER_EXCLUDED_PREFIXES = (
    ".claude/archive",
    ".claude/codex/_runtime",
    ".claude/coordination",
    ".claude/evidence",
    ".claude/specs/archive",
)


def _is_ephemeral(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(
        rel == prefix or rel.startswith(prefix + "/")
        for prefix in MARKER_EXCLUDED_PREFIXES
    ):
        return True
    return any(
        part in EPHEMERAL_PARTS or part.startswith(EPHEMERAL_PREFIXES)
        for part in path.relative_to(ROOT).parts
    )


def target_files() -> list[Path]:
    out = [ROOT / "CLAUDE.md"]
    c = ROOT / ".claude"
    if c.is_dir():
        for directory, dirnames, filenames in os.walk(c, topdown=True):
            parent = Path(directory)
            dirnames[:] = [
                name
                for name in dirnames
                if not _is_ephemeral(parent / name)
            ]
            out.extend(
                parent / name
                for name in filenames
                if name.lower().endswith(".md")
                and not _is_ephemeral(parent / name)
            )
    return [p for p in out if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="refresh inventory.json + marker blocks")
    g.add_argument("--check", action="store_true", help="fail if anything is out of date")
    args = ap.parse_args()

    inv = build()
    payload = json.dumps(inv, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    drift: list[str] = []
    used: set[str] = set()

    for path in target_files():
        original = path.read_text(encoding="utf-8", errors="replace")
        if "<!-- inv:" not in original:
            continue
        updated, keys = apply_markers(original, inv)
        used.update(keys)
        if updated != original:
            if args.write:
                path.write_text(updated, encoding="utf-8")
            else:
                drift.append(f"stale inventory block: {path.relative_to(ROOT).as_posix()}")

    if args.write:
        GENERATED.mkdir(parents=True, exist_ok=True)
        INVENTORY_JSON.write_text(payload, encoding="utf-8")
        print(f"wrote {INVENTORY_JSON.relative_to(ROOT).as_posix()}")
        print(f"marker keys substituted: {', '.join(sorted(used)) or '(none)'}")
        return 0

    if not INVENTORY_JSON.is_file():
        drift.append(".claude/generated/inventory.json missing — run --write")
    elif INVENTORY_JSON.read_text(encoding="utf-8") != payload:
        drift.append(".claude/generated/inventory.json is stale — run --write")

    if drift:
        print("INVENTORY DRIFT DETECTED\n", file=sys.stderr)
        for d in drift:
            print(f"  - {d}", file=sys.stderr)
        print("\nFix: python scripts/diagnostics/generate_inventory.py --write", file=sys.stderr)
        return 1

    print(f"inventory OK ({inv['dags']['total_ids']} DAGs, "
          f"{inv['frontend']['api_routes']} API routes, {inv['knowledge']['specs']} specs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
