"""CTR-DB-TRUTH-MATRIX-001 (BL-36) — the matrix has to agree with the repo it describes.

Why this file exists
--------------------
`.claude/specs/platform/db-truth-matrix.md` is the document that decides which tables get
retired, which get a canonical writer and which are declared authoritative. It had **zero
executable coverage**: inverting the decision on `bi.fact_*` from *"DEPRECATED / 7 ficheros
la referencian"* to *"AUTORITATIVA (escritor único) / 0 ficheros"* — a claim of exclusive
authorship over an attribute that already has five writers, plus a reference count off by
seven — left every gate in the repo green. `grep db_inventory_matrix|db-truth-matrix|
CTR-DB-TRUTH` found only the generator and prose.

What is checked (and what is deliberately NOT)
----------------------------------------------
The matrix's own evidence model (§0.2) is the contract: **E1/E2 are static and derivable,
E3 (rows) is a backup snapshot, and everything about the live DB is NOT VERIFIED.** So the
tests below only assert what E1/E2 can prove:

* every `W`/`R` column and every prose count of writers/readers/referencing files must
  equal what `.claude/generated/db-inventory.json` measured (§2's own definition of those
  columns);
* a claim of a SINGLE authoritative writer (FABRIC §31) must survive contact with the
  measured writer list;
* a table declared DEPRECATED/RETIRO that still has readers in the code must DISCLOSE
  them, with a count, in its own row.

Nothing here asserts row counts, liveness in production, or that a table exists in the
real database — the matrix does not claim those either, and a test that did would be
inventing the evidence the document honestly declares missing.

The reader perimeter is DERIVED (K-029): it is a glob over `airflow/`, `services/`,
`scripts/` and `usdcop-trading-dashboard/`, matched with the generator's OWN regexes
(`scripts/diagnostics/db_inventory_matrix._patterns`), never a hand-kept file list.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / ".claude" / "specs" / "platform" / "db-truth-matrix.md"
INVENTORY = ROOT / ".claude" / "generated" / "db-inventory.json"

BACKTICK = re.compile(r"`([^`]+)`")
#: A bare/qualified SQL identifier. Anything else in backticks (paths, globs, prose) is
#: simply not a table and is skipped rather than guessed at.
IDENTIFIER = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)?")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _inventory() -> dict[str, dict]:
    assert INVENTORY.is_file(), f"falta {INVENTORY} — corre db_inventory_matrix.py --write"
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert data.get("contract") == "CTR-DB-TRUTH-MATRIX-001"
    return {o["key"]: o for o in data["objects"]}


def _resolve(name: str) -> dict | None:
    """`bi.fact_forecasts` / `usdcop_m5_ohlcv` -> the inventory entry, or None."""
    n = name.strip().strip('"').lower()
    if not IDENTIFIER.fullmatch(n):
        return None
    return _inventory().get(n if "." in n else f"public.{n}")


def _expand(name: str) -> list[dict]:
    """Resolve a backticked token to inventory entries, expanding `fact_*`-style globs.

    §9 names its groups with globs (``BI `fact_*` ``), so a rule that only understood exact
    names could be disarmed by deleting the one exact name the row happened to mention —
    which is precisely how the disclosure test first failed to notice its own mutation.
    A glob without a schema matches the table name in ANY schema: over-inclusive on
    purpose, because for a retirement decision the conservative error is to consider one
    table too many, never one too few.
    """
    entry = _resolve(name)
    if entry is not None:
        return [entry]
    token = name.strip().strip('"').lower()
    if "*" not in token:
        return []
    schema, _, bare = token.rpartition(".")
    pattern = re.compile(re.escape(bare).replace(r"\*", r"[a-z0-9_]*") + r"\Z")
    return [
        e for e in _inventory().values()
        if pattern.match(e["table"]) and (not schema or e["schema"] == schema)
    ]


@lru_cache(maxsize=1)
def _document() -> tuple[tuple, tuple]:
    """The matrix split into (table_rows, paragraphs).

    `table_rows` are `(lineno, header_cells_plain, raw_cells)`; `paragraphs` are
    `(lineno, joined_text)` for the prose between tables, where §6 keeps most of its
    counts. Headings and fenced blocks are dropped: §11 is SQL, not claims.
    """
    rows: list[tuple] = []
    paragraphs: list[tuple] = []
    header: list[str] | None = None
    para: list[str] = []
    para_start: int | None = None
    fenced = False

    def flush():
        nonlocal para, para_start
        if para:
            paragraphs.append((para_start, " ".join(para)))
        para, para_start = [], None

    for lineno, raw in enumerate(MATRIX.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line.startswith("```"):
            fenced = not fenced
            flush()
            continue
        if fenced:
            continue
        if line.startswith("|") and line.endswith("|") and len(line) > 2:
            flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue                       # separator row
            plain = [re.sub(r"[*`]", "", c).strip() for c in cells]
            if header is None:
                header = plain
                continue
            rows.append((lineno, tuple(header), tuple(cells)))
            continue
        header = None
        if not line or line.startswith("#"):
            flush()
            continue
        if para_start is None:
            para_start = lineno
        para.append(line)
    flush()
    return tuple(rows), tuple(paragraphs)


# ---------------------------------------------------------------------------
# Derived reader perimeter (K-029: globbed, never a hand list)
# ---------------------------------------------------------------------------

PERIMETER_DIRS = ("airflow", "services", "scripts", "usdcop-trading-dashboard")


@lru_cache(maxsize=1)
def _perimeter_blobs() -> tuple[tuple[str, str], ...]:
    from scripts.diagnostics.db_inventory_matrix import CODE_EXT, SKIP_PARTS

    out: list[tuple[str, str]] = []
    for name in PERIMETER_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or SKIP_PARTS & set(path.parts):
                continue
            if path.suffix.lower() not in CODE_EXT:
                continue
            try:
                out.append((path.relative_to(ROOT).as_posix(),
                            path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                continue
    return tuple(out)


def _readers_in_perimeter(entry: dict) -> list[str]:
    """Files in the four trees that READ the table, matched with the generator's regexes."""
    from scripts.diagnostics.db_inventory_matrix import _patterns

    writers, readers = _patterns(entry["schema"], entry["table"])
    hits = []
    for rel, text in _perimeter_blobs():
        if entry["table"] not in text.lower():
            continue
        if any(rx.search(text) for rx in writers):
            continue                            # a writer, not a reader
        if any(rx.search(text) for rx in readers):
            hits.append(rel)
    return sorted(hits)


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

#: "8 escritores" / "**53** lectores" / "cero escritores".
COUNT_RE = re.compile(r"(?:\*\*)?(\d+|cero)(?:\*\*)?\s+(?:\*\*)?(escritores?|lectores?)", re.I)
#: "`inference_features_5m` (18c, W3/R8)" / "(`trades_history` W2/R4)".
WR_RE = re.compile(r"`([^`]+)`[^`]{0,40}?\bW(\d+)/R(\d+)\b")
#: "E2: `inference_features_5m` R8".
R_ONLY_RE = re.compile(r"`([^`]+)`\s+R(\d+)\b")
#: "7 ficheros referencian `bi.fact_forecasts`" — W+R, the drop-cost of the table.
REFS_RE = re.compile(
    r"(\d+|cero)\s+ficheros?\s+(?:que\s+)?(?:la|las|lo|los)?\s*referencian?\s*`([^`]+)`", re.I)


def _last_name_before(text: str, pos: int) -> str | None:
    best = None
    for m in BACKTICK.finditer(text):
        if m.end() > pos:
            break
        best = m.group(1)
    return best


def _count_claims(text: str, lineno: int, fallback_names: tuple[str, ...]):
    """Yield `(lineno, entry, kind, claimed, quote)` for every count claim in `text`.

    Binding rule: a count belongs to the last table named in its own segment; if the
    segment names none (the common shape in §3, where the justification cell says "53
    lectores" about the row's subject) it belongs to the row's subject cell.
    """
    for m in COUNT_RE.finditer(text):
        claimed = 0 if m.group(1).lower() == "cero" else int(m.group(1))
        kind = "writers" if m.group(2).lower().startswith("escritor") else "readers"
        name = _last_name_before(text, m.start())
        entry = _resolve(name) if name else None
        targets = [entry] if entry else [_resolve(n) for n in fallback_names]
        for target in filter(None, targets):
            yield lineno, target, kind, claimed, m.group(0).strip()

    for m in WR_RE.finditer(text):
        entry = _resolve(m.group(1))
        if entry:
            yield lineno, entry, "writers", int(m.group(2)), m.group(0).strip()
            yield lineno, entry, "readers", int(m.group(3)), m.group(0).strip()

    for m in R_ONLY_RE.finditer(text):
        entry = _resolve(m.group(1))
        if entry:
            yield lineno, entry, "readers", int(m.group(2)), m.group(0).strip()

    for m in REFS_RE.finditer(text):
        entry = _resolve(m.group(2))
        if entry:
            claimed = 0 if m.group(1).lower() == "cero" else int(m.group(1))
            yield lineno, entry, "references", claimed, m.group(0).strip()


def _measured(entry: dict, kind: str) -> int:
    if kind == "references":
        return len(set(entry["writers"]) | set(entry["readers"]))
    return len(entry[kind])


# ---------------------------------------------------------------------------
# (c) declared reference counts must equal the measured ones
# ---------------------------------------------------------------------------

def test_w_and_r_columns_match_the_measured_inventory():
    """Las columnas `W`/`R` de §3 y §4 son, por definición de §2, el nº de ficheros con
    ruta de escritura/lectura medido por E2. Si no coinciden, la matriz decide con cifras
    que ya no existen.

    Rojo con: en `.claude/specs/platform/db-truth-matrix.md`, poner `| 0 | 0 |` en las
    columnas W/R de la fila `bi.fact_forecasts` (§4.4), que E2 mide como W5/R2.
    """
    mismatches, covered = [], 0
    for lineno, header, cells in _document()[0]:
        if "W" not in header or "R" not in header:
            continue
        iw, ir = header.index("W"), header.index("R")
        if iw >= len(cells) or ir >= len(cells):
            continue
        w = re.sub(r"[*` ]", "", cells[iw])
        r = re.sub(r"[*` ]", "", cells[ir])
        if not (w.isdigit() and r.isdigit()):
            continue                            # "0-2": una fila que agrupa varias tablas
        for name in BACKTICK.findall(cells[0]):
            entry = _resolve(name)
            if entry is None:
                continue                        # glob (`bi.fact_*`) o nombre no declarado
            covered += 1
            real = (len(entry["writers"]), len(entry["readers"]))
            if real != (int(w), int(r)):
                mismatches.append(
                    f"L{lineno} {entry['key']}: la matriz declara W={w}/R={r}, "
                    f"E2 mide W={real[0]}/R={real[1]}")

    assert covered >= 40, f"solo {covered} filas W/R comprobadas: el parser dejó de enganchar"
    assert not mismatches, "columnas W/R desalineadas con el inventario:\n" + "\n".join(mismatches)


def test_prose_reference_counts_match_the_measured_inventory():
    """Los conteos en prosa ("8 escritores", "53 lectores", "W2/R4", "7 ficheros
    referencian `x`") son los que sostienen las decisiones de §6, §9 y §10.

    Rojo con: en §9 de la matriz, cambiar `7 ficheros referencian \\`bi.fact_forecasts\\``
    por `0 ficheros referencian \\`bi.fact_forecasts\\`` (E2 mide 5 escritores + 2 lectores).
    """
    rows, paragraphs = _document()
    claims = []
    for lineno, _header, cells in rows:
        subject = tuple(BACKTICK.findall(cells[0]))
        for cell in cells[1:]:
            claims.extend(_count_claims(cell, lineno, subject))
    for lineno, text in paragraphs:
        claims.extend(_count_claims(text, lineno, ()))

    mismatches = [
        f"L{lineno} {entry['key']}.{kind}: la matriz dice {claimed} ({quote!r}), "
        f"E2 mide {_measured(entry, kind)}"
        for lineno, entry, kind, claimed, quote in claims
        if _measured(entry, kind) != claimed
    ]
    assert len(claims) >= 20, f"solo {len(claims)} claims en prosa: el parser dejó de enganchar"
    assert not mismatches, "conteos en prosa desalineados:\n" + "\n".join(mismatches)


# ---------------------------------------------------------------------------
# (a) no attribute may be declared to have a single writer when it has several
# ---------------------------------------------------------------------------

#: FABRIC §31 language for "this object owns the attribute".
EXCLUSIVE_RE = re.compile(
    r"autoritativ\w*|escritor\w*\s+únic\w*|únic\w*\s+escritor|un solo escritor|canonical writer",
    re.I)
#: The same words used to say the authority is MISSING ("Ninguna de las dos tiene un
#: escritor único identificable", "fuente NO autoritativa") are a finding, not a claim.
NEGATION_RE = re.compile(r"\bningun\w*|\bno\s+(?:es|hay|tiene|lo|autoritativa)\b", re.I)


def _enclosing_clause(text: str, start: int, end: int) -> str:
    """The cell/sentence the match sits in — never the neighbouring one.

    A fixed character window would import a negation from the cell next door and silence
    a real §31 violation, so the context stops at the nearest `|` or sentence boundary.
    """
    lo = max(text.rfind("|", 0, start), text.rfind(". ", 0, start), 0)
    hi = min(
        (i for i in (text.find("|", end), text.find(". ", end)) if i != -1),
        default=len(text),
    )
    return text[lo:hi]


def test_no_table_is_declared_sole_writer_of_an_attribute_it_shares():
    """FABRIC §31: ningún atributo con dos escritores. Si la matriz declara autoría única
    sobre una tabla, E2 no puede estar midiendo varios escritores para ella.

    Rojo con: en §9 de la matriz, cambiar la propuesta de `BI \\`fact_*\\`` de `DEPRECATED`
    a `AUTORITATIVA (escritor único)` — `bi.fact_forecasts` tiene 5 escritores medidos.
    """
    rows, paragraphs = _document()
    segments = [(lineno, " | ".join(cells), tuple(n for c in cells for n in BACKTICK.findall(c)))
                for lineno, _h, cells in rows]
    segments += [(lineno, text, ()) for lineno, text in paragraphs]

    violations, covered = [], 0
    for lineno, text, row_names in segments:
        for m in EXCLUSIVE_RE.finditer(text):
            if NEGATION_RE.search(_enclosing_clause(text, m.start(), m.end())):
                continue                        # "ninguna tiene un escritor único"
            names = row_names or tuple(BACKTICK.findall(text[:m.start()]))
            entries = {e["key"]: e for e in map(_resolve, names) if e}
            for key, entry in entries.items():
                covered += 1
                if len(entry["writers"]) > 1:
                    violations.append(
                        f"L{lineno} {key}: la matriz reclama autoría única ({m.group(0)!r}) "
                        f"pero E2 mide {len(entry['writers'])} escritores: {entry['writers']}")

    assert covered >= 1, "ninguna afirmación de autoría única quedó enganchada a una tabla"
    assert not violations, "§31 violado dentro de la propia matriz:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# (b) a DEPRECATED table with live readers must disclose them
# ---------------------------------------------------------------------------

#: Columns where the matrix states what it PROPOSES to do with the object.
DECISION_COLUMNS = ("clasificación propuesta", "propuesta", "hipótesis")
RETIRE_RE = re.compile(r"\b(DEPRECATED|RETIRO|RETIRAR|DROP|ELIMINAR)\b", re.I)
ANY_COUNT_RE = re.compile(
    r"(?:\*\*)?(\d+)(?:\*\*)?\s+(?:\*\*)?(?:escritores?|lectores?|ficheros?)"
    r"|\bW(\d+)/R(\d+)\b|`[^`]+`\s+R(\d+)\b")


def test_deprecated_tables_disclose_the_readers_that_still_exist():
    """Retirar una tabla que sigue teniendo lectores tiene un coste; la matriz solo es
    honesta si lo publica en la misma fila en la que propone el retiro.

    El perímetro de lectores se DERIVA (K-029): glob sobre `airflow/`, `services/`,
    `scripts/` y `usdcop-trading-dashboard/` con las regex del propio generador. Las
    tablas de nombre ambiguo (`signals`, `models`, …) quedan fuera: §2 dice explícitamente
    que no deben usarse para decidir un drop sin verificación manual.

    Rojo con: en §9 de la matriz, borrar `7 ficheros referencian \\`bi.fact_forecasts\\``
    de la fila `BI \\`fact_*\\`` (DEPRECATED) — el retiro pasaría a proponerse sin decir que
    rompe 7 ficheros.
    """
    undisclosed, covered = [], 0
    for lineno, header, cells in _document()[0]:
        idxs = [i for i, h in enumerate(header) if h.lower() in DECISION_COLUMNS]
        if not idxs:
            continue
        decision = " ".join(cells[i] for i in idxs if i < len(cells))
        keyword = RETIRE_RE.search(decision)
        if not keyword:
            continue
        entries = {
            e["key"]: e
            for name in (n for c in cells for n in BACKTICK.findall(c))
            for e in _expand(name)
            if not e["ambiguous_name"]
        }
        if not entries:
            continue                            # fila sin sujeto resoluble ("OMS legacy")
        live = {k: v for k, v in ((k, _readers_in_perimeter(e)) for k, e in entries.items()) if v}
        covered += 1
        if not live:
            continue
        row = " | ".join(cells)
        disclosed = [int(g) for m in ANY_COUNT_RE.finditer(row) for g in m.groups() if g]
        if not any(n > 0 for n in disclosed):
            undisclosed.append(
                f"L{lineno}: propone {keyword.group(0)} para {sorted(live)} pero la fila no "
                f"declara ningún conteo de referencias; lectores reales: "
                + "; ".join(f"{k} <- {v}" for k, v in live.items()))

    assert covered >= 4, f"solo {covered} filas de retiro evaluadas: el parser dejó de enganchar"
    assert not undisclosed, "retiro propuesto sin declarar los lectores vivos:\n" + "\n".join(undisclosed)


# ---------------------------------------------------------------------------
# guard: the tests above are only as good as the evidence file they read
# ---------------------------------------------------------------------------

def test_matrix_declares_the_inventory_it_is_checked_against():
    """La matriz debe seguir apuntando al artefacto E2 con el que aquí se contrasta.

    Rojo con: borrar `.claude/generated/db-inventory.json` de los `code_anchors` del
    front-matter de la matriz — los tests estarían validando contra un fichero que el
    documento ya no reconoce como su evidencia.
    """
    front = MATRIX.read_text(encoding="utf-8").split("---", 2)
    assert len(front) >= 3, "la matriz perdió su front-matter"
    for anchor in ("scripts/diagnostics/db_inventory_matrix.py",
                   ".claude/generated/db-inventory.json"):
        assert anchor in front[1], f"la matriz ya no declara {anchor} como code_anchor"
        assert (ROOT / anchor).exists(), f"code_anchor muerto: {anchor}"


@pytest.mark.parametrize("key", ["bi.fact_forecasts", "public.usdcop_m5_ohlcv",
                                 "public.macro_indicators_daily"])
def test_matrix_subjects_still_exist_in_the_inventory(key):
    """Las tres tablas sobre las que la matriz toma sus decisiones más caras (§9, §6.5)
    tienen que seguir existiendo en el inventario, o los tests de arriba se volverían
    silenciosamente vacíos al no resolver ningún nombre.

    Rojo con: borrar el objeto correspondiente de `.claude/generated/db-inventory.json`.
    """
    assert _inventory().get(key), f"{key} desapareció del inventario E2"
