"""/replay is a READ-ONLY research surface — perimeter lock (BL-34).

Contract: `.claude/rules/approval-gates.md` invariante 3 ("los botones de aprobación
viven solo en /dashboard") + CTR-QUANT-CONSTITUTION-001 §7 (el Voto 2 decide sobre el
bundle publicado, nunca sobre el preview del replay).

WHY THIS FILE EXISTS
--------------------
BL-34 shipped a runtime-correct `/replay` (Playwright 3/3 green) with **no static
protection**. Two mutations proved it, both leaving `578 passed`, `npm run rbac:check`
OK and `npm run rbac:test` PASS — zero delta:

  M-A  `components/production/ForecastingBacktestSection.tsx:1619`
       `const canPromote = userRole === 'admin' && !readOnly`  →  `const canPromote = true`
       (the Vote-2 ApprovalPanel + DeployPanel mount on /replay, for ANY role).
  M-B  a new `components/production/RogueApprovalWidget.tsx` with
       "Aprobar (Voto 2/2)" / "Rechazar" / "Desplegar a producción" buttons and a
       `POST /api/production/approve`, mounted from `app/replay/page.tsx`.

The only test that caught either was `tests/e2e/replay-read-only.spec.ts`, and
`grep -rn playwright .github/workflows` returns 0 — no workflow runs it.

SHAPE OF THE LOCK (regla K-029: el perímetro se DERIVA, jamás se enumera)
------------------------------------------------------------------------
Same pattern as `test_forecasting_caveat_present.py::test_forecasting_has_no_action_capabilities`
(S-06): start at the route entry point `app/replay/page.tsx` and follow every statically
resolvable local import transitively. A file is inside the muralla the moment the surface
imports it — wherever the author chose to put it. That is what kills M-B: a hardcoded
directory list is side-stepped by picking another directory; an import closure is not.

THE ASYMMETRY WITH /forecasting (read this before "simplifying" the file)
------------------------------------------------------------------------
/forecasting reaches NO approval code, so its lock is a flat blacklist over the whole
closure. /replay is different: it deliberately shares ONE implementation with /dashboard
(`BacktestTerminalPage` variant + `ForecastingBacktestSection readOnly`), so the approval
machinery IS legitimately inside the closure, gated by `readOnly`. A flat blacklist would
be red on a pristine tree — i.e. useless.

So the perimeter is split:
  * every OTHER file in the closure  → flat blacklist (kills M-B and every future twin);
  * the declared dual-variant file    → exempted from the blacklist, and pinned instead by
    STRUCTURAL tests (`canPromote` must be a pure conjunction containing `!readOnly`; the
    Vote-2 panels must sit inside `{canPromote && ...}`) plus the RENDER twin in
    `usdcop-trading-dashboard/tests/unit/components/replay-read-only-render.test.tsx`,
    which mounts the replay variant with an ADMIN session and asserts zero action buttons.
    That render test is what actually kills M-A behaviourally; the structural test kills it
    statically and names the line.
Every waiver is itself checked (`test_perimeter_waivers_are_all_still_needed`): a waiver
that stops matching fails, so a stale waiver cannot silently cover the next real offender.

EVASIONS COVERED (learned from the BL-06 defect: exact-literal blacklists are trivial to dodge)
----------------------------------------------------------------------------------------------
The scan runs over a NORMALISED view of each file, not over raw lines:
  ✔ case            `/API/Production/Approve`, `APROBAR`
  ✔ accents         `Aprobación` ≡ `Aprobacion` (NFKD, combining marks dropped)
  ✔ concatenation   `fetch('/api/produc' + 'tion/approve')`, across newlines, N pieces,
                    mixed quote styles (`'a' + "b" + \x60c\x60`), and with comments
                    between the pieces
  ✔ escapes         `'/api/production/appro\x76e'`, `'approve'`
  ✔ template holes  `` `/api/produc${''}tion/approve` `` (static parts are joined; the
                    interpolated code is still scanned, just separately)
  ✔ new directory   any file, anywhere, the moment `app/replay/page.tsx` reaches it
  ✔ opaque imports  `import(name)` is rejected outright instead of silently skipped —
                    static analysis cannot follow it, so it would be a hole in the muralla
NOT COVERED (stated, not hidden — this is defence in depth, not a proof):
  ✘ runtime-assembled specifiers: `String.fromCharCode(...)`, `atob('...')`,
    `['appro','ve'].join('')`, `x['app'+'rove']`, a URL read from config/props at runtime
  ✘ homoglyphs inside a URL path (a Cyrillic 'а' in `/аpi/...` would not match — the
    request would also 404, so it is not a working attack, but the lock does not prove it)
  ✘ a rogue widget added INSIDE the exempted dual-variant file
  ✘ semantics: nothing here proves the buttons work, only that they are absent
The behavioural backstops for the last two are the render test (any rendered button whose
accessible name says aprobar/rechazar/desplegar/promover fails, wherever it was written)
and the Playwright spec.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = (ROOT / "usdcop-trading-dashboard").resolve()

#: Route entry point. `app/replay/page.tsx` IS /replay as far as a user is concerned.
REPLAY_ROUTE = DASHBOARD / "app" / "replay" / "page.tsx"

#: Files that MUST end up in the derived closure. Without this, a resolver that silently
#: stops resolving (a rename, a new path alias, a moved barrel) would shrink the perimeter
#: to nothing and every lock below would pass by covering zero files — the exact vacuity
#: failure mode S-06 documents for /forecasting.
_PERIMETER_ANCHORS = (
    "usdcop-trading-dashboard/app/replay/page.tsx",
    "usdcop-trading-dashboard/components/production/BacktestTerminalPage.tsx",
    "usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx",
    "usdcop-trading-dashboard/components/gm/TerminalShell.tsx",
)

_MIN_PERIMETER_FILES = 15

#: The ONE shared dashboard/replay implementation. Exempted from the flat blacklist and
#: pinned by the structural + render tests instead (see module docstring). Adding an entry
#: here is a deliberate act: it must be a file that /dashboard and /replay genuinely share.
DUAL_VARIANT_FILES: dict[str, str] = {
    "usdcop-trading-dashboard/components/production/ForecastingBacktestSection.tsx":
        "Implementación ÚNICA de /dashboard (aprobación) y /replay (readOnly). Contiene la "
        "maquinaria de Voto 2 por diseño (DRY); su neutralización en /replay se prueba "
        "estructuralmente (canPromote) y en render (replay-read-only-render.test.tsx).",
}

# ---------------------------------------------------------------------------
# Import closure
# ---------------------------------------------------------------------------

# `from '<spec>'`, `import('<spec>')`, `require('<spec>')` — the three spellings that pull
# another module into the surface.
_IMPORT_SPEC = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*)['"]([^'"]+)['"]"""
)
# A dynamic import whose specifier is NOT a literal: `import(name)`, `import(`${b}/x`)`.
# Static analysis cannot follow it ⇒ rejected outright rather than silently skipped.
_OPAQUE_DYNAMIC_IMPORT = re.compile(r"\bimport\s*\(\s*(?!['\"])(?!/\*\s*@vite-ignore)")
#: Comment-only lines: prose is documentation, never a capability (BL-06 precedent).
_COMMENT_LINE = re.compile(r"^\s*(//|\*|/\*|\{/\*)")

_TS_SUFFIXES = (".tsx", ".ts", ".jsx", ".js")


def _resolve_import(spec: str, importer: Path) -> Path | None:
    """Resolve a local module specifier to a file, or None for a package import."""
    if spec.startswith("@/"):
        base = DASHBOARD / spec[2:]
    elif spec.startswith("./") or spec.startswith("../"):
        base = importer.parent / spec
    else:
        return None                      # bare specifier ⇒ node_modules, not ours
    try:
        base = base.resolve()
    except OSError:                      # pragma: no cover — malformed path
        return None
    candidates: list[Path] = []
    if base.suffix in _TS_SUFFIXES:
        candidates.append(base)
    candidates += [base.with_suffix(s) for s in _TS_SUFFIXES]
    candidates += [base / f"index{s}" for s in _TS_SUFFIXES]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _replay_perimeter_files() -> list[Path]:
    """Every source file the /replay route reaches, transitively (sorted, deterministic)."""
    seen: set[Path] = set()
    stack: list[Path] = [REPLAY_ROUTE.resolve()] if REPLAY_ROUTE.is_file() else []
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        text = current.read_text(encoding="utf-8", errors="replace")
        for spec in _IMPORT_SPEC.findall(text):
            target = _resolve_import(spec, current)
            if target and target not in seen and DASHBOARD in target.parents:
                stack.append(target)
    return sorted(seen)


def _rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


# ---------------------------------------------------------------------------
# Normalised scan view — the anti-evasion layer (see module docstring)
# ---------------------------------------------------------------------------

_ESCAPES = {"n": " ", "r": " ", "t": " ", "b": " ", "f": " ", "v": " ", "0": " "}


def _decode_js_escapes(body: str) -> str:
    """Resolve `\\xNN`, `\\uNNNN`, `\\u{...}` and simple backslash escapes.

    `'/api/production/appro\\x76e'` must read as `/api/production/approve`, otherwise the
    blacklist is defeated by one escape sequence.
    """
    out: list[str] = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\" or i + 1 >= n:
            out.append(c)
            i += 1
            continue
        nxt = body[i + 1]
        if nxt == "x" and re.match(r"[0-9a-fA-F]{2}", body[i + 2:i + 4] or ""):
            out.append(chr(int(body[i + 2:i + 4], 16)))
            i += 4
        elif nxt == "u" and body[i + 2:i + 3] == "{":
            end = body.find("}", i + 3)
            hexs = body[i + 3:end] if end != -1 else ""
            if end != -1 and re.fullmatch(r"[0-9a-fA-F]{1,6}", hexs):
                out.append(chr(int(hexs, 16)))
                i = end + 1
            else:                        # pragma: no cover — malformed escape
                out.append(nxt)
                i += 2
        elif nxt == "u" and re.match(r"[0-9a-fA-F]{4}", body[i + 2:i + 6] or ""):
            out.append(chr(int(body[i + 2:i + 6], 16)))
            i += 6
        else:
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
    return "".join(out)


def _skip_ws_and_comments(src: str, i: int) -> int:
    """Index of the next significant character at/after `i` (whitespace + comments skipped).

    Comments matter here: `'/api/produc' /* nope */ + 'tion/approve'` must still merge.
    """
    n = len(src)
    while i < n:
        if src[i].isspace():
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j == -1 else j + 2
        else:
            break
    return i


def _read_string(src: str, i: int) -> tuple[str, str, int]:
    """Read the literal starting at `src[i]`.

    Returns `(static_body, interpolated_code, end_index)`. For template literals the
    static chunks are JOINED (so `` `/api/produc${''}tion/approve` `` collapses to the
    real path) and the `${...}` code is returned separately so it is still scanned, just
    not glued into the path.
    """
    quote = src[i]
    n = len(src)
    j = i + 1
    body: list[str] = []
    interp: list[str] = []
    while j < n:
        c = src[j]
        if c == "\\":
            body.append(src[j:j + 2])
            j += 2
            continue
        if c == quote:
            j += 1
            break
        if c == "\n" and quote != "`":   # unterminated literal — stop at EOL
            break
        if quote == "`" and c == "$" and src[j + 1:j + 2] == "{":
            depth, k = 0, j + 1
            while k < n:
                if src[k] == "{":
                    depth += 1
                elif src[k] == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
                k += 1
            interp.append(src[j + 2:k - 1])
            j = k
            continue
        body.append(c)
        j += 1
    return _decode_js_escapes("".join(body)), " ".join(interp), j


def _scan_view(src: str) -> tuple[str, list[int]]:
    """Return `(text, lines)`: `src` with comments dropped, adjacent string literals merged
    across `+` (any quote style, across newlines, comments between pieces), escapes decoded,
    accents stripped and casefolded. `lines[k]` is the original 1-based line of `text[k]`.

    Comments are DROPPED rather than scanned: explanatory prose naming `/api/production/approve`
    is documentation, not a capability (same rule as BL-06). The docstring of this very module
    would otherwise fail its own lock.
    """
    chars: list[str] = []
    lines: list[int] = []

    def emit(s: str, ln: int) -> None:
        for ch in s:
            nfkd = unicodedata.normalize("NFKD", ch)
            for c2 in nfkd:
                if unicodedata.combining(c2):
                    continue
                for c3 in c2.casefold():
                    chars.append(c3)
                    lines.append(ln)

    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            emit("\n", line)
            line += 1
            i += 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            line += src.count("\n", i, j)
            i = j
            continue
        if c in "\"'`":
            start_line = line
            body, interp, j = _read_string(src, i)
            line += src.count("\n", i, j)
            i = j
            extra = [interp]
            # Fold every `+ '<literal>'` that follows into the SAME body: this is the
            # `'/api/produc' + 'tion/approve'` evasion, and it is the whole reason this
            # function exists instead of a per-line regex.
            while True:
                k = _skip_ws_and_comments(src, i)
                if k >= n or src[k] != "+":
                    break
                k2 = _skip_ws_and_comments(src, k + 1)
                if k2 >= n or src[k2] not in "\"'`":
                    break
                body2, interp2, j2 = _read_string(src, k2)
                line += src.count("\n", i, j2)
                body += body2
                extra.append(interp2)
                i = j2
            emit(body, start_line)
            emit(" ", start_line)        # never glue a literal to the next identifier
            for e in extra:
                if e.strip():
                    emit(e + " ", start_line)
            continue
        emit(c, line)
        i += 1
    return "".join(chars), lines


def _line_of(lines: list[int], idx: int) -> int:
    return lines[idx] if 0 <= idx < len(lines) else 0


# ---------------------------------------------------------------------------
# What /replay may never reach
# ---------------------------------------------------------------------------

#: Endpoints that mutate approval / deployment / execution state. Matched on the
#: normalised, concatenation-collapsed view — the leading slash is deliberately omitted
#: so `'/api' + '/production/approve'` and `` `${base}api/production/approve` `` both hit.
FORBIDDEN_ENDPOINTS = (
    "api/production/approve",
    "api/production/deploy",
    "api/registry/promote",
    "api/execution/",
    "api/trading/order",
)

#: Approval callbacks: the wiring, even when the endpoint string lives elsewhere.
FORBIDDEN_IDENTIFIERS = ("onapprove", "onreject", "handleapprove", "handlereject")

#: Action verbs as VISIBLE TEXT / labels. Word-bounded and matched on the normalised view,
#: so `APROBAR`, `Aprobación` and `aprobar` are the same thing. Deliberately excludes the
#: STATE spellings that legitimately appear all over the closure: `'APPROVED'`,
#: `REJECTED = 'rejected'`, `deployStatus`, `DeployResponse`, `handlePromote` — a state or a
#: symbol name is not a call to action, and blacklisting them would force a false positive
#: on honest code (which is how blacklists get disabled).
FORBIDDEN_ACTION_VERBS = (
    r"\baprobar\b",
    r"\baprueb\w*",
    r"\baprobaci[o]n\b",
    r"\brechazar\b",
    r"\brechaz(a|e|en|ad)\w*",
    r"\bdesplegar\b",
    r"\bdesplieg\w*",
    r"\bpromover\b",
    r"\bpromocionar\b",
    r"\bpromueve\w*",
    r"\bvoto\s*2",
    r"\bapprove\b",
    r"\breject\b",
    r"\bdeploy\b",
    r"\bpromote\b",
)

#: Declared, reasoned waivers on the flat blacklist — file AND pattern, never a whole file
#: and never a whole pattern. Checked for staleness by
#: `test_perimeter_waivers_are_all_still_needed`.
PERIMETER_WAIVERS: dict[tuple[str, str], str] = {
    ("usdcop-trading-dashboard/lib/i18n/gm.ts", r"\baprobaci[o]n\b"):
        "navApproval: 'Aprobación' — etiqueta de NAVEGACIÓN del shell compartido que "
        "enlaza a /dashboard (donde vive el Voto 2). Navegar a la superficie de "
        "aprobación no es aprobar; BL-34 exige exactamente esa separación "
        "(Backtest→/replay, Aprobación→/dashboard).",
}


def _perimeter_offenders(path: Path, *, apply_waivers: bool = True) -> list[str]:
    """'file:line: <what> :: <snippet>' hits inside one perimeter file."""
    rel = _rel(path)
    text, lines = _scan_view(path.read_text(encoding="utf-8", errors="replace"))
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    def snippet(lineno: int) -> str:
        return raw_lines[lineno - 1].strip()[:110] if 0 < lineno <= len(raw_lines) else ""

    offenders: list[str] = []
    for token in FORBIDDEN_ENDPOINTS + FORBIDDEN_IDENTIFIERS:
        for m in re.finditer(re.escape(token), text):
            ln = _line_of(lines, m.start())
            offenders.append(f"{rel}:{ln}: endpoint/callback {token!r} :: {snippet(ln)}")
    for pat in FORBIDDEN_ACTION_VERBS:
        if apply_waivers and (rel, pat) in PERIMETER_WAIVERS:
            continue
        for m in re.finditer(pat, text):
            ln = _line_of(lines, m.start())
            offenders.append(f"{rel}:{ln}: verbo de acción /{pat}/ :: {snippet(ln)}")
    return offenders


# ---------------------------------------------------------------------------
# 1. The perimeter itself
# ---------------------------------------------------------------------------


def test_replay_perimeter_is_derived_not_hardcoded():
    """Meta-lock: the perimeter must really be the import closure of the route.

    A lock over an empty or truncated file set is green by vacuity. This pins that the
    closure is discovered, that it contains the known anchors, and that no file inside it
    hides a module behind a non-literal dynamic import (unfollowable ⇒ a back door).

    ROJO: añadir `const mod = 'x'; import(mod);` a app/replay/page.tsx (import opaco), o
    romper/mover la ruta.
    """
    assert REPLAY_ROUTE.is_file(), (
        f"{_rel(REPLAY_ROUTE)} disappeared — if the route moved, REPLAY_ROUTE must move "
        "with it, or the muralla covers nothing."
    )
    files = _replay_perimeter_files()
    rels = {_rel(f) for f in files}
    missing = [a for a in _PERIMETER_ANCHORS if a not in rels]
    assert not missing, (
        f"The derived /replay perimeter lost {missing}. The import resolver is broken or a "
        "surface file moved; every lock below would silently stop covering it."
    )
    assert len(files) >= _MIN_PERIMETER_FILES, (
        f"The derived perimeter collapsed to {len(files)} files (expected >= "
        f"{_MIN_PERIMETER_FILES}): {sorted(rels)}"
    )
    opaque: list[str] = []
    for f in files:
        src = f.read_text(encoding="utf-8", errors="replace")
        # Checked on the RAW line (comment lines skipped): the distinction between
        # `import('@/x')` and `import(mod)` is the QUOTE, which both _mask_code and
        # _scan_view remove. Same rule as the /forecasting lock.
        for ln, line in enumerate(src.splitlines(), start=1):
            if _COMMENT_LINE.match(line):
                continue
            if _OPAQUE_DYNAMIC_IMPORT.search(line):
                opaque.append(f"{_rel(f)}:{ln}: {line.strip()[:110]}")
    assert not opaque, (
        "A /replay perimeter file imports a module through a non-literal specifier. The "
        "perimeter is computed statically, so such an import is an unauditable back door "
        "into the read-only surface:\n  " + "\n  ".join(opaque)
    )


def test_scan_view_collapses_the_known_evasions():
    """Self-test of the anti-evasion primitive. A blacklist whose measuring tape can be
    bent by a `+` is not a blacklist — BL-06 shipped exactly that. Every string below is a
    concrete mutation that defeated a naive literal scan.

    ROJO: quitar el plegado de concatenación (o el decodificador de escapes) de _scan_view.
    """
    cases = [
        "fetch('/api/produc' + 'tion/approve')",
        "fetch('/API/PRODUC' + 'TION/APPROVE')",
        'fetch("/api/produc" + \'tion/appr\' + "ove")',
        "fetch('/api/produc'\n  + /* mid */ 'tion/approve')",
        "fetch('/api/production/appro\\x76e')",
        "fetch('/api/production/\\u0061pprove')",
        "fetch(`/api/produc${''}tion/approve`)",
    ]
    for src in cases:
        text, _ = _scan_view(src)
        assert "api/production/approve" in text, (
            f"_scan_view failed to collapse a known evasion:\n  {src!r}\n  -> {text!r}"
        )
    # Accents/case on visible verbs.
    text, _ = _scan_view("const label = 'APROBACIÓN pendiente';")
    assert "aprobacion" in text
    # Comments are documentation, not capability (this module's own docstring relies on it).
    text, _ = _scan_view("// llama a /api/production/approve\nconst x = 1;")
    assert "api/production/approve" not in text
    # Line attribution survives the transformation.
    text, lines = _scan_view("const a = 1;\nfetch('/api/produc'\n + 'tion/approve');\n")
    assert _line_of(lines, text.index("api/production/approve")) == 2


# ---------------------------------------------------------------------------
# 2. The flat blacklist (kills M-B)
# ---------------------------------------------------------------------------


def test_replay_perimeter_has_no_action_capabilities():
    """M-B: EVERY file /replay reaches — except the declared dual-variant implementation —
    must stay free of approval/deploy/execution endpoints, approval callbacks and action
    verbs, in any spelling.

    /replay is a research surface: the day it can approve, deploy or even *offer* those
    verbs, the 2-vote gate has a second door that nobody guards (approval-gates.md inv. 3).

    ROJO: crear components/production/RogueApprovalWidget.tsx con botones
      "Aprobar (Voto 2/2)" / "Rechazar" / "Desplegar a producción" y
      `fetch('/api/production/approve', { method: 'POST' })`, importado desde
      app/replay/page.tsx — y también su variante EVASIVA
      `fetch('/api/produc' + 'tion/approve')` con las etiquetas en minúscula.
    """
    offenders: list[str] = []
    for f in _replay_perimeter_files():
        if _rel(f) in DUAL_VARIANT_FILES:
            continue
        offenders.extend(_perimeter_offenders(f))
    assert not offenders, (
        "The /replay read-only surface grew action capabilities (BL-34 / approval-gates.md "
        "invariante 3: los botones de aprobación viven SOLO en /dashboard):\n  "
        + "\n  ".join(offenders)
    )


def test_perimeter_waivers_are_all_still_needed():
    """A waiver nobody re-reads is a waiver that covers the next real offender.

    Both kinds are checked: a per-(file,pattern) waiver must still match something, and a
    dual-variant exemption must (a) still be inside the perimeter and (b) still actually
    contain approval capability — if the shared file stops sharing, the exemption must be
    deleted, not inherited.

    ROJO: borrar el fetch a /api/production/approve de ForecastingBacktestSection.tsx (la
    exención queda obsoleta), o renombrar navApproval en lib/i18n/gm.ts.
    """
    stale: list[str] = []
    for (rel, pat), _reason in PERIMETER_WAIVERS.items():
        path = ROOT / rel
        hits = (
            [o for o in _perimeter_offenders(path, apply_waivers=False) if f"/{pat}/" in o]
            if path.is_file() else []
        )
        if not hits:
            stale.append(f"waiver {rel} :: /{pat}/ ya no coincide con nada")
    perimeter = {_rel(f) for f in _replay_perimeter_files()}
    for rel in DUAL_VARIANT_FILES:
        path = ROOT / rel
        if rel not in perimeter:
            stale.append(
                f"dual-variant {rel} ya no está en el perímetro de /replay — la exención "
                "es innecesaria (o la ruta dejó de compartir implementación)"
            )
            continue
        # "Sigue haciendo falta" se mide por la CAPACIDAD (un endpoint que muta estado de
        # aprobación/deploy/promoción), no por el copy: si la maquinaria se mudó a un
        # fichero solo-/dashboard, la exención debe borrarse, no heredarse.
        if not any(
            "endpoint/callback" in o
            for o in _perimeter_offenders(path, apply_waivers=False)
        ):
            stale.append(
                f"dual-variant {rel} ya no llama a ningún endpoint de aprobación/deploy/"
                "promoción — la exención sobra y debe borrarse (si no, el próximo widget "
                "rogue entra gratis por este fichero)"
            )
    assert not stale, (
        "Waivers obsoletos (bórralos; un waiver rancio tapa silenciosamente al siguiente "
        "infractor):\n  " + "\n  ".join(stale)
    )


# ---------------------------------------------------------------------------
# 3. The dual-variant file: structural gate (kills M-A statically)
# ---------------------------------------------------------------------------

SHARED_SECTION = (
    DASHBOARD / "components" / "production" / "ForecastingBacktestSection.tsx"
)

#: Vote-2 surfaces that may only ever mount behind the gate.
_VOTE2_MOUNTS = ("<ApprovalPanel", "<DeployPanel")


def _mask_code(src: str) -> str:
    """`src` with comments and string bodies blanked out, same length.

    Brace/paren balance is the measuring tape of every structural check below; counting
    braces that live inside a comment or a string makes it trivially bendable (a single
    `/* } */` rebalances any conditional). Blanking keeps every index identical.
    """
    out = list(src)
    n = len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] not in "\r\n":
                out[k] = " "

    i = 0
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
            continue
        if c in "\"'`":
            quote = c
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == quote:
                    j += 1
                    break
                if src[j] == "\n" and quote != "`":
                    break
                j += 1
            blank(i, j)
            i = j
            continue
        i += 1
    return "".join(out)


def _balanced_block(src: str, open_idx: int, opener: str = "{", closer: str = "}") -> tuple[int, int]:
    """(start, end) of the balanced block whose opener sits at `open_idx`."""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == opener:
            depth += 1
        elif src[i] == closer:
            depth -= 1
            if depth == 0:
                return open_idx, i + 1
    return open_idx, len(src)


def test_replay_variant_gate_is_a_pure_conjunction_with_not_readonly():
    """M-A: `canPromote` — the single switch that decides whether /replay renders the
    Vote-2 surface — must be a PURE CONJUNCTION that contains `!readOnly`.

    Pure conjunction is the point, not decoration: `&&` only, no `||`, no ternary, no
    boolean literal. `const canPromote = true` fails (no `!readOnly`); so does the evasive
    `userRole === 'admin' && !readOnly || true`, which keeps the pinned substring while
    inverting the meaning — the exact shape of a mutation that "looks compliant".

    ROJO: `const canPromote = true;`  ·  `const canPromote = userRole === 'admin' && !readOnly || true;`
    """
    assert SHARED_SECTION.is_file(), f"{_rel(SHARED_SECTION)} disappeared"
    masked = _mask_code(SHARED_SECTION.read_text(encoding="utf-8", errors="replace"))
    m = re.search(r"\bcanPromote\s*=\s*([^;\n]*(?:\n(?![ \t]*(?:const|let|var|//))[^;\n]*)*);", masked)
    assert m, (
        "No `canPromote = <expr>;` declaration found in ForecastingBacktestSection.tsx. The "
        "read-only gate must stay a single, greppable expression — if it moved, this lock "
        "and the reviewer both lose the only line that separates /replay from /dashboard."
    )
    expr = re.sub(r"\s+", " ", m.group(1)).strip()
    line = masked.count("\n", 0, m.start()) + 1
    where = f"{_rel(SHARED_SECTION)}:{line}: canPromote = {expr}"
    assert re.search(r"!\s*readOnly\b|readOnly\s*===\s*false", expr), (
        f"{where}\nThe /replay gate no longer negates `readOnly`. M-A verbatim: "
        "`const canPromote = true` mounts the Vote-2 ApprovalPanel + DeployPanel on the "
        "read-only research surface, for ANY role (approval-gates.md invariante 3)."
    )
    for bad, why in (
        (r"\|\|", "un `||` puede reabrir la puerta que el `!readOnly` cierra"),
        (r"\?", "un ternario esconde la rama que realmente decide"),
        (r"\btrue\b", "un literal `true` neutraliza la conjunción entera"),
    ):
        assert not re.search(bad, expr), (
            f"{where}\nEl gate de solo-lectura debe ser una conjunción pura: {why}."
        )
    assert "&&" in expr and "readOnly" in expr, f"{where}\nGate degenerado."


def test_vote2_panels_only_mount_behind_the_gate():
    """The gate is worthless if a panel mounts next to it. Every `<ApprovalPanel` /
    `<DeployPanel` mount must sit INSIDE a `{canPromote && ...}` JSX block.

    Measured on the masked source (comments/strings blanked) so a `/* } */` cannot
    rebalance the braces — the S-07 attack against the forecasting lock.

    ROJO: sacar `<ApprovalPanel .../>` fuera del bloque `{canPromote && (...)}` (o
    cambiarlo por `{approval && (<ApprovalPanel .../>)}`).
    """
    masked = _mask_code(SHARED_SECTION.read_text(encoding="utf-8", errors="replace"))
    gates = [
        _balanced_block(masked, m.start())
        for m in re.finditer(r"\{\s*canPromote\s*&&", masked)
    ]
    assert gates, (
        "No `{canPromote && ...}` JSX gate found — the Vote-2 surface is no longer gated "
        "at all on the shared dashboard/replay implementation."
    )
    ungated: list[str] = []
    for mount in _VOTE2_MOUNTS:
        for m in re.finditer(re.escape(mount) + r"\b", masked):
            if any(a <= m.start() < b for a, b in gates):
                continue
            line = masked.count("\n", 0, m.start()) + 1
            ungated.append(f"{_rel(SHARED_SECTION)}:{line}: {mount} fuera de {{canPromote && …}}")
    assert not ungated, (
        "Vote-2 panels mount outside the read-only gate — /replay would render them "
        "(approval-gates.md invariante 3):\n  " + "\n  ".join(ungated)
    )


# ---------------------------------------------------------------------------
# 4. The read-only note (BL-34's user-visible half of the contract)
# ---------------------------------------------------------------------------

_READONLY_NOTE_TESTID = "replay-readonly-note"


def test_replay_readonly_note_is_present_in_the_perimeter():
    """The note that tells the operator WHERE Vote 2 lives is part of the contract, not
    decoration: without it /replay is a surface that silently drops a capability the user
    expects. It must exist somewhere in the derived perimeter, gated on `readOnly`, and
    keep pointing at /dashboard.

    ROJO: borrar `data-testid="replay-readonly-note"` (o su bloque `{readOnly && (...)}`).
    """
    hosts = [
        f for f in _replay_perimeter_files()
        if _READONLY_NOTE_TESTID in _mask_code(
            f.read_text(encoding="utf-8", errors="replace")
        ) or _READONLY_NOTE_TESTID in f.read_text(encoding="utf-8", errors="replace")
    ]
    assert hosts, (
        f'data-testid="{_READONLY_NOTE_TESTID}" no aparece en NINGÚN fichero del perímetro '
        "de /replay. Es el ancla DOM del Playwright de BL-34 y la única señal al operador "
        "de que el Voto 2 vive en /dashboard."
    )
    host = hosts[0]
    src = host.read_text(encoding="utf-8", errors="replace")
    masked = _mask_code(src)
    idx = src.index(_READONLY_NOTE_TESTID)
    gates = [
        _balanced_block(masked, m.start())
        for m in re.finditer(r"\{\s*readOnly\s*&&", masked)
    ]
    assert any(a <= idx < b for a, b in gates), (
        f"{_rel(host)}: la nota de solo-lectura ya no está dentro de un bloque "
        "`{readOnly && …}` — o se muestra también en /dashboard (donde es falsa), o dejó "
        "de depender de la variante."
    )
    assert "/dashboard" in src, (
        f"{_rel(host)}: la nota de solo-lectura ya no nombra /dashboard — decirle al "
        "operador que no puede aprobar SIN decirle dónde sí, es media verdad."
    )


@pytest.mark.parametrize("rel", sorted(DUAL_VARIANT_FILES))
def test_dual_variant_file_declares_the_readonly_prop(rel: str):
    """The exemption is only legitimate while the file really is variant-aware: it must
    accept a `readOnly` prop. A shared file that stopped reading the variant cannot be
    trusted to neutralise anything.

    ROJO: quitar la prop `readOnly` de la firma de ForecastingBacktestSection.
    """
    src = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    assert re.search(r"readOnly\s*[?:=]", src), (
        f"{rel} ya no declara la prop `readOnly`: la exención del blacklist se apoya en "
        "que este fichero distingue /dashboard de /replay. Si no la distingue, la "
        "maquinaria de aprobación no puede vivir aquí."
    )
