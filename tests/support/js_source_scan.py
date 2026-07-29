"""Static-scan primitives for JS/TS sources — the shared measuring tape of the CI murallas.

WHY THIS MODULE EXISTS
----------------------
Two regression locks derive a route's perimeter by import closure and then look for
capabilities that must not be there:

  * `tests/regression/test_forecasting_caveat_present.py`  (BL-06 — /forecasting is a
    diagnostic surface: no approve/deploy/execute wiring, no order verbs)
  * `tests/regression/test_replay_is_read_only.py`         (BL-34 — /replay is read-only:
    the Vote-2 machinery lives only on /dashboard)

Both need the SAME primitive: a normalised VIEW of a source file in which the cheap
obfuscations are already collapsed, so the search itself can stay a simple substring /
word-bounded regex. BL-34 grew that primitive; BL-06 shipped without it and was defeated by
three characters of concatenation:

    fetch(`${P}/appro` + 've')          // endpoint blacklist: no match
    fetch('/api/exec' + 'ution/orders') // endpoint blacklist: no match
    <button>Comprar ahora</button>      // verb blacklist was UPPERCASE-only: no match

Keeping one copy per lock would mean two implementations of one concept, and then the
question "which of the two is the real rule?" has no answer (K-035). It lives here once.

WHAT `scan_view` COLLAPSES
--------------------------
  ✔ case              `/API/Production/Approve`, `APROBAR`, `Comprar`
  ✔ accents           `Aprobación` ≡ `Aprobacion` (NFKD, combining marks dropped)
  ✔ concatenation     `'/api/produc' + 'tion/approve'` — N pieces, mixed quote styles,
                      across newlines, with comments between the pieces
  ✔ escapes           `'appro\\x76e'`, `'\\u0061pprove'`, `'appro\\u{76}e'`
  ✔ template holes    `` `/api/produc${''}tion/approve` `` (static chunks are joined)
  ✔ const indirection `const P = '/api/production'; fetch(`${P}/appro` + 've')` — a
                      `${IDENT}` whose identifier is bound EXACTLY ONCE in the file to a
                      literal string is substituted INLINE, so the pieces meet. Without
                      this, one extra variable defeats the whole tape, and that is the
                      cheapest evasion there is. Any other interpolation is kept and
                      scanned SEPARATELY, never glued into the path, so `${runtimeBase}`
                      cannot be forged into a false positive.
  ✔ comments dropped  prose naming an endpoint is documentation, not a capability (both
                      locks depend on this: their own docstrings name the endpoints)

WHAT IT DOES NOT COLLAPSE (stated, not hidden — defence in depth, not a proof)
-----------------------------------------------------------------------------
  ✘ runtime-assembled strings: `String.fromCharCode(...)`, `atob('...')`,
    `['appro','ve'].join('')`, `x['app'+'rove']` as a computed member, a URL that arrives
    from config/props/env at runtime
  ✘ indirection the const pass cannot see: an identifier assigned more than once (dropped
    on purpose — a second binding makes the value ambiguous and guessing would invent
    false positives), one imported from another module, a chain
    `const A='/api'; const B=`${A}/production`` (substitution is single-pass, not
    recursive), or an object/array member `CFG.base`
  ✘ homoglyphs (a Cyrillic `а` in `/аpi/…` survives NFKD unchanged)
  ✘ semantics of any kind — absence of a token is not absence of a capability

Line attribution survives every transformation: `scan_view` returns `(text, lines)` where
`lines[k]` is the original 1-based line of `text[k]`, so a hit still names a real line.
"""
from __future__ import annotations

import re
import unicodedata

__all__ = [
    "scan_view",
    "line_of",
    "decode_js_escapes",
    "read_string",
    "read_folded_string",
    "string_constants",
    "skip_ws_and_comments",
    "mask_code",
]

#: Escapes that carry no textual content — folded to a space so `'a\nb'` cannot glue
#: `a` and `b` into a token that neither piece spells.
_ESCAPES = {"n": " ", "r": " ", "t": " ", "b": " ", "f": " ", "v": " ", "0": " "}


def decode_js_escapes(body: str) -> str:
    """Resolve `\\xNN`, `\\uNNNN`, `\\u{...}` and simple backslash escapes.

    `'/api/production/appro\\x76e'` must read as `/api/production/approve`, otherwise a
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


def skip_ws_and_comments(src: str, i: int) -> int:
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


#: A `${...}` hole whose content is a single bare identifier — the only shape the const
#: pass will substitute inline. Anything with a call, a member access or an operator stays
#: opaque and is scanned separately.
_BARE_IDENT = re.compile(r"^[A-Za-z_$][\w$]*$")

#: `const NAME =` / `let` / `var` immediately followed by a string literal.
_STR_BINDING = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?=['\"`])")


def read_string(
    src: str, i: int, consts: dict[str, str] | None = None
) -> tuple[str, str, int]:
    """Read the literal starting at `src[i]`.

    Returns `(static_body, interpolated_code, end_index)`. For template literals the static
    chunks are JOINED (so `` `/api/produc${''}tion/approve` `` collapses to the real path).
    A `${...}` hole is either substituted inline (bare identifier present in `consts`) or
    returned separately so it is still scanned, just not glued into the path.
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
            code = src[j + 2:k - 1]
            name = code.strip()
            if consts and _BARE_IDENT.match(name) and name in consts:
                # `const P = '/api/production'` + `` `${P}/appro` `` must MEET, or one
                # extra variable defeats the whole tape.
                body.append(consts[name])
            else:
                interp.append(code)
            j = k
            continue
        body.append(c)
        j += 1
    return decode_js_escapes("".join(body)), " ".join(interp), j


def read_folded_string(
    src: str, i: int, consts: dict[str, str] | None = None
) -> tuple[str, list[str], int]:
    """Read the literal at `src[i]` PLUS every `+ '<literal>'` that follows it.

    This is the `'/api/produc' + 'tion/approve'` evasion, and the reason the scan cannot be
    a per-line regex: the pieces may use different quote styles, sit on different lines and
    have comments between them. Returns `(joined_body, interpolated_code_chunks, end)`.
    """
    body, interp, j = read_string(src, i, consts)
    extras = [interp]
    n = len(src)
    while True:
        k = skip_ws_and_comments(src, j)
        if k >= n or src[k] != "+":
            break
        k2 = skip_ws_and_comments(src, k + 1)
        if k2 >= n or src[k2] not in "\"'`":
            break
        body2, interp2, j = read_string(src, k2, consts)
        body += body2
        extras.append(interp2)
    return body, extras, j


def string_constants(src: str) -> dict[str, str]:
    """Module-level-ish `const NAME = '<string>'` bindings, for `${NAME}` substitution.

    Deliberately conservative — this map exists to close ONE evasion (an extra variable
    between the pieces), not to emulate a JS engine:

    * only bindings whose value STARTS as a string literal are recorded (the folded
      `'a' + "b"` chain counts);
    * an identifier bound MORE THAN ONCE is dropped entirely. Guessing which binding wins
      would invent false positives, and a lock that cries wolf gets deleted;
    * substitution is single-pass: a const built from another const stays unresolved.

    Comments are skipped so a commented-out binding cannot poison the map, and no scoping
    is modelled: file-wide uniqueness is the (stated) approximation.
    """
    values: dict[str, str] = {}
    dropped: set[str] = set()
    i, n = 0, len(src)
    while i < n:
        j = skip_ws_and_comments(src, i)
        if j != i:
            i = j
            continue
        if src[i] in "\"'`":             # skip over literals: no bindings inside them
            _, _, i = read_string(src, i)
            continue
        m = _STR_BINDING.match(src, i)
        if not m:
            i += 1
            continue
        name = m.group(1)
        body, _extras, end = read_folded_string(src, m.end())
        if name in values and values[name] != body:
            dropped.add(name)
        values[name] = body
        i = end
    for name in dropped:
        values.pop(name, None)
    return values


def scan_view(src: str) -> tuple[str, list[int]]:
    """Return `(text, lines)`: `src` with comments dropped, adjacent string literals merged
    across `+` (any quote style, across newlines, comments between pieces), escapes decoded,
    accents stripped and casefolded. `lines[k]` is the original 1-based line of `text[k]`.

    Comments are DROPPED rather than scanned: explanatory prose naming
    `/api/production/approve` is documentation, not a capability. Both locks rely on this —
    their own module docstrings would otherwise fail their own rule.
    """
    consts = string_constants(src)
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
            # Fold the whole `'a' + "b" + `c`` chain into ONE body (see read_folded_string).
            body, extra, j = read_folded_string(src, i, consts)
            line += src.count("\n", i, j)
            i = j
            emit(body, start_line)
            emit(" ", start_line)        # never glue a literal to the next identifier
            for e in extra:
                if e.strip():
                    emit(e + " ", start_line)
            continue
        emit(c, line)
        i += 1
    return "".join(chars), lines


def line_of(lines: list[int], idx: int) -> int:
    """1-based original line of `text[idx]` (0 when the index is out of range)."""
    return lines[idx] if 0 <= idx < len(lines) else 0


# ---------------------------------------------------------------------------
# Structural view: same length, comments and string BODIES blanked
# ---------------------------------------------------------------------------


def mask_code(src: str) -> str:
    """Return `src` with comments and string bodies blanked out, SAME LENGTH.

    Brace/paren balance is the measuring tape of every structural check in both locks
    (`{canPromote && …}` gating, JSX depth of a banner). Counting braces that live inside a
    comment or a string makes that tape trivially bendable — the demonstrated attack was a
    single comment character:

        {__isInternal && (
          /* } el candado cuenta llaves literales, tambien en comentarios */
          <ForecastDisclaimer variant="weekly" />
        )}

    which rebalances the count to 0 while the banner is invisible to everyone who is not an
    admin. Blanking (rather than deleting) keeps every character index identical, so callers
    can keep using positions from the ORIGINAL source.

    Template literals keep their `${...}` interpolations visible — they are real code with
    real braces — and are masked recursively.
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
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            blank(i, j)
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == c:
                    j += 1
                    break
                if src[j] == "\n":       # unterminated literal — stop at EOL
                    break
                j += 1
            blank(i, j)
            i = j
            continue
        if c == "`":
            blank(i, i + 1)
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    blank(j, j + 2)
                    j += 2
                    continue
                if src[j] == "`":
                    blank(j, j + 1)
                    j += 1
                    break
                if src[j] == "$" and j + 1 < n and src[j + 1] == "{":
                    blank(j, j + 1)      # the '$' is text; '{...}' is code
                    depth = 0
                    k = j + 1
                    while k < n:
                        if src[k] == "{":
                            depth += 1
                        elif src[k] == "}":
                            depth -= 1
                            if depth == 0:
                                k += 1
                                break
                        k += 1
                    out[j + 1:k] = list(mask_code(src[j + 1:k]))
                    j = k
                    continue
                blank(j, j + 1)
                j += 1
            i = j
            continue
        i += 1
    return "".join(out)
