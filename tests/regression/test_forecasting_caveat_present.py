"""The forecasting surfaces must carry the diagnostic caveat.

Contract: CTR-QUANT-CONSTITUTION-001

The model zoo's directional accuracy is ~0.52 with the best model at p=0.11 unadjusted and
p~0.66 adjusted for the 9 models tried; the best model-by-horizon cell reaches p_adj = 1.0
over the 63 cells examined. The dashboard nonetheless displayed "Direction Accuracy: 52%" with
zero context, which reads as "the models work" to anyone who does not carry the significance
tables in their head.

Worse: the zoo family is NOT purely informational — `train_and_export_smart_simple.py` derives
an executed trade direction from the same model family. The caveat is the line between a
diagnostic surface and an implied recommendation.

This is a source-level check (the dashboard is a standalone build; runtime rendering is not
reachable from this suite). It pins the presence of the caveat markup in both surfaces that
show DA. Removing the banner makes this fail, which is the point: a disclaimer that can be
silently deleted is decoration, not disclosure.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "usdcop-trading-dashboard" / "components"

FORECASTING_VIEW = DASH / "gm" / "views" / "ForecastingView.tsx"
LEGACY_DASHBOARD = DASH / "forecasting" / "ForecastingDashboard.tsx"
LEGACY_WEEKLY = DASH / "forecasting" / "WeeklyInferenceView.tsx"

# BL-04 moved the caveat copy/testid to a shared SSOT constant; CXD-032 moved the
# BANNER itself to a shared component consuming that SSOT.
DISCLAIMER_SSOT = (
    ROOT / "usdcop-trading-dashboard" / "lib" / "ui" / "forecast-disclaimer.ts"
)
DISCLAIMER_COMPONENT = DASH / "forecasting" / "ForecastDisclaimer.tsx"

# Every surface that shows a forecast/DA number must reference the SHARED banner
# (CXD-032: "componente compartido ForecastDisclaimer montado en cada superficie").
# The legacy dashboard wraps it in DiagnosticCaveat to append the derived-DA line.
SURFACES = {
    "forecasting/ForecastingDashboard.tsx": ("DiagnosticCaveat", "ForecastDisclaimer"),
    "forecasting/WeeklyInferenceView.tsx": ("ForecastDisclaimer",),
    "gm/views/ForecastingView.tsx": ("ForecastDisclaimer",),
}

# Tokens that mean "this file renders a forecast-quality number" (DA in any of its
# spellings, incl. the weekly-inference DA 2025 tiles).
_DA_PRESENCE_TOKENS = ("direction_accuracy", "Direction Accuracy", "da_2025_pct", "DA 2025")


@pytest.mark.parametrize("rel,markers", SURFACES.items(), ids=list(SURFACES))
def test_da_surface_carries_caveat(rel: str, markers: tuple):
    p = DASH / rel
    if not p.is_file():
        pytest.skip(f"{rel} absent")
    src = p.read_text(encoding="utf-8", errors="replace")
    if not any(t in src for t in _DA_PRESENCE_TOKENS):
        pytest.skip(f"{rel} no longer shows DA")
    assert all(m in src for m in markers), (
        f"{rel} displays Direction Accuracy but the shared caveat ({markers!r}) is gone. "
        "A ~52% DA shown without context reads as 'the models work'; the statistics say "
        "coin flip (p_adj 0.66 across models, 1.0 across model-by-horizon cells)."
    )


# ---------------------------------------------------------------------------
# BL-01 — the caveat banner itself (testid + honest NO-SIGNAL clause), and its
# gating. Hardened after Codex review: the first version pinned only the prefix
# "Superficie de diagn", so a deceptive mutation like
#   'Superficie de diagnóstico: señal validada, opere con confianza.'
# passed. The lock now (a) parses the SSOT string constants, (b) requires the
# FULL no-signal clauses, and (c) rejects promotional/action language outright.
# ---------------------------------------------------------------------------

def _mask_code(src: str) -> str:
    """Return `src` with comments and string bodies blanked out, same length.

    S-07: every structural check below reasons about JSX brace balance
    (`prefix.count("{") - prefix.count("}")`). Counting braces that live inside a
    comment or a string literal makes the check trivially defeatable — the
    demonstrated attack was one comment character:

        {__isInternal && (
          /* } el candado cuenta llaves literales, tambien en comentarios */
          <ForecastDisclaimer variant="weekly" />
        )}

    which rebalances the count to 0 while the banner is invisible to everyone who
    is not an admin. Blanking (rather than deleting) keeps every character index
    identical, so the callers can keep using positions from the original source.

    Template literals keep their `${...}` interpolations visible (they are real
    code, with real braces) and are masked recursively.
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
                    out[j + 1:k] = list(_mask_code(src[j + 1:k]))
                    j = k
                    continue
                blank(j, j + 1)
                j += 1
            i = j
            continue
        i += 1
    return "".join(out)


def _jsx_depth_from_enclosing_return(src: str, pos: int) -> int | None:
    """Brace depth of `pos` relative to its enclosing `return (`, comments and
    strings excluded. `None` when there is no enclosing JSX return."""
    masked = _mask_code(src)
    ret = masked.rfind("return (", 0, pos)
    if ret == -1:
        return None
    prefix = masked[ret + len("return ("): pos]
    return prefix.count("{") - prefix.count("}")


def test_brace_depth_ignores_comments_and_strings():
    """Self-test of the structural primitive (S-07). A lock whose measuring tape
    can be bent by a comment is not a lock — this pins the tape."""
    honest = 'function C(){\n  return (\n    <div>\n      <Banner />\n    </div>\n  );\n}'
    assert _jsx_depth_from_enclosing_return(honest, honest.index("<Banner")) == 0
    # The exact demonstrated attack: a conditional wrapper rebalanced by a `}`
    # written inside a comment.
    attacked = (
        'function C(){\n  return (\n    <div>\n      {isInternal && (\n'
        '        /* } rebalanceo */\n        <Banner />\n      )}\n    </div>\n  );\n}'
    )
    assert _jsx_depth_from_enclosing_return(attacked, attacked.index("<Banner")) == 1
    # Same trick with a string literal instead of a comment.
    via_string = (
        'function C(){\n  return (\n    <div>\n      {isInternal && (\n'
        '        <Banner title="}" />\n      )}\n    </div>\n  );\n}'
    )
    assert _jsx_depth_from_enclosing_return(via_string, via_string.index("<Banner")) == 1
    # Template interpolation is CODE: its braces must still count.
    tmpl = (
        'function C(){\n  return (\n    <div>\n'
        '      {list.map((x) => (\n        <Banner k={`a${x.b}`} />\n      ))}\n'
        '    </div>\n  );\n}'
    )
    assert _jsx_depth_from_enclosing_return(tmpl, tmpl.index("<Banner")) == 1


def _norm(s: str) -> str:
    """Accent-stripped, casefolded, whitespace-collapsed — so 'SEÑAL'/'señal'/'senal'
    all compare equal and a mutation cannot hide behind diacritics or case."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).casefold().strip()


# Every `export const NAME = <value>;` in the module — value captured up to the
# first bare `;` (the SSOT is concatenated string literals by design).
_TS_CONST = re.compile(r"export\s+const\s+(\w+)\s*=\s*([^;]*);")

# A string literal in ANY of the three TS spellings: '...', "...", `...`.
# Red-team bypass #1: the previous parser only understood single quotes, so a
# new constant written with double quotes or backticks was invisible to every
# check below. All three forms are now first-class.
_TS_STR_PIECE = re.compile(
    r"'(?:[^'\\]|\\.)*'"
    r'|"(?:[^"\\]|\\.)*"'
    r"|`(?:[^`\\]|\\.)*`"
)


def _ts_string_constants(src: str) -> dict[str, str]:
    """Parse every `export const NAME = <string literals joined by +>;`.

    Zero silent escapes (red-team bypass #1): if the module contains ANY
    `export const` whose value cannot be statically extracted as a pure
    concatenation of string literals (non-string expression, function call,
    template literal with `${...}` interpolation, ...), this fails loudly
    instead of silently checking nothing.
    """
    out: dict[str, str] = {}
    unparsed: list[str] = []
    for name, raw in _TS_CONST.findall(src):
        pieces = _TS_STR_PIECE.findall(raw)
        residue = re.sub(r"[+\s]", "", _TS_STR_PIECE.sub("", raw))
        interpolated = [p for p in pieces if p.startswith("`") and "${" in p]
        if not pieces or residue or interpolated:
            unparsed.append(name)
            continue
        # Strip the quotes and resolve simple backslash escapes (\', \", \`, \\).
        out[name] = "".join(re.sub(r"\\(.)", r"\1", p[1:-1]) for p in pieces)
    assert not unparsed, (
        f"forecast-disclaimer.ts exports constants {unparsed} whose values could "
        "not be statically extracted as plain string literals. The disclaimer "
        "SSOT must stay statically verifiable — every export const must be a "
        "concatenation of '...'/\"...\"/`...` literals with no interpolation "
        "(BL-01, zero silent escapes)."
    )
    return out


# The complete honest clauses (normalized). Full sentences, NOT prefixes — the
# entire point is that 'Superficie de diagnóstico' + marketing tail must fail.
REQUIRED_CLAUSES = {
    "FORECAST_DISCLAIMER_HEADLINE": ["no es una senal de inversion"],
    "FORECAST_DISCLAIMER_ZOO_TITLE": ["superficie de diagnostico, no de senales"],
    "FORECAST_DISCLAIMER_DIRECTIONAL_TITLE": ["no senal ejecutable"],
    # The body must keep the coin-flip honesty, not just any statistics-sounding prose.
    "FORECAST_DISCLAIMER_ZOO_BODY": ["indistinguible de una moneda al aire"],
    # CXD-032 (2nd rejection): the rule-based weekly surface (Gold/BTC) needs its OWN
    # branch. It does not run the model zoo, so the zoo body was a false statement about
    # the nature of the product, not a stylistic detail.
    "FORECAST_DISCLAIMER_WEEKLY_TITLE": ["superficie de diagnostico, no de senales"],
    "FORECAST_DISCLAIMER_WEEKLY_BODY": [
        "basada en reglas",
        "no hay conjunto de modelos ml ni probabilidad calibrada",
    ],
}

# Promotional / action language that inverts the disclaimer's meaning. Checked on
# normalized text. Each pattern was chosen against concrete attack strings that
# defeated earlier versions of this lock:
#   M1: 'Superficie de diagnóstico: señal validada, opere con confianza.'
#   M2: 'NO ES UNA SEÑAL CUALQUIERA: ES NUESTRA SEÑAL DE COMPRA MÁS CONFIABLE'
#   M3: 'Precisión direccional del 90% garantizada. Ejecute estas señales...'
#   M4 (red-team bypass #3, thesaurus attack): 'Invierta ya: señal de entrada
#       alcista confirmada, precisión demostrada.'
#
# HONEST LIMIT (defense in DEPTH, not completeness): a blacklist can never beat
# the thesaurus — Spanish has unbounded ways to phrase "act on this". These
# patterns raise the cost of a deceptive mutation; they are NOT the guarantee.
# The hard guarantee is REQUIRED_CLAUSES above: the complete no-signal clauses
# must be present verbatim (normalized), so any surviving synonym attack still
# ships inside a banner that states, in full, that this is not an investment
# signal. Do not read this list as "everything not matched is fine".
#
# Two scopes:
#   FORBIDDEN_MARKETING          — applied to the SSOT constants AND to the JSX
#                                  content of the forecasting surface files.
#   FORBIDDEN_MARKETING_SSOT_ONLY — broader words that the disclaimer copy must
#                                  never contain, but that the surfaces use
#                                  legitimately in diagnostic labels
#                                  ('Confianza proxy', 'Sin posición').
FORBIDDEN_MARKETING = [
    # Imperative action verbs (financial CTA style).
    r"\bopere\b",
    r"\bejecute\b",
    r"\bcompre\b",
    r"\bvenda\b",
    r"\binviert\w*",                       # invierta / inviertan / invierte (ya)
    r"\bactu(e|a|en|ad)\b",                # actúe/actúa ahora (not 'actual...')
    r"apuest",                             # apueste al alza / apuesta segura
    # Taking a position, phrased any common way.
    r"(tome|toma|abra|abre|entre en) (una )?posicion",
    r"posicion (larga|corta|alcista|bajista)",
    # Signal-flavored nouns with promotional qualifiers.
    r"(senal(es)?|punto(s)?) de (compra|venta|entrada|inversion segura)",
    r"senal(es)? (validada|confiable|segura|fuerte|ganadora|comprobada)",
    r"es (nuestra|la) (mejor )?senal",
    r"(alcista|bajista|tendencia|senal|direccion)e?s? confirmad",
    # Certainty / performance promises.
    r"demostrad",
    r"garantiz",
    r"validad",
    r"rentabilidad",
    r"recomendad",
    r"sin riesgo",
    r"asegurad",
    r"confiab",
    r"con (plena |total )?confianza",
]

# Only for the disclaimer SSOT: the caveat copy has no legitimate use for these,
# but the surfaces do ('Confianza proxy' is the mandated BL-03 label; 'Sin
# posición' is the honest flat state), so applying them to JSX would force the
# surfaces to drop honest diagnostic wording.
FORBIDDEN_MARKETING_SSOT_ONLY = [
    r"confianza",
    r"\bposicion(es)?\b",
    r"\bentrada\b",
    r"confirmad",
]


def _disclaimer_constants() -> dict[str, str]:
    consts = _ts_string_constants(
        DISCLAIMER_SSOT.read_text(encoding="utf-8", errors="replace")
    )
    missing = [k for k in REQUIRED_CLAUSES if k not in consts]
    assert not missing, (
        f"forecast-disclaimer.ts no longer exports {missing} as plain string "
        "constants — the disclaimer SSOT must stay statically verifiable (BL-01)."
    )
    return consts


def test_disclaimer_copy_carries_no_hardcoded_numbers():
    """CXD-032 (2nd rejection): NO disclaimer constant may contain a digit.

    The rejected version froze '≈52%' and '9 modelos' into the shared zoo body. That copy
    is (at best) true for the USD/COP model zoo and false everywhere else it was mounted:
    BTC's measured DA is ≈0.46, and the Gold/BTC weekly surfaces are rule-based policies
    that run no zoo at all. A performance number is only honest where it is DERIVED from
    the data of the surface showing it (the legacy dashboard's `useMemo` line is the
    correct pattern, and it is pinned by test_caveat_is_not_hardcoded_to_a_stale_number).

    Banning digits outright is the only version of this rule that cannot be bypassed by
    moving the stale number into a new constant or a different variant.
    """
    consts = _disclaimer_constants()
    offenders = [
        f"{name}: {value!r}"
        for name, value in sorted(consts.items())
        if name != "FORECAST_DISCLAIMER_TESTID" and any(ch.isdigit() for ch in value)
    ]
    assert not offenders, (
        "Disclaimer copy hardcodes a performance/quantity figure. Every number shown next "
        "to the caveat must be derived from the data of THAT surface, or not shown "
        "(quant-constitution: ningun numero de performance sin su fuente publicada):\n  "
        + "\n  ".join(offenders)
    )


def test_shared_disclaimer_component_is_ssot_and_unconditional():
    """CXD-032: the shared <ForecastDisclaimer/> component exists, consumes the SSOT
    (testid + headline + BOTH title/body pairs) and renders unconditionally — no
    `return null`, no hidden/aria-hidden attributes, no display:none, and the testid
    sits at brace-depth 0 of its JSX return (no `{cond && ...}` wrapper)."""
    assert DISCLAIMER_COMPONENT.is_file(), (
        "components/forecasting/ForecastDisclaimer.tsx disappeared — the shared banner "
        "is the root fix for CXD-032 (BL-02/BL-04); surfaces must not re-inline copy."
    )
    src = DISCLAIMER_COMPONENT.read_text(encoding="utf-8", errors="replace")
    for const in (
        "FORECAST_DISCLAIMER_TESTID",
        "FORECAST_DISCLAIMER_HEADLINE",
        "FORECAST_DISCLAIMER_ZOO_TITLE",
        "FORECAST_DISCLAIMER_ZOO_BODY",
        "FORECAST_DISCLAIMER_WEEKLY_TITLE",
        "FORECAST_DISCLAIMER_WEEKLY_BODY",
        "FORECAST_DISCLAIMER_DIRECTIONAL_TITLE",
        "FORECAST_DISCLAIMER_DIRECTIONAL_BODY",
    ):
        assert const in src, (
            f"ForecastDisclaimer.tsx no longer consumes {const} — headline AND body must "
            "come from lib/ui/forecast-disclaimer.ts (BL-04: no hardcoded copy)."
        )
    assert "forecast-disclaimer" in src, "ForecastDisclaimer.tsx must import the SSOT module"
    assert "return null" not in src, (
        "ForecastDisclaimer.tsx grew a `return null` branch — the shared banner must be "
        "impossible to suppress from inside (BL-02: 'hidden por rama' is the rejected bug)."
    )
    for tok in (" hidden", "aria-hidden", "display:none", "display: 'none'", "visibility"):
        assert tok not in src, (
            f"ForecastDisclaimer.tsx contains {tok!r} — the shared banner must not ship "
            "its own hiding mechanism (CXD-032 mutation: hidden/CSS => rojo)."
        )
    m = re.search(r"data-testid=\{FORECAST_DISCLAIMER_TESTID\}", src)
    assert m, "ForecastDisclaimer.tsx lost the SSOT testid attribute"
    # Depth measured with comments and strings masked (S-07): counting literal
    # braces let a single `/* } */` rebalance any conditional wrapper.
    depth = _jsx_depth_from_enclosing_return(src, m.start())
    assert depth is not None, "testid appears outside the component's JSX return"
    assert depth <= 1, (
        # depth 1 = inside the banner's own opening tag attribute braces; anything
        # deeper means a conditional JSX expression wraps the banner.
        f"The banner markup inside ForecastDisclaimer.tsx is nested in a JSX "
        f"conditional (depth {depth})."
    )


def test_caveat_banner_present():
    """BL-01: the GM forecasting view carries the da-caveat banner (now via the
    shared component) AND the SSOT still carries the complete no-signal clauses
    (not just their prefixes)."""
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    has_banner = (
        'data-testid="da-caveat"' in src
        or "FORECAST_DISCLAIMER_TESTID" in src
        or "ForecastDisclaimer" in src
    )
    assert has_banner, (
        "ForecastingView.tsx lost the da-caveat banner (no literal testid, no "
        "FORECAST_DISCLAIMER_TESTID, no shared <ForecastDisclaimer/>). The DA surface "
        "must not render without its diagnostic disclaimer (BL-01)."
    )
    # BL-04: the honest copy lives in the shared SSOT; the view carries it inline,
    # imports the SSOT module, or mounts the shared component (which the companion
    # test pins to the SSOT).
    ssot_src = DISCLAIMER_SSOT.read_text(encoding="utf-8", errors="replace")
    assert "Superficie de diagn" in ssot_src, (
        "lib/ui/forecast-disclaimer.ts lost the honest phrase ('Superficie de "
        "diagnóstico, no de señales') — the disclaimer text is part of the contract, "
        "not decoration (BL-01/BL-04)."
    )
    assert (
        "Superficie de diagn" in src
        or "forecast-disclaimer" in src
        or "ForecastDisclaimer" in src
    ), (
        "ForecastingView.tsx neither inlines the honest phrase nor references the "
        "disclaimer SSOT/shared component (BL-01/BL-04)."
    )
    consts = _disclaimer_constants()
    # The testid indirection must still resolve to the pinned testid.
    assert consts.get("FORECAST_DISCLAIMER_TESTID") == "da-caveat", (
        "FORECAST_DISCLAIMER_TESTID no longer resolves to 'da-caveat' — the e2e/DOM "
        "anchor of the disclaimer would silently detach (BL-01)."
    )
    # Full no-signal clauses, normalized — a prefix plus a marketing tail fails here.
    for name, clauses in REQUIRED_CLAUSES.items():
        text = _norm(consts[name])
        for clause in clauses:
            assert clause in text, (
                f"{name} lost the required clause {clause!r}. The disclaimer must state "
                "the complete no-signal meaning; a truncated or reworded version is a "
                "different contract (BL-01, Codex review: prefix-only pin rejected)."
            )


def test_caveat_copy_resists_deceptive_mutation():
    """BL-01 (hardening): the disclaimer copy must not contain promotional or
    action language. Guards against mutations that keep the pinned honest prefix
    but invert the meaning ('señal validada, opere con confianza', 'garantizada',
    'ejecute', 'señal de compra', ...). Attack strings M1/M2/M3 that defeated the
    prefix-only lock all fail here."""
    consts = _disclaimer_constants()
    offenders: list[str] = []
    for name in sorted(k for k in consts if k != "FORECAST_DISCLAIMER_TESTID"):
        text = _norm(consts[name])
        for pat in FORBIDDEN_MARKETING + FORBIDDEN_MARKETING_SSOT_ONLY:
            if re.search(pat, text):
                offenders.append(f"{name}: /{pat}/ -> {consts[name]!r}")
    assert not offenders, (
        "Disclaimer copy contains promotional/action language — a diagnostic caveat "
        "that recommends acting is worse than no caveat:\n  " + "\n  ".join(offenders)
    )


# Characters the disclaimer copy may use: printable ASCII plus the Spanish
# repertoire and the few typographic symbols the honest copy actually needs.
# Anything else — Cyrillic/Greek homoglyphs, zero-width characters, combining
# marks — is an evasion vector: 'о' (U+043E CYRILLIC SMALL LETTER O) survives
# NFKD unchanged, so 'cоnfianza' sails past every substring/regex check above
# (red-team bypass #2). A whitelist is the only shape of this check that fails
# closed on the next homoglyph instead of enumerating Unicode.
_ALLOWED_NON_ASCII = set("áéíóúñüÁÉÍÓÚÑÜ¿¡—–·≈’«»°")


def test_caveat_copy_uses_only_whitelisted_characters():
    """BL-01 (hardening, red-team bypass #2): every disclaimer constant must be
    built exclusively from printable ASCII + the whitelisted Spanish/typographic
    repertoire. A single Cyrillic/Greek homoglyph or invisible (zero-width)
    character fails with its codepoint named."""
    consts = _disclaimer_constants()
    offenders: list[str] = []
    for name in sorted(consts):
        for ch in consts[name]:
            if 0x20 <= ord(ch) <= 0x7E or ch in _ALLOWED_NON_ASCII:
                continue
            offenders.append(
                f"{name}: U+{ord(ch):04X} "
                f"{unicodedata.name(ch, '<unnamed>')} ({ch!r})"
            )
    assert not offenders, (
        "Disclaimer copy contains characters outside the whitelist — homoglyphs "
        "and invisible characters are how a deceptive mutation evades the "
        "normalized blacklist (e.g. 'cоnfianza' with U+043E):\n  "
        + "\n  ".join(offenders)
    )


# Banner mount in EVERY spelling: literal testid, the shared SSOT constant, or the
# shared component (CXD-032). (Codex review 2026-07-27 rejected the old literal-only
# count as "universalidad sorteable por constante"; CXD-032 rejected the old
# view-only check as "hidden por rama pasa" — now every surface is pinned.)
BANNER_ATTR = re.compile(
    r'data-testid=(?:\{FORECAST_DISCLAIMER_TESTID\}|"da-caveat")'
    r"|<ForecastDisclaimer\b"
    r"|<DiagnosticCaveat\b"
)

# Surface → file. The legacy dashboard mounts the shared banner through its
# DiagnosticCaveat wrapper (derived-DA line); the wrapper itself is pinned below.
_BANNER_SURFACES = {
    "gm/views/ForecastingView.tsx": lambda: FORECASTING_VIEW,
    "forecasting/ForecastingDashboard.tsx": lambda: LEGACY_DASHBOARD,
    "forecasting/WeeklyInferenceView.tsx": lambda: LEGACY_WEEKLY,
}


@pytest.mark.parametrize("rel", list(_BANNER_SURFACES), ids=list(_BANNER_SURFACES))
def test_caveat_not_gated_on_any_surface(rel: str):
    """BL-02 (hardened, CXD-032): the caveat renders UNCONDITIONALLY on EVERY
    forecasting surface — GM view (all assets/modes) AND legacy dashboard AND
    legacy weekly inference (Gold/BTC). La muralla es por superficie.

    Constant-proof + structural: the banner mount is detected as literal testid,
    SSOT-constant testid, or shared-component mount, and at least one mount must be
    an unconditional direct child of the enclosing `return (` JSX — the brace
    balance of the prefix between the `return (` and the mount must be zero. Any
    `{isModelZoo && (...)}` / `{data && (...)}` / ternary wrapper (however the
    banner is spelled) leaves an unbalanced `{` in that prefix and fails. The
    RENDER-level twin of this check (visibility, hidden attrs/classes, mutations)
    lives in usdcop-trading-dashboard/tests/unit/components/
    forecasting-caveat-surfaces.test.tsx.
    """
    src = _BANNER_SURFACES[rel]().read_text(encoding="utf-8", errors="replace")
    # Mount sites are searched in the MASKED source: a `<ForecastDisclaimer` named
    # inside a comment or a string is prose, not a mount (S-07).
    masked = _mask_code(src)
    matches = [
        m for m in BANNER_ATTR.finditer(masked)
        # ignore import lines mentioning the component name
        if "import" not in src[src.rfind("\n", 0, m.start()) + 1: m.start()]
    ]
    assert matches, (
        f"{rel} has no da-caveat banner in literal, constant, or shared-component "
        "form (BL-02/CXD-032)."
    )
    depths = []
    for m in matches:
        depth = _jsx_depth_from_enclosing_return(src, m.start())
        assert depth is not None, f"{rel}: banner appears outside any JSX return"
        depths.append(depth)
    assert any(d == 0 for d in depths), (
        f"Every da-caveat banner in {rel} is nested inside a JSX expression (brace "
        f"depths from enclosing return: {depths}). BL-02 requires the banner to render "
        "unconditionally on every surface and mode — wrapping it in a conditional is "
        "the exact regression this test exists to block, no matter how it is spelled."
    )


# ---------------------------------------------------------------------------
# CXD-032 (2nd rejection) — the VARIANT is a factual claim about the surface,
# the early-return branches are part of the surface, and the hit column may not
# be communicated by glyph+colour alone.
# ---------------------------------------------------------------------------


def _balanced_paren_block(src: str, open_idx: int) -> str:
    """Substring from the '(' at `open_idx` to its matching ')'."""
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return src[open_idx: i + 1]
    return src[open_idx:]


def _weekly_surface_sources() -> dict[str, str]:
    """The two files that render a weekly-inference table (GM AssetWeeklyBody lives inside
    ForecastingView.tsx). Whole-file scope on purpose: the shared imports and helpers sit
    above the component, and a mutation is just as real if it lands there."""
    return {
        "gm/views/ForecastingView.tsx":
            FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace"),
        "forecasting/WeeklyInferenceView.tsx":
            LEGACY_WEEKLY.read_text(encoding="utf-8", errors="replace"),
    }


def test_rule_based_weekly_surface_does_not_claim_the_model_zoo():
    """CXD-032 (2nd rejection, finding #1): Gold/BTC weekly is a RULE-BASED policy — it
    runs no model zoo. Mounting `variant="zoo"` there made the banner assert '9 modelos'
    and a ~52% mean DA about a surface where neither exists: a false statement about the
    nature of the product, printed inside the honesty banner itself.

    The legacy weekly view must mount the rule branch and never the zoo branch, and the GM
    view must pick 'weekly' whenever the asset is not in model-zoo mode."""
    weekly_src = LEGACY_WEEKLY.read_text(encoding="utf-8", errors="replace")
    mounts = re.findall(r"<ForecastDisclaimer\s+variant=\"(\w+)\"", weekly_src)
    assert mounts, (
        "WeeklyInferenceView.tsx mounts <ForecastDisclaimer/> without an explicit variant "
        "— the default branch is the model zoo one, which is false on a rule-based "
        "surface (CXD-032)."
    )
    assert set(mounts) == {"weekly"}, (
        f"WeeklyInferenceView.tsx mounts the disclaimer with variants {sorted(set(mounts))}. "
        "Gold/BTC weekly inference is rule-based: only the 'weekly' branch states the truth "
        "about it; 'zoo' claims a 9-model ML ensemble that this surface does not run."
    )
    gm_src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"<ForecastDisclaimer\s*\n?\s*variant=\{([^}]*)\}", gm_src)
    assert m, "ForecastingView.tsx no longer selects the disclaimer variant explicitly"
    expr = re.sub(r"\s+", " ", m.group(1))
    assert "'weekly'" in expr, (
        f"ForecastingView.tsx variant expression ({expr!r}) never yields 'weekly'. Assets "
        "rendered through AssetWeeklyBody are rule-based; the zoo copy is false for them."
    )
    assert "isModelZoo" in expr, (
        f"ForecastingView.tsx variant expression ({expr!r}) no longer branches on "
        "isModelZoo — the branch is what keeps the claim true per surface."
    )


def test_weekly_early_returns_carry_the_caveat():
    """CXD-032 (2nd rejection, finding #2 — the BTC gap): WeeklyInferenceView leaves
    through TWO early returns before the main JSX (`loading && !data`, and the
    error/locked branch). With BTC gated by plan (403) or still loading, the forecasting
    surface rendered with NO caveat at all. La muralla es por superficie, no por estado de
    carga: EVERY JSX return of the component must carry the shared banner."""
    src = LEGACY_WEEKLY.read_text(encoding="utf-8", errors="replace")
    fn = src.find("export function WeeklyInferenceView")
    assert fn != -1, "WeeklyInferenceView disappeared"
    body = src[fn:]
    # Only JSX returns (`return (` immediately followed by `<`); `return () => ...`
    # cleanup callbacks are not renders.
    # Only the COMPONENT's own render paths: `return (` at indentation <= 4 (top level or
    # inside one `if`). Nested JSX returns (the `weeks.map(w => return (<tr .../>))` row
    # renderer) are not surfaces and are excluded by indentation. `return () => ...`
    # cleanup callbacks are excluded by the `(?=<)` lookahead (they are not renders).
    blocks = [
        _balanced_paren_block(body, m.start(1))
        for m in re.finditer(r"^ {2,4}return\s*(\()\s*(?=<)", body, re.MULTILINE)
    ]
    assert len(blocks) >= 3, (
        f"WeeklyInferenceView has {len(blocks)} JSX returns; the loading and error early "
        "returns must still exist as distinct render paths (CXD-032)."
    )
    naked = [i for i, b in enumerate(blocks) if "<ForecastDisclaimer" not in b]
    assert not naked, (
        f"WeeklyInferenceView JSX return(s) #{naked} render the forecasting surface with "
        "no <ForecastDisclaimer/>. The loading and 403/404 branches are exactly where BTC "
        "shipped uncovered (CXD-032 finding #2)."
    )


# The weekly tables' hit column: a ✓/· glyph plus a colour is invisible to a screen
# reader. Both weekly surfaces must name the column and expose Sí/No per row.
_HIT_A11Y_TOKENS = (
    "FORECAST_HIT_COLUMN_LABEL",
    "FORECAST_HIT_YES_LABEL",
    "FORECAST_HIT_NO_LABEL",
    'aria-hidden="true"',
    "sr-only",
)


def test_weekly_hit_column_is_not_symbol_and_colour_only():
    """CXD-032 (2nd rejection, finding #3): 'la columna ✓ comunica acierto con
    símbolos/color sin nombre ni Sí/No accesibles'. Both weekly surfaces must import the
    shared labels, hide the decorative glyph from assistive tech and render the Sí/No
    text. The RENDER-level twin lives in forecasting-caveat-surfaces.test.tsx
    (assertHitColumnAccessible)."""
    for name, seg in _weekly_surface_sources().items():
        missing = [t for t in _HIT_A11Y_TOKENS if t not in seg]
        assert not missing, (
            f"{name}: the weekly hit column lacks {missing} — success/failure communicated "
            "only by glyph and colour is not communicated at all to a screen reader "
            "(CXD-032 finding #3)."
        )


# LONG/SHORT are ORDER labels: they name what an executor would do. On a DIAGNOSTIC
# surface the honest rendering is the bias they describe (FABRIC §24.3: "sin colores ni
# etiquetas imperativas").
_RAW_DIRECTION_RENDER = re.compile(r"\{\s*(?:w|forward)\.direction\s*\}")


def test_weekly_direction_is_not_rendered_as_an_order_label():
    """BL-03, re-remediated: the weekly surfaces must map `direction` through the shared
    neutral labels instead of printing the raw LONG/SHORT token.

    NOTE for the archaeologist: the previous delivery's Vitest test *required*
    `getAllByText(/\\b(LONG|SHORT)\\b/)` to match — a test that demanded the exact thing
    BL-03 forbids. It was rewritten, not deleted: a test that contradicts its spec is a
    bug in the test."""
    for name, seg in _weekly_surface_sources().items():
        raw = _RAW_DIRECTION_RENDER.findall(seg)
        assert not raw, (
            f"{name} renders the raw direction token as JSX text ({len(raw)} site(s)). "
            "LONG/SHORT are imperative order labels; a diagnostic surface shows the bias "
            "(directionLabel(...) from the shared SSOT)."
        )
        assert "directionLabel(" in seg, (
            f"{name} no longer maps direction through the shared neutral label helper "
            "(BL-03 / FABRIC §24.3)."
        )
        for const in (
            "FORECAST_DIRECTION_LABEL_UP",
            "FORECAST_DIRECTION_LABEL_DOWN",
            "FORECAST_DIRECTION_LABEL_FLAT",
        ):
            assert const in seg, (
                f"{name} does not consume {const} — the neutral wording has ONE source "
                "(lib/ui/forecast-disclaimer.ts), same rule as the caveat copy (BL-04)."
            )


def test_legacy_diagnostic_caveat_wrapper_cannot_hide():
    """CXD-032: the legacy DiagnosticCaveat wrapper (derived-DA line) must not be able
    to suppress the shared banner — no `return null` (the old `if (!stats) return null`
    was exactly the 'hidden por rama' Codex flagged), and it must mount
    <ForecastDisclaimer/>."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8", errors="replace")
    start = src.find("function DiagnosticCaveat")
    assert start != -1, "DiagnosticCaveat disappeared from the legacy dashboard"
    body = src[start: src.find("\nfunction ", start + 10)]
    assert "<ForecastDisclaimer" in body, (
        "DiagnosticCaveat no longer mounts the shared <ForecastDisclaimer/> — the "
        "legacy banner must come from the shared SSOT component (BL-04/CXD-032)."
    )
    # The stats useMemo may legitimately yield null (no DA rows); the COMPONENT may
    # not: after the memo closes, no `return null` is allowed before the JSX return.
    memo_end = body.find("}, [data])")
    component_tail = body[memo_end if memo_end != -1 else 0:]
    assert "return null" not in component_tail, (
        "DiagnosticCaveat grew a `return null` branch again — the banner must render "
        "even with zero DA stats (CXD-032: 'hidden por rama pasa' is the rejected bug; "
        "only the DERIVED sentence may be conditional)."
    )


# ---------------------------------------------------------------------------
# BL-06 — CI muralla: forecasting surfaces are read-only diagnostic surfaces.
# They must never grow approve/deploy/execution wiring or order verbs.
# ---------------------------------------------------------------------------

FORBIDDEN_ACTION_TOKENS = [
    "api/production/approve",
    "api/production/deploy",
    "/api/execution",
    "onApprove",
    "onReject",
]

# Order verbs as UI text (word-bounded, case-sensitive — Spanish uppercase CTA style).
_ORDER_VERBS = re.compile(r"\b(COMPRAR|VENDER)\b")

# Comment-only lines are tolerated (explanatory prose is not an action capability).
_COMMENT_LINE = re.compile(r"^\s*(//|\*|/\*|\{/\*)")


def _non_comment_offenders(path: Path) -> list[str]:
    """Return 'file:line: token' hits for forbidden tokens outside comment lines."""
    offenders: list[str] = []
    rel = path.relative_to(ROOT).as_posix()
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if _COMMENT_LINE.match(line):
            continue
        for token in FORBIDDEN_ACTION_TOKENS:
            if token in line:
                offenders.append(f"{rel}:{lineno}: {token}")
        m = _ORDER_VERBS.search(line)
        if m:
            offenders.append(f"{rel}:{lineno}: {m.group(0)}")
    return offenders


# ---------------------------------------------------------------------------
# The SURFACE is whatever the /forecasting routes reach — not a hand-written list.
#
# S-06 (auto-red-team, 2026-07-28): the previous perimeter was exactly two paths
# (`components/gm/views/ForecastingView.tsx` + `components/forecasting/*`). Mounting
# a new component from ANY other directory escaped it completely: a
# `components/gm/views/ForecastingApprovePanel.tsx` with
# `fetch('/api/production/approve')`, `fetch('/api/execution/orders')` and a
# `COMPRAR` button, imported by ForecastingView and rendered inside its JSX, left
# the suite at 25/25 green. The BL-06 doc claimed "el día que forecasting pueda
# aprobar o ejecutar, este test se pone rojo"; it was false for any file the author
# of the mutation chose to put elsewhere.
#
# The perimeter is now DERIVED: start at the route entry points that Next.js
# serves for /forecasting and /legacy/forecasting, and follow every statically
# resolvable local import transitively. A new component is inside the muralla the
# moment the surface imports it, wherever it lives — which is the only definition
# of "the forecasting surface" that a mutation cannot side-step by choosing a
# directory.
# ---------------------------------------------------------------------------

DASHBOARD = ROOT / "usdcop-trading-dashboard"

#: Route entry points. `app/**/page.tsx` IS the surface as far as a user is
#: concerned; anything they reach is served under /forecasting.
FORECASTING_ROUTES = (
    DASHBOARD / "app" / "forecasting" / "page.tsx",
    DASHBOARD / "app" / "legacy" / "forecasting" / "page.tsx",
)

#: Files that MUST end up in the derived closure. Without this, a resolver that
#: silently stops resolving (a rename, a new path alias, an `index.ts` barrel that
#: moves) would shrink the perimeter back to nothing and every lock below would go
#: green by covering zero files — the exact failure mode S-06 documents.
_SURFACE_ANCHORS = (
    "usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx",
    "usdcop-trading-dashboard/components/forecasting/ForecastingDashboard.tsx",
    "usdcop-trading-dashboard/components/forecasting/WeeklyInferenceView.tsx",
    "usdcop-trading-dashboard/components/forecasting/ForecastDisclaimer.tsx",
    "usdcop-trading-dashboard/components/gm/TerminalShell.tsx",
    "usdcop-trading-dashboard/lib/ui/forecast-disclaimer.ts",
)

_MIN_SURFACE_FILES = 20

# `from '<spec>'`, `import('<spec>')`, `require('<spec>')` — the three spellings
# that pull another module into the surface.
_IMPORT_SPEC = re.compile(
    r"""(?:\bfrom\s*|\bimport\s*\(\s*|\brequire\s*\(\s*)['"]([^'"]+)['"]"""
)
# A dynamic import whose specifier is NOT a literal: `import(name)`,
# `import(`${base}/x`)`. Static analysis cannot follow it, so it is a hole in the
# muralla and is rejected outright rather than silently skipped.
_OPAQUE_DYNAMIC_IMPORT = re.compile(r"\bimport\s*\(\s*(?!['\"])(?!/\*\s*@vite-ignore)")

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
    candidates = [base.with_suffix(s) for s in _TS_SUFFIXES]
    candidates += [base / f"index{s}" for s in _TS_SUFFIXES]
    if base.suffix in _TS_SUFFIXES:
        candidates.insert(0, base)
    for c in candidates:
        if c.is_file():
            return c
    return None


def _forecasting_surface_files() -> list[Path]:
    """Every source file the /forecasting routes reach, transitively.

    Deterministic (sorted) so failure output is stable. Files outside the
    dashboard are ignored; `node_modules` is never entered (bare specifiers are
    not resolved at all).
    """
    seen: set[Path] = set()
    stack: list[Path] = [r.resolve() for r in FORECASTING_ROUTES if r.is_file()]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        text = current.read_text(encoding="utf-8", errors="replace")
        for spec in _IMPORT_SPEC.findall(text):
            target = _resolve_import(spec, current)
            if target and target not in seen and DASHBOARD.resolve() in target.parents:
                stack.append(target)
    return sorted(seen)


# Kept as the name the older tests used; it now returns the DERIVED surface.
def _forecasting_component_files() -> list[Path]:
    return _forecasting_surface_files()


def test_forecasting_surface_is_derived_not_hardcoded():
    """S-06 meta-lock: the perimeter must actually be the import closure.

    A lock over an empty (or truncated) file set is green by vacuity — that is
    precisely how BL-06 shipped. This pins that the closure is discovered, that it
    contains the known surface anchors, and that no file inside it hides a module
    behind a non-literal dynamic import (which static analysis cannot follow, so
    it would reopen the same hole).
    """
    for route in FORECASTING_ROUTES:
        assert route.is_file(), (
            f"{route.relative_to(ROOT).as_posix()} disappeared — if the route moved, "
            "FORECASTING_ROUTES must move with it, or the muralla covers nothing."
        )
    files = _forecasting_surface_files()
    rels = {f.resolve().relative_to(ROOT).as_posix() for f in files}
    missing = [a for a in _SURFACE_ANCHORS if a not in rels]
    assert not missing, (
        f"The derived forecasting surface lost {missing}. The import resolver is "
        "broken or a surface file moved; every BL-06 lock below would silently stop "
        "covering it (S-06)."
    )
    assert len(files) >= _MIN_SURFACE_FILES, (
        f"The derived surface collapsed to {len(files)} files (expected >= "
        f"{_MIN_SURFACE_FILES}): {sorted(rels)}"
    )
    opaque: list[str] = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(
            f.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if _COMMENT_LINE.match(line):
                continue
            if _OPAQUE_DYNAMIC_IMPORT.search(line):
                opaque.append(f"{rel}:{lineno}: {line.strip()[:110]}")
    assert not opaque, (
        "A forecasting surface file imports a module through a non-literal "
        "specifier. The perimeter is computed statically, so such an import is an "
        "unauditable back door into the surface (S-06):\n  " + "\n  ".join(opaque)
    )


def test_forecasting_has_no_action_capabilities():
    """BL-06: EVERY file the /forecasting routes reach must stay free of
    approval/deploy/execution endpoints, approval callbacks and BUY/SELL order verbs.

    Forecasting is a diagnostic surface (quant-constitution): the day it can approve,
    deploy or phrase an order, it stops being disclosure and becomes a signal product
    that bypassed the 2-vote gate.

    S-06: the perimeter is the import closure of the routes, not two hardcoded
    paths. The mutation that used to pass — a new `ForecastingApprovePanel.tsx`
    under `components/gm/views/` mounted inside ForecastingView's JSX — now fails
    here, because importing it is what puts it inside the muralla.
    """
    offenders: list[str] = []
    for f in _forecasting_surface_files():
        offenders.extend(_non_comment_offenders(f))
    assert not offenders, (
        "Forecasting surfaces grew action capabilities (forbidden outside comments):\n  "
        + "\n  ".join(offenders)
    )


def test_legacy_forecasting_same_rules():
    """BL-06: the legacy ForecastingDashboard.tsx obeys the same muralla — legacy pages
    are still served (admin-only /legacy) and must not be the back door."""
    offenders = _non_comment_offenders(LEGACY_DASHBOARD)
    assert not offenders, (
        "Legacy ForecastingDashboard.tsx grew action capabilities:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# BL-01 (red-team bypass #4) — the marketing blacklist applies to the SURFACE
# files too, not only to the SSOT. Guarding the disclaimer constants is useless
# if 'señal validada: opere con confianza' can simply be written inline in the
# JSX next to the banner: the rendered page is the product, not the SSOT file.
# ---------------------------------------------------------------------------

# Same exclusions as BL-06: comment lines are prose, and import lines only name
# modules/identifiers.
_IMPORT_LINE = re.compile(r"^\s*(import\b|export\s*\{|\}\s*from\s)")


def _surface_marketing_offenders(path: Path, *, apply_exemptions: bool = True) -> list[str]:
    """'file:line: /pattern/' hits of FORBIDDEN_MARKETING in a surface file,
    matched on accent-stripped casefolded text (same _norm as the SSOT check),
    skipping comment and import lines (BL-06 precedent)."""
    offenders: list[str] = []
    rel = path.relative_to(ROOT).as_posix()
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if _COMMENT_LINE.match(line) or _IMPORT_LINE.match(line):
            continue
        text = _norm(line)
        for pat in FORBIDDEN_MARKETING:
            if not re.search(pat, text):
                continue
            if apply_exemptions and (rel, pat) in MARKETING_EXEMPTIONS:
                continue
            offenders.append(f"{rel}:{lineno}: /{pat}/ :: {line.strip()[:110]}")
    return offenders


# Declared, reasoned exemptions to the marketing scan on the WIDER surface (S-06:
# the closure now includes shared chrome and i18n). An exemption names the exact
# file AND the exact pattern — never a whole file, never a whole pattern — and is
# itself checked: an exemption that stops matching fails the test, so a stale
# waiver cannot quietly cover a future offender.
MARKETING_EXEMPTIONS: dict[tuple[str, str], str] = {
    ("usdcop-trading-dashboard/lib/i18n/gm.ts", r"garantiz"):
        "disclaimer legal del pie: 'los resultados pasados NO garantizan resultados "
        "futuros'. Es la negación de una promesa — el opuesto exacto de lo que el "
        "patrón persigue.",
}


def test_marketing_exemptions_are_all_still_needed():
    """A waiver that no longer matches anything is a waiver nobody re-reads. Each
    declared exemption must correspond to a real, current hit."""
    stale = []
    for (rel, pat), _reason in MARKETING_EXEMPTIONS.items():
        path = ROOT / rel
        hits = [o for o in _surface_marketing_offenders(path, apply_exemptions=False)
                if f"/{pat}/" in o] if path.is_file() else []
        if not hits:
            stale.append(f"{rel} :: /{pat}/")
    assert not stale, (
        "Marketing exemptions that no longer match anything (delete them; a stale "
        "waiver silently covers the next real offender):\n  " + "\n  ".join(stale)
    )


def test_forecasting_surfaces_carry_no_marketing_language():
    """BL-01 (hardening, red-team bypass #4): every file the /forecasting routes
    reach must be free of the promotional/action language (case- and
    accent-insensitive), outside comments and imports.

    The SSOT-only patterns are NOT applied here: the surfaces legitimately say
    'Confianza proxy' (mandated BL-03 label) and 'Sin posición' (honest flat
    state). Same honest limit as the SSOT check: this is defense in depth
    against inline marketing, not a proof of honesty — the hard guarantee
    remains the mandatory, unconditional SSOT banner (BL-01/BL-02)."""
    offenders: list[str] = []
    for f in _forecasting_component_files():
        offenders.extend(_surface_marketing_offenders(f))
    assert not offenders, (
        "Forecasting surface files contain promotional/action language inline — "
        "marketing next to the banner defeats the disclaimer no matter how "
        "honest the SSOT copy is:\n  " + "\n  ".join(offenders)
    )


def test_caveat_is_not_hardcoded_to_a_stale_number():
    """The main dashboard banner must compute its DA from the loaded data.

    A hardcoded '52%' would silently become false the day the data changes — in either
    direction. The banner's honesty comes from being derived, not asserted.
    """
    src = (DASH / "forecasting/ForecastingDashboard.tsx").read_text(encoding="utf-8",
                                                                    errors="replace")
    assert "useMemo" in src.split("function DiagnosticCaveat")[1].split("function ")[0], (
        "DiagnosticCaveat must derive its statistics from the data prop, not a literal"
    )


# ---------------------------------------------------------------------------
# BL-03 — probabilistic wording + neutral DA colors on diagnostic surfaces.
# Codex review 2026-07-27: "falta probabilidad weekly y color DA".
# ---------------------------------------------------------------------------

# CXD-032 overruled the previous version of this lock ("falta probabilidad weekly"):
# `confidence` is rule CONVICTION / regime strength, NOT a probability — there is no
# probability_up field in the weekly contract and none may be invented. The mandated
# label, on EVERY weekly surface (GM AssetWeeklyBody + legacy WeeklyInferenceView):
_CONVICTION_LABEL_NORM = "conviccion de regla (proxy; no probabilidad)"


def _weekly_segments() -> dict[str, str]:
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    fn = src.find("function AssetWeeklyBody")
    assert fn != -1, "AssetWeeklyBody (weekly inference surface) disappeared"
    # The label may live in a const declared just above the function — include it.
    const = src.find("const WEEKLY_CONVICTION_LABEL")
    start = min(fn, const) if const != -1 else fn
    end = src.find("export function ForecastingView", fn)
    return {
        "gm/AssetWeeklyBody": src[start:end if end != -1 else len(src)],
        "forecasting/WeeklyInferenceView.tsx":
            LEGACY_WEEKLY.read_text(encoding="utf-8", errors="replace"),
    }


def test_weekly_confidence_is_conviction_not_probability():
    """BL-03 (CXD-032): every weekly-inference surface must expose the per-week
    `confidence` number labeled as 'Convicción de regla (proxy; no probabilidad)' —
    the number is NOT hidden, but it is NEVER sold as a probability: no
    'probabilidad estimada' wording in the weekly segments (the directional-replay
    panel keeps its wording because `probability_up` IS a real contract field there),
    and the non-calibrated nature stays explicit."""
    for name, seg in _weekly_segments().items():
        norm = _norm(seg)
        assert _CONVICTION_LABEL_NORM in norm, (
            f"{name} lost the mandated conviction label "
            f"({_CONVICTION_LABEL_NORM!r}). `confidence` must be surfaced AND labeled "
            "as rule conviction, not probability (CXD-032)."
        )
        assert "confidence" in seg, (
            f"{name}: the conviction column must be driven by the data's `confidence` "
            "field (rule-conviction proxy), not a literal."
        )
        assert "probabilidad estimada" not in norm, (
            f"{name} phrases `confidence` as 'probabilidad estimada' — there is no "
            "probability_up in the weekly contract; calling conviction a probability "
            "fabricates calibration the rule engine does not have (CXD-032)."
        )
        assert re.search(r"no calibrad|no es una probabilidad", norm), (
            f"{name} must keep the explicit non-calibrated disclaimer on the "
            "conviction proxy."
        )


# DA is a DIAGNOSTIC metric (~coin flip after adjusting for models tried): it must
# never be painted green/red ("works/doesn't") on the forecasting surfaces. One
# neutral tone; the caveat banner provides the context.
_DA_LINE_TOKENS = (
    "direction_accuracy",          # covers model_avg_/wf_ prefixed fields too
    "da_2025_pct",
    "balanced_accuracy",
    "fmtDa(",
    "formatDA(",
    "DA 2025",
)
_FORBIDDEN_DA_COLORS = (
    "GM.pos", "GM.neg", "GM.warn",
    "'pos'", "'neg'", "'warn'",
    "emerald", "text-red", "text-amber",
    "#10B981", "#10b981", "#EF4444", "#ef4444", "#22C55E", "#22c55e",
)


def _da_color_offenders(path: Path) -> list[str]:
    offenders: list[str] = []
    rel = path.relative_to(ROOT).as_posix()
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        if _COMMENT_LINE.match(line):
            continue
        if not any(tok in line for tok in _DA_LINE_TOKENS):
            continue
        for color in _FORBIDDEN_DA_COLORS:
            if color in line:
                offenders.append(f"{rel}:{lineno}: {color} :: {line.strip()[:110]}")
    return offenders


def test_da_is_never_colored_pos_neg():
    """BL-03: no DA/BDA rendering on any forecasting surface may carry green/red
    (or threshold-warn) color semantics — a DA painted green reads as 'the model
    works' while the statistics say coin flip. Applies to the GM view AND the
    legacy components (still served under /legacy, admin-only)."""
    offenders: list[str] = []
    for f in _forecasting_component_files():
        offenders.extend(_da_color_offenders(f))
    assert not offenders, (
        "Direction-accuracy values are still color-coded pos/neg/warn on a "
        "diagnostic surface (BL-03 'color DA'):\n  " + "\n  ".join(offenders)
    )


def test_directional_replay_metric_cells_are_neutral():
    """BL-03 (companion to the line-scan): the shared `metricCell` renderer of the
    'DA OOS por horizonte' table colors its value on a *different* line than the DA
    tokens, so the line-scan alone cannot see it. Pin the whole helper body: no
    pos/neg/warn tones. The only colored verdict in that table is the pre-declared
    'Generaliza' badge, whose criterion is stated in the table footer."""
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    start = src.find("const metricCell")
    if start == -1:
        pytest.skip("metricCell helper no longer exists")
    body = src[start: src.find(");", start) + 2]
    hits = [c for c in ("GM.pos", "GM.neg", "GM.warn") if c in body]
    assert not hits, (
        f"metricCell colors DA/BDA/recall cells with {hits} — diagnostic metrics "
        "must render in a neutral tone (BL-03)."
    )


# ---------------------------------------------------------------------------
# BL-04 — the caveat copy has ONE source. Codex review 2026-07-27: "legacy
# conserva hardcode 'Direccion con senal'".
# ---------------------------------------------------------------------------

# Accent/case-proof: matches 'Direccion con senal', 'Dirección con señal', etc.
_SIGNAL_CLAIM = re.compile(r"direcci[oó]n\s+con\s+se[nñ]al", re.IGNORECASE)


def test_no_hardcoded_signal_claim_anywhere_in_dashboard():
    """BL-04: no forecasting surface — GM or legacy — may hardcode a 'Direccion con
    senal' headline. That branch flipped the diagnostic caveat into a green signal
    claim the day mean DA crosses 55%, converting a weak metric into an implied
    recommendation outside the 2-vote gate (quant-constitution). The caveat headline
    comes ONLY from lib/ui/forecast-disclaimer.ts."""
    dash_root = ROOT / "usdcop-trading-dashboard"
    offenders: list[str] = []
    for sub in ("components", "app", "lib"):
        for f in (dash_root / sub).rglob("*.ts*"):
            if "node_modules" in f.parts:
                continue
            src = f.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(src.splitlines(), start=1):
                # Comment-only lines are tolerated (explanatory prose about the
                # removed claim is not a rendered claim — same rule as BL-06).
                if _COMMENT_LINE.match(line):
                    continue
                if _SIGNAL_CLAIM.search(line):
                    offenders.append(
                        f"{f.relative_to(ROOT).as_posix()}:{lineno}: {line.strip()[:110]}"
                    )
    assert not offenders, (
        "Hardcoded 'Direccion con senal' claim found (BL-04):\n  "
        + "\n  ".join(offenders)
    )


def test_legacy_caveat_headline_and_body_come_from_ssot():
    """BL-04 (CXD-032): the legacy DiagnosticCaveat must render headline AND body
    from the shared SSOT — via the shared <ForecastDisclaimer/> component (whose
    SSOT wiring the companion test pins), never from local strings, and never behind
    a beats-the-bar ternary. The old version passed with only the TITLE from SSOT
    while the body stayed hardcoded prose next to it (split-JSX): mounting the
    shared component is what closes that hole, so this test requires the mount, not
    just a constant reference."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8", errors="replace")
    body_start = src.find("function DiagnosticCaveat")
    assert body_start != -1, "DiagnosticCaveat disappeared from the legacy dashboard"
    body = src[body_start: src.find("\nfunction ", body_start + 10)]
    assert "<ForecastDisclaimer" in body, (
        "Legacy DiagnosticCaveat no longer mounts the shared <ForecastDisclaimer/> — "
        "headline AND body must come from the SSOT component; a locally rebuilt banner "
        "reopens the split-JSX/hardcoded-body hole (BL-04/CXD-032)."
    )
    assert not re.search(r"\?\s*['\"`][^'\"`]*['\"`]\s*:\s*<ForecastDisclaimer", body), (
        "Legacy DiagnosticCaveat gates the shared banner behind a ternary — the "
        "banner must be unconditional (BL-04)."
    )
    # The legacy file must not re-inline any disclaimer copy next to the mount: the
    # honest phrases live ONLY in the SSOT (drift-proof).
    for phrase in ("Superficie de diagn", "NO ES UNA SE"):
        assert phrase not in body, (
            f"Legacy DiagnosticCaveat re-inlines disclaimer copy ({phrase!r}) — the "
            "copy has ONE source: lib/ui/forecast-disclaimer.ts via ForecastDisclaimer "
            "(BL-04)."
        )
