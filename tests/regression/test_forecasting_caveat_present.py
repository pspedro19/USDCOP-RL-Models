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

# BL-04 moved the caveat copy/testid to a shared SSOT constant; surfaces may carry the
# marker either literally or via the imported constant name.
DISCLAIMER_SSOT = (
    ROOT / "usdcop-trading-dashboard" / "lib" / "ui" / "forecast-disclaimer.ts"
)

SURFACES = {
    "forecasting/ForecastingDashboard.tsx": ("DiagnosticCaveat",),
    "gm/views/ForecastingView.tsx": ("da-caveat", "FORECAST_DISCLAIMER_TESTID"),
}


@pytest.mark.parametrize("rel,markers", SURFACES.items(), ids=list(SURFACES))
def test_da_surface_carries_caveat(rel: str, markers: tuple):
    p = DASH / rel
    if not p.is_file():
        pytest.skip(f"{rel} absent")
    src = p.read_text(encoding="utf-8", errors="replace")
    if "direction_accuracy" not in src and "Direction Accuracy" not in src:
        pytest.skip(f"{rel} no longer shows DA")
    assert any(m in src for m in markers), (
        f"{rel} displays Direction Accuracy but the caveat ({markers!r}) is gone. A ~52% DA "
        "shown without context reads as 'the models work'; the statistics say coin flip "
        "(p_adj 0.66 across models, 1.0 across model-by-horizon cells)."
    )


# ---------------------------------------------------------------------------
# BL-01 — the caveat banner itself (testid + honest NO-SIGNAL clause), and its
# gating. Hardened after Codex review: the first version pinned only the prefix
# "Superficie de diagn", so a deceptive mutation like
#   'Superficie de diagnóstico: señal validada, opere con confianza.'
# passed. The lock now (a) parses the SSOT string constants, (b) requires the
# FULL no-signal clauses, and (c) rejects promotional/action language outright.
# ---------------------------------------------------------------------------

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


def test_caveat_banner_present():
    """BL-01: the GM forecasting view carries the da-caveat banner AND the SSOT
    still carries the complete no-signal clauses (not just their prefixes)."""
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    has_testid = (
        'data-testid="da-caveat"' in src
        or "FORECAST_DISCLAIMER_TESTID" in src
    )
    assert has_testid, (
        "ForecastingView.tsx lost the da-caveat banner (neither the literal testid nor "
        "the FORECAST_DISCLAIMER_TESTID constant is referenced). The DA surface must "
        "not render without its diagnostic disclaimer (BL-01)."
    )
    # BL-04: the honest copy lives in the shared SSOT; the view must either carry
    # it inline or import the SSOT module that does.
    ssot_src = DISCLAIMER_SSOT.read_text(encoding="utf-8", errors="replace")
    assert "Superficie de diagn" in src or (
        "forecast-disclaimer" in src and "Superficie de diagn" in ssot_src
    ), (
        "The honest phrase ('Superficie de diagnóstico, no de señales') is neither "
        "inline in ForecastingView.tsx nor provided via lib/ui/forecast-disclaimer.ts "
        "— the disclaimer text is part of the contract, not decoration (BL-01/BL-04)."
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


# Banner attribute in BOTH spellings: literal testid or the shared SSOT constant.
# (Codex review 2026-07-27 rejected the old literal-only count as "universalidad
# sorteable por constante": once the testid was expressed through
# FORECAST_DISCLAIMER_TESTID the old check passed vacuously — re-gating the banner
# under `isModelZoo` would NOT have failed.)
BANNER_ATTR = re.compile(r'data-testid=(?:\{FORECAST_DISCLAIMER_TESTID\}|"da-caveat")')


def test_caveat_not_gated_only_to_model_zoo():
    """BL-02 (hardened): the caveat renders UNCONDITIONALLY in /forecasting — la
    muralla es por superficie, no por asset ni por modo de render.

    Constant-proof + structural: the banner attribute is detected in literal AND
    constant form, and at least one banner must be an unconditional direct child of
    the enclosing `return (` JSX — the brace balance of the prefix between the
    `return (` and the banner must be zero. Any `{isModelZoo && (...)}` / ternary
    wrapper (regardless of how the testid is spelled) leaves an unbalanced `{` in
    that prefix and fails.
    """
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    matches = list(BANNER_ATTR.finditer(src))
    assert matches, (
        "ForecastingView.tsx has no da-caveat banner in either literal or "
        "FORECAST_DISCLAIMER_TESTID form (BL-02)."
    )
    depths = []
    for m in matches:
        ret = src.rfind("return (", 0, m.start())
        assert ret != -1, "banner appears outside any JSX return"
        prefix = src[ret + len("return ("): m.start()]
        depths.append(prefix.count("{") - prefix.count("}"))
    assert any(d == 0 for d in depths), (
        "Every da-caveat banner in ForecastingView.tsx is nested inside a JSX "
        f"expression (brace depths from enclosing return: {depths}). BL-02 requires "
        "the banner to render unconditionally for EVERY forecast mode (model zoo, "
        "directional replay AND weekly inference) — wrapping it in "
        "`{isModelZoo && (...)}` or any conditional is the exact regression this "
        "test exists to block, no matter how the testid is spelled."
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


def _forecasting_component_files() -> list[Path]:
    files = [FORECASTING_VIEW]
    files.extend(sorted((DASH / "forecasting").glob("*.ts*")))
    return [f for f in files if f.is_file()]


def test_forecasting_has_no_action_capabilities():
    """BL-06: ForecastingView + every components/forecasting/* file must stay free of
    approval/deploy/execution endpoints, approval callbacks and BUY/SELL order verbs.

    Forecasting is a diagnostic surface (quant-constitution): the day it can approve,
    deploy or phrase an order, it stops being disclosure and becomes a signal product
    that bypassed the 2-vote gate.
    """
    offenders: list[str] = []
    for f in _forecasting_component_files():
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


def _surface_marketing_offenders(path: Path) -> list[str]:
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
            if re.search(pat, text):
                offenders.append(f"{rel}:{lineno}: /{pat}/ :: {line.strip()[:110]}")
    return offenders


def test_forecasting_surfaces_carry_no_marketing_language():
    """BL-01 (hardening, red-team bypass #4): ForecastingView.tsx and every
    components/forecasting/* file must be free of the promotional/action
    language (case- and accent-insensitive), outside comments and imports.

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

def test_weekly_inference_shows_probabilistic_wording():
    """BL-03: the weekly-inference branch (Gold rule-based surface) must carry the
    probabilistic wording, not only the USD/COP directional-replay branch.

    The only honest per-week number available in weekly_inference_<year>.json is
    `confidence` (rule-conviction proxy, 0..1) — it must be surfaced with the
    'probabilidad estimada' wording AND labeled as a non-calibrated proxy, so the
    surface neither hides the number nor overstates it as a calibrated probability
    (quant-constitution: no fabricated calibration).
    """
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    start = src.find("function AssetWeeklyBody")
    assert start != -1, "AssetWeeklyBody (weekly inference surface) disappeared"
    end = src.find("export function ForecastingView", start)
    seg = src[start:end if end != -1 else len(src)]
    assert "probabilidad estimada" in seg, (
        "AssetWeeklyBody (weekly inference) shows directional predictions with no "
        "probabilistic wording. BL-03 requires 'probabilidad estimada ...' on every "
        "DIAGNOSTIC forecast surface, weekly inference included."
    )
    assert "confidence" in seg, (
        "The weekly probabilistic wording must be driven by the data's `confidence` "
        "field (rule-conviction proxy), not a literal."
    )
    assert re.search(r"proxy[^<\n]*no calibrada|no calibrada[^<\n]*proxy", seg), (
        "The weekly 'probabilidad estimada' must be explicitly labeled as a "
        "non-calibrated conviction proxy — presenting it as a calibrated probability "
        "would fabricate precision the rule engine does not have."
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


def test_legacy_caveat_headline_comes_from_ssot():
    """BL-04: the legacy DiagnosticCaveat must render its headline from the shared
    SSOT constant, unconditionally — not from any local string, and never behind a
    beats-the-bar ternary."""
    src = LEGACY_DASHBOARD.read_text(encoding="utf-8", errors="replace")
    body_start = src.find("function DiagnosticCaveat")
    assert body_start != -1, "DiagnosticCaveat disappeared from the legacy dashboard"
    body = src[body_start: src.find("function ", body_start + 10)]
    assert "FORECAST_DISCLAIMER_ZOO_TITLE" in body, (
        "Legacy DiagnosticCaveat no longer uses the SSOT headline constant "
        "(lib/ui/forecast-disclaimer.ts) — duplicated caveat copy drifts (BL-04)."
    )
    assert not re.search(r"\?\s*['\"`][^'\"`]*['\"`]\s*:\s*`?\$\{FORECAST_DISCLAIMER",
                         body), (
        "Legacy DiagnosticCaveat gates the SSOT headline behind a ternary — the "
        "headline must be unconditional (BL-04)."
    )
