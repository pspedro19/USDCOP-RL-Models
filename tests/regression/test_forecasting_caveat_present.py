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


_TS_CONST = re.compile(
    r"export\s+const\s+(\w+)\s*=\s*((?:'(?:[^'\\]|\\.)*'\s*\+?\s*)+);"
)


def _ts_string_constants(src: str) -> dict[str, str]:
    """Parse `export const NAME = '...' + '...';` string constants from a TS module.

    The disclaimer SSOT is plain concatenated single-quoted literals by design;
    if that ever stops parsing, the assertion below fails loudly instead of
    silently checking nothing.
    """
    out: dict[str, str] = {}
    for name, raw in _TS_CONST.findall(src):
        pieces = re.findall(r"'((?:[^'\\]|\\.)*)'", raw)
        out[name] = "".join(p.replace("\\'", "'") for p in pieces)
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
# defeated the previous prefix-only lock (M1/M2/M3 in the BL-01 review):
#   M1: 'Superficie de diagnóstico: señal validada, opere con confianza.'
#   M2: 'NO ES UNA SEÑAL CUALQUIERA: ES NUESTRA SEÑAL DE COMPRA MÁS CONFIABLE'
#   M3: 'Precisión direccional del 90% garantizada. Ejecute estas señales...'
FORBIDDEN_MARKETING = [
    r"\bopere\b",
    r"\bejecute\b",
    r"\bcompre\b",
    r"\bvenda\b",
    r"senal(es)? de (compra|venta|inversion segura)",
    r"senal(es)? (validada|confiable|segura|fuerte|ganadora|comprobada)",
    r"es (nuestra|la) (mejor )?senal",
    r"confianza",
    r"confiab",
    r"garantiz",
    r"validad",
    r"rentabilidad",
    r"recomendad",
    r"sin riesgo",
    r"asegurad",
    r"\bopere con\b",
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
        for pat in FORBIDDEN_MARKETING:
            if re.search(pat, text):
                offenders.append(f"{name}: /{pat}/ -> {consts[name]!r}")
    assert not offenders, (
        "Disclaimer copy contains promotional/action language — a diagnostic caveat "
        "that recommends acting is worse than no caveat:\n  " + "\n  ".join(offenders)
    )


def test_caveat_not_gated_only_to_model_zoo():
    """BL-02 (expected green after it lands): the caveat must also render for
    weekly_inference mode, not only under `isModelZoo`.

    Passes when either (a) the testid appears more than once (a second render path /
    shared component for the rule-based mode), or (b) the single banner is no longer
    wrapped exclusively in `{isModelZoo && (...)}`.
    """
    src = FORECASTING_VIEW.read_text(encoding="utf-8", errors="replace")
    occurrences = src.count('data-testid="da-caveat"')
    gated_only = re.search(
        r"\{isModelZoo\s*&&\s*\(\s*<div\s+data-testid=\"da-caveat\"", src
    )
    assert occurrences >= 2 or gated_only is None, (
        "The da-caveat banner renders only under isModelZoo — the weekly_inference "
        "(rule-based) surface shows results with no diagnostic caveat. BL-02 must "
        "render the caveat for both forecast modes."
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
