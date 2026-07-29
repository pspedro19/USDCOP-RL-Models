"""
Strategy Passport + Control Tower contract (CTR-PASSPORT-001)
=============================================================

BL-32 / FABRIC §24.4-§24.5. TS mirror:
``usdcop-trading-dashboard/lib/contracts/passport.contract.ts``.
Spec: ``.claude/specs/platform/passport-control-tower.md``.

The Passport is a **derived, read-only** view: identity + governance + lineage +
performance across five environments + execution/risk. FABRIC's rule is that it is
"un SELECT", never hand-written. Today the fact tables that would back that SELECT
(BL-18 metric engine, BL-21/BL-22 exec/fact_pnl, BL-24 lineage) DO NOT EXIST, so
this contract is deliberately built around one primitive:

    Sourced[T] = {"value": T | None, "source": {"path", "status", "pending"}}

Every single number carries the artifact it came from. A number with no published
artifact is NOT rendered as 0, "-" or an estimate: it is ``value=None`` with
``status="unavailable"`` and a ``pending`` string naming the backlog item that will
supply it. That is what makes the degradation honest instead of decorative, and it
is what turns the missing CODEX pieces into a *documented interface* rather than a
silent hole.

Constitutional invariants encoded here (they are asserted, not commented):

* §6 small sample — with ``n_trades < 20`` only count + PnL survive; Sharpe,
  p-value and DSR are nulled (:func:`suppress_small_sample`).
* §2 N_MAX is a SPEND CAP and never enters the DSR (:data:`N_MAX_TRIALS`), and the
  three N (family/cluster/global) are mandatory disclosure.
* strategy-contract §2 — no ``Infinity``/``NaN``/``undefined`` ever reaches JSON.
* §7 / approval-gates — the Passport is DIAGNOSTIC: it MUST NOT expose any action.
  :data:`FORBIDDEN_PASSPORT_ACTIONS` is the machine-readable form of that rule.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

PASSPORT_CONTRACT_ID = "CTR-PASSPORT-001"
PASSPORT_CONTRACT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Vocabularies (mirrored 1:1 in passport.contract.ts)
# ---------------------------------------------------------------------------

#: A field is either backed by a published artifact or it is not. There is no
#: third state — "estimated" would be a modelling decision, not engineering.
SOURCE_STATUSES: tuple[str, ...] = ("published", "unavailable")

#: FABRIC §24.4: "la misma métrica del mismo motor en las cinco columnas".
#: The single metric engine is BL-18; until it lands, each column declares its own
#: source and the Passport declares ``metric_engine`` unavailable.
PASSPORT_ENVS: tuple[str, ...] = ("backtest", "held_out", "paper", "canary", "live")

#: FABRIC §24.5 LIBRO: "conteo por estado".
BOOK_STATES: tuple[str, ...] = ("CHAMPION", "CANARY", "PAPER", "REDUCED", "QUARANTINED")

#: Retirement traffic light (quant-constitution §5). ``unknown`` is a first-class
#: value: a strategy with no signed withdrawal protocol is not "green".
RETIREMENT_SIGNALS: tuple[str, ...] = ("green", "yellow", "red", "unknown")

#: Three-clock monitoring (FABRIC §23 / BL-25): datos · modelo · **PnL**.
#: The names belong to the PRODUCER (``src/monitoring/system_health_contract.py::Clock``),
#: not to this consumer: calling the third one ``exec`` made the composer discard the
#: ``pnl`` clock whenever ``system_health.json`` published it (CODEX F-07).
#: ``tests/unit/test_passport_contract.py`` pins this tuple to the producer enum.
HEALTH_CLOCKS: tuple[str, ...] = ("data", "model", "pnl")

#: Trial lineage (ADR-0022). FT = predictive, AT = economic.
TRIAL_KINDS: tuple[str, ...] = ("forecast", "action")

# ---------------------------------------------------------------------------
# Constitutional constants
# ---------------------------------------------------------------------------

#: quant-constitution §6 — below this, only count and PnL are publishable.
MIN_TRADES_FOR_RATIOS = 20

#: FABRIC §9.7 — commitment device against p-hacking. **Spend cap only: it NEVER
#: enters the DSR or any statistical formula.** Disclosed next to N_global so the
#: reader sees how much of the budget has been burned.
N_MAX_TRIALS = 989

#: quant-constitution §2 — the bar for any edge claim.
DSR_BAR = 0.95

#: Inferential quantities that :func:`suppress_small_sample` nulls out.
SMALL_SAMPLE_SUPPRESSED_FIELDS: tuple[str, ...] = (
    "sharpe", "sortino", "calmar", "p_value", "dsr_family", "dsr_cluster",
    "dsr_global", "psr", "bootstrap_ci_low", "bootstrap_ci_high",
)

#: The §6 verdict when the published source does not let us determine N.
#: **Fail-closed** (S-04): the guard used to return untouched on an unknown N
#: ("absence of N is not evidence of N<20"), which is backwards for a
#: *publication* guard — the manifests that omit the count are exactly the ones
#: with 1-3 trades. ``btc_hodl_b1`` published Sharpe 0.793 and p=0.0242 off ONE
#: trade with both guards green in both languages.
UNDETERMINABLE_N_REASON = (
    "N no determinable desde la fuente publicada (fail-closed, quant-constitution §6: "
    "sin conteo de trades no se publica Sharpe/p-value/DSR)"
)

#: The Passport/Control Tower is a DIAGNOSTIC surface (approval-gates.md §3,
#: plan 00 ACTION vs DIAGNOSTIC). Vote 2 lives ONLY on /dashboard. Any of these
#: appearing in a Passport payload is a contract violation, not a feature.
FORBIDDEN_PASSPORT_ACTIONS: tuple[str, ...] = (
    "approve", "reject", "promote", "deploy", "vote", "kill_switch", "execute",
)

# ---------------------------------------------------------------------------
# Contenido obligatorio por bloque (espejo de las interfaces TS)
# ---------------------------------------------------------------------------

#: Lo que CADA bloque del Passport tiene que declarar.
#:
#: Es el espejo en tiempo de ejecución de las ``export interface`` del contrato TS
#: (``PassportIdentity``/``PassportGovernance``/``PassportLineage``/
#: ``PassportLiveState``/``PassportRisk``). Los tipos de TS se borran al compilar, así
#: que **ningún runtime puede derivar esta lista**: cada lenguaje la lleva literal en
#: UN solo sitio y las dos son espejo la una de la otra —
#:
#:   * Python -> este mapa
#:   * TS     -> ``PASSPORT_BLOCK_FIELDS`` en ``passport.contract.ts``
#:
#: ``tests/unit/test_passport_contract.py::test_block_fields_mirror_the_ts_contract``
#: parsea las interfaces Y el mapa TS, y se pone rojo en cuanto uno de los tres deriva.
#:
#: Por qué existe: ``validate_strategy_passport`` solo comprobaba que la CLAVE
#: estuviera, así que ``"governance": {}`` — cero trials, cero DSR, cero N — validaba
#: exactamente igual que un bloque completo.
PASSPORT_BLOCK_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": (
        "strategy_id", "asset_id", "display_name", "surface", "engine_type",
        "status", "active_version", "timeframe",
    ),
    "governance": (
        "n_trials_total", "n_trials_forecast", "n_trials_action",
        "n_family", "n_cluster", "n_global", "dsr_family", "dsr_bar",
        "approval_status", "gates", "withdrawal_protocol",
        "retirement_signal", "retirement_reason",
    ),
    "lineage": (
        "model_versions", "spec_fingerprint", "feature_set_hash", "policy_hash",
        "lineage_graph",
    ),
    "live": (
        "open_orders", "last_fill_at", "quarantined", "reconciled",
        "kill_switch_engaged", "deploy_status", "last_signal_at",
    ),
    "risk": (
        "current_exposure", "vol_target_pct", "vol_forecast_pct", "m_forward",
        "m_dd", "rho_max", "turnover",
    ),
}

#: De los anteriores, los que el contrato declara como valor DIRECTO (no ``Sourced``):
#: identificadores, la barra constitucional y el semáforo de retiro. Del resto se exige
#: la forma ``Sourced`` completa.
PASSPORT_PLAIN_FIELDS: dict[str, tuple[str, ...]] = {
    "identity": ("strategy_id", "asset_id", "display_name", "surface",
                 "engine_type", "status", "timeframe"),
    "governance": ("dsr_bar", "retirement_signal", "retirement_reason"),
    "lineage": (),
    "live": (),
    "risk": (),
}

#: ``EnvPerformance`` — cada una de las cinco columnas del §24.4. Mismo espejo, mismo
#: motivo: ``"performance": {"live": {}}`` declaraba el entorno sin decir nada de él.
ENV_PERFORMANCE_FIELDS: tuple[str, ...] = (
    "env", "period_label", "return_pct", "n_trades", "max_dd_pct", "win_rate_pct",
    "profit_factor", "sharpe", "calmar", "p_value", "dsr_family", "timing_ratio",
    "insufficient_trades",
)

#: Los dos campos del entorno que NO son ``Sourced``: la etiqueta del entorno y la
#: bandera que deja el guard §6.
ENV_PERFORMANCE_PLAIN_FIELDS: tuple[str, ...] = ("env", "insufficient_trades")

#: **Un hueco declarado no es un error.** ``policy_hash`` no tiene productor hasta
#: BL-45, ``dsr_family`` está ``unavailable`` para casi todas las estrategias y un
#: activo sin trials es un estado legítimo. Lo que se exige no es un valor: es la
#: forma alternativa que el propio contrato ya define para un hueco —
#: ``value=None`` + ``status="unavailable"`` + ``pending`` no vacío. La regla es
#: **"sin agujeros MUDOS", no "sin agujeros"**: un ``{}`` no puede expresar quién debe
#: el dato; ``None`` + ``pending="BL-45 …"`` sí.

# ---------------------------------------------------------------------------
# Sourced values — the core primitive
# ---------------------------------------------------------------------------


def sanitize_number(value: Any) -> Any:
    """Return ``None`` for NaN/Infinity; pass everything else through.

    strategy-contract.md §2: no published JSON may carry ``Infinity``/``NaN``.
    Applied at the value level so a bad float can never reach the browser.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return None if not math.isfinite(float(value)) else value
    return value


def sourced(value: Any, path: str, *, note: str | None = None) -> dict:
    """A value that came from a published artifact.

    ``path`` is repo-root-relative and must be a real, published file — that is
    the whole point: "ningún número de performance sin fuente publicada".
    """
    return {
        "value": sanitize_number(value),
        "source": {"path": path, "status": "published", "pending": None,
                   "note": note},
    }


def unavailable(pending: str, *, note: str | None = None) -> dict:
    """A value NO published artifact can supply yet.

    ``pending`` names the backlog item / system that will supply it (e.g.
    ``"BL-22 fact_pnl"``). This is the documented acople surface: when the
    producing side lands, only the composer changes — the contract does not.
    """
    return {
        "value": None,
        "source": {"path": None, "status": "unavailable", "pending": pending,
                   "note": note},
    }


def is_available(field: Any) -> bool:
    """True when a Sourced field actually carries a published value."""
    return (
        isinstance(field, dict)
        and isinstance(field.get("source"), dict)
        and field["source"].get("status") == "published"
        and field.get("value") is not None
    )


# ---------------------------------------------------------------------------
# Small-sample guard (quant-constitution §6)
# ---------------------------------------------------------------------------


def _format_n(n: int | float) -> str:
    """Render N identically in both runtimes (TS ``String(3.0)`` is ``"3"``)."""
    return str(int(n)) if float(n).is_integer() else repr(float(n))


def resolve_n_trades(block: Any) -> int | float | None:
    """The trade count of a performance/sleeve block, or ``None`` when it cannot
    be determined **with certainty** from what was published.

    ``None`` covers all three shapes the real artifacts produce: the key is
    absent, the field is ``unavailable``, or the field is *published with value
    null* (``sourced(None, path)`` — what the composer emits when a manifest
    headline carries no trade count). All three mean "we do not know N".
    """
    if not isinstance(block, dict):
        return None
    field = block.get("n_trades")
    n = field.get("value") if isinstance(field, dict) else field
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        return None
    return n


def small_sample_reason(n: int | float | None) -> str:
    """The single sentence both runtimes attach to a suppressed field."""
    if n is None:
        return UNDETERMINABLE_N_REASON
    return (f"N={_format_n(n)} < {MIN_TRADES_FOR_RATIOS} "
            f"(quant-constitution §6: solo conteo y PnL)")


def small_sample_violations(block: Any, label: str) -> list[str]:
    """Inferential fields published on a block whose N does not license them.

    A ratio is publishable **only** when the block itself carries a published
    trade count ``>= 20``. Unknown N is a violation, not a pass.
    """
    n = resolve_n_trades(block)
    if n is not None and n >= MIN_TRADES_FOR_RATIOS:
        return []
    detail = (f"N={_format_n(n)} < {MIN_TRADES_FOR_RATIOS}" if n is not None
              else "N no determinable desde la fuente publicada (fail-closed)")
    return [
        f"{label}.{key}: published with {detail} (quant-constitution §6)"
        for key in SMALL_SAMPLE_SUPPRESSED_FIELDS
        if is_available(block.get(key) if isinstance(block, dict) else None)
    ]


def suppress_small_sample(env_perf: dict) -> dict:
    """Null every inferential field of an env-performance block unless a
    published trade count of at least 20 licenses it.

    Descriptive quantities (return, PnL, drawdown, win rate, counts) survive —
    they describe what happened. Ratios and p-values do not: "con N < 20 trades
    se reporta solo conteo y PnL". Mutates and returns ``env_perf``.

    **Fail-closed on an unknown N** (S-04): the suppressed field keeps its
    Sourced shape but flips to ``unavailable`` with the reason, so the UI says
    WHY. Publishing a Sharpe requires *proving* N >= 20 from the artifact; the
    absence of the count is not a licence, it is the most common way the count
    is missing precisely because the sample is tiny.
    """
    n = resolve_n_trades(env_perf)
    if n is not None and n >= MIN_TRADES_FOR_RATIOS:
        return env_perf
    env_perf["insufficient_trades"] = True
    reason = small_sample_reason(n)
    for key in SMALL_SAMPLE_SUPPRESSED_FIELDS:
        if key in env_perf:
            env_perf[key] = unavailable(reason)
    return env_perf


def can_show_ratios(n_trades: int | None) -> bool:
    """Mirror of ``canShowRatios`` in ui.contract.ts."""
    return (n_trades or 0) >= MIN_TRADES_FOR_RATIOS


# ---------------------------------------------------------------------------
# Validators — shared verdicts with the TS mirror
# ---------------------------------------------------------------------------


def _err(errors: list[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def validate_sourced(field: Any, label: str) -> list[str]:
    """Structural validation of one Sourced field."""
    errors: list[str] = []
    if not isinstance(field, dict):
        return [f"{label}: not a Sourced object"]
    _err(errors, "value" in field, f"{label}: missing 'value'")
    src = field.get("source")
    if not isinstance(src, dict):
        return errors + [f"{label}: missing/invalid 'source'"]
    status = src.get("status")
    _err(errors, status in SOURCE_STATUSES, f"{label}: bad source.status {status!r}")
    if status == "published":
        _err(errors, bool(src.get("path")),
             f"{label}: published fields MUST name their artifact path")
    if status == "unavailable":
        _err(errors, field.get("value") is None,
             f"{label}: unavailable fields MUST have value=None")
        _err(errors, bool(src.get("pending")),
             f"{label}: unavailable fields MUST declare what they are pending on")
    value = field.get("value")
    _err(errors, sanitize_number(value) == value or value is None,
         f"{label}: non-finite number reached the contract")
    return errors


def _is_sourced_shaped(node: Any) -> bool:
    """True when ``node`` at least *looks* like a Sourced field.

    Mirrors the test :func:`_walk_sourced` applies, so the two agree on what the
    tree-walk already covers and what a block check still has to catch.
    """
    return (isinstance(node, dict) and "value" in node
            and isinstance(node.get("source"), dict))


def validate_block(field_values: Any, label: str, fields: Iterable[str],
                   plain_fields: Iterable[str] = ()) -> list[str]:
    """Content validation of ONE declared block (identity/governance/…/an env).

    Every field the contract declares must be present, and every field that is not
    in ``plain_fields`` must carry the full ``Sourced`` shape. A field may perfectly
    well be empty — but only as ``value=None`` + ``status="unavailable"`` +
    ``pending``: a declared hole, never a mute one.
    """
    if not isinstance(field_values, dict):
        return [f"{label}: not an object"]
    plain = set(plain_fields)
    errors: list[str] = []
    for field in fields:
        if field not in field_values:
            errors.append(f"{label}: missing '{field}'")
            continue
        if field in plain:
            continue
        value = field_values[field]
        # Los Sourced BIEN FORMADOS los valida ya el walk sobre el payload entero;
        # aquí solo hace falta atrapar lo que ni siquiera tiene esa forma (un ``{}``,
        # un número pelado, un ``None``), que el walk no visita y por eso no veía.
        if not _is_sourced_shaped(value):
            errors.extend(validate_sourced(value, f"{label}.{field}"))
    return errors


def _walk_sourced(node: Any, prefix: str = "") -> Iterable[tuple[str, dict]]:
    """Yield every Sourced-shaped dict in a payload tree."""
    if isinstance(node, dict):
        if "value" in node and isinstance(node.get("source"), dict):
            yield prefix or "<root>", node
            return
        for key, val in node.items():
            yield from _walk_sourced(val, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for i, val in enumerate(node):
            yield from _walk_sourced(val, f"{prefix}[{i}]")


def validate_strategy_passport(payload: Any) -> list[str]:
    """Validate a ``v_strategy_passport`` composition. Returns error strings."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["passport: not an object"]
    _err(errors, payload.get("contract") == PASSPORT_CONTRACT_ID,
         f"passport.contract must be {PASSPORT_CONTRACT_ID}")
    for key in ("strategy_id", "generated_at", "identity", "governance",
                "lineage", "performance", "live", "risk"):
        _err(errors, key in payload, f"passport: missing '{key}'")

    # Presencia de la clave NO es contenido: un `"governance": {}` publicaba un
    # Passport sin gobernanza, sin linaje y sin riesgo con cero errores.
    for block, fields in PASSPORT_BLOCK_FIELDS.items():
        if block in payload:
            errors.extend(validate_block(
                payload[block], f"passport.{block}", fields,
                PASSPORT_PLAIN_FIELDS.get(block, ()),
            ))

    gov = payload.get("governance")
    if isinstance(gov, dict):
        # §2: la barra viaja CON el bloque — un DSR sin su barra es un número
        # que nadie puede juzgar. Es una constante declarada, no una medición.
        if "dsr_bar" in gov:
            _err(errors, gov.get("dsr_bar") == DSR_BAR,
                 f"passport.governance.dsr_bar must be {DSR_BAR} (quant-constitution §2)")
        if "retirement_signal" in gov:
            _err(errors, gov.get("retirement_signal") in RETIREMENT_SIGNALS,
                 f"passport.governance.retirement_signal: bad value "
                 f"{gov.get('retirement_signal')!r}")

    perf = payload.get("performance")
    if isinstance(perf, dict):
        missing = [e for e in PASSPORT_ENVS if e not in perf]
        _err(errors, not missing,
             f"passport.performance must declare all five envs; missing {missing}")
        for env, block in perf.items():
            _err(errors, env in PASSPORT_ENVS, f"passport.performance: unknown env {env!r}")
            errors.extend(validate_block(
                block, f"passport.performance.{env}",
                ENV_PERFORMANCE_FIELDS, ENV_PERFORMANCE_PLAIN_FIELDS,
            ))
            errors.extend(small_sample_violations(block, f"passport.performance.{env}"))
    else:
        errors.append("passport.performance: not an object")

    for key in FORBIDDEN_PASSPORT_ACTIONS:
        _err(errors, key not in payload,
             f"passport: DIAGNOSTIC surface must not expose action {key!r}")

    for label, field in _walk_sourced(payload):
        errors.extend(validate_sourced(field, f"passport.{label}"))
    return errors


def validate_control_tower(payload: Any) -> list[str]:
    """Validate a Control Tower snapshot (§24.5 LIBRO/SLEEVES/DATOS)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["tower: not an object"]
    _err(errors, payload.get("contract") == PASSPORT_CONTRACT_ID,
         f"tower.contract must be {PASSPORT_CONTRACT_ID}")
    for key in ("generated_at", "book", "sleeves", "data", "pending_interfaces"):
        _err(errors, key in payload, f"tower: missing '{key}'")

    book = payload.get("book")
    if isinstance(book, dict):
        counts = book.get("state_counts")
        if isinstance(counts, dict):
            unknown = [s for s in counts if s not in BOOK_STATES]
            _err(errors, not unknown, f"tower.book.state_counts: unknown states {unknown}")
        else:
            errors.append("tower.book.state_counts: not an object")

    sleeves = payload.get("sleeves")
    if isinstance(sleeves, list):
        for i, sleeve in enumerate(sleeves):
            if not isinstance(sleeve, dict):
                errors.append(f"tower.sleeves[{i}]: not an object")
                continue
            _err(errors, bool(sleeve.get("strategy_id")),
                 f"tower.sleeves[{i}]: missing strategy_id")
            signal = sleeve.get("retirement_signal")
            _err(errors, signal in RETIREMENT_SIGNALS,
                 f"tower.sleeves[{i}].retirement_signal: bad value {signal!r}")
            # §6 applies to the SLEEVE row too: it is a decision surface, and it
            # is where `btc_hodl_b1: sharpe=0.793 n_trades=null` was published.
            errors.extend(small_sample_violations(sleeve, f"tower.sleeves[{i}]"))
    else:
        errors.append("tower.sleeves: not a list")

    data = payload.get("data")
    if isinstance(data, dict):
        n_max = data.get("n_max_trials")
        n_max_value = n_max.get("value") if isinstance(n_max, dict) else n_max
        _err(errors, n_max_value == N_MAX_TRIALS,
             f"tower.data.n_max_trials must be {N_MAX_TRIALS} (spend cap, never in the DSR)")

    for key in FORBIDDEN_PASSPORT_ACTIONS:
        _err(errors, key not in payload,
             f"tower: DIAGNOSTIC surface must not expose action {key!r}")

    for label, field in _walk_sourced(payload):
        errors.extend(validate_sourced(field, f"tower.{label}"))
    return errors


__all__ = [
    "PASSPORT_CONTRACT_ID", "PASSPORT_CONTRACT_VERSION",
    "SOURCE_STATUSES", "PASSPORT_ENVS", "BOOK_STATES", "RETIREMENT_SIGNALS",
    "HEALTH_CLOCKS", "TRIAL_KINDS",
    "MIN_TRADES_FOR_RATIOS", "N_MAX_TRIALS", "DSR_BAR",
    "SMALL_SAMPLE_SUPPRESSED_FIELDS", "FORBIDDEN_PASSPORT_ACTIONS",
    "UNDETERMINABLE_N_REASON",
    "PASSPORT_BLOCK_FIELDS", "PASSPORT_PLAIN_FIELDS",
    "ENV_PERFORMANCE_FIELDS", "ENV_PERFORMANCE_PLAIN_FIELDS",
    "sanitize_number", "sourced", "unavailable", "is_available",
    "resolve_n_trades", "small_sample_reason", "small_sample_violations",
    "suppress_small_sample", "can_show_ratios",
    "validate_sourced", "validate_block", "validate_strategy_passport",
    "validate_control_tower",
]
