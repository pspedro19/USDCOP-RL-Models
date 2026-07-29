"""CTR-PASSPORT-001 contract tests (BL-32) — Python side of the Py↔TS mirror.

TS twin: ``usdcop-trading-dashboard/tests/unit/contracts/passport-contract.test.ts``.
Both runners assert the same vocabularies and the same verdicts; a drift on either
side goes red. What is locked here is what makes the Passport honest:

1. an unavailable field can never carry a value nor hide who owes it;
2. a published field can never be anonymous;
3. N<20 can never publish a Sharpe / p-value / DSR;
4. the DIAGNOSTIC surface can never grow an action.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from src.contracts.passport import (
    BOOK_STATES,
    DSR_BAR,
    FORBIDDEN_PASSPORT_ACTIONS,
    HEALTH_CLOCKS,
    MIN_TRADES_FOR_RATIOS,
    N_MAX_TRIALS,
    PASSPORT_CONTRACT_ID,
    PASSPORT_ENVS,
    RETIREMENT_SIGNALS,
    SOURCE_STATUSES,
    is_available,
    sanitize_number,
    sourced,
    suppress_small_sample,
    unavailable,
    validate_control_tower,
    validate_sourced,
    validate_strategy_passport,
)

TS_CONTRACT = "usdcop-trading-dashboard/lib/contracts/passport.contract.ts"


# --------------------------------------------------------------- vocabularies

def test_five_environments_in_order():
    assert PASSPORT_ENVS == ("backtest", "held_out", "paper", "canary", "live")


def test_five_book_states():
    assert BOOK_STATES == ("CHAMPION", "CANARY", "PAPER", "REDUCED", "QUARANTINED")


def test_only_two_source_statuses():
    """A third status ("estimated") would be a modelling decision, not engineering."""
    assert SOURCE_STATUSES == ("published", "unavailable")


def test_unknown_is_a_first_class_retirement_signal():
    assert "unknown" in RETIREMENT_SIGNALS


def test_three_clocks_use_the_producers_vocabulary():
    """F-07: el tercer reloj del §23 es `pnl` (datos/modelo/PnL), jamás `exec`."""
    assert HEALTH_CLOCKS == ("data", "model", "pnl")


def test_clock_names_match_the_producer_enum():
    """Frontera consumidor↔productor: `control__system_health` es quien EMITE los
    relojes (`src/monitoring/system_health_contract.py::Clock`). Si el Passport
    nombra uno distinto, el reloj publicado se descarta en silencio — que es
    exactamente lo que pasaba con `exec`. Este test lo hace imposible."""
    from src.monitoring.system_health_contract import Clock

    assert HEALTH_CLOCKS == tuple(c.value for c in Clock)


def test_constitutional_constants():
    assert MIN_TRADES_FOR_RATIOS == 20
    assert N_MAX_TRIALS == 989          # spend cap only — never in the DSR
    assert DSR_BAR == 0.95


def test_ts_mirror_declares_the_same_vocabularies():
    """Cheap structural parity: the TS file must literally contain the same tuples."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    text = (root / TS_CONTRACT).read_text(encoding="utf-8")
    for token in (*PASSPORT_ENVS, *BOOK_STATES, *RETIREMENT_SIGNALS, *SOURCE_STATUSES):
        assert f"'{token}'" in text, f"TS mirror is missing {token!r}"
    assert f"N_MAX_TRIALS = {N_MAX_TRIALS}" in text
    assert f"MIN_TRADES_FOR_RATIOS = {MIN_TRADES_FOR_RATIOS}" in text


# ----------------------------------------------------------- Sourced primitive

def test_published_fields_must_name_their_artifact():
    assert validate_sourced(sourced(1.23, "public/data/x.json"), "f") == []
    anonymous = {"value": 1, "source": {"path": None, "status": "published", "pending": None}}
    assert any("MUST name their artifact" in e for e in validate_sourced(anonymous, "f"))


def test_unavailable_fields_must_be_null_and_name_their_owner():
    assert validate_sourced(unavailable("BL-22 fact_pnl"), "f") == []
    with_value = {"value": 0, "source": {"path": None, "status": "unavailable", "pending": "BL-22"}}
    assert any("MUST have value=None" in e for e in validate_sourced(with_value, "f"))
    no_owner = {"value": None, "source": {"path": None, "status": "unavailable", "pending": None}}
    assert any("pending on" in e for e in validate_sourced(no_owner, "f"))


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_never_reaches_json(bad):
    assert sanitize_number(bad) is None
    assert sourced(bad, "p")["value"] is None
    assert "Infinity" not in json.dumps(sourced(bad, "p"))


def test_is_available():
    assert is_available(sourced(1, "p"))
    assert not is_available(unavailable("BL-x"))
    assert not is_available(sourced(None, "p"))


# --------------------------------------------------------- small sample (§6)

def _env(n_trades):
    return {
        "env": "live",
        "period_label": sourced("2026", "p"),
        "return_pct": sourced(3.36, "p"),
        "n_trades": unavailable("n/d") if n_trades is None else sourced(n_trades, "p"),
        "max_dd_pct": sourced(1.5, "p"),
        "win_rate_pct": sourced(72.7, "p"),
        "profit_factor": sourced(2.408, "p"),
        "sharpe": sourced(1.9, "p"),
        "calmar": sourced(2.24, "p"),
        "p_value": sourced(0.03, "p"),
        "dsr_family": sourced(0.42, "p"),
        "timing_ratio": sourced(0.02, "p"),
        "insufficient_trades": False,
    }


def test_small_sample_strips_inferential_keeps_descriptive():
    out = suppress_small_sample(_env(11))
    assert out["insufficient_trades"] is True
    for key in ("sharpe", "calmar", "p_value", "dsr_family"):
        assert out[key]["value"] is None
        assert out[key]["source"]["status"] == "unavailable"
        assert "N=11" in out[key]["source"]["pending"]
    # Descriptive quantities describe what happened — they survive.
    assert out["return_pct"]["value"] == 3.36
    assert out["n_trades"]["value"] == 11
    assert out["max_dd_pct"]["value"] == 1.5


def test_small_sample_noop_at_or_above_twenty():
    assert suppress_small_sample(_env(20))["sharpe"]["value"] == 1.9


def test_unknown_n_is_fail_closed():
    """S-04: absence of N is NOT permission to publish a ratio.

    The first version of this guard returned untouched when ``n_trades`` was
    ``None`` ("absence of N is not evidence of N<20"). That reasoning is exactly
    backwards for a publication guard: the manifests that omit the trade count
    are precisely the ones with 1-3 trades (btc_hodl_b1 published Sharpe 0.793 /
    p=0.0242 off a SINGLE trade). §6 says with N<20 only count and PnL are
    publishable; a number whose N cannot be determined from the published source
    cannot be shown to satisfy that, so it is suppressed.
    """
    out = suppress_small_sample(_env(None))
    for key in ("sharpe", "calmar", "p_value", "dsr_family"):
        assert out[key]["value"] is None, f"{key} survived an undeterminable N"
        assert out[key]["source"]["status"] == "unavailable"
        assert "no determinable" in out[key]["source"]["pending"]
    assert out["insufficient_trades"] is True
    # Descriptive quantities still survive — they are not inferential.
    assert out["return_pct"]["value"] == 3.36
    assert out["max_dd_pct"]["value"] == 1.5


def test_published_but_null_n_is_also_fail_closed():
    """The real shape of the defect: ``n_trades`` IS published, with value null.

    ``sourced(None, path)`` is what the composer emits when the manifest headline
    has no trade count. It is ``status: published``, so a naive "is n published?"
    check would wave it through.
    """
    env = _env(11)
    env["n_trades"] = sourced(None, "p")
    out = suppress_small_sample(env)
    assert out["sharpe"]["value"] is None
    assert out["insufficient_trades"] is True


# ------------------------------------------------------------- payload shapes

# The blocks below are DELIBERATELY populated. The first version of this fixture used
# `"identity": {}, "governance": {}, "lineage": {}, "live": {}, "risk": {}`, and the
# consequence was that a Passport with **cero trials, cero DSR, cero linaje y cero
# riesgo** validated exactly like a complete one: shrinking the mandatory-key tuple in
# `validate_strategy_passport` from eight keys to three left the suite at 44 passed.
# A fixture that is emptier than the real payload cannot detect an emptier payload.
# Field names mirror `passport.contract.ts` 1:1 — `test_python_fixture_mirrors_the_ts_blocks`
# pins that mirror so the two cannot drift.

def _identity():
    return {
        "strategy_id": "smart_simple_v11",
        "asset_id": "usdcop",
        "display_name": "Smart Simple v11",
        "surface": "action",
        "engine_type": "composite",
        "status": "production",
        "active_version": sourced("v11", "public/data/strategies/registry.json"),
        "timeframe": "H5",
    }


def _governance():
    return {
        "n_trials_total": sourced(63, "data/control-tower/governance.json"),
        "n_trials_forecast": sourced(21, "data/control-tower/governance.json"),
        "n_trials_action": sourced(42, "data/control-tower/governance.json"),
        "n_family": sourced(12, "data/control-tower/governance.json"),
        "n_cluster": sourced(30, "data/control-tower/governance.json"),
        "n_global": sourced(63, "data/control-tower/governance.json"),
        "dsr_family": sourced(0.83, "data/approvals/smart_simple_v11.json"),
        "dsr_bar": DSR_BAR,
        "approval_status": sourced("APPROVED", "data/approvals/smart_simple_v11.json"),
        "gates": sourced([], "data/approvals/smart_simple_v11.json"),
        "withdrawal_protocol": sourced(
            ".claude/specs/assets/usdcop/WITHDRAWAL-PROTOCOL.md",
            "data/control-tower/governance.json"),
        "retirement_signal": "unknown",
        "retirement_reason": "sin evaluación de retiro POR ESTRATEGIA publicada (BL-25)",
    }


def _lineage():
    return {
        "model_versions": sourced([], "public/data/strategies/smart_simple_v11/manifest.json"),
        "spec_fingerprint": unavailable("BL-17 — fingerprints canónicos"),
        "feature_set_hash": unavailable("BL-39 — feature contracts por estrategia-versión"),
        "policy_hash": unavailable("BL-45 — motor de políticas (policy_hash/params_hash)"),
        "lineage_graph": unavailable("BL-24 — nodes/edges de linaje"),
    }


def _live():
    return {
        "open_orders": unavailable("BL-21 — event sourcing exec.*"),
        "last_fill_at": unavailable("BL-21 — event sourcing exec.*"),
        "quarantined": unavailable("BL-21 — cuarentena por reconciliación"),
        "reconciled": unavailable("BL-21/BL-22 — reconciliación contra fills"),
        "kill_switch_engaged": unavailable("BL-30 — kill switch independiente de Airflow"),
        "deploy_status": unavailable("sin deploy_status.json publicado"),
        "last_signal_at": sourced("2026-07-27T13:00:00Z",
                                  "public/data/strategies/smart_simple_v11/manifest.json"),
    }


def _risk():
    return {
        "current_exposure": unavailable("BL-26 — portfolio_snapshot"),
        "vol_target_pct": unavailable("BL-27 — allocator v1"),
        "vol_forecast_pct": unavailable("BL-27 — allocator v1"),
        "m_forward": unavailable("BL-27 — multiplicadores m_forward/m_dd"),
        "m_dd": unavailable("BL-27 — multiplicadores m_forward/m_dd"),
        "rho_max": unavailable("BL-26 — matriz de correlación entre sleeves"),
        "turnover": unavailable("BL-22 — fact_position"),
    }


#: The eight keys `validate_strategy_passport` declares mandatory, with the builder that
#: produces a realistic instance of each. Driving the negative tests off this map is what
#: makes "the block is mandatory" testable one block at a time.
PASSPORT_BLOCK_BUILDERS = {
    "identity": _identity,
    "governance": _governance,
    "lineage": _lineage,
    "live": _live,
    "risk": _risk,
}


def _passport(**overrides):
    perf = {env: {**_env(25), "env": env} for env in PASSPORT_ENVS}
    payload = {
        "contract": PASSPORT_CONTRACT_ID,
        "contract_version": "1.0.0",
        "strategy_id": "smart_simple_v11",
        "generated_at": "2026-07-28T00:00:00Z",
        "identity": _identity(),
        "governance": _governance(),
        "lineage": _lineage(),
        "performance": perf,
        "live": _live(),
        "risk": _risk(),
    }
    payload.update(overrides)
    return payload


def test_valid_passport_passes():
    assert validate_strategy_passport(_passport()) == []


# ------------------------------------------------- mandatory blocks (BL-32 mutation)

#: Everything `validate_strategy_passport` refuses to compose a Passport without.
MANDATORY_PASSPORT_KEYS = ("strategy_id", "generated_at", "identity", "governance",
                           "lineage", "performance", "live", "risk")


@pytest.mark.parametrize("key", MANDATORY_PASSPORT_KEYS)
def test_every_declared_passport_block_is_mandatory(key):
    """Rojo con: recortar la tupla de claves obligatorias de `validate_strategy_passport`
    (`src/contracts/passport.py`) a `("strategy_id", "generated_at", "performance")`.

    Ese recorte deja pasar un Passport SIN gobernanza, SIN linaje y SIN riesgo — es decir,
    sin trials, sin DSR, sin `policy_hash` y sin exposición — y la suite seguía en 44
    passed porque ningún test borraba jamás un bloque. Cada bloque se comprueba por
    separado: un solo test agregado se pondría verde en cuanto UNA de las ocho claves
    sobreviviera.
    """
    payload = _passport()
    del payload[key]
    errors = validate_strategy_passport(payload)
    assert f"passport: missing '{key}'" in errors, (
        f"validate_strategy_passport aceptó un Passport sin '{key}': {errors}")


#: The Control Tower half has exactly the same hole and exactly the same fix.
MANDATORY_TOWER_KEYS = ("generated_at", "book", "sleeves", "data", "pending_interfaces")


@pytest.mark.parametrize("key", MANDATORY_TOWER_KEYS)
def test_every_declared_tower_block_is_mandatory(key):
    """Rojo con: recortar la tupla de `validate_control_tower` (`src/contracts/passport.py`)
    a `("generated_at",)` — la Torre validaría sin LIBRO, sin SLEEVES y sin DATOS."""
    payload = _tower()
    del payload[key]
    errors = validate_control_tower(payload)
    assert f"tower: missing '{key}'" in errors, (
        f"validate_control_tower aceptó una Torre sin '{key}': {errors}")


# ------------------------------------------------------- Py↔TS block-shape parity

_TS_BLOCK_INTERFACES = {
    "identity": "PassportIdentity",
    "governance": "PassportGovernance",
    "lineage": "PassportLineage",
    "live": "PassportLiveState",
    "risk": "PassportRisk",
}


def _ts_interface_fields(interface: str) -> set[str]:
    """Field names declared by one `export interface` of the TS mirror.

    The TS contract is the only place in the repo where the SHAPE of each Passport block
    is declared (the Python validator only checks that the key exists). Parsing it is
    therefore the only way a Python test can assert "governance declara n_trials_total".
    """
    root = Path(__file__).resolve().parents[2]
    text = (root / TS_CONTRACT).read_text(encoding="utf-8")
    match = re.search(rf"export interface {interface} \{{(.*?)\n\}}", text, re.S)
    assert match, f"{TS_CONTRACT} no declara `export interface {interface}`"
    body = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)   # drop doc comments
    body = re.sub(r"//[^\n]*", "", body)
    return set(re.findall(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\??\s*:", body, re.M))


@pytest.mark.parametrize("block,interface", sorted(_TS_BLOCK_INTERFACES.items()))
def test_python_fixture_mirrors_the_ts_blocks(block, interface):
    """Cada bloque del Passport declara EXACTAMENTE los campos del contrato TS.

    Rojo con: borrar `n_trials_total: SourcedNumber;` de `export interface
    PassportGovernance` en `usdcop-trading-dashboard/lib/contracts/passport.contract.ts`
    (o vaciar cualquiera de las cinco interfaces).

    Por qué existe: `validate_strategy_passport` solo comprueba PRESENCIA de la clave, así
    que `governance: {}` valida igual que un bloque completo. La forma de cada bloque solo
    está declarada en el espejo TS; este test la ata al fixture Python para que un bloque
    vaciado en cualquiera de los dos lados se vea, y para que el fixture no pueda volver a
    ser `{}` (que es lo que hacía inerte a toda la suite).
    """
    declared = _ts_interface_fields(interface)
    fixture = set(PASSPORT_BLOCK_BUILDERS[block]().keys())
    assert fixture == declared, (
        f"passport.{block} diverge del contrato TS {interface}: "
        f"faltan {sorted(declared - fixture)}, sobran {sorted(fixture - declared)}")


def test_governance_declares_trials_dsr_and_the_bar():
    """El bloque que la mutación TS (`governance: {} as never`) vaciaba sin que nadie lo
    notase: cero trials, cero DSR, cero N.

    Rojo con: borrar `n_trials_total`/`dsr_family`/`dsr_bar` de `PassportGovernance` en
    `passport.contract.ts` (el test lee del contrato, no de una lista a mano).
    """
    declared = _ts_interface_fields("PassportGovernance")
    for field in ("n_trials_total", "n_trials_forecast", "n_trials_action",
                  "n_family", "n_cluster", "n_global", "dsr_family", "dsr_bar"):
        assert field in declared, f"PassportGovernance ya no declara {field!r}"
    gov = _governance()
    assert gov["dsr_bar"] == DSR_BAR                      # §2: la barra viaja con el bloque
    assert validate_sourced(gov["n_trials_total"], "governance.n_trials_total") == []
    assert validate_sourced(gov["dsr_family"], "governance.dsr_family") == []


def test_all_five_environments_must_be_declared():
    p = _passport()
    del p["performance"]["canary"]
    assert any("missing ['canary']" in e for e in validate_strategy_passport(p))


def test_sharpe_published_with_small_n_is_rejected():
    """The exact §6 violation this contract exists to make impossible."""
    p = _passport()
    p["performance"]["live"] = _env(3)          # deliberately NOT suppressed
    errors = " ".join(validate_strategy_passport(p))
    assert "performance.live.sharpe: published with N=3" in errors
    assert "performance.live.p_value" in errors


def test_sharpe_published_with_undeterminable_n_is_rejected():
    """S-04: the validator must not need to KNOW N to reject a ratio.

    ``btc_hodl_b1`` shipped Sharpe 0.793 and p=0.0242 with ``n_trades.value =
    null`` and ``validate_strategy_passport(...) == []``. Absence of the count is
    the common case (only the 3 smart_simple manifests publish `headline.trades`);
    a guard that only fires on a known small N is dead exactly where N is smallest.
    """
    p = _passport()
    env = _env(3)
    env["n_trades"] = unavailable("el manifiesto no publica el conteo de trades")
    p["performance"]["live"] = env
    errors = " ".join(validate_strategy_passport(p))
    assert "performance.live.sharpe" in errors
    assert "N no determinable" in errors
    assert "performance.live.p_value" in errors


def test_sharpe_published_with_null_n_is_rejected():
    """Same verdict when ``n_trades`` is published-but-null (the real artifact shape)."""
    p = _passport()
    env = _env(3)
    env["n_trades"] = sourced(None, "public/data/strategies/x/manifest.json")
    p["performance"]["live"] = env
    errors = " ".join(validate_strategy_passport(p))
    assert "performance.live.sharpe" in errors
    assert "N no determinable" in errors


@pytest.mark.parametrize("action", FORBIDDEN_PASSPORT_ACTIONS)
def test_passport_cannot_grow_an_action(action):
    errors = " ".join(validate_strategy_passport(_passport(**{action: True})))
    assert f"must not expose action '{action}'" in errors


def _tower(**overrides):
    payload = {
        "contract": PASSPORT_CONTRACT_ID,
        "contract_version": "1.0.0",
        "generated_at": "2026-07-28T00:00:00Z",
        "book": {"state_counts": {"CHAMPION": 1, "PAPER": 2, "CANARY": None,
                                  "REDUCED": None, "QUARANTINED": None}},
        "sleeves": [{"strategy_id": "smart_simple_v11", "retirement_signal": "unknown"}],
        "data": {"n_max_trials": sourced(N_MAX_TRIALS, "src/contracts/passport.py")},
        "paired_tests": [],
        "pending_interfaces": [],
    }
    payload.update(overrides)
    return payload


def test_valid_tower_passes():
    assert validate_control_tower(_tower()) == []


def test_n_max_is_pinned():
    t = _tower(data={"n_max_trials": sourced(500, "x")})
    assert any("n_max_trials must be 989" in e for e in validate_control_tower(t))


def test_unknown_book_state_rejected():
    t = _tower(book={"state_counts": {"CHAMPION": 1, "WINNER": 3}})
    assert any("unknown states ['WINNER']" in e for e in validate_control_tower(t))


def test_invented_retirement_signal_rejected():
    t = _tower(sleeves=[{"strategy_id": "x", "retirement_signal": "probably_fine"}])
    assert any("retirement_signal: bad value" in e for e in validate_control_tower(t))


def test_tower_sleeve_cannot_publish_sharpe_without_a_determinable_n():
    """S-04 (tower half): the Control Tower row is a decision surface too.

    The observed defect: ``REAL sleeve btc_hodl_b1: sharpe=0.793 n_trades=null
    insufficient=false``. A sleeve row must carry its N to carry a ratio.
    """
    t = _tower(sleeves=[{
        "strategy_id": "btc_hodl_b1",
        "retirement_signal": "unknown",
        "n_trades": unavailable("el manifiesto no publica el conteo de trades"),
        "sharpe": sourced(0.793, "public/data/registry.json"),
        "dsr_family": sourced(0.8357, "public/data/production/approval_state.json"),
    }])
    errors = " ".join(validate_control_tower(t))
    assert "sleeves[0].sharpe" in errors
    assert "N no determinable" in errors
    assert "sleeves[0].dsr_family" in errors


def test_tower_sleeve_with_small_n_cannot_publish_sharpe():
    t = _tower(sleeves=[{
        "strategy_id": "x", "retirement_signal": "unknown",
        "n_trades": sourced(1, "p"), "sharpe": sourced(0.793, "p"),
    }])
    assert any("published with N=1" in e for e in validate_control_tower(t))


def test_tower_sleeve_with_enough_trades_keeps_its_sharpe():
    t = _tower(sleeves=[{
        "strategy_id": "x", "retirement_signal": "unknown",
        "n_trades": sourced(34, "p"), "sharpe": sourced(3.35, "p"),
    }])
    assert validate_control_tower(t) == []


@pytest.mark.parametrize("action", FORBIDDEN_PASSPORT_ACTIONS)
def test_tower_cannot_grow_an_action(action):
    errors = " ".join(validate_control_tower(_tower(**{action: {}})))
    assert f"must not expose action '{action}'" in errors


# ------------------------------------- governance CONTENT (producer ↔ consumer, BL-32)

COMPOSER_TS = "usdcop-trading-dashboard/lib/passport/compose.ts"


def _fields_the_composer_reads() -> set[str]:
    """Per-asset governance fields the TS composer actually dereferences.

    DERIVED from the composer source (`gov?.x` / `gov.x`), never a hand-kept list: the
    whole failure mode being closed here is a list that stops matching the code.
    """
    root = Path(__file__).resolve().parents[2]
    text = (root / COMPOSER_TS).read_text(encoding="utf-8")
    return set(re.findall(r"\bgov\??\.([A-Za-z_][A-Za-z0-9_]*)", text))


def test_governance_projection_supplies_every_field_the_passport_reads():
    """El bloque `governance` del Passport se compone ENTERO desde esta proyección
    Python: si un activo llega como `{}`, el Passport publica cero trials, cero DSR y
    cero N — exactamente el vacío que la mutación TS `governance: {} as never` producía,
    pero desde el lado productor.

    Rojo con: en `scripts/pipeline/export_control_tower.py::read_assets`, sustituir el
    diccionario por `out[asset_id] = {}`.

    Nota de honestidad: NO se exige que los valores sean no-nulos — un activo sin trials
    todavía es un estado legítimo. Se exige que la CLAVE esté declarada (con `None`
    explícito si no hay dato), que es la diferencia entre "no hay dato" y "nadie preguntó".
    """
    from scripts.pipeline.export_control_tower import build

    needed = _fields_the_composer_reads()
    assert needed, f"no se pudo derivar ningún campo `gov.*` de {COMPOSER_TS}"

    payload = build()
    assets = payload.get("assets")
    assert isinstance(assets, dict) and assets, "la proyección no declara ningún activo"

    missing = {
        asset_id: sorted(needed - set(block or {}))
        for asset_id, block in assets.items()
        if not isinstance(block, dict) or (needed - set(block))
    }
    assert not missing, (
        "la proyección de gobernanza no declara campos que el composer del Passport lee "
        f"({COMPOSER_TS}): {missing}")


def test_governance_projection_never_invents_a_trial_count():
    """`n_trials_total` sale COPIADO del front-matter del HYPOTHESIS-REGISTRY del activo
    (la SSOT con la que se deflacta el DSR), jamás degradado a null en silencio.

    Rojo con: en `scripts/pipeline/export_control_tower.py`, `FRONTMATTER_RE = re.compile(
    r"^---\\r\\n(.*?)\\r\\n---\\r\\n", re.S)` — el front-matter deja de parsearse, `_frontmatter`
    devuelve `{}` sin quejarse y el N del DSR se publica como `None` en los cuatro activos.
    """
    import yaml

    from scripts.pipeline.export_control_tower import ASSET_DOCS, ROOT, build

    payload = build()
    for asset_id, block in payload["assets"].items():
        registry = ROOT / ASSET_DOCS[asset_id]["hypothesis"]
        declared = None
        if registry.exists():
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n",
                             registry.read_text(encoding="utf-8", errors="replace"), re.S)
            if match:
                declared = (yaml.safe_load(match.group(1)) or {}).get("n_trials_total")
        published = block.get("n_trials_total")
        assert published == declared, (
            f"{asset_id}: la proyección publica n_trials_total={published!r} "
            f"pero el HYPOTHESIS-REGISTRY declara {declared!r}")
