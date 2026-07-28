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

def _passport(**overrides):
    perf = {env: {**_env(25), "env": env} for env in PASSPORT_ENVS}
    payload = {
        "contract": PASSPORT_CONTRACT_ID,
        "contract_version": "1.0.0",
        "strategy_id": "smart_simple_v11",
        "generated_at": "2026-07-28T00:00:00Z",
        "identity": {}, "governance": {}, "lineage": {},
        "performance": perf, "live": {}, "risk": {},
    }
    payload.update(overrides)
    return payload


def test_valid_passport_passes():
    assert validate_strategy_passport(_passport()) == []


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
