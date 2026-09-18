"""Strict admission and replayable PAPER accounting; never evidence of fills."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from numbers import Real
from pathlib import Path

import numpy as np

from src.research.cost_contract import CONTRACT_PATH, COST_CONTRACT, load_cost_contract
from src.research.cost_model import CostParameters
from src.research.session_env import EXPOSURE_LEVELS, run_session

from .canonical import GENESIS_HASH, chain_hash, hash_payload

ROOT = Path(__file__).resolve().parents[3]
ENGINE_FILES = ("src/research/session_env.py", "src/research/cost_model.py",
                "src/research/cost_contract.py", "src/research/llm_forward/settlement_accounting.py")
ENGINE_HASHES = {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in ENGINE_FILES}
CONTRACT_HASH = sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def number(value, name: str, *, minimum=None, maximum=None) -> float:
    """Reject coercions before numpy can turn malformed inputs into a policy."""
    if isinstance(value, bool | np.bool_) or not isinstance(value, Real):
        raise ValueError(f"{name}: expected a real number, not a coercible value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}: nonfinite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name}: below minimum")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name}: above maximum")
    return result


def vector(value, length: int, name: str, *, positive=False, weights=False) -> np.ndarray:
    if not isinstance(value, list | tuple | np.ndarray) or len(value) != length:
        raise ValueError(f"{name}: expected {length} entries")
    values = [number(v, name) for v in value]
    if positive and any(v <= 0 for v in values):
        raise ValueError(f"{name}: prices must be positive")
    if weights and any(v not in EXPOSURE_LEVELS for v in values):
        raise ValueError(f"{name}: not in frozen action space")
    return np.asarray(values, dtype=float)


def timestamp(value, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name}: timestamp must be a string")
    try:
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name}: invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name}: timezone required")
    return result.astimezone(UTC)


def session_open(day) -> datetime:
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("session_date: ISO date required")
    parsed = date.fromisoformat(day)
    return datetime(parsed.year, parsed.month, parsed.day, 13, tzinfo=UTC)


def digest(value, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name}: lowercase sha256 required")
    return value


def identity(record: dict) -> tuple[str, str, str, str, str]:
    day = record["session_date"]
    session_open(day)
    parts = record["decision_id"].split("::")
    if len(parts) not in (2, 3) or parts[0] != day or not parts[1]:
        raise ValueError("decision_id: inconsistent session/arm")
    for key in ("provider", "model"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key}: identity required")
    if record["provider"].strip().lower() == "stub":
        raise ValueError("stub provider is not experimental evidence")
    prereg = digest(record.get("preregistration_sha256"), "preregistration_sha256")
    digest(record.get("record_hash"), "record_hash")
    digest(record.get("prompt_sha256"), "prompt_sha256")
    return day, parts[1], record["provider"], record["model"], prereg


def sealed(record: dict) -> bool:
    """Daily optional-None fallback; stream flags/timestamps remain mandatory."""
    try:
        schedule = record.get("decision_schedule")
        if schedule is not None and schedule != "first_bar_hold":
            return False
        if schedule == "first_bar_hold" and (
            type(record.get("bar_index")) is not int or record["bar_index"] != 0
            or record.get("sealed_before_open") is not False
        ):
            return False
        opening = session_open(record["session_date"])
        if timestamp(record["session_open_utc"], "session_open_utc") != opening:
            return False
        emitted = timestamp(record["emitted_at_utc"], "emitted_at_utc")
        cutoff = timestamp(record["cutoff_utc"], "cutoff_utc")
        corpus = record.get("corpus")
        if not isinstance(corpus, list):
            return False
        for doc in corpus:
            if not isinstance(doc, dict):
                return False
            published = timestamp(doc["published_at_utc"], "published_at_utc")
            if not (published < cutoff and published <= emitted):
                return False
        bar = record.get("bar_index")
        if bar is None:
            # A pre-session path can only be known if sealed before the open.
            # A complete stream is validated row-by-row, never by this branch.
            flag = record.get("sealed_before_next_bar")
            if flag is None:
                flag = record.get("sealed_before_open")
            # cutoff is a planned upper bound, often exactly the opening. It may
            # postdate emission; actual documents above must already be published.
            return (flag is True and record.get("sealed_before_open") is True
                    and cutoff <= opening and emitted < opening)
        if type(bar) is not int or not 0 <= bar < 59:
            return False
        close = opening + timedelta(minutes=5 * (bar + 1))
        received = timestamp(record["bar_received_at_utc"], "bar_received_at_utc")
        return (
            record.get("sealed_before_next_bar") is True
            and cutoff == close and close <= received <= emitted < close + timedelta(minutes=5)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def aggregate(records: list[dict]) -> dict | None:
    """No implicit deletion/coercion, no incomplete sessions represented as zero."""
    if len(records) != 59:
        return None
    try:
        if any(type(row.get("bar_index")) is not int for row in records):
            return None
        rows = sorted(records, key=lambda row: row["bar_index"])
        if [row["bar_index"] for row in rows] != list(range(59)):
            return None
        identities = {identity(row) for row in rows}
        if len(identities) != 1:
            return None
        day, arm, _, _, _ = identities.pop()
        spreads = set()
        weights = []
        for bar, row in enumerate(rows):
            if row.get("decision_schedule") is not None:
                return None
            if row["decision_id"] != f"{day}::{arm}::b{bar:02d}":
                return None
            if not sealed(row) or row.get("abstained") is not False:
                return None
            if row.get("decision_path") is not None:
                return None
            spreads.add(number(row["spread_pips"], "spread_pips", minimum=0))
            weight = decision_score(row)
            if weight not in EXPOSURE_LEVELS:
                return None
            weights.append(weight)
        if len(spreads) != 1:
            return None
        candidate = dict(rows[0])
        candidate.update(
            decision_id=f"{day}::{arm}::stream", decision_path=weights,
            spread_pips=spreads.pop(), bar_index=None, bar_count=59,
            source_decisions=[{"decision_id": row["decision_id"],
                               "record_hash": row["record_hash"]} for row in rows],
        )
        return candidate
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return None


def check_record_hash(row: dict) -> None:
    stored = digest(row.get("record_hash"), "record_hash")
    previous = digest(row.get("prev_hash"), "prev_hash")
    payload = {key: value for key, value in row.items() if key not in {"prev_hash", "record_hash"}}
    if chain_hash(previous, payload) != stored:
        raise ValueError("record content hash mismatch")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def read_ledger(path: Path) -> list[dict]:
    """No last-key-wins parsing; reading must not create files or directories."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line, object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("ledger rows must be objects")
    return rows


def decision_score(record: dict) -> float:
    decision = record.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("decision object required")
    score = number(decision.get("score"), "score", minimum=-1, maximum=1)
    direction = "flat" if score == 0 else ("long" if score > 0 else "short")
    if decision.get("direction") != direction:
        raise ValueError("decision direction contradicts score")
    number(decision.get("confidence"), "confidence", minimum=0, maximum=1)
    if not isinstance(decision.get("rationale"), str):
        raise ValueError("decision rationale must be a string")
    return score


def check_chain(rows: list[dict]) -> None:
    """Validate the supplied snapshot, not a different second read of the ledger."""
    previous = GENESIS_HASH
    identifiers = set()
    for seq, row in enumerate(rows):
        if type(row.get("seq")) is not int or row["seq"] != seq:
            raise ValueError("ledger sequence mismatch")
        key = row.get("decision_id")
        if not isinstance(key, str) or not key or key in identifiers:
            raise ValueError("ledger duplicate/invalid decision_id")
        if row.get("prev_hash") != previous:
            raise ValueError("ledger chain mismatch")
        check_record_hash(row)
        previous = row["record_hash"]
        identifiers.add(key)


def check_decision_schedules(rows: list[dict]) -> None:
    """One modality per arm, not two returns for one arm/day under distinct IDs.

    Hash validation is separate so a producer can preflight an unsigned candidate
    while holding the shared writer lock. Unknown modes never fall back to legacy.
    """
    modes = {}
    for row in rows:
        identifier = row.get("decision_id")
        if not isinstance(identifier, str):
            continue  # identity/admission reports malformed historical rows
        parts = identifier.split("::")
        if len(parts) not in (2, 3) or not parts[1] or parts[0] != row.get("session_date"):
            continue
        schedule = row.get("decision_schedule")
        if schedule is None:
            schedule = "stream" if row.get("bar_index") is not None else "preopen_daily"
        elif not isinstance(schedule, str) or schedule != "first_bar_hold":
            raise ValueError("unknown decision schedule; no legacy fallback")
        previous = modes.setdefault(parts[1], schedule)
        if previous != schedule:
            raise ValueError("decision schedule changed within an arm; separate treatment required")


def candidate_from_sources(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("source decisions required")
    for row in rows:
        if "source_decisions" in row:
            raise ValueError("source row impersonates an internal aggregate")
        check_record_hash(row)
    if len(rows) == 59:
        candidate = aggregate(rows)
        if candidate is None:
            raise ValueError("invalid complete stream")
        return candidate
    if len(rows) != 1:
        raise ValueError("expected one daily decision or 59 stream decisions")
    row = rows[0]
    identity(row)
    number(row.get("spread_pips"), "spread_pips", minimum=0)
    held = row.get("decision_schedule") == "first_bar_hold"
    if held:
        if (type(row.get("bar_index")) is not int or row["bar_index"] != 0
                or row.get("provider") != "rl_frozen_stream"):
            raise ValueError("first_bar_hold requires RL first-bar identity")
    elif row.get("bar_index") is not None:
        raise ValueError("one stream bar is not a complete session")
    if not sealed(row) or row.get("abstained") is not False:
        raise ValueError("daily decision is not admissible")
    if len(row["decision_id"].split("::")) != 2:
        raise ValueError("daily decision id must not masquerade as a stream")
    score = decision_score(row)
    if held and row.get("decision_path") is None:
        raise ValueError("first_bar_hold requires its constant 59-weight path")
    if row.get("decision_path") is not None:
        weights = vector(row["decision_path"], 59, "weights", weights=True)
        if weights[0] != score:
            raise ValueError("path summary differs from first position")
        if held and not np.all(weights == score):
            raise ValueError("first_bar_hold path must be constant")
    return row


def candidate_weights(record: dict) -> np.ndarray:
    path = record.get("decision_path")
    if path is not None:
        return vector(path, 59, "weights", weights=True)
    score = number(record["decision"]["score"], "score", minimum=-1, maximum=1)
    levels = np.asarray(EXPOSURE_LEVELS)
    return np.full(59, levels[int(np.argmin(np.abs(levels - score)))])


def api_cost_summary(rows: list[dict]) -> dict:
    costs = []
    for row in rows:
        usage = row.get("usage")
        if usage is not None and not isinstance(usage, dict):
            raise ValueError("usage must be an object or unknown")
        if usage is not None and usage.get("cost_usd") is not None:
            costs.append(number(usage["cost_usd"], "usage.cost_usd", minimum=0))
    total = number(math.fsum(costs), "api_cost total", minimum=0)
    return {"unit": "usd", "basis": "provider_usage_report_not_invoice",
            "known_decisions": len(costs), "unknown_decisions": len(rows) - len(costs),
            "known_subtotal_usd": total,
            "total_usd": total if len(costs) == len(rows) else None,
            "included_in_daily_return": False}


def _snapshot() -> dict:
    current = {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in ENGINE_FILES}
    contract_hash = sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    if current != ENGINE_HASHES or contract_hash != CONTRACT_HASH:
        raise ValueError("engine/contract changed since import; restart and review")
    if load_cost_contract() != COST_CONTRACT:
        raise ValueError("cost contract differs from imported defaults")
    params = CostParameters()
    if (params.commission_per_side != COST_CONTRACT.commission_per_side
            or params.slippage_coefficient != COST_CONTRACT.slippage_coef):
        raise ValueError("cost defaults differ from contract")
    return {"engine_sha256": current, "cost_contract_sha256": contract_hash}


def _computed(closes, weights, spread, parameters, day) -> dict:
    result = run_session(closes, weights, spread, date=day, cost_parameters=parameters)
    gross_bars = weights * result.bar_returns
    turnover = np.abs(np.diff(np.concatenate([[0.0], weights, [0.0]])))
    fields = {
        "gross_return": result.gross_return, "total_cost": result.total_cost,
        "daily_return": result.daily_return, "terminal_cost": result.terminal_cost,
        "n_changes": result.n_changes, "mean_abs_exposure": result.mean_abs_exposure,
        "sum_abs_dw": float(turnover.sum()), "gross_bars": gross_bars.tolist(),
        "costs": result.costs.tolist(),
    }
    for key in ("gross_return", "total_cost", "daily_return", "terminal_cost",
                "mean_abs_exposure", "sum_abs_dw"):
        number(fields[key], key)
    vector(fields["gross_bars"], 59, "gross_bars")
    costs = vector(fields["costs"], 60, "costs")
    if (costs < 0).any():
        raise ValueError("negative trading cost")
    return fields


def build_accounting(record: dict, closes, source_rows: list[dict]) -> dict:
    """Seal exactly what was computed, with explicit provenance limitations."""
    source = candidate_from_sources(source_rows)
    weights = candidate_weights(source)
    prices = vector(closes, 60, "closes", positive=True)
    spread = number(source.get("spread_pips"), "spread_pips", minimum=0)
    if record["decision_id"] != source["decision_id"] or record["session_date"] != source["session_date"]:
        raise ValueError("candidate/source identity mismatch")
    if not np.array_equal(candidate_weights(record), weights):
        raise ValueError("candidate/source weights mismatch")
    if number(record.get("spread_pips"), "spread_pips", minimum=0) != spread:
        raise ValueError("candidate/source spread mismatch")
    provenance = _snapshot()
    parameters = CostParameters()
    ordered = sorted(source_rows, key=lambda row: row.get("bar_index") or 0)
    refs = [{"decision_id": row["decision_id"], "record_hash": row["record_hash"]} for row in ordered]
    inputs = {"closes": prices.tolist(), "weights": weights.tolist()}
    return {
        "schema": "thesis-session-accounting-v1", "session_date": source["session_date"],
        "return_unit": "decimal", "price_unit": "cop_per_usd", "evidence": "paper_assumed_costs",
        "scope": {"close_timestamps_verified": False, "executable_quotes_verified": False,
                  "fee_parameters_sealed_at_decision": False, "api_cost_invoice_verified": False},
        "spread_cop_per_usd": spread,
        "commission_per_side": number(parameters.commission_per_side, "commission", minimum=0),
        "slippage_coefficient": number(parameters.slippage_coefficient, "slippage", minimum=0),
        **inputs, "inputs_sha256": hash_payload(inputs), "source_decisions": refs,
        **provenance, "api_cost": api_cost_summary(ordered),
        **_computed(prices, weights, spread, parameters, source["session_date"]),
    }


def validate_accounting(settlement: dict, decision_rows: list[dict]) -> float | None:
    """Return verified net or None for legacy unknown. Caller checks both chains."""
    payload = settlement.get("accounting")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("accounting must be an object")
    try:
        by_id = {row["decision_id"]: row for row in decision_rows}
        if len(by_id) != len(decision_rows):
            raise ValueError("duplicate source decision")
        refs = payload["source_decisions"]
        if not isinstance(refs, list) or len(refs) not in (1, 59):
            raise ValueError("source reference cardinality")
        source_rows = [by_id[ref["decision_id"]] for ref in refs]
        candidate = candidate_from_sources(source_rows)
        prices = vector(payload["closes"], 60, "closes", positive=True)
        expected = build_accounting(candidate, prices, source_rows)
        # Canonical equality enforces types as well as values and includes every vector,
        # provenance field and reference. A different engine needs its archived reader.
        if hash_payload(payload) != hash_payload(expected):
            raise ValueError("accounting replay mismatch (values, types, references or engine)")
        if type(settlement["bars_observed"]) is not int or settlement["bars_observed"] != 60:
            raise ValueError("bars_observed must be integer 60")
        if (settlement["session_date"] != candidate["session_date"]
                or settlement["decision_id"] != candidate["decision_id"]):
            raise ValueError("outer/inner identity mismatch")
        if timestamp(settlement["settled_at_utc"], "settled_at_utc") < session_open(candidate["session_date"]) + timedelta(hours=5):
            raise ValueError("settlement predates full session close")
        expected_outer = {
            "open_price": prices[0], "close_price": prices[-1],
            "realized_return": round(float(prices[-1] / prices[0] - 1.0), 8),
            "signed_return": round(expected["gross_return"], 8),
        }
        for key, value in expected_outer.items():
            if number(settlement[key], key) != value:
                raise ValueError(f"outer/inner {key} mismatch")
        return expected["daily_return"]
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("malformed accounting or missing source decision") from exc
