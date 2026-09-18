"""Record schemas for the forward ledger.

Two record types, deliberately kept in separate files and written by separate
jobs:

* ``DecisionRecord`` — captures inputs/output under its declared schedule:
  pre-open legacy, per-bar stream or explicit first-bar hold. Never modified afterwards.
* ``SettlementRecord`` — written *after* the session closes. References a decision
  by id and adds the realized outcome.

Separate records make outcome backfills easier to detect. This is not access
control, an external durable timestamp or a guarantee of executable prices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal

Direction = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class CorpusDoc:
    """One document the model was allowed to read, recorded by reference.

    The full text lives in a content-addressed store (``data/corpus/<sha>.txt``),
    not in the ledger. The ledger stays small and diffable; the hash still proves
    exactly which bytes were shown.
    """

    doc_id: str
    url: str
    published_at_utc: str
    title: str
    text_sha256: str
    char_count: int


@dataclass(frozen=True)
class LlmUsage:
    """Cost and latency of one model call.

    Alpha Arena's most transferable lesson was that the fee line, not the
    prediction, decided the ranking. The analogue here is inference cost per
    decision, so it is a first-class recorded field rather than an afterthought.
    """

    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    cost_usd: float
    system_fingerprint: str | None = None


@dataclass(frozen=True)
class Decision:
    """The model's output, after schema validation."""

    score: float          # s_d in [-1, 1]; this is what feeds the observation vector
    direction: Direction
    confidence: float     # [0, 1]
    rationale: str


@dataclass
class DecisionRecord:
    """A decision captured under an explicitly validated observation schedule."""

    seq: int
    decision_id: str
    session_date: str            # YYYY-MM-DD, the session being predicted
    emitted_at_utc: str          # when this process ran
    cutoff_utc: str              # no document published at/after this was shown
    session_open_utc: str        # first bar of the session
    sealed_before_open: bool     # False requires a valid explicit intraday schedule
    preregistration_sha256: str
    prompt_sha256: str
    model: str
    provider: str
    temperature: float
    seed: int | None
    corpus: list[CorpusDoc]
    decision: Decision | None    # None when the corpus was empty (abstention)
    abstained: bool
    abstain_reason: str | None
    usage: LlmUsage | None

    # --- anadidos al vendorizar (CTR-RESEARCH-FORWARD-001) --------------------
    # `decision` lleva UN score, que basta para un brazo que decide una vez. El brazo RL
    # nativo decide 59 veces por sesion y ahi la decision ES la senda: sellar solo `w_0`
    # dejaria 58 decisiones fuera del ledger, que es justo el agujero que el ledger existe
    # para cerrar.
    decision_path: list[float] | None = None

    # El spread esperado del dia sale del posterior de regimen del cierre de d-1, asi que se
    # conoce ANTES de la apertura y se sella con la decision. Recalcularlo al liquidar
    # permitiria que el contrato de costos cambiara entre el sellado y el resultado — que es
    # exactamente la puerta que este diseno cierra.
    spread_pips: float | None = None

    # Declarada, no escondida: el brazo RL ve la barra 0 para construir su observacion y el
    # LLM no. Los dos sellan antes de que exista r_1, pero no con la misma informacion.
    information_edge: str | None = None

    # Campos aditivos para brazos que sellan por barra. Los registros históricos no los
    # llevan; lectores deben hacer fallback a ``sealed_before_open``.
    sealed_before_next_bar: bool | None = None
    bar_index: int | None = None
    bar_received_at_utc: str | None = None

    prev_hash: str = ""
    record_hash: str = ""

    # C049: exactly one post-close(0) decision held for all 59 paper returns.
    # None/absent preserves legacy; never infer this mode or migrate old rows.
    decision_schedule: str | None = None

    def hashable_payload(self) -> dict[str, Any]:
        """The record minus its own hash fields.

        A record cannot contain its own hash, so the two chain fields are
        stripped before hashing and re-attached afterwards.
        """
        payload = asdict(self)
        payload.pop("prev_hash")
        payload.pop("record_hash")
        return payload


@dataclass
class SettlementRecord:
    """The realized outcome for one decision, written after the close."""

    seq: int
    decision_id: str
    session_date: str
    settled_at_utc: str
    open_price: float
    close_price: float
    realized_return: float       # (close - open) / open, sign convention documented
    signed_return: float         # gross strategy return, NEVER net of costs
    bars_observed: int
    prev_hash: str = ""
    record_hash: str = ""

    # C048 additive: old signed_return remains GROSS, rounded to eight decimals.
    # Missing/None means net accounting UNKNOWN, never zero. Original rows are immutable.
    accounting: dict[str, Any] | None = None

    def hashable_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("prev_hash")
        payload.pop("record_hash")
        return payload


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with explicit offset.

    Naive datetimes are banned throughout this package. A timestamp without a
    zone is exactly the ambiguity the whole design exists to eliminate.
    """

    return datetime.now(UTC).isoformat(timespec="seconds")
