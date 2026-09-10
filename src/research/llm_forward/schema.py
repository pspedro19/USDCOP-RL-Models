"""Record schemas for the forward ledger.

Two record types, deliberately kept in separate files and written by separate
jobs:

* ``DecisionRecord`` — sealed *before* the session opens. Contains what the model
  saw and what it emitted. Never modified afterwards.
* ``SettlementRecord`` — written *after* the session closes. References a decision
  by id and adds the realized outcome.

Keeping them apart is not tidiness. If one process could write both, a bug (or a
tired operator) could backfill an outcome into a decision row, and the whole
forward guarantee would evaporate with no trace. Separation makes that class of
mistake structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
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
    """A sealed pre-session decision."""

    seq: int
    decision_id: str
    session_date: str            # YYYY-MM-DD, the session being predicted
    emitted_at_utc: str          # when this process ran
    cutoff_utc: str              # no document published at/after this was shown
    session_open_utc: str        # first bar of the session
    sealed_before_open: bool     # False => excluded from analysis, not deleted
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

    prev_hash: str = ""
    record_hash: str = ""

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
    signed_return: float         # realized_return * position sign, gross of costs
    bars_observed: int
    prev_hash: str = ""
    record_hash: str = ""

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
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
