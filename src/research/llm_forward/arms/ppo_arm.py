"""First-bar-held PPO adapter for the research PAPER ledger.

An M5 bar stamped 08:00 COT closes at 08:05, not 08:00. The record contains one
inference made after that close and its constant 59-weight path. This is not an
executable-fill certificate: using historical C0 after inference still assumes a
price proxy. Native 59-decision policies must use the per-bar runner, never a
complete-session replay in this writer. Existing frozen identity/parity failures
remain blocking; this adapter does not refit or rebind those artifacts.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.research.dataset import SEED_M5  # noqa: E402
from src.research.live_spec import build_live_spec_partial  # noqa: E402
from src.research.llm_forward.decide import arm_spec, load_preregistration  # noqa: E402
from src.research.llm_forward.ledger import Ledger, LedgerError  # noqa: E402
from src.research.llm_forward.paths import DECISIONS_PATH, PREREG_PATH  # noqa: E402
from src.research.llm_forward.settlement_accounting import (  # noqa: E402
    candidate_from_sources,
    check_chain,
    check_decision_schedules,
    read_ledger,
    session_open,
)
from src.research.session_env import EXPOSURE_LEVELS, OPERABLE_RETURNS  # noqa: E402
from src.research.session_gym import SessionTradingEnv, position_state  # noqa: E402

MODELS_DIR = REPO / "data" / "thesis" / "ppo"


def _direction(w: float) -> str:
    return "flat" if w == 0.0 else ("long" if w > 0 else "short")


def decide_weights(spec, model_path: Path, hold_all_session: bool) -> np.ndarray:
    """Senda de exposición de la política congelada para esta sesión.

    Con `hold_all_session` se toma **solo la primera decisión** y se sostiene hasta el cierre
    — el equivalente de `k=59` en la curva de frecuencia del hold-out. Sin él, el agente
    decide en las 59 barras, que es su política nativa.
    """
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    obs, _ = env.reset()

    if hold_all_session:
        action, _ = model.predict(obs, deterministic=True)
        return np.full(OPERABLE_RETURNS, float(EXPOSURE_LEVELS[int(action)]))

    weights, done = [], False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(int(action))
        weights.append(float(EXPOSURE_LEVELS[int(action)]))
    return np.asarray(weights, dtype=float)


def decide_first_weight(partial, model_path: Path) -> float:
    """Primera decisión usando exclusivamente la barra 0 ya cerrada."""
    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device="cpu")
    obs = np.concatenate([partial.market[0], position_state(0.0, 0, 0.0, 0.0, 0),
                          partial.context]).astype(np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return float(EXPOSURE_LEVELS[int(action)])


def observation_for_closed_bar(
    partial,
    *,
    previous_weight: float,
    bars_in_position: int,
    unrealized: float,
    drawdown: float,
    n_changes: int,
) -> np.ndarray:
    """Build the PPO observation for the last bar in a causal prefix.

    ``partial`` must come from :func:`build_live_spec_partial`; only its last
    closed market row is used. Position features are supplied by the persisted
    stream state, never reconstructed from future bars.
    """
    if partial.bars_received < 1 or len(partial.market) != partial.bars_received:
        raise ValueError("partial live spec has no consistent closed-bar prefix")
    if not -1.0 <= previous_weight <= 1.0:
        raise ValueError("previous_weight outside the action space")
    obs = np.concatenate([
        partial.market[-1],
        position_state(previous_weight, bars_in_position, unrealized, drawdown, n_changes),
        partial.context,
    ]).astype(np.float32)
    return np.clip(obs, -5.0, 5.0)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def decide_first_bar_hold(model, *, session_date, arm_id, model_id,
                          preregistration_sha256, partial, bar_received_at_utc):
    """One validated inference; no whole-session builder or outcome is needed."""
    from .ppo_stream import PPOStreamingRunner

    if (getattr(partial, "date", None) != session_date
            or type(getattr(partial, "bars_received", None)) is not int
            or partial.bars_received != 1):
        raise ValueError("first_bar_hold requires the requested session and exactly one bar")
    close = session_open(session_date) + timedelta(minutes=5)
    record = PPOStreamingRunner(
        model, arm_id=arm_id, model_id=model_id,
        preregistration_sha256=preregistration_sha256,
    ).decide(
        session_date=session_date, bar_index=0, partial=partial,
        previous_weight=0.0, bars_in_position=0, unrealized=0.0, drawdown=0.0,
        n_changes=0, bar_close_utc=close, bar_received_at_utc=bar_received_at_utc,
    )
    return replace(
        record, decision_id=f"{session_date}::{arm_id}",
        decision_path=[record.decision.score] * OPERABLE_RETURNS,
        decision_schedule="first_bar_hold",
        information_edge=(
            "One decision after first close (08:05 COT), held for 59 paper returns. "
            "Emission before close(1) is NOT proof of execution at C0: historical-close "
            "price proxy, no executable fill or durable-storage acknowledgement."
        ),
    )


def run(session_date: str, arm_id: str = "ppo_regime_fwd_k59",
        now_override: datetime | None = None, dry_run: bool = False) -> int:
    """Write only a first-bar-held decision; current clock cannot be overridden."""
    if now_override is not None:
        raise ValueError("now_override cannot backdate a forward decision; mock clocks in tests")
    from src.research.llm_forward.stream_runner import _writer_lock

    ledger = Ledger(DECISIONS_PATH, key_field="decision_id")
    with _writer_lock(ledger.path):
        return _run_held_locked(session_date, arm_id, dry_run, ledger)


def _run_held_locked(session_date, arm_id, dry_run, ledger):
    spec_yaml, prereg_hash = load_preregistration(PREREG_PATH)
    arm = arm_spec(spec_yaml, arm_id)
    if arm["kind"] != "rl_frozen":
        raise ValueError(f"{arm_id} es de tipo {arm['kind']!r}; este job sella brazos RL.")

    count = arm.get("decisions_per_session")
    if type(count) is not int or count != 1:
        raise ValueError("59 decisions require the per-bar runner; batch replay cannot seal forward")
    model_path = REPO / arm["model"]
    rows = read_ledger(ledger.path)
    check_chain(rows)
    decision_id = f"{session_date}::{arm_id}"
    check_decision_schedules([*rows, {
        "session_date": session_date, "decision_id": decision_id,
        "decision_schedule": "first_bar_hold", "bar_index": 0,
    }])
    existing = next((row for row in rows if row["decision_id"] == decision_id), None)
    if existing is not None:
        candidate_from_sources([existing])
        if (existing.get("preregistration_sha256") != prereg_hash
                or existing.get("model") != model_path.name):
            raise ValueError("existing held decision has a different frozen identity")
        print(f"  ya registrada {decision_id}; no new inference or ledger write")
        return 0
    opening = session_open(session_date)
    close = opening + timedelta(minutes=5)
    if not close <= _utc_now() < close + timedelta(minutes=5):
        raise ValueError("first_bar_hold must start after close(0) and before close(1)")

    import pandas as pd
    from stable_baselines3 import PPO

    if not model_path.is_file():
        raise FileNotFoundError(f"falta el modelo congelado {model_path}")
    m5 = pd.read_parquet(SEED_M5)
    received = _utc_now()  # local snapshot read, not an exchange receipt or fill
    times = pd.to_datetime(m5["time"])
    if times.dt.tz is None:
        raise ValueError("source bar timestamps must have an explicit timezone")
    first = m5[times == opening]
    if "symbol" in first.columns:
        symbols = first["symbol"].astype(str).str.upper().str.replace("/", "", regex=False)
        first = first[symbols == "USDCOP"]
    if len(first) != 1:
        raise ValueError("exactly one USDCOP first bar required; missing/duplicate source")
    partial = build_live_spec_partial(session_date, first)
    model = PPO.load(str(model_path), device="cpu")
    record = decide_first_bar_hold(
        model, session_date=session_date, arm_id=arm_id, model_id=model_path.name,
        preregistration_sha256=prereg_hash, partial=partial, bar_received_at_utc=received,
    )
    if dry_run:
        print(f"{arm_id} {session_date}: w0={record.decision.score:+.2f} "
              f"on_time={record.sealed_before_next_bar} (dry-run, no ledger write)")
        return 0

    try:
        written = ledger.append(record)
    except LedgerError as exc:
        print(f"  [rechazado] {exc}", file=sys.stderr)
        return 1

    print(f"  sellado seq={written['seq']} {arm_id} w0={record.decision.score:+.2f} "
          f"hash={written['record_hash'][:12]}...")
    return 0


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Sella la decision del brazo RL congelado.")
    ap.add_argument("--session-date", required=True)
    ap.add_argument("--arm", default="ppo_regime_fwd_k59")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return run(args.session_date, arm_id=args.arm, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
