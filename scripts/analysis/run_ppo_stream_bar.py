#!/usr/bin/env python
"""Seal exactly one causal PPO bar decision.

The caller must provide a prefix file containing only bars received through the requested
bar.  The command refuses a longer/shorter prefix instead of inferring or filling data.
"""
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.live_spec import build_live_spec_partial  # noqa: E402
from src.research.llm_forward.decide import arm_spec, load_preregistration  # noqa: E402
from src.research.llm_forward.paths import PREREG_PATH  # noqa: E402
from src.research.llm_forward.stream_runner import LiveSessionRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-date", required=True)
    parser.add_argument("--bar-index", required=True, type=int)
    parser.add_argument("--bars-path", required=True, type=Path,
                        help="parquet/CSV prefix through the requested closed bar")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--prereg", type=Path, default=PREREG_PATH)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--bar-received-at", type=str)
    args = parser.parse_args()
    if args.bar_index < 0 or args.bar_index >= 59:
        parser.error("--bar-index must be in [0, 58]")
    if not args.bars_path.is_file() or not args.model.is_file():
        parser.error("bars-path and model must exist")
    try:
        prereg, prereg_hash = load_preregistration(args.prereg)
        arm = arm_spec(prereg, args.arm_id)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"invalid forward preregistration: {exc}")
    if arm.get("kind") != "rl_frozen":
        parser.error(f"arm {args.arm_id!r} is not an rl_frozen arm")
    if int(arm.get("decisions_per_session", 0)) != 59:
        parser.error("stream runner requires a 59-decision RL arm")
    bars = (pd.read_parquet(args.bars_path) if args.bars_path.suffix.lower() == ".parquet"
            else pd.read_csv(args.bars_path))
    # Receipt is captured before feature construction/model load, never backdated.
    received = args.bar_received_at or datetime.now(UTC).isoformat()
    if len(bars) != args.bar_index + 1:
        parser.error("prefix length must equal bar-index + 1; future/missing bars are rejected")
    bars["time"] = pd.to_datetime(bars["time"])
    partial = build_live_spec_partial(args.session_date, bars)
    from stable_baselines3 import PPO
    model = PPO.load(str(args.model), device="cpu")
    runner = LiveSessionRunner(
        model=model, arm_id=args.arm_id, model_id=args.model_id,
        preregistration_sha256=prereg_hash, ledger_path=args.ledger,
        state_path=args.state,
    )
    record = runner.step(
        session_date=args.session_date, partial=partial,
        # L0 labels M5 bars by OPEN (08:00 COT); its close is observable at 08:05.
        bar_close_utc=bars.iloc[-1]["time"].to_pydatetime() + timedelta(minutes=5),
        bar_received_at_utc=received,
    )
    print({"written": record is not None, "bar_index": args.bar_index,
           "ledger": str(args.ledger), "state": str(args.state)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
