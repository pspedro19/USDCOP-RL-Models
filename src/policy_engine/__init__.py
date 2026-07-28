"""
Packaged policy evaluation library (BL-46 R4).

ONE library, imported by every runner — Airflow batch, backtest replay, paper
and the live loop (invariant 4 of `.claude/rules/strategy-engines.md`: two
implementations mean the backtest lies). Only the SOURCE of the feature
snapshot changes; the evaluation path is this module.

Public surface::

    build_policy(spec)                       -> Policy
    evaluate_policy(policy, snapshot, ctx)   -> StrategyDecision
    publish_signal(decision, ...)            -> StrategySignalRecord
    write_policy_version_index(records, ...) -> path

Spec: .claude/specs/planes/05-rule-based-strategies.md §4, §8
Contract: CTR-POLICY-BACKEND-001
"""

from src.policy_engine.runner import (  # noqa: F401
    FALLBACK_MODES,
    POLICY_ENGINE_VERSION,
    build_policy,
    evaluate_policy,
    publish_signal,
    write_policy_version_index,
)

__all__ = [
    "FALLBACK_MODES",
    "POLICY_ENGINE_VERSION",
    "build_policy",
    "evaluate_policy",
    "publish_signal",
    "write_policy_version_index",
]
