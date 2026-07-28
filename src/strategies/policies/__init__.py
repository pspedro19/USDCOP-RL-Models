"""Coded policies (spec §3.2 mode A) + spec loader — BL-47 R6-R8.

A coded policy is the escape hatch the DSL deliberately does NOT cover:
arithmetic sizing (vol targeting), state, sequential rules. It is still a
``Policy`` (``src/contracts/policy.py``): same ``evaluate(snapshot, context)``
in backtest, paper and live, same ``StrategyDecision``, same ``rule_trace``.

Migration rule (BL-47, 0 trials): a coded policy NEVER re-derives the frozen
strategy. It reproduces, per bar, the exact arithmetic of the frozen module
declared in ``config/strategy_manifests/<asset>.yaml``. Any divergence is
declared in the spec (``divergence:``) and escalated — never silently fixed.
"""

from src.strategies.policies.loader import (  # noqa: F401
    ALLOWED_MODULE_PREFIX,
    PolicySpecError,
    build_policy,
    load_policy_spec,
    load_all_policy_specs,
    policy_specs_dir,
)

__all__ = [
    "ALLOWED_MODULE_PREFIX",
    "PolicySpecError",
    "build_policy",
    "load_policy_spec",
    "load_all_policy_specs",
    "policy_specs_dir",
]
