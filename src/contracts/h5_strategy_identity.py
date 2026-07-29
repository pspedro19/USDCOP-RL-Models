"""Canonical strategy identity for the H5 weekly tables.

Migration 064 (``database/migrations/064_h5_strategy_id.sql``) replaced the
uniqueness on ``signal_date`` with ``(signal_date, strategy_id)`` on the three
strategy-scoped H5 tables so that v11 and a challenger (v12) can coexist on the
same Monday.  That has two consequences which this module exists to serve:

1. **Writers** must name ``strategy_id`` explicitly and use
   ``ON CONFLICT (signal_date, strategy_id)``.  PostgreSQL cannot infer the
   composite constraint from ``ON CONFLICT (signal_date)`` and raises 42P10
   *before writing any row* — which is why ``forecast_h5_*`` silently froze.
2. **Readers** must filter by ``strategy_id``.  A reader that does not filter
   will start mixing strategies the moment a second one writes: v11 sizing fed
   by v12 executions, v11 live joined against v12 paper.  That is data
   corruption, not a cosmetic issue.

The value below is the identity of the **production** COP strategy.  It is not
a free-floating literal: ``tests/regression/test_h5_strategy_identity_reads.py``
asserts it stays equal to migration 064's column DEFAULT and to the production
entry in ``config/strategy_registry.yaml``, and that every other copy of the
literal in the codebase (services and dashboard, which cannot import this
module) agrees with it.

Contract: CTR-STRAT-REGISTRY-001 (extends) · see also ``.claude/rules/strategy-contract.md``
"""

from __future__ import annotations

# The COP production strategy. Mirrors migration 064's
#   ADD COLUMN strategy_id TEXT NOT NULL DEFAULT 'smart_simple_v11'
# so the historical backfill (which IS v11) and new writes agree.
H5_PRODUCTION_STRATEGY_ID = "smart_simple_v11"

# The three H5 tables that migration 064 made strategy-scoped.
# forecast_h5_predictions is deliberately NOT here (predictions are MODEL output,
# shared across strategies) and neither is forecast_h5_subtrades (it inherits the
# identity through execution_id).
H5_STRATEGY_SCOPED_TABLES = (
    "forecast_h5_signals",
    "forecast_h5_executions",
    "forecast_h5_paper_trading",
)

# The composite conflict target installed by migration 064.
H5_CONFLICT_TARGET = ("signal_date", "strategy_id")

__all__ = [
    "H5_PRODUCTION_STRATEGY_ID",
    "H5_STRATEGY_SCOPED_TABLES",
    "H5_CONFLICT_TARGET",
]
