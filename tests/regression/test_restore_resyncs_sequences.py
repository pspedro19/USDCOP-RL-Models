"""A restore must leave the database able to accept new rows.

Contract: CTR-DQ-OPS-002

The feature-data backup carries explicit `id` values, so restoring writes rows without ever
calling `nextval`. The sequence stays where it was, and the next NATURAL insert draws an id
that already exists -> primary-key UniqueViolation.

Observed 2026-07-20 on a live database: `news_articles_id_seq.last_value = 3` against
`MAX(id) = 37`. `news_daily_pipeline::ingest_all_sources` failed on all 3 retries. The same
drift was present in forecast_h5_{predictions,subtrades,paper_trading,signals,executions} and
news_feature_snapshots -- 7 tables, all of them in the restore set. A cold restore therefore
left the entire H5 signal and execution path unable to write.

The tables were populated and read fine, so every "is the data there?" check passed. Only a
WRITE reveals it. That is why this test asserts on the resync call rather than on row counts.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "scripts" / "ops" / "backup" / "feature_data_backup.py"


def _tree() -> ast.Module:
    return ast.parse(MODULE.read_text(encoding="utf-8", errors="replace"))


def test_resync_helper_exists():
    fns = {n.name for n in ast.walk(_tree()) if isinstance(n, ast.FunctionDef)}
    assert "_resync_sequences" in fns, (
        "restore must advance sequences past the ids it inserted, or the next natural "
        "insert collides with a restored id"
    )


def test_restore_calls_resync():
    """The call must be executable code inside restore(), not a comment or a docstring."""
    restore = next(
        n for n in ast.walk(_tree())
        if isinstance(n, ast.FunctionDef) and n.name == "restore"
    )
    calls = [
        n for n in ast.walk(restore)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_resync_sequences"
    ]
    assert calls, (
        "restore() never calls _resync_sequences(). Restored tables would read correctly "
        "and reject every new row -- the failure mode that broke news ingestion."
    )


def test_resync_runs_before_commit():
    """Resync belongs in the same transaction as the insert.

    If it ran after the commit and the process died in between, the database would be left
    in exactly the broken state this fix exists to prevent.
    """
    src = MODULE.read_text(encoding="utf-8", errors="replace")
    resync_at = src.index("_resync_sequences(cur, table)")
    commit_at = src.index("conn.commit()", resync_at)
    between = src[resync_at:commit_at]
    assert between.count("\n") <= 2, (
        "_resync_sequences must run immediately before conn.commit(), inside the same "
        "transaction as the insert"
    )
