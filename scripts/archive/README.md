# `scripts/archive/` — superseded / one-off scripts

Kept for git history and reference; **not** part of any live pipeline. Nothing in `airflow/`,
`Makefile`, `dvc.yaml`, `services/`, or CI references these. Do not wire them back without review.

| File | Why archived | Superseded by |
|------|--------------|---------------|
| `cron_m5_fetch.py` | Parallel cron impl of a live DAG; unwired | Airflow `l0_ohlcv_realtime.py` |
| `cron_macro_update.py` | Parallel cron impl of a live DAG; unwired | Airflow `l0_macro_update.py` |
| `cron_monitor.py` | Parallel cron impl of a live DAG task; unwired | `forecast_h5_l7_multiday_executor.py` (monitor task) |
| `cron_week_end.py` | Parallel cron impl of a live DAG task; unwired | `forecast_h5_l7_multiday_executor.py` (close_week task) |
| `ensure_db_tables.py` | Self-labeled DEPRECATED in its own docstring | `scripts/ops/db_migrate.py` |
| `build_course_pptx.py` | v1; writes the same output file as v2 | `scripts/presentation/build_course_pptx_v2.py` |
| `regenerate_l4_trades_correct.py` | One-off L4-trade regeneration | (one-time task, done) |
| `regenerate_l4_trades_2025_only.py` | One-off L4-trade regeneration (2025 subset) | (one-time task, done) |
| `populate_l4_trades.py` | One-off L4-trade backfill | (one-time task, done) |
| `validate_100_percent.py` | Milestone gate for a completed remediation | (milestone reached) |
| `validate_blockers.py` | P0-blocker gate for a completed milestone | (milestone reached) |
| `validate_dvc.py` | Phase-0.4 DVC-config gate (completed) | (milestone reached) |
| `validate_phase1.py` | V22 Phase-1 gate (completed) | (milestone reached) |
| `validate_phase4.py` | V22 Phase-4 gate (completed) | (milestone reached) |
| `validate_v7_deployment.py` | V7.1 post-deploy gate (completed) | (milestone reached) |

### Non-Python (shell / SQL)

| File | Why archived | Superseded by |
|------|--------------|---------------|
| `cron_runner.sh` | Master cron orchestrator that drives the archived `cron_*.py` | Airflow DAGs |
| `dvc_setup.sh` | Older/smaller DVC setup dup (91 lines, unreferenced) | `scripts/ops/setup_dvc.sh` (287 lines) |
| `v20_migration.sql` | Buggy original; a corrected version exists | `scripts/migrations/v20_migration_fixed.sql` |
| `verify_remediation.sh` | Verification gate for a completed remediation milestone | (milestone reached) |
| `verify_course_delivery.sh` | One-off course-delivery check | (one-time task, done) |

> Reusable validators (health/smoke/pre-training gates) live in `scripts/validation/`, not here.
> `ensure_db_tables.py` was updated to `from scripts.ops.db_migrate import …` so it still imports if run.
