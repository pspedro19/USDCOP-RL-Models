---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Seed/backup reconciliation

The read-only checker `scripts/analysis/reconcile_seed_backups.py` compares the
canonical seeds with their backup copies without modifying either file. It
normalizes timestamp values in memory to UTC, reports source-specific rows,
coverage, duplicate/invalid timestamps, column drift and SHA-256 hashes.

Evidence: `evidence/seed-backup-reconciliation.json`.

Current status is expected to be `REVIEW_REQUIRED` when seed and backup
coverage differs (for example, historical backfill rows exist only in the
backup). This is a reconciliation finding, not permission to overwrite the
canonical seed. The next controlled step is an approved backfill into a new
versioned artifact, followed by quality gates and manifest promotion.

The unit test confirms both datasets are inspected, hashes are present and the
operation is read-only.
