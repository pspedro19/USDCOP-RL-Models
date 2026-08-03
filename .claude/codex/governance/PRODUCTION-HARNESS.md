---
kind: audit
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Production harness

`src.validation.production_harness` is the deterministic lifecycle gate used
before promotion, scheduled retraining and rollback. It checks artifact
lineage (`model_version`, `dataset_hash`), feature PSI drift, latency/error/
uptime SLOs, Sharpe/max drawdown risk limits, and champion–challenger
regression. A kill switch forces `go=false`. Evidence is written as JSON and
is safe to archive for audit; no code path silently promotes a model.

Run: `python scripts/validation/run_production_harness.py manifest.json
--champion champion.json --out .claude/codex/production-harness-latest.json`.
Exit code 0 means all gates passed; 2 means blocked and requires rollback or
human review. Retraining jobs should invoke this gate with a frozen dataset
hash and retain the resulting evidence artifact.
