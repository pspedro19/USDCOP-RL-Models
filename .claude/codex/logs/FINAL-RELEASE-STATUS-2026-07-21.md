---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Final release status — 2026-07-21

## Decision

The engineering and product contracts are ready, but the release harness correctly
returns **NO-GO** for production trading. Commerce, RBAC, news safety and the SP500
quantitative contract pass. Data lineage, acquisition/provider evidence and real PIT
OOS evidence remain blocked; these cannot be resolved by code-only changes.

## Evidence

- Harness output: `evidence/harness-latest.json`
- Passing gates: commerce-contracts, rbac-contracts, news-safety,
  quantitative-contracts.
- Blocked gates: market-data-statistics, acquisition-assets,
  acquisition-manifests, real-data-oos, provider-e2e.

## Closure checklist

1. Reconcile every seed/backup/provider manifest and attach successful provider-run
   evidence.
2. Produce point-in-time datasets and lineage for USDCOP macro vintages and BTC
   crypto-native features.
3. Run volatility-focused OOS/forward evaluation for all four assets, including
   costs, calibration, drawdown and DSR/PBO controls.
4. Re-run the harness with sandbox provider credentials and a disposable tenant.
5. Promote only after all gates are PASS and an explicit approval checkpoint is
   recorded.

Until then, forecasts remain diagnostic and sizing research only; no directional
model-zoo output is authorized for execution.

## Latest verification

- Backup audit: **PASS** (29/29 files readable, 474,228 rows, all declared source
  families present).
- Acquisition asset audit: **REVIEW_REQUIRED** (33 artifacts, 3 blockers).
- Manifest validation: **REVIEW_REQUIRED** (0 provider execution manifests).
- Retraining readiness: **NO-GO** (statistics, PIT vintages, OOS and promotion are
  still incomplete).

The daily DAG cross-reference defect was corrected: clusters now carry database
article IDs and populate `news_cross_reference_articles`. A regression test protects
this contract (`tests/unit/test_cross_reference_persistence_contract.py`).

The USDCOP approval/execution ensemble was also reconciled to the live contract:
Ridge + BayesianRidge mean, with XGBoost disabled until a separately promoted
immutable bundle exists. This prevents Vote 2 from approving a model that execution
does not run.
