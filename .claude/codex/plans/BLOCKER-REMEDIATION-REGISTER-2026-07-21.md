---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Blocker remediation register

| Blocker | Evidence | Required closure |
|---|---|---|
| PIT lineage | Public snapshots explicitly set `pit_vintage=false` and `promotion_eligible=false`; macro audit says vintages require verification | Acquire release-vintage history and validate `available_at >= release_date` for every feature |
| Provider execution | Acquisition audit has 33 artifacts but no provider execution manifests | Run each provider in sandbox, persist request window, response checksum, timestamp and row counts |
| Backup durability | Local restore is readable, but remote/offsite durability is not evidenced | Execute a restore drill to an isolated target and record checksum parity plus offsite location |
| Feature quality | Macro series have large missingness and 522 gaps over three days; `usdcop_1h` has no parsed date range | Profile cadence per source, fix timestamp parsing, and define feature-specific missingness policies |
| OOS promotion | Retraining readiness is NO-GO: no PIT/OOS/promotion evidence | Run volatility-first OOS/forward evaluation with costs, calibration, drawdown and DSR/PBO |

No local artifact can legitimately close the first, second or fifth blocker without
new provider/PIT evidence. The harness must remain NO-GO until those artifacts exist.
