# `specs/` — On-Demand Reference

> Dense, descriptive specs organized by domain. **Not auto-loaded** — read when a task needs depth.
> Always-true rules live in `../rules/` (auto-loaded). Navigation: `../README.md`.

## Domains

| Folder | What lives here |
|--------|-----------------|
| `architecture-overview.md` | As-built map of the whole system (infra, DAGs, contracts, drift) — read first. |
| `platform/` | Cross-cutting contracts: `mlops-lifecycle`, `frontend-architecture`, `dashboard-integration`, `registry-lifecycle`, `execution-bridge`, `risk-management`, `observability`, `cicd-testing`, `authentication`. |
| `data/` | `backup-recovery` (disaster scenarios). Freshness thresholds are in `../rules/data-freshness.md`. |
| `pipelines/` | `training-l2-l3-l4`, `inference-l1-l5` (RL pipeline internals). |
| `operations/` | `elite-operations` — **SSOT for the DAG schedule / collision-free timeline** + recovery playbooks. |
| `tracks/` | **One file/folder per strategy track** (scalable). |
| `assets/` | **One file/folder per tradeable asset** (scalable) + the onboarding playbook & as-built. |
| `audit/` | Point-in-time audit findings as tasks-to-fix. `AUDIT-2026-07-remediation.md` (10-agent code↔spec audit, ~114 findings, P0/P1/P2 backlog). |

## Scalable sub-trees

**`tracks/`** — a strategy track = a way of trading (weekly TP/HS, daily trailing, news-driven…).
- `h5-smart-simple.md` — COP weekly production track.
- `news-analysis/` — News Engine + Analysis package (`_summary.md` + `NN_*.md` detail docs).
- **Add a track**: new `tracks/<track>.md` from `../templates/spec-template.md`, register in
  `../rules/strategy-contract.md` (StrategyRegistry) + the dynamic registry (`platform/registry-lifecycle.md`).

**`assets/`** — a tradeable asset = a symbol with its own session/timezone/drivers.
- `_onboarding-playbook.md` — prescriptive "how to add an asset" (AssetProfile, tests A1–F1).
- `_asbuilt-implementation.md` — multi-asset as-built (per-asset **session/tz/annualization** rules).
- `xauusd/` — Gold package (SPEC-00..12, ADR, roadmap, status).
- **Add an asset**: copy `../templates/asset-profile.example.yaml` → `config/assets/<id>.yaml`,
  then `assets/<id>/` (or `<id>.md`) from `../templates/spec-template.md`. Follow the playbook.

> Underscore-prefixed files (`_onboarding-playbook.md`, `_asbuilt-implementation.md`, `_summary.md`)
> are cross-asset/cross-track meta-docs; non-underscore = a specific asset/track.
