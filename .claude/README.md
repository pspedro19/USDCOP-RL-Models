# `.claude/` — Project Knowledge Base

> Single entry-point to the USDCOP/Gold trading system's specs, rules, and process docs.
> Start here. Every doc has a `Contract · Version · Status` header and cross-links.

---

## The one rule that governs everything: **auto-load boundary**

Claude Code injects **`rules/*.md` into every session** (always-on context budget). Everything
else in `.claude/` is **on-demand** (read only when a task needs it). Design accordingly:

| Folder | Loaded | Put here |
|--------|--------|----------|
| **`rules/`** | **Every session (auto)** | Thin, imperative, always-true rules & contracts. Keep it small. |
| `specs/` | On demand | Dense reference (how things are built/wired). |
| `experiments/` | On demand | RL experiment logs, queue, plans (process artifacts). |
| `templates/` | On demand | Scaffolds to copy when adding an asset / spec / experiment. |

> Adding a big reference doc? It goes in `specs/`, **not** `rules/` — or it bloats every session.
> Only genuine always-apply rules (governance, contracts, gates, DO-NOTs) belong in `rules/`.

---

## Map

```
.claude/
├── README.md                     ← you are here
├── rules/                        ← AUTO-LOADED. Always-true rules & contracts.
│   ├── 00-INDEX.md                  rule map → points to specs/ for depth
│   ├── data-governance.md           L0 data rules (timezone golden rule, OHLCV/macro contracts)
│   ├── data-freshness.md            freshness thresholds + recovery (SSOT)
│   ├── strategy-contract.md         universal strategy/trade/gate schemas
│   ├── approval-gates.md            2-vote approval + 5 gates
│   ├── experiment-protocol.md       experiment discipline (1 var, 5 seeds, stats)
│   └── ssot-versioning.md           frozen experiment configs
│
├── specs/                        ← ON-DEMAND reference, by domain.
│   ├── architecture-overview.md     as-built map of the whole system
│   ├── platform/                    cross-cutting contracts (mlops, frontend-architecture, dashboard,
│   │                                registry, execution, risk, observability, cicd, authentication)
│   ├── data/                        backup-recovery
│   ├── pipelines/                   training (L2-L3-L4) + inference (L1-L5)
│   ├── operations/                  elite-operations (DAG schedule SSOT, recovery playbooks)
│   ├── tracks/                      ONE folder/file per strategy track  ← scalable
│   │   ├── h5-smart-simple.md          production track (COP weekly)
│   │   └── news-analysis/              News Engine + Analysis package
│   ├── assets/                      ONE folder/file per tradeable asset ← scalable
│   │   ├── _onboarding-playbook.md     how to add an asset (prescriptive)
│   │   ├── _asbuilt-implementation.md  multi-asset as-built (session/tz/annualization)
│   │   └── xauusd/                     Gold spec package (SPEC-00..12, ADR, status)
│   └── audit/                       point-in-time audits → tasks-to-fix
│       └── AUDIT-2026-07-remediation.md  10-agent code↔spec audit (~114 findings, P0/P1/P2)
│
├── experiments/                  ← ON-DEMAND. RL experiment log, queue, plans.
└── templates/                    ← ON-DEMAND. Scaffolds for new asset/spec/experiment.
```

---

## How to extend (scalability)

| I want to… | Do this |
|------------|---------|
| **Add a tradeable asset** (BTC, …) | Read `specs/assets/_onboarding-playbook.md`; copy `templates/asset-profile.example.yaml` → `config/assets/<id>.yaml`; create `specs/assets/<id>/` (or a single `<id>.md`) from `templates/spec-template.md`. |
| **Add a strategy track** | Create `specs/tracks/<track>.md` (or a folder) from `templates/spec-template.md`; register it in `rules/strategy-contract.md` (StrategyRegistry) and the dynamic registry. |
| **Add a reference spec** | `specs/<domain>/<name>.md` from `templates/spec-template.md`. Never add dense reference to `rules/`. |
| **Add an always-apply rule** | Append to the right `rules/*.md` (keep it thin). Update `rules/00-INDEX.md`. |
| **Run an experiment** | Copy `templates/experiment-config-template.md`; follow `rules/experiment-protocol.md`; log in `experiments/`. |

---

## Conventions

- **Naming**: kebab-case topic names. `NN-` prefix only inside ordered packages (`assets/xauusd/specs/SPEC-NN`, `tracks/news-analysis/NN_`).
- **Header**: every spec starts with `> Contract · Version · Status · cross-refs`.
- **SSOT ownership** (avoid duplication): freshness thresholds/recovery → `rules/data-freshness.md`; DAG schedule/timeline → `specs/operations/elite-operations.md`; strategy schemas → `rules/strategy-contract.md`; approval gates → `rules/approval-gates.md`. Other docs **link**, never re-tabulate.
- **`CLAUDE.md`** (repo root) is the always-loaded master; its "SDD Architecture" section indexes this tree.
