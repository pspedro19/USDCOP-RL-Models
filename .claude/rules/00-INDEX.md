---
kind: rule
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# Rules Index (auto-loaded)

> These `rules/*.md` are injected into **every** session — keep them thin and always-true.
> Dense reference lives in `../specs/` (on-demand). This index maps each rule to its deep spec.

| Rule (auto-loaded) | Governs | Deep reference (on-demand) |
|--------------------|---------|-----------------------------|
| `data-governance.md` | L0 OHLCV + macro: timezone golden rule (America/Bogota), multi-pair table, UPSERT, BRL quirk | `../specs/pipelines/l0-data-reference.md` |
| `data-freshness.md` | Freshness **thresholds** (OHLCV 3d, macro 7d, models 10d) **(SSOT)** | `../specs/operations/freshness-recovery.md` (runbooks + migraciones) |
| `strategy-contract.md` | Invariantes de estrategia (id universal, JSON safety, per-asset) | `../specs/platform/strategy-schemas.md` (schemas completos) |
| `approval-gates.md` | Invariantes del doble voto | `../specs/platform/approval-lifecycle.md` (secuencia + schema) |
| `experiment-protocol.md` | Experiment discipline: 1 variable, 5 seeds, statistical validation | `../templates/experiment-config-template.md`, `../experiments/` |
| `quant-constitution.md` | **Transversal anti-selección** (todos los tracks): no grid-search sobre test, trials+DSR, baselines B1′/tonto/costos, anti-look-ahead 3 capas, retiro pre-firmado | `../specs/assets/btcusdt/design/constitution-modeling.md`, `../specs/audit/PLAN-completar-sistema-2026-07.md` |
| `rbac.md` | **RBAC + monetización** (CTR-RBAC-001): deny-by-default, rol≠plan, entitlements server-side, PreTradeGate paper-first, billing por webhook, audit append-only | `../specs/platform/rbac-monetization.md` |
| `strategy-engines.md` | **Motores de estrategia** (rule_based/ml/rl/composite): un contrato, un motor de evaluación, DSL whitelist, trace no-recalculado, reglas también cobran trials | `../specs/planes/05-rule-based-strategies.md` |
| `ssot-versioning.md` | Frozen experiment SSOT configs, versioning lifecycle | `experiment-protocol.md` |

## SSOT ownership (do not duplicate across docs)

| Concern | Authoritative file | Everyone else |
|---------|--------------------|---------------|
| Freshness thresholds + recovery | `data-freshness.md` | link to it |
| DAG schedule / collision-free timeline | `../specs/operations/elite-operations.md` | link to it |
| Strategy/trade/gate schemas | `strategy-contract.md` | link to it |
| Approval gates | `approval-gates.md` | link to it |
| Timezone / OHLCV governance | `data-governance.md` | link to it |
| Per-asset session/tz/annualization | `../specs/assets/_asbuilt-implementation.md` | link to it |
| Anti-selección / trials / DSR / baselines (todos los tracks) | `quant-constitution.md` | link to it |

> Full navigation + how to add an asset/track/spec: see `../README.md`.
