---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---
# `.claude/` — Project Knowledge Base

> Single entry-point to the USDCOP/Gold trading system's specs, rules, skills and agents.
> Start here. Every document carries typed YAML front matter (see **Conventions**), and the
> architectural counts below are **generated from source** — never hand-maintained.

---

## The one rule that governs everything: **auto-load boundary**

Claude Code injects **`rules/*.md` into every session** (always-on context budget). Everything
else in `.claude/` is **on-demand** (read only when a task needs it). Design accordingly:

| Folder | Loaded | Put here |
|--------|--------|----------|
| **`rules/`** | **Every session (auto)** | Thin, imperative, always-true rules & contracts. **Presupuesto: ≤3.000 palabras, verificado en CI.** |
| `specs/` | On demand | Dense reference (how things are built/wired). |
| `skills/` | On demand (invocadas) | Flujos operativos repetidos, ejecutables. |
| `agents/` | On demand (invocados) | Revisores especializados, **read-only**. |
| `experiments/` | On demand | RL experiment logs, queue, plans (process artifacts). |
| `templates/` | On demand | Scaffolds to copy when adding an asset / spec / experiment. |
| `generated/` | Nunca a mano | Inventario derivado del código (`inventory.json`). |
| [`codex/`](codex/README.md) | On demand | Canal de revisión independiente de Codex: `audits/`, `plans/`, `proposals/`, `logs/`, `inventories/`, `governance/`. |
| [`coordination/`](coordination/README.md) | Nunca (runtime) | Estado del protocolo dual CLAUDE↔CODEX. **Exento del esquema tipado** — son heartbeats, no documentos. |
| `evidence/` | Fuera del vault | Artefactos probatorios crudos; se abren por ruta desde la spec o auditoría que los gobierna. |
| `archive/` | Nunca | Binarios y material legado retirado del árbol vivo. Excluido del vault. |

> Adding a big reference doc? It goes in `specs/`, **not** `rules/` — or it bloats every session.
> Only genuine always-apply rules (governance, contracts, gates, DO-NOTs) belong in `rules/`.

---

## Este repo es también un vault de Obsidian

La raíz del repo es la raíz del vault (`.obsidian/app.json`). El vault indexa **conocimiento**
(`.claude/**`, `docs/**` y los charters raíz) y excluye **código, datos, evidencia cruda y
estado runtime**. Así, los README adyacentes al código y los worktrees de coordinación no
contaminan el grafo.

Tres convenciones que sostienen esto:

1. **Solo enlaces Markdown relativos.** `app.json` fuerza `useMarkdownLinks: true` +
   `newLinkFormat: "relative"`, así que Obsidian los genera en el formato correcto por
   defecto. Los `[[wikilinks]]` **no** se usan: el link-checker no los parsea, y un enlace que
   CI no puede verificar se pudre en silencio. Ver [`../AGENTS.md`](../AGENTS.md) §4.
2. **El front matter tipado ya es el frontmatter de Obsidian.** `kind`/`status`/`version`/
   `last_verified` son propiedades navegables en la app y, a la vez, contrato verificado en
   `specs-gate.yml`. No hay dos esquemas.
3. **El scratch de los agentes no entra al vault** — runtime de coordinación, evidencia,
   archivos retirados y `codex/_runtime/` quedan fuera. Del protocolo dual permanecen visibles
   únicamente su mapa, la fuente de verdad y la asignación estable.

Los dos charters que leen los agentes viven en la raíz del repo:
[`../CLAUDE.md`](../CLAUDE.md) (Claude Code) y [`../AGENTS.md`](../AGENTS.md) (Codex, que lo
carga automáticamente). `AGENTS.md` no duplica reglas: apunta aquí.

---

## Map — árbol real de `specs/` (generado)

> Escrito por `scripts/diagnostics/generate_inventory.py --write`. **No editar a mano.**
> Aquí vivía un árbol mantenido a mano que acabó describiendo un `assets/` con solo `xauusd/`
> mientras existían `btcusdt/` y `usdcop/` — y era el punto de entrada del que depende todo el
> esquema. Se eliminó a propósito: un índice manual del propio árbol siempre se queda atrás.

<!-- inv:specs_tree -->
```
. (1)
adr/ (3)
assets/ (5)
assets/btcusdt/ (2)
assets/btcusdt/adr/ (1)
assets/btcusdt/design/ (3)
assets/btcusdt/design/adr/ (6)
assets/btcusdt/design/specs/ (13)
assets/btcusdt/specs/ (1)
assets/spx500/ (3)
assets/usdcop/ (6)
assets/xauusd/ (4)
assets/xauusd/adr/ (1)
assets/xauusd/specs/ (13)
audit/ (5)
data/ (1)
operations/ (2)
pipelines/ (3)
planes/ (9)
planes/backlog/ (47)
platform/ (22)
tracks/ (1)
tracks/news-analysis/ (13)
```
<!-- /inv -->

<!-- inv:knowledge -->
**10 rules** (~2,910 palabras auto-cargadas) · **165 specs** · **32 skills** · **3 agents**
<!-- /inv -->

## Capacidades y responsabilidades (generado)

> Catálogo derivado del front matter de cada definición. No se mantiene a mano.

<!-- inv:capabilities -->
### Agentes especializados (solo lectura)

| Agente | Responsabilidad |
|---|---|
| [data-lineage](agents/data-lineage.md) | Traces a feature back to its source — ingestion, timezone handling, T-1 shifts, normalization scope, and contract parity between training and inference. Use when a feature… |
| [quant-reviewer](agents/quant-reviewer.md) | Reviews any claim of trading edge against the quant constitution — trial accounting, Deflated Sharpe, mandatory baselines, look-ahead in three layers, and cost stress. Use before… |
| [spec-auditor](agents/spec-auditor.md) | Audits a spec (or the whole .claude/ tree) against the code it claims to describe — dead code anchors, stale "pending" claims, hand-maintained counts, broken links, SSOT… |

<details>
<summary><strong>Skills operativas y de dominio</strong></summary>

| Skill | Cuándo usarla |
|---|---|
| [approval-cycle](skills/approval-cycle/SKILL.md) | Drive the 2-vote approval and production deploy for a strategy — run the backtest export, read the gates, prepare the human Vote 2, and monitor the deploy DAG. Use when promoting… |
| [bet-sizing](skills/bet-sizing/SKILL.md) | Determine how much capital to allocate to individual positions within a portfolio. Use when the user asks about position sizing, the Kelly criterion, fractional Kelly, risk… |
| [browser-use](skills/browser-use/SKILL.md) | Direct browser control via CDP for web interaction: automation, scraping, testing, screenshots, and site/app work. |
| [commodities](skills/commodities/SKILL.md) | Analyze commodity markets including futures curve dynamics, roll yield, and supply/demand fundamentals. Use when the user asks about commodity investing, commodity ETFs… |
| [contract-change](skills/contract-change/SKILL.md) | Safely change a contract that is mirrored across Python, TypeScript, OpenAPI or RBAC. Use when editing anything under src/contracts, src/core/contracts, services/**/contracts, or… |
| [currencies-and-fx](skills/currencies-and-fx/SKILL.md) | Analyze currency markets, exchange rate mechanics, and FX risk management for international portfolios. Use when the user asks about exchange rates, FX hedging, interest rate… |
| [dag-change](skills/dag-change/SKILL.md) | Add, rename, reschedule, pause or deprecate an Airflow DAG without breaking the registry, the collision-free timeline or downstream sensors. Use when touching anything under… |
| [data-recovery](skills/data-recovery/SKILL.md) | Diagnose and repair stale data in the USDCOP trading system. Use when training is blocked by a freshness gate, when a dashboard page shows "Sin datos", when… |
| [defuddle](skills/defuddle/SKILL.md) | Extract clean markdown content from web pages using Defuddle CLI, removing clutter and navigation to save tokens. Use instead of WebFetch when the user provides a URL to read or… |
| [digital-assets](skills/digital-assets/SKILL.md) | Analyze digital assets including cryptocurrency fundamentals, blockchain mechanics, DeFi protocols, and on-chain metrics. Use when the user asks about crypto investing, Bitcoin… |
| [forward-risk-var](skills/forward-risk-var/SKILL.md) | Estimate potential future losses using VaR, Expected Shortfall, Monte Carlo simulation, and stress testing. Use when the user asks about Value-at-Risk, CVaR, Expected Shortfall… |
| [historical-risk](skills/historical-risk/SKILL.md) | Quantify realized risk from historical data using volatility estimators, drawdown analysis, and downside risk metrics. Use when the user asks about historical volatility, maximum… |
| [json-canvas](skills/json-canvas/SKILL.md) | Create and edit JSON Canvas files (.canvas) with nodes, edges, groups, and connections. Use when working with .canvas files, creating visual canvases, mind maps, flowcharts, or… |
| [obsidian-bases](skills/obsidian-bases/SKILL.md) | Create and edit Obsidian Bases (.base files) with views, filters, formulas, and summaries. Use when working with .base files, creating database-like views of notes, or when the… |
| [obsidian-cli](skills/obsidian-cli/SKILL.md) | Interact with Obsidian vaults using the Obsidian CLI to read, create, search, and manage notes, tasks, properties, and more. Also supports plugin and theme development with… |
| [obsidian-markdown](skills/obsidian-markdown/SKILL.md) | Create and edit Obsidian Flavored Markdown while honoring the vault's repository link policy, plus embeds, callouts, properties, and other Obsidian-specific syntax. Use when… |
| [onboard-asset](skills/onboard-asset/SKILL.md) | Add a new tradeable asset (like Gold or BTC) end to end — AssetProfile, data ingestion, drivers, features, regime fit, backtest, dashboard, monitoring. Use when onboarding a new… |
| [performance-metrics](skills/performance-metrics/SKILL.md) | Evaluate investment performance on a risk-adjusted basis using industry-standard ratios and capture analysis. Use when the user asks about Sharpe ratio, Sortino ratio… |
| [quant-algo-trading](skills/quant-algo-trading/SKILL.md) | Router for the quantitative trading skill library — regime detection, position sizing, risk, backtest validation, performance metrics and cross-asset instrument knowledge for… |
| [rbac-qa](skills/rbac-qa/SKILL.md) | Run the full access-control and UI quality gate for the dashboard — RBAC route coverage, contract invariants, role matrix, functional and visual QA. Use after touching routes… |
| [regen-dashboard](skills/regen-dashboard/SKILL.md) | Regenerate the data the dashboard serves — forecasting CSV/PNGs, per-asset weekly inference JSON, chart OHLCV, and strategy bundles. Use when a dashboard page is empty on a fresh… |
| [release-and-rollback](skills/release-and-rollback/SKILL.md) | Assess release readiness and rollback options before shipping. Use when asked to deploy, promote a canary, cut a release, or roll back — and to check whether the deployment… |
| [return-calculations](skills/return-calculations/SKILL.md) | Compute and compare investment return metrics including TWR, MWR (dollar-weighted IRR on portfolio cash flows), CAGR, and annualized returns. Use when the user asks about… |
| [run-experiment](skills/run-experiment/SKILL.md) | Run a training experiment under the repo's hard rules — one variable, 5 seeds, frozen SSOT config, statistical validation, mandatory report format. Use when the user asks to test… |
| [security-testing](skills/security-testing/SKILL.md) | Test for security vulnerabilities using OWASP principles. Use when conducting security audits, testing auth, or implementing security practices. |
| [spec-sync](skills/spec-sync/SKILL.md) | Verify a spec against the code it claims to describe, then refresh its last_verified date — or archive it if it has gone stale. Use when a spec looks out of date, after changing… |
| [stack-operations](skills/stack-operations/SKILL.md) | Bring the Docker stack up or down in the right mode, verify services are actually healthy, and diagnose a service that will not start. Use when starting work for the day, when a… |
| [statistics-fundamentals](skills/statistics-fundamentals/SKILL.md) | Apply statistical methods to financial data including descriptive statistics, covariance estimation, regression, hypothesis testing, and resampling. Use when the user asks about… |
| [volatility-modeling](skills/volatility-modeling/SKILL.md) | Model, forecast, and interpret volatility using time-series models and options-implied measures. Use when the user asks about EWMA, GARCH models, implied volatility, volatility… |
| [webapp-testing](skills/webapp-testing/SKILL.md) | Toolkit for interacting with and testing local web applications using Playwright. Supports verifying frontend functionality, debugging UI behavior, capturing browser screenshots… |
| [weekly-verify](skills/weekly-verify/SKILL.md) | Run the weekly operator verification for the trading system — confirm training, signal, backups and guardrails after the Sunday/Monday cycle. Use on Monday mornings, after a… |
| [xasset-alpha-engine](skills/xasset-alpha-engine/SKILL.md) | Cross-asset quantitative engine for FX (majors and emerging markets), crypto, equity indices and commodities. NOTE: Deflated/Probabilistic Sharpe here are thin wrappers over… |

</details>
<!-- /inv -->

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
- **Front matter obligatorio** (validado en CI por `test_knowledge_frontmatter.py`):

  ```yaml
  ---
  kind: rule | as-built | roadmap | adr | audit | historical
  status: IMPLEMENTED | PARTIAL | PLANNED | PAUSED | DEPRECATED | SUPERSEDED | HISTORICAL | ARCHIVED
  version: 1.0.0
  last_verified: YYYY-MM-DD     # "lo verifiqué contra el código", no "toqué el archivo"
  supersedes: []
  code_anchors: []              # rutas reales que la spec describe; si no existen, el gate falla
  ---
  ```

  `skills/` y `agents/` usan su propio front matter (`name` + `description`), que parsea el harness.
- **Ningún número vive en prosa.** Los conteos van en bloques `<!-- inv:key --> … <!-- /inv -->`
  que rellena `scripts/diagnostics/generate_inventory.py --write`.
- **SSOT ownership** (avoid duplication): freshness thresholds/recovery → `rules/data-freshness.md`; DAG schedule/timeline → `specs/operations/elite-operations.md`; strategy schemas → `rules/strategy-contract.md`; approval gates → `rules/approval-gates.md`. Other docs **link**, never re-tabulate.
- **`CLAUDE.md`** (repo root) is the always-loaded master; its "SDD Architecture" section indexes this tree.

## Documentos de este directorio

<!-- idx:auto -->

**Subdirectorios:** [`codex/`](codex/README.md) · [`experiments/`](experiments/README.md) · [`rules/`](rules/00-INDEX.md) · [`specs/`](specs/README.md) · [`templates/`](templates/README.md)

<!-- /idx -->
