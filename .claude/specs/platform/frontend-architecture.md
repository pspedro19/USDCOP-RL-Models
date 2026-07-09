# SDD Spec: Frontend Architecture (Dashboard)

> **Responsibility**: Authoritative source for **how the Next.js dashboard is built and behaves** —
> app structure, routing, the BFF API layer, the data-flow/dynamism model, the contracts boundary,
> state management, component organization, and the scalability patterns that let new
> strategies/assets/models appear without editing page code.
> This is the **architecture** view. The **data contract** (Python→JSON file layouts, JSON schemas,
> page data flows) lives in `dashboard-integration.md` — do not duplicate schemas here.
>
> Contract: CTR-FE-ARCH-001
> Version: 1.0.0
> Date: 2026-07-05
> Status: IMPLEMENTED (execution/live-trading surfaces PARTIALLY REALIZED — see §11)
> Cross-refs: `dashboard-integration.md` (data contract), `registry-lifecycle.md` (dynamic strategy
> registry + replay), `strategy-contract.md` (StrategySelector, EXIT_REASONS), `authentication.md`
> (NextAuth + SignalBridge auth), `../tracks/news-analysis/08_DASHBOARD_FRONTEND.md` (analysis page detail).

---

## 1. Mental model (the load-bearing truth)

The dashboard is a **Next.js 15 App-Router application with an embedded BFF** (Backend-for-Frontend:
the `app/api/**` route handlers). It is a **thin read/interaction layer over pipeline outputs** — it
computes almost nothing itself. Three invariants define it:

1. **Pipeline-produced, frontend-rendered.** Metrics, trades, charts, and analysis are produced by
   Python (H5/H1/forecasting/analysis pipelines) and written to `public/data/**` (JSON/CSV/PNG) or a
   Postgres DB. The frontend reads them; it does not recompute alpha.
2. **Hybrid data flow with graceful degradation.** Live surfaces try the DB first, fall back to
   committed files, then to an `unavailable` state — the page never hard-crashes on missing data
   (`hooks/useLiveProduction.ts`).
3. **Contract-driven boundary.** 17 TypeScript `*.contract.ts` files (10 root + 7 execution) mirror the
   Python `src/contracts/` schemas. The Python↔TS boundary is *typed*, so a schema drift surfaces as a
   `tsc`/CI typecheck error. **Caveat**: `next build` currently sets `ignoreBuildErrors` +
   `ignoreDuringBuilds` (`next.config.ts`), so drift does **not** fail the production build today — type
   safety lives in the editor / standalone `tsc` / CI, not the build. Re-enabling build-time checks is a
   tracked debt item (§11).

**Dynamism** = dynamic `strategy_id` lookup + the strategy **registry** + **adaptive polling** + the
**replay** engine. **Scalability** = you add a strategy, asset, or model version and it *appears*
in the UI through the registry/contract layer, with **no page-component edits**.

---

## 2. Tech stack & build

| Concern | Choice | Evidence |
|---------|--------|----------|
| Framework | **Next.js 15.5.2**, App Router (`app/`) | `package.json` |
| UI runtime | **React 19.1.0** (Server + Client Components) | `package.json` |
| Language | TypeScript (strict), path alias `@/*` | `tsconfig.json` |
| Styling | **Tailwind CSS 4.1.13**, dark theme | `tailwind.config.ts`, `postcss.config.mjs` |
| Charts | **lightweight-charts 5** (candles) + **recharts 3** (metrics) | `package.json` |
| Client state | **zustand 5** (`stores/`) + React Context (`contexts/`) | see §7 |
| Data fetching | native `fetch` + **@tanstack/react-query** + bespoke polling hooks (SWR is a dep but **unused** — 0 `useSWR`) | see §5 |
| Auth | **NextAuth 4 (JWT)** at the edge via `middleware.ts` (+ a parallel localStorage/zustand auth for `/execution`) | `middleware.ts`, `authentication.md` |
| Build | `output: 'standalone'`, manual `splitChunks`, **`ignoreBuildErrors`+`ignoreDuringBuilds`=true** | `next.config.ts` |
| Tests | **vitest** (unit) + **Playwright** (e2e) | `vitest.config.ts`, `playwright.config.ts` |
| Container | `Dockerfile.dev` / `Dockerfile.prod`, `healthcheck.js` | repo root of app |
| Lint | ESLint (flat `eslint.config.mjs`; legacy `.eslintrc.json` still present — see §11 debt) | — |

`middleware.ts` runs NextAuth JWT checks at the edge. Public allowlist: `/`, `/login`, `/execution`
(self-managed auth), `/api/auth`, `/api/health`, `/data`, `/forecasting`, static. Protected API prefixes
(`/api/{trading,signals,backtest,pipeline,agent}`) return 401 JSON without a token; `ADMIN_ROUTES` require
`token.role==='admin'` (else 403 / redirect to `/hub`). It also injects security headers
(`X-Frame-Options`, `X-Content-Type-Options`, XSS, Referrer, Permissions-Policy) and forwards
`x-user-id`/`x-user-role` to API routes. Cookie: `next-auth.session-token`.

---

## 3. Routing — page inventory (13 route files = 8 sections + 5 execution sub-pages)

| Route | Kind | Purpose | Primary data |
|-------|------|---------|--------------|
| `/` | section | Landing / entry | static |
| `/hub` | section | Navigation hub to all modules | static |
| `/forecasting` | section | **Multi-asset** (pair selector): USD/COP 9-model ML zoo (CSV+PNG); Gold/BTC rule-based whole-year weekly inference | `public/forecasting/` CSV+PNG + `<asset>/weekly_inference_<year>.json` |
| `/dashboard` | section | 2025 backtest review + **human approval (Vote 2/2)**. **Decision numbers = bundle** (audit I-4, fixed 2026-07-06): the `GatesPanel` next to Approve always renders `approval.gates` from the bundle; `dynamicMetrics` (recomputed from visible trades) only activates during replay or a USER date-filter (compared vs the auto-initialized window, not `summary.year`) and is labeled "PREVIEW DEL REPLAY" — it never feeds Vote 2 | `summary_2025.json` + approval + trades |
| `/production` | section | 2026 YTD **read-only** live monitoring | DB-live + `summary.json` fallback |
| `/analysis` | section | Weekly/daily AI market analysis + macro overlays + chat, with a per-**asset** selector (USD/COP, Gold, BTC) | `public/data/analysis/<asset>/**` |
| `/execution` | section | SignalBridge OMS / exchange integration (own auth) | DB + exchange APIs |
| `/execution/dashboard` | sub | Execution overview | DB/API |
| `/execution/exchanges` | sub | Exchange connect / balances / credentials | exchange APIs |
| `/execution/executions` | sub | Order/execution history | DB |
| `/execution/settings` | sub | Kill-switch, limits, config | SignalBridge API |
| `/execution/login` | sub | Execution-module login | localStorage auth |
| `/login` | section | Dashboard NextAuth login | NextAuth |

> The `/execution/*` sub-tree is a self-contained module with its **own** auth (localStorage), which is
> why `middleware.ts` exempts `/execution` from NextAuth.

---

## 4. BFF API layer — 54 route handlers across 16 groups

> **RBAC (2026-07-06, CTR-RBAC-001):** todas las rutas están gobernadas por
> `lib/contracts/rbac.contract.ts` (SSOT) + `middleware.ts` **deny-by-default**; los estáticos
> `/data/**` + `/forecasting/**` exigen sesión en el edge (antes eran públicos); grupo nuevo
> **billing (3)**: `billing/{checkout,webhook,me}` (webhook público firma-verificada). Páginas
> nuevas: `/pricing` (pública) + `/account/billing`. Gates de CI: `npm run rbac:check` +
> `rbac:test` (cableados en `contracts-check.yml`). Spec: `rbac-monetization.md`.

Route handlers under `app/api/**/route.ts`. Each is a thin adapter that reads a **file**, queries the
**DB**, proxies an **external** exchange/SignalBridge API, or opens a **stream** (SSE).

| Group (count) | Representative routes | Source | Purpose |
|---------------|-----------------------|--------|---------|
| **execution (13)** | `execution/signal-bridge/{status,statistics,history,kill-switch}`, `execution/exchanges/**`, `execution/auth/{login,register}` | EXTERNAL/DB | SignalBridge OMS, exchange connect/balances, module auth |
| **experiments (7)** | `experiments`, `experiments/[id]/{approve,reject,trades}`, `experiments/pending`, `experiments/by-model/[modelId]` | DB/FILE | RL/ML experiment tracking + approval |
| **production (6)** | `production/{live,status,monitor,approve,deploy,deploy/status}` | **DB+FILE** | Live monitor, approval Vote-2, manifest-driven deploy |
| **backtest (5)** | `backtest`, `backtest/real`, `backtest/{real/stream,stream}`, `backtest/status/[modelId]` | DB/STREAM | On-demand backtests + SSE progress |
| **analysis (5)** | `analysis/{assets,weeks,week/[year]/[week],calendar,chat}` | FILE/LLM | Asset list (`assets`) + weekly views + economic calendar + chat; all **asset-aware** via `?asset=` (chat via `body.asset`), default `usdcop` |
| **trading (3)** | `trading/signals`, `trading/trades/history`, `trading/performance/multi-strategy` | DB | Signals, trade history, cross-strategy perf |
| **registry (2)** | `registry`, `registry/promote` | FILE | Dynamic strategy index + promote active version |
| **models (2)** | `models/[modelId]/{metrics,equity-curve}` | DB/FILE | Per-model metrics for the model zoo |
| **market (2)** | `market/candlesticks-filtered`, `market/realtime-price` | DB | OHLCV candles + latest price (no fallback fabrication) |
| **strategies (1)** | `strategies/[strategyId]/manifest` | FILE | One strategy's bundle manifest (versions/backtests) |
| **replay (1)** | `replay/load-trades` | FILE | Immutable per-version trades for client replay |
| **data (1)** | `data/[...path]` | FILE | Generic fs-backed JSON reader for `public/data/**` — serves **post-build** bundle artifacts that Next static 404s (standalone gotcha, §5.A); JSON-only, traversal-guarded |
| **pipeline (1)** | `pipeline/dates` | DB/FILE | Available data dates |
| **health (1)** | `health` | — | Liveness/readiness |
| **auth (1)** | `auth/[...nextauth]` | NextAuth | Session/JWT handler |

> **File vs DB rule of thumb**: approval/backtest **artifacts** are file-based (committed contract in
> `public/data/production/`, regenerable analysis/forecasting output); **live** surfaces
> (`production/live`, `market/*`, `trading/*`) query Postgres. Candlestick/price routes **never
> fabricate fallback data** (fixed 2026-04-06 — see `dashboard-integration.md`).
>
> **Analysis asset resolution**: the asset-aware analysis routes resolve files through the server-side
> resolver `lib/analysis-paths.ts` (`readAnalysisJson`, `analysisDir`), which maps `?asset=<id>` to the
> per-asset dir `public/data/analysis/<asset>/…` (with a legacy-root fallback for the default `usdcop`).
> The id is path-safe — unknown/traversal values collapse to `usdcop` via `resolveAnalysisAsset`
> (`lib/contracts/analysis-assets.ts`, the SSOT asset list mirrored in `config/analysis/analysis_assets.yaml`).
> `GET /api/analysis/assets` returns that SSOT list and drives the `/analysis` selector.

---

## 5. Data flow & dynamism (the core)

Five coexisting server-side source patterns, chosen per route:

- **A — File-based BFF.** Route reads a committed/regenerated JSON/CSV from `public/data/**` and returns
  it typed. Backbone of `/dashboard`, `/forecasting`, `/analysis`, registry, and the approval flow.
  E.g. `registry/route.ts` reads `registry.json` (synthesizes a legacy index if absent);
  `strategies/[id]/manifest` reads with a path-traversal guard; `production/status` reads
  `approval_state.json` (returns a default on miss).
  > **Standalone post-build gotcha (learned 2026-07, CTR-STRAT-REGISTRY-001).** In the `output:
  > 'standalone'` prod image, Next's **static** handler only serves `public/` files that existed at
  > **image-build** time. Bundle artifacts published *after* the build — e.g. a freshly promoted
  > immutable `strategies/<sid>/backtests/<v>/{summary,trades}_<year>.json` written by the Python
  > `BundlePublisher` into the mounted `public/data` — are physically present yet **404 through
  > `/data/...`**, which silently broke the `/dashboard` backtest **replay** for any new version.
  > **Fix:** a generic fs-backed route **`app/api/data/[...path]/route.ts`** reads the file from disk at
  > request time (JSON-only, path-traversal-guarded), and the replay component
  > (`ForecastingBacktestSection.tsx`) fetches **`/api/data/<relPath>`** instead of static `/data/<relPath>`.
  > Post-build bundles are now always served without a rebuild. **Mount must be `:rw`** — the approve /
  > promote / deploy routes `fs.writeFile` into `public/data`; a `:ro` mount raised `EROFS` 500s (fixed in
  > `docker-compose.yml` + `docker-compose.compact.yml`, both `public/data` mounts now `:rw`).
- **B — DB-direct.** `production/live` runs raw SQL (`lib/db/postgres-client`) against the H5 tables
  (`forecast_h5_*`, `usdcop_m5_ohlcv`), computing equity/Sharpe/drawdown/guardrails **server-side** with
  `Promise.allSettled` per query and a `data_source: 'db'|'file'|'unavailable'` + `partial_errors[]`
  envelope. `market/*`, `trading/*` are DB-backed too.
- **C — Backend proxy.** `models/[modelId]/metrics`, `trading/performance/multi-strategy` proxy to
  `INFERENCE_API_URL` (default `:8003`) with a ~3 s `AbortController` timeout, returning a
  `source:'default'` empty payload on failure.
- **D — Streaming (SSE).** `backtest/real/stream`, `backtest/stream` push incremental progress during
  long backtests.
- **E — WebSocket / NRT.** `hooks/useNRTWebSocket.ts`, `hooks/execution/useSignalBridgeWebSocket.ts`, and
  a standalone `server/websocket-server.js` (`npm run ws`) serve execution/market feeds. **Note**: the
  live production monitor uses **polling, not WS**.

**Client-side fetch mechanisms are mixed**: bespoke polling hooks (`useLiveProduction`), **react-query**
(`useWeeklyAnalysis`, `useBacktest`, `useModelMetrics`, execution hooks; `QueryProvider` staleTime 5 min,
retry 1), and `setInterval` polling (`ModelContext` model list; `/forecasting` market status every 60 s).

**Adaptive live loop** (`hooks/useLiveProduction.ts`) is the reference implementation of the dynamism +
degradation model:

```
try /api/production/live (DB)
  └─ DB has trades?  → use DB  (data_source = 'db', isLive = true)
  └─ else            → fall back to /data/production/summary.json + trades/{strategy_id}.json
                       (data_source = 'file'); or 'unavailable'
poll cadence = 30 s when market open, 5 min when closed   (POLL_MARKET_OPEN_MS / _CLOSED_MS)
+ new-trade detection, + countdown timer, + manual refresh()
```

**Dynamic `strategy_id` (no hardcoding).** The active strategy key is read from the data, never
literal-coded: `summary.strategies[summary.strategy_id]` and `trades/${sid}.json`
(`useLiveProduction.ts` L146, L265). This is what lets a *different* strategy become active without a
frontend change (see the KNOWN BUGS entry #2 this fixed). **Known residual**: the live DB path still
defaults `STRATEGY_ID='smart_simple_v11'` — tracked in §11.

**Replay engine.** `hooks/useReplay*.ts` (state machine, animation, keyboard) + `/api/replay/load-trades`
+ `/api/strategies/[strategyId]/manifest` drive client-side, per-version replay over **immutable**
strategy bundles. See `registry-lifecycle.md`.

---

## 6. Contracts layer — the scalability backbone

`lib/contracts/*.ts` are the typed mirror of Python `src/contracts/` (10 root contracts + a 7-file
`execution/` sub-package = 17). The dashboard consumes data **only** through these types, so a Python
schema change that isn't mirrored surfaces as a `tsc`/CI typecheck error (not a `next build` failure —
see §1 caveat). Two contracts add **runtime** validation on top of the static types:
`ssot.contract.ts` compares `OBSERVATION_DIM` against a live `/api/ssot` at runtime (`useSSOT.ts`), and
`forecasting.contract.ts` uses Zod schemas.

| TS contract | Types | Python mirror |
|-------------|-------|---------------|
| `strategy.contract.ts` | `StrategyTrade`, `StrategyStats`, `StrategySummary`, `EXIT_REASON_COLORS` | `strategy_schema.py` |
| `production-approval.contract.ts` | `ApprovalState`, `GateResult`, `ProductionSummary`, `ProductionTradeFile` | `strategy_schema.py` |
| `production-monitor.contract.ts` | `LiveProductionResponse`, `CurrentSignal`, `ActivePosition`, `Guardrails`, `MarketState` | live monitor API |
| `weekly-analysis.contract.ts` | `WeeklyViewData`, `DailyAnalysisEntry`, `MacroVariableSnapshot` | `analysis_schema.py` |
| `forecasting.contract.ts` | Forecasting CSV row + view types | forecasting export |
| `backtest.contract.ts` / `backtest-ssot.contract.ts` | Backtest result + SSOT config types | backtest export |
| `experiments.contract.ts` | Experiment tracking types | experiment registry |
| `model.contract.ts` | Model registry types | model registry |
| `ssot.contract.ts` | RL feature-contract mirror (20-feat v3.1.0, hash-pinned) | `src/core/contracts/feature_contract.py` |

**Rule**: adding a field or exit reason means editing **both** the TS contract and the Python schema
(see `strategy-contract.md` DO-NOTs). The contract is the API. Because `next build` currently ignores
type errors, treat `tsc`/CI as the enforcement point — not the build.

---

## 7. State management

| Layer | Item | Purpose |
|-------|------|---------|
| zustand (`stores/`) | `useAnalysisChatStore.ts` (+ execution `auth-storage`) | Analysis chat state (carries `contextAsset` for asset-scoped chat); execution auth token |
| Context (`contexts/`) | `LanguageContext.tsx` | i18n toggle (ES/EN) — landing page |
| Context (`contexts/`) | `ModelContext.tsx` | Selected model (fetches `/api/models`, comparison ≤4, default on 5xx) |
| Providers | `QueryProvider` (react-query), `NotificationProvider`, `ErrorBoundary` | Query cache, toasts, crash isolation |
| Hooks (`hooks/`, ~20) | `useLiveProduction`, `useWeeklyAnalysis` (asset-scoped: `useAnalysisAssets`, `useAnalysisIndex(asset)`, `useWeeklyView(y,w,asset)`, `useUpcomingEvents(asset)`), `useBacktest`, `useModelMetrics`, `useTradesHistory`, `useIntegratedChart`, `useNRTWebSocket`, `useReplay{,Animation,Keyboard,StateMachine,Timeline}`, `execution/*` | Fetching, polling, streaming, replay, live feeds |

Most cross-page shared state is **server-owned** (files/DB) and re-fetched, not held client-side — a
deliberate simplicity choice consistent with the "thin read layer" model.

---

## 8. Component organization (`components/`, ~87 components across 15 subdirs)

> Shared design system in `components/ui/` (`button`, `badge`, `card`, `table`, `skeleton`,
> notifications) built on Radix primitives + `class-variance-authority` + `clsx`, composed via the
> `cn()` helper (`@/lib/utils`). `components/execution/_pages_reference/` is a non-routed reference copy
> (dead weight — see §11).

| Subdir | Role |
|--------|------|
| `production/` | `/production` + approval widgets (`ForecastingBacktestSection`, `ApprovalPanel`, `ApprovalStatusCard`) |
| `forecasting/` | Model-zoo panels, rankings, forward/consensus views |
| `analysis/` | Weekly analysis, macro overlays, chat widget, asset selector (`AssetSelector.tsx`) |
| `execution/` | SignalBridge OMS, exchange, kill-switch UIs |
| `charts/` | Candlestick + metric chart wrappers (lightweight-charts/recharts) |
| `trading/`, `mlops/`, `monitoring/`, `operations/`, `alerts/` | Domain panels |
| `navigation/`, `landing/`, `common/`, `providers/`, `ui/` | Shell, layout, shared primitives, context providers, design-system components |

---

## 9. Scalability patterns (in the code today)

| Pattern | How | Where |
|---------|-----|-------|
| **Dynamic strategy registry** | `registry.json` + per-strategy `manifest.json` + `registry/promote` flip the active version; UI renders whatever the registry lists | `registry-lifecycle.md`, `/api/registry*`, `ForecastingBacktestSection.tsx` |
| **Dynamic `strategy_id`** | Active key read from data, never hardcoded | `useLiveProduction.ts` |
| **StrategySelector** | `/api/production/strategies` scans `summary*.json`; dropdown appears only when >1 strategy | `strategy-contract.md` §StrategySelector |
| **Graceful degradation** | DB→file→unavailable; PNG `onError` hides container; registry UI no-ops on legacy builds | `useLiveProduction.ts`, `dashboard-integration.md` §PNG |
| **Contract-typed boundary** | Build fails on unmirrored schema drift | `lib/contracts/**` |
| **Exit-reason registry** | `EXIT_REASON_COLORS` maps reasons→style; unknown reasons render neutral, no crash | `strategy.contract.ts` |
| **i18n** | `LanguageContext` ES/EN | `contexts/LanguageContext.tsx` |
| **Manifest-driven deploy** | `/api/production/deploy` reads `deploy_manifest` from `approval_state.json` — no per-strategy TS | `approval-gates.md` |

**Add a strategy (no page edits)**: pipeline exports `summary*.json` + `trades/{sid}.json` + registry
bundle → `/api/production/strategies` and `/api/registry` pick it up → selector/version UI shows it.

**Add a page/section**: create `app/<route>/page.tsx`; if it needs data, add an `app/api/<group>/route.ts`
adapter returning a contract type; add the contract to `lib/contracts/` mirroring the Python schema.

---

## 10. Important files map

| Path | Role |
|------|------|
| `middleware.ts` | Edge NextAuth gate (public allowlist + JWT) |
| `app/**/page.tsx` | 13 route pages (§3) |
| `app/api/**/route.ts` | 50 BFF handlers (§4) |
| `lib/contracts/*.ts` | 10 typed Python↔TS contracts (§6) |
| `lib/contracts/analysis-assets.ts` | SSOT analysis asset list (`ANALYSIS_ASSETS`, `DEFAULT_ANALYSIS_ASSET`, `resolveAnalysisAsset`, `isValidAnalysisAsset`) — mirrored by `config/analysis/analysis_assets.yaml` |
| `lib/analysis-paths.ts` | Server-side per-asset analysis file resolver (`readAnalysisJson`, `analysisDir`; path-safe, legacy-root fallback) |
| `hooks/useLiveProduction.ts` | Reference live-data loop (DB-first, adaptive poll, file fallback) |
| `hooks/useReplay*.ts` | Client replay over immutable bundles |
| `stores/useAnalysisChatStore.ts` | Analysis chat state |
| `contexts/{LanguageContext,ModelContext}.tsx` | i18n + model selection |
| `components/production/ForecastingBacktestSection.tsx` | Backtest review + approval + version replay |
| `public/data/production/**` | **Committed** approval-flow contract (summary/approval/trades) |
| `public/data/{registry.json,strategies/**}` | Dynamic strategy registry bundles |
| `public/{forecasting,data/analysis/<asset>}/**` | **Regenerable** pipeline output, per-asset for analysis (untracked — see §11) |

---

## 11. Roadmap / future plans (this system keeps evolving)

> Integrated here so the frontend spec tracks what's next, not just what's built.

1. **Regenerable dashboard data is now untracked** (2026-07). `public/forecasting/` (PNGs) and
   `public/data/analysis/` (JSON) were removed from git as regenerable pipeline output; only
   `public/data/production/**` (the approval-flow contract) stays committed. **Consequence**: a fresh
   clone/deploy shows `/forecasting` and `/analysis` empty until the weekly pipeline regenerates them.
   **Plan**: publish these as a CI/pipeline artifact **or** serve them from MinIO/object storage so a
   fresh deploy is populated without committing binaries. Until then, graceful degradation keeps the
   pages from breaking (`onError` PNG hide, empty-state components).
2. **Model artifacts → MinIO.** Per the CLAUDE.md roadmap, `models/**` binaries migrate to object
   storage; `models/[modelId]/*` routes should then read from MinIO instead of the filesystem.
3. **Multi-strategy comparison.** Today only ONE active `summary.json`. Planned: per-strategy
   `summary_{strategy_id}.json` so `/production` can compare strategies side-by-side
   (`strategy-contract.md` notes this as the future enhancement).
4. **Multi-asset surfacing.** Gold (`assets/xauusd/`) and BTC (`assets/btcusdt/`) are onboarded and the
   `assets/` tree is scalable. **Shipped (2026-07)**: `/analysis` now has a per-asset switcher
   (`components/analysis/AssetSelector.tsx`) driven by the SSOT list — USD/COP, Gold, BTC — with
   asset-scoped weekly+daily analysis (Gold/BTC data from `src/analysis/asset_analysis_generator.py`;
   USD/COP unchanged). `/forecasting` **now also** carries the same pair selector (reusing
   `AssetSelector`): USD/COP → the 9-model ML zoo; Gold/BTC → `WeeklyInferenceView.tsx` (rule-based
   whole-year weekly positioning, backtest-2025 default / production-2026, from
   `public/forecasting/<asset>/weekly_inference_<year>.json`). **Next**: extend the same asset-switcher
   pattern to `/production` as more assets go live.
5. **Execution → live trading.** The `/execution` module is PARTIALLY REALIZED — testnet→live
   promotion, SignalBridge auth consolidation, and kill-switch UX are tracked in
   `authentication.md` + `execution-bridge.md`.
6. **NRT/WebSocket expansion.** `useNRTWebSocket` + inference feeds broaden as the RL/NRT path matures.
7. **H1 daily track re-enable.** `/forecasting` H1 views are paused pending H5 v2.0 validation; re-surface
   when the H1 DAGs resume (`h5-smart-simple.md`).
8. **Build/type hygiene debt** (found in the 2026-07 frontend audit — fix before scaling):
   - `next.config.ts` sets `ignoreBuildErrors` + `ignoreDuringBuilds` = true → **type/lint drift does
     not fail the build**. Re-enable so the contract layer actually gates deploys.
   - **Residual hardcode**: the live DB path defaults `STRATEGY_ID = 'smart_simple_v11'`
     (`app/api/production/live/route.ts`, `hooks/useLiveProduction.ts`) despite the dynamic registry
     elsewhere — make it registry-driven for true multi-strategy live monitoring.
   - **Dual auth systems**: NextAuth (main app) + localStorage/zustand (execution). Consolidate per
     `authentication.md`.
   - Remove dead `components/execution/_pages_reference/`; drop the unused `swr` dependency.
   - Consolidate ESLint to flat `eslint.config.mjs`; remove legacy `.eslintrc.json`.

---

## Cross-References

| Concern | Spec |
|---------|------|
| Python→JSON/CSV/PNG **data contract**, page data flows, JSON safety | `dashboard-integration.md` |
| Strategy/trade/gate schemas, StrategySelector, exit reasons | `../../rules/strategy-contract.md` |
| Dynamic strategy registry + immutable bundles + replay | `registry-lifecycle.md` |
| Approval gates + 2-vote lifecycle | `../../rules/approval-gates.md` |
| NextAuth + SignalBridge auth as-built | `authentication.md` |
| Analysis page (news/analysis) detail | `../tracks/news-analysis/08_DASHBOARD_FRONTEND.md` |
| Whole-system as-built map | `../architecture-overview.md` |

## DO NOT

- Do NOT hardcode `strategy_id` in a page/component — read it from `summary.strategy_id` (dynamism breaks otherwise).
- Do NOT add a data field without updating **both** the TS `*.contract.ts` and the Python `src/contracts/` schema.
- Do NOT fabricate fallback market data in `market/*` routes — return empty/error, never fake candles/prices.
- Do NOT assume `public/forecasting/` or `public/data/analysis/` exist on a fresh deploy — they are regenerable/untracked; rely on graceful degradation.
- Do NOT bypass `middleware.ts` auth for protected routes; `/execution` is exempt only because it carries its own auth.
