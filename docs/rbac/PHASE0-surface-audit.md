# PHASE0 — Surface audit (generado 2026-07-06)

> Cada ruta real de `app/**` y su permiso efectivo según `lib/contracts/rbac.contract.ts`
> (regenerar: `node scripts/generate-surface-audit.mjs`). Estáticos: `/data/**` y
> `/forecasting/**` exigen sesión en el edge (middleware); `public/` restante = branding.

## API routes (53)

| Ruta | Permiso |
|---|---|
| `/api/analysis/assets` | `analysis:read` |
| `/api/analysis/calendar` | `analysis:read` |
| `/api/analysis/chat` | `analysis:read` |
| `/api/analysis/week/[year]/[week]` | `analysis:read` |
| `/api/analysis/weeks` | `analysis:read` |
| `/api/auth/[...nextauth]` | `public` |
| `/api/backtest` | `research:read` |
| `/api/backtest/real` | `research:read` |
| `/api/backtest/real/stream` | `research:read` |
| `/api/backtest/status/[modelId]` | `research:read` |
| `/api/backtest/stream` | `research:read` |
| `/api/billing/checkout` | `authenticated` |
| `/api/billing/webhook` | `public` |
| `/api/data/[...path]` | `authenticated` |
| `/api/execution/auth/login` | `public` |
| `/api/execution/auth/register` | `public` |
| `/api/execution/exchanges` | `execution:self` |
| `/api/execution/exchanges/[exchange]/balance` | `execution:self` |
| `/api/execution/exchanges/[exchange]/connect` | `execution:self` |
| `/api/execution/exchanges/balances` | `execution:self` |
| `/api/execution/exchanges/credentials` | `execution:self` |
| `/api/execution/executions` | `execution:self` |
| `/api/execution/signal-bridge/history` | `execution:self` |
| `/api/execution/signal-bridge/kill-switch` | `execution:self` |
| `/api/execution/signal-bridge/statistics` | `execution:self` |
| `/api/execution/signal-bridge/status` | `execution:self` |
| `/api/execution/signal-bridge/user/[userId]/limits` | `execution:self` |
| `/api/experiments` | `research:read` |
| `/api/experiments/[id]` | `research:read` |
| `/api/experiments/[id]/approve` | `research:read` |
| `/api/experiments/[id]/reject` | `research:read` |
| `/api/experiments/[id]/trades` | `research:read` |
| `/api/experiments/by-model/[modelId]` | `research:read` |
| `/api/experiments/pending` | `research:read` |
| `/api/health` | `public` |
| `/api/market/candlesticks-filtered` | `market:read` |
| `/api/market/realtime-price` | `market:read` |
| `/api/models/[modelId]/equity-curve` | `research:read` |
| `/api/models/[modelId]/metrics` | `research:read` |
| `/api/pipeline/dates` | `authenticated` |
| `/api/production/approve` | `approval:vote` |
| `/api/production/deploy` | `approval:vote` |
| `/api/production/deploy/status` | `approval:vote` |
| `/api/production/live` | `signals:read` |
| `/api/production/monitor` | `signals:read` |
| `/api/production/status` | `signals:read` |
| `/api/registry` | `research:read` |
| `/api/registry/promote` | `approval:vote` |
| `/api/replay/load-trades` | `research:read` |
| `/api/strategies/[strategyId]/manifest` | `research:read` |
| `/api/trading/performance/multi-strategy` | `signals:read` |
| `/api/trading/signals` | `signals:read` |
| `/api/trading/trades/history` | `signals:read` |

## Pages (14)

| Página | Permiso |
|---|---|
| `/` | `public` |
| `/analysis` | `analysis:read` |
| `/dashboard` | `research:read` |
| `/execution` | `execution:self` |
| `/execution/dashboard` | `execution:self` |
| `/execution/exchanges` | `execution:self` |
| `/execution/executions` | `execution:self` |
| `/execution/login` | `execution:self` |
| `/execution/settings` | `execution:self` |
| `/forecasting` | `forecast:read` |
| `/hub` | `authenticated` |
| `/login` | `public` |
| `/pricing` | `public` |
| `/production` | `signals:read` |
