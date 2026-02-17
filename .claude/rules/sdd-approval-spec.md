# SDD Spec: Approval Lifecycle & Gates

> **Responsibility**: Authoritative source for approval gates, 2-vote system, and `approval_state.json` schema.
> Other specs reference this file for approval details — do not duplicate gate definitions elsewhere.
>
> Contracts:
> - TypeScript: `lib/contracts/production-approval.contract.ts` (re-exports from strategy.contract.ts)
> - Python: `src/contracts/strategy_schema.py`

---

## Two-Vote Approval System

Every strategy requires two independent approvals before production deployment:

```
                    BACKTEST COMPLETE
                          |
                          v
              ┌───────────────────────┐
  Vote 1/2   │  Python Export Script  │  Automatic gate evaluation
  (Automatic) │  --phase backtest     │  Writes approval_state.json
              └───────────┬───────────┘
                          |
                          v
              ┌───────────────────────┐
              │  approval_state.json  │  status: PENDING_APPROVAL
              │  5 gates evaluated    │  recommendation: PROMOTE/REVIEW/REJECT
              └───────────┬───────────┘
                          |
                          v
              ┌───────────────────────┐
  Vote 2/2   │  Operator on          │  Reviews KPIs, trades, gates
  (Human)    │  /dashboard           │  Clicks "Aprobar" or "Rechazar"
              └───────────┬───────────┘
                          |
                ┌─────────┴─────────┐
                v                   v
           APPROVED            REJECTED
                |                   |
                v                   v
         Stage 6: Deploy     Stay in backtest
         --phase production   --reset-approval to retry
```

### Where Each Vote Happens

| Vote | Actor | Location | Trigger |
|------|-------|----------|---------|
| Vote 1/2 | Python script | CLI (`--phase backtest`) | Automatic during export |
| Vote 2/2 | Human operator | `/dashboard` page | Click Approve/Reject |

**Key**: Vote 2 happens on `/dashboard`, NOT on `/production`.
The `/production` page shows a read-only `ApprovalStatusCard` — no interactive buttons.

---

## Approval Lifecycle

```
PENDING_APPROVAL ──→ APPROVED ──→ LIVE
        |                          ^
        |                   (manual promotion)
        └──→ REJECTED
                |
                └──→ PENDING_APPROVAL  (reset for re-evaluation)
```

| Status | Meaning | Set By | Next Action |
|--------|---------|--------|-------------|
| `PENDING_APPROVAL` | Backtest complete, awaiting human review | Python export script | Operator reviews on `/dashboard` |
| `APPROVED` | Human clicked Approve | Dashboard API | Run `--phase production` (Stage 6) |
| `REJECTED` | Human clicked Reject | Dashboard API | Fix issues, run `--reset-approval` |
| `LIVE` | Strategy deployed to real trading | Manual promotion | Monitor via Stage 7 |

---

## Gate System

Each strategy defines a list of gates. A gate is a pass/fail check on a backtest metric.

### Gate Schema

```json
{
  "gate": "min_return_pct",
  "label": "Retorno Minimo",
  "passed": true,
  "value": 20.03,
  "threshold": -15.0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `gate` | string | Machine-readable gate ID |
| `label` | string | Human-readable label (Spanish) |
| `passed` | bool | Whether the metric meets the threshold |
| `value` | number | Actual metric value |
| `threshold` | number | Gate threshold |

### Default Gates (5)

| Gate | Label | Comparator | Threshold | Description |
|------|-------|------------|-----------|-------------|
| `min_return_pct` | Retorno Minimo | `>` | -15% | Must not lose more than 15% |
| `min_sharpe_ratio` | Sharpe Minimo | `>` | 0.0 | Must have positive Sharpe |
| `max_drawdown_pct` | Max Drawdown | `<` | 20% | Cannot exceed 20% drawdown |
| `min_trades` | Trades Minimos | `>=` | 10 | Must have enough trades |
| `statistical_significance` | Significancia (p<0.05) | `<` | 0.05 | Must be statistically significant |

### Backtest Recommendation

Computed automatically from gate results:

| Recommendation | Condition | Dashboard Display |
|----------------|-----------|-------------------|
| **PROMOTE** | All 5 gates passed | Green badge, Approve button prominent |
| **REVIEW** | Some gates passed, none critical failed | Yellow badge, both buttons visible |
| **REJECT** | Critical gate failed (return or drawdown) | Red badge, Reject button prominent |

`backtest_confidence` = fraction of gates passed (0.0 to 1.0).

---

## Approval Sequence (End-to-End)

```
1. Operator runs: python scripts/train_and_export_smart_simple.py --phase backtest
   |
   ├── Train models on 2020-2024 expanding window
   ├── Evaluate on 2025 OOS data
   ├── Compute metrics (return, Sharpe, MaxDD, WR, p-value)
   ├── Evaluate 5 gates (Vote 1/2)
   └── Write files:
       ├── summary_2025.json          (backtest metrics)
       ├── approval_state.json        (gates + PENDING_APPROVAL)
       └── trades/smart_simple_v11_2025.json  (trade details)

2. Operator navigates to /dashboard
   |
   ├── ForecastingBacktestSection loads summary_2025.json
   ├── Displays: 5 KPI cards, p-value badge, candlestick chart, trade table
   ├── Gates panel shows 5/5 passed (or partial)
   └── Interactive Approve/Reject panel visible

3. Operator clicks "Aprobar" (Approve)
   |
   ├── POST /api/production/approve { action: "approve", notes: "..." }
   ├── API updates approval_state.json:
   |   ├── status: "APPROVED"
   |   ├── approved_by: "operator"
   |   ├── approved_at: "2026-02-16T..."
   |   └── reviewer_notes: "..."
   ├── Auto-fires POST /api/production/deploy (fire-and-forget)
   └── /production shows read-only APPROVED badge

4. Deploy API reads deploy_manifest from approval_state.json
   |
   ├── Spawns: python {manifest.script} {manifest.args}
   ├── Retrain on 2020-2025 (full data including OOS year)
   ├── Generate 2026 production trades
   ├── Seed DB tables via --seed-db (UPSERT, idempotent)
   └── Write: summary.json + trades/smart_simple_v11.json
```

---

## Post-Approval Pipeline

After APPROVED, deploy starts **automatically** via the deploy API. The approve
endpoint fires a non-blocking POST to `/api/production/deploy`, which reads the
`deploy_manifest` from `approval_state.json` and spawns the correct pipeline script.

The deploy API is **manifest-driven**: each backtest script embeds a `deploy_manifest`
in `approval_state.json` specifying the script path, CLI args, and DB tables to seed.
This makes the deploy universal — any strategy (H1, H5, RL) auto-deploys with no
hardcoded strategy detection in TypeScript.

Manual deploy is also supported:

```bash
# Manual deploy (same as what the API fires automatically):
python scripts/train_and_export_smart_simple.py --phase production --no-png --seed-db
```

This retrains models on the expanded window (2020-2025) and:
- Exports `summary.json` — 2026 YTD metrics
- Exports `trades/{strategy_id}.json` — 2026 production trades
- Seeds 5 H5 DB tables via UPSERT (`--seed-db` flag)
- Optional PNGs for `/production` page

After deployment, Stage 7 (Airflow monitoring) takes over with automated weekly retraining.

---

## File Convention

All files live under `usdcop-trading-dashboard/public/data/production/`.

| File | Purpose | Written By | Read By |
|------|---------|------------|---------|
| `approval_state.json` | Gate results + approval status | Python export / Dashboard API | `/dashboard` (interactive), `/production` (read-only) |
| `summary_{year}.json` | OOS backtest metrics | Python export (`--phase backtest`) | `/dashboard` |
| `summary.json` | Production metrics (current year) | Python export (`--phase production`) | `/production` |
| `trades/{sid}.json` | Production trades | Python export (`--phase production`) | `/production` |
| `trades/{sid}_{year}.json` | Backtest trades | Python export (`--phase backtest`) | `/dashboard` |

### File Naming Rules

- `strategy_id` is the key used in filenames and JSON keys
- Trade files: `{strategy_id}.json` (production), `{strategy_id}_{year}.json` (backtest)
- Summary files: `summary.json` (production), `summary_{year}.json` (backtest)
- Only ONE strategy can be active at a time (one `summary.json`, one `approval_state.json`)

### Multi-Strategy Support

When multiple strategies have exported data, each has its own trade files:
```
trades/
├── smart_simple_v11.json
├── smart_simple_v11_2025.json
├── forecast_vt_trailing.json
└── forecast_vt_trailing_2025.json
```

The `summary.json` and `approval_state.json` belong to the currently active strategy.
See `sdd-strategy-spec.md` for the StrategySelector contract.

### approval_state.json Schema

```json
{
  "status": "PENDING_APPROVAL",
  "strategy": "smart_simple_v11",
  "strategy_name": "Smart Simple v1.1.0",
  "backtest_year": 2025,
  "backtest_recommendation": "PROMOTE",
  "backtest_confidence": 1.0,
  "gates": [
    {
      "gate": "min_return_pct",
      "label": "Retorno Minimo",
      "passed": true,
      "value": 20.03,
      "threshold": -15.0
    },
    {
      "gate": "min_sharpe_ratio",
      "label": "Sharpe Minimo",
      "passed": true,
      "value": 3.516,
      "threshold": 0.0
    },
    {
      "gate": "max_drawdown_pct",
      "label": "Max Drawdown",
      "passed": true,
      "value": 3.83,
      "threshold": 20.0
    },
    {
      "gate": "min_trades",
      "label": "Trades Minimos",
      "passed": true,
      "value": 24,
      "threshold": 10
    },
    {
      "gate": "statistical_significance",
      "label": "Significancia (p<0.05)",
      "passed": true,
      "value": 0.0097,
      "threshold": 0.05
    }
  ],
  "backtest_metrics": {
    "return_pct": 20.03,
    "sharpe": 3.516,
    "max_dd_pct": 3.83,
    "p_value": 0.0097,
    "trades": 24,
    "win_rate_pct": 70.8
  },
  "deploy_manifest": {
    "pipeline_type": "ml_forecasting",
    "script": "scripts/train_and_export_smart_simple.py",
    "args": ["--phase", "production", "--no-png", "--seed-db"],
    "config_path": "config/execution/smart_simple_v1.yaml",
    "db_tables": ["forecast_h5_predictions", "forecast_h5_signals",
                  "forecast_h5_executions", "forecast_h5_subtrades",
                  "forecast_h5_paper_trading"]
  },
  "approved_by": null,
  "approved_at": null,
  "reviewer_notes": null,
  "created_at": "2026-02-16T...",
  "last_updated": "2026-02-16T..."
}
```

---

## Reset Flow

To re-evaluate a strategy after retraining or parameter changes:

```bash
python scripts/train_and_export_smart_simple.py --reset-approval
```

This overwrites `approval_state.json` with:
- Fresh gate results from the latest backtest
- `status: PENDING_APPROVAL`
- `approved_by: null`, `approved_at: null`
- New `created_at` and `last_updated` timestamps

The operator must then re-approve on `/dashboard` (Vote 2/2 again).

---

## Dashboard Components

| Component | Page | Purpose |
|-----------|------|---------|
| `ApprovalPanel` | `/dashboard` | Interactive: shows gates, Approve/Reject buttons |
| `ApprovalStatusCard` | `/production` | Read-only: shows approval badge (APPROVED/REJECTED/PENDING) |

### ApprovalPanel (on `/dashboard`)

Visible when `approval_state.status === "PENDING_APPROVAL"`:
- Displays all 5 gates with pass/fail icons
- Shows recommendation badge (PROMOTE/REVIEW/REJECT)
- "Aprobar" button → POST `/api/production/approve` with `action: "approve"`
- "Rechazar" button → POST `/api/production/approve` with `action: "reject"`
- Optional text field for `reviewer_notes`

### ApprovalStatusCard (on `/production`)

Always visible, read-only:
- Green badge if APPROVED
- Red badge if REJECTED
- Yellow badge if PENDING_APPROVAL
- Shows `approved_by` and `approved_at` if available

---

## Related Specs

- `sdd-mlops-lifecycle.md` — **Master lifecycle document** (Stages 0-7)
- `sdd-strategy-spec.md` — Universal strategy schemas + StrategySelector contract
- `sdd-dashboard-integration.md` — JSON/PNG file schemas, page data flows
- `sdd-pipeline-lifecycle.md` — 8-stage pipeline quick reference
