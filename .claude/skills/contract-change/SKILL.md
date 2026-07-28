---
name: contract-change
description: Safely change a contract that is mirrored across Python, TypeScript, OpenAPI or RBAC. Use when editing anything under src/contracts, src/core/contracts, services/**/contracts, or usdcop-trading-dashboard/lib/contracts — or when changing an SSOT constant (thresholds, feature order, enums) that both backend and dashboard read.
---

# Contract change

## Why this is dangerous

Mirrors are maintained **by convention, and the convention is not filename-based**. Only 2 of the
9 `src/contracts/*.py` files pair with a TS file of a similar name; the real map is below and
cannot be inferred. Two structural gaps make silent drift easy:

- **`contracts-check.yml` never fires on a TypeScript-only change** — it watches only
  `src/**/*.py`, `services/**/*.py`, `airflow/**/*.py` (`.github/workflows/contracts-check.yml:6-15`).
- **The TS side of the SSOT is barely tested.** `test_action_threshold_ssot.py` exists *because
  Python said 0.35 and the TS mirror said 0.33* — and the test it left behind guards the Python
  fallback and the YAML, **not the TS**. The tests that do open a `.contract.ts`
  (`tests/unit/test_newsengine_analysis.py:1096-1112`) only assert that interface names appear as
  strings; that is not field parity.

`specs-gate.yml` does watch `lib/contracts/**`, but it validates the knowledge system, not
contract semantics.

## The mirror map (do not infer pairs from filenames)

| Authoritative source | Mirror |
|---|---|
| `src/contracts/strategy_schema.py` | `usdcop-trading-dashboard/lib/contracts/strategy.contract.ts` |
| `src/contracts/strategy_manifest.py` | `usdcop-trading-dashboard/lib/contracts/strategy-manifest.contract.ts` |
| `src/contracts/forecast_output.py` | `usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts` |
| `src/contracts/policy.py` + `src/contracts/rule_trace.py` (whitelist: `src/contracts/policy_dsl.py`) | `usdcop-trading-dashboard/lib/contracts/policy.contract.ts` |
| `src/contracts/analysis_schema.py` | `usdcop-trading-dashboard/lib/contracts/weekly-analysis.contract.ts` |
| `src/config/backtest_ssot.py` | `usdcop-trading-dashboard/lib/contracts/backtest-ssot.contract.ts` |
| `config/analysis/analysis_assets.yaml` | `usdcop-trading-dashboard/lib/contracts/analysis-assets.ts` |
| `src/core/contracts/feature_contract.py` + `src/core/contracts/action_contract.py` + `config/pipeline_ssot.yaml` | `usdcop-trading-dashboard/lib/contracts/ssot.contract.ts` |
| `services/inference_api/contracts/forecasting.py` | `usdcop-trading-dashboard/lib/contracts/forecasting.contract.ts` |
| `services/signalbridge_api/app/contracts/auth.py` | `usdcop-trading-dashboard/lib/contracts/execution/auth.contract.ts` |
| `services/signalbridge_api/app/contracts/exchange.py` | `usdcop-trading-dashboard/lib/contracts/execution/exchange.contract.ts` |
| `services/signalbridge_api/app/contracts/signal_bridge.py` | `usdcop-trading-dashboard/lib/contracts/execution/signal-bridge.contract.ts` |
| `services/signalbridge_api/app/contracts/signal.py` | `usdcop-trading-dashboard/lib/contracts/execution/signal.contract.ts` |
| `services/signalbridge_api/app/contracts/execution.py` | `usdcop-trading-dashboard/lib/contracts/execution/execution.contract.ts` |
| `services/signalbridge_api/app/contracts/trading.py` | `usdcop-trading-dashboard/lib/contracts/execution/trading-config.contract.ts` |

**No single-file Python mirror** (TS is the SSOT or it is composed): `rbac.contract.ts`,
`admin-console.contract.ts`, `catalog.contract.ts`, `backtest.contract.ts`,
`experiments.contract.ts`, `model.contract.ts`, `ui.contract.ts`.

> This table is enforced by `tests/regression/test_contract_mirrors.py` — if a path moves or
> disappears, the test fails. A hand-written map that nothing checks would rot exactly like the
> counts this repo just finished degenerating.

## Procedure

1. **Find the authoritative side in the map above.** Never guess from the filename.
2. **Change the source first** (Python / YAML / Pydantic).
3. **Update the mirror by hand**: names, enums, optionality, literal values, defaults.
4. **If a FastAPI schema changed**, regenerate OpenAPI:
   `python scripts/tools/export_openapi.py --output docs/api/openapi_v2.yaml`
5. **If RBAC changed**, update `rbac.contract.ts` plus every server/middleware/nav consumer, and
   run the RBAC scripts manually — CI may not fire for a TS-only edit.
6. **Add a parity test for what you changed.** This is the step that actually stops recurrence;
   without it you are relying on the same convention that already failed once.
7. **Regenerate the inventory** if contract counts changed.

## Verify

```powershell
python -m pytest tests/regression/test_contract_mirrors.py tests/regression/test_action_threshold_ssot.py -q
python -m pytest tests/contracts/ -q
python -m pytest tests/regression/test_feature_order_ssot.py tests/regression/test_action_enum_ssot.py -q
python scripts/diagnostics/generate_inventory.py --check
cd usdcop-trading-dashboard; node scripts/check-rbac-coverage.mjs; node --experimental-strip-types scripts/test-rbac-contract.mjs
```

## DO NOT

- Do NOT change only one side of a mirrored pair.
- Do NOT infer the mirror from the filename — `analysis_schema.py` ↔ `weekly-analysis.contract.ts`.
- Do NOT assume `contracts-check.yml` ran: it does not trigger on TS-only changes.
- Do NOT treat a string-presence test as schema parity.
- Do NOT conflate the two threshold families: dashboard/pipeline action thresholds are **0.35 /
  -0.35** (`ssot.contract.ts`, `pipeline_ssot.yaml`); backtest/experiment thresholds are **0.50 /
  -0.50** (`backtest-ssot.contract.ts`, `experiment_ssot.yaml`). Both are correct; they are
  different configs. Do not "fix" one to match the other.
- Do NOT add an exit reason, action or feature in one language only.
