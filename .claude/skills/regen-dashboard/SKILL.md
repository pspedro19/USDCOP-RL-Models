---
name: regen-dashboard
description: Regenerate the data the dashboard serves — forecasting CSV/PNGs, per-asset weekly inference JSON, chart OHLCV, and strategy bundles. Use when a dashboard page is empty on a fresh clone, after a data recovery, or when /forecasting or /dashboard shows stale results.
---

# Regenerate dashboard data

## Why this is a skill and not a doc

The two places that documented this **contradict each other**:
`specs/operations/elite-operations.md` lists 3 commands and omits the per-asset ones that
`rules/data-freshness.md` includes. Follow the shorter list and **Gold and BTC silently stay
stale** — the pages render, just with old numbers, which is worse than an obvious failure.

This skill is the single list.

## Full regeneration (in order)

```bash
# 1. USD/COP forecasting — whole-year CSV + PNGs (~30-45 min)
python scripts/pipeline/generate_weekly_forecasts.py --num-weeks 30

# 2. Per-asset weekly inference (Gold + BTC) — DO NOT SKIP
python scripts/pipeline/generate_asset_weekly_forecast.py --asset all --year all

# 3. Chart OHLCV for the /dashboard backtest chart (all assets)
python -m scripts.data.export_chart_ohlcv

# 4. Strategy bundles + production summaries
python scripts/pipeline/train_and_export_smart_simple.py --phase both

# 5. Weekly analysis (/analysis)
python scripts/pipeline/generate_weekly_analysis.py
```

## Targeted regeneration

Prefer the narrowest command that fixes the symptom:

| Symptom | Command |
|---|---|
| `/forecasting` vacío (USD/COP) | paso 1 |
| `/forecasting` vacío solo Gold/BTC | paso 2 |
| Chart de backtest sin velas | paso 3 |
| KPIs de `/dashboard` o `/production` viejos | paso 4 |
| `/analysis` sin semanas | paso 5 |
| `/analysis` con charts macro vacíos | `python scripts/ops/patch_analysis_macro_charts.py` |

## Verify

Cada paso escribe archivos concretos — compruébalos, no asumas:

- `public/forecasting/bi_dashboard_unified.csv` + PNGs
- `public/forecasting/<asset>/weekly_inference_<year>.json`
- `public/data/market/<SYMBOL>_daily.json`
- `public/data/production/summary*.json` + `trades/`

## Constraints

- Estos outputs **no están trackeados** (son regenerables): un clone fresco muestra
  `/forecasting` vacío hasta correr esto. Eso es esperado, no un bug.
- El paso 1 tarda 30-45 min. No lo lances si el síntoma es solo de Gold/BTC.
- Paso 4 con `--phase both` **resetea el estado de aprobación** → requiere Vote 2 humano otra vez.
  Si solo querías refrescar el backtest, usa `--phase backtest`.
- Gold/BTC también se regeneran solos vía el stage `l5_weekly_forecast` de sus DAGs semanales.
