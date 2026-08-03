---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Data granularity inventory

## Market data

- `usdcop_m5_ohlcv`: canonical 5-minute multi-symbol/intraday table; SQL inference
  views derive 1h/4h returns from these bars.
- `asset_daily_ohlcv`: canonical multi-asset daily OHLCV table (BTC, XAU, SPX and
  other daily assets).
- Seeds additionally contain `usdcop_1h_ohlcv.parquet`, plus M5 files for BTC,
  XAU, USD/BRL and USD/MXN. These are artifacts; they are not separate promoted
  SQL contracts unless loaded into the canonical table.

## Macro and crypto-native data

- `macro_indicators_daily`: daily macro panel with release/availability metadata.
- `crypto_derivatives_daily`: daily funding, open-interest and basis aggregates.
- `crypto_onchain_daily` and `crypto_flows_daily`: schema contracts exist for
  on-chain and flow data, pending complete evidence/backfill.

## News and analysis

- `news_articles`, `news_ingestion_log`, `news_cross_references`,
  `news_cross_reference_articles`, `news_daily_digests` and
  `news_feature_snapshots`.
- `weekly_analysis`, `daily_analysis` and `macro_variable_snapshots` store the AI
  analysis layer.

The important distinction is that a Parquet seed proves an artifact exists, not that
the corresponding database table is loaded, fresh, PIT-safe or promotion-eligible.
