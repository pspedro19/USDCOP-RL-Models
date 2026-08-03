---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Daily accumulation / Friday close evidence

The production contract is now protected by
`tests/regression/test_news_weekly_rollup_contract.py`.

- News ingestion runs at 07:00, 12:00 and 18:00 UTC Monday–Friday.
- The L8 analysis runs at 19:00 UTC Monday–Friday, after the news DAG sensor.
- Each weekday calls `generate_for_date(today)` and appends/replaces that day's
  entry in the current weekly JSON.
- Friday is detected with `today.weekday() == 4` and calls `generate_for_week` to
  close the weekly analysis and export the final synthesis.
- The NewsEngine separately emits daily and Monday weekly statistical digests.

This proves the scheduling and roll-up contract statically. A production health
claim still requires one successful Airflow run with provider, database, LLM-agent,
JSON-export and checksum evidence.
