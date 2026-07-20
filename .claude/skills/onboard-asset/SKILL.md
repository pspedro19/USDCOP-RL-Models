---
name: onboard-asset
description: Add a new tradeable asset (like Gold or BTC) end to end — AssetProfile, data ingestion, drivers, features, regime fit, backtest, dashboard, monitoring. Use when onboarding a new symbol or when re-running a stage of an existing asset's DS cycle.
---

# Onboard a tradeable asset

Stages A-I. El playbook completo con la tabla de 17 archivos a tocar y el contrato de 17 tests
está en `.claude/specs/assets/_onboarding-playbook.md` — esta skill es el orden de ejecución y
los errores que ya se cometieron.

## Stages

| # | Stage | Entregable | Tests |
|---|-------|-----------|-------|
| A | PROFILE | `config/assets/<id>.yaml` congelado (CTR-ASSET-PROFILE-001) | — |
| B | DATA | símbolo en los DAGs L0 → `seeds/latest/<id>_*.parquet`, validado | A1-A4 |
| C | DRIVERS | drivers propios del activo en `macro_variables_ssot.yaml`; **quitar los COP-only** | B1 |
| D | FEATURES | lista por activo + hash de contrato + `norm_stats` train-only | C1-C3 |
| E | REGIME FIT | re-ajustar umbrales de Hurst — **jamás copiar los de COP** | D1 |
| F | BACKTEST | walk-forward OOS + 5 gates (Vote 1) | E1-E5 |
| G | DASHBOARD | `strategy_id` registrado + bundle publicado | F1 |
| H | APPROVE | → skill `approval-cycle` | — |
| I | MONITOR | añadir al `config/assets/pipelines.yaml` (la factory emite el DAG semanal) | — |

## Errores ya cometidos (no repetirlos)

- **Umbrales de régimen copiados de COP** (0.52/0.42). Cada activo se re-ajusta: es Stage E, no
  un detalle.
- **Drivers COP-only heredados** (EMBI, IBR, TPM, WTI, yields Colombia). BTC usa un set de 19
  features honesto que los descarta.
- **Anclaje de fecha en barras diarias**: `tz_convert(→ET).normalize()` sobre un sello 00:00-UTC
  corre todas las barras un día atrás. Es el bug "Sunday pile-up" de Gold. Ancla en UTC.
- **Anualización**: por activo. BTC es 24/7 → √365. Nunca compares métricas entre activos.

## Constraints

- El AssetProfile se **congela** antes de Stage B.
- Todo seed pasa por `src/data_quality/ohlcv_validators.py` — barra en día no-sesión = ERROR duro.
- Gate honesto: un candidato debe batir **B1 (buy&hold) y el baseline tonto** en Sharpe **y**
  Calmar, con stress de costos ×2 (`.claude/rules/quant-constitution.md`).
- No declares edge sin DSR trial-aware > 0.95.
- La estrategia llega al dashboard **publicando un bundle**, sin tocar el frontend.
