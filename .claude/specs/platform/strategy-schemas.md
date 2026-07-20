---
kind: as-built
status: IMPLEMENTED
contract: CTR-STRATEGY-SCHEMA-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - src/contracts/signal_contract.py
  - src/contracts/signal_adapters.py
  - src/contracts/execution_strategies.py
  - usdcop-trading-dashboard/lib/contracts/strategy.contract.ts
---
# SDD Spec: Schemas de estrategia (referencia)

> **Responsibility**: schemas completos (TradeRecord, MetricsSummary, ExitReasonRegistry,
> UniversalSignalRecord), registro de estrategias, contrato del selector y CLI.
> Las **invariantes** viven en `../../rules/strategy-contract.md` (auto-cargada).

---

## 1. StrategyManifest

| Campo | Tipo | Ejemplo | Requerido |
|-------|------|---------|-----------|
| `strategy_id` | string | `"smart_simple_v11"` | Sí |
| `strategy_name` | string | `"Smart Simple v2.0.0"` | Sí |
| `strategy_type` | enum | `ml_supervised`/`rl_ppo`/`hybrid`/`rule_based` | Opcional |
| `timeframe` | enum | `intraday_5m`/`daily`/`weekly` | Opcional |
| `pair` / `version` | string / semver | `"USD/COP"` / `"2.0.0"` | Opcional |

`strategy_id` es la clave universal: localiza stats en `summary.json → strategies[id]`, el archivo
`trades/{id}.json` y el badge de nombre.

## 2. Registro de estrategias

**3 activos · 10 estrategias** publicadas en `public/data/strategies/`, indexadas por
`registry.json`. Métricas anualizadas **por activo** — nunca comparar entre activos.

| Strategy ID | Activo | Tipo | Estado | Runner |
|-------------|--------|------|--------|--------|
| `smart_simple_v11` | usdcop | ml_supervised | **PRODUCTION** | `scripts/pipeline/train_and_export_smart_simple.py` |
| `smart_simple_aggr` | usdcop | ml_supervised | experimental (A/B) | idem |
| `forecast_vt_trailing` | usdcop | ml_supervised | **PAUSED** | `scripts/pipeline/generate_weekly_forecasts.py` |
| `rl_v215b` | usdcop | rl_ppo | DEPRIORITIZED | `scripts/pipeline/run_ssot_pipeline.py` |
| `gold_long_only_b1` · `gold_trend_b2` · `gold_regime_gated_v1` | xauusd | rule_based / hybrid | experimental | `scripts/pipeline/run_gold_pipeline.py` |
| `btc_trend_b2` · `btc_hodl_b1` · `btc_exposure_s3` | btcusdt | rule_based / hybrid | experimental | `scripts/pipeline/run_btc_pipeline.py` |

**Estados**: `PAPER_TRADING` (visible en `/dashboard`+`/production`) · `PRODUCTION` ·
`DEPRIORITIZED` (oculto) · `ARCHIVED` (oculto).

### Añadir una estrategia

1. Script con convención `--phase backtest|production|both`.
2. Registrarla en esta tabla.
3. Exportar JSON conforme a `StrategySummary` + `StrategyTradeFile`.
4. Añadir exit reasons nuevas al registro **en TS y Python**.
5. `--phase backtest` genera `approval_state.json` y arranca el flujo de aprobación.

## 3. TradeRecord

| Campo | Tipo |
|-------|------|
| `trade_id` | int |
| `timestamp` / `exit_timestamp` | ISO8601 (COT -05:00) |
| `side` | `LONG`/`SHORT` |
| `entry_price` / `exit_price` | float |
| `pnl_usd` / `pnl_pct` | float |
| `exit_reason` | string |
| `equity_at_entry` / `equity_at_exit` | float |
| `leverage` | float |

Campos extra específicos de estrategia están permitidos; el dashboard ignora claves desconocidas.

## 4. MetricsSummary

`final_equity`, `total_return_pct`, `sharpe`, `max_dd_pct`, `win_rate_pct`, `profit_factor`,
`trading_days`, `exit_reasons` (dict), `n_long`, `n_short`.

**`profit_factor` es `null` cuando no hay pérdidas — NUNCA `Infinity`.** Cap de display 999.99.

## 5. ExitReasonRegistry

| Reason | Color | Usado por |
|--------|-------|-----------|
| `take_profit` | emerald | Smart Simple, VT+Trail |
| `trailing_stop` | emerald | VT+Trail |
| `hard_stop` | red | ambas |
| `week_end` | blue | Smart Simple |
| `session_close` | blue | VT+Trail |
| `circuit_breaker` | amber | cualquiera |
| `no_bars` | slate | cualquiera |

Razones desconocidas renderizan en gris (no rompen).

## 6. StrategySelector

```typescript
interface StrategyOption {
  strategy_id: string; strategy_name: string;
  status: 'PENDING_APPROVAL' | 'APPROVED' | 'REJECTED' | 'LIVE';
  year: number; return_pct: number;
  has_backtest: boolean; has_production: boolean;
}
```

`GET /api/production/strategies` escanea `public/data/production/` buscando `summary*.json` y
construye la lista dinámicamente. **Sin IDs hardcodeados.**

## 7. UniversalSignalRecord

Desacopla generación de señal de ejecución: cualquier estrategia produce
`UniversalSignalRecord[]` y un único `ReplayBacktestEngine` los ejecuta a `StrategyTrade[]`.

Campos clave: `signal_id`, `strategy_id`, `signal_date`, `direction` (+1/-1/0), `magnitude`,
`confidence`, `skip_trade`, `leverage`, `hard_stop_pct`, `take_profit_pct`,
`trailing_activation_pct`, `trailing_distance_pct`, `entry_price`, `entry_type`, `horizon_bars`,
`bar_frequency`, `metadata`.

| Adapter | Estrategia | Execution strategy |
|---------|-----------|--------------------|
| `H5SmartSimpleAdapter` | smart_simple_v11 | `WeeklyTPHSExecution` |
| `H1ForecastVTAdapter` | forecast_vt_trailing | `DailyTrailingStopExecution` |
| `RLPPOAdapter` | rl_v215b | `IntradaySLTPExecution` |

Almacenamiento: `data/signals/{strategy_id}_{year}.parquet`

```bash
python scripts/pipeline/generate_universal_signals.py --strategy smart_simple_v11 --year 2025
python scripts/pipeline/replay_backtest_universal.py --strategy smart_simple_v11 --year 2025 --export-dashboard
```

## 8. Tests estadísticos

En `summary.json → statistical_tests`: `p_value` (req.), `significant` (req.), `t_stat`,
`bootstrap_95ci_ann`.

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Invariantes de estrategia (auto-cargadas) | `../../rules/strategy-contract.md` |
| Gates de aprobación | `../../rules/approval-gates.md` · `approval-lifecycle.md` |
| Registry dinámico y bundles | `registry-lifecycle.md` |
| Contrato de datos del dashboard | `dashboard-integration.md` |
| Ciencia de las estrategias rule-based | `../assets/_strategy-science.md` |
