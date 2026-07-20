---
kind: rule
status: IMPLEMENTED
contract: CTR-STRATEGY-SCHEMA-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - usdcop-trading-dashboard/lib/contracts/strategy.contract.ts
---
# Rule: Contrato universal de estrategia

> **SSOT de las invariantes de estrategia.** Schemas completos, registro, adapters, CLI:
> `../specs/platform/strategy-schemas.md`.

## Invariantes

1. **`strategy_id` es la clave universal.** Determina `trades/{id}.json`,
   `summary.json → strategies[id]` y el badge. **Nunca hardcodear un id en el dashboard** —
   resolver dinámicamente desde `summary.strategy_id`.
2. **`profit_factor` es `null` cuando no hay pérdidas.** JAMÁS `Infinity`. Ningún JSON exportado
   puede contener `Infinity`, `NaN` ni `undefined` — usar `safe_json_dump()`.
3. **Una exit reason nueva se añade en LOS DOS contratos**: `strategy.contract.ts` y
   `strategy_schema.py`. Las desconocidas renderizan en gris, no rompen.
4. **Toda estrategia expone `--phase backtest|production|both`** y exporta JSON conforme a
   `StrategySummary` + `StrategyTradeFile`.
5. **Métricas anualizadas por activo.** Gold/BTC/COP corren en relojes distintos: **nunca
   compararlos en una misma tabla de ranking**.
6. **Con N < 20 trades no se reportan Sharpe ni p-value** — solo conteo y PnL
   (ver `quant-constitution.md` §6).
7. **La señal se desacopla de la ejecución**: las estrategias producen `UniversalSignalRecord[]`
   y un único motor de replay los ejecuta.

## DO NOT

- Do NOT hardcodear `strategy_id` en el dashboard — usa el lookup dinámico.
- Do NOT emitir `Infinity`/`NaN`/`undefined` en JSON — `null` vía `safe_json_dump()`.
- Do NOT añadir exit reasons en un solo lenguaje.
- Do NOT comparar métricas entre activos como si compartieran reloj.
- Do NOT reportar Sharpe/p-value con N<20 trades.
- Do NOT publicar una estrategia sin bundle en el registry (es la vía de entrada al dashboard).
