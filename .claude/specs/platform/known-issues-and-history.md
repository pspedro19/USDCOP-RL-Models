---
kind: as-built
status: IMPLEMENTED
contract: CTR-HISTORY-001
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - src/contracts/strategy_schema.py
  - scripts/data/build_unified_fx_seed.py
---
# SDD Spec: Bugs conocidos e historial de versiones

> **Responsibility**: bugs ya corregidos (para no re-introducirlos) e historial de resultados por
> track. Se movió aquí desde `CLAUDE.md` para bajar el presupuesto de auto-carga —
> es referencia, no una regla que aplique en cada sesión.

---

## 1. Bugs conocidos (ya corregidos — no re-introducir)

| # | Bug | Corrección |
|---|-----|-----------|
| 1 | `Infinity` en JSON — `profit_factor: float("inf")` rompe `JSON.parse()` | `None` + `safe_json_dump()` |
| 2 | `strategy_id` hardcodeado — el dashboard leía `forecast_vt_trailing` pero el export escribía `smart_simple_v11` | lookup dinámico de `strategy_id` |
| 3 | Exit reasons desconocidas rompían la UI | registro universal `EXIT_REASON_COLORS` |
| 4 | Bypass de `min_hold_bars` — CLOSE y reversals deben chequearlo ANTES de ejecutar | check previo |
| 5 | Estados LSTM en backtest — RecurrentPPO requiere `model.predict(obs, state=…, episode_start=…)` | pasar estado |
| 6 | `close_reason` no se propagaba de env → info dict → reward calculator | propagación explícita |
| 7 | `flat_reward_weight` != 0 creaba sesgo HOLD | forzado a 0.0 |
| 8 | `max_drawdown` de backtest a 15% mataba la curva de equity | 99% |
| 9 | `volume_zscore` muerto — el volumen OHLCV es 100% ceros | feature eliminada |
| 10 | Encoding de `dow` con /7.0 (días calendario) | /5.0 (días de trading) |
| 11 | Seed USDCOP con timestamps UTC etiquetados como COT (`tz_localize` en vez de `tz_convert`) | corregido en `build_unified_fx_seed.py` |
| 12 | Barra diaria de Gold corrida un día a domingo (`tz_convert(→ET).normalize()` sobre sello 00:00-UTC) | anclar la fecha en UTC; validador OHLCV como gate |

---

### Abierto: correr la suite reescribe evidencia TRACKEADA (2026-08-06)

Correr `pytest tests/unit tests/regression` deja modificados tres ficheros **versionados**:

    .claude/codex/evidence/acquisition-backups-audit.json
    .claude/codex/evidence/market-data-statistics.json
    .claude/codex/evidence/seed-backup-reconciliation.json

**Diff medido: `1 1` en cada uno — sólo el campo `generated_at`.** El contenido decisorio (los
snapshots, los conteos, las reconciliaciones) queda **idéntico**; lo único que cambia es la marca
de tiempo de regeneración.

**Por qué está aquí y no en CI.** No es un fallo de test: es que **basta ejecutar la suite para
que `git status` salga sucio** sin que nadie haya decidido nada. En este repo eso tiene un coste
concreto y ya cobrado: el árbol sucio es la señal que los agentes usan para detectar trabajo sin
lease, y ha producido **dos falsas alarmas** de protocolo. Además, un timestamp que cambia solo
puede **enmascarar un cambio real de contenido** en el mismo commit.

**Estado**: registrado, no corregido. Los ficheros viven bajo `.claude/codex/**` y su corrección
—escribir sólo si el contenido decisorio cambia, o excluir el campo del artefacto versionado— es
decisión de ese carril, no de éste.

## 2. Historial de versiones (condensado)

| Track | Estrategia | Retorno 2025 | Sharpe | p-value | $10K → | Estado |
|-------|-----------|--------------|--------|---------|--------|--------|
| **H5 Weekly** | **Smart Simple v2.0** | **+25.63%** | **3.35** | **0.006** | **$12,563** | **PRODUCTION** |
| H5 Weekly | v2.0 (2026 YTD) | +0.61% | — | — | $10,061 | Gate activo (1 trade) |
| H1 Daily | Forecast+VT+Trailing | +36.84% | 3.135 | 0.0178 | $13,684 | PAUSED |
| RL | V21.5b | +2.51% | 0.321 | 0.272 | $10,251 | NO significativo |
| Baseline | Buy & Hold | -12.29% | — | — | $8,771 | — |

> **Advertencia metodológica (auditoría 2026-07-06)**: los p-values de 2025 salieron de iterar
> sobre el mismo OOS (grid de 42 celdas, "#8 de 42"). El **DSR trial-aware de v11 = 0.50-0.92 <
> 0.95** en todos los escenarios. El backtest 2025 **no prueba edge** tras selección; v11 está
> CONGELADA y el forward 2026 es el único juez. Ver `../../rules/quant-constitution.md`.

### v1.1 → v2.0 (2026-03-18)

Añadidos: regime gate (Hurst), effective HS (cap 3.5% de portafolio), XGBoost al ensemble,
features `vol_regime_ratio` + `trend_slope_60d`, leverage dinámico, circuit breaker.
Retraining semanal restaurado (era mensual en v1.1.0 — bug metodológico).

La auditoría de 10 agentes reveló **R² < 0 en ambos años**: el alpha del modelo es negativo vs
"siempre SHORT". **El gate es el MVP**: bloqueó 11 de 12 semanas mean-reverting en Q1 2026,
convirtiendo -5.17% en +0.61%.

### Historial RL

V20 falló · V21 falló · V21.5 superada · V22 mixta (1/5) · **V21.5b la mejor (4/5)** ·
EXP-ASYM-001 falló · EXP-HOURLY/DAILY fallaron.
Baselines: buy-and-hold -14.66%, random -4.12%, CI bootstrap [-0.69%, +6.15%].

### Multi-activo (stacks rule-based, NO comparables con COP)

| Activo | Mejor estrategia | Retorno OOS | Sharpe | p-value | Rec |
|--------|------------------|-------------|--------|---------|-----|
| XAU/USD | `gold_trend_b2` (2004→2026) | +55.3% | 0.362 | 0.041 | PROMOTE |
| BTC/USDT | `btc_trend_b2` (2018→2026) | +351% | 1.40 | 0.0 | PROMOTE |

Cada uno anualizado por activo — **nunca comparar entre activos**.

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Disciplina anti-selección (DSR, trials) | `../../rules/quant-constitution.md` |
| Ciencia de las estrategias rule-based | `../assets/_strategy-science.md` |
| Ciclo DS por activo × estrategia | `../assets/_ds-cycle-asbuilt.md` |
| Protocolo de experimentos | `../../rules/experiment-protocol.md` |
