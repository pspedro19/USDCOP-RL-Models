# SPEC-09 — Validación y Backtest

## Propósito
Protocolo de validación honesto: walk-forward con purga/embargo, métricas ajustadas por riesgo con IC por bootstrap, **atribución por régimen**, Deflated Sharpe con conteo real de trials, y el gate de decisión contra los dos baselines.

## Walk-forward con purga y embargo (López de Prado)
- **Nunca un solo split.** Ventana expansiva o rodante; reportar **solo OOS**.
- **Purge:** eliminar del train las muestras cuyo horizonte de etiqueta se solapa con el set de test.
- **Embargo ≥ lookback máximo:** zona muerta tras el test. Regla concreta: embargo ≥ `horizonte_de_recompensa + lookback_LSTM`; y para features lentas (Hurst a 250d) **congelarlas** o extender el embargo a su ventana. Sin esto, autocorrelación filtra futuro.

```python
def make_walkforward_folds(index, train_span, test_span, embargo, step) -> list[Fold]: ...
# cada Fold: {train_slice, test_slice, embargo_slice, fold_id}
```

## Métricas (por fold OOS, agregadas mediana/IQR sobre seeds)
Globales: **CAGR, Sharpe, Sortino, Max Drawdown, Calmar, win rate, trade promedio, turnover, exposición media**.

## Atribución por régimen (el diagnóstico más informativo)
Desglosar **PnL, Sharpe y Drawdown por régimen Daily** (SPEC-04). Si todo el alpha vive en un solo régimen (típicamente TREND), lo que tienes es un trend-follower caro — y B2 te lo confirma. Este desglose decide si el sistema es robusto o un one-trick pony.

```python
def attribution_by_regime(trades, regime_labels) -> pd.DataFrame:
    """PnL, Sharpe, DD, #trades por régimen."""
```

## Blindaje estadístico
- **Deflated Sharpe Ratio / Probabilistic Sharpe** (López de Prado): corrige por multiple testing. **Necesita el número REAL de trials** → tomarlo de MLflow (SPEC-08), contando cada configuración probada. Sin registro, se subestima y el DSR es teatro.
- **Block bootstrap** (sobre retornos diarios, bloque ~5–20 días para preservar autocorrelación) → **IC de Sharpe y CAGR**. Un punto estimado sin intervalo no dice nada con 2–3 años de OOS. Usar `arch.bootstrap` o implementación propia.
- Comparar **siempre** contra B1 y B2 en las mismas ventanas.

```python
def deflated_sharpe(sr, n_trials, T, skew, kurt) -> float: ...
def block_bootstrap_ci(returns, stat_fn, block=10, n=10_000) -> tuple[float,float]: ...
```

## Red flags (automatizar chequeos donde se pueda)
- Sharpe > 4–5 → casi siempre bug (look-ahead, fills irreales, in-sample). **Falla el reporte con warning fuerte.**
- Drawdown ridículamente bajo → máquina de dinero gratis = no existe.
- Equity curve demasiado suave (test de "suavidad" anómala).
- Resultados hipersensibles a costos o al seed (IQR de seeds enorme).
- PnL concentrado en un régimen o en 3–4 trades (de la atribución).
- Referencia de cordura: Sharpe ~3.35 del sistema COP es lo *creíble*; mucho más en oro, sospechar.

## Gate de decisión (Fase 6)
La **mediana de ≥5 seeds** del agente debe superar a **B1 y B2** OOS, ajustado por riesgo (Sharpe/Sortino/Calmar con IC por bootstrap no solapados). Si no → el mejor baseline ES la estrategia; iterar features/reward antes de añadir complejidad (v2 ensemble, SPEC estrategia §5).

## Salida
- `reports/walkforward_{run_id}.html`: métricas por fold, agregados, atribución por régimen, DSR, ICs, comparación baselines, checklist de red flags.
- Todo loggeado en MLflow.

## Publicación al registro (contrato de salida — ver SPEC-12)
La validación NO termina en el HTML: si pasa el gate, la **tarea final publica un bundle** vía `register_bundle` (`airflow/dags/utils/register_bundle.py`). Reglas:
- **Inmutabilidad + versionado (keystone):** escribe `public/data/strategies/<sid>/backtests/<model_version>/summary_<year>.json` + `trades_<year>.json` + `signals_<year>.parquet`. **NUNCA sobreescribe** `summary_<year>.json` — así v1 y v2 del mismo año coexisten y son **replayables** (`model_version` = config congelada de SPEC-08). Migrar el export legacy que sí sobreescribe.
- **Gate atómico:** valida JSON-safe (sin Inf/NaN), gates completos (Vote 1/2), `signals` parquet presente si `capabilities.replay`, lineage `feature_hash==norm_stats_hash`; luego upsert de `manifest.json` + `registry.json`. Si algo falla, NO toca `registry.json`.
- **Aditivo:** los archivos legacy se siguen escribiendo en paralelo; no se rompe ningún consumo del front.
- Tests **R3** (v2 no modifica `backtests/1.x/**`), **R5** (bundle inválido no entra), **R9** (`register_bundle` corre último, idempotente, con provenance).

## Criterios de aceptación
- [ ] Folds con purga+embargo correctos (test: ninguna muestra de train solapa test/embargo).
- [ ] Métricas calculadas por fold y agregadas mediana/IQR sobre seeds.
- [ ] Atribución por régimen implementada y en el reporte.
- [ ] Deflated Sharpe usando conteo de trials de MLflow.
- [ ] Block bootstrap CI para Sharpe y CAGR.
- [ ] Chequeos de red flags automáticos que marcan el reporte.
- [ ] Comparación contra B1 y B2 en el mismo reporte; gate evaluado explícitamente.

## Dependencias
SPEC-05/06 (entorno/riesgo), SPEC-07 (baselines), SPEC-08 (modelos, trials MLflow).
