---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - config/feature_registry.yaml
  - config/experiment_ssot.yaml
  - src/core/contracts/feature_contract.py
  - src/validation/data_quality_gate.py
---

# Auditoría de datos, features y estadística

## Hallazgos medidos

| ID | Hallazgo | Impacto | Estado |
|---|---|---|---|
| FEAT-P0-01 | `feature_registry.yaml` declara 15 dimensiones, pero `experiment_ssot.yaml` declara 20 | consumidor legado puede entrenar/inferir con orden incorrecto | RED |
| FEAT-P0-02 | `DataQualityGate` fija 15 columnas y 13 predictors antiguos | un dataset v3 válido puede fallar o uno legado pasar como producción | RED |
| STAT-P0-01 | gates activos aceptan retorno OOS −20% y Sharpe −3 | “PASS” técnico puede representar destrucción económica | RED |
| STAT-P1-01 | hay DSR, block bootstrap y walk-forward en partes del repo, pero no son campos obligatorios del manifest | la UI/promoción puede mostrar evidencia incompleta | RED |
| DATA-P1-01 | hay controles de NaN, orden, rangos, fechas y duplicados | buena base, falta perfil longitudinal obligatorio por dataset/version | AMBER |

## Métricas que aún deben hacerse obligatorias

Datos: cobertura por fuente, freshness, gaps por calendario, duplicados, revisiones, stale runs,
missingness por régimen, outliers robustos, continuidad OHLC, zona horaria, lineage y checksum.

Features: PSI/KS o detector equivalente, estabilidad de media/varianza/cuantiles, correlación/VIF,
importance stability por fold, tasa de clipping, disponibilidad temporal, fallback/imputación y paridad
offline-online bit a bit o con tolerancia declarada.

Estrategia: retorno neto, alpha contra baseline, Sharpe/Sortino/Calmar, max drawdown y duración,
CVaR, turnover, hit rate, payoff, profit factor, exposición, capacidad, slippage y sensibilidad a costos.
Todos por fold, seed, régimen y ventana, con intervalos de confianza.

Forecast: error contra naive/random-walk, MASE, directional accuracy, Brier/log-loss cuando aplique,
calibración, cobertura y ancho de intervalos, estabilidad por horizonte y utilidad después de costos.

## Corrección recomendada

Generar cualquier registro secundario desde `experiment_ssot.yaml`; eliminar listas de features embebidas.
El manifest de promoción debe exigir evidencia OOS, baseline, DSR/multiplicidad, costos, dataset hash,
feature hash, seeds/folds, limitaciones y ventana de validez.
