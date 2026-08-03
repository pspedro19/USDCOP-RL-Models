---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - config/experiment_ssot.yaml
  - src/validation/data_quality_gate.py
  - src/contracts/strategy_manifest.py
  - .claude/codex/harness/tests/test_assurance_contracts.py
---

# Ciclo de validación checkpoint por checkpoint

Una estrategia no avanza por tener un backtest atractivo. Cada checkpoint produce un artefacto
versionado, assertions reproducibles y una decisión `PASS`, `FAIL` o `WAIVER` con propietario y caducidad.

| CP | Objetivo | Gate mínimo | Evidencia obligatoria |
|---|---|---|---|
| 0. Hipótesis | Evitar HARKing y experimentación sin límite | hipótesis, mecanismo, métrica primaria, baseline, universo, horizonte y regla de abandono pre-registrados | ficha firmada + ID inmutable |
| 1. Ingesta | Datos auténticos y completos | esquema, zona horaria, calendario, OHLC, duplicados=0, gaps dentro del SLA, fuente y checksum | perfil por partición + lineage |
| 2. Descriptiva | Entender la muestra antes de modelar | N, cobertura, missingness, cuantiles, MAD/IQR, skew, kurtosis, ACF, cambios de régimen | reporte train/validation/test separado |
| 3. Features | Causalidad y paridad | `available_at <= decision_at`, orden/hash/dtype iguales, warm-up explícito, invariancia batch/online | parity + leakage + metamorphic tests |
| 4. Selección | Controlar multiplicidad | selección solo en train/validation; FDR/DSR y número efectivo de trials | ledger de todos los trials, incluidos fallidos |
| 5. Validación | Generalización temporal | walk-forward purgado/embargo; OOS intacto; bootstrap por bloques; benchmark honesto | resultados por fold/seed/regime |
| 6. Economía | Rentabilidad ejecutable | retorno y Sharpe OOS positivos; CI de alpha; costos 1x/2x/3x; turnover, capacidad y latencia | curvas netas, trades y sensibilidad |
| 7. Robustez | Evitar un óptimo frágil | estabilidad paramétrica, perturbación de datos, seeds, subperiodos, crisis y activos correlacionados | heatmaps y distribución, no solo mejor punto |
| 8. Paper | Probar el sistema real | shadow/paper 30–90 días, reconciliación, stale-data, restart, kill switch, límites de riesgo | eventos y SLO operativos |
| 9. Promoción | Decisión independiente | manifest completo; aprobación dual; artefactos firmados; rollback probado | propuesta + hashes + auditoría |
| 10. Producción | Control continuo | drift, PnL attribution, slippage, exposición, latencia, error budget y circuit breakers | dashboard + alertas ensayadas |
| 11. Retiro | Contener deterioro | triggers objetivos, revocación, posiciones abiertas y retención de evidencia | runbook de rollback/retirement |

## Reglas estadísticas de decisión

- La unidad de remuestreo debe preservar dependencia temporal; usar moving/stationary block bootstrap, no IID.
- Reportar tamaño de efecto e intervalo de confianza junto al p-value. Un p-value aislado no promueve.
- Corregir selección múltiple con DSR/FDR y registrar el número efectivo de estrategias probadas.
- Separar intervalos de selección, calibración y prueba final. El OOS final se consume una sola vez.
- Para forecasts: MAE/RMSE/MASE, directional accuracy con baseline, calibración, cobertura/ancho de intervalos y pérdida económica neta.
- Para RL: distribución por seed, estabilidad de política, action imbalance, reward hacking, constraint violations y desempeño de una política nula.

## Regla de parada

Un checkpoint rojo impide promoción. Un waiver no transforma el gate en verde: exige riesgo residual,
compensación, responsable, fecha de expiración y aprobación independiente.
