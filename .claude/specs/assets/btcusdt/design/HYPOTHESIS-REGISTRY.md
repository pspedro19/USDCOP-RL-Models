# HYPOTHESIS REGISTRY — Registro de Hipótesis y Trials

> **Insumo directo del Deflated Sharpe Ratio.** El DSR necesita saber **cuántas hipótesis se
> probaron** (incluidas las descartadas) para corregir el Sharpe por selección. Este
> documento es la **cuenta oficial de trials**. Toda hipótesis probada —cada componente en
> aislamiento, cada variante de peso, cada análisis de sensibilidad, cada re-fit del
> clasificador— se registra aquí ANTES de correr su test, con su H0/H1 y su criterio de
> decisión pre-registrado.
>
> Se sincroniza con MLflow (cada fila = un `run` etiquetado). El conteo `N_trials` alimenta
> la fórmula del DSR en SPEC-11 §9.

## Convenciones

- **Estado:** `PENDIENTE` (registrada, no corrida) · `RECHAZA_H0` (aporta) · `NO_RECHAZA`
  (descartada) · `INCONCLUSA`.
- **α global** = 0.05, con corrección de comparaciones múltiples (Benjamini-Hochberg FDR
  sobre la familia de tests de aporte de componentes; ver SPEC-11 §9).
- Cada fila cuenta como **≥ 1 trial** en el DSR aunque termine `NO_RECHAZA`. Las variantes de
  sensibilidad cuentan como trials adicionales.

## Registro

| ID | Spec | Hipótesis nula (H0) | Estadístico / test | Criterio de decisión | N_trials | Estado |
|---|---|---|---|---|---|---|
| H-REG-01 | SPEC-01 | Los regímenes del HMM no se relacionan con el retorno forward ajustado por downside | Kruskal-Wallis sobre Sortino forward por régimen + permutación | p < α tras BH | 1 | PENDIENTE |
| H-REG-02 | SPEC-01 | El re-fit del HMM produce labels inestables (≠ del histórico congelado) | % de labels que cambian entre re-fits consecutivos en solape | **> 20 %** ⇒ inestable, NO entra | 1 | PENDIENTE |
| H-POS-01 | SPEC-02 | El funding extremo no precede a caídas de vol-adjusted return | Diferencia de CVaR forward tras funding p>95 vs. resto (bootstrap) | IC 95 % del delta no cruza 0 | 1 | PENDIENTE |
| H-CMB-01 | SPEC-03 | La combinación en riesgo (R) **no** mejora Calmar OOS vs. multiplicación ingenua ciclo×funding | Bootstrap pareado sobre ΔCalmar OOS | IC 95 % de ΔCalmar > 0 | 1 | PENDIENTE |
| H-CMB-02 | SPEC-03 | z_ciclo y z_funding son ortogonales (no requieren combinación) | \|ρ rodante 90d\| sostenido | **> 0.4** ⇒ combinar (rechaza ortogonalidad) | 1 | PENDIENTE |
| H-LIQ-01 | SPEC-04 | El gate de liquidez no aporta Calmar sobre el núcleo | Bootstrap sobre ΔCalmar (núcleo+M_liq vs núcleo) | IC 95 % > 0 + DSR | 1 | PENDIENTE |
| H-LIQ-02 | SPEC-04 | El aporte de ETF flows/stablecoins es momentum disfrazado | ΔCalmar controlando por momentum de precio (regresión parcial) | Aporte sobrevive control ⇒ se conserva | 1 | PENDIENTE |
| H-EVT-01 | SPEC-05 | El clasificador de eventos no tiene poder de detección (recall = base) | Recall sobre corpus etiquetado (cota superior, §7.8) + FP/año | Recall≈100 % clases catastróficas ∧ FP/año aceptable | 1 | PENDIENTE |
| H-EVT-02 | SPEC-05 | El gate de eventos no mejora Calmar/CVaR (las aplanadas cuestan más de lo que salvan) | PnL diferencial por aplanada (bootstrap) | Suma de PnL diferencial > 0 | 1 | PENDIENTE |
| H-ENG-01 | SPEC-06 | S3 (motor completo de reglas) no supera a B2 en Calmar OOS | Bootstrap pareado ΔCalmar(S3, B2) | IC 95 % > 0, dentro de presupuesto de turnover | 1 | PENDIENTE |
| H-ENG-02 | SPEC-06 | S3 no supera a B1 (HODL vol-targeted) en Calmar OOS | Bootstrap pareado ΔCalmar(S3, B1) | IC 95 % > 0 | 1 | PENDIENTE |
| H-MET-01 | SPEC-08 | El meta-modelo no discrimina señales buenas de malas (AUC = 0.5) | Test de DeLong sobre AUC OOS | AUC > 0.5 con p < α | 1 | PENDIENTE |
| H-MET-02 | SPEC-08 | S4 (con meta-labeling) no supera a S3 en Calmar OOS | Bootstrap pareado ΔCalmar(S4, S3) | IC 95 % > 0 | 1 | PENDIENTE |
| H-RL-01 | SPEC-09 | S5 (con RL táctico) no supera a S4 en Calmar OOS (mediana de seeds) | Bootstrap sobre mediana de ΔCalmar(S5, S4) por seed | IC 95 % > 0 sobre mediana ≥5 seeds | 1 | PENDIENTE |
| H-SYS-01 | SPEC-11 | El sistema final tiene Sharpe ≤ 0 tras deflación | **Deflated Sharpe Ratio** con N_trials de este registro | DSR > 0 con confianza ≥ 95 % | 1 | PENDIENTE |

## Sensibilidades (trials adicionales, pre-registradas)

Cada celda es un trial extra en el DSR. **No se elige la mejor**: se reportan todas (SPEC-11).

| Familia | Celdas | Trials |
|---|---|---|
| σ_objetivo | {25 %, 30 %, 35 %} | 3 |
| Banda de rebalanceo | {10 %, 12.5 %, 15 %} | 3 |
| Pesos R (ciclo/funding) | {60/40, 70/30, 80/20} | 3 |
| Re-versión del clasificador LLM | por versión de modelo | ≥ 1 c/u |

## Conteo para el DSR

```
N_trials_total = Σ(filas del registro principal) + Σ(celdas de sensibilidad) + Σ(re-versiones LLM) + componentes descartados
```

> **Regla dura:** ningún claim de edge (Sharpe/Calmar positivo) es válido sin recomputar el
> DSR con el `N_trials_total` actualizado. Añadir un experimento y no actualizar el conteo es
> una violación de la constitución (§2.2.3).


## Resultado H-POS-01 (2026-07-06, OLA 5 — btc_trend_funding_s4)

S4 = B2 trend × funding brake (k=0.25, floor=0.4, priors ex-ante, `src/btc_strategy/strategies.py`).
Data: 2492 días de funding Binance (`crypto_derivatives_daily`), features causales shift(1).

| Métrica | B2 | S4 |
|---|---|---|
| DSR (N=4 trials) | 0.9987 | 0.9976 |
| OOS-2025 return | −1.37% | **−1.42%** |
| OOS-2025 MaxDD | −8.6% | **−7.3%** |

**Veredicto honesto:** S4 pasa el DSR pero NO cumple el gate pre-registrado (OOS-2025 positivo ∧
bate a B2 en Sharpe+Calmar). El freno por funding reduce levemente el DD pero no aporta retorno
OOS ⇒ **NO se promueve**; se documenta como "funding-brake solo no rescata el año lateral".
Siguiente palanca: z_ciclo on-chain (H-REG-01/H-BTC-CYCLE-02) — requiere extractor on-chain (B3).
Trial añadido: +1 (N=4).
