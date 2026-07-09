# PRE-REGISTRATION — Parámetros Congelados del Sistema

> **La "constitución numérica".** Todos los parámetros del sistema se fijan **aquí, por
> prior económico, ANTES de ver un solo resultado OOS**. Cambiar cualquiera de estos valores
> mirando resultados de backtest es feature selection disfrazada e **infla el Sharpe por
> selección**. Cada cambio a esta tabla **cuenta como un trial en el Deflated Sharpe Ratio**
> (ver `HYPOTHESIS-REGISTRY.md`).
>
> Materializado en `config/*.yaml`. Este documento es la SSOT humana; el YAML es la SSOT de
> máquina. Deben coincidir (test de consistencia en SPEC-11).

Fecha de congelamiento: **antes del primer backtest OOS** (Fase 3 del roadmap).

---

## 1. Instrumento y universo (ADR-0008)

| Parámetro | Valor | Fuente del prior |
|---|---|---|
| Instrumento | **Spot-only** BTC/USDT–USDC | Elimina liquidación por diseño (§5.0) |
| Rango de exposición | **[0, 1]** (sin apalancamiento, sin cortos) | Spot-only |
| Barra canónica | Cierre **UTC 00:00** | SSOT de datos (§3.2) |
| Perpetuos | Solo fuente de datos (funding, OI, basis) | ADR-0008 |

## 2. Vol-targeting (SPEC-06)

| Parámetro | Valor | Prior |
|---|---|---|
| σ_objetivo | **30 % anualizada** | ~⅓–½ de la vol realizada típica de BTC (60–100 %) |
| Estimador de vol | **Semi-desviación downside**, EWMA | Las caídas de BTC son más rápidas que las subidas |
| Span EWMA | **30 días** | Ventana de vol de corto plazo estándar |
| Anualización | **√365** | Cripto es 24/7 |
| scalar (rango) | **[0.1, 1.0]** | Piso evita división patológica; techo = spot-only |

## 3. Score interno de riesgo (SPEC-03 / §6.2)

| Parámetro | Valor | Prior |
|---|---|---|
| Fórmula | **R = 0.7·z_ciclo + 0.3·z_funding** | El ciclo manda (diferenciador BTC, meses); funding matiza (días) |
| Mapeo | **M_interno = 0.25 + 0.75 / (1 + exp(a·(R − b)))** | Sigmoide decreciente: satura extremos |
| Pendiente a | **1.5** | Transición suave sin saltos |
| Centro b | **0.5** | Con R≈0 → M≈0.85, levemente defensivo por defecto |
| Piso M_interno | **0.25** | Nunca flat por valoración sola (flat = solo eventos+breaker) |
| Techo M_interno | **1.0** | Spot-only; nunca apalancar en capitulación (§7.6) |
| z_ciclo | promedio z-scores rodantes **4a** de {MVRV-Z, NUPL, Puell} | Métricas de ciclo, ventana multi-anual |
| z_funding | z-score del percentil rodante **1a** del funding, con signo | Señal rápida de posicionamiento |

## 4. Regla de ortogonalidad / combinación (SPEC-03, ADR-0009)

| Parámetro | Valor | Prior |
|---|---|---|
| Umbral de correlación | **\|ρ rodante 90d\| > 0.4 sostenido** (>50 % de los días) | Encima de 0.4 hay doble conteo material |
| Acción si se supera | Combinar aditivamente en riesgo (no multiplicar) | §7.1 |
| Revisión | Trimestral | Detecta deriva estructural de correlaciones |

## 5. Gate de liquidez (SPEC-04)

| Parámetro | Valor |
|---|---|
| Rango M_liquidez | **[0.5, 1.25]** |
| Pesos (fijos, NO optimizados) | tasas reales 25 %, DXY 20 %, VIX 15 %, ETF flows 20 %, stablecoin 10 %, M2-lag 5 %, dominancia 5 % |
| Cap conjunto ETF+stablecoin | **30 %** (endogeneidad, §5.3) |
| Ventana de correlación rodante | 90–180 d |
| Lag de ETF flows | **D+1** (publican al cierre US) |

## 6. Gate de eventos (SPEC-05)

| Clase | Acción | Timer |
|---|---|---|
| EXCHANGE_FAILURE / STABLECOIN_STRESS | 1.ª señal G=0.5 → confirmación G=0 | Re-entrada manual |
| REGULATORY_SHOCK | G=0.25 | 48 h |
| MACRO_SURPRISE | G=0.5 | 24 h |
| ETF_STRUCTURAL | G=0.5 | 12 h |
| WHALE_MOVEMENT | G=0.75 | 12 h |
| NOISE | G=1.0 | — |
| Vol-spike breaker | ATR(H1) > **3× EWMA(ATR)** ⇒ G=min(G, 0.5) | Independiente del LLM |
| Objetivo de recall (clases catastróficas) | **~100 %** (acepta baja precisión) | §7.5 |
| Confirmación de flat | 2 fuentes en 15 min, o 1 fuente + spike de vol | §7.5 |

## 7. Ejecución y CostModel (SPEC-07)

| Parámetro | Valor |
|---|---|
| Fee | **10 bps por lado** (taker spot) |
| ½ spread | **~1 bp** (BTC/USDT, prior) |
| Slippage | **2 bps** + impacto k·(orden/vol_barra) si orden > 0.1 % del vol |
| Banda de rebalanceo | **±12.5 % NAV** |
| Mínimo entre rebalanceos ordinarios | **24 h** (los eventos/breaker saltan la banda) |
| Tamaño mínimo de ajuste | **2 % NAV** |
| Presupuesto de turnover | drag de costos **< 25 %** del exceso de Calmar sobre B1 |

## 8. Meta-labeling (SPEC-08)

| Parámetro | Valor |
|---|---|
| Evento etiquetable | Cruce de banda de rebalanceo (±12.5 % NAV) |
| Horizonte H | **20 días hábiles** (o siguiente evento si llega antes) |
| Etiqueta | ΔSortino realizado (post-costos) con ajuste vs. contrafactual sin ajuste |
| Umbral de probabilidad | **p < 0.45 ⇒ posponer el rebalanceo** |
| Rol | **Solo frena** trades; nunca crea ni amplía |
| Embargo | **≥ H** (para no filtrar la etiqueta) |

## 9. RL táctico (SPEC-09)

| Parámetro | Valor |
|---|---|
| Arquitectura | LSTM (encoder) → PPO (RecurrentPPO) |
| Acción | Discreta {−1, 0, +1} como desviación táctica |
| Desviación máxima | **±10 % NAV** respecto a la exposición del motor |
| Recompensa | Sortino diferencial post-costos |
| Semillas | **≥ 5**, se reporta mediana e IQR |

## 10. Cap de portafolio (SPEC-06 / §7.6)

| Parámetro | Valor |
|---|---|
| Método | **Kelly fraccional ≤ ¼ Kelly** sobre supuestos pesimistas |
| Edge asumido | mitad del exceso de Calmar del backtest **post-haircut** |
| Escenario de cola | pérdida total con probabilidad no nula incluida |
| Inmutabilidad | **Ningún estado de los gates lo modifica.** Fijado ex-ante |

## 11. Protocolo de retiro (SPEC-12 / §14)

| Condición | Umbral | Acción |
|---|---|---|
| Suspensión — drawdown | DD en vivo > **1.25 ×** max DD del backtest OOS | Exposición → 0 |
| Suspensión — divergencia | shadow-vs-real > **2 % NAV** acumulado en 30 d | Exposición → 0 |
| Suspensión — drift | PSI > **0.25** sostenido 2 semanas | Exposición → 0 |
| Suspensión — datos | QA rojo sin resolver en **48 h** | Exposición → 0 |
| Decaimiento | Calmar 12m < B1 por **2 trimestres** | Media exposición + revisión |
| Muerte | 3 suspensiones en 12 m, **o** Calmar 12m < B1 por **4 trimestres**, o quiebre estructural | Capital → B1/cash; exige v4 |

## 12. Validación (SPEC-11)

| Parámetro | Valor |
|---|---|
| Esquema | Walk-forward con purga + embargo |
| Live haircut | **30–50 %** |
| Sub-eras | pre-2020 / 2020–2023 / 2024+ |
| Sensibilidades reportadas (no búsquedas) | σ_objetivo {25/30/35 %}, banda {10/12.5/15 %}, pesos {60/40, 70/30, 80/20} |
| Métrica primaria de graduación | **Calmar** (y Sortino) |
| DSR | cuenta **todos** los trials, incluidos descartados |

> **Regla final:** nada de esta tabla se re-optimiza mirando resultados. Las sensibilidades
> se reportan completas; si el resultado solo vive en una celda, el sistema es frágil y se
> rechaza.
