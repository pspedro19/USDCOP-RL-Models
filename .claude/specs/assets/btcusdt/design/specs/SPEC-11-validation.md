# SPEC-11 — Validación & Framework de Tests de Hipótesis ★

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | TODAS las specs (las envuelve), CostModel (SPEC-07), labels replay (SPEC-01) |
| Materializa | `validation/` (walk_forward, purge_embargo, dsr, bootstrap, attribution, subera) |
| Pre-registro | Guía v3 §2.2, §4.6, §12; ADR (López de Prado) |
| Es | El motor de backtest y la maquinaria estadística de TODO el sistema |

## 1. Propósito y alcance

Especificar la **maquinaria de validación** que envuelve cada componente: walk-forward con
purga/embargo, cómputo del **Deflated Sharpe Ratio**, intervalos de confianza por bootstrap,
**atribución por componente**, reporte por **sub-era**, análisis de **sensibilidad**, y el
protocolo del backtest del clasificador de eventos (con contaminación). Es el árbitro; ningún
claim de edge es válido sin pasar por aquí.

## 2. Entradas (contrato)

| Entrada | Tipo | Fuente |
|---|---|---|
| Serie de exposición de una estrategia | pd.Series | cualquier SPEC (B1…S5) |
| `returns_1d`, `Close` | pd.Series | `btc_daily` |
| CostModel | fn | SPEC-07 |
| Labels replay de régimen | pd.Series | SPEC-01 (versionados en DVC) |
| `HYPOTHESIS-REGISTRY` | doc | conteo de trials para el DSR |

## 3. Salidas (contrato)

| Salida | Tipo | Descripción |
|---|---|---|
| Métricas OOS | dict | Calmar, Sortino, Sharpe, maxDD, Ulcer, SQN, K-Ratio, turnover + drag |
| `DSR` | float | Sharpe deflactado por `N_trials_total` |
| IC bootstrap | dict | intervalos 95 % de cada métrica y de los deltas entre estrategias |
| Atribución | tabla | aporte en Calmar de cada componente en aislamiento |
| Reporte sub-era | tabla | métricas por pre-2020 / 2020–2023 / 2024+ |
| Grid de sensibilidad | tabla | todas las celdas (no la mejor) |

## 4. Interfaz (API)

```python
def walk_forward(strategy_fn, data, train_win, test_win, step,
                 purge: int, embargo: int) -> list[FoldResult]: ...
def deflated_sharpe(sr_observed: float, n_trials: int, skew: float,
                    kurtosis: float, n_obs: int) -> float: ...
def block_bootstrap_ci(metric_fn, returns, block_len: int, n_boot: int = 10_000) -> CI: ...
def paired_bootstrap_delta(metric_fn, ret_a, ret_b, block_len: int) -> CI: ...
def attribution(core_fn, component_fns, data) -> pd.DataFrame: ...
def subera_report(result, boundaries=("2020-01-01","2024-01-01")) -> pd.DataFrame: ...
```

## 5. Algoritmo / lógica (la maquinaria)

**5.1 Walk-forward con purga + embargo (López de Prado):**
```
para cada fold:
    train = ventana pasada
    PURGA: elimina del train las observaciones cuyas etiquetas se solapan con el test
    EMBARGO: buffer de `embargo` barras entre train y test (≥ H para meta-labeling)
    test  = ventana siguiente (OOS)
    entrena en train, evalúa en test  →  usa labels replay (SPEC-01), NO el modelo final
concatena resultados OOS de todos los folds
```

**5.2 Deflated Sharpe Ratio (la corrección por multiple-testing):**
```
SR_esperado_max ≈ f(N_trials, varianza de SR entre trials)      # Sharpe esperado por azar dado N trials
DSR = Prob( SR_verdadero > 0 | SR_observado, N_trials, skew, kurtosis, n_obs )
# N_trials = N_trials_total del HYPOTHESIS-REGISTRY (componentes + variantes + sensibilidades + descartados)
```
> El DSR es el guardián contra el data-mining. Añadir un experimento sin actualizar
> `N_trials` es una violación de la constitución (§2.2.3).

**5.3 Bootstrap por bloques (ICs que respetan autocorrelación):**
```
block_len ≈ f(autocorrelación de retornos)      # bloques preservan estructura temporal
IC_95 = percentiles [2.5, 97.5] de la métrica sobre 10⁴ remuestreos por bloques
# paired_bootstrap_delta para comparar estrategias (S3 vs B1, combinado vs multiplicado, ...)
```

**5.4 Atribución por componente (§12):**
```
para cada componente c en {z_ciclo, z_funding, M_liquidez, G_eventos}:
    ΔCalmar_c = Calmar(núcleo + c) − Calmar(núcleo)      # en aislamiento
    si ΔCalmar_c no es > 0 con IC 95 % ⇒ el componente SE ELIMINA (y cuenta como trial)
```

**5.5 Reporte por sub-era:** métricas separadas para pre-2020 / 2020–2023 / 2024+; un
parámetro que solo funciona en una era se marca **frágil**.

**5.6 Sensibilidad (declarada, no búsqueda):** se reportan **todas** las celdas de
σ_objetivo {25/30/35 %}, banda {10/12.5/15 %}, pesos {60/40, 70/30, 80/20}. **No se elige la
mejor**; si el edge solo vive en una celda, el sistema es frágil y se rechaza.

**5.7 Backtest del clasificador de eventos (contaminación, §7.8):** corpus tiempo-de-titular;
recall = **cota superior**; estimación insesgada solo post-cutoff en shadow; versión de LLM =
trial.

## 6. Invariantes y post-condiciones

- Ningún backtest usa split ingenuo (siempre walk-forward + purga + embargo).
- Todo backtest paga CostModel.
- Todo claim de edge recomputa el DSR con `N_trials` actual.
- Los labels de régimen son replay como-en-vivo, nunca el modelo final.
- Las sensibilidades se reportan completas (no se cherry-pickea).

## 7. Tests unitarios

- [ ] Purga elimina correctamente las observaciones solapadas (caso sintético).
- [ ] Embargo inserta el buffer exacto entre train y test.
- [ ] `deflated_sharpe`: a más `N_trials`, menor DSR para el mismo SR observado.
- [ ] `block_bootstrap_ci` con serie iid ≈ IC analítico; con serie autocorrelada, más ancho.
- [ ] `attribution` marca para eliminar un componente con ΔCalmar ≤ 0.
- [ ] `subera_report` particiona correctamente en las tres eras.

## 8. Tests de integración

- [ ] Pipeline completo B1…S5 pasa por `walk_forward`; métricas OOS concatenadas coherentes.
- [ ] El DSR del sistema usa el `N_trials_total` del registro (no un número hardcodeado).
- [ ] La atribución reproduce la decisión de incluir/excluir cada gate.
- [ ] Un backtest con split ingenuo es rechazado por el framework (guardarraíl).

## 9. Test de hipótesis (nivel sistema)

**H-SYS-01 — ¿el sistema tiene edge real tras deflación?**
- **H0:** el Sharpe verdadero del sistema es ≤ 0 tras deflación por `N_trials_total`.
- **Estadístico:** **Deflated Sharpe Ratio** (Bailey & López de Prado).
- **Criterio:** DSR > 0 con confianza ≥ 95 %. **Sin esto, no hay go-live.**
- **Corrección múltiple:** el DSR *es* la corrección; `N_trials` incluye todo lo probado.

**Familia de tests de aporte (FDR):** todos los H-*-0x de aporte de componentes se corrigen
por **Benjamini-Hochberg** para controlar la tasa de falso descubrimiento de la familia.

## 10. Protocolo de backtest / validación (meta)

Esta spec **es** el protocolo. Regla de oro: **replay de labels como-en-vivo + purga/embargo
+ CostModel + DSR con conteo completo + sub-eras + sensibilidad completa + live haircut
30–50 %.** Cualquier resultado que no pase por aquí no es válido.

## 11. Criterios de aceptación (DoD)

- [ ] `walk_forward`, `deflated_sharpe`, `block_bootstrap`, `attribution`, `subera_report`
      implementados y testeados.
- [ ] El DSR se computa con `N_trials_total` del registro (sincronía verificada).
- [ ] Guardarraíl contra split ingenuo activo.
- [ ] Reporte de sensibilidad completo (todas las celdas) generado automáticamente.
- [ ] Live haircut aplicado a todo resultado de producción.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.4 selección/multiple-testing (RESUELTO AQUÍ):** DSR con conteo completo + FDR.
- **Capa modelos:** replay de labels (no el modelo final).
- **§7.7 historia corta:** sub-eras + sensibilidad declarada.
- **§7.8 contaminación LLM:** recall como cota superior; insesgado post-cutoff.

## 13. Parámetros pre-registrados

Walk-forward con purga/embargo (embargo ≥ H); block bootstrap 10⁴; DSR con `N_trials_total`;
sub-eras pre-2020/2020–2023/2024+; sensibilidades {σ, banda, pesos} completas; live haircut
30–50 %; α = 0.05 con BH-FDR.
