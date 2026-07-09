# SPEC-10 — Baselines y Escalera de Graduación (B1…S5)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | `btc_daily`, SPEC-01 (régimen), CostModel (SPEC-07) |
| Materializa | `strategy/baselines.py` |
| Pre-registro | Guía v3 §9 |
| Orden | **Se construye en Fase 2, ANTES de todo lo demás.** Sin benchmark no hay gate |

## 1. Propósito y alcance

Definir los **baselines** contra los que se juzga cada escalón del sistema, y la **escalera
de graduación** con sus gates. B1 (HODL vol-targeted) es el rival brutal; el objetivo del
sistema es ganarle en **Calmar**, no en retorno absoluto.

## 2. Entradas (contrato)

| Entrada | Tipo | Fuente |
|---|---|---|
| `returns_1d`, `Close` | pd.Series | `btc_daily` |
| `regime`, `z_ciclo` | — | SPEC-01 (para B2) |
| CostModel | fn | SPEC-07 (todos los baselines pagan costos) |

## 3. Salidas (contrato)

| Salida | Tipo | Descripción |
|---|---|---|
| Curvas de equity | pd.Series × 5 | B1, B2, S3, S4, S5 |
| Métricas | dict × 5 | Calmar, Sortino, Sharpe, maxDD, turnover, por sub-era |

## 4. Interfaz (API)

```python
def b1_hodl_voltarget(returns: pd.Series, sigma_target: float = 0.30) -> pd.Series:
    """HODL con vol-targeting downside. El baseline brutal."""
def b2_cycle_rules(daily: pd.DataFrame, z_ciclo: pd.Series) -> pd.Series:
    """Exposición escalada por umbrales simples de MVRV-Z/NUPL. Smart-beta on-chain."""
# S3 = SPEC-06 ; S4 = SPEC-08 ; S5 = SPEC-09
```

## 5. Algoritmo / lógica (la escalera)

```
B1 — HODL/DCA vol-targeted     : exposure = vol_target_scalar (SPEC-06 §5), sin gates
B2 — Reglas de ciclo sin ML    : exposure escalada por umbrales de MVRV-Z/NUPL (sin HMM, sin funding)
S3 — Motor completo de reglas  : SPEC-06 (capas 1–4, sin ML supervisado)
S4 — S3 + meta-labeling        : SPEC-08
S5 — S4 + RL táctico           : SPEC-09  (opcional)
```

**Gate de cada escalón:** ganarle al anterior **OOS en Calmar y Sortino**, con IC por
bootstrap, DSR (registro de trials), mediana multi-seed donde aplique, atribución por
componente, **dentro del presupuesto de turnover**. El escalón que no supera al anterior **se
descarta** y el sistema se queda en el previo.

## 6. Invariantes y post-condiciones

- Todos los baselines pagan CostModel (comparación justa).
- Todos usan el mismo esquema de validación (walk-forward, sub-eras).
- Todas las exposiciones ∈ [0, 1] (spot-only, comparabilidad).

## 7. Tests unitarios

- [ ] B1 con σ objetivo alto ⇒ exposición cercana a 1 (poco recorte).
- [ ] B2 en euforia (MVRV-Z alto) ⇒ exposición reducida; en capitulación ⇒ elevada.
- [ ] Todos los baselines producen curvas de equity finitas y exposiciones en [0,1].

## 8. Tests de integración

- [ ] Las 5 curvas se computan sobre el mismo histórico con el mismo CostModel.
- [ ] Métricas por sub-era coherentes; B1 domina en retorno absoluto en el bull (sanity).

## 9. Test de hipótesis

Los gates de la escalera **son** las hipótesis H-ENG-01/02 (S3 vs B2/B1), H-MET-02 (S4 vs S3),
H-RL-01 (S5 vs S4). Esta spec provee los baselines; los tests viven en las specs respectivas
y en el registro de hipótesis.

**Marco de decisión de la escalera:**
```
if not (S3 > B2 en Calmar OOS):        el motor no aporta sobre reglas simples → revisar
if not (S3 > B1 en Calmar OOS):        no se le gana a HODL ajustado por riesgo → honesto, quizá quedarse en B1
if not (S4 > S3):                      quedarse en S3
if not (mediana S5 > S4):              quedarse en S4  (lo más probable, §4.1)
```

## 10. Protocolo de backtest / validación

- Walk-forward con purga/embargo, CostModel, sub-eras, sensibilidades — idéntico para todos.
- **Live haircut 30–50 %** aplicado a todas las curvas antes de comparar en términos de
  producción.

## 11. Criterios de aceptación (DoD)

- [ ] B1 y B2 implementados y validados **antes** de construir S3+ (Fase 2).
- [ ] Las 5 curvas comparables (mismo histórico, mismos costos, misma validación).
- [ ] Aceptado que el resultado honesto puede ser **S3** (o incluso **B1**) como estrategia
      final.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.4 selección:** cada escalón es un trial en el DSR.
- **Comparación honesta:** todos pagan costos y usan la misma validación; live haircut.
- **Baseline brutal:** B1 es HODL bien gestionado, no un hombre de paja.

## 13. Parámetros pre-registrados

B1: σ_objetivo 30 % (SPEC-06). B2: umbrales de ciclo de prior (SPEC-01 §5). Gates de
graduación: Calmar/Sortino OOS + bootstrap + DSR + presupuesto de turnover. Live haircut
30–50 %.
