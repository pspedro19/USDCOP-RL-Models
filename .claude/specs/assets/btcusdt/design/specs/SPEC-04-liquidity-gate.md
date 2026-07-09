# SPEC-04 — Gate de Liquidez (M_liquidez)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | `btc_daily`: tasas reales, DXY, VIX, ETF flows, stablecoin supply, M2, dominancia |
| Materializa | `strategy/liquidity.py` |
| Pre-registro | Guía v3 §5.3; ADR-0009 (ortogonalidad) |
| Rol | **Multiplicativo** (semi-ortogonal al score interno) |

## 1. Propósito y alcance

Producir `M_liquidez ∈ [0.5, 1.25]`, un multiplicador que refleja si el ciclo tiene **viento
de cola macro**. Multiplica al score interno (SPEC-03) por ser semi-ortogonal. Incluye el
**test de endogeneidad** que recorta ETF flows/stablecoins si resultan ser momentum
disfrazado.

## 2. Entradas (contrato)

| Feature | Peso | Tipo | Fuente | Disponibilidad | Rol |
|---|---|---|---|---|---|
| Tasas reales (TIPS 10Y) | **25 %** | float | FRED | D−1 | Caída = viento de cola |
| DXY | **20 %** | float | FRED/Yahoo | D−1 | Correlación rodante |
| VIX | **15 %** | float | FRED/Yahoo | D−1 | >30 = risk-off |
| ETF flows | **20 %** | float | Farside | **D+1** | Demanda estructural (endógena parcial) |
| Stablecoin supply | **10 %** | float | DefiLlama | D−1 | Pólvora seca (endógena) |
| M2 global (lag) | **5 %** | float | FRED+BCs | D−1 | Contexto lento, bajo peso |
| Dominancia BTC | **5 %** | float | TradingView | D−1 | Apetito intra-cripto |

Pesos **fijos y documentados, NO optimizados**. Cap conjunto ETF+stablecoin = **30 %**.

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Invariante |
|---|---|---|---|
| `M_liquidez_t` | float | **[0.5, 1.25]** | mapeo lineal del score compuesto |
| `liquidity_score_t` | float | ~[−3, 3] | suma ponderada de z-scores rodantes |
| `endogeneity_report` | dict | — | aporte de ETF/stablecoins tras controlar por momentum |

## 4. Interfaz (API)

```python
def m_liquidez(daily_until_t: pd.DataFrame, weights: dict[str, float]) -> float:
    """Suma ponderada de z-scores rodantes → mapeo lineal a [0.5, 1.25]. Solo info ≤ t (ETF: D+1)."""

def endogeneity_check(contrib: pd.Series, price_momentum: pd.Series) -> float:
    """Aporte de ETF/stablecoins a Calmar tras regresión parcial sobre momentum de precio."""
```

## 5. Algoritmo / lógica

```
for f in features:
    z_f_t = zscore(f_t, rolling_window=90..180d)          # correlaciones/betas RODANTES, no fijas
liquidity_score_t = Σ weight_f · z_f_t                     # signos según prior económico
M_liquidez_t = map_linear(liquidity_score_t, out=[0.5, 1.25])   # con bandas + histéresis
```

- **Todas las relaciones son rodantes.** El M2-lead de 10–13 semanas es la relación más
  popularizada y por tanto la más sospechosa de sobreajuste narrativo: contexto lento de bajo
  peso (5 %), jamás timing.
- **Endogeneidad declarada (§5.3):** ETF flows y stablecoins son parcialmente *consecuencia*
  del precio (los flujos persiguen momentum). Por eso: (a) lag D+1 estricto en ETF, (b) peso
  conjunto capeado en 30 %, (c) test de endogeneidad (H-LIQ-02): si su aporte no sobrevive el
  control por momentum, **se recortan**.

## 6. Invariantes y post-condiciones

- `M_liquidez ∈ [0.5, 1.25]` siempre.
- ETF flows nunca se usan antes de D+1 (test de lag I10).
- Σ pesos = 1.0; peso conjunto ETF+stablecoin ≤ 0.30.
- Todas las normalizaciones son rodantes (no-fuga).

## 7. Tests unitarios

- [ ] Score todo-neutro (z=0) ⇒ `M_liquidez` = punto medio del mapeo.
- [ ] Rango: inputs extremos ⇒ `M_liquidez` clip a [0.5, 1.25].
- [ ] Lag ETF: `m_liquidez` en fecha D no usa el flujo publicado en D (solo ≤ D−1 real ⇒ D+1
      disponibilidad).
- [ ] Cap conjunto: ETF+stablecoin no exceden 30 % del score aunque sus z sean enormes.
- [ ] Pesos rodantes: cambiar un dato futuro no altera `M_liquidez` de una fecha pasada.

## 8. Tests de integración

- [ ] En 2024 (era ETF), entradas sostenidas de ETF elevan `M_liquidez` — sanity.
- [ ] En shocks de VIX conocidos (2020-03), `M_liquidez` cae — sanity risk-off.
- [ ] `endogeneity_check` corre sobre el histórico y produce un aporte controlado por momentum.

## 9. Test de hipótesis

**H-LIQ-01 — ¿el gate de liquidez aporta Calmar sobre el núcleo?**
- **H0:** Calmar OOS(núcleo + M_liquidez) ≤ Calmar OOS(núcleo).
- **Estadístico:** bootstrap pareado por bloques sobre `ΔCalmar`.
- **Criterio:** IC 95 % > 0 **y** DSR positivo contándolo como trial. Si no ⇒ **se descarta**
  (default = descartar).

**H-LIQ-02 — ¿el aporte de ETF/stablecoins es momentum disfrazado?**
- **H0:** el aporte de ETF flows/stablecoins a Calmar **desaparece** al controlar por momentum
  de precio.
- **Estadístico:** ΔCalmar del sub-bloque ETF/stablecoin tras regresión parcial sobre momentum
  (o purga de la componente de momentum).
- **Criterio:** si el aporte **sobrevive** el control ⇒ se conserva; si no ⇒ **se recorta** el
  sub-bloque (es beta de momentum ya capturada por vol-targeting).

## 10. Protocolo de backtest / validación

- Aislamiento: núcleo+M_liquidez vs. núcleo, con motor y CostModel completos.
- Sub-eras: ETF flows solo existen 2024+ ⇒ su aporte solo es medible en la última sub-era
  (declarado; peso estadístico fino).
- Sensibilidad de la ventana rodante {90, 120, 180} d reportada.

## 11. Criterios de aceptación (DoD)

- [ ] H-LIQ-01 rechaza H0 (o el gate se descarta honestamente).
- [ ] H-LIQ-02 resuelto (endogeneidad controlada; sub-bloque conservado o recortado).
- [ ] Tests en verde; lag ETF y cap conjunto auditados.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **Capa datos:** correlaciones rodantes, ETF con lag D+1 (test I10).
- **§5.3 endogeneidad:** control por momentum (H-LIQ-02).
- **§7.1 ortogonalidad:** multiplica solo mientras `|ρ(M_interno, M_liquidez)| ≤ 0.4`; si se
  supera, se absorbe al score (SPEC-03) y cuenta como trial.

## 13. Parámetros pre-registrados

Pesos §2 (fijos); rango [0.5, 1.25]; cap conjunto ETF+stablecoin 30 %; ventana rodante
90–180 d; lag ETF D+1. Ninguno se optimiza.
