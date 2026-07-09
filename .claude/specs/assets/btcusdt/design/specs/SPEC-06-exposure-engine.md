# SPEC-06 — Motor de Exposición (vol-targeting + integración + bandas + cap)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | SPEC-03 (`M_interno`), SPEC-04 (`M_liquidez`), SPEC-05 (`G_eventos`), `btc_daily` (retornos) |
| Materializa | `strategy/engine.py` |
| Pre-registro | Guía v3 §5.1, §6.1, §7.2, §7.3, §7.6 |
| Es | **S3** (el sistema completo de reglas, sin ML) |

## 1. Propósito y alcance

Integrar todos los gates en la **exposición final ∈ [0, 1]**: computar el vol-targeting
downside, multiplicar por los gates según la regla de combinación, aplicar bandas de
rebalanceo, circuit breakers y el **cap de portafolio Kelly-fraccional**. Es el corazón
determinista del sistema (**S3**).

## 2. Entradas (contrato)

| Entrada | Tipo | Rango | Fuente |
|---|---|---|---|
| `returns_1d` | pd.Series | ℝ | `btc_daily` (para σ downside) |
| `M_interno_t` | float | [0.25, 1.0] | SPEC-03 |
| `M_liquidez_t` | float | [0.5, 1.25] | SPEC-04 |
| `G_eventos_t` | float | {1, .5, .25, 0} | SPEC-05 |
| `exposure_prev` | float | [0, 1] | estado (para bandas) |
| `nav`, `portfolio_nw` | float | > 0 | para el cap de portafolio |

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Invariante |
|---|---|---|---|
| `exposure_target_t` | float | **[0, 1]** | tras gates + cap |
| `exposure_final_t` | float | **[0, 1]** | tras bandas (puede = `exposure_prev` si no se cruzó banda) |
| `rebalance_flag` | bool | — | `True` solo si `|target − prev| > 12.5 %` y ≥ 24 h y ≥ 2 % NAV |
| `vol_scalar_t` | float | [0.1, 1.0] | del vol-targeting |

## 4. Interfaz (API)

```python
def vol_target_scalar(returns: pd.Series, sigma_target: float = 0.30,
                      span: int = 30) -> float:
    """σ_down = sqrt(EWMA(min(r,0)², span))·sqrt(365); scalar = clip(σ_target/σ_down, 0.1, 1.0)."""

def compute_exposure(vol_scalar: float, m_interno: float, m_liquidez: float,
                     g_eventos: float, portfolio_cap: float) -> float:
    """exposure = clip(vol_scalar · m_interno · m_liquidez · g_eventos, 0, min(1.0, portfolio_cap))."""

def apply_rebalance_band(target: float, prev: float, band: float = 0.125,
                         min_hours: int = 24, min_size: float = 0.02,
                         breaker: bool = False) -> tuple[float, bool]:
    """Escalón: solo mueve si |target-prev|>band y pasó min_hours y min_size. Breaker salta la banda."""
```

## 5. Algoritmo / lógica

```
# 1) Vol-targeting downside (§5.1)
σ_down_t   = sqrt( EWMA( min(r_i, 0)² , span=30 ) ) · sqrt(365)          # semi-desv anualizada
vol_scalar = clip( 0.30 / σ_down_t , 0.1 , 1.0 )                          # σ_objetivo=30% (prior)

# 2) Integración de gates (regla de combinación, §6.1)
#    M_interno ya combinó ciclo+funding EN RIESGO (SPEC-03).
#    Liquidez y eventos MULTIPLICAN (ortogonales).
raw_target = vol_scalar · M_interno · M_liquidez · G_eventos

# 3) Cap de portafolio (§7.6) — INMUNE a los gates
portfolio_cap = kelly_fraccional_cap(...)                                 # ≤ ¼ Kelly, ex-ante
exposure_target = clip( raw_target , 0 , min(1.0, portfolio_cap) )

# 4) Bandas de rebalanceo (§7.3) — escalón, no continuo
exposure_final, rebalance = apply_rebalance_band(
    exposure_target, exposure_prev, band=0.125, min_hours=24, min_size=0.02,
    breaker=(G_eventos < 1.0 or vol_spike))                               # protección salta la banda
```

**Cap de portafolio Kelly-fraccional (§7.6, la cola que ningún gate cubre):**
```
edge_pesimista = 0.5 · (exceso de Calmar del backtest POST-haircut)      # supuesto conservador
incluir escenario de pérdida total con probabilidad no nula
portfolio_cap = min( ¼ · kelly(edge_pesimista, ...) , tope_absoluto_patrimonio )
# Fijado EX-ANTE. NINGÚN estado de los gates lo modifica.
```

## 6. Invariantes y post-condiciones

- `exposure_final ∈ [0, 1]` SIEMPRE (spot-only; §7.2 resuelto por diseño).
- `exposure_final ≤ portfolio_cap` SIEMPRE (cap inmune a gates; §7.6).
- Sin rebalanceo si `|target − prev| ≤ 12.5 %` (salvo breaker).
- Un evento `G < 1` o vol-spike **salta** la banda y el mínimo de 24 h (la protección no
  espera).
- `vol_scalar ∈ [0.1, 1.0]`.

## 7. Tests unitarios

- [ ] σ_down de retornos todos positivos → pequeña (solo downside cuenta) ⇒ scalar alto (clip
      a 1.0).
- [ ] `compute_exposure` con `G=0` ⇒ 0 exacto.
- [ ] Rango: 10⁴ combinaciones ⇒ exposure ∈ [0, 1].
- [ ] Cap: si `portfolio_cap = 0.5`, exposure nunca > 0.5 aunque los gates pidan 1.0.
- [ ] Banda: `|target − prev| = 10 %` ⇒ no rebalancea; `= 15 %` ⇒ rebalancea.
- [ ] Breaker salta banda: `G=0.5` con delta 5 % ⇒ rebalancea igual.
- [ ] Mínimo 24 h respetado para rebalanceos ordinarios.

## 8. Tests de integración

- [ ] Cadena completa SPEC-01→06 sobre el histórico ⇒ serie de exposure ∈ [0,1] sin violar
      invariantes.
- [ ] En euforia+funding alto+risk-off: exposure cae fuerte (todos los gates alinean, pero sin
      doble conteo por SPEC-03).
- [ ] En capitulación+funding negativo+viento macro: exposure sube hacia 1.0, **nunca >1**.
- [ ] Turnover anualizado dentro del presupuesto (§5.7) con las bandas activas.

## 9. Test de hipótesis

**H-ENG-01 — ¿S3 le gana a B2 (reglas de ciclo) en Calmar OOS?**
- **H0:** Calmar OOS(S3) ≤ Calmar OOS(B2).
- **Estadístico:** bootstrap pareado por bloques sobre `ΔCalmar(S3, B2)`.
- **Criterio:** IC 95 % > 0 **y dentro del presupuesto de turnover**.

**H-ENG-02 — ¿S3 le gana a B1 (HODL vol-targeted) en Calmar OOS?**
- **H0:** Calmar OOS(S3) ≤ Calmar OOS(B1). *(El baseline brutal.)*
- **Estadístico:** bootstrap pareado; también Sortino y max drawdown.
- **Criterio:** IC 95 % > 0. **Es honesto y aceptable no rechazar** — significaría que S3 no
  supera a HODL ajustado por riesgo y que hay que reconsiderar (o quedarse en B1).

## 10. Protocolo de backtest / validación

- Motor completo con CostModel (SPEC-07), labels replay (SPEC-01), walk-forward purga/embargo.
- **Atribución por componente:** núcleo, núcleo+z_ciclo, +z_funding, +M_liquidez, +G_eventos —
  cada aporte medido y contado como trial (SPEC-11).
- Reporte por sub-era; sensibilidades (σ_objetivo, banda) completas.

## 11. Criterios de aceptación (DoD)

- [ ] Invariante `exposure ∈ [0,1]` y cap inmune verificados sobre todo el histórico.
- [ ] **H-ENG-01 rechaza H0** (S3 > B2). Si no, S3 no aporta sobre reglas simples.
- [ ] H-ENG-02 evaluado y reportado honestamente (gane o no a B1).
- [ ] Turnover dentro del presupuesto.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.2 apalancamiento/ruina:** spot-only, exposure ∈ [0,1], σ downside. Resuelto por diseño.
- **§7.3 turnover:** bandas + mínimo 24 h + tamaño mínimo + presupuesto.
- **§7.6 supervivencia:** cap de portafolio Kelly-fraccional inmune a gates.

## 13. Parámetros pre-registrados

σ_objetivo 30 %, downside EWMA span 30d, √365; scalar [0.1, 1.0]; banda ±12.5 %, mínimo 24 h,
tamaño mínimo 2 % NAV; cap ≤ ¼ Kelly ex-ante. Ninguno se optimiza.
