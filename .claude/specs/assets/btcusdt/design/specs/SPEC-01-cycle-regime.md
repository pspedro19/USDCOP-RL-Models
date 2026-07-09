# SPEC-01 — Régimen de Ciclo (HMM fit congelado + z_ciclo)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | `btc_daily` (on-chain: MVRV_ZScore, NUPL, Puell_Multiple, SOPR, Realized_Price) |
| Materializa | `strategy/regime.py` |
| Pre-registro | Guía v3 §4.3, §5.2, §7.7; ADR-0010 |
| Modo de fallo cubierto | §7.7 (historia corta), capa de anti-look-ahead **modelos** |

## 1. Propósito y alcance

Producir dos salidas: (a) el **régimen de ciclo** discreto (4 estados con histéresis) y
(b) `z_ciclo`, un score continuo de "estiramiento de valoración". Ambos con **protocolo de
fit congelado walk-forward**, para que el pasado nunca se re-etiquete con información del
futuro.

## 2. Entradas (contrato)

| Feature | Tipo | Unidad | Fuente | Disponibilidad | Rango válido |
|---|---|---|---|---|---|
| `MVRV_ZScore` | float | z | BGeometrics | D−1 | ~[−1.5, 12] |
| `NUPL` | float | ratio | BGeometrics | D−1 | [−0.6, 1.0] |
| `Puell_Multiple` | float | ratio | BGeometrics | D−1 | [0.2, 12] |
| `SOPR` | float | ratio | BGeometrics | D−1 | ~[0.9, 1.1] |
| `Realized_Price` | float | USD | BGeometrics | D−1 | > 0 |

Precondición: cada serie respeta su fecha de inicio real (NaN antes); ventana de
calentamiento de 4 años para los z-scores (sin z-score fuera de esa ventana).

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Invariante |
|---|---|---|---|
| `regime_t` | `enum{DEEP_VALUE, ACCUMULATION, MARKUP, EUPHORIA}` | 4 estados | dwell ≥ 5–10 días (histéresis) |
| `z_ciclo_t` | float | ~[−3, 3] | promedio de z-scores rodantes 4a de {MVRV-Z, NUPL, Puell}; **alto = caro** |
| `regime_confidence` | float | [0, 1] | probabilidad posterior del HMM del estado asignado |
| `model_version` | str | — | id del modelo congelado que produjo la etiqueta (auditoría) |

## 4. Interfaz (API)

```python
class RegimeModel(Protocol):
    def fit_frozen(self, train: pd.DataFrame, window_id: str) -> "FrozenRegimeModel":
        """Ajusta el HMM sobre la ventana de entrenamiento y lo CONGELA. Idempotente."""

class FrozenRegimeModel(Protocol):
    def label_forward(self, obs_until_t: pd.DataFrame) -> RegimeLabel:
        """Etiqueta SOLO la barra t con info hasta t (D−1 de exógenas). Nunca re-etiqueta."""
    def z_ciclo(self, obs_until_t: pd.DataFrame) -> float: ...
    version: str
```

## 5. Algoritmo / lógica

```
# z_ciclo (continuo)
for m in {MVRV_ZScore, NUPL, Puell_Multiple}:
    z_m_t = (m_t − rolling_mean(m, 4y)) / rolling_std(m, 4y)     # ventana rodante, sin fuga
z_ciclo_t = mean(z_MVRV, z_NUPL, z_Puell)

# Régimen (HMM Gaussiano, 4 estados) con fit congelado
model = fit_frozen(train=[2013..2018], window_id="w0")           # EM sobre vector on-chain
regime_raw_t = model.label_forward(obs_until_t)                   # Viterbi online / posterior
regime_t     = apply_hysteresis(regime_raw_t, min_dwell=5..10d)   # anti-flip-flop

# Umbrales de referencia (priors, NO optimizados) para mapear estados a semántica:
#   DEEP_VALUE  : MVRV-Z<0, NUPL<0,      SOPR<1,  Puell<0.5
#   ACCUMULATION: MVRV-Z bajo, NUPL 0–0.25
#   MARKUP      : MVRV-Z medio, NUPL 0.25–0.5
#   EUPHORIA    : MVRV-Z>7 (o percentil-4a>0.9), NUPL>0.75, Puell>4
```

**Caveat era ETF (§7.7):** además de los umbrales absolutos se computan **percentiles
rodantes de 4 años**; la divergencia sostenida (>1 año) entre absoluto y percentil es señal
de cambio estructural (input del retiro, SPEC-12 §14.3).

**Protocolo de fit congelado (obligatorio, ADR-0010):**
1. Fit inicial sobre 1.ª ventana, congelado.
2. Etiqueta solo hacia adelante con info ≤ D−1.
3. Re-fit anual sobre ventana expandida **con purga**; el nuevo modelo etiqueta solo desde su
   despliegue. **Jamás re-etiqueta el pasado.**
4. La secuencia de labels como-se-habría-producido se versiona en DVC.

## 6. Invariantes y post-condiciones

- `z_ciclo` no usa datos > t (test de no-fuga automatizado).
- La etiqueta de una fecha nunca cambia una vez producida en vivo (append-only en DVC).
- `regime_t` no cambia más de una vez por `min_dwell` (histéresis).
- Todo label lleva el `model_version` que lo generó.

## 7. Tests unitarios

- [ ] `z_ciclo` de una serie constante = 0; de una rampa monótona = creciente y acotado.
- [ ] Ventana de calentamiento: sin z-score antes de 4 años de datos (retorna NaN, no error).
- [ ] Histéresis: una oscilación de 1 día no cambia `regime_t`.
- [ ] `label_forward` con el mismo input dos veces = mismo output (determinismo).
- [ ] No-fuga: `label_forward(obs_until_t)` es idéntico si se le pasan o no filas > t.
- [ ] `fit_frozen` es idempotente (mismo train ⇒ mismo modelo, misma `version`).

## 8. Tests de integración

- [ ] Sobre el histórico real, la secuencia de labels replay coincide con la reconstrucción
      walk-forward (no usa el modelo final).
- [ ] Los 4 regímenes aparecen en el histórico con dwell coherente (meses, no días).
- [ ] DEEP_VALUE coincide temporalmente con mínimos de precio conocidos (2015, 2018-12,
      2022-11) y EUPHORIA con techos (2017-12, 2021-04/11) — sanity económico.

## 9. Test de hipótesis

**H-REG-01 — ¿los regímenes tienen contenido predictivo de riesgo?**
- **H0:** la distribución del retorno forward ajustado por downside (Sortino a H días) es
  **igual** entre los 4 regímenes.
- **H1:** al menos un régimen difiere.
- **Estadístico:** Kruskal-Wallis sobre Sortino forward agrupado por régimen; validación por
  **permutación** (shuffle de etiquetas, 10⁴ permutaciones) para robustez ante no-normalidad
  y autocorrelación (permutación por bloques).
- **Criterio:** p < α tras Benjamini-Hochberg; tamaño de efecto (ε²) reportado.
- **Corrección múltiple:** cuenta como 1 trial en el DSR.

**H-REG-02 — ¿el modelo de régimen es estable?**
- **H0:** dos re-fits consecutivos producen labels equivalentes en su ventana de solape.
- **Estadístico:** % de labels que cambian entre re-fits.
- **Criterio:** **> 20 % ⇒ inestable ⇒ NO entra a producción** (regla dura, §4.3.5).

## 10. Protocolo de backtest / validación

- Walk-forward con el modelo congelado; **jamás** el modelo final sobre todo el histórico.
- Reporte por sub-era (pre-2020 / 2020–2023 / 2024+): el régimen debe tener sentido en cada
  una; si solo funciona en una era, se marca frágil.
- Sensibilidad de `min_dwell` {5, 7, 10} d reportada (no optimizada).

## 11. Criterios de aceptación (DoD)

- [ ] H-REG-01 rechaza H0 tras BH; H-REG-02 pasa (< 20 % de disenso).
- [ ] Tests unitarios y de integración en verde.
- [ ] Secuencia de labels replay versionada en DVC.
- [ ] Sin look-ahead de modelo (auditoría de fit congelado).
- [ ] `z_ciclo` aporta a S3 en aislamiento (verificado vía SPEC-03/06, no aquí).

## 12. Anti-look-ahead / modos de fallo cubiertos

- **Capa modelos (§4.3):** fit congelado ⇒ el pasado nunca ve el futuro. **Este es el punto
  central de la spec.**
- **Capa datos:** z-scores rodantes 4a, exógenas D−1.
- **§7.7 historia corta:** umbrales de prior (no grid), pocos regímenes, bandas anchas,
  percentiles rodantes, reporte por sub-era.

## 13. Parámetros pre-registrados

`z_ciclo` = mean de z-scores rodantes 4a de {MVRV-Z, NUPL, Puell}; HMM 4 estados; `min_dwell`
5–10 d; re-fit anual con purga; umbral de inestabilidad 20 %; umbrales semánticos de régimen
por prior (§5). Ninguno se optimiza.
