# SPEC-03 — Combinación en Riesgo (R → M_interno) ★

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | SPEC-01 (`z_ciclo`), SPEC-02 (`z_funding`) |
| Materializa | `strategy/risk_combination.py` |
| Pre-registro | Guía v3 §6.2, §7.1; ADR-0009 |
| Criticidad | **Máxima.** Es el componente donde la arquitectura vive o muere (§7.1) |

## 1. Propósito y alcance

Combinar `z_ciclo` (lento, meses) y `z_funding` (rápido, días) en un **único multiplicador
interno de riesgo** `M_interno ∈ [0.25, 1.0]`, **aditivamente en espacio de riesgo, NO por
multiplicación**. Resuelve el modo de fallo §7.1: ciclo y funding están correlacionados
(ambos miden "crowd estirado"), y multiplicarlos doble-contaría el mismo riesgo,
sub-participando sistemáticamente en el melt-up de fin de ciclo.

Incluye además la **regla de mantenimiento de ortogonalidad** que decide qué otros
componentes deben absorberse al score si dejan de ser ortogonales.

## 2. Entradas (contrato)

| Entrada | Tipo | Rango | Fuente |
|---|---|---|---|
| `z_ciclo_t` | float | ~[−3, 3] | SPEC-01 |
| `z_funding_t` | float | ~[−3, 3] | SPEC-02 |
| `M_liquidez_t` | float | [0.5, 1.25] | SPEC-04 (solo para el chequeo de ortogonalidad) |
| `G_eventos_t` | float | {1, .5, .25, 0} | SPEC-05 (solo para el chequeo de ortogonalidad) |

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Invariante |
|---|---|---|---|
| `R_t` | float | ~[−3, 3] | score de riesgo aditivo; **alto = caro/riesgo** |
| `M_interno_t` | float | **[0.25, 1.0]** | sigmoide decreciente de R; monótona no creciente en R |
| `orthogonality_report` | dict | — | matriz de correlación rodante 90d entre {M_interno, M_liquidez, G_eventos} |

## 4. Interfaz (API)

```python
def combinar_en_riesgo(z_ciclo: float, z_funding: float,
                       w_ciclo: float = 0.7, w_funding: float = 0.3,
                       a: float = 1.5, b: float = 0.5) -> float:
    """R = w_ciclo·z_ciclo + w_funding·z_funding
       M_interno = 0.25 + 0.75 / (1 + exp(a·(R − b)))   ∈ [0.25, 1.0], decreciente en R."""

def check_orthogonality(series: dict[str, pd.Series], window: int = 90,
                        threshold: float = 0.4) -> OrthogonalityReport:
    """Correlación rodante entre gates; marca los pares con |ρ| > threshold sostenido."""
```

Pesos y `(a, b)` son **parámetros pre-registrados con defaults congelados**; se exponen solo
para el análisis de sensibilidad de SPEC-11, nunca para optimización.

## 5. Algoritmo / lógica (con la matemática completa)

```
# 1) Score de riesgo aditivo (combinación en espacio de riesgo)
R_t = 0.7 · z_ciclo_t + 0.3 · z_funding_t

# 2) Mapeo sigmoide decreciente a multiplicador
M_interno_t = 0.25 + 0.75 / (1 + exp( 1.5 · (R_t − 0.5) ))

#    Propiedades (verificables):
#    - R → −∞  ⇒ M_interno → 1.00   (barato/capitulación: exposición plena, NUNCA >1)
#    - R = 0.5 ⇒ M_interno = 0.625   (centro)
#    - R = 0   ⇒ M_interno ≈ 0.85    (neutro: levemente defensivo por diseño)
#    - R → +∞  ⇒ M_interno → 0.25   (euforia: piso; nunca flat por valoración sola)
```

**Por qué aditivo y no multiplicativo (el corazón del fix, §7.1):**
combinar señales **correlacionadas** por producto castiga el mismo riesgo dos veces. En
espacio de riesgo (z-scores), la suma ponderada es la forma correcta de agregar evidencia
correlacionada sobre un mismo factor latente ("crowd estirado"). El producto se reserva para
factores **ortogonales** (liquidez, eventos), donde sí son hipótesis independientes.

**Justificación de cada número (todos priors, ninguno grid-searched):**
- **0.7 / 0.3:** el ciclo es la hipótesis principal (diferenciador on-chain de BTC, opera en
  meses); el funding aporta la información rápida (días). Codifica "el ciclo manda, el funding
  matiza".
- **Sigmoide:** satura los extremos (MVRV-Z 9 ≈ MVRV-Z 7, ambos "euforia clara"); evita que
  un outlier de una métrica domine.
- **a = 1.5:** pendiente suave, sin saltos bruscos de exposición.
- **b = 0.5:** con R≈0, M≈0.85 (defensivo por defecto: el sistema gana en Calmar recortando
  colas, no maximizando beta).
- **Piso 0.25 / techo 1.0:** nunca flat por valoración sola (flat = solo eventos+breaker);
  nunca apalancar en capitulación (§7.6, "barato" ≈ "yéndose a cero" en tiempo real).

**Regla de mantenimiento de ortogonalidad (pre-registrada, trimestral):**
```
if |ρ_rolling_90d(M_interno, M_liquidez)| > 0.4 sostenido (>50% de días):
    → M_liquidez deja de multiplicar; se integra a R con peso re-derivado de priors
    → ESE CAMBIO CUENTA COMO TRIAL EN EL DSR (HYPOTHESIS-REGISTRY)
```

## 6. Invariantes y post-condiciones

- `M_interno ∈ [0.25, 1.0]` para todo R (verificable analíticamente).
- `M_interno` es **monótona no creciente** en R (más riesgo ⇒ menos exposición).
- `M_interno(R=0.5) = 0.625` exacto (test de anclaje numérico).
- `combinar_en_riesgo` es determinista y sin estado.
- La suma de pesos `w_ciclo + w_funding = 1.0` (chequeo de configuración).

## 7. Tests unitarios

- [ ] Anclajes numéricos: `M(R→−∞)=1.0`, `M(R=0)≈0.85`, `M(R=0.5)=0.625`, `M(R→+∞)=0.25`.
- [ ] Monotonicidad: para R1 < R2, `M(R1) ≥ M(R2)` (muestreo denso del dominio).
- [ ] Rango: 10⁴ inputs aleatorios ⇒ `M ∈ [0.25, 1.0]` siempre.
- [ ] Determinismo: mismo `(z_ciclo, z_funding)` ⇒ mismo `M_interno`.
- [ ] Pesos: `w_ciclo + w_funding ≠ 1` ⇒ error de configuración.
- [ ] `check_orthogonality` detecta un par sintético con ρ=0.8 y no marca uno con ρ=0.1.

## 8. Tests de integración

- [ ] Sobre el histórico: en EUPHORIA (SPEC-01) con funding alto (SPEC-02), `M_interno` cae a
      la zona 0.25–0.4; en DEEP_VALUE con funding negativo, sube a 0.9–1.0.
- [ ] **Test anti-doble-conteo:** comparar `M_interno` (aditivo) vs. `M_ciclo_naive ×
      M_funding_naive` (multiplicativo) en euforia: el multiplicativo debe ser
      **materialmente más bajo** (evidencia del doble conteo que evitamos).
- [ ] La matriz de correlación rodante entre {M_interno, M_liquidez, G_eventos} se computa y
      loggea sin error sobre todo el histórico.

## 9. Test de hipótesis

**H-CMB-01 — ¿la combinación en riesgo mejora Calmar vs. multiplicación ingenua?** *(el test
que justifica toda la spec)*
- **H0:** el Calmar OOS del sistema con `M_interno` combinado en riesgo es **≤** al del sistema
  con `M_ciclo × M_funding` multiplicados ingenuamente.
- **H1:** la combinación en riesgo mejora el Calmar OOS.
- **Estadístico:** **bootstrap pareado** sobre la diferencia de Calmar OOS
  (`ΔCalmar = Calmar_combinado − Calmar_multiplicado`), remuestreo por bloques (preserva
  autocorrelación), 10⁴ réplicas.
- **Criterio:** IC 95 % de `ΔCalmar` **> 0**. Se reporta también Sortino y turnover (la
  combinación no debe ganar Calmar a costa de disparar turnover).
- **Corrección múltiple:** 1 trial en el DSR.

**H-CMB-02 — ¿ciclo y funding son ortogonales (no requieren combinación)?**
- **H0:** `|ρ_rolling_90d(z_ciclo, z_funding)| ≤ 0.4` sostenido.
- **Estadístico:** fracción de días con `|ρ| > 0.4`.
- **Criterio:** si > 50 % de los días superan 0.4 ⇒ **rechazamos ortogonalidad ⇒ la
  combinación aditiva está justificada** (que es la hipótesis de diseño). Si NO se rechaza,
  se documenta que la combinación es conservadora pero inocua.

## 10. Protocolo de backtest / validación

- El backtest de H-CMB-01 usa el motor completo (SPEC-06) con CostModel (SPEC-07), variando
  **solo** el bloque de combinación (aditivo vs. multiplicativo) — todo lo demás fijo.
- Reporte por sub-era: la ventaja de la combinación debe verse (o no) en cada era; especial
  atención al melt-up 2021 (donde el doble conteo más castiga).
- Sensibilidad de pesos {60/40, 70/30, 80/20} y de `(a,b)` reportada completa (SPEC-11); cada
  celda es un trial.

## 11. Criterios de aceptación (DoD)

- [ ] **H-CMB-01 rechaza H0** (la combinación gana Calmar OOS, dentro del presupuesto de
      turnover). Si NO, se reconsidera la arquitectura — este es el gate más importante.
- [ ] H-CMB-02 resuelto (ortogonalidad medida, no asumida).
- [ ] Anclajes numéricos y monotonicidad en verde.
- [ ] La regla de mantenimiento de ortogonalidad está implementada y loggea a MLflow.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.1 doble conteo (RESUELTO AQUÍ):** combinación aditiva en riesgo de los correlacionados;
  producto solo para ortogonales; ortogonalidad medida trimestralmente.
- **§7.4 selección:** cambios de combinación cuentan como trials en el DSR.
- Trade-off nombrado: combinar sacrifica algo de auditabilidad pura; se recupera vía
  atribución en aislamiento (SPEC-11): núcleo+z_ciclo y núcleo+z_funding se miden por separado
  antes de entrar al score.

## 13. Parámetros pre-registrados

`R = 0.7·z_ciclo + 0.3·z_funding`; `M_interno = 0.25 + 0.75/(1+exp(1.5·(R−0.5)))` ∈ [0.25, 1.0];
umbral de combinación `|ρ 90d| > 0.4` sostenido; revisión trimestral. Ninguno se optimiza;
sensibilidades se reportan completas.
