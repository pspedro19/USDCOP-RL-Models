# ADR-0009 — Combinación en riesgo vs. multiplicación de gates

**Estado:** Aceptado · **Fecha:** pre-registro · **Spec:** SPEC-03 · **Modo de fallo:** §7.1

## Contexto

La arquitectura multiplicativa `exp = vt × M_ciclo × M_liq × M_pos × G_evt` trata cuatro
gates como cuatro hipótesis independientes. Pero ciclo y posicionamiento están **altamente
correlacionados** (ambos miden "crowd estirado"): multiplicar `0.3 × 0.5 = 0.15` castiga el
mismo riesgo dos veces, sub-participando sistemáticamente en el melt-up de fin de ciclo
(donde ocurre buena parte del retorno).

## Decisión

Los componentes **correlacionados** (`|ρ|` rodante 90d > **0.4** sostenido) se **combinan
aditivamente en espacio de riesgo**, no se multiplican:
```
R = 0.7·z_ciclo + 0.3·z_funding ; M_interno = 0.25 + 0.75/(1+exp(1.5·(R−0.5))) ∈ [0.25, 1.0]
```
Solo los componentes **~ortogonales** (liquidez, eventos) multiplican. La ortogonalidad se
**mide** trimestralmente; un cambio de combinación cuenta como trial en el DSR.

## Alternativas descartadas

- **Multiplicación ingenua de los 4 gates:** doble conteo (el fallo).
- **Aprender los pesos por optimización:** reintroduce overfitting; se prohíbe (los pesos son
  priors de timescale).
- **Usar el mínimo de gates correlacionados:** menos aggressive que el producto, pero pierde
  la información de magnitud; la suma en riesgo la conserva.

## Consecuencias

Se sacrifica algo de la auditabilidad de la factorización pura (los correlacionados ya no se
atribuyen 100 % por separado), recuperada vía atribución en aislamiento (SPEC-11). A cambio,
no se doble-cuenta. **H-CMB-01 valida que esto mejora Calmar OOS vs. multiplicar** — si no lo
hiciera, se reabre la decisión.
