---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-08-25
supersedes: []
code_anchors:
  - src/research/cost_model.py
  - src/research/session_env.py
  - src/research/session_gym.py
  - scripts/analysis/thesis_baselines.py
---

# BL-48 — El costo de ejecución es la variable dominante en el intradía de USD/COP

**Fuente**: [`planes/06-RESULTADOS.md`](../06-RESULTADOS.md) §6 · **Ola**: — · **Esfuerzo**: L ·
**Trials**: los que cobre cada línea, por separado

## Por qué existe este ítem

La tesis del brazo PPO (`H-TESIS-RL-01`, RECHAZADA) no produjo una estrategia, pero sí produjo
una **medición del terreno** que ninguna otra parte del repositorio tenía, y que estaba
quedándose dentro de un documento largo sin ser un ítem accionable.

Con el contrato de costos de §9.3 —spread esperado del régimen + 0,5 pips de comisión por lado
+ slippage proporcional a la volatilidad— una política acotada por sesión paga **4,81 pips por
sesión de media**. Sobre las 584 sesiones del hold-out eso es **71,4% del nocional**.

**El intradía de este par es de suma negativa antes de que entre ningún skill.** Cualquier
política que opere necesita un edge bruto de ~0,12% diario solo para llegar a cero, sobre una
desviación diaria del orden del 0,4%.

Corolario que gobierna este backlog: **el siguiente trabajo empieza por bajar el costo por
operación, no por mejorar el predictor.** Las dos configuraciones de PPO evaluadas ejecutaron
433-813 cambios en 234 sesiones (33-47% del nocional en costos) y ninguna batió a `always_flat`.

## Las cinco líneas, cada una con la medición que la motiva

| # | Línea | Medición que la respalda |
|---|---|---|
| 1 | **Bajar la frecuencia de decisión** — una o dos por sesión, no 59 | El costo escala con `\|Δw\|`, no con el acierto. **MATIZADO 2026-08-25**: medido, no basta — ver abajo |
| 2 | **Ejecución pasiva** (limit orders) en vez de cruzar el spread | El componente de spread es ~94% del costo por operación: 3,0 de los 3,16 pips del round-trip típico |
| 3 | **Penalizar el turnover en el reward**, no solo en la contabilidad | Los agentes mantuvieron \|exposición\| ≈ 0,75-0,86: nunca aprendieron que operar es caro |
| 4 | **Ablacionar el régimen contra un backbone SIN features de volatilidad** | `ppo_backbone` reproduce el condicionamiento por régimen sin recibir los posteriores, porque el HMM se ajusta sobre volatilidad y el agente ya ve siete features de vol. Es la única forma de separar H2 de esa redundancia |
| 5 | **Ampliar el horizonte más allá de la sesión** | §9.1 fuerza plano al cierre y eso paga un round-trip diario garantizado. B1 pasivo, que no lo paga, gasta 0,21% en costos frente al 71,4% del intradía |

## Evidencia nueva (2026-08-25): el alfa existe, y la cadencia sola no lo rescata

La descomposición del hold-out (`CTR-RESEARCH-DECOMP-001`, 0 trials) aporta dos números que
este backlog no tenía y que **cambian su primera línea**:

1. **El alfa existe y está cuantificado**: bruto +27,95%, Sharpe bruto +2,23 con IC que excluye
   el cero, 10/10 corridas positivas. Hay algo que rescatar, no es una expedición a ciegas.
2. **Pero el alfa vive en la alta frecuencia igual que el costo.** Re-scoring de las mismas
   decisiones cada `k` barras:

   | k | dec/sesión | BRUTO | Neto |
   |---|---|---|---|
   | 1 | 59 | +27,95% | −79,2% |
   | 30 | 2 | +9,67% | −54,1% |
   | 59 | 1 | **−5,95%** | −33,2% |

   Bajar la cadencia mejora el neto **destruyendo la señal**, no preservándola.

**Consecuencia para este backlog**: la línea 1 (bajar frecuencia) **no es viable por sí sola**,
y perseguirla aislada gastaría un trial en un rechazo previsible. Tiene que ir acompañada de la
línea 2 (ejecución pasiva), que reduce el costo **sin** tocar la cadencia y por tanto sin
arrastrar el alfa.

Y una cota dura para cualquier intento: el spread de break-even es **negativo** (`s* = −0,29`
pips). Ni a spread cero el alfa cubre la comisión de 0,5 pips por lado. **Cualquier línea que
no reduzca también la comisión —otro venue, maker rebates— parte de un suelo imposible.**

---

## Lo que NO es este ítem

No es «reintentar la tesis». El universo de `H-TESIS-RL-01` está **cerrado** y su hold-out
abierto una sola vez; nada de aquí puede reabrir esa búsqueda. Cada línea es una hipótesis
nueva que **cobra su propio trial** en el `HYPOTHESIS-REGISTRY` antes de mirar dato alguno
(constitución §2), y la línea 4 en particular exige un pre-registro propio.

La línea 2 además no es solo modelado: cambiar a ejecución pasiva cambia el contrato de costos
de §9.3, y por tanto **invalida la comparación con los resultados actuales**. Sería un
experimento distinto, no una mejora del mismo.
