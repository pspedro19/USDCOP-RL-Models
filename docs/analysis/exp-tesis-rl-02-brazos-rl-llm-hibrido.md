---
kind: analysis
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-12
supersedes: []
code_anchors:
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/thesis_hybrid.py
  - scripts/analysis/run_thesis_llm.py
  - scripts/analysis/settle_thesis_llm.py
  - src/research/ppo_recipe.py
---

# Brazos RL, LLM e híbrido — EXP-TESIS-RL-02 (bloque de selección)

Complementa el [capítulo de resultados](exp-tesis-rl-02-capitulo-resultados.md), que cubre
integridad de datos, baselines y el brazo supervisado. Aquí van los tres brazos medidos con el
dataset v2 y la receta que superó la compuerta sintética: **PPO**, **LLM** e **híbrido**.

> **Alcance.** Todo lo que sigue es **diagnóstico retrospectivo**. El bloque de selección (2023)
> ya se miró y motivó las correcciones de v2, así que ninguna cifra de aquí es confirmatoria.
> El juez confirmatorio es el carril forward desde el freeze. Está declarado así en el
> pre-registro v3 y repetido en cada tabla a propósito.

## Qué hacía falta arreglar antes de que estos números valieran algo

**La compuerta de sanidad no gobernaba al entrenador.** `thesis_train_ppo.py` leía el informe,
imprimía `receta=flat_init_no_turn` y entrenaba otra cosa: el sesgo inicial de `action_net` —que
*es* la sonda que abrió S1–S4— vivía sólo dentro del script de sanidad. Una compuerta que valida
un informe mientras corre otra receta no es una compuerta. Corregido en
[`src/research/ppo_recipe.py`](../../src/research/ppo_recipe.py): una sola implementación para
sanidad y mercado, sonda desconocida aborta, y cada artefacto graba `recipe_probe`.

**Azure respondía en contrato y el arnés lo contaba como fallo.** `gpt-4o-mini` envuelve el JSON
en una valla markdown; DeepSeek lo devuelve pelado. `json.loads` reventaba sobre la valla y la
decisión se sellaba `numeric_field_invalid`. Sin arreglarlo, el brazo de robustez habrían sido
13.334 llamadas pagadas cuyo resultado dice «el modelo no supo responder», cuando respondió bien
las 13.334 veces. La corrección se aplica **igual a los dos proveedores**: hacerlo sólo para Azure
sería el mismo trato desigual por el otro lado.

**Una caída de red quedó sellada como decisión.** 884 barras en DeepSeek y 882 en Azure cayeron
**en el mismo instante** —14 sesiones completas por brazo—, firma de problema local y no de
proveedor. La política congelada `retain_previous_weight` las habría liquidado como jornadas
enteras de «mantener», indistinguibles de una convicción del modelo. Esa política existe para un
modelo que responde **mal**, que es comportamiento medible; un modelo al que **no se llegó** no
decidió nada. Se apartaron y se volvieron a pedir: los dos ledgers cierran con **0 no respondidas
y 0 inválidas**.

## Resultados — selección, 226 sesiones, 2023-01-03 → 2023-12-29

| brazo | n | neto % | Sharpe | IC 95 % | DD % |
|---|---:|---:|---:|---|---:|
| `always_flat` | 226 | **0,00** | +0,00 | — | 0,00 |
| `NULL_A_corto_1x` | 226 | −11,97 | −1,11 | [−2,71, +0,48] | −16,58 |
| `B1_pasivo` | 226 | −20,15 | −1,46 | [−3,53, +0,50] | −22,94 |
| `B1_sesion_1x` | 226 | −32,54 | −3,53 | [−5,38, −1,79] | −33,20 |
| `ppo_backbone_mean5` | 226 | −33,09 | −8,56 | [−10,74, −6,74] | −33,15 |
| `ppo_regime_mean5` | 226 | −35,52 | −6,82 | [−9,31, −4,71] | −35,94 |
| supervisado (LogReg) | 226 | −55,08 | −9,73 | — | — |
| híbrido PPO+DeepSeek | 226 | −69,24 | −14,33 | [−16,00, −12,95] | −69,14 |
| híbrido PPO+Azure | 226 | −70,59 | −11,01 | — | −70,57 |
| DeepSeek `deepseek-chat` | 226 | −78,93 | −22,78 | [−25,45, −20,66] | −78,82 |
| Azure `gpt-4o-mini` | 226 | −82,89 | −14,84 | — | −82,82 |

Los dos ledgers cierran **13.334/13.334 decisiones, 0 no respondidas, 0 inválidas**, y las dos
liquidaciones dan **226 de 226 sesiones sin una exclusión**.

**Ninguno bate a no operar.** Contra `always_flat`, todos con **p = 0,0002** — que es el suelo de
resolución de 10.000 réplicas Monte Carlo, no un p-valor diminuto, y así se reporta.

El orden es lo llamativo: cuanto más sofisticado el método, peor el resultado. No operar gana a
una regla tonta, que gana al RL, que gana al supervisado, que gana al LLM.

## No hay un fracaso, hay dos

Descompuesto en bruto y coste, los brazos fallan por razones distintas y fundirlos en «ninguno
bate a flat» perdería lo único interesante del experimento.

| brazo | bruto | coste | cambios/sesión |
|---|---:|---:|---:|
| PPO (mediana de semillas) | **+9,90 %** | 55,16 % | 5,94 |
| híbrido PPO+DeepSeek | −12,35 % | 104,90 % | 9,91 |
| DeepSeek | **−15,73 %** | 139,26 % | 18,86 |

**El PPO tiene un problema de coste de transacción.** Genera alfa bruta positiva y opera
demasiado para conservarla. La política encuentra algo; el peaje se lo come.

**El LLM tiene un problema de señal.** Sus decisiones pierden dinero **antes de pagar nada**.
Ninguna reducción de costes lo salva.

## El hallazgo del híbrido: el acuerdo del LLM es anti-informativo

La regla del híbrido se congeló en el pre-registro con los ledgers al 7 % y al 3 %, sin poder ver
lo que produce: se conserva la exposición del PPO sólo cuando el LLM coincide en signo. Se queda
con **3.048 de 11.315 posiciones (26,9 %)** y aun así empeora.

La descomposición dice por qué, y es un resultado **sobre el LLM**, no sobre el veto:

```
bruto PPO sobre TODAS sus barras : +9,90 %
  en las que el LLM avala        : -12,35 %   (3.048 barras)
  en las que el LLM rechaza      : +22,25 %
```

Si el tramo avalado tuviera el mismo signo que el total, el veto sería sólo una muestra más
pequeña. Con el signo **invertido**, el acuerdo del LLM no es ruido: **selecciona justamente las
posiciones perdedoras del PPO**. Como filtro no es inútil, es anti-informativo — tomar el 27 % que
aprueba es peor que tomarlas todas.

**Y se replica en un proveedor independiente, más fuerte todavía.** Azure `gpt-4o-mini` avala el
30,5 % de las posiciones y su descomposición es:

```
bruto PPO sobre TODAS sus barras : +9,90 %
  en las que Azure avala         : -23,41 %
  en las que Azure rechaza       : +33,31 %
```

Dos modelos de vendedores distintos, con prompts idénticos y sin contacto entre sí, seleccionan
el mismo subconjunto perdedor. Eso saca el hallazgo del terreno de la casualidad: no es que *un*
modelo fallara, es que la clase de señal que un LLM extrae de este contexto está sistemáticamente
invertida respecto de lo que conviene operar.

Verificado por una vía independiente, recomputando `w_ppo · r` barra a barra desde el portable y
el ledger, antes de escribirlo aquí.

## Una corrección a nuestro propio pre-registro

Al congelar la regla escribimos que *«un veto sólo puede reducir rotación; no puede inventar
bruto»*. **La primera mitad es falsa.** El híbrido hace **9,91 cambios por sesión contra 5,94 del
PPO solo**: casi el doble.

Un veto no recorta una posición, la **interrumpe**. Donde el PPO mantenía +0,5 durante veinte
barras seguidas, el veto la corta cada vez que el LLM discrepa y la restaura al volver a
coincidir, convirtiendo un tramo continuo en una alternancia. Reduce el *tiempo* en posición y
aumenta el *número de cambios* — y lo que cuesta dinero son los cambios.

**La regla no se tocó.** Sustituirla al ver lo que hace sería la selección que el pre-registro
existe para impedir. Se corrigió el argumento y se fijó con un contraejemplo mínimo en
`test_hybrid_rule_is_frozen.py`: seis barras planas del PPO son 1 cambio y 6 bajo el veto.

## Qué queda cerrado y qué no

**Cerrado**: la objeción de que el PPO perdía por una receta rota. La misma receta que se abstiene
sobre ruido puro (S1, 5/5 semillas) opera cuatro veces por sesión sobre USD/COP y pierde en las
diez. El rechazo pasa de «resultado de una implementación con fuga macro y receta sin validar» a
«resultado de un dataset sin fuga con una receta que supera los controles de solución conocida».

**No cerrado**: nada de esto es confirmatorio. El DSR trial-aware de ambas configuraciones PPO es
**0,0000 con n_trials = 115**, y el stress de costes las mata a ×2 y ×3. La única comparación que
salió a favor de algo es la ablación H2 — `ppo_regime` supera a `ppo_backbone` (ΔSharpe +1,745,
p = 0,0406): los posteriores de régimen aportan, pero aportan para perder menos.

**Declaración de búsqueda**: la receta `flat_init_no_turn` se eligió tras probar **ocho sondas**
sobre fixtures sintéticas. No cobra trial de mercado —las fixtures no son USD/COP— pero es una
búsqueda, no un prior económico, y quien lea esto tiene derecho a saberlo.
