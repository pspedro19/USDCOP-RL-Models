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

**Ninguno bate a no operar**, y el contraste correcto no es el que publiqué primero.

> **Corrección 2026-09-13.** Una versión anterior decía «todos con **p = 0,0002**». **Falso en dos
> sentidos.** Ese número salía de las colas percentiles del bootstrap, que sirven para describir
> pero no para contrastar. Recomputado con la media diaria **centrada bajo H0**, bilateral, con
> corrección de continuidad (Codex, `results.json::paired_primary_mean_return_tests`):
>
> - **Nueve de diez brazos**: `p = 1/10001 = 0,00009999`, **Holm** `0,0009999`.
> - **`NULL_A_corto_1x`: `p = 0,1794`** (1.793 excedencias) — **NO significativo**. El corto 1×
>   pierde 11,97 % y aun así **no es distinguible de no operar**: su varianza se come la
>   diferencia.
> - **`B1_pasivo` queda fuera** del contraste de Sharpe: tiene **1 operación**, y la regla de
>   N < 20 de la constitución §6 lo prohíbe.
>
> La cola percentil `0,00019998` sigue existiendo en los artefactos, etiquetada **DESCRIPTIVO**.
> No debe citarse como p-valor.

Los brazos con modelo —PPO, supervisado, LLM e híbridos— pierden contra la abstención con
`p_Holm ≈ 0,001`. El único que no se distingue de no operar es el baseline tonto.

El orden es lo llamativo: cuanto más sofisticado el método, peor el resultado. No operar gana a
una regla tonta, que gana al RL, que gana al supervisado, que gana al LLM.

## No hay un fracaso, hay dos

Descompuesto en bruto y coste, los brazos fallan por razones distintas y fundirlos en «ninguno
bate a flat» perdería lo único interesante del experimento.

Bruto y coste de esta tabla son **sumas aritméticas** (ver la corrección de arriba); el bruto
compuesto del PPO es **+10,0769 %**.

| brazo | bruto (suma) | coste (suma) | cambios/sesión |
|---|---:|---:|---:|
| PPO (mediana de exposiciones) | **+9,90 %** | 55,16 % | 5,94 |
| híbrido PPO+DeepSeek | −12,35 % | 104,90 % | 9,91 |
| DeepSeek | **−15,73 %** | 139,26 % | 18,86 |

**El PPO tiene un problema de coste de transacción.** Su bruto es positivo y el peaje se lo
come. **No es «alfa demostrada»** —Codex objetó esa palabra y con razón—: el bruto de este bloque
está concentrado en el lado corto (+9,22 % de los cortos contra +0,68 % de los largos) en un año
en que el par cayó un 20,2 %. Un bruto que vive del lado que coincide con la tendencia del bloque
no está distinguido de exposición direccional afortunada, y un solo bloque no puede separarlos.

**El LLM tiene un problema de señal.** Sus decisiones pierden dinero **antes de pagar nada**.
Ninguna reducción de costes lo salva.

**La prueba más limpia de que son dos fracasos distintos no es mía, es del bundle de Codex**
(`research_grade_20260912_v1/cost_stress.json`, 24/24 artefactos verificados por hash). Apagando
el peaje **por completo**:

| brazo | coste cero | ×1 | ×2 | ×3 |
|---|---:|---:|---:|---:|
| PPO (mediana de exposiciones) | **+10,0769 %** | −36,67 % | −63,63 % | −79,15 % |
| DeepSeek | **−14,6664 %** | −78,93 % | −94,85 % | −98,75 % |
| Azure | **−28,9768 %** | −82,89 % | −95,93 % | −99,04 % |
| híbrido PPO+DeepSeek | −11,73 % | −69,24 % | −89,36 % | — |

Sin coste alguno, el PPO sigue en positivo y los dos LLM siguen en negativo. Eso separa «problema
de coste» de «problema de señal» sin depender de ninguna suma de brutos, y lo produjo una
implementación independiente de la mía: sus cifras a ×1 coinciden con las publicadas aquí hasta
la segunda decimal.

> **Corrección 2026-09-13 — y es el error que esta misma tesis ya había retirado una vez.**
> Escribí que la diferencia entre **+10,0769 %** (Codex) y **+9,9007 %** (mío) era el coste
> terminal. **Falso.** Lo midió Codex y lo confirmé recomputando: `+9,9007` es la **suma
> aritmética** de `w·r` sesión a sesión y `+10,0769` es el **compuesto** de las mismas sesiones.
> Los 0,1761 pp son el interés compuesto, nada más.
>
> Es exactamente la confusión que el CORRIGENDUM de `06-RESULTADOS.md` retiró en v1.1.0 —el
> «bruto +27,95 %» que era suma presentado junto a un neto compuesto—, cometida otra vez por mí
> en el capítulo que la documenta. La cifra canónica es la **compuesta**: **+10,0769 %**.
>
> **Todas las descomposiciones de bruto de esta sección son sumas aritméticas**, y sólo así
> cuadran: `+9,9007 = −12,35 + 22,25` se sostiene porque la suma es aditiva y el compuesto no lo
> es. Se etiquetan como sumas a propósito; **no deben compararse contra cifras compuestas**.

> **Dos agregaciones distintas, y no son intercambiables.** La fila `ppo_regime_mean5` de la tabla
> anterior es la **media de retornos de cinco semillas** (−35,52 %). La fila «PPO» de esta tabla es
> la **mediana de exposiciones barra a barra** (−36,67 % sobre las mismas sesiones), que es la que
> entra en el híbrido porque el veto opera barra a barra. Son objetos distintos y el artefacto del
> híbrido los reporta por separado (`ppo_median_same_sessions`). Lo señaló Codex; una versión
> anterior de este capítulo los ponía en la misma columna sin decirlo.

## El híbrido y el sesgo direccional — corregido 2026-09-13

> **Retractación.** La primera versión de esta sección afirmaba que el acuerdo del LLM era
> «anti-informativo», que el hallazgo se «replicaba en un proveedor independiente» y que eso
> «descartaba la casualidad». **Las tres cosas son overreach y las retiro.** Lo señaló Codex con
> los conteos que aparecen abajo, y al verificarlos aparece una explicación más simple que la mía.

La regla se congeló en el pre-registro con los ledgers al 7 % y al 3 %: se conserva la exposición
del PPO sólo cuando el LLM coincide en signo. El resultado medido:

| | avala | de ellas largas | bruto avalado | bruto total PPO |
|---|---:|---:|---:|---:|
| DeepSeek | 3.048 barras (26,9 %) | 62 % | **−12,35 %** | +9,90 % |
| Azure | 3.455 barras (30,5 %) | **99 %** | **−23,41 %** | +9,90 % |

El tramo avalado tiene el signo contrario al total, y eso es real. Pero **la causa no es que el
LLM extraiga señal invertida**. Es más mundana:

```
USD/COP en selección (2023): 4862 -> 3882   (-20,2 %)
bruto PPO en posiciones CORTAS: +9,22 %  (3.208 barras)
bruto PPO en posiciones LARGAS: +0,68 %  (8.107 barras)
```

**Todo el alfa del PPO está en el lado corto**, porque el peso se apreció un 20 % ese año. Y los
dos LLM tienen **sesgo largo**: Azure decide largo en el 98 % de sus posiciones (4.399 largas
contra 90 cortas), DeepSeek en el 42 %. Avalar sesgado a largo equivale a **vetar el lado donde
está el alfa**, y el bruto avalado sale negativo por el lado largo en ambos casos (−17,01 % y
−22,42 %).

Dicho de otro modo: el veto no descubrió las posiciones perdedoras del PPO, **descartó sus
posiciones ganadoras** porque apuntaban en la dirección que el LLM no quería tomar.

**Por qué «replicación independiente» era falso**, y es la parte que más me equivoqué:

- Comparten **los mismos datos y el mismo prompt**. No son dos experimentos, son dos lecturas
  del mismo estímulo.
- No avalan el mismo subconjunto: intersección **1.842** barras sobre 3.048 y 3.455 (unión 4.661).
  La correlación de sus pesos es **0,626**, lejos de la coincidencia que yo describí.
- Lo que sí comparten es el **sesgo largo**, que es la causa común y no una confirmación mutua.

**Qué queda en pie, con el alcance correcto**: el brazo LLM pierde, y pierde por una razón
identificable —un prior direccional equivocado para este bloque— y no por costes. Eso es
suficiente para rechazarlo aquí y **no es suficiente** para afirmar nada general sobre si un LLM
puede extraer señal de este contexto. Un año en el que el par hubiera subido produciría el
resultado contrario por el mismo mecanismo, y este diseño no puede distinguirlos.

**Confundido adicional declarado** (también de Codex): el prompt entrega las features
**normalizadas y sin unidades**, así que un modelo que recibe z-scores no puede razonar sobre
magnitudes. No se corrige ni se re-corre: el prompt está congelado por hash y cambiarlo viendo
el resultado sería selección. Un prompt con unidades es una hipótesis nueva y cobra su trial.

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

**Reproducibilidad, declarada.** Estas cifras se midieron contra el portable
`7f332df17a2492a0…` (fichero inalterado). El **código** cambió después —Codex reconstruyó las
fixtures sintéticas y el modelo de costes—, así que `dataset_identity()` en HEAD ya no coincide
(`1fa989d2…` esperado contra `7f042564…` grabado) y `load_portable` **se niega a cargarlo sin
`allow_stale`**. Los números son los medidos; **no son reproducibles contra HEAD** hasta que se
reconcilie la identidad. Se dice aquí en vez de dejar que alguien lo descubra.

**El posterior de régimen se entregó truncado — reparametrizado, no mutilado.**
Hallazgo abierto por el guard que Codex añadió en `dataset.py`, y acotado por él mismo después.

`regime_hmm.py:78` deja que el BIC elija K entre 2 y 5; el esquema congela **cuatro** huecos; y
`dataset.py:199-205` hace `probs[:N_REGIMES]`. Con K=5, la quinta coordenada no se escribe. Los
dos portables lo llevan grabado dentro —`regime_meta: k=5`, etiquetas
`['calmo','intermedio','intermedio','intermedio','shock']`— y el BIC prefiere 5 en ambos.

**Lo que esto NO es, y lo dije mal antes.** Escribí que la política entrenó «con cuatro quintos
del estado, sin el shock». **Falso, y la corrección es de Codex**: cuatro coordenadas de un
símplex de cinco **determinan la quinta**, `p₄ = 1 − Σp₀..₃`. La información está presente; una
capa lineal puede recuperarla. No hubo pérdida inevitable, hubo **reparametrización no
declarada**: el vector deja de ser una distribución y pasa a ser cuatro de sus cinco coordenadas,
sin que nada en el contrato lo diga.

**Lo que sí es, medido por Codex sobre el portable archivado** (`SHA 7f332df1…`, K=5, HMM
`0a4b9ec5…`, verificado por un revisor independiente con *unpickling* restringido), en float64
sobre las 226 sesiones de selección:

- **78 sesiones** con déficit `> 1e-6` —es decir, donde la quinta coordenada no era despreciable—;
- **19 sesiones** en las que, al reconstruir `p₄`, **el régimen más probable cambia**;
- reparto por estado `[14, 27, 106, 60, 19]`;
- el spread reconstruido coincide con el original a `1,15e-7`, así que la contabilidad no se movió.

O sea: en **19 de 226 sesiones (8,4 %)** el estado dominante real era uno que el vector entregado
no señalaba como dominante. El contrato y la clasificación descriptiva son erróneos; los pesos y
el P&L, no.

**V1 NO ESTÁ DEMOSTRADO, y también es corrección suya.** El portable sin sufijo (`SHA 8483…`)
tiene bloques de **488/226/520** sesiones, no los **499/234/584** del EXP-TESIS-RL-01 publicado:
es una reconstrucción, no el artefacto original. **Nada de lo anterior prueba con qué K entrenó la
tesis publicada**; haría falta el linaje histórico. Aquí se afirma sólo de v2, que es lo que está
verificado.

**La identidad del modelo LLM tampoco está fijada.** El ledger graba `model_id: "deepseek-chat"`
y `gpt-4o-mini`, que son **alias de despliegue, no versiones**. No hay `served_model`,
`model_snapshot`, `system_fingerprint` ni `model_version` en ninguna de las 13.334 filas de
ninguno de los dos brazos.

Codex comprobó la página oficial de precios el 2026-09-13 y **`deepseek-chat` no aparece en la
tabla vigente**: figuran `deepseek-flash` (DeepSeek-V4.1-Flash) y `deepseek-v4-pro`
(DeepSeek-V4-Pro-0813).

*(Errata: una versión anterior de este párrafo decía, citando a Codex, que la página describía
`deepseek-chat` como alias de V4-Flash *non-thinking* con deprecación el 2026-07-24. **Codex
retiró esa afirmación**: procedía de un resultado de búsqueda indexado meses antes, no del cuerpo
actual de la página. Se retira aquí también. Ni la versión cacheada ni la viva acreditan qué
modelo respondió a las llamadas históricas.)*

El identificador con el que se llamó **ya no existe en la tabla de precios del proveedor**, y no
hay forma, desde la evidencia archivada, de establecer qué versión sirvió cada una de las 13.334
respuestas.

Consecuencia: los brazos LLM **no son reproducibles a nivel de modelo**. Sus números son los que
se obtuvieron, con el prompt y el muestreo congelados y hasheados, pero nadie puede volver a
pedirle lo mismo *al mismo modelo* y esperar lo mismo. Para el brazo de robustez, que existe para
comparar proveedores, la limitación es mayor: compara dos alias, no dos modelos.

**Identidad macro verificada, disponibilidad NO.** El artefacto
`research_grade_macro_20260912/identity_live_network.json`
(SHA `400ba9a5833f4829f91a471c644e0409cd185910d9dc95ded31867c3762e716e`, verificado por mí)
confirma que las cuatro series coinciden **exactamente** con su fuente declarada: Brent
`FRED_DCOILBRENTEU` 9.973 filas, DGS2 `FRED_DGS2` 12.566, DXY `INVESTING_DXY` 1.747, IBR
`BANREP_IBR` 4.558 — **0 discrepancias y `max_abs_diff` 0,0 en las cuatro**.

Y declara por sí mismo lo que **no** verifica: `historical_availability_verified: **false**` y
`source_independence_verified: **false**`.

Eso segundo importa más de lo que parece en una tesis cuyo defecto central fue el look-ahead.
Que la serie de hoy coincida con la fuente de hoy **no demuestra que ese valor estuviera
disponible en la fecha que el dataset le atribuye**. Las revisiones y los *vintages* quedan
fuera del alcance de esta comprobación. El gate de causalidad (`macro_causality_t_minus_1`,
PASS) verifica la regla de desplazamiento del pipeline, que es otra cosa: comprueba que **no
usamos** el valor del mismo día, no que el valor de hace tres años fuera el que hoy leemos.

Es la brecha abierta más relevante que queda, y el artefacto tiene el mérito de nombrarla en vez
de dejarla implícita en un `True` global.

**Vintages capturados (2026-09-12).** Codex capturó ALFRED para las dos series que lo publican:
**452 capturas, cero errores**, manifiesto `SHA f1b903ae5e9716d444ea8c11e691987746e320bf5f9ba584996c6430be031c66`.
Resultado: **226 `NOT_IN_PRIOR_DATE_VINTAGE` por serie**, en DGS2 y en Brent — el valor que el
pipeline usa como T−1 no aparece en el vintage del día anterior en **ninguna** de las 226 sesiones.

**Lo que eso significa y lo que no**, con la acotación que puso él mismo: **no** autoriza a decir
«226 sesiones usaron datos no publicados a las 08:00». Que el valor no estuviera en el vintage de
ayer no excluye que se publicara hoy antes de la apertura, ni que estuviera disponible por otra
fuente. Lo que sí establece es que **la disponibilidad en el momento de decidir nunca se
verificó**, y que hay una vía concreta —ALFRED— para verificarla en dos de las cuatro series.

Si al cerrar el replay resultara que el valor no estaba disponible antes de la apertura, sería
una fuga por **latencia de publicación**: distinta de la fuga de mismo día que este programa
corrigió, invisible para el gate de causalidad, y viva en todos los resultados de esta tesis.

**Cerrado**: la objeción de que el PPO perdía por una receta rota. La misma receta que se abstiene
sobre ruido puro (S1, 5/5 semillas) opera cuatro veces por sesión sobre USD/COP y pierde en las
diez. El rechazo pasa de «resultado de una implementación con fuga macro y receta sin validar» a
«resultado de un dataset sin fuga con una receta que supera los controles de solución conocida».

**No cerrado**: nada de esto es confirmatorio. El DSR trial-aware de ambas configuraciones PPO es
**0,0000 con n_trials = 115**, y el stress de costes las mata a ×2 y ×3.

**Retractación (2026-09-12).** Una versión anterior de este capítulo decía que la ablación H2 salía
a favor de `ppo_regime` (ΔSharpe +1,745, p = 0,0406). **Esa afirmación no se sostiene** y la retiro.
El contraste se hizo sobre la *serie media de las cinco semillas*, que trata la media como si fuera
una observación y descarta la varianza entre semillas — justo la fuente de incertidumbre que
domina aquí. Tomando la semilla como unidad aleatoria:

```
Sharpe por semilla  regime  : -2,43  -6,76  -4,65  -3,42  -5,26
Sharpe por semilla  backbone: -4,83  -3,91  -6,97  -4,60  -3,49
diferencia media +0,253 · sd entre semillas 2,421
IC 95 % bootstrap sobre semillas: [-1,62, +2,12]  -> INCLUYE CERO
```

Codex lo señaló de forma independiente con un modelo jerárquico, IC `[-2,08, +4,46]`: distinto
método, mismo veredicto. **No hay evidencia de que los posteriores de régimen aporten nada**, y
con cinco semillas y esta dispersión no la habría aunque aportaran. Queda como indecidible, no
como negativo.

Era la única comparación que salía a favor de algo en todo el experimento, lo que la hacía
exactamente la que más escrutinio merecía.

**Compuerta sintética: cerrada 2026-09-12.** Codex ejecutó S1–S4 completas sobre la fixture
definitiva —la que desactiva comisión y slippage en S2, no sólo el spread— con **5/5 semillas en
train y en unseen**, protocolo `SHA 33873c610252bb674879bb89e89272875d4d67b24062f8b9d4e62edac8fc21e9`.
**La receta sigue siendo exactamente `flat_init_no_turn`**: no hubo búsqueda de cuál pasa primero,
fue la comprobación de una receta pre-congelada en 20 corridas únicas. Añadió además un replay de
20 checkpoints, 2.000 sesiones y 118.000 decisiones con **diferencia 0**.

Eso **cierra** el párrafo siguiente, que se conserva porque describe un estado real del programa
y porque la corrección vino de fuera, no de mí.

**Caveat de runtime, que sí sigue abierto.** Misma receta nominal no certifica automáticamente
las diez corridas de mercado: se ejecutaron con otro intérprete y **mis artefactos no graban las
versiones de librería**. Lo que sí consta: el proceso de entrenamiento fue
`…\Programs\Python\Python312\python.exe`, que hoy reporta Python 3.12.2 · SB3 2.9.0 ·
torch 2.14.0 · gymnasium 1.3.0 — el mismo entorno que declara el bundle de Codex. Es una
**observación del proceso en vivo, no un dato grabado**, y las versiones pueden haber cambiado
desde entonces. El manifiesto por corrida de Codex (`library_versions`) cierra esta brecha para
lo que venga; mis diez corridas no la tienen.

**Cobertura real de la compuerta sintética (histórico, ya resuelto).** Los artefactos S2 y S3 que abrieron la compuerta se
produjeron el 2026-09-11 a las 10:54 y 11:27, **siete horas antes** del commit `989da5d7` que
corrigió que ambas fixtures fueran byte a byte idénticas. La compuerta verificó por tanto **tres
condiciones, no cuatro**: S2 («¿aprende la señal siquiera, sin coste?») nunca se probó de verdad,
fue S3 corrido dos veces. La pregunta de S2 queda respondida *implícitamente* por S3 —no se puede
operar cuando el alfa supera el coste sin haber aprendido la señal— pero la afirmación de cobertura
era falsa. S2 se está re-corriendo con la fixture corregida; el resultado se añadirá aquí tal como
salga. Lo detectó Codex en su réplica.

**Declaración de búsqueda**: la receta `flat_init_no_turn` se eligió tras probar **ocho sondas**
sobre fixtures sintéticas. No cobra trial de mercado —las fixtures no son USD/COP— pero es una
búsqueda, no un prior económico, y quien lea esto tiene derecho a saberlo.
