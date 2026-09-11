---
kind: as-built
status: PARTIAL
contract: CTR-RESEARCH-RESULTS-001
version: 1.1.0
last_verified: 2026-09-11
supersedes: []
code_anchors:
  - scripts/analysis/thesis_baselines.py
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/thesis_statistics.py
  - scripts/analysis/thesis_open_holdout.py
  - scripts/presentation/generar_resultados_y_figuras.py
  - src/research/inference.py
  - src/research/session_gym.py
---

# RESULTADOS Y CONCLUSIONES — tesis USD/COP (brazo PPO)

> Documento maestro: [`06-tesis-rl-llm-hibrido.md`](06-tesis-rl-llm-hibrido.md) ·
> Pre-registro firmado: [`06-PRE-REGISTRATION.md`](06-PRE-REGISTRATION.md) ·
> Registro de trials: [`HYPOTHESIS-REGISTRY`](../assets/usdcop/HYPOTHESIS-REGISTRY.md) ·
> Log: [`EXPERIMENT_LOG`](../../experiments/EXPERIMENT_LOG.md)
>
> **Ninguna cifra de este documento está escrita a mano.** Todas salen de
> `outputs/thesis/statistics_<bloque>.json` y de las 10 corridas en `data/thesis/ppo/`,
> generadas por `scripts/presentation/generar_resultados_y_figuras.py` (§14 del plan).

---

## CORRIGENDUM (2026-09-11, v1.1.0) — leer antes que nada

> **Este documento se publicó el 2026-08-25 con defectos de implementación y de redacción
> estadística.** Dos auditorías independientes los encontraron:
> [auditoría Codex](../../../docs/analysis/exp-tesis-rl-01-auditoria-2026-09-10.md) (con
> [evidencia hasheada](../../../docs/analysis/exp-tesis-rl-01-evidence-2026-09-10.json) y
> diagnóstico reproducible `scripts/diagnostics/audit_thesis_rl_integrity.py`) y la revisión
> Claude del mismo día. **Cada cifra de esta sección sale de esa evidencia**, no de una
> relectura del texto anterior. Las secciones §0-§8 se conservan **sin reescribir** para que
> la corrección sea auditable contra lo que efectivamente se publicó; donde contradigan a
> esta sección, **manda esta sección**.

### Lo que se sostiene

**El rechazo económico.** Las políticas PPO evaluadas pierden frente a la abstención bajo el
modelo de costos especificado, en los dos bloques, con las diez corridas netas negativas y
supervivencia nula al doble de costos. Re-precio sobre las **mismas acciones publicadas**
(`net_d(k) = gross_d − k·cost_d`, compuesto):

| Configuración | Costos ×1 | ×2 | ×3 |
|---|---:|---:|---:|
| `ppo_regime` | −54,8658 % | −84,5866 % | −94,7491 % |
| `ppo_backbone` | −56,4364 % | −86,1460 % | −95,6075 % |

**La causa contable inmediata:** los costos acumulados superan al bruto. Eso está medido.

### Lo que NO se sostiene y queda retirado

**1. «El agente aprende una política con señal real» (§0, §5b.1).** Retirado como afirmación.
Tres razones independientes, cualquiera de ellas suficiente:

- **Fuga macro (P0, comprobada).** `attach_macro_features` resolvía `merge_asof(backward)`
  sobre la fecha de sesión: el valor fechado `d` —cierre de DXY/Brent publicado *después* del
  cierre colombiano de las 12:55— entraba en la observación de las 08:00 de ese mismo día.
  Perturbar la fila macro del 2023-06-15 cambiaba el contexto del agente en `max_abs = 0,0953`
  (`causality_gate_pass = False`). El nombre `*_ret_prev` no correspondía al valor calculado.
- **Bruto no pre-registrado.** El primario registrado era el ΔSharpe **neto** contra
  `always_flat`; el bruto apareció después del rechazo. Es exploratorio por construcción.
- **Es una media de semillas, no una política.** Ver punto 3.

**2. Suma aritmética presentada junto a compuesto (§0).** El «+27,95 % bruto» era **suma de
retornos diarios** mientras el «−54,87 % neto» era **compuesto**. Comparables (compuestos):

| Hold-out, media de 5 semillas | Bruto suma | **Bruto compuesto** | **Neto compuesto** |
|---|---:|---:|---:|
| `ppo_regime` | +27,9497 % | **+31,8449 %** | **−54,8658 %** |
| `ppo_backbone` | +31,5108 % | **+36,5743 %** | **−56,4364 %** |

**3. «Sharpe bruto +2,23, IC excluye cero, 10/10 semillas positivas» (§0).** El +2,2254 es el
Sharpe de la **serie media de cinco semillas**, que es una cartera de cinco políticas, no la
semilla típica. Por semilla (hold-out, bruto):

| Semilla | 42 | 123 | 456 | 789 | 1337 |
|---|---:|---:|---:|---:|---:|
| `ppo_regime` compuesto | +58,72 % | +26,15 % | **+0,69 %** | +27,64 % | +51,02 % |
| `ppo_regime` Sharpe | 2,011 | 1,245 | **0,072** | 1,266 | 2,156 |
| `ppo_backbone` compuesto | +38,79 % | +71,04 % | **−0,02 %** | +23,84 % | +57,30 % |

Y el «10/10 positivas» **es falso en compuesto**: `ppo_backbone` semilla 456 da
**−0,0211 %**. La diferencia frente a su suma aritmética (+1,269 %) es arrastre por
volatilidad, no un error de cálculo.

**4. «ρ cayó de 0,97 a 0,58» (§0, §3, §6.5, §7).** **Nunca ocurrió.** Los valores
recalculados y los propios JSON originales dan **0,5853 en selección** y **0,5821 en
hold-out**. Queda retirada toda la explicación de pérdida de potencia o de efecto del refit
que se apoyaba en esa caída. H2 en hold-out: ΔSR = 0,4319, IC [−0,8828; 1,8606], p = 0,5358
— indecidible por potencia insuficiente **en ambos bloques**, no por un desvanecimiento.

**5. `p < 0,0001` (§0, §2, §2b).** Es resolución de Monte Carlo, no precisión medida. El dato
honesto: **0 excedencias en 10 000 réplicas** del bootstrap estacionario. El Sharpe de
`always_flat` es una convención (σ = 0), no una medición.

**6. «La política rentable no existe dentro del espacio de acción» (§4).** Falso: `w = 0` está
en el espacio y obtiene exactamente 0 %. Lo correcto es **«esta receta de PPO no la
encontró»**, y los modelos refit pierden incluso dentro de su propio conjunto de
entrenamiento. Es fallo del optimizador y/o del objetivo, no una propiedad demostrada del
mercado.

> **Medido el 2026-09-11, y es más fuerte que lo anterior.** Se corrió la misma receta sobre
> una serie **sintética de ruido iid con costo positivo**, donde la política óptima es
> demostrablemente no operar. Las cinco semillas del protocolo, 100.000 pasos:
>
> | Semilla | 42 | 123 | 456 | 789 | 1337 |
> |---|---:|---:|---:|---:|---:|
> | Exposición media | 0,527 | 0,966 | 0,485 | 0,985 | 0,968 |
>
> **0 de 5 se quedan planas.** El agente opera con la mitad o la totalidad del capital sobre
> datos **sin señal alguna**, pagando costo en cada cambio. Aquí no hay mercado al que culpar:
> los datos son ruido generado. **La receta no converge a la solución trivial ni cuando esa
> solución es la única correcta**, así que cualquier conclusión sobre el intradía de USD/COP
> obtenida con ella describe al optimizador tanto como al activo. Evidencia:
> `outputs/thesis-repair/sanity_S1_protocol.json`; registro en
> [`EXPERIMENT_LOG`](../../experiments/EXPERIMENT_LOG.md) § EXP-TESIS-RL-02-SANITY-S1.

**7. «El alfa vive en la alta frecuencia» (§5b.3).** La descomposición dice lo contrario:
**+48,9078 puntos** provienen del signo de la exposición **media diaria** y **−20,9582
puntos** son residuo de timing intradía. Además esa atribución es **retrospectiva**: la
exposición media incorpora acciones posteriores al inicio del día, así que no describe una
señal disponible a las 08:00.

**8. PBO 0,116 / 0,2113.** Reproducen, pero su alcance es otro: el CSCV se construyó sobre
**diez políticas por semilla dentro de un bloque**, así que mide selección hipotética entre
semillas, no el procedimiento candidato/OOF que la regla pre-registrada describía. Invocar
esa regla sobre este estadístico fue un error de aplicación.

**9. Stress de costos ×2/×3 (§2b).** El publicado usaba un P&L **sintético** (exposición media
× retorno cierre-a-cierre − k·costo medio), no el de la estrategia. La tabla de arriba lo
sustituye con el re-precio sobre las posiciones reales. El veredicto no cambia; el artefacto
estaba mal etiquetado.

**10. H1 cambió de identidad.** El pre-registro fija `PPO régimen` frente a
`baseline_matched`; el documento llamó H1 al contraste frente a `always_flat`. **El
comparador registrado nunca se construyó.** Lo reportado es subdesempeño frente a la
abstención, no el rechazo de la H1 registrada.

**11. `N ≥ 125` trials.** Afirmación mía, no demostrada. Hay **115 documentados**; la
conciliación de cadencias y sensibilidades está pendiente y se hace en el registro, no aquí.

### Defectos de implementación encontrados (todos corregidos y con test)

| Defecto | Efecto | Estado |
|---|---|---|
| Macro del mismo día en la observación | Fuga de disponibilidad temporal | Corregido: regla `available_at` en `config/research/macro_availability.yaml`; gate `causality_gate = True` |
| Reward omitía el costo de liquidación terminal | El agente optimizaba un juego con salida gratis y se puntuaba en otro que la cobra | Corregido en `session_gym.py` + test de paridad invertido |
| Cierre terminal valorado con `c58`/`σ58` | Precio equivocado en el paso que la cronología declara en `c59` | Corregido en `session_env.py`/`cost_model.py` |
| 82,41 % de barras con O=H=L=C (100 % en todo el desarrollo) | `parkinson_12` y `garman_klass_12` constantes en train y saturando el clip en el hold-out | Excluidas en el schema v2 (39 → 37 features) |
| Ventanas cruzando el salto nocturno | 1 378 sesiones con `logret_1` de apertura distinto de cero, contra §6.4 | Corregido: ventanas intra-sesión |
| Máscara sin reglas §6.3 ni festivos de EE. UU. | Sesiones que la spec excluía entraban | Corregido; `train_valid` separado de `valid` |
| Caché del dataset sin invalidar por código/contenido | Un arreglo semántico podía cargar el pickle viejo | Corregido: identidad por sha256 |
| Fuentes macro sin identidad certificada | Brent mezcla futuros y parche spot; DGS2 de Investing no coincide con FRED; DXY admite fallback a otro índice | Declarado en `macro_availability.yaml`; reconciliación pendiente |
| Ledger forward con sellado falso | `sealed_before_open = True` hardcodeado y `build_live_spec` exigiendo 60 barras | Corregido: specs parciales y metadatos de sellado por barra |

### Lo que esta corrección **no** autoriza a decir

No se ha demostrado que USD/COP intradía sea imposible de negociar con beneficio, ni que
corregir estos defectos produzca rentabilidad. Tampoco se puede asignar qué porcentaje de la
pérdida publicada corresponde a cada defecto: para eso hace falta el reentrenamiento sin
fugas, que es un experimento nuevo con sus propios trials. **Mejorar los datos puede incluso
reducir el bruto**, porque parte de él venía de la fuga.

Y una regla que esta corrección no levanta: **selección 2023 y hold-out 2024-26 ya se
miraron, y motivaron estas correcciones.** Cualquier replay de la versión corregida sobre
esos bloques es **diagnóstico retrospectivo**. El juez confirmatorio de la versión v2 es el
carril **forward**, desde su fecha de congelación.

### Texto defendible mientras tanto

> En los artefactos auditados, las políticas PPO evaluadas presentan rendimiento neto
> negativo frente a la abstención bajo el modelo de costos especificado. La atribución del
> rendimiento bruto a una señal negociable no está establecida, por problemas de
> disponibilidad temporal en las entradas macro, de representación de la serie de precios y
> de convención estadística en el reporte. El programa de reparación y re-evaluación está
> especificado en el pre-registro v3.

> Programa de reparación: [`06-PRE-REGISTRATION-v3.md`](06-PRE-REGISTRATION-v3.md) ·
> Backlog: [`BL-48`](backlog/BL-48-costos-ejecucion-intradia.md),
> [`BL-49`](backlog/BL-49-tests-2-y-14-tesis.md),
> [`BL-50`](backlog/BL-50-reparacion-tesis-rl.md)

---

## 0. El resultado, en una frase

**El agente aprende una política con señal real, y el costo de ejecutarla la anula cuatro
veces. La restricción que ata no es la predicción: es la ejecución.**

El rechazo se mantiene —PPO no bate a `always_flat`, y la diferencia es decidible en contra—
pero ahora tiene mecanismo, y el mecanismo cambia lo que hay que concluir.

| Hold-out (584 sesiones) | `ppo_regime` | `ppo_backbone` |
|---|---|---|
| **BRUTO** (costo cero, **cota superior inalcanzable**) | **+27,95%** | **+31,51%** |
| Costos de ejecución | 107,1% | 114,3% |
| Neto | −79,2% | −82,7% |
| Sharpe bruto · IC 95% | +2,23 · [+1,21, +3,29] | +2,37 · [+1,33, +3,43] |

El IC del Sharpe bruto **excluye el cero** en ambas configuraciones, y **10 de 10 corridas dan
bruto positivo**. La señal no es ruido.

**El bruto exige costo cero, que no existe.** Es una cota superior contrafactual; no es un
retorno, no es alcanzable, y no es un claim de edge. Se etiqueta así en toda tabla donde
aparece.

El rechazo, que no cambia. IC del 95% que excluye el cero por un margen amplio en los dos
bloques:

| Bloque | n | ppo_regime | always_flat | ΔSharpe | p |
|---|---|---|---|---|---|
| Selección 2023 | 234 | −29,85% | **0,00%** | −4,640 | <0,0001 |
| **Hold-out 2024-26** | **584** | **−54,87%** | **0,00%** | **−6,215** | **<0,0001** |

0 de 10 semillas positivas en cada bloque, DSR 0,0000 (deflactado con los **115** trials
del activo, no con los 111 heredados: los 2 que cobró esta tesis también deflactan su propio
claim, y los 2 de la rama forward posterior se suman por §2 de la constitución — **el DSR da
0,0000 con 113 y con 115**, así que el veredicto no depende de ese conteo), y muerte al doble
de costos.
La única hipótesis confirmatoria (H2) resultó **decidible en selección e INDECIDIBLE en el
hold-out** — y no se puede afirmar que el efecto «se desvaneciera»: la correlación entre brazos
cayó de 0,97 a 0,58 y con ella la potencia del contraste pareado (§3). PBO del hold-out 0,211,
por encima del umbral 0,20 pre-registrado.

---

## 1. El listón, y por qué no es el que uno esperaría

Lo primero que produjo este trabajo no fue un modelo sino una **medición del terreno**, y
condiciona todo lo demás.

Con el contrato de costos de §9.3 —spread esperado del régimen + 0,5 pips de comisión por lado
+ slippage proporcional a la volatilidad—, una política acotada por sesión paga en el hold-out
una media de **4,81 pips por sesión**. Sobre 584 sesiones eso es **71,4% del nocional**.

| Baseline (hold-out, 584 sesiones) | Ops | Retorno | Anual | Sharpe | IC 95% | Costos |
|---|---|---|---|---|---|---|
| B1 pasivo (overnight) | 2 | −21,3% | −8,7% | −0,69 | [−1,83, **+0,45**] | 0,21% |
| B1 sesión (1× intradía) | 584 | −56,1% | −26,8% | −3,00 | [−4,17, −1,83] | **71,4%** |
| NULL-A (corto 1×) | 584 | −46,9% | −21,3% | −2,30 | [−3,43, −1,17] | 71,4% |
| **always-flat** | **0** | **0,0%** | — | — | — | **0%** |
| random | 584 | −100,0% | −99,9% | −41,0 | — | 1832% |

Tres consecuencias:

1. **El rival a batir es `always_flat` = 0%.** El intradía de este par es de suma negativa
   *antes* de que entre ningún skill. El agente no compite por ganar más: compite contra **no
   operar**.
2. **Ni B1 ni NULL-A sobreviven al stress ×2** ⇒ REJECT por constitución §3.4. El baseline
   «obvio» tampoco es una estrategia.
3. **El IC del buy&hold pasivo cruza el cero.** Con 584 sesiones ni siquiera el benchmark
   pasivo es distinguible de cero — eso acota lo que este trabajo puede afirmar sobre
   diferencias finas, y decirlo forma parte del resultado.

---

## 2. Bloque de selección (2023, n = 234) — el veredicto

| Configuración | Retorno | Sharpe | IC 95% | MaxDD | \|exp\| | Costos |
|---|---|---|---|---|---|---|
| **always_flat** | **0,00%** | — | — | 0,00% | 0 | 0% |
| NULL_A corto 1× | −12,47% | −1,14 | [−2,72, +0,43] | −16,64% | 1,00 | — |
| B1 pasivo | −20,6% | — | — | — | 1,00 | 0,2% |
| ppo_regime (media 5 semillas) | **−29,85%** | **−4,64** | [−6,79, −2,80] | −30,90% | 0,86 | 33-41% |
| B1 sesión 1× | −34,7% | — | — | — | 1,00 | — |
| ppo_backbone (media 5 semillas) | **−35,43%** | **−6,76** | [−8,71, −5,13] | −35,50% | 0,75 | 42-47% |

### Contrastes pareados (bootstrap estacionario, 10.000 réplicas, bloques 5-20)

| Contraste | ΔSharpe | IC 95% | p | ρ | Veredicto |
|---|---|---|---|---|---|
| **H2**: ppo_regime vs ppo_backbone | **+2,120** | [+0,238, +4,087] | **0,0276** | 0,97 | **DECIDIBLE a favor** |
| **H1**: ppo_regime vs always_flat | −4,640 | [−6,793, −2,796] | <0,0001 | — | **DECIDIBLE EN CONTRA** |
| ppo_regime vs B1 pasivo | −3,196 | [−5,080, −1,290] | 0,0014 | — | DECIDIBLE en contra |
| ppo_backbone vs always_flat | −6,760 | [−8,714, −5,134] | <0,0001 | — | DECIDIBLE en contra |

### Gates de la constitución

| Gate | Resultado |
|---|---|
| ≥3/5 semillas positivas por configuración | **0/5 y 0/5** |
| DSR > 0,95 (n_trials = **115** del activo) | **0,0000** en ambas |
| PBO < 0,20 | 0,116 — pasa, pero es irrelevante: no hay ganador que preservar |
| Stress de costos ×2 | **MUERE** en ambas ⇒ REJECT |
| B1′ (exposición emparejada 0,86×) | −16,61%/año — tampoco rescata nada |

---

## 2b. Bloque de HOLD-OUT (2024-01-02 → 2026-08-24, n = 584) — apertura única

> Abierto el **2026-08-25 16:43 UTC**, una sola vez, con el pre-registro firmado.
> Constancia: `outputs/thesis/holdout_opening.json` (máscara `f6958be1b59e9767`, modelos
> `refit`, 111 trials heredados). Los gates de la Regla B habían devuelto código 2 hasta
> ese momento.

| Estrategia | Retorno | Sharpe | IC 95% | MaxDD |
|---|---|---|---|---|
| **always_flat** | **0,00%** | — | — | 0,00% |
| B1 pasivo | −21,3% | −0,69 | [−1,83, +0,45] | — |
| NULL_A corto 1× | −46,95% | −2,30 | [−3,50, −1,16] | −50,14% |
| **ppo_regime** (media 5) | **−54,87%** | **−6,21** | [−7,54, −4,96] | −54,84% |
| B1 sesión 1× | −56,04% | −3,00 | [−4,21, −1,77] | −57,03% |
| **ppo_backbone** (media 5) | **−56,44%** | **−6,65** | [−8,52, −5,02] | −56,53% |

| Contraste | ΔSharpe | IC 95% | p | Veredicto |
|---|---|---|---|---|
| ppo_regime vs **always_flat** | −6,215 | [−7,537, −4,963] | <0,0001 | **DECIDIBLE EN CONTRA** |
| ppo_regime vs B1 pasivo | −5,523 | [−7,100, −4,115] | <0,0001 | DECIDIBLE en contra |
| ppo_backbone vs always_flat | −6,647 | [−8,515, −5,019] | <0,0001 | DECIDIBLE en contra |
| **H2**: regime vs backbone | +0,432 | [−0,883, +1,861] | 0,55 | **INDECIDIBLE** |

| Gate | Resultado |
|---|---|
| ≥3/5 semillas positivas | **0/5 y 0/5** |
| DSR > 0,95 (n_trials **115**) | **0,0000** en ambas |
| PBO < 0,20 | **0,211 — POR ENCIMA** del umbral pre-registrado |
| Stress ×2 | **MUERE** en ambas ⇒ REJECT |
| B1′ (exposición 0,71×) | −6,05%/año |

**El veredicto del hold-out coincide con el de selección y lo agrava**: las dos
configuraciones pierden más de la mitad del capital, con costos del **89-119% del nocional**
sobre 584 sesiones. El compromiso firmado en el pre-registro §9 —«si el hold-out saliera
positivo, eso NO revierte el rechazo»— no ha tenido que invocarse: salió en la misma dirección.

> **Verificación cruzada que conviene señalar.** Los baselines de esta tabla los recomputa
> `thesis_statistics.py` desde los `SessionSpec`, mientras que los de la Fase E los produjo
> `thesis_baselines.py` por un camino distinto. Coinciden al segundo decimal (B1 sesión
> −56,04% vs −56,1%; NULL-A −46,95% vs −46,9%). Dos implementaciones independientes sobre el
> mismo contrato de costos dan el mismo número.

---

## 3. H2: el agente SÍ usa el régimen, pero la ventaja no sobrevive al hold-out

H2 es la única hipótesis confirmatoria. Sale **decidible a favor en selección** (ΔSharpe
+2,120, p = 0,0276) e **indecidible en el hold-out** (+0,432, IC [−0,883, +1,861]). Lo que
sigue explica el mecanismo, que sí es sólido, y por qué eso no basta para sostener la
hipótesis.

La figura 5 muestra el mecanismo sin ambigüedad. `ppo_regime` **condiciona su comportamiento
al régimen**: en `intermedio_2` va largo +1 el 86% de las barras, y en `shock` se pone corto
(50% en −1, 22% en −0,5). `ppo_backbone`, sin esa información, reparte sus acciones casi
igual en los cuatro regímenes.

Es decir: la señal de régimen **es real y el agente la usa**. Lo que ocurre es que la usa para
perder menos, no para ganar. Ambos brazos quedan por debajo de no operar, y la familia de Holm
tiene un solo miembro, así que no hay corrección por multiplicidad que aplicar.

> ### H2 NO REPLICA en el hold-out
>
> | Bloque | n | ΔSharpe (regime − backbone) | IC 95% | p | Veredicto |
> |---|---|---|---|---|---|
> | Selección 2023 | 234 | **+2,120** | [+0,238, +4,087] | 0,0276 | DECIDIBLE |
> | **Hold-out 2024-26** | **584** | **+0,432** | **[−0,883, +1,861]** | 0,55 | **INDECIDIBLE** |
>
> **Y una corrección que hay que hacer explícita.** La lectura tentadora es «el efecto se
> desvaneció». Los datos no permiten afirmarlo, porque **la correlación entre los dos brazos
> cayó de ρ = 0,97 en selección a ρ = 0,58 en el hold-out**, y el bootstrap pareado pierde
> potencia justamente cuando esa correlación baja. Medido por simulación a n = 584:
>
> | ρ entre brazos | ΔSharpe 0,4 | 0,8 | 1,2 | 2,0 |
> |---|---|---|---|---|
> | 0,92 | 25% | 75% | 95% | 100% |
> | **0,58** | 20% | 20% | **35%** | **60%** |
>
> Con la correlación observada en el hold-out, el contraste **no habría resuelto con fiabilidad
> ni siquiera el +2,12 de selección**. Así que hay dos causas que estos datos no separan: que
> el efecto sea menor, o que el test perdiera potencia. Lo único que se puede afirmar es que
> **H2 queda INDECIDIBLE en el hold-out** (§11.1) — que es distinto de «no hay efecto» y
> distinto de «el efecto desapareció».
>
> El **PBO del hold-out sale 0,211, por encima del umbral 0,20 pre-registrado**, lo que apunta
> en la misma dirección de cautela: una ventaja seleccionada en un bloque que no se confirma
> en el otro.
>
> ### Por qué probablemente no replica: el régimen es casi redundante
>
> La figura 5 del **hold-out** muestra algo que la de selección no dejaba ver: **`ppo_backbone`
> también condiciona su comportamiento al régimen** — 0,70 de barras en corto durante `shock`
> frente a 0,43 en `calmo`— **pese a recibir los cuatro posteriores puestos a cero**.
>
> No es un error de la ablación: es que puede inferirlo. El HMM se ajusta sobre un vector de
> observación dominado por volatilidad (rv, log-rv, ATR normalizado, rango/ATR), y el vector
> del agente ya incluye **siete features de volatilidad** (`rv_12`, `rv_78`, `rv_ratio`,
> `atr_14`, `atr_norm`, `parkinson_12`, `garman_klass_12`) que el backbone sí ve. El posterior
> del HMM es, en buena medida, **una función de información que el agente ya tenía**.
>
> Eso da una explicación concreta de por qué la ventaja no persiste: lo que H2 mide no es «¿el
> régimen informa?» sino «¿el régimen informa *por encima* de la volatilidad cruda?», y la
> respuesta parece ser «poco». En selección esa diferencia pequeña resultó decidible; en el
> hold-out, con menos correlación entre brazos, no.
>
> Es una hipótesis consistente con lo medido, no una conclusión demostrada: separarla
> requeriría una ablación adicional —quitar las features de volatilidad al backbone— que este
> trabajo no ejecutó y que cobraría su propio trial.

> **Conclusión honesta sobre H2**: la información de régimen cambia el comportamiento del
> agente de forma medible y consistente (figura 5), pero **no hay evidencia de que esa
> diferencia de comportamiento se traduzca en una diferencia de desempeño sostenida**. El
> resultado de selección no se presenta como hallazgo confirmado.

> **Matiz obligado sobre la palabra «régimen»** (§8.3). Las persistencias diagonales del HMM
> son 0,405 · 0,390 · 0,454 · 0,484, o sea duraciones medias de **1,6 a 1,9 días**. Lo que el
> modelo encuentra son **tipos de día**, no regímenes macro de semanas o meses. Para alimentar
> el spread esperado del día siguiente eso es exactamente lo adecuado; lo que no se puede
> escribir es que el agente «reconoce regímenes de mercado».

---

## 4. El hallazgo del refit: ni siquiera gana sobre datos que vio

El pre-registro compromete un refit sobre desarrollo + selección antes de abrir el hold-out.
Al ejecutarlo aparece un dato que refuerza el diagnóstico:

**Los modelos refit siguen perdiendo en 2023, con 2023 dentro de su conjunto de
entrenamiento** (−8,6% a −17,9% según semilla, `ppo_regime`).

No es un problema de generalización: es que **la política rentable no existe dentro del
espacio de acción y el contrato de costos dados**. Un agente que no logra ser rentable
in-sample no está sobreajustando; está topando con una restricción estructural.

---

## 5. Por qué falla — el mecanismo, no la interpretación

Los agentes mantienen |exposición| ≈ 0,75-0,86 y ejecutan **433-813 cambios en 234 sesiones**,
lo que consume **33-47% del nocional en costos**. A 4,81 pips por sesión, cualquier política
que opere necesita un edge bruto superior a ~0,12% diario solo para llegar a cero.

Ese umbral no es alcanzable con la información disponible: el retorno diario de USD/COP en la
sesión tiene una desviación del orden del 0,4%, así que se pediría una ratio señal-ruido que
ningún resultado de este repositorio —ni el track de producción H5, cuyo modelo tiene R² < 0—
sugiere que exista.

---

## 5b. Descomposición: por qué perdió (CTR-RESEARCH-DECOMP-001)

> **0 trials.** Es descomposición descriptiva de un resultado ya obtenido, no selección de una
> variante nueva; el universo sigue cerrado y el hold-out abierto una sola vez. Mismo criterio
> con el que no se cobró la selección de K del HMM.
>
> **Fiabilidad**: el replay reproduce la evaluación original con delta **0,00e+00** en las 10
> corridas. Si no cuadrara, estas cifras vendrían de un modelo distinto del publicado.

### 5b.1 Sí hay señal, y es pequeña

Sharpe bruto **+2,23** (IC [+1,21, +3,29]), 10/10 corridas positivas. El agente predice algo.

Nótese que el Sharpe bruto del bloque de **selección** sale +5,49 — y **no se reporta como
resultado**: para los modelos refit, selección es in-sample. El único bruto honesto es el del
hold-out.

### 5b.2 El alfa no cubre ni la comisión

| | `ppo_regime` | `ppo_backbone` |
|---|---|---|
| Alfa por operación | 0,67 pips | 0,72 pips |
| Costo por operación | 2,55 pips | 2,56 pips |
| **Spread de break-even `s*`** | **−0,29 pips** | **−0,21 pips** |

**`s*` es negativo**, y eso es lo más importante de todo el trabajo: significa que **ni con un
spread de cero** el alfa cubriría la comisión de 0,5 pips por lado. La estrategia no es viable
a **ningún** spread.

Y con eso **desaparece la amenaza principal a la validez de la tesis**. `SPREAD_PIPS_BY_LEVEL
= (2.0, 3.0, 6.0)` (`src/research/regime_hmm.py:71`) es una constante declarada en §8.4 que
**nunca se midió** —el seed solo tiene OHLCV, no hay bid/ask en el repositorio— y era razonable
temer que la conclusión dependiera de ella. **No depende**: se sostiene en el límite de spread
cero, y por tanto también bajo el supuesto de producción (`src/config/backtest_ssot.py:34-35`:
2,5 bps + 1 bps ≈ 0,89 pips/lado), que es 2,5× más barato que el de la tesis.

### 5b.3 La frecuencia no es el culpable único

Re-scoring de las **mismas decisiones** del agente, muestreadas cada `k` barras (media de 5
semillas, `ppo_regime`):

| k | decisiones/sesión | BRUTO | Costo | Neto |
|---|---|---|---|---|
| 1 | 59 | **+27,95%** | 107,1% | −79,2% |
| 5 | 12 | +15,86% | 93,2% | −77,3% |
| 15 | 4 | +13,06% | 78,6% | −65,5% |
| 30 | 2 | +9,67% | 63,8% | −54,1% |
| 59 | 1 | **−5,95%** | 27,3% | −33,2% |

El neto mejora al bajar la frecuencia (−79% → −33%), **pero el bruto se desploma con él**, y a
una decisión por sesión ya es **negativo**.

**El alfa vive en la alta frecuencia igual que el costo.** Bajar la frecuencia no rescata la
estrategia: la mata por el otro lado. Esto **matiza la línea 1 de `BL-48`**: reducir la
frecuencia por sí sola no basta, haría falta que la reducción de costo no arrastrara la señal
— y eso exige cambiar la ejecución, no solo la cadencia.

> **Limitación**: submuestrear la senda de un agente entrenado a 59 decisiones **no es** lo
> mismo que entrenar uno a `k`. La curva acota; no demuestra qué haría un agente entrenado a
> esa frecuencia, que sería una hipótesis nueva con su propio trial.

### 5b.4 La paradoja del always-flat

`w = 0` está en el espacio de acción congelado (§2, decisión 4). **La política óptima del
bloque —no operar, 0,00%— era alcanzable, y el agente no la encontró**: terminó en −79%.

Distribución de acciones de `ppo_regime` en el hold-out: `−1,0` el 34,8% de las barras, `+1,0`
el 24,2%, y **`0,0` solo el 16,7%**. De 2.920 sesiones-corrida, apenas **68 enteras en flat**.

Y el plan temía **exactamente lo contrario**: §10.1 lista *«Colapso a flat»* entre los
criterios de trial degenerado y fija `ent_coef = 0.01` para evitarlo, con el comentario *«con
costos penalizados el agente colapsa a "siempre neutral" si la exploración es baja»*. **Ocurrió
la inversión del riesgo previsto.** Es un hallazgo sobre RL con recompensa densa y negativa, no
sobre el mercado.

---

## 6. Conclusiones

1. **No operar es la estrategia.** Es el compromiso firmado antes de ver resultados
   (pre-registro §8.3) y la versión concreta de «el baseline ES la estrategia»
   (constitución §3). `always_flat` bate a las dos configuraciones de PPO, a las dos versiones
   de B1 y a NULL-A, con diferencia decidible.

2. **El agente SÍ aprende: la restricción es la ejecución, no la predicción.** Bruto +27,95%,
   Sharpe bruto +2,23 con IC que excluye el cero, 10/10 corridas positivas. Decir «PPO no
   funciona» sería falso; lo que no funciona es **ejecutar** lo que PPO aprende.

3. **El alfa no cubre ni la comisión, así que ningún supuesto de costo lo salva.** El spread de
   break-even es **negativo** (−0,29 pips): ni a spread cero. Esto es lo que hace la conclusión
   **robusta al supuesto de costos que nunca se midió**, que era la mayor debilidad del trabajo.

4. **Bajar la frecuencia no basta**, y esto corrige la intuición de partida: el alfa vive en la
   alta frecuencia igual que el costo. A una decisión por sesión el bruto ya es negativo
   (−5,95%). Reducir cadencia mejora el neto destruyendo la señal, no preservándola.

5. **El agente usa el régimen de forma medible, pero H2 no se sostiene.** El cambio de
   comportamiento por régimen es nítido y consistente (figura 5), y en selección se traduce en
   una diferencia decidible de desempeño. **En el hold-out esa diferencia queda en ΔSharpe
   +0,432 y deja de ser decidible.** No se puede atribuir a que el efecto desapareciera: la
   correlación entre brazos cayó de 0,97 a 0,58 y con ρ=0,58 el contraste a n=584 solo alcanza
   ~35% de potencia frente a un ΔSharpe de 1,2. H2 se reporta como **no confirmada** — ni
   refutada ni desvanecida: **indecidible**, que es lo que §11.1 obliga a escribir.

6. **Lo que este trabajo NO demuestra**: que PPO no pueda funcionar. Demuestra que **esta
   receta** (hiperparámetros congelados de `v215b_baseline.yaml`, 300k pasos, 39 features,
   5 niveles de exposición) sobre **estos datos** contra **estos costos** no funciona. Sin HPO
   no hay evidencia sobre el mejor PPO alcanzable, y esa limitación es consecuencia declarada
   de la decisión de §15, no un descuido.

7. **El régimen del HMM es casi redundante con la volatilidad cruda.** El backbone, sin
   recibir los posteriores, reproduce el mismo condicionamiento por régimen (figura 5 del
   hold-out) porque ya ve siete features de volatilidad, y el HMM se ajusta sobre volatilidad.
   La pregunta que H2 mide de hecho no es «¿el régimen informa?» sino «¿informa **por encima**
   de la volatilidad que el agente ya tiene?».

8. **Los resultados negativos se publican** (constitución §3.6, pre-registro §8.3). Son *el*
   resultado de este trabajo, no su ausencia.

### Qué haría distinto un trabajo siguiente

No son recomendaciones genéricas: cada una sale de una medición de este trabajo.

| Cambio | Porque aquí se midió |
|---|---|
| Bajar la **frecuencia** de decisión (una o dos por sesión, no 59) | 433-813 cambios consumen 33-47% del nocional; el costo escala con `|Δw|`, no con el acierto |
| **Ejecución pasiva** (limit orders) en vez de cruzar el spread | el componente de spread es ~94% del costo por operación (3,0 de 3,16 pips en el round-trip típico) |
| Penalizar el turnover **en el reward**, no solo en la contabilidad | el agente mantiene \|exposición\| ≈ 0,75-0,86: no aprendió que operar es caro |
| Ablacionar el régimen **contra un backbone sin features de volatilidad** | es la única forma de separar H2 de la redundancia descrita en el punto 5 |
| Ampliar el horizonte más allá de la sesión | §9.1 fuerza plano al cierre y eso paga un round-trip diario garantizado; B1 pasivo, que no lo paga, pierde 0,21% en costos frente al 71,4% del intradía |

---

## 6b. Aplicación de la regla de PBO pre-registrada

El PBO del hold-out es **0,211**, por encima del umbral de 0,20 que el pre-registro §4 fijó
antes de calcularlo. La regla escrita entonces se aplica ahora sin interpretación:

> *«Si PBO > 0.20: NO se modifica el universo candidato ni se retunea nada; se suspende la
> pretensión confirmatoria del trabajo; el hold-out puede abrirse ÚNICAMENTE como evaluación
> de un sistema con riesgo alto de selección, etiquetado así en TODAS las tablas.»*

Y la elección registrada ex-ante era **abrir igualmente, con etiqueta**. Por tanto:

1. **No se retunea nada.** El universo sigue siendo `{ppo_regime, ppo_backbone}` y no se añade
   ni se quita un candidato.
2. **La pretensión confirmatoria queda suspendida.** H2 no se presenta como hallazgo
   confirmado ni siquiera con el resultado favorable de selección.
3. **Etiqueta obligatoria**: todos los resultados del hold-out de este trabajo son una
   **evaluación con riesgo alto de selección**, no un juicio confirmatorio limpio.

Conviene precisar qué NO cambia con esta etiqueta. El rechazo de **H1 no depende de ella**:
un PBO alto advierte de que el *ganador* elegido puede no replicar, y aquí no hay ganador que
proteger — las dos configuraciones pierden contra `always_flat` con p < 0,0001 en los dos
bloques. La etiqueta afecta a lo que se podría haber afirmado a favor; el resultado en contra
se sostiene por sí mismo.

---

## 7. Limitaciones declaradas

| Limitación | Alcance |
|---|---|
| Sin HPO | No hay evidencia sobre el mejor PPO alcanzable; White RC y SPA quedan sin sentido (universo de 2) |
| Sin brazo LLM ni híbrido | H3 y H4 no se responden; el corpus textual no se construyó |
| 2025 dentro del hold-out ya fue mirado | 42 celdas del track H5; declarado ex-ante, DSR deflactado con el N heredado |
| «Régimen» = tipo de día | 1,6-1,9 días de duración media; la narrativa se ajusta en §8.3 |
| Grupo de volumen inexistente | 100% de barras con volumen 0; el par es OTC |
| Slippage subestimado en la barra 0 | `rv_12` no cruza sesiones; ~5% optimista, cubierto por el stress ×2/×3 |
| Tests 2 y 14 de §13 sin implementar | shuffle test y etiqueta OOF/OOS; el 7 sí se hizo, con referencia independiente en vez de `vectorbt` |
| n = 584 no resuelve diferencias finas | Mínimo detectable ΔSharpe ≈ 0,7 al 80% **a ρ=0,92**; con la ρ=0,58 del hold-out cae a ≳2 |
| El régimen es casi redundante con la volatilidad | El backbone infiere comportamiento por régimen desde sus 7 features de vol; separar ambos exigiría una ablación adicional no ejecutada |
| El diagnóstico inicial del guard macro era erróneo | Se escribió que `RangeValidator` «solo se instancia en tests»; **es falso**, sí está cableado. Los huecos reales son otros cuatro (ver la corrección en §6.5 del plan) |
| El spread nunca se midió | `SPREAD_PIPS_BY_LEVEL = (2.0, 3.0, 6.0)` es una constante declarada en §8.4; el seed solo tiene OHLCV. **Deja de ser amenaza a la validez** porque el break-even es negativo: la conclusión se sostiene incluso a spread cero (§5b.2) |
| El bruto es una cota superior inalcanzable | Exige costo cero. No es un retorno ni un claim de edge, y así se etiqueta en cada tabla |
| La curva de frecuencia acota, no demuestra | Submuestrear la senda de un agente entrenado a 59 decisiones no equivale a entrenar uno a `k` |
| Sin ensamble multi-semilla | Las hipótesis se probaron sobre la media de 5 semillas, no sobre el ensamble por voto de §10.1 |
| Manifiesto no materializado como fichero | Sus hashes viven repartidos en artefactos versionados; la firma está en el pre-registro §9 |

---

## 7b. Estado de verificación

> Cifras **medidas** el 2026-08-25 tras regenerar, no arrastradas de una corrida anterior.

| Comprobación | Resultado |
|---|---|
| Tests del carril de investigación | **80 en verde** en 10 ficheros (features, entorno, costos, paridad de motor, paridad gym, escalado, inferencia, + los 4 del carril forward) |
| Reporte de coherencia §19.4 | **7/7 en ambos bloques** (`all_ok: true`) |
| Ledger de trials encadenado | **OK** — 243 asientos, sumas exactas por activo, cadena de hashes íntegra, `usdcop: 115` |
| Regresión (`tests/regression`) | **1.976 en verde**, 0 fallos |
| Gates de conocimiento | **1.017 en verde** |
| Reproducibilidad de la regeneración | Regenerar `statistics_{selection,holdout}.json` reprodujo **todos** los Sharpe, p-valores, IC, PBO y stress bit a bit; el único delta fue `n_trials` 113→115 y su `sr0` derivado. El bootstrap de 10.000 réplicas es determinista |

**Fuera del carril de investigación**, contabilizado en vez de resumido:

| Conjunto | Resultado | Naturaleza |
|---|---|---|
| `tests/contracts` + `data_quality` + `onboarding` + 2 de raíz | **117 verde, 2 fallos** | los 2 son preexistentes: `test_observation_parity` (`Config dimension 15 != expected 20`) y `test_trading_calendar::test_empty_dataframe_handling` (`IndexError` con DataFrame vacío) |
| `tests/unit` | **6 errores de colección** | `pyspark` incompatible con Python 3.12; módulo `regime` inexistente |
| `tests/integration` | **92 fallos + 80 errores** (391 verde, 151 saltados) | `norm_stats` sin 4 de las 13 features esperadas (58 de los 80 errores salen de ahí, en `test_feature_builder` y `test_observation_parity`); módulo `train_ssot` inexistente; varias exigen Postgres/Redis |

Pase completo de ese tramo (33 min, 716 tests): **94 fallos y 80 errores, y todos salvo 2
caen dentro de `tests/integration`**. Los 2 restantes son los de raíz de la primera fila.

**Esos 2 fallos son preexistentes y verificables**: `trading_calendar.py`,
`observation_builder.py` y `config/feature_config.json` están intactos respecto al último
commit. Todos pertenecen al carril de producción RL, no a esta tesis, y **no se presentan como
verdes**.

### Un tercer defecto, este de reporte: CERRADO (2026-08-25)

`tabla_4_2_particion.md` titulaba «efectivas» al conteo posterior a la máscara de evaluación
(558 / 235 / 584). **Ese no es el `n` con el que se hizo el análisis**: construir los
`SessionSpec` descarta además las sesiones sin posterior de régimen —calentamiento del HMM— y
las que no traen 60 barras. Los `n` reales son **499 / 234 / 584**, y así los reportaban todas
las demás tablas, los IC y los bootstraps.

En development la diferencia son **59 sesiones**, y el rango real empieza el 2020-05-07, no el
2020-01-02. **El hold-out no está afectado** (584 = 584), así que el veredicto nunca estuvo en
juego: era integridad del reporte. La tabla publica ahora los tres conteos —existen, tras
máscara, usables— y el rango realmente usado.

---

### Los dos defectos de infraestructura que destapó la tesis: CERRADOS (2026-08-25)

| Defecto | Cómo se cerró |
|---|---|
| **Postgres moría por OOM** al recorrer `asset_daily_ohlcv` (2.430 chunks de 7 días para 60.542 filas) | `scripts/ops/fix_daily_hypertable_chunks.py`: **2.431 chunks → 6**, respaldo CSV verificado de las 60.542 filas y las 3 vistas dependientes recreadas desde sus definiciones capturadas. La consulta que lo mataba pasa a **0,7 s**; los 7 tests del fichero pasan |
| **Un valor macro fuera de rango llegaba a la BD** pese a tener su rango declarado | Cuarentena por fila en el **punto de escritura** (`CTR-L0-QUARANTINE-001`), no en una tarea del DAG: aparta la fila mala, la registra con su motivo y deja pasar el resto. 18 tests de regresión, incluido el replay del incidente real de Brent |

**El primer diagnóstico del segundo defecto era incorrecto** y está corregido en cinco sitios:
se escribió que `RangeValidator` «solo se instancia en tests», y es falso — sí está cableado.
Los huecos reales eran cuatro (ingesta diaria sin validar, `validate_data` que nunca lanzaba,
restore que la saltaba, y config ausente que desactivaba el guard en silencio), más un quinto
más profundo: **una violación de rango solo era error si superaba el 10% de los valores**, así
que los 59 días de Brent no habrían sido bloqueados ni por un validador perfectamente cableado.

Las líneas de trabajo que este resultado deja abiertas están registradas como
[`BL-48`](backlog/BL-48-costos-ejecucion-intradia.md) (costos de ejecución) y
[`BL-49`](backlog/BL-49-tests-2-y-14-tesis.md) (tests 2 y 14).

---

## 7c. Trabajo posterior: la rama comparativa forward (CTR-RESEARCH-FORWARD-001)

> **Exploratoria por diseño, no por modestia.** El hold-out se abrió una vez y el PBO quedó en
> 0,211 > 0,20. Un brazo que arranca hoy **no rescata H1 ni H2**, y
> `config/research/preregistration_forward.yaml` lo fija en `status: exploratory` para que el
> fichero mismo impida la afirmación más adelante. **Nada de esta sección cambia una cifra de
> las anteriores.**

El brazo LLM del plan original (§10.4) nunca fue recuperable: no hay corpus histórico —181
artículos en 12 días frente a 1.383 sesiones— y, aunque lo hubiera, un LLM que hoy lee una
noticia de 2021 ya sabe cómo terminó 2021. Eso está en los pesos y ningún filtro lo quita.

La única evaluación limpia es **prospectiva**, y se montó como una comparación de cuatro brazos
sobre **las mismas sesiones**, con **el mismo contrato de costos** y **el mismo motor de
liquidación** que produjo las tablas de este documento:

| Brazo | Decisiones/sesión | Sella | Trials |
|---|---|---|---|
| `llm_direct_fwd_v1` | 1 | 07:15 COT | +1 AT |
| `ppo_regime_fwd_k59` | 1 | 08:00 COT | +1 AT |
| `ppo_regime_fwd_k1` | 59 | 08:00 COT | 0 — evidencia forward de la política ya evaluada |
| `always_flat` | 0 | — | 0 — el listón |

Trials del activo: 113 → **115**.

**La verificación que lo ancla a este documento**: liquidar una senda del hold-out por el carril
forward reproduce **exactamente** —bruto, costo, neto y turnover— lo que registró
`decomposition_holdout.json`. Y `build_live_spec` reproduce el spec del batch con delta
**0,000e+00** en features y contexto sobre fechas de los tres bloques. Sin esas dos igualdades
el carril mediría otra cosa sin decirlo.

**Asimetría declarada**: el RL ve la barra 0 para construir su observación y el LLM no. Los dos
sellan antes de que exista `r_1`, así que los dos son causalmente limpios, pero no con la misma
información.

**Lo que no va a producir**: potencia. ~85 sesiones hasta diciembre, cuando en el hold-out ni
584 resolvieron H2. Es un **piloto de factibilidad y un artefacto de ingeniería**.

**La compuerta**: inventario de fuentes de 5 días hábiles. Día 1 (2026-08-25) dio **52
documentos pre-apertura**, pero los tres feeds viables son del mismo medio — un solo punto de
fallo con tres URL, y así queda declarado. Si la mediana cae por debajo de 1, el brazo se cierra
y se escribe.

---

## 8. Artefactos

**Tablas** (`outputs/thesis/tablas/`): 4.1 datos · 4.2 partición · 4.3 desempeño · 4.4 por
régimen · 4.5 ablación · 4.6 costos · 4.7 DSR/PBO · 4.10 semillas.

**Figuras** (`outputs/thesis/figuras/`): curvas de capital · underwater · Sharpe móvil ·
sensibilidad a costos · acciones por régimen · perfiles por semilla · particiones · Sharpe por
régimen · sesión representativa.

**Reproducción de cero**:

```bash
# 1. Dataset (en el HOST: necesita hmmlearn, que el contenedor de Airflow no tiene).
#    ~6 min la primera vez; luego se cachea.
python -c "import sys;sys.path.insert(0,'.');from src.research.dataset import load_or_build,save_portable;save_portable(load_or_build())"

# 2. Entrenamiento, en Airflow (~25 min por corrida, 6 en paralelo).
airflow dags trigger research_thesis_ppo_training                        # 10 corridas
airflow dags trigger research_thesis_ppo_training -c '{"refit": true}'   # refit F8

# 3. Estadística de selección — aquí se toma el veredicto.
python scripts/analysis/thesis_statistics.py --block selection

# 4. Hold-out. Los dos comandos salen con código 2 si el pre-registro no está firmado.
#    Se abre por configuración porque evaluar los 10 modelos son ~345.000 predicciones;
#    el script es reanudable (salta lo ya evaluado) y la constancia ACUMULA.
python scripts/analysis/thesis_open_holdout.py --models refit --config ppo_regime
python scripts/analysis/thesis_open_holdout.py --models refit --config ppo_backbone
python scripts/analysis/thesis_statistics.py --block holdout

# 5. Tablas, figuras y reporte de coherencia (--no-replay salta las 2 figuras que
#    recargan modelos y tardan ~15 min).
python scripts/presentation/generar_resultados_y_figuras.py --block selection
python scripts/presentation/generar_resultados_y_figuras.py --block holdout
```

El paso 5 devuelve código distinto de cero si alguna de las seis comprobaciones de coherencia
falla, así que sirve de gate: si las tablas y las figuras no son mutuamente consistentes, no se
publican en silencio.
