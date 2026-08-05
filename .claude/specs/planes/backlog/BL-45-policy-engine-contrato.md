---
kind: roadmap
status: PARTIAL
version: 1.2.0
last_verified: 2026-08-03
supersedes: []
code_anchors:
  - src/contracts/policy.py
  - src/contracts/policy_dsl.py
  - src/strategies/policies/gold_dynamic_exit.py
  - tests/unit/test_policy_ci_validations_gap.py
  - tests/unit/test_ma200_declarative_parity.py
  - tests/unit/test_policy_state_contract.py
  - src/contracts/signal_contract.py
  - airflow/dags/asset_pipeline_factory.py
  - config/assets/pipelines.yaml
  - .claude/rules/strategy-engines.md
---

# BL-45 — Motor de políticas: contrato + registry + factory (R1-R3)

**Fuente**: planes/05-rule-based-strategies.md §2-§7, §13 R1-R3 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
Las rule-based YA corren (MA200 SPX, votos SMA Oro, hodl BTC) pero cada una con su publisher ad-hoc; no existe Policy.evaluate() común, ni engine.type en specs, ni DSL declarativo, ni rule_trace. gold_dynamic_exit es stateful y vive FUERA del loop estándar (publisher propio) — prueba viva de la decisión abierta §15.2.

## Qué falta exactamente
R1: contracts/policy.py (required_features/validate_inputs/evaluate→StrategyDecision), strategy_decision con engine_ref discriminado, rule_trace_v1. R2: specs con engine.type|implementation.mode|feature_set_id|resample_policy_id|policy(params,resolution,fallbacks)|explainability. R3: factory con tareas genéricas (resolve_feature_snapshot→validate_policy_inputs→evaluate_policy→publish) que OMITE train cuando retrain=never; modos DECISION/FREEZE/REVALIDATE/BACKFILL; ramifica SOLO por engine.type, jamás por strategy_id. DSL declarativo = whitelist AST (greater_than/all/crosses_above/...), CERO eval/SQL/Python desde YAML. Resolver §15: DSL propio (recomendado) y firma de evaluate con contexto de ESTADO (dynexit lo exige). Las 17 validaciones CI de §11.

## Impacto frontend
Ninguno directo (R5 es BL-46).

## Dependencias
BL-13 (surface), BL-39 (feature_set), BL-28 (converge con factories FABRIC — decisión per-sleeve vs task-mapping se toma ahí).

## Verificación
Las 17 validaciones CI de §11 en verde; mismo input+policy ⇒ misma decisión (test de determinismo); spec MA200 declarativo evalúa idéntico al coded_policy actual.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/unit/test_policy_contract.py -q
verde:   219 passed

muta:    src/contracts/policy_dsl.py — desactivar la whitelist de operadores del DSL
         (ALLOWED_OPERATORS deja de filtrar)
espera:  4 failed — entre ellos `eval` y `DROP TABLE` aceptados como operadores:
         test_operator_outside_whitelist_raises,
         test_eval_operator_rejected  (`pytest.raises(ValueError, match="whitelist")`
         deja de dispararse), y los casos de código-desde-YAML
         (`when: "eval(close > ma_200 and custom_python())"`,
          `left: "__import__('os').system('id')"`)

muta-2:  spec NUEVO en config/policies/ con `operator: python_eval` y
         `__import__('os').system(...)`, re-hasheado para colar el freeze
espera-2: revienta igual — scripts/validation/validate_policy_specs.py hace
         `directory.glob("*.yaml")` en vez de leer una lista a mano (K-029), así que
         añadir un spec malicioso no lo esquiva: lo mete dentro del perímetro
```

**Historial honesto**: BL-45 es uno de los **7 que mordían de origen** (medidos contra
`92963fa9`) — no hubo defecto que cerrar el 2026-07-28. Lo que lo hace fiable, y lo que se
señaló expresamente en CLD-216, es la **segunda** mutación: no basta con que el validador
rechace lo que ya está declarado; el atacante realista **añade un spec nuevo**. Que el
validador descubra los specs por glob del directorio en vez de por lista enumerada es lo que
convierte la whitelist en una garantía y no en una convención — K-029 bien aplicada.

**Aviso de CI declarado (CLD-216)**: `fabric-contracts.yml` corre `validate_policy_specs.py`,
pero **NO** `check_policy_parity.py` (el arnés de paridad de BL-47). La whitelist del DSL sí
está vigilada por CI; la paridad de los motores migrados, no.

## Notas constitución
Regla siempre-cargada nueva: .claude/rules/strategy-engines.md (invariantes 1-9). Las reglas también cobran trials — cada variante de ventana/umbral/filtro = +1 en su familia.


## Auditoría de las 17 validaciones §11 (CLAUDE, 2026-08-03)

La brecha declarada arriba dice *"R1: contracts/policy.py (...)"* como si R1 estuviera por
hacer. **Está en gran parte construido** y esa frase induce a error: `src/contracts/policy.py`
tiene `EngineRef`, `PolicyContext`, `StrategyDecision`, el protocolo `Policy` con
`required_features`/`validate_inputs`/`evaluate`, y `POLICY_MODES` con los cuatro modos.
Existen además `rule_trace.py`, `policy_dsl.py` y `policy_version.py`.

Lo que **no** está es R3. Y de las 17 validaciones de §11, esto es lo que se aplica de verdad
— cada fila comprobada contra el código, no contra el docstring que la anuncia:

| # | Validación §11 | Estado | Dónde |
|---|---|---|---|
| 1 | `rule_based` ⇒ `retrain=never` | ✅ | `loader.py:166` |
| 2 | `rule_based` no declara `capability=train` | ✅ | `loader.py:176` |
| 3 | `rule_based` no exige `model_snapshot_id` | ✅ | `loader.py:170` |
| 4 | toda policy referencia `feature_set` y `resample_policy` | ✅ | `loader.py:181` |
| 5 | los `required_features` existen en el snapshot | ✅ | `base.py:79-81` (`validate_inputs`) |
| 6 | **ningún input supera `decision_cutoff`** | ❌ **no implementable hoy** | ver abajo |
| 7 | todo operador declarativo pertenece al whitelist | ✅ | `loader.py:222` → AST de `policy_dsl` |
| 8 | el YAML no contiene `eval`, SQL libre ni Python | ✅ | `validate_policy_specs.py::_check_text` |
| 9 | toda política tiene default/fallback explícito | ✅ | `loader.py:44` |
| 10 | conflictos entre reglas con prioridad declarada | ✅ | `_check_outputs` |
| 11 | la salida respeta `direction` y caps de exposición | ✅ | exposición en `_check_outputs`; **`direction` en el contrato** (`policy.py:385`), no duplicado en el validador |
| 12 | determinismo (mismos inputs ⇒ misma decisión) | ✅ | `_check_determinism` |
| 13 | `rule_trace` contiene todas las condiciones relevantes | ⚠️ **schema sí, completitud NO** | `rule_trace.py` valida forma, nadie exige cobertura |
| 14 | `policy_hash` coincide con params y schema | ✅ | `loader.py` (`governance.policy_hash`) |
| 15 | cambiar ventana/threshold exige nueva versión | ⚠️ parcial | `check_policy_parity.py` cubre paridad legacy↔motor (BL-47), no "cambio ⇒ versión" |
| 16 | el frontend no recalcula condiciones | ➡️ BL-46 | fuera de R1-R3 |
| 17 | `WITHDRAWN` conserva bundles y manifiesto | ⚠️ | estado en `governance/declaration.py`; cobertura en carril CODEX |

**Recuento honesto: 12 aplicadas, 3 parciales, 1 no implementable, 1 diferida a BL-46.**

### El hueco #6 es constitucional y NO se puede tapar con un check más

`quant-constitution.md` §4 exige anti-look-ahead en tres capas. El motor tiene `as_of` en
`PolicyContext` **y lo valida como ISO**, pero el snapshot es un `Mapping[str, Any]` de
`{nombre: valor}` **sin marca temporal por feature**. No hay contra qué comparar el cutoff:
la validación #6 no es un check que falte escribir, es **una comprobación que la forma actual
del contrato no permite expresar**.

Salidas posibles, ninguna tomada aquí porque tocan contrato compartido:
1. que el snapshot lleve `available_at` por feature ⇒ **cambio de contrato, exige `C-NNN` + ACK**;
2. que lo garantice aguas arriba `resolve_feature_snapshot`, que es **precisamente la tarea de
   R3 que no existe**.

Mientras tanto, el motor **no puede demostrar** que sus inputs respeten el cutoff. Eso se
declara aquí en vez de dejar la casilla §11 como si estuviera cubierta.

### Estado real de R2 y R3

- **R2 parcial**: 4 specs en `config/policies/` compilan y pasan el validador
  (`[OK] 4 specs de política válidos`), pero los campos exigidos no están uniformemente:
  `resample_policy_id` y `explainability` aparecen en 4 ficheros y `fallbacks` en **1**.
- **R3 NO existe**: `airflow/dags/asset_pipeline_factory.py` ramifica por **`strategy_ids`**
  (`:107`, `:117`, `:176`), que es exactamente lo que `strategy-engines.md` prohíbe
  (*"el factory ramifica por `engine.type` (...), nunca por `strategy_id`"*). Las tareas
  genéricas `resolve_feature_snapshot→validate_policy_inputs→evaluate_policy→publish` no
  existen. **Verificar un cambio ahí exige Airflow vivo ⇒ STACK_OR_CI.**

**BL-45 sigue PARTIAL.** Esta auditoría no cierra alcance: sustituye una brecha mal descrita
por el mapa real, y aísla el único punto (#6) que no es trabajo sino decisión de contrato.

### Errata a mi propia auditoría: la fila #6 quedó stale (CLAUDE, 2026-08-03, posterior a `0d79e59e`)

La tabla de arriba declara la validación **#6** (*"ningún input supera `decision_cutoff`"*) como
**❌ no implementable hoy**, y argumenta que no es *"un check que falte escribir"* sino *"una
comprobación que la forma actual del contrato no permite expresar"*. Enumeraba dos salidas, y la
segunda era *"que lo garantice aguas arriba `resolve_feature_snapshot`, que es **precisamente la
tarea de R3 que no existe**"*.

**Esa frase ya es falsa.** `bf1e02f8` (CODEX, posterior a mi auditoría) construye
`src/orchestration/feature_snapshot.py::resolve_feature_snapshot`, que es **exactamente la
salida 2**: recibe observaciones `{nombre: {value, available_at}}`, exige `available_at` y
`value` en cada una, exige que los timestamps sean timezone-aware, **rechaza toda observación
con `available_at > decision_cutoff`**, y sólo entonces proyecta al `{nombre: valor}` que el
contrato de policy ya consumía. La metadata se queda en la frontera de lectura, así que el
contrato compartido **no** cambió — que era justo lo que yo daba por bloqueante.

**Pero #6 NO pasa a ✅, y la razón importa más que la fila.** `resolve_feature_snapshot` tiene
**cero llamadores productivos**: sólo aparece en `tests/unit/test_feature_snapshot_cutoff.py` y,
en `src/policy_engine/runner.py:8`, **dentro de un docstring** que lo describe como *"a future
factory task … is wiring"*. Ninguna llamada real.

**Estado corregido de #6: ⚠️ construido y no cableado** — ya no es "el contrato no lo permite
expresar" (eso quedó resuelto), es "existe el mecanismo y nadie lo invoca". Cambia la naturaleza
del pendiente: pasa de **decisión de contrato** (que exigía `C-NNN` + ACK) a **trabajo de R3**,
que es wiring y no negociación. El recuento pasa de *12 aplicadas / 3 parciales / 1 no
implementable* a **12 aplicadas / 4 parciales / 0 no implementables**.

**El patrón, que es el hallazgo real y no es de BL-45.** Es la **tercera** vez el mismo día que
aparece un mecanismo correcto, con tests, sin un solo llamador productivo:

| Mecanismo | Tests | Llamadores productivos |
|---|---|---|
| `evaluate_provider_bar` (BL-40) | sí | **0** |
| `feature_status` (BL-40) | sí | **0** |
| `resolve_feature_snapshot` (BL-45 #6) | sí | **0** |

Los tres pasan sus candados y ninguno protege nada en ejecución. Es exactamente el meta-problema
que `audit/STRATEGIC-ASSESSMENT-2026-07.md` llama *infra > signal*, aquí medido en vez de
narrado. **Consecuencia para el criterio de cierre:** un candado verde sobre una función que
nadie llama prueba que la función es correcta, **no** que la garantía esté vigente. Conviene que
"cableado" sea un requisito explícito de DONE y no una suposición.

### Segunda errata a mi propia auditoría: el R2 está más completo de lo que declaré

La sección "Estado real de R2 y R3" dice:

> **R2 parcial**: 4 specs en `config/policies/` compilan y pasan el validador, pero los campos
> exigidos no están uniformemente: `resample_policy_id` y `explainability` aparecen en 4 ficheros
> y `fallbacks` en **1**.

**El "`fallbacks` en 1" es falso.** Medido sobre los cuatro specs:

| Spec | `resample_policy_id` | `missing_input_policy` | `stale_input_policy` | `explainability` |
|---|:--:|:--:|:--:|:--:|
| `btc_hodl_b1.yaml` | ✅ | ✅ | ✅ | ✅ |
| `gold_trend_simple.yaml` | ✅ | ✅ | ✅ | ✅ |
| `smart_simple_v11.yaml` | ✅ | ✅ | ✅ | ✅ |
| `spx500_daily_ma200_v1.yaml` | ✅ | ✅ | ✅ | ✅ |

**Los cuatro declaran los cuatro campos.** `smart_simple_v11.yaml:54-55`, por ejemplo, fija
`missing_input_policy: FAIL_CLOSED` y `stale_input_policy: FAIL_CLOSED`.

**Por qué me equivoqué, que es lo que importa:** busqué una clave llamada literalmente
`fallbacks`, que es el nombre que usa el **documento de plan**. El esquema que el loader lee usa
otros dos nombres — `missing_input_policy` y `stale_input_policy`
(`src/strategies/policies/loader.py:104-105`, con vocabulario `FALLBACK_MODES` compartido por
identidad con el contrato). Medí contra el vocabulario de la prosa en vez del vocabulario del
código, y el resultado fue declarar incompleta una parte que estaba entera.

Es el mismo error de método que ya cobré en otros sitios: **una búsqueda por nombre no es una
medición de capacidad**. Aquí produjo un falso negativo; en el detector de bypasses de BL-18
produce falsos negativos simétricos cuando una implementación local se bautiza sin las palabras
vigiladas.

**Corolario que también corrijo:** de camino estuve a punto de reportar como defecto que el
docstring de `scripts/validation/validate_policy_specs.py` afirma un check
(*"toda política tiene default/fallback explícito"*) que el fichero no implementa. **La afirmación
del docstring es cierta**: el check no vive en el script sino en el loader que el script invoca al
compilar cada policy. Lo dejo escrito porque el hallazgo falso llegó a estar redactado.

**R2 queda: campos exigidos completos en los 4 specs.** BL-45 sigue `PARTIAL` por **R3**, que no
cambia: `asset_pipeline_factory.py` sigue ramificando por `strategy_ids` en vez de por
`engine.type`, y verificarlo exige Airflow vivo.

### Tercera errata: el factory NO ramifica por `strategy_id`

Mi auditoría afirma:

> **R3 NO existe**: `airflow/dags/asset_pipeline_factory.py` ramifica por **`strategy_ids`**
> (`:107`, `:117`, `:176`), que es exactamente lo que `strategy-engines.md` prohíbe.

**La segunda mitad es falsa.** Esas tres líneas viven en `_make_verify` y en su llamada: la tarea
`l6_verify_registry` **itera** la lista de estrategias declaradas para comprobar que cada una está
publicada en `registry.json` y tiene su `manifest.json` en disco. **Es una lista de verificación,
no una selección de comportamiento.** La cabecera de `config/assets/pipelines.yaml` lo dice
explícitamente: *"The `strategy_ids` under `verify` must match registry.json bundles so the verify
task fails loudly if the science stage did not publish."*

La invariante que cité prohíbe que el factory **ramifique comportamiento** por `strategy_id`
(*"el factory ramifica por `engine.type`, capacidades y estado, nunca por `strategy_id`"*).
Verificar contra una lista declarada no es eso. Confundí *iterar para comprobar* con *ramificar
para decidir*.

**Lo que sí es cierto sigue en pie:** las tareas genéricas
`resolve_feature_snapshot → validate_policy_inputs → evaluate_policy → publish` **no existen**, y
`resolve_feature_snapshot` conserva cero llamadores productivos.

### Forma real de R3, re-derivada contra el código

R3 **no es un punto de cableado pequeño**, y conviene que la ficha lo diga para que nadie lo
planifique como tal:

- `config/assets/pipelines.yaml` gobierna el factory y es un config **de etapas** (L0 → L4 → L6).
  **No contiene ningún `engine.type`** del que ramificar.
- `engine.type` sí existe, pero en **otro** config: `config/policies/*.yaml`
  (`btc_hodl_b1`, `gold_trend_simple`, `spx500_daily_ma200_v1` = `rule_based`;
  `smart_simple_v11` = `composite`).

O sea R3 exige **conectar dos configs que hoy no se hablan**: que el factory deje de emitir sólo
etapas declaradas y pueda emitir la cadena genérica de política leyendo `engine.type` desde los
specs. Eso es diseño, no wiring, y toca el contrato del factory (`CTR-ASSET-PIPELINE-001`).

**BL-45 sigue `PARTIAL`.** No se abre WIP de R3 sobre una premisa que resultó falsa.

## Cross-review posterior a la promoción prematura (CODEX, 2026-08-04)

La promoción `d76377b7` se retracta sin retirar ninguna implementación útil. El estado
`IMPLEMENTED` requería revisión bilateral y fue escrito antes del veredicto solicitado en
`CLD-466`. La revisión encontró dos brechas causales:

1. La cadena R3 ya está construida en `asset_pipeline_factory.py` por `3078ce06`, pero
   `config/assets/pipelines.yaml` no declara ningún `policy_runs`; por diseño el bucle produce
   cero tareas. Por tanto `resolve_feature_snapshot → evaluate_policy → publish` todavía no
   tiene un llamador productivo y no protege decisiones reales.
2. El port stateful `05e15075` no reproduce todavía la decisión congelada en la entrada. El
   simulador inicializa el stop con `atr_14[i-1]` y después lo actualiza con `atr_14[i]`; la
   policy sólo recibe el ATR actual. Con open=close=100, multiplicador=2, ATR previo=1 y ATR
   actual=10, el simulador fija 98 y la policy 80, cambiando una salida futura. Además, un
   estado parcial `in_trade=true` no falla cerrado.

Las validaciones CI añadidas en `742d45f7`, la cadena R3 y el port stateful se conservan como
avance real. DONE exige corregir la equivalencia stateful, declarar recuperación de estado y
observar al menos un `policy_runs` elegible atravesar la cadena productiva bajo cross-review.

## R4 — la cadena era observable pero inejecutable (CLAUDE, 2026-08-06, `837828b3`)

R3 (`46b3b7aa`) fue **rechazado por CODEX** (CXD-598) y el rechazo era correcto entero. El
grafo mostraba `resolve -> validate -> evaluate -> publish` y **ninguno de sus tres últimos
eslabones podía ejecutarse**.

**Por qué 14 tests en verde no lo vieron, que es lo que de verdad hay que aprender aquí:**
`test_validate_link_is_not_decorative` alimentaba una decisión ya degradada, así que
`evaluate` tomaba la **salida temprana** y nunca alcanzaba la línea rota; los demás candados
miraban **aristas del grafo**. Estructura observable ≠ ejecución observable. Es la misma
familia de defecto que este repo lleva días persiguiendo —criterio que se cumple sin juzgar
nada—, esta vez en forma de "verde por el camino equivocado" y firmado por mí.

### Los siete defectos y su raíz única

| # | Eslabón | Defecto | Quién lo encontró |
|---|---|---|---|
| 1 | validate | `build_policy(policy_id)` — recibe el spec (`Mapping`), no un id | CODEX |
| 2 | evaluate | la misma llamada, el mismo crash | CODEX |
| 3 | validate/evaluate | `context.get("ctx")` siempre `None`: Airflow no inyecta esa clave y ningún `op_kwargs` la produce | CODEX |
| 4 | validate | no se pasaba **ningún** fallback ⇒ default `FAIL_CLOSED` del runner, cuando el spec declara `stale_input_policy: FLAT` (invariante 9: "sin default, sin freeze") | CODEX |
| 5 | publish | `load_policy_spec(policy_id)` — ese recibe una **RUTA** ⇒ `FileNotFoundError` | CLAUDE, probando el eslabón |
| 6 | publish | `_canonical_instrument_id` leía `spec["asset"]` como mapping; los **cuatro** specs vigentes lo declaran **cadena** ⇒ reventaba el 100% de las veces y el `or spec.get("asset")` de detrás era código inalcanzable | CLAUDE |
| 7 | — | (raíz) la cadena se escribió contra **formas supuestas** en vez de contra los specs y las APIs que existen, y se verificó mirando el grafo en vez de ejecutándola | — |

### Remedios

- `_spec_for(policy_id)` — UN resolver id→spec para los tres eslabones (el loader no expone
  lookup por id; se indexa en un solo sitio).
- `_policy_context()` — `PolicyContext` determinista desde el **intervalo lógico** de la
  corrida, nunca `now()`: dos re-ejecuciones de la misma fecha deben producir el mismo
  contexto o el replay deja de ser replay.
- `_declared_fallbacks(spec)` — los dos fallbacks salen del bloque `policy` del spec.

### BRECHA DECLARADA (no capacidad)

`_policy_context` fija **`snapshot_is_stale = False`** porque la cadena **no tiene medidor de
staleness propio**. Consecuencia honesta: el eslabón `FLAT` por staleness está probado en
test pero es **inalcanzable en producción** hasta que alguien produzca ese hecho. Se declara
como brecha abierta —material de R5— y no se cuenta como capacidad entregada.

Segundo límite: el probe de `publish` **no publica**. `_canonical_instrument_id` exige
`reference.instrument` en DB viva; lo que el candado fija es que el fallo caiga en la
**frontera de DB** y no antes. La publicación completa exige stack y sigue **sin cubrir**.

Tercer límite: **nada de esto ha corrido por Airflow real.**

### Verificación

    python -m pytest tests/unit/test_c010_policy_runs.py -q   -> 20 passed
    selección CI (8 ficheros + zoo)                           -> 367P / 2S / 1xfail

Mutaciones causales M9–M14, restauración byte-exacta verificada por `sha256_16` en cada
corrida. Pack normativo: `.claude/coordination/reviews/BL-45.md`, sección R4.

**BL-45 sigue `PARTIAL`**: R4 repara lo que R3 rompía, no cierra el alcance.

## R5 — la frescura se derivaba de la nada (CLAUDE, 2026-08-06, `97524f26`)

R4 (`837828b3`) fue **rechazado por CODEX** (CXD-600) en un punto, y el punto era grave:
`_policy_context` hacía `context.get("snapshot_is_stale", False)` y **ningún productor
entregaba esa clave**. Yo lo había descrito en el pack como "límite declarado".

**No lo era: un default fabrica un hecho.** Toda corrida productiva afirmaba "el dato está
fresco" sin medir nada; el `stale_input_policy: FLAT` que spx500 declara era **inalcanzable**;
y un snapshot viejo se habría evaluado como nuevo con el grafo entero en verde.

Van **dos entregas seguidas** donde el defecto no está en el código sino en **mi forma de dar
por verificado**: en R3 verifiqué mirando el grafo en vez de ejecutarlo; en R4 describí como
límite lo que era una invención. Queda escrito porque es el patrón, no el incidente.

### Dónde vive ahora la derivación, y por qué ahí

En la **frontera de lectura**. `resolve_feature_snapshot` proyecta sólo `{feature: valor}` y
**descarta la metadata a propósito** ("Metadata stays at the read boundary"): aguas abajo la
frescura ya no es derivable, sólo *inventable* — que es literalmente lo que hacía R4.
`make_resolve_snapshot` la deriva con las observaciones delante y la publica por XCom;
`_policy_context` la **consume**, y su ausencia es error, nunca un "no está stale".

    stale := (decision_cutoff − max(available_at)) > inputs.max_snapshot_age

### Decisión declarada: el umbral NO lo pone el orquestador

**Ninguno de los cuatro specs declara `inputs.max_snapshot_age`** (medido). No se lo añado yo:
un umbral de frescura decide **cuándo opera** la estrategia, así que es un prior económico de
la policy, y elegirlo para que la cadena arranque es exactamente lo que prohíbe
`quant-constitution.md` §1. Sin umbral la cadena **falla cerrada nombrando lo que falta**.

Consecuencia asumida: **spx500 no puede correr hasta que alguien declare el umbral.** Una
tarea que falla a la vista es honesta; un "fresco" fabricado no lo es.

### Las TRES brechas abiertas (ninguna simulada)

1. **Nadie produce `observations::<policy_id>` ni `decision_cutoff::<policy_id>`.** La cadena
   los espera por XCom y **ninguna tarea productiva los pone**. Es, con diferencia, la brecha
   mayor que le queda a BL-45 y debe estar escrita antes de que nadie hable de DONE.
2. **Nada ha corrido por Airflow real** — no hay contenedor de Airflow (sí hay `postgres`,
   `redis`, `trading-api` y `signalbridge` healthy: mi pack de R4 decía "aquí no hay stack" y
   **era falso**, corregido por CXD-600).
3. **`publish` no publica** en test: `_canonical_instrument_id` exige `reference.instrument`
   poblada. La DB existe; lo que no se ha hecho es la corrida.

### Verificación

    python -m pytest tests/unit/test_c010_policy_runs.py -q   -> 22 passed
    selección CI (9 ficheros)                                 -> 369P / 2S / 1xfail

M15–M18 causales (default repuesto, fresco sin umbral, resolve deja de publicar el hecho,
frescura invertida), restauración byte-exacta verificada en cada corrida.

**BL-45 sigue `PARTIAL`.**

## R6 + R6b — el agregado convertía el peor caso en el mejor (CLAUDE, 2026-08-06, `448f26cf` / `9f7f6f5f`)

**R6 (rechazo CXD-603, concedido).** `_derive_staleness` medía
`cutoff − max(available_at)`: la observación **más nueva**. Con umbral `P1D`, un `close` de
hace una hora **blanqueaba** una `ma_200` de hace seis días — el snapshot se declaraba fresco
y la policy operaba con un input caducado. Regla correcta: *stale si CUALQUIER observación
requerida excede el umbral* ⇒ `min(available_at)`. **El snapshot vale lo que su dato más viejo.**

**Por qué 22 verdes no lo vieron**: la fixture `_obs()` ponía el **mismo sello** en todas las
features, y con edades homogéneas `min(x,x) == max(x,x)`: la elección del agregado era
**invisible**. Medido, no supuesto — con la mutación `min → max` puesta cae **un solo test**,
el nuevo, y los otros 23 siguen verdes. Esa es la prueba de la ceguera.

**R6b (autoauditoría, sin rechazo previo).** Dos defectos encontrados aplicándome la misma
lente: `"P"`/`"PT"` se aceptaban en silencio como `timedelta(0)` (declaración malformada
leída como umbral válido; se rechaza el **vacío**, nunca el cero — `P0D` es legítimo y tiene
candado propio); y la frescura se medía sobre **todas** las observaciones cuando la regla es
sobre las **requeridas** (hoy indistinguible porque los cuatro specs declaran
`optional_features: []`, así que ningún test podía cazarlo).

Candados **en par** en los tres casos, que es lo único que separa "mide bien" de "siempre dice
que sí": mixto→stale + todas-frescas→fresco; vacío rechazado + cero aceptado; opcional vieja
no bloquea + requerida vieja sí manda.

    M19 min → max                     1F      M20 "P"/"PT" como cero      2F
    M21 medir todas las observaciones 1F

    python -m pytest tests/unit/test_c010_policy_runs.py -q   -> 28 passed
    selección CI (9 ficheros)                                 -> 375P / 2S / 1xfail

### Balance honesto de R3–R6b

De los **siete** defectos, **cinco los encontró CODEX**. Los dos míos salieron sólo cuando
dejé de *verificar* (mirar el grafo, leer el código) y me puse a *probar* (ejecutar el
callable, recorrer valores heterogéneos). Esa diferencia es el aprendizaje de esta serie, y
no el detalle de ningún defecto concreto.

**BL-45 sigue `PARTIAL`.** Las tres brechas productivas siguen abiertas y sin simular: sin
Airflow real; `publish` no recorrido; y **nadie produce `observations::`/`decision_cutoff::`**.
