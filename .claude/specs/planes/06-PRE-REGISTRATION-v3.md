---
kind: roadmap
status: SIGNED
version: 3.0.0
last_verified: 2026-09-11
supersedes: [specs/planes/06-PRE-REGISTRATION.md]
code_anchors:
  - config/experiments/thesis_ppo_v2.yaml
  - config/research/feature_schema_v2.json
  - config/research/macro_availability.yaml
  - data/thesis/research_data_portable_v2.pkl
  - scripts/analysis/thesis_train_ppo.py
  - scripts/analysis/thesis_statistics.py
---

# Pre-registro v3 — EXP-TESIS-RL-02

operator_signature: SIGNED

Este documento gobierna la versión corregida después de la auditoría EXP-TESIS-RL-01. Está en
`PARTIAL` hasta que el operador lo firme antes de la primera corrida v2 de mercado. Los replays
de 2023 y 2024–2026 son diagnósticos retrospectivos y no pueden convertirse en evidencia
confirmatoria.

**Nota de estado vigente (2026-09-11):** la compuerta sintética sí fue superada por la receta
`flat_init_no_turn`; la sección histórica inmediatamente inferior conserva el diagnóstico de la
receta original y no debe interpretarse como el estado actual. Véase la corrección formal y el
agregado firmado más abajo.


## Compuerta de sanidad — NO SUPERADA (2026-09-11)

Este pre-registro exige congelar una receta que pase las fixtures de solución conocida
**antes** de tocar datos de mercado. Ejecutada S1 (ruido iid con costo, óptimo = no operar) con
las cinco semillas y las cuatro sondas ordenadas: **ninguna receta pasa**, y la de referencia da
0/5 semillas planas.

Mientras eso siga así, **la Etapa 4 de [`BL-50`](backlog/BL-50-reparacion-tesis-rl.md) no se
ejecuta**: reentrenar v2 con una receta que no encuentra el flat sobre ruido puro produciría
otra conclusión confundida entre optimizador y mercado, que es exactamente el defecto que este
programa existe para corregir. La búsqueda de receta continúa en terreno sintético, donde no se
gasta ningún trial de mercado.

## Corrección de la compuerta de sanidad (2026-09-11)

La sección anterior quedó obsoleta después de ejecutar la receta sintética corregida. El artefacto
[`sanity_protocol_v2.json`](../../../outputs/thesis-repair/sanity_protocol_v2.json) confirma que
`flat_init_no_turn` pasa S1–S4 con cinco semillas y cobra cero trials de mercado. La receta PPO
original se conserva como diagnóstico fallido sobre S1, pero no se reutiliza para v2. Esta
corrección permite avanzar al rebuild v2 únicamente si además pasa el gate macro y la identidad
del portable; los controles sintéticos no son evidencia de rentabilidad en USD/COP.

## Identidad congelada

| Elemento | Valor |
|---|---|
| Schema | `feature_schema_v2.json`, 37 features, declared SHA `a0568db4b953604cabb6d64eab73631419f65184d0fc96bd61a6f994f75d2e8b` |
| Dataset portable | `research_data_portable_v2.pkl`, identidad esperada `7f042564204ad6d61baa016556265828b508714d92b6f1eab1e9453adbd56d61` |
| HMM portable | `regime_hmm_frozen.json`, hash de parámetros requerido; el hash se valida al cargar |

> **Corrección 2026-09-11.** El SHA del schema que figuraba aquí terminaba en
> `…46f35f36c695487484cb`, que es la cola de la identidad del dataset portable: una pegada
> defectuosa había reemplazado los últimos 18 caracteres. Dos hashes independientes no
> comparten sufijo, y un congelamiento cuyo hash no se puede verificar no congela nada. El
> valor correcto se lee del propio artefacto, y `tests/regression/test_prereg_v3_identity.py`
> vuelve a comprobarlo en cada corrida para que no se degrade en silencio.
| Macro | `macro_availability.yaml`, disponibilidad estricta anterior a apertura |
| Semillas | `42, 123, 456, 789, 1337` |
| Anualización | 221 sesiones/año |

## Universo y cargos

El universo v2 contiene únicamente `ppo_regime_v2` y `ppo_backbone_v2`, cinco semillas cada
uno, más `always_flat`, B1, B1′, NULL-A, random y las reglas intradía fijas. Las correcciones,
replays retrospectivos y fixtures sintéticas cobran cero trials de mercado. El control shuffle
cobra un FT; cada configuración v2 evaluada en selección cobra un AT; cada brazo forward cobra
un AT. El ledger se actualiza antes de mirar el resultado.

## Hipótesis y juez

- H1′: una política PPO v2 supera `always_flat` en retorno diario neto pareado.
- H2: la información de régimen mejora a `ppo_backbone_v2`; solo se interpreta con potencia
  declarada y bootstrap jerárquico por semilla.
- Evaluación retrospectiva: desarrollo y selección, con retraso 0/1/2, surrogate intrasesión y
  escenarios de costo bajo/central/alto.
- Juez confirmatorio: únicamente forward después del freeze, una mirada a `n=120` sesiones,
  barras no selladas = NA.

## Regla de éxito

No se reclama edge salvo que el resultado por semilla, la mediana, el DSR (`N` del ledger), los
baselines, el stress ×2 y el retraso de una barra sean favorables. Si v2 pierde contra flat en
selección, la familia PPO se cierra y el forward queda como replicación descriptiva.

## Prohibiciones

- No elegir hiperparámetros mirando selección u hold-out.
- No usar macro del mismo día, cero-fill ni fallback de fuente.
- No reportar Sharpe/p con menos de 20 operaciones.
- No llamar confirmatorio a un replay retrospectivo.
- No ejecutar el juez forward antes de 120 sesiones ni abrir una segunda mirada.

## Campos que faltaban para firmar — cerrados 2026-09-11

Estos cuatro huecos eran lo único que impedía firmar. Se cierran **antes** de la primera llamada
a un proveedor, que es el único orden que hace que un pre-registro signifique algo.

| Campo | Valor congelado |
|---|---|
| **Híbrido** | **PPO + LLM**, fijado ex-ante por el operador. Queda prohibido evaluar además PPO+FinMA-ES y quedarse con el que salga mejor: sería exactamente la selección que este documento existe para impedir. |
| **Protocolo LLM** | `prompt_version: thesis-llm-trader-v1` · `temperature: 0.1` · `top_p: 0.9` · `max_tokens: 256` · `decisions_per_session: 59` · `max_retries_invalid_json: 1` · respuesta inválida ⇒ `retain_previous_weight` · sin fallback entre proveedores. Fuente: `config/research/llm_thesis.yaml` (CTR-RESEARCH-LLM-THESIS-001). |
| **Hash del prompt** | `system_prompt` SHA-256 `143a0e72fc50bf915feed8d766d85531170823465fb71091573418867bee0621`, congelado el 2026-09-11 sobre los 13 334 contextos de `outputs/thesis-repair/llm_selection_contexts_v2.jsonl`. |
| **Modelos** | Primario DeepSeek `deepseek-chat`; robustez Azure OpenAI, deployment aportado por el operador en ejecución. **Ledgers separados**; el proveedor NO se elige después mirando el PnL. |

**Identidad de datos verificada al firmar**: el portable `research_data_portable_v2.pkl` tiene
SHA-256 de fichero `7f332df17a2492a0533f713bc143316fec9b9eadedcfbdad59d5eb0a36b7b0b5`, que es
exactamente el `dataset_sha256` que llevan los 13 334 contextos. Si alguno difiere, el runner
aborta: no se puede ejecutar el brazo LLM sobre un dataset distinto del declarado.

**Conteo de trials**: el brazo LLM sobre *selección* es **retrospectivo y diagnóstico**
(`--allow-retrospective`), así que no abre juez confirmatorio ni cobra trial de mercado. Los dos
brazos confirmatorios (DeepSeek y Azure en forward, tras el freeze) son **+2 AT** en su primera
liquidación mirada.

**Declaración de búsqueda de receta**: la receta PPO `flat_init_no_turn` se eligió tras probar
**ocho sondas** sobre fixtures sintéticas. No cobra trial de mercado —las fixtures no son
USD/COP— pero se declara aquí porque es una búsqueda, no un prior económico, y quien lea los
resultados tiene derecho a saberlo.

## Regla del híbrido — congelada 2026-09-11, antes de ver ningún ledger

El pre-registro nombraba el híbrido (**PPO + LLM**) pero no decía **cómo se combinan**. Un
híbrido sin regla no es una hipótesis: es una licencia para probar combinaciones hasta que una
salga positiva. Se cierra aquí, con los dos ledgers **incompletos** (DeepSeek 917/13 334, Azure
353/13 334) y por tanto sin que nadie haya podido ver el resultado que la regla produce.

**Componente PPO**: exposición por barra de `ppo_regime_v2`, política determinista (sin
muestreo), **mediana de las cinco semillas** barra a barra, ajustada a la rejilla congelada
`{-1, -0.5, 0, +0.5, +1}`. Se usa la mediana y no la mejor semilla: elegir semilla sería
selección.

**Componente LLM**: peso por barra del ledger de **DeepSeek** (proveedor primario declarado).

**Regla de combinación — acuerdo de signo obligatorio**:

    w_hib(b) = w_ppo(b)   si signo(w_ppo(b)) == signo(w_llm(b)) y ambos ≠ 0
    w_hib(b) = 0          en cualquier otro caso

**Por qué esta regla y no otra, declarado ex-ante**: el fallo medido del PPO v2 no es falta de
bruto — la mediana del bruto es **+13,50 %** — sino exceso de operación: **4,13 cambios por
sesión** y un coste mediano del **47,17 %** que se come el bruto dos veces y media. Un veto que
exige confirmación independiente **sólo puede reducir rotación; no puede inventar bruto**. Eso
hace la hipótesis falsable y estrecha: si el híbrido sigue perdiendo, el problema no es el
filtro de entrada, y la familia se cierra.

**Brazo de robustez**: la **misma** regla sustituyendo DeepSeek por Azure `gpt-4o-mini`. Se
reporta siempre; **no se elige el proveedor que salga mejor** — esa comparación mide robustez,
no rendimiento.

**Prohibido sin cobrar trial nuevo**: promediar exposiciones, ponderar por confianza del LLM,
usar el LLM sólo para dimensionar, invertir el papel de veto, o cualquier otra regla de
combinación. Cada una es una hipótesis distinta y se registra antes de mirarla.

## Cómo se aplicó esta firma

`status: SIGNED` lo escribió Claude el 2026-09-11 **por instrucción explícita del operador en
sesión**, que autorizó firmar en su nombre tras habérsele presentado el bloqueo
(`run_thesis_llm.py:144`) y las alternativas. Se deja constancia aquí porque una firma cuyo
autor no consta no es una firma. La autorización no convierte ningún replay retrospectivo en
evidencia confirmatoria ni levanta la obligación de publicar los resultados negativos —
incluido el que ya está medido: **0/10 semillas PPO v2 baten a `always_flat` en selección**.

## Firma del operador — 2026-09-11

El operador autoriza explícitamente continuar con el experimento bajo este protocolo, incluyendo
DeepSeek como proveedor primario y Azure OpenAI como prueba de robustez. Los proveedores deben
usar ledgers separados, prompt/modelo fijados, sin fallback entre proveedores y sin selección
retrospectiva por PnL. La autorización no convierte replays retrospectivos en evidencia
confirmatoria y conserva la obligación de publicar resultados negativos.

Identidad firmada: schema declared SHA `a0568db4b953604cabb6d64eab73631419f65184d0fc96bd61a6f994f75d2e8b`;
portable identity `7f042564204ad6d61baa016556265828b508714d92b6f1eab1e9453adbd56d61`;
macro/cross-source vigentes en `outputs/thesis-repair/e2e_status_current.json`.
