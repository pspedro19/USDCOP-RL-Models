---
title: Tablero de mutación — lote CLAUDE (23 BLs)
contract: CTR-MUTATION-SCOREBOARD-001
status: LIVE
owner: CLAUDE
measured_against: 92963fa9
date: 2026-07-28
supersedes: none
---

# Tablero de mutación — lote CLAUDE

**Qué mide.** Para cada BL: se rompe **a propósito** la garantía que promete, **en el código de
producción**, y se comprueba si su test cae. Un BL cuyo test sigue verde con la garantía rota
**no está protegido**, por muy correcto que sea el código.

**Cómo se midió.** Contra `git archive 92963fa9` en copias aisladas; el repo real nunca se mutó.
Cada veredicto trae la mutación literal y el mensaje del fallo en el mensaje de canal citado.

**Por qué existe.** Porque el 2026-07-28 se descubrió que se puede destruir el fencing de una
orden, forzar una identidad contable a no fallar nunca, o **borrar el panel A/B entero de
`/production`**, y las suites no pierden un solo test. "Verde" dejó de ser evidencia.

---

## Resumen

| Veredicto | N | BLs |
|---|---|---|
| **MUERDE limpio** | 7 | BL-01, BL-04, BL-12, BL-15, BL-45, BL-46, BL-47 |
| **MUERDE con matiz** | 2 | BL-02, BL-03 |
| **CERRADO hoy** | 14 | BL-05, BL-06, BL-09, BL-11, BL-13, BL-14, BL-20, BL-25, BL-31, BL-32, BL-34, BL-36, BL-39, BL-42 |
| **NO MUERDE — abiertos** | 0 | — |
| **SIN TEST** | 0 | — |

**7 + 2 + 14 + 0 + 0 = 23.** El total debe cuadrar: un BL que cambia de fila y no aparece en ninguna es exactamente el tipo de hueco que este tablero existe para cazar.

---

## MUERDE limpio

| BL | Mutación aplicada | Resultado |
|---|---|---|
| BL-01 | Vaciar el sentido del disclaimer conservando el `data-testid` | rojo: exige las cláusulas completas normalizadas, no prefijos |
| BL-04 | Re-inlinear el copy imitando el banner compartido | 2 rojos: exige el **montaje** del componente, no el texto |
| BL-12 | Neutralizar `check_provenance_wall`; borrar `forecast_trial_ids` de una familia | 4 rojos + 1 rojo: la muralla FT→AT es estructural |
| BL-15 | Anular la finitud (Py) y quitar `!Number.isFinite` (TS); `raise` → `continue` en el muro de publicación | 9 + 5 + 2 rojos sobre la **misma tabla compartida** |
| BL-45 | Desactivar la whitelist del DSL; spec nuevo con `python_eval` + `__import__('os').system` re-hasheado | 4 rojos, y el spec nuevo **también** revienta: el validador hace glob del directorio (K-029) |
| BL-46 | Re-evaluar la condición en React en vez de renderizar el veredicto del backend | 2 rojos, uno con **traza contradictoria** (invariante 7 de `strategy-engines.md`) |
| BL-47 | `>=` → `>` en el motor de Gold | `865/5610 barras divergen, primera idx=9, legacy=0.6449 motor=0.0` — compara contra el **productor congelado real** |

## MUERDE con matiz

| BL | Muerde, pero | Qué falta |
|---|---|---|
| BL-02 | protege una **rama muerta**: los tres activos son `model_zoo`, luego `isModelZoo` es siempre true y `AssetWeeklyBody` es inalcanzable | test de render con `forecast_mode: 'weekly_inference'` inyectado + test de coherencia que falle si ningún activo usa la rama que el candado congela |
| BL-03 | muerde **Vitest** (lee el DOM); el candado Python solo comprueba subcadenas y se sortea vaciando el cuerpo de `directionLabel` | comprobación de comportamiento en Python, o declarar explícitamente la cobertura delegada a Vitest |
| BL-13 | `surface` muerde en 3 mutaciones distintas; el **hash canónico LF de los manifiestos no tiene código de producción** (`_canonical_lf` vive dentro del test) | extraer el método a un módulo de producción que **el test y el escritor del freeze** importen |
| BL-14 | detecta que el bloque `components` **existe**; un `current_model_snapshot` completamente inventado pasa verde | validar `as_of`, forma del `pointer`, hashes hex ≠ `0*16`, y `registered_in`; y que los `forecast_trial_ids` existan en el ledger |
| BL-31 | el gate de orden y el rollback muerden; una observación **INVALID** no rompe la racha verde pese a que el docstring lo promete | test que meta un `INVALID` en medio de una racha y exija `streak == 0` |
| BL-39 | muerde por **drift de hash**, no por semántica: re-registrar el hash deja pasar la fuga T-1. El bit-check v11 **skipea siempre en CI** | test de causalidad sobre un frame sintético con un salto en T |

## NO MUERDE → cerrado el 2026-07-28

Cada cierre trae su rojo **re-verificado por la raíz**, no solo reportado por quien lo escribió.

| BL | Hueco | Cierre | Rojo verificado |
|---|---|---|---|
| BL-05 | anular el montaje de `PaperCandidatesPanel` borraba el A/B de `/production` y la suite seguía 13/578 verde; y la §6 era un test de **eco**, no de prohibición | monta `ProductionView` **entera** con la cadena de datos real y `fetch` enrutado por URL (muerde también si cambia la URL del ledger); + prohibición §6 real; + caso inverso rol `free` | `{false && paperLedger && …}` ⇒ 1 failed |
| BL-06 | reescribir el widget como `fetch('/api/produc' + 'tion/approve')` y `<button>Comprar ahora</button>` ⇒ 28 passed | primitivo anti-evasión **extraído a `tests/support/js_source_scan.py`** y compartido por los dos candados (−177 líneas): plegado de concatenaciones, escapes, NFKD, minúsculas, y **sustitución conservadora de constantes** — un `${IDENT}` ligado una sola vez a un literal se sustituye; ligado dos veces se descarta en vez de adivinar | la evasión completa desde `lib/telemetry/` ⇒ 3 hits, 1 failed |
| BL-09 | `run_all_checks` podía quedarse con **1 de 11 checks**; la cadena `prev_hash` sin cobertura (solo se detectaba edición, no supresión/inserción/reorden) | cableado parametrizado por **introspección** (un check nuevo queda cubierto solo) + tres ataques a la cadena que exigen que *todos* los errores sean `prev_hash roto` | `return errors` tras el 1er check ⇒ 10 failed |
| BL-11 | `check_families` se apagaba entera con un `return []` | dos tests de contenido: celda a trial inexistente, `trials_charged` que subcuenta | incluido en el anterior |
| BL-20 | se podían **fabricar** las contribuciones SHAP; `grep additivity tests/` ⇒ **0 aserciones**. La ruta TreeSHAP no la ejecutaba ningún test | **oráculo** que rehace el fit y pregunta al modelo sus predicciones crudas; identidad anclada en forma agregada (global + por año, 7 testigos) + TreeSHAP ejecutada de verdad | `phi = np.ones_like(Z)` ⇒ 3 failed |
| BL-25 | el umbral **3σ no estaba anclado**: ×1000 pasaba verde porque el fixture usaba `paper − constante` (`sd(d)` ~1e-18) | gemelos con ruido independiente y semilla fija a **3.50σ** (rojo) y **2.47σ** (verde), + guard de dispersión que falla si alguien vuelve a degenerar el fixture | ×1000 y ÷1000 rompen cada gemelo |
| BL-32 | recortar los bloques obligatorios de 8 a 3 ⇒ 44 passed idéntico (el fixture del propio test usaba `{}`); vaciar `governance` a `{}` en TS ⇒ 30/30 | los 8 bloques exigidos **uno a uno**; la forma de cada bloque clavada contra las interfaces TS (único sitio donde está declarada); el payload real debe poblar cada campo con un `Sourced` bien formado. Regla: **sin agujeros mudos**, no sin agujeros — `null + pending:"BL-45 …"` es válido, `{}` no | 8 bloques → 3 ⇒ 5 failed |
| BL-34 | `canPromote = true` montaba el Voto 2 en `/replay` para cualquier rol, y un widget de aprobación nuevo en `app/replay/page.tsx` también pasaba ⇒ **578 passed, rbac:check OK, rbac:test PASS** | perímetro derivado del cierre de imports (29 ficheros) con el **fichero dual exento y auto-verificado**, fijado por tests estructurales + render jsdom en 5 roles × 2 estados, con **control positivo** para no ser verde por vacuidad | `canPromote = true` ⇒ pytest 1 failed + vitest 2 failed |
| BL-42 | `round(v, 2)` → `round(v / 100.0, 6)` —un decimal bajo sufijo `_pct`— no movía un test: se validaba el **artefacto**, no el productor | ejercita `_compute_result_metrics` con un ledger sintético que compone 10.000 → 11.446,16, o sea **+14,46 pp derivados fuera del código bajo prueba** | `/100.0` ⇒ 2 failed |

## NO MUERDE — abiertos

| BL | Mutación que pasa verde | Test que falta |
|---|---|---|

## BL-36 — cerrado el 2026-07-28

Tenía **cero** cobertura: se podía invertir cualquier decisión de la matriz de verdad, declarando dos
escritores para el mismo atributo, y ningún gate se movía. Ahora 8 tests contrastan **77 claims** (50 de
columnas W/R + 27 en prosa) contra `.claude/generated/db-inventory.json`, derivando el perímetro de
lectores por glob sobre 1277 ficheros con las regex del propio generador (K-029, cero listas a mano).

Dos límites declarados: a nivel **atributo** es imposible (la matriz no tiene columna atributo→escritor,
así que se implementó a nivel tabla); y *"una DEPRECATED no puede tener lectores"* **saldría rojo en
prístino**, porque §9 declara `bi.fact_*` DEPRECATED y a la vez publica que 7 ficheros la referencian —
eso es el estado honesto, no el defecto. Se convirtió en: *un retiro propuesto sobre una tabla con
lectores vivos debe declarar su conteo de referencias*.

**Incoherencias numéricas encontradas: cero.** La matriz estaba bien; lo que no había era nada que lo
comprobara.

---

## Lo que este tablero enseña, más allá de los BLs

El patrón no es "los tests comprueban subcadenas". Es más amplio y aplica a los dos ingenieros:
**se prueban las PIEZAS y no el CABLEADO.** `check_families` neutralizable y `claim_order_dispatch`
destruible son el mismo defecto; `ExecutionService` sin importadores y `PaperCandidatesPanel` sin
montaje, también. De ahí sale la regla operativa: **todo validador que alimente un gate necesita un
test que demuestre que el gate lo invoca**, y todo componente que sea una garantía necesita un test
que demuestre que está montado donde dice.
