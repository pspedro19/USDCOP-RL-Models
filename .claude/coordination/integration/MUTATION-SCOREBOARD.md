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
| **MUERDE con matiz** | 6 | BL-02, BL-03, BL-13, BL-14, BL-31, BL-39 |  <!-- BL-25 salio: el 3-sigma ya esta acotado -->
| **NO MUERDE → CERRADO hoy** | 5 | BL-05, BL-09, BL-11, BL-20, BL-25*, BL-42 |
| **NO MUERDE — abiertos** | 3 | BL-06, BL-32, BL-34 |
| **SIN TEST** | 1 | BL-36 |

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
| BL-25 | los tres relojes muerden; el umbral **3σ no está anclado** (×1000 pasa verde: el fixture usa diferencia constante ⇒ `sd(d)` ~1e-18) | escenario con ruido real a ~3.5σ (rojo) y su gemelo a ~2.5σ (verde) |
| BL-31 | el gate de orden y el rollback muerden; una observación **INVALID** no rompe la racha verde pese a que el docstring lo promete | test que meta un `INVALID` en medio de una racha y exija `streak == 0` |
| BL-39 | muerde por **drift de hash**, no por semántica: re-registrar el hash deja pasar la fuga T-1. El bit-check v11 **skipea siempre en CI** | test de causalidad sobre un frame sintético con un salto en T |

## NO MUERDE → cerrado el 2026-07-28

| BL | Hueco | Cierre |
|---|---|---|
| BL-09 | `run_all_checks` podía quedarse con **1 de 11 checks**; el encadenamiento `prev_hash` sin cobertura (solo se detectaba edición, no supresión/inserción/reorden) | 4 tests: cableado parametrizado por **introspección** + tres ataques a la cadena. Rojo verificado por la raíz: 10 failed |
| BL-11 | `check_families` se apagaba entera con un `return []` | 2 tests de contenido: celda a trial inexistente, `trials_charged` que subcuenta |

## NO MUERDE — abiertos

| BL | Mutación que pasa verde | Test que falta |
|---|---|---|
| **BL-34** | `canPromote = true` (Voto 2 en `/replay` para cualquier rol) ⇒ **578 passed, rbac:check OK, rbac:test PASS**. Y un widget de aprobación **nuevo** montado en `app/replay/page.tsx` ⇒ también verde | candado de **perímetro derivado** sobre el cierre de imports de `/replay` + render jsdom que exija 0 botones de aprobación. La spec de Playwright que lo caza **no la ejecuta ningún workflow** |
| **BL-05** | anular el montaje de `PaperCandidatesPanel` en `ProductionView.tsx:982` ⇒ el A/B v11/v12/v14 **desaparece de `/production`** y la suite sigue 13/578 verde. Y añadir "Sharpe 3.35 · p=0.006" junto a `n_trades=11` pasa verde | test de render de `ProductionView` **completa**; y test de **prohibición** §6 (que el `textContent` de la fila NO empate `/sharpe|p\s*=|calmar|sortino/i`), no de eco |
| **BL-20** | `phi = np.ones_like(Z)` — contribuciones SHAP **fabricadas, constantes 1.0** ⇒ 27 passed idéntico. `grep additivity tests/` ⇒ **0 aserciones** | aditividad recomputada (`\|Σφ + base − pred\| < 1e-9`) y no-degeneración: perturbar un coeficiente debe cambiar el ranking. K-041 en casa propia |
| **BL-32** | recortar los bloques obligatorios del passport de 8 a 3 ⇒ 44 passed idéntico (el fixture del propio test usa `{}`); vaciar `governance` a `{}` en TS ⇒ 30/30 | test negativo por bloque: `governance` sin `n_trials_total`/`dsr_family`, `identity` sin `strategy_id` |
| **BL-42** | `round(total_return, 2)` → `round(total_return / 100.0, 6)` — **un decimal bajo un sufijo `_pct`**, literalmente el bug que el BL prohíbe ⇒ 26 passed idéntico | ejercitar el **código**, no el artefacto: ledger sintético de retorno conocido y aserción sobre la salida en memoria |
| **BL-06** | reescribir el widget como `fetch('/api/produc' + 'tion/approve')` y `<button>Comprar ahora</button>` ⇒ 28 passed | colapsar concatenaciones antes de buscar y normalizar mayúsculas; o prohibir todo `POST/PUT/DELETE` en la superficie sea cual sea la URL |

## SIN TEST

**BL-36** — se puede invertir cualquier decisión de la matriz de verdad, **declarando dos
escritores para el mismo atributo**, y ningún gate se mueve. `grep db-truth-matrix` solo
encuentra el generador y prosa. Falta un test que parsee la matriz contra
`.claude/generated/db-inventory.json` y falle si una tabla tiene más de un escritor, si una
tabla DEPRECATED conserva lectores, o si una decisión cambia de valor sin ADR.

---

## Lo que este tablero enseña, más allá de los BLs

El patrón no es "los tests comprueban subcadenas". Es más amplio y aplica a los dos ingenieros:
**se prueban las PIEZAS y no el CABLEADO.** `check_families` neutralizable y `claim_order_dispatch`
destruible son el mismo defecto; `ExecutionService` sin importadores y `PaperCandidatesPanel` sin
montaje, también. De ahí sale la regla operativa: **todo validador que alimente un gate necesita un
test que demuestre que el gate lo invoca**, y todo componente que sea una garantía necesita un test
que demuestre que está montado donde dice.
