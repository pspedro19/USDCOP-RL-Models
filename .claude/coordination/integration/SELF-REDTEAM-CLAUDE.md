---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/contracts/forecast_output.py
  - src/contracts/policy.py
  - src/contracts/policy_version.py
  - usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts
  - usdcop-trading-dashboard/lib/contracts/policy-version.contract.ts
  - usdcop-trading-dashboard/lib/passport/compose.ts
  - usdcop-trading-dashboard/tests/setup.ts
  - tests/fixtures/forecast_output_cases.v1.json
  - tests/fixtures/policy_backend_cases.v1.json
  - tests/regression/test_forecasting_caveat_present.py
  - scripts/validation/report_ledger_dsr.py
  - scripts/pipeline/export_control_tower.py
  - scripts/analysis/generate_interpretability.py
  - scripts/diagnostics/db_inventory_matrix.py
  - src/strangler/parity.py
---

# AUTO-RED-TEAM — 7 commits de 2026-07-28 (`6e06df4..cdd6494`)

**Veredicto global: RECHAZO.** 8 BLOQUEANTE, 14 GRAVE, 13 MENOR.
Arreglados en esta sesión: **7** (todos con rojo→verde demostrado). El resto se
documenta sin arreglar porque exige decisiones de operador o commitear ficheros.

**El hallazgo transversal, y el que más duele:** casi todo lo entregado hoy se
verificó **sobre un árbol sucio y ejecutando los tests a mano**. Nada de lo nuevo
está en CI, y dos cosas que el autor declaró verdes están **rojas en un checkout
limpio**. Cuatro auditorías independientes llegaron a la misma conclusión por
caminos distintos.

---

## 0. DELTA REAL vs BASELINE (medido en checkout limpio, no en el árbol sucio)

`BASELINE.md` dice "47 fallos de frontmatter pre-existentes, todo lo demás VERDE".
Ese 47 está medido **sobre el árbol sucio** (incluye `.claude/codex/*` sin trackear).
La comparación honesta es checkout limpio de `6e06df4^` vs checkout limpio de `cdd6494`:

```
$ git archive 6e06df4^ | tar -x -C clean_before   &&  pytest tests/regression -q
8 failed, 991 passed, 24 skipped
$ git archive cdd6494  | tar -x -C clean_cdd6494  &&  pytest tests/regression -q
ERROR tests/regression/test_strangler_cop.py
!!!!! Interrupted: 1 error during collection !!!!!
$ ... --ignore=tests/regression/test_strangler_cop.py
9 failed, 1039 passed, 24 skipped

$ comm -13 before.txt after.txt     # fallos NUEVOS
FAILED tests/regression/test_knowledge_frontmatter.py::test_code_anchors_point_at_real_files[.claude/specs/platform/db-truth-matrix.md]
$ comm -23 before.txt after.txt     # arreglados
(vacío)
```

**DELTA = +1 fallo + 1 error de colección.** El commit `14687cd` afirma
*"Verificado por la raiz: frontmatter 47 failed = BASELINE (DELTA 0)"*. Eso es
**falso en árbol limpio**: DELTA = +1, y el fallo nuevo es el documento que ese
mismo commit añade.

---

## BLOQUEANTES

### S-01 · Paridad Py↔TS rota en 5 entradas — el punto exacto donde CODEX ya rechazó 4 veces · **ARREGLADO**

`src/contracts/forecast_output.py:95-98,358-360,402` · `usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts:160-230`

BL-15 afirma *"Grammar accepted by BOTH runtimes"*, *"exact mirror"*, *"same
characters"*, y respalda la afirmación con 90 casos compartidos y pin SHA-256.
Los 90 casos son buenos. Ninguno ataca los cinco huecos siguientes, y **los cinco
divergen**:

| # | Entrada | Python | TypeScript |
|---|---|---|---|
| A | `٢٠٢٦-٠١-٠١T٠٠:٠٠:٠٠Z` (dígitos árabo-índicos) | **VÁLIDO** `errors=[]` | RECHAZADO |
| B | `0000-01-01T00:00:00Z` | **CRASH** `ValueError: year 0 is out of range` | **VÁLIDO** `errors=[]` |
| B2 | `0099-12-31` → `0100-01-01` | **VÁLIDO** | RECHAZADO `target_time must be > as_of` |
| C | entero de 401 dígitos | **CRASH** `OverflowError` | RECHAZADO (`Infinity`) |
| D | `lower=9007199254740993` | RECHAZADO (fuera de intervalo) | **VÁLIDO** |

Salida real (Python):
```
=== A) UNICODE DIGITS (Arabic-Indic) ===
py errors: []
py parsed: 2026-01-01 00:00:00+00:00
=== B) YEAR 0000 ===
py RAISED ValueError : year 0 is out of range
=== C) HUGE INT point ===
py RAISED OverflowError : int too large to convert to float
```
Salida real (TS, mismo payload):
```
A_TS_ERRORS= ["as_of is not strict ISO8601 ...: \"٢٠٢٦-٠١-٠١T٠٠:٠٠:٠٠Z\"", ...]
B_TS_ERRORS= []
B2_TS_ERRORS= ["target_time must be > as_of"]
D_TS_PARSED= {"point":9007199254740992,"lower":9007199254740992,...} ERRORS= []
```

**Tres causas raíz distintas, todas de manual:**
1. **`\d` no es `[0-9]` en Python.** `re` con patrón `str` casa **cualquier dígito
   decimal Unicode**; el `\d` de JavaScript es ASCII puro. `int()` convierte
   alegremente `٢٠٢٦`. La afirmación "mirrored character by character" es cierta
   *literalmente* y falsa *semánticamente*.
2. **`Date.UTC` remapea los años 0-99 a 1900-1999** (quirk legacy de `Date`). Por
   eso `0099-12-31` se volvió 1999-12-31 mientras `0100-01-01` se quedó en el año
   100, invirtiendo el orden. Python usa `datetime`, que no remapea — pero **no
   admite el año 0**, así que el otro extremo del rango también rompe.
3. **`math.isfinite(int)` revienta** con enteros fuera del rango float64, y
   `JSON.parse` **redondea** enteros sobre 2^53 antes de que la validación los vea.

Lo más grave no es que diverjan: es que B y C hacen **crashear** una función cuyo
contrato documentado es *"Returns the list of contract violations"* / *"Raises
`ForecastOutputError` (never `TypeError`)"*. El muro de ingesta es fail-**crash**,
no fail-closed.

**Arreglo (rojo→verde demostrado):** 6 casos nuevos en el fixture compartido,
pin re-generado (90→96 casos), pins actualizados en **ambos** runners.

Rojo, antes de tocar el código — nótese que los fallos son **complementarios**,
que es la firma de una asimetría bilateral real:
```
# Python
FAILED ...test_case[intl_unicode_digits_as_of]
FAILED ...test_case[date_year_0000]
FAILED ...test_case[num_point_int_beyond_double_range]
3 failed, 94 passed
# TypeScript
× date_year_0000 -> invalid ... expected 'valid' to be 'invalid'
× valid_year_0099_to_0100 -> valid ... errors=["target_time must be > as_of"]
× num_bounds_beyond_safe_integer -> invalid ... expected 'valid' to be 'invalid'
Tests 3 failed | 99 passed
```
Verde, después:
```
132 passed          (Python)
Tests 102 passed    (TypeScript)
```
Fixes: `\d`→`[0-9]` en el patrón Python; `Date.UTC` sustituido por aritmética
entera civil de Hinnant (`daysFromCivil`) — el **mismo** algoritmo que BL-46 ya
usaba bien, así que el patrón correcto ya existía en el repo; año 0000 rechazado
explícitamente en ambos lados (dominio 0001..9999); y regla de finitud bilateral
(`_is_finite` / `finite`) que rechaza enteros fuera de rango **y** fuera de
precisión float64 en los dos runtimes.

Bonus arreglado de paso: el TS comparaba instantes en un contador único de
microsegundos, que **desborda la precisión exacta de 2^53 pasado ~el año 2255** —
dos timestamps lejanos separados por 1 µs habrían comparado iguales en TS y
distintos en Python. Ahora se comparan `(segundos, micros)` por separado.

### S-02 · La misma clase de defecto en BL-46, en el contrato de políticas · **ARREGLADO**

`src/contracts/policy.py:81-85` · `src/contracts/policy_version.py:143,674`

BL-46 heredó dos de los tres defectos de S-01. Salida real:
```
BL46 unicode ts regex match: True
BL46 require_instant OK -> '٢٠٢٦-...'
BL46 epoch_seconds   -> 1767225600           <-- epoch REAL desde dígitos árabes
BL46 require_finite_number(huge) RAISED OverflowError : int too large to convert to float
```
TS sobre la misma cadena: `BL46_TS_regex_match= false`.

Un `frozen_at` / `valid_from` con dígitos no-ASCII entra en Python, produce un
epoch real y ordena ventanas de señal; en TS ni siquiera parsea. `policy_hash`
viaja a cada `strategy_signal` para atribución exacta — es el peor sitio posible
para que los dos lados no coincidan sobre qué es un instante válido.

**Arreglo:** 5 casos nuevos en `tests/fixtures/policy_backend_cases.v1.json`,
pin regenerado. Rojo:
```
FAILED ...test_case[pv_unicode_digits_frozen_at]
FAILED ...test_case[sig_unicode_digits_valid_from]
FAILED ...test_case[sig_target_exposure_int_beyond_double_range]
FAILED ...test_case[cfgvals_number_int_beyond_double_range]
FAILED ...test_case[cfgvals_number_beyond_safe_integer]
5 failed, 119 passed        (Python)
Tests 1 failed | 103 passed (TypeScript)
```
Verde: `256 passed` (Python) · `Tests 298 passed` (TS, los 5 ficheros de contratos).
Se extrajo `is_bilaterally_finite()` y se usa en los **dos** sitios que
crasheaban (`require_finite_number` y `ConfigField.validate_value`), con espejo
`finiteNumber()` en TS aplicado también a `validateConfigValue` (que llamaba a
`Number.isFinite` directamente, saltándose el helper).

### S-03 · BL-31 no importa en un clon limpio: `src/identity/` **no está commiteado**

`src/strangler/parity.py:20`

```
$ git ls-files src/identity/          -> (vacío)
$ git status --porcelain src/identity/ -> ?? src/identity/
# en checkout limpio de cdd6494:
src\strangler\parity.py:20: in <module>
    from src.identity.canonical import CanonicalizationError, canonical_json_bytes, semantic_hash
E   ModuleNotFoundError: No module named 'src.identity'
ERROR tests/regression/test_strangler_cop.py
```

El commit `6f76934` entrega 1.345 líneas de `src/strangler/**` + 622 líneas de
tests que **no se pueden ni recolectar** fuera de la máquina del autor. El módulo
entero está muerto en un clon limpio y su evidencia de tests es irreproducible.
Es el único error de importación **nuevo** de hoy (los otros 6 — `regime`,
`features.calculators`, etc. — son pre-existentes, verificado en ambos checkouts).

**NO arreglado:** el arreglo es commitear `src/identity/`, y tengo prohibido
commitear. Es la primera acción a tomar.

### S-04 · `/passport` publica Sharpe y p-value de una estrategia con **1 trade**

`usdcop-trading-dashboard/lib/passport/compose.ts:346-363,628-630`

La constitución §6 y `strategy-contract.md` §6 prohíben reportar Sharpe o
p-value con N<20. El guard existe y **funciona** — en ambos lenguajes, con
mensajes byte-idénticos — pero se activa sobre `n_trades.value`, y si es `null`
**retorna sin tocar nada**. El compositor lee el conteo de `headline.trades`, una
clave que la mayoría de manifests publicados no tiene. Ejecutado contra el
`public/data` real:

```
REAL btc_hodl_b1 backtest: n_trades={"value":null} | sharpe={"value":0.793} | p_value={"value":0.0242} | insufficient_trades=false
validateStrategyPassport(real btc_hodl_b1) = []
REAL sleeve btc_hodl_b1: sharpe=0.793 n_trades=null insufficient=false dsr=0.8357
```
`btc_hodl_b1` tiene **1 trade** (`trades_2025.json` → 1 elemento; `summary_2025.json`
→ `n_long:1, n_short:0`). El fallback correcto existe 300 líneas más arriba
(`compose.ts:322-323` suma `n_long+n_short`) y no se usa aquí.

El mensaje del commit afirma *"El validador rechaza en AMBOS lenguajes un Sharpe
publicado con N<20"*. Es cierto para el caso declarado en el fixture y **falso en
la superficie real**: el guard es ciego, simétricamente, en los dos lenguajes.

### S-05 · Ningún candado de hoy está en CI. Ninguno.

```
$ grep -rn "vitest|npm run test|npm test" .github/workflows/     -> (vacío)
$ grep -rn "playwright|e2e" .github/workflows/                   -> (vacío)
$ grep -rn "check_trial_ledger|report_ledger_dsr|export_control_tower" .github/workflows/ Makefile -> (vacío)
$ grep -rn "db_inventory_matrix|db-inventory" .github/ Makefile   -> (vacío)
```
`ci.yml:139,206` corre `pytest tests/unit/` y `tests/integration/` — nunca
`tests/regression/`. `specs-gate.yml:49-67` corre una **lista enumerada** de 6
tests de regresión; ninguno de los de hoy está en ella.

Quedan fuera de CI: el muro de honestidad (`test_forecasting_caveat_present.py`),
toda la suite vitest de render, la spec Playwright de BL-05, los 22 mutation-tests
de gobernanza, `test_bl09_bl11_bl12_governance.py`, `test_strangler_cop.py`, el
gate DSR y los dos `--check` de artefactos. Un BL titulado literalmente
*"CI muralla frontend"* se cerró como `IMPLEMENTED` con **cero CI**.

### S-06 · El muro de BL-06 no muerde: basta con crear el fichero fuera de su perímetro

`tests/regression/test_forecasting_caveat_present.py:684-687`

`_forecasting_component_files()` escanea exactamente dos rutas. Mutación real:
se creó `components/gm/views/ForecastingApprovePanel.tsx` con
`fetch('/api/production/approve')`, `fetch('/api/execution/orders')` y un botón
`COMPRAR`, y se montó dentro del JSX de `ForecastingView`:
```
MUT-7 applied: panel de aprobacion/ejecucion montado en /forecasting
============================= 25 passed in 0.67s ==============================
```
25/25 verde con un botón que aprueba a producción y manda órdenes desde la
superficie diagnóstica. `BL-06-...md:66-67` afirma *"el día que forecasting pueda
aprobar o ejecutar, este test se pone rojo"*. **Es falso.**
(El candado **sí** muerde dentro del perímetro: la misma mutación en
`ForecastingView.tsx` da `1 failed, 24 passed` señalando la línea. El problema es
el alcance, no la lógica.)

### S-07 · El caveat de honestidad se puede ocultar a `free`/`subscriber` con los dos candados verdes

`tests/regression/test_forecasting_caveat_present.py:453` ·
`usdcop-trading-dashboard/tests/unit/components/forecasting-caveat-surfaces.test.tsx:55-57`

```jsx
{__isInternal && (
  /* } el candado estructural cuenta llaves literales, tambien en comentarios */
  <ForecastDisclaimer variant="weekly" />
)}
```
```
--- PYTEST ---  25 passed
--- VITEST ---  Tests 21 passed (21)
```
El disclaimer queda invisible exactamente para la población a la que protege, y
**nada se pone rojo**. Dos causas independientes, ambas necesarias:
1. `:453` `depths.append(prefix.count("{") - prefix.count("}"))` cuenta llaves
   literales **incluidas las de comentarios y strings** → se reequilibra trivialmente.
2. El test de render mockea `next-auth/react` globalmente a `role:'admin'` →
   **ningún test ejercita jamás una sesión no-admin**.

Reabre CXD-032 en la forma que la remediación afirma explícitamente haber cerrado.

### S-08 · `db-inventory.json` no es reproducible desde el commit que lo entrega

`.claude/generated/db-inventory.json` · `.claude/specs/platform/db-truth-matrix.md`

Sus conteos salen de **12 ficheros SQL sin trackear** (`058`, `059`, `070`-`078`,
`init-scripts/27`). Ejecutando el generador **commiteado** dentro de un checkout
limpio del **propio commit**:

| | JSON commiteado | checkout limpio del mismo commit |
|---|---|---|
| objetos declarados | **242** (168 tablas + 74 vistas) | **200** (129 + 71) |
| ORPHAN_DDL_UNVERIFIED | 143 | **109** |

**42 de 242 objetos declarados (17%) no existen en git.** El hallazgo estelar de
BL-36 — *"el repo declara 242 objetos … ~109 probablemente no existen"* — no es
derivable del repo. Y el gate que lo protegería está rojo y sin cablear:
```
$ python scripts/diagnostics/db_inventory_matrix.py --check
...db-inventory.json is stale - run with --write     EXIT=1
```
mientras `db-truth-matrix.md:500` afirma que está *"verde (JSON determinista y al día)"*.

---

## GRAVES (resumen; detalle completo en los anexos de cada auditoría)

| ID | Qué está mal | Fichero |
|---|---|---|
| S-09 | El gate DSR deflacta con `dsr_family` (N=60), el más **permisivo**; constitución §2 y ADR-0022 exigen el total del activo (111) o global (239). `DSR_family ≥ DSR_cluster ≥ DSR_global` siempre. Verdicto de hoy no cambia (0.61-0.64 < 0.95); el **mecanismo** sí está roto. | `scripts/validation/report_ledger_dsr.py:120-121` |
| S-10 | La rejilla sigma **no está pre-declarada**: cada YAML declara la suya, nada la valida. La declarada omite las dos sigmas más punitivas que el propio repo ha medido (0.0473 y 0.094229). Con la empírica, DSR cae de 0.6368 a **0.2113**. | `report_ledger_dsr.py:50-61`, `registries/families/smart_simple.yaml:38` |
| S-11 | El mismo YAML embarca una **segunda construcción** de la misma serie 2025 y afirma "no altera el veredicto (ambas < 0.95)". **Falso por este código**: construcción B da DSR **0.9639-0.9822 > 0.95 a todo N**. Serie + sigma + decisión pendiente = tres palancas de selección sobre la barra. | `registries/families/smart_simple.yaml:43-52` |
| S-12 | `governance.json` está **obsoleto dentro de su propio commit**: 8/10 familias con `path:null` y "SIN ARCHIVO DE FAMILIA DECLARADO", cuando ese commit crea los 10 YAML. `export_control_tower.py --check` → `EXIT=1`, no cableado. | `usdcop-trading-dashboard/public/data/control-tower/governance.json` |
| S-13 | `n_family`/`n_cluster`/`n_global` publicados son un `max()` de contadores corrientes por activo, no conteos. `spx500.n_family=94` y `btcusdt.n_family=78` no son el N de ninguna familia que esos activos posean. El Passport los renderiza como el N que deflacta esa estrategia. | `scripts/pipeline/export_control_tower.py:112-118` |
| S-14 | Tres DSR distintos publicados del mismo objeto (0.0587 con N=59 en el gate de Vote 2 / 0.6368 con N=60 / 0.7235 con N=72), y el N del ledger (111) no llega a ninguna superficie de decisión. Sin SSOT ni test cruzado. | `approval_state.json`, `HYPOTHESIS-REGISTRY.md:16` |
| S-15 | El candado anti-blanqueo de multiplicidad se derrota con un **rename consistente**: renombrando `hypothesis_key` en los 4 `*_vol` y vaciando `sibling_families`, N pasa de 6 a 1 por familia y el validador sale `EXIT=0`. | `check_trial_ledger.py:658-697` |
| S-16 | BL-12 implementa **1 de los 3** campos de provenance que nombra la constitución: `action_trial_id` y `research_cluster` aparecen en **cero** YAML y en **cero** código de validación. | `check_trial_ledger.py:535-593` |
| S-17 | El "guard de prosa estructural" gobierna **solo** el bloque delimitado `LEDGER-TOTALS`. Inyectando "DSR 0.99, Sharpe 3.35, p=0.0001 — edge confirmado" fuera de él: `EXIT=0`, 46 passed, cero fallos nuevos. | `registries/README.md:70`, `check_trial_ledger.py` |
| S-18 | `--force-rollback-only` permite registrar la capa **de ejecución** como `MIGRATED` con 15/15 criterios pendientes, y la línea del ledger **no marca que fue forzada**. Sin test que cubra la ruta forzada. | `scripts/validation/check_strangler_parity.py:202-208,243-247` |
| S-19 | El ledger strangler es `open("ab")` sin cadena de hashes ni chequeo de estado previo, y `derive_states` confía en "gana la última transición" → una línea a mano marca cualquier capa como migrada. Contrasta con BL-09, donde el ledger **sí** está encadenado. | `src/strangler/parity.py:187`, `gates.py:41-51` |
| S-20 | La cabecera constitucional se hardcodea idéntica en **todos** los artefactos de interpretabilidad, pero la ruta lineal atribuye **1649/1654 filas (99,7%) del propio train** bajo un cartel que dice "solo test-folds". La UI muestra el cartel y, dos líneas más abajo, el `scope` que lo contradice. | `scripts/analysis/generate_interpretability.py:66,96-110` |
| S-21 | Correr pytest **muta artefactos trackeados** (`data/interpretability/**`) y los valores derivan (`n_days 7943→7942`, `pnl_gross 1.0916→1.0905`): la evidencia commiteada es una foto de una extracción viva bajo un directorio `version` congelado. | `tests/unit/test_interpretability_artifacts.py:47-53` |
| S-22 | La evidencia de lectores está estructuralmente subcontada: un `elif` hace que todo fichero que escribe **y** lee se registre solo como escritor. **116 aristas lector perdidas**. Las decisiones D-01..D-12 ("0 lectores rotos tras cada drop") descansan justo en esas listas. | `scripts/diagnostics/db_inventory_matrix.py:324-327` |

---

## MENORES (13, condensados)

- **S-23 · ARREGLADO** — El test nuevo de BL-46 `StrategyEngineExplanation.test.tsx`
  pasa **solo** y falla **en suite**: `tests/setup.ts` nunca llamaba `cleanup()` de
  Testing Library, así que el DOM se acumulaba entre ficheros.
  Rojo: `npx vitest run forecasting-caveat-surfaces StrategyEngineExplanation` →
  `Tests 4 failed | 22 passed`; `... PaperCandidatesPanel StrategyEngineExplanation` →
  `Tests 4 failed | 14 passed`. Verde tras añadir `cleanup()` global:
  `26 passed` y `18 passed`. Suite completa `tests/unit/`: **52 failed → 44 failed**
  (los 44 restantes son `Button.test.tsx` y `replayApiClient.test.ts`, verificados
  como **no tocados** por los 7 commits).
- **S-24** — El pin anti-drift del fixture de BL-46 es `toBeGreaterThanOrEqual(90)`,
  no un conteo exacto como el de BL-15: se puede **borrar** un caso y regenerar el
  SHA sin que nada proteste. (Los 5 casos que añadí sí lo dejan en 102.)
- **S-25** — `\d` con el mismo problema Unicode en `src/strangler/contracts.py:248`
  (`re.fullmatch(r"\d+\.\d+\.\d+", version)`): `١.٢.٣` pasa como versión válida.
  Sin espejo TS, por eso es menor. **No arreglado** (BL-31 está muerto en clon limpio, S-03).
- **S-26** — Umbral constitucional `20` hardcodeado 4 veces en `compose.ts:389,630,691,692`
  aunque el fichero ya importa `MIN_TRADES_FOR_RATIOS` del contrato. Y el test espejo
  Py↔TS no comprueba `DSR_BAR`, que puede derivar sin detección.
- **S-27** — `withdrawal_protocol_signed = wd_path.exists()`; la UI lo renderiza como
  "protocolo **firmado**". Que un fichero exista no es una firma (§5 exige firma ex-ante).
- **S-28** — `n_max_trials` se publica con `source.path = 'src/contracts/passport.py'`:
  un fichero fuente presentado como artefacto publicado.
- **S-29** — Los "bar pre-firmados" tienen `declared_at` 2026-07-27/28 sobre 239 filas
  de trials observados en 2025-2026: es una barra **retro-ajustada**, y el README la
  llama "pre-firmada".
- **S-30** — `fit.n_train = 4959` es la suma de los 5 folds sobre un dataset de 1703 filas,
  servido bajo el mismo nombre de campo que la rama lineal usa para un conteo real.
- **S-31** — Dos tests contradictorios conviven: `test_paths_follow_...` exige artefactos
  bajo `public/` (rojo desde `57c3e1c`) y `test_artifacts_moved_out_of_public` exige lo
  contrario. BL-20 declaró COMPLETE citando solo la suite que pasa.
- **S-32** — Tests tautológicos: `gross ≈ beta + timing` es cierto por construcción
  (`timing := gross - beta`); poner `beta = 0.0` deja la suite verde. Y 4/38 casos de
  BL-31 re-afirman invariantes que el propio `contracts.py` ya impone al cargar.
- **S-33** — El lock de leakage de la ruta lineal no muerde: cambiar
  `StandardScaler().fit(Xtr)` a fit sobre **todo** el dataset deja la suite igual, y el
  artefacto sigue emitiendo `"scaler": "StandardScaler train-only"` — string que no
  asegura nadie.
- **S-34** — "14 esquemas" en la prosa vs **23** en el JSON que cita como evidencia, de los
  cuales `'create'` y `'for'` son falsos positivos del regex sobre comentarios SQL.
  Ninguno de los dos números es correcto.
- **S-35** — Los tres MD de BL-09/11/12 siguen en `status: PARTIAL` describiendo el estado
  *previo*, y su sección "Verificación" promete un CI que no existe.

---

## Ángulos que salieron LIMPIOS (dicho explícitamente)

- **Anti-look-ahead en TreeSHAP (BL-20): CORRECTO.** Leí la aritmética de folds línea
  a línea. `prev = df[df.date < cut]`, `train = prev.iloc[:-HORIZON]`: la fila
  superviviente `k-5` tiene `y5` que mira hasta `k`, y la primera fila de test es
  `k+1`. **La purga está en el lado correcto, es ajustada y no tiene off-by-one.**
  `StandardScaler` se ajusta **por fold**. Las features son causales, el macro va
  `shift(1)` + `merge_asof(backward)`. **No encontré fuga.** El defecto es que nada
  de esto está testeado: borrar la purga entera deja la suite idéntica (S-02 del anexo BL-20).
- **Path traversal en `/api/passport/[strategyId]`: LIMPIO.** Rechaza todo lo que no
  case `^[a-z0-9_-]+$` antes de tocar el fs; `../../../package`, `..%2F..%2Fpackage`
  y `a/../registry` devuelven `null` sin leer fichero.
- **JSON safety: LIMPIO en las tres superficies nuevas.** `export_control_tower.py:277`
  usa `safe_json_dumps`; los 6 artefactos de interpretabilidad escanean `badtokens=[]`;
  `governance.json` tiene 0 ocurrencias de `Infinity`/`NaN`/`undefined`. Ningún
  `profit_factor` infinito.
- **§7 (no recomputar en frontend): LIMPIO.** `PassportView.tsx` solo formatea; los
  gates se copian literalmente de `approval_state*.json`. `compose.ts`, pese a sus 870
  líneas, está en capas y no es un God object.
- **Aritmética del ledger de trials: LIMPIA.** Recontada independiente: 239 total,
  FT=55/AT=184, `usdcop 111, xauusd 77, btcusdt 34, spx500 17`. Suma exacta, casa con
  el bloque `LEDGER-TOTALS`, con cada front-matter y con cada `trials_charged`. Cadena
  de hashes correcta.
- **N<20 en interpretabilidad y gobernanza: LIMPIO.** 0 hits de `sharpe|p_value` en
  `data/interpretability`. (El fallo de N<20 está en el Passport, S-04.)
- **Los candados de a11y de BL-05 son el mejor trabajo de los 7 commits: 4/4 muerden**
  y afirman semántica real (`scope`, `aria-hidden`, sr-only, `tabindex`), no snapshots.
  Su debilidad es la entrega (S-05), no el diseño.
- **Los 22 mutation-tests de gobernanza son reales**: cada uno falla de verdad cuando
  se aplica su mutación. El número del commit es exacto. El problema es que nadie los corre.

---

## Qué NO sobreviviría una re-review estricta de CODEX, y por qué

**1. BL-15 y BL-46 tal como se commitearon.** Es el terreno donde CODEX ya rechazó
cuatro veces, y donde va a mirar primero y con lupa. Los cinco huecos de S-01/S-02
son exactamente el catálogo que se ataca cuando alguien escribe "paridad bilateral":
Unicode, límites de calendario, límites numéricos de JS. El fixture de 90 casos es
buen trabajo y **transmite una confianza que no estaba ganada** — es más peligroso
que no tener fixture, porque invita a dejar de mirar. Ya está arreglado; sin ese
arreglo, rechazo seguro.

**2. Todo lo que se declaró "verificado".** Tres commits dicen
`IMPLEMENTATION_COMPLETE_UNVERIFIED`, lo cual es honesto. Pero `14687cd` afirma
"DELTA 0" y es **+1 en árbol limpio**; `db-truth-matrix.md` afirma que su gate está
verde y **sale 1**; `passport-control-tower.md` afirma que `--check` está al día y
**sale 1**; el mensaje de BL-32 afirma que N<20 se rechaza en ambos lenguajes y la
superficie real **publica Sharpe de 1 trade**. El patrón es único y sistemático:
**se verificó sobre el árbol sucio del autor**. CODEX audita desde el commit. Cada
una de esas cuatro afirmaciones se cae con un comando.

**3. BL-06 cerrado como `IMPLEMENTED`.** Un BL llamado "CI muralla frontend" sin una
sola línea en `.github/workflows/`, con el muro rodeable creando un fichero fuera de
dos rutas hardcodeadas, y con el caveat ocultable a los usuarios de pago con los dos
candados verdes. No es cerrable; hay que bajarlo a `PARTIAL`.

**4. BL-31 entero.** No importa en un clon limpio. No hay discusión posible sobre un
módulo que no carga.

**5. El gate DSR como *control*.** Como **contabilidad** es honesto y la aritmética
reconcila exactamente — eso lo defiendo. Como **control** no existe: no está en CI,
deflacta con el N más pequeño disponible, y sus tres entradas (serie, sigma, decisión
pendiente) las elige la misma persona que se beneficia. Y el propio YAML embarca una
construcción alternativa que **pasa la barra a todo N**, bajo una nota que afirma lo
contrario. Es el hallazgo que arreglaría primero después de los BLOQUEANTES.

**Lo que sí defiendo sin reservas:** la purga walk-forward de BL-20 (correcta, la
verifiqué línea a línea), los candados a11y de BL-05 (4/4 muerden con semántica real),
la aritmética del ledger, la seguridad de la ruta del Passport, la JSON-safety, y la
decisión de BL-46 de usar aritmética civil entera en lugar de `Date.parse` — que es
precisamente el patrón correcto que BL-15 no usó y que he portado allí.

---

## Cambios aplicados en esta sesión (7 arreglos, ninguno commiteado)

```
src/contracts/forecast_output.py                              |  50 +-
src/contracts/policy.py                                       |  11 +-
src/contracts/policy_version.py                               |  38 +-
usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts |  90 +-
usdcop-trading-dashboard/lib/contracts/policy-version.contract.ts  |  17 +-
usdcop-trading-dashboard/tests/setup.ts                       |  12 +-
tests/fixtures/forecast_output_cases.v1.json                  | 146 +-   (90 -> 96 casos)
tests/fixtures/policy_backend_cases.v1.json                   | 139 +-   (97 -> 102 casos)
tests/unit/test_forecast_output_contract.py                   |   4 +-   (pin 90 -> 96)
usdcop-trading-dashboard/tests/unit/contracts/forecast-output-parity.test.ts | 6 +-  (pin)
```
Verificación final: `pytest` contratos **256 passed** · `vitest tests/unit/contracts/`
**298 passed** · `tsc --noEmit` **635 líneas** (BASELINE ~639, 0 errores nuevos en los
ficheros tocados) · `vitest tests/unit/` **52 failed → 44 failed**.
