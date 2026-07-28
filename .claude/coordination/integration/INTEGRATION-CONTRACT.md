---
kind: spec
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - .claude/coordination/ASSIGNMENTS.md
  - .claude/coordination/integration/BDD-MATRIX.md
  - .claude/coordination/integration/TDD-GAPS.md
  - src/identity/canonical.py
  - src/strangler/parity.py
  - src/policy_engine/runner.py
  - src/strategies/policies/loader.py
  - src/contracts/passport.py
  - usdcop-trading-dashboard/lib/passport/compose.ts
  - scripts/validation/check_trial_ledger.py
  - database/migrations/075_fact_position_pnl.sql
---

# INTEGRATION-CONTRACT — dónde se tocan CLAUDE y CODEX

> **PROPUESTA** por `claude-root-9c3f1e42` el 2026-07-28. ACTIVE cuando CODEX
> cofirme frontera por frontera.
> Este documento no describe lo que debería pasar: describe **lo que hoy pasa**,
> con evidencia de línea, y separa lo acordado de lo divergente.
> Toda evidencia es lectura directa o ejecución de esta sesión. Lo que no verifiqué
> está marcado `NO VERIFICADO`.

---

## 0. Advertencia sobre el árbol en movimiento

`git status` reporta **255 rutas untracked**, y prácticamente **todo el lote de
CODEX está sin commitear**: `src/{identity,metrics,lineage,portfolio,market,
governance,orchestration}/`, `src/execution/{events,service}.py`,
`src/validation/*_harness.py`, y las nueve migraciones `070`-`078`.

Consecuencias que condicionan cada frontera de abajo:

1. **No hay hash que revisar.** El protocolo exige cross-review contra un hash
   inmutable (`PROTOCOL.md` §1.5, canal §5.7). Sobre ficheros untracked eso es
   imposible: lo que audito hoy puede no ser lo que se commitee.
2. **El árbol se movió durante la auditoría.** Caso concreto: un subagente reportó
   un `ImportError` vivo en `src/orchestration/semantic_diff.py:10`
   (`canonical_bytes`, `canonical_hash` — nombres inexistentes); al re-verificarlo
   yo mismo minutos después, la línea ya importaba
   `canonical_json_bytes, semantic_hash as canonical_semantic_hash`. **El hallazgo
   era cierto cuando se hizo y falso cuando se comprobó.** Trata todo hallazgo
   sobre untracked como fechado.
3. **Hay falso-verde estructural.** Ver F-01.

**Precondición de esta fase**: CODEX commitea su lote (aunque sea
`IMPLEMENTED_UNVERIFIED`) **antes** de que se escriba un solo test de frontera.
Sin eso, la fase de verificación mide un árbol que no existe en git.

---

## 1. Índice de fronteras

| # | Frontera | Dueño del código | Veredicto hoy |
|---|---|---|---|
| **F-01** | `src/identity/` ← consumido por `src/strangler/` | CODEX (módulo) / CLAUDE (consumidor) | **BLOQUEANTE — consumidor remediado, ROJO A PROPÓSITO hasta que CODEX commitee** |
| **F-02** | Hash canónico: identidad vs familia policy | CODEX (canonical) / CLAUDE (policy) | **PARCIAL — 4→1 dentro de la familia policy (0 digests movidos); frontera con `canonical.py` = decisión del operador** |
| **F-03** | Ledger de trials + validador | CODEX (fichero) / CLAUDE (checks 8-13) | **ACUERDO** (aditivo verificado) |
| **F-04** | DSR — SSOT constitucional | CODEX (`services/common/metrics.py`) | **DIVERGENCIA GRAVE (duplicación viva)** |
| **F-05** | Vocabulario de entorno (`held_out`) | CLAUDE (contrato) / CODEX (CHECK SQL) | **DIVERGENCIA BLOQUEANTE** |
| **F-06** | Grano de identidad: `strategy_id` vs `sleeve_id` | ambos | **DIVERGENCIA BLOQUEANTE** |
| **F-07** | Tercer reloj: `exec` vs `pnl` | CODEX (productor) / CLAUDE (consumidor) | **DIVERGENCIA GRAVE — ya perdiendo dato** |
| **F-08** | Passport ← 9 interfaces de CODEX | ambos | **0/9 COINCIDEN** |
| **F-09** | Motor de políticas vs factories FABRIC | CLAUDE / CODEX | **PARCIAL — una sola `build_policy` (lado CLAUDE cerrado); dos registries y la contradicción v11 siguen abiertos** |
| **F-10** | Contratos espejo Py↔TS | ver tabla F-10 | **PARCIAL** (3 con fixture, resto paralelo) |
| **F-11** | CI como frontera compartida | CODEX | **AUSENTE** |
| **F-12** | Migraciones DB: escribe CODEX, revisa CLAUDE | CODEX | **RIESGO OPERATIVO** |
| **F-13** | Vocabulario de fallback | ambos | **DIVERGENCIA MENOR (4 vocabularios)** |
| **F-14** | Métricas: catálogo vs escritores | CODEX (interna) | **DIVERGENCIA GRAVE (bypass propio)** |

---

## F-01 · `src/identity/` ← `src/strangler/` · **BLOQUEANTE**

| | |
|---|---|
| **Dueño** | CODEX es dueño de `src/identity/` (BL-17). CLAUDE es dueño del consumidor `src/strangler/` (BL-31). |
| **Shape acordado** | `canonicalize()`, `canonical_json_bytes()`, `semantic_hash()`, `CanonicalArtifact`, `CanonicalizationError` — expuestos desde `src/identity/canonical.py`. |
| **Test que la protege** | **NINGUNO HOY** → `TDD-GAPS G-01`. |
| **Divergencia** | **SÍ, y es la más peligrosa del lote.** |

**Evidencia (verificada por mí, no por subagente):**

```
$ grep -n "from src.identity" src/strangler/parity.py
20:from src.identity.canonical import CanonicalizationError, canonical_json_bytes, semantic_hash

$ git ls-files src/identity/ | wc -l
0
```

`src/strangler/parity.py` está **commiteado** (`6f76934`). `src/identity/` **no
está en git**. Por tanto:

- **El HEAD de CLAUDE no importa en un clone limpio.**
- Los **38 tests verdes** de `tests/regression/test_strangler_cop.py` son
  **falso-verdes**: pasan porque el directorio untracked existe en esta copia.
- Misma exposición para `tests/unit/test_codex_fabric_contracts.py` y para todo
  consumidor de `src/{execution,orchestration,portfolio,metrics,lineage,market,
  governance,validation}`.

**Nota positiva que hay que decir explícitamente**: la sospecha del brief —que
`semantic_hash` de CLAUDE y `fingerprints.py` de CODEX fueran dos implementaciones
del mismo concepto— es **FALSA**. `semantic_hash` existe en **un solo sitio**
(`src/identity/canonical.py:121`) y CLAUDE la **consume**;
`src/identity/fingerprints.py:9` también la consume vía `canonical_json_bytes`. Eso
es **composición correcta**, y es el mejor ejemplo de integración limpia que hay
entre los dos lotes. La duplicación real está en otro sitio (F-02).

**Qué debe ser verdad para cerrar**: `git ls-files src/identity/` > 0 y G-01 verde.

### REMEDIACIÓN CLAUDE · 2026-07-28 · lado consumidor cerrado, frontera SIGUE ROJA

Lo que se arregló es **la honestidad del consumidor**, no la ausencia del módulo
(eso solo lo cierra CODEX commiteando):

1. **Guard de import explícito** — `src/strangler/parity.py:20-45`. Si
   `src.identity.canonical` no está, el módulo lanza un `ImportError` que **nombra
   al dueño** (CODEX/BL-17), dice que no está commiteado y explica por qué NO hay
   fallback. Verificado simulando el clone limpio con un bloqueador en
   `sys.meta_path`: el guard dispara con ese texto.
2. **G-01 escrito y ROJO A PROPÓSITO** —
   `tests/regression/test_repo_self_contained.py::test_committed_tree_has_no_untracked_imports`.
   Recorre `git ls-files src` (git, no el filesystem: el filesystem miente sobre lo
   que contiene un clone) y parsea los imports de nivel de módulo. Único huérfano
   del árbol: `src/strangler/parity.py:21 -> src.identity.canonical`.
   El docstring prohíbe explícitamente "arreglarlo" borrándolo.
3. **Se acabó el falso-verde del strangler** —
   `tests/regression/test_strangler_cop.py::test_strangler_dependency_is_committed`
   falla mientras la dependencia esté untracked, así que la suite **no puede
   citarse como "38/38 verdes"**. En un clone limpio el guard convierte el módulo
   en error de colección: falla ruidoso, nunca skip silencioso.

Dos tests acompañan al gate para que su verde signifique algo: uno simula "el
dueño ya commiteó" (sin stagear ficheros ajenos) y demuestra que el rojo se
limpia solo con el commit; el otro es prueba de mutación del parser.
**Hallazgo propio durante el fix**: la primera versión del detector solo miraba
`tree.body`, así que el propio guard `try/import` lo cegaba — el detector se
declaraba verde sobre el defecto que acababa de envolver. Corregido descendiendo
a `try`/`if`/`with` de nivel de módulo, con test de mutación para esa forma exacta.

**Sigue rojo, por diseño, hasta `git add src/identity/` por CODEX.**

---

## F-02 · Hash canónico: identidad vs familia policy · **GRAVE (DRY/SSOT)**

| | |
|---|---|
| **Dueño** | `src/identity/canonical.py` → CODEX. `src/contracts/policy*.py` + `src/strategies/policies/loader.py` → CLAUDE. |
| **Shape acordado** | Un objeto lógico ⇒ un hash. Prefijo `sha256:` + 64 hex. |
| **Test que la protege** | **NINGUNO** → `TDD-GAPS G-02`. |
| **Divergencia** | **SÍ, silenciosa.** |

**Cinco implementaciones del mismo idiom, cuatro de ellas divergentes del canónico:**

| Fichero:línea | Serialización | Divergencia vs canónico |
|---|---|---|
| `src/identity/canonical.py:112` | `ensure_ascii=False`, `allow_nan=False`, `separators=(",",":")`, `sort_keys=True`, floats → `Decimal` `ROUND_HALF_EVEN`, `NFC`, `datetime`→`_utc_z` | **el canónico** |
| `src/contracts/policy.py:374` | `json.dumps(sort_keys, separators, allow_nan=False)` | `ensure_ascii` por defecto `True`; float crudo; sin NFC; `datetime` ⇒ `TypeError` |
| `src/contracts/policy_dsl.py:131` | ídem | ídem |
| `src/contracts/policy_version.py:210` | ídem | ídem |
| `src/strategies/policies/loader.py:98` | ídem | ídem |

**Prueba empírica sobre `{"b":1,"a":"café","f":0.1+0.2}`:**

```
identity bytes: b'{"a":"caf\xc3\xa9","b":1,"f":"0.3"}'
policy   bytes: b'{"a":"caf\\u00e9","b":1,"f":0.30000000000000004}'
EQUAL? False
```

Coinciden **solo** con payloads ASCII y sin floats. Y la divergencia es
**invisible**: `src/contracts/policy.py:72`
`HASH_PATTERN = ^sha256:[0-9a-f]{8,64}$` acepta 8-64 hex, así que ambas familias
validan. Además `src/contracts/policy.py:347` **trunca a hex16** mientras el
canónico devuelve 64.

**Qué debe ser verdad para cerrar**: los 4 helpers delegan en
`canonical_json_bytes`/`semantic_hash`, o se declara formalmente que son **dos
dominios de hash distintos con fronteras documentadas** (y entonces `HASH_PATTERN`
deja de aceptar ambos indistintamente).

**DECISIÓN DEL OPERADOR**: unificar cambia hashes ya congelados (v11/v12/v14,
`config/policies/*.yaml`). Es un re-freeze deliberado como el de `3861568`, o no
se hace. **No la tomo yo.**

### REMEDIACIÓN CLAUDE · 2026-07-28 · las 4 copias son 1; la frontera con CODEX sigue abierta

La auditoría mezclaba dos problemas distintos. Separados:

**(a) Duplicación DENTRO de la familia policy — ARREGLADO, con cero cambio de bytes.**
Las cuatro copias eran **idiom byte-idéntico entre sí** (todas
`json.dumps(sort_keys=True, separators=(",",":"), allow_nan=False)`), así que
colapsarlas en una sola función **no mueve ningún digest**. No hay re-freeze, no
hay decisión de operador, no hay trials. SSOT:
`src/contracts/policy.py::policy_canonical_hash`. Delegan
`policy.py::_fingerprint`, `policy_dsl.py::_canonical_policy_hash`,
`policy_version.py::_canonical_hash` y `loader.py::canonical_policy_hash` (esta
última conserva su PROYECCIÓN — qué claves deciden — y delega solo la
serialización).
Prueba de que nada se movió: `tests/unit/test_canonical_hash_ssot.py` pinnea los
digests **calculados con el código pre-refactor** sobre entradas adversariales
(`café`, `0.1+0.2`, `-0.0`, NFC/NFD, claves desordenadas) y comprueba que los
cuatro `governance.policy_hash` congelados de `config/policies/*.yaml` siguen
validando. `check_policy_parity.py` sigue dando exposición **idéntica** en las 3
políticas con datos reales.

**(b) Divergencia ENTRE familias (policy ↔ `src/identity/canonical.py`) — NO TOCADA.**
Sigue siendo cierto que producen bytes distintos para el mismo payload
(`ensure_ascii`, cuantización Decimal, NFC, `datetime`). Unificarlas movería
**todos** los hashes congelados del repo ⇒ re-freeze de evidencia publicada ⇒
**DECISIÓN DEL OPERADOR, sin tomar.** Queda declarado en el código
(`src/contracts/policy.py`, nota sobre el SSOT) como *dos dominios de hash con
frontera documentada*, que es la segunda salida que este mismo documento admitía.

**(c) `HASH_PATTERN` endurecido — la divergencia ya no puede ser silenciosa.**
`^sha256:[0-9a-f]{8,64}$` → `^sha256:[0-9a-f]{64}$`, en Python **y** en el espejo
TS (`policy.contract.ts`). Aceptaba 57 longitudes distintas, así que un digest de
otro idiom —o el fragmento hex16 truncado que va dentro de `signal_id`— validaba
como si fuera el fingerprint canónico. Ningún dato real usaba longitud corta
(specs, fixtures y registries: todos 64); los únicos afectados eran placeholders
`"sha256:deadbeef"` de mis propios tests, ahora digests reales.
Paridad bilateral re-verificada tras el cambio: **170/170 vitest** sobre
`policy_contract_cases.v1.json` + `policy_backend_cases.v1.json` (fixtures sin
tocar, `content_sha256` intacto).

---

## F-03 · Ledger de trials + validador · **ACUERDO** ✅

| | |
|---|---|
| **Dueño** | `scripts/validation/check_trial_ledger.py` y `registries/ledger.jsonl` → **CODEX** (BL-10). Checks 8-13 → CLAUDE (BL-09/11/12). |
| **Shape acordado** | Checks 1-7 de CODEX intactos; CLAUDE solo añade. |
| **Test que la protege** | `tests/regression/test_trial_ledger.py` (10) + `test_bl10_legacy_estimate_contract.py` (2) + `test_bl09_bl11_bl12_governance.py` (34, 22 mutaciones). **51 tests, todos verdes.** |
| **Divergencia** | **NO.** Es la frontera mejor resuelta de todo el backlog. |

**Evidencia de que el hardening fue aditivo, no una edición del código ajeno:**

```
$ git diff --stat 4c4fdf5 HEAD -- scripts/validation/check_trial_ledger.py
433 insertions(+), 0 deletions(-)
```

Cero borrados: los checks 1-7 de CODEX son byte-idénticos. Los 13 checks son
**todos fail-closed** (`main()` devuelve 1 si hay cualquier error; no hay
warnings).

**Conteos verificados por mí** (coinciden exactamente con lo declarado por CLAUDE):
239 líneas · FT=55 · AT=184 · usdcop=111 · xauusd=77 · btcusdt=34 · spx500=17 ·
génesis `FT-0001` con `prev_hash` = 64 ceros · `registries/README.md`
LEDGER-TOTALS byte-a-byte igual.

**Ejecución real:**
```
$ python scripts/validation/check_trial_ledger.py          EXIT=0
$ python scripts/validation/report_ledger_dsr.py           EXIT=0
  smart_simple_v11: DSR family/cluster/global = 0.6368/0.6212/0.6116  (bar 0.95)
```

**Tres reservas honestas que CODEX debe conocer aunque la frontera esté sana:**

1. **Nada de esto corre en CI** (F-11 / G-04).
2. **El 100 % del ledger es `env: legacy_backfill`** (239/239) y las 10 familias
   declaran `claims_edge: false`. Todas las ramas prospectivas fail-closed son
   **código muerto frente a datos reales**, y el gate DSR **hoy no gatea nada**.
   Se ejercitan solo en fixtures de mutación.
3. **La completitud es indemostrable por diseño.** Los 13 checks verifican
   consistencia interna (ledger↔YAML↔README↔front-matter). Un trial ejecutado y
   nunca cargado es invisible para todos. **El ledger prueba que nadie lo editó,
   no que alguien lo registró todo.** Debe quedar declarado, no “resuelto” con un
   test que mienta.

**Además — dos gates DSR con tres N distintos, sin cross-referencia:**
`scripts/pipeline/train_and_export_smart_simple.py:1235` usa
`n_trials = fm["n_trials_total"]` ⇒ **N=111 (total del activo)** para el gate de
Vote 1; `report_ledger_dsr.py` usa **N=60/138/239**. Ambos son de CLAUDE, pero
consumen el ledger de CODEX. Hoy no hay conflicto porque **ninguno pasa el bar**;
lo habrá el día que alguno se acerque a 0.95.

---

## F-04 · DSR — el SSOT constitucional · **GRAVE**

| | |
|---|---|
| **Dueño** | `services/common/metrics.py::deflated_sharpe_ratio` — SSOT declarado por `quant-constitution.md` §2. |
| **Shape acordado** | Todo el mundo importa; nadie reimplementa. |
| **Test que la protege** | `tests/regression/test_quant_library_gate.py::test_promoted_skills_do_not_shadow_the_metrics_ssot` — **con un punto ciego**. |
| **Divergencia** | **SÍ, duplicación viva y git-tracked.** |

**Quién delega correctamente** (esto está bien y hay que decirlo):
`.claude/skills/xasset-alpha-engine/scripts/xasset/validation.py:170-193` es un
wrapper fino que se **niega explícitamente** a llevar fallback;
`src/metrics/engine.py:15-19` importa el SSOT;
`src/validation/quant_harness.py` y `sp500_oos_gate.py` (CODEX) **no hacen
matemática de DSR** — reciben el float y lo comprueban de rango.

**Quién duplica**: `src/strategies/spx500_regime_gated_v1/deflated_sharpe.py` —
194 líneas, **git-tracked**, con `expected_max_sharpe`, `probabilistic_sharpe`,
`deflated_sharpe`, `min_track_record_length`. Usa `scipy.stats.norm` donde el SSOT
usa un `_norm_ppf` propio (Acklam) y devuelve `float` donde el SSOT devuelve
`{sr0, dsr, significant}`. **Está viva**: `kernels.py:18` la importa.

**Lo relevante de la frontera**: el gate que existe para impedir exactamente esto
itera **solo `.claude/skills/**.py`**. No puede ver `src/`. **El único duplicado
real del repo está en el punto ciego del test escrito para detectarlo.**

**Debilidad adicional en el lado de CODEX**: `quant_harness.py` y
`sp500_oos_gate.py` validan que el DSR **exista y esté en rango**, nunca que fuera
**computado correctamente**. Un manifiesto con `dsr: 0.99` inventado pasa el gate.

**Qué debe ser verdad para cerrar**: G-03 verde con el iterador ampliado a `src/`,
`scripts/` y `services/`.

---

## F-05 · Vocabulario de entorno · **BLOQUEANTE**

| | |
|---|---|
| **Dueño** | CLAUDE define `PASSPORT_ENVS`; CODEX define los CHECK SQL. **Nadie declaró ser SSOT.** |
| **Shape acordado** | **NO HAY.** Nunca pasó por `CONTRACTS.md`. |
| **Test que la protege** | **NINGUNO** → `TDD-GAPS G-06`. |
| **Divergencia** | **SÍ.** |

| Lado | Valores |
|---|---|
| CLAUDE — `src/contracts/passport.py:58` (+ espejo TS) | `backtest`, `held_out`, `paper`, `canary`, `live` |
| CODEX — `074:11`, `075:10`, `075:30`, `070:83` | `replay`, `paper`, `canary`, `live` (+`research` en 070) |

Dos consecuencias duras:

1. `backtest` y `replay` son **el mismo concepto con dos nombres**.
2. **`held_out` es literalmente inalmacenable** en `fact.*` y `exec.*`. La columna
   `held_out` del Passport **no puede llenarse ni cuando BL-23 exista** sin
   cambiar un CHECK. Y BL-23 no tiene productor: lo más cercano,
   `scripts/data/backfill_catalog_facts.py:114`, inserta con `'replay'`
   **hardcodeado** pese a declarar `"anti_survivorship_population": True`.

**DECISIÓN DEL OPERADOR**: qué nombre gana. Renombrar `backtest`→`replay` toca
evidencia ya publicada; añadir `held_out` al CHECK toca migraciones ya escritas.

---

## F-06 · Grano de identidad: `strategy_id` vs `sleeve_id` · **BLOQUEANTE**

| | |
|---|---|
| **Dueño** | ambos, sin acuerdo previo. |
| **Shape acordado** | **NO HAY.** |
| **Test que la protege** | **NINGUNO** (cubierto indirectamente por G-08). |
| **Divergencia** | **SÍ, estructural.** |

El Passport de CLAUDE hace `LEFT JOIN exec_order o ON o.strategy_id = s.strategy_id`
y `JOIN lineage_node l USING (strategy_id)`
(`.claude/specs/platform/passport-control-tower.md:79-83, 106`).

La realidad de CODEX:

| Tabla | Llave de negocio real | Evidencia |
|---|---|---|
| `exec.order_header` | `sleeve_id TEXT NOT NULL` | `074:13` |
| `exec.fill_event` | **ninguna de las dos** | `074:44-56` |
| `lineage.node` | `node_id` PK, UNIQUE `(node_type, semantic_hash)` | `076:7,22` |
| `fact.position` / `fact.pnl` | `sleeve_id` | `075:6-24, 26-49` |
| `exec.reconciliation_event` | `account_id` + `instrument_id` | `078:7-8` |
| `portfolio.kill_switch_event` | `account_id` | `077:85` |

**Ninguna de las tablas de hechos tiene `strategy_id`.** Y el desacuerdo ya está en
los datos: `config/assets/fabric_factories.yaml` declara la sleeve
`usdcop_smart_simple_v11` mientras `config/policies/smart_simple_v11.yaml` declara
`id: smart_simple_v11`. **Ya divergen.**

**Qué debe ser verdad para cerrar**: una tabla de correspondencia
`strategy_id ↔ sleeve_id` con dueño único, o un solo identificador. Elegir cuál es
**DECISIÓN DEL OPERADOR** (afecta a bundles publicados).

---

## F-07 · El tercer reloj · **GRAVE — la única divergencia que ya está perdiendo dato**

| | |
|---|---|
| **Dueño** | productor CODEX (`src/monitoring/system_health_contract.py`, DAG `control_system_health.py`, entregado en `254ce8f`). Consumidor CLAUDE (Passport). |
| **Shape acordado** | “tres relojes”. **El tercero nunca se nombró en `CONTRACTS.md`.** |
| **Test que la protege** | **NINGUNO** → `TDD-GAPS G-17`. |
| **Divergencia** | **SÍ, activa hoy.** |

- Productor: `Clock.DATA="data"` · `MODEL="model"` · **`PNL="pnl"`**
  (`src/monitoring/system_health_contract.py:52-57`).
- Consumidor: `HEALTH_CLOCKS = ['data','model','exec']`
  (`src/contracts/passport.py:68`, espejado en `passport.contract.ts:55`), y
  `composeData()` busca `health.clocks['exec']`
  (`usdcop-trading-dashboard/lib/passport/compose.ts:752-753`).

El Passport **descarta silenciosamente el reloj `pnl` que sí se publica**, y la
nota que se renderiza al usuario —*"system_health publica data y model"*
(`compose.ts:757`)— **ya es falsa**: publica tres. Es un fallo de honestidad
causado por un nombre, no por mala fe: el productor de BL-25 existe, funciona y
está commiteado.

**Qué debe ser verdad para cerrar**: el consumidor itera lo que el productor
publica, y la nota se **deriva** del dato en vez de ser un literal.

---

## F-08 · Passport ← las 9 interfaces de CODEX · **0/9 COINCIDEN**

| | |
|---|---|
| **Dueño** | CLAUDE consume; CODEX produce (BL-18/21/22/23/24/25/26/27/28). |
| **Shape acordado** | declarado unilateralmente por CLAUDE en `PENDING_INTERFACES` (`usdcop-trading-dashboard/lib/passport/compose.ts:237-295`) — **sin ACK de CODEX**. |
| **Test que la protege** | `tests/unit/test_passport_contract.py` valida la **forma del documento**, no la existencia de las fuentes. |
| **Divergencia** | **SÍ: ninguna de las 9 coincide literalmente.** |

**Corrección al enunciado del encargo**: `PENDING_INTERFACES` tiene **8 entradas,
no 9**. BL-28 no está en el array (solo como `unavailable()` inline en
`compose.ts:790`), y la 8ª entrada real es **BL-32 (SQL)**, cuyas vistas
`v_strategy_passport_live` y `mv_strategy_performance_daily` **no existen en
ninguna migración**.

| # | Interfaz | BL | ¿Existe? | Veredicto | Evidencia |
|---|---|---|---|---|---|
| 1 | `metric_event(metric_id, env, as_of, n_trades, metric_engine_version)` | 18 | parcial | **DIVERGE** | `070:134-156` — es `metric_namespace`+`metric_name`, `environment`, `event_time`; `n_trades` y `metric_engine_version` no son columnas |
| 2 | `exec_order.state='OPEN'`, `exec_fill.filled_at`, `exec_quarantine.active` | 21 | parcial | **DIVERGE** | `074:6-56` — estado es proyección `v_order_state.status` (TEXT **sin CHECK**, `074:37`); `filled_at`→`fill_time` (`074:47`); no existe `exec_quarantine` |
| 3 | `fact_pnl.timing_ratio`, `turnover`, `book.capital`, `pnl_d/m/y_pct` | 22 | parcial | **DIVERGE** | `075:31-37` — `pnl_timing` es un **monto**, no un ratio; no hay capital ni NAV ⇒ los tres `*_pct` no son derivables |
| 4 | `performance.held_out` | 23 | **NO** | **AUSENTE** | `backfill_catalog_facts.py:114` inserta `'replay'` hardcodeado |
| 5 | `lineage_graph USING (strategy_id)`, `last_vintage_revision` | 24 | parcial | **DIVERGE** | `076:7,22` — sin `strategy_id`; `revision_event` existe (`076:41-49`) pero falta la vista |
| 6 | `clocks.exec`, semáforo de retiro por estrategia | 25 | parcial | **DIVERGE** | ver F-07; el semáforo por estrategia no existe |
| 7 | `portfolio_snapshot.{gross,net}_exposure, cvar_pct, correlation_matrix, capital, rho_max` | 26 | nombre sí, contenido no | **DIVERGE** | `077:6-17` — la tabla real guarda `cutoff_time, required_sleeves, stale_signals, missing_sleeves, fallback_applied`. **Cero solapamiento** |
| 8 | `m_forward`, `m_dd`, `vol_target_pct`, `vol_forecast_pct`, `tier` | 27 | parcial | **DIVERGE** | `077:36,40` — `multiplier_forward`/`multiplier_drawdown` (mismo concepto, otro nombre); vol **nunca se persiste** (`allocator.py:35` la recibe y no la devuelve) |
| 9 | `data.replay_parity` | 28 | lógica sí, dato no | **DIVERGE** | `semantic_diff.py:45-51` existe; **nada lo persiste ni publica**; único caller = un test |
| +1 | `v_strategy_passport_live`, `mv_strategy_performance_daily` | 32 | **NO** | **AUSENTE** | no existen en `database/` |

**Tres causas raíz** (todo lo demás es cosmética encima de ellas): **(a)** el grano
(F-06), **(b)** el vocabulario de entorno (F-05), **(c)** el nombre del tercer
reloj (F-07).

**Un hallazgo en la dirección contraria, que corrige al Passport:**
`compose.ts:637` deja `tier: null // BL-27` y `compose.ts:730` afirma que
`CANARY`/`REDUCED` “no tienen productor”. **Sí lo tienen**:
`control.strategy_declaration.capital_tier CHECK IN ('ZERO','SHADOW','CANARY',
'FULL','REDUCED','EXIT_ONLY')` (`070:17-19`) + `operational_state`
(`070:20-22`), **y con `strategy_id` real** (`070:9`). Es la **única** de las
nueve donde el shape de CODEX es directamente utilizable — y es justo la que el
Passport no lee.

---

## F-09 · Motor de políticas vs factories FABRIC · **GRAVE**

| | |
|---|---|
| **Dueño** | CLAUDE: `src/policy_engine/`, `src/strategies/policies/`, `config/policies/`. CODEX: `src/orchestration/factories.py`, `airflow/dags/fabric_factories.py`, `config/assets/fabric_factories.yaml`. |
| **Shape acordado** | **NINGUNO.** |
| **Test que la protege** | **NINGUNO** → G-12/G-13. |
| **Divergencia** | **SÍ, conceptual y estructural.** |

**Los dos mundos no se conocen**: `grep "from src.contracts" src/orchestration
src/governance src/identity src/lineage src/market src/portfolio src/metrics
src/validation` ⇒ **0 resultados**.

| Concepto | CLAUDE | CODEX | Colisión |
|---|---|---|---|
| identidad | `sleeve_id` validado por `ID_PATTERN` (`policy.py:75`) | `sleeve_id` = clave del YAML (`factories.py:110`), sin validar forma | mismo nombre, dos autoridades — y ya divergen (F-06) |
| registro de versiones | `control.policy_version` vía `write_policy_version_index` (`runner.py:246`) | ninguno; el DAG dispara scripts congelados | **dos registries de facto** |
| fingerprint de decisión | `policy.py:359-376` (JSON `sort_keys`, hex16) | `fingerprints.py:44` (namespace + `canonical_json_bytes`) | **dos hashes incompatibles para “la misma” decisión** (F-02) |
| ciclo de vida | `migration.status` (SPEC_ONLY/PARITY_*/CUTOVER, `loader.py:48`) | `ResearchState × CapitalTier × OperationalState` (`declaration.py:19-41`) | **dos máquinas de estado** |
| cómo se ejecuta | `Policy.evaluate(snapshot, ctx)` en proceso | `subprocess.run(script)` por tarea (`fabric_factories.py:32`) | **la lane ACTION de FABRIC no pasa por el motor de políticas** |

**Contradicción concreta y grave**: `config/assets/fabric_factories.yaml` declara
`strat__usdcop_smart_simple_v11` como estrategia **ejecutable** por DAG, mientras
`config/policies/smart_simple_v11.yaml:70` la declara `SPEC_ONLY` — es decir, *sin
política ejecutable, fail-closed*. **Las dos vías afirman cosas opuestas sobre la
misma estrategia viva.**

**Lo que sí está bien y hay que reconocer**: el frontend **no re-evalúa**.
`policy.contract.ts` (706 líneas) exporta solo validadores, cero `evaluate`/
`buildPolicy`. Invariante 7 de `strategy-engines.md` respetada.

**Qué debe ser verdad para cerrar**: un solo registro con discriminador
`engine.type` (invariante 1 de `strategy-engines.md`). O bien FABRIC declara
explícitamente que orquesta **scripts legacy** y no estrategias, y deja de usar el
vocabulario `strategies:` en su YAML.

### REMEDIACIÓN CLAUDE · 2026-07-28 · un solo `build_policy`; la contradicción de estado NO la resuelvo yo

**Arreglado (lado CLAUDE, código mío): las dos `build_policy` son una.**
`src/policy_engine/runner.py::build_policy` **ya no tiene cuerpo propio**: es el
mismo objeto función que `src/strategies/policies/loader.py::build_policy` (se
comprueba con `is`, no con "se comporta igual", para que una divergencia futura no
pueda esconderse detrás de un adaptador fino). Con él se retiraron del runner su
`ALLOWED_POLICY_ROOTS` propio y su rama `coded_policy` duplicada.

**Por qué el loader es la SSOT** (criterio técnico, documentado en
`runner.py`): (1) honra `migration.status` — el runner ignoraba `migration` por
completo; (2) verifica el freeze `governance.policy_hash` vs contenido económico
derivado; (3) su allowlist de import es la **estrecha**
(`src.strategies.policies.`) frente a `src.strategies.` + `strategies.policies.`;
(4) corre la validación estructural completa incluyendo el AST del whitelist DSL;
(5) es el que tiene **callers reales** (los dos validadores de
`scripts/validation/` y la suite de specs) — el del runner tenía **cero** y aun
así era el exportado como API pública, o sea **la puerta laxa era la anunciada**.

Evidencia del rojo previo, que es más grave de lo que decía la auditoría: al pasar
el spec `SPEC_ONLY` a `runner.build_policy`, **no lo rechazaba por SPEC_ONLY** —
seguía adelante y construía una `DeclarativePolicy`, fallando solo de rebote por
una clave ausente, porque además leía **otra forma de spec**
(`spec["implementation"]` en vez de `spec["engine"]["implementation"]`). No eran
"dos implementaciones equivalentes": eran dos contratos distintos.
Gate: `tests/unit/test_policy_specs.py::test_spec_only_policy_cannot_be_built_by_any_gate`
(las 3 puertas exportadas rechazan SPEC_ONLY) + `..._solo_existe_un_build_policy_...`
+ `..._ninguna_puerta_importa_fuera_del_allowlist_estrecho`.

**NO resuelto — DECISIÓN DEL OPERADOR (contradicción de estado sobre una
estrategia viva):**

| Fuente | Dueño | Afirma |
|---|---|---|
| `config/assets/fabric_factories.yaml:52-63` | CODEX | `usdcop_smart_simple_v11: enabled: true`, con tareas `backtest_replay/gates/signal/verify` y `produces: action://usdcop_smart_simple_v11/strategy_output/v2` ⇒ **ejecutable, y emite señal** |
| `config/policies/smart_simple_v11.yaml:69-70` | CLAUDE | `migration.status: SPEC_ONLY` ⇒ **sin política ejecutable, fail-closed** |

Resolverla exige decidir **si esa estrategia es ejecutable**, que es una decisión
de producto/riesgo sobre la estrategia en producción, no un refactor. **No la
tomo.** Matiz honesto para quien decida: puede que no sea contradicción real sino
vocabulario compartido — FABRIC lanza por `subprocess` los **scripts congelados**
que el propio spec nombra en `migration.legacy_producer`
(`train_and_export_smart_simple.py` + DAGs `forecast_h5_*`), no el motor de
políticas. Si es eso, la salida barata es la que ya propone este documento: que
FABRIC declare que orquesta scripts legacy y deje de llamarlos `strategies:`.
Nótese que los identificadores tampoco coinciden (`usdcop_smart_simple_v11` vs
`id: smart_simple_v11`), que es F-06 otra vez.

---

## F-10 · Contratos espejo Py↔TS

| | |
|---|---|
| **Dueño** | por par (tabla). Guardián del mapa: `tests/regression/test_contract_mirrors.py` (CODEX). |
| **Shape acordado** | Py y TS **rechazan exactamente lo mismo** (canal §4). |
| **Divergencia** | **PARCIAL.** |

| Par | SSOT | Nivel real | En el mapa |
|---|---|---|---|
| `policy.py`+`rule_trace.py` ↔ `policy.contract.ts` | Python | **fixture compartida real** (`policy_contract_cases.v1.json`, 65 casos, `content_sha256` recomputado por ambos) | sí |
| `policy_version.py` ↔ `policy-version.contract.ts` | Python | **fixture compartida real** (`policy_backend_cases.v1.json`, 97 casos) | **NO** |
| `forecast_output.py` ↔ `forecast-output.contract.ts` | Python | **fixture compartida real** (`forecast_output_cases.v1.json`, 90 casos) | sí |
| `passport.py` ↔ `passport.contract.ts` | Python | **solo dos definiciones paralelas**, sin fixture | **NO** |
| `strategy_manifest.py` ↔ `strategy-manifest.contract.ts` | Python | parcial (whitelist de surfaces en runtime) | sí |
| `strategy_schema.py` ↔ `strategy.contract.ts` | Python | paralelas | sí |
| `analysis_schema.py` ↔ `weekly-analysis.contract.ts` | Python | paralelas | sí |
| `backtest_ssot.py` ↔ `backtest-ssot.contract.ts` | Python | pin por regex | sí |
| `feature_contract.py`+`pipeline_ssot.yaml` ↔ `ssot.contract.ts` | Python/YAML | hash `FEATURE_ORDER` (`contracts-check.yml`) + pin TS | sí |
| 6× SignalBridge `app/contracts/*.py` ↔ `lib/contracts/execution/*.ts` | Python | paralelas, con test de emparejamiento | sí |

**Solo en un lenguaje** (legítimo si es deliberado; hay que declararlo):
Py-only — `policy_dsl.py` (backend-only por diseño), `execution_strategies.py`,
`signal_contract.py`, `signal_adapters.py`, `replay_engine.py`, `asset_profile.py`,
`news_engine_schema.py`. TS-only — `rbac.contract.ts` (**TS es SSOT, declarado**),
`admin-console`, `catalog`, `backtest`, `experiments`, `model`, `ui`,
`production-approval`, `production-monitor`.

**Dos defectos de la frontera:**

1. **El guardián no detecta pares FALTANTES.**
   `tests/regression/test_contract_mirrors.py:62-77` solo verifica que las rutas
   **que el mapa nombra** existan. `passport` y `policy_version` son pares reales
   (uno con fixture compartida) **fuera del mapa**, y el test queda verde. La única
   excepción que sí exige presencia es `test_execution_contracts_are_paired`, y
   solo para `lib/contracts/execution/`. → **G-23**.
2. **La mitad TS nunca corre en CI** → F-11 / G-05. Las tres fixtures compartidas
   se verifican **solo del lado Python**, lo que convierte “paridad bilateral
   demostrada” en una afirmación cierta *localmente* y no garantizada *en el
   repositorio*.

---

## F-11 · CI como frontera compartida · **AUSENTE**

| | |
|---|---|
| **Dueño** | CODEX (`.github/workflows/` es su frontera por ASSIGNMENTS). |
| **Shape acordado** | ninguno explícito. |
| **Divergencia** | **SÍ: hay 12 workflows y ninguno ejecuta la gobernanza nueva.** |

Workflows reales (12): `ci`, `deploy`, `security`, `security-scan`,
`contracts-check`, `drift-check`, `dvc-validate`, `experiment`, `canary-promote`,
`a11y`, `rbac-gate`, `specs-gate`.
Nota: `.claude/specs/platform/cicd-testing.md` dice **9** y no menciona `a11y`,
`rbac-gate` ni `specs-gate` ⇒ **la spec está stale** (`last_verified: 2026-07-20`).

Qué **no** corre hoy en ningún workflow:

| Ausente | Consecuencia |
|---|---|
| `vitest` (`grep` ⇒ 0 resultados) | la mitad TS de las 3 fixtures compartidas no está guardada |
| `check_trial_ledger.py` · `report_ledger_dsr.py` | 13 checks + 51 tests de gobernanza rompibles con CI verde |
| `validate_policy_specs.py` · `check_policy_parity.py` | la prohibición de `eval` desde YAML y la paridad del strangler no se verifican |
| `tests/regression/test_trial_ledger.py` y los dos de BL-09/10/11/12 | `specs-gate.yml` corre `tests/regression/` con **allowlist de 6 ficheros** que no los incluye |

Y una omisión de disparo: **`contracts-check.yml` no dispara con
`usdcop-trading-dashboard/lib/contracts/**`** (sus paths son `src/**`,
`services/**`, `airflow/**`). Un cambio que rompa la mitad TS de un contrato
espejo **no enciende ninguna luz roja**.

**Qué debe ser verdad para cerrar**: G-04 + G-05.

---

## F-12 · Migraciones DB · **RIESGO OPERATIVO**

| | |
|---|---|
| **Dueño** | **SOLO CODEX escribe** `database/migrations/*`; CLAUDE revisa (`ASSIGNMENTS.md` §Fronteras duras). Esta frontera está bien definida y se ha respetado. |
| **Divergencia** | no de propiedad, **sí de seguridad de aplicación**. |

`scripts/ops/db_migrate.py:120` hace `sorted(MIGRATIONS_DIR.glob("*.sql"))` **sin
allowlist**. Por tanto las nueve migraciones `070`-`078`, hoy untracked y **sin
cross-review**, se aplicarían en el próximo arranque de la pila.

Esto choca frontalmente con el guardarraíl vigente del protocolo: *“BL-41 sin DDL
ni cutover hasta Vault real, roles no-super y TDD”*. **El guardarraíl es una
promesa que el sistema no respalda.** → **G-07**.

**Segundo problema**: **9 grupos de tablas creadas sin ningún consumidor** —
`exec.*` (5 objetos), `portfolio.*` (4), `fact.*` (2),
`control.artifact_identity`, `control.strategy_declaration`, `market.raw_bar`,
`forecast.*` (4), `reference.*` (6), `quality.*` (2). Ni writer ni reader fuera de
la migración. Y `src/metrics/engine.py::MetricEngine` **no tiene ni un caller, ni
en tests**. Crear esquema sin lector es deuda, no progreso — y `reference.*`
además **bloquea todo lo demás**: son FK obligatorias y nadie las puebla (G-25).

---

## F-13 · Vocabulario de fallback · **MENOR (pero es 4 vocabularios)**

| Fuente | Valores | Evidencia |
|---|---|---|
| CLAUDE — runner | `FAIL_CLOSED`, `FLAT` | `src/policy_engine/runner.py:51` |
| CLAUDE — loader | `FAIL_CLOSED`, `FLAT`, `HOLD` | `src/strategies/policies/loader.py:44` |
| CLAUDE — DSL | ninguno (raise hardcodeado) | `src/contracts/policy_dsl.py:511-515` |
| CODEX — snapshot | `FLAT`, `KEEP_POSITION_UNTIL_EXPIRY`, `EXIT_ONLY`, `USE_LAST_VALID_WITH_MAX_AGE` | `src/portfolio/snapshot.py:16-20` |

Un spec con `HOLD` **pasa la validación y revienta con `ValueError` en
`runner.py:186-187`**. Nadie usa `HOLD` hoy: bomba latente. → **G-15**.

`strategy-engines.md` invariante 9 exige `missing_input_policy`/
`stale_input_policy` declarados y **“sin default explícito no hay freeze”**. Hoy
lo declarado en el YAML **no se honra en ningún camino unificado**. → **G-14**.

---

## F-14 · Métricas: catálogo vs escritores · **GRAVE (interna de CODEX, la consume CLAUDE)**

| | |
|---|---|
| **Dueño** | CODEX (BL-18). Consumidor: el Passport de CLAUDE. |
| **Shape acordado** | “motor único de métricas”. |
| **Divergencia** | **SÍ, CODEX se contradice consigo mismo.** |

- `config/metrics/catalog.yaml` define **5** métricas: `strategy.sharpe` (:3),
  `strategy.calmar` (:13), `strategy.max_drawdown` (:23),
  `strategy.timing_ratio` (:34), `research.dsr` (:42).
- `scripts/data/backfill_catalog_facts.py:46-53,105-115` inserta en
  `control.metric_event` **sin pasar por `MetricEngine` ni por el catálogo**, con
  `metric_namespace='performance'` y 8 `metric_name`
  (`sharpe, sortino, calmar, max_drawdown_pct, total_return_pct, n_trades,
  p_value, dsr`). **Ninguno está en el catálogo**; `MetricCatalog.get()` los
  rechazaría (`src/metrics/engine.py:119-123`).
- Y el Passport pide **tres ids que no existen en ninguno de los dos conjuntos**:
  `return_pct`, `max_dd_pct`, `dsr_family`
  (`.claude/specs/platform/passport-control-tower.md:90-94`). `dsr_family` (por
  familia) contra `research.dsr` (genérica) **ni siquiera es el mismo grano**.

**El motor único tiene un bypass en el camino de producción.** → **G-16**.

---

## 2. Qué debe ser verdad para declarar la integración cerrada

```
1. CODEX commitea su lote                    → F-01 deja de ser falso-verde
2. CI corre gobernanza + vitest              → F-11 cerrada; 11 escenarios dejan de ser humo
3. Allowlist de migraciones                  → F-12 deja de contradecir al guardarraíl
4. Se acuerdan TRES vocabularios:
     env (F-05) · grano (F-06) · reloj (F-07)  → F-08 pasa de 0/9 a evaluable
5. Un hash (F-02) y un DSR (F-04)            → cierran las dos duplicaciones de SSOT
6. Un registry de estrategias (F-09)         → invariante 1 de strategy-engines
7. El mapa de espejos detecta faltantes (F-10) → la paridad deja de ser opcional
```

Ninguna frontera se declara cerrada por acuerdo verbal: cada una necesita **su
test de integración verde con la mutación demostrada** (canal §5.5).

---

## FIRMAS

- `claude-root-9c3f1e42` · 2026-07-28 · **PROPUESTO**. Auditoría adversarial de
  ambos lotes, incluido el mío: de las 14 fronteras, **5 divergencias tienen a
  CLAUDE como responsable total o parcial** (F-01 consumidor, F-02 familia policy,
  F-07 consumidor, F-08 lecturas, F-09 motor). No arranqué Docker, no commiteé, y
  las decisiones de modelado que aparecieron están marcadas DECISIÓN DEL OPERADOR
  y **sin tomar**.
- `claude-root-9c3f1e42` · 2026-07-28 · **REMEDIACIÓN F-01 / F-02 / F-09** (los tres
  defectos con CLAUDE como responsable del código). TDD con rojo demostrado antes de
  cada fix. Cero decisiones de modelado tomadas: siguen sin resolver, y marcadas,
  (i) unificar el hash policy con `src/identity/canonical.py` (re-freeze de evidencia
  publicada) y (ii) si `smart_simple_v11` es ejecutable (FABRIC dice sí, su spec dice
  `SPEC_ONLY`). **0 trials.** No commiteé, no arranqué Docker, no toqué ficheros de
  CODEX. F-01 queda ROJO deliberadamente: es la señal, no el defecto.
- `codex-root-880ff498` · _pendiente de cofirma / objeción por frontera_
