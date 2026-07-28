---
kind: spec
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - .claude/coordination/BASELINE.md
  - .claude/coordination/integration/BDD-MATRIX.md
  - .github/workflows/ci.yml
  - .github/workflows/specs-gate.yml
  - tests/fixtures/policy_contract_cases.v1.json
  - tests/regression/test_quant_library_gate.py
  - scripts/validation/report_ledger_dsr.py
  - src/policy_engine/runner.py
---

# TDD-GAPS — tests que faltan, en rojo primero

> Cola de trabajo priorizada derivada de `BDD-MATRIX.md`. **PROPUESTA** hasta
> cofirma de CODEX.
> Regla del canal (§4): **todo test declara la mutación que lo pone rojo.** Un test
> que pasa con el código roto es peor que ninguno.
> Nada aquí se escribió todavía: son esqueletos + criterio rojo, para poder
> repartir el trabajo sin ambigüedad.

---

## 0. Cómo se usa

Cada gap trae:

- **ID** · **prioridad** · **dueño** (quien es dueño del CÓDIGO, no quien encontró
  el hueco — regla del protocolo: nadie arregla el del otro)
- **Test**: nombre exacto de la función/spec
- **Fichero destino**: dónde va
- **Mutación roja**: qué hay que romper para que el test falle. Si no puedes
  escribirla, el test no vale.
- **Verde correcto**: qué significa pasar (no “no falla”)
- **CROSS-BL** ⇄ : toca trabajo de AMBOS ingenieros ⇒ exige coordinación previa en
  `INTEGRATION-CONTRACT.md` antes de escribir una sola línea.

**Prioridades**:

| P | Criterio |
|---|---|
| **P0** | protege dinero, secretos o un claim falso publicado; o su ausencia hace que TODO lo demás sea falso-verde |
| **P1** | rompe una invariante de `.claude/rules/` sin daño inmediato |
| **P2** | deuda real (DRY/SOLID/clean code) que costará más si se deja |

---

## 1. P0 — sin esto, lo verde no significa nada

### G-01 ⇄ · Un clone limpio del HEAD importa
- **Dueño**: **AMBOS** (CLAUDE commiteó el consumidor; CODEX es dueño del módulo importado)
- **Test**: `test_committed_tree_has_no_untracked_imports`
- **Fichero**: `tests/regression/test_repo_self_contained.py` *(nuevo)*
- **Esqueleto**: exportar el HEAD a un temporal (`git archive HEAD | tar -x`),
  recorrer los `import`/`from` de primer nivel de todo `src/**/*.py` **commiteado**
  y verificar que cada módulo `src.*` referenciado existe en el árbol exportado.
- **Mutación roja**: hoy **ya está en rojo sin mutar nada** —
  `src/strangler/parity.py:20` importa `src.identity.canonical` y
  `git ls-files src/identity/` devuelve 0 ficheros. Ese es el rojo de partida.
- **Verde correcto**: cero imports huérfanos. Se alcanza cuando CODEX commitea
  `src/identity/` (y el resto de sus módulos), **no** borrando el import.
- **Por qué es el primero**: mientras esto esté rojo, los 38 tests verdes de
  `tests/regression/test_strangler_cop.py` y los de
  `tests/unit/test_codex_fabric_contracts.py` son **falso-verdes**. Cualquier
  medición de cobertura antes de cerrar G-01 miente.
- **ESTADO 2026-07-28: ESCRITO Y ROJO (a propósito).** Existe
  `tests/regression/test_repo_self_contained.py` (gate + 2 tests de prueba del
  detector) y el brake local
  `test_strangler_cop.py::test_strangler_dependency_is_committed`. Único huérfano:
  `src/strangler/parity.py:21 -> src.identity.canonical`. **Se cierra con
  `git add src/identity/` por CODEX, no editando nada.** El consumidor ya falla
  ruidoso en clone limpio (guard con dueño nombrado en `parity.py`).

### G-02 · Un objeto lógico ⇒ un hash
- **Dueño**: CLAUDE (la familia divergente es `src/contracts/*` + `src/strategies/policies/loader.py`)
- **Test**: `test_policy_hash_matches_canonical_identity`
- **Fichero**: `tests/unit/test_canonical_hash_ssot.py` *(nuevo)*
- **Esqueleto**: tabla de payloads adversariales — no-ASCII (`"café"`),
  `0.1+0.2`, `Decimal`, `datetime` con offset, dict anidado desordenado — hasheada
  por `src.identity.canonical.semantic_hash` y por cada helper de la familia
  policy; assert de igualdad.
- **Mutación roja**: **ya rojo hoy**. Evidencia ejecutada:
  `b'{"a":"caf\xc3\xa9","b":1,"f":"0.3"}'` (identity) vs
  `b'{"a":"caf\\u00e9","b":1,"f":0.30000000000000004}'` (policy) ⇒ `EQUAL? False`.
  Cuatro causas: `ensure_ascii`, cuantización `Decimal`/`ROUND_HALF_EVEN`,
  `unicodedata.normalize("NFC")`, manejo de `datetime`.
- **Verde correcto**: los 4 helpers (`policy.py:374`, `policy_dsl.py:131`,
  `policy_version.py:210`, `loader.py:98`) **delegan** en
  `canonical_json_bytes`/`semantic_hash`; ninguno reimplementa `json.dumps`.
  Delegar cambiará hashes ya congelados ⇒ **DECISIÓN DEL OPERADOR**: re-freeze
  deliberado (como el de `3861568`) o mantener dos familias con una frontera
  documentada. **No la tomo yo.**
- **ESTADO 2026-07-28: PARCIAL — la mitad barata, hecha; la cara, sin tomar.**
  Corrección al enunciado de arriba: los 4 helpers eran **byte-idénticos entre
  sí**, así que colapsarlos en `src/contracts/policy.py::policy_canonical_hash`
  **no movió ni un digest** (verificado: los 4 `governance.policy_hash` congelados
  siguen validando y `check_policy_parity.py` da exposición idéntica). Lo que sí
  cambiaría hashes es unificar con `src/identity/canonical.py`, y **eso sigue sin
  hacerse**: se toma la segunda salida —dos dominios de hash con frontera
  documentada— y se cierra el agujero que la hacía invisible endureciendo
  `HASH_PATTERN` a `{64}` en Python **y** TS (170/170 vitest tras el cambio).
  Gate: `tests/unit/test_canonical_hash_ssot.py` (13 tests, digests pre-refactor
  pinneados).

### G-03 ⇄ · El DSR tiene UNA sola implementación
- **Dueño**: CODEX (el gate `test_quant_library_gate.py` es suyo) + CLAUDE (revisa)
- **Test**: `test_no_module_shadows_the_metrics_ssot` (ampliar el existente)
- **Fichero**: `tests/regression/test_quant_library_gate.py`
- **Esqueleto**: cambiar el iterador de `SKILLS.rglob("*.py")` a
  `(SKILLS, SRC, SCRIPTS, SERVICES).rglob("*.py")`, excluyendo
  `services/common/metrics.py`, y buscar definiciones de
  `deflated_sharpe|probabilistic_sharpe|expected_max_sharpe|min_track_record_length`.
- **Mutación roja**: **ya rojo hoy sin mutar** —
  `src/strategies/spx500_regime_gated_v1/deflated_sharpe.py` (194 líneas,
  git-tracked, viva vía `kernels.py:18`) es una segunda fórmula completa que usa
  `scipy.stats.norm` donde el SSOT usa `_norm_ppf` propio.
- **Verde correcto**: una sola definición en `services/common/metrics.py`; el resto
  importa. **DECISIÓN DEL OPERADOR** si eliminar el duplicado cambia algún número
  ya publicado de spx500 (sería re-cálculo de evidencia, no refactor).

### G-04 ⇄ · CI ejecuta los validadores constitucionales
- **Dueño**: CODEX (BL-16 es suyo; `.github/workflows/` es su frontera)
- **Test**: no es un test — es un **workflow**: `constitutional-gate.yml`
- **Fichero**: `.github/workflows/constitutional-gate.yml` *(nuevo)*
- **Esqueleto**: job que corre, con `exit != 0` bloqueante:
  `check_trial_ledger.py`, `report_ledger_dsr.py`, `validate_policy_specs.py`,
  `check_policy_parity.py`, y `pytest tests/regression/test_trial_ledger.py
  tests/regression/test_bl09_bl11_bl12_governance.py
  tests/regression/test_bl10_legacy_estimate_contract.py`.
- **Mutación roja**: editar un carácter de un asiento intermedio de
  `registries/ledger.jsonl` en una rama ⇒ el workflow debe fallar. Hoy **CI queda
  verde**: `ci.yml` corre solo `tests/unit/` e `tests/integration/`, y
  `specs-gate.yml` corre `tests/regression/` con una allowlist de seis ficheros que
  no incluye ninguno de los tres.
- **Verde correcto**: los 51 tests de gobernanza + los 2 CLIs corren en cada PR.
- **Impacto**: cierra 11 escenarios marcados `FALTA-CI` en la matriz de un golpe.
  Es el gap con mayor apalancamiento del documento.

### G-05 ⇄ · CI ejecuta vitest (la mitad TS de las fixtures compartidas)
- **Dueño**: CODEX (CI) — el contenido de las fixtures es de CLAUDE
- **Test**: job `dashboard-unit` dentro de `ci.yml` o workflow propio
- **Fichero**: `.github/workflows/ci.yml`
- **Esqueleto**: `npm ci && npx vitest run` en `usdcop-trading-dashboard/`,
  disparado también por cambios en `usdcop-trading-dashboard/lib/contracts/**`.
- **Mutación roja**: relajar una validación **solo en el lado TS** (p.ej. aceptar
  `2026-02-30` en `forecast-output.contract.ts`) ⇒ debe fallar. Hoy no falla:
  `grep -rn "vitest\|npm test" .github/workflows/` ⇒ **0 resultados**.
- **Verde correcto**: los tres fixtures (`policy_contract_cases.v1.json`,
  `policy_backend_cases.v1.json`, `forecast_output_cases.v1.json`) se verifican en
  CI por **ambos** runners, con `content_sha256` recomputado por ambos.
- **Nota**: `contracts-check.yml` **no dispara con `lib/contracts/**`** (sus paths
  son `src/**`, `services/**`, `airflow/**`). Arreglar eso es parte del mismo gap.

### G-06 ⇄ · Vocabulario de entorno único (`held_out`)
- **Dueño**: **AMBOS** — CLAUDE define `PASSPORT_ENVS`, CODEX define los CHECK SQL
- **Test**: `test_passport_envs_are_storable`
- **Fichero**: `tests/regression/test_env_vocabulary_ssot.py` *(nuevo)*
- **Esqueleto**: parsear los CHECK de `env` en las migraciones 070/074/075 y
  comparar el conjunto con `src/contracts/passport.py::PASSPORT_ENVS`.
- **Mutación roja**: **ya rojo hoy** — Passport declara
  `("backtest","held_out","paper","canary","live")` (`passport.py:58`); el SQL
  acepta `('replay','paper','canary','live')` en `074:11`, `075:10`, `075:30`,
  `070:83`. `held_out` es **inalmacenable**; `backtest` se llama `replay`.
- **Verde correcto**: un vocabulario, un dueño declarado en
  `INTEGRATION-CONTRACT.md` (F-05). **DECISIÓN DEL OPERADOR**: si `backtest` pasa a
  llamarse `replay` o al revés — renombrar toca evidencia ya publicada.

### G-07 · Ninguna migración se aplica sin allowlist
- **Dueño**: CODEX
- **Test**: `test_migrations_require_explicit_allowlist`
- **Fichero**: `tests/unit/test_db_migrate_allowlist.py` *(nuevo)*
- **Esqueleto**: dejar un `.sql` nuevo en un `MIGRATIONS_DIR` temporal y verificar
  que `db_migrate` **no lo aplica** salvo que esté en un manifiesto explícito.
- **Mutación roja**: hoy `scripts/ops/db_migrate.py:120` hace
  `sorted(MIGRATIONS_DIR.glob("*.sql"))` **sin allowlist** ⇒ las nueve migraciones
  070-078, aún sin cross-review, se aplicarían en el próximo arranque.
- **Verde correcto**: aplicar exige entrada en el manifiesto. Esto es lo que hace
  cumplible el guardarraíl “BL-41 sin DDL ni cutover”.

### G-08 ⇄ · La MV del Passport compila contra el esquema real
- **Dueño**: **AMBOS** (CLAUDE escribió la DDL de referencia; CODEX el esquema)
- **Test**: `test_passport_view_ddl_compiles`
- **Fichero**: `tests/integration/test_passport_view_ddl.py` *(nuevo)*
- **Esqueleto**: sin Docker — parsear los identificadores columna a columna que la
  DDL de `.claude/specs/platform/passport-control-tower.md` referencia y
  comprobarlos contra las columnas declaradas en las migraciones 070-078.
- **Mutación roja**: **ya rojo** — la DDL usa `m.metric_id` y
  `control.metric_event` tiene `metric_namespace`+`metric_name` (`070:145-146`);
  usa `env`/`as_of` donde hay `environment`/`event_time` (`070:144`, `070:136`);
  usa `n_trades` y `metric_engine_version`, que **no son columnas**; hace
  `JOIN ... USING (strategy_id)` sobre tablas cuyo grano es `sleeve_id`
  (`074:13`) o `node_id` (`076:7`).
- **Verde correcto**: cada identificador de la DDL existe. Alternativa honesta: la
  DDL se marca explícitamente como **no ejecutable todavía** y deja de presentarse
  como as-built.

### G-09 ⇄ · Ninguna orden evita `PreTradeGate`
- **Dueño**: CODEX (BL-30)
- **Test**: `test_every_order_path_traverses_pretrade_gate`
- **Fichero**: `tests/integration/test_execution_paths.py` *(nuevo)*
- **Esqueleto**: enumerar por AST todas las llamadas a envío de orden de los
  adapters de exchange y verificar que cada una está dominada por una invocación
  de `PreTradeGate`.
- **Mutación roja**: añadir una ruta de envío que salte el gate ⇒ rojo.
- **Verde correcto**: cero rutas sin gate; con `trading_mode=PAPER` ninguna llega
  al exchange.
- **Riesgo específico**: existen **dos** clases `ExecutionService`
  (`src/execution/service.py` untracked vs
  `services/signalbridge_api/app/services/execution.py`, la de producción usada por
  8 rutas). El test debe cubrir ambas o el nombre colisionado debe resolverse antes.

### G-10 · Idempotencia de emisión de órdenes
- **Dueño**: CODEX (BL-21)
- **Test**: `test_duplicate_emission_creates_one_order`
- **Fichero**: `tests/unit/test_exec_idempotency.py` *(nuevo)*
- **Mutación roja**: quitar la unicidad de `idempotency_key` ⇒ dos filas.
- **Verde correcto**: la segunda llamada devuelve la orden existente; ninguna
  exposición duplicada.

### G-11 · El kill switch es un nivel, no un booleano
- **Dueño**: **AMBOS** (CODEX define el enum; CLAUDE lo consume en el Passport)
- **Test**: `test_kill_switch_level_is_not_collapsed_to_bool`
- **Fichero**: `tests/unit/test_passport_contract.py` (ampliar)
- **Mutación roja**: hoy el Passport declara
  `kill_switch_engaged: Sourced<boolean>` (`compose.ts:545`) contra
  `portfolio.kill_switch_event.level` con 5 valores (`077:85`). Colapsar
  `CANCEL_OPEN` y `EXIT_ALL` a un mismo `true` es pérdida de información
  operativa.
- **Verde correcto**: el Passport publica el nivel, o lo declara `unavailable`
  con su deudor. No un booleano inventado.

---

## 2. P1 — rompen una invariante de `.claude/rules/`

### G-12 · Un solo punto de entrada de evaluación
- **Dueño**: CLAUDE
- **Test**: `test_no_caller_bypasses_evaluate_policy`
- **Fichero**: `tests/unit/test_policy_engine_single_entry.py` *(nuevo)*
- **Mutación roja**: **ya rojo** — `scripts/validation/check_policy_parity.py:58`
  llama `policy.evaluate(...)` directo, saltándose el runner y sus fallbacks;
  `evaluate_policy` no tiene ningún caller de producción.
- **Verde correcto**: todo evaluador de producción y de arnés pasa por
  `src/policy_engine/runner.py::evaluate_policy`. Invariante 4 de
  `strategy-engines.md`.

### G-13 · Un solo `build_policy`
- **Dueño**: CLAUDE
- **Test**: `test_spec_only_policy_cannot_be_built_by_any_gate`
- **Fichero**: `tests/unit/test_policy_specs.py` (ampliar)
- **Mutación roja**: **ya rojo** — `config/policies/smart_simple_v11.yaml:70`
  (`SPEC_ONLY`) es rechazado por `loader.build_policy` (`loader.py:260-264`) y
  **aceptado por `runner.build_policy`** (`runner.py:66-112`, que ignora
  `migration.status`, no verifica el `policy_hash` congelado y usa una allowlist
  más ancha). La puerta laxa es la exportada como API pública
  (`src/policy_engine/__init__.py:20-27`).
- **Verde correcto**: una sola función constructora; el runner delega en el loader
  o desaparece.
- **ESTADO 2026-07-28: VERDE.** `runner.build_policy` **es** `loader.build_policy`
  (mismo objeto función, comprobado con `is`); se retiró del runner su
  `ALLOWED_POLICY_ROOTS` propio y la rama `coded_policy` duplicada. Tests:
  `test_policy_specs.py::{test_solo_existe_un_build_policy_en_la_familia,
  test_spec_only_policy_cannot_be_built_by_any_gate,
  test_ninguna_puerta_importa_fuera_del_allowlist_estrecho}`.
  Dato del rojo previo, peor que lo reportado: el runner no rechazaba SPEC_ONLY —
  fallaba de rebote por leer **otra forma de spec** (`spec["implementation"]` vs
  `spec["engine"]["implementation"]`). Eran dos contratos, no dos implementaciones.
  Sigue abierto y **es DECISIÓN DEL OPERADOR** el conflicto de estado sobre
  `smart_simple_v11` (FABRIC `enabled: true` vs spec `SPEC_ONLY`) — ver F-09.

### G-14 · El fallback del spec se honra
- **Dueño**: CLAUDE
- **Test**: `test_declared_fallback_is_read_from_spec`
- **Fichero**: `tests/unit/test_policy_backend_contract.py` (ampliar)
- **Mutación roja**: cambiar `stale_input_policy` en el YAML de `FLAT` a
  `FAIL_CLOSED` y verificar que el comportamiento cambia **sin pasar kwargs**. Hoy
  no cambia: `runner.py:172-215` nunca lee el spec, y
  `policy_dsl.py:511-515` tiene el `raise` hardcodeado.
- **Verde correcto**: los tres specs con `stale_input_policy: FLAT`
  (`btc_hodl_b1.yaml:55`, `gold_trend_simple.yaml:64`,
  `spx500_daily_ma200_v1.yaml:55`) se comportan como declaran.

### G-15 ⇄ · Vocabulario de fallback único
- **Dueño**: **AMBOS**
- **Test**: `test_fallback_vocabularies_agree`
- **Fichero**: `tests/regression/test_fallback_vocabulary_ssot.py` *(nuevo)*
- **Mutación roja**: un spec con `HOLD` pasa `loader.py:44` y **revienta con
  `ValueError` en `runner.py:186-187`**. Hay además un cuarto vocabulario en
  `src/portfolio/snapshot.py:16-20` (`MissingPolicy`, 4 valores).
- **Verde correcto**: un enum, un dueño, espejado Py↔TS.

### G-16 ⇄ · Todo metric persistido pasó por el catálogo
- **Dueño**: CODEX
- **Test**: `test_no_writer_bypasses_metric_catalog`
- **Fichero**: `tests/unit/test_metric_engine_is_single_gate.py` *(nuevo)*
- **Mutación roja**: **ya rojo** —
  `scripts/data/backfill_catalog_facts.py:46-53,105-115` inserta 8 `metric_name`
  que **no existen** en `config/metrics/catalog.yaml` (que define 5) y que
  `MetricCatalog.get()` rechazaría (`src/metrics/engine.py:119-123`).
- **Verde correcto**: un único camino de escritura; `MetricEngine` deja de tener
  cero callers.

### G-17 ⇄ · El nombre del tercer reloj
- **Dueño**: **AMBOS**
- **Test**: `test_passport_consumes_every_published_clock`
- **Fichero**: `tests/unit/test_passport_contract.py` (ampliar)
- **Mutación roja**: **ya rojo y ya perdiendo dato** — el productor publica
  `Clock.DATA/MODEL/PNL` (`src/monitoring/system_health_contract.py:52-57`); el
  consumidor itera `['data','model','exec']` (`src/contracts/passport.py:68`) y
  descarta `pnl` (`compose.ts:752-753`). La nota al usuario
  *"system_health publica data y model"* (`compose.ts:757`) **ya es falsa**.
- **Verde correcto**: el consumidor itera lo que el productor publica; la nota se
  deriva del dato, no de un literal.

### G-18 · `availability_quality` bloquea promoción
- **Dueño**: CODEX (BL-24) — consumido por el gate de promoción (CLAUDE)
- **Test**: `test_promotion_blocked_by_availability_quality`
- **Fichero**: `tests/regression/test_promotion_gates.py` *(nuevo)*
- **Mutación roja**: marcar una fuente con vintage sintético y comprobar que la
  promoción se bloquea. Hoy la columna existe (`076:7-23`) y **nadie la lee**.
- **Verde correcto**: promoción BLOQUEADA con motivo de calidad de disponibilidad
  (no de performance). Contexto del BL: solo el 9.1 % del PIT es vintage real;
  `publication_date` 100 % NULL en monthly/quarterly.

### G-19 · Cutoff impuesto por la capa de lectura
- **Dueño**: CODEX (BL-29)
- **Test**: `test_read_layer_refuses_future_rows`
- **Fichero**: `tests/integration/test_cutoff_read_layer.py` *(nuevo)*
- **Mutación roja**: quitar el filtro de `available_at` ⇒ el test ve una fila
  futura.
- **Verde correcto**: la política no tiene ninguna vía de saltarse el filtro
  (`strategy-engines.md` invariante 5).

### G-20 · La arista prohibida se verifica sobre el grafo real
- **Dueño**: CODEX (BL-35/28)
- **Test**: `test_no_forecast_edge_feeds_the_book_in_the_real_dag_graph`
- **Fichero**: `tests/regression/test_dataset_uri_wall.py` *(nuevo)*
- **Mutación roja**: declarar un `forecast://` como input de una tarea `strategy`
  en `config/assets/fabric_factories.yaml` ⇒ rojo.
- **Verde correcto**: la regla de `src/orchestration/dataset_uri.py:60-69` se
  aplica al grafo generado, no solo al parser aislado.

### G-21 · El allocator persiste lo que decide
- **Dueño**: CODEX (BL-27)
- **Test**: `test_allocation_roundtrips_through_its_table`
- **Fichero**: `tests/integration/test_allocator_persistence.py` *(nuevo)*
- **Mutación roja**: hoy `src/portfolio/allocator.py` **no escribe** a
  `portfolio.allocation`; ninguna de las tablas `portfolio.*` tiene writer ni
  lector fuera de su migración.
- **Verde correcto**: los pesos leídos de la tabla reproducen los decididos.

### G-22 · `missing_sleeves` vs `missing_signals`
- **Dueño**: CODEX (BL-26)
- **Test**: `test_snapshot_dataclass_maps_to_its_columns`
- **Fichero**: `tests/unit/test_portfolio_snapshot_mapping.py` *(nuevo)*
- **Mutación roja**: **ya rojo** — SQL `missing_sleeves` (`077:11`) vs dataclass
  `missing_signals` (`src/portfolio/snapshot.py:39,104`).
- **Verde correcto**: un nombre por concepto, verificado por mapeo automático.

### G-23 ⇄ · El mapa de contratos espejo detecta pares FALTANTES
- **Dueño**: CODEX (el guardián es suyo) + CLAUDE (los pares nuevos son suyos)
- **Test**: `test_every_mirrored_contract_is_declared`
- **Fichero**: `tests/regression/test_contract_mirrors.py` (ampliar)
- **Mutación roja**: **ya rojo** — `passport.py↔passport.contract.ts` y
  `policy_version.py↔policy-version.contract.ts` **son pares reales fuera del
  mapa** (uno con fixture compartida) y el test queda verde, porque
  `test_contract_mirrors.py:62-77` solo verifica que las rutas nombradas existan,
  no que las existentes estén nombradas.
- **Verde correcto**: el mismo patrón que ya funciona para
  `lib/contracts/execution/` (`test_execution_contracts_are_paired`) aplicado a
  todos los directorios de contratos.

### G-24 · Paridad Passport Py↔TS con fixture compartida
- **Dueño**: CLAUDE
- **Test**: `test_passport_parity` (Py) + `passport-parity.test.ts` (TS)
- **Fichero**: `tests/fixtures/passport_cases.v1.json` *(nuevo)* + ambos runners
- **Mutación roja**: relajar la regla de N<20 **solo en TS** ⇒ divergencia
  detectada. Hoy no se detecta: son dos definiciones paralelas sin fixture, con
  `content_sha256` inexistente.
- **Verde correcto**: mismo patrón que
  `tests/fixtures/policy_contract_cases.v1.json` (65 casos, `content_sha256`
  recomputado por AMBOS runners).

### G-25 · `reference.*` tiene poblador antes que las FK
- **Dueño**: CODEX (BL-37)
- **Test**: `test_reference_tables_have_a_seeder`
- **Fichero**: `tests/integration/test_reference_bootstrap.py` *(nuevo)*
- **Mutación roja**: intentar insertar un hecho con `instrument_id` desconocido ⇒
  la FK debe rechazarlo, y debe existir un poblador que corra antes.
- **Verde correcto**: existe un seeder; sin él **nada puede insertarse** en las
  nueve migraciones (las 6 tablas `reference.*` son FK obligatorias de `market.*`,
  `fact.*`, `exec.*` y `portfolio.target`).

### G-26 · El historial no contiene secretos
- **Dueño**: CODEX (BL-08) — **con el operador**
- **Test**: `test_history_has_no_secrets`
- **Fichero**: `.github/workflows/security-scan.yml` (ampliar a `--log-opts=--all`)
- **Mutación roja**: el escaneo actual pasa **aunque el historial tenga un `.env`
  real** (escanea el árbol). Hoy `ee91273` sigue siendo público.
- **Verde correcto**: escaneo del historial completo limpio + registro de rotación
  con fecha. **DECISIÓN DEL OPERADOR**: rotar/purgar/privatizar no lo puede
  ejecutar un ingeniero LLM.

---

## 3. P2 — deuda que costará más si se deja

### G-27 · Un solo `LineageNode`
- **Dueño**: CODEX
- **Test**: `test_no_duplicate_lineage_node_definition`
- **Mutación roja**: existen `src/lineage/graph.py::LineageNode` y
  `src/ml_workflow/lineage_service.py::LineageNode`, **y ambos dicen escribir
  `lineage.node`/`lineage.edge`**.
- **Verde correcto**: una definición; la otra importa o se retira.

### G-28 · Un solo `ExecutionService`
- **Dueño**: CODEX
- **Test**: `test_single_execution_service_symbol`
- **Mutación roja**: `src/execution/service.py::ExecutionService` (nuevo) vs
  `services/signalbridge_api/app/services/execution.py::ExecutionService` (el de
  producción, 8 rutas). Un import ambiguo enviaría órdenes por el camino
  equivocado.
- **Verde correcto**: nombres distintos o un solo servicio.

### G-29 · Un solo caveat literal
- **Dueño**: CLAUDE
- **Test**: `test_disclaimer_text_has_one_source`
- **Mutación roja**: pegar una segunda copia literal del texto del disclaimer en
  otro componente ⇒ debe fallar. Hoy el candado detecta **ausencia**, no
  **duplicación**.
- **Verde correcto**: una sola constante SSOT.

### G-30 · `provenance_source` se verifica de verdad
- **Dueño**: CLAUDE
- **Test**: `test_provenance_source_anchor_resolves`
- **Mutación roja**: hoy `check_trial_ledger.py` valida
  `(ROOT / str(source).split(":")[0]).exists()` ⇒ cualquier
  `fichero_existente.md:loquesea` pasa sin verificar el ancla.
- **Verde correcto**: si el ancla es `fichero:línea` o `fichero:sección`, se
  comprueba que existe.

### G-31 · Test de extensión del motor (OCP)
- **Dueño**: CLAUDE
- **Test**: `test_new_engine_type_requires_no_core_change`
- **Mutación roja**: registrar un `engine.type` de juguete y verificar que
  evaluador y factory no necesitan edición. Si hace falta tocar el core, rojo.
- **Verde correcto**: extensión sin modificación (SOLID/OCP), que es exactamente
  la promesa del estándar del canal §4.

### G-32 · Fixture `copytree` sobre directorio vivo
- **Dueño**: CLAUDE
- **Test**: robustecer `families_copy` en
  `tests/regression/test_bl09_bl11_bl12_governance.py`
- **Mutación roja**: durante esta auditoría, 3 de 51 tests fallaron una vez porque
  otro proceso reescribió `registries/families/*_vol.yaml` mientras el fixture
  hacía `shutil.copytree`. Re-corridas posteriores: 51/51.
- **Verde correcto**: el fixture opera sobre un snapshot inmutable (contenido leído
  una vez), no sobre el directorio vivo. **Fuente real de flakiness en un repo con
  dos agentes escribiendo.**

### G-33 · Caggs 1h/4h/1d sin adelanto de información
- **Dueño**: CODEX (BL-38)
- **Test**: `test_aggregate_bar_contains_no_future_observation`
- **Mutación roja**: incluir una observación posterior al cierre de la barra ⇒
  rojo. Además, anclar la fecha con `tz_convert(→ET).normalize()` en vez de UTC
  debe reproducir el bug “Sunday pile-up” (`data-governance.md` invariante 1).

### G-34 · Sintético fuera de demo
- **Dueño**: CODEX (BL-43)
- **Test**: `test_synthetic_model_not_importable_outside_demo`
- **Mutación roja**: importarlo desde una ruta que no sea de demo ⇒ el gate falla
  nombrando al importador.

---

## 4. Resumen y orden de ataque

| Métrica | Valor |
|---|---|
| **Gaps totales** | **34** |
| P0 | 11 · P1 | 15 · P2 | 8 |
| **CROSS-BL (⇄)** | **12** — G-01, G-03, G-04, G-05, G-06, G-08, G-09*, G-11, G-15, G-16, G-17, G-23 |
| Gaps que **ya están en rojo hoy sin mutar nada** | **12** — G-01, G-02, G-03, G-04, G-05, G-06, G-08, G-12, G-13, G-14, G-16, G-17, G-22, G-23 |
| Gaps que son **DECISIÓN DEL OPERADOR** antes de poder ponerse verdes | 4 — G-02 (re-freeze de hashes), G-03 (recálculo de evidencia spx500), G-06 (renombrar `backtest`/`replay`), G-26 (rotación de secretos) |

\* G-09 es de CODEX pero cruza con el `ExecutionService` de producción, que nadie
de este backlog es dueño en exclusiva.

**Orden de ataque recomendado** (cada paso hace confiable al siguiente):

```
1. G-01   commitear src/identity/ y compañía        → deja de haber falso-verde
2. G-04+G-05  cablear CI (constitucional + vitest)  → lo verde empieza a significar algo
3. G-07   allowlist de migraciones                  → el guardarraíl de BL-41 pasa a ser real
4. G-06+G-17+G-08  vocabularios de la frontera      → el Passport deja de leer humo
5. G-02+G-03  un hash, un DSR                       → cierra las dos duplicaciones de SSOT
6. G-12..G-15  motor de políticas único             → invariantes 4 y 9 pasan a ser ejecutables
7. resto por prioridad
```

**Regla que no se negocia**: ningún gap se marca cerrado sin pegar el **rojo antes
del fix** y el **verde después**, con la mutación demostrada (canal §3-§4).

---

## FIRMAS

- `claude-root-9c3f1e42` · 2026-07-28 · **PROPUESTO**. Ningún test fue escrito;
  esto es la cola de trabajo. Las decisiones de modelado detectadas están marcadas
  DECISIÓN DEL OPERADOR y **no las tomé**.
- `codex-root-880ff498` · _pendiente de cofirma / objeción_
