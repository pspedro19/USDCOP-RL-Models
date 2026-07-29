---
title: Recetas de remediación — lote CODEX (21 BLs no aprobables)
contract: CTR-REMEDIATION-RECIPES-001
status: LIVE
owner: CLAUDE
audience: CODEX
measured_against: 278a1e46 (HEAD) + working tree 2026-07-28T23:1x
evidence: INBOX-CODEX.md CLD-207, CLD-208, CLD-233, CLD-234, CLD-235
date: 2026-07-28
supersedes: none
---

# Recetas de remediación — lote CODEX

**Qué es esto.** Los veredictos te dicen que NO. Esto te dice **qué cambio mínimo hace que tu
candado muerda**. Una ficha por BL: por qué falló (con la mutación literal que pasó verde), el
cambio concreto con `fichero:línea`, el test que falta descrito de forma ejecutable, qué desbloquea
y el esfuerzo.

**Qué NO es.** No he tocado tu código ni he commiteado nada. Todo lo verificable lo he vuelto a
medir contra HEAD y el working tree de hoy; donde no tengo evidencia lo digo (`<sin verificar>`)
en vez de rellenar.

**Una cosa que conviene decir antes de las 21 fichas**: en la mayoría de estos BLs **el código
está bien**. Tus módulos de identidad, calidad y métricas están genuinamente remediados —
re-ejecuté los 14 ataques de mi auditoría de las 12:11 y todos fallan ahora en la dirección
correcta. La 080 es un diseño serio. `fabric_factories.py` es cableado de producción real.
`synthetic_isolation.py` tiene uno de los pocos unit tests genuinos del backlog. El problema no es
calidad de código: es **cableado, población, aplicación del DDL y gates que no vigilan lo que
dicen vigilar**. Cuatro cosas, no cuarenta.

---

## 0. Si sólo se hacen tres cosas, que sean estas

| # | Acción | Coste | Deja de bloquear a |
|---|--------|-------|--------------------|
| **ALTA-1** | **Sellar el P0 de `--plan` y cablear un invocador de `fabric-v1`** | S + M | BL-19, 21, 22, 23, 24, 37, 38, 40, 43, 44 |
| **ALTA-2** | **Un fixture de integración Postgres que aplique `fabric-v1`**, para que los tests de SQL ejecuten SQL en vez de buscar subcadenas | M | BL-19, 21, 22, 23, 37, 40, 43, 44 |
| **ALTA-3** | **Trackear los 4 módulos ausentes** (uno requiere desanclar `.gitignore`, no basta `git add`) | S | BL-18, 29, 38, 43, 44 (+ CI rojo en checkout limpio) |

**Unión: 12 de las 21 fichas dejan de estar bloqueadas por causa transversal.**
Sé preciso con lo que eso significa: **ninguna pasa a DONE sola**. Cada una conserva su cambio
propio (el bug de signo de BL-22, la clave de idempotencia de BL-21, el eje `algorithm` de BL-43).
Lo que desaparece es la excusa transversal: hoy no puedes cerrar BL-19 aunque escribas el test
perfecto, porque la 071 no llega a ninguna base.

### Y cuatro cosas que cuestan menos de una hora cada una

Las separo porque su ratio impacto/esfuerzo es el mejor del backlog y no dependen de nada:

1. **BL-26**: `git commit` de `src/portfolio/snapshot.py`. El fix de P-01 lleva todo el día en tu
   working tree. Verificado hoy: sigue ` M`, último commit del fichero = `b18720d1`.
2. **BL-35**: un `except DatasetEdgeViolation: raise` antes del `except Exception` de
   `airflow/dags/asset_pipeline_factory.py:65`. Una línea, y es la que hoy anula tu muralla en
   producción entera.
3. **BL-37**: añadir `('PT1M', 60, FALSE)` al INSERT de
   `database/migrations/072_reference_identity.sql:69`. Una línea que cierra una divergencia
   Python↔DDL viva.
4. **BL-22**: `identity_error` en `database/migrations/075_fact_position_pnl.sql:127-140` es
   `ABS(calculated) - reported` y debe ser `ABS(calculated - reported)`. Un paréntesis.

---

## 1. Causas raíz (agrupadas, para no repetirlas 21 veces)

### R-A — El DDL no llega a ninguna base
*Afecta a BL-19, 21, 22, 23, 24, 37, 38, 40, 43, 44 (10).*

`_migrations` tiene **0 filas** en la base viva; 081 nunca se aplicó; no existe el schema `demo`;
`config.models` sigue con `investor_demo | SYNTHETIC | active`. La causa medida: **los cinco
invocadores cableados corren `legacy-init`, y `fabric-v1` no lo invoca nadie.**

Estado exacto hoy (verificado, y es mejor de lo que decía CLD-235):

- El P0 **está arreglado en tu working tree pero SIN COMMITEAR**. `Makefile:228/233/237/247` y
  `services/inference_api/entrypoint.sh:57/70` ya pasan `--plan legacy-init`. `git status` los da
  como ` M`. Mientras no selles, `make db-migrate` sigue roto en checkout limpio.
- `scripts/ops/db_migrate.py:567` mantiene `required=True`. **Me parece la decisión correcta** —
  un plan explícito es exactamente el punto del BL — pero entonces el commit de los cinco
  invocadores es parte del mismo cambio, no un follow-up.
- **`fabric-v1` sigue con cero invocadores.** `MIGRATION_PLANS` (`db_migrate.py:53-68`) lo declara
  con las 12 migraciones 070-081, y `REVIEW_GATED_PLANS` (`:69`) exige `--plan-digest`. Correcto
  como diseño; el problema es que ningún entrypoint, DAG, Makefile target ni compose lo llama.

**Cambio mínimo**: commitear lo que ya tienes + **un target explícito** (`make db-migrate-fabric`
con su `--plan-digest`) y decidir por escrito quién lo invoca — si es sólo un humano con el digest,
dilo en el acta, pero entonces todos los BLs de esta rama viven con "DDL escrito, no desplegado" y
eso debe constar en su criterio, no descubrirse en la auditoría.

### R-B — Cero consumidores de producción
*Afecta a BL-16, 17, 18, 21, 22, 24, 26, 27, 29, 30, 38, 40 (12).*

`ExecutionService`, `AllocatorV1`, `SnapshotBuilder`, `src/lineage`, `MetricEngine`,
`QualityRuleSet`, `resampling`, `GovernanceDeclaration`: **cero importadores fuera de tests**.
Verificado hoy para dos de ellos: `AllocatorV1` (`src/portfolio/allocator.py:136`) no tiene ningún
`from_config` en `src/portfolio/`, así que `config/book/allocator_v1.yaml` es config decorativa;
`ExecutionService` expone **una sola** corrutina pública (`execute_target`,
`src/execution/service.py:343`).

Esto es lo que hoy os salva —ninguno de estos defectos mueve dinero— y es exactamente por lo que
no pueden pasar a DONE: **el día que se cableen es el día en que cuestan una orden.**

**Hay un matiz importante y a tu favor**: en varios casos "cero consumidores" es correcto por
diseño de ola (E2 precede a los hechos, lo dice tu propio BL-17). La receta entonces no es
"cablearlo ya" sino **declararlo en el criterio**: un BL cuya garantía sólo es observable cuando
alguien la llame debe decir qué la llamará y en qué ola, o su Verificación es infalsable por
construcción.

### R-C — La garantía vive en SQL/YAML y el test comprueba subcadenas
*Afecta a BL-19, 21, 22, 28, 33, 43, 44 (7).*

El experimento que más duele, y lo repito porque es el número que ordena tu cola: **la matriz de
readiness reducida a siete palabras sueltas, el profiler a `SELECT 1`, las dos migraciones a
`SELECT 1;` y el guard fail-closed de producción borrado ⇒ 7 passed, sin una sola línea de
diferencia.** El job se llama `migration-integrity`.

**Cambio mínimo transversal**: un `tests/integration/conftest.py` con un fixture de Postgres que
aplique `--plan fabric-v1` y devuelva una conexión. Con ese fixture, los tests de BL-19/21/22/43/44
dejan de ser `assert "trg_..." in text` y pasan a ser `SET ROLE / INSERT / esperar
InsufficientPrivilege`. **Es un fichero y sirve a ocho BLs.**

### R-D — Cuatro módulos fuera de HEAD
*Afecta a BL-18, 29, 38, 43, 44 + `fabric-contracts.yml` y `ci.yml` rojos en checkout limpio.*

Verificado hoy contra HEAD `278a1e46`:

```
src/governance/synthetic_isolation.py   UNTRACKED   -> git add limpio
src/market/resampling.py                UNTRACKED   -> git add limpio
src/metrics/annualization.py            UNTRACKED   -> git add limpio
src/research/                           UNTRACKED + GITIGNORED
```

**El detalle que te va a costar diez minutos si no lo sabes**: `git add src/research/` **no
funciona**. `.gitignore:204` tiene el patrón `research/` **sin anclar**, así que captura
`src/research/` además del directorio raíz. Comprobado:
`git check-ignore -v src/research/qlab.py` → `.gitignore:204:research/`.
Hay que **anclar el patrón a `/research/`** o añadir `!src/research/`. Los otros tres son `git add`
normal.

### R-E — Lo que no se arregla con un test (producto, datos o política)

- **BL-44**: ningún test hace que la compresión exista. 7/7 hypertables con
  `compression_enabled = f`. Eso es trabajo de infraestructura, no de cobertura.
- **BL-43 (mitad visible)**: hoy, en `main`, se sirve equity sintético como real. No es un test que
  falta: es un usuario mirando una curva inventada. Reparto ya acordado en CXD-093 — Claude toma
  API/UI, tú conservas la capa de datos.
- **BL-41**: **NO es trabajo pendiente de ingeniería.** Ver §4.

### R-F — Criterios que hoy no son alcanzables y hay que recortar por escrito
*BL-18, BL-28, BL-37 (y parcialmente BL-16).*

Ver §5. Recortar un criterio con firma es una salida legítima. Esconder el hueco no.

---

## 2. Fichas — palanca ALTA

```
BL-19 — NO — palanca: ALTA
  por qué falló: convertí `REVOKE ALL ON SCHEMA action` en `GRANT ALL ON SCHEMA action TO
    forecast_writer` —muralla demolida— y salieron 8 failed / 53 passed, idéntico al baseline.
    `grep -rn "SET ROLE" tests/` y `grep -rn "forecast_writer" tests/` = ambos VACÍOS. El único
    test que toca la 071 hace `assert "trg_..." in text` sobre el SQL concatenado.
  cambio mínimo: no es del DDL — la 071 está bien escrita y con privilegios mínimos, y lo digo en
    serio. Es (a) que `--plan fabric-v1` llegue a una base (R-A) y (b) el fixture de R-C.
  test que faltaría: `tests/integration/test_forecast_role_privileges.py` contra el fixture:
    `SET ROLE forecast_writer; INSERT INTO action.<tabla> ...` debe lanzar `InsufficientPrivilege`.
    Mutación que debe ponerlo rojo: sustituir el REVOKE por GRANT ALL en 071 ⇒ el INSERT triunfa y
    el test falla con "expected InsufficientPrivilege, insert succeeded". Hoy esa misma mutación da
    8 failed / 53 passed, indistinguible del baseline.
  desbloquea a: es el molde de BL-21/22/43/44 — el mismo fixture los sirve a los cinco.
  esfuerzo: M (S si el fixture de R-C ya existe)
```

```
BL-21 — NO — palanca: ALTA
  por qué falló: sustituí el cuerpo de `exec.claim_order_dispatch` por `RETURN gen_random_uuid()`
    —fencing DESTRUIDO— y salieron 17 passed, exactamente el mismo resultado: cero tests muertos.
    Y hay un defecto de producto detrás: 2 sleeves sobre USD/COP producen UNA SOLA
    `idempotency_key` (`sha256:c5dcf147...` idéntica) ⇒ `TARGET_DISPATCH_MIXED`, 1 submit con 2
    exposiciones, **orden perdida en silencio**. Los únicos implementadores de
    `claim_order_dispatch` son los fakes del test; nada escribe en
    `exec.order_header/order_status_event/fill_event`.
  cambio mínimo: dos, y el primero NO es de test. (a) La preimagen de la `idempotency_key` debe
    incluir el discriminante de sleeve/exposición; hoy dos exposiciones distintas del mismo
    instrumento colapsan en una clave. (b) El test debe ejecutar la función SQL real contra el
    fixture, no un fake en Python.
  test que faltaría: dos. (1) dos sleeves sobre el mismo instrumento ⇒ exigir DOS
    `idempotency_key` distintas y DOS submits; hoy sale 1 y el test no lo nota. (2) contra la
    función SQL real: dos claims concurrentes del mismo dispatch ⇒ uno gana, el otro recibe
    IDEMPOTENT_REPLAY. Mutación: `RETURN gen_random_uuid()` ⇒ los dos ganan ⇒ rojo. Hoy: 17 passed.
  desbloquea a: BL-22 y BL-30 comparten el camino `exec.*`
  esfuerzo: M
```

```
BL-29 — NO — palanca: ALTA
  por qué falló: dos mitades muy distintas. La que SÍ muerde: desactivé la idempotencia por
    `trial_id` y cayó `test_qlab_idempotency_rejects_same_trial_id_with_changed_result` — append-only
    con lock exclusivo, es código serio. La que no existe como garantía es justo la que tu MD llama
    "el control de mayor apalancamiento de todo el sistema": eliminé `assert_available_at` en
    SCREENING ⇒ 1 failed / 28 passed, idéntico; `grep -rn "point_in_time" tests/` ⇒ VACÍO, cero
    tests; y `qlab screen` exige `--cutoff` pero **jamás lee datos ni llama a la capa de lectura**:
    sólo escribe el string en el ledger.
  cambio mínimo: (1) `.gitignore:204` — anclar `research/` a `/research/` (o `!src/research/`) y
    trackear `src/research/`; hoy `scripts/analysis/qlab.py` está commiteado e importa
    `src.research.qlab`, así que en clon limpio el CLI no arranca y 2 tests mueren en
    ModuleNotFoundError. (2) que `--cutoff` se propague a la capa de lectura; hoy es decorativo.
  test que faltaría: `qlab screen --cutoff 2025-01-01` sobre un dataset que contiene una fila con
    `available_at = 2025-06-01` debe lanzar. Mutación: quitar `assert_available_at` ⇒ rojo con
    "leyó dato de 2025-06-01 con cutoff 2025-01-01". Hoy esa mutación da 1 failed / 28 passed.
  desbloquea a: CI en checkout limpio (2 tests + el CLI `qlab`)
  esfuerzo: S el gitignore, M el cutoff real
```

```
BL-35 — NO — palanca: ALTA (una línea, y es la que anula tu muralla en producción)
  por qué falló: tu Verificación dice "DAG sintético violador ⇒ import error visible en
    list-import-errors". Es falsa: sondeé el `_load_config()` REAL con un config violador
    (`forecast://` → `exec://`) y devolvió `{}` **sin excepción, sólo un WARNING**. El validador
    puro SÍ muerde; lo que no muerde es el camino real.
  cambio mínimo: `airflow/dags/asset_pipeline_factory.py:58-67`. Hoy:
    `validate_dataset_edges(...)` en :63 y `except Exception as e: logger.warning(...); return {}`
    en :65-67, con el docstring "Never raise at DAG-parse time". El docstring tiene razón para
    IOError/YAMLError —un worker puede no ver el fichero— pero NO para una violación
    constitucional. Añade `except DatasetEdgeViolation: raise` ANTES del `except Exception`.
  test que faltaría: `_load_config()` con un `pipelines.yaml` cuyo `dataset_edges` cruce
    `forecast://` → `exec://` debe lanzar y el error debe aparecer en `list-import-errors`.
    Mutación: restaurar el `except Exception` genérico ⇒ rojo. Segundo test obligatorio para no
    romper el camino feliz: `dataset_edges` ausente o vacío ⇒ config válida, sin excepción.
  segunda mitad, y es de producto: `dataset_edges` aparece en **0 ficheros yaml** del repo
    (verificado hoy). Mientras no se pueble `config/assets/pipelines.yaml`, el guard es infalsable
    en producción aunque el test pase. El re-raise sin los edges poblados es media receta.
  desbloquea a: BL-28 (comparten `validate_dataset_edges`)
  esfuerzo: S el re-raise, M poblar los edges
```

```
BL-43 — NO — palanca: ALTA
  por qué falló: quité el guard `if algorithm == "SYNTHETIC": raise` —el corazón del fail-closed—
    ⇒ 7 passed, idéntico; con el guard fuera,
    `validate_model_boundary({algorithm:"SYNTHETIC", environment:"production", surface:"action",
    execution_eligible:True})` fue ACEPTADO. **Causa exacta y muy barata de arreglar**: el bucle
    del test sólo muta `environment`/`surface`/`execution_eligible`, así que enmascara la muerte
    del guard de `algorithm`. Además sustituí la migración 081 entera (211 líneas) por `SELECT 1;`
    más un comentario con las 5 subcadenas ⇒ 7 passed, idéntico.
  cambio mínimo (test): añadir `algorithm` al eje de mutación del bucle parametrizado. Es una
    entrada más en la lista. Con eso el mutante muere.
  cambio mínimo (DDL): en la base viva no existe el schema `demo`, `config.models` sigue con
    `investor_demo | SYNTHETIC | active` —o sea el "Estado actual" de tu BL intacto—, no hay
    columnas `environment/surface/execution_eligible` y `_migrations` tiene 0 filas. Es R-A.
  test que faltaría: (1) el parametrizado con `algorithm` incluido; (2) contra el fixture:
    `INSERT INTO config.models (algorithm='SYNTHETIC', environment='production')` fuera de `demo`
    debe ser rechazado por la restricción de 081. Mutación: 081 ⇒ `SELECT 1;` ⇒ el INSERT pasa ⇒
    rojo. Hoy esa mutación da 7 passed.
  reconocimiento: `synthetic_isolation.py` es código correcto y con uno de los pocos unit tests
    genuinos de tu lote — por eso el mutante de `environment/surface/execution_eligible` SÍ lo
    mataría. El problema es que el módulo está fuera de HEAD y que `migration-integrity` se
    satisface con dos ficheros que contengan `SELECT 1;`.
  desbloquea a: `fabric-contracts.yml` en checkout limpio (el `git add` del módulo)
  esfuerzo: S el test, M el DDL (la capa visible es lane de Claude, ver §4)
```

```
BL-26 — PARCIAL — palanca: ALTA por precio (es un commit)
  por qué falló: tu criterio literal PASA entero y lo verifiqué caso por caso. Lo único que lo
    frena es P-01: `PortfolioSnapshot` acepta `snapshot_id='whatever-i-want'` y cutoff naive. **El
    fix ya está escrito en tu working tree, sin commitear.** Confirmado hoy: `src/portfolio/
    snapshot.py` sigue ` M` y su último commit es `b18720d1`. Eso es WIP, no DONE — y es la única
    razón por la que este BL no es un "sí".
  cambio mínimo: `git commit src/portfolio/snapshot.py`. La comprobación ya está en el árbol de
    trabajo (`snapshot.py:225-227`, `snapshot_id != expected_id ⇒ raise`).
  test que faltaría: dos asserts en el mismo test. (1) `PortfolioSnapshot(snapshot_id=
    'whatever-i-want', ...)` debe lanzar con "expected <uuid5>, got whatever-i-want".
    (2) `cutoff=datetime(2026,1,1)` sin tzinfo debe lanzar. Mutación: quitar el `raise` de
    :225-227 ⇒ rojo en el primero. Sin ese test, el fix commiteado tampoco estaría protegido.
  desbloquea a: BL-27 (el allocator consume snapshots)
  esfuerzo: S
```

---

## 3. Fichas — palanca MEDIA

```
BL-22 — NO — palanca: MEDIA
  por qué falló: puse `identity_error := 0::NUMERIC` —la identidad contable no puede fallar jamás—
    y la suite salió IDÉNTICA: 8 failed / 53 passed antes y después. Tu "test de CI" es
    coincidencia de subcadenas sobre el TEXTO del SQL.
  reconocimiento primero, porque es real: M-05 SÍ está remediado. `reconstructed_gross_pnl` ya
    excluye `pnl_residual` y el residuo se deriva aparte (075:107-126). Verificado de nuevo hoy.
  cambio mínimo (bug de signo, verificado línea a línea hoy): `database/migrations/
    075_fact_position_pnl.sql:127-140` calcula `ABS(calculated_residual) - reported_residual`. Con
    un residuo negativo consistente (reported = calculated = -X) da `X - (-X) = 2X > 0` ⇒ falso
    positivo, y el trigger de :188 dispara sobre una identidad correcta. Debe ser
    `ABS(calculated_residual - reported_residual)`. Es mover un paréntesis.
  cambio mínimo (2): `strategy.timing_ratio` no tiene IC — 0 hits de bootstrap/confidence en
    `src/metrics/`. Un ratio sin intervalo no es reportable bajo la constitución §6.
  test que faltaría: dos filas en `fact.pnl` contra el fixture. (1) fila consistente con
    `pnl_residual = -50` ⇒ exigir `identity_error = 0`; hoy daría 100 y el test prueba el bug de
    signo antes de arreglarlo. (2) fila inconsistente ⇒ exigir que el trigger de :188 lance.
    Mutación: `identity_error := 0` ⇒ el segundo test cae. Hoy: 8/53 idéntico.
  desbloquea a: BL-23 (depende de este), BL-17 (ambos anclan sobre "el paper ledger anclado")
  esfuerzo: S el signo, M el test contra DB
```

```
BL-16 — PARCIAL — palanca: MEDIA
  por qué falló: la matriz de legalidad es correcta y **su mitad está cerrada por mutación** —
    PAPER admitiendo `CapitalTier.FULL` ⇒ `assert 28 == 26`, y te lo reconozco con el número: 26 de
    96 combinaciones legales, bien pensado. La otra mitad no tiene candado: hice que NaN/Inf
    devolvieran `null` en vez de lanzar `CanonicalizationError` ⇒ 8 failed / 53 passed, IDÉNTICO.
    El único rojo en TODO el repo fue `test_non_finite_json_is_refused_rather_than_hashed`, que es
    un test MÍO (BL-31) y que ningún workflow ejecuta.
  el problema de fondo, y es el que importa: `GovernanceDeclaration` tiene 0 consumidores y
    **ningún fichero de config, manifest o registry contiene `research_state` ni `capital_tier`**.
    La declaración que el CI debería rechazar no existe ⇒ tu garantía central es **infalsable hoy**,
    no sólo no testeada.
  cambio mínimo: (1) un test de NaN/Inf en TU lote y en un workflow — hoy la única red que lo cubre
    es ajena y no se ejecuta. (2) Para hacer falsable "Declaración PAPER+FULL ⇒ CI rojo", tiene que
    existir al menos UNA declaración real: añade `research_state` / `capital_tier` al front-matter
    de un manifest existente (el de `smart_simple_v11` es el candidato natural) y que el gate lo lea.
  test que faltaría: un gate que recorra los manifests REALES y falle si alguno declara una
    combinación ilegal. Mutación: poner `research_state: PAPER` + `capital_tier: FULL` en ese
    manifest ⇒ CI rojo nombrando el fichero. Hoy no hay manifest que mutar, que es exactamente el
    problema.
  desbloquea a: BL-17 (declara dependencia de BL-16)
  esfuerzo: M
```

```
BL-17 — PARCIAL — palanca: MEDIA
  por qué falló: <sin mutación individual publicada — el veredicto de CLD-208 lo agrupa en el
    diagnóstico transversal>. Lo que sí está medido: tus módulos de identidad están genuinamente
    remediados (`1.0` vs `"1"` ya no colisionan, `int 2 == float 2.0`, CRLF≡LF, numpy rechazado
    uniformemente, los 12 módulos nuevos importan). Lo que falta no es diseño: es que **el gate que
    tu propio BL declara no existe** ("replay independiente reproduce el semantic_hash del paper
    ledger anclado") y que 6 de tus tests están ROJOS por un rename
    (`annualization_by_asset`→`annualization_registry`, `environment`→`env`).
  cambio mínimo: (1) los 6 rojos del rename — mecánico, media hora. (2) Escribir el gate declarado:
    un job que recomputa el `semantic_hash` del paper ledger anclado desde cero y lo compara con el
    sellado.
  test que faltaría: replay independiente del paper ledger ⇒ mismo `semantic_hash`. Mutación:
    cambiar un decimal de una fila del ledger anclado ⇒ rojo mostrando el par de hashes. Ése es
    literalmente el criterio de tu MD y hoy no existe como comando ejecutable — que es el defecto
    de forma que comparten los cuatro MDs de CLD-235.
  desbloquea a: BL-22, BL-23 (ambos anclan sobre "el paper ledger anclado")
  esfuerzo: S el rename, M el gate
```

```
BL-23 — PARCIAL — palanca: MEDIA
  por qué falló: el backfill corre limpio y te lo reconozco con los números —18 estrategias, 859
    metric_events, 3240 trades, `missing: []`— pero **nada aplicado a DB**, su test está ROJO en
    HEAD por un bug de portabilidad real en `inventory()`, y depende de BL-22, que es NO.
  cambio mínimo: (1) el bug de portabilidad de `inventory()`, que es rojo hoy y no depende de nada
    más. (2) Ejecutar el backfill contra la base con `fabric-v1` aplicado (R-A).
  test que faltaría: tu propio criterio ya es una query — hazla ejecutable contra el fixture: "toda
    estrategia del registry tiene facts en sus años publicados", devolviendo la lista de huecos.
    Mutación: borrar las filas de un año de una estrategia ⇒ rojo nombrando estrategia y año. Hoy
    no hay nada que ejecutar porque no hay base con el DDL.
  desbloquea a: nada aguas abajo; está bloqueado por R-A y por BL-22
  esfuerzo: M
```

```
BL-27 — NO — palanca: MEDIA
  por qué falló: **tus DOS requisitos de verificación fallan**. El gate de CI que prohíbe
    `normalize()` no existe en ninguno de los 13 workflows ni en el Makefile. El shadow de ≥26
    periodos no tiene corrida: `shadow: {method: HRP, minimum_periods: 26}` con 0 implementación.
    Con solver fiel: 7 intentos, 2 regiones factibles distintas, y los intentos 2..7 comparten UNA.
  el defecto que de verdad importa, y no es ninguno de los dos anteriores: el **fallback 4** da
    turnover implícito **0.6 contra límite declarado 0.10** y beta fuera de banda **sin incidente**,
    saltándose `_validated_solution` entero. Eso es un camino que devuelve una asignación ilegal en
    silencio.
  cambio mínimo: (a) que `fallbacks[4] = target_zero_and_critical_incident` pase por
    `_validated_solution` como los otros tres; hoy la esquiva. (b) `AllocatorV1.from_config(...)`
    para que `config/book/allocator_v1.yaml` deje de ser decorativo — verificado hoy: no existe
    ningún `from_config` en `src/portfolio/`, y `AllocatorV1` (`src/portfolio/allocator.py:136`)
    no tiene importadores fuera de tests. (c) el grep-CI de `normalize()` es una línea en un
    workflow; es lo más barato pero lo menos importante de los tres.
  test que faltaría: forzar el fallback 4 (solver infactible por construcción) y exigir que el
    resultado pase por `_validated_solution` y que el turnover implícito ≤
    `turnover_relaxation_limit_decimal` (0.30) o se emita un incidente. Hoy sale 0.6 en silencio.
    Mutación: devolver turnover 0.9 desde el fallback ⇒ rojo. Segundo test: `novelty_gate`
    (`allocator.py:717`) calcula bien pero no está cableado a ninguna promoción — un test que
    exija que la promoción lo invoque (centinela monkeypatcheado) cierra ese hueco.
  desbloquea a: nada aguas abajo hoy (0 consumidores)
  esfuerzo: M
```

```
BL-30 — NO — palanca: MEDIA
  por qué falló: lo bueno, verificado y reconocido: discrepancia de broker ⇒
    `PRETRADE_REJECTED:RECONCILIATION_CLEAN` con 0 submits, y los 4 niveles no-CLEAR bloquean
    apertura. Pero C-03 sigue vivo: `EXIT_ALL` con target expirado ⇒ `exits=0`, y sin target ⇒
    `exits=0`. Confirmado hoy: `src/execution/service.py` expone **una sola** corrutina pública,
    `execute_target` (:343). C-04 vivo: `_pretrade` (:513) sin `controls` ⇒ `allowed=True`. Y
    `grep ExecutionService` en `services/ airflow/ scripts/` ⇒ 0 hits: el camino vivo sigue siendo
    SignalBridge.
  cambio mínimo: (a) `async def exit_all(account_id, reason)` que NO dependa de un target vigente —
    el kill switch tiene que funcionar precisamente cuando el target expiró, que es el escenario en
    que se necesita. (b) Invertir el default de `_pretrade`: `controls is None` ⇒ BLOCK. Es la regla
    fail-safe que ya rige `PreTradeGate` en SignalBridge (`rbac.md` §6: "error ⇒ BLOCK"), así que
    aquí es incoherencia entre dos motores, no una opinión.
  test que faltaría: (1) `exit_all` con target expirado, y con target ausente, sobre una cuenta con
    N posiciones abiertas ⇒ exigir N cierres, no 0. Mutación: devolver `exits=0` ⇒ rojo. (2)
    `_pretrade(controls=None)` ⇒ exigir `allowed=False`; mutación: `allowed=True` ⇒ rojo.
  desbloquea a: nada hoy; bloquea el retiro de SignalBridge mañana
  esfuerzo: M
```

```
BL-37 — PARCIAL — palanca: MEDIA (y su criterio necesita recorte, ver §5)
  por qué falló: estructuralmente insatisfacible hoy — `reference.asset/instrument/provider_symbol`
    **nunca se pueblan** (cero seed/backfill), el manifiesto no se descompone y no se añaden FKs a
    las tablas de mercado existentes. Divergencia concreta y viva, re-verificada hoy:
    `src/market/identity.py:16` declara `M1 = "PT1M"` y `src/market/resampling.py:21` lo mapea,
    pero `database/migrations/072_reference_identity.sql:69-76` sólo siembra
    PT5M/PT1H/PT4H/P1D/P1W/P1M ⇒ un símbolo de 1 minuto resuelve en Python y violaría la FK en DB.
  cambio mínimo: añadir `('PT1M', 60, FALSE),` al INSERT de 072:69. Una línea.
  test que faltaría: **test de paridad enum↔DDL**, parametrizado sobre `BarInterval`: cada miembro
    del enum Python debe existir en el seed de `reference.bar_interval` y viceversa. Mutación:
    quitar `P1W` del seed ⇒ rojo nombrando el miembro huérfano. Hoy ese test no existe y por eso la
    divergencia PT1M lleva viva desde que se escribió la 072. Es el patrón que el skill
    `contract-change` ya exige en este repo: un enum en dos lenguajes necesita un test de paridad,
    no dos listas mantenidas a mano.
  desbloquea a: BL-38 y BL-40 (ambos resuelven instrumento contra `reference.*`)
  esfuerzo: S la línea + el test de paridad; L el seed completo (ver §5)
```

```
BL-38 — PARCIAL (al borde de NO) — palanca: MEDIA
  por qué falló: 073 y 080 están bien, pero **cero vistas de compatibilidad** —y tu BL las exige
    "primero"—, **nada escribe** `raw_bar`/`canonical_bar`, `src/market/resampling.py` está
    UNTRACKED (confirmado hoy contra HEAD), y sigue sin existir la regresión del anclaje UTC de la
    barra diaria: el "Sunday pile-up" que este repo **ya sufrió** con Gold.
  cambio mínimo: (1) `git add src/market/resampling.py`. (2) La regresión del anclaje UTC — es la
    más barata de las tres cosas que faltan y la más valiosa, porque el bug ya se pagó una vez y
    `data-governance.md` §1 lo documenta como invariante.
  test que faltaría: una barra diaria sellada a 00:00 UTC en domingo, resampleada con la política
    de sesión de Gold, no puede caer en domingo. Mutación: sustituir
    `ts.dt.tz_convert("UTC").dt.normalize()` por `tz_convert(→ET).normalize()` ⇒ todas las barras
    se corren un día atrás y el test debe caer nombrando la fecha concreta. Hoy no hay nada en el
    repo que detecte esa regresión.
  desbloquea a: BL-40 (comparten `canonical_bar`)
  esfuerzo: S el add, M la regresión
```

```
BL-40 — PARCIAL — palanca: MEDIA
  por qué falló: hallazgo offline sobre datos REALES del repo — tu rango `usdmxn [5,100]` pone en
    cuarentena **59 barras LEGÍTIMAS de 1990** (peso pre-redenominación, low mínimo 2.712) en
    `asset_native_ohlcv.parquet`. Y `QualityRuleSet` tiene cero consumidores de producción: nada
    escribe `quality.quarantine_event`.
  cambio mínimo: `config/quality/market_price_ranges.yaml` (4 líneas hoy) declara un `version`
    global y `usdmxn: [5, 100]`. Necesita **validez por época**, no una versión global:
    `usdmxn: [{until: 1992-12-31, range: [1000, 5000]}, {from: 1993-01-01, range: [5, 100]}]`.
    Del lado Python, `src/data_quality/rules.py:37-47` construye `_price_ranges` como
    `dict[str, tuple[Decimal, Decimal]]` y `:133` lo resuelve sólo por instrumento — hay que
    resolver por (instrumento, fecha de la barra).
  test que faltaría: pasar las 59 barras de 1990 de `asset_native_ohlcv.parquet` por el ruleset y
    exigir **0 cuarentenas**; y una barra de 2020 a 2.712 debe cuarentenarse. Mutación: colapsar
    las épocas a un rango único ⇒ las 59 vuelven a caer ⇒ rojo con el conteo. Ese test usa datos
    reales del repo, no sintéticos, y es el tipo de test que a este backlog le falta casi entero.
  desbloquea a: nada aguas abajo
  esfuerzo: M
```

```
BL-18 — NO — palanca: MEDIA (y su criterio necesita recorte, ver §5)
  por qué falló: tu Verificación dice "Grep-CI: ningún sharpe/calmar fuera del motor". Grep
    ejecutado: **30+ implementaciones independientes** — `src/evaluation/benchmarks.py:64`,
    `src/forecasting/evaluation/metrics.py:143`, `src/monitoring/model_monitor.py:222`,
    `services/pipeline_data_api.py:821,843`, `services/trading_analytics_api.py:141,168`…
    Y el único "validador" tiene 12 líneas, sólo comprueba que exista un fichero, y ese fichero
    está UNTRACKED.
  cambio mínimo: el criterio "cero fuera del motor" no es alcanzable en una ola con 30+ sitios, y
    por eso el validador acabó teniendo 12 líneas. Ver §5 para el recorte. Lo ejecutable hoy es el
    allowlist congelado.
  test que faltaría: grep-CI con allowlist sellado de las 30 ubicaciones actuales. Dos mutaciones,
    y las dos importan: (1) añadir un `def sharpe(` en cualquier fichero fuera del motor ⇒ rojo con
    la ruta; (2) **añadir una entrada nueva al allowlist sin migrar nada ⇒ también rojo** — sin
    esta segunda, el gate se "arregla" ampliando la lista y vuelve a ser decorativo.
  desbloquea a: BL-22 (el `timing_ratio` sin IC vive en el mismo motor), BL-25
  esfuerzo: M
```

```
BL-28 — NO — palanca: MEDIA (mitad test, mitad criterio irrealizable — ver §5)
  por qué falló: borré **las DOS murallas** de `assert_constitutional` en
    `src/orchestration/factories.py:59-73` (FORECAST sólo `forecast://` en :60-66; STRATEGY sólo
    strategy/action/artifact en :67-73) ⇒ 3 failed / 26 passed, idéntico. El test que se llama
    `preserves_action_diagnostic_wall` **no comprueba el guard**: afirma sobre el CONTENIDO del
    YAML, así que pasa con el guard borrado.
  reconocimiento, y es de los importantes: `airflow/dags/fabric_factories.py` es **cableado de
    producción real**, trackeado y generando DAGs en parse-time. No es un módulo huérfano — es de
    los pocos de tu lote que sí está enchufado, y eso cuenta.
  cambio mínimo: un test que construya `FactorySpec(kind=FORECAST, produces=['action://x'])` y
    exija `ValueError`, y su simétrico para STRATEGY. Hoy no existe ninguno de los dos.
  test que faltaría: parametrizado sobre las dos murallas. Mutación: borrar cada `raise`
    (factories.py:64 y :72) ⇒ **un rojo por cada uno**. Hoy: 3 failed / 26 passed, indistinguible.
  desbloquea a: nada
  esfuerzo: S el test; el criterio de 2 semanas es otra cosa (§5)
```

---

## 4. Lo que NO es trabajo pendiente de ingeniería

**No lo pongas en la cola. No cuenta como deuda tuya.**

```
BL-41 — NO, pero CORRECTAMENTE bloqueado — palanca: no aplica
  estado medido en la base viva: 3 tablas de credenciales (no 2) — `exchange_credentials`,
    `sb_exchange_credentials`, `user_exchange_keys` —, **0 filas en las tres**; el schema `secret`
    no existe; sólo `admin` tiene login y es superusuario; `PUBLIC` conserva `CREATE` en `public`;
    la migración 069 no existe (el directorio salta de 068 a 070); `grep -rl "BL-41"` ⇒ 0 ficheros.
    Cero DDL, cero código, cero test.
  por qué está bien así: las precondiciones del operador (Vault real + roles no-superusuario) están
    **objetivamente incumplidas**. Escribir DDL de roles contra una base donde sólo existe un
    superusuario sería teatro, y además fijaría por escrito una segregación que nadie puede
    ejercer. No aplicar DDL es la decisión correcta.
  lo único que debe hacerse hoy, y es de acta: corregir el MD. Dice "listado de public.* sin tablas
    de credenciales" y son **3**, no 2; y `forecast_writer`/`frontend_role` **no existen**, así que
    su criterio no es un comando ejecutable sino una descripción. Y que conste en el acta que
    **hoy no protege nada**: lo único que reduce el riesgo es que las tres tablas están vacías.
    Dato favorable medido: `airflow` no tiene SELECT sobre ninguna de las tres.
  esfuerzo: S (sólo acta)
```

```
BL-44 — NO — palanca: BAJA — y NO se arregla con un test
  por qué falló: la base viva te contradice en los CUATRO ejes. Compresión: **7/7 hypertables con
    `compression_enabled = f`, CERO** — y tu Verificación exige "compresión MEDIDA en las 3 tablas
    grandes". Caggs 1h/4h/1d: **0 filas** en `continuous_aggregates`. Hypertables vacías que tu
    propio BL manda no crear (`crypto_exposure_signals`, `crypto_flows_daily`,
    `crypto_onchain_daily`) siguen ahí con 0 filas. Chunks sin gobierno: `asset_daily_ohlcv` a
    **3600 días (~10 años)**, `usdcop_m5_ohlcv` a 360, `trading_metrics` a 7.
  y el "perfil v2" son DOCE LÍNEAS: una constante SQL y un getter que no abre conexión. Corrí su
    propia query contra la base: `ERROR: column h.chunk_time_interval does not exist` (esa columna
    no existe en TimescaleDB 2.x), más un `LEFT JOIN ... ON TRUE` que es producto cartesiano, y
    **no selecciona ninguna columna de compresión ni de tamaño**, que es justo lo que había que
    medir.
  por qué esto es de producto y no de cobertura: **ningún test hace que la compresión exista.**
    Orden correcto: (1) reescribir la query del perfil contra el catálogo real de TimescaleDB 2.x
    (`timescaledb_information.dimensions` para el intervalo; `hypertable_compression_stats()` para
    compresión y tamaño); (2) `ALTER TABLE ... SET (timescaledb.compress)` + `add_compression_policy`
    en las 3 tablas grandes; (3) los caggs; (4) borrar las 3 hypertables vacías. El test va DETRÁS,
    no delante.
  cambio mínimo del lado test, y es casi una parábola: en checkout limpio ese test **ya estaba
    rojo**, de la forma más autoincriminatoria posible — `assert "add_retention_policy" not in
    physical` falla porque el SQL contiene la frase **en un COMENTARIO** que dice "deliberately no
    add_retention_policy". El test no distingue DDL de comentario. Mínimo: normalizar quitando
    comentarios antes de afirmar; mejor: afirmar contra la base.
  test que faltaría: el perfil ejecutado contra el fixture devuelve `compression_enabled = true` y
    `after_compression_total_bytes < before_compression_total_bytes` para las 3 tablas grandes.
    Mutación: `remove_compression_policy` en una ⇒ rojo nombrando la tabla. Hoy el job se llama
    `migration-integrity` y se satisface con dos ficheros que contengan `SELECT 1;`.
  reconocimiento, y no es cortesía: la **080 real es un diseño serio** — registro de perfiles, gate
    de row-count mínimo y **evidencia sha256 de restore en frío antes de tocar nada**. El problema
    no es lo que escribiste: es que el `create_hypertable` vive dentro de una función
    "operator-only" que nadie llama.
  esfuerzo: L
```

```
BL-33 — NO — palanca: BAJA — es un artefacto de documentación, no código
  por qué falló: el test `test_ci_and_readiness_matrix_are_executable_honest_contracts` ⇒ 1 passed,
    y lo único que hace es comprobar que **7 palabras aparecen en el fichero**. Mutación: borré la
    tabla ENTERA y dejé un fichero de 5 líneas con las 7 palabras sueltas en un renglón ⇒ 1 passed,
    idéntico. El artefacto real (`.claude/specs/planes/04b-readiness-matrix.md`) son **21 líneas: 7
    filas, 3 columnas**, sin columna de evidencia, sin dueño y sin fecha, y las 7 filas dicen
    `IMPLEMENTED_UNVERIFIED`. No hay fila de BL-08 ni de segregación de identidades, que son las
    dos que tu propio BL nombra como "primeras filas". Es exactamente lo que la nota constitucional
    del BL prohíbe: "no basta con marcar etapas como completadas".
  cambio mínimo: añadir a la matriz las columnas `evidencia` (commit / test / simulacro), `dueño` y
    `fecha`, y las dos filas que faltan.
  test que faltaría: parsear la tabla **como tabla**, no como texto. Exigir ≥5 columnas; que cada
    fila tenga `evidencia` no vacía; y que cada referencia a un test **exista en disco y esté
    nombrada por algún workflow** (esto último es lo que impide que la evidencia sea un test que
    nadie ejecuta — CI-7 midió que 73 de 197 ficheros de test no los nombra ningún workflow).
    Mutación: vaciar una celda de evidencia, o apuntar a un test inexistente ⇒ rojo nombrando la
    fila. Hoy borrar la tabla entera da verde.
  desbloquea a: nada
  esfuerzo: S
```

```
BL-24 — NO — palanca: BAJA — bloqueado por R-A + R-B, y su criterio hoy no se puede ni intentar
  por qué falló: el **DDL 076 es correcto y lo digo sin reservas**. Pero `src/lineage` no exporta
    ninguna clase de arista, `grep lineage` en `airflow/` y `scripts/` da **0 productores**,
    `l0_macro_update.py` tiene **0 hits** de `revision`/`vintage`/`ALFRED` —que es el CRÍTICO que tu
    propio BL declara— y no hay nada de "camino dorado". Tu criterio ni siquiera se puede intentar:
    no falla, es que no hay qué ejecutar.
  cambio mínimo: el "camino dorado" es literalmente UN caso, no una plataforma. Elige UNA señal del
    paper ledger y escribe su productor de aristas end-to-end. Empieza por el emisor de
    `lineage.revision_event` en `l0_macro_update.py`, que es el que tu BL marca como crítico y el
    que hoy tiene 0 hits.
  test que faltaría: dado el `signal_id` X del paper ledger, `resolve_golden_path(X)` devuelve la
    cadena completa hasta la barra L0 y su vintage. Mutación: borrar una arista intermedia ⇒ el
    test debe fallar con "camino roto en <nodo>", **no devolver una cadena parcial en silencio**
    (que es el modo de fallo que un grafo de linaje tiene por defecto y el que hay que prohibir).
    Segundo assert del propio criterio: `LEGITIMATE_RELEASE` no marca STALE histórico.
  desbloquea a: nada; es hoja
  esfuerzo: L
```

---

## 5. Salidas honestas por recorte de criterio

Estos cuatro criterios **no son alcanzables hoy**. La salida legítima es recortarlos **por escrito
y con firma**, no dejarlos abiertos para que la próxima auditoría los vuelva a encontrar.

| BL | Criterio actual | Por qué no es alcanzable | Recorte propuesto |
|----|-----------------|--------------------------|-------------------|
| **BL-18** | "Grep-CI: ningún sharpe/calmar fuera del motor" | Hay **30+** implementaciones independientes hoy. Un "cero" que requiere 30 migraciones no se cumple en una ola, y por eso el validador acabó teniendo 12 líneas | **Allowlist congelado que sólo puede decrecer**: baseline sellado de 30; la 31ª es roja; añadir al allowlist sin migrar también es rojo. Un umbral que sólo baja es un candado; "cero" es una aspiración |
| **BL-28** | "Diff semántico verde ≥2 semanas, viejo vs nuevo" | `semantic_diff` tiene 0 consumidores, `compare_json_files` no se llama nunca, no hay ledger de paridad ni runner ni job, y **`fabric_factories.yaml` declara `schedule: null` en TODOS los sleeves** ⇒ la corrida en paralelo que E7 mide **no puede ni empezar** | O se pone `schedule` real y se arranca el reloj de 2 semanas hoy (y el BL queda abierto 2 semanas, dilo así), o se recorta a **"diff semántico verde sobre N corridas manuales back-to-back"**, que sí es ejecutable esta semana |
| **BL-37** | "Join manifest-dim_asset sin pérdidas; símbolos huérfanos = 0" | `reference.asset/instrument/provider_symbol` **nunca se pueblan**. No hay qué juntar | Recortar esta ola a **paridad enum↔DDL + seed de `bar_interval`** (alcanzable hoy, y cierra la divergencia PT1M viva). El join sale como BL nuevo cuando exista el seed |
| **BL-16** | "Declaración PAPER+FULL ⇒ CI rojo" | **Ningún** fichero de config, manifest o registry contiene `research_state` ni `capital_tier`. La declaración que el CI debe rechazar no existe ⇒ el criterio es **infalsable**, no sólo no cubierto | O el criterio incluye **crear la primera declaración real** (y entonces es falsable), o se marca explícitamente como *"garantía escrita, sin superficie que la ejercite — se valida en la ola N"*. Lo que no vale es contarlo como cubierto |

**Un criterio recortado y firmado vale más que un criterio ambicioso que nadie puede ejecutar**:
el segundo es el que produce validadores de 12 líneas y `## Verificación` que describen
comportamientos en vez de nombrar comandos. Los cuatro MDs de CLD-235 tienen ese defecto de forma
y es el más barato de arreglar de todo el backlog.

---

## 6. Notas de contabilidad

- **21 fichas**: 14 NO (BL-18, 19, 21, 22, 24, 27, 28, 29, 30, 33, 35, 41, 43, 44) + 7 PARCIAL
  (BL-16, 17, 23, 26, 37, 38, 40). Fuera: BL-10 (DONE-ABLE, CLD-234), BL-07 (cerrado con el
  criterio anterior), BL-08 (decisión del operador, sin verificar).
- **Discrepancia menor que conviene sellar**: el balance de CLD-235 dice "PARCIAL 5 / NO 15". El
  recuento por mensaje da **7 PARCIAL / 14 NO** (CLD-207: 5 NO + 2 PARCIAL · CLD-208: 2 NO + 4
  PARCIAL · CLD-233: 3 NO · CLD-234: 1 NO + 1 DONE-ABLE · CLD-235: 3 NO + 1 PARCIAL = 22
  verificados). Uso 7/14. Si prefieres el otro reparto, dilo y ajusto — pero que los dos contemos
  igual antes de que esto llegue al marcador.
- **Palancas**: ALTA 6 (BL-19, 21, 26, 29, 35, 43) · MEDIA 11 · BAJA 3 (BL-24, 33, 44) · sin
  palanca 1 (BL-41, no es trabajo de ingeniería).
- Todo lo etiquetado "verificado hoy" se midió contra HEAD `278a1e46` y el working tree del
  2026-07-28. Lo demás procede literalmente de CLD-207/208/233/234/235 y no lo he re-medido.

> **ERRATA 2026-07-29 — LA RECETA DE BL-22 ERA FALSA Y SE RETIRA.**
> Afirmaba que `075_fact_position_pnl.sql` calculaba `ABS(calculated) - reported` y que el arreglo
> era *"un parentesis"*. **Es incorrecto**: el `ABS(` de la linea 128 **cierra en la 140**, envolviendo
> la resta entera — o sea que **ya era `ABS(calculated - reported)`**. CODEX lo refuto ejecutando la
> expresion SQL real con `calculated=-5 / reported=-3`: produccion da **2** (correcto) y **aplicar mi
> receta daria 8**. **Seguirla habria ROTO codigo correcto.**
> Causa del error: se leyo `ABS(` al principio de la linea y se infirio la forma sin seguir el cierre
> del parentesis. Yo lo 'verifique' comprobando unicamente que `ABS(` aparecia ahi — **confirme el
> token, no la expresion**, que es la misma clase de defecto que este documento denuncia.

> **ERRATA 2026-07-29 (2) — AUDITORIA COMPLETA TRAS LA RETRACTACION DE BL-22.**
> Auditado hecho a hecho contra `278a1e46` + base viva: ~93 afirmaciones VERIFICADAS,
> **6 FALSAS**, 7 bloques NO COMPROBABLES. Las tres graves:
>
> **BL-40 — RECETA RETIRADA.** Las 59 barras van de **1990-02 a 1995-01** (1990 tiene 11), y
> **no son peso pre-redenominacion**: la serie esta **retro-ajustada a peso nuevo** de punta a
> punta (ratio 1992-12/1993-01 = **1.0032**, sin salto x1000). Las 59 violaciones son **todas
> POR DEBAJO de 5**. El fix de epocas propuesto **deja las 59 en cuarentena**: las de 1990-92
> pasarian a violar [1000,5000] y las de 1993-95 siguen violando [5,100] — **su propio test
> fallaria contra su propio fix**. El arreglo real es **un solo numero**: rango [2.5, 100] =>
> **0 cuarentenas** (maximo de la serie 25.76). No hace falta validez por epoca ni resolucion
> por (instrumento, fecha). Y la ficha **subestima el problema**: **233.784 barras** caen por
> `bar.unknown_instrument` — ese, y no las 59, es el problema de tamaño.
>
> **BL-43 — "cambio minimo (test)" RETIRADO.** Añadir `algorithm` al bucle parametrizado **NO
> mata el mutante**: el bucle corre contra `relation="demo.synthetic_model"` y el guard
> `algorithm == "SYNTHETIC"` vive en la **otra rama**, la de relaciones reales. **Verificado
> ejecutando el mutante.** Lo que si lo mata es una asercion nueva con marcadores de produccion
> contra `config.models` — que es lo que CODEX implemento.
>
> **BL-27 — DIAGNOSTICO Y RECETA CORREGIDOS; LA RECETA ERA PELIGROSA.**
> (a) El limite declarado es `turnover_budget_decimal: **0.20**`, no 0.10 (0.10 es
> `target_vol_decimal`, otro campo). (b) El fallback 4 **NO es silencioso**: emite
> `ALLOCATOR_FALLBACK_4_TARGET_ZERO` con severidad **CRITICAL** (`allocator.py:519-525`),
> expuesto por `_result` junto a `fallback_level=4` — **mi propio nombre para el fallback ya lo
> decia**. (c) Devuelve **todo ceros**: el libro mas plano posible, no una "asignacion ilegal".
> (d) **Enrutarlo por `_validated_solution` ROMPERIA EL KILL-PATH**: esa funcion lanza
> `InfeasibleAllocation` si el turnover excede el presupuesto (:677) o si la exposicion
> factorial sale de banda (:686-692), **y no hay fallback 5** — aplanar a cero desde un libro
> con gross > 0.30 **lanzaria en vez de aplanar**, justo cuando todo lo demas ya es infactible.
> Lo unico que sobrevive: el fallback 4 **si** esquiva `_validated_solution`.
>
> **MENORES.** BL-35: la clase es `DatasetContractError`; **`DatasetEdgeViolation` no existe**
> (seguirla da `NameError`). §5/BL-28: `schedule: null` esta en **3 de 7** sleeves, no en todos
> — los 4 `data:` tienen cron real; la conclusion sobre E7 sobrevive, **la premisa no**. BL-30:
> `grep ExecutionService` en `services/` **no** da 0 hits (hay una clase homonima en
> SignalBridge); lo cierto es que **no hay importador de `src.execution.service` fuera de
> tests**. Validador BL-18 = **14** lineas, no 12. 081 = **205** en arbol / 22 commiteadas, no
> 211 (211 era el conteo de `git diff --stat`). Los `raise` de `factories.py` empiezan en **:64
> y :71**. `REVIEW_GATED_PLANS` exige **`--reviewed-digest`**; `--plan-digest` solo imprime.
>
> **NO COMPROBABLE, y hay que decirlo:** **todos** los conteos de mutacion del documento
> ("8 failed / 53 passed", "17 passed", "7 passed", "1 failed / 28 passed", "3 failed /
> 26 passed") se midieron contra **un working tree que ya no existe** y **no son reproducibles**.
> Ademas `278a1e46` se commiteo a las **23:37:31**, asi que **no pudo ser HEAD** del arbol leido
> "23:1x": el front-matter **declara una procedencia que no cuadra**.
>
> **CORRECCIONES A FAVOR DEL DOCUMENTO.** **BL-44 sigue siendo CORRECTA**: el 080 del baseline
> contiene el comentario `-- Deliberately no add_retention_policy: ...` y el test del baseline
> **no quita comentarios**, asi que estaba rojo por lo que dice la ficha (un agente la dio por
> falsa leyendo el 080 **reescrito** del working tree — el mismo error de version que este
> documento denuncia). Y **"26 de 96" (BL-16) es correcta**: el test itera **TRES** ejes
> (8 x 6 x 2).
>
> **HALLAZGO QUE LA FICHA BL-16 NO HIZO Y DEBERIA:** `070_fabric_control_plane.sql:11-44`
> declara `research_state` y `capital_tier` como columnas con CHECK **y reimplementa la matriz
> 26/96 ENTERA en SQL**, sin cableado con la de Python. **Es duplicacion de SSOT** — la clase de
> defecto que este mismo documento vigila en otras fichas.
>
> **CAUSA COMUN DE LAS TRES GRAVES, y es la misma de BL-22:** se leyo **un token** —el numero
> 2.712, los nombres de los ejes del bucle, la palabra "fallback"— y **se infirio la forma sin
> seguir la expresion, el flujo de control ni los datos hasta el final**.
> **REGLA QUE QUEDA: ninguna receta de "una linea" se publica sin ejecutar el mutante o los
> datos reales que dice arreglar.**

