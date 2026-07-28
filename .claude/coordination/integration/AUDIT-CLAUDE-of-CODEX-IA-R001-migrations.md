---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - database/migrations/070_fabric_control_plane.sql
  - database/migrations/071_forecast_schema_roles.sql
  - database/migrations/072_reference_identity.sql
  - database/migrations/073_market_quality.sql
  - database/migrations/074_exec_event_sourcing.sql
  - database/migrations/075_fact_position_pnl.sql
  - database/migrations/076_lineage_graph.sql
  - database/migrations/077_portfolio_control.sql
  - database/migrations/078_exec_reconciliation.sql
  - database/migrations/079_fabric_integrity_remediation.sql
  - scripts/ops/db_migrate.py
  - scripts/data/backfill_catalog_facts.py
  - services/inference_api/entrypoint.sh
  - Makefile
---

# AUDIT-CLAUDE of CODEX — IA-R001 · migraciones 070-079, migrador y backfill

> **Objeto sellado**: `b18720d10a2d84c5217c3800fe421ea23607f921`.
> Todo lo auditado se leyó con `git show b18720d:<path>` / `git archive`, **no** del working
> tree. Verificado además que los 12 ficheros de mi alcance son **byte-idénticos** entre
> `b18720d` y el WT actual (`git diff --quiet b18720d -- <path>` → 12/12 sin delta), por lo
> que los `code_anchors` apuntan exactamente al contenido auditado.
>
> **Cero Docker, cero base de datos, cero DDL ejecutado, cero commits, cero edición del
> código auditado.** Las sondas que ejecuté viven en el scratchpad de sesión y corren
> contra emulaciones en memoria o contra ficheros del repo; se transcriben íntegras abajo.
>
> **Alcance**: `database/migrations/07{0..9}_*.sql`, `scripts/ops/db_migrate.py`,
> `scripts/data/backfill_catalog_facts.py`. Los módulos Python de `src/**` y el frontend
> los cubren los otros dos auditores.

---

## 0. Verificación del digest declarado — REPRODUCE

CODEX declara `sha256:05948df2561a8122b528e7c11a0d856d6c51f32b6d2fe0a1f439a7524b04dba9`
(CXD-061). Lo recalculé yo mismo con el algoritmo real de
`db_migrate.py::get_plan_digest` (`name || \0 || bytes || \0`, en el orden del plan),
tomando los bytes **del object store de git**, no del disco:

```
GIT-BLOB(LF): sha256:05948df2561a8122b528e7c11a0d856d6c51f32b6d2fe0a1f439a7524b04dba9
WORKTREE    : sha256:05948df2561a8122b528e7c11a0d856d6c51f32b6d2fe0a1f439a7524b04dba9
DECLARED    : sha256:05948df2561a8122b528e7c11a0d856d6c51f32b6d2fe0a1f439a7524b04dba9
```

**Coinciden los tres.** Además `.gitattributes` fija `*.sql text eol=lf`
(`git check-attr text eol` → `text: set`, `eol: lf`), así que la clase de fallo CRLF que
nos costó BL-13 está cerrada aquí: el digest es reproducible en cualquier OS. Esto está
bien hecho y lo digo sin regatear.

Lo que **no** reproduce es la *utilidad* del digest como gate — ver B-03.

---

## 1. Hallazgos

### B-01 — BLOQUEANTE — `--plan` obligatorio rompe los 6 invocadores reales; el contenedor deja de migrar en silencio

**Fichero:línea**: `scripts/ops/db_migrate.py:497-502` (`required=True`) ·
`services/inference_api/entrypoint.sh:57,68` · `Makefile:228,233,237,247`

**Qué está mal**: `--plan` se declaró `required=True`. **Ningún** invocador existente pasa
`--plan`. Argparse aborta con código 2 antes de tocar nada.

**Escenario de fallo concreto**: `services/inference_api/entrypoint.sh` arranca el
contenedor así:

```sh
python /app/scripts/ops/db_migrate.py || {
    echo "WARNING: Migration script failed, continuing anyway..."
}
...
python /app/scripts/ops/db_migrate.py --validate || {
    echo "WARNING: Schema validation failed, some features may not work"
}
```

El `|| { echo WARNING; }` se traga el exit 2. Resultado: **desde este commit, cada arranque
del `inference_api` no aplica ninguna migración y no valida ningún esquema, y lo único que
queda es una línea WARNING en el log**. Igual `make db-migrate`, `make db-status`,
`make db-validate`, `make db-reset` y el alias legacy `make migrate`: los cinco fallan.

**Evidencia (EJECUTADO)**:

```
$ python scripts/ops/db_migrate.py --status
usage: db_migrate.py [-h] [--status] [--validate] --plan
                     {legacy-init,fabric-v1} [--plan-digest]
                     [--reviewed-digest REVIEWED_DIGEST]
db_migrate.py: error: the following arguments are required: --plan
EXIT=2
```

**Remedio**: o `--plan` con `default="legacy-init"` (preservando el contrato previo) y
`required=True` solo para los planes review-gated, o actualizar **los 6 invocadores** en el
mismo commit. Y el entrypoint debe **fallar cerrado** (`|| exit 1`) en la migración: tragarse
un fallo de esquema es exactamente cómo se llega a producción con tablas que no existen.

**Test que lo cierra**: regresión que hace `grep` de todo invocador de `db_migrate.py` en
`Makefile`, `**/entrypoint.sh`, `docker-compose*.yml` y `airflow/**`, y afirma que cada uno
pasa un `--plan` válido; más un test que ejecuta el CLI tal cual lo llama el entrypoint y
exige exit 0 o 1, nunca 2.

---

### B-02 — BLOQUEANTE — un solo fallo de migración envenena ese fichero **para siempre**

**Fichero:línea**: `scripts/ops/db_migrate.py:284-310` (`run_migration`, insert de éxito
`ON CONFLICT (filename) DO NOTHING`) vs `:314-322` (insert de fallo, misma clave)

**Qué está mal**: cuando una migración lanza, el `except` registra
`INSERT INTO _migrations (... success=FALSE ...) ON CONFLICT (filename) DO NOTHING`. Esa
fila ocupa la clave `UNIQUE (filename)`. En el reintento, `classify_migrations` sí la
considera pendiente (filtra por `success = TRUE`), la SQL se ejecuta bien… y entonces el
insert de éxito **choca con la fila de fallo**, `DO NOTHING` devuelve `NULL`, y el código
interpreta ese `NULL` como "reclamación concurrente":

```python
if inserted is None:
    raise MigrationDriftError(
        f"concurrent migration claim for {filepath.name}; rerun status"
    )
```

Ese `raise` está **dentro** de `async with conn.transaction()`, así que el DDL recién
aplicado se revierte. Bucle cerrado: la migración nunca se aplica, el mensaje culpa a una
concurrencia inexistente, y la única salida es cirugía manual (`DELETE FROM _migrations`).

**Escenario de fallo concreto**: `078` hace dos `ALTER TABLE exec.order_header ADD
CONSTRAINT … FOREIGN KEY … REFERENCES portfolio.target(...)`. En un despliegue donde `074`
ya existe y hay órdenes, la validación de la FK falla → 078 queda envenenada → 079 nunca
corre → el plan queda a medias de forma **irrecuperable por CLI**.

**Evidencia (EJECUTADO)** — sonda que importa el `db_migrate.py` **sellado** y ejecuta la
función real `run_migration` contra una emulación en memoria de `_migrations` (UNIQUE
filename) y de `conn.transaction()`:

```
### attempt 1: migration raises (e.g. FK validation / transient error)
   result: False | relation "portfolio.target" does not exist
   _migrations rows: [{'filename': '078.sql', 'checksum': 'f30d59154d973b3946bdba7fa0591ee3',
                       'success': False, 'error': 'relation "portfolio.target" does not exist'}]

### attempt 2: the cause is fixed, SQL now succeeds. Re-run.
   result: False | concurrent migration claim for 078.sql; rerun status
   _migrations rows: [... success: False ...]     # sin cambio
   DDL actually committed: []                     # rollback

### attempt 3 (and every future attempt)
   result: False | concurrent migration claim for 078.sql; rerun status

VERDICT: POISONED (never applies again)
EXIT=1
```

Sonda: `<scratchpad>/probes/probe_migrator_poison.py` (te la paso íntegra si la quieres en
el repo; no la commiteo yo).

**Remedio**: no reutilizar `filename` como clave de idempotencia y de bitácora de fallos a
la vez. O bien (a) `PRIMARY KEY (filename)` solo para éxitos y una tabla aparte
`_migration_failures`, o (b) `ON CONFLICT (filename) DO UPDATE SET checksum=…, success=TRUE,
…` cuando la fila previa tiene `success=FALSE` y el checksum coincide, o (c) registrar el
fallo en una transacción **separada** y borrar esa fila al reintentar.

**Test/mutación que lo cierra**: exactamente la sonda de arriba como test: fallo → reintento
exitoso → `success=TRUE` y DDL committeado. Hoy sale rojo.

---

### B-03 — BLOQUEANTE — el "review gate" es autorreferencial: no compara contra ningún digest fijado

**Fichero:línea**: `scripts/ops/db_migrate.py:242-254` (`plan_is_authorized`)

**Qué está mal**:

```python
def plan_is_authorized(plan, reviewed_digest):
    if plan not in REVIEW_GATED_PLANS:
        return True
    expected = get_plan_digest(plan)     # <- recomputado del DISCO, ahora mismo
    if reviewed_digest != expected:
        return False
    return True
```

`expected` no es un valor fijado en el repo: es el hash de los mismos bytes que se van a
ejecutar, leídos en el mismo instante. Confirmé por grep que **el digest esperado no está
pinneado en ningún sitio ejecutable** — ni constante, ni YAML de config, ni Makefile, ni
workflow de CI. Solo aparece en prosa (`CODEX-STATUS.md`, `INBOX-CLAUDE.md`).

**Escenario de fallo concreto**: quien pueda escribir en `database/migrations/` edita `079`
para añadir `GRANT ALL ON ALL TABLES IN SCHEMA exec TO PUBLIC;`, corre
`--plan fabric-v1 --plan-digest`, copia el nuevo digest, y lo pasa como `--reviewed-digest`.
El gate pasa. La única protección real es la memoria del operador sobre un hash de 64
caracteres que no está escrito en ninguna parte que el sistema pueda comprobar.

**Evidencia**: ESTÁTICO + grep ejecutado sobre `*.py|*.yml|*.yaml|Makefile|*.sh|*.json`:
únicas apariciones de `reviewed-digest`/`fabric-v1` son el propio `db_migrate.py` y los dos
tests que lo ejercitan; ningún literal `05948df2…` fuera de documentos de coordinación.

**Remedio**: pinnear el digest esperado en el repo (`PLAN_DIGESTS = {"fabric-v1":
"sha256:05948df2…"}` o `config/migrations/plans.yaml` bajo control de cambios) y que
`plan_is_authorized` compare **contra el valor pinneado**; `--reviewed-digest` queda como
segundo factor (dos-hombres-una-llave), no como la única referencia. Añadir un job de CI que
falle si `get_plan_digest("fabric-v1") != PLAN_DIGESTS["fabric-v1"]` — ese sí detecta la
edición.

**Test/mutación que lo cierra**: mutar un byte de `075` y afirmar que `plan_is_authorized`
devuelve `False` **aunque el operador pase el digest recién generado**. Hoy devuelve `True`.

---

### B-04 — BLOQUEANTE — `backfill_catalog_facts` descarta en silencio el 63 % de los hechos de PnL que calcula, y mezcla versiones de modelo en una sola posición

**Fichero:línea**: `scripts/data/backfill_catalog_facts.py:305-357` (pnl) y `:359-407`
(position) · grano definido en `database/migrations/075_fact_position_pnl.sql:26,65-67`

**Qué está mal**: el script agrupa PnL por `(strategy_id, asset_id, exit_time, version,
year)` — es decir, **sí** distingue `model_version` y año. Pero la PK de `fact.pnl` es
`(as_of, strategy_id, sleeve_id, instrument_id, environment, pnl_component)` y el script
escribe `sleeve_id = strategy_id`, `environment='backtest'` e `instrument_id` derivado del
activo. **`version` y `year` no forman parte del grano.** El insert usa
`ON CONFLICT DO NOTHING` **sin target**, así que cada colisión es un no-op mudo:
`cursor.rowcount` = 0 y el reporte no lo distingue de "ya estaba".

Para `fact.position` es peor: `position_events` se indexa por
`(strategy_id, asset_id, timestamp)` **sin versión**, y `running_qty` acumula
`signed_qty` de todas las versiones en la misma serie. No es una colisión, es una **fusión**:
la cantidad persistida es la suma de backtests independientes.

**Escenario de fallo concreto**: el registry real tiene 17 de 19 estrategias con varias
`model_version` cubriendo el **mismo año** (p.ej. `smart_simple_v11` con
`1.0.0/1.1.0/2.0.0/3.0.0-A/3.0.0-B` sobre 2025). El backfill que se vende como
anti-supervivencia ("the catalog is the population; status is a dimension, never a filter")
guarda una versión arbitraria por instante y tira el resto sin decirlo.

**Evidencia (EJECUTADO)** — sonda que importa el `backfill_catalog_facts.py` **sellado**,
corre su `inventory()` real sobre `usdcop-trading-dashboard/public/data/registry.json` y
reproduce la agrupación de `_write` (sin tocar ninguna DB):

```
strategies=18 metric_facts=859 trades=3240 missing=0

fact.pnl grains written            : 1189
grains with >1 (version,year) row  : 881
  ... of which values DIFFER       : 170   <-- descartados en silencio por ON CONFLICT DO NOTHING
pnl rows the script computes but never persists: 4102 (x2 components)
   example grain: btc_hodl_b1 2026-07-05T00:00:00+00:00 ->
       [('1.1.0',2026,27309.13), ('1.2.0',2026,27309.13), ('1.0.0',2026,27146.25), ('1.2.1',2026,27309.13)]
   example grain: gold_dynamic_exit 2025-05-01T21:00:00+00:00 ->
       [('1.0.0',2026,1391.97), ('1.0.0',2025,1363.67)]

fact.position instants mixing >1 model_version into ONE qty: 1162 / 2178
EXIT=1
```

Sonda: `<scratchpad>/probes/probe_backfill_grain.py`.

**Remedio**: o el grano de `fact.pnl`/`fact.position` incorpora la identidad del productor
(`run_id`/`derivation_id`/`model_version`) — y entonces hay que tocar `075` con una `080`
aditiva —, o el backfill deja de fabricar filas que sabe que colisionan. En cualquier caso:
**`ON CONFLICT DO NOTHING` sin target está prohibido en un cargador de hechos**; usa target
explícito, `RETURNING`, y cuenta e informa los descartes divergentes como incidente, no como
cero.

**Test/mutación que lo cierra**: dos backtests del mismo `strategy_id`/año con `pnl`
distinto ⇒ el cargador debe persistir ambos o abortar; nunca terminar con exit 0 habiendo
tirado uno. Y la sonda de arriba convertida en test con umbral `divergent == 0`.

---

### G-01 — GRAVE — el plan `legacy-init` sigue siendo un glob sin allowlist y sin gate de revisión

**Fichero:línea**: `scripts/ops/db_migrate.py:43` y `:61`

```python
"legacy-init": tuple(sorted((PROJECT_ROOT / "init-scripts").glob("*.sql"))),
...
REVIEW_GATED_PLANS = frozenset({"fabric-v1"})
```

**Qué está mal**: la remediación cerró el agujero **solo para `fabric-v1`**. Contesto
literalmente la pregunta que me pediste atacar:

- *¿Se puede renombrar un fichero y colarlo?* En `fabric-v1`, **no**: los nombres son una
  tupla literal y `get_migration_files` lanza `FileNotFoundError` si falta alguno — falla
  cerrado. Bien hecho.
- *¿Y en `legacy-init`?* **Sí, trivialmente**: cualquier `.sql` depositado en `init-scripts/`
  entra en el plan, sin allowlist, sin digest y sin `--reviewed-digest` (devuelve `True` en
  la primera línea de `plan_is_authorized`). Hoy el directorio tiene 16 ficheros; el nº 17
  se ejecuta solo.
- *¿El checksum cubre el contenido o solo el nombre?* Cubre el **contenido**
  (`hashlib.<...>(filepath.read_bytes())`). Esa parte de tu afirmación se sostiene — con el
  matiz de G-02.

Además, `database/migrations/001..068` siguen **sin ninguna ruta de aplicación**: ni el
plan `legacy-init` (que apunta a `init-scripts/`) ni `fabric-v1` (070-079) los incluyen.
Mi M-01 anterior está cerrada **solo para 070-079**.

**Evidencia**: ESTÁTICO + `git ls-tree` (16 ficheros en `init-scripts/`, 60 en
`database/migrations/` de los cuales 10 en un plan).

**Remedio**: convertir `legacy-init` también en tupla explícita (o exigir un manifiesto
firmado en `init-scripts/`), y declarar un plan explícito para 001-068 o archivarlas con una
nota de "aplicadas históricamente vía init-scripts".

**Test**: crear un fichero temporal en `init-scripts/` y afirmar que `get_migration_files
("legacy-init")` **no** lo incluye. Hoy sale rojo.

---

### G-02 — GRAVE — el "DRIFT fatal" se apoya en MD5

**Fichero:línea**: `scripts/ops/db_migrate.py:182`
(`return hashlib.md5(filepath.read_bytes()).hexdigest()`)

**Qué está mal**: la detección de drift es la defensa declarada contra "alguien cambió una
migración ya aplicada". MD5 admite colisiones de prefijo elegido en la práctica. El actor
del modelo de amenaza es exactamente quien puede escribir el fichero. El digest de plan sí
usa SHA-256 (`:209`), lo que hace la inconsistencia más llamativa. `_migrations.checksum` ya
es `VARCHAR(64)`, así que SHA-256 hex cabe sin migrar el schema.

**Evidencia**: ESTÁTICO.

**Remedio**: `hashlib.sha256`, más una `080` que reescriba/invalide los checksums MD5
existentes (o un campo `checksum_algo`).

**Test**: afirmar `get_file_checksum` devuelve 64 hex chars y que
`hashlib.sha256(bytes).hexdigest() == get_file_checksum(path)`.

---

### G-03 — GRAVE — `run_migrations` sigue tras un fallo, propagando el envenenamiento de B-02 a todo el plan

**Fichero:línea**: `scripts/ops/db_migrate.py:373-386`

**Qué está mal**: el bucle incrementa `error_count` y **continúa** con la siguiente
migración. Como 071-079 dependen del esquema de 070, un fallo transitorio en 070 hace fallar
las nueve siguientes; con B-02, las diez quedan envenenadas de forma permanente en una sola
corrida.

**Escenario**: caída de conexión durante `070` (`CREATE EXTENSION pgcrypto` en un servidor
sin el paquete, por ejemplo) ⇒ 10 filas `success=FALSE` ⇒ el plan `fabric-v1` es
inaplicable sin `DELETE` manual.

**Evidencia**: ESTÁTICO (la mecánica de B-02 está EJECUTADA).

**Remedio**: `break` al primer error (las migraciones son ordenadas y dependientes), y
mensaje explícito de "plan detenido en X".

**Test**: plan de 3 ficheros, el 1º falla ⇒ afirmar que 2º y 3º **no** se intentan.

---

### G-04 — GRAVE — 079 convierte una violación CRITICAL en un no-op mudo, y la única evidencia que sobrevive vive en una tabla borrable y truncable

**Fichero:línea**: `database/migrations/079_fabric_integrity_remediation.sql:220-256`
(`RETURN NULL`) · `database/migrations/070_fabric_control_plane.sql:148-159`
(`control.incident` sin triggers)

**Verificación de tu afirmación, punto por punto**:

- *"serializa derivation con advisory lock"* — **cierto en el papel**:
  `PERFORM pg_advisory_xact_lock(hashtextextended(NEW.derivation_id, 0))` con
  `derivation_id TEXT` (070:175) resuelve a `hashtextextended(text, bigint)` → `bigint` →
  `pg_advisory_xact_lock(bigint)`. La clave está bien derivada; el lock es de transacción y
  el `SELECT` posterior corre en una función plpgsql VOLATILE, que en READ COMMITTED toma
  snapshot fresco por comando, así que el segundo escritor **sí** ve la fila del primero
  tras esperar. No es un lock decorativo. Dicho eso: **no puedo ejecutarlo** (sin Postgres),
  y una afirmación de serialización sin un test de concurrencia real contra PG no debería
  pasar a VERIFIED. Ver §3.
- *"conserva el incidente sin RAISE rollback"* — **cierto**: `RETURN NULL` en un
  `BEFORE INSERT` suprime la fila sin abortar, y el `INSERT INTO control.incident` previo
  permanece.
- *"artifact/forecast append-only + no-TRUNCATE"* — **cierto y verificado**: matriz completa
  en §2.

**Qué está mal, entonces**: dos cosas que la descripción no cubre.

1. **El llamante no recibe ninguna señal.** Un `INSERT` que registra una derivación
   no-determinista devuelve `INSERT 0 0` y la aplicación sigue creyendo que registró el
   artefacto. Se cambió "aborta y pierde el incidente" por "conserva el incidente y pierde
   el dato, en silencio". Para una violación etiquetada `CRITICAL` eso es fail-open.
2. **`control.incident` es la única evidencia superviviente y no está protegida**: sin
   trigger de `UPDATE/DELETE`, sin `no-TRUNCATE`, sin entrada en `audit_log`. Compárese con
   `metric_event`, `legacy_metric_observation` y `strategy_declaration_event`, que sí los
   recibieron en 070. Un `DELETE FROM control.incident WHERE incident_type =
   'NONDETERMINISTIC_DERIVATION'` borra toda la traza y no deja rastro.

**Escenario de fallo concreto**: dos corridas del mismo `derivation_id` producen hashes
distintos. La segunda se suprime, el pipeline reporta éxito, y quien quiera tapar el
incidente lo borra con una sentencia. El sistema queda idéntico a como estaba antes de 079.

**Evidencia**: ESTÁTICO + matriz ejecutada (§2).

**Remedio**: (a) devolver la señal al llamante sin abortar — p.ej. escribir la fila con un
flag `quarantined = TRUE` en vez de suprimirla, o exponer una función de registro que
devuelva el veredicto; (b) `BEFORE UPDATE OR DELETE` + `BEFORE TRUNCATE` sobre
`control.incident` permitiendo únicamente la transición de `status`/`resolution` vía
función `SECURITY DEFINER`; (c) espejar el incidente en `public.audit_log` como ya hacen
079:193 y 077:764.

**Test/mutación**: insertar dos identidades divergentes ⇒ afirmar (1) que existe la fila de
incidente, (2) que el llamante **sabe** que su fila no entró, y (3) que
`DELETE FROM control.incident` lanza.

---

### G-05 — GRAVE — la supersesión del kill-switch no ordena por severidad: un evento débil **sí** degrada uno fuerte, justo lo contrario de lo que dice el COMMENT

**Fichero:línea**: `database/migrations/077_portfolio_control.sql:460-498`
(tabla + `validate_kill_switch_supersession`) ·
`database/migrations/078_exec_reconciliation.sql:185-232` (`effective_kill_switch` y su
`COMMENT`)

**Qué está mal**: el validador de supersesión comprueba **solo** que el evento superseído
exista y tenga el mismo ámbito de cuenta:

```sql
IF NOT FOUND OR prior_account_id IS DISTINCT FROM NEW.account_id THEN
    RAISE EXCEPTION 'kill-switch supersession must reference an event for the same account scope';
END IF;
```

No hay ninguna comprobación de rango de severidad, de autor, ni de evidencia. Y el resolver
excluye explícitamente todo evento con un sucesor activo. El `COMMENT` de 078:231-232
afirma: *"weaker events never downgrade stronger controls"*.

**Escenario de fallo concreto**: existe `E1 = ACCOUNT_FREEZE` activo (rank 4). Una sola
sentencia

```sql
INSERT INTO portfolio.kill_switch_event
  (account_id, level, actor, reason, supersedes_event_id, effective_at)
VALUES (NULL, 'BLOCK_NEW', 'whoever', 'x', '<E1>', NOW());
```

hace que `effective_kill_switch()` devuelva `BLOCK_NEW`: `E1` queda excluido por tener
sucesor activo. Con `level='CLEAR'` el resultado es rank 0, es decir **kill levantado**. La
tabla es append-only, sí, pero eso solo garantiza que quede la fila; no impide el efecto.
Cruza además con `rules/rbac.md` §4 ("solo `admin` acciona el kill global"): el esquema no
distingue actores.

**Evidencia**: ESTÁTICO (lectura de las tres piezas: CHECK, trigger y resolver).

**Remedio**: en `validate_kill_switch_supersession`, exigir
`rank(NEW.level) >= rank(prior.level)` **salvo** para `CLEAR`, y para `CLEAR` exigir
evidencia (`reason` estructurado + fingerprint `^sha256:…$` + actor en una allowlist), igual
que 079 exige `recovery_evidence_fingerprint` para volver a `NOMINAL`. Alternativamente,
resolver por severidad **ignorando** supersesiones que bajan de rango.

**Test/mutación**: `ACCOUNT_FREEZE` activo + insert de `BLOCK_NEW` que lo supersede ⇒ debe
lanzar; y `effective_kill_switch` debe seguir devolviendo `ACCOUNT_FREEZE`. Hoy devuelve
`BLOCK_NEW`.

---

### G-06 — GRAVE — 074: la unicidad de `broker_fill_id` solo se impone en el camino de corrección, no en el de inserción

**Fichero:línea**: `database/migrations/074_exec_event_sourcing.sql:422`
(`UNIQUE (order_id, broker_fill_id)` sobre valores **base**) · `:559-568` (chequeo contra
`exec.v_effective_fill`) · `:425-452` (`validate_fill_event`, que **no** consulta la vista)

**Qué está mal**: la vista `v_effective_fill` es la verdad económica (la usa
`v_order_state` para `filled_qty`). El trigger de corrección impide que una corrección
colisione con el `broker_fill_id` **efectivo** de otro fill. Pero el trigger de inserción de
`fill_event` no hace esa comprobación, y el `UNIQUE` de tabla solo ve los valores base.

**Escenario de fallo concreto**:

1. `F1` se inserta con `broker_fill_id = 'B1'`.
2. Corrección sobre `F1`: `broker_fill_id 'B1' → 'B2'`. El trigger comprueba la vista, no
   hay conflicto, pasa.
3. Se inserta `F2` con `broker_fill_id = 'B2'`. El `UNIQUE (order_id, broker_fill_id)` mira
   base: `F1.base='B1'`, `F2='B2'` ⇒ **no conflicta**. `validate_fill_event` no mira la vista.

Resultado: dos fills con el **mismo** `broker_fill_id` efectivo dentro de la misma orden.
`v_order_state.filled_qty` los suma a ambos ⇒ doble conteo de un mismo fill del bróker. Esta
es exactamente la clase de M-07 que ya te reporté (fills duplicados) reintroducida por la
capa de correcciones nueva.

**Evidencia**: ESTÁTICO.

**Remedio**: añadir a `validate_fill_event` el mismo `EXISTS` contra `exec.v_effective_fill`
que ya tiene `validate_fill_correction`, o —mejor— materializar la unicidad efectiva con
un índice único sobre una tabla de proyección mantenida por trigger.

**Test/mutación**: la secuencia 1-2-3 de arriba ⇒ el paso 3 debe lanzar
`broker_fill_id cannot collide within order`. Hoy pasa.

---

### G-07 — GRAVE — las dos tablas de *fencing* que evitan doble envío al bróker son las únicas de `exec`/`portfolio` sin protección de `DELETE`/`TRUNCATE`

**Fichero:línea**: `074:160-177` (`exec.order_dispatch`) · `077:500-516`
(`portfolio.kill_switch_action`)

**Qué está mal**: `order_dispatch` es la reclamación con lease que impide que dos
ejecutores manden la misma orden al bróker; `kill_switch_action` es la equivalente para
`cancel_open`/`exit_all`. Ambas son mutables por diseño (llevan estado de claim) — eso está
bien. Lo que no está bien es que sean las **únicas dos** tablas de sus esquemas sin
`BEFORE TRUNCATE` (matriz completa en §2). Sus tablas de eventos hermanas
(`order_dispatch_event`, `kill_switch_action_event`) sí lo tienen, lo que confirma que la
omisión no es una decisión declarada.

**Escenario de fallo concreto**: `TRUNCATE exec.order_dispatch;` (o un `DELETE` puntual)
borra todas las reclamaciones. El siguiente ciclo de ejecución vuelve a reclamar cada orden
—`INSERT … ON CONFLICT (order_id) DO NOTHING` ahora **no** conflicta— y **reenvía órdenes ya
enviadas al bróker**. La bitácora `order_dispatch_event` conserva el rastro, pero el dinero
ya salió dos veces.

**Evidencia**: matriz EJECUTADA (§2): `exec.order_dispatch` → `OPEN/OPEN`,
`portfolio.kill_switch_action` → `OPEN/OPEN`.

**Remedio**: `BEFORE TRUNCATE … block_*_mutation()` en ambas, y `BEFORE DELETE` bloqueando
(el `UPDATE` debe seguir permitido, o mejor: revocar `UPDATE`/`DELETE` a todos los roles y
mutar solo por las funciones `SECURITY DEFINER` que ya existen). Nota adicional: las
funciones `claim_order_dispatch`/`finalize_order_dispatch` son `SECURITY DEFINER` con
`search_path` fijado — eso sí está bien hecho — pero la tabla queda accesible por detrás.

**Test/mutación**: `TRUNCATE exec.order_dispatch` debe lanzar; y tras un `DELETE` bloqueado,
`claim_order_dispatch` sobre una orden ya `COMPLETED` debe seguir devolviendo `NULL`.

---

### G-08 — GRAVE — 074 permite corregir un fill; 075 hace imposible reflejar esa corrección en los hechos

**Fichero:línea**: `075:26` y `075:65-67` (PKs sin versión) · `075:204-211` (immutables) vs
`074:454-468` (`fill_correction_event`)

**Qué está mal**: `fact.position` y `fact.pnl` tienen la PK sobre el grano de negocio, **sin
ninguna columna de versión/revisión**, y triggers que bloquean `UPDATE` **y** `DELETE`. Son
tablas de escritura única e irreversible por grano.

`market.canonical_bar` sí resolvió este mismo problema con `canonical_version INTEGER` +
`UNIQUE (…, canonical_version)`. Los hechos de PnL no.

**Escenario de fallo concreto**: el bróker corrige la cantidad de un fill. 074 lo modela
correctamente (`fill_correction_event`), `v_effective_fill` refleja el nuevo valor, y la PnL
correcta **no se puede escribir**: el `INSERT` choca con la PK y el `UPDATE` lo bloquea el
trigger. El libro de hechos queda permanentemente desincronizado de la ejecución, y la única
salida es desactivar el trigger. Dos migraciones del mismo plan se contradicen.

**Evidencia**: ESTÁTICO. (El backfill lo evidencia de rebote: recurre a
`ON CONFLICT DO NOTHING` precisamente porque no tiene otra salida — ver B-04.)

**Remedio**: `080` aditiva con `fact_version INTEGER NOT NULL DEFAULT 1` en la PK + una
vista `v_current_position`/`v_current_pnl` que tome el máximo, más una columna
`supersedes_fact_version`. Es el patrón que 073 ya usa.

**Test**: fill corregido ⇒ el recomputo de PnL debe persistir como versión 2 y la vista
corriente debe devolverla.

---

### G-09 — GRAVE — la identidad contable de 075 existe como función, pero nada la impone ni la llama

**Fichero:línea**: `075:146-197` (`fact.assert_pnl_identity`)

**Qué está mal**: la función es correcta ahora (mi M-05 anterior — el residual entraba en la
reconstrucción y la identidad era vacía por construcción — **está genuinamente arreglado**:
`reconstructed_gross_pnl` ya no incluye `pnl_residual`, y `identity_error` compara declarado
vs calculado). Pero **ningún trigger la invoca y ningún código la llama**: grep sobre
`src/`, `services/`, `airflow/`, `scripts/` en el árbol sellado ⇒ 0 llamadas. La única
referencia fuera de 075 es mi propia auditoría anterior.

Consecuencia: se puede insertar `gross_pnl = 1000` sin ningún componente y sin residual, y
la base lo acepta sin objeción. La garantía está **declarada, no impuesta** — el patrón que
ya te señalé una vez.

**Evidencia (EJECUTADO)**: `git grep -n "assert_pnl_identity" b18720d -- src services airflow
scripts` ⇒ solo `database/migrations/075_fact_position_pnl.sql:146`.

**Remedio**: `CREATE CONSTRAINT TRIGGER … DEFERRABLE INITIALLY DEFERRED AFTER INSERT ON
fact.pnl` que invoque `assert_pnl_identity` con la tolerancia del grano al COMMIT — el mismo
patrón que 077 ya usa en `trg_target_complete`. Y fijar la tolerancia en un SSOT, no como
parámetro libre del llamante.

**Test/mutación**: `gross=1000`, componentes que suman 200, `residual=0` ⇒ el COMMIT debe
fallar. Hoy commitea.

---

### G-10 — GRAVE — el backfill escribe aunque la población esté incompleta

**Fichero:línea**: `scripts/data/backfill_catalog_facts.py:444-456`

**Qué está mal**: el docstring declara la política anti-supervivencia ("the catalog is the
population"). `inventory()` acumula `missing` (manifests/summaries/trades ausentes o trades
inválidos), y `main()` **primero escribe con `--apply`** y **después** devuelve `2` por
`missing`. La base queda con una población parcial y el operador solo tiene un código de
salida post-commit para enterarse.

**Evidencia**: ESTÁTICO (hoy `missing=0` en el registry real, según mi sonda — es decir,
está latente, no activo).

**Remedio**: `if missing and args.apply and not args.allow_incomplete: raise SystemExit(2)`
**antes** de conectar. Fail-closed por defecto.

**Test**: manifest ausente + `--apply` ⇒ exit 2 y **cero** escrituras.

---

### N-01 — MENOR — dinero en `float` alimentando columnas `NUMERIC`

**Fichero:línea**: `backfill_catalog_facts.py:46-71` (`TradeFact` con `float`), `:320`
(`amount = sum(item.pnl …)`), `:292-297` (`signed_qty` por división en coma flotante),
`:400` (`qty * price` como `market_value`)

Toda la aritmética monetaria es binaria de 64 bits y luego se guarda en `NUMERIC`. El
esquema se toma un trabajo considerable en rechazar `NaN`/`Infinity` en cada columna; el
productor introduce error de redondeo antes de llegar ahí, y ese error entra además en
`derivation_id` (que hashea `__dict__`), haciendo el fingerprint dependiente de la
representación flotante. **Remedio**: `decimal.Decimal` en `TradeFact` y en las sumas.
**Test**: dos ordenaciones distintas de los mismos trades ⇒ mismo `amount` bit a bit.

---

### N-02 — MENOR — `promotion_evidence` es "inmutable" solo *después* de dejar de ser `'{}'`

**Fichero:línea**: `079:44-47` y `079:8-11`

El guard es `IF OLD.promotion_evidence <> '{}' AND NEW … IS DISTINCT FROM OLD`. Mientras
valga `'{}'` se puede escribir cualquier cosa. El backfill de 079:8-11 copia
`transition_evidence`, que para declaraciones CHAMPION anteriores puede ser el `'{}'` por
defecto (070:29): esas filas quedan **CHAMPION con evidencia de promoción escribible a
voluntad, para siempre**, con un `UPDATE` que no cambia ningún estado y por tanto no dispara
ningún gate. **Remedio**: hacer el guard `NEW IS DISTINCT FROM OLD` a secas para
`research_state = 'CHAMPION'`, y en el backfill marcar las filas sin evidencia como
`operational_state='QUARANTINED'` en vez de dejarlas con `'{}'`.

---

### N-03 — MENOR — `transition_evidence` es reescribible por un `UPDATE` no-op, y eso inyecta filas en un ledger append-only

**Fichero:línea**: `070:117-135` + `070:142-146` (trigger AFTER) · `079:173-216`

Un `UPDATE control.strategy_declaration SET transition_evidence = '<lo que sea>'` sin
cambiar estados no atraviesa ningún gate (el bloque de capital exige `capital_tier IS
DISTINCT`, el de transición admite `NEW.research_state = OLD.research_state`) y sin embargo
**dispara el trigger de auditoría**, que escribe una fila en
`control.strategy_declaration_event` (append-only, imborrable) y en `public.audit_log`. Se
puede ensuciar el libro de transiciones con "evidencia" arbitraria sin transición alguna.
El propio backfill 079:8-11 genera N filas espurias con `from == to`. **Remedio**: no
auditar cuando ningún estado cambió, o exigir que `transition_evidence` solo se pueda
escribir junto con una transición válida.

---

### N-04 — MENOR — `quality.quarantine_event` promete en su `COMMENT` lo que su DDL no impone

**Fichero:línea**: `073:34-47` y `073:174-175`

El `COMMENT` dice *"anomalous records are events, never silent UPDATE/clip operations"*. La
tabla es completamente `UPDATE`-able, `DELETE`-able y `TRUNCATE`-able (matriz §2), a
diferencia de `quality.correction_event`, que sí está sellada. Si el flujo de resolución
necesita mutar `status`/`resolution`/`correction_event_id` (razonable), dilo en el comentario
y bloquea el resto de columnas con un trigger de columnas inmutables + `no-TRUNCATE`.

---

### N-05 — MENOR — dos capas de 077 se contradicen sobre si un sleeve puede tener más de un instrumento

**Fichero:línea**: `077:292` (`UNIQUE (target_id, sleeve_id, instrument_id)`) vs
`077:414-421` (`assert_target_complete`)

El `UNIQUE` permite N instrumentos por sleeve. El trigger diferido compara
`ARRAY_AGG(sleeve_id)` (con duplicados) contra los sleeves requeridos (distintos), así que
un segundo instrumento en el mismo sleeve hace fallar el COMMIT con un mensaje que habla de
cobertura, no de cardinalidad. Elige una: o `UNIQUE (target_id, sleeve_id)`, o
`ARRAY_AGG(DISTINCT …)`.

---

### N-06 — MENOR — `portfolio.pretrade_decision` referencia sin integridad

**Fichero:línea**: `077:438-458`

`reconciliation_id UUID` no tiene FK a `exec.reconciliation_event` (entiendo que por evitar
la dependencia circular 077↔078, pero 078 ya añade FKs en sentido contrario, así que podría
añadirse allí). `kill_switch_level TEXT` no tiene `CHECK` contra el mismo dominio que
`kill_switch_event.level`. La decisión pre-trade es la prueba de por qué se dejó pasar una
orden; sus dos referencias clave no están tipadas.

---

### N-07 — MENOR — cero scripts de rollback para 070-079

`git ls-tree b18720d database/migrations | grep rollback` ⇒ **únicamente**
`rollback_033_event_triggers.sql`. Diez migraciones que crean 9 esquemas, 46 tablas y ~30
triggers, y ninguna vía de reversión documentada. Dado B-02 (un fallo es irrecuperable por
CLI), la ausencia de rollback deja de ser higiene y pasa a ser el único plan de recuperación.

---

### N-08 — MENOR — las dos `ADD CONSTRAINT` de 078 validan filas existentes

**Fichero:línea**: `078:143-155`

Sobre una base fresca son inocuas. Sobre una base donde 074 ya vive y hay órdenes, ambas
`ALTER TABLE … ADD CONSTRAINT … FOREIGN KEY` toman `ACCESS EXCLUSIVE` y **validan** todo el
histórico; cualquier `target_id` huérfano aborta 078. Combinado con B-02, ese aborto es
terminal. **Remedio**: `ADD CONSTRAINT … NOT VALID` seguido de `VALIDATE CONSTRAINT` en una
migración posterior, con un informe previo de filas huérfanas.

---

### N-09 — MENOR — los booleanos de evidencia se aceptan por coerción

**Fichero:línea**: `079:70-81`, `:101-108`, `:121-124`, `:141-144`

`(NEW.transition_evidence->>'vote2_approved')::BOOLEAN` acepta `1`, `"t"`, `"yes"`, `"on"`.
El gate de `dsr` sí es estricto (`jsonb_typeof(...) IS DISTINCT FROM 'number'`); los
booleanos no. Aplica el mismo `jsonb_typeof(...) = 'boolean'` a los cuatro.

---

## 2. Matriz de inmutabilidad — verificada, no asumida

Extraída programáticamente del texto de las diez migraciones (EJECUTADO). Solo se listan las
tablas creadas en el plan. `OPEN` = ningún trigger lo bloquea.

| Tabla | UPDATE/DELETE | TRUNCATE |
|---|---|---|
| `control.strategy_declaration` | OPEN *(por diseño: máquina de estados)* | **OPEN** |
| `control.strategy_declaration_event` | blocked | blocked |
| `control.strategy_sleeve` | OPEN | OPEN |
| **`control.incident`** | **OPEN** | **OPEN** ← G-04 |
| `control.artifact_identity` | blocked | blocked |
| `control.metric_event` | blocked | blocked |
| `control.legacy_metric_observation` | blocked | blocked |
| `forecast.forecast_output` / `_score` / `model_horizon_result` / `calibration_result` | blocked | blocked |
| `reference.*` (7 tablas) | OPEN | OPEN |
| `market.raw_bar` / `canonical_bar` / `canonical_bar_source` | blocked | blocked |
| **`quality.quarantine_event`** | **OPEN** | **OPEN** ← N-04 |
| `quality.correction_event` | blocked | blocked |
| `quality.feature_status` | OPEN | OPEN |
| `exec.order_header` / `order_status_event` / `fill_event` / `fill_correction_event` / `order_dispatch_event` | blocked | blocked |
| **`exec.order_dispatch`** | **OPEN** | **OPEN** ← G-07 |
| `fact.position` / `fact.pnl` | blocked | blocked |
| `lineage.node` / `edge` / `strategy_node` / `revision_event` | OPEN | OPEN |
| `portfolio.snapshot` / `snapshot_signal` / `allocation` / `target` / `target_exposure` / `pretrade_decision` / `kill_switch_event` / `kill_switch_action_event` | blocked | blocked |
| **`portfolio.kill_switch_action`** | **OPEN** | **OPEN** ← G-07 |

La cobertura de `no-TRUNCATE` que reclamas (mi M-15 anterior) **es real y amplia**: 26 de 46
tablas selladas en ambos ejes. Los cuatro huecos que marco son los que importan por lo que
protegen, no por el conteo.

---

## 3. Lo que está BIEN hecho (y lo digo sin regatear)

Ocho hallazgos míos anteriores están **genuinamente cerrados**, verificados contra el SQL,
no contra tu descripción:

- **M-02** (guard de reconciliación degradaba el kill): 078 ahora inserta `BLOCK_NEW` con
  `supersedes_event_id` NULL y el resolver es severity-first ⇒ ya no degrada. *(Aunque G-05
  abre el mismo daño por otro vector.)*
- **M-03** (kill global no cubría ninguna cuenta): `WHERE (e.account_id = p_account_id OR
  e.account_id IS NULL)` — arreglado (078:206).
- **M-04** (kill mutable y sin traza): 077 añade immutable + no-TRUNCATE + espejo a
  `public.audit_log` (077:764-799). Arreglado.
- **M-05** (identidad contable vacía): `reconstructed_gross_pnl` ya **no** incluye el
  residual; `identity_error` compara declarado vs calculado (075:107-140). Arreglado
  matemáticamente — falta imponerlo (G-09).
- **M-06** (convención de signo de costes): `CHECK (pnl_component NOT IN
  ('commissions','slippage','financing') OR amount >= 0)` (075:78-81). Arreglado.
- **M-07** (fills duplicados con `broker_fill_id` NULL): ahora `NOT NULL` + `btrim <> ''` +
  `UNIQUE (order_id, broker_fill_id)` (074:418,422). Arreglado en el eje que reporté.
- **M-08** (`v_order_state` ignoraba correcciones): ahora agrega sobre
  `exec.v_effective_fill` (074:705). Arreglado.
- **M-09** (`GRANT UPDATE` sobre `forecast.*`): 071 concede solo `SELECT, INSERT`
  (071:113,116) y 079 sella las cuatro tablas append-only. Arreglado por partida doble.
- **M-12** (`feature_status` sin instrumento): dos índices únicos parciales (073:79-84).
  Arreglado.
- **M-13** (`DECLARED/ZERO → CHAMPION/FULL` en un UPDATE): **ataqué esto expresamente y no
  pude romperlo.** El `CHECK` de tabla (070:32-38) acota el `capital_tier` por
  `research_state`, y 079 cubre los tres saltos que cruzan estados (`PAPER→CHAMPION` fuerza
  `CANARY`, `CANARY→FULL` exige evidencia canary + Vote-2, `CHAMPION→RETIRING` fuerza
  `EXIT_ONLY` + evidencia de retiro). Probé también `RETIRING→WITHDRAWN` con capital
  arbitrario: lo bloquea el `CHECK`. Los gates de capital **se sostienen**.
- **M-17** (target apuntando a allocation de otro snapshot): FK compuesta
  `(allocation_id, snapshot_id, sleeve_id)` (077:296). Arreglado.
- **M-18** (fallback sin `max_age`): `validate_snapshot_contract` exige que las claves de
  `max_age_by_sleeve` sean **exactamente** `required_sleeves` (077:68-74). Arreglado.
- **M-19** (cadena multiplicativa no verificada): `CHECK (final_risk_budget = base * … )` y
  `CHECK (ABS(signed_weight) = final_risk_budget)` en `NUMERIC` exacto (077:212-221).
  Arreglado, y elegante.

Además, revisé y **no encontré nada** en las categorías clásicas de daño:

- Cero `DROP TABLE`, cero `DROP COLUMN`, cero `TRUNCATE` ejecutado, cero `ALTER TYPE` con
  pérdida, cero `SET NOT NULL` sobre tabla poblada sin default.
- Cero `CREATE INDEX CONCURRENTLY` (que habría reventado dentro de la transacción del
  migrador).
- Cero `TIMESTAMP` sin `TZ` en columnas de instante — 100 % `TIMESTAMPTZ`. La regla de oro
  del repo se respeta.
- Cero `GRANT ALL`, cero `TO PUBLIC` con privilegios; 071/074/077 hacen `REVOKE … FROM
  PUBLIC` sobre las funciones sensibles, y las funciones de dispatch son `SECURITY DEFINER`
  **con `SET search_path` fijado** (074:275-276, 331-332, 384-385) — eso es lo correcto y
  casi nadie lo hace.
- `float` donde debería haber `NUMERIC`: **no en el DDL**. Todo lo monetario (`qty`,
  `price`, `commission`, `amount`, `nav_amount`, OHLC) es `NUMERIC`. Los `DOUBLE PRECISION`
  están donde corresponde (métricas, probabilidades, forecasts) y todos con `CHECK` anti
  `NaN`/`±Infinity` — que además funciona en PostgreSQL porque `NaN = NaN` es cierto ahí.
  El `float` es un problema del **productor** Python (N-01), no del esquema.
- El allowlist de `fabric-v1` es una tupla literal con `FileNotFoundError` si falta un
  fichero: renombrar **no** cuela nada en ese plan. Tu afirmación se sostiene.
- El checksum de drift cubre **contenido**, no nombre. Tu afirmación se sostiene (modulo MD5).

---

## 4. Veredicto global

**RECHAZADO para promoción. `IA-R001` no pasa a VERIFIED.**

El trabajo es sustancialmente mejor que la iteración anterior: 14 hallazgos míos cerrados de
verdad, la cobertura `no-TRUNCATE` es real, los gates de capital resisten un ataque directo,
el digest reproduce y el `.gitattributes` cierra la clase CRLF. Nada de esto lo regateo.

Pero **la remediación no se puede desplegar tal cual**:

1. **B-01** deja al `inference_api` sin migrar ni validar en cada arranque, con el fallo
   silenciado por un `|| echo WARNING`. Es una regresión introducida por este mismo commit.
2. **B-02** convierte el primer fallo de migración en un estado terminal sin salida por CLI,
   y **B-03** hace que el gate de revisión no autorice nada: valida el disco contra el disco.
   Los tres juntos significan que la ruta de aplicación que la remediación dice haber creado
   no es utilizable ni auditable.
3. **B-04** hace que el cargador anti-supervivencia descarte en silencio el 63 % de lo que
   calcula sobre el registry real y mezcle versiones de modelo en 1 162 hechos de posición.
   Eso no es un bug de borde: es la afirmación central del script siendo falsa.
4. Y sobre lo que me pediste atacar expresamente: **`079` cumple lo que dice en la letra
   (advisory lock bien derivado, incidente sin RAISE, append-only con triggers reales,
   identidad/evidencia y gates de capital que aguantan) pero falla en el espíritu en dos
   puntos** — el `RETURN NULL` convierte una violación `CRITICAL` en un no-op mudo (G-04), y
   la única evidencia que sobrevive vive en `control.incident`, la tabla que quedó sin
   sellar. Súmale G-05: el `COMMENT` de 078 afirma que un evento débil nunca degrada un
   control fuerte, y una sola sentencia `INSERT` levanta un `ACCOUNT_FREEZE`.

**Condiciones de aceptación (rojo→verde con hash nuevo)**: B-01, B-02, B-03, B-04, G-04,
G-05, G-07 y G-09, cada uno con su test/mutación arriba. G-01/02/03/06/08/10 y los MENOR
pueden ir en una segunda tanda si los declaras como deuda con fecha.

---

## 5. Límites de esta auditoría (honestidad)

- **No hay PostgreSQL disponible** (Docker y DB excluidos por directiva). Todo lo relativo a
  semántica SQL en tiempo de ejecución —el comportamiento del advisory lock bajo
  concurrencia real, el snapshot de una función plpgsql VOLATILE en READ COMMITTED, el orden
  de disparo de triggers, la validación de las FK de 078 sobre datos— está razonado del
  texto y de la documentación de PostgreSQL, **no ejecutado**. Los marco como ESTÁTICO. Por
  el mismo motivo, tu afirmación de serialización en `detect_nondeterministic_derivation`
  la doy por **plausible y bien construida**, pero **no verificada**: exige un test de
  concurrencia real contra PG antes de VERIFIED, y ese test no lo puede sustituir ninguna
  sonda mía.
- Lo que **sí ejecuté** y está transcrito literalmente: el recálculo del digest de plan
  (3 formas), el CLI real de `db_migrate.py` (exit 2), la sonda de envenenamiento sobre la
  función real `run_migration`, la sonda de grano sobre `inventory()` real contra el
  `registry.json` real, la matriz de inmutabilidad y los greps de invocadores/llamadas.
- Fuera de mi alcance por asignación: `src/**` (identidad canónica, allocator/CVXPY, target,
  metrics, execution service), `airflow/`, `config/` y el frontend. Si alguno de mis
  hallazgos de esquema tiene un mitigante en esas capas, no lo he visto.
- Las sondas viven en el scratchpad de sesión, no en el repo. **No he creado, editado ni
  commiteado nada del código auditado.**
