---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - database/migrations/058_billing_webhook_idempotency.sql
  - database/migrations/059_checkout_order_ledger.sql
  - database/migrations/070_fabric_control_plane.sql
  - database/migrations/071_forecast_schema_roles.sql
  - database/migrations/073_market_quality.sql
  - database/migrations/074_exec_event_sourcing.sql
  - database/migrations/075_fact_position_pnl.sql
  - database/migrations/077_portfolio_control.sql
  - database/migrations/078_exec_reconciliation.sql
  - init-scripts/26-restore-features.sh
  - scripts/ops/db_migrate.py
  - usdcop-trading-dashboard/app/api/billing/webhook/route.ts
---

# AUDIT-CLAUDE — Revisión adversarial de las migraciones 058/059 + 070-078 (CODEX)

> **Naturaleza**: auditoría **estática y de solo lectura**. No se arrancó Docker, ni la base
> de datos, ni ningún servicio. **No se ejecutó DDL alguno.** No se editó ni commiteó el
> código auditado. Todo hallazgo se sostiene sobre el texto de los ficheros y el código que
> los referencia; donde no pude construir un escenario de fallo concreto, bajé la severidad
> o lo marqué OBSERVACIÓN.
>
> **Objeto**: 11 ficheros sin commitear (`git status` = `??`), 39 tablas nuevas, 8 esquemas
> nuevos (`control`, `forecast`, `action`, `portfolio`, `exec`, `reference`, `market`,
> `quality`, `fact`, `lineage`), 1 rol, 12 funciones/triggers, 3 vistas.
>
> **Contexto declarado por el operador**: trabajo en curso, revisión temprana deliberada.
> Varios hallazgos son "aún no cableado", y eso es legítimo en WIP — los he separado
> explícitamente de los que serían daño real el día que esto se aplique.

---

## 0. Resumen ejecutivo

**Veredicto: RECHAZAR en el estado actual.** No por la calidad del modelado —que es
notablemente alta y en varios puntos mejor que el DDL heredado del repo— sino por:

1. **M-01 (BLOQUEANTE)**: las migraciones 070-078 **no tienen ninguna ruta de aplicación**.
   Ningún runner del repo las ve. Se pueden mergear y el sistema reportará verde con 36
   tablas inexistentes.
2. **M-02 (BLOQUEANTE)**: el guard de reconciliación de 078 **degrada** un kill-switch más
   fuerte ya activo. Un control de seguridad que debilita otro control de seguridad.
3. **M-03/M-04 (GRAVE)**: el kill-switch **global** no cubre ninguna cuenta, y la tabla de
   kill-switch es **borrable sin traza** — contradice `rbac.md` regla 4.
4. **M-05 (GRAVE)**: la identidad contable de 075 es **vacía por construcción**: `pnl_residual`
   entra en la reconstrucción, así que el residuo siempre da 0 y el assert siempre pasa.
   Es exactamente lo contrario de lo que pide BL-22.
5. **M-09/M-13 (GRAVE)**: `GRANT UPDATE` sobre `forecast.*` permite re-etiquetar el pasado
   (constitución §4), y `control.strategy_declaration` permite saltar de `DECLARED/ZERO` a
   `CHAMPION/FULL` en un solo UPDATE sin traza, sin DSR y sin Vote 2.

**Conteo honesto**: 2 BLOQUEANTES, 11 GRAVES, 8 MENORES, 4 OBSERVACIONES.

### Lo que está BIEN hecho (y merece decirse)

Verificado por inspección exhaustiva, no por impresión:

| Ángulo de ataque | Resultado |
|---|---|
| **Timezone (regla de oro)** | **100% `TIMESTAMPTZ`.** `grep -nE "TIMESTAMP( \|$\|,)"` sobre los 11 ficheros devuelve **cero** columnas naive. Es la primera tanda de DDL del repo que cumple BL-41 de nacimiento. |
| **Dinero en float** | **Ninguno.** `qty`/`price`/`commission` (074), `amount` (075), OHLCV (073), `base/final_risk_budget`/`signed_weight` (077), `amount_cents BIGINT` (059) → todos `NUMERIC`/entero. `DOUBLE PRECISION` queda reservado a forecasts (071) y métricas (070), que es donde corresponde. |
| **Destructividad** | **Cero.** No hay `DROP TABLE`, `DROP COLUMN`, `TRUNCATE`, `ALTER … TYPE`, `DELETE FROM`, ni `NOT NULL` retro-añadido sobre tabla poblada. Ninguna tabla existente se toca. Contra `db-truth-matrix.md`: no hay riesgo de pérdida de datos reales. |
| **Idempotencia** | **Correcta en los 11.** `CREATE TABLE/INDEX/SCHEMA/EXTENSION IF NOT EXISTS`, `DROP TRIGGER IF EXISTS` antes de cada `CREATE TRIGGER`, `DROP CONSTRAINT IF EXISTS` antes de `ADD CONSTRAINT` (073:63-68), `INSERT … ON CONFLICT DO NOTHING` (072:69-76), guarda `DO $$ … IF NOT EXISTS (SELECT 1 FROM pg_roles …)` para `CREATE ROLE` (071:10-16). Re-aplicar produce el mismo estado. |
| **Transaccionalidad** | **Correcta.** Ningún `BEGIN`/`COMMIT` explícito y ningún `CREATE INDEX CONCURRENTLY` (que habría fallado: `db_migrate.py:130` manda el fichero entero por `asyncpg conn.execute`, es decir una transacción implícita única). Un fallo a mitad revierte el fichero completo. |
| **Privilegios** | **Mínimos.** No hay `GRANT ALL`, ni grant a `PUBLIC`, ni rol superusuario, ni credenciales/secretos en el DDL. `forecast_writer` es `NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT`. |
| **Hypertables** | **Ausencia correcta.** BL-44 dice literalmente "NO crear hypertables para tablas pequeñas o vacías". No crearlas es cumplimiento, no omisión. |
| **BL-21 idempotencia de orden** | `exec.order_header.idempotency_key TEXT NOT NULL UNIQUE CHECK (~'^sha256:…')` (074:9) es exactamente lo pedido. |
| **BL-22 convivencia paper/live** | `env` en la PK de `fact.position` y `fact.pnl` (075:23,48). Cumplido. |
| **Invariantes PIT de 073** | Cinco timestamps explícitos + `available_at >= event_time` + `retrieved_at >= available_at` + `high >= GREATEST(open,close,low)`. Es DDL de calidad de datos genuino. |

Nada de lo que sigue contradice esto. El modelado es bueno; el problema es que varios
mecanismos **declaran** una garantía que el DDL **no impone**.

---

## 1. Hallazgos

### M-01 — BLOQUEANTE — Las migraciones 070-078 no tienen ninguna ruta de aplicación

**Fichero:línea**: `init-scripts/26-restore-features.sh:30` · `scripts/ops/db_migrate.py:41` ·
`scripts/migrations/run_all_migrations.sh:192-224` · `docker-compose.compact.yml:69-71`

**Qué está mal**. Hay tres runners en el repo y **ninguno** aplica 070-078:

1. `scripts/ops/db_migrate.py:41` → `MIGRATIONS_DIR = PROJECT_ROOT / "init-scripts"`. Ese
   directorio tiene 14 `.sql` y **ninguna** de las 11 auditadas. El script hace `glob("*.sql")`
   sobre `init-scripts/` y nada más.
2. `scripts/migrations/run_all_migrations.sh:192-224` usa un array `MIGRATIONS` **hardcodeado**
   que termina en `database/migrations/037_experiment_contracts.sql`.
3. `init-scripts/26-restore-features.sh:30` es el único que lee `database/migrations/`, y lo
   hace con un glob explícito:
   ```
   ls /feature-migrations/04[3-9]_*.sql /feature-migrations/05[0-9]_*.sql /feature-migrations/06[0-9]_*.sql
   ```
   **No hay patrón `07[0-9]_`.** 058 y 059 sí entran (por `05[0-9]`); **070-078 no entran por
   ningún lado**.

**Escenario de fallo concreto**. Clon limpio → `make docker-up` → `26-restore-features.sh`
aplica hasta 068 → arranca el stack. Después el operador ejecuta
`python scripts/ops/db_migrate.py --status`, que responde *"No pending migrations. Database is
up to date."* (`db_migrate.py:220`) porque solo mira `init-scripts/`. **36 tablas de las 39
declaradas no existen y el sistema reporta verde.** El primer `INSERT INTO control.metric_event`
de `scripts/data/backfill_catalog_facts.py:105` falla con `relation "control.metric_event" does
not exist`, y `airflow/dags/control_system_health.py` no tiene dónde escribir.

**Agravante (pre-existente, no atribuible a esta tanda pero que la afecta)**:
`docker-compose.compact.yml:69-71` — el stack de uso diario según `CLAUDE.md` (`make compact`) —
**no monta** `./database/migrations:/feature-migrations`. Solo lo hace `docker-compose.yml:72`.
En compact, `26-restore-features.sh:38` imprime *"/feature-migrations not mounted, skipping
migrations"* y **ni siquiera 043-068 se aplican**. Además `26-*` solo corre en
`docker-entrypoint-initdb.d`, es decir **solo en el primer boot de un volumen nuevo**: sobre
una DB existente nada aplica nada, jamás.

**Agravante 2**: `26-restore-features.sh:24,35` usa `ON_ERROR_STOP=0` y
`$PSQL -f "$f" >/dev/null 2>&1 || echo warn`. Un fallo de migración es **invisible**: se
descarta stdout y stderr y se degrada a un `warn` en el log de arranque de postgres.

**Remedio**. (a) Añadir `07[0-9]_*.sql` al glob de `26-restore-features.sh:30` (mínimo
imprescindible); (b) montar `./database/migrations:/feature-migrations:ro` también en
`docker-compose.compact.yml`; (c) apuntar `db_migrate.py:41` a **ambos** directorios y hacerlo
la única autoridad; (d) test de regresión: *"todo `database/migrations/*.sql` aparece en al
menos una ruta de aplicación"*, que falle en CI al añadir un fichero nuevo.

---

### M-02 — BLOQUEANTE — 078: el guard de reconciliación DEGRADA un kill-switch más fuerte

**Fichero:línea**: `database/migrations/078_exec_reconciliation.sql:25-44` (trigger) +
`:46-57` (vista `exec.v_effective_kill_switch`)

**Qué está mal**. El trigger inserta **siempre** `level='BLOCK_NEW'` ante un MISMATCH
(078:32), con `effective_at = NEW.observed_at` (078:34). La vista resuelve el nivel vigente
con `DISTINCT ON (COALESCE(account_id,'*')) … ORDER BY … effective_at DESC, created_at DESC`
(078:47,57): **gana el más reciente, no el más severo.** No existe ningún orden de severidad
entre `CLEAR < BLOCK_NEW < CANCEL_OPEN < EXIT_ALL < ACCOUNT_FREEZE` en ninguna parte del DDL.

**Escenario de fallo concreto**.
```
10:00  admin: INSERT INTO portfolio.kill_switch_event
              (account_id,level,actor,reason,effective_at)
       VALUES ('acct-1','EXIT_ALL','admin','pérdida diaria excedida','2026-07-28 10:00Z');
17:00  EOD reconciliation: INSERT INTO exec.reconciliation_event
              (account_id, phase, observed_at, status, actor)
       VALUES ('acct-1','EOD','2026-07-28 17:00Z','MISMATCH','execution_service');
       -> trigger inserta ('acct-1','BLOCK_NEW', …, effective_at='17:00Z')
17:01  SELECT level FROM exec.v_effective_kill_switch WHERE account_id='acct-1';
       -> 'BLOCK_NEW'
```
El `EXIT_ALL` del operador ha sido **silenciosamente degradado** a `BLOCK_NEW` por el propio
mecanismo de seguridad. Se bloquean aperturas nuevas, pero las posiciones abiertas dejan de
liquidarse. Es un control de seguridad que **debilita** otro control de seguridad, y lo hace
sin error, sin log y sin traza.

**Variante del mismo defecto**: el sentido inverso también rompe. Si la reconciliación EOD
corre a las 17:00 pero con `observed_at` = cierre de mercado (12:55) y el operador emitió un
`CLEAR` a las 14:00, el `BLOCK_NEW` nace con `effective_at` **anterior** al `CLEAR` → la vista
sigue devolviendo `CLEAR` → **el mismatch nunca bloquea nada** (fail-open).

**Remedio**. Introducir un retículo de severidad explícito (tabla `kill_level_rank` o
`CASE` en la vista) y resolver por `MAX(rank)` sobre los eventos vigentes, no por recencia.
Un guard automático nunca debe poder escribir un nivel inferior al vigente: o eleva, o no
escribe. Y `effective_at` de un guard automático debe ser `NOW()`, no `observed_at`.

---

### M-03 — GRAVE — 078: el kill-switch GLOBAL no cubre ninguna cuenta

**Fichero:línea**: `database/migrations/078_exec_reconciliation.sql:46-57` ·
`database/migrations/077_portfolio_control.sql:84` (`account_id TEXT` nullable = "global")

**Qué está mal**. `DISTINCT ON (COALESCE(account_id,'*'))` agrupa el evento global
(`account_id IS NULL`) **en su propio cubo**, separado de cada cuenta. La vista no propaga el
kill global a las cuentas; devuelve una fila con `account_id = NULL` que ningún consumidor
filtrado por cuenta verá.

**Escenario de fallo concreto**.
```sql
-- admin acciona el kill global (rbac.md regla 4)
INSERT INTO portfolio.kill_switch_event (account_id, level, actor, reason, effective_at)
VALUES (NULL, 'EXIT_ALL', 'admin', 'incidente de mercado', NOW());

-- execution_service, tal como el COMMENT de la vista (078:61-62) describe su uso:
SELECT level FROM exec.v_effective_kill_switch WHERE account_id = 'acct-1';
-- -> 0 filas -> el consumidor interpreta "sin kill" -> las órdenes siguen fluyendo
```
`rbac.md` regla 4 nombra el "kill global" como acción de primera clase de `admin`. Aquí es
inoperante salvo que cada consumidor conozca y aplique a mano la semántica del `NULL` — y el
COMMENT de la vista (078:61-62) promete lo contrario: *"Live read model queried directly by
execution_service"*.

**Bonus menor del mismo `COALESCE`**: una cuenta llamada literalmente `'*'` colisiona con el
cubo global.

**Remedio**. La vista debe resolver por cuenta el máximo entre sus eventos y los globales.
Mejor aún, una función `exec.effective_kill_level(p_account_id TEXT) RETURNS TEXT` que sea la
única API de lectura, para que la semántica no dependa de que cada consumidor escriba bien
el `WHERE`.

---

### M-04 — GRAVE — 077: `kill_switch_event` es mutable y borrable, y no llega al `audit_log`

**Fichero:línea**: `database/migrations/077_portfolio_control.sql:82-92`

**Qué está mal**. Es la única tabla de decisión de seguridad de la tanda **sin trigger
append-only**, en un lote donde `control.metric_event` (070:170-173), `market.raw_bar`
(073:119-122) y las cuatro tablas de `exec.*` (074:110-125) sí lo tienen. Además `actor TEXT`
es texto libre sin FK a `sb_users`, y nada escribe en `audit_log`.

**Escenario de fallo concreto**.
```sql
DELETE FROM portfolio.kill_switch_event
 WHERE reason LIKE 'broker reconciliation discrepancy%';
```
`exec.v_effective_kill_switch` vuelve a devolver el `CLEAR` anterior, el trading se reanuda y
**no queda evidencia de que el kill existió**. Idéntico con `UPDATE … SET level='CLEAR'`.

Esto viola directamente `rbac.md` regla 4: *"Vote 2 / promover / kill global: solo `admin`,
siempre al `audit_log` (append-only, trigger que bloquea UPDATE/DELETE)"*. La migración crea
un **segundo mecanismo de kill, paralelo al de `055`**, que no hereda ninguna de esas
propiedades. Lo mismo aplica a `portfolio.target.approved_by TEXT` (077:61): una segunda vía
de aprobación sin FK a `sb_users`, sin comprobación de rol admin y sin entrada en `audit_log`
— es decir, un segundo Vote 2 fuera del gobierno de `approval-gates.md`.

**Remedio**. (a) Trigger append-only sobre `kill_switch_event` (copiar el patrón de
`055_rbac_monetization.sql:52-60`); (b) `actor UUID REFERENCES sb_users(id)` y
`approved_by UUID REFERENCES sb_users(id)`; (c) trigger `AFTER INSERT` que espeje a
`audit_log` con `action='kill_global'|'kill_user'` y `action='vote2_approve'`, para que siga
existiendo **un solo** libro de auditoría.

---

### M-05 — GRAVE — 075: la identidad contable es vacía por construcción

**Fichero:línea**: `database/migrations/075_fact_position_pnl.sql:59-78` (vista) y
`:82-106` (`assert_pnl_identity`)

**Qué está mal**. `reconstructed_gross_pnl` **incluye** `+ pnl_residual` (075:65 y 075:76), y
`absolute_residual = |gross_pnl − reconstructed|`.

**Escenario de fallo concreto**. El productor calcula el residuo como residuo (que es lo que
significa la palabra):
```
pnl_residual := gross_pnl − (beta + timing + carry − commissions − slippage − financing)
```
Sustituyendo en 075:59-66:
```
reconstructed = (beta+timing+carry−comm−slip−fin) + pnl_residual
              = (beta+timing+carry−comm−slip−fin) + gross − (beta+timing+carry−comm−slip−fin)
              = gross
absolute_residual = |gross − gross| = 0     ← SIEMPRE, para CUALQUIER atribución
```
`fact.assert_pnl_identity()` (075:101) compara `0 > tolerance * …` → **siempre pasa**. Una
atribución completamente inventada (beta=0, timing=0, carry=0, residual=todo) supera el test
de identidad contable con residuo 0. El único invariante contable del sistema no puede fallar.

BL-22 pide literalmente *"identidad contable como test de CI (|residual|/|gross| ≤ tol)"*: el
residuo debe **medirse contra una tolerancia**, no absorberse dentro de la suma. La
implementación hace lo contrario de la especificación que la justifica.

**Agravante**: `p_tolerance NUMERIC` (075:87) es un parámetro libre del llamante. No hay
default, ni SSOT, ni cota. `SELECT fact.assert_pnl_identity(…, 1e9)` pasa siempre. Un umbral
de decisión debe ser un prior declarado del operador (constitución §1), no un argumento.

**Remedio**.
```sql
-- reconstrucción SIN el plug
explained := beta + timing + carry − commissions − slippage − financing
absolute_residual := ABS(gross_pnl − explained)
-- y además: el pnl_residual ALMACENADO debe coincidir con el medido
CHECK: ABS(stored_residual − (gross_pnl − explained)) ≤ eps
ASSERT: absolute_residual ≤ tol * GREATEST(ABS(gross_pnl), 1)
```
con `tol` leído de un SSOT versionado, no pasado por el llamante.

---

### M-06 — GRAVE — 075: la convención de signo de los componentes de coste no está declarada ni acotada

**Fichero:línea**: `database/migrations/075_fact_position_pnl.sql:37` (`amount NUMERIC NOT
NULL`, sin CHECK) y `:62-64,73-75` (la vista **resta** `commissions`, `slippage`, `financing`)

**Qué está mal**. La vista asume que los tres componentes de coste se almacenan como
**magnitudes positivas**. Nada lo impone, y nada lo documenta: no hay `CHECK`, no hay
`COMMENT ON COLUMN`, y el nombre `amount` no sugiere signo.

**Escenario de fallo concreto**. Un productor —razonablemente— almacena una comisión como PnL
negativo, que es la convención habitual en una serie de PnL:
```
INSERT INTO fact.pnl (..., pnl_component='commissions', amount=-50, ...)
```
La vista computa `− (−50) = +50`. El PnL reconstruido queda **sobrestimado en 100** respecto a
la intención (−50). Combinado con M-05, el error se traga entero por el plug del residuo y
`assert_pnl_identity` sigue en verde. Dos productores distintos (backtest y live) con
convenciones distintas producirían facts incomparables sin que nada lo detecte.

**Remedio**. `CHECK (pnl_component NOT IN ('commissions','slippage','financing') OR amount >= 0)`
más `COMMENT ON COLUMN fact.pnl.amount` fijando la convención. Alternativa más robusta:
almacenar **todos** los componentes con signo aditivo y que la reconstrucción sea un `SUM()`
puro, con lo que el problema desaparece por diseño.

---

### M-07 — GRAVE — 074: `exec.fill_event` admite fills duplicados si `broker_fill_id` es NULL

**Fichero:línea**: `database/migrations/074_exec_event_sourcing.sql:52` (`broker_fill_id TEXT`
nullable) y `:55` (`UNIQUE (order_id, broker_fill_id)`)

**Qué está mal**. En PostgreSQL, `UNIQUE` es `NULLS DISTINCT` por defecto: **dos filas con
`broker_fill_id IS NULL` no colisionan**. La restricción que aparenta deduplicar fills no
deduplica precisamente en el caso que más se repite.

**Escenario de fallo concreto**. `executor_type='deterministic_simulator'` (074:12) — el
simulador de paper que BL-21 exige que escriba en **las mismas** tablas — no tiene ningún
`broker_fill_id` que reportar, luego insertará NULL. Un retry (timeout del cliente, reintento
de Airflow, re-ejecución de una task) inserta el fill dos veces:
```
fill_event: (order=O1, qty=100, price=3900, broker_fill_id=NULL)
fill_event: (order=O1, qty=100, price=3900, broker_fill_id=NULL)   <- aceptada
```
`exec.v_order_state.filled_qty` (074:96) devuelve **200** para una orden de 100.
`fact.position.qty` derivado queda al doble, y la reconciliación de 078 dispara MISMATCH
permanente (que a su vez degrada el kill-switch, M-02).

La criba de aceptación de BL-21 es *"Retry inyectado ⇒ cero órdenes duplicadas"*. A nivel de
**orden** está cubierta impecablemente por `idempotency_key … UNIQUE` (074:9). A nivel de
**fill** no lo está.

**Remedio**. El stack corre PG15 (`docker-compose.compact.yml:57`
`timescale/timescaledb:latest-pg15`), luego está disponible:
```sql
UNIQUE NULLS NOT DISTINCT (order_id, broker_fill_id)
```
Mejor todavía: `broker_fill_id TEXT NOT NULL` y que el simulador genere un id determinista
(p. ej. `sha256(order_id || seq)`), coherente con el resto de la tanda, que ya usa
fingerprints deterministas en todas partes.

---

### M-08 — GRAVE — 074: la proyección `v_order_state` ignora las correcciones de fill

**Fichero:línea**: `database/migrations/074_exec_event_sourcing.sql:94-101` (agrega solo
`exec.fill_event`) frente a `:58-67` (`exec.fill_correction_event`, que **nadie lee**)

**Qué está mal**. Los fills son inmutables por trigger (074:118-121), así que la **única** vía
para corregir un fill erróneo es insertar un `fill_correction_event`. Pero la vista que define
"el estado de la orden" no consulta esa tabla en absoluto. El COMMENT (074:127-128) afirma
*"Current order state is a projection of immutable events"* — y omite una de las cuatro clases
de evento.

**Escenario de fallo concreto**.
```
T+0  fill_event: (order=O1, qty=100, price=3900, broker_fill_id='B1')
T+1  el bróker corrige: la cantidad real era 10
     fill_correction_event: (fill_id=…, field='qty', old_value='100', new_value='10')
T+1  SELECT filled_qty FROM exec.v_order_state WHERE order_id='O1';  -> 100
```
La posición contable queda 10× inflada de forma permanente, y la reconciliación contra el
bróker (078) reportará MISMATCH indefinidamente sin que exista ningún camino para cerrarlo.

**Agravante de tipado**: `old_value`/`new_value` son `TEXT` (074:63-64), de modo que aunque la
vista se corrigiera tendría que castear texto a `NUMERIC` para recomputar cantidades — sin
`CHECK` que garantice que `field='qty'` implica un valor numérico.

**Remedio**. Modelar la corrección como evento tipado (`qty_delta NUMERIC`,
`price_override NUMERIC`, `voided BOOLEAN`) y plegarla en la vista, de modo que
`filled_qty = SUM(fill.qty) + SUM(correction.qty_delta)`. Un ledger event-sourced cuyo
proyector ignora una clase de evento no es event sourcing.

---

### M-09 — GRAVE — 071: `GRANT UPDATE` sobre `forecast.*` permite re-etiquetar el pasado

**Fichero:línea**: `database/migrations/071_forecast_schema_roles.sql:103` y `:105-106`
(`ALTER DEFAULT PRIVILEGES … GRANT SELECT, INSERT, UPDATE`)

**Qué está mal**. `forecast.forecast_output` es la tabla de predicciones. Es la **única**
tabla de la tanda con vocación inmutable que **no** tiene trigger append-only — compárese con
`market.raw_bar` (073:119-122), `control.metric_event` (070:170-173) y las cuatro de `exec.*`
(074:110-125) — y encima se concede `UPDATE` explícitamente al rol escritor, también para
tablas futuras vía `ALTER DEFAULT PRIVILEGES`.

**Escenario de fallo concreto**. Después de que `target_time` haya pasado y el valor real sea
conocido:
```sql
SET ROLE forecast_writer;
UPDATE forecast.forecast_output SET point = <valor realizado>
 WHERE as_of < NOW() - INTERVAL '1 week';
```
`forecast.forecast_score` y `forecast.model_horizon_result` recomputados sobre esa tabla
arrojan DA ≈ 1.0 y error ≈ 0. No queda ninguna traza: no hay `updated_at`, no hay tabla de
revisiones, no hay hash de fila. La `quant-constitution.md` §4 lo nombra explícitamente en la
capa "Modelos": *"re-ajustar y re-etiquetar el pasado → fit congelado walk-forward; nunca
re-etiquetar"*. Este `GRANT` es la llave para hacerlo.

No hace falta mala fe: basta un job de "recalibración" que haga UPSERT en vez de INSERT.

**Remedio**. `GRANT SELECT, INSERT` (sin UPDATE) + trigger append-only sobre
`forecast_output`. Si hacen falta enmiendas, `forecast_revision_event` siguiendo el patrón que
la propia tanda ya usa en `073` (`quality.correction_event`) y `076`
(`lineage.revision_event`). La coherencia interna del diseño ya existe; solo falta aplicarla
aquí.

**Nota relacionada (MENOR, mismo fichero)**: BL-19 define su criterio de verificación como
`SET ROLE forecast_writer; INSERT INTO action... ⇒ denegado`. La migración crea el esquema
`action` (071:6) **vacío**: no hay ninguna tabla en él, luego el test de aceptación de BL-19
no se puede escribir todavía. Los `REVOKE ALL ON SCHEMA action/portfolio/exec FROM
forecast_writer` (071:108-110) son además no-ops: en un esquema recién creado, `PUBLIC` no
tiene privilegios por defecto (eso solo ocurre con `public`), así que no había nada que
revocar. No es un fallo — es teatro defensivo inocuo —, pero conviene no confundirlo con una
muralla probada.

---

### M-10 — GRAVE — 059: la máquina de transiciones contradice el código que dice servir

**Fichero:línea**: `database/migrations/059_checkout_order_ledger.sql:29-33` (transiciones
legales) frente a `usdcop-trading-dashboard/app/api/billing/webhook/route.ts:65` y `:75`

**Qué está mal**. El trigger admite exactamente: `created→pending`,
`pending→{paid,failed,cancelled,expired}`, `paid→{refunded,charged_back}`. El webhook ejecuta
transiciones que **no** están en esa lista.

**Escenario de fallo concreto A — `paid → cancelled`**:
```
1. usuario paga            -> checkout_orders.status = 'paid'
2. el proveedor envía 'subscription.cancelled'
3. route.ts:70-71 calcula terminal = 'cancelled'
4. route.ts:75  UPDATE checkout_orders SET status='cancelled'
                 WHERE reference=$2 AND status IN ('pending','paid','created')
   -> el WHERE casa la fila 'paid'
   -> trigger 059:32  RAISE EXCEPTION 'illegal checkout order transition: paid -> cancelled'
5. no hay try/catch alrededor de route.ts:75 -> 500 al proveedor
6. route.ts:80 (audit 'plan_payment_failed') NUNCA se ejecuta
```
Cancelación de suscripción → 500 permanente + sin registro de auditoría. Lo mismo con
`paid → failed`.

**Escenario de fallo concreto B — `created → paid` y estado partido**: `route.ts:65` hace
`SET status='paid' … WHERE status IN ('created','pending')`. Si la fila está en `created`
(estado por defecto de 059:10; hoy `cart/checkout/route.ts:75` inserta `'pending'` explícito,
pero el DEFAULT del DDL y el `WHERE` del código dicen que ese camino se contempla), el trigger
lanza. Y como el route **no es transaccional** (cada `query()` es su propia transacción), en
ese punto ya se han commiteado:
- `route.ts:48` la fila de idempotencia en `billing_webhook_events`, y
- `route.ts:62` el `UPDATE sb_users SET entitlements = …` (¡el plan ya está concedido!).

El reintento del proveedor choca con el UNIQUE de 058, cae en el `catch` de `route.ts:52-54` y
responde `{received:true, duplicate:true}` — es decir, **el error se vuelve permanente e
invisible**: pedido colgado en `created`, entitlement concedido, sin `audit_log` de
`plan_change`, y 500 en la primera entrega.

**Remedio**. Decidir cuál es la autoridad y alinear la otra: (a) ampliar la tabla de
transiciones a `created→paid` y `paid→{cancelled,failed,expired}`, o (b) corregir los `WHERE`
del route. **Y, en cualquier caso**, envolver las escrituras del webhook en **una sola
transacción**, para que la fila de idempotencia, el entitlement y el estado del pedido
commiteen o reviertan juntos. Hoy un fallo a mitad deja tres tablas en desacuerdo.

---

### M-11 — GRAVE — 059: el "immutable server-side quote" ni es inmutable ni se lee nunca

**Fichero:línea**: `database/migrations/059_checkout_order_ledger.sql:1` (el comentario que lo
declara), `:37` (`BEFORE UPDATE **OF status**`) · `app/api/billing/webhook/route.ts:38-42`

**(a) No es inmutable.** El trigger es `BEFORE UPDATE OF status` (059:37), es decir **solo se
dispara cuando `status` está en la lista de columnas del UPDATE**. Por tanto:
```sql
UPDATE checkout_orders SET amount_cents = 1, plan = 'auto' WHERE reference = '...';
```
no dispara nada, no valida nada y ni siquiera actualiza `updated_at`. La cotización que la
cabecera del fichero llama "immutable server-side quote" es libremente reescribible, incluida
la cantidad a cobrar y el plan concedido.

**(b) Nunca se lee.** El webhook valida el importe contra un precio **recomputado**
(`planPriceCents(decoded.plan) + addons`, `route.ts:38-42`), no contra
`checkout_orders.amount_cents`. La columna es de solo escritura.

**Escenario de fallo concreto** (combinando ambas mitades — es el caso de uso para el que la
tabla existe):
```
T0  el usuario abre checkout: amount_cents = 79_900 (precio vigente) -> guardado y cobrado
T1  se sube el precio en lib/billing/prices.ts a 99_900
T2  llega el webhook payment.approved con amountInCents = 79_900
    route.ts:38  expected = 99_900   (recomputado con el precio NUEVO)
    route.ts:40  79_900 !== 99_900   -> 400 'amount mismatch'
```
El usuario ha pagado, el pago no se acredita, el pedido no cambia de estado y no hay
`audit_log`. La tabla que existe justo para evitar esto tiene el dato correcto guardado y
nadie lo consulta.

**Remedio**. (a) Trigger `BEFORE UPDATE` (sin `OF status`) que rechace cualquier cambio en
`user_id`, `plan`, `addon_assets`, `amount_cents`, `currency`, `reference`, permitiendo solo
`status`/`updated_at`; (b) que el webhook compare `verification.event.amountInCents` contra
`checkout_orders.amount_cents WHERE reference = $1`, que es la cotización que el servidor
realmente emitió.

---

### M-12 — GRAVE — 073: `quality.feature_status` no puede registrar features sin instrumento

**Fichero:línea**: `database/migrations/073_market_quality.sql:72` (`instrument_id UUID
REFERENCES …`, declarado **nullable**) y `:77` (`PRIMARY KEY (feature_id, instrument_id,
observed_at)`)

**Qué está mal**. En PostgreSQL, incluir una columna en la `PRIMARY KEY` le impone `NOT NULL`
implícitamente. La columna se declara opcional en la línea 72 y se vuelve obligatoria en la
77. El DDL se contradice a sí mismo a cinco líneas de distancia.

**Escenario de fallo concreto**. BL-40 cubre el estado de calidad de **todas** las features,
y buena parte de las del repo no tienen instrumento: `dxy_level`, `vix_level`, `embi_spread`,
`wti_price`, `fed_funds`, todas las macro de `MACRO_DAILY_CLEAN`.
```sql
INSERT INTO quality.feature_status
  (feature_id, instrument_id, status, reason_code, observed_at)
VALUES ('embi_spread', NULL, 'STALE', 'PROVIDER_LAG', NOW());
-- ERROR: null value in column "instrument_id" violates not-null constraint
```
La salida obvia bajo presión es inventar un UUID centinela ("instrumento global"), que es
**exactamente** el colapso de identidades que BL-37/072 existe para prohibir
(`COMMENT ON TABLE reference.instrument`: *"aliases never collapse identity"*).

**Remedio**. PK sustituta (`status_id UUID DEFAULT gen_random_uuid()`) más
`UNIQUE NULLS NOT DISTINCT (feature_id, instrument_id, observed_at)` — disponible en PG15 —,
o dos índices únicos parciales (`WHERE instrument_id IS NULL` / `IS NOT NULL`).

---

### M-13 — GRAVE — 070: `control.strategy_declaration` no tiene máquina de transiciones ni traza

**Fichero:línea**: `database/migrations/070_fabric_control_plane.sql:7-45`

**Qué está mal**. Los tres `CHECK` (070:31-44) restringen con elegancia las **combinaciones**
válidas de `(research_state, capital_tier, dag_declared, surface)` — el COMMENT presume, con
razón, de reducir 96 combinaciones nominales a 26. Pero **ninguno restringe la transición**.
Y la tabla es mutable: no hay trigger append-only, no hay `updated_at`, no hay tabla de
historial, no hay vínculo con `audit_log`.

**Escenario de fallo concreto**.
```sql
-- estado inicial: ('smart_simple_v13','1.0.0','DECLARED','ZERO','NOMINAL', dag_declared=FALSE)
UPDATE control.strategy_declaration
   SET research_state = 'CHAMPION',
       capital_tier   = 'FULL',
       dag_declared   = TRUE
 WHERE strategy_id = 'smart_simple_v13';
-- CHECK 070:34: CHAMPION admite FULL            -> OK
-- CHECK 070:38: dag_declared = (state IN FROZEN,PAPER,CHAMPION,RETIRING) -> TRUE = TRUE -> OK
-- CHECK 070:41: surface='action'                -> OK
-- ACEPTADO
```
Una estrategia pasa de `DECLARED` a capital **FULL** en una sentencia, **sin pasar por
`SCREENED`, `DESIGN_RUN`, `FROZEN` ni `PAPER`**, sin DSR, sin conteo de trials, sin Vote 2 y
sin dejar rastro de que alguna vez estuvo en otro estado. La constitución §2 exige DSR > 0.95
para cualquier claim, `approval-gates.md` exige doble voto, y §5 exige protocolo de retiro
pre-firmado; el plano de control que debería materializar todo eso no registra ninguno de
esos hechos.

**Contraste interno demoledor**: `059_checkout_order_ledger.sql:26-38` **sí** implementa un
trigger de transiciones legales… para un pedido de checkout. El mismo autor sabe cómo hacerlo;
simplemente no se aplicó donde se decide el capital.

**Remedio**. (a) Trigger de transición con el grafo legal de estados
(`DECLARED→SCREENED→DESIGN_RUN→FROZEN→PAPER→CHAMPION→RETIRING→WITHDRAWN`, con las únicas
regresiones que el operador declare); (b) `control.strategy_declaration_event` append-only con
`actor`, `reason` y `at`; (c) espejo a `audit_log`; (d) columnas `n_trials_total` y
`dsr_at_promotion` con `CHECK (research_state <> 'CHAMPION' OR dsr_at_promotion > 0.95)`, para
que el bar constitucional viva en el esquema y no en la prosa.

---

### M-14 — MENOR — 070: el incidente de no-determinismo se pierde con el ROLLBACK

**Fichero:línea**: `database/migrations/070_fabric_control_plane.sql:110-122`

**Qué está mal**. El trigger inserta en `control.incident` (070:110-120) y **acto seguido**
lanza `RAISE EXCEPTION` (070:121-122). Ambas cosas ocurren en la misma transacción: la
excepción la aborta, y el `INSERT` del incidente **se revierte con ella**.

**Escenario de fallo concreto**. Una rederivación no determinista ocurre de verdad. El
`INSERT` en `artifact_identity` se rechaza (correcto, es el objetivo), pero
`SELECT count(*) FROM control.incident WHERE incident_type='NONDETERMINISTIC_DERIVATION'`
devuelve **0**. `airflow/dags/control_system_health.py` no ve nada, no hay alerta, y el único
rastro es el mensaje de error en el log del proceso que insertaba. El mecanismo de detección
detecta y no registra.

**Segundo defecto, menor, del mismo trigger**: la comprobación (070:102-107) lee filas ya
commiteadas. Bajo `READ COMMITTED`, dos transacciones concurrentes que inserten derivaciones
divergentes con el mismo `derivation_id` no se ven mutuamente y **ambas** pasan. La detección
solo funciona en serie.

**Tercer defecto relacionado**: `control.artifact_identity` no tiene trigger append-only
(a diferencia de `control.metric_event`, 070:170-173). Un `DELETE` de la fila anterior seguido
de un `INSERT` divergente pasa el chequeo sin ruido.

**Remedio**. Emitir el incidente fuera de la transacción (`dblink`/`pg_background`), o no
lanzar excepción y marcar la fila como `QUARANTINED` dejando que el guard aguas abajo la
bloquee. Añadir un `UNIQUE (derivation_id, semantic_hash)` da además la garantía a nivel de
índice, inmune a la carrera. Y un trigger append-only sobre `artifact_identity`.

---

### M-15 — MENOR — La inmutabilidad por trigger no cubre `TRUNCATE`

**Fichero:línea**: `070:170-173`, `073:119-122`, `074:110-125`

**Qué está mal**. Todos los triggers append-only son `BEFORE UPDATE OR DELETE … FOR EACH ROW`.
`TRUNCATE` no es `DELETE`: no dispara triggers de fila y vacía la tabla entera.

**Escenario de fallo concreto**. `TRUNCATE market.raw_bar;` borra el registro inmutable de
observaciones del proveedor sin encontrar resistencia, pese al COMMENT (073:124-125) que la
declara inmutable y al mensaje del propio trigger (*"publish a correction_event"*).

**Honestidad**: `audit_log` (`055:57-60`) tiene exactamente la misma laguna, así que esto es
**consistencia con el repo, no una regresión**. Lo reporto porque estas tablas se están
creando ahora y arreglarlo cuesta tres líneas. Igualmente, el dueño de la tabla puede
`ALTER TABLE … DISABLE TRIGGER`; eso solo se cierra si la app deja de conectarse como
propietaria, que es precisamente el alcance de BL-41.

**Remedio**. Añadir `BEFORE TRUNCATE … FOR EACH STATEMENT` a las tablas append-only (y, de
paso, a `audit_log`).

---

### M-16 — MENOR — 074/078: la identidad de instrumento es opcional y reabre la dualidad que 072 cierra

**Fichero:línea**: `074:17-18` (`instrument_id UUID` nullable **junto a** `instrument TEXT NOT
NULL`) · `078:8` (`instrument_id UUID` nullable) · frente a `075:9,29` (`instrument_id UUID
**NOT NULL**`)

**Qué está mal**. El ledger de ejecución permite escribir una orden identificando el
instrumento **solo por texto libre**, mientras que las tablas de hechos exigen el UUID
canónico. Es la dualidad de identidades que 072 (BL-37) existe para abolir, reintroducida en
la capa de ejecución.

**Escenario de fallo concreto**. El adaptador de bróker escribe `instrument='USDCOP'` con
`instrument_id = NULL`. Al construir `fact.position` hay que resolver `'USDCOP'` contra
`reference.instrument_alias`; si el alias no existe (o resuelve a otro instrumento —
`SPX` vs `SPY` vs `ES`, el ejemplo del propio COMMENT en 072:78-79), la fila no se puede
insertar en `fact.position` (que exige `NOT NULL`). Resultado: la reconciliación de 078
compara `internal_qty` (vacío) con `broker_qty` (real) → MISMATCH permanente → y por M-02
eso degrada el kill-switch vigente. Tres defectos encadenados a partir de un `NULL` permitido.

**Remedio**. `instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id)` en
`exec.order_header` y `exec.reconciliation_event`; degradar `instrument TEXT` a columna
desnormalizada de visualización (o eliminarla). La resolución de identidad debe ocurrir **al
escribir**, no al reconciliar.

---

### M-17 — MENOR — 077: un `target` puede apuntar a una `allocation` de OTRO snapshot

**Fichero:línea**: `database/migrations/077_portfolio_control.sql:51-52`

**Qué está mal**. `portfolio.target` referencia `snapshot_id` y `allocation_id` mediante dos
FK **independientes**. `portfolio.allocation` ya referencia su propio `snapshot_id` (077:31).
Nada obliga a que `target.snapshot_id = allocation.snapshot_id`.

**Escenario de fallo concreto**. Un bug de orquestación (o un reintento parcial) crea un
target con `snapshot_id` = snapshot de hoy y `allocation_id` = una allocation calculada sobre
el snapshot de ayer. Los `CHECK` pasan, las dos FK pasan, y `decision_fingerprint` +
`semantic_hash` (077:58-59) sellan criptográficamente **un libro que nunca existió**: pesos de
ayer atribuidos al cutoff de hoy. Es literalmente la "foto movida" que BL-26 pone como caso de
prueba, solo que ocurriendo dentro del propio plano de control.

**Remedio**. `ALTER TABLE portfolio.allocation ADD UNIQUE (allocation_id, snapshot_id);` y en
`portfolio.target` sustituir las dos FK por una compuesta
`FOREIGN KEY (allocation_id, snapshot_id) REFERENCES portfolio.allocation(allocation_id, snapshot_id)`.

---

### M-18 — MENOR — 077: BL-26 prohíbe el fallback sin `max_age`, y el DDL lo permite

**Fichero:línea**: `database/migrations/077_portfolio_control.sql:9,13`

**Qué está mal**. `required_sleeves TEXT[] NOT NULL` y `max_age_by_sleeve JSONB NOT NULL`
están declarados obligatorios, pero `'{}'::text[]` y `'{}'::jsonb` **satisfacen** `NOT NULL`.
Nada relaciona ambas columnas.

**Escenario de fallo concreto**.
```sql
INSERT INTO portfolio.snapshot (cutoff_time, required_sleeves, max_age_by_sleeve,
                                semantic_hash, run_id)
VALUES (NOW(), '{usdcop,xauusd,btcusdt}', '{}'::jsonb, 'abc', 'run-1');
-- ACEPTADO
```
Un snapshot con tres sleeves requeridos y **cero políticas de antigüedad declaradas**. BL-26
dice literalmente: *"política de faltante declarada (USE_LAST_VALID sin max_age PROHIBIDO)"*,
y su criterio de verificación es *"Libro con señal COP de hoy + Oro de ayer ⇒ rechazado sin
políticas declaradas"*. Con este DDL, ese caso se acepta.

**Remedio**. Trigger `BEFORE INSERT` que exija
`required_sleeves <@ ARRAY(SELECT jsonb_object_keys(max_age_by_sleeve))` y
`cardinality(required_sleeves) > 0` (no se puede expresar con `CHECK` porque requiere
subconsulta).

---

### M-19 — MENOR — 077: la cadena multiplicativa del allocator se documenta pero no se verifica

**Fichero:línea**: `database/migrations/077_portfolio_control.sql:35-42`

**Qué está mal**. El esquema modela con cuidado los cinco multiplicadores de gobierno
(`forward`, `liquidity`, `diversification`, `operations`, `drawdown`) con sus rangos, pero
`final_risk_budget` solo tiene `CHECK (>= 0)`: **nada lo ata al producto de la cadena**.
`signed_weight` (077:42) no tiene cota alguna, y no hay ninguna restricción por snapshot sobre
la suma de pesos ni sobre el apalancamiento agregado.

**Escenario de fallo concreto**.
```sql
INSERT INTO portfolio.allocation (..., base_risk_budget=0.10,
       multiplier_forward=1, multiplier_liquidity=1, multiplier_diversification=1,
       multiplier_operations=1, multiplier_drawdown=1,
       final_risk_budget=5.0, signed_weight=12.0, fallback_level=0, ...);
-- ACEPTADO: 50x el presupuesto base, apalancamiento 12x
```
La fila es auditable, tiene todos sus multiplicadores en verde, y su presupuesto final no
guarda ninguna relación con ellos. La cadena de gobierno es decorativa.

**Remedio**. `CHECK (ABS(final_risk_budget − base_risk_budget * multiplier_forward *
multiplier_liquidity * multiplier_diversification * multiplier_operations *
multiplier_drawdown) < 1e-9)`, o mejor, columna generada. Y una cota explícita sobre
`signed_weight` derivada del prior de apalancamiento declarado por el operador.

---

### M-20 — MENOR — 074: `CREATE OR REPLACE VIEW … SELECT h.*` rompe el re-run tras evolucionar el schema

**Fichero:línea**: `database/migrations/074_exec_event_sourcing.sql:78`

**Qué está mal**. `CREATE OR REPLACE VIEW` en PostgreSQL exige que la lista de columnas
resultante sea idéntica en nombre, tipo y **orden**, permitiendo únicamente añadidos al final.
`SELECT h.*` hace que la vista dependa del orden físico de columnas de `exec.order_header`, y
la vista añade siete columnas **después** del `h.*`.

**Escenario de fallo concreto**. Una futura migración `079` hace
`ALTER TABLE exec.order_header ADD COLUMN venue TEXT;`. Al re-ejecutar `074` —cosa que el
runner hace automáticamente si el fichero cambia, `db_migrate.py:216`— el
`CREATE OR REPLACE VIEW` falla con `cannot change name of view column "status_time" to "venue"`.
La migración se marca `success=FALSE` (`db_migrate.py:153-162`) por una razón completamente
benigna, y el operador persigue un fantasma.

**Remedio**. Enumerar las columnas de `h` explícitamente en la vista.

---

### M-21 — MENOR — 059: `billing_events.provider_event_id` no contiene el id del proveedor

**Fichero:línea**: `database/migrations/059_checkout_order_ledger.sql:18-19` ·
`app/api/billing/webhook/route.ts:51` y `:74`

**Qué está mal**. La columna se llama `provider_event_id` y es `UNIQUE`, pero el único
escritor le pasa un valor **sintetizado**: `` `${reference}:${type}` ``. No es el id del
evento del proveedor; es una clave derivada (referencia + tipo). El nombre de la columna miente
sobre su contenido, y el `UNIQUE` protege algo distinto de lo que aparenta.

**Escenario de fallo concreto**. El proveedor emite dos `payment.refunded` distintos para la
misma referencia (reembolso parcial y luego el resto). Ambos generan la misma
`provider_event_id`; el `ON CONFLICT (provider_event_id) DO NOTHING` (route.ts:73) **descarta
en silencio el payload del segundo**. El libro de eventos de facturación —que es la evidencia
frente a una disputa— pierde un reembolso real sin error ni log.

Simétricamente: dos entregas del **mismo** evento con distinto id de proveedor no se
distinguen, así que tampoco cubre el caso de idempotencia real que el nombre promete.

**Adicional**: `order_reference TEXT NOT NULL` (059:19) no tiene FK a `checkout_orders(reference)`,
que **es** `UNIQUE` (059:9) y por tanto referenciable. Se admiten eventos de facturación
huérfanos.

**Remedio**. Guardar el id real del evento del proveedor (el `verifyWebhook` ya recibe el
payload completo) y añadir `FOREIGN KEY (order_reference) REFERENCES checkout_orders(reference)`.

---

### M-22 — OBSERVACIÓN — 058 y 059 son dos libros de idempotencia solapados

`billing_webhook_events` (058, `UNIQUE (reference, event_type)`) se escribe **solo** para
`payment.approved` (route.ts:48); `billing_events` (059, `UNIQUE (provider_event_id)`) se
escribe para todos los tipos (route.ts:49,72). Dos tablas, dos claves, dos semánticas, ninguna
es el SSOT. No causa daño hoy —para `payment.approved` la clave de 058 es adecuada—, pero es
exactamente el patrón "dos escritores para un mismo atributo" que `db-truth-matrix.md` §7
levanta como problema estructural. **Ambas están vacías ahora mismo**: es el momento barato de
colapsarlas en una.

---

### M-23 — OBSERVACIÓN — Huecos de numeración y precedente de colisión

- **No existe `069`**: la serie salta de `068` a `070`. Si es deliberado, conviene un
  `069_RESERVED.md`; si no, hay una migración perdida.
- **058/059 aterrizan por debajo de 060-068**, que ya están escritas (fechas de fichero:
  058/059 = 27-jul; 060-063 = 21-jul). Ambos runners son idempotentes por nombre de fichero,
  y las 11 migraciones son mutuamente independientes y aditivas, así que **el orden no rompe
  nada hoy**. Lo señalo porque el repo ya tiene un precedente real de colisión:
  `056_admin_console_is_test.sql` **y** `056_rbac_dynamic_roles.sql`. Como
  `run_all_migrations.sh` indexa por `basename` y `db_migrate.py` por `filename`, un tercer
  `056_*` con el mismo basename se saltaría en silencio.
- **070-078 no asumen nada de 058/059** (verificado: cero referencias cruzadas). Las
  dependencias internas de la tanda sí son correctas por número: `072` (reference) antes de
  `073`/`074`/`075`/`077`/`078`, y `077` (portfolio.kill_switch_event) antes de que `078` la
  use — que además es *late-binding* plpgsql, luego tolerante.

---

### M-24 — OBSERVACIÓN — Ninguna de las 11 tiene rollback

El repo tiene precedente explícito: `database/migrations/rollback_033_event_triggers.sql`.
Aquí se crean 39 tablas, 10 esquemas, 1 rol, 12 funciones/triggers y 3 vistas **sin ninguna
ruta de retorno**. Para DDL puramente aditivo es defendible, pero `control`, `forecast` y
`exec` van a ser el destino del strangler (BL-31), y cuando eso empiece, deshacer sin un
rollback escrito será caro. Recomendación: un `rollback_070_078_fabric.sql` que haga
`DROP SCHEMA … CASCADE` sobre los esquemas nuevos, escrito **ahora** que están vacíos y su
ejecución es trivialmente segura.

---

### M-25 — OBSERVACIÓN — 072: `reference.asset.annualization` crea un segundo SSOT

**Fichero:línea**: `database/migrations/072_reference_identity.sql:20`

`annualization` es un parámetro de modelado (252 vs 365 vs sesiones COT) cuyo SSOT declarado
en `.claude/rules/00-INDEX.md` es `.claude/specs/assets/_asbuilt-implementation.md`, y cuya
encarnación en código son los `config/assets/*.yaml`. Ponerlo en una columna sin mecanismo de
sincronización ni check de CI invita a que el número que anualiza el backtest y el que anualiza
el reporte diverjan sin que nadie lo note — el tipo de divergencia silenciosa que
`strategy-contract.md` §5 ("nunca compararlos en una misma tabla de ranking") intenta evitar.
**No es un bug**: la tabla está vacía y nadie la escribe. Pero conviene declarar quién manda
antes de la primera fila.

**Al margen — sobre el ángulo 10 (constitución) en general**: revisé si algún esquema
hornea un umbral o una selección de modelado. **Encontré muy poco**, y es un mérito: los
`CHECK` de 077 (`multiplier_* BETWEEN 0 AND 1`, `fallback_level BETWEEN 0 AND 4`) son cotas
estructurales, no priors calibrados; los enums de estado son taxonomía, no parámetros; ningún
fichero contiene una ventana, un lookback ni un umbral de señal. Las dos excepciones reales ya
están reportadas: `p_tolerance` como argumento libre (M-05) y la ausencia de columnas de
trials/DSR en el punto donde se promueve capital (M-13).

---

## 2. Veredicto global

**RECHAZAR.** No apto para aplicar contra ninguna base de datos en el estado actual.

Es un rechazo con matices, y quiero que el matiz quede escrito: **el modelado conceptual de
esta tanda es el mejor DDL del repositorio**. Timestamptz al 100%, `NUMERIC` para todo lo que
es dinero, cero destructividad, idempotencia real fichero a fichero, privilegios mínimos,
fingerprints con `CHECK` de formato, `env` en las PK de hechos, invariantes OHLC y PIT en las
barras. La distancia entre esto y `init-scripts/` es enorme y va en la dirección correcta.

El problema es de otro tipo, y es sistemático: **hay mecanismos que declaran una garantía en
un COMMENT o en un nombre, y no la imponen en el esquema.**

- "immutable server-side quote" que se puede reescribir (M-11a) y que nadie lee (M-11b).
- "identidad contable" que no puede fallar (M-05).
- "projection of immutable events" que ignora una clase de evento (M-08).
- "append-only" en seis tablas pero no en la del kill-switch (M-04).
- Máquina de estados constitucional que no restringe transiciones (M-13), en un fichero cuyo
  hermano (059) sí la implementa para un pedido de checkout.
- Un guard de reconciliación que degrada el kill-switch que debía reforzar (M-02).

Ese patrón es más peligroso que la ausencia de la garantía, porque genera confianza
injustificada: alguien leerá `COMMENT ON TABLE market.raw_bar IS 'immutable…'` y construirá
encima asumiendo que lo es.

**Condiciones mínimas para levantar el rechazo** (por orden):

1. **M-01** — ruta de aplicación real y verificada en CI. Sin esto, lo demás es teórico.
2. **M-02, M-03, M-04** — el kill-switch es el último control antes del dinero. Los tres son
   fail-open o degradación silenciosa.
3. **M-05, M-06** — la identidad contable debe poder fallar; si no, no es un test.
4. **M-07, M-08** — el ledger de ejecución debe deduplicar fills y aplicar correcciones.
5. **M-09, M-13** — quitar `UPDATE` de `forecast.*` y poner máquina de transiciones + traza en
   `strategy_declaration`. Son los dos puntos donde el esquema permite violar la constitución.
6. **M-10, M-11, M-12** — contradicciones DDL↔código y DDL↔DDL que producen 500 o INSERT
   imposible el primer día.

M-14 a M-25 pueden ir en una segunda pasada sin bloquear, salvo M-16, que recomiendo cerrar
antes de escribir la primera fila en `exec.*` (después es una migración de datos, no un
`ALTER`).

---

## 3. Tests que exigiría antes de dar esto por bueno

### 3.1 Tests de aplicación de migración

| # | Test | Criterio de fallo |
|---|---|---|
| T-01 | **Cobertura de runners**: todo fichero de `database/migrations/*.sql` aparece en al menos una ruta de aplicación (glob de `26-restore-features.sh`, array de `run_all_migrations.sh`, o `MIGRATIONS_DIR` de `db_migrate.py`). | Falla al añadir un fichero no cubierto. Es el test que habría cazado M-01 solo. |
| T-02 | **Doble aplicación**: aplicar los 11 ficheros dos veces seguidas sobre una DB limpia. | Cualquier error en la segunda pasada. |
| T-03 | **Diff de esquema tras doble aplicación**: `pg_dump --schema-only` tras 1 pasada vs tras 2. | Diff no vacío. |
| T-04 | **Atomicidad**: inyectar un error sintáctico al final de cada fichero y comprobar que ninguna tabla del fichero queda creada. | Cualquier objeto parcial superviviente. |
| T-05 | **Paridad compact/full**: el esquema resultante de `docker-compose.compact.yml` y de `docker-compose.yml` es idéntico. | Diff no vacío (hoy falla: M-01, agravante). |
| T-06 | **Rollback**: aplicar los 11, ejecutar el rollback, y verificar que el esquema vuelve al estado previo. | Requiere primero escribir el rollback (M-24). |

### 3.2 SQL de verificación (ejecutable contra una DB de test)

```sql
-- V-01 (M-15/M-04/M-09): toda tabla declarada append-only tiene guard de UPDATE, DELETE y TRUNCATE
WITH append_only(t) AS (VALUES
  ('control.metric_event'),('control.artifact_identity'),('market.raw_bar'),
  ('exec.order_header'),('exec.order_status_event'),('exec.fill_event'),
  ('exec.fill_correction_event'),('forecast.forecast_output'),
  ('portfolio.kill_switch_event'),('public.audit_log'))
SELECT t, bool_or(tg.tgtype & 16 > 0) AS has_update,   -- UPDATE
          bool_or(tg.tgtype & 8  > 0) AS has_delete,   -- DELETE
          bool_or(tg.tgtype & 32 > 0) AS has_truncate  -- TRUNCATE
FROM append_only
LEFT JOIN pg_trigger tg ON tg.tgrelid = t::regclass AND NOT tg.tgisinternal
GROUP BY t HAVING NOT (bool_or(tg.tgtype&16>0) AND bool_or(tg.tgtype&8>0) AND bool_or(tg.tgtype&32>0));
-- Esperado: 0 filas.  Hoy: fallan forecast_output, kill_switch_event, artifact_identity (UPDATE/DELETE)
--                          y las 10 (TRUNCATE).

-- V-02 (M-12): ninguna columna declarada nullable acaba dentro de una PRIMARY KEY
SELECT c.table_schema, c.table_name, c.column_name
FROM information_schema.columns c
JOIN information_schema.key_column_usage k USING (table_schema, table_name, column_name)
JOIN information_schema.table_constraints tc USING (constraint_name, table_schema)
WHERE tc.constraint_type='PRIMARY KEY'
  AND c.table_schema IN ('control','forecast','reference','market','quality','exec','fact','lineage','portfolio')
  AND c.is_nullable='YES';
-- Esperado: 0 filas.

-- V-03 (M-09): ningún rol no-propietario tiene UPDATE o DELETE en forecast.*
SELECT grantee, table_name, privilege_type
FROM information_schema.role_table_grants
WHERE table_schema='forecast' AND privilege_type IN ('UPDATE','DELETE','TRUNCATE');
-- Esperado: 0 filas.  Hoy: forecast_writer tiene UPDATE en las 4 tablas.

-- V-04 (M-16): identidad canónica obligatoria en las tablas de ejecución
SELECT table_name, column_name, is_nullable
FROM information_schema.columns
WHERE column_name='instrument_id' AND table_schema IN ('exec','fact','portfolio','market','quality');
-- Esperado: is_nullable='NO' en exec.order_header y exec.reconciliation_event.

-- V-05 (M-01): toda tabla que el código referencia existe realmente
--   (alimentar desde .claude/generated/db-inventory.json: writers/readers != 0 => debe existir)

-- V-06 (M-03): tz — ninguna columna de instante sin zona en los esquemas nuevos
SELECT table_schema, table_name, column_name, data_type
FROM information_schema.columns
WHERE data_type = 'timestamp without time zone'
  AND table_schema IN ('control','forecast','reference','market','quality','exec','fact','lineage','portfolio');
-- Esperado: 0 filas.  Hoy: PASA. Este test debe quedarse para que siga pasando.
```

### 3.3 Tests de trigger / comportamiento (pytest contra DB de test)

| # | Test | Aserción |
|---|---|---|
| T-10 | **Inmutabilidad**: por cada tabla append-only, `UPDATE`, `DELETE` y `TRUNCATE` lanzan excepción. | 3 × 10 casos rojos hoy en `TRUNCATE`. |
| T-11 | **Determinismo (M-14)**: insertar dos artefactos con mismo `derivation_id` y distinto `semantic_hash`; el segundo debe fallar **y** `control.incident` debe contener 1 fila `NONDETERMINISTIC_DERIVATION`. | Hoy: falla la segunda mitad (el incidente se revierte). |
| T-12 | **Determinismo concurrente**: dos sesiones simultáneas insertando derivaciones divergentes. | Exactamente una debe pasar. Hoy pasan las dos. |
| T-13 | **Retry de fill (M-07)**: insertar dos veces el mismo fill sin `broker_fill_id`; `v_order_state.filled_qty` debe ser la cantidad de un solo fill. | Hoy devuelve el doble. Es el test §30.3 que BL-21 ya exige. |
| T-14 | **Corrección de fill (M-08)**: fill 100 → corrección a 10 → `v_order_state.filled_qty = 10`. | Hoy devuelve 100. |
| T-15 | **Identidad contable rompible (M-05)**: insertar una atribución deliberadamente **incorrecta** (gross=1000, componentes que suman 200, residual=0) y comprobar que `assert_pnl_identity(…, 0.01)` **lanza**. | Hoy no lanza. Un test de identidad que no puede fallar no es un test. |
| T-16 | **Signo de costes (M-06)**: `commissions = −50` debe rechazarse en el INSERT. | Hoy se acepta y falsea el PnL. |
| T-17 | **Kill global (M-03)**: kill con `account_id IS NULL` ⇒ `effective_kill_level('acct-1') = 'EXIT_ALL'`. | Hoy: sin nivel. |
| T-18 | **No-degradación (M-02)**: `EXIT_ALL` vigente + MISMATCH de reconciliación ⇒ el nivel efectivo **sigue** siendo `EXIT_ALL`. | Hoy: baja a `BLOCK_NEW`. |
| T-19 | **Mismatch retroactivo (M-02 variante)**: MISMATCH con `observed_at` anterior a un `CLEAR` ⇒ debe bloquear igualmente. | Hoy: fail-open. |
| T-20 | **Transiciones de estrategia (M-13)**: `DECLARED/ZERO → CHAMPION/FULL` en un UPDATE debe rechazarse; toda transición aceptada debe dejar fila en el historial y en `audit_log`. | Hoy: aceptada y sin traza. |
| T-21 | **Checkout ↔ webhook (M-10)**: simular las 4 secuencias reales del proveedor (`approved`, `refunded`, `charged_back`, `cancelled` tras `paid`) contra el trigger de 059. | 2 de 4 lanzan hoy. |
| T-22 | **Inmutabilidad de la cotización (M-11a)**: `UPDATE checkout_orders SET amount_cents=1` debe rechazarse. | Hoy se acepta. |
| T-23 | **La cotización se usa (M-11b)**: webhook con `amountInCents` ≠ `checkout_orders.amount_cents` ⇒ 400; con importe **igual al guardado** pero distinto del precio vigente ⇒ **200**. | Hoy el segundo caso da 400. |
| T-24 | **feature_status sin instrumento (M-12)**: `INSERT (feature_id='embi_spread', instrument_id=NULL, …)` debe aceptarse. | Hoy viola NOT NULL. |
| T-25 | **Coherencia snapshot↔allocation (M-17)**: target con allocation de otro snapshot debe rechazarse. | Hoy se acepta. |
| T-26 | **Política de faltante (M-18)**: snapshot con `required_sleeves` no cubiertos por `max_age_by_sleeve` debe rechazarse. Es el caso de prueba textual de BL-26. | Hoy se acepta. |
| T-27 | **Cadena del allocator (M-19)**: `final_risk_budget` ≠ producto de la cadena debe rechazarse. | Hoy se acepta. |
| T-28 | **Muralla física BL-19 (M-09)**: `SET ROLE forecast_writer; INSERT INTO action.<tabla>` ⇒ denegado; `UPDATE forecast.forecast_output` ⇒ denegado. | El primero no se puede escribir aún (esquema `action` vacío); el segundo pasa hoy y no debería. |

---

## 4. Alcance y límites de esta auditoría (honestidad)

Lo que **no** puedo afirmar y no afirmo:

- **No sé qué hay en la base de datos real.** No se arrancó nada. Todos los cruces de
  "¿la tabla tiene datos?" se hicieron contra `db-truth-matrix.md`, que a su vez declara
  (§0.1) que tampoco tuvo acceso a la DB. Mi conclusión de que **no hay riesgo de pérdida de
  datos** se apoya en un hecho verificable sin DB: las 11 migraciones **no contienen ninguna
  sentencia destructiva** y **no tocan ninguna tabla existente** salvo creando el trigger de
  su propia tabla nueva. Eso es sólido con independencia del contenido de la DB.
- **No ejecuté ninguno de los escenarios de fallo.** Todos están derivados del texto SQL y de
  la semántica documentada de PostgreSQL 15 (versión confirmada en
  `docker-compose.compact.yml:57`). Los que dependen de comportamiento sutil del motor —
  `UNIQUE NULLS DISTINCT` (M-07), `PRIMARY KEY` ⇒ `NOT NULL` implícito (M-12), rollback de
  `INSERT` ante `RAISE EXCEPTION` en el mismo trigger (M-14), `TRUNCATE` no dispara triggers
  de fila (M-15), `BEFORE UPDATE OF col` no dispara si la columna no está en el `SET` (M-11a)
  — están señalados como tales para que se verifiquen empíricamente con T-10..T-24 antes de
  actuar sobre ellos.
- **Orfandad de DDL (ángulo 8)**: verificado por `grep` sobre `airflow/`, `services/`,
  `scripts/`, `src/`, `usdcop-trading-dashboard/`. De las 39 tablas nuevas, **solo
  `control.metric_event` tiene escritor** (`scripts/data/backfill_catalog_facts.py:105`), y
  058/059 tienen lector/escritor en el dashboard. Las 36 restantes son DDL sin cablear. **No
  lo cuento como hallazgo**: `db-truth-matrix.md` §5 ya clasifica explícitamente
  `reference.*`, `market.*`, `quality.*`, `forecast.*`, `control.*` como *"destino recién
  creado y aún no cableado — huérfano por ahora, por diseño"*, y el operador declaró que esto
  es WIP. Lo registro para que quede el número exacto y no se confunda con el legacy retirable
  de BL-36.
- **No revisé** el resto del trabajo de CODEX (código Python/TS asociado, specs, tests). Solo
  los 11 ficheros SQL y el código estrictamente necesario para juzgarlos.
