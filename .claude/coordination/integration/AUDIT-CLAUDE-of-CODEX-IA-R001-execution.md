---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/identity/canonical.py
  - src/identity/fingerprints.py
  - src/execution/events.py
  - src/execution/service.py
  - src/execution/__init__.py
  - src/portfolio/target.py
  - src/lineage/graph.py
  - src/lineage/__init__.py
  - src/market/identity.py
  - src/data_quality/rules.py
  - src/data_quality/ohlcv_validators.py
  - src/data_quality/__init__.py
  - src/orchestration/factories.py
  - src/orchestration/dataset_uri.py
  - src/orchestration/semantic_diff.py
  - scripts/analysis/qlab.py
  - airflow/dags/fabric_factories.py
  - config/assets/fabric_factories.yaml
  - config/quality/market_price_ranges.yaml
  - database/migrations/074_exec_event_sourcing.sql
  - database/migrations/077_portfolio_control.sql
  - services/signalbridge_api/app/services/pretrade.py
  - tests/unit/test_codex_safety_contracts.py
  - tests/unit/test_codex_adversarial_remediations.py
---

# AUDIT-CLAUDE-of-CODEX — IA-R001 · Ejecución, identidad canónica, calidad y orquestación

**Objeto auditado (sellado e inmutable):** `b18720d10a2d84c5217c3800fe421ea23607f921`
— *"[codex] FABRIC contracts and adversarial integrity remediation"* (45 ficheros, +10 418 líneas).

**Método.** El árbol se extrajo con `git archive b18720d | tar -x` a un scratchpad aislado y **todas
las sondas se ejecutaron contra esa copia**, no contra el working tree. Verificado que mi alcance
no ha mutado desde el sello:

```
$ git diff --stat b18720d HEAD -- src/identity src/execution src/lineage src/market \
      src/data_quality src/orchestration scripts/analysis/qlab.py \
      airflow/dags/fabric_factories.py config/assets/fabric_factories.yaml config/quality tests/unit
(vacío)
```

**Alcance:** `src/identity/**`, `CanonicalArtifact`, `src/execution/**` + kill switch,
`src/lineage/**`, `src/market/**`, `src/data_quality/**`, `src/orchestration/**`,
`scripts/analysis/qlab.py`, `airflow/dags/fabric_factories.py`, `config/assets/fabric_factories.yaml`.
Migraciones SQL y portfolio/métricas los cubren los otros dos auditores; aquí solo se tocan en lo
que sostiene una afirmación de ejecución.

**Baseline de la suite sellada** (para que conste que parto de verde):

```
$ python -m pytest tests/unit/test_codex_adversarial_remediations.py \
                   tests/unit/test_codex_safety_contracts.py -q
tests\unit\test_codex_adversarial_remediations.py .......                [ 28%]
tests\unit\test_codex_safety_contracts.py ..................             [100%]
============================= 25 passed in 1.53s ==============================
```

---

## 0. Lo que CODEX cerró de mi auditoría anterior (verificado por ejecución)

Esto no es cortesía: son remediaciones reales y en varios casos mejores de lo que pedí.

| ID previo | Estado | Evidencia ejecutada |
|---|---|---|
| `1.0` y `"1"` producían el mismo hash | **CERRADO** | `C({"x":1.0})=b'{"x":1}'` vs `C({"x":"1"})=b'{"x":"1"}'` → `SAME=False`. Y `1` vs `1.0` → `SAME=True`. El marcador privado `_CanonicalNumber` (`canonical.py:31-41`) es la solución correcta: emite número JSON real sin colisionar con strings. |
| CRLF **dentro** de payloads string (hueco que seguía abierto) | **CERRADO** | `C({"x":"a\r\nb"}) == C({"x":"a\nb"}) == C({"x":"a\rb"})` → todos `b'{"x":"a\\nb"}'` (`canonical.py:116`). |
| Unicode NFC/NFD | **CERRADO, y en claves además de valores** | NFD≡NFC en valor y en clave; además detecta colisión de claves tras normalizar: `C({"é":1,"é":2})` → `CanonicalizationError: NFC key collision` (`canonical.py:137-151`). |
| `-0.0`, `Decimal`, NaN/Inf | **CERRADO** | `-0.0`≡`0.0`≡`Decimal("-0")` → `b'{"x":0}'`; `float('inf')`/`nan` → `CanonicalizationError: NaN and Infinity are forbidden`. `Decimal("1.000")`≡`1`. Escalares numpy y datetimes naive rechazados explícitamente. |
| Efectos del kill se re-disparaban (4 llamadas ⇒ 4 `exit_all`) | **CERRADO** | 4 × `execute_target` con `EXIT_ALL` ⇒ `[('cancel_open',…), ('exit_all',…)]`, **`exit_all count = 1`**. Claim con `action_key = event_id:account:action` + fencing token (`service.py:697-738`). |
| `ACCOUNT_FREEZE` dejaba pasar órdenes | **CERRADO** | Barrido de los 5 niveles: `CLEAR→submits=1`; `BLOCK_NEW/CANCEL_OPEN/EXIT_ALL/ACCOUNT_FREEZE → submits=0` con `PRETRADE_REJECTED:…KILL_SWITCH_ALLOWS`. |
| Clave de idempotencia omitía `env` | **CERRADO** | `replay/paper/canary/live` → 4 hashes distintos (`events.py:73-81`). |
| Barras con `Infinity` aceptadas / instrumento sin rango pasaba | **CERRADO** | `float('inf')`, `'Infinity'`, `nan`, `'-inf'` → `QUARANTINED / bar.numeric`. `usdcop/xauusd/btcusdt/spx500` (sin rango declarado) → `QUARANTINED / bar.unknown_instrument`. |
| Módulos que no importaban | **CERRADO para `src/**`** | 17/17 módulos nuevos de `src/` importan limpio. (Excepción en `scripts/`: ver **C-01**.) |
| Carrera de publicación `CanonicalArtifact` | **CORRECTA** — ver §1 |

### El ataque a la carrera de `CanonicalArtifact.write` NO encontró fallo

Esto es lo que CODEX pidió explícitamente que atacara, y **aguanta**:

```
A2. 8 procesos, MISMO contenido, barrera de sincronización
    8 × OK  |  destino = b'{"n":1}'  |  restos .tmp: []
A3. 8 procesos, contenido DIVERGENTE
    ganadores=1  rechazados=7  total=8
    destino: b'{"n":0}'  |  restos .tmp: []
A4. FS sin hard links (os.link → OSError EPERM simulado)
    write() falló: PermissionError  |  destino creado: False  |  restos .tmp: []
```

Gana **exactamente uno**; los 7 divergentes reciben `CanonicalizationError: refusing divergent
canonical artifact publication`. No hay lectura parcial posible porque el `os.link` ocurre
**después** de `write()+flush()+fsync()` sobre el inodo temporal. **No hay fallback silencioso** si
el FS no soporta enlaces duros: `os.link` sólo captura `FileExistsError`, cualquier otro `OSError`
propaga ⇒ *fail-closed sin ventana*. Es la respuesta correcta a la pregunta que él mismo planteó.

Las migraciones SQL de fencing también son atómicas **en su texto**: `INSERT … ON CONFLICT DO
NOTHING` + `SELECT … FOR UPDATE` + toma de lease vencido (`074:199-274`, `077:529-616`). El
problema no es el SQL: es que nadie lo ejecuta (**C-05**).

---

## 1. Hallazgos

Severidad: **HIGH** = puede perder dinero, perder una orden o dejar el kill switch sin efecto ·
**MEDIUM** = garantía declarada que no se sostiene · **LOW** = corrección/DRY/durabilidad.

---

### C-01 · HIGH · `scripts/analysis/qlab.py:10`

**Qué está mal.** El CLI commiteado importa `src.research.qlab`, y `src/research/` **no existe en el
repositorio**: está en `.gitignore:204` (`research/`). El árbol sellado no lo contiene y `git ls-files
src/research` está vacío. Es exactamente la clase de bug que te encontré antes (`canonical_bytes`
inexistente), reincidente y sin test que lo detecte.

**Escenario de fallo.** Clon limpio del repo → `python scripts/analysis/qlab.py family-declare …`
revienta antes de parsear argumentos. `registries/ledger.jsonl` y `registries/families/*.yaml` **sí**
están trackeados (12 ficheros): el estado del ledger de trials viaja, el código que lo gobierna no.
Cualquier CI, DAG o auditor que intente cobrar un trial en un checkout limpio no puede.

**Evidencia ejecutada.**

```
$ git ls-tree -r --name-only b18720d src/research/
(vacío)
$ git check-ignore -v src/research/qlab.py
.gitignore:204:research/	src/research/qlab.py
$ git ls-files src/research
(vacío)
$ python -c "importlib.util.spec_from_file_location('qlab','scripts/analysis/qlab.py') …"
FAIL qlab -> ModuleNotFoundError No module named 'src.research'
```

**Remedio.** O `!src/research/` negado en `.gitignore` y `src/research/{__init__,qlab,point_in_time}.py`
commiteados, o mover el módulo a un paquete trackeado (`src/governance/qlab.py`). El patrón
`research/` sin ancla `/` del `.gitignore` está capturando un paquete de código de primera clase.

**Test que lo cierra.** Un smoke de importación **de scripts**, no sólo de `src/`:
```python
@pytest.mark.parametrize("path", sorted(Path("scripts").rglob("*.py")))
def test_every_committed_script_imports(path): ...  # exec_module, no sólo compile()
```
Nótese que `py_compile` NO lo detecta: el fallo es en tiempo de import, no de sintaxis.

---

### C-02 · HIGH · `src/execution/events.py:73-81` + `src/portfolio/target.py:151-153`

**Qué está mal.** `order_idempotency_key` se compone de
`{account_id, instrument, target_version, decision_fingerprint, environment, rebalance_cutoff}`.
**No incluye `sleeve_id`, `allocation_id` ni `strategy_id`.** Y `PortfolioTarget.__post_init__` sólo
prohíbe `sleeve_id` duplicado (línea 153) — **permite dos sleeves sobre el mismo instrumento**, que es
el caso canónico de una cartera multi-estrategia (trend y carry sobre USD/COP).

**Escenario de fallo concreto.** Un target con `sleeve-trend` (peso 0.10) y `sleeve-carry` (peso 0.20)
sobre USD/COP genera dos `OrderIntent` con **idéntica** `idempotency_key`. `claim_order` deduplica,
`claim_order_dispatch` devuelve `None` para el segundo, y la segunda exposición se descarta como
`IDEMPOTENT_REPLAY`. **La cartera se ejecuta a la mitad, sin excepción, sin alerta**, y el estado
global que se devuelve es `TARGET_DISPATCH_MIXED` — indistinguible de un mix legítimo.

**Evidencia ejecutada** (ledger fake que deduplica exactamente como el DDL de `074`):

```
I3. Target con DOS sleeves sobre el MISMO instrumento
  PortfolioTarget ACEPTA 2 sleeves sobre el mismo instrumento:
      [('sleeve-carry', 'USD/COP'), ('sleeve-trend', 'USD/COP')]

  status global: TARGET_DISPATCH_MIXED
    {'status': 'SUBMITTED'}                        {'order_id': 'order-1', 'idempotency_key': 'sha256:768a6ac2…'}
    {'status': 'IDEMPOTENT_REPLAY', 'inserted': False} {'order_id': 'order-1', 'idempotency_key': 'sha256:768a6ac2…'}
  ORDENES REALMENTE ENVIADAS AL BROKER: [('sleeve-carry', '20.0', 'sha256:768a6ac27a88c10')]
  exposiciones en el target: 2 | ordenes enviadas: 1
```

Hay además un segundo filo del mismo defecto: `AccountRepository.state_for(executable)` devuelve
`current_qty` **del instrumento**, no del sleeve, así que cada exposición calcula su delta contra la
posición completa. O se netea por instrumento (una orden), o se clavetea por sleeve (dos claves
distintas). Hoy no hace ninguna de las dos.

**Remedio.** Decidir explícitamente el grano y hacerlo cumplir por contrato:
(a) netear en `execute_target` por `instrument_id` antes de construir intents y prohibir el reparto
por sleeve en la capa de orden; **o** (b) añadir `sleeve_id` (y `allocation_id`) al payload de
`order_idempotency_key` y al `execution_fingerprint`. En cualquier caso, `PortfolioTarget` debe
declarar cuál de los dos invariantes sostiene (`UNIQUE(instrument_id)` o clave por sleeve).

**Test que lo cierra.**
```python
def test_two_sleeves_on_one_instrument_do_not_collapse_into_one_order():
    target = _target_with(["sleeve-trend", "sleeve-carry"], instrument="USD/COP")
    result = await service.execute_target(target.target_id)
    assert broker.sent_notional == Decimal("300")   # 0.10+0.20 sobre NAV 1000
    assert "IDEMPOTENT_REPLAY" not in {o["status"] for o in result["orders"]}
```
**Mutación:** quitar `instrument` del payload de la clave debe romper otro test; añadir `sleeve_id`
debe hacer pasar éste.

---

### C-03 · HIGH · `src/execution/service.py:350-361` (+ `680`)

**Qué está mal.** Los efectos del kill switch (`cancel_open`, `exit_all`) se disparan **sólo** desde
`_enforce_switch_side_effects`, que es privado y tiene **un único invocador**: `execute_target`,
línea 361. Y `execute_target` llega ahí **después** de tres puertas que abortan con excepción:
`TARGET_NOT_FOUND` (351), `INVALID_TARGET_CONTRACT` (354) y `TARGET_OUTSIDE_VALIDITY_WINDOW` (356).

Es decir: **el kill switch está condicionado a que exista un target vivo y dentro de su ventana.**
Un kill switch es precisamente el mecanismo que debe actuar cuando el flujo normal no está corriendo.

**Escenario de fallo concreto.** Sábado 03:00. El operador eleva el kill a `EXIT_ALL` porque hay una
posición abierta con el broker y una noticia de gap. No hay `PortfolioTarget` vigente (el último
venció el viernes). Quien llame a `execute_target` recibe `TARGET_OUTSIDE_VALIDITY_WINDOW` y
**`exit_all` nunca se envía al broker**. El evento de kill queda registrado; el efecto no ocurre.
Esto es exactamente la mitad de la pregunta que planteaste ("¿el efecto puede aplicarse sin su
evento, o el evento registrarse sin efecto?"): la primera mitad está cerrada, la segunda no.

**Evidencia ejecutada.**

```
P1. KILL SWITCH EXIT_ALL con target FUERA de ventana de validez
  execute_target raised: ExecutionRejected TARGET_OUTSIDE_VALIDITY_WINDOW
  broker calls: []
  => EXIT_ALL never dispatched: True

P2. KILL SWITCH EXIT_ALL con target INEXISTENTE
  execute_target raised: ExecutionRejected TARGET_NOT_FOUND
  broker calls: [] => EXIT_ALL never dispatched: True
```

```
$ grep -rn "_enforce_switch_side_effects" --include=*.py .
./src/execution/service.py:361   ← único invocador de producción
./src/execution/service.py:680   ← definición
./tests/unit/test_codex_safety_contracts.py:465,466,506,511,516   ← los tests lo llaman en privado
```

Nótese que los propios tests (`test_broker_timeout_is_unknown_and_kill_actions_are_one_shot`,
`test_kill_action_transport_failure_is_recorded_and_safely_retried`) **invocan el método privado
directamente**. Eso es la señal: la única forma de ejercitar el kill switch es saltándose la API
pública, porque la API pública no lo expone.

**Remedio.** Método público `async def enforce_kill_switch(self, account_id, *, now=None)` que
resuelva `kill_switch.effective_state` y aplique efectos **sin tocar targets**, invocable por un DAG
de watchdog y por la ruta admin del kill global. `execute_target` pasa a llamarlo *antes* de resolver
el target (o a delegarlo). Además: elevar el kill debería disparar el efecto por evento, no por
polling desde el path de ejecución.

**Test que lo cierra.**
```python
async def test_exit_all_fires_with_no_target_at_all():
    service = _service(targets=EmptyTargets())          # get_target -> None
    await service.enforce_kill_switch("account-1")
    assert broker.exits == 1
async def test_exit_all_fires_when_the_only_target_is_expired(): ...
```

---

### C-04 · MEDIUM · `src/execution/service.py:524-534` vs `369-383`

**Qué está mal.** La misma clase define **dos defaults opuestos** para `ExecutionControls`:

- `execute_target:369-383` — si no hay `ControlRepository`, construye `ExecutionControls(False×8, PAPER)`:
  **fail-closed**, todo se rechaza. Correcto.
- `_pretrade:524-534` — si `controls is None`, construye `health_nominal=True, operational_nominal=True,
  price_within_collar=True, liquidity_sufficient=True, market_session_open=True, currency_settled=True,
  account_enabled=True, user_trading_enabled=True, trading_mode=target.environment`:
  **fail-OPEN**, y además hace coincidir el modo de trading con el entorno del target por construcción,
  anulando el check `trading_mode_matches_environment`.

Un default permisivo escondido dentro del propio método que ES la puerta de riesgo. Hoy no es
explotable desde `execute_target` (siempre pasa un `controls` no-`None`), pero es una mina: cualquier
llamador futuro, test o refactor que invoque `_pretrade` sin `controls` obtiene permiso total.

**Escenario de fallo.** Alguien añade una ruta "dry-run pretrade" o un endpoint de simulación que
llama `service._pretrade(target=…, account=…, limits=…, reconciliation=…, kill_level=…, now=…)`
sin pasar `controls`. La respuesta dice `allowed=True` con `reason_codes=()` aunque el mercado esté
cerrado, la cuenta deshabilitada y el usuario sin opt-in.

**Evidencia ejecutada.**

```
P5. Default de ExecutionControls: execute_target vs _pretrade
  _pretrade(controls omitido) -> allowed = True | reason_codes = ()
  checks de control: {'health_nominal': True, 'operational_nominal': True, 'account_enabled': True,
                      'user_trading_enabled': True, 'price_collar': True, 'liquidity': True,
                      'market_session': True, 'trading_mode_matches_environment': True}
```

**Remedio.** Borrar el default de `_pretrade` y hacer `controls: ExecutionControls` obligatorio
(sin `| None`). Un solo default, el fail-closed, y que viva en un único sitio
(`ExecutionControls.denied()` como constructor nombrado).

**Test/mutación.** `pytest.raises(TypeError): service._pretrade(**kwargs_sin_controls)`. Mutación de
control: cambiar cualquier `False` del default de `execute_target:369-383` a `True` debe romper un
test; hoy ningún test cubre el default de `_pretrade`.

---

### C-05 · MEDIUM · `tests/unit/test_codex_safety_contracts.py:524-580` + ausencia de `EventLedger` real

**Qué está mal.** El cierre del TOCTOU (dos envíos reales concurrentes) descansa entero en que
`exec.claim_order_dispatch` y `portfolio.claim_kill_switch_action` sean atómicas. Los tests que
"verifican" eso hacen **coincidencia de subcadenas sobre el texto de la migración**:

```python
assert "create or replace function exec.claim_order_dispatch" in sql
assert "for update" in sql
```

Eso pasa aunque el cuerpo de la función sea `RETURN gen_random_uuid();`. Y no existe **ninguna
implementación de producción del `Protocol EventLedger`**: el único código que implementa
`claim_order_dispatch` / `claim_kill_switch_action` en todo el repo son los fakes del propio fichero
de tests. El SQL existe, el Protocol existe, **nada los conecta**.

**Escenario de fallo.** Alguien "optimiza" la función de claim (quita el `FOR UPDATE`, cambia el
`ON CONFLICT DO NOTHING` por `DO UPDATE`, o refactoriza la toma de lease). La suite sigue verde. El
TOCTOU vuelve, con dinero real, y el único guardián es la revisión humana del diff SQL.

**Evidencia ejecutada — mutación.** Sustituí el cuerpo de `exec.claim_order_dispatch` por un no-op
que **siempre concede el claim** (reabriendo exactamente el bug de los dos envíos concurrentes),
dejando intacto el resto del fichero:

```sql
CREATE OR REPLACE FUNCTION exec.claim_order_dispatch(…) RETURNS UUID AS $$
BEGIN
    -- MUTANT: no fencing at all. Every caller wins the claim.
    RETURN pg_catalog.gen_random_uuid();
END; $$ LANGUAGE plpgsql;
```

```
mutant injected; 'for update' still present in file: True
$ python -m pytest tests/unit/test_codex_safety_contracts.py -q
collected 18 items
tests\unit\test_codex_safety_contracts.py ..................             [100%]
============================= 18 passed in 0.94s ==============================
```

**18/18 verde con el fencing destruido.**

```
$ grep -rn "claim_order_dispatch\|claim_kill_switch_action" --include=*.py . | grep -v service.py
./tests/unit/test_codex_safety_contracts.py:261,313   ← sólo fakes de test
```

**Remedio.** (1) Una implementación real `PostgresEventLedger` que llame a las funciones SQL.
(2) Un test de integración con Postgres efímero (testcontainers o el compose ya existente) que corra
**dos claims concurrentes reales** y asserte un solo token. Mientras eso no exista, el enunciado
correcto en la documentación es "fencing **diseñado**", no "fencing **cerrado**".

**Test que lo cierra.**
```python
@pytest.mark.integration
async def test_two_concurrent_claims_yield_exactly_one_token(pg):
    await pg.execute(Path("database/migrations/074_exec_event_sourcing.sql").read_text())
    a, b = await asyncio.gather(claim(pg, order_id, fp), claim(pg, order_id, fp))
    assert len([t for t in (a, b) if t is not None]) == 1
```
Este test **muere** con el mutante de arriba. Los actuales no.

---

### C-06 · MEDIUM · `src/execution/service.py:315` vs `services/signalbridge_api/app/services/execution.py:29`

**Qué está mal.** Existen ahora **dos clases `ExecutionService`** y **dos `PreTradeDecision`** con
formas incompatibles (`allowed: bool` vs `action: PreTradeAction`), y **dos cadenas de checks
pre-trade** independientes. Dos implementaciones del mismo concepto ⇒ una miente.

La regla dura 6 de `.claude/rules/rbac.md` (modo global PAPER por defecto, kill-switch por usuario
`sb_trading_configs.trading_enabled`, caps de notional, **fail-safe error⇒BLOCK**) vive **sólo** en
`services/signalbridge_api/app/services/pretrade.py::PreTradeGate`. El nuevo
`src/execution/service.py::_pretrade` no consulta `settings.trading_mode`, ni `sb_trading_configs`,
ni `user_risk_limits`: delega todo eso a un `ControlRepository` inyectable **del que no existe
ninguna implementación**.

**Estado real hoy:** el módulo nuevo **no está cableado a ninguna ruta ni DAG** —
`grep "from src.execution.service"` sólo devuelve `tests/unit/test_codex_fabric_contracts.py:553`.
Por eso **no hay bypass activo de `PreTradeGate`**: el camino de órdenes en vivo sigue pasando por
`execution.py:221-223 → PreTradeGate(self.db).check(...)`. Lo reporto porque es el bypass
**pre-construido**: en el momento en que alguien conecte un `Broker` real a `ExecutionService`, la
regla 6 deja de aplicarse sin que ningún gate lo note.

**Evidencia ejecutada.**

```
$ grep -rn "ExecutionService" --include=*.py . | grep -v signalbridge
./src/strangler/contracts.py:369:  """Interface to BL-30 (external execution service). NOT implemented here."""
./tests/unit/test_codex_fabric_contracts.py:553,555,564
$ grep -n "pretrade\|PreTradeGate" services/signalbridge_api/app/services/execution.py
221:        from app.services.pretrade import PreTradeGate
223:        decision = await PreTradeGate(self.db).check(
```

**Remedio.** Antes de cablear nada: o el `ControlRepository` de producción se implementa **como
adaptador sobre `PreTradeGate`** (una sola fuente de verdad del modo global y el kill por usuario),
o los dos servicios se fusionan. Renombrar además una de las dos `PreTradeDecision` — dos tipos con
el mismo nombre en el mismo repo es una trampa de import.

**Test que lo cierra.**
```python
def test_no_order_path_bypasses_the_pretrade_gate():
    """Todo módulo que llame a un adaptador de broker debe importar PreTradeGate."""
    for mod in modules_calling(("place_order", "submit", "create_order")):
        assert "PreTradeGate" in mod.read_text(), f"{mod} envía órdenes sin gate"
```

---

### C-07 · MEDIUM · `src/orchestration/factories.py:47-73`

**Qué está mal.** `assert_constitutional` restringe el namespace publicable **sólo** para
`FactoryKind.FORECAST` (líneas 60-66) y `FactoryKind.STRATEGY` (67-73). `DATA` y `BACKFILL` no tienen
ninguna restricción: una fábrica de datos puede publicar en `action://` y `exec://`, que son
precisamente los namespaces económicos que la separación de carriles existe para proteger.

**Escenario de fallo.** Alguien añade a la sección `data:` de `fabric_factories.yaml` una tarea que
publica `exec://usdcop/orders/v1`. El parseo del DAG es verde, `assert_constitutional` calla, y una
fábrica del carril de datos queda autorizada como productora de datasets de ejecución.

**Evidencia ejecutada.**

```
F1. Una DATA factory puede publicar en el namespace ACTION/EXEC?
  ACEPTADO: asset__rogue__data produce ('action://rogue/strategy_output/v1', 'exec://rogue/orders/v1')
```
(Contraste: la regla que sí funciona —
`F2: forecast:// → action://` dentro de un mismo DAG STRATEGY → *rechazado correctamente*.)

**Remedio.** Tabla explícita `KIND → frozenset(schemes_permitidos)` y comprobarla para **los cuatro**
kinds, incluido `BACKFILL`, incluidos los `consumes` (hoy los `consumes` no se validan en absoluto:
`F7` acepta `consumes: [exec://…, control://…]` en una STRATEGY).

**Test que lo cierra.** Parametrizar sobre `FactoryKind` × `ALLOWED_SCHEMES` y asertar la matriz
completa (4×8 = 32 celdas), no sólo las dos filas hoy cubiertas.

---

### C-08 · MEDIUM · `src/orchestration/factories.py:172-190`

**Qué está mal.** El docstring dice *"Return all regular factories after validating **cross-DAG**
dataset edges"*. La comprensión que sigue es:

```python
for spec in specs for source in spec.consumes for target in spec.produces
```

Eso es el producto cartesiano **dentro de cada spec** — exactamente el mismo que ya hizo
`assert_constitutional` línea 54-59. **Ninguna arista productor→consumidor entre DAGs se valida
jamás.** El guard es una duplicación sin efecto con un docstring falso.

Consecuencia práctica: un DAG que **sólo consume** (`produces: []`) obtiene **cero validación**,
porque el cartesiano `consumes × produces` es vacío.

**Escenario de fallo concreto.** Un DAG del carril de acción que consume `forecast://zoo/panel/v1`
y coloca órdenes como efecto lateral de su script (sin declarar `produces`) pasa limpio. La regla
constitucional "el allocator acepta strategy_output, nunca forecast_output" queda burlada declarando
menos, no más. Y hay una segunda vía en dos saltos: `forecast:// → artifact://` está permitido para
STRATEGY (línea 70 incluye `artifact` en el whitelist), y `artifact://` no está restringido para
nadie.

**Evidencia ejecutada.**

```
F6. DAG que SOLO consume forecast:// (sin produces) => 0 validacion
  ACEPTADO: strat__exec_side consumes= ('forecast://zoo/panel/v1',) produces= ()

F7. consumes con namespace arbitrario/no validado
  aceptado: ('forecast://a/b/c', 'exec://x/y/z', 'control://k/i/ll')

F4. FALSO NEGATIVO en dos saltos
  ACEPTADO: forecast://zoo/panel/v1 -> artifact://launder/bundle/v1  (dags: strat__launder, forecast__zoo)
```

**Remedio.** Construir el grafo real: `producer_of[uri] = spec` a partir de `produces`, y para cada
`spec.consumes` resolver el productor y validar la arista `productor.scheme → consumidor.kind`.
Además: rechazar `consumes` de `forecast://` en cualquier spec de kind STRATEGY, con `produces` vacío
o no. Y arreglar el docstring (o borrar la llamada duplicada).

**Test que lo cierra.**
```python
def test_action_dag_cannot_consume_forecast_even_without_produces():
    with pytest.raises(DatasetContractError):
        build_all_specs({"strategies": {"s": {"consumes": ["forecast://z/p/v1"],
                                              "produces": [], "tasks": [...]}}})
def test_forecast_cannot_reach_action_through_an_artifact_hop(): ...
```

---

### C-09 · MEDIUM · `src/orchestration/semantic_diff.py:55-57, 91-96`

**Qué está mal.** `SemanticDiff` se autocontradice. `equal` se calcula por hash canónico (donde
`1 ≡ 1.0`), pero `_first_difference` compara `type(left) is not type(right)` **sobre los tipos Python
crudos**. Resultado: `equal=True` y `first_difference="$.x: type int != float"` a la vez.

**Escenario de fallo concreto — y es el escenario que el propio repo declara.**
`config/assets/fabric_factories.yaml:3-5` dice: *"These DAGs coexist with the legacy asset factory
until a minimum of two consecutive weekly semantic hashes match."* La herramienta que decide ese
corte es `compare_json_files`. El bundle legacy pasa por `json.load` (donde `1` se lee `int`) y el
bundle FABRIC lleva `Decimal`/`float`; el comparador reporta indefinidamente una "primera diferencia"
que no existe. Quien mire `first_difference` nunca corta; quien mire `equal` recibe un informe que
señala una diferencia falsa. Una de las dos respuestas miente, siempre.

**Evidencia ejecutada.**

```
S1. SemanticDiff se contradice a si mismo?
  compare({'x':1},{'x':1.0}) ->
    SemanticDiff(equal=True, left_hash='sha256:5041bf1f…', right_hash='sha256:5041bf1f…',
                 first_difference='$.x: type int != float')
   CONTRADICCION: True
  list vs tuple -> True | $.x: type list != tuple
```

**Remedio.** `_first_difference` debe operar sobre el árbol ya **canonicalizado**
(`canonicalize(value)`), no sobre los tipos Python. O, mínimo: `first_difference=None if equal else …`.

**Test que lo cierra.**
```python
@given(st.recursive(...))
def test_first_difference_is_none_iff_equal(a, b):
    d = compare(a, b)
    assert (d.first_difference is None) == d.equal
```

---

### C-10 · MEDIUM · `src/identity/canonical.py:117-120`

**Qué está mal.** Arreglaste la colisión escalar-sin-etiqueta **para números** (`_CanonicalNumber`)
pero la dejaste abierta **para fechas**: `datetime` y `date` se serializan como strings desnudos, así
que un `datetime` y el string literal con su forma ISO producen bytes idénticos. Es la misma clase de
bug que acabas de cerrar, en el tipo de al lado.

**Escenario de fallo.** `decision_fingerprint(decision_inputs=…)` acepta un `Mapping` libre
(`fingerprints.py:45-58`). Dos decisiones económicamente distintas — una con
`{"as_of": datetime(2026,1,5,tzinfo=utc)}` y otra con `{"as_of": "2026-01-05T00:00:00Z"}` porque vino
de un JSON — comparten fingerprint. Peor con `date`: un campo de texto libre `{"note": "2026-01-05"}`
colisiona con `{"note": date(2026,1,5)}`. En un sistema donde el fingerprint **es** la identidad
económica y alimenta `order_idempotency_key`, dos identidades que colisionan significan que una orden
se descarta como réplica de otra.

**Evidencia ejecutada.**

```
datetime vs su forma string   b'{"t":"2024-01-01T00:00:00Z"}'
                              b'{"t":"2024-01-01T00:00:00Z"}'  SAME=True
date vs su forma string       b'{"d":"2024-01-01"}'
                              b'{"d":"2024-01-01"}'  SAME=True
```

**Remedio.** Mismo patrón que ya aplicaste a los números: un marcador privado
`_CanonicalTagged(tag="ts"|"date", text=…)` que emita un objeto etiquetado
(`{"@ts":"2024-01-01T00:00:00Z"}`) o, si se quiere conservar JSON plano, un prefijo reservado que las
strings de usuario tengan prohibido (y se valide). Lo mismo aplica, con menor riesgo, a
`tuple`≡`list` y `Enum`≡`value` (`canonical.py:105-106, 153-162`), que también son colisiones de tipo
— decisión de diseño defendible, pero debe estar **declarada** en el docstring del contrato.

**Test que lo cierra.**
```python
def test_typed_scalars_never_collide_with_their_string_rendering():
    assert semantic_hash({"t": datetime(2024,1,1,tzinfo=timezone.utc)}) != semantic_hash({"t": "2024-01-01T00:00:00Z"})
    assert semantic_hash({"d": date(2024,1,1)}) != semantic_hash({"d": "2024-01-01"})
```
Es el gemelo exacto del test que ya escribiste en
`test_codex_safety_contracts.py:26` (`semantic_hash({"x":1.0}) != semantic_hash({"x":"1"})`).

---

### C-11 · MEDIUM · `config/assets/fabric_factories.yaml:18`

**Qué está mal.** La fábrica `asset__usdcop__data` declara la tarea
`{id: quality, script: scripts/validation/audit_partial_ohlcv_bars.py}`. **Ese fichero no existe** en
ninguna parte del árbol sellado. `fabric_factories.py::_run` valida la existencia con
`if not path.is_file(): raise FileNotFoundError` **en tiempo de ejecución de la tarea**, no en el
parseo del DAG.

**Escenario de fallo.** El DAG aparece registrado y sano en la UI de Airflow. Cada lunes 05:00 el
run llega a `quality` y muere con `FileNotFoundError`. Como la cadena es lineal
(`previous >> task`, línea 71-72), `data_verify` no corre nunca y el dataset
`asset://usdcop/canonical_bar/1d` no se marca como producido — sin que ningún gate de parseo lo haya
avisado.

**Evidencia ejecutada.**

```
=== scripts referenced by fabric_factories.yaml: exist? ===
  OK      scripts/data/ingest_asset_ohlcv.py
  OK      scripts/data/build_unified_fx_seed.py
  MISSING scripts/validation/audit_partial_ohlcv_bars.py     ← única ausencia de las 16
  OK      scripts/validation/validate_dataset_calendar.py
  … (13 más, todas OK)
$ find . -name "*audit_partial*" -o -name "*partial_ohlcv*"
(vacío)
```

**Remedio.** Crear el script o quitar la tarea; y mover la comprobación de existencia a
**parse time**: `build_all_specs` debe validar que cada `TaskSpec.callable_path` existe bajo
`PROJECT_ROOT`, de forma que un script fantasma tumbe el import del módulo de DAGs (el bloque
`try/except` de `fabric_factories.py:79-84` ya está preparado para reventar visiblemente).

**Test que lo cierra.**
```python
def test_every_fabric_task_script_exists():
    cfg = yaml.safe_load(Path("config/assets/fabric_factories.yaml").read_text())
    for spec in build_all_specs(cfg):
        for t in spec.tasks:
            assert (REPO / t.callable_path).is_file(), f"{spec.dag_id}:{t.task_id} -> {t.callable_path}"
```

---

### C-12 · LOW · `src/identity/canonical.py:214-219`

**Qué está mal.** `CanonicalArtifact.build` calcula **un** digest y lo asigna a los **dos** campos:
`semantic_hash=digest, bytes_hash=digest`. Los dos campos no pueden discrepar jamás, así que
`bytes_hash` no aporta ninguna verificación independiente — es el mismo concepto con dos nombres.
Mientras tanto `src/lineage/graph.py:44-47` los valida como conceptos distintos, lo que sugiere que
aguas abajo se los cree diferentes.

**Escenario de fallo.** Alguien introduce compresión, un footer o un envelope en el artefacto
publicado. `semantic_hash` (del contenido lógico) y `bytes_hash` (de los bytes en disco) deben
divergir en ese momento — pero el código los deriva de la misma variable, así que la divergencia se
vuelve invisible en vez de detectada.

**Evidencia ejecutada.**

```
A1. bytes_hash vs semantic_hash: son el MISMO valor siempre?
  semantic_hash = sha256:d52fc366a693bd8cb045c75e6857fa91d16ae891810fef7abdbe068a381843b9
  bytes_hash    = sha256:d52fc366a693bd8cb045c75e6857fa91d16ae891810fef7abdbe068a381843b9
  identical     = True
  con quantum   identical = True
```

**Remedio.** O `bytes_hash` se calcula sobre los bytes realmente escritos (property que lea el
fichero publicado / se calcule tras cualquier envoltura), o se elimina el campo y `LineageNode` deja
de fingir que son dos.

---

### C-13 · LOW · `src/identity/canonical.py:227-245`

**Qué está mal.** Un crash duro entre la materialización del temporal (línea 228) y el `os.link`
(236) deja un huérfano `.{nombre}.{pid}.{uuid}.tmp`. El `finally` (244-245) sólo protege excepciones,
no `SIGKILL`/OOM/corte de luz. Nadie los recoge.

**Escenario de fallo.** El worker de Airflow es OOM-killed publicando bundles. Cada intento deja un
temporal del tamaño del artefacto en el directorio de publicación; crecen sin límite. No corrompen
nada (empiezan por `.`, así que `glob("*.json")` no los ve) pero llenan el volumen.

**Evidencia ejecutada** (`os.link` reemplazado por `os._exit(9)` para simular SIGKILL):

```
A5. Crash entre materializacion y enlace: huerfanos?
  exit code hijo: 9
  destino publicado: False
  huerfanos que quedaron: ['.x.json.34044.765ec97b969a45eaae4055780ca4b1d8.tmp']
```

**Remedio.** Barrido de temporales con `mtime` mayor a un umbral al inicio de `write()`
(el nombre ya lleva pid y uuid, así que es seguro), o publicar desde un directorio `.tmp/` dedicado
que un job de mantenimiento pueda vaciar.

---

### C-14 · LOW · `src/identity/canonical.py:231-236`

**Qué está mal.** Se hace `os.fsync` del **fichero** (231) pero nunca del **directorio padre**. En
POSIX, un `os.link` no está durable hasta que se sincroniza el directorio: tras un corte de energía
el inodo existe con sus datos, pero la entrada de directorio puede haberse perdido. Para un
publicador content-addressed que promete "publica el inodo completamente escrito", falta el último
eslabón.

**Evidencia ejecutada.**

```
A6. fsync del directorio padre tras publicar?
  'fsync' aparece 1 vez/veces; sobre el DIRECTORIO: False
```

**Remedio.** Tras el `os.link` exitoso: `fd = os.open(destination.parent, os.O_RDONLY); os.fsync(fd);
os.close(fd)` en POSIX (no-op documentado en Windows).

---

### C-15 · LOW · `src/identity/canonical.py:194-196`

**Qué está mal.** El guard de "quantum declarado que no casó con ningún campo numérico" convierte un
**array vacío legítimo** en un error duro de canonicalización. Está incluso consagrado en tu test
`test_schema_quantum_applies_inside_arrays_and_unmatched_paths_fail`, así que es deliberado — pero
la consecuencia es una trampa de disponibilidad.

**Escenario de fallo.** El mismo esquema de quantums (`{"/legs/qty": "0.001"}`) se aplica a todos los
targets. Un target legítimamente plano (`legs: []`, ninguna exposición) **no se puede publicar**:
`CanonicalizationError`. El caso "hoy no operamos nada" es el que más importa registrar.

**Evidencia ejecutada.**

```
== empty array + declared quantum ==
ERR CanonicalizationError field quantum paths matched no numeric field: ['/legs/qty']
```

**Remedio.** Distinguir "path sintácticamente imposible en este esquema" (error de configuración,
detectable una vez contra el esquema) de "path que no casó en esta instancia" (normal). Sugerencia:
validar los quantums contra el esquema en el arranque y hacer que `canonical_json_bytes` sólo falle
si el path **ni siquiera es alcanzable**, o exponer `strict_quantums: bool = True` y ponerlo a
`False` en el path de publicación.

---

### C-16 · LOW · `src/identity/canonical.py:43-48`

**Qué está mal.** `_schema_path` elimina **cualquier** segmento decimal, no sólo los índices de
array. Un objeto con claves numéricas (años, ids, buckets) hereda silenciosamente las reglas de
quantum pensadas para miembros de array. Además `str.isdecimal()` acepta dígitos no-ASCII.

**Evidencia ejecutada.**

```
quantum /a/qty aplicado a la clave de OBJETO '0':
  b'{"a":{"0":{"qty":1.23}}}'         ← el quantum de /a/qty se aplicó a /a/0/qty
clave decimal árabe-índica '٢':
  b'{"a":{"\xd9\xa2":{"qty":1.23}}}'  ← idem
```

**Remedio.** Propagar un flag "este segmento vino de un índice de array" durante el recorrido en vez
de reinferirlo del texto del path.

---

### C-17 · LOW · `src/data_quality/rules.py:95-152` vs `src/data_quality/ohlcv_validators.py` · `src/data_quality/__init__.py`

**Qué está mal.** Hay ahora **dos módulos de calidad OHLCV** con conjuntos de checks **disjuntos** y
sin relación de autoridad declarada:

| | `ohlcv_validators.py` (CTR-DQ-OHLCV-001, mandado por `data-governance.md` §2) | `rules.py` (BL-40, nuevo) |
|---|---|---|
| Día/hora fuera de sesión | ERROR duro (el gate del bug de Gold) | no mira el timestamp |
| Timestamps duplicados | ERROR | no |
| NaN / high<low / integridad OHLC | ERROR | sí (finito, positivo, orden) |
| Rango económico declarado | no | sí (fail-closed si falta) |
| Alias de proveedor | no | sí (biyección) |

`src/data_quality/__init__.py` **no se actualizó**: sigue exportando y documentando sólo
CTR-DQ-OHLCV-001, y `rules.py` tiene **cero consumidores fuera de tests**. Y
`config/quality/market_price_ranges.yaml` declara únicamente `usdmxn` y `usdclp` — **ninguno** de los
cuatro activos de producción del propio `fabric_factories.yaml`.

**Escenario de fallo.** Si `rules.py` se cablea al ingest tal como está: (a) el 100% de las barras de
usdcop/xauusd/btcusdt/spx500 se cuarentena (fail-closed, seguro pero inoperante), y (b) desaparece el
check de día-no-sesión que `data-governance.md` §2 declara ERROR duro.

**Evidencia ejecutada.**

```
instrumento declarado en el YAML? usdcop   accepted=False rule=bar.unknown_instrument
instrumento declarado en el YAML? xauusd   accepted=False rule=bar.unknown_instrument
instrumento declarado en el YAML? btcusdt  accepted=False rule=bar.unknown_instrument
instrumento declarado en el YAML? spx500   accepted=False rule=bar.unknown_instrument
instrumento declarado en el YAML? usdmxn   accepted=True  rule=None

barra en sábado 03:00 UTC con volumen -5:
  QualityDecision(accepted=True, status='VALID', rule_id=None, …)
```

**Remedio.** Declarar cuál es la autoridad. O `rules.py` **compone** `validate_ohlcv_seed` (rango +
alias como capa adicional sobre el gate de sesión), o se fusionan. Y `market_price_ranges.yaml` debe
cubrir los activos reales antes de que el módulo se cablee a nada.

> **Nota fuera del sello (working tree, sin commitear al cierre de esta auditoría).**
> `git diff src/data_quality/__init__.py` muestra que alguien ya está resolviendo la ambigüedad —
> pero **por sustitución, no por composición**: la API del paquete pasa de exportar
> `validate_ohlcv_seed / OHLCVValidationError / ValidationIssue / ERROR / WARN` a exportar sólo
> `QualityDecision, QualityRuleSet`, y el docstring CTR-DQ-OHLCV-001 desaparece. Es decir, el gate
> de día-no-sesión mandado por `data-governance.md` §2 deja de estar en la API pública del paquete.
> Hoy no rompe nada (`grep "from src.data_quality import"` sólo encuentra código vendorizado ajeno,
> los consumidores reales importan el submódulo directo), pero **agrava C-17 en vez de cerrarlo**:
> la lectura natural del `__init__` pasa a ser "la calidad de datos de este proyecto es `rules.py`".
> Reexportar **ambos** conjuntos y dejar el contrato de autoridad escrito en el docstring.

---

### C-18 · LOW · `src/data_quality/rules.py:154-178`

`QualityRuleSet.feature_status` tiene `all_values_identical: bool = False` y `source_enabled: bool =
True` — dos defaults **permisivos** en un módulo cuyo docstring de cabecera dice *"Fail-closed …
quality rules"*. Un llamador que omita el parámetro obtiene `AVAILABLE`:

```
QualityRuleSet.feature_status(feature_id="f", measured=True)
-> QualityDecision(accepted=True, status='AVAILABLE', …)
```

**Remedio.** Hacerlos obligatorios (keyword-only sin default). En un módulo fail-closed, el default
de un check es "no verificado", no "aprobado".

---

### C-19 · LOW · `src/orchestration/dataset_uri.py:33-48`

`DatasetURI.parse` no normaliza la autoridad, así que `asset://USDCOP/…` y `asset://usdcop/…` son dos
datasets **distintos** para Airflow: un productor y un consumidor que difieran en mayúsculas nunca se
enlazan y nadie avisa.

```
asset://USDCOP/canonical_bar/1d | asset://usdcop/canonical_bar/1d | iguales = False
```

Además, la comprobación de traversal opera sobre el path sin decodificar: `asset://a/%2e%2e/b` pasa
(irrelevante para seguridad — son identificadores opacos de Airflow, no rutas de fichero — pero es
otra forma de que dos URIs semánticamente iguales no coincidan).

**Remedio.** `authority.strip().lower()` + `unicodedata.normalize("NFC", …)` en `parse`, y
`unquote()` antes de la comprobación de segmentos.

---

### C-20 · LOW · `src/lineage/__init__.py:1` · `src/execution/__init__.py`

Deuda de coherencia paquete↔implementación:
- `src/lineage/__init__.py` documenta *"Lineage node/edge/revision contracts"* pero **no existe
  ninguna clase de arista**: sólo el enum `EdgeType`. El "grafo" tiene nodos y tipos de arista, sin
  aristas.
- `src/execution/__init__.py` no se actualizó: los nuevos `events`/`service` no están en la API del
  paquete (`grep`: el fichero es una única línea de comentario).
- `src/lineage`, `src/governance`, `src/metrics` y `src/data_quality.rules` tienen **cero
  importadores fuera de tests**. Es código de especificación, y está bien que exista antes que el
  cableado — pero conviene que la doc lo diga, porque hoy se lee como infraestructura activa.

---

### OBS-21 · INFO (NO EJECUTADO) · `airflow/dags/fabric_factories.py:56-72`

`_build` pone `inlets=` en la primera tarea y `outlets=` en la última. En Airflow 2.8 (versión
pineada: `docker/Dockerfile.airflow-ml:5 → apache/airflow:2.8.1-python3.11`), `outlets` **sí** publica
actualizaciones de Dataset, pero `inlets` es **sólo linaje**: la programación dirigida por datasets
requiere `schedule=[Dataset(...)]` a nivel de DAG. Si esto es así, el bloque `consumes:` del YAML es
decorativo: ningún DAG espera realmente a su insumo.

**No he ejecutado Airflow**, así que lo dejo como observación a verificar, no como hallazgo. La
comprobación es barata: `airflow dags list-import-errors` + inspeccionar
`DagModel.schedule_dataset_references` tras cargar el módulo.

Nota menor relacionada: el `try: from airflow.sdk import Asset` (líneas 21-24) prepara Airflow 3,
pero el módulo usa `from airflow.utils.dates import days_ago`, que no existe en Airflow 3. La rama
"Airflow 3" no puede funcionar tal cual. Irrelevante hoy (2.8.1), pero es una compatibilidad
declarada que no es real.

---

## 2. Resumen

| ID | Sev | Fichero:línea | Título |
|---|---|---|---|
| C-01 | HIGH | `scripts/analysis/qlab.py:10` | Importa `src.research`, que está gitignored y ausente del repo |
| C-02 | HIGH | `src/execution/events.py:73` · `src/portfolio/target.py:153` | Dos sleeves en un instrumento colisionan en la clave de idempotencia ⇒ orden perdida en silencio |
| C-03 | HIGH | `src/execution/service.py:350-361` | El kill switch no tiene vía de actuación sin un target vigente |
| C-04 | MED | `src/execution/service.py:524` | Default fail-OPEN de `ExecutionControls` dentro de la propia puerta de riesgo |
| C-05 | MED | `tests/unit/test_codex_safety_contracts.py:524+` | El fencing se "verifica" por subcadenas; mutación destructiva ⇒ 18/18 verde; no hay `EventLedger` real |
| C-06 | MED | `src/execution/service.py:315` | Segundo `ExecutionService`/`PreTradeDecision` que no pasa por `PreTradeGate` (regla dura 6) |
| C-07 | MED | `src/orchestration/factories.py:47-73` | Una fábrica DATA puede publicar en `action://` y `exec://` |
| C-08 | MED | `src/orchestration/factories.py:172-190` | "cross-DAG edges" no valida ninguna arista entre DAGs; `produces: []` ⇒ cero validación |
| C-09 | MED | `src/orchestration/semantic_diff.py:91-96` | `equal=True` con `first_difference` no nulo — y es la herramienta del corte de paralelo |
| C-10 | MED | `src/identity/canonical.py:117-120` | `datetime`/`date` siguen serializándose sin etiqueta ⇒ colisionan con su string ISO |
| C-11 | MED | `config/assets/fabric_factories.yaml:18` | Script inexistente; falla en runtime, no en parseo |
| C-12 | LOW | `src/identity/canonical.py:219` | `bytes_hash` es una copia literal de `semantic_hash` |
| C-13 | LOW | `src/identity/canonical.py:227-245` | Huérfanos `.tmp` tras crash duro, sin recolector |
| C-14 | LOW | `src/identity/canonical.py:231-236` | Falta `fsync` del directorio padre |
| C-15 | LOW | `src/identity/canonical.py:194-196` | Array vacío + quantum declarado ⇒ no se puede publicar |
| C-16 | LOW | `src/identity/canonical.py:43-48` | `_schema_path` borra cualquier segmento decimal, no sólo índices |
| C-17 | LOW | `src/data_quality/rules.py` | Segundo validador OHLCV con checks disjuntos; YAML sin los activos reales |
| C-18 | LOW | `src/data_quality/rules.py:154-178` | Defaults permisivos en módulo fail-closed |
| C-19 | LOW | `src/orchestration/dataset_uri.py:33-48` | Autoridad de URI sin normalizar |
| C-20 | LOW | `src/lineage/__init__.py` · `src/execution/__init__.py` | Doc↔implementación: "edges" que no existen; `__init__` sin actualizar |
| OBS-21 | INFO | `airflow/dags/fabric_factories.py:56-72` | `consumes` probablemente decorativo (NO ejecutado) |

**Veredicto.** No apto para promover el carril de ejecución tal cual: **C-02** y **C-03** son pérdidas
económicas silenciosas y **C-01** rompe un CLI commiteado en cualquier clon limpio. **C-05** es el
hallazgo estructural: la afirmación "TOCTOU cerrado" no está demostrada por ningún test que pueda
fallar.

Dicho eso — y lo digo en serio — la remediación de identidad canónica y de la carrera de publicación
es de calidad alta. El `_CanonicalNumber` es la solución correcta al bug que reporté, no un parche;
la publicación por hard-link resiste el ataque concurrente que me pediste que lanzara, con el
comportamiento fail-closed correcto cuando el FS no coopera; y las 8 remediaciones previas están
cerradas de verdad, verificadas por ejecución, no por declaración. Los hallazgos nuevos son en su
mayoría del mismo tipo que los que ya aceptaste: **contratos declarados cuyo cableado o cuya prueba
todavía no existe**.

---

## Apéndice · Reproducción

Todas las sondas viven en el scratchpad de sesión y operan sobre la extracción sellada:

```bash
git archive b18720d10a2d84c5217c3800fe421ea23607f921 | tar -x -C <scratch>/sealed
cd <scratch>/sealed
python ../probe_kill.py        # C-03, C-04 + verificación de kill re-fire y ACCOUNT_FREEZE
python ../probe_idem.py        # C-02 + verificación de env en la clave
python ../probe_artifact.py .  # C-12..C-16 + carrera multiproceso (usa probe_artifact_child.py)
# canonical / data-quality / factories / semantic-diff: heredocs inline documentados arriba
```

Mutación de C-05 (destruye el fencing, la suite sigue verde):

```bash
# sustituir el cuerpo de exec.claim_order_dispatch por `RETURN pg_catalog.gen_random_uuid();`
python -m pytest tests/unit/test_codex_safety_contracts.py -q   # 18 passed
```
