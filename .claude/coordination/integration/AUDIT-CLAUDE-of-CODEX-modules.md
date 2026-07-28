---
kind: review
status: ACTIVE
version: 1.0.0
last_verified: 2026-07-28
supersedes: []
code_anchors:
  - src/identity/canonical.py
  - src/identity/fingerprints.py
  - src/metrics/engine.py
  - config/metrics/catalog.yaml
  - src/market/identity.py
  - src/data_quality/rules.py
  - src/execution/events.py
  - src/execution/service.py
  - src/governance/declaration.py
  - services/common/metrics.py
  - tests/unit/test_codex_fabric_contracts.py
---

# AUDIT — Claude (adversarial) sobre los módulos sin commitear de CODEX

- **Auditor**: Claude (revisor independiente, rol adversarial)
- **Objeto**: trabajo en curso sin commitear de CODEX — `src/identity/`, `src/metrics/` +
  `config/metrics/catalog.yaml`, `src/market/identity.py`, `src/governance/declaration.py`,
  `src/data_quality/rules.py`, `src/execution/{events,service}.py`
- **Método**: auditoría estática + **scripts de ataque ejecutados** (fuera del repo, en scratchpad).
  Sin Docker, sin DB, sin servicios. Sin ediciones al código auditado, sin commits.
- **Veredicto global**: **RECHAZADO para merge** — 4 BLOQUEANTES y 16 GRAVES.
  El diseño es bueno; la implementación aún no cumple sus propios contratos.

## Contexto honesto (leer antes que los hallazgos)

`tests/unit/test_codex_fabric_contracts.py` es de CODEX y hoy está **11 rojos / 1 verde**:

```
11 failed, 1 passed in 1.88s
FAILED test_canonical_mapping_rejects_non_string_key_with_domain_error
FAILED test_spec_fingerprint_commits_data_and_feature_snapshots
FAILED test_fabric_factory_config_builds_and_preserves_action_diagnostic_wall
FAILED test_semantic_diff_ignores_only_declared_volatile_fields
FAILED test_snapshot_rejects_signal_not_available_at_cutoff
FAILED test_snapshot_hash_commits_policy_max_age_and_signal_payload
FAILED test_inverse_volatility_baseline_is_normalized_before_caps
FAILED test_allocator_multiplier_reduces_without_renormalizing_survivors
FAILED test_quality_rule_quarantines_decimal_nan_instead_of_crashing
FAILED test_qlab_idempotency_rejects_same_trial_id_with_changed_result
FAILED test_execution_flip_is_opening_and_exit_all_never_submits_target
```

Esto es coherente con una fase **RED de TDD deliberada**, y así lo trato: los hallazgos que ya
tienen un test rojo de CODEX se marcan **[YA-ROJO]** (conocidos, pendientes de implementar) y NO
cuentan como valor de esta auditoría. Los demás son **puntos ciegos**: ningún test los cubre y el
código los pasa hoy en silencio. El peso de este informe está en los puntos ciegos.

Ninguno de los módulos está cableado a producción todavía (`grep` de importadores: solo se importan
entre sí + `src/portfolio`, `src/orchestration`, `src/strangler`). Eso acota el daño **hoy**, no la
gravedad de fusionarlos así.

---

## Lo que está bien hecho (y no debe tocarse)

- **El writer escribe bytes, no texto** (`canonical.py:134`, `write_bytes`) — inmune a la traducción
  CRLF de Windows que ya causó un incidente en este repo (commit `3861568`).
- **NFC de claves y valores + detección de colisión NFC** (`canonical.py:83-86`): `{"señal"}` NFC y
  NFD dan el **mismo** hash y una colisión de claves post-normalización es error duro. Verificado.
- **Normalización temporal correcta**: `datetime` con offset −05:00 y su equivalente UTC dan el mismo
  hash, y coinciden con la forma string ISO-Z. Naive timestamps prohibidos (`canonical.py:22-23`).
- **Ruido de float resuelto**: `0.1+0.2` ≡ `0.3`, y `-0.0` ≡ `0.0`. Verificado ejecutando.
- **`None` ≠ clave ausente** (hashes distintos) — correcto: distingue "declarado nulo" de "no declarado".
- **Reloj inyectable** en `ExecutionService.execute_target(..., now=...)` y **Protocols** para todas
  las dependencias (DIP real, no clases concretas). El servicio es testeable sin Airflow, que es
  literalmente el objetivo de BL-30.
- **Enums cerrados y fail-closed** en `governance/declaration.py`; la matriz de 26 combinaciones
  legales sobre 96 nominales es correcta y está bien contada.
- **`src/execution/events.py` deriva la idempotency_key exactamente con los 5 campos que pide BL-21**
  (account ⊕ instrument ⊕ target_version ⊕ decision_fp ⊕ cutoff).

---

## BLOQUEANTES

### X-01 — El "hash canónico" colisiona entre número y string (identidad rota en la raíz)

**Severidad**: BLOQUEANTE · **Fichero**: `src/identity/canonical.py:74-77` (+ `99-115`)

`float` se serializa como **string decimal** sin etiqueta de tipo, mientras `int` se serializa como
número JSON. Consecuencia: **cualquier float colisiona con el string que lo representa**, y `1` (int)
NO colisiona con `1.0` (float). Es la ambigüedad clásica de canonicalización, y afecta a la primitiva
de la que dependen `spec_fingerprint`, `decision_fingerprint`, `execution_fingerprint`,
`derivation_id`, `order_idempotency_key`, `portfolio.snapshot` y `strangler.parity`.

**Escenario reproducible** (ejecutado):

```
== type-confusion collision: float 1.0 vs string '1' ==
   sha256:2739c05c2b0485fa413088c09ebc7bd13487240713bafad3d27aedac9d139d76
   sha256:2739c05c2b0485fa413088c09ebc7bd13487240713bafad3d27aedac9d139d76
   COLLIDE? True
   1.5 vs '1.5' COLLIDE? True
== same effect through spec_fingerprint (the real API) ==
  spec A: strategy_spec={"tp_mult": 2.0, "eps": 1e-13}
  spec B: strategy_spec={"tp_mult": "2",  "eps": 1e-20}
  COLLIDE? True
  int 2 vs float 2.0 -> same fingerprint? False
```

Dos specs de estrategia **distintas** (`tp_mult` numérico vs string, `eps` 1e-13 vs 1e-20) producen
**el mismo `spec_fingerprint`**. Un YAML que hoy dice `tp_mult: 2.0` y mañana `tp_mult: "2"` es
indistinguible; y el mismo YAML releído con `2` en vez de `2.0` cambia de identidad. Esto rompe
simultáneamente la detección de cambio de receta (BL-14/BL-28) y la trazabilidad del Passport.

**Remedio**: etiquetar el tipo en la forma canónica. O bien un envelope
(`{"$num": "2"}` / `{"$str": "2"}`), o bien un prefijo reservado en el string (`"n:2"` vs `"s:2"`),
o serializar los decimales como número JSON con render fijo y prohibir strings numéricos ambiguos.
Además: unificar `int` y `float` bajo una única representación decimal (hoy divergen), o rechazar
`float` y exigir `Decimal` en las superficies de fingerprint.

---

### X-02 — La idempotency_key ignora el entorno: una corrida paper **cancela** la orden live

**Severidad**: BLOQUEANTE · **Fichero**: `src/execution/events.py:25-42` + `src/execution/service.py:217-219`

La clave es `SHA256(account ⊕ instrument ⊕ target_version ⊕ decision_fp ⊕ cutoff)` — literal de
BL-21 — pero BL-21 también exige **4 entornos** (`ExecutionEnvironment` REPLAY/PAPER/CANARY/LIVE
está definido en el mismo fichero y **no se usa en ninguna parte**). Con el mismo target aprobado,
paper y live comparten clave; el segundo path devuelve `IDEMPOTENT_REPLAY` y **nunca llega al broker**.

**Escenario reproducible** (ejecutado, mismo ledger, mismo target, distinto `environment`):

```
== 4. idempotency key ignores environment, side and quantity ==
   key(paper) == key(live) ? True -> env is not in the key at all
   paper run: SUBMITTED
   live run : IDEMPOTENT_REPLAY | live broker submits: 0 <- the LIVE order was never sent
```

La estrategia queda **desalineada del target real** sin ningún error: el sistema cree que la orden
está colocada. El fallo simétrico también existe (una corrida live "consume" el paper).

**Remedio**: incluir `env` (y `executor_type`) en el payload de la clave, y validar
`environment` contra `ExecutionEnvironment` en `PortfolioTarget` (ver X-17). El UNIQUE de BD debe
ser sobre `(env, idempotency_key)`.

---

### X-03 — TOCTOU: dos invocaciones concurrentes envían **dos órdenes** con la misma clave

**Severidad**: BLOQUEANTE · **Fichero**: `src/execution/service.py:217-230`

El patrón es *read-then-write*: `find_order_by_idempotency_key()` → `append_order()` → `submit()`.
No hay inserción atómica ni conflicto declarado en el `Protocol EventLedger`, así que la unicidad no
está en ninguna parte del código. BL-21 fija como criterio de aceptación
*"Retry inyectado ⇒ cero órdenes duplicadas"*.

**Escenario reproducible** (ejecutado; el ledger fake modela un SELECT con latencia de red, que es
exactamente lo que hace Postgres):

```
statuses: ['SUBMITTED', 'SUBMITTED']
broker.submit calls for the SAME idempotency key: 2
ledger append_order calls: 2
```

Dos órdenes reales al exchange por el mismo target. Con un reintento de Airflow/cron solapado —el
caso que BL-21 nombra— esto es doble exposición.

**Remedio**: `append_order` debe ser *insert-or-get* atómico (`INSERT ... ON CONFLICT (env,
idempotency_key) DO NOTHING RETURNING`), y el servicio debe tratar "0 filas insertadas" como replay.
El `Protocol` debe documentar esa semántica; sin UNIQUE en la DDL el contrato es indemostrable.

---

### X-04 — El motor "gobernado" publica Sharpe = 4.09e16 y Sharpe con N=5 como `status: OK`

**Severidad**: BLOQUEANTE · **Fichero**: `src/metrics/engine.py:138-143`, `249-251`,
`config/metrics/catalog.yaml` (sin `min_obs`)

Dos violaciones directas de reglas auto-cargadas (`quant-constitution.md` §6, `strategy-contract.md`
§6: *"con N < 20 trades no se reportan Sharpe ni p-value"*), más una **regresión** respecto a BL-07,
que ya está `IMPLEMENTED` y honra `MIN_TRADES_FOR_STATS` de `src/contracts/strategy_schema.py`
(`scripts/analysis/timing_ratio_oneoff.py:262` → `attribution_status: INSUFFICIENT_TRADES`).

**Escenario reproducible** (ejecutado):

```
== 1. Sharpe with N=5 trades ==
   value: 8.143987248386798 status: OK  N = 5
== 7. Sharpe of a strategy with zero variance ==
   value: 4.087048463993355e+16 status: OK
timing_ratio with N=5 -> -3.3257964991947446 OK | BL-07 demands INSUFFICIENT_TRADES + suppression
```

El Sharpe de 4.09e16 sale de `services/common/metrics.py:41` (`std_excess == 0` no captura el
1.7e-18 de ruido de coma flotante en `[0.01]*30`), y `engine.py:250` solo verifica `isfinite`, que
un 4e16 pasa. La constitución §6 dice literalmente que Sharpe > 4-5 es look-ahead hasta demostrar lo
contrario; aquí el motor lo sella con `catalog_version`, `formula_version` y `status: OK`.

**Remedio**: (a) `min_obs`/`min_trades` en el catálogo por métrica, resuelto contra
`MIN_TRADES_FOR_STATS`, con estado `INSUFFICIENT_SAMPLE` y `metric_value: null`; (b) banda de
plausibilidad por unidad (`ratio` ⇒ |x| ≤ 10 salvo override declarado) que degrade a
`IMPLAUSIBLE`, no a `OK`; (c) tolerancia relativa en el guard de varianza cero.

---

## GRAVES

### X-05 — Todo float menor que 1e-12 colapsa a `0` (colisión silenciosa) y los `field_quantums` de arrays no se aplican nunca

**Severidad**: GRAVE · **Fichero**: `src/identity/canonical.py:29-39` (quantum por defecto `1e-12`),
`91-95` (paths con índice)

`fingerprints.py` **nunca** pasa `field_quantums`, así que las 4 huellas del spine usan el quantum
por defecto. Ejecutado:

```
== D. tiny floats collapse (default quantum 1e-12) ==
       1e-13 -> {'p': '0'}   1e-20 -> {'p': '0'}   5e-13 -> {'p': '0'}   0.0 -> {'p': '0'}
  1e-13 == 1e-20 ? True      1e-13 == 0.0 ? True
```

Un `p_value`, una tolerancia o un `eps` distintos son la misma identidad. Peor: el quantum por path
**se ignora en silencio** dentro de listas, porque el path real lleva índice (`/legs/0/qty`) y
`_quantum` solo hace lookup exacto o `"*"`:

```
== N. field_quantums path for list items ==
  no quantums      : {'legs': [{'qty': '1.005'}, {'qty': '2.005'}]}
  quantum /legs/qty: {'legs': [{'qty': '1.005'}, {'qty': '2.005'}]}   <- ignorado, sin error
  quantum '*'=0.01 : {'legs': [{'qty': '1'}, {'qty': '2'}]}           <- '*' arrasa con todo
```

Crees haber cuantizado a 0.01 y no lo hiciste. Silencio total.

**Remedio**: rechazar claves de `field_quantums` que no matchean ningún path (fail-closed); soportar
comodín de índice (`/legs/*/qty`); prohibir quantum por defecto en superficies de fingerprint
(exigir schema explícito) o documentar 1e-12 como pérdida de precisión aceptada por contrato.

### X-06 — Un float ≥ 1e16 revienta con `decimal.InvalidOperation`, fuera del tipo de error declarado

**Severidad**: GRAVE · **Fichero**: `src/identity/canonical.py:45` (`Decimal.quantize`)

```
== E. large floats vs 1e-12 quantum ==
  1e+15 -> sha256:ffbc8e9eb605df81
  1e+16 -> RAISED InvalidOperation: [<class 'decimal.InvalidOperation'>]
  1e+20 -> RAISED InvalidOperation: [<class 'decimal.InvalidOperation'>]
```

Cuantizar a 1e-12 un número de 17 dígitos excede la precisión de 28 del contexto Decimal. `1e16` no
es exótico aquí: un notional en COP, un epoch en nanosegundos (`pd.Timestamp.value` ≈ 1.7e18), un
volumen en satoshis. La excepción **no** hereda de `CanonicalizationError`, así que todo caller que
haga `except CanonicalizationError` se cae igual.

**Remedio**: `localcontext(prec=...)` dimensionado, o cuantización relativa (dígitos significativos)
en vez de absoluta; y envolver cualquier `InvalidOperation` en `CanonicalizationError`.

### X-07 — CRLF vs LF dentro de un string cambia la huella (el incidente que este repo ya tuvo)

**Severidad**: GRAVE · **Fichero**: `src/identity/canonical.py:66-67`

```
== F. CRLF vs LF inside a string payload ==
  LF  : sha256:48621d33edf2ff88
  CRLF: sha256:a5b5887c3ffb00e4
  equal? False
```

El writer normaliza Unicode a NFC pero no los finales de línea. Cualquier payload que incluya texto
leído de disco (una receta YAML embebida, un `rule_trace`, una nota de spec) produce huellas
distintas en un checkout Windows y uno Linux. El commit `3861568` de este repo existe precisamente
por esto ("HASH CANONICO LF reproducible desde blob git en cualquier OS").

**Remedio**: normalizar `\r\n` y `\r` → `\n` en `canonicalize` para todo string, junto al NFC. Es una
línea, y cierra una clase entera de incidentes ya vivida.

### X-08 — `1` y `1.0` producen fingerprints distintos (round-trip de YAML rompe identidad)

**Severidad**: GRAVE · **Fichero**: `src/identity/canonical.py:64` vs `74-77`

Verificado: `spec_fingerprint(strategy_spec={"tp_mult": 2})` ≠ `spec_fingerprint({"tp_mult": 2.0})`.
YAML/JSON/Postgres numeric no preservan esa distinción de forma estable; un `yaml.safe_load` de
`2.0` y de `2` difieren en tipo Python pero no en semántica económica. Convive con X-01 (donde
`1.0` sí colisiona con `"1"`): la relación de equivalencia implementada no es ni reflexiva de tipos
ni consistente. **Es el mismo defecto raíz que X-01 y debe arreglarse en el mismo cambio.**

### X-09 — Los umbrales de `research.dsr` están por debajo de la barra constitucional: DSR 0.94 se reporta `OK`

**Severidad**: GRAVE · **Fichero**: `config/metrics/catalog.yaml:48-49`

`quant-constitution.md` §2 y `approval-gates.md` §7 fijan **DSR > 0.95** como barra (cambiarla exige
ADR). El catálogo declara `warning: 0.90`, `critical: 0.50`. Ejecutado:

```
== 5. catalog thresholds vs constitution bar (DSR > 0.95) ==
   catalog warning/critical: 0.9 0.5
   a DSR of 0.94 would be reported as: OK
   a DSR of 0.91 would be reported as: OK
   a DSR of 0.6  would be reported as: WARNING
```

Un DSR de 0.94 —que la constitución RECHAZA— sale verde en el evento gobernado que el dashboard
consumirá. Es exactamente el "default silencioso" que convierte una regla dura en decoración.

**Remedio**: `warning: 0.95`, `critical: 0.95` (o un `gate: 0.95` explícito con estado `FAIL`), y un
test que compare el umbral del catálogo contra el SSOT de la constitución.

### X-10 — El DSR se calcula asumiendo normalidad aunque los retornos están en el contexto

**Severidad**: GRAVE · **Fichero**: `src/metrics/engine.py:159-182`

`_dsr` pasa `skew=context.get("skew", 0.0)` y `kurtosis=context.get("kurtosis", 3.0)`. Si el caller
olvida los momentos, el DSR se evalúa bajo normal. Sobre la MISMA serie que ya está en `context`,
`services/common/metrics.py::trial_aware_moments` los mide:

```
== 4. DSR: default skew/kurtosis vs moments measured from the SAME returns ==
   engine DSR (skew=0, kurt=3 defaults): 0.0
   DSR with measured moments            : 0.0  (skew=-5.690 kurt=39.254)
```

En ese ejemplo ambos dan 0.0, pero los momentos reales (skew −5.69, kurtosis 39.25) son brutalmente
no normales: la elección del default **no es neutra** y sesga al alza el DSR de estrategias con cola
izquierda gorda — justo el perfil de una estrategia con stops. Que la métrica que decide PROMOTE
dependa de que el caller "se acuerde" es fail-open.

**Remedio**: `_dsr` debe llamar a `trial_aware_moments(returns)` y usar los momentos medidos;
permitir override solo explícito y registrado en `lineage`. Además reutiliza `sharpe_per_period` en
vez de reimplementar `mean/std` (hoy hay dos cálculos de Sharpe en el mismo fichero).

### X-11 — `annualization: from_asset_registry` no lee ningún registro, y `window` es una etiqueta sin efecto

**Severidad**: GRAVE · **Fichero**: `src/metrics/engine.py:277-289` y `236-240`, `273`

BL-18 dice literalmente: *"'annualization: from_asset_registry' resuelve mecánicamente la regla de
relojes"*. La implementación hace `context.get("annualization")` — es decir, **el caller decide**.
El registro por activo ya existe (`config/assets/btcusdt.yaml:33-34` `bars_per_year: 105120`/
`trading_days_per_year: 365`; `config/assets/xauusd.yaml:29` `72000`;
`config/analysis/analysis_assets.yaml:32,51,66`) y no se consulta.

```
== 2. Same returns, two different annualizations ==
   annualization=  52 -> sharpe=-0.0758 status=CRITICAL
   annualization= 252 -> sharpe=-0.1669 status=CRITICAL
   annualization= 365 -> sharpe=-0.2008 status=CRITICAL
   asset_id=usdcop with annualization=252 -> -0.1669 CRITICAL   <- nadie lo impide
== 3. window label is not enforced ==
   52w over 26 obs: -1.4971  | 52w over 60 obs: -0.0758
   same metric_id+window, different numbers, both status: CRITICAL CRITICAL
```

Tres números "gobernados" para la misma serie, y dos números distintos bajo la misma etiqueta `52w`.
`strategy-contract.md` §5 prohíbe comparar activos con relojes distintos; con esto la tabla de
ranking vuelve a ser posible y además indetectable.

**Remedio**: resolver `annualization` desde el registro por `asset_id` dentro del motor (fallar si
`asset_id` no está o no está catalogado); derivar la ventana del propio motor a partir de
`as_of` + `window` sobre los datos, o exigir en `lineage` el rango `[start,end]` y el hash del
input, y validar que el conteo coincide con la ventana declarada.

### X-12 — Calmar de una estrategia sin drawdown se publica como `0.0` con estado `CRITICAL`

**Severidad**: GRAVE · **Fichero**: `src/metrics/engine.py:146-151` → `services/common/metrics.py:161-162`

```
== 6. Calmar of a strictly-winning strategy (max_dd == 0) ==
   value: 0.0 status: CRITICAL   <- 30 straight wins
```

Una métrica **indefinida** (división por cero) se publica como el número 0.0 y se sella `CRITICAL`.
Un `null` con estado `N_A` ya está soportado por `MetricEvent` y `_status`; no usarlo es
precisamente "default silencioso". El mismo patrón afecta a `calculate_sharpe_ratio` (retorna 0.0
si `len<2` o `std==0`) y a `calculate_profit_factor`, que devuelve `float('inf')` — prohibido por
`strategy-contract.md` §2 si alguna vez se cataloga.

**Remedio**: el motor debe distinguir "no calculable" de "cero". Envolver las funciones de
`services/common` en adaptadores que devuelvan `None` en los casos degenerados, o corregirlas allí.

### X-13 — `Infinity` en una barra se ACEPTA como `VALID` (fail-open en un módulo que se declara fail-closed)

**Severidad**: GRAVE · **Fichero**: `src/data_quality/rules.py:28-57` (docstring línea 1:
*"Fail-closed, non-clipping quality rules"*)

```
== 12. NaN / Infinity in a bar ==
   {open: NaN, ...}  -> RAISED InvalidOperation   [YA-ROJO: test_quality_rule_quarantines_decimal_nan…]
   {open: inf, high: inf, low: 3900, close: 3950} -> QualityDecision(accepted=True, status='VALID')
```

El caso NaN ya tiene test rojo de CODEX. **El caso `Infinity` no lo tiene**: `Decimal('Infinity')`
compara bien, pasa el orden OHLC, y como USDCOP no tiene rango declarado, sale `VALID`. Además, sin
rango declarado, **cualquier** anomalía pasa:

```
== 11. USD/COP bar at an absurd price (no declared range) ==
   USDCOP 4000     -> VALID
   USDCOP 0.0001   -> VALID
   USDCOP 999999   -> VALID
   BTCUSDT -50     -> VALID     <- precio negativo
   XAUUSD 0        -> VALID
```

Los rangos MXN/CLP implementados **sí** corresponden a lo que BL-40 pide explícitamente, y eso es
correcto; lo que falla es el **default** para todo lo demás: instrumento sin rango ⇒ aceptar. En un
sistema de trading el default debe ser el contrario.

**Remedio**: (a) rechazar no-finitos y no-positivos **antes** de comparar (y dentro del `try`);
(b) instrumento sin rango declarado ⇒ `QUARANTINED`/`UNKNOWN_INSTRUMENT`, nunca `VALID`;
(c) mover `_PRICE_RANGES` a config versionada (hoy es una constante hardcodeada en el código, en un
repo cuya regla SSOT prohíbe justamente eso).

### X-14 — Las reglas de rango se saltan con solo escribir el símbolo distinto

**Severidad**: GRAVE · **Fichero**: `src/data_quality/rules.py:45` (`instrument.upper().replace("/", "")`)

```
== 15. instrument alias mismatch ==
   'usdmxn'  -> QUARANTINED (bar.range.USDMXN)
   'USD_MXN' -> VALID
   'MXN=X'   -> VALID
```

La normalización es ad-hoc (mayúsculas + quitar `/`). Los alias reales de proveedor —`MXN=X`
(Yahoo), `USD_MXN`, `USDMXN.FX`— **evitan la cuarentena en silencio**. Y esto ocurre teniendo
`src/market/identity.py` (BL-37) en el mismo árbol, que existe para resolver alias: los dos módulos
no se hablan. BL-37 lo dice: *"una identidad, N alias mapeados"*.

**Remedio**: `evaluate_bar` debe recibir un `instrument_id` ya canónico (o resolverlo vía el registro
de BL-37) y **fallar** si el símbolo no resuelve.

### X-15 — El kill switch: un giro completo long→short se clasifica `reduce_only` y pasa

**Severidad**: GRAVE · **Fichero**: `src/execution/service.py:247-248`, `256`, `314-320`
· **[YA-ROJO parcial]** (`test_execution_flip_is_opening_and_exit_all_never_submits_target`)

`opening = abs(target_notional) > abs(current_notional)` no mira el **signo**. Ejecutado:

```
== 1. FULL REVERSAL long->short ==
   kill_switch=BLOCK_NEW -> status=SUBMITTED side=SELL qty=200.00 reduce_only=True
   (la posición pasa de LONG 100 a SHORT 100 mientras se declara reduce_only)
== 2. ACCOUNT_FREEZE (level 4) still lets 'reduce_only' orders through ==
   result: SUBMITTED | broker.submit calls: 1 | cancel_open: 1 | exit_all: 1
== 3. exit_all/cancel_open re-fire on every invocation ==
   after 4 total calls -> cancel_open: 4  exit_all: 4
```

Lo que **NO** cubre el test rojo de CODEX y añado aquí: (a) `ACCOUNT_FREEZE` (nivel 4, el más
severo) también deja pasar `reduce_only` — un freeze de cuenta debe bloquear todo; (b)
`_enforce_switch_side_effects` **no es idempotente**: cada invocación vuelve a disparar
`cancel_open` y `exit_all` (4 llamadas en 4 invocaciones), de modo que un scheduler que reintente
mientras el switch está en EXIT_ALL machaca al broker con salidas repetidas; (c) los efectos
laterales se ejecutan **y además** se envía la orden en la misma pasada.

**Remedio**: `reduce_only` = mismo signo **y** `|target| ≤ |current|`; tabla explícita nivel→acciones
con `ACCOUNT_FREEZE` ⇒ bloqueo total; efectos laterales idempotentes (guardados por evento en el
ledger, no por invocación).

### X-16 — Un timeout del broker se registra como `REJECTED` (la orden puede estar viva en el exchange)

**Severidad**: GRAVE · **Fichero**: `src/execution/service.py:222-226`

```
== 6. broker timeout is recorded as REJECTED ==
   raised: gateway timeout - order state UNKNOWN
   ledger status trail: [('o1', 'REJECTED', 'BROKER_SUBMIT_ERROR')]
```

`except Exception` trata por igual "el broker rechazó" y "no sé qué pasó". Un timeout de gateway es
el caso clásico en que la orden **sí** llegó. Marcarla `REJECTED` hace que la proyección de estado
mienta y que un reintento posterior… se bloquee por idempotencia (bien) pero sobre un estado
`REJECTED` falso (mal). BL-21: *"estado de orden = proyección de eventos"* — un evento falso
envenena la proyección.

**Remedio**: estado `SUBMIT_UNKNOWN` / `PENDING_RECONCILIATION` para errores no clasificables como
rechazo explícito del venue, y forzar reconciliación antes de cualquier acción posterior sobre esa
cuenta.

### X-17 — No se consulta `PreTradeGate`, ni `trading_mode`, ni el kill-switch por usuario; `environment` es un string libre

**Severidad**: GRAVE · **Fichero**: `src/execution/service.py:190-230`, `232-269`

`rbac.md` regla dura 6: *"`PreTradeGate` (services/signalbridge_api/app/services/pretrade.py) es el
último gate antes del exchange — modo global (`trading_mode`, default PAPER simula y NO envía),
kill-switch por usuario (`sb_trading_configs.trading_enabled`, default False), caps de notional y
trades/día. Fail-safe: error ⇒ BLOCK"*. Los 16 checks de `_pretrade` son un gate **paralelo** que no
incluye ninguno de esos cuatro:

```
== 8. checks evaluated ==
['approved_immutable_target','broker_internal_position_match','daily_loss_cap',
 'decision_fingerprint_present','gross_exposure_cap','instrument_allowed','kill_switch_allows',
 'minimum_notional','order_notional_cap','position_notional_cap','positive_mark','positive_nav',
 'quantity_positive','reconciliation_clean','snapshot_and_allocation_linked','target_time_valid']
```

Y el entorno no se valida contra el enum que el propio paquete define:

```
== 7. environment is a free string ==
   env='lve' accepted -> SUBMITTED | intent.environment = lve
```

Un typo (`"lve"`) pasa por todos los gates. Faltan también: `trades/día`, y ninguna diferencia de
comportamiento entre `paper` y `live` (paper **envía** al `Broker` igual).

**Remedio**: `environment: ExecutionEnvironment` tipado en `PortfolioTarget`; el modo global y el
switch por usuario como checks del `_pretrade` con fail-safe BLOCK; y decidir explícitamente si
`ExecutionService` sustituye o delega en `PreTradeGate` — hoy son dos caminos a exchange con reglas
distintas, que es la peor de las dos opciones.

### X-18 — La moneda se declara y nunca se usa: notional y límites pueden estar en unidades distintas

**Severidad**: GRAVE · **Fichero**: `src/execution/service.py:56-76`, `242-268`

`AccountState.currency` existe y **no se lee en ninguna línea**. `RiskLimits` no tiene moneda.
`target_notional = nav * target_weight` hereda la unidad del NAV; `max_order_notional` es un Decimal
sin unidad. Con USD/COP la diferencia entre pesos y dólares es ~4000×: un cap de "1.000" pensado en
USD deja pasar una orden de 4.000.000 COP, o al revés bloquea todo. `BL-42` existe justamente por
esto ("unidades y decimales de la señal normalizada").

**Remedio**: moneda obligatoria en `RiskLimits` + assert de igualdad con `AccountState.currency`, o
un tipo `Money(amount, currency)` con aritmética que rechace mezclas.

### X-19 — La matriz de legalidad solo se aplica si entras por `from_mapping()`

**Severidad**: GRAVE · **Fichero**: `src/governance/declaration.py:90-133`

`GovernanceDeclaration` es un dataclass congelado **sin `__post_init__`** (a diferencia de
`ProviderSymbol`, que sí lo tiene). Construirlo directamente evita `validate_declaration`:

```
== governance: illegal combination via direct construction ==
   constructed PAPER+FULL without error -> PAPER FULL | blocks_new_orders: False | dag_should_exist: True
   validate_declaration would have rejected it: PAPER cannot use FULL; allowed=['SHADOW','ZERO']
```

BL-16 fija como verificación *"Declaración PAPER+FULL ⇒ CI rojo"*. Hoy PAPER+FULL es construible en
proceso y sus propiedades (`blocks_new_orders`, `dag_should_exist`) responden como si fuese legal.
Cualquier consumidor que reciba el objeto ya construido (no el mapping) opera sin matriz.

**Remedio**: `__post_init__` que llame a la validación de la matriz; `from_mapping` queda como
parser. Un objeto ilegal no debe poder existir.

### X-20 — `src/orchestration/semantic_diff.py` no importa: llama a una API que no existe

**Severidad**: GRAVE · **Fichero**: `src/orchestration/semantic_diff.py:10` · **[YA-ROJO]**

```
FAIL  src.orchestration.semantic_diff -> ImportError: cannot import name 'canonical_bytes'
      from 'src.identity.canonical'
```

`canonical.py` exporta `canonical_json_bytes` y `semantic_hash`; `semantic_diff` importa
`canonical_bytes` y `canonical_hash`. Módulo muerto en el árbol. Señala que **no hay un smoke test
de import** para los módulos nuevos: un `pytest --collect-only` sobre `src/**` lo habría cazado
antes de escribir una sola línea de lógica.

**Remedio**: fijar los nombres de la API pública en `src/identity/__init__.py` y añadir un test de
import de todos los módulos nuevos (es el test más barato del repo y aquí ya habría pagado).

---

## MENORES / OBSERVACIONES

- **X-21** (`canonical.py:64,74`): `np.float64` se acepta silenciosamente (es subclase de `float`)
  pero `np.int64`, `np.bool_` y `np.float32` lanzan `CanonicalizationError: unsupported canonical
  type`. En un repo donde casi todo nace de un DataFrame, `df.to_dict()` explota en la mitad de los
  tipos y pasa en la otra mitad. Decidir explícitamente: rechazar todo numpy, o convertirlo todo.
- **X-22** (`canonical.py:80`) **[YA-ROJO]**: claves de tipos mezclados → `TypeError` crudo de
  `sorted()` en vez de `CanonicalizationError`. Igual con `asdict()` sobre una *clase* dataclass
  (`canonicalize(Foo)` → `TypeError`).
- **X-23** (`service.py:227-229`): el `broker_order_id` se pasa en el parámetro `reason_code` de
  `append_status`. Abuso de campo; el id del broker acabará en una columna de motivos. Además
  `OrderHeader` (events.py:45) **no se usa**: el ledger devuelve un `Mapping` sin tipar y el servicio
  hace `header["order_id"]` a ciegas.
- **X-24** (`engine.py:256`): `metric_event_id = uuid.uuid4()` — recalcular la misma métrica genera
  dos eventos distintos y no deduplicables, teniendo `src/identity` en el mismo árbol. Un
  `derivation_id(inputs, code_hash, params)` haría el evento idempotente y reproducible, que es lo
  que BL-18 pide de `control.metric_event`.
- **X-25** (`market/identity.py:54-64`): BL-37 pide un registro `provider_symbol → instrument_id`
  con biyectividad; lo implementado es un dataclass sin registro ni unicidad, y sin normalizar:
  `ProviderSymbol("twelvedata","USD/COP","usdcop")` y `(..., "USD-COP")` coexisten, y
  `" usdcop "` conserva los espacios. Además `normalize_interval` rechaza `1m`, `15m` y `PT1M`
  (no hay intervalo de 1 minuto en el enum, ojo con el choque futuro `P1M` mes vs `PT1M` minuto).
  **De BL-38 (raw_bar/canonical_bar, `bar_method`, resampleo, anclaje UTC de la barra diaria,
  semántica de los 5 timestamps) no hay nada todavía** — el bug histórico "Sunday pile-up" no está
  ni defendido ni testeado en este módulo.
- **X-26** (`engine.py:103-111`): `warning`/`critical` se leen del YAML sin validar tipo; un
  `warning: "0.25"` (string) revienta con `TypeError: '<=' not supported between 'float' and 'str'`
  en el momento de evaluar, no al cargar el catálogo. `MetricCatalog.load` tampoco verifica que cada
  métrica catalogada tenga fórmula registrada.
- **X-27** (`canonical.py:118-131`): `CanonicalArtifact` fija `semantic_hash == bytes_hash` por
  construcción (misma expresión). BL-17 pide como verificación *"derivation_id igual con semantic
  distinto ⇒ incidente"*; con ambos hashes idénticos por definición, ese detector es vacío.
- **X-28** (`portfolio/snapshot.py:55`): import de `semantic_hash` **dentro** de una función. Import
  perezoso oculto, sin motivo aparente de ciclo; rompe la trazabilidad de dependencias.

---

## Veredicto

**RECHAZADO para merge.** Ordenado por lo que hay que arreglar primero:

1. **Identidad (X-01, X-05..X-08)** — es la base de todo lo demás. Mientras el hash confunda `1.0`
   con `"1"` y colapse `1e-13` a `0`, cada fingerprint, cada Passport y cada idempotency_key
   construido encima hereda el defecto. **Arreglar esto antes que nada; todo lo aguas abajo hay que
   re-hashearlo después.**
2. **Ejecución (X-02, X-03, X-15..X-18)** — es el único bloque que puede perder dinero real.
   Ninguno de estos módulos debe tocar un `Broker` no simulado hasta que X-02/X-03 tengan UNIQUE en
   la DDL y test de concurrencia verde.
3. **Métricas (X-04, X-09..X-12)** — el motor "gobernado" hoy sella con `status: OK` números que las
   reglas auto-cargadas prohíben publicar. Es peor que no tener motor: da autoridad a lo que la
   constitución rechaza.
4. **Calidad y gobernanza (X-13, X-14, X-19)** — defaults fail-open en módulos cuyo docstring dice
   fail-closed.

Lo que **sí** está bien —el writer en bytes, NFC, tz-aware, `None ≠ ausente`, los Protocols, el reloj
inyectable, la matriz de 26 combinaciones y el enfoque red-first— es una base sólida. El problema no
es el diseño: es que 11 de 12 contratos declarados están rojos y hay al menos 20 comportamientos
peligrosos que **ningún test cubre todavía**.

---

## Tests TDD/BDD que exigiría para dar esto por bueno

Todos deben demostrarse **ROJOS contra el código actual** antes del fix (los marcados 🔴 ya los
reproduje ejecutando; los demás derivan de lectura + ejecución del mismo probe).

### Identidad — `tests/unit/test_canonical_identity.py`

| Test | Debe estar rojo hoy porque |
|---|---|
| `test_number_and_its_string_form_never_share_a_hash` 🔴 | `semantic_hash({"x":1.0}) == semantic_hash({"x":"1"})` |
| `test_int_and_float_of_equal_value_share_a_hash` 🔴 | `spec_fingerprint({"tp_mult":2}) != spec_fingerprint({"tp_mult":2.0})` |
| `test_distinct_subquantum_floats_do_not_collide` 🔴 | `1e-13`, `1e-20` y `0.0` dan el mismo hash |
| `test_large_float_raises_canonicalization_error_not_invalid_operation` 🔴 | `1e16` lanza `decimal.InvalidOperation` |
| `test_crlf_and_lf_strings_share_a_hash` 🔴 | `"a\r\nb"` y `"a\nb"` difieren |
| `test_unmatched_field_quantum_path_is_rejected` 🔴 | `{"/legs/qty": "0.01"}` se ignora en silencio dentro de arrays |
| `test_numpy_scalars_are_handled_consistently` 🔴 | `np.float64` pasa, `np.int64` lanza |
| `test_all_new_modules_import` 🔴 | `src.orchestration.semantic_diff` → `ImportError` |
| `test_hash_is_stable_across_line_ending_checkouts` | gate de CI equivalente al de `3861568` |

### Métricas — `tests/unit/test_metric_engine_contract.py`

| Test | Debe estar rojo hoy porque |
|---|---|
| `test_sharpe_below_min_trades_is_suppressed` 🔴 | N=5 → `8.14`, `status OK` |
| `test_timing_ratio_below_min_trades_is_insufficient_sample` 🔴 | N=5 → `-3.33`, `status OK` (regresión vs BL-07) |
| `test_zero_variance_series_never_yields_astronomical_sharpe` 🔴 | `[0.01]*30` → `4.087e16`, `status OK` |
| `test_undefined_calmar_is_null_not_zero` 🔴 | `max_dd == 0` → `0.0` + `CRITICAL` |
| `test_dsr_threshold_matches_constitutional_bar` 🔴 | DSR 0.94 → `OK` (catálogo 0.90/0.50 vs 0.95) |
| `test_dsr_uses_measured_skew_and_kurtosis` 🔴 | defaults 0/3 con los retornos disponibles en contexto |
| `test_annualization_is_resolved_from_asset_registry` 🔴 | `asset_id=usdcop` + `annualization=252` se acepta |
| `test_same_asset_cannot_publish_two_annualizations` 🔴 | 52/252/365 sobre la misma serie, los tres `OK` |
| `test_window_label_must_match_the_evaluated_range` 🔴 | 26 y 60 observaciones, ambas `52w` |
| `test_catalog_thresholds_must_be_numeric` 🔴 | `warning: "0.25"` → `TypeError` al evaluar |
| `test_metric_event_id_is_content_addressed` 🔴 | `uuid4` ⇒ no idempotente |

### Calidad — `tests/unit/test_quality_rules_fail_closed.py`

| Test | Debe estar rojo hoy porque |
|---|---|
| `test_non_finite_bar_is_quarantined` 🔴 | `inf` → `VALID` (NaN ya cubierto por el test de CODEX) |
| `test_non_positive_price_is_quarantined` 🔴 | `-50` y `0` → `VALID` |
| `test_instrument_without_declared_range_is_quarantined` 🔴 | USDCOP `0.0001` y `999999` → `VALID` |
| `test_provider_aliases_resolve_to_the_same_rule` 🔴 | `MXN=X`/`USD_MXN` esquivan el rango de USDMXN |
| `test_price_ranges_come_from_versioned_config` 🔴 | hoy es constante hardcodeada |

### Ejecución — `tests/unit/test_execution_service_safety.py`

| Test | Debe estar rojo hoy porque |
|---|---|
| `test_paper_and_live_targets_never_share_an_idempotency_key` 🔴 | live → `IDEMPOTENT_REPLAY`, 0 envíos |
| `test_concurrent_execute_target_submits_exactly_one_order` 🔴 | 2 `submit` con la misma clave |
| `test_position_reversal_is_not_reduce_only` 🔴 | long→short marcado `reduce_only=True` |
| `test_account_freeze_blocks_even_reduce_only` 🔴 | nivel 4 → `SUBMITTED` |
| `test_kill_switch_side_effects_are_idempotent` 🔴 | 4 invocaciones → 4 `exit_all` |
| `test_broker_timeout_is_not_recorded_as_rejected` 🔴 | timeout → `REJECTED` |
| `test_environment_must_be_a_valid_execution_environment` 🔴 | `"lve"` aceptado |
| `test_pretrade_consults_global_trading_mode_and_user_kill_switch` 🔴 | ninguno de los dos está en los 16 checks |
| `test_paper_environment_never_reaches_the_broker` 🔴 | paper llama a `Broker.submit` igual que live |
| `test_risk_limits_and_nav_must_share_currency` 🔴 | `currency` declarada y nunca leída |

### Gobernanza / identidad de mercado

| Test | Debe estar rojo hoy porque |
|---|---|
| `test_illegal_declaration_cannot_be_constructed_directly` 🔴 | `GovernanceDeclaration(PAPER, FULL, NOMINAL)` se construye |
| `test_provider_symbol_maps_to_exactly_one_instrument` 🔴 | mismo `provider_symbol` → dos `instrument_id` |
| `test_instrument_identifiers_are_normalized` 🔴 | `" usdcop "` conserva espacios |
| `test_daily_bar_date_is_anchored_in_utc_before_close_offset` | BL-38 sin implementar (regresión "Sunday pile-up") |

**Criterio de aceptación global**: los ~34 tests anteriores verdes, los 11 rojos actuales de
`tests/unit/test_codex_fabric_contracts.py` verdes, y un gate de CI que (a) importe todos los módulos
nuevos, (b) compare el umbral DSR del catálogo contra el SSOT constitucional, y (c) reproduzca los
fingerprints en un checkout con `core.autocrlf=true` y otro con `false`.
