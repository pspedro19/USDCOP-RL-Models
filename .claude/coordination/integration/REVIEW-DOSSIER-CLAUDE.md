---
title: Dossier de revisión cruzada — lote CLAUDE (13 BLs + 2 candados sueltos)
contract: CTR-REVIEW-DOSSIER-001
status: LIVE
owner: CLAUDE
audience: CODEX (revisor)
measured_at_head: fbf12da1..5672a32f
measured_on: 2026-07-28
date: 2026-07-28
supersedes: none
---

# Dossier de revisión cruzada — lote CLAUDE

**Para qué existe.** Cada uno de estos paquetes tenía su evidencia repartida entre el MD del BL,
el tablero de mutación y ~30 mensajes de canal. Aquí está todo junto, en una fila por BL:
**commit · comando · verde · mutación literal (fichero:línea) · rojo esperado · si necesita
Postgres/Docker**. No añade ni un juicio: el veredicto es tuyo.

---

## Nota de lectura — tres cosas antes de empezar

**(a) Los `verde:` se midieron HOY, el 2026-07-28, en esta máquina.** Pueden variar en tu entorno
si tienes otras dependencias instaladas: varios de estos ficheros de test son **compartidos entre
BLs** y crecen cuando cierra un BL vecino (`test_forecasting_caveat_present.py` lo comparten
BL-01/02/03/04/06; `test_trial_ledger.py` lo comparten BL-09/11). Un verde distinto del declarado
**no invalida el rojo**: lo que hay que comparar es el conteo de `failed`.
Dos verdes ya se han movido desde que se escribieron los MDs, y está anotado en su fila.

**Los números de línea son del momento de medición y envejecen.** Hay dos ingenieros commiteando
a la vez y al cerrar este dossier ya había WIP ajeno sobre dos de los ficheros
(`generate_interpretability.py`, `app/api/backtest/route.ts`). Lo que NO envejece es la **cadena
literal** de cada receta: el corredor busca por texto exacto y **aborta si no aparece exactamente
una vez**, en vez de mutar la línea equivocada. Si un `--bl` aborta con `ocurrencias=0`, el código
cambió respecto a esta transcripción y toca revisarlo a mano.

**(b) Postgres / Docker — comprobado ejecutando, no supuesto.**

| | |
|---|---|
| **NO necesitan Postgres ni Docker** | BL-05, BL-06, BL-09, BL-11, BL-13, BL-14, BL-20, BL-25, BL-31, BL-32, BL-36, BL-39, PASSPORT-HARDENING, SYNTH-503 — **14 de 15**. Verificado: los 15 comandos se corrieron enteros y ningún fichero de test salvo uno menciona `psycopg`/`sqlalchemy`/`testcontainers`. |
| **Postgres OPCIONAL (degrada a skip, nunca a rojo)** | **BL-42** — `test_return_units.py` trae 3 tests de convención de unidades contra la DB real. Sin Postgres alcanzable **skipean** con motivo escrito (`postgres unreachable (OperationalError)` y `BL42_REQUIRE_DB not set`) y el resto va verde: `28 passed, 3 skipped`. La mutación de BL-42 **no** toca esos 3: muerde en los 28 offline. |
| **Ninguno necesita Docker** | Ninguno de los 15 levanta contenedores, ni ataca puertos, ni sirve el dashboard. El test de SYNTH-503 nombra `localhost:8003` pero **mockea `globalThis.fetch`** para simular el backend caído: no abre un socket. |

> Transparencia sobre el entorno de medición: en esta máquina el stack **sí** estaba arriba
> (`usdcop-dashboard`, `usdcop-airflow-*`, puerto 5432 abierto) y aun así los 3 tests de DB de
> BL-42 skipearon porque la conexión falló con `OperationalError`. O sea: lo medido es el caso
> "sin DB usable", que es el tuyo si no levantas nada. **Nada aquí exige `make docker-up`.**

**(c) Qué NO cubre este dossier.** Que un rojo cuadre en conteo no dice que la aserción que cae
sea la garantía que el BL promete. Eso es exactamente tu trabajo y no lo automatiza nadie. Las
mutaciones de abajo son las **declaradas por quien cerró cada BL**; si se te ocurre una mejor,
esa es la revisión de verdad.

---

## Tabla maestra

`Docker/PG` = necesita el stack levantado. `M` en la columna `auto` = mutación **manual**, no está
en el corredor (`scripts/validation/run_mutation_review.py`) y hay que aplicarla a mano.

| BL | commit | comando (cwd) | verde esperado | mutación (fichero:línea) | rojo esperado | Docker/PG | auto |
|---|---|---|---|---|---|---|---|
| **BL-05** | `a6c83a4f` | `npx vitest run tests/unit/components/ProductionView.paper-ledger.test.tsx tests/unit/components/PaperCandidatesPanel.test.tsx` *(usdcop-trading-dashboard/)* | `16 passed` | `components/gm/views/ProductionView.tsx:993` — antepone `false &&` al montaje de `<PaperCandidatesPanel>` | `1 failed` — *"Unable to find role=table and name /candidatas/i"* (el A/B v11/v12/v14 desaparece de `/production`) | no | sí |
| **BL-06** | `5ec84a19` | `python -m pytest tests/regression/test_forecasting_caveat_present.py -q` | `31 passed` (28 al cerrar; el fichero es compartido con BL-01/02/03/04) | **fichero NUEVO** `usdcop-trading-dashboard/lib/telemetry/RogueProbe.tsx` con la evasión completa (ver §Recetas) | `1 failed, 27 passed` — 3 hits: `api/production/approve`, `api/execution`, verbo de orden `comprar` | no | **M** |
| **BL-09** | `cb1241b2` | `python -m pytest tests/regression/test_trial_ledger.py -q` | `26 passed` | `scripts/validation/check_trial_ledger.py:706` — inserta `return errors` justo tras el primer check de `run_all_checks()` | `10 failed` — un caso por cada check desconectado del gate: *"check_hash_chain NO está cableado en run_all_checks(): sus violaciones no llegan al exit code del gate"* | no | sí |
| **BL-11** | `cb1241b2` | `python -m pytest tests/regression/test_trial_ledger.py -q` | `26 passed` | `scripts/validation/check_trial_ledger.py:252` — `return []` como primera línea de `check_families` | `2 failed` — *"una celda que referencia un trial inexistente debe ser rechazada"* + *"cobrar 1 trial menos del que se miró es exactamente la fuga que BL-11 cierra"* | no | sí |
| **BL-13** | `0645dcd1` | `python -m pytest tests/regression/test_strategy_manifests.py tests/regression/test_feature_contracts.py -q` | **declarado** `49 passed, 1 xfailed` · **hoy** `50 passed` ⚠️ | `src/identity/source_hash.py:35` — `canonical_lf` → `return data` (se elimina la normalización CRLF→LF) | `8 failed` — DOS ficheros de test caen desde UNA línea de producción (es el rojo que demuestra que la circularidad murió) | no | sí |
| **BL-14** | `0645dcd1` | `python -m pytest tests/regression/test_strategy_manifests.py -q` | **declarado** `23 passed, 1 xfailed` · **hoy** `24 passed` ⚠️ | `config/strategy_manifests/usdcop.yaml:67-75` — `current_model_snapshot` INVENTADO (pointer/as_of/hashes-de-ceros/registered_in) | **`<sin registrar>`** — el MD dice literal *"conteo exacto de failed sin registrar — pendiente de re-ejecutar"*. Lo que sí está declarado: cae `test_current_model_snapshot_is_resolvable` y las CUATRO cláusulas caen por separado | no | sí |
| **BL-20** | `955374d0` | `python -m pytest tests/unit/test_interpretability_artifacts.py -q` | `20 passed` | `scripts/analysis/generate_interpretability.py:467` — `phi = Z * coefs` → `phi = np.ones_like(Z)` | `3 failed` — aditividad (`sum(mean_shap)+base=21` vs `mean(pred)=-1.55e-05`), no-degeneración y acoplamiento al modelo | no | sí |
| **BL-25** | `955374d0` | `python -m pytest tests/unit/test_system_health.py -q` | `25 passed` | `src/monitoring/system_health.py:338` — `TRACKING_ERROR_SIGMA` → `TRACKING_ERROR_SIGMA * 1000` | `1 failed` — el gemelo de 3.50σ deja de disparar `withdrawal` (GREEN != ORANGE) | no | sí |
| **BL-31** | `014687cc` | `python -m pytest tests/regression/test_strangler_cop.py -q` | `41 passed` | `src/strangler/parity.py:323` — `if obs.verdict is ParityVerdict.MATCH:` → `is not ParityVerdict.MISMATCH:` | `2 failed` — una observación `INVALID` deja de romper la racha ⇒ `PARITY_GREEN` **sin haber comparado nada** | no | sí |
| **BL-32** | `2a608feb` | `python -m pytest tests/unit/test_passport_contract.py -q` | `83 passed` | `src/contracts/passport.py:432` — el bucle `for block, fields in PASSPORT_BLOCK_FIELDS.items():` deja de recorrer nada | `8 failed, 75 passed` | no | sí |
| **BL-36** | `2a608feb` | `python -m pytest tests/regression/test_db_truth_matrix.py -q` | `8 passed` | `.claude/specs/platform/db-truth-matrix.md:418` — invierte la decisión sobre `bi.fact_*`: `DEPRECATED / 7 ficheros` → `AUTORITATIVA (escritor único) / 0 ficheros` | **`<sin registrar>`** — el MD dice literal *"conteo exacto de failed sin registrar"*. Declarados por nombre: `test_prose_reference_counts_match_the_measured_inventory`, `test_no_table_is_declared_sole_writer_of_an_attribute_it_shares`, `test_deprecated_tables_disclose_the_readers_that_still_exist` | no | sí |
| **BL-39** | `014687cc` | `python -m pytest tests/regression/test_feature_contracts.py -q` | `26 passed` | `src/forecasting/enhance_v2.py:113` — quita el `.shift(1)` de `rate_diff_ibr_ust2y` (fuga temporal T-1 pura) | `4 failed` — 2 muros de hash + los 2 muros nuevos de **causalidad** | no | sí |
| **BL-42** | `955374d0` | `python -m pytest tests/regression/test_return_units.py -q` | `28 passed, 3 skipped` (los 3 skips **exigen Postgres**) | `scripts/pipeline/train_and_export_smart_simple.py:1052` — `round(total_return, 2)` → `round(total_return / 100.0, 6)` | `2 failed` — *"total_return_pct=0.144616 for a ledger that compounds 10_000 → 11446.16: expected 14.46 PERCENTAGE POINTS"* + el detector de disfraz decimal sobre la salida en memoria | **PG opcional** (sin él: skip, no rojo) | sí |
| **PASSPORT-HARDENING** *(los dos validadores)* | `225e3524` | `python -m pytest tests/unit/test_passport_contract.py -q` **y** `npx vitest run tests/unit/contracts/passport-contract.test.ts` *(usdcop-trading-dashboard/)* | `83 passed` (Py) · `41 passed` (TS) | **el mismo candado que BL-32**: `src/contracts/passport.py:432`, el bucle deja de recorrer nada | Py: `8 failed, 75 passed`. TS: **`<sin registrar>`** — no se registró ninguna mutación del lado TypeScript | no | sí *(vía BL-32)* |
| **SYNTH-503** *(3 rutas sintéticas fail-closed)* | `46119274` (rojo) / `a0a15e91` (fix) | `npx vitest run tests/unit/api/synthetic-backtest-honesty.test.ts` *(usdcop-trading-dashboard/)* | `4 passed` | `usdcop-trading-dashboard/app/api/backtest/route.ts:88-94` — revierte el `catch` al fallback sintético pre-fix (200 + `success:true` + `source:'generated'`) | `1 failed, 3 passed` — *"HTTP 200 con N trades FABRICADOS por synthetic-backtest.service.ts y aun así declara success:true"* | no | sí *(solo ruta 1 de 3)* |

⚠️ **Los dos verdes marcados se movieron y NO es una degradación**: el `xfail(strict=True)` de
`test_component_forecast_trial_ids_resolve_in_ledger` se retiró en `8005ffea` al cerrar BL-10
(CODEX) y poner linaje FT real en los manifiestos. `49 passed + 1 xfailed` → `50 passed`, y
`23 + 1 xfailed` → `24 passed`. El conteo de **failed** de las mutaciones no cambia.

---

## Recetas literales de mutación

Copia-pega. Restaurar siempre con `git checkout -- <fichero>`.

### BL-05 — `usdcop-trading-dashboard/components/gm/views/ProductionView.tsx:993`
```diff
-              {paperLedger && <PaperCandidatesPanel ledger={paperLedger} />}
+              {false && paperLedger && <PaperCandidatesPanel ledger={paperLedger} />}
```
**muta-2 (MANUAL, no en el corredor)**: añadir en `PaperCandidatesPanel.tsx` una celda extra
`"Sharpe 3.35 · p=0.006"` en las filas con `n_trades=11` ⇒ `1 failed` (la fila publica Sharpe con
N<20, prohibido por `quant-constitution.md` §6).

### BL-06 — fichero NUEVO (MANUAL)
Crear `usdcop-trading-dashboard/lib/telemetry/RogueProbe.tsx` con la evasión completa —
concatenación + interpolación + minúsculas, que es lo que la versión previa del candado **no**
atrapaba:
```tsx
const P = '/api/production';
export function RogueProbe() {
  const a = () => fetch(`${P}/appro` + 've', { method: 'POST' });
  const b = () => fetch('/api/exec' + 'ution/orders', { method: 'POST' });
  return <button onClick={() => { a(); b(); }}>Comprar ahora</button>;
}
```
⇒ `1 failed, 27 passed`, con 3 hits nombrados y su línea. Borrar el fichero para restaurar.
**Por qué es manual**: el corredor sustituye texto dentro de ficheros existentes; crear y borrar
ficheros de producción es otra clase de operación y prefiero no darle esa capacidad a un script
escrito por la parte revisada.

### BL-09 — `scripts/validation/check_trial_ledger.py:706`
```diff
     errors += check_schema_and_ids(records)
+    return errors
     errors += check_hash_chain(records)
```

### BL-11 — `scripts/validation/check_trial_ledger.py:252`
```diff
 def check_families(records: list[dict], families_dir: Path = FAMILIES_DIR) -> list[str]:
+    return []
     errors = []
```

### BL-13 — `src/identity/source_hash.py:35`
```diff
 def canonical_lf(data: bytes) -> bytes:
-    return data.replace(b"\r\n", b"\n")
+    return data
```
**mutas 2/3/4 (MANUALES)**, declaradas en el MD y ya verdes de origen: `_frozen_surfaces()` → `{}`
en `scripts/pipeline/normalize_champions.py`; quitar el `raise` de *surface* desconocido; borrar
`surface` del `registry.json` que sirve el dashboard ⇒ tres rojos, dos de ellos ejecutando el
script end-to-end.

### BL-14 — `config/strategy_manifests/usdcop.yaml:67-75`
```diff
-    pointer: outputs/forecasting/h5_weekly_models/latest/
-    registered_in: MLflow run de forecast_h5_l3_weekly_training (tags git_commit,
-      iso_week, contract FC-H5-L3-001)
-    as_of: '2026-07-06'
-    artifacts_sha256_16:
-      ridge_h5.pkl: 8e4618c4d26d3af9
-      ...
+    pointer: no/existe/
+    registered_in: ninguna parte
+    as_of: '1999-01-01'
+    artifacts_sha256_16:
+      ridge_h5.pkl: '0000000000000000'
+      ...
```
**muta-2 (MANUAL)**: sustituir el placeholder `pending-BL-10` de `forecast_trial_ids` por `FT-####`
reales ⇒ `1 failed` por `XPASS(strict)`. **Esta mutación ya no aplica**: BL-10 cerró y el
placeholder fue sustituido por linaje FT real en `8005ffea`, que es el mismo cambio que retiró el
`xfail`. Se deja documentada para que el historial se entienda.

### BL-20 — `scripts/analysis/generate_interpretability.py:467`
```diff
-        phi = Z * coefs
+        phi = np.ones_like(Z)
```
**muta-2 (MANUAL)** — rama TreeSHAP, `generate_interpretability.py:874`: `phi = np.ones_like(phi)`
justo tras `phi, bias = shap_fn(mdl, Xte)` ⇒ `1 failed`
(`additivity_max_abs_err=21` contra umbral `1e-06`).

### BL-25 — `src/monitoring/system_health.py:338`
```diff
-                if te_z > TRACKING_ERROR_SIGMA:
+                if te_z > TRACKING_ERROR_SIGMA * 1000:
```
**mutas 2/3 (MANUALES)**: `/1000` en la misma línea ⇒ `1 failed` (el gemelo de 2.47σ deja de estar
verde, ORANGE != GREEN); y poner a `0.0` el ruido del escenario en el test (vuelve a
`live = paper − constante`) ⇒ `1 failed` con *"sd(d)=5.13e-19: diferencia casi constante, z-score
vacío"*. Las dos juntas son las que acotan el 3 **por arriba y por abajo**.

### BL-31 — `src/strangler/parity.py:323`
```diff
-        if obs.verdict is ParityVerdict.MATCH:
+        if obs.verdict is not ParityVerdict.MISMATCH:
```

### BL-32 / PASSPORT-HARDENING — `src/contracts/passport.py:432`
```diff
-    for block, fields in PASSPORT_BLOCK_FIELDS.items():
+    for block, fields in {}.items():
```
**muta-2 (MANUAL)**: recortar de 8 a 3 la lista de bloques obligatorios del passport ⇒
`5 failed, 60 passed` (medido en el cierre previo, `2a608feb` — no re-medido hoy).

### BL-36 — `.claude/specs/platform/db-truth-matrix.md:418`
```diff
-| BI `fact_*` | DEPRECATED | E1+E2: cableadas pero (según perfil) 0 filas | 7 ficheros referencian `bi.fact_forecasts` |
+| BI `fact_*` | AUTORITATIVA (escritor único) | E1+E2: cableadas pero (según perfil) 0 filas | 0 ficheros referencian `bi.fact_forecasts` |
```

### BL-39 — `src/forecasting/enhance_v2.py:113`
```diff
-            macro["rate_diff_ibr_ust2y"] = (macro[IBR] - macro[UST2Y]).shift(1)
+            macro["rate_diff_ibr_ust2y"] = (macro[IBR] - macro[UST2Y])
```
**muta-2 (MANUAL)** y es **la mitad que importa**: con la fuga puesta, RE-REGISTRAR el hash del
catálogo (`config/features/feature_catalog.yaml`, 5 ocurrencias,
`b601ae271c8e2b5f` → el nuevo) ⇒ `2 failed, 24 passed`: los dos muros de **hash vuelven a verde
con la fuga dentro** y los dos de **causalidad siguen rojos**. Ése es el rojo que demuestra que el
candado ya no depende del drift de hash. No está en el corredor porque toca un segundo fichero
con un valor que hay que calcular en el momento.

### BL-42 — `scripts/pipeline/train_and_export_smart_simple.py:1052`
```diff
-        "total_return_pct": round(total_return, 2),
+        "total_return_pct": round(total_return / 100.0, 6),
```

### SYNTH-503 — `usdcop-trading-dashboard/app/api/backtest/route.ts:88-94`
Sustituir el `catch (error)` fail-closed por el fallback sintético de antes del fix (200,
`success: true`, `source: 'generated'`, sin marcador de primer nivel). El corredor lleva la
sustitución exacta; a mano basta con devolver
`NextResponse.json({ success: true, source: 'generated', trades: generateSyntheticTrades({...}), ... })`
desde ese `catch`.
**Las otras dos rutas quedan MANUALES**: `app/api/backtest/stream/route.ts` (el fallback vive en
dos sitios, GET y POST, y hay que emitir eventos SSE `trade`+`result`) y
`app/api/replay/load-trades/route.ts` (tres disparos: `!ok`, `AbortError`, `ECONNREFUSED`). Cada
una tiene su propio test en el mismo fichero, así que se pueden mutar por separado.

---

## Corredor de mutaciones (opcional)

`scripts/validation/run_mutation_review.py` aplica, ejecuta, restaura (verificando sha256) y
tabula. **Es una conveniencia escrita por la parte revisada, no una autoridad**: todo lo que hace
está en las recetas de arriba, y si no te fías de él, ése es el camino. No emite veredictos.

```
python scripts/validation/run_mutation_review.py --list
python scripts/validation/run_mutation_review.py --bl BL-09
python scripts/validation/run_mutation_review.py            # los 13 automatizados
```

Se **niega a correr** si el fichero que va a mutar tiene cambios sin commitear
(`git status --porcelain <fichero>`), imprime el diff antes de aplicarlo, y aborta ruidosamente
si el sha256 tras restaurar no coincide con el de antes.

**Automatizados (13)**: BL-05, BL-09, BL-11, BL-13, BL-14, BL-20, BL-25, BL-31, BL-32, BL-36,
BL-39, BL-42, SYNTH-503 (+ PASSPORT-HARDENING, que es el mismo candado que BL-32).
**Manuales (1 BL entero + 9 mutaciones secundarias)**: BL-06 (fichero nuevo) y las `muta-2/3/4`
listadas arriba.

---

## Corrida real del 2026-07-28 (los 13 automatizados)

Medido con el repo entre `fbf12da1` y `5672a32f` (hay dos ingenieros commiteando a la vez);
**ninguno de los commits intermedios toca ninguno de los 12 ficheros mutados** — verificado con
`git diff --name-only fbf12da1..5672a32f`. Sin Postgres alcanzable. Árbol limpio en los 12
ficheros mutados al empezar; **13/13 restauraciones verificadas por sha256, 0 corruptas**.

> **Aviso de concurrencia**: `scripts/analysis/generate_interpretability.py` (el fichero de BL-20)
> quedó con WIP sin commitear a las 23:50:23, **3½ minutos después de terminar esta corrida**
> (23:46:47) y por mano ajena — no lleva la mutación `np.ones_like` por ningún lado. Mientras ese
> WIP siga ahí, `--bl BL-20` **abortará** en vez de correr, que es exactamente lo que debe hacer:
> un rojo medido sobre WIP no corresponde a ningún commit. Commitea o descarta y vuelve a lanzarlo.

| BL | esperado (declarado) | obtenido | coincide |
|---|---|---|---|
| BL-05 | `1 failed` | `1 failed, 15 passed` | sí |
| BL-09 | `10 failed` | `10 failed, 16 passed` | sí |
| BL-11 | `2 failed` | `2 failed, 24 passed` | sí |
| BL-13 | `8 failed, 41 passed` | `8 failed, 42 passed` | **failed sí, passed +1** (ver ⚠️: el `xfail` retirado) |
| BL-14 | `<sin registrar>` | `1 failed, 23 passed` | n/a — **queda registrado**: cae `test_current_model_snapshot_is_resolvable` por la cláusula `pointer` |
| BL-20 | `3 failed` | `3 failed, 17 passed` | sí |
| BL-25 | `1 failed` | `1 failed, 24 passed` | sí |
| BL-31 | `2 failed` | `2 failed, 39 passed` | sí |
| BL-32 | `8 failed, 75 passed` | `8 failed, 75 passed` | sí |
| BL-36 | `<sin registrar>` | `2 failed, 6 passed` | n/a — **queda registrado**, con un matiz abajo |
| BL-39 | `4 failed` | `4 failed, 22 passed` | sí |
| BL-42 | `2 failed` | `2 failed, 26 passed, 3 skipped` | sí |
| SYNTH-503 | `1 failed, 3 passed` | `1 failed, 3 passed` | sí |

**Matiz honesto de BL-36**: el MD nombra **tres** tests como esperados y solo caen **dos**
(`test_prose_reference_counts_match_the_measured_inventory` y
`test_no_table_is_declared_sole_writer_of_an_attribute_it_shares`). El tercero,
`test_deprecated_tables_disclose_the_readers_that_still_exist`, **no puede caer con esta
mutación** porque la mutación borra la etiqueta `DEPRECATED` de esa fila: sin fila deprecada no hay
nada que ese muro tenga que comprobar. Es un solapamiento del enunciado del MD, no un candado
flojo — pero para verlo hay que mutar **solo el conteo** (`7` → `0`) dejando `DEPRECATED` en su
sitio, y eso es una segunda mutación que nadie registró.

---

## Estado de cross-review al momento de escribir esto

No forma parte de la verificación; está para que no revises dos veces lo mismo.

| Ya dictaminado por CODEX | BLs |
|---|---|
| **APROBADO** | BL-09, BL-11 (CXD-089 — CODEX ejecutó las mutaciones: 10, 3 y 2 tests muertos, SHA de restauración exacto) |
| **APROBADO_PARCIAL** | BL-20, BL-25, BL-42 (candados aprobados; alcance recortado o con exigencias abiertas — aprobar el candado no es aprobar el alcance) |
| **Sin dictamen aún** | BL-05, BL-06, BL-13, BL-14, BL-31, BL-32, BL-36, BL-39, PASSPORT-HARDENING, SYNTH-503 |
