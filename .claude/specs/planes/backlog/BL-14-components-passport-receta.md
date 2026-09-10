---
kind: roadmap
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-08-05
supersedes: []
code_anchors:
  - config/strategy_manifests/usdcop.yaml
  - scripts/pipeline/train_and_export_smart_simple.py
---

# BL-14 — Bloque components: receta congelada del predictor de v11

**Fuente**: plan 02 §1 / FABRIC §16 · **Ola**: 2 · **Esfuerzo**: M · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
code_hash global del manifiesto cubre fuentes, pero no distingue el componente Ridge/BR ni registra sus model_snapshots semanales.

## Qué falta exactamente
Bloque `components:` (component_id, role=decision_input, spec_fingerprint de la RECETA, retrain_policy=weekly_expanding, current_model_snapshot rotativo, forecast_trial_ids heredados). CI: 'la receta está congelada; todo snapshot queda registrado' — nunca 'los pesos no cambian'.

## Impacto frontend
Passport (BL-32) lo muestra.

## Dependencias
BL-12 (herencia FT).

## Verificación
Manifest test extendido; snapshot semanal nuevo aparece con linaje.

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_strategy_manifests.py -q
verde:   24 passed   (2026-08-05; era `23 passed, 1 xfailed` — el xfail desapareció
         al cerrar BL-10, no al relajar nada)

muta:    config/strategy_manifests/usdcop.yaml — current_model_snapshot INVENTADO:
           pointer: "no/existe/"
           as_of: "1999-01-01"
           artifacts_sha256_16: {ridge_h5.pkl: "0000000000000000"}
           registered_in: "ninguna parte"
espera:  rojo en test_current_model_snapshot_is_resolvable, y las CUATRO cláusulas caen
         POR SEPARADO — **medido y registrado el 2026-08-05**, una mutación por cláusula:
           pointer: no/existe/            -> 1 failed, 23 passed
           registered_in: ninguna parte   -> 1 failed, 23 passed
           as_of: '1999-01-01'            -> 1 failed, 23 passed
           ridge_h5.pkl: '0000000000000000' -> 1 failed, 23 passed
         (las cuatro en test_current_model_snapshot_is_resolvable; restauración
          verificada byte-exacta con `git checkout --` tras cada una)

muta-2:  linaje FT — el placeholder `pending-BL-10` YA no existe (BL-10 cerró el
         2026-07-28 y el xfail(strict) se retiró: la suite no tiene xfailed).
         El ataque vigente es al revés, sobre el linaje REAL:
           FT-0048 -> FT-9999 (id que no está en el ledger)
             -> 1 failed: test_component_forecast_trial_ids_resolve_in_ledger
           renombrar la clave forecast_trial_ids_legacy (equivale a borrarla)
             -> 2 failed: + test_component_declares_forecast_lineage_key
```

**`registered_in` de v12/v14: era DISEÑO, no deuda (medido 2026-08-05).** BL-13 lo pasó a
esta ficha como deuda («`usdcop_v12/v14` declaran un `current_model_snapshot` rotativo SIN
`registered_in`»), y la sospecha razonable era un criterio verde **por ausencia** — la clase
de falso verde que más caro sale. Medido: la cláusula es **condicional a propósito** y las
DOS ramas muerden:

    campeona (v11) SIN registered_in                 -> 1 failed, 23 passed
    candidata paper (v12) CON registered_in inventado -> 1 failed, 23 passed
    (ambas en test_current_model_snapshot_is_resolvable; restauración byte-exacta)

La regla es «quien SIRVE los pesos debe declarar dónde queda registrada cada corrida»: la
campeona está obligada, una candidata paper no; pero si la declara, tiene que nombrar un DAG
real de `dag_registry.py`. Omitirla no es un agujero — declararla en falso sí, y eso cae.
**Deuda cerrada por medición, no por decreto.**

**El conteo que faltaba escondía algo (2026-08-05).** Esta ficha declaraba que las cuatro
cláusulas caían «POR SEPARADO» pero dejaba el conteo sin registrar. Al medirlo aparece el
matiz: mutar **las cuatro a la vez** da **1 failed**, no cuatro — los asserts son secuenciales
y el test muere en la primera cláusula (`pointer`), dejando las otras tres **sin demostrar**.
La afirmación sólo es cierta mutando **una cláusula por vez**, que es como se ha medido ahora.
Un conteo sin registrar no es un detalle de forma: era exactamente el sitio donde vivía la
diferencia entre «cuatro candados» y «un candado con cuatro cláusulas de las que sólo se
probaba la primera».

**Historial honesto**: hasta el 2026-07-28 este candado **comprobaba que el bloque EXISTE, no
que fuera CIERTO**. Un `current_model_snapshot` completamente inventado —exactamente el de
la mutación de arriba— daba **20 passed, idéntico**, porque `test_composite_declares_components`
solo verifica que la CLAVE está. Y borrar el linaje FT del componente tampoco mordía: su
valor real hoy es el placeholder `pending-BL-10` y ningún candado lo detectaba.

**Dos decisiones que conviene leer**: (1) resolubilidad REAL, no forma — el snapshot se cruza
contra el snapshot de normalización (otro artefacto EN GIT) que sella independientemente el
hash del scaler, el `ordered_feature_hash` y el directorio del puntero; dos registros del
mismo binario que no coincidan significan que uno miente, y funciona en CI sin los `.pkl`,
que están gitignored. (2) `as_of <= manifest_frozen_at`, **NO** `>=`: el dato real es al revés
(`as_of 2026-07-06` es la última corrida L3 ANTES del sello `2026-07-28`), y la invariante
honesta es que un manifiesto congelado solo puede declarar un snapshot que YA EXISTÍA al
sellarlo.

**Estado del linaje FT a 2026-08-05: RESUELTO.** El párrafo siguiente describe el estado
del 2026-07-28 y se conserva como historia. BL-10 cerró (`IMPLEMENTED`), el placeholder
`pending-BL-10` fue sustituido por FT-0001..FT-0048 **derivados** del ledger (misma consulta
que valida `check_trial_ledger.py::check_provenance_wall`, no elegidos a mano), el
`xfail(strict=True)` se retiró y su test es hoy normal y verde. Verificado por mutación
arriba: un FT inexistente y la clave borrada ponen rojo. La suite no tiene ningún `xfailed`.

**Cross-review de CODEX (2026-08-05): APROBADO el flip**, con un ataque independiente que
mejora mi propia lectura. Le pregunté si ensanchar `manifest_frozen_at` a `2026-08-05` con el
re-freeze **relajaba** el candado `as_of <= manifest_frozen_at`. Respuesta medida: mutar
`as_of 2026-07-06 -> 2026-08-01` (dentro del nuevo sello, o sea **pasando** la comparación de
fechas) sigue dando **1 failed**, y cae en la igualdad cruzada contra
`normalization_snapshot.training.as_of`. El candado efectivo no es la fecha del sello sino la
**conjunción** reloj/sello + igualdad con el snapshot versionado; mover ambos registros y sus
hashes sería un re-freeze gobernado, no un bypass silencioso.

**El linaje FT vale hoy `pending-BL-10`, y BL-10 es de CODEX**, así que se entrega como
`xfail(strict=True)` con la razón escrita: se pondrá ROJO el día que BL-10 cierre y el
placeholder deje de ser aceptable. Va acompañado de un test NO-xfail que exige la presencia
de la clave y que el aplazamiento apunte a un BL que exista en disco — cierra hoy el hueco de
"borrarlo no muerde" sin forzar el rojo de un BL ajeno.

## Notas constitución
Precisión FABRIC §16: v11 reentrena cada domingo — lo congelado es la receta.
