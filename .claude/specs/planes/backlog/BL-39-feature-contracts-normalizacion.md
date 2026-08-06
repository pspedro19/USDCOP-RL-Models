---
kind: roadmap
status: PARTIAL
version: 1.0.2
last_verified: 2026-08-05
supersedes: []
code_anchors:
  - config/features/feature_catalog.yaml
  - scripts/validation/validate_feature_catalog.py
  - tests/regression/test_feature_contracts.py
  - src/core/contracts/feature_contract.py
  - scripts/pipeline/train_and_export_smart_simple.py
  - src/forecasting/enhance_v2.py
---

# BL-39 — Feature contracts por estrategia-versión + normalización al artefacto

**Fuente**: Plan Consolidado §1.2-1.4 / DATA-STRATEGY §40-48 (D4) · **Ola**: 2-3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built/perfil 2026-07-27)
config.feature_definitions (30) mezcla definición+normalización+código: z-scores con media/sigma HARDCODEADAS ((vix-21.16)/7.89 — dependen del período de entrenamiento), python_function/sql_formula como strings no autoritativos, source_table apuntando a tablas por renombrar. El contrato de 20 del RL es el patrón correcto a generalizar.

## Qué falta exactamente
Catálogo estable (feature_id, causality_policy, source_contract, transformation, lookback, code_reference+code_hash) en Git; feature_set(strategy_version, feature_id, order, required) por estrategia; normalization_snapshot_id → artefacto MinIO/MLflow (mean, std, training_cutoff, semantic_hash). La matriz por estrategia de §41-47 se convierte en fixture de CI (v11 = 25 feats; las rule-based declaran su set mínimo — MA200 solo close).

## Impacto frontend
La vista SHAP admin (BL-20) consume el feature_set versionado.

## Dependencias
Coordina con BL-14 (components) y BL-16. NO toca v11 en runtime: el snapshot actual se registra como legacy_v1 bit-idéntico.

## Verificación
Reproducir la señal v11 de la última semana desde feature_set+snapshot == bit-check; CI rechaza features sin causality_policy o sin prior de signo (Anexo A.4).

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/regression/test_feature_contracts.py -q
verde:   24 passed, 2 skipped   (2026-08-05; NO "26 passed" — ver más abajo)

muta:    quitar el .shift(1) de un feature macro (fuga temporal T-1 pura)
espera:  4 failed — 2 muros de hash + los 2 muros nuevos de CAUSALIDAD

muta-2:  RE-REGISTRAR el hash del catálogo con la fuga dentro
         (5 ocurrencias, b601ae271c8e2b5f -> 45b16e0da0928467)
espera-2: 2 failed, 24 passed — los dos muros de hash VUELVEN A VERDE con la fuga dentro
         y los dos tests de causalidad SIGUEN ROJOS.
         Restaurado: producción y catálogo sin diff.
```

**Corrección 2026-08-05: tampoco hay evidencia LOCAL, y el número de arriba estaba mal.** Más
abajo esta ficha ya declaraba que el bit-check "skipea siempre en CI" y lo llamaba *evidencia
local-only*. Medido: **no existe esa evidencia local**. La corrida real aquí es
**24 passed, 2 skipped** —no las `26 passed` que esta ficha publicaba—, y los dos `skipped` son
`test_disk_artifacts_are_bit_identical_to_frozen_registration` y
`test_bitcheck_v11_signal_from_feature_set_plus_snapshot`, o sea la primera línea de esta
Verificación. No hay artefactos H5 **ni en el working tree ni dentro del contenedor de Airflow**
(`find /opt -name feature_cols_h5.json` ⇒ 0 resultados), y la metadata de Airflow sólo registra
ejecuciones de tres DAGs (`control_system_health`, `core_l0_01_ohlcv_backfill`,
`rbac_entitlements_daily`): H5-L3 nunca ha entrenado en este entorno.

La diferencia importa: "skipea en CI pero se verifica en local" describe una evidencia que existe
en otro sitio; "no hay artefactos en ningún sitio" describe una que **no se ha producido nunca**.
Lo que sostiene el `PARTIAL` es la mitad **contractual** del criterio —catálogo, priors de signo,
causalidad, muros de hash y la mutación de look-ahead—, que sí corre. Lo que convertiría el
bit-check en evidencia es una corrida real de `train_and_export_smart_simple.py` que produzca los
artefactos congelados; entonces se publica `26 passed` **medido**. Hasta entonces el skip se
declara y no se cuenta como verde (`K-051`).

### El bit-check NO se cierra regenerando: por qué, medido (2026-08-05)

Esta ficha proponía como salida «una corrida real de `train_and_export_smart_simple.py` que
produzca los artefactos congelados; entonces se publica `26 passed` medido». **Se fue a hacer y
no se puede, y la razón es más interesante que el obstáculo.**

Primero, el productor no es ése: los `.pkl` los escribe
`airflow/dags/forecast_h5_l3_weekly_training.py:299-349` (`MODELS_DIR = outputs/forecasting/
h5_weekly_models/latest/`), no el script de export.

Segundo, y decisivo — `bitcheck_v11_signal.py:112-119` compara el hash del disco contra
`manifest.components[0].current_model_snapshot.artifacts_sha256_16`, que es la **registración
as-of 2026-07-06**. Una corrida de hoy no puede coincidir por **dos** motivos independientes:

| Motivo | Medido |
|---|---|
| Versión de sklearn | snapshot declara `sklearn_version_at_fit: 1.9.0` (contenedor Airflow); **local es 1.6.1** → los bytes del pickle difieren aunque el modelo sea idéntico |
| Ventana de entrenamiento | v11 usa ventana **expansiva** hasta el último viernes; hoy es un mes más larga que el `as_of 2026-07-06` → otros pesos |

Y aquí está el fondo: **este test no se puede satisfacer regenerando, por construcción.** Si se
regenera y no casa, es rojo; si se ajustara algo para que casara, sería **circular** — la misma
trampa que esta ficha ya documenta en su `muta-2` (re-registrar el hash devuelve el verde con la
fuga dentro). El bit-check sólo es evidencia **allí donde el pipeline congelado corrió de verdad y
dejó sus artefactos**. Aquí nunca corrió, así que el skip es la respuesta honesta, no una tarea
pendiente de código.

**Consecuencia para el cierre**: lo que falta de BL-39 no es trabajo de código sino que el pipeline
H5-L3 se ejecute en un entorno con el contenedor de Airflow. Mientras tanto el `PARTIAL` lo
sostiene la mitad contractual —que sí corre y sí muerde— y el skip queda declarado (`K-051`), nunca
contado como verde.

### C032 — identidad por activo y serie física (2026-08-05)

C032 cerró el falso verde medido por CLD-503: la resolución global por `feature_id` ligaba
`close` de BTCUSDT, XAUUSD y SPX500 al contrato USDCOP en `cop_per_usd`. El catálogo v2 exige
`(asset_id, feature_id)`, `series_id`, fuente de mercado discriminada por activo y resolución
exact-one de todos los feature sets. DXY/WTI/VIX/UST10Y-2Y usan el `canonical_name` del SSOT macro;
la paridad física cubre unidad/fuente/transformación/código, pero excluye correctamente el prior
por consumidor y la materialización local `asbuilt_source`.

TDD medido: antes de implementar, **6 failed / 24 passed / 2 skipped**. Después: suite focal
**31 passed / 2 skipped** y validador CLI verde sobre el catálogo real. El estado sigue `PARTIAL`:
C032 no fabrica los artefactos H5 ausentes y los dos bit-checks continúan en skip explícito.

**Historial honesto**: hasta el 2026-07-28 el look-ahead **solo se detectaba por DRIFT DE
HASH**. Quitar el `.shift(1)` de un feature macro daba rojos que decían *"drifted from the
registered catalog hash"*; **re-registrando el hash, la suite volvía a VERDE con la fuga
dentro**. Es decir: ningún test detectaba el look-ahead por sí mismo. El muro nuevo es
semántico e independiente del hash: frame `MACRO_DAILY_CLEAN` sintético con UN salto de nivel
en T, y se exige `feature[T] == spread[T-1]` (el salto es invisible), `feature[T+1] ==
spread[T]` (la feature no está muerta) y la igualdad T-1 en TODAS las barras. Un tercer test
es un meta-assert: la sección **no puede leer** `_sha16` / `sha256_16` / `_catalog()`, o sea
que no puede volver a apoyarse en el hash.

**Cobertura declarada, no total** (escrito en el propio fichero, no es olvido): cubre
`rate_diff_ibr_ust2y` y `term_spread`. NO cubre `usdmxn/usdclp_ret_1d_lag`
(`include_xlead` default `False`, experimento pre-registrado) ni los cuatro `*_close_lag1` de
`dataset_loader.py` (el lag vive en otro módulo y necesita su propio fixture).

**La otra mitad de la Verificación declarada arriba —el bit-check v11— SKIPEA SIEMPRE en CI**
porque los `.pkl` están gitignored: es evidencia local-only, no de integración continua.

## Notas constitución
Cambiar UNA constante de normalización tras reentrenar sin snapshot versionado = leakage silencioso — exactamente lo que este BL elimina.

## Hallazgo: `required_features` y `ordered_features` nunca se habían cruzado (2026-08-06)

**Registrado aquí por petición de CXD-609** (el hallazgo es interfaz BL-39 × BL-45 y debe constar
en ambas fichas; el gate ejecutable vive en
`tests/regression/test_cross_ssot_feature_declarations.py`, sellado en `e8815afe`).

### Qué se midió

Cada policy declara `inputs.required_features` (lo que EXIGE) y su feature-set declara
`ordered_features` (lo que ALGUIEN materializa). **Nadie los había cruzado nunca**:

| policy | req | ordenadas | exigidas que su set NO declara |
|---|---|---|---|
| `btc_hodl_b1` | 1 | 1 | `realized_vol_20` |
| `gold_trend_simple` | 5 | 1 | `sma_63`, `sma_126`, `sma_252`, `realized_vol_20` |
| `smart_simple_v11` | 3 | **25** | `predictor_return_5d`, `hurst_exponent`, `realized_vol_20d` |
| `spx500_daily_ma200_v1` | 2 | 1 | `ma_200` → **cerrado** (decisión C, `97ebb4c9`+`76423175`) |

`smart_simple_v11` descarta la explicación fácil: declara **25** features ordenadas y aun así
ninguna de sus tres requeridas está entre ellas. `required_features` y `ordered_features` hablaban
de cosas distintas.

**No se puede alegar `derived_in_policy`**: el DSL no tiene ningún operador de ventana (medido), y
las dos policies *coded* lo dicen en su propio código — `gold.py`: *"the policy only consumes
them"*; `btc.py`: `required = ("realized_vol_20",)`. La prosa no exime; sólo eximiría un contrato
estructurado más código que demuestre derivación interna.

### Un verde por vacuidad de mi propia auditoría, declarado

La primera pasada midió *"¿toda `ordered_feature` está en el catálogo?"* y dio **0 sin catalogar** —
verde perfecto. Es verde **porque el denominador es diminuto**: los tres feature-sets rule-based
declaran UNA feature ordenada cada uno y empujan el resto a `derived_in_policy`. Medí la dirección
que no importaba. La que importa da **6 huérfanas ejecutables** (tras cerrar SPX quedan **5**).

### Clasificación (corregida por CXD-609)

Mi conteo inicial de "9 huérfanas homogéneas" era impreciso. Son **dos clases**:
1. **ejecutables** — la policy corre y le falta el input (SPX 1 cerrado, Gold 4, BTC 1);
2. **componentes de `smart_simple_v11`**, que es `SPEC_ONLY`, sin implementación y con
   `governance.required_features_verified: false`. Sus 25 ordenadas son la receta *upstream* del
   predictor y sus 3 requeridas son componentes de decisión *downstream*: exigir subconjunto ahí
   mezclaría capas y daría un **rojo falso**, que gasta la misma credibilidad que un verde falso.
   Tiene test separado que fija su estado declarado.

## `inputs.feature_set_hash` — PILOTO SPX (`91400773`), deuda 3/4 viva

`canonical_policy_payload` incluía el `feature_set_id` pero **no su contenido**: un set podía ganar
o perder features bajo una policy congelada sin que el `policy_hash` se moviera un bit. Peor que el
caso de `max_snapshot_age`, porque el `feature_set_id` **sí** viajaba y daba impresión de cobertura.

Cerrado **sólo para `spx500_daily_ma200_v1`**. `gold_trend_simple`, `btc_hodl_b1` y
`smart_simple_v11` **conservan el hueco a propósito**, con candado que enrojece si esa deuda cambia
sin declararlo. **Esto no es cierre sistémico de la identidad de feature-set** y no debe leerse así.

**BL-39 sigue `PARTIAL`.**

## Deuda cross-SSOT ejecutable = CERO (2026-08-06)

Cerradas las tres policies construibles, cada una con la forma que su caso pedía — **no la misma
receta tres veces**:

| policy | huérfanas | forma | commit |
|---|---|---|---|
| `spx500_daily_ma200_v1` | `ma_200` | **no existía productor**: se escribió uno y el harness dejó de tener su copia | `97ebb4c9` + `76423175` |
| `btc_hodl_b1` | `realized_vol_20` | **productor congelado ya existente**: el catálogo apunta a `build_daily_features`, con contrato de invocación declarado | `f7109afd` + `080305b5` |
| `gold_trend_simple` | `sma_63/126/252`, `realized_vol_20` | **MIXTO**: la vol tenía productor congelado; las tres SMA no existían en ninguna parte | `773c7ccb` |

**Lo que hizo falta para que fuera honesto en los tres casos:**

* **cero fórmulas nuevas donde ya había una congelada.** En BTC y en la vol de Gold el
  `code_reference` apunta al código congelado real, así que el `sha256_16` congela **la fórmula**.
  Yo había propuesto un adaptador que delegara; con él, el hash habría congelado el adaptador y la
  fórmula habría quedado **fuera del muro**.
* **contrato de invocación DECLARADO, no inferido.** `build_daily_features(df) -> df` calcula ~10
  features de golpe; `compute_ma_200(close) -> Series` es otra convención. El catálogo declara
  `producer_contract` + `output_column` y el resolver ramifica por ese valor: inferir por firma
  haría que renombrar un argumento cambiara cómo se invoca a un productor congelado.
* **ninguna feature con ventana parametrizable.** Una `sma(close, window=63)` permitiría publicar
  la media de 63 bajo la identidad de la de 126 —mismo `series_id`, mismo hash, valor distinto—.
  Las ventanas van hard-coded y la identidad la fija `output_column`. Candados **por firma**.
* **misma `feature_id`, distinta receta por activo**: `realized_vol_20` usa √365 en BTC (24/7) y
  √252 en Gold (calendario de bolsa). El `transformation` los separa (`_ann365` / `_ann252`): con
  el mismo nombre parecerían una sola receta, y copiar el reloj equivocado movería la exposición
  ~20% sin que nada fallara. El candado lo comprueba **sobre los datos**, no sobre la etiqueta.
* **0 trials probado, no declarado**: paridad de serie completa por `resolve_feature_series` —la
  misma función que usa producción— contra la referencia legacy reescrita a mano, más el harness
  real de decisión (Gold 5618 barras y BTC 3239, exposición idéntica en float64).

### El gate perdió su allowlist, a propósito

Mientras hubo deuda, el gate llevaba allowlist con `xfail(strict=True)` por policy. Con la deuda a
cero esa forma se vuelve **peligrosa**: una lista vacía **pasa por vacuidad**, un `xfail` sin
sujeto no juzga nada, y la excepción quedaría para que una regresión futura la reutilizara. Ahora
es **un juez directo** sobre toda policy construible, con su propia guarda anti-vacuidad.

### Piloto `feature_set_hash`: 3 de 4, y la cuarta NO debe cerrarse

`smart_simple_v11` conserva el hueco **deliberadamente**. Medido: `engine.type: composite` y
`retrain: weekly` —la cadena rechaza ambos—, su módulo de implementación **no existe**, sus tres
requeridas **no están en el catálogo** y su spec declara `required_features_verified: false`.
Firmar un `feature_set_hash` sobre un contrato de inputs que el propio spec declara **no
verificado** daría apariencia de garantía donde no la hay.

**BL-39 sigue `PARTIAL`.**
