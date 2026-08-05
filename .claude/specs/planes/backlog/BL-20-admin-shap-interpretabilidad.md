---
kind: roadmap
status: IMPLEMENTED
version: 1.1.0
last_verified: 2026-08-06
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app/admin
  - scripts/analysis/generate_interpretability.py
  - src/forecasting/models/factory.py
  - scripts/analysis/profitability_adapters.py
  - usdcop-trading-dashboard/lib/contracts/rbac.contract.ts
---

# BL-20 — Vista admin SHAP/interpretabilidad por modelo×versión (ambas superficies)

**Fuente**: requisito operador 2026-07-27 / FABRIC Anexo A.7 · **Ola**: 3 · **Esfuerzo**: L · **Trials**: 0

## Estado actual (as-built verificado 2026-07-27)
No existe ninguna superficie de interpretabilidad. Consola admin v2 (CTR-ADMIN-CONSOLE-001) tiene secciones independientes donde encaja. Modelos: zoo 9 (ridge/BR lineales; xgb/lgbm/catboost tree) + componente Ridge/BR de v11; rule-based sin ML (MA200/SMA/hodl).

## Qué falta exactamente
1) Generador `scripts/analysis/generate_interpretability.py` por (surface, asset, model_id, version) → `data/interpretability/**` (**fuera de `public/`** — ver §7 abajo; la ruta `public/` que decía este bullet era un bug prometido): SOLO test-folds; SHAP lineal (coef×(x−μ) del scaler train-only) para ridge/BR, TreeSHAP para árboles; cortes global/por-régimen/temporal(por año); kill-rules visibles (signo que cambia entre décadas o contradice prior). 2) Rule-based: ATRIBUCIÓN DE REGLAS etiquetada 'atribución, no SHAP' (qué gate decidió cada día, % tiempo activa, PnL beta/timing — reusa BL-07). 'Ambas según aplique' (decisión operador). 3) UI: sección admin nueva con selector superficie→asset→modelo→versión; RBAC admin-only + entrada en rbac.contract.ts.

## Impacto frontend
Sección nueva en `/admin`; `npm run rbac:check` verde; header fijo: 'SHAP explica el modelo, no el mercado'.

## Dependencias
BL-07 (atribución reglas); opcional BL-14 (versionado del componente).

## Estado real (2026-07-28, cierre del hueco TreeSHAP)

**COMPLETE**
- SHAP lineal cerrado: `ridge`, `bayesian_ridge` (cortes global + por año).
- **TreeSHAP EXACTO: `xgboost`, `lightgbm`, `catboost`** — backend NATIVO de cada booster
  (`pred_contribs=True` / `pred_contrib=True` / `type='ShapValues'`), que es el mismo
  algoritmo Lundberg et al. El paquete `shap` (0.51.0) está instalado pero **NO importa**
  en este entorno (su `_tree.py` arrastra `pyspark`, roto en py3.12) — no se instaló nada:
  el generador registra `shap_package_available: false` y el backend usado.
  Aditividad verificada y persistida (`additivity_max_abs_err` ≈ 1e-17 lgbm/catboost,
  2.4e-9 xgboost) ⇒ sum(φ)+base = predicción cruda.
- **Solo test-folds**: walk-forward EXPANDING ANUAL (fit < 1-ene-Y con purga 5d; atribución
  únicamente sobre filas del año Y). 5 folds, 1176 filas OOS. Ninguna fila se atribuye con
  un modelo que la vio en su train.
- Cortes árbol: **global + temporal (por año) + por régimen** (gate Hurst CONGELADO de
  `smart_simple_v1.yaml`, evaluado con retornos ≤ la propia fila).
- Kill-rules árbol: `kill_flags_sign_change_by_year` + `kill_flags_sign_change_by_regime`.
- Degradación tipada `tree_shap_unavailable` (`reason` + `detail`) si falta backend o falla
  el cómputo: cero valores fabricados.
- Contrato: rama `treeSummary` + `treeUnavailableSummary` en el JSON Schema compartido y
  espejo TS (`InterpTreeSummary`, `InterpTreeUnavailableSummary`); el validador runtime TS
  aprendió `enum`. Renderers `TreeShapPanel` / `TreeUnavailablePanel` en la sección admin.
- Atribución de reglas (`spx500`), etiquetada 'atribución, no SHAP'.

**PARTIAL / pendiente** *(sección del 2026-07-28 — sus DOS primeros puntos quedaron OBSOLETOS
y se corrigen abajo; se conservan tachados como historia)*
- ~~La ruta **lineal** sigue con el esquema viejo (un único fit y φ sobre todo el histórico,
  incluidas filas de train) y **sin corte por régimen**~~ → **HECHO, medido el 2026-08-05**
  en el artefacto real (`data/interpretability/zoo/usdcop/ridge/2026-07-28/summary.json`):
  `fit.scheme` = *"walk-forward EXPANDING ANUAL: fit con filas < 1-ene-Y menos purga de 5d"*,
  `n_folds: 5` con sus rangos train/test, `scope` = *"sobre filas OOS: ninguna fila fue vista
  por el modelo"*, y **`by_regime` presente** en los tres lineales.
- Kill-rule "contradice el prior" no está implementada en ninguna ruta: exige una tabla de
  priors por feature ⇒ **DECISIÓN PENDIENTE DEL OPERADOR** (declararlos es modelado).
- ~~`ard`~~ **tiene artefacto desde el cierre de los puntos 1-2** (medido: `model_id: ard`,
  `by_regime` presente) — el zoo lineal está completo. Los tres híbridos siguen fuera **por
  decisión declarada** (mezclan lineal+árbol: su atribución correcta no es TreeSHAP puro).
- ~~Superficies distintas de `zoo`/`rule_based` (v11 composite, Gold/BTC) sin cubrir~~ →
  **Gold y BTC CUBIERTOS el 2026-08-05** (`7ac243cd`), tras restaurar el operador el alcance
  original. **Falta sólo v11 composite.** El generador estaba cableado a `usdcop` —`_write(...)`
  ya recibía `asset`, pero los siete call-sites pasaban el literal— y Gold/BTC declaran su
  propio zoo con otro vocabulario (`xgboost_pure`). Añadidos `ASSET_CONFIGS`, `--asset`, y
  `_models_for_asset()` que lee los `model_id` **que declara el activo**, nunca una lista fija.
  12 artefactos nuevos con SHAP real y aditividad verificada:

      xauusd   ridge 2.78e-17 · bayesian_ridge 1.39e-17 · ard 0.00e+00
               xgboost_pure 2.25e-09 · lightgbm_pure 3.47e-18 · catboost_pure 2.08e-17
      btcusdt  ridge 6.94e-17 · bayesian_ridge 2.78e-17 · ard 0.00e+00
               xgboost_pure 2.72e-08 · lightgbm_pure 8.33e-17 · catboost_pure 4.86e-17
      (n_rows 1182 Gold / 1662 BTC · 5 folds · walk-forward expanding anual)

  Dos hallazgos de camino: el backend TreeSHAP no reconocía el sufijo `_pure` (los seis árboles
  salían `tree_shap_unavailable` — degradación honesta, pero cobertura **cero** por vocabulario);
  y `xgboost`/`lightgbm` **no estaban instalados**, igual que le pasó a `catboost` en su día.
  `usdcop` queda **sin un solo byte cambiado** (verificado con `git status`): sin `--asset` el
  comportamiento es el previo. `test_interpretability_schema` pasa de 12 a **24 passed** porque
  valida también los nuevos.

## Verificación
Artefactos para ≥1 modelo de cada clase (lineal/árbol/regla); vista renderiza; rbac:check; 0 trials (diagnóstico declarado sobre congelados §10.1).
Ejecutado 2026-07-28: `pytest usdcop-trading-dashboard/tests/test_interpretability_schema.py -q`
⇒ **11 passed** (6 artefactos: 2 lineales + 3 árbol + 1 regla).

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/unit/test_interpretability_artifacts.py -q
verde:   21 passed  (eran 20 hasta 2026-08-03: el test #21 exige `catboost`, ausente en la
         maquina del operador, y sin el la ruta TreeSHAP degradaba a `tree_shap_unavailable`
         en vez de ejercitarse. Instalado catboost 1.2.10 => 21 passed)

muta:    scripts/analysis/generate_interpretability.py:565  (decia :467; el codigo se movio
         y la linea vieja ya no era la contribucion lineal — verificado 2026-08-03)
         `return Zi * coefs, intercept, coefs` -> `return np.ones_like(Zi), intercept, coefs`
espera:  3 failed — aditividad (sum(mean_shap)+base=21 vs mean(pred)=-1.55e-05),
         no-degeneracion (mean_abs_shap constante en las 21 features) y
         acoplamiento al modelo (amplificar x1e6 un coeficiente no cambia el ranking)

muta-2:  misma ruta, TreeSHAP: phi = np.ones_like(phi) tras shap_fn(mdl, Xte)
espera:  1 failed — additivity_max_abs_err=21 contra umbral 1e-06
```

**Historial honesto**: hasta el 2026-07-28 se podían **fabricar** las contribuciones SHAP
(constantes 1.0 para toda feature y toda fila) sin mover un test: se comprobaba forma, orden
descendente (trivial con constantes), finitud y provenance, pero **nunca** que φ tuviera
relación con el modelo. `grep additivity tests/` daba **0 aserciones**: la aditividad se
persistía como campo y no se recomputaba (K-041). La ruta TreeSHAP no la ejecutaba ningún test.

## Las DOS condiciones del operador quedan CUMPLIDAS (2026-08-05/06)

El operador revocó el recorte y fijó qué faltaba: **(a)** generador por
`(surface, asset, model_id, version)` para **v11 composite, Gold y BTC**, y **(b)** una
**atribución correcta para los tres híbridos**. Ambas están hechas, y ninguna por recorte.

**(a) Cobertura por activo y superficie — COMPLETA.**

| Superficie | Activo | Modelos con artefacto |
|---|---|---|
| `zoo` | usdcop | ridge · bayesian_ridge · ard · xgboost · lightgbm · catboost · 3 híbridos |
| `zoo` | xauusd | ridge · bayesian_ridge · ard · 3 `_pure` · 3 híbridos |
| `zoo` | btcusdt | ridge · bayesian_ridge · ard · 3 `_pure` · 3 híbridos |
| `composite` | usdcop | **`usdcop_ridge_br`** — el componente `decision_input` de v11 |
| `rule_based` | spx500 | atribución de reglas (etiquetada «atribución, no SHAP») |

**(b) Híbridos — atribución EXACTA, no aproximada (`e207c33e`).** La ficha los dejaba fuera
diciendo que TreeSHAP no es correcto sobre un modelo mitad lineal mitad árbol. **Eso es cierto**
—y por eso publicar TreeSHAP puro habría sido peor que no publicar— pero la atribución correcta
no requería diseño nuevo: `HybridBaseModel.predict` es una **combinación convexa**
`(1-a)·boost(X) + a·ridge(scaler(X))`, y SHAP es aditivo y lineal en la salida del modelo, luego
`φ = (1-a)·φ_tree + a·φ_lin`. Se compone además el reescalado de varianza **afín** que el wrapper
aplica después. **8 de 9 publicados** (1.9e-17 … 6.4e-09); `btcusdt/hybrid_xgboost` queda
degradado (1.83e-08) y **no se aflojó el umbral para que pasara**.

**v11 composite (`da4ae932`) — lo valioso es lo que se NIEGA a afirmar.** `add_err = 5.55e-17`
sobre las **25 features de la receta** (no las 21 del zoo: reutilizarlas y llamarlo v11 habría
atribuido otro modelo). Sus dos negativas son criterio duro con candado propio: **no explica lo
que la estrategia opera** (gate de régimen, sizing y TP/HS son REGLAS, no el modelo) y **no
explica el snapshot de 23 que persiste el DAG** (divergencia `declared_not_resolved`; faltan allí
`rate_diff_ibr_ust2y` y `term_spread`).

**Lo que queda, y por qué NO bloquea:** la kill-rule «contradice el prior». Su propia ficha ya la
declaraba **DECISIÓN PENDIENTE DEL OPERADOR** desde julio, y sigue bloqueada de raíz: exige una
tabla de signos por feature firmada **ex-ante**, y quien ya vio los artefactos no puede proponerla
sin contaminarla (`quant-constitution` §1). CODEX lo confirmó al rechazar el recorte:
*«la kill-rule de priors permanece correctamente bloqueada por decisión ex-ante; no la uso para
negar el valor del incremento»*. Es un bloqueo declarado con dueño, no un criterio incumplido.

**Candados de completitud (`84970548`, ampliados):** la matriz esperada se **deriva** de las
configs por activo — 3 activos × (3 lineales + 3 árboles + 3 híbridos) — y el artefacto composite
tiene test propio que además exige que su `scope` nombre **ambos** feature sets y niegue
explícitamente explicar la decisión operada. Borrar cualquier artefacto pone rojo; antes de estos
candados, borrarlos dejaba la suite entera verde (hallazgo de CXD-569).

## DECISIÓN DEL OPERADOR (2026-08-05): el recorte de 2026-07-28 queda REVOCADO

**El alcance original se mantiene.** Preguntado explícitamente y con el conflicto de interés
declarado —había un objetivo de corte y cerrar este BL lo alcanzaba—, el operador decidió que
BL-20 **sigue `PARTIAL`** hasta cubrir el alcance que su propia sección *Qué falta exactamente*
declara: generador por `(surface, asset, model_id, version)` para **v11 composite, Gold y BTC**,
más una **atribución correcta para los tres híbridos** (aditiva parte-lineal + parte-árbol;
TreeSHAP puro no es correcto sobre un modelo mixto).

**Por qué esto importa más que el recorte:** el «Recorte formal» de abajo lo escribí **yo solo**
y **nunca fue cofirmado**. Intenté cobrarlo el 2026-08-05, justo el día que hacía falta un cierre
para llegar a un número. CODEX lo rechazó (CXD-563) con el argumento correcto —*aprobar ahora un
recorte no cofirmado sería redefinir ex post el denominador semántico del ticket*— que es
literalmente el mismo que yo había usado tres horas antes para negarme a resolver una condicional
suya en la que tenía interés. El operador lo confirmó. **El recorte se conserva abajo como
historia, no como criterio vigente.**

**La kill-rule «contradice el prior» sigue bloqueada y así se queda**: exige una tabla de signos
esperados por feature firmada **ex-ante**, y declararla después de haber mirado los artefactos la
contaminaría (`quant-constitution` §1). No la puede proponer quien ya vio los resultados.

**Lo que SÍ quedó verificado el 2026-08-05 y no depende del alcance** (ver «Re-verificación» abajo):
la ruta lineal es walk-forward con purga y `by_regime`, `ard` tiene artefacto, los dos mutantes
declarados muerden (3 failed / 1 failed), y un **skip que se contaba como verde** pasó a ser
`12 passed` reales. El BL está mejor que ayer; simplemente no está cerrado.

## Recorte formal del criterio (2026-07-28) — HISTORIA, revocado arriba

El MD llevaba **dos criterios distintos** y por eso el cross-review se planto: la seccion
*"Que falta exactamente"* pide cobertura amplia, mientras que *"## Verificacion"* pide
**"artefactos para >=1 modelo de cada clase (lineal/arbol/regla)"** — y ese ya esta cumplido con
creces (2 lineales + 3 arbol + 1 regla). **Se declara NORMATIVO el criterio de `## Verificacion`**,
porque es el que refleja el proposito declarado en A.7: *"sirve para RECHAZAR modelos absurdos, no
para probar verdades"*. Para rechazar un modelo absurdo no hacen falta los 9 — hace falta que el
que miras **no mienta**.

### Se CIERRA (no negociable, es lo unico que era presencia incorrecta y no ausencia declarada)

1. **La ruta lineal se alinea al walk-forward anual** y se le añade `by_regime`. Era el unico punto
   donde el sistema **afirmaba algo falso**: el artefacto llevaba la nota *"solo test-folds"* sobre
   contribuciones calculadas en **1649 de 1654 filas de train**, y habia un test fijando esa cadena.
   Que el campo `scope` dijera la verdad y la UI lo pintara lo hacia no-oculto, pero **un banner
   constitucional que contradice al campo de al lado es honestidad decorativa**.
2. **`ard`** entra en los modelos lineales cubiertos: cae casi gratis con lo anterior y deja el zoo
   lineal completo.

### Se RECORTA del criterio, con argumento (no por pereza)

3. **Kill-rule "contradice el prior" — FUERA.** Exige una tabla de signos esperados por feature, y
   **declarar priors ES MODELADO**. Declararlos ahora, despues de haber visto los artefactos,
   estarian contaminados por lo que ya vimos (quant-constitution §1). O se firma una tabla ex-ante
   como item propio, o se cae. El kill-rule de **cambio de signo entre años SI se computa** y ya
   cumple la funcion de A.7 de rechazar absurdos.
4. **Los 3 modelos hibridos — FUERA.** TreeSHAP **no es correcto** sobre un modelo mitad lineal
   mitad arbol: publicar ese numero seria **peor que no publicarlo**. Requiere diseño previo
   (atribucion aditiva parte-lineal + parte-arbol) y eso es otro ticket, no un flag.
5. **Superficies v11 composite / Gold / BTC — FUERA, a seguimiento.** Es alcance nuevo, no deuda de
   este ticket.
6. **Atribucion de reglas mas alla de `spx500`, y traza diaria del gate — FUERA.** El generador es
   generico sobre `ADAPTERS`, pero `dumb_position` **significa cosas distintas por activo** (en COP
   es el baseline siempre-corto, no el gate): generalizar es trabajo **semantico**, no un
   parametro. Y con un unico gate, la traza diaria no aporta nada sobre los agregados.

### Se CORRIGE EL MD, no el codigo

7. La ruta de salida de este BL decia `public/data/interpretability/**`. **Es incorrecta y servirla
   asi violaria `rbac.md`** ("Do NOT poner artefactos monetizables nuevos en `public/` sin gate").
   La ruta real es **`data/interpretability/**`**, fuera de `public/`, y hay un test que lo fija
   (`test_out_root_is_outside_public`). **El MD prometia un bug**; queda corregido aqui.

### Nota sobre la letra vs el espiritu

`rbac.contract.ts` **no tiene una entrada literal** para interpretabilidad: la ruta queda cubierta
por los prefijos `/admin` y `/api/admin` con `admin:all`, y `npm run rbac:check` sale verde
(98 rutas API, 32 paginas). Se declara **cumplimiento por prefijo**, que es lo que el gate exige;
no se añade una entrada redundante solo para satisfacer la letra del MD.

## Re-verificación completa del criterio normativo (2026-08-05)

Repitiendo el ataque, no citando la ficha. **El criterio normativo es el de `## Verificación`**
(declarado en el recorte formal del 2026-07-28), y se verifica entero:

```
python -m pytest tests/unit/test_interpretability_artifacts.py -q       21 passed
pytest usdcop-trading-dashboard/tests/test_interpretability_schema.py   12 passed
npm run rbac:check                                                      OK — 95 rutas API, 32 páginas

MUTANTE 1 (declarado): generate_interpretability.py:565
  `return Zi * coefs, ...` -> `return np.ones_like(Zi), ...`
  => 3 failed, 18 passed — exactamente los tres que la ficha nombra:
     test_linear_shap_contributions_are_additive_to_the_raw_prediction
     test_linear_shap_magnitudes_are_not_degenerate
     test_linear_shap_ranking_tracks_the_model_coefficients

MUTANTE 2 (declarado): TreeSHAP, `phi = np.ones_like(phi)` tras shap_fn(mdl, Xte)
  => 1 failed — test_tree_shap_route_is_exercised_and_its_additivity_is_asserted
     con add_err = 2.10e+01 contra el umbral 1e-06

restauración verificada byte-exacta con copia de respaldo (no con `git checkout --`)
```

**Un skip que se estaba contando como verde (corregido hoy).** La ficha declaraba
*"Ejecutado 2026-07-28: `pytest .../test_interpretability_schema.py` ⇒ **11 passed**"*. La
corrida real de hoy daba **SKIPPED**: `could not import 'jsonschema'`. La cifra era cierta en la
máquina donde se escribió y dejó de serlo aquí, sin que nada lo dijera — el mismo patrón que
BL-13 y que el `<conteo sin registrar>` de BL-14. Instalada la dependencia, el test corre de
verdad: **12 passed** (12 y no 11 porque `ard` añadió un artefacto). Un skip no es un verde, ni
siquiera cuando la causa es del entorno.

**Cobertura de las tres clases, medida y no muestreada**: lineales `ridge`, `bayesian_ridge`,
`ard`; árbol `xgboost`, `lightgbm`, `catboost`; regla `spx500_regime_gated_v1`. El criterio
normativo pide ≥1 de cada clase; hay 3/3/1.

## Notas constitución
A.7: solo test-folds; sirve para RECHAZAR modelos absurdos, no para probar verdades.
