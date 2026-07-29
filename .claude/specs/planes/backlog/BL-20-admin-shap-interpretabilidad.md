---
kind: roadmap
status: PARTIAL
version: 1.1.0
last_verified: 2026-07-28
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
1) Generador `scripts/analysis/generate_interpretability.py` por (surface, asset, model_id, version) → `public/data/interpretability/**`: SOLO test-folds; SHAP lineal (coef×(x−μ) del scaler train-only) para ridge/BR, TreeSHAP para árboles; cortes global/por-régimen/temporal(por año); kill-rules visibles (signo que cambia entre décadas o contradice prior). 2) Rule-based: ATRIBUCIÓN DE REGLAS etiquetada 'atribución, no SHAP' (qué gate decidió cada día, % tiempo activa, PnL beta/timing — reusa BL-07). 'Ambas según aplique' (decisión operador). 3) UI: sección admin nueva con selector superficie→asset→modelo→versión; RBAC admin-only + entrada en rbac.contract.ts.

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

**PARTIAL / pendiente**
- La ruta **lineal** sigue con el esquema viejo (un único fit y φ sobre todo el histórico,
  incluidas filas de train) y **sin corte por régimen**. Alinearla al walk-forward anual y
  añadirle `by_regime` es trabajo pendiente (no se tocó para no romper el artefacto vivo).
- Kill-rule "contradice el prior" no está implementada en ninguna ruta: exige una tabla de
  priors por feature ⇒ **DECISIÓN PENDIENTE DEL OPERADOR** (declararlos es modelado).
- `ard` y los tres híbridos del zoo no tienen artefacto (los híbridos mezclan lineal+árbol:
  su atribución correcta no es TreeSHAP puro).
- Superficies distintas de `zoo`/`rule_based` (v11 composite, Gold/BTC) sin cubrir.

## Verificación
Artefactos para ≥1 modelo de cada clase (lineal/árbol/regla); vista renderiza; rbac:check; 0 trials (diagnóstico declarado sobre congelados §10.1).
Ejecutado 2026-07-28: `pytest usdcop-trading-dashboard/tests/test_interpretability_schema.py -q`
⇒ **11 passed** (6 artefactos: 2 lineales + 3 árbol + 1 regla).

### Verificación ejecutable (CTR-MUTATION-SCOREBOARD-001)

```
comando: python -m pytest tests/unit/test_interpretability_artifacts.py -q
verde:   20 passed

muta:    scripts/analysis/generate_interpretability.py:467
         phi = Z * coefs  ->  phi = np.ones_like(Z)
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

## Notas constitución
A.7: solo test-folds; sirve para RECHAZAR modelos absurdos, no para probar verdades.
