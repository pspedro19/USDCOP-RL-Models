---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app/admin
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

## Verificación
Artefactos para ≥1 modelo de cada clase (lineal/árbol/regla); vista renderiza; rbac:check; 0 trials (diagnóstico declarado sobre congelados §10.1).

## Notas constitución
A.7: solo test-folds; sirve para RECHAZAR modelos absurdos, no para probar verdades.
