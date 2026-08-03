# TRIAGE — los PARTIAL del carril CLAUDE

> Sustituye la clasificación que retiré entera el 2026-08-03. Aquella derivaba el estado de
> **una sonda única** y cinco de sus filas fueron refutadas el mismo día. Aquí cada fila lleva
> **el comando declarado por la propia ficha**, **su resultado medido hoy**, y **la brecha que la
> ficha declara viva** — que es cosa distinta del comando. Verde ≠ cerrable: BL-05 estaba verde
> en unitarios y CODEX lo rechazó con razón en `CXD-188` porque su ficha declara E2E pendiente.
>
> Medido: 2026-08-03 · Autor: CLAUDE (`claude-root-152c263e`) · Entorno: PostgreSQL 16.4 portable
> vivo, **sin Docker**, `fabric-v1` sin aplicar.

## Conteo

Mi carril son **23 BLs**, de los cuales **7 IMPLEMENTED** (BL-01, 02, 04, 09, 11, 12, 34) ⇒
**16 PARTIAL**, no 20. El "20" venía del corte en que sólo 3 estaban cerrados.

| Clase | N | BLs |
|---|---:|---|
| LOCAL_CLOSABLE | 4 | BL-03, BL-15, BL-20, BL-45 |
| LOCAL_pero_CARRIL_CODEX | 1 | BL-06 |
| STACK_OR_CI | 4 | BL-05, BL-25, BL-32, BL-46 |
| OPERATOR | 2 | BL-36, BL-42 |
| FROZEN_DRIFT | 2 | BL-13, BL-14 |
| TIME_GATED | 2 | BL-31, BL-47 |
| MIXTO (parcial local + artefacto ausente) | 1 | BL-39 |

## Tabla

| BL | Comando declarado en la ficha | Resultado medido | Brecha viva declarada | Clase |
|---|---|---|---|---|
| BL-03 | `pytest tests/regression/test_forecasting_caveat_present.py -q` | **31 passed** | GM decide por `forecast_mode` (`ForecastingView.tsx:1032`), legacy por `isUsdcop` (`components/legacy/ForecastingLegacy.tsx`). El mismo BTC/Oro recibe relatos incompatibles. | **LOCAL_CLOSABLE** |
| BL-05 | `npx vitest run ProductionView.paper-ledger + PaperCandidatesPanel` | **16 passed** | La ficha declara E2E 375px/landscape/teclado/consola **no ejecutado**, y `fabric-contracts.yml` no incluye `paper-candidates-a11y.spec.ts`. | **STACK_OR_CI** |
| BL-06 | `pytest tests/regression/test_forecasting_caveat_present.py -q` | **31 passed** | El scanner funciona pero **ningún workflow invoca el gate Python** (`CXD-190`). | **LOCAL_pero_CARRIL_CODEX** |
| BL-13 | `pytest tests/regression/test_strategy_manifests.py` | **4 failed / 20 passed** | `surface: action\|diagnostic` en manifiestos + re-freeze consciente. Bloqueado por el drift de `code_hash`. | **FROZEN_DRIFT** |
| BL-14 | `pytest tests/regression/test_strategy_manifests.py -q` | **4 failed / 20 passed** | Bloque `components:` con `spec_fingerprint` de la receta. Mismos 4 rojos. | **FROZEN_DRIFT** |
| BL-15 | `pytest tests/unit/test_forecast_output_contract.py` | **132 passed** | **No existe todavía un `book_construction` real** que consuma `strategy_output`. | **LOCAL_CLOSABLE** |
| BL-20 | `pytest tests/unit/test_interpretability_artifacts.py -q` | **21 passed** (era 1F/20P) | El 1F era `catboost` ausente ⇒ TreeSHAP degradaba a `tree_shap_unavailable`. **Instalado ⇒ verde.** Queda que la ficha dice generar a `public/data/interpretability/**`, lo que **contradice CXD-057** (deben ser privados). Texto stale. | **LOCAL_CLOSABLE** |
| BL-25 | `pytest tests/unit/test_system_health.py -q` | **25 passed** | Motor único sobre `facts` + `metric_event`; la tabla llega con la migración 070. | **STACK_OR_CI** (`fabric-v1`) |
| BL-31 | `pytest tests/regression/test_strangler_cop.py -q` | **41 passed** | Paralelo **≥2 semanas por capa** con paridad `semantic_hash`; L7 al final. | **TIME_GATED** |
| BL-32 | `pytest tests/unit/test_passport_contract.py -q` | **83 passed** | `v_strategy_passport_live` + `mv_strategy_performance_daily` nocturna: son objetos de DB. | **STACK_OR_CI** |
| BL-36 | `pytest tests/regression/test_db_truth_matrix.py -q` | **8 passed** | Decisión por grupo escrita en la matriz de verdad. ASSIGNMENTS lo marca *"con el operador"*. | **OPERATOR** |
| BL-39 | `pytest tests/regression/test_feature_contracts.py -q` | **24 passed / 2 skipped** | Catálogo de features + `feature_set` por estrategia. Los 2 skips: *"H5 model artifacts not present (`outputs/` is gitignored)"* ⇒ el bit-check no es ejecutable en checkout limpio. | **MIXTO** |
| BL-42 | `pytest tests/regression/test_return_units.py -q` | **28 passed / 3 skipped** | Convención canónica de unidades mientras los productores divergen — ya listada como decisión del operador. | **OPERATOR** |
| BL-45 | `pytest tests/unit/test_policy_contract.py -q` | **219 passed** | R1/R2 del motor de políticas. `src/contracts/policy.py` **ya existe** (16 defs), así que parte de la brecha declarada puede estar stale y hay que auditar qué falta de verdad. | **LOCAL_CLOSABLE** |
| BL-46 | `npx vitest run StrategyEngineExplanation.test.tsx` | **5 passed** | `control.policy_version` + `strategy_signal` normalizada + endpoints: tabla de `fabric-v1`. | **STACK_OR_CI** |
| BL-47 | `pytest tests/unit/test_policy_specs.py -q` | **26 passed** | R6 exige paridad `semantic_hash` **antes** de apagar el camino viejo; depende de BL-45/46. | **TIME_GATED** |

## Lo que este triage NO afirma

- **No afirma que los 13 verdes estén cerrados.** Un comando verde prueba que su candado pasa,
  no que la brecha de la ficha esté cubierta. Diez de los trece siguen PARTIAL por brechas que
  el comando ni toca.
- **No reclasifica ningún BL ajeno.** BL-06 es wiring de CI, que por ASSIGNMENTS es carril CODEX;
  lo marco y no lo tomo.
- **No propone re-freeze de hashes.** BL-13/14 sólo salen del rojo con un re-freeze consciente,
  que es decisión del operador. Actualizar hashes mecánicamente para poner verde está prohibido.

## Patrón que aparece tres veces y merece decisión propia

`reports/*.csv` (paridad H1), `outputs/**` (bit-check BL-39) y `data/experiments/**/*.parquet`
son **artefactos inmutables de investigación que ningún checkout limpio tiene**. Tres candados
distintos dependen de ellos y los tres son irreproducibles fuera del árbol donde se generaron.
O se trackean, o los tests que los exigen deben declararse `skip` explícito con motivo — hoy
unos saltan en silencio y otro muere con `FileNotFoundError`.
