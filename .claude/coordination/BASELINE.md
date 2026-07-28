# BASELINE de monitores (comparar DELTAS, no totales)
# Última medición completa: 2026-07-28T17:35-05:00 (todo lo de abajo re-ejecutado, no citado de memoria)
# Regla: una aprobación = "sin fallos NUEVOS vs esta lista". Un gate de CI que compare
# TOTALES nace rojo por deuda ajena y se desactiva el primer día — compara DELTA.

## Python — pytest

- **test_knowledge_frontmatter: 47 failed pre-existentes** (ADR-0021 sin front-matter,
  BOOK-LEVERAGE, EXP-DIR, `.claude/codex/*` legacy, etc. — anteriores al protocolo).
  Re-medido 2026-07-28: sigue en 47. Saneo = BL futuro, no bloqueo.

- **test_backlog_status_is_honest: 17 failed pre-existentes** (candado NUEVO 2026-07-28,
  `tests/regression/test_backlog_status_is_honest.py`, CTR-BACKLOG-HONESTY-001).
  Nace rojo A PROPÓSITO: mide una deuda real del tablero, no un defecto del test.
  Los 17 son BLs con `status: PLANNED` cuyo código ya está escrito y trackeado
  (la cabecera del artefacto se declara a sí misma deliverable del BL).
  **Los 17 son de propiedad CODEX** (ASSIGNMENTS.md); CLAUDE: 0.
  Ids exactos (test ids `test_planned_item_has_no_shipped_deliverable[BL-NN]`):

  | BL | Dueño | Evidencia que contradice `PLANNED` |
  |----|-------|-------------------------------------|
  | BL-10 | CODEX | `registries/README.md`, `registries/families/{trend_regime,usdcop_direction}.yaml`, `scripts/validation/check_trial_ledger.py`, `tests/regression/{test_bl10_legacy_estimate_contract,test_trial_ledger}.py` |
  | BL-16 | CODEX | `database/migrations/070_fabric_control_plane.sql`, `src/governance/declaration.py`, `src/identity/canonical.py` |
  | BL-17 | CODEX | `src/identity/fingerprints.py` |
  | BL-18 | CODEX | `src/metrics/engine.py` |
  | BL-19 | CODEX | `database/migrations/071_forecast_schema_roles.sql` |
  | BL-21 | CODEX | `database/migrations/074_exec_event_sourcing.sql`, `src/execution/events.py` |
  | BL-22 | CODEX | `database/migrations/075_fact_position_pnl.sql` |
  | BL-24 | CODEX | `database/migrations/076_lineage_graph.sql`, `src/lineage/graph.py` |
  | BL-26 | CODEX | `database/migrations/077_portfolio_control.sql`, `src/portfolio/{snapshot,target}.py` |
  | BL-27 | CODEX | `src/portfolio/allocator.py` |
  | BL-30 | CODEX | `database/migrations/078_exec_reconciliation.sql`, `src/execution/service.py` |
  | BL-35 | CODEX | `src/orchestration/dataset_uri.py` |
  | BL-37 | CODEX | `database/migrations/072_reference_identity.sql`, `src/market/identity.py` |
  | BL-38 | CODEX | `database/migrations/073_market_quality.sql`, `database/migrations/080_market_physical_profile.sql` |
  | BL-40 | CODEX | `src/data_quality/rules.py` |
  | BL-43 | CODEX | `database/migrations/081_synthetic_demo_isolation.sql` |
  | BL-44 | CODEX | `database/migrations/080_market_physical_profile.sql` |

  Se cierra cambiando `status:` en el MD del BL (dueño = CODEX), NO relajando el test.
  PLANNED honestos hoy (sin código publicado): BL-08, BL-23, BL-28, BL-29, BL-33, BL-41.

- **Colección de pytest — `pytest tests/` SIGUE ABORTANDO** (verificado 2026-07-28):
  `tests/scripts/test_feature_builder.py:45` llama `sys.exit(1)` en tiempo de import
  (falla `get_config` del RL) ⇒ `INTERNALERROR> SystemExit: 1`, **exit code 3**, con
  2176 tests ya recolectados. Cualquier job de CI que invoque `pytest tests/` a secas
  muere aquí. Workaround mientras no se arregle: `--ignore=tests/scripts` (o invocar
  subdirectorios explícitos).
- Colección: `tests/integration/test_mlflow_dataset_tracking.py` ⇒
  `ModuleNotFoundError: No module named 'train_ssot'` (pre-existente, 1 error de
  colección, no aborta la sesión).
- Todo lo demás en Python (`test_scripts_layout`, `test_strategy_manifests`, manifests,
  caveat/honesty): VERDE = 0 fallos.

## Frontend — Vitest y tsc

- **`npx vitest run tests/unit` (dashboard): 46 tests failed / 626 passed (672) en
  5 ficheros de 29.** Medido 2026-07-28T17:35, árbol limpio, 20.1 s.
  **Son PRE-EXISTENTES y AJENOS a esta fase** (dos módulos que ya no existen + un setup
  de React roto); ninguno lo introdujo el protocolo dual. Sin este registro, el CI que
  CODEX está cableando nace rojo por deuda ajena.

  | Fichero | Fallos | Causa medida |
  |---------|--------|--------------|
  | `tests/unit/components/Button.test.tsx` | 43 | `ReferenceError: React is not defined` (el test usa `React.*` sin importar React) |
  | `tests/unit/api/interpretability-security.test.ts` | 2 | la ruta devuelve 500 donde el test espera 200 (`artefacto válido ⇒ 200`, `campos desconocidos se STRIPean`) |
  | `tests/unit/replayApiClient.test.ts` | 1 | `TypeError: Cannot read properties of undefined (reading 'modelId')` |
  | `tests/unit/components/OptimizedChart.test.tsx` | suite entera | no resuelve `@/components/charts/OptimizedChart` — **el módulo no existe** (`components/charts/` solo tiene `TradingChartWithSignals.tsx`) |
  | `tests/unit/technical-indicators.test.ts` | suite entera | no resuelve `@/lib/technical-indicators` — **el módulo no existe** |

  Las dos últimas fallan en colección (Vitest las cuenta como "Failed Suites 2", 0 tests),
  por eso 43+2+1 = 46 y no más. Bloques de error que imprime Vitest: 48 (46 tests + 2 suites).
  **Corrección respecto a la estimación previa de "~44": el número real de hoy es 46**;
  ninguno de los cinco ficheros ha dejado de fallar.
  Los dos suites huérfanos se cierran borrando el test o restaurando el módulo — decisión
  de producto, no de esta fase.

- **`tsc --noEmit` dashboard: ~639 líneas de error pre-existentes** (ForecastingView
  `DIRECTION_TONE` ya eliminado; el resto en tests/rutas API no tocados).
- `rbac:check`: VERDE = 0 fallos.

## Cómo usar este fichero en CI

1. Ejecutar el monitor.
2. Restar esta lista.
3. Si el delta es 0 ⇒ verde, aunque el total sea rojo.
Añadir una entrada aquí exige haber MEDIDO el fallo y haber dicho de quién es la deuda.
