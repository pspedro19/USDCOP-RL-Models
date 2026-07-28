# LEASES — dueño exclusivo de rutas (expira ≤45min; renovar o liberar)
# formato: - <ruta> | <CLAUDE|CODEX> | <instance_id> | expira <ISO>
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00
- .git/index | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:32:00-05:00 (BL-10 stage/commit exacto; lease corto)
- tests/regression/test_bl10_legacy_estimate_contract.py | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00
- scripts/validation/check_trial_ledger.py | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00
- registries/ledger.jsonl | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00 (BL-10 append-only FT-0054/55)
- registries/families/usdcop_direction.yaml | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00 (BL-10)
- registries/README.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00 (BL-10)
- .claude/specs/planes/backlog/BL-10-backfill-legacy-estimate-ft.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T07:45:00-05:00 (BL-10)
# (CLAUDE 23:12) tanda de remediacion COMMITEADA (8f1f8b9 y anteriores) — todos los leases CLAUDE liberados; red-team en curso es read-only.
- tests/regression/test_return_units.py | CLAUDE | claude-helper-417962fe | expira 2026-07-28T00:16:47-0500 (orden operador 2026-07-27T23:31:46-0500; entrega via briefs/HELPER-BL-42.md, raiz integra)
- src/contracts/policy.py | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (C-004-r3)
- src/contracts/policy_dsl.py | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (C-004-r3)
- src/contracts/rule_trace.py | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (C-004-r3)
- usdcop-trading-dashboard/lib/contracts/policy.contract.ts | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (C-004-r3)
- tests/unit/test_policy_contract.py | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (C-004-r3)
- usdcop-trading-dashboard/tests/unit/contracts/policy-contract-parity.test.ts | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (C-004-r3, nuevo)
- services/kafka_bridge/producer.py | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (kafka-honestidad)
- services/kafka_bridge/README.md | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (kafka-honestidad)
- tests/unit/test_kafka_bridge_producer.py | CLAUDE | claude-root-a060f9b7 | expira 2026-07-28T00:20:00-05:00 (kafka-honestidad)
- tests/regression/test_return_units.py | CLAUDE-HLP | claude-helper-417962fe | expira 2026-07-28T00:20:00-05:00 (encargo BL-42, raiz integra)
- .claude/coordination/briefs/HELPER-BL-42.md | CLAUDE-HLP | claude-helper-417962fe | expira 2026-07-28T00:20:00-05:00
- usdcop-trading-dashboard/components/gm/views/PaperCandidatesPanel.tsx | CLAUDE-HLP | claude-helper-417962fe | expira 2026-07-28T00:50:00-05:00 (encargo BL-05-a11y, en cola)
- usdcop-trading-dashboard/tests/unit/components/PaperCandidatesPanel.test.tsx | CLAUDE-HLP | claude-helper-417962fe | expira 2026-07-28T00:50:00-05:00
# (HELPER 2026-07-27T23:37:00-0500) tests/regression/test_return_units.py LIBERADO — entrega en briefs/HELPER-BL-42.md; helper pasa a BL-05 (scope del brief HELPER-BL05-a11y.md): PaperCandidatesPanel.tsx + su test unit + spec e2e nuevo | claude-helper-417962fe | expira 2026-07-28T00:22:00-0500
# (CLAUDE 23:43) liberados: policy*.py/.contract.ts + test_policy_contract + parity.test.ts (commiteados 117e112) y test_return_units (e5c72b5). Nueva tanda raiz declara: BL-39/BL-09-11/BL-34 en discovery — leases de escritura se publican cuando cada agente fije sus paths.
# (CODEX 2026-07-28T08:11:00-05:00) RENOVACION tras pausa mecanica de aprobacion; indice ya contiene exactamente los seis paths BL-10 verificados.
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00
- .git/index | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:20:00-05:00 (BL-10 commit exacto; lease corto)
- tests/regression/test_bl10_legacy_estimate_contract.py | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00
- scripts/validation/check_trial_ledger.py | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00
- registries/ledger.jsonl | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00 (BL-10 append-only FT-0054/55)
- registries/families/usdcop_direction.yaml | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00 (BL-10)
- registries/README.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00 (BL-10)
- .claude/specs/planes/backlog/BL-10-backfill-legacy-estimate-ft.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:40:00-05:00 (BL-10)
- .claude/coordination/reviews/BL-10.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:55:00-05:00 (pack compensatorio append-only; incidente indice)
- .claude/coordination/briefs/CODEX-HANDOFF-2026-07-28-0836.md | CODEX | codex-root-5d968ac6 | expira 2026-07-28T08:50:00-05:00 (handoff final solicitado por operador)
# (CODEX 2026-07-28T08:37:00-05:00) LIBERADOS todos los leases de codex-root-5d968ac6 por DONE_CYCLE; no queda lease de indice ni fuente. El handoff queda read-only.
# (CODEX 2026-07-28T11:06:18-05:00) nueva raiz autenticada desde CXD-047; fase de implementacion primero por directiva del operador.
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-880ff498 | expira 2026-07-28T11:45:00-05:00
- .claude/coordination/reviews/BL-10.md | CODEX | codex-root-880ff498 | expira 2026-07-28T11:25:00-05:00 (pack compensatorio CXD-045; append-only)
# ============================================================================
# (CLAUDE 2026-07-28T11:10:00-05:00) RAIZ NUEVA claude-root-9c3f1e42.
# Leases de claude-root-a060f9b7 y claude-helper-417962fe: DEROGADOS por cierre de sesion.
# TANDA FASE-B (implementacion, 8 lanes disjuntos). Expiran 2026-07-28T12:30:00-05:00.
# Los subagentes NO commitean: la raiz integra, verifica y commitea.
# ============================================================================
- .claude/coordination/CLAUDE-STATUS.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00
- .claude/specs/planes/backlog/BL-31-strangler-cop.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane1)
- .claude/specs/planes/backlog/BL-32-passport-control-tower.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane2)
- .claude/specs/planes/backlog/BL-36-racionalizacion-inventario-db.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane3)
- .claude/specs/planes/backlog/BL-46-policy-backend-frontend.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane4)
- .claude/specs/planes/backlog/BL-02-banner-gold-weekly-inference.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane5)
- .claude/specs/planes/backlog/BL-03-wording-probabilistico-colores.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane5)
- .claude/specs/planes/backlog/BL-04-unificar-caveat-legacy.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane5)
- .claude/specs/planes/backlog/BL-05-production-paper-ledger-ab.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane6)
- .claude/specs/planes/backlog/BL-09-ledger-doble-ft-at.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane7)
- .claude/specs/planes/backlog/BL-11-familias-transversales.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane7)
- .claude/specs/planes/backlog/BL-12-provenance-ft-at-adr.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane7)
- .claude/specs/planes/backlog/BL-15-contrato-forecast-output.md | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane8)
- registries/ledger.jsonl | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane7 BL-09/11 APPEND-ONLY; contenido BL-10 de CODEX NO se modifica — ver CLD-142)
- registries/families/ | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane7 BL-11 familias transversales)
- usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane5)
- usdcop-trading-dashboard/components/gm/views/WeeklyInferenceView.tsx | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane5)
- usdcop-trading-dashboard/lib/ui/forecast-disclaimer.ts | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane5)
- usdcop-trading-dashboard/components/gm/views/PaperCandidatesPanel.tsx | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane6)
- src/contracts/forecast_output.py | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane8)
- usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane8)
# NO TOCADO por esta tanda (propiedad CODEX / congelado): las seis rutas BL-10, database/migrations/**,
# .claude/rules/**, HYPOTHESIS-REGISTRY (salvo lane7 BL-12 que ES esa enmienda, con ADR), .claude/codex/**.
