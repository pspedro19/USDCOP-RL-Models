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
- .claude/coordination/LEASES.md | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (registro fase auditoria)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (heartbeat fase auditoria)
- .claude/coordination/INBOX-CLAUDE.md | CODEX | codex-root-880ff498 | expira 2026-07-28T12:20:00-05:00 (CXD-050 solicitud review)
- .claude/coordination/INTEGRATION-AUDIT.md | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (nuevo registro compartido; inicializacion)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 TDD focal)
- src/identity/{canonical.py,fingerprints.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 rojo-verde)
- src/orchestration/{dataset_uri.py,factories.py,semantic_diff.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 rojo-verde)
- src/portfolio/{snapshot.py,allocator.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 rojo-verde)
- src/data_quality/rules.py | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 rojo-verde)
- src/research/qlab.py | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 rojo-verde)
- src/execution/service.py | CODEX | codex-root-880ff498 | expira 2026-07-28T12:35:00-05:00 (IA-R001 rojo-verde)
- src/contracts/forecast_output.py | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane8)
- usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts | CLAUDE | claude-root-9c3f1e42 | expira 2026-07-28T12:30:00-05:00 (lane8)
# NO TOCADO por esta tanda (propiedad CODEX / congelado): las seis rutas BL-10, database/migrations/**,
# .claude/rules/**, HYPOTHESIS-REGISTRY (salvo lane7 BL-12 que ES esa enmienda, con ADR), .claude/codex/**.
- database/migrations/{059,070,071,073,074,075,076,077,078}_*.sql | CODEX | codex-root-880ff498 | expira 2026-07-28T14:30:00-05:00 (IA-R001 remedios SQL)
- src/{identity,governance,execution,market,data_quality,metrics}/ | CODEX | codex-root-880ff498 | expira 2026-07-28T14:30:00-05:00 (IA-R001 remedios modulos)
- scripts/{ops/db_migrate.py,data/backfill_catalog_facts.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T14:30:00-05:00 (migracion segura + backfill)
- tests/unit/test_codex_{fabric,safety}_contracts.py | CODEX | codex-root-880ff498 | expira 2026-07-28T14:30:00-05:00 (31 contratos)
- .claude/coordination/{INBOX-CLAUDE.md,INTEGRATION-AUDIT.md,CODEX-STATUS.md,LEASES.md} | CODEX | codex-root-880ff498 | expira 2026-07-28T14:30:00-05:00 (auditoria)
- database/migrations/{059,070,071,073,074,075,076,077,078}_*.sql | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (IA-R001 TDD rojo-verde SQL; renovación)
- src/{identity,governance,execution,market,data_quality,metrics,portfolio,orchestration,research}/ | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (IA-R001 remedios módulos; renovación)
- scripts/{ops/db_migrate.py,data/backfill_catalog_facts.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (migración segura + backfill; renovación)
- tests/unit/test_codex_{fabric,safety}_contracts.py | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (39 contratos TDD)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md,integration/**,INTEGRATION-AUDIT.md} | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (auditoría cruzada y SSOT)
- RELEASE .claude/coordination/integration/** | CODEX | codex-root-880ff498 | liberado 2026-07-28T14:08:00-05:00 (lease amplio sustituido inmediatamente; no bloquear paths Claude)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md,INTEGRATION-AUDIT.md} | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (coordinación propia)
- .claude/coordination/integration/{AUDIT-CODEX-of-CLAUDE-IA-R001.md,SELF-REDTEAM-CODEX.md,isolated_contract_pytest.py,probes/**} | CODEX | codex-root-880ff498 | expira 2026-07-28T16:30:00-05:00 (artefactos propios)
- .claude/coordination/integration/{README.md,BDD-MATRIX.md,TDD-GAPS.md,INTEGRATION-CONTRACT.md} | CODEX+CLAUDE | shared-append-only | expira 2026-07-28T16:30:00-05:00 (solo secciones firmadas; sin reescribir al otro)
# (CODEX 2026-07-28T16:18:00-05:00) renovación FASE-II: correcciones adversariales y backlog residual, rojo->verde local; Docker/DB prohibidos.
- scripts/ops/db_migrate.py | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (recuperación de migración fallida + exclusión concurrente)
- src/metrics/{engine.py,annualization.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-18 annualization SSOT)
- src/market/resampling.py | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-38 resample determinista)
- src/governance/synthetic_isolation.py | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-43 fail-closed)
- scripts/{validation/validate_fabric_contracts.py,diagnostics/timescale_profile_v2.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-16/44)
- database/migrations/{080,081}_*.sql | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-38/43/44; DDL estático solamente)
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (TDD residual)
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-16 CI)
- .claude/specs/planes/04b-readiness-matrix.md | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-33)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md,INTEGRATION-AUDIT.md} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (heartbeat/auditoría)
- .claude/coordination/integration/{AUDIT-CODEX-of-CLAUDE-IA-R001.md,SELF-REDTEAM-CODEX.md,isolated_contract_pytest.py,probes/**} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (artefactos firmados)
- .claude/coordination/integration/{README.md,BDD-MATRIX.md,TDD-GAPS.md,INTEGRATION-CONTRACT.md} | CODEX+CLAUDE | shared-append-only | expira 2026-07-28T17:00:00-05:00 (solo append de secciones firmadas)
- config/metrics/{catalog.yaml,legacy_bypass_allowlist.yaml} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-18 contrato y guard no-new-bypass)
- tests/unit/{test_codex_fabric_contracts.py,test_codex_safety_contracts.py,test_codex_adversarial_remediations.py,test_codex_phase2_backlog.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (API de annualization + TDD)
- services/demo_mode/config.py | CODEX | codex-root-880ff498 | expira 2026-07-28T17:00:00-05:00 (BL-43 SSOT demo.*)
# (CODEX 2026-07-28T16:40:00-05:00) auditorías CLD-175/176 recibidas; lote bloqueantes rojo->verde y cierre bilateral CXD-063.
- database/migrations/{059_checkout_order_ledger.sql,080_*.sql,081_*.sql,082_*.sql,083_*.sql} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:20:00-05:00 (baseline fresh + upgrade aditivo + hardening auditoría)
- src/{metrics,portfolio,execution,research}/** | CODEX | codex-root-880ff498 | expira 2026-07-28T17:20:00-05:00 (P-01..P-20/E findings)
- config/{metrics,book}/** | CODEX | codex-root-880ff498 | expira 2026-07-28T17:20:00-05:00 (SSOT métricas/allocator)
- scripts/{ops/db_migrate.py,data/backfill_catalog_facts.py,analysis/qlab.py,validation/validate_fabric_contracts.py,diagnostics/timescale_profile_v2.py} | CODEX | codex-root-880ff498 | expira 2026-07-28T17:20:00-05:00 (M/P remediación)
- services/inference_api/entrypoint.sh | CODEX | codex-root-880ff498 | expira 2026-07-28T17:20:00-05:00 (ruta de aplicación fail-closed)
- tests/unit/test_codex_*.py | CODEX | codex-root-880ff498 | expira 2026-07-28T17:20:00-05:00 (TDD adversarial)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md,INTEGRATION-AUDIT.md,integration/**} | CODEX+CLAUDE | shared-append-only | expira 2026-07-28T17:20:00-05:00 (secciones propias; no reescritura cruzada)
# (CODEX 2026-07-28T20:53:00-05:00) SUCESION EXPLICITA: codex-root-880ff498 stale y PID 21928 cerrado por orden del operador; nueva raiz unica codex-root-39684-20c0.
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-39684-20c0 | expira 2026-07-28T21:35:00-05:00 (heartbeat nueva raiz)
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-39684-20c0 | expira 2026-07-28T21:35:00-05:00 (Playwright/BDD CI, ownership CXD-066)
- .claude/coordination/integration/SELF-REDTEAM-CODEX.md | CODEX | codex-root-39684-20c0 | expira 2026-07-28T21:35:00-05:00 (tablero mutacion Codex)
- .claude/coordination/integration/AUDIT-CODEX-of-CLAUDE-IA-R001.md | CODEX | codex-root-39684-20c0 | expira 2026-07-28T21:35:00-05:00 (auditoria bilateral)
# (CODEX 2026-07-28T21:00:25-05:00) test BDD nuevo y disjunto para que el job CI tenga screenshot + consola/red fail-closed sin editar specs Claude.
- usdcop-trading-dashboard/tests/e2e/ci-public-readonly.spec.ts | CODEX | codex-root-39684-20c0 | expira 2026-07-28T21:35:00-05:00 (Playwright CI publico estable; evidencia de navegador)
# (CODEX 2026-07-28T22:33:00-05:00) renovacion raiz unica: cierre FASE-C, gate CI sobre freeze 3f568220 y mensajes append-only.
- .claude/coordination/{CODEX-STATUS.md,INBOX-CLAUDE.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:18:00-05:00 (heartbeat, veredictos y leases propios)
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:18:00-05:00 (Playwright/BDD CI, ownership CXD-066)
- usdcop-trading-dashboard/tests/e2e/ci-public-readonly.spec.ts | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:18:00-05:00 (Playwright CI publico estable)
- .claude/coordination/integration/{SELF-REDTEAM-CODEX.md,AUDIT-CODEX-of-CLAUDE-IA-R001.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:18:00-05:00 (scoreboard y auditoria propios)
- .claude/coordination/{CODEX-STATUS.md,INBOX-CLAUDE.md,INBOX-CODEX.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:35:00-05:00 (relevo, heartbeat y ACKs append-only)
- src/governance/synthetic_isolation.py | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:35:00-05:00 (CI checkout limpio + BL-43 guard algorithm)
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:35:00-05:00 (mutacion fail-closed BL-43)
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:35:00-05:00 (Playwright/BDD CI, ownership CXD-066; renovacion)
- usdcop-trading-dashboard/tests/e2e/ci-public-readonly.spec.ts | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:35:00-05:00 (Playwright CI publico; renovacion)
# (CODEX 2026-07-28T23:07:20-05:00) alcance acotado por operador; remediacion P0 de invocadores DB.
- scripts/ops/db_migrate.py | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (P0 `--plan` sin caller)
- Makefile | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (P0 db-migrate/status/validate)
- services/inference_api/entrypoint.sh | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (P0 plan de migracion explicito)
- Makefile | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (P0 invocadores migrador con plan explicito)
- services/inference_api/entrypoint.sh | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (P0 startup migraciones fail-closed)
- tests/unit/test_codex_adversarial_remediations.py | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (TDD P0 migration callers)
- .claude/specs/planes/backlog/BL-10-backfill-legacy-estimate-ft.md | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:45:00-05:00 (cierre APROBADO bloqueado por tripwire manifest)
# RELEASE 2026-07-28T23:05:44-05:00: freeze servido 3f568220 cerrado; CI/K-044/Playwright diferidos a fase final. WIP workflow/spec e2e intactos.
# (CODEX 2026-07-28T23:13:12-05:00) prioridad operador: revisión cruzada de 11 packs Claude en snapshots temporales; cero escritura en paths de producción Claude.
- .claude/coordination/{CODEX-STATUS.md,INBOX-CLAUDE.md,INBOX-CODEX.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:50:00-05:00 (heartbeat, ACKs y veredictos append-only)
# (CODEX 2026-07-28T23:24:54-05:00) sello bilateral BL-10 tras monitor independiente verde.
- .claude/coordination/reviews/BL-10.md | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:40:00-05:00 (addendum final append-only)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:32:00-05:00 (commit exacto de dos documentos BL-10)
# (CODEX 2026-07-28T23:32:41-05:00) cierre formal de revisiones Tanda A, sin editar producción Claude.
- .claude/coordination/reviews/{BL-20.md,BL-25.md,BL-42.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:52:00-05:00 (veredictos append-only PARTIAL con evidencia)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:42:00-05:00 (commit exacto packs Tanda A)
# (CODEX 2026-07-28T23:36:00-05:00) ampliar P0 migrador a consejos/runtime callers descubiertos adversarialmente.
- services/inference_api/{main.py,routers/health.py} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:00:00-05:00 (plan explícito y comando operativo válido)
- scripts/validation/validate_fresh_install.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:00:00-05:00 (consejo de migración válido)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-28T23:52:00-05:00 (commit exacto remediación P0 callers migrador)
# (CODEX 2026-07-28T23:52:02-05:00) reparar módulos fuente ausentes del corte limpio.
- .gitignore | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:10:00-05:00 (excepción explícita src/research)
- src/research/** | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:10:00-05:00 (paquete qlab/PIT)
- src/{governance/synthetic_isolation.py,metrics/annualization.py,market/resampling.py} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:10:00-05:00 (módulos importados por tests/producción)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:02:00-05:00 (commit exacto módulos fuente ausentes)
# (CODEX 2026-07-28T23:59:00-05:00) remediación BL-29 cutoff real antes de cobrar.
- scripts/analysis/qlab.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:20:00-05:00 (screening PIT)
- tests/unit/test_qlab_point_in_time.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:20:00-05:00 (TDD/mutación causal)
- .claude/specs/planes/backlog/BL-29-qlab-cli-cutoff-lectura.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:20:00-05:00 (estado honesto para review)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:10:00-05:00 (commit exacto BL-29 PIT)
# (CODEX 2026-07-29T00:06:00-05:00) P0 invocador operativo explícito fabric-v1 y coordinación de revisión BL-29.
- Makefile | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:40:00-05:00 (invocador fabric-v1 con digest revisado)
- tests/unit/test_codex_adversarial_remediations.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:40:00-05:00 (candado causal de caller fabric-v1)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:40:00-05:00 (heartbeat y mensajes append-only)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:30:00-05:00 (staging exacto P0 fabric-v1)
- scripts/ops/db_migrate.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:40:00-05:00 (digest pinneado; cerrar bypass autorreferencial B-03)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T00:40:00-05:00 (plan fabric-v1 completo + pin contractual)
# (CODEX 2026-07-29T00:34:00-05:00) dictámenes restantes de la tanda cruzada, sólo reviews/append.
- .claude/coordination/reviews/{BL-39.md,BL-32.md,BL-36.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:05:00-05:00 (dictamen inmutable, sin producción Claude)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:05:00-05:00 (mensajes y heartbeat append-only)
# (CODEX 2026-07-29T01:07:00-05:00) última tanda de revisiones Claude; snapshots temporales, producción sólo lectura.
- .claude/coordination/reviews/{BL-05.md,BL-06.md,BL-13.md,BL-14.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:37:00-05:00 (addenda de dictamen inmutable)
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:37:00-05:00 (mensajes y heartbeat append-only)
# (CODEX 2026-07-29T01:16:08-05:00) remediación causal BL-26: identidad/cutoff del snapshot y mutación dedicada.
- src/portfolio/snapshot.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:50:00-05:00 (BL-26 identidad inmutable y cutoff aware)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:50:00-05:00 (BL-26 candados directos y mutación)
- .claude/specs/planes/backlog/BL-26-portfolio-snapshot.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T01:50:00-05:00 (estado y evidencia honesta)
# (CODEX 2026-07-29T01:25:32-05:00) remediación BL-35: parse fail-closed del contrato y aristas productivas neutrales.
- airflow/dags/asset_pipeline_factory.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:05:00-05:00 (re-raise DatasetContractError)
- config/assets/pipelines.yaml | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:05:00-05:00 (dataset_edges actuales sin decidir ML vs reglas)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:05:00-05:00 (camino real _load_config y mutación)
- .claude/specs/planes/backlog/BL-35-dataset-uris-arista-prohibida.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:05:00-05:00 (estado y evidencia)
# (CODEX 2026-07-29T01:33:58-05:00) remediación BL-37: paridad enum Python ↔ seed DDL.
- database/migrations/072_reference_identity.sql | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:10:00-05:00 (seed PT1M; digest fabric permanece cerrado)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:10:00-05:00 (paridad bidireccional y mutación)
- .claude/specs/planes/backlog/BL-37-identidades-canonicas.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:10:00-05:00 (recorte honesto de alcance y evidencia)
# (CODEX 2026-07-29T01:41:30-05:00) ACK CLD-250: BL-26 R2 valida causalidad, sanea errores y alinea persistencia.
- src/portfolio/snapshot.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:30:00-05:00 (invariantes causales en rehidratación)
- database/migrations/077_portfolio_control.sql | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:30:00-05:00 (UUIDv5 + missing_policy persistible)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:30:00-05:00 (forjas autoconsistentes y mutaciones)
- .claude/specs/planes/backlog/BL-26-portfolio-snapshot.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:30:00-05:00 (R2 y corrección de evidencia dirty-tree)
# (CODEX 2026-07-29T02:00:46-05:00) remediación BL-29 R2: auditabilidad, identidad y cutoff local por activo.
- scripts/analysis/qlab.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:40:00-05:00 (CLI PIT fail-closed y evidencia canónica)
- src/research/point_in_time.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:40:00-05:00 (sync/async y SQL sin predicado crudo)
- src/research/qlab.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:40:00-05:00 (identidad trial y campos auditables)
- tests/unit/test_qlab_point_in_time.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:40:00-05:00 (TDD y mutaciones BL-29 R2)
- .claude/specs/planes/backlog/BL-29-qlab-cli-cutoff-lectura.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:40:00-05:00 (estado y evidencia honesta R2)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:30:00-05:00 (staging exacto y commit BL-29 R2)
# (CODEX 2026-07-29T02:26:08-05:00) BL-29 R2 sellado en b162e4a0; leases de producción liberados para cross-review.
# (CODEX 2026-07-29T02:30:34-05:00) BL-22: ejecutar la expresión contable real y refutar/confirmar el supuesto paréntesis.
- database/migrations/075_fact_position_pnl.sql | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:00:00-05:00 (sólo mutación temporal/restauración exacta)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:00:00-05:00 (oráculo semántico identidad PnL)
- .claude/specs/planes/backlog/BL-22-fact-position-pnl.md | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:00:00-05:00 (estado real/evidencia)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T02:45:00-05:00 (staging exacto y commit BL-22)
# (CODEX 2026-07-29T02:39:36-05:00) BL-22 sellado en 67a38a99; leases de producción/test/MD liberados.
# (CODEX 2026-07-29T02:44:12-05:00) P0 064↔writers: candado rojo Codex; rutas COP quedan para lease Claude.
- tests/regression/test_h5_strategy_upsert_contract.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:15:00-05:00 (derivar targets UPSERT desde 064)
# (CODEX 2026-07-29T02:51:42-05:00) Reconciliación factual del frontmatter de doce entregas parciales Codex; cero promoción a DONE.
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:08:00-05:00 (staging exacto de doce MDs de honestidad)
# (CODEX 2026-07-29T08:01:45-05:00) Errata factual post-CLD257, comprobada contra flujo, mutante y parquet reales.
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T08:14:00-05:00 (commit compensatorio exacto de cinco MDs)
- [2026-07-29T08:31:23-05:00] CLAUDE toma lease de los writers/readers H5 para el P0 del ON CONFLICT (CXD-137):
  forecast_h5_l5_weekly_signal.py, forecast_h5_l6_weekly_monitor.py, forecast_h5_l7_multiday_executor.py,
  forecast_h5_l5_vol_targeting.py, los 4 UPSERT H5 de train_and_export_smart_simple.py y el join de
  control_system_health.py. Carril exclusivo CLAUDE por ASSIGNMENTS. CODEX no toca estas rutas.
# (CODEX 2026-07-29T08:33:18-05:00) Comunicación de backtests aislados, ACK H5 y heartbeat; sin tocar writers Claude.
- .claude/coordination/{INBOX-CLAUDE.md,CODEX-STATUS.md,LEASES.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T08:53:00-05:00 (CXD-141 + estado reproducible)
# (CODEX 2026-07-29T08:36:42-05:00) Backtests: cerrar tres fugas de frontera con un único split causal y mutaciones.
- scripts/analysis/{weekly_forecasting_oos_canonical.py,evaluate_2025_feature_selection_2026.py,evaluate_colombia_macro_candidates.py,_causal_backtest.py} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T09:20:00-05:00 (labels deben madurar estrictamente antes del OOS)
- tests/unit/test_codex_backtest_causality.py | CODEX | codex-root-39684-20c0 | expira 2026-07-29T09:20:00-05:00 (TDD frontera y mutaciones off-by-one)
# (CODEX 2026-07-30T17:59:16-05:00) probe de escritura solicitado por el operador.
- .tmp/codex-write-probe.txt | CODEX | codex-write-probe-20260730 | expira 2026-07-30T18:14:16-05:00
# (CODEX 2026-07-30T18:39:37-05:00) carril disjunto de validación del grafo; navegación/índices quedan intactos para Claude.
- scripts/validation/check_knowledge_graph.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:20:00-05:00 (gate estático de perímetro y conectividad)
- tests/regression/test_knowledge_graph.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:20:00-05:00 (TDD sintético del gate)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:20:00-05:00 (heartbeat de esta raíz)
# (CODEX 2026-07-30T19:13:30-05:00) renovación del gate y wiring CI/documentación del comando; carril de índices/config sigue en Claude.
- scripts/validation/check_knowledge_graph.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:50:00-05:00 (gate estático de perímetro y conectividad)
- tests/regression/test_knowledge_graph.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:50:00-05:00 (TDD sintético del gate)
- .github/workflows/specs-gate.yml | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:50:00-05:00 (wiring read-only del gate)
- AGENTS.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:50:00-05:00 (comandos locales que reflejan CI)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T19:50:00-05:00 (heartbeat de esta raíz)
# (CODEX 2026-07-30T19:28:03-05:00) paridad del checker de enlaces con el parser balanceado del gate de grafo.
- scripts/validation/check_knowledge_links.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T20:05:00-05:00 (poda de runtime y enlaces con labels anidados)
- tests/regression/test_knowledge_links.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T20:05:00-05:00 (TDD del parser y perímetro)
- scripts/validation/knowledge_markdown.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T20:05:00-05:00 (parser Markdown compartido por ambos gates)
# (CODEX 2026-07-30T21:27:29-05:00) takeover explícito del carril de navegación: CXD-142..150 sin ACK, último write Claude 19:16:34 y leases visibles vencidos.
- scripts/diagnostics/generate_doc_indexes.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (generador podado, idempotente y cleanup de índices runtime)
- scripts/diagnostics/generate_inventory.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (conteo skills por SKILL.md y catálogo generado)
- tests/regression/test_knowledge_inventory.py | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (candado del inventario de capacidades)
- .obsidian/app.json | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (perímetro de conocimiento/runtime)
- .obsidian/graph.json | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (grafo sólo de notas existentes)
- .claude/README.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (entrypoint y catálogo de capacidades)
- docs/INDEX.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (entrypoint de docs)
- .claude/{codex,experiments,specs,templates}/**/README.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (índices Markdown generados)
- docs/**/README.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (índices Markdown generados)
- .claude/coordination/{README.md,briefs/README.md,reviews/README.md,integration/README.md} | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (retirar bloques/archivos generados dentro de runtime)
- scripts/validation/{check_knowledge_graph.py,check_knowledge_links.py,knowledge_markdown.py} | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (gates del grafo y enlaces)
- tests/regression/{test_knowledge_graph.py,test_knowledge_links.py} | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (TDD de los gates)
- .github/workflows/specs-gate.yml | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (wiring CI)
- AGENTS.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (comandos locales)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (heartbeat)
- .claude/skills/xasset-alpha-engine/SKILL.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (convertir referencia interna en enlace verificable)
- .claude/skills/obsidian-markdown/SKILL.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:10:00-05:00 (hacer explícita la precedencia de la política relativa del repo)
# (CODEX 2026-07-30T21:47:26-05:00) reducción del presupuesto auto-loaded sin tocar quant-constitution ni trasladar invariantes.
- .claude/rules/{00-INDEX.md,strategy-engines.md,data-freshness.md,data-governance.md,rbac.md} | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:25:00-05:00 (eliminar duplicación y dejar referencia densa en specs existentes)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-kg-43736 | expira 2026-07-30T22:25:00-05:00 (heartbeat)
# (CODEX 2026-07-30T21:56:59-05:00) RELEASE: todos los leases de `codex-root-kg-43736` quedan liberados; entrega CXD-151.
# (CODEX 2026-07-31T12:14:52-05:00) carril disjunto: registro de DAG shadow; PROGRESS permanece multiwriter cofirmado y no recibe lease exclusivo.
- airflow/dags/contracts/dag_registry.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T12:55:00-05:00 (cerrar inventario de tres DAG shadow contra disco)
- tests/regression/test_knowledge_inventory.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T12:55:00-05:00 (gate existente; sólo ampliar si el fix revela un caso no cubierto)
# (CLAUDE 2026-07-31T12:18:19-05:00) raiz nueva `claude-root-152c263e`; toma los DOS rojos abiertos (CXD-151 declaraba uno).
- airflow/dags/contracts/dag_registry.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T13:30:00-05:00 (clasificar los 3 forecast_h1_*shadow* por evidencia del modulo)
- airflow/dags/forecast_h1_daily_shadow_v1.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T13:30:00-05:00 (solo lectura salvo que la evidencia exija deprecar)
- airflow/dags/forecast_h1_regime_shadow.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T13:30:00-05:00 (idem)
- airflow/dags/forecast_h1_regime_shadow_v2.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T13:30:00-05:00 (idem)
- .claude/skills/webapp-testing/** | CLAUDE | claude-root-152c263e | expira 2026-07-31T13:30:00-05:00 (cerrar gate de skills promovidas sin tests)
- tests/regression/test_quant_library_gate.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T13:30:00-05:00 (solo si el liston cambia; por defecto NO se toca el candado)
# (CODEX 2026-07-31T12:22:44-05:00) RELEASE por carrera de lectura: cede a CLAUDE los leases CODEX 12:14 sobre `dag_registry.py` y `test_knowledge_inventory.py`; no hubo writes de implementación. PROGRESS sigue multiwriter sin lease exclusivo.
# (CODEX 2026-07-31T12:32:32-05:00) BL-33 disjunto: reemplazar matriz decorativa por evidencia enlazada y un candado estructural; cero rutas COP/skills Claude.
- .claude/specs/planes/04b-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:15:00-05:00 (registro institucional con control/expectativa/evidencia/estado/dueño/fecha)
- .claude/specs/planes/backlog/BL-33-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:15:00-05:00 (estado honesto PLANNED→PARTIAL sólo tras evidencia)
- tests/regression/test_readiness_matrix.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:15:00-05:00 (TDD contra filas genéricas o sin enlaces verificables)
# (CODEX 2026-07-31T13:12:00-05:00) renovación BL-33 tras pausa de aprobación del runner; mismos paths, sin ampliar alcance.
- .claude/specs/planes/04b-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:55:00-05:00 (incorporar resultados reales y gaps hallados)
- .claude/specs/planes/backlog/BL-33-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:55:00-05:00 (PLANNED→PARTIAL con límites y evidencia)
- tests/regression/test_readiness_matrix.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:55:00-05:00 (gate estructural y mutación)
# (CODEX 2026-07-31T13:18:27-05:00) índice compartido contiene cuatro paths pre-staged ajenos; lease corto para commit BL-33 con `--only`, preservándolos byte por byte.
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:30:00-05:00 (stage del test nuevo + commit --only de los tres paths BL-33; no tocar cuatro entries preexistentes)
# (CODEX 2026-07-31T13:20:53-05:00) RELEASE `.git/index`: BL-33 sellado por `--only` en ba602b69835a1c45780f2d5a89bcb069a90034a2; los cuatro entries pre-staged ajenos permanecen con el mismo name-status.
# (CODEX 2026-07-31T13:32:04-05:00) RELEASE BL-33 implementación: matriz, ficha y test quedaron sellados en ba602b69835a1c45780f2d5a89bcb069a90034a2; sólo se abre el paquete runtime para cross-review.
- .claude/coordination/reviews/BL-33.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:10:00-05:00 (paquete inmutable de evidencia y límites; no cambia el commit revisado)
# (CODEX 2026-07-31T13:35:11-05:00) corrección factual antes de publicar pack: separar el resultado del gate propio del de la batería amplia; el agregado era cierto pero estaba atribuido a una sola invocación.
- .claude/specs/planes/04b-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:15:00-05:00 (evidencia 199P+3P, sin cambiar estados)
- .claude/specs/planes/backlog/BL-33-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:15:00-05:00 (misma errata reproducible)
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T13:50:00-05:00 (commit compensatorio `--only`; preservar cuatro entries ajenos)
# (CODEX 2026-07-31T13:41:18-05:00) RELEASE dos MDs BL-33 + `.git/index`: corrección sellada en 1812ae3bd1771c7e2d7dcf129eb2557286a9c988; los cuatro entries pre-staged ajenos permanecen idénticos. Continúa sólo el lease del review pack.
# (CODEX 2026-07-31T13:43:24-05:00) gate de índices detectó deriva causada por el título BL-33; regeneración mecánica con herramienta oficial.
- .claude/specs/planes/README.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:05:00-05:00 (`generate_doc_indexes.py --write`; verificar diff único)
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:05:00-05:00 (commit `--only` del índice generado; preservar cuatro entries ajenos)
# (CODEX 2026-07-31T13:45:53-05:00) RELEASE índice `specs/planes` + `.git/index`: índice generado sellado solo en 75a1e86d; staged ajeno intacto. El gate detecta nueva reescritura externa de Obsidian.
- .obsidian/graph.json | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:15:00-05:00 (restaurar `hideUnresolved: true`, único cambio permitido)
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:15:00-05:00 (commit `--only` de config; preservar cuatro entries ajenos)
# (CODEX 2026-07-31T13:50:13-05:00) RELEASE `.obsidian/graph.json` + `.git/index`: config canónica trackeada en 793837592e965c2850c555da9246ac46cc29165c; cuatro entries staged ajenos intactos. Índice libre para entrega CLAUDE.
# (CODEX 2026-07-31T14:01:20-05:00) renovación exclusiva del review pack BL-33; `.git/index` permanece LIBRE para CLAUDE desde 13:50:13.
- .claude/coordination/reviews/BL-33.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:35:00-05:00 (pack de corte 79383759; sin lease de índice todavía)
# (CLAUDE 2026-07-31T13:50:00-05:00) RENOVACION (los de 12:18 vencieron a las 13:30) + ampliacion para el candado fail-closed del DAG retirado.
- airflow/dags/forecast_h1_regime_shadow.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:45:00-05:00 (fence fail-closed: negarse a escribir el ledger si pertenece a otro experimento)
- tests/regression/test_h1_shadow_ledger_ownership.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:45:00-05:00 (TDD del fence, fichero nuevo)
- airflow/dags/contracts/dag_registry.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:45:00-05:00 (pendiente de commit cuando CODEX libere .git/index)
- tests/regression/test_quant_library_gate.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:45:00-05:00 (idem, ya verde 8P)
- .claude/specs/operations/elite-operations.md | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:45:00-05:00 (idem, timeline SSOT)
# (CLAUDE 2026-07-31T13:56:00-05:00) lease corto `.git/index` para commit `--only` de 5 rutas; los 4 entries pre-staged ajenos NO se tocan.
- .git/index | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:10:00-05:00 (commit --only; verificar name-status ajeno antes/despues)
# (CLAUDE 2026-07-31T14:09:00-05:00) RENOVACION del lease `.git/index` (vencia 14:10) + AMPLIACION de alcance: el stack shadow H1 estaba SIN TRACKEAR entero y entra en el mismo commit.
- .git/index | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:35:00-05:00 (commit --only; los 4 entries pre-staged ajenos se verifican por name-status antes y despues)
- airflow/dags/forecast_h1_*shadow*.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:35:00-05:00 (trackear los 3 modulos)
- scripts/{pipeline,validation,data}/*usdcop_h1_*shadow* | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:35:00-05:00 (generadores/validadores/evaluadores/baselines del experimento pre-registrado)
- config/forecast_experiments/usdcop_h1_*shadow*.yaml | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:35:00-05:00 (contratos pre-registrados)
- tests/unit/test_*usdcop_h1_*shadow*.py | CLAUDE | claude-root-152c263e | expira 2026-07-31T14:35:00-05:00 (sus tests, hoy invisibles al CI)
# (CODEX 2026-07-31T14:13:40-05:00) BL-41 fase estática fail-closed; C-007 permanece condicionado y no se crea/ejecuta DDL.
- config/governance/bl41_secret_cutover.yaml | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:55:00-05:00 (adoptar SSOT BLOCKED_OPERATOR sin evidencias inventadas)
- scripts/validation/check_bl41_secret_cutover.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:55:00-05:00 (validador estático; cero conexión DB/Vault)
- tests/regression/test_bl41_secret_cutover.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:55:00-05:00 (TDD/mutaciones de precondiciones y secreto reference-only)
- .claude/specs/planes/backlog/BL-41-seguridad-db-p0.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:55:00-05:00 (PLANNED→PARTIAL sólo tras gates; sin claim de cutover)
# (CLAUDE 2026-07-31T14:16:30-05:00) RELEASE de TODOS los leases de `claude-root-152c263e`: sellado en 749250dfaa26e360cf94970cf237f8a8b1270700; los 4 entries pre-staged ajenos verificados identicos por name-status antes y despues. `.git/index` libre.
# (CODEX 2026-07-31T14:25:29-05:00) lease corto de índice para sellar BL-41 estático; pack BL-33 y cuatro entries ajenos quedan fuera.
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T14:45:00-05:00 (stage de tres artefactos nuevos + commit `--only` de cuatro paths BL-41)
# (CODEX 2026-07-31T14:28:04-05:00) RELEASE BL-41 implementación + `.git/index`: cuatro paths sellados en 46d36e89aa7ce2d61b6e43e347ce6d4ed7e2200f; staged ajeno y pack BL-33 intactos.
- .claude/coordination/reviews/BL-41.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:05:00-05:00 (pack inmutable de fase estática; BL-41 permanece PARTIAL)
# (CODEX 2026-07-31T14:46:50-05:00) remediación adversarial CLD-267; no abre BL nuevo y no toca baseline/clasificación Claude.
- scripts/validation/check_bl41_secret_cutover.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (gate static-preflight que nunca autoriza cutover + evidencia tipada)
- tests/regression/test_bl41_secret_cutover.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (TDD contra `evidence: ok`, paths/hash y autorización YAML)
- config/governance/bl41_secret_cutover.yaml | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (declarar modo STATIC_PREFLIGHT_ONLY, defaults bloqueados)
- .claude/specs/planes/backlog/BL-41-seguridad-db-p0.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (documentar límite corregido sin claim externo)
- tests/regression/test_readiness_matrix.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (pin reviewed target-set de las 36 filas)
- .claude/specs/planes/04b-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (documentar correspondencia y evidencia de mutación)
- .claude/specs/planes/backlog/BL-33-readiness-matrix.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:35:00-05:00 (incorporar hallazgo/reparación CLD-267; permanece PARTIAL)
# (CODEX 2026-07-31T15:00:20-05:00) índice corto para commit compensatorio R2 de siete paths trackeados; preservar cuatro entries pre-staged ajenos.
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:15:00-05:00 (`git commit --only` de BL-33/41 R2; no stagear ni capturar otros paths)
# (CODEX 2026-07-31T15:02:13-05:00) RELEASE `.git/index` + siete paths R2: commit exacto `16f8b611f21a3981d864519506ae3d1b350d7290`; cuatro entries pre-staged ajenos permanecen idénticos. Índice libre.
# (CODEX 2026-07-31T15:02:13-05:00) packs compensatorios R2; implementación queda congelada para re-review Claude.
- .claude/coordination/reviews/BL-33.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:25:00-05:00 (addendum CLD-267 + target correcto de rango/commit compensatorio)
- .claude/coordination/reviews/BL-41.md | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:25:00-05:00 (addendum static-preflight-only + regresión evidence-ok)
# (CODEX 2026-07-31T15:05:13-05:00) índice corto para versionar los dos packs R2 hoy untracked; preservar cuatro entries pre-staged ajenos.
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T15:15:00-05:00 (`git add` exacto de BL-33.md/BL-41.md + commit `--only`; ninguna otra ruta)
# (CODEX 2026-07-31T15:06:48-05:00) RELEASE packs BL-33/41 + `.git/index`: versionados en `623e69a9af0c81cc4783e78c669cb4527c3267a5`; cuatro staged ajenos idénticos. Todos los leases CODEX de R2 quedan liberados.
# (CLAUDE 2026-07-31T14:56:00-05:00) lease para el recalculo del baseline de monitores (ACK de CODEX en CXD-162: herramienta oficial, diff propuesto, commit aparte).
- .claude/coordination/BASELINE.monitors.json | CLAUDE | claude-root-152c263e | expira 2026-07-31T15:40:00-05:00 (re-registro de pytest_knowledge_frontmatter con --update-baseline, NUNCA a mano)
- .claude/coordination/BASELINE.md | CLAUDE | claude-root-152c263e | expira 2026-07-31T15:40:00-05:00 (documentar el pago de deuda)
# (CLAUDE 2026-07-31T15:12:00-05:00) RELEASE BASELINE.monitors.json + BASELINE.md SIN ESCRIBIR: 42/47 de la 'deuda pagada' son ficheros movidos sin trackear; re-registrar grabaria un 0 falso. Evidencia en CLD-269.
# (CODEX 2026-07-31T15:42:09-05:00) BL-22 reparacion local acotada de provenance path; no toca DB, MetricEngine dirty ni plan fabric-v1.
- scripts/data/backfill_catalog_facts.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:10:00-05:00 (hacer robusta la referencia de source cuando inventory usa un PUBLIC aislado)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:10:00-05:00 (pin causal de source estable sin path absoluto de tmp)
# (CODEX 2026-07-31T15:58:07-05:00) renueva BL-22 e indice corto para sellar exactamente dos paths; preservar 3R100+1D staged ajenos.
- scripts/data/backfill_catalog_facts.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:20:00-05:00 (provenance estable + traversal fail-closed ya verificados)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:20:00-05:00 (dos regresiones focales verdes)
- .git/index | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:10:00-05:00 (commit --only de los dos paths BL-22; cuatro entries pre-staged quedan fuera)
# (CODEX 2026-07-31T16:03:01-05:00) RELEASE BL-22 + indice: dos paths sellados en 8aa3a75f; 3R100+1D staged ajenos conservan blobs exactos. Sin leases CODEX activos.
# (CODEX 2026-07-31T16:10:31-05:00) cross-review adversarial BL-02/04: mutaciones locales temporales, sin commit; restauracion por hash antes de liberar.
- usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:30:00-05:00 (hacer condicional el disclaimer BL-02 y demostrar que los gates muerden)
- usdcop-trading-dashboard/components/forecasting/ForecastingDashboard.tsx | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:30:00-05:00 (romper temporalmente el consumidor SSOT BL-04 y demostrar deteccion)
# (CODEX 2026-07-31T16:20:08-05:00) RELEASE ambos leases de mutacion BL-02/04: restaurados byte-exactos a SHA256 `848B220C...65E537` y `509947EB...7E904C`; paths limpios y suites finales verdes.
# (CODEX 2026-07-31T16:23:49-05:00) cross-review adversarial BL-01/03: mutaciones locales temporales, sin commit; consultar este lease antes de medir forecasting.
- usdcop-trading-dashboard/lib/ui/forecast-disclaimer.ts | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:40:00-05:00 (invertir temporalmente el copy honesto conservando el ancla BL-01)
- usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:40:00-05:00 (devolver token crudo en directionLabel BL-03)
# (CODEX 2026-07-31T16:29:56-05:00) RELEASE ambos leases BL-01/03: restaurados byte-exactos a SHA256 `506BB286...F01A8A` y `848B220C...65E537`; paths y pruebas medidos limpios.
# (CODEX 2026-07-31T16:34:49-05:00) cross-review adversarial BL-12: neutralizacion temporal del muro FT->AT, sin commit; restauracion SHA antes de liberar.
- scripts/validation/check_trial_ledger.py | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T16:50:00-05:00 (hacer return vacio en check_provenance_wall y probar cuatro ataques)
# (CODEX 2026-07-31T16:39:25-05:00) RELEASE lease BL-12: validador restaurado byte-exacto SHA256 `EFA0984A...FE2AAC`, path limpio y suite final 34P.
# (CLAUDE 2026-07-31T16:34:00-05:00) promocion autorizada por CXD-189 de BL-01/02/04 (owner=CLAUDE). BL-12/13/14 CEDIDOS a CODEX (K-021): no los toco.
- .claude/specs/planes/backlog/BL-01-test-caveat-forecasting.md | CLAUDE | claude-root-152c263e | expira 2026-07-31T17:05:00-05:00 (status PARTIAL->IMPLEMENTED con evidencia y hash de cross-review)
- .claude/specs/planes/backlog/BL-02-banner-gold-weekly-inference.md | CLAUDE | claude-root-152c263e | expira 2026-07-31T17:05:00-05:00 (idem)
- .claude/specs/planes/backlog/BL-04-unificar-caveat-legacy.md | CLAUDE | claude-root-152c263e | expira 2026-07-31T17:05:00-05:00 (idem)
# (CODEX 2026-07-31T16:47:19-05:00) BL-20: retirar dos artefactos lineales historicos falsos/schema-invalidos; nuevas versiones 2026-07-28 permanecen. Cero API/schema WIP.
- data/interpretability/zoo/usdcop/ridge/2026-07-27/summary.json | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T17:10:00-05:00 (nota dice solo-test-folds pero scope atribuye train; schema vigente lo rechaza)
- data/interpretability/zoo/usdcop/bayesian_ridge/2026-07-27/summary.json | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T17:10:00-05:00 (mismo defecto; retiro recuperable por Git)
# (CODEX 2026-07-31T16:55:29-05:00) renovacion BL-20 hasta 17:20 para gates/commit exacto; mismos dos paths, sin ampliar a API dirty.
- data/interpretability/zoo/usdcop/ridge/2026-07-27/summary.json | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T17:20:00-05:00 (borrado WT, pendiente gates y sellado)
- data/interpretability/zoo/usdcop/bayesian_ridge/2026-07-27/summary.json | CODEX | codex-root-backlog-20260731-1214 | expira 2026-07-31T17:20:00-05:00 (borrado WT, pendiente gates y sellado)
# (CLAUDE 2026-08-03T08:09:04-05:00) RELEASE leases de promocion BL-01/02/04: status IMPLEMENTED con bloque de cierre; honesty gate 105P/47S y frontmatter 994P DESPUES del cambio. Marcador 5/47 -> 8/47, pendiente cofirma CODEX.
# (CLAUDE 2026-08-03T08:22:00-05:00) DIRECTIVA OPERADOR: commitear todo el arbol por carriles, SIN PUSH. Lease de indice; el carril CODEX se firma [claude] citando su autoria (anula CXD-154 por orden del dueño).
- .git/index | CLAUDE | claude-root-152c263e | expira 2026-08-03T09:15:00-05:00 (commits por carril; publico hashes al terminar)
- src/metrics/persistence.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:42:00-05:00 (BL-18: sink PostgreSQL append-only/idempotente para MetricEvent; fichero nuevo)
- src/metrics/__init__.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:42:00-05:00 (BL-18: exportar sink sin cambiar contratos compartidos)
- tests/unit/test_metric_persistence.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:42:00-05:00 (TDD: SQL parametrizado, JSON finito, idempotencia y rechazo de conflicto)
# (CODEX 2026-08-03T11:02:00-05:00) BL-18 fase de persistencia local. No toca migraciones, catalogo, allowlist ni workflow hasta cerrar este incremento.
# (CODEX 2026-08-03T11:08:00-05:00) lease corto de indice para sellar solo el incremento BL-18; canales runtime quedan fuera.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:18:00-05:00 (commit --only de persistence.py, __init__.py y test_metric_persistence.py)
# (CODEX 2026-08-03T11:12:00-05:00) RELEASE BL-18 persistence + `.git/index`: tres paths exactos sellados en a89931c7; package-lock externo excluido. Sin leases CODEX activos.
# (CODEX 2026-08-03T11:18:00-05:00) BL-18 allowlist monotono: inventario AST de implementaciones Sharpe/Calmar y wiring CI. No toca implementaciones metricas.
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:00:00-05:00 (reemplazar lista vacia falsa por baseline factual y techo monotono)
- scripts/validation/validate_fabric_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:00:00-05:00 (scanner AST fail-closed; API CLI existente)
- tests/unit/test_metric_bypass_allowlist.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:00:00-05:00 (TDD: implementacion nueva y expansion sin retiro quedan rojas)
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:00:00-05:00 (invocar el gate en CI Python)
# (CODEX 2026-08-03T11:27:00-05:00) lease corto de indice para BL-18 allowlist; solo cuatro paths, canales runtime excluidos.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:37:00-05:00 (commit exacto gate+YAML+test+workflow)
# (CODEX 2026-08-03T11:31:00-05:00) RELEASE BL-18 allowlist + `.git/index`: cuatro paths sellados en 55fcefc6; sin leases CODEX activos.
# (CODEX 2026-08-03T11:36:00-05:00) BL-18 primer consumidor: SPX economic_metrics tiene formula ddof=1 identica al SSOT; preservar API/None->0 y decrementar inventario.
- src/strategies/spx500_regime_gated_v1/economic_metrics.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:15:00-05:00 (delegar Sharpe al SSOT sin cambiar resultados)
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:15:00-05:00 (30->29 y retirar solo el bypass migrado)
- tests/unit/test_metric_consumer_migration.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:15:00-05:00 (paridad numerica y casos degenerados)
# (CODEX 2026-08-03T11:43:00-05:00) lease corto indice para primer consumer BL-18; tres paths exactos.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:53:00-05:00 (SPX consumer + decremento allowlist + test)
# (CODEX 2026-08-03T11:47:00-05:00) RELEASE consumer SPX + indice: tres paths sellados en 8765adee.
# (CODEX 2026-08-03T11:47:00-05:00) BL-18 limpiar wrappers SPX delegados sin cambiar API publica.
- src/strategies/spx500_regime_gated_v1/economic_metrics.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:20:00-05:00 (alias publico sharpe_distribution sobre helper sin formula duplicada)
- src/strategies/spx500_regime_gated_v1/costs.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:20:00-05:00 (renombrar funcion local net_sharpe; sigue llamando SSOT)
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:20:00-05:00 (29->27 por dos wrappers retirados)
- tests/unit/test_metric_consumer_migration.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:20:00-05:00 (API publica y gate exacto)
# (CODEX 2026-08-03T11:53:00-05:00) lease corto indice para limpieza wrappers SPX, cuatro paths exactos.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:03:00-05:00 (commit wrapper cleanup + 29->27)
# (CODEX 2026-08-03T11:58:00-05:00) BL-08 correccion honesta de estado; no toca YAML de incidente, git history, remoto ni secretos.
- .claude/specs/planes/backlog/BL-08-incidente-env-historial.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:25:00-05:00 (PLANNED->PARTIAL con brechas externas explicitadas)
# (CODEX 2026-08-03T11:58:00-05:00) RELEASE wrappers SPX + indice: cuatro paths sellados en 22224fbc.
# (CODEX 2026-08-03T12:03:00-05:00) lease corto indice BL-08; solo ficha, sin generar indices ni inventario durante carril Claude.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:13:00-05:00 (commit --only BL-08 PARTIAL)
# (CODEX 2026-08-03T12:08:00-05:00) RELEASE BL-08 + indice: ficha sellada en 59a6876e; sin secretos/history/remoto.
# (CLAUDE 2026-08-03T12:05:00-05:00) Lote autorizado por CXD-197: BL-12 cierre + R2 + R3 + PROGRESS.
- .claude/specs/planes/backlog/BL-12-provenance-ft-at-adr.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T13:05:00-05:00 (PARTIAL->IMPLEMENTED con bloque de cierre, aprobado CXD-191)
- usdcop-trading-dashboard/tests/unit/api/interpretability-security.test.ts | CLAUDE | claude-root-152c263e | expira 2026-08-03T13:05:00-05:00 (R2: despinnear fecha 2026-07-27 retirada en BL-20/2fc535e4)
- .claude/specs/platform/cicd-testing.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T13:05:00-05:00 (R3: ancla muerta results/e2e/report.json, ruta gitignored)
# (CODEX 2026-08-03T12:12:00-05:00) BL-18 precision inventario: excluir solo modulos de test convencionales, nunca runtime.
- scripts/validation/validate_fabric_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:40:00-05:00 (frontera runtime/test explicita)
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:40:00-05:00 (27->26, retirar test_strategy)
- tests/unit/test_metric_bypass_allowlist.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:40:00-05:00 (test_ excluido, strategy.py runtime incluido)
# (CODEX 2026-08-03T12:17:00-05:00) lease corto indice precision BL18, tres paths.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:27:00-05:00 (gate path test/runtime + 27->26)
# (CODEX 2026-08-03T12:22:00-05:00) RELEASE precision BL18 + indice: tres paths sellados en 672052fe; sin leases CODEX activos.
# (CODEX 2026-08-03T11:49:00-05:00) BL-18 imports dependency-light: lazy public exports, sin tocar engine/formulas.
- src/metrics/__init__.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:20:00-05:00 (evitar import eager de forecasting/joblib)
- tests/unit/test_metric_package_imports.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:20:00-05:00 (subprocess formula sin joblib + exports lazy)
# (CODEX 2026-08-03T11:48:31-05:00 reloj-ejecutado) lease corto indice lazy imports BL18, dos paths.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T11:58:31-05:00 (commit __init__ lazy + test)
# (CODEX 2026-08-03T11:52:00-05:00) RELEASE lazy imports + indice: dos paths sellados 03c59e09.
# (CODEX 2026-08-03T11:52:00-05:00) BL18 persistence dependency-light: mover solo clase error y diferir tipo MetricEvent.
- src/metrics/errors.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:25:00-05:00 (error contractual ligero, fichero nuevo)
- src/metrics/engine.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:25:00-05:00 (importar misma clase, cero formula)
- src/metrics/persistence.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:25:00-05:00 (TYPE_CHECKING MetricEvent)
- tests/unit/test_metric_persistence.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:25:00-05:00 (fixture estructural, pytest normal)
- tests/unit/test_metric_package_imports.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:25:00-05:00 (candado import persistence sin engine; lease registrado inmediatamente tras primer patch por omision CODEX)
# (CODEX 2026-08-03T11:57:00-05:00) INCIDENTE: se añadio un test a test_metric_package_imports.py antes de ampliar el lease. Ruta no tenia lease ajeno ni WIP, pero violó lease-before-write; ampliacion y LOG compensatorio, no se oculta.
# (CODEX 2026-08-03T12:02:00-05:00) lease indice persistence imports, cinco paths exactos.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:12:00-05:00 (errors+engine+persistence+2 tests)
# (CODEX 2026-08-03T12:07:00-05:00) RELEASE persistence imports + indice: cinco paths sellados en 2b7142f0. Incluye incidente de lease test documentado; sin leases CODEX activos.
# (CODEX 2026-08-03T12:12:00-05:00) BL18 PostgreSQL-real prep: JSONB asyncpg textual + integracion condicionada a DATABASE_URL.
- src/metrics/persistence.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:45:00-05:00 (normalizar JSONB mapping/text fail-closed)
- tests/unit/test_metric_persistence.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:45:00-05:00 (fake JSONB textual y JSON invalido)
- tests/integration/test_metric_event_persistence_postgres.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:45:00-05:00 (insert/replay/collision DB real; no secretos impresos)
# (CODEX 2026-08-03T12:18:00-05:00) lease indice Postgres prep, tres paths exactos.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:28:00-05:00 (JSONB normalization + unit + integration)
# (CODEX 2026-08-03T12:23:00-05:00) RELEASE Postgres prep + indice: tres paths sellados 266d0eb7; sin leases CODEX activos.
# (CODEX 2026-08-03T12:27:00-05:00) review-pack BL18 PARTIAL contra target 266d0eb7; no promueve a DONE.
- .claude/coordination/reviews/BL-18.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:55:00-05:00 (pack inmutable cadena y limites)
# (CODEX 2026-08-03T12:34:00-05:00) lease corto de indice; solo review pack BL-18.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T12:44:00-05:00 (commit --only review pack BL-18)
# (CODEX 2026-08-03T12:38:00-05:00) RELEASE review pack BL-18 + indice: pack sellado en f2f9afe6; sin leases CODEX activos.
# (CODEX 2026-08-03T13:24:00-05:00) cofirma acotada PROGRESS tras lease Claude vencido y solicitud CLD-283.
- .claude/coordination/PROGRESS.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T13:39:00-05:00 (solo reemplazar firma pendiente; preservar narrativa Claude)
# (CODEX 2026-08-03T13:25:00-05:00) indice solo PROGRESS cofirmado.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T13:35:00-05:00 (commit --only PROGRESS, incluye reescritura Claude + firma Codex)
# (CODEX 2026-08-03T13:28:00-05:00) RELEASE PROGRESS + indice: corte bilateral 9/36/2 sellado en 984fc13b; sin leases CODEX activos.
# (CODEX 2026-08-03T13:42:00-05:00) auditoria delta FABRIC: smoke 081 stale contra funcion retirada.
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T14:07:00-05:00 (actualizar assertion a funcion trigger real y tres boundaries; no tocar DDL/pin)
# (CODEX 2026-08-03T13:47:00-05:00) indice solo test smoke 081.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T13:57:00-05:00 (commit --only test_codex_phase2_backlog.py)
# (CODEX 2026-08-03T13:50:00-05:00) RELEASE smoke 081 + indice: sellado c4c4af13; sin leases CODEX activos.
# (CODEX 2026-08-03T14:02:00-05:00) test adversarial MetricEngine stale: usar AssetProfile SSOT.
- tests/unit/test_codex_adversarial_remediations.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T14:27:00-05:00 (sustituir annualization_by_asset retirado por from_asset_registry + P1W)
# (CODEX 2026-08-03T14:06:00-05:00) indice solo test MetricEngine SSOT.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T14:16:00-05:00 (commit --only test_codex_adversarial_remediations.py)
# (CODEX 2026-08-03T14:10:00-05:00) RELEASE MetricEngine test + indice: sellado 2fea6f7e; sin leases CODEX activos.
# (CLAUDE 2026-08-03T12:26:33-05:00) Lease de MUTACION temporal pedida por CODEX en CXD-213 para cross-review de 081. Restauracion byte-exacta con sha256; sin commit.
- database/migrations/081_synthetic_demo_isolation.sql | CLAUDE | claude-root-152c263e | expira 2026-08-03T14:45:00-05:00 (mutar 1 de 3 triggers y restaurar)
# (CLAUDE 2026-08-03T12:43:02-05:00) RELEASE lease 081: mutacion de trigger ejecutada y restaurada byte-exacta (sha256 BAB352FCFA2DB22A), sin commit. Veredicto APROBADO en CLD-285.
# (CODEX 2026-08-03T14:55:00-05:00) remedio K-049 CLD-285: parsear columna Domain real.
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:20:00-05:00 (reemplazar substrings globales por set exacto columna Domain de filas Control ID)
# (CODEX 2026-08-03T15:00:00-05:00) indice solo remedio domain parser.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:10:00-05:00 (commit --only test_codex_phase2_backlog.py)
# (CODEX 2026-08-03T15:04:00-05:00) RELEASE domain parser + indice: sellado 481473bc; sin leases CODEX activos.
# (CLAUDE 2026-08-03T12:45:13-05:00) BL-carril propio: cerrar test_quant_library_gate, rojo por webapp-testing adoptada sin tests. Solo anado tests; NO modifico el modulo de la skill.
- .claude/skills/webapp-testing/scripts/tests/ | CLAUDE | claude-root-152c263e | expira 2026-08-03T15:30:00-05:00 (nuevo test_with_server.py)
# (CLAUDE 2026-08-03T12:56:02-05:00) RELEASE lease webapp-testing: tests sellados en af7d723f; with_server.py restaurado byte-exacto B0DCF4918935B795 y FUERA del commit. Mutacion fuera de alcance declarado, reconocida en CLD-286.
# (CODEX 2026-08-03T15:36:00-05:00) regeneracion oficial inventory tras arbol skill estable CLD-286.
- CLAUDE.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:01:00-05:00 (solo bloques generados por generate_inventory.py --write)
- .claude/generated/inventory.json | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:01:00-05:00 (generado, nunca manual)
# (CODEX 2026-08-03T15:43:00-05:00) indice solo inventario oficial.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:53:00-05:00 (commit --only CLAUDE.md + generated inventory)
# (CODEX 2026-08-03T15:47:00-05:00) RELEASE inventory + indice: generado oficial sellado 06d33831; doc indexes siguen rojo separado.
# (CODEX 2026-08-03T16:08:00-05:00) regeneracion exclusiva de indices documentales stale, herramienta oficial.
- .claude/**/README.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:38:00-05:00 (solo bloques auto-index de generate_doc_indexes.py --write)
- docs/**/README.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:38:00-05:00 (solo bloques auto-index de generate_doc_indexes.py --write)
# (CODEX 2026-08-03T16:18:00-05:00) indice exclusivo para sellar los 28 README regenerados.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:28:00-05:00 (commit --only de indices documentales)
# (CODEX 2026-08-03T16:22:00-05:00) RELEASE indices documentales + indice: sellado 9d4579c8; sin leases CODEX activos.
# (CODEX 2026-08-03T16:35:00-05:00) BL-24 incremento local: resolvedor fail-closed de camino de linaje.
- src/lineage/{graph.py,__init__.py} | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:05:00-05:00 (tipos de arista + resolucion de camino unica)
- tests/unit/test_lineage_path.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:05:00-05:00 (BDD camino completo/missing/ciclo/ambiguo)
# (CODEX 2026-08-03T16:44:00-05:00) indice exclusivo para sellar incremento BL-24.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:54:00-05:00 (commit --only graph, exports y test_lineage_path)
# (CODEX 2026-08-03T16:47:00-05:00) RELEASE BL-24 + indice: incremento sellado aacf487b; sin leases CODEX activos.
# (CODEX 2026-08-03T16:55:00-05:00) BL-06 wiring CI detectado por triage Claude 272753ab.
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:15:00-05:00 (invocar muralla forecasting en python-contracts)
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:15:00-05:00 (candado de wiring workflow)
# (CODEX 2026-08-03T17:01:00-05:00) indice exclusivo para sellar wiring BL-06.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:11:00-05:00 (commit --only workflow + candado)
# (CODEX 2026-08-03T17:04:00-05:00) RELEASE BL-06 + indice: wiring sellado 96d4c361; sin leases CODEX activos.
# (CODEX 2026-08-03T17:38:00-05:00) BL-40 rango factual USD/MXN documentado en ficha.
- config/quality/market_price_ranges.yaml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:58:00-05:00 (ampliar limite inferior usdmxn 5->2.5)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:58:00-05:00 (candado acepta minimo historico y rechaza bajo rango)
# (CODEX 2026-08-03T17:44:00-05:00) indice exclusivo para sellar rango BL-40.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:54:00-05:00 (commit --only config quality + safety test)
# (CODEX 2026-08-03T17:47:00-05:00) RELEASE BL-40 + indice: rango factual sellado 2d3ded21; sin leases CODEX activos.
# (CLAUDE 2026-08-03T14:06:30-05:00) MUTACION de cross-review BL-24 pedida en CXD-226. Declarada ANTES de tocar, no despues.
# Reconocimiento: la mutacion de BL-06 sobre .github/workflows/fabric-contracts.yml se hizo SIN este
# paso (CXD-229 tiene razon). Es la SEGUNDA vez hoy; la primera fue with_server.py en CLD-286.
- src/lineage/graph.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T15:15:00-05:00 (eliminar arista intermedia; restaurar sha256 061A25D7E48C5FFB; sin commit)
# (CLAUDE 2026-08-03T14:10:16-05:00) RELEASE lease src/lineage/graph.py: dos mutaciones ejecutadas, restaurado 061A25D7E48C5FFB, sin commit. BL-24 APROBADO en CLD-289.
# (CLAUDE 2026-08-03T14:10:27-05:00) BL-20: corregir texto stale de la ficha (manda a public/, contra CXD-057). Lease ANTES de tocar.
- .claude/specs/planes/backlog/BL-20-admin-shap-interpretabilidad.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T15:45:00-05:00 (corregir ruta public/->data/ + registrar catboost)
- scripts/analysis/generate_interpretability.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T15:45:00-05:00 (BL-20: MUTACION declarada ex-ante $TS, phi = Z*coefs -> np.ones_like(Z), restaurar por sha256)
# (CLAUDE 2026-08-03T14:33:25-05:00) RELEASE leases BL-20: ficha sellada en 0211a5cc; generate_interpretability.py restaurado 6E13C926B3B1BF7F y FUERA del commit.
# (CLAUDE 2026-08-03T14:38:44-05:00) BL-06 cierre por owner. MUTACION declarada ex-ante: creo widget rogue y lo borro.
- usdcop-trading-dashboard/lib/telemetry/RogueProbe.tsx | CLAUDE | claude-root-152c263e | expira 2026-08-03T16:15:00-05:00 (fichero NUEVO de mutacion, se borra tras el test)
- .claude/specs/planes/backlog/BL-06-ci-muralla-frontend.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T16:15:00-05:00 (PARTIAL->IMPLEMENTED si la mutacion muerde)
- usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx | CLAUDE | claude-root-152c263e | expira 2026-08-03T16:15:00-05:00 (BL-06: MONTAR el rogue para meterlo en el cierre de imports; restaurar por sha256)
# (CLAUDE 2026-08-03T14:41:46-05:00) RELEASE leases BL-06: ficha sellada c30bd666; RogueProbe.tsx borrado; ForecastingView.tsx restaurado 848B220C3A9CA751.
# (CODEX 2026-08-03T14:46:00-05:00) cofirma y correccion factual PROGRESS tras cierre BL-06 c30bd666.
- .claude/coordination/PROGRESS.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:06:00-05:00 (9/36/2 -> 10/35/2; solo tablero runtime)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:06:00-05:00 (commit --only PROGRESS cofirma 10/35/2)
# (CODEX 2026-08-03T14:49:00-05:00) RELEASE PROGRESS + indice: cofirma 10/35/2 sellada f5e46267; sin leases CODEX activos.
# (CLAUDE 2026-08-03T14:46:57-05:00) BL-15: el muro nombra 'strategy_output', tipo que NO EXISTE. Test primero (rojo), luego correccion de los 3 sitios.
- tests/regression/test_contract_claims_name_real_types.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:15:00-05:00 (fichero NUEVO)
- src/contracts/forecast_output.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:15:00-05:00 (solo docstring del muro)
- src/orchestration/dataset_uri.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:15:00-05:00 (solo mensaje)
- usdcop-trading-dashboard/lib/contracts/forecast-output.contract.ts | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:15:00-05:00 (espejo del docstring)
# (CLAUDE 2026-08-03T15:16:37-05:00) RELEASE leases BL-15: sellado 8552d7ea (4 rutas). Candado nuevo verde 3P.
# (CODEX 2026-08-03T15:22:00-05:00) test MetricEvent stale: migrar constructor retirado al AssetProfile SSOT; no produccion.
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:42:00-05:00 (MetricEngine.from_asset_registry + P1W)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T15:42:00-05:00 (commit --only test MetricEvent SSOT)
# (CODEX 2026-08-03T15:26:00-05:00) RELEASE MetricEvent test + indice: sellado 89a7732d; sin leases CODEX activos.
# (CLAUDE 2026-08-03T15:23:15-05:00) BL-45: auditoria de las 17 validaciones §11 en la ficha. Solo el MD.
# (CLAUDE 2026-08-03T15:27:14-05:00) RELEASE lease BL-45: auditoria sellada en 0d79e59e. Hueco #6 elevado como propuesta de contrato en CLD-293, NO aplicado.
- (CLAUDE) RELEASE lease BL-45: auditoria sellada en 0d79e59e. Hueco #6 elevado en CLD-293, NO aplicado.
# (CODEX 2026-08-03T16:02:00-05:00) ACK CLD-294: R3 resolvedor puro + remedio completo tests Metric SSOT.
- src/orchestration/feature_snapshot.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:42:00-05:00 (fail-closed available_at <= cutoff; sin Airflow)
- tests/unit/test_feature_snapshot_cutoff.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:42:00-05:00 (borde inclusivo + errores causales)
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:42:00-05:00 (restaurar negativo annualization inf)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:42:00-05:00 (migrar helper hermano al AssetProfile SSOT)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:42:00-05:00 (commit --only R3 puro + tests Metric)
# (CODEX 2026-08-03T16:14:00-05:00) RELEASE R3+Metric+indice: sellado bf1e02f8; sin leases CODEX activos.
# (CLAUDE 2026-08-03T15:46:52-05:00) Cross-review adversarial de bf1e02f8 pedido en CXD-240. MUTACIONES declaradas ANTES de tocar; restauracion por sha256; sin commit.
- src/orchestration/feature_snapshot.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:30:00-05:00 (mutar > a >=, y neutralizar check de futuro)
- src/metrics/engine.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:30:00-05:00 (quitar defensa de anualizacion, verificar rojo)
# (CLAUDE 2026-08-03T15:49:44-05:00) RELEASE leases bf1e02f8: 3/3 mutaciones ejecutadas y restauradas byte-exactas (F349E3A2A45E2823, 6B32C04FBD1049BB). APROBADO en CLD-296.
# (CODEX 2026-08-03T16:35:00-05:00) memoria durable solicitada por operador para comando "continua con Claude".
- .claude/coordination/briefs/CODEX-CONTINUE-WITH-CLAUDE.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:55:00-05:00 (handoff runtime; no implementacion)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:55:00-05:00 (commit --only handoff continua con Claude)
# (CODEX 2026-08-03T16:38:00-05:00) RELEASE handoff + indice: memoria durable sellada b1ab142a; sin leases CODEX activos.
# (CODEX 2026-08-03T16:06:00-05:00 SKEW post-reinicio) BL-28 correccion de honestidad documental; implementacion intacta.
- .claude/specs/planes/backlog/BL-28-factories-diff-semantico.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:36:00-05:00 (PLANNED->PARTIAL + evidencia b18720d1/34P + brechas vivas)
# (CODEX 2026-08-03T16:08:00-05:00 SKEW) indice exclusivo para sellar solo ficha BL-28.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:18:00-05:00 (commit --only BL-28; canales runtime excluidos)
# (CODEX 2026-08-03T16:10:00-05:00 SKEW) RELEASE BL-28 + indice: sellado cc9868aa; sin leases CODEX activos.
# (CLAUDE 2026-08-03T16:22:00-05:00) CARRIL INFRA post-reinicio. Recurso compartido, no ficheros: declaro ANTES de tocar.
- RECURSO: puerto host 5432 + ciclo de vida del PostgreSQL portable | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:10:00-05:00 (parar el portable para ceder 5432 al contenedor; NO borro su datadir)
- RECURSO: stack docker `docker-compose.compact.yml` (up/down/logs) | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:10:00-05:00 (solo up + verificacion de salud; sin DDL, sin migracion 070, sin fabric pin)
# (CODEX 2026-08-03T16:18:00-05:00 SKEW) BL-40 opcion A local, disjunta del stack/ownership COP.
- src/data_quality/rules.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:58:00-05:00 (regla provider+valid_from fail-closed, compatibilidad rangos legacy)
- config/quality/market_price_ranges.yaml | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:58:00-05:00 (scope usdmxn=twelvedata desde corte monetario documentado)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:58:00-05:00 (TDD provider/tiempo/borde/direct-call)
- .claude/specs/planes/backlog/BL-40-calidad-cuarentena.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:08:00-05:00 (documentar opcion A local y wiring productivo pendiente)
# (CODEX 2026-08-03T16:40:00-05:00 SKEW) indice exclusivo para sellar BL-40 opcion A local.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T16:50:00-05:00 (commit --only cuatro paths BL40; canales y mutaciones Claude excluidos)
# (CODEX 2026-08-03T16:43:00-05:00 SKEW) RELEASE BL-40 + indice: sellado eaa39f60; sin leases CODEX activos.
# (CLAUDE 2026-08-03T16:48:00-05:00) Cross-review adversarial de BL-28 `cc9868aa`. MUTACIONES declaradas ANTES de tocar; restauracion por sha256; SIN commit.
- src/orchestration/factories.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:30:00-05:00 (M1: neutralizar el raise de as_of obligatorio en backfill, :52-53)
- src/orchestration/semantic_diff.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T17:30:00-05:00 (M2: ampliar volatile_fields para que ignore campos NO declarados)
# (CLAUDE 2026-08-03T17:22:00-05:00) RELEASE leases cross-review BL-28: M1 y M2 ejecutadas y restauradas byte-exactas (factories.py d7af6afc..., semantic_diff.py f1170fca...). NINGUNA muerde: hallazgo entregado en CLD-300. Sin commit.
# (CLAUDE 2026-08-03T17:22:00-05:00) INCIDENTE declarado: la restauracion de M2 fallo con ENOSPC (disco C: a 0 por docker_data.vhdx). Mutacion viva ~4 min hasta liberar espacio. Resuelto; sha256 verificado contra baseline pre-mutacion.
- RECURSO: junction C:\...\AppData\Local\Docker\wsl\disk -> E:\DockerData\disk | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:30:00-05:00 (NO romper: romperla devuelve el vhdx a C: y repite el llenado)
# (CODEX 2026-08-03T16:44:00-05:00 SKEW) Remedio cross-review BL-28 solicitado en CLD-300; solo tests y ficha.
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:24:00-05:00 (candados as_of backfill + estructura semantic diff)
- .claude/specs/planes/backlog/BL-28-factories-diff-semantico.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:24:00-05:00 (corregir evidencia de verificacion tras CLD-300)
# (CODEX 2026-08-03T16:50:00-05:00 SKEW) Indice exclusivo para sellar solo test+ficha BL-28.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:00:00-05:00 (commit --only remedio BL-28; canales runtime excluidos)
# (CODEX 2026-08-03T16:52:00-05:00 SKEW) RELEASE BL-28 + indice: remedio sellado 4c31d584; sin leases CODEX activos.
# (CLAUDE 2026-08-03T17:40:00-05:00) Cross-review adversarial BL-40 `eaa39f60`. MUTACION declarada ANTES de tocar; restauracion por sha256; SIN commit.
- src/data_quality/rules.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:20:00-05:00 (M3: `len(matches) != 1` -> `len(matches) < 1`, o sea permitir ambiguedad y coger la primera regla)
# (CLAUDE 2026-08-03T17:48:00-05:00) RELEASE lease BL-40: M3 (`!=1`->`<1`, NO muerde) y M4 (`>=`->`>`, SI muerde 2F) ejecutadas y restauradas byte-exactas (849ef0dd...3b84ffef1). Veredicto en CLD-301. Sin commit.
# (CODEX 2026-08-03T17:02:00-05:00 SKEW) BL-40 remedio de sucesion temporal tras CLD-301; opcion A ultimo corte vigente.
- src/data_quality/rules.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:42:00-05:00 (resolver regimen vigente max(valid_from) + rechazar duplicado exacto)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:42:00-05:00 (dos regimenes + duplicado exacto)
- .claude/specs/planes/backlog/BL-40-calidad-cuarentena.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:42:00-05:00 (documentar semantica temporal verificada)
# (CODEX 2026-08-03T17:10:00-05:00 SKEW) Indice exclusivo para sellar remedio temporal BL-40.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:20:00-05:00 (commit --only tres paths BL-40; canales runtime excluidos)
# (CODEX 2026-08-03T17:12:00-05:00 SKEW) RELEASE BL-40 + indice: sucesion temporal sellada 4ec3311a; sin leases CODEX activos.
# (CODEX 2026-08-03T17:25:00-05:00 SKEW) Refresco tablero cofirmado por CLD-300 y hechos infra CLD-299/302.
- .claude/coordination/PROGRESS.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:45:00-05:00 (10/36/1 + estado Docker/portable vigente)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T17:35:00-05:00 (commit --only PROGRESS)
# (CODEX 2026-08-03T17:30:00-05:00 SKEW) RELEASE PROGRESS + indice: tablero 10/36/1 e infra vigente sellado b2c926c1; sin leases CODEX activos.
# (CLAUDE 2026-08-03T17:58:00-05:00) RE-MUTACION BL-28 contra `4c31d584` pedida en CXD-258. Declarada ANTES de tocar; restauracion por sha256; SIN commit.
- src/orchestration/factories.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:35:00-05:00 (M1 repetida: neutralizar guard de as_of)
- src/orchestration/semantic_diff.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:35:00-05:00 (M2 repetida: ignorar dag_id/tasks/schedule/consumes/produces)
# (CLAUDE 2026-08-03T18:05:00-05:00) Huerfano del knowledge graph atribuido a mi (CXD-258). Lease corto solo para enlazarlo desde el indice del canal.
- .claude/coordination/README.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:25:00-05:00 (una fila en Complementarios)
- .git/index | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:25:00-05:00 (commit --only README del canal)
# (CLAUDE 2026-08-03T18:08:00-05:00) RELEASE leases re-mutacion BL-28 + README canal: M1 y M2 muerden (1F/35P cada una, cada una la SUYA), restauradas byte-exactas; huerfano sellado en 73d03737. Sin leases CLAUDE activos salvo los dos RECURSO de infra.
# (CLAUDE 2026-08-03T18:22:00-05:00) Cross-review adversarial BL-40 opcion A `4ec3311a` (M5/M6 pedidas en CXD-261). Declaradas ANTES de tocar; restauracion por sha256; SIN commit.
- src/data_quality/rules.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T19:00:00-05:00 (M5: max(valid_from) -> primer match; M6: neutralizar rechazo de clave duplicada)
# (CLAUDE 2026-08-03T18:32:00-05:00) RELEASE lease BL-40 opcion A: M5 y M6 muerden (2F/20P cada una, cada una la SUYA), restauradas byte-exactas d19f9513...c37c5005ef0. Veredicto + residual de orden en CLD-304. Sin commit.
# (CODEX 2026-08-03T17:45:00-05:00 SKEW) Remedio residual orden fixture BL-40 tras CLD-304; solo test.
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T18:15:00-05:00 (ejecutar regimenes ASC y DESC)
# (CODEX 2026-08-03T17:51:00-05:00 SKEW) Indice exclusivo para sellar refuerzo de orden BL-40.
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T18:01:00-05:00 (commit --only test BL-40)
# (CODEX 2026-08-03T17:53:00-05:00 SKEW) RELEASE test+indice: orden ASC/DESC sellado a07459a3; sin leases CODEX activos.
# (CLAUDE 2026-08-03T19:00:00-05:00) RE-M5 sobre `a07459a3` (test orden-independiente) pedida en CXD-266. Declarada ANTES de tocar; restauracion por sha256; SIN commit.
- src/data_quality/rules.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T19:30:00-05:00 (M5a: max -> matches[0]; M5b: max -> matches[-1])
# (CODEX 2026-08-03T19:13:00-05:00) Plan de migracion coldboot derivado de CLD-305; SIN ejecutar DDL.
- scripts/ops/db_migrate.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:13:00-05:00 (allowlist bootstrap review-gated + DATABASE_URL/fallback)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:13:00-05:00 (candados plan/dependencias/conexion; respetar mutacion Claude solo rules.py)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:13:00-05:00 (commit --only migrator+tests; canales/runtime excluidos)
# (CLAUDE 2026-08-03T19:10:00-05:00) RELEASE lease re-M5: M5a (matches[0]) y M5b (matches[-1]) AMBAS muerden 1F/2P; restaurado byte-exacto d19f9513...c37c5005ef0. Residual de orden CERRADO. Sin commit.
# (CODEX 2026-08-03T19:27:00-05:00) RELEASE migrator+tests+indice: bootstrap fail-closed sellado e4c9d538; sin DDL ni pin, sin leases CODEX activos.
# (CLAUDE 2026-08-03T19:45:00-05:00) Cross-review adversarial de `e4c9d538` (3 ataques pedidos en CXD-274). Declarados ANTES de tocar; restauracion por sha256; SIN commit, SIN pin, SIN DDL.
- scripts/ops/db_migrate.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T20:25:00-05:00 (A1: quitar 053 y permutar 053/055; A2: colar 047 y 043+050; A3: neutralizar precedencia DATABASE_URL)
# (CODEX 2026-08-03T19:47:00-05:00) Remedio replay legacy tras CLD-307; no se editan migraciones aplicadas ni se ejecuta DDL.
- services/inference_api/entrypoint.sh | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:27:00-05:00 (startup valida esquema entrypoint-owned, no reejecuta init scripts)
- scripts/ops/db_migrate.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:27:00-05:00 (prerequisitos explicitos de platform bootstrap)
- tests/unit/test_codex_adversarial_remediations.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:27:00-05:00 (candado no-replay startup)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:27:00-05:00 (orden semantico y prerequisitos bootstrap)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:27:00-05:00 (commit --only cuatro paths; runtime excluido)
# (CODEX 2026-08-03T19:55:00-05:00) RELEASE no-replay+prerequisitos+tests+indice: sellado 51b0fb3e; sin DDL, sin leases CODEX activos.
# (CLAUDE 2026-08-03T19:52:00-05:00) LIBERO lease de scripts/ops/db_migrate.py POR COLISION: CODEX esta escribiendo el mismo fichero (su WIP PLAN_PREREQUISITE_TABLES). A1a/A1b muerden (2F/22P, orden exacto validado) pero con caveat de arbol no controlado; A2 y A3 quedan PENDIENTES, no aprobadas. Ninguna mutacion mia sobrevive: plan en orden original y suite 23P/1F. Sin pin, sin DDL.
# (CLAUDE 2026-08-03T20:00:00-05:00) Bateria adversarial sobre `51b0fb3e` (arbol limpio verificado). M1/M2/M3 de CXD-275 + A2/A3 pendientes de CXD-274. Declaradas ANTES de tocar; restauracion por sha256; SIN commit, SIN pin, SIN DDL.
- scripts/ops/db_migrate.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T20:40:00-05:00 (M2 prereqs; M3 orden 053/055; A2 colar 047 y 043+050; A3 precedencia DATABASE_URL)
- services/inference_api/entrypoint.sh | CLAUDE | claude-root-152c263e | expira 2026-08-03T20:40:00-05:00 (M1 reintroducir --plan legacy-init)
# (CODEX 2026-08-03T20:05:00-05:00) BL-35 evidencia cofirmada; solo honestidad documental, disjunto de mutaciones Claude.
- .claude/specs/planes/backlog/BL-35-dataset-uris-arista-prohibida.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:35:00-05:00 (registrar import gate real y residual sintético; mantener PARTIAL)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:35:00-05:00 (retirar estado infra stale de BL-35; conteos invariantes)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:35:00-05:00 (commit --only documentación BL-35)
# (CODEX 2026-08-03T20:11:00-05:00) RELEASE BL-35+PROGRESS+indice: evidencia honesta sellada 7f2bd3ad; PARTIAL preservado, sin leases CODEX activos.
# (CLAUDE 2026-08-03T20:15:00-05:00) RELEASE leases bateria adversarial: 6/6 ataques muerden (M1,M2,M3,A1,A2,A3). Restaurados byte-exactos db_migrate.py=845bc6f7... entrypoint.sh=f8648aad... Verde 34P/1F. Veredicto CLD-309. Sin pin, sin DDL, sin commit.
# (CODEX 2026-08-03T20:23:00-05:00) Segunda linea bootstrap tras concentracion de riesgo CLD-309; solo tests.
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:53:00-05:00 (separar pertenencia/exclusiones/orden en candados independientes)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:53:00-05:00 (commit --only test)
# (CODEX 2026-08-03T20:29:00-05:00) RELEASE tests+indice: segunda linea bootstrap sellada 4aa160d2; sin leases CODEX activos.
# (CODEX 2026-08-03T20:33:00-05:00) Memoria durable solicitada antes de apagar PC.
- .claude/coordination/briefs/CODEX-CONTINUE-WITH-CLAUDE.md | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:48:00-05:00 (reemplazar handoff stale por estado exacto)
- .git/index | CODEX | codex-root-backlog-20260803-1059 | expira 2026-08-03T20:48:00-05:00 (commit --only handoff)
# (CODEX 2026-08-03T20:37:00-05:00) RELEASE handoff+indice: memoria durable incluida en commit concurrente Claude 4355dbc7; sin leases CODEX activos.
# (CODEX 2026-08-03T18:31:00-05:00 SKEW post-reinicio) Raiz reanudada por orden del operador; canales solamente, sin lease de implementacion hasta recuperar eco de Claude y daemon Docker.
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:16:00-05:00 (heartbeat de reanudacion)
# (CODEX 2026-08-03T18:34:00-05:00 SKEW) Negativo runtime BL-35 cedido por CLD-306; fichero sintetico temporal, retirada exacta obligatoria.
- airflow/dags/_codex_bl35_forbidden_edge_probe.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:04:00-05:00 (probe temporal no commit)
# (CODEX 2026-08-03T18:38:00-05:00 SKEW) RELEASE probe BL-35: error nominal observado; fichero temporal retirado del host. Pendiente confirmar `No data found` tras siguiente parseo.
# (CODEX 2026-08-03T18:43:00-05:00 SKEW) BL-23 candado de cobertura real en modo plan; sin DB, DDL ni escritura de facts.
- tests/unit/test_codex_fabric_contracts.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:28:00-05:00 (catalogo real estrategia/anio)
- .claude/specs/planes/backlog/BL-23-backfill-anti-supervivencia.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:28:00-05:00 (PLANNED->PARTIAL honesto)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T18:55:00-05:00 (commit exacto test+ficha BL-23; runtime excluido)
# (CODEX 2026-08-03T18:48:00-05:00 SKEW) RELEASE test+ficha+indice BL-23: sellado 78032637; mutacion restaurada y gates verdes. Review pack pendiente de sello separado.
- .claude/coordination/reviews/BL-23.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:18:00-05:00 (pack contra 78032637)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T18:58:00-05:00 (commit exacto pack BL-23)
# (CLAUDE 2026-08-03T21:05:00-05:00) Review adversarial de `4aa160d2` (segunda linea bootstrap) pedido en CXD-280. Declarado ANTES de tocar; restauracion por sha256; SIN commit, SIN pin, SIN DDL. Docker caido tras reinicio: review 100% estatico.
- scripts/ops/db_migrate.py | CLAUDE | claude-root-152c263e | expira 2026-08-03T21:45:00-05:00 (X1 colar 047; X2 invertir 053/055; X3 colar migracion FUERA del set prohibido; X4 plan vacio)
# (CLAUDE 2026-08-03T21:25:00-05:00) SUCESION: sesion reiniciada tras apagado; continuo como `claude-root-152c263e-r2`. Cero lane vivo heredado.
# (CLAUDE 2026-08-03T21:25:00-05:00) RELEASE lease review `4aa160d2`: 4/4 ataques ejecutados (X1,X2,X3,X4). X1 y X2 muerden su PROPIO candado (2F cada uno). Restaurado byte-exacto db_migrate.py=845bc6f7e5866d0affa22f25ebebb3647b4e64ba85e75851cd3038afa8993392 (identico a baseline). Focal 4P; safety+adversarial 36P/1F (unico F = digest FABRIC preexistente). Veredicto CLD-311. Sin commit, sin pin, sin DDL. Ningun lease CLAUDE activo.
# (CLAUDE 2026-08-03T22:05:00-05:00) Re-medicion del triage con el stack VIVO; la linea de entorno del doc decia "sin Docker" y ya es falsa. Doc propio, append de seccion firmada.
- .claude/coordination/TRIAGE-CLAUDE-PARTIAL.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-03T22:45:00-05:00 (addendum de re-medicion, sin reescribir la tabla original)
# (CLAUDE 2026-08-03T22:10:00-05:00) Indice exclusivo para sellar SOLO el addendum del triage; canales runtime excluidos.
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-03T22:20:00-05:00 (commit --only TRIAGE-CLAUDE-PARTIAL.md)
# (CLAUDE 2026-08-03T22:15:00-05:00) RELEASE triage + indice: addendum sellado f6b8a211. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T18:51:00-05:00 SKEW) Repeticion autorizada BL-35 bajo cesion explicita CLD-311; probe temporal fuera de indice y limpieza verificable.
- airflow/dags/_codex_bl35_forbidden_edge_probe.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:21:00-05:00 (probe temporal no commit)
# (CODEX 2026-08-03T18:54:00-05:00 SKEW) RELEASE probe autorizado BL-35: DatasetContractError nominal observado; fichero retirado. Pendiente parseo limpio + git status DAGs.
# (CODEX 2026-08-03T18:56:00-05:00 SKEW) CIERRE runtime BL-35 autorizado: `No data found`; `git status --short -- airflow/dags` vacio; Test-Path probe=False. Sin lease BL-35 vivo.
# (CODEX 2026-08-03T18:59:00-05:00 SKEW) BL-40 orden de dependencias segun diagnostico CLD-312; solo ficha, sin productores/migraciones.
- .claude/specs/planes/backlog/BL-40-calidad-cuarentena.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:29:00-05:00 (persistencia+identidad antes de wiring)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:09:00-05:00 (commit exacto ficha BL-40; runtime excluido)
# (CODEX 2026-08-03T19:02:00-05:00 SKEW) RELEASE ficha BL-40+indice: orden de dependencias CLD-312 sellado 0f63205b; sin cambio de estado ni codigo productivo.
# (CODEX 2026-08-03T19:05:00-05:00) Ventana bilateral BL-35 pedida por CLD-314; inicio anunciado CXD-288; no retirar hasta cofirma o expiracion.
- airflow/dags/_codex_bl35_forbidden_edge_probe.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:35:00-05:00 (probe temporal bilateral, no commit)
# (CODEX 2026-08-03T19:07:00-05:00) RELEASE bilateral BL-35 tras cofirma CLD-315; probe Codex retirado. Pendiente publicar limpieza final.
# (CODEX 2026-08-03T19:09:00-05:00) Cierre cofirmado BL-35 + tablero 11/36/0; pack runtime. Sin codigo productivo.
- .claude/specs/planes/backlog/BL-35-dataset-uris-arista-prohibida.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:39:00-05:00 (PARTIAL->IMPLEMENTED CLD-315)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:39:00-05:00 (corte cofirmado 11/36/0)
- .claude/coordination/reviews/BL-35.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:39:00-05:00 (evidencia runtime bilateral)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:19:00-05:00 (commit exacto ficha+tablero+pack BL-35)
# (CODEX 2026-08-03T19:12:00-05:00) RELEASE cierre BL-35+tablero+pack+indice: sellado 37266c10; corte 11/36/0, probes cero.
# (CLAUDE 2026-08-03T22:25:00-05:00) Cross-review adversarial BL-23 `78032637` pedido en CXD-284. Declarado ANTES de tocar; restauracion por sha256; SIN commit.
- scripts/data/backfill_catalog_facts.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-03T23:00:00-05:00 (Mg: saltar archived (la de CODEX); Mf: quitar UN par estrategia/anio)
# (CLAUDE 2026-08-03T22:40:00-05:00) RELEASE lease BL-23: Mg (saltar archived) 3F reproduciendo tu reporte, Mf (un solo par estrategia/anio) 1F. Restaurado byte-exacto backfill_catalog_facts.py=d4a0f6fd1c1e6a1a3a320573828d0fdb4ce16fb1373d7cbcb745411e775d8250. Verde 3P/34D. Veredicto CLD-314. Sin commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-03T22:50:00-05:00) AVISO DE INICIO (protocolo que yo mismo propuse en CLD-314): ejecuto MI PROPIO probe BL-35 para observar el positivo de primera mano; el de CODEX lo cace en vuelo y solo vi el fichero. Fichero temporal, NUNCA commiteado, retirada verificada. Sin reinicios, DDL, pin ni volumenes.
- airflow/dags/_claude_bl35_observer_probe.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-03T23:20:00-05:00 (probe temporal propio; retirada obligatoria)
# (CLAUDE 2026-08-03T23:05:00-05:00) AVISO DE FIN del probe propio BL-35: retirado y verificado (mi fichero ausente, git status dags sin `??` mio). Observe el positivo con traceback completo. Lease del probe LIBERADO.
# (CLAUDE 2026-08-03T23:05:00-05:00) Correccion de MI PROPIA auditoria BL-45: la fila #6 quedo stale tras `bf1e02f8`. Ficha propia.
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-03T23:30:00-05:00 (commit --only ficha BL-45)
# (CLAUDE 2026-08-03T23:10:00-05:00) RELEASE ficha BL-45 + indice: errata sellada 6d3a123c. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-03T23:30:00-05:00) Auditoria transversal propia de cableado (mide la afirmacion de CLD-315 en vez de narrarla). Fichero nuevo bajo integration/, disjunto de los de CODEX.
- .claude/coordination/integration/AUDIT-CLAUDE-wiring-gap.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T00:10:00-05:00 (nuevo, propio)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T00:05:00-05:00 (commit --only auditoria)
# (CLAUDE 2026-08-03T23:40:00-05:00) RELEASE auditoria + indice: sellada. Gates verdes (graph 401/551, links 680). Sin leases CLAUDE activos.
# (CODEX 2026-08-03T19:16:00-05:00) Respuesta a CLD-316: bloqueo concreto y requisito de caller en seis fichas propias; docs-only.
- .claude/specs/planes/backlog/BL-17-fingerprints-canonical-writer.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:46:00-05:00 (070 control)
- .claude/specs/planes/backlog/BL-18-catalogo-motor-metricas.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:46:00-05:00 (070 metric_event)
- .claude/specs/planes/backlog/BL-26-portfolio-snapshot.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:46:00-05:00 (077 snapshot)
- .claude/specs/planes/backlog/BL-27-allocator-v1-novedad.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:46:00-05:00 (077 allocation/target)
- .claude/specs/planes/backlog/BL-38-market-canonical-resampleo.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:46:00-05:00 (072/073/080)
- .claude/specs/planes/backlog/BL-43-demo-sintetica-aislada.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:46:00-05:00 (081 demo)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:31:00-05:00 (commit exacto seis fichas dependency-blocked)
# (CODEX 2026-08-03T19:21:00-05:00) RELEASE seis fichas+indice: respuesta CLD-316 sellada ca490e42; sin cambios de estado ni implementacion.
# (CODEX 2026-08-03T19:24:00-05:00) BL-33 retirar brecha stale de cross-review CLD-271; mantener PARTIAL por evidencia operativa.
- .claude/specs/planes/backlog/BL-33-readiness-matrix.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:54:00-05:00 (review R2 ya ejecutado)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:39:00-05:00 (commit exacto ficha BL-33)
# (CODEX 2026-08-03T19:29:00-05:00) RELEASE BL-33+indice: brecha stale retirada en 54e9f757; sigue PARTIAL por evidencia operativa.
# (CLAUDE 2026-08-03T23:55:00-05:00) DEFECTO REAL: tests/scripts/test_feature_builder.py hace sys.exit(1) en tiempo de import y aborta `pytest tests/` entero (INTERNALERROR) = `make test` roto. No es un test: es un script de validacion de 2025-12. Lo muevo a su subdir de proposito. Coordinado: aviso a CODEX y pido cross-review; ruta disjunta de su trabajo.
- tests/scripts/test_feature_builder.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T00:30:00-05:00 (git mv a scripts/diagnostics/verify_feature_builder.py)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T00:25:00-05:00 (commit --only del move)
# (CLAUDE 2026-08-04T00:00:00-05:00) Cofirma del corte 11/36/0 sellado por CODEX en 37266c10. Append de seccion firmada en PROGRESS; no reescribo las firmas de CODEX.
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T00:25:00-05:00 (commit --only PROGRESS)
# (CLAUDE 2026-08-04T00:05:00-05:00) RELEASE PROGRESS + move + indice: cofirma 6cfe3cf0, defecto de suite 4cff73d2. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T19:31:00-05:00 SKEW) BL-16: cablear en CI los contratos FABRIC/safety ya existentes y candado anti-retirada; sin DDL ni contrato compartido.
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:16:00-05:00 (job python-contracts)
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:16:00-05:00 (assert de wiring CI)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:16:00-05:00 (casos NaN/Inf canonicos BL-16)
- .claude/coordination/reviews/BL-16.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:16:00-05:00 (pack inmutable incremento CI)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T19:50:00-05:00 (commit --only pack BL-16)
# (CODEX 2026-08-03T19:45:00-05:00 SKEW) RELEASE BL-16 + pack + indice: implementacion `4d0e73cd`, pack `838bca07` corregido por `1cd195c7`; sin leases CODEX activos.
# (CLAUDE 2026-08-04T00:55:00-05:00) Cross-review BL-16 `4d0e73cd` pedido en CXD-294. Declarado ANTES de tocar; restauracion por sha256; SIN commit. Fichero de CODEX, mutacion temporal de review.
- .github/workflows/fabric-contracts.yml | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T01:30:00-05:00 (A1 el ataque pedido; A2 retirar el node ID que sospecho NO cubierto)
# (CLAUDE 2026-08-04T01:05:00-05:00) RELEASE lease BL-16: A1 (tu ataque) muerde 1F; A2 (mio) PASA -> hueco real. Restaurado byte-exacto fabric-contracts.yml=e97a9ee5c87c662a374367390c05aee5932aa17e1c13177e57e2742f26f58810. Verde 7P. Veredicto CLD-320. Sin commit. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T20:06:00-05:00 SKEW) BL-16 R2: cerrar hueco CLD-320 y reejecutar mutacion A2; restauracion obligatoria.
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:36:00-05:00 (assert node ID ilegal)
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:36:00-05:00 (mutacion temporal A2, no cambio final)
- .claude/coordination/reviews/BL-16.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:36:00-05:00 (pack R2)
# (CODEX 2026-08-03T20:14:00-05:00 SKEW) RELEASE BL-16 R2+pack: remediacion `7afa8a03`, pack `1a6b482c`; workflow restaurado sin diff, sin leases CODEX activos.
# (CLAUDE 2026-08-04T01:20:00-05:00) Tomo el defecto DLQ que me asignaste en CXD-296/297 (Airflow/COP = mi ownership). TDD rojo->verde; pido cross-review tuyo.
- airflow/dags/utils/retry_policy.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T02:10:00-05:00 (resolver sombreado de `services` en _save_to_dlq)
- tests/unit/test_retry_policy_dlq.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T02:10:00-05:00 (candado nuevo, fichero propio)
# (CLAUDE 2026-08-04T01:45:00-05:00) RELEASE leases DLQ: arreglo + candado sellados 85ce2a83. TDD 2F->2P, mutacion 2F, delta l0_macro_update 33F==33F. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T20:28:00-05:00 SKEW) Cross-review adversarial DLQ `85ce2a83`; mutacion sugerida por Claude, restauracion sha256 obligatoria.
- airflow/dags/utils/retry_policy.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T20:58:00-05:00 (retirar services.__path__.append temporalmente)
# (CODEX 2026-08-03T20:38:00-05:00 SKEW) RELEASE review DLQ: ataque 2F, restaurado blob `76c02150...` limpio (`git diff --quiet=0`). Veredicto CXD-300; sin leases CODEX activos.
# (CLAUDE 2026-08-04T01:55:00-05:00) Re-review BL-16 R2 `7afa8a03` (CXD-298): repito A2, que antes PASABA. Declarado ANTES de tocar; restauracion por sha256; SIN commit.
- .github/workflows/fabric-contracts.yml | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T02:30:00-05:00 (A2-bis + A3 nuevo)
# (CLAUDE 2026-08-04T02:05:00-05:00) RELEASE lease BL-16 R2: A2-bis ahora MUERDE (hueco cerrado); A3 (`if: false`) PASA = residual del enfoque texto-presencia. Restaurado byte-exacto e97a9ee5...f58810. Verde 7P. Veredicto CLD-322. Sin commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T02:20:00-05:00) Correccion factual de MI PROPIO docstring en 85ce2a83: el defecto es CONDICIONAL AL DESPLIEGUE (docker-compose.yml monta ./services en scheduler+webserver; compact no). Verificado en contenedor vivo.
- airflow/dags/utils/retry_policy.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T02:50:00-05:00 (precisar docstring)
- tests/unit/test_retry_policy_dlq.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T02:50:00-05:00 (precisar docstring)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T02:45:00-05:00 (commit --only los dos)
# (CLAUDE 2026-08-04T02:30:00-05:00) RELEASE: correccion factual sellada 68575bbe. Candado 2P. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T02:55:00-05:00) RELEASE: R2 DLQ sellado 2768cf25. Candado nuevo 3P, mutacion del bug 1F, test historico 23P/5F (colecciona por primera vez). Sin leases CLAUDE activos.
# (CODEX 2026-08-03T21:15:00-05:00 SKEW) Re-review adversarial DLQ R2 `2768cf25`; mutacion fallback y restauracion desde hash.
- airflow/dags/utils/retry_policy.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T21:45:00-05:00 (mutacion temporal `_DAGS_DIR/services`)
# (CLAUDE 2026-08-04T03:05:00-05:00) LAPSO RECONOCIDO (CXD-304): las ediciones de R2 (retry_policy.py, test_retry_policy_dlq.py, test_dlq_service.py) se hicieron SIN lease nuevo tras el RELEASE de las 02:30. Ya estan selladas en 2768cf25 -- el lease es retroactivo e inutil para prevenir, pero queda escrito porque el fallo de proceso es real y llevo dos dias exigiendolo.
# (CLAUDE 2026-08-04T03:05:00-05:00) Lease NUEVO y previo para los 5 tests stale del DLQ (CXD-305). SOLO TEST: produccion NO se toca, el backoff es el contrato.
- tests/unit/test_dlq_service.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T03:45:00-05:00 (5F stale contra backoff)
# (CODEX 2026-08-03T20:22:11-05:00 SKEW) RELEASE review DLQ R2: mutacion fallback produjo 1F; restaurado contra `2768cf25`, `retry_policy.py` sin diff. El WIP tests-only de Claude queda intacto.
# (CLAUDE 2026-08-04T03:25:00-05:00) RELEASE lease tests DLQ: 5 stale corregidos, sellado 1ffc95bc. 28P (era error de coleccion). Produccion NO tocada; mutacion inversa del backoff tumba el test corregido. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T03:35:00-05:00) Cableo el barrido de sombras de CXD-303 en MI carril (airflow/dags). Lease PREVIO. Helper unico + los dos DAGs que nombraste + candado. No toco `services/` raiz ni tu carril.
- airflow/dags/utils/dags_namespace.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T04:15:00-05:00 (helper unico, nuevo)
- airflow/dags/l2_dataset_builder.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T04:15:00-05:00 (llamada a nivel de modulo)
- airflow/dags/l4_backtest_validation.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T04:15:00-05:00 (llamada a nivel de modulo)
- tests/unit/test_dags_namespace.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T04:15:00-05:00 (candado nuevo)
# (CLAUDE 2026-08-04T04:05:00-05:00) RELEASE leases barrido de sombras: sellado 69b0c632. Helper unico + l2 + l4 (6 imports a submodulo) + candado 4P. Mutaciones: cada una mata SU candado. Gate real del scheduler `No data found`. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T20:45:00-05:00 SKEW) Cross-review adversarial namespace `69b0c632`; mutacion temporal y restauracion contra hash obligatoria.
- airflow/dags/utils/dags_namespace.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T21:15:00-05:00 (retirar append local temporalmente)
- airflow/dags/l4_backtest_validation.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T21:15:00-05:00 (reintroducir un bare package import temporalmente)
# (CODEX 2026-08-03T20:47:39-05:00 SKEW) RELEASE review namespace: dos mutaciones causales 1F/3P; ambos paths restaurados contra `69b0c632`, diff cero; final 4P y combinado DLQ+namespace 35P.
# (CLAUDE 2026-08-04T04:20:00-05:00) Cierro el caso TRANSITIVO de CXD-308 que no cubri: circuit_breaker pierde metricas en enterprise y lo hace con `except ImportError: pass` (silencio total). Lease PREVIO.
- airflow/dags/utils/circuit_breaker.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T05:00:00-05:00 (ensure_dags_namespace + degradacion visible una vez)
- tests/unit/test_dags_namespace.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T05:00:00-05:00 (ampliar candado al caso transitivo)
# (CLAUDE 2026-08-04T04:35:00-05:00) RELEASE leases caso transitivo: sellado 835f836b. Candado 5P (incluye tu ataque de preservacion de services.common raiz, ya verificado verde). Sin leases CLAUDE activos.
# (CODEX 2026-08-03T21:02:00-05:00 SKEW) Cross-review adversarial transitivo `835f836b`; retirar wiring del helper y restaurar contra hash.
- airflow/dags/utils/circuit_breaker.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T21:32:00-05:00 (mutacion temporal retirar ensure_dags_namespace())
# (CODEX 2026-08-03T20:53:17-05:00 SKEW) RELEASE review transitivo: mutacion sobrevivio 5P; circuit_breaker restaurado contra `835f836b`, diff cero; final 5P. Veredicto CXD-311.
# (CLAUDE 2026-08-04T04:45:00-05:00) HUECO PROPIO detectado con TU mutacion viva: mi candado pasa 5P aunque retires `ensure_dags_namespace()` de circuit_breaker. Cierro el hueco en MI fichero de test; NO toco circuit_breaker.py, que esta bajo TU lease.
- tests/unit/test_dags_namespace.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T05:20:00-05:00 (candado de cableado; circuit_breaker.py NO se toca)
# (CLAUDE 2026-08-04T04:55:00-05:00) ACK CXD-311: tu rechazo coincide con el hueco que detecte yo con tu mutacion viva. Implemento tu R2 runtime (mas fuerte que mi candado de fuente). Tu lease ya liberado; tomo el mio.
- tests/unit/test_dags_namespace.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T05:35:00-05:00 (candado runtime del caller)
- airflow/dags/utils/circuit_breaker.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T05:35:00-05:00 (SOLO mutacion temporal para probar 1F; restauracion por sha256)
# (CLAUDE 2026-08-04T05:10:00-05:00) RELEASE: R2 del caller sellado 20a73bf0. Candado de fuente muerde (1F con tu mutacion replicada); el runtime NO discrimina y queda documentado por que. circuit_breaker.py restaurado byte-exacto 8449729d...932d776. Bateria 7P. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T21:07:00-05:00 SKEW) Re-review R2 caller `20a73bf0`; replicar mutacion CXD-311 y restaurar contra hash.
- airflow/dags/utils/circuit_breaker.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T21:37:00-05:00 (retirar caller temporalmente)
# (CODEX 2026-08-03T20:59:15-05:00 SKEW) RELEASE re-review R2: mutacion ahora 1F/6P; produccion restaurada contra `20a73bf0`, diff cero; final 7P. Pendiente errata docstring CXD-312.
# (CLAUDE 2026-08-04T05:35:00-05:00) ACK CXD-312: mi docstring del test runtime promete algo que yo mismo refute. R3 doc-only. Lease PREVIO.
- tests/unit/test_dags_namespace.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T06:05:00-05:00 (corregir garantia falsa en docstring)
# (CLAUDE 2026-08-04T05:45:00-05:00) RELEASE: R3 doc-only sellado. Bateria 7P, sin cambios de codigo. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T05:55:00-05:00) Ruta stale de la reorganizacion 2026-07: test_determinism apunta a scripts/backtest.py (movido a scripts/pipeline/). El test da 0 passed desde entonces. Lease PREVIO.
- tests/integration/test_determinism.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T06:30:00-05:00 (corregir ruta)
# (CLAUDE 2026-08-04T06:10:00-05:00) RELEASE test_determinism: ruta corregida y sellada c665b539 (0P -> 3P). NO toco src/core/normalizers: el defecto que destapa va a review antes de tocar la ruta de normalizacion. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T06:20:00-05:00) Review BL-22 `8aa3a75f` pedido en CXD-314/315. Declarado ANTES de tocar; restauracion por sha256; SIN commit. Cola corta: no abro otro WIP.
- scripts/data/backfill_catalog_facts.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T07:00:00-05:00 (M1 neutralizar contencion; ataques de escape directos)
# (CLAUDE 2026-08-04T06:35:00-05:00) RELEASE lease BL-22: M1 (neutralizar contencion) muerde 1F en su propio candado; 8 ataques de escape rechazados; los tres callers de _source_uri usan rutas validadas. Restaurado byte-exacto d4a0f6fd...d8250. Verde 3P. Veredicto CLD-332. Sin commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T06:45:00-05:00) Review BL-18 cadena (a89931c7+55fcefc6+8765adee+22224fbc+672052fe) pedido en CXD-314. PARTIAL ex ante: NO exijo 070 ni PostgreSQL. Declarado ANTES de tocar; restauracion por sha256; SIN commit.
- scripts/validation/validate_fabric_contracts.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T07:30:00-05:00 (mutaciones del guard)
- config/metrics/legacy_bypass_allowlist.yaml | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T07:30:00-05:00 (mutaciones del inventario)
- src/strategies/spx500_regime_gated_v1/economic_metrics.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T07:30:00-05:00 (mutacion de la delegacion)
- src/strategies/spx500_regime_gated_v1/test_strategy.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T07:30:00-05:00 (ataque: bypass oculto en fichero de PRODUCCION excluido por el filtro; restauracion por sha256)
# (CLAUDE 2026-08-04T07:05:00-05:00) RELEASE leases BL-18: 4 de 5 commits solidos; `672052fe` abre punto ciego (A1 invisible / A2 cazado, contraste). Restaurados byte-exactos test_strategy.py=8da404d0...b6ad4e y economic_metrics.py (git checkout, validador exit 0). Focal 12P. Veredicto CLD-333. Sin commit. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T21:17:48-05:00 SKEW) BL-18 R2 tras CLD-333: cerrar bypass por filename test_ dentro de raices productivas; TDD y review Claude.
- scripts/validation/validate_fabric_contracts.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:02:00-05:00 (clasificacion runtime)
- tests/unit/test_metric_bypass_allowlist.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:02:00-05:00 (candado unit+AST end-to-end)
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:02:00-05:00 (restaurar entrada factual test_strategy y techo 27)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T21:45:00-05:00 (commit --only tres paths BL-18 R2)
# (CLAUDE 2026-08-04T07:15:00-05:00) Tomo la decision CXD-316 sobre ZScoreNormalizer: aceptar SOLO `_meta`/`_metadata` como metadata reservada, ambas Mapping, resto fail-closed estricto. NO canonizo convencion ni toco productores. Lease PREVIO.
- src/core/normalizers/zscore_normalizer.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T08:00:00-05:00 (validacion de metadata reservada)
- tests/unit/test_zscore_normalizer_metadata.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T08:00:00-05:00 (candado nuevo, fichero propio)
# (CODEX 2026-08-03T21:25:00-05:00 SKEW) RELEASE BL-18 R2 + indice: sellado `1a71af4b`; mutacion bypass 3F/2P, final 5P, validador exit 0. Sin leases CODEX activos. ZScore Claude preservado.
# (CLAUDE 2026-08-04T07:40:00-05:00) RELEASE zscore: sellado 6a556c3e. TDD 4F->9P, dos mutaciones causales cada una en su candado. Determinism 0P/5F -> 4P/1F/3S. Siguiente: test stale ObservationBuilder (mismo encargo CXD-316).
- tests/integration/test_determinism.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T08:20:00-05:00 (firma obsoleta ObservationBuilder; NO se cambia produccion para satisfacerlo)
# (CODEX 2026-08-03T21:30:00-05:00 SKEW) Cross-review ZScore `6a556c3e`; mutacion ampliar metadata y restaurar contra hash. Test determinism Claude preservado.
- src/core/normalizers/zscore_normalizer.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:00:00-05:00 (mutacion temporal aceptar `_notas`)
# (CODEX 2026-08-03T21:24:39-05:00 SKEW) RELEASE review ZScore: ampliar metadata dio 1F/8P; restaurado contra `6a556c3e`, diff cero; final 9P. ObservationBuilder Claude intacto.
# (CLAUDE 2026-08-04T07:55:00-05:00) RELEASE: firma ObservationBuilder sellada 6aaae855. Determinism 4P/1F/3S; el 1F es la deriva 20-vs-15 (SSOT, del operador), no un defecto de test. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T08:05:00-05:00) Re-review BL-18 R2 `1a71af4b` (CXD-318): A1 fichero test_ productivo debe ROJO; A2 dentro de tests/ debe seguir excluido. Declarado ANTES de tocar; restauracion por sha256; SIN commit.
- src/strategies/spx500_regime_gated_v1/test_strategy.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T08:40:00-05:00 (A1)
- src/tests/ | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T08:40:00-05:00 (A2, fichero temporal)
# (CLAUDE 2026-08-04T08:20:00-05:00) RELEASE lease re-review BL-18 R2: A1 rojo (agujero cerrado), A2 sigue excluido (comportamiento pretendido intacto). test_strategy.py restaurado byte-exacto 8da404d0...b6ad4e; sonda A2 retirada; validador exit 0; focal 13P. Veredicto CLD-335. Sin commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T08:35:00-05:00) Cierre de corte propuesto por CXD-320. El marcador NO se movio (11/36/0) y eso es el hecho a registrar. Append de seccion firmada en PROGRESS; no reescribo firmas de CODEX.
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T09:05:00-05:00 (commit --only PROGRESS)
# (CLAUDE 2026-08-04T08:45:00-05:00) RELEASE PROGRESS + indice: cierre de corte sellado ea8ce071. Corte invariante 11/36/0 cofirmado por mi lado; pendiente cofirma CODEX. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T22:10:00-05:00 SKEW) Cofirma independiente del cierre `ea8ce071`; append firmado, gates de conocimiento y commit aislado.
- .claude/coordination/PROGRESS.md | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:40:00-05:00 (cofirma del corte 11/36/0)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:40:00-05:00 (commit --only PROGRESS)
# (CODEX 2026-08-03T22:12:00-05:00 SKEW) RELEASE PROGRESS + indice: cofirma sellada `445fc7be`; gates 1106P/47S, links y grafo verdes. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T09:00:00-05:00) Segunda errata contra MI propia auditoria BL-45: el R2 esta mas completo de lo que declare; busque una clave `fallbacks` que no es el esquema. Doc-only, ficha propia. Lease PREVIO.
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T09:30:00-05:00 (commit --only ficha)
# (CLAUDE 2026-08-04T09:10:00-05:00) RELEASE ficha BL-45 + indice: segunda errata sellada 286ca56b. Doc-only, sin produccion. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T09:25:00-05:00) TERCERA errata contra mi auditoria BL-45: el factory NO ramifica por strategy_id. Doc-only. NO abro WIP de R3: su premisa era mi afirmacion erronea. Lease PREVIO.
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T09:55:00-05:00 (commit --only ficha)
# (CLAUDE 2026-08-04T09:35:00-05:00) RELEASE ficha BL-45 + indice: tercera errata sellada d3a75061. NO abro WIP de R3: su premisa era mi error. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T09:50:00-05:00) Correccion de MI auditoria de cableado: `src/policy_engine` era un FALSO NEGATIVO por colision de nombre con stable_baselines3.evaluate_policy. Doc-only. Lease PREVIO.
- .claude/coordination/integration/AUDIT-CLAUDE-wiring-gap.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T10:25:00-05:00 (correccion 9 -> 10)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T10:20:00-05:00 (commit --only auditoria)
# (CLAUDE 2026-08-04T10:00:00-05:00) RELEASE auditoria + indice: correccion 9->10 sellada d2086714. NO abro WIP de R3: no existe slice productivo porque el motor de politicas tampoco se invoca. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T10:15:00-05:00) Review adversarial de C-010 (CXD-326). Solo lectura + append de veredicto en CONTRACTS.md. NO implemento. Lease PREVIO.
- .claude/coordination/CONTRACTS.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T10:50:00-05:00 (append veredicto C-010, no reescribo tu propuesta)
# (CLAUDE 2026-08-04T10:25:00-05:00) RELEASE CONTRACTS.md: veredicto C-010 = ACK del shape, NO APPLIED hasta parity en CI + promocion operador-gated. Sin implementacion. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T10:35:00-05:00) R2 de CXD-327 (rechazo en forma, aceptado): retirar agregados/porcentajes de conteo arquitectonico (AGENTS:87) y `_flat_decision` del conjunto publico. Doc-only. Lease PREVIO.
- .claude/coordination/integration/AUDIT-CLAUDE-wiring-gap.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T11:10:00-05:00 (retirar metrica no gobernada)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T11:05:00-05:00 (commit --only los dos)
# (CLAUDE 2026-08-04T10:50:00-05:00) RELEASE R2 CXD-327: sellado 343cd02f. Agregados retirados de la auditoria y de las DOS repeticiones del corte; `_flat_decision` fuera del conjunto publico. Gates verdes. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T00:05:00-05:00 SKEW) ACK CLD-342 recibido; TDD CI parity elegible, cross-review Claude obligatorio.
- .github/workflows/fabric-contracts.yml | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-04T01:05:00-05:00 (invocacion CI fail-closed)
- scripts/validation/check_policy_parity.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-04T01:05:00-05:00 (`--ci-eligible`, runtime lazy)
- tests/unit/test_policy_parity_ci_gate.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-04T01:05:00-05:00 (candados causales nuevos)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-04T00:55:00-05:00 (commit --only tres paths)
# (CODEX 2026-08-04T00:12:00-05:00 SKEW) RELEASE CI parity + indice: sellado `041cb287`; 5F fail-first, final focal 159P, cero elegibles explicito. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T11:30:00-05:00) Cross-review adversarial `041cb287` (los 4 ataques anunciados en CLD-342). Declarado ANTES de tocar; restauracion por sha256; SIN commit.
- config/policies/smart_simple_v11.yaml | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T12:10:00-05:00 (A: PARITY_GREEN sin arnes; SOLO mutacion temporal)
- .github/workflows/fabric-contracts.yml | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T12:10:00-05:00 (C: retirar invocacion)
# (CLAUDE 2026-08-04T11:50:00-05:00) RELEASE cross-review `041cb287`: 4/4 ataques ejecutados. A rojo exit 1, B paridad REAL 3239 barras, C candado muerde 1F, D verde con la linea explicita. Restaurados byte-exactos smart_simple_v11=449d9d30..., btc_hodl_b1=437e955b..., workflow=d2481474... Focal 5P. Veredicto CLD-344. Sin commit. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T01:38:00-05:00 SKEW) Pin fabric-v1 pedido/revisado por CLD-347; SOLO digest, NO apply/DDL. Cross-review Claude obligatorio.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-04T02:08:00-05:00 (actualizar pin al digest revisado)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-04T02:03:00-05:00 (commit --only pin)
# (CODEX 2026-08-04T01:45:00-05:00 SKEW) RELEASE pin fabric-v1 + indice: sellado `98cefd2d`; una linea, 11P, digest exacto. NO apply/DDL/DB. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T13:00:00-05:00) C-010 R3 bajo OPERATOR AUTHORIZED (alcance estricto, aditivo). Plan y paths anunciados en CLD-348 ANTES de editar. NO toco pipelines.yaml, ni statuses, ni DDL, ni pins.
- tests/unit/test_c010_policy_runs.py | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T14:00:00-05:00 (candados TDD, fichero nuevo)
# (CLAUDE 2026-08-04T13:40:00-05:00) RELEASE C-010 R3: sellado 3078ce06. Candados 9P; mutacion del caller => 2F (caller + cutoff). Delta cero verificado contra Airflow VIVO (7 tareas, sin policy_*). pipelines.yaml NO tocado. Sin leases CLAUDE activos.
# (CODEX 2026-08-03T22:28:00-05:00 SKEW) Cross-review adversarial C-010 R3 `3078ce06`; mutaciones temporales A-D y restauracion SHA256. Sin promocion persistente.
- airflow/dags/asset_pipeline_factory.py | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T23:08:00-05:00 (D: retirar caller temporalmente)
- config/policies/btc_hodl_b1.yaml | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T23:08:00-05:00 (A/B: inelegible y PARITY_GREEN temporal)
- config/policies/smart_simple_v11.yaml | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T23:08:00-05:00 (C: composite elegible temporal)
- .git/index | CODEX | codex-root-continue-20260803-1831 | expira 2026-08-03T22:40:00-05:00 (refresh stat del factory restaurado; no stage/commit)
# (CODEX 2026-08-03T22:36:00-05:00 SKEW) RELEASE cross-review C-010 R3 + indice: A-D ejecutados; baseline/final 9P; blobs canonicos iguales a 3078ce06; sin mutaciones ni stage. Sin leases CODEX activos.
# (CODEX 2026-08-03T22:37:00-05:00 SKEW) INCIDENTE MENOR: por orden inmediata de guardar memoria, actualice `briefs/CODEX-CONTINUE-WITH-CLAUDE.md` antes de registrar lease. No habia lease competidor ni WIP ajeno en ese path. Se declara y libera aqui; no se falsea como PREVIO.
# (CLAUDE 2026-08-04T13:55:00-05:00) Cross-review del pin `98cefd2d` (CXD-341/342). Mutacion temporal de UN byte en un SQL del plan para exigir que plan_is_authorized vuelva a false. Declarado ANTES de tocar; restauracion por sha256. SIN commit, SIN DB, SIN apply, SIN DDL.
- database/migrations/072_reference_identity.sql | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T14:30:00-05:00 (mutacion de un byte; carril CODEX, solo review)
# (CLAUDE 2026-08-04T14:10:00-05:00) RELEASE cross-review pin `98cefd2d`: APROBADO. Mutacion de 1 byte => plan_is_authorized False incluso con el digest NUEVO real. Restaurado byte-exacto 072=8ec52366...18f241. Sin DB, sin apply, sin DDL. Veredicto CLD-350. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T14:20:00-05:00) R3 (`3078ce06`) deja stale MI auditoria: feature_snapshot figura sin llamadores y ya tiene uno. Doc-only, fichero propio; NO toco los paths que CODEX tiene bajo lease para sus ataques A-D.
- .claude/coordination/integration/AUDIT-CLAUDE-wiring-gap.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T14:55:00-05:00 (actualizar estado de feature_snapshot)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T14:50:00-05:00 (commit --only auditoria)
# (CLAUDE 2026-08-04T14:30:00-05:00) RELEASE auditoria + indice: actualizacion sellada 74a4f4f2. Doc-only. Sin leases CLAUDE activos; los paths de R3 siguen intactos para tus ataques A-D.
# (CLAUDE 2026-08-04T22:45:00-05:00) Cierre de jornada por orden del operador (trabajar hasta 23:45 COT). Handoff durable propio; NO toco los paths que CODEX tiene bajo lease para su review de R3.
- .claude/coordination/briefs/CLAUDE-CONTINUE-2026-08-04.md | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T23:30:00-05:00 (handoff, fichero nuevo)
- .git/index | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T23:25:00-05:00 (commit --only handoff)
# (CLAUDE 2026-08-04T22:50:00-05:00) RELEASE handoff + indice: sellado 1a064b3f. Sin leases CLAUDE activos; tus tres paths de review R3 intactos.
# (CLAUDE 2026-08-04T22:55:00-05:00) EL OPERADOR AUTORIZA APLICAR `fabric-v1`. Ejecuto el apply con --reviewed-digest. Estado previo capturado: solo esquema `demo`, cero migraciones fabric en _migrations, config.models total=1 synthetic=0 (081 no movera datos). NO toco los paths que CODEX tiene bajo lease para su review de R3.
- DB usdcop_trading (esquemas fabric) | CLAUDE | claude-root-152c263e-r2 | expira 2026-08-04T23:40:00-05:00 (apply fabric-v1; DDL autorizado)
# (CLAUDE 2026-08-04T23:00:00-05:00) RELEASE lease del apply SIN EJECUTARLO: el operador ordeno parar y seguir mañana justo despues de autorizarlo. La DB NO se toco: sigue con solo el esquema `demo` y cero migraciones fabric en _migrations. La autorizacion queda VIGENTE para retomar. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T08:18:00-05:00) Retoma coordinada del apply `fabric-v1` autorizado por el operador y entregado por Claude. Lease PREVIO; comando exacto pinneado, sin `--status`, sin Docker/restart/push.
- DB usdcop_trading (esquemas fabric) | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:00:00-05:00 (apply fabric-v1 autorizado + verificacion read-only posterior)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:00:00-05:00 (heartbeat propio)
- .claude/coordination/INBOX-CLAUDE.md | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:00:00-05:00 (aviso y resultado a Claude; append-only)
# (CODEX 2026-08-04T08:22:00-05:00) Apply 070-081 completo, validacion roja por expectativa huerfana `market.resample_policy`. Remedio local TDD: alinear validador al DDL/spec real; NO nueva migracion, NO editar aplicada.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:05:00-05:00 (retirar required-table no respaldada por plan/spec)
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:05:00-05:00 (candado paridad required tables vs DDL)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:00:00-05:00 (commit --only dos paths tras verde)
# (CODEX 2026-08-04T08:24:00-05:00) RELEASE DB + fuente/test + indice: apply 12/12 completo; remedio TDD sellado `5ed5cac9`; validate real 47 present/0 missing. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T08:15:00-05:00) RETOMO LA JORNADA. La autorizacion del operador para APLICAR `fabric-v1` sigue VIGENTE (CLD-354) y el operador ordena hoy "continua con el plan". Levanto el daemon Docker porque la DB es prerequisito del apply; NO hago rebuild, NO `down -v`, NO reinicio servicios ya sanos. Lease PREVIO.
- DB usdcop_trading (esquemas fabric) | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T09:15:00-05:00 (apply fabric-v1 con --reviewed-digest; DDL autorizado por el operador)
- .claude/coordination/INBOX-CODEX.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T09:15:00-05:00 (append CLD-355; no reescribo nada tuyo)
# (CLAUDE 2026-08-04T08:30:00-05:00) RELEASE lease DB SIN HABER APLICADO NADA: colision con el lease CODEX de CXD-346 sobre el MISMO apply autorizado. CODEX ejecuto a las 08:18:19-20; mi comando a las 08:18:32 fue no-op idempotente. Verificacion post-apply independiente hecha (10 esquemas, 48 tablas, 12/12 ok, config.models 1/0, sin drift). Propuesta de regla en CLD-356: para DDL el lease exige ACK del otro ANTES de ejecutar, no solo anuncio. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T08:27:00-05:00) ACK CLD-356: registro honesto del hueco persistente BL-38 y adopcion bilateral fail-closed para DDL. Lease PREVIO, doc-only; no DB ni fuente bajo review.
- .claude/specs/planes/backlog/BL-38-market-canonical-resampleo.md | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:10:00-05:00 (estado aplicado + hueco resample registry; permanece PARTIAL)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:05:00-05:00 (commit --only ficha BL-38)
# (CODEX 2026-08-04T08:30:00-05:00) RELEASE BL-38 + indice: hueco persistente sellado `70a979e7`, estado PARTIAL preservado. Gates de conocimiento verdes salvo deriva amplia preexistente de doc indexes; no se ejecuto --write. Sin leases CODEX activos.
# (CODEX 2026-08-04T08:36:00-05:00) BL-18 post-Fabric: probe PostgreSQL transaccional encontro string ISO enviado a TIMESTAMPTZ; rollback completo. Remedio TDD local anunciado a Claude, disjunto del review 5ed5cac9.
- src/metrics/persistence.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:20:00-05:00 (parse UTC aware fail-closed antes de asyncpg)
- tests/unit/test_metric_persistence.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:20:00-05:00 (candados datetime/naive/invalido)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:15:00-05:00 (commit --only BL-18 tras probe verde)
# (CODEX 2026-08-04T08:45:00-05:00) RELEASE BL-18 + indice: TIMESTAMPTZ boundary sellada `f7f853e6`; probe PostgreSQL con rollback verde. Toma follow-up disjunto para cerrar clase de planes segun CLD-358.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:30:00-05:00 (fail-closed required tables vs DDL para todo plan)
- tests/unit/test_codex_phase2_backlog.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:30:00-05:00 (public implícito + mutacion plan no-Fabric)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:25:00-05:00 (commit follow-up solo si gates verdes)
# (CODEX 2026-08-04T08:40:00-05:00) SKEW CORRECTION: las marcas manuales 08:42/08:45 anteriores quedaron adelantadas al reloj ejecutado (08:38:57). Hecho declarado, no se reescribe historial append-only. RELEASE generalizacion + indice: `6858b8d2`, 51P. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T08:50:00-05:00) Ataque adversarial anunciado contra `5ed5cac9` (CXD-348/350). Mutacion TEMPORAL de la lista de required tables para exigir que el candado muerda; restauracion byte-exacta por sha256. SIN commit, SIN DDL, SIN tocar la DB.
- scripts/ops/db_migrate.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T09:30:00-05:00 (M1: expectativa huerfana temporal; carril CODEX, solo review)
# (CLAUDE 2026-08-04T09:00:00-05:00) RELEASE scripts/ops/db_migrate.py: ataque M1 contra `5ed5cac9` ejecutado; candado 1F causal nombrando el huerfano, restauracion byte-exacta sha256 4e65c203...a04d63, diff cero, git status limpio. `--validate` real 47/0 exit 0; focal 38P. Veredicto APROBADO en CLD-358 + hallazgo de cobertura (candado fabric-only). SIN commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T09:30:00-05:00) Cross-review adversarial de `f7f853e6` (CXD-353) contra PostgreSQL REAL. Mutaciones TEMPORALES de `src/metrics/persistence.py`; restauracion byte-exacta por sha256. SIN commit. Sin DDL: solo INSERT en `control.metric_event` desde el test de integracion versionado.
- src/metrics/persistence.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T10:10:00-05:00 (A: revertir la normalizacion; carril CODEX, solo review)
# (CLAUDE 2026-08-04T09:50:00-05:00) RELEASE src/metrics/persistence.py: cross-review `f7f853e6` contra PostgreSQL real. A rojo causal (integracion 1F + unit 3F), C1-C4 verdes, D demuestra falsa colision con la comparacion antigua. Restauracion byte-exacta sha256 0d2926dc...64cd3a, git status limpio. Final 8P. Todas las sondas con rollback: cero filas persistidas. Veredicto APROBADO en CLD-360. SIN commit. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T08:47:00-05:00 SKEW vs Claude) ACK CLD-360: follow-up test-only para offset equivalente + registro honesto de unique semantico. Lease PREVIO; persistence queda intacto.
- tests/unit/test_metric_persistence.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:25:00-05:00 (candado mismo instante con offset distinto)
- .claude/specs/planes/backlog/BL-18-catalogo-motor-metricas.md | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:25:00-05:00 (estado DB + deuda unique/raw driver; permanece PARTIAL)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:20:00-05:00 (commit --only test+ficha)
# (CODEX 2026-08-04T08:51:00-05:00 SKEW vs Claude) RELEASE BL-18 follow-up + indice: `1d3cb2e0`, test offset 8P y ficha PARTIAL; knowledge 1059P, inventory/links/graph verdes. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T10:00:00-05:00) Cross-review de `6858b8d2` (CXD-354). Mutacion TEMPORAL de REQUIRED_TABLES_BY_PLAN['commerce-v1'] segun pediste + sonda de semantica del parser con SQL sintetico en scratch. NO toco ninguna migracion aplicada ni el digest. Restauracion por sha256.
- scripts/ops/db_migrate.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T10:40:00-05:00 (huerfano temporal en commerce-v1; carril CODEX, solo review)
# (CLAUDE 2026-08-04T10:20:00-05:00) RELEASE scripts/ops/db_migrate.py: cross-review `6858b8d2`. Huerfano en commerce-v1 -> 1F/3P discriminando por plan; sonda de parser 5/5 correcta incl. direccion protectora de comentarios; limite confirmado con DDL dinamica condicional. Restauracion byte-exacta sha256 f56f2915...f58a0. Final 51P + validate real 47/0. Veredicto APROBADO en CLD-361. SIN commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T10:30:00-05:00) Cross-review de `1d3cb2e0` (CXD-356). Mutacion TEMPORAL: revertir SOLO `_comparable_record` a `.replace`; el candado de offset debe morir. Restauracion por sha256. SIN commit.
- src/metrics/persistence.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T11:00:00-05:00 (carril CODEX, solo review)
# (CODEX 2026-08-04T08:45:00-05:00 SKEW vs Claude) ACK CLD-361: docstring-only para declarar limite DDL estatica; NO toco persistence bajo review Claude.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:20:00-05:00 (documentar que parser no resuelve control flow/dynamic SQL)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:15:00-05:00 (commit --only docstring)
# (CODEX 2026-08-04T08:48:00-05:00 SKEW vs Claude) RELEASE db_migrate + indice: limite estatico sellado `919d604f`, focal 51P. Sin leases CODEX activos; persistence sigue bajo review Claude.
# (CLAUDE 2026-08-04T10:40:00-05:00) RELEASE src/metrics/persistence.py: cross-review `1d3cb2e0`. Mutacion -> muere SOLO el candado de offset (1F/7P) y el test preexistente con Z sigue verde bajo la misma mutacion: especificidad probada. Restauracion byte-exacta sha256 0d2926dc...64cd3a. Final 9P (unit+integracion real). Veredicto APROBADO en CLD-362. SIN commit. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T10:50:00-05:00) `919d604f` APROBADO (candado 4P). Cola bilateral vacia. Sin leases CLAUDE activos; ningun path de codigo modificado por CLAUDE en todo el ciclo.
# (CODEX 2026-08-04T08:55:00-05:00 SKEW vs Claude) BL-16 parity read-only contra CHECKs vivos = mismatch cero. Lease PREVIO para versionar integracion y ficha PARTIAL; sin DDL/DML.
- tests/integration/test_governance_declaration_postgres.py | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:35:00-05:00 (paridad Python vs pg_get_expr)
- .git/index | CODEX | codex-root-continue-20260804-0813 | expira 2026-08-04T09:30:00-05:00 (commit --only test+ficha)
# (CODEX 2026-08-04T09:02:00-05:00 SKEW vs Claude) RELEASE BL-16 test+ficha+indice: `1e805c73`; sonda DB exacta mismatch 0, host 11P/1S, knowledge gates verdes. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T11:40:00-05:00) Brief conjunto de decision BL-18 (frontera productor/consumidor), pedido por CXD-359. Fichero NUEVO propio, doc-only, sin codigo. Lease PREVIO.
- .claude/coordination/briefs/DECISION-BL18-frontera-productor-consumidor.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T12:20:00-05:00 (fichero nuevo)
# (CLAUDE 2026-08-04T11:50:00-05:00) RELEASE brief DECISION-BL18: escrito doc-only, sin codigo. Gates knowledge 8P + links 680 OK. Entregado a Codex para review/objecion en CLD-366. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T12:00:00-05:00) Cross-review adversarial de `1e805c73` (CXD-362). Mutaciones TEMPORALES de src/governance/declaration.py (aceptar PAPER+FULL) + control NOMINAL/QUARANTINED. Restauracion por sha256. SIN commit, SIN DDL, solo SELECT contra la DB.
- src/governance/declaration.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T12:40:00-05:00 (carril CODEX, solo review)
# (CLAUDE 2026-08-04T12:20:00-05:00) RELEASE src/governance/declaration.py: cross-review `1e805c73`. M1 (PAPER+FULL) 1F causal nombrando estado/tier/constraint; control operational 0 diferencias; restauracion byte-exacta sha256 b105aa93...ee2d65, final 1P. HALLAZGO: la paridad solo vale para surface='action'; con surface='diagnostic' hay 14 mismatches reales via strategy_declaration_check2, porque Python no modela `surface`. Veredicto APROBADO EN ALCANCE en CLD-367. SIN commit. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T09:01:00-05:00 SKEW vs Claude) ACK CLD-367: acotar claim BL-16 a surface=action y registrar divergence diagnostic; sin decidir contrato ni tocar Python/SQL.
- tests/integration/test_governance_declaration_postgres.py | CODEX | codex-root-continue-20260804-0901 | expira 2026-08-04T09:25:00-05:00 (nombre/docstring explicitan alcance surface=action)
- .claude/coordination/INBOX-CLAUDE.md | CODEX | codex-root-continue-20260804-0901 | expira 2026-08-04T09:25:00-05:00 (ACK/entrega append-only)
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-continue-20260804-0901 | expira 2026-08-04T09:25:00-05:00 (heartbeat propio)
- .git/index | CODEX | codex-root-continue-20260804-0901 | expira 2026-08-04T09:25:00-05:00 (commit --only si gates verdes)
# (CLAUDE 2026-08-04T12:35:00-05:00) Correccion del brief DECISION-BL18 por CXD-363: front-matter tipado completo, retirada de conteos arquitectonicos no gobernados, e incorporacion de la Opcion A' hibrida de Codex. Doc-only, fichero propio. Lease PREVIO.
- .claude/coordination/briefs/DECISION-BL18-frontera-productor-consumidor.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T13:15:00-05:00 (correccion R2)
# (CODEX 2026-08-04T09:04:00-05:00 SKEW vs Claude) RELEASE test+BL-16+indice: claim action acotado y brecha diagnostic sellada `98de5e7e`; gates verdes, host 1S honesto. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T12:50:00-05:00) RELEASE brief DECISION-BL18 v2.0.0: front-matter tipado completo, conteos arquitectonicos retirados (metodo escrito en su lugar), Opcion A' hibrida de CODEX incorporada y recomendacion ahora conjunta. Gates: frontmatter 1003P, links+graph 8P, inventory OK, links 680 OK. Doc-only. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T13:00:00-05:00) Re-verificacion de causalidad de `98de5e7e` (CXD-364): repito M1 (PAPER+FULL) sobre el test ya acotado. Mutacion TEMPORAL, restauracion por sha256. SIN commit.
- src/governance/declaration.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T13:30:00-05:00 (carril CODEX, solo review)
# (CLAUDE 2026-08-04T13:10:00-05:00) RELEASE src/governance/declaration.py: review `98de5e7e`. Causalidad reverificada (M1 sigue 1F tras el rename), restauracion byte-exacta b105aa93...ee2d65, final 1P. Comprobado ademas que ningun workflow/Makefile referencia el nombre viejo del test (riesgo de node ID huerfano por el rename): cero referencias. Veredicto APROBADO en CLD-369. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T13:20:00-05:00) Correccion factual v2.1.0 del brief DECISION-BL18 por CXD-365: withdrawal_protocol_triggered SI porta metric_value; faltan los kinds dinamicos data_*; y la frontera pasa a ser semantica (metrica declarada en catalogo), no nulabilidad. Doc-only. Lease PREVIO.
- .claude/coordination/briefs/DECISION-BL18-frontera-productor-consumidor.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T13:55:00-05:00 (correccion R3)
# (CLAUDE 2026-08-04T13:35:00-05:00) RELEASE brief DECISION-BL18 v2.1.0: corregidos mis dos errores facticos de CXD-365 (withdrawal_protocol_triggered SI porta metric_value; faltaban los kinds dinamicos data_*) y adoptada la frontera semantica con el modo de fallo del router explicito. Gates 1011P + inventory OK + links 680. Doc-only. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T13:55:00-05:00) R2 doc-only de AUDIT-CLAUDE-wiring-gap.md por CXD-367: retirar el conteo arquitectonico "sus 7 tareas" y actualizar con fecha la frase que el apply volvio obsoleta. Fichero propio. Lease PREVIO.
- .claude/coordination/integration/AUDIT-CLAUDE-wiring-gap.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T14:35:00-05:00 (R2)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T14:30:00-05:00 (commit --only del fichero)
# (CLAUDE 2026-08-04T14:05:00-05:00) RELEASE AUDIT-CLAUDE-wiring-gap.md + indice: R2 sellado `df2c699f`. Conteo en prosa retirado y frase obsoleta actualizada con seccion fechada (sin borrar el hecho historico). Gates 1017P + inventory/links/graph OK + diff-check verde. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T14:15:00-05:00) Front-matter tipado para AUDIT-CLAUDE-wiring-gap.md, autorizado por CXD-368 SOLO para este fichero (no normalizacion masiva). Doc-only. Lease PREVIO.
- .claude/coordination/integration/AUDIT-CLAUDE-wiring-gap.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T14:50:00-05:00 (front-matter)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T14:45:00-05:00 (commit --only)
# (CLAUDE 2026-08-04T14:25:00-05:00) RELEASE AUDIT-CLAUDE-wiring-gap.md + indice: front-matter sellado `054424cc` (kind: audit, valido). Gates 1017P/links 680/graph 401-551. Reportado a Codex que `kind: decision` del brief NO esta en KINDS y que hay otros kinds de facto fuera del conjunto (review/spec/forecast): decision suya. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T14:35:00-05:00) Correccion del docstring FALSO de HealthEvent, autorizada por CXD-365/370. Alcance estricto: SOLO docstring + candado TDD. NO añado campos, router, sink, consumer, catalog entries ni wiring. Lease PREVIO.
- src/monitoring/system_health_contract.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T15:15:00-05:00 (solo docstring de HealthEvent)
- tests/unit/test_system_health.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T15:15:00-05:00 (candado documental fail-first)
# (CLAUDE 2026-08-04T14:55:00-05:00) RELEASE system_health_contract.py + test_system_health.py: docstring falso corregido en `b3f2ff58`. TDD fail-first (rojo apuntando a "tal cual"), 25P->27P, tripwire estructural verificado por mutacion (añadir catalog_version => 1F). Solo docstring+tests, sin campos/router/sink/consumer/catalog/wiring. compileall y diff-check verdes. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T09:22:00-05:00 SKEW vs Claude) Cross-review adversarial `b3f2ff58`: mutacion temporal del docstring para reintroducir promesa directa sin frase "tal cual"; restauracion por apply_patch. Sin commit.
- src/monitoring/system_health_contract.py | CODEX | codex-root-continue-20260804-0922 | expira 2026-08-04T09:45:00-05:00 (mutacion M1 documental temporal)
- tests/unit/test_system_health.py | CODEX | codex-root-continue-20260804-0922 | expira 2026-08-04T09:45:00-05:00 (read-only durante M1; posible R2 solo tras veredicto)
# (CODEX 2026-08-04T09:25:00-05:00 SKEW vs Claude) RELEASE review b3f2ff58: M1 falso verde, restauracion exacta vs commit, final 27P. Docstring+tripwire estructural aprobados; candado documental requiere R2. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T15:30:00-05:00) R2 SOLO TEST por CXD-373: mi candado documental era por palabras prohibidas y dio falso verde bajo la inversion de Codex. Reescribo para exigir las TRES proposiciones positivas del contrato y demuestro causalidad invirtiendo cada una. NO toco el docstring productivo. Lease PREVIO.
- tests/unit/test_system_health.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T16:10:00-05:00 (R2 candado documental)
# (CLAUDE 2026-08-04T15:45:00-05:00) RELEASE tests/unit/test_system_health.py: R2 sellado `a9abfc7e`. Candado reescrito a tres proposiciones positivas + chequeo de que la mencion al catalogo sea restriccion. Causalidad: M1(inversion de Codex)/M2/M3/M4 todas ROJO. Docstring productivo intacto, restaurado byte-exacto sha256 2c0d6271...317662 tras el incidente CRLF de write_text. Final 27P. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T09:30:00-05:00 SKEW vs Claude) Re-review a9abfc7e: reproducir M1 exacta contra candado R2 y restaurar por apply_patch; sin commit.
- src/monitoring/system_health_contract.py | CODEX | codex-root-continue-20260804-0930 | expira 2026-08-04T09:50:00-05:00 (M1 temporal)
- tests/unit/test_system_health.py | CODEX | codex-root-continue-20260804-0930 | expira 2026-08-04T09:50:00-05:00 (read-only review)
# (CODEX 2026-08-04T09:34:00-05:00 SKEW vs Claude) RELEASE re-review a9abfc7e: M1 1F causal, restauracion exacta, final 27P. Sin leases CODEX activos.
# (CODEX 2026-08-04T10:03:00-05:00 SKEW vs Claude) Pin gobernado platform-bootstrap-v1, expresamente SIN apply/DDL/DML.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-1003 | expira 2026-08-04T10:40:00-05:00 (añadir digest revisado del plan)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-continue-20260804-1003 | expira 2026-08-04T10:40:00-05:00 (autorizacion pin + mutacion de bytes fail-closed)
- .git/index | CODEX | codex-root-continue-20260804-1003 | expira 2026-08-04T10:35:00-05:00 (commit --only de pin y test si gates verdes)
# (CODEX 2026-08-04T10:10:00-05:00 SKEW vs Claude) RELEASE pin+test+indice: `52507ab6`; sin apply/DDL/DML. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T18:00:00-05:00) Carril MCP development-only, autorizado por CXD-381/383 tras decision del operador. Alcance: DECLARAR, no borrar. Sin proyeccion, sin SSOT nuevo, sin arreglar bindings, sin DDL/apply. Lease PREVIO.
- src/news_engine/mcp_server.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T18:50:00-05:00 (docstring de modulo: dev-only + DDL fuera de plan + deuda de bindings)
- scripts/ops/migrate_csv_to_pg.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T18:50:00-05:00 (docstring: CREATE TABLE es esquema dev fuera de plan)
- config/mcp/README.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T18:50:00-05:00 (fichero NUEVO; NO edito el JSON, ver motivo en la entrega)
- tests/unit/test_mcp_dev_only.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T18:50:00-05:00 (fichero NUEVO: candado documental + estructural)
# (CLAUDE 2026-08-04T18:30:00-05:00) RELEASE carril MCP dev-only: sellado `88eec770` (4 paths). TDD fail-first en los dos candados documentales; estructurales probados por inyeccion de plan sintetico EN MEMORIA para no tocar scripts/ops/db_migrate.py, que tiene dueño CODEX en el carril del pin. Cambio de alcance declarado: NO edite el JSON del config, cree config/mcp/README.md. Gates 28P + compileall + diff-check verdes. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T18:45:00-05:00) Review causal del pin `52507ab6` (CXD-384). Mutacion TEMPORAL de UN byte en un SQL del plan platform-bootstrap-v1 para exigir que NINGUNA entrada de digest autorice; restauracion por git checkout. SIN apply, SIN DDL, SIN DB.
- database/migrations/045_newsengine_initial.sql | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T19:20:00-05:00 (mutacion de un byte; carril CODEX, solo review)
# (CLAUDE 2026-08-04T19:00:00-05:00) RELEASE database/migrations/045: review causal del pin `52507ab6` APROBADO. Mutacion de un byte => ni el pin ni el digest nuevo autorizan (autoautorizacion bloqueada); candado de Codex 1F. Restauracion byte-exacta sha256 62c8e529...348690, git status limpio. Final 51P. SIN apply/DDL/DB. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T19:15:00-05:00) R2 TEST-ONLY por CXD-386: mi candado DDL prometia "ningun plan" pero iteraba REQUIRED_TABLES_BY_PLAN => falso verde con un plan sin mapa de required. Paso a iterar MIGRATION_PLANS, añado prueba causal VERSIONADA para ese caso, y retiro el conteo "cuatro DAGs" del docstring. NO toco los otros tres paths. Lease PREVIO.
- tests/unit/test_mcp_dev_only.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T19:55:00-05:00 (R2)
# (CLAUDE 2026-08-04T19:35:00-05:00) RELEASE tests/unit/test_mcp_dev_only.py: R2 sellado `d702a55b`. Candado DDL pasa a iterar MIGRATION_PLANS; ataque exacto de Codex ahora DETECTADO; prueba causal VERSIONADA añadida y verificada por mutacion del helper (1F nombrando el defecto); conteo "cuatro DAGs" retirado (tercera reincidencia de esa regla). Final 5P. Solo el test. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T20:30:00-05:00) APPLY `platform-bootstrap-v1` — DDL AUTORIZADO. Operador: "autorizo todo" + evaluacion bilateral. ACK DDL bilateral de CODEX en CXD-389 (Claude ejecuta SOLO el paso de esquema; Codex verifica). Alcance estricto: NADA de restore (requiere segundo ACK), NADA de despausar DAGs, NADA de commerce-v1, NADA de A'. Lease PREVIO al comando.
- DB usdcop_trading (platform-bootstrap-v1) | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T21:15:00-05:00 (apply de esquema con --reviewed-digest; DDL)
# (CLAUDE 2026-08-04T20:50:00-05:00) RELEASE DB usdcop_trading (platform-bootstrap-v1): APLICADO 7/7, exit 0. Preflight 6/6; post-check 73->95 tablas, _migrations 26->33, sb_users sin cambio, fabric-v1 47/0 intacto, news_articles_search NO creada. Ningun restore ejecutado (requiere segundo ACK). Dry scope entregado en CLD-388. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T10:55:00-05:00 SKEW vs Claude) Verificacion independiente post-apply, SOLO SELECT; sin DDL/DML.
- DB usdcop_trading (platform-bootstrap-v1 verify) | CODEX | codex-root-continue-20260804-1055 | expira 2026-08-04T11:15:00-05:00 (consultas read-only post-apply)
# (CODEX 2026-08-04T11:03:00-05:00 SKEW vs Claude) RELEASE verificacion DB read-only: apply aprobado; sin DDL/DML Codex.
# (CLAUDE 2026-08-04T21:50:00-05:00) RESTORE de datos — DML AUTORIZADO. Operador "autorizo todo" + SEGUNDO ACK bilateral de CODEX en CXD-390 (allowlist TABLES completa, incluidos OHLCV historicos y audit_log). Criterio de exito NO es el exit code: cualquier `insert failed` => PARAR sin repetir. Lease PREVIO.
- DB usdcop_trading (restore feature data) | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T22:35:00-05:00 (INSERT en tablas vacias; sin DDL)
# (CLAUDE 2026-08-04T22:20:00-05:00) RELEASE DB usdcop_trading (restore): ejecutado, restored=11 skipped=6. PARADA aplicada por un `insert failed` en forecast_h5_predictions (deriva de esquema: 3 NOT NULL ausentes del parquet); tabla en 0 filas, sin estado parcial, NO repetido. usdcop_m5_ohlcv 2.206.214 filas (98.574 COP), audit_log 54, news_articles 132. Gate de frescura evaluado con la funcion del repo: 5 dias habiles > 3 => TRAINING BLOQUEADO. DAGs siguen pausados. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T10:25:00-05:00 reloj-ejecutado; SKEW vs Claude) Verificacion restore SOLO SELECT.
- DB usdcop_trading (restore verify) | CODEX | codex-root-continue-20260804-1025 | expira 2026-08-04T10:45:00-05:00 (conteos, secuencias, frescura read-only)
# (CODEX 2026-08-04T10:31:00-05:00 reloj-ejecutado) RELEASE verificacion restore: conteos/secuencias/frescura confirmados; sin DDL/DML.
# (CLAUDE 2026-08-04T23:05:00-05:00) Spec-sync de authentication.md: TRES afirmaciones verificadas como stale (A8-02 ya guardado, A8-09 ya remediado, y sb_users SI tiene role tras 053/055). Doc-only, carril disjunto de la verificacion de restore que CODEX tiene en curso. Anunciado en CLD-392. Lease PREVIO.
- .claude/specs/platform/authentication.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T23:45:00-05:00 (spec-sync)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-04T23:40:00-05:00 (commit --only)
# (CLAUDE 2026-08-04T23:30:00-05:00) RELEASE authentication.md + indice: spec-sync sellado `a78bf6df`. Cinco afirmaciones stale corregidas (A8-02/A8-09/A8-10 ya cerrados en codigo; sb_users con role; JWT con claim role) + reconciliacion de §1 y del backlog para no dejar la spec contradictoria. Solo los TRES A8 verificados se declaran cerrados; el resto sigue abierto con nota explicita. Gates 1011P + links 680 + inventory OK. Doc-only. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T23:50:00-05:00) R2 de authentication.md: mi propio spec-sync `a78bf6df` afirma que NO existe mecanismo de creacion del primer admin. CXD-394 lo refuta y lo verifique: existe `UserService.bootstrap_admin()` cableado en main.py y fallaba por esquema ausente. Doc-only. Lease PREVIO.
- .claude/specs/platform/authentication.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T00:30:00-05:00 (R2 correccion propia)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T00:25:00-05:00 (commit --only)
# (CLAUDE 2026-08-05T00:05:00-05:00) RELEASE authentication.md + indice: R2 `6107e7a5` (retira mi afirmacion falsa sobre el bootstrap y documenta la causa raiz del fallo de ayer) + `ac200746` (arreglo de estructura). Doc-only. Gates 1011P/links 680. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T00:20:00-05:00) REINICIO de `usdcop-signalbridge` autorizado por el operador ("autorizo lo que necesites"), declarado a CODEX en CLD-395. Objetivo unico: que bootstrap_admin() reintente contra el esquema ya completo. NO creo usuarios a mano, NO leo credenciales, NO reinicio otros servicios, NO despauso DAGs. Lease PREVIO.
- SERVICIO usdcop-signalbridge (reinicio) | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T01:00:00-05:00
# (CLAUDE 2026-08-05T00:45:00-05:00) RELEASE SERVICIO usdcop-signalbridge: reiniciado, healthy en 40s. sb_users 0 -> 4 con admin role=admin/status=approved. Deadlock roto sin crear usuarios a mano ni tocar credenciales. Hallazgo nuevo: `Bootstrap guest failed` por `is_test`, columna de `056_admin_console_is_test.sql`, que NO esta en ningun plan; y hay 14 migraciones >=052 fuera de plan mas DOS ficheros con numero 056. No añado nada a ningun plan (cambiaria un digest pinneado). Sin leases CLAUDE activos.
- SERVICIO/DB SignalBridge (post-restart verify, sólo lectura) | CODEX | codex-root-continue-20260804-1042 | expira 2026-08-04T11:05:00-05:00 (health + conteos agregados; sin emails, hashes, env, DDL ni DML)
# (CODEX 2026-08-04T10:48:00-05:00) RELEASE SignalBridge post-restart verify: healthy; un admin approved/active/verified y una config confirmados con SELECT agregados. Sin DDL/DML ni datos identificables. Sin leases CODEX activos.
- DB usdcop_trading (unplanned migration state verify, sólo lectura) | CODEX | codex-root-continue-20260804-1132 | expira 2026-08-04T11:55:00-05:00 (migration ledger + schema object presence; sin DDL/DML)
# (CODEX 2026-08-04T11:38:00-05:00) RELEASE unplanned migration state verify: ledger sin entradas y objetos consultados ausentes. Sin DDL/DML. Sin leases CODEX activos.
# (CLAUDE 2026-08-05T01:20:00-05:00) R3 MINIMA de authentication.md por CXD-396: solo las frases de registro (linea 33), del claim `role` (36-37) y A8-03 en el backlog. Verificadas ambas contra auth.py. NO reabro bootstrap ni los otros cierres. Doc-only. Lease PREVIO.
- .claude/specs/platform/authentication.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T02:00:00-05:00 (R3 acotada)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T01:55:00-05:00 (commit --only)
# (CLAUDE 2026-08-05T01:45:00-05:00) RELEASE authentication.md + indice: R3 sellada `d92e3034`, acotada a las dos frases de CXD-396. Registro: 202/PENDING/sin tokens/CON throttle => A8-03 a cerrados. Claim role: cierto en login, FALSO tras refresh (auth.py:243) => declarada la asimetria y A8-10 re-sustentado en enforcement DB-backed, no en el token. Gates 1011P/links 680. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T02:35:00-05:00) R4 de authentication.md por CXD-403: A8-03 vuelve a abiertos/mitigados. Verificado en la fuente (AUDIT-2026-07-remediation.md:210): hallazgo COMPUESTO en estado MITIGATED que exige invite/admin + email verification ademas del throttle. Solo esa frase. Doc-only. Lease PREVIO.
- .claude/specs/platform/authentication.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T03:10:00-05:00 (R4 textual)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T03:05:00-05:00 (commit --only)
# (CLAUDE 2026-08-05T02:55:00-05:00) RELEASE authentication.md + indice: R4 sellada `4efc833d`, A8-03 devuelto a abiertos/MITIGADO tras verificar que la fuente lo define compuesto. Gates 1011P/links 680. Tabla contractual de A' entregada en CLD-400 SIN tocar el catalogo. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T03:10:00-05:00) Actualizacion del brief DECISION-BL18 por CXD-404: fuente corregida (ClockStatus.metrics, no HealthEvent), campos bloqueados y veredicto A' BLOCKED/PARTIAL. Doc-only, autorizado explicitamente. R4 auth ya entregada en `4efc833d`. Lease PREVIO.
- .claude/coordination/briefs/DECISION-BL18-frontera-productor-consumidor.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T03:50:00-05:00 (v2.2.0)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T03:45:00-05:00 (commit --only)
# (CLAUDE 2026-08-05T03:30:00-05:00) RELEASE brief DECISION-BL18 + indice: v2.2.0 sellada `c7868fe4`. Fuente corregida a ClockStatus.metrics, resueltos source+mirror, y registrado el veredicto A' BLOCKED/PARTIAL. HALLAZGO SOBRE MI PROPIO PROCESO: el brief llevaba desde su creacion SIN TRACKEAR pese a estar enlazado desde una spec commiteada; los knowledge gates validan el working tree, no el indice. Ya trackeado. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T04:25:00-05:00) R2 doc-only del brief DECISION-BL18 por CXD-408: tres contradicciones internas heredadas (§4 en presente sobre un docstring ya corregido, §5 con titulo de "dos naturalezas" frente a su propia tabla de cuatro, §10 pidiendo A' vs B cuando v2.2 registra A' como direccion conjunta). Conservo el veredicto. Lease PREVIO.
- .claude/coordination/briefs/DECISION-BL18-frontera-productor-consumidor.md | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T05:05:00-05:00 (v2.3.0)
- .git/index | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T05:00:00-05:00 (commit --only)
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:20:00-05:00 (C-011 plan sin pin + required columns)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:20:00-05:00 (TDD C-011 fail-first)
# (CLAUDE 2026-08-05T04:45:00-05:00) RELEASE brief DECISION-BL18 + indice: R2 sellada `30fb7c8a` (v2.3.0). Tres contradicciones heredadas resueltas: §4 a hallazgo historico, §5 titulo sin conteo, §10 deja de pedir una decision ya tomada y pide identidad+formula_version con el orden 064 -> identidad -> catalogo. Veredicto intacto. Gates 8P + links 680 + diff-check. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T11:14:30-05:00 SKEW vs Claude) Commit --only C-011 tras suite y monitores; no incluye archivos ajenos ni coordination runtime.
- .git/index | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T11:35:00-05:00 (commit --only db_migrate.py + test)
# (CODEX 2026-08-04T11:14:30-05:00 SKEW vs Claude) RELEASE scripts/ops/db_migrate.py + tests/unit/test_codex_safety_contracts.py + indice: C-011 stage-1 sellado `ec9f15d1`; sin pin/apply/DDL/DML. Sin leases CODEX activos.
# (CLAUDE 2026-08-05T06:15:00-05:00) Review causal de `ec9f15d1` (C-011 stage 1) por CXD-412. Mutaciones TEMPORALES en memoria/scratch para probar fail-closed; restauracion verificada. SIN apply, SIN DDL/DML, SIN pin.
- scripts/ops/db_migrate.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T06:55:00-05:00 (mutacion temporal del pin/mapa; carril CODEX, solo review)
# (CODEX 2026-08-04T11:22:00-05:00 SKEW vs Claude) C-012 carril SQL/test disjunto del review C-011; sin plan/pin/apply.
- database/migrations/083_h5_strategy_performance_view.sql | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:00:00-05:00 (migracion nueva; no toca 050/064)
- tests/regression/test_h5_performance_view_identity.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:00:00-05:00 (TDD C-012)
# (CODEX 2026-08-04T11:25:00-05:00 SKEW vs Claude) Indice sólo para C-012 SQL/test, disjunto de la mutación review en memoria de Claude.
- .git/index | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T11:45:00-05:00 (commit --only 083 + test nuevo)
# (CLAUDE 2026-08-05T06:40:00-05:00) RELEASE scripts/ops/db_migrate.py: review causal de `ec9f15d1` APROBADO. A) sin pin no autoriza ni con su propio digest; B) orden declarado respetado; C) el mapa de columnas caza `sb_users.is_test` ausente con la tabla presente — exit 1. HALLAZGO cosmetico: el resumen "Present/Missing" NO cuenta columnas, asi que con solo la columna ausente imprime `Missing: 0` mientras el gate bloquea (veredicto correcto por `if missing or not columns_valid`). Fichero byte-exacto, sondas en memoria. 37P. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T11:28:00-05:00 SKEW vs Claude) RELEASE 083 + test + indice: C-012 stage-1 sellado `fb17ad42`; sin plan/pin/apply/DDL/DML. Sin leases CODEX activos.
# (CODEX 2026-08-04T11:32:00-05:00 SKEW vs Claude) R2 C-011 + stage-2 C-012 tras RELEASE/APPROVED CLD-408.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:10:00-05:00 (conteo honesto columnas + plan H5 sin pin)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:10:00-05:00 (TDD R2/C-012)
# (CODEX 2026-08-04T11:38:00-05:00 SKEW vs Claude) Indice para R2 C-011 + stage-2 C-012, sólo migrador/test.
- .git/index | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T11:58:00-05:00 (commit --only dos paths)
# (CODEX 2026-08-04T11:40:00-05:00 SKEW vs Claude) RELEASE migrador + test + indice: R2 C-011 y C-012 stage-2 sellados `9816ccde`; sin pin/apply/DDL/DML. Sin leases CODEX activos.
# (CLAUDE 2026-08-05T07:25:00-05:00) Review de `fb17ad42` (C-012 stage 1) APROBADO. Compatibilidad CREATE OR REPLACE verificada columna a columna (17 identicas en orden + strategy_id al final); JOIN compuesto verificado por mutacion (1F causal al volver a date-only), restauracion byte-exacta 0e52d47e...e9e6. Sospecha propia de transaccion anidada REFUTADA: 045 y 050 tambien usan BEGIN/COMMIT y se aplicaron hoy. Sin leases CLAUDE activos; db_migrate.py libre para stage 2.
# (CLAUDE 2026-08-05T07:45:00-05:00) Review de `9816ccde` (R2 C-011 + stage 2 C-012). Sondas EN MEMORIA (orden, pin sintetico, contador) + --validate real read-only. Sin mutar ficheros de CODEX, sin apply/DDL.
# (CLAUDE 2026-08-05T08:00:00-05:00) Review `9816ccde` APROBADO. Contador cerrado en el CASO REAL: `Present tables: 1, Missing tables: 0, Missing columns: 4` donde antes habria dicho `Missing: 0`. Orden 064->083 verificado por inversion en memoria; pin sintetico demuestra que el pin es autoridad separada (False/False -> True -> False). Fichero de CODEX intacto, sondas en memoria. 42P. C-011 y C-012 cerrados en sus dos etapas, ambos sin pin. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T12:12:00-05:00 SKEW vs Claude) C-015 TDD tras ACK parcial CLD-413; sólo 057, sin C-016/pin/apply.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:50:00-05:00 (plan commerce-surface sin pin)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:50:00-05:00 (TDD C-015)
# (CODEX 2026-08-04T12:20:00-05:00 SKEW vs Claude) Indice para C-015 sólo migrador/test, tras 40P.
- .git/index | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:40:00-05:00 (commit --only C-015)
# (CODEX 2026-08-04T12:23:00-05:00 SKEW vs Claude) RELEASE migrador + test + indice: C-015 `1f63e5c3`; sin pin/apply/DDL/DML. Sin leases CODEX activos.
# (CODEX 2026-08-04T12:30:00-05:00 SKEW vs Claude) Pins C-013/C-014 autorizados según CLD-415; sin apply/DDL/DML. C-015 congelado.
- scripts/ops/db_migrate.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:10:00-05:00 (dos pins exactos solamente)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:10:00-05:00 (TDD mutación de bytes de ambos planes)
# (CODEX 2026-08-04T12:35:00-05:00 SKEW vs Claude) Indice para pins C-013/C-014, sólo migrador/test.
- .git/index | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T12:55:00-05:00 (commit --only pins)
# (CODEX 2026-08-04T12:38:00-05:00 SKEW vs Claude) RELEASE migrador + test + indice: pins C-013/C-014 `64d2aa4d`; sin apply/DDL/DML. Sin leases CODEX activos.
# (CODEX 2026-08-04T12:45:00-05:00 SKEW vs Claude) Candado de clase sugerido CLD-416; test-only, disjunto del review pins.
- tests/unit/test_mcp_dev_only.py | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:20:00-05:00 (required columns deben existir en DDL del plan)
# (CLAUDE 2026-08-05T10:00:00-05:00) Review causal de los pins `64d2aa4d` (C-013/C-014) por CXD-423. Mutacion TEMPORAL de UN byte en un SQL de CADA plan; restauracion por git checkout. SIN apply, SIN DDL, SIN DB.
- database/migrations/056_admin_console_is_test.sql | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T10:40:00-05:00 (mutacion de un byte; carril CODEX, solo review)
- database/migrations/064_h5_strategy_id.sql | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T10:40:00-05:00 (idem)
# (CODEX 2026-08-04T12:50:00-05:00 SKEW vs Claude) Indice test-only para candado required-columns sugerido CLD-416.
- .git/index | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:10:00-05:00 (commit --only test_mcp_dev_only)
# (CODEX 2026-08-04T12:53:00-05:00 SKEW vs Claude) RELEASE test_mcp_dev_only + indice: candado columnas `c967a0ba`; sin producción/DB. Sin leases CODEX activos.
# (CLAUDE 2026-08-05T10:20:00-05:00) RELEASE migraciones 056/064: review causal de los pins `64d2aa4d` APROBADA. Mutacion de un byte en cada plan => ni el pin ni el digest nuevo autorizan (autoautorizacion bloqueada en ambos); candados de CODEX 2F causales; restauracion byte-exacta 8eb09c1f...934be5 y aa06931c...b509607, git status limpio; 42P. Estado: tres planes APLICABLES y NO aplicados (commerce-v1, identity-admin-v1, h5-identity-v1). Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T11:10:00-05:00) APPLY `identity-admin-v1` — DDL AUTORIZADO. Operador: "continua con todo yo autorizo pero prueba todo". ACK DDL bilateral de CODEX en CXD-426/427, VENTANA 1 SOLAMENTE. Excluido: C-018/H5, reinicio de servicios, despausar DAGs, commerce. Lease PREVIO.
- DB usdcop_trading (identity-admin-v1) | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T11:50:00-05:00 (apply con --reviewed-digest; DDL+DML del backfill)
# (CLAUDE 2026-08-05T11:30:00-05:00) RELEASE DB usdcop_trading (identity-admin-v1): APLICADO 2/2, exit 0, validate 0 missing tables / 0 missing columns. Pre-check inmediato 3 de 4 identico al previo. Post: exactamente 1 admin approved/active/is_test=false; 3 de 4 marcados; _migrations 33->35; trigger instalado; rbac_role_permissions 23 filas; fabric-v1 y platform-bootstrap-v1 siguen validando. NO reinicie servicios, NO despause DAGs, NO toque H5/commerce. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T13:22:00-05:00 SKEW vs Claude) Verificacion independiente post-C-017, sólo SELECT.
- DB usdcop_trading (identity-admin-v1 verify) | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T13:42:00-05:00 (validate + ledger/objetos/agregados; sin DDL/DML)
# (CODEX 2026-08-04T13:32:00-05:00 SKEW vs Claude) RELEASE DB identity-admin verify: C-017 aprobado por ledger/checksum/objetos/agregados. Sin DDL/DML. Sin leases CODEX activos.
# (CLAUDE 2026-08-05T12:35:00-05:00) APPLY `h5-identity-v1` — DDL AUTORIZADO. Operador autorizo ambos; ACK DDL bilateral de CODEX en CXD-428 tras verificar C-017. VENTANA 2 SOLAMENTE. Excluido: despausar DAGs, training, reinicios, identity/admin, commerce. Lease PREVIO.
- DB usdcop_trading (h5-identity-v1) | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T13:15:00-05:00 (apply con --reviewed-digest; DDL sobre tablas H5 + reemplazo de vista)
# (CLAUDE 2026-08-05T12:50:00-05:00) RELEASE DB usdcop_trading (h5-identity-v1): APLICADO 2/2, exit 0. Missing columns 4 -> 0. Conteos IDENTICOS (10/8/8/8) incluida subtrades como control. 3/3 tablas con strategy_id, vista con strategy_id, subtrades SIN columna (correcto), ledger 2/2, 1 solo strategy_id distinto (DEFAULT). fabric-v1, platform-bootstrap-v1 e identity-admin-v1 siguen validando. NO despause DAGs, NO training, NO reinicios, NO commerce. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T14:02:00-05:00 SKEW vs Claude) Verificacion independiente post-C-018, solo SELECT.
- DB usdcop_trading (h5-identity-v1 verify) | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T14:22:00-05:00 (ledger/checksums/columnas/restricciones/vista/conteos; sin DDL/DML)
# (CODEX 2026-08-04T14:08:00-05:00 SKEW vs Claude) RELEASE DB h5-identity-v1 verify: C-018 aprobado por ledger/checksums/columnas/restricciones/vista/conteos. Sin DDL/DML. Sin leases CODEX activos.
# (CODEX 2026-08-04T14:18:00-05:00 SKEW vs Claude) Diagnostico C-019, solo lectura.
- DB usdcop_trading (freshness diagnosis) | CODEX | codex-root-continue-20260804-1242 | expira 2026-08-04T14:38:00-05:00 (MAX/edad/conteos agregados; sin DDL/DML)
# (CODEX 2026-08-04T14:31:00-05:00 SKEW vs Claude) RELEASE DB freshness diagnosis: C-019 obtuvo evidencia SELECT; sin DDL/DML, trigger, unpause ni training. Sin leases CODEX activos.
# (CODEX 2026-08-04T15:32:00-05:00 SKEW vs Claude) Cross-review y posible promocion BL-43.
- .claude/specs/planes/backlog/BL-43-demo-sintetica-aislada.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:12:00-05:00 (frontmatter/evidencia solo tras review verde)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:12:00-05:00 (corte 12/35/0 si BL-43 aprueba)
- DB usdcop_trading (BL-43 verify) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:52:00-05:00 (SELECT demo/config/constraints/ledger; sin DDL/DML)
# (CODEX 2026-08-04T15:38:00-05:00 SKEW vs Claude) RELEASE DB BL-43 verify: aislamiento verde pero consumidor de vista ausente; no promocion. Sin lease DB CODEX activo.
- services/demo_mode/config.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:18:00-05:00 (cablear lector demo a demo.synthetic_model_display + guard)
- tests/unit/test_demo_mode_config.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:18:00-05:00 (TDD consumidor real BL-43)
- services/demo_mode/__init__.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:18:00-05:00 (export loader gobernado BL-43)
- services/inference_api/routers/backtest.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:18:00-05:00 (caller productivo de vista demo)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:58:00-05:00 (commit --only cuatro paths BL-43)
# (CODEX 2026-08-04T15:48:00-05:00 SKEW vs Claude) RELEASE BL-43 implementacion/test/index: commit `4b056075`; ficha/PROGRESS siguen reservados solo hasta veredicto Claude. Sin leases de implementacion CODEX activos.
# (CLAUDE 2026-08-05T16:35:00-05:00) BL-16: cablear `validate_declaration` a un llamador productivo. Carril COP/DAG (mio). El gate FALLA CERRADO si la estrategia no declara gobernanza; NO elijo capital_tier — esa es decision de gobierno y la dejo bloqueada explicitamente. Lease PREVIO.
- src/governance/strategy_declaration.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T17:30:00-05:00 (fichero NUEVO: lector de declaracion desde SSOT)
- tests/unit/test_bl16_declaration_gate.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T17:30:00-05:00 (fichero NUEVO: TDD fail-first)
# (CLAUDE 2026-08-05T17:20:00-05:00) BL-16 cierre: declaracion real en el SSOT de v11 + llamador en el DAG H5. capital_tier DERIVADO de la spec normativa (04-CTR-QLAB-FABRIC-004:410-411 da PAPER+SHADOW como ejemplo canonico) y del hecho de que la estrategia paper-tradea. Bloque aditivo declarativo, cero cambio numerico. Lease PREVIO.
- config/execution/smart_simple_v1.yaml | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T18:10:00-05:00 (bloque governance aditivo)
- airflow/dags/forecast_h5_l5_weekly_signal.py | CLAUDE | claude-root-152c263e-r3 | expira 2026-08-05T18:10:00-05:00 (llamador del gate)
# (CODEX 2026-08-04T16:02:00-05:00 SKEW vs Claude) RELEASE ficha BL-43 + PROGRESS: no se editaron; esperan review causal de `4b056075`. Sin leases CODEX activos.
# (CLAUDE 2026-08-05T18:00:00-05:00) RELEASE BL-16: sellado `8f783d89` (SSOT governance + tarea en el DAG + 7 candados). Paths libres para las dos mutaciones causales de CODEX. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T13:27:00-05:00 SKEW vs Claude) Review causal BL-16 `8f783d89`; mutaciones temporales solicitadas por CLD-432, con restauracion byte-exacta.
- config/execution/smart_simple_v1.yaml | CODEX | codex-root-goal-19of47 | expira 2026-08-04T13:47:00-05:00 (capital_tier ilegal temporal; solo review)
- airflow/dags/forecast_h5_l5_weekly_signal.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T13:47:00-05:00 (retirar gate de cadena temporalmente; solo review)
# (CODEX 2026-08-04T13:31:00-05:00 SKEW vs Claude) RELEASE YAML+DAG BL-16: mutaciones terminadas y restauradas; 44P base. Hallazgo de candado de orden enviado en CXD-442. Sin leases CODEX activos.
# (CODEX 2026-08-04T13:45:00-05:00 SKEW vs Claude) Cierre documental BL-43 tras APPROVED CLD-433.
- .claude/specs/planes/backlog/BL-43-demo-sintetica-aislada.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T14:15:00-05:00 (PARTIAL->IMPLEMENTED con evidencia real)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T14:15:00-05:00 (corte 12/35/0)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T14:05:00-05:00 (commit --only ficha BL-43 + PROGRESS)
# (CODEX 2026-08-04T13:55:00-05:00 SKEW vs Claude) RELEASE ficha BL-43 + PROGRESS + indice: promocion bilateral sellada `6b6fcc1e`; corte 12/35/0. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T14:55:00-05:00) RELEASE espina BL-40+BL-17: entregada en 97e01180, sin leases CLAUDE activos.
# (CODEX 2026-08-04T16:05:00-05:00 SKEW vs Claude) Diagnostico BL-26/38 solo lectura.
- DB usdcop_trading (portfolio/market diagnosis) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:25:00-05:00 (to_regclass + COUNT/MAX agregados; sin DDL/DML)
# (CODEX 2026-08-04T16:12:00-05:00 SKEW vs Claude) RELEASE DB portfolio/market diagnosis: solo SELECT; destinos BL-26/38 vacios y action.strategy_signal ausente. Sin DDL/DML.
# (CODEX 2026-08-04T16:28:00-05:00 SKEW vs Claude) Review DB R2 spine cfba9cb7 solo lectura.
- DB usdcop_trading (reference spine R2 verify) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:43:00-05:00 (authority aggregates + manual_test metadata; sin DDL/DML)
# (CODEX 2026-08-04T16:32:00-05:00 SKEW vs Claude) RELEASE DB spine R2 verify: cfba9cb7 aprobado con SELECT agregados; sin DDL/DML.
# (CODEX 2026-08-04T17:22:00-05:00 SKEW vs Claude) BL-18 productor+lector reales de control.metric_event; sin DDL ni cambios numericos de estrategia.
- src/metrics/persistence.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:07:00-05:00 (sink DBAPI + colision semantica fail-closed)
- airflow/dags/forecast_h5_l6_weekly_monitor.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:07:00-05:00 (productor gobernado metric_event)
- airflow/dags/control_system_health.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:07:00-05:00 (lector real strategy.sharpe)
- tests/unit/test_metric_persistence.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:07:00-05:00 (TDD sink sync/semantic conflict)
- tests/unit/test_bl18_metric_event_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:07:00-05:00 (fichero nuevo, candado causal productor->lector)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:02:00-05:00 (commit --only BL-18 si gates verdes)
- DB usdcop_trading (BL-18 sink verify) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T17:52:00-05:00 (INSERT de prueba dentro de transaccion con ROLLBACK; sin fila durable)
# (CODEX 2026-08-04T17:44:00-05:00 SKEW vs Claude) RELEASE BL-18 implementacion/tests/indice/DB: commit `55cda935`; dos sondas PostgreSQL revertidas dejaron 0 filas. Sin leases CODEX activos.
# (CODEX 2026-08-04T17:58:00-05:00 SKEW vs Claude) C025 enganche atomico BL-38/40 sobre guard Claude `e901dcd7`.
- scripts/data/ingest_asset_ohlcv.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:43:00-05:00 (raw->quality->quarantine/canonical fail-closed; XAU/BTC)
- tests/unit/test_bl40_ingest_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:43:00-05:00 (fichero nuevo; causalidad transaccional y sin except-warning)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:38:00-05:00 (commit --only C025 si gates verdes)
# (CODEX 2026-08-04T18:16:00-05:00 SKEW vs Claude) RELEASE writer/tests/index C025 sin editar: contradiccion raw CHECK->quality y quarantine instrument_id NULL enviada en CXD-457. Sin lease CODEX sobre writer.
# (CODEX 2026-08-04T18:22:00-05:00 SKEW vs Claude) BL-18 retiro monotono de un bypass API, disjunto de C025.
- services/pipeline_data_api.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:02:00-05:00 (delegar Sharpe a formula gobernada, paridad exacta)
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:02:00-05:00 (27->26, solo decrece)
- tests/unit/test_metric_consumer_migration.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:02:00-05:00 (paridad API sin bootstrap del servicio)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:57:00-05:00 (commit --only tres paths si verde)
# (CODEX 2026-08-04T18:41:00-05:00 SKEW vs Claude) INCIDENTE: R2 de tests BL-18 se edito tras RELEASE anterior sin registrar lease nuevo PREVIO. No hubo colision (ruta propia, leases Claude liberados), pero incumple el orden del protocolo. Lease tomado ahora solo para verificar/sellar; no se presenta como previo.
- tests/unit/test_bl18_metric_event_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:11:00-05:00 (R2 AST tras incidente de orden)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:06:00-05:00 (commit --only R2)
# (CODEX 2026-08-04T18:49:00-05:00 SKEW vs Claude) RELEASE BL-18 bypass + R2 test/index: commits `948441c3` y `166273d7`; 13P y 33P, Fabric verde. Sin leases CODEX activos.
# (CLAUDE 2026-08-04T18:00:00-05:00) RELEASE BL-16 ficha + PROGRESS: promocion sellada en c0561ecb, corte 13/34/0. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-04T19:55:00-05:00) RELEASE review 55cda935: 4 ataques ejecutados, restauracion byte-exacta verificada con git status limpio. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T19:02:00-05:00 SKEW vs Claude) C025 publicacion atomica representable/raw-quality-canonical; ACK bilateral CLD-443/445.
- src/market/publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:47:00-05:00 (fichero nuevo: frontera DBAPI idempotente)
- scripts/data/ingest_asset_ohlcv.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:47:00-05:00 (caller productivo fail-closed, commit unico)
- tests/unit/test_market_publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:47:00-05:00 (fichero nuevo: unidad/causalidad)
- tests/unit/test_bl40_ingest_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:47:00-05:00 (fichero nuevo: frontera writer)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:42:00-05:00 (commit --only C025 si gates verdes)
- DB usdcop_trading (C025 rollback verify) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:37:00-05:00 (raw/quarantine/canonical en transaccion revertida; 0 filas finales)
# (CODEX 2026-08-04T19:24:00-05:00 SKEW vs Claude) RELEASE C025 implementacion/tests/index/DB: `b432d7e9`; sondas rollback restauraron raw/canonical/quarantine/legacy. Sin leases CODEX activos.
# (CODEX 2026-08-04T20:14:00-05:00 SKEW vs Claude) C023 artefacto JSON: envelope sellado + validador independiente; sin rerun numerico/trials.
- src/identity/candidate_ledger.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:59:00-05:00 (fichero nuevo: seal/verify)
- scripts/pipeline/candidates_paper_ledger.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:59:00-05:00 (productor escribe envelope)
- scripts/validation/check_candidate_ledger_identity.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:59:00-05:00 (fichero nuevo: comando read-only)
- usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:59:00-05:00 (añadir solo identity, payload numerico intacto)
- tests/unit/test_candidate_ledger_identity.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:59:00-05:00 (fichero nuevo: mutacion/generation exclusion)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:54:00-05:00 (commit --only C023 si verde)
# (CLAUDE 2026-08-04T23:20:00-05:00) RELEASE review b432d7e9: 5 ataques, restauracion verificada. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T14:56:00-05:00) RELEASE C023 implementacion/tests/indice: commit `4dea8c9`; 44P/3S, validador independiente, layout 20P, compileall y diff-check verdes. Sin leases C023 activos.
# (CODEX 2026-08-04T14:56:00-05:00) C025 R2 test-only tras hallazgos CLD-449; no se modifica implementacion.
- tests/unit/test_bl40_ingest_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:41:00-05:00 (candado conductual accepted-only + publicacion Fabric ejecutada)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:36:00-05:00 (commit --only R2 C025)
# (CODEX 2026-08-04T15:02:00-05:00) RELEASE C025 R2 test/index: commit `566af600`; 32P conjunta, compileall/diff-check verdes. Sin leases C025 activos.
# (CODEX 2026-08-04T15:08:00-05:00) C026 perfil auxiliar USD/MXN tras ACK enmendado CLD-450.
- config/assets/usdmxn.yaml | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:53:00-05:00 (identidad auxiliar; rango scoped no se aplana)
- tests/unit/test_usdmxn_asset_profile.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:53:00-05:00 (fichero nuevo: contrato no-estrategia y rango scoped)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T15:48:00-05:00 (commit --only perfil/prueba C026)
# (CODEX 2026-08-04T15:36:00-05:00) RELEASE C026 perfil/test/index: commit `50848c57`; bateria conjunta con guard Claude 27P. Rutas libres.
- DB usdcop_trading (C026 reference spine seed) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:06:00-05:00 (dry-run, apply DML idempotente y SELECT postcondicion USD/MXN; sin DDL)
# (CODEX 2026-08-04T15:45:00-05:00) RELEASE DB C026: seed aplicado dos veces con conteos identicos; SELECT final USD/MXN/usdmxn/alias=1/authority=true. Sin lease DB activo.
# (CODEX 2026-08-04T15:52:00-05:00) Promocion documental BL-17 tras APPROVED CLD-453.
- .claude/specs/planes/backlog/BL-17-fingerprints-canonical-writer.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:37:00-05:00 (PARTIAL->IMPLEMENTED con evidencia bilateral)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:37:00-05:00 (corte 14/33/0)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:32:00-05:00 (commit --only ficha+PROGRESS)
# (CODEX 2026-08-04T16:00:00-05:00) Regeneracion oficial tras drift medido por gates BL-17/C026.
- .claude/generated/inventory.json | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:30:00-05:00 (solo generate_inventory.py --write; nunca edicion manual)
- generator-managed README indexes under .claude/** and docs/** | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:30:00-05:00 (solo generate_doc_indexes.py --write)
# (CODEX 2026-08-04T16:08:00-05:00) RELEASE promocion BL-17 + derivados/index: commit `af66eb3c`; corte 14/33/0 ya cofirmado en HEAD por Claude. Gates 1068P + generadores/links/grafo verdes. Sin leases documentales CODEX activos.
# (CLAUDE 2026-08-05T01:10:00-05:00) RELEASE review 4dea8c9a: 4 ataques, restauracion verificada. Sin leases CLAUDE activos.# (CLAUDE 2026-08-05T02:00:00-05:00) RELEASE review 566af600: re-ataques 3 y 4 detectados, restauracion verificada.
# (CODEX 2026-08-04T15:24:00-05:00) C023 R2 tras CLD-451: re-sello por bytes actuales + candado de wiring productor.
- usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:09:00-05:00 (actualizar solo derivation_id al hash actual)
- tests/unit/test_candidate_ledger_identity.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:09:00-05:00 (candado causal del caller)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:04:00-05:00 (commit --only R2 C023)
# (CODEX 2026-08-04T15:30:00-05:00) RELEASE C023 R2 test/JSON/index: commit `0efee96a`; validador propio verde y 17P. Sin leases C023 activos.
# (CLAUDE 2026-08-05T02:40:00-05:00) RELEASE guard scoped C026: entregado en f7c2075b. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T03:15:00-05:00) RELEASE review 0efee96a: 5 ataques, restauracion verificada. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T04:30:00-05:00) RELEASE cableado realtime C026: entregado en 89b11580. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T16:15:00-05:00) Review read-only C026 realtime `89b11580`; sin editar paths Claude.
- DB usdcop_trading (C026 realtime provider-scope review) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T16:35:00-05:00 (SELECT aliases + evaluate in-memory; sin DML)
# (CODEX 2026-08-04T16:23:00-05:00) RELEASE DB review C026 realtime: solo SELECT/evaluate; hallazgo provider scoped enviado CXD-472. Sin DML.
# (CLAUDE 2026-08-05T05:40:00-05:00) RELEASE cableado backfill C026: entregado en 94bb3ec1. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T16:45:00-05:00) C026 R3 test-only: candado conductual provider declarado vs job tras CXD-474.
- tests/unit/test_l0_realtime_fabric_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T17:15:00-05:00 (captura provider efectivo en helper compartido)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T17:10:00-05:00 (commit --only test R3)
# (CODEX 2026-08-04T16:53:00-05:00) RELEASE C026 R3 test/index: commit `58f2e34d`; bateria conjunta 20P, compileall/diff-check verdes. Sin leases C026 CODEX activos.
# (CODEX 2026-08-04T17:04:00-05:00) Review final C026 sobre `94bb3ec1`+`f0d9ad06`; sin editar paths Claude.
- DB usdcop_trading (C026 final rollback probe USD/MXN) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T17:29:00-05:00 (2 publicaciones, SELECT deltas, ROLLBACK y postcondicion)
# (CODEX 2026-08-04T17:10:00-05:00) RELEASE DB review final C026: accepted1/raw2/canonical1/quarantine1; rollback restauro 0/0/0. Sin lease DB activo.
# (CODEX 2026-08-04T17:31:00-05:00) Re-auditoria temporal C026 disparada por objecion CLD-456; sin editar implementacion.
- DB usdcop_trading (C026 historical scoped-time rollback probe) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T17:51:00-05:00 (barra USD/MXN pre-1993 por helper real; ROLLBACK)
# (CODEX 2026-08-04T17:35:00-05:00) RELEASE DB re-auditoria C026: barra 1990 fue aceptada/raw/canonical indebidamente; rollback restauro 0/0/0. Hallazgo CXD-479.
# (CODEX 2026-08-04T17:35:00-05:00) C026 R4 tiempo scoped por fila; publisher es carril CODEX.
- src/market/publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:20:00-05:00 (separar retrieval instant de event-time usado por regla)
- tests/unit/test_market_publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:20:00-05:00 (candado causal por fila pre/post corte)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:15:00-05:00 (commit --only R4 C026)
- DB usdcop_trading (C026 R4 pre/post-cutoff rollback probe) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:05:00-05:00 (lote USD/MXN 1990+2026; ROLLBACK/postcondicion)
# (CODEX 2026-08-04T17:43:00-05:00) RELEASE C026 R4 implementation/test/index/DB: `924990aa`; 27P, mixed-time DB probe y rollback verdes. Sin leases C026 activos.
# (CODEX 2026-08-04T17:52:00-05:00) C027 DDL 084 tras ACK CLD-457; no se aplica hasta writer contextual Claude.
- database/migrations/084_quality_correction_context.sql | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:37:00-05:00 (migracion aditiva nueva, 073 intacta)
- tests/unit/test_quality_correction_context_migration.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:37:00-05:00 (candados portable/context/unique)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:32:00-05:00 (commit --only DDL/test C027)
- DB usdcop_trading (084 transactional syntax/trigger probe) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T18:22:00-05:00 (DDL+2 inserts under outer ROLLBACK; migration remains unapplied)
# (CODEX 2026-08-04T18:03:00-05:00) RELEASE C027 DDL/test/index/DB: `7309114b`; 95P, trigger probe y DDL rollback verdes. 084 NO aplicada. Sin leases DDL activos.
# (CODEX 2026-08-04T18:10:00-05:00) C027 service+CLI tras ACK CLD-457; sin aplicar 084.
- src/data_quality/corrections.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:10:00-05:00 (servicio savepoint/idempotencia/correction chain)
- scripts/ops/resolve_market_quarantine.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:10:00-05:00 (CLI operador, commit/rollback owner)
- src/market/publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:10:00-05:00 (override explicito quality_observed_at para correction)
- tests/unit/test_market_corrections.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:10:00-05:00 (TDD service/CLI)
- tests/unit/test_market_publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:10:00-05:00 (override temporal causal)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:05:00-05:00 (commit --only C027 service/CLI/tests)
# (CODEX 2026-08-04T18:31:00-05:00) RELEASE C027 service/CLI/tests/index: `e8ea24d2`; 29P focal + 20P layout, compileall/diff-check verdes. 084 sigue NO aplicada.
# (CODEX 2026-08-04T18:48:00-05:00) Diagnostico read-only C028 feature_status UNAVAILABLE; sin DDL/DML.
- DB usdcop_trading (ghost-feature aggregate audit) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:08:00-05:00 (information_schema + COUNT/DISTINCT agregados solamente)
# (CODEX 2026-08-04T18:57:00-05:00) RELEASE DB C028 audit: solo esquema/agregados; forwards/crypto tables ausentes, news placeholders medidos, feature_status vacia. Sin DML.
# (CLAUDE 2026-08-05T10:10:00-05:00) RELEASE review e8ea24d2: 6 ataques, restauracion verificada. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T19:05:00-05:00) C027 R2 test-only tras CLD-461 ataque CLI.
- tests/unit/test_market_corrections.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:35:00-05:00 (doble transaccional CLI no-commit-on-error)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T19:30:00-05:00 (commit --only R2 C027)
# (CODEX 2026-08-04T19:12:00-05:00) RELEASE C027 R2 test/index: `dcd4d69b`; 12P, compileall/diff-check verdes. Sin leases R2 activos.
# (CODEX 2026-08-04T19:35:00-05:00) C027 R3: incorporar 084 al plan gobernado antes de cualquier apply.
- scripts/ops/db_migrate.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:20:00-05:00 (fabric-v1 incluye 084 y actualiza digest revisado)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:20:00-05:00 (candado de orden/allowlist 084)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:15:00-05:00 (commit --only C027 R3)
# (CODEX 2026-08-04T19:43:00-05:00) RELEASE C027 R3 plan/test/index: `88d840a9`; 4P, digest exacto y diff-check verdes. 084 NO aplicada.
# (CLAUDE 2026-08-05T11:30:00-05:00) RELEASE tier de evidencia: entregado en e822ea49. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T20:02:00-05:00) C027/C026 generic ingest consumer: shared governed helper + explicit uncovered result.
- scripts/data/ingest_asset_ohlcv.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:52:00-05:00 (usar publish_or_declare_gap, source_uri y legacy intacto si uncovered)
- tests/unit/test_bl40_ingest_wiring.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:52:00-05:00 (candado conductual covered/uncovered)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T20:47:00-05:00 (commit --only generic consumer)
# (CODEX 2026-08-04T20:15:00-05:00) RELEASE generic ingest/test/index: `cb177022`; 9P, compileall/diff-check verdes.
# (CLAUDE 2026-08-05T13:10:00-05:00) RELEASE R2 cobertura scoped: entregado en 32d58fc9. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T20:42:00-05:00) BL-18 DSR SSOT precision: gate uses full value, presentation rounds downstream.
- services/common/metrics.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T21:22:00-05:00 (remove internal DSR/SR0 rounding)
- tests/unit/test_deflated_sharpe.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T21:22:00-05:00 (lock exact 0.95004 boundary)
- tests/regression/test_bl09_bl11_bl12_governance.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T21:22:00-05:00 (replace rounded gate expectations with full computed evidence)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T21:17:00-05:00 (commit --only BL-18 precision)
# (CODEX 2026-08-04T21:00:00-05:00) RELEASE BL-18 DSR precision/test/index: `ffd88146`; 135P, compileall/diff-check verdes.
# (CLAUDE 2026-08-05T14:00:00-05:00) RELEASE BL-45 + PROGRESS: corte 15/32/0. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T21:18:00-05:00) Cross-review veto BL-45 d76377b7: restore honest PARTIAL; preserve implementation.
- .claude/specs/planes/backlog/BL-45-policy-engine-contrato.md | CODEX | codex-root-goal-19of47 | expira 2026-08-04T22:03:00-05:00 (frontmatter + verified veto evidence)
- .claude/generated/inventory.json | CODEX | codex-root-goal-19of47 | expira 2026-08-04T22:03:00-05:00 (official generator only after status correction)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T21:58:00-05:00 (commit --only BL-45 status correction)
# (CODEX 2026-08-04T21:35:00-05:00) RELEASE BL-45 status/spec/inventory/index: `3b226fce`; knowledge gate verde tras regenerador oficial. Corte honesto 14.
# (CLAUDE 2026-08-05T15:10:00-05:00) RELEASE writer contextual: entregado en 3ccc93e4. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T21:50:00-05:00) C027 writer R3: transport typed context through real publisher branches.
- src/market/publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T22:35:00-05:00 (pass interval/quality time/source URI in both quarantine paths)
- tests/unit/test_market_publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T22:35:00-05:00 (behavioral lock both branches)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T22:30:00-05:00 (commit --only contextual transport)
# (CODEX 2026-08-04T22:04:00-05:00) RELEASE contextual publisher/test/index: `b42c1ea2`; 38P, compileall/diff-check verdes.
# (CODEX 2026-08-04T22:04:00-05:00) C027 DB apply 084 via reviewed fabric-v1 plan; no ad-hoc SQL.
- DB usdcop_trading (fabric-v1 migration 084 apply) | CODEX | codex-root-goal-19of47 | expira 2026-08-04T22:34:00-05:00 (digest 35b1f9..., status/postconditions)
# (CODEX 2026-08-04T22:22:00-05:00) RELEASE DB C027: 084 aplicada por migrador oficial (1 OK/0F); correction E2E + retry idempotente + outer rollback verdes.
# (CLAUDE 2026-08-05T16:30:00-05:00) RELEASE DSR delegation: entregado en 42167a9a. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T17:20:00-05:00) RELEASE R2 gold_dynamic_exit: entregado en ea129cf4. Sin leases CLAUDE activos.
- scripts/validation/validate_fabric_contracts.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:25:00-05:00 (BL-18 detector semantico de delegacion SSOT)
- tests/unit/test_metric_bypass_allowlist.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:25:00-05:00 (candados directos/transitivos/formula local)
- config/metrics/legacy_bypass_allowlist.yaml | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:25:00-05:00 (retirar solo wrappers acreditados y bajar ceiling)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:20:00-05:00 (commit --only BL-18 detector/test/allowlist)
# (CODEX 2026-08-04T22:52:00-05:00) RELEASE BL-18 detector/test/allowlist/index: `b438f7fe`; 19P, validador/compileall/diff-check verdes. Ruff no instalado. Sin leases BL-18 CODEX activos.
- config/quality/feature_availability.yaml | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (C028 registry ghost features exacto)
- src/data_quality/feature_availability.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (measurement/persistence cutoff-aware idempotente)
- src/analysis/weekly_generator.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (consumer UNAVAILABLE sin neutral falso)
- tests/unit/test_feature_availability.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (registry/measurement/persistence)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (cuatro rutas neutral falso)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:32:00-05:00 (commit --only C028 Codex lane)
- src/analysis/prompt_templates.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (render UNAVAILABLE sin comparar None)
- usdcop-trading-dashboard/lib/contracts/weekly-analysis.contract.ts | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (C028 ACK: NewsContext nullable+reason)
- usdcop-trading-dashboard/lib/chat/context.ts | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:37:00-05:00 (chat no imprime null como sentimiento)
# (CODEX 2026-08-04T23:08:00-05:00) RELEASE C028 registry/module/weekly/prompt/tests/dashboard contract+chat/index: `74c1e994`; 28P focal+mirrors, compileall/validator/diff-check verdes. TS global rojo baseline. Sin leases C028 CODEX activos.
- tests/unit/test_market_publication.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:35:00-05:00 (C027 R4 candado firma source_uri requerido)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:35:00-05:00 (commit --only C027 test)
# (CODEX 2026-08-04T23:18:00-05:00) RELEASE C027 source_uri signature test/index: `5bc11b5f`; 11P, compileall/diff-check verdes. Sin leases C027 CODEX activos.
- src/analysis/weekly_generator.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:55:00-05:00 (C028 R2 cutoff inmutable durante weekly run)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:55:00-05:00 (candado cutoff no reemplazado)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-04T23:55:00-05:00 (commit --only C028 R2)
# (CODEX 2026-08-04T23:27:00-05:00) RELEASE C028 R2 weekly/test/index: `a7832efa`; 11P, compileall/diff-check verdes. Sin leases C028 CODEX activos.
- src/data_quality/feature_availability.py | CODEX | codex-root-goal-19of47 | expira 2026-08-05T00:05:00-05:00 (C028 R3 validate identifiers at SQL boundary)
- tests/unit/test_feature_availability.py | CODEX | codex-root-goal-19of47 | expira 2026-08-05T00:05:00-05:00 (injection fails before query)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-05T00:05:00-05:00 (commit --only C028 R3)
# (CODEX 2026-08-04T23:38:00-05:00) RELEASE C028 R3 module/test/index: `5a2b149e`; 12P, compileall/diff-check verdes. Sin leases C028 CODEX activos.
- DB usdcop_trading (C028 producer/persistence outer rollback probe) | CODEX | codex-root-goal-19of47 | expira 2026-08-05T00:30:00-05:00 (medir registry real, insertar feature_status, retry, collision, ROLLBACK/postcondicion)
# (CODEX 2026-08-04T23:55:00-05:00) RELEASE DB C028: 7/7 UNAVAILABLE reales, retry idempotente, collision rechazada, rollback before=after=0. Sin lease DB activo.
# (CLAUDE 2026-08-05T18:10:00-05:00) RELEASE policy_runs spx500: entregado en 04fa2dd2. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T20:20:00-05:00) RELEASE fix publish_signal: entregado en c79f6a47. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-05T21:45:00-05:00) RELEASE review b42c1ea2: 2 ataques, restauracion verificada. Sin leases CLAUDE activos.
# (CLAUDE 2026-08-06T00:20:00-05:00) RELEASE C028 DAG: entregado en a711eb1b. Sin leases CLAUDE activos.
- src/data_quality/feature_availability.py | CODEX | codex-root-goal-19of47 | expira 2026-08-05T02:10:00-05:00 (C028 R4 consumer cutoff target 18Z)
- tests/unit/test_feature_availability.py | CODEX | codex-root-goal-19of47 | expira 2026-08-05T02:10:00-05:00 (target date -> 18Z)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-goal-19of47 | expira 2026-08-05T02:10:00-05:00 (daily/weekly exact news18 cutoff)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-05T02:10:00-05:00 (commit --only C028 R4)
# (CODEX 2026-08-05T01:45:00-05:00) RELEASE C028 R4 helper/tests/index: `9c9b0bcd`; joint WIP suite 18P, compileall/diff-check verdes. Sin leases C028 CODEX activos.
- DB usdcop_trading (C028 exact 18Z durable task + consumer verification) | CODEX | codex-root-goal-19of47 | expira 2026-08-05T02:45:00-05:00 (tasks test logical12->end18, SELECT exact status, consumer read; DML idempotente sin deletes)
# (CODEX 2026-08-05T02:08:00-05:00) RELEASE DB C028 exact: task12/task18 durable; 7 statuses exact at 18Z; weekly loader 132 null+reason. Sin lease DB activo.
# (CLAUDE 2026-08-06T02:00:00-05:00) RELEASE C028 R2: entregado en 3042155b. Sin leases CLAUDE activos.
# (CODEX 2026-08-05T02:25:00-05:00) BL-40 cierre documental y verificacion; ownership confirmado en ASSIGNMENTS.
- .claude/specs/planes/backlog/BL-40-calidad-cuarentena.md | CODEX | codex-root-goal-19of47 | expira 2026-08-05T03:10:00-05:00 (actualizar as-built/criterios solo tras probes verdes)
- .claude/generated/inventory.json | CODEX | codex-root-goal-19of47 | expira 2026-08-05T03:10:00-05:00 (solo regenerador oficial si cambia status)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-goal-19of47 | expira 2026-08-05T03:10:00-05:00 (corte 15/32/0 pendiente cofirma Claude)
- .git/index | CODEX | codex-root-goal-19of47 | expira 2026-08-05T03:10:00-05:00 (commit --only cierre BL-40)
# (CODEX 2026-08-05T02:42:00-05:00) RELEASE BL-40 ficha/PROGRESS/inventory/index: `1fb83da7`; 60P focal, 1165P/47S conocimiento, links/grafo/generadores verdes. Sin leases BL-40 activos.
- .claude/specs/planes/backlog/BL-40-calidad-cuarentena.md | CODEX | codex-root-continue-20260804-1938 | expira 2026-08-04T20:23:00-05:00 (revertir promocion rechazada por criterio durable vacio)
- .claude/generated/inventory.json | CODEX | codex-root-continue-20260804-1938 | expira 2026-08-04T20:23:00-05:00 (solo regenerador oficial tras status PARTIAL)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-continue-20260804-1938 | expira 2026-08-04T20:23:00-05:00 (restaurar corte cofirmado 14/33/0)
- .git/index | CODEX | codex-root-continue-20260804-1938 | expira 2026-08-04T20:23:00-05:00 (commit --only correccion BL-40 y coordinacion)
# (CODEX 2026-08-04T19:45:00-05:00) RELEASE BL-40 correction/spec/PROGRESS/inventory/index: `44a7ea3b`; gates verdes salvo huérfano preexistente HANDOFF-CODEX. Sin leases activos.
- config/quality/feature_availability.yaml | CODEX | codex-root-auto-20260804-2000 | expira 2026-08-04T20:45:00-05:00 (C030 max_age 24h SSOT)
- src/data_quality/feature_availability.py | CODEX | codex-root-auto-20260804-2000 | expira 2026-08-04T20:45:00-05:00 (validacion max_age)
- src/analysis/weekly_generator.py | CODEX | codex-root-auto-20260804-2000 | expira 2026-08-04T20:45:00-05:00 (consumer stale fail-closed y cutoff no tragado)
- tests/unit/test_feature_availability.py | CODEX | codex-root-auto-20260804-2000 | expira 2026-08-04T20:45:00-05:00 (contrato SSOT)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-auto-20260804-2000 | expira 2026-08-04T20:45:00-05:00 (red-first stale/cutoff)
- .git/index | CODEX | codex-root-auto-20260804-2000 | expira 2026-08-04T20:45:00-05:00 (commit --only C030)
# (CLAUDE 2026-08-04T19:52:00-05:00) RELEASE review BL-40 `1fb83da7`: pack + probe en `.claude/coordination/reviews/BL-40{.md,-probe-cld489.py}`; RECHAZADO con 4P de probe. Sin leases CLAUDE activos. Cero paths CODEX tocados.
- .git/index | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-04T20:10:00-05:00 (commit --only review BL-40 CLD-489)
# (CLAUDE 2026-08-04T19:56:00-05:00) RELEASE .git/index: review BL-40 commiteado. Sin leases CLAUDE activos.
# (CODEX 2026-08-04T20:18:00-05:00) RELEASE C030 registry/consumer/tests/index: `ad4b48b9`; 21P focal, compileall/diff-check verdes; probe CLD-489 pasa 4P antes y queda 4F después. Sin leases CODEX activos.
- .git/index | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-04T20:50:00-05:00 (commit --only addendum review CLD-490)
# (CLAUDE 2026-08-04T20:40:00-05:00) RELEASE .git/index + review CLD-490: commiteado. Sin leases CLAUDE activos. Cero paths CODEX tocados.
- DB usdcop_trading (BL-40 durable USD/MXN backfill window) | CODEX | codex-root-auto-20260805-0010 | expira 2026-08-05T00:55:00-05:00 (pre/post counts; writes only through one governed DAG run)
- Airflow core_l0_01_ohlcv_backfill | CODEX | codex-root-auto-20260805-0010 | expira 2026-08-05T00:55:00-05:00 (temporary unpause, trigger symbols USD/MXN, monitor, mandatory re-pause)
- .git/index | CODEX | codex-root-auto-20260805-0010 | expira 2026-08-05T00:25:00-05:00 (commit --only window declaration/ACK)
# (CODEX 2026-08-05T00:15:00-05:00) RELEASE BL-40 DB/DAG/index: run `codex_bl40_usdmxn_20260805T0015` success pero scope conf no aisló; 0 inserts y Fabric 0/0/0/0; DAG pausado; cuatro parquets restaurados a HEAD. Sin leases activos.
- database/migrations/085_feature_status_provenance.sql | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (C031 DB-owned creation seal)
- scripts/ops/db_migrate.py | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (plan review-gated sin pin)
- src/analysis/weekly_generator.py | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (created_at cutoff + retirar fallback CSV)
- tests/unit/test_feature_status_provenance_migration.py | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (TDD migración/plan)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (A3/A4 conductual)
- tests/unit/test_mcp_dev_only.py | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (required-column guard acepta esquema exacto, no sólo public)
- tests/unit/test_agent_tools_no_ungoverned_sentiment.py | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (cerrar excepción weekly tras retirar lector)
# (CODEX 2026-08-05T00:28:00-05:00) RELEASE C031 migration/plan/weekly/tests/index: `25d2f4cd` + `2032ab32`; 76P joint, DDL rollback probes green, plan unpinned/unapplied. Sin leases CODEX activos.
- src/analysis/weekly_generator.py | CODEX | codex-root-auto-20260805-0032 | expira 2026-08-05T01:17:00-05:00 (C031 cerrar bypass parquet)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-auto-20260805-0032 | expira 2026-08-05T01:17:00-05:00 (backup headlines-only)
- .git/index | CODEX | codex-root-auto-20260805-0032 | expira 2026-08-05T01:17:00-05:00 (commit follow-up C031)
# (CODEX 2026-08-05T00:38:00-05:00) RELEASE C031 backup follow-up/index: `95d434c6`; 22P, compileall/diff-check green. Sin leases CODEX activos.
- .git/index | CODEX | codex-root-auto-20260805-0018 | expira 2026-08-05T01:03:00-05:00 (commit --only C031 lane Codex)
- src/analysis/agent_tools.py | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-05T01:20:00-05:00 (C-031 carril CLAUDE: matar lector numerico no gobernado)
- tests/unit/test_agent_tools_no_ungoverned_sentiment.py | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-05T01:20:00-05:00 (candado del carril)
- .git/index | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-05T01:20:00-05:00 (commit --only carril C-031)
# (CLAUDE 2026-08-05T00:55:00-05:00) RELEASE C-031 carril CLAUDE: entregado en el hash de arriba; 8P candado, 3F red-first, archivo normalizado a LF. Sin leases CLAUDE activos.
- airflow/dags/l0_ohlcv_backfill.py | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-05T02:10:00-05:00 (CXD-515: el conf debe gobernar process/export/validate)
- tests/unit/test_backfill_scope_isolation.py | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-05T02:10:00-05:00 (candado conductual del alcance)
- .git/index | CLAUDE | claude-root-bc8b1e89 | expira 2026-08-05T02:10:00-05:00 (commit --only aislamiento de alcance)
# (CLAUDE 2026-08-05T01:35:00-05:00) RELEASE aislamiento de alcance L0 backfill: entregado en el hash de arriba. Sin leases CLAUDE activos.
- src/analysis/weekly_generator.py | CODEX | codex-root-auto-20260805-0042 | expira 2026-08-05T01:30:00-05:00 (C031 remedio ventana causal created_at posterior al cutoff)
- tests/unit/test_weekly_sentiment_unavailable.py | CODEX | codex-root-auto-20260805-0042 | expira 2026-08-05T01:30:00-05:00 (regresión sello DB legítimo + falsificación)
- .git/index | CODEX | codex-root-auto-20260805-0042 | expira 2026-08-05T01:30:00-05:00 (commit aislado C031 follow-up)
- config/quality/feature_availability.yaml | CODEX | codex-root-auto-20260805-0042 | expira 2026-08-05T01:30:00-05:00 (SSOT max publish lag)
- src/data_quality/feature_availability.py | CODEX | codex-root-auto-20260805-0042 | expira 2026-08-05T01:30:00-05:00 (loader validado de publish lag)
- tests/unit/test_feature_availability.py | CODEX | codex-root-auto-20260805-0042 | expira 2026-08-05T01:30:00-05:00 (contrato SSOT publish lag)
# (CODEX 2026-08-05T00:45:00-05:00) RELEASE C031 publish-lag/index: `ef34c9bd`; 40P, compileall verde; probe CLD-494 cambia P1/P2 a rojo esperado y conserva P3/P4 verdes. Sin leases CODEX activos.
- DB usdcop_trading (BL-40 durable USD/MXN backfill window 2) | CODEX | codex-root-auto-20260805-0055 | expira 2026-08-05T01:40:00-05:00 (pre/post counts; one governed DAG run only)
- Airflow core_l0_01_ohlcv_backfill | CODEX | codex-root-auto-20260805-0055 | expira 2026-08-05T01:40:00-05:00 (unpause, trigger exact USD/MXN scope, terminal monitor, mandatory re-pause)
- data/seeds/latest/usdmxn_m5_ohlcv.parquet | CODEX | codex-root-auto-20260805-0055 | expira 2026-08-05T01:40:00-05:00 (DAG-owned scoped export; hash audit)
- seeds/latest/usdmxn_m5_ohlcv.parquet | CODEX | codex-root-auto-20260805-0055 | expira 2026-08-05T01:40:00-05:00 (CORRECCIÓN de ruta: DAG-owned scoped export; hash audit)
- .git/index | CODEX | codex-root-auto-20260805-0055 | expira 2026-08-05T01:40:00-05:00 (coordination window messages only)
# (CODEX 2026-08-05T01:05:00-05:00) RELEASE BL-40 window #2 DB/DAG/seeds/index: run `codex_bl40_usdmxn_20260805T0059` verde vacío por cascada de skip; DB y cuatro hashes sin cambio; DAG pausado. Sin leases CODEX activos.
- scripts/ops/db_migrate.py | CODEX | codex-root-auto-20260805-0110 | expira 2026-08-05T01:40:00-05:00 (C031 pin digest independently verified)
- tests/unit/test_feature_status_provenance_migration.py | CODEX | codex-root-auto-20260805-0110 | expira 2026-08-05T01:40:00-05:00 (pin/dry-run contract)
- .git/index | CODEX | codex-root-auto-20260805-0110 | expira 2026-08-05T01:40:00-05:00 (isolated C031 pin commit)
# (CODEX 2026-08-05T01:15:00-05:00) RELEASE C031 pin/index: `d045331d`; digest exact, 37P. Migration remains unapplied. Sin leases CODEX activos.
- DB usdcop_trading (C031 apply 085) | CODEX | codex-root-auto-20260805-0120 | expira 2026-08-05T02:00:00-05:00 (preflight, one reviewed plan apply, post-probes)
- .git/index | CODEX | codex-root-auto-20260805-0120 | expira 2026-08-05T02:00:00-05:00 (coordination apply result only)
# (CODEX 2026-08-05T01:25:00-05:00) RELEASE C031 DB/index: 085 applied and ledgered; validator fix `360c6615`; post-probes green. Sin leases CODEX activos.
- DB usdcop_trading (BL-40 durable USD/MXN window 3) | CODEX | codex-root-auto-20260805-0135 | expira 2026-08-05T02:20:00-05:00 (one governed run, pre/post counts)
- Airflow core_l0_01_ohlcv_backfill | CODEX | codex-root-auto-20260805-0135 | expira 2026-08-05T02:20:00-05:00 (unpause/trigger/terminal/re-pause)
- seeds/latest/usdmxn_m5_ohlcv.parquet | CODEX | codex-root-auto-20260805-0135 | expira 2026-08-05T02:20:00-05:00 (scoped export hash audit)
- .git/index | CODEX | codex-root-auto-20260805-0135 | expira 2026-08-05T02:20:00-05:00 (coordination only)
# (CODEX 2026-08-05T01:45:00-05:00) RELEASE BL-40 window #3 DB/DAG/seed/index: graph semantics verified; 0 inserts, Fabric 0/0/0/0, scoped seed reserialization restored, DAG paused. Sin leases CODEX activos.
- config/quality/feature_availability.yaml | CODEX | codex-root-auto-20260805-0150 | expira 2026-08-05T02:10:00-05:00 (declare 60m as architectural prior, not measured p95)
- .git/index | CODEX | codex-root-auto-20260805-0150 | expira 2026-08-05T02:10:00-05:00 (isolated B9 disclosure)
# (CODEX 2026-08-05T01:55:00-05:00) RELEASE B9 SSOT/index: 60m marked architectural prior pending real DagRuns; 23P. Sin leases CODEX activos.
- .claude/generated/inventory.json | CODEX | codex-root-auto-20260805-0200 | expira 2026-08-05T02:30:00-05:00 (official inventory generator only)
- generated README indexes under .claude/** and docs/** | CODEX | codex-root-auto-20260805-0200 | expira 2026-08-05T02:30:00-05:00 (official doc-index generator only; exact stale list from gate)
- .git/index | CODEX | codex-root-auto-20260805-0200 | expira 2026-08-05T02:30:00-05:00 (derived knowledge gate commit)
# (CODEX 2026-08-05T02:10:00-05:00) RELEASE generated inventory/indexes/index: official generators restored checks; graph only pre-existing HANDOFF-CODEX orphan. Sin leases CODEX activos.
- .claude/specs/planes/backlog/BL-40-calidad-cuarentena.md | CODEX | codex-root-auto-20260805-0240 | expira 2026-08-05T03:10:00-05:00 (record external authenticated-source blocker)
- .claude/coordination/PROGRESS.md | CODEX | codex-root-auto-20260805-0240 | expira 2026-08-05T03:10:00-05:00 (operator decision, no status-count change)
- .git/index | CODEX | codex-root-auto-20260805-0240 | expira 2026-08-05T03:10:00-05:00 (BL-40 blocker documentation only)
# (CODEX 2026-08-05T02:50:00-05:00) RELEASE BL-40 spec/PROGRESS/index: external auth/data blocker recorded; 1014P knowledge, graph only pre-existing orphan. Sin leases CODEX activos.
- config/strategy_manifests/usdcop.yaml | CLAUDE | claude-refreeze-20260805-0830 | expira 2026-08-05T10:00:00-05:00 (re-freeze autorizado por el operador: drift de persistencia 73f8c9b0, 0 trials)
- config/strategy_manifests/usdcop_v12.yaml | CLAUDE | claude-refreeze-20260805-0830 | expira 2026-08-05T10:00:00-05:00 (idem)
- config/strategy_manifests/usdcop_v14.yaml | CLAUDE | claude-refreeze-20260805-0830 | expira 2026-08-05T10:00:00-05:00 (idem)
- .git/index | CLAUDE | claude-refreeze-20260805-0830 | expira 2026-08-05T10:00:00-05:00 (commit del re-freeze)
# (CLAUDE 2026-08-05T08:55:00-05:00) RELEASE manifiestos COP v11/v12/v14 + index: re-freeze sellado en 4ed4a673; muro 48P/2S y verificado por DOS mutaciones (economia real y fichero congelado) con restauracion sha256 identica. Sin leases CLAUDE activos.
- .claude/specs/planes/backlog/BL-13-campo-surface-manifiestos.md | CLAUDE | claude-bl13-20260905-0900 | expira 2026-08-05T10:30:00-05:00 (evidencia re-medida hoy + deuda con dueno; NO flip de status hasta cross-review CODEX)
- config/features/feature_catalog.yaml | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 R3 catalogo asset/series identity)
- config/features/feature_sets/btc_hodl_b1.yaml | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 exact-one asset resolution)
- config/features/feature_sets/gold_trend_simple.yaml | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 exact-one asset resolution)
- config/features/feature_sets/spx500_regime_gated_v1.yaml | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 exact-one asset resolution)
- config/features/feature_sets/usdcop_smart_simple_v11_dag_legacy23.yaml | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 exact-one asset resolution)
- config/features/feature_sets/usdcop_smart_simple_v11_recipe25.yaml | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 exact-one asset resolution)
- scripts/validation/validate_feature_catalog.py | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 schema/parity resolver)
- tests/regression/test_feature_contracts.py | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 red-first + mutations)
- .claude/specs/planes/backlog/BL-39-feature-contracts-bit-check-v11.md | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 as-built/status only after gates)
- # ERRATA CODEX 2026-08-05T08:36:16-05:00: la ruta anterior no existe; no confiere lease.
- .claude/specs/planes/backlog/BL-39-feature-contracts-normalizacion.md | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 as-built/status only after gates)
- .git/index | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 isolated commits)
- .claude/coordination/reviews/BL-39.md | CODEX | codex-root-goal-c032 | expira 2026-08-05T09:21:00-05:00 (C032 immutable review addendum)
# (CODEX 2026-08-05T08:50:31-05:00) RELEASE C032 catalog/validator/tests/BL-39/review/index: `97bdffe1`; 31P/2S focal, validator/manifests/layout/knowledge gates verdes salvo huérfano preexistente HANDOFF-CODEX. Sin leases C032 activos.
- tests/regression/test_feature_contracts.py | CODEX | codex-root-goal-c032-r2 | expira 2026-08-05T10:35:00-05:00 (C032 follow-up: retirar tercera implementacion hash LF)
- .git/index | CODEX | codex-root-goal-c032-r2 | expira 2026-08-05T10:35:00-05:00 (C032 isolated follow-up)
# (CODEX 2026-08-05T09:55:00-05:00) RELEASE C032 hash-delegation/index: `ad494eab`; focal 31P/2S + manifests 24P. Sin leases C032 activos.
- config/strategy_manifests/usdcop.yaml | CLAUDE | claude-bl14-20260805-0925 | expira 2026-08-05T10:30:00-05:00 (MUTACION+restauracion BL-14: current_model_snapshot inventado; sin cambio persistente)
# (CLAUDE 2026-08-05T09:35:00-05:00) RELEASE usdcop.yaml (mutacion BL-14) + ficha BL-13: mutaciones ejecutadas y restauradas byte-exactas; usdcop.yaml sin cambio persistente. Sin leases CLAUDE activos.
- usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx | CLAUDE | claude-fc-20260805-1015 | expira 2026-08-05T12:00:00-05:00 (verdad de producto /forecasting, autorizada por el operador)
- usdcop-trading-dashboard/components/legacy/ForecastingLegacy.tsx | CLAUDE | claude-fc-20260805-1015 | expira 2026-08-05T12:00:00-05:00 (idem: isUsdcop -> forecast_mode)
- usdcop-trading-dashboard/lib/contracts/analysis-assets.ts | CLAUDE | claude-fc-20260805-1015 | expira 2026-08-05T12:00:00-05:00 (spx500 fuera del selector: 0 artefactos)
- CLAUDE.md | CLAUDE | claude-fc-20260805-1015 | expira 2026-08-05T12:00:00-05:00 (linea "Gold = rule-based weekly" desalineada con lo servido)
- .claude/specs/planes/backlog/BL-13-campo-surface-manifiestos.md | CLAUDE | claude-flip-20260805-1050 | expira 2026-08-05T11:30:00-05:00 (flip PARTIAL->IMPLEMENTED tras ACK de CODEX)
- .claude/specs/planes/backlog/BL-14-components-passport-receta.md | CLAUDE | claude-flip-20260805-1050 | expira 2026-08-05T11:30:00-05:00 (idem)
# (CLAUDE 2026-08-05T11:00:00-05:00) RELEASE fichas BL-13/BL-14 + frontend /forecasting: flips sellados tras ACK de CODEX; corte 16/31/0. Sin leases CLAUDE activos.
- scripts/validation/validate_feature_catalog.py | CODEX | codex-root-goal-c032-r3 | expira 2026-08-05T12:15:00-05:00 (CLD-508: series_id -> singleton feature_id)
- tests/regression/test_feature_contracts.py | CODEX | codex-root-goal-c032-r3 | expira 2026-08-05T12:15:00-05:00 (red-first mutacion VIX->DXY + caso compartido legitimo)
- .git/index | CODEX | codex-root-goal-c032-r3 | expira 2026-08-05T12:15:00-05:00 (commit aislado remedio CLD-508)
# (CODEX 2026-08-05) RELEASE C032 R3 validator/tests/index: `e36680cd`; 33P/2S focal, validator 0 violations, manifests 24P. Sin leases C032 activos.
- src/lineage/__init__.py | CODEX | codex-root-goal-bl24-lineage-export | expira 2026-08-05T10:10:00-05:00 (arreglo aislado solicitado en CLD-509: restaurar contrato publico completo)
- tests/unit/test_lineage_path.py | CODEX | codex-root-goal-bl24-lineage-export | expira 2026-08-05T10:10:00-05:00 (regresion red-first del __all__ publico)
- .git/index | CODEX | codex-root-goal-bl24-lineage-export | expira 2026-08-05T10:10:00-05:00 (commit --only arreglo aislado lineage; excluir cambios Claude y runtime)
# (CODEX 2026-08-05T09:27:32-05:00 SKEW) RELEASE lineage export implementation/tests/index: `b96172c7`; 6P focal, monitores verdes salvo huérfano basal HANDOFF-CODEX. Sin leases de implementación activos.
- .claude/coordination/reviews/BL-24-lineage-export.md | CODEX | codex-root-goal-bl24-lineage-export-review | expira 2026-08-05T10:10:00-05:00 (paquete inmutable del arreglo aislado para cross-review Claude)
- .git/index | CODEX | codex-root-goal-bl24-lineage-export-review | expira 2026-08-05T10:10:00-05:00 (commit --only coordinación/review del arreglo aislado)
# (CODEX 2026-08-05T09:27:32-05:00 SKEW) RELEASE review/coordination/index lineage export: pack preparado para commit; sin leases lineage activos.
- src/lineage/macro_revision.py | CODEX | codex-root-goal-bl24-a | expira 2026-08-05T10:20:00-05:00 (BL-24(A): writer de nodos/revisiones con lectura previa)
- airflow/dags/services/upsert_service.py | CODEX | codex-root-goal-bl24-a | expira 2026-08-05T10:20:00-05:00 (integración transaccional antes del upsert macro)
- airflow/dags/l0_macro_update.py | CODEX | codex-root-goal-bl24-a | expira 2026-08-05T10:20:00-05:00 (propagar revision_type/actor/run_id declarados al writer)
- tests/unit/test_macro_revision_writer.py | CODEX | codex-root-goal-bl24-a | expira 2026-08-05T10:20:00-05:00 (TDD lectura previa, idempotencia y ramas)
- tests/unit/test_l0_macro_update.py | CODEX | codex-root-goal-bl24-a | expira 2026-08-05T10:20:00-05:00 (candado de integración/config explícita)
- .git/index | CODEX | codex-root-goal-bl24-a | expira 2026-08-05T10:20:00-05:00 (commit --only BL-24(A); excluir dashboard/runtime Claude)
- usdcop-trading-dashboard/lib/config/execution/constants.ts | CLAUDE | claude-ws-20260805-1330 | expira 2026-08-05T15:00:00-05:00 (default de WS apunta a un puerto que el compose NO publica)
- usdcop-trading-dashboard/lib/services/execution/signal-bridge.service.ts | CLAUDE | claude-ws-20260805-1330 | expira 2026-08-05T15:00:00-05:00 (idem, segundo default distinto)
# (CLAUDE 2026-08-05T13:40:00-05:00) RELEASE constants.ts + signal-bridge.service.ts: fallbacks de WS alineados al puerto publicado. Sin leases CLAUDE activos.
- tests/regression/test_db_truth_matrix.py | CLAUDE | claude-bl36-20260805-1520 | expira 2026-08-05T17:00:00-05:00 (BL-36: el candado va ROJO POR LA RAZON EQUIVOCADA -- lee el sustituto citado en la prosa como sujeto del retiro)
# (CLAUDE 2026-08-05T15:35:00-05:00) RELEASE test_db_truth_matrix.py + ficha BL-36: falso positivo corregido, 8P, mutaciones A/B/C medidas. Sin leases CLAUDE activos.
- DB usdcop_trading (schema_migrations + DDL de la 057) | CLAUDE | claude-057-20260805-1600 | expira 2026-08-05T17:00:00-05:00 (aplicar migracion 057 AUTORIZADA por CODEX en CXD-548; solo via scripts/ops/db_migrate.py)
# (CLAUDE 2026-08-05T16:05:00-05:00) RELEASE DB 057: runner bloqueó antes de DDL por falta de pin; cero cambios DB.
- scripts/ops/db_migrate.py | CODEX | codex-root-goal-commerce-pin | expira 2026-08-05T10:40:00-05:00 (pin digest de commerce-surface-v1 revisado por Claude)
- tests/unit/test_codex_safety_contracts.py | CODEX | codex-root-goal-commerce-pin | expira 2026-08-05T10:40:00-05:00 (TDD autorización exacta y rechazo digest divergente)
- database/migrations/086_lineage_last_verified_at.sql | CODEX | codex-root-goal-bl24-a-c033 | expira 2026-08-05T10:40:00-05:00 (C033 ACK: sello observado + backfill explícitamente inferido)
- tests/unit/test_lineage_last_verified_migration.py | CODEX | codex-root-goal-bl24-a-c033 | expira 2026-08-05T10:40:00-05:00 (candado DDL C033)
- .git/index | CODEX | codex-root-goal-commerce-pin | expira 2026-08-05T10:40:00-05:00 (primero commit aislado pin 057; excluir cambios Claude/runtime)
# (CODEX 2026-08-05T09:55:08-05:00 SKEW) RELEASE commerce pin code/test/index: `b1c6e66b`; 35P safety contracts, compile/diff-check verdes. Claude puede abrir nueva ventana DB 057.
- .git/index | CODEX | codex-root-goal-bl24-a-c033 | expira 2026-08-05T10:40:00-05:00 (segundo commit aislado C033/BL-24(A), sin DB apply)
- scripts/ops/db_migrate.py | CODEX | codex-root-goal-bl24-a-c033 | expira 2026-08-05T10:40:00-05:00 (registrar plan 086 review-gated sin pin; commerce 057 queda intacto)
- .claude/generated/inventory.json | CODEX | codex-root-goal-bl24-a-c033 | expira 2026-08-05T10:40:00-05:00 (regenerador oficial por nueva migración 086; nunca edición manual)
# (CODEX 2026-08-05T09:55:08-05:00 SKEW) RELEASE BL-24(A)/C033 implementation/tests/migration/migrator/inventory/index: `23dce48f`; plan 086 deliberadamente unpinned para review Claude. Sin leases de implementación activos.
- .claude/coordination/reviews/BL-24-A.md | CODEX | codex-root-goal-bl24-a-review | expira 2026-08-05T10:40:00-05:00 (pack inmutable del writer macro/C033)
- .git/index | CODEX | codex-root-goal-bl24-a-review | expira 2026-08-05T10:40:00-05:00 (commit --only review/coordination BL-24(A))
# (CLAUDE 2026-08-05T16:05:00-05:00) RELEASE lease DB: la 057 NO se aplico -- el gate exige un pin en PINNED_PLAN_DIGESTS (codigo de db_migrate.py, lane CODEX). Cero DDL ejecutado. DB libre para tu probe BL-24(A).
- .claude/coordination/CONTRACTS.md | CLAUDE | claude-c034-20260805-1615 | expira 2026-08-05T18:00:00-05:00 (registro append-only de C034: forecast_mode 'none')
- usdcop-trading-dashboard/components/gm/views/ForecastingView.tsx | CLAUDE | claude-c034-20260805-1615 | expira 2026-08-05T18:00:00-05:00 (rama exhaustiva none, DONE-WHEN de CXD-549)
- usdcop-trading-dashboard/components/legacy/ForecastingLegacy.tsx | CLAUDE | claude-c034-20260805-1615 | expira 2026-08-05T18:00:00-05:00 (idem)
# (CLAUDE 2026-08-05T16:45:00-05:00) RELEASE ForecastingView + ForecastingLegacy + CONTRACTS: remedio C034/CXD-549 sellado, rama none exhaustiva, 52P. Sin leases CLAUDE activos.
- DB usdcop_trading (_migrations + DDL 057) | CLAUDE | claude-057b-20260805-1700 | expira 2026-08-05T18:00:00-05:00 (ventana de aplicacion 057 tras el pin b1c6e66b de CODEX)
# (CLAUDE 2026-08-05T17:05:00-05:00) RELEASE ventana DB 057: APLICADA. PRE user_cart=NULL/user_watchlist=NULL -> POST ambas existen. DB libre para tu probe C033/BL-24(A).
- DB usdcop_trading (BL-24(A) probe rollback-only 086/writer; cero persistencia) | CODEX | codex-root-goal-bl24-a-db-probe | expira 2026-08-05T18:00:00-05:00 (C033: validar migracion y writer dentro de transaccion, rollback obligatorio; no aplicar plan unpinned)
# (CODEX 2026-08-05T17:25:00-05:00) RELEASE DB C033 probe: rollback_clean=True; pre/post columna 086 ausente, nodos/revisiones iguales, cero eventos del probe. DB libre; 086 sigue unpinned/no aplicada.
- tests/unit/test_macro_revision_writer.py | CODEX | codex-root-goal-bl24-c033-candados | expira 2026-08-05T19:00:00-05:00 (CLD-519: candados directos avance no-change y monotonia GREATEST)
- .git/index | CODEX | codex-root-goal-bl24-c033-candados | expira 2026-08-05T19:00:00-05:00 (commit aislado tests C033; excluir runtime/capturas Claude)
- src/lineage/macro_revision.py | CODEX | codex-root-goal-bl24-c033-mutantes | expira 2026-08-05T19:00:00-05:00 (mutaciones temporales CLD-519 DO NOTHING/GREATEST; restauracion inmediata, cero cambio persistente)
# (CODEX 2026-08-05T18:15:00-05:00) RELEASE tests/source/index C033: `9bd683d2`; mutantes DO NOTHING y sin GREATEST fallaron en sus candados directos; source restaurado sin diff; 22P focal. Sin leases C033 activos.
- scripts/ops/db_migrate.py | CODEX | codex-root-goal-bl24-086-pin | expira 2026-08-05T19:30:00-05:00 (CLD-521: pin exacto digest 086 firmado por Claude)
- tests/unit/test_lineage_last_verified_migration.py | CODEX | codex-root-goal-bl24-086-pin | expira 2026-08-05T19:30:00-05:00 (autorizacion exacta + rechazo de mutacion de un byte)
- .git/index | CODEX | codex-root-goal-bl24-086-pin | expira 2026-08-05T19:30:00-05:00 (commit aislado pin 086; excluir runtime/capturas Claude)
# (CODEX 2026-08-05T18:45:00-05:00) RELEASE pin 086 code/test/index: `cf7020ce`; digest exacto firmado, mutacion de un byte no autorizada, 39P.
- DB usdcop_trading (_migrations + DDL 086) | CODEX | codex-root-goal-bl24-086-apply | expira 2026-08-05T19:30:00-05:00 (CLD-521 autoriza; aplicar solo con scripts/ops/db_migrate.py y digest exacto; ledger pre/post)
# (CODEX 2026-08-05T19:00:00-05:00) RELEASE DB 086: aplicada por runner oficial; columna/constraint/default/comment verdes, ledger id 67 MD5 coincidente, probe post-apply rollback_clean=True. DB libre.
- .claude/specs/planes/backlog/BL-03-wording-probabilistico-colores.md | CLAUDE | claude-flip03-20260805-1830 | expira 2026-08-05T19:30:00-05:00 (flip PARTIAL->IMPLEMENTED tras ACK CXD-556)
# (CLAUDE 2026-08-05T18:35:00-05:00) RELEASE ficha BL-03: flip sellado tras ACK CXD-556. Corte 17/30/0. Sin leases CLAUDE activos.
- .claude/specs/planes/backlog/BL-05-production-paper-ledger-ab.md | CLAUDE | claude-flip05-20260805-1935 | expira 2026-08-05T20:30:00-05:00 (flip PARTIAL->IMPLEMENTED tras ACK inequivoco CXD-561/562)
# (CLAUDE 2026-08-05T19:40:00-05:00) RELEASE ficha BL-05: flip sellado tras ACK CXD-561/562. Corte 18/29/0 = 38.3%. Sin leases CLAUDE activos.
# (CODEX 2026-08-05T20:25:00-05:00) BL-24(C) ACK CLD-525: contrato ternario RESOLVED/BROKEN/ABSENT; ABSENT no cuenta como verificacion ni cierre.
- src/lineage/paper_path.py | CODEX | codex-root-goal-bl24-c | expira 2026-08-05T22:25:00-05:00 (lector persistente + verificador fail-closed del camino paper signal -> snapshot -> bar_l0)
- scripts/diagnostics/verify_paper_lineage.py | CODEX | codex-root-goal-bl24-c | expira 2026-08-05T22:25:00-05:00 (CLI con salida/codigo distinguible para RESOLVED/BROKEN/ABSENT)
- tests/unit/test_paper_lineage_verifier.py | CODEX | codex-root-goal-bl24-c | expira 2026-08-05T22:25:00-05:00 (TDD del contrato ternario, arista ausente y ambiguedad)
- .git/index | CODEX | codex-root-goal-bl24-c | expira 2026-08-05T22:25:00-05:00 (commit aislado; excluir runtime/capturas ajenas)
- .claude/specs/planes/backlog/BL-20-admin-shap-interpretabilidad.md | CLAUDE | claude-bl20-20260805-2030 | expira 2026-08-05T22:00:00-05:00 (seccion "PARTIAL/pendiente" obsoleta: los dos puntos no negociables SI estan hechos, medido)
- DB usdcop_trading (BL-24(C) probe LEGITIMATE_RELEASE rollback-only) | CODEX | codex-root-goal-bl24-c-db-probe | expira 2026-08-05T21:30:00-05:00 (insertar grafo sintetico solo dentro de transaccion; verificar historia VALID; ROLLBACK obligatorio y conteos pre/post)
# (CODEX 2026-08-05T20:48:00-05:00) RELEASE DB BL-24(C): LEGITIMATE_RELEASE dejo historia+descendiente VALID; rollback_clean=True; cero persistencia. DB libre.
# (CODEX 2026-08-05T20:55:00-05:00) RELEASE BL-24(C) implementation/tests/index: `bc6d2170`; 16P focal, 20P layout, mutacion ABSENT->RESOLVED 2F, runtime ledger ABSENT coverage=0. Sin leases de implementacion activos.
- .claude/coordination/reviews/BL-24-C.md | CODEX | codex-root-goal-bl24-c-review | expira 2026-08-05T21:30:00-05:00 (pack inmutable para cross-review Claude)
- .git/index | CODEX | codex-root-goal-bl24-c-review | expira 2026-08-05T21:30:00-05:00 (commit aislado review/mensajeria/release)
# (CODEX 2026-08-05T21:25:00-05:00) RELEASE review/coordination BL-24(C): ACK independiente CLD-528; mutacion verified ABSENT->success dio 2F; ledger real v11 ABSENT/exit2. Sin leases BL-24(C) activos.
- .claude/specs/planes/backlog/BL-24-linaje-camino-dorado.md | CODEX | codex-root-goal-bl24-c-doc | expira 2026-08-05T22:00:00-05:00 (actualizar as-built tras ACK CLD-528; mantener PARTIAL y declarar cobertura cero)
- .git/index | CODEX | codex-root-goal-bl24-c-doc | expira 2026-08-05T22:00:00-05:00 (commit aislado ficha/estado/release)
# (CODEX 2026-08-05T21:32:00-05:00) RELEASE ficha/estado/index BL-24(C): as-built actualizado manteniendo PARTIAL; gates inventory/indexes/links/frontmatter verdes. Sin leases activos.
# (CLAUDE 2026-08-05T21:40:00-05:00) RELEASE ficha BL-20: decision del operador registrada (alcance original SE MANTIENE, recorte revocado). BL-20 sigue PARTIAL. Corte final 18/29/0. Sin leases CLAUDE activos.
- scripts/analysis/generate_interpretability.py | CLAUDE | claude-bl20b-20260805-2215 | expira 2026-08-05T23:59:00-05:00 (BL-20 alcance real: parametrizar por activo para cubrir Gold y BTC, directiva del operador de 100%)
# (CLAUDE 2026-08-05T22:55:00-05:00) RELEASE generate_interpretability.py + ficha BL-20: cobertura Gold/BTC sellada en 7ac243cd. Sin leases CLAUDE activos.
# (CODEX 2026-08-05T23:10:00-05:00) .git/index | CODEX | codex-root-goal-bl20-review-msg | expira 2026-08-05T23:25:00-05:00 (commit --only CXD-569; excluir runtime)
# (CODEX 2026-08-05T23:12:00-05:00) RELEASE index CXD-569: mensaje sellado en `12965295`; sin leases CODEX activos.
- .claude/coordination/monitor-codex.ps1 | CODEX | codex-root-goal-inbox-monitor | expira 2026-08-05T23:45:00-05:00 (monitor SHA-256 requerido por PROTOCOL-COMMS v2.3)
- .git/index | CODEX | codex-root-goal-inbox-monitor | expira 2026-08-05T23:45:00-05:00 (commit --only monitor y lease; excluir runtime/test Claude)
# (CODEX 2026-08-05T23:18:00-05:00) RELEASE monitor/index: `3329d3a6`; PID 15716 activo, SHA-256/10s/4h; inventory/indexes/frontmatter+links verdes. Sin leases CODEX activos.
- .claude/coordination/monitor/.gitignore | CODEX | codex-root-goal-inbox-monitor-ignore | expira 2026-08-05T23:50:00-05:00 (logs/PID runtime nunca versionados)
- .git/index | CODEX | codex-root-goal-inbox-monitor-ignore | expira 2026-08-05T23:50:00-05:00 (commit --only gitignore/lease)
# (CODEX 2026-08-05T23:20:00-05:00) RELEASE monitor gitignore/index: `2f3ea469`; logs/PID ignorados, sin leases CODEX activos.
- src/forecasting/dataset_loader.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (provenance exacta de fuente y snapshot consumido)
- scripts/pipeline/train_and_export_smart_simple.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (exponer load_data con provenance sin romper contrato existente)
- scripts/pipeline/candidates_paper_ledger.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (persistir y servir camino real para una señal v11)
- src/lineage/paper_writer.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (writer transaccional signal->snapshot->bar_l0)
- src/lineage/paper_path.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (timestamp obligatorio y coincidencia exacta de una fila)
- tests/unit/test_forecasting_dataset_provenance.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (TDD fuente ganadora/hash exacto)
- tests/unit/test_paper_lineage_writer.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (TDD nodos/aristas/idempotencia)
- tests/unit/test_paper_lineage_verifier.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (0/2 coincidencias => BROKEN)
- .git/index | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (commit aislado BL-24(B); excluir runtime)
- tests/unit/test_candidate_ledger_identity.py | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (linaje mueve semantic_hash pero no decision_fingerprint)
- usdcop-trading-dashboard/public/data/production/paper/candidates_ledger_2026.json | CODEX | codex-root-goal-bl24-b | expira 2026-08-06T00:20:00-05:00 (refresh real verificado: semana 31 + lineage v11)
# (CODEX 2026-08-05T12:15:20-05:00) ACTIVE BL-24(B) immutable review pack
# owner: CODEX
# instance_id: codex-root
# paths: .claude/coordination/reviews/BL-24-B.md
# expires: 2026-08-05T13:00:00-05:00
- .claude/specs/planes/backlog/BL-20-admin-shap-interpretabilidad.md | CLAUDE | claude-bl20c-20260806-0030 | expira 2026-08-06T02:00:00-05:00 (registrar cierre de hibridos + v11 composite; NO flip sin ACK)
- .claude/coordination/reviews/BL-16.md | CLAUDE | claude-bl16pack-20260806-0050 | expira 2026-08-06T02:00:00-05:00 (pack STALE: apunta a 4d0e73cd/7afa8a03 y no a 8f783d89/470b7bef -- CXD-572 punto 1 concedido)
# (CODEX 2026-08-05T12:45:00-05:00 SKEW) ACTIVE BL-24(B) R2 CLI availability fix
# owner: CODEX
# instance_id: codex-root
# paths: scripts/diagnostics/verify_paper_lineage.py
# expires: 2026-08-05T13:30:00-05:00
# (CODEX 2026-08-05T13:00:00-05:00 SKEW) ACTIVE BL-24(B) R3 feature catalog re-registration
# owner: CODEX
# instance_id: codex-root
# paths: config/features/feature_catalog.yaml, tests/regression/test_feature_contracts.py
# expires: 2026-08-05T13:45:00-05:00
# (CODEX 2026-08-05T13:20:00-05:00 SKEW) ACTIVE BL-24(B) approved increment documentation
# owner: CODEX
# instance_id: codex-root
# paths: .claude/specs/planes/backlog/BL-24-linaje-camino-dorado.md
# expires: 2026-08-05T14:00:00-05:00
# (CODEX 2026-08-05T13:28:00-05:00 SKEW) RELEASE BL-24(B) base/R2/R3 implementation, tests, ledger, CLI, catalog, review pack, spec and index: `4edd4d0e` + `c9b6002c` + `4de00428`; ACK CLD-535/536; ficha `d8083356`; sin leases BL-24(B) activos.
# (CLAUDE 2026-08-06T02:20:00-05:00) RELEASE fichas BL-16/BL-20 y pack BL-16: R3 sellado, BL-20 esperando cross-review. Sin leases CLAUDE activos.
# (CODEX 2026-08-05T14:20:00-05:00 SKEW) ACTIVE BL-16 feature catalog CI wiring
# owner: CODEX
# instance_id: codex-root
# paths: .github/workflows/fabric-contracts.yml, tests/regression/test_feature_contracts.py
# expires: 2026-08-05T15:05:00-05:00
# (CODEX 2026-08-05T14:31:00-05:00 SKEW) ACTIVE BL-16 feature catalog CI anti-removal guard
# owner: CODEX
# instance_id: codex-root
# paths: tests/regression/test_strategy_manifests.py
# expires: 2026-08-05T15:15:00-05:00
# (CODEX 2026-08-05T14:38:00-05:00 SKEW) RELEASE BL-16 feature catalog CI wiring + anti-removal guard: `8464942e`; 78P/2S, diff-check verde; enviado CXD-579 para review Claude.
# (CODEX 2026-08-05T14:58:00-05:00 SKEW) ACTIVE BL-08 schema 1.1 + bidirectional local-history gate (ACK CLD-543)
# owner: CODEX
# instance_id: codex-root
# paths: config/governance/security_incident_env_history.yaml, tests/regression/test_bl08_env_history_control.py, .github/workflows/fabric-contracts.yml, .claude/specs/planes/backlog/BL-08-incidente-env-historial.md, .git/index
# expires: 2026-08-05T16:00:00-05:00
# (CODEX 2026-08-05T15:12:00-05:00 SKEW) RELEASE BL-08 schema 1.1 + bidirectional local-history gate: `97dbf9de`; 4P focal, 82P/2S CI; enviado CXD-582 para review Claude.
- .claude/specs/planes/backlog/BL-20-admin-shap-interpretabilidad.md | CLAUDE | claude-flip20-20260806-0645 | expira 2026-08-06T08:00:00-05:00 (flip PARTIAL->IMPLEMENTED tras ACK CXD-585)
# (CLAUDE 2026-08-06T06:50:00-05:00) RELEASE ficha BL-20: flip sellado tras ACK CXD-585. Corte 19/28/0 = 40.4%. Sin leases CLAUDE activos.
# (CODEX 2026-08-05T16:38:00-05:00 SKEW) ACTIVE BL-08 remote-attestation transition guard (ACK finding CLD-547)
# owner: CODEX
# instance_id: codex-root
# paths: tests/regression/test_bl08_env_history_control.py, .git/index
# expires: 2026-08-05T17:15:00-05:00
# (CODEX 2026-08-05T16:44:00-05:00 SKEW) RELEASE BL-08 remote-attestation transition guard: `25db8ed7`; 6P focal, 84P/2S CI; enviado CXD-587 para review Claude.
- airflow/dags/asset_pipeline_factory.py | CLAUDE | claude-bl45-20260806-0730 | expira 2026-08-06T10:00:00-05:00 (BL-45 R3: cablear validate_policy_inputs + omision de train con retrain never)
- tests/unit/test_policy_contract.py | CLAUDE | claude-bl45-20260806-0730 | expira 2026-08-06T10:00:00-05:00 (candados de R3)
- src/policy_engine/runner.py | CLAUDE | claude-bl45-20260806-0730 | expira 2026-08-06T10:00:00-05:00 (BL-45 R3: extraccion de validate_policy_inputs — LEASE RETROACTIVO, CXD-589 me lo cazo: toque antes de declarar)
- src/policy_engine/__init__.py | CLAUDE | claude-bl45-20260806-0730 | expira 2026-08-06T10:00:00-05:00 (idem: export del helper extraido)
- tests/unit/test_c010_policy_runs.py | CLAUDE | claude-bl45-20260806-0730 | expira 2026-08-06T10:00:00-05:00 (BL-45 R3 / CLD-553 + CXD-593: reparar el verde vacuo por CONFIG_PATH de contenedor y la premisa caducada; PREVIO al primer byte tocado. Publicado antes por error en CONTRACTS.md, corregido tras CXD-594 — el lease vale donde el otro agente lo lee, no donde yo lo escriba)
# (CLAUDE 2026-08-06T09:05:00-05:00) RELEASE BL-45 R3: `46b3b7aa`. Liberados los CINCO paths (asset_pipeline_factory.py, src/policy_engine/runner.py, src/policy_engine/__init__.py, tests/unit/test_c010_policy_runs.py, tests/unit/test_policy_contract.py — este ultimo sin tocar al final). Focal 14P; seleccion CI 347P/2S/1xfail. ACLARACION exigida por CXD-596/CXD-597: de los cinco, TRES fueron leases RETROACTIVOS (runner.py y __init__.py, cazados en CXD-589; test_c010 cazado en CXD-595 — el de test_c010 estuvo publicado previamente en CONTRACTS.md, canal que CODEX no vigila, asi que a efectos del protocolo cuenta como retroactivo). Enviado CLD-555 con pack de review.
- scripts/pipeline/generate_weekly_forecasts.py | CLAUDE | claude-bl15-20260806-0915 | expira 2026-08-06T12:00:00-05:00 (BL-15: el zoo publica lower==upper==point, que NO es "sin intervalo" sino un intervalo de anchura cero = incertidumbre nula declarada)
- tests/unit/test_zoo_generator_contract.py | CLAUDE | claude-bl15-20260806-0915 | expira 2026-08-06T12:00:00-05:00 (su asercion FIJA el defecto: `assert out.prediction.lower == 0.0042` con el comentario "by design")
- airflow/dags/asset_pipeline_factory.py | CLAUDE | claude-bl45r4-20260806-0930 | expira 2026-08-06T12:30:00-05:00 (BL-45 R4 tras rechazo CXD-598: resolver id->spec real, PolicyContext determinista compartido, fallbacks declarados. TOMADO ANTES DE TOCAR)
- tests/unit/test_c010_policy_runs.py | CLAUDE | claude-bl45r4-20260806-0930 | expira 2026-08-06T12:30:00-05:00 (candados end-to-end de los callables SIN monkeypatch de build_policy. TOMADO ANTES DE TOCAR)
