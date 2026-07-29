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
- .claude/specs/planes/backlog/{BL-16-ci-constitucional-etapa0.md,BL-17-fingerprints-canonical-writer.md,BL-18-catalogo-motor-metricas.md,BL-19-schema-forecast-roles-db.md,BL-21-event-sourcing-exec.md,BL-24-linaje-camino-dorado.md,BL-27-allocator-v1-novedad.md,BL-30-execution-service-externo.md,BL-38-market-canonical-resampleo.md,BL-40-calidad-cuarentena.md,BL-43-demo-sintetica-aislada.md,BL-44-timescale-ops-perfil-fisico.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:25:00-05:00 (PLANNED→PARTIAL + estado/gaps reales)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T03:08:00-05:00 (staging exacto de doce MDs de honestidad)
# (CODEX 2026-07-29T08:01:45-05:00) Errata factual post-CLD257, comprobada contra flujo, mutante y parquet reales.
- .claude/specs/planes/backlog/{BL-16-ci-constitucional-etapa0.md,BL-27-allocator-v1-novedad.md,BL-30-execution-service-externo.md,BL-40-calidad-cuarentena.md,BL-43-demo-sintetica-aislada.md} | CODEX | codex-root-39684-20c0 | expira 2026-07-29T08:30:00-05:00 (compensación factual sin tocar producción)
- .git/index | CODEX | codex-root-39684-20c0 | expira 2026-07-29T08:14:00-05:00 (commit compensatorio exacto de cinco MDs)
- [2026-07-29T08:31:23-05:00] CLAUDE toma lease de los writers/readers H5 para el P0 del ON CONFLICT (CXD-137):
  forecast_h5_l5_weekly_signal.py, forecast_h5_l6_weekly_monitor.py, forecast_h5_l7_multiday_executor.py,
  forecast_h5_l5_vol_targeting.py, los 4 UPSERT H5 de train_and_export_smart_simple.py y el join de
  control_system_health.py. Carril exclusivo CLAUDE por ASSIGNMENTS. CODEX no toca estas rutas.
