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
- .claude/coordination/PROGRESS.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T13:05:00-05:00 (reescritura cofirmable al corte 9/36/2 tras BL-08 de CODEX)
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
- .claude/specs/planes/backlog/BL-45-policy-engine-contrato.md | CLAUDE | claude-root-152c263e | expira 2026-08-03T18:00:00-05:00 (auditoria 17 checks + correccion de brecha stale)
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
