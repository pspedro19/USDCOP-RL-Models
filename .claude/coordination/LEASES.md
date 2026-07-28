# LEASES — dueño exclusivo de rutas (expira ≤45min; renovar o liberar)
# formato: - <ruta> | <CLAUDE|CODEX> | <instance_id> | expira <ISO>
- .claude/coordination/CODEX-STATUS.md | CODEX | codex-root-5d968ac6 | expira 2026-07-27T23:50:00-05:00
- tests/regression/test_bl10_legacy_estimate_contract.py | CODEX | codex-root-5d968ac6 | expira 2026-07-27T23:55:00-05:00
- scripts/validation/check_trial_ledger.py | CODEX | codex-root-5d968ac6 | expira 2026-07-27T23:55:00-05:00
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
