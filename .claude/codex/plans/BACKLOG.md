---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/audits/COMMERCE-IDENTITY-BILLING-AUDIT.md
  - .claude/codex/proposals/QUALITY-HARNESS-SPEC.md
  - .claude/codex/proposals/SECURITY-ASSURANCE-SPEC.md
---

# Backlog Codex consolidado

| Prioridad | ID | Trabajo | Justificación | Evidencia de cierre |
|---|---|---|---|---|
| P0 | PAY-001 | Orden server-side y total plan+add-ons | Hoy se conceden add-ons no incluidos en el monto | tests de monto + webhook E2E |
| P0 | PAY-002 | Idempotencia/replay y ledger de eventos | Un webhook puede reprocesarse | constraint única + concurrencia verde |
| P0 | PAY-003 | Verificar monto, moneda y orden | Firma válida no demuestra que se pagó lo correcto | negativas firmadas sin grant |
| P0 | SEC-001 | BOLA/BFLA por rol y tenant | Superficie multiusuario financiera | matriz adversarial completa |
| P0 | SEC-003 | Remediar 3 critical/17 high npm | Supply-chain explotable conocida | audit/SBOM verde o excepción |
| P0 | A11Y-001 | Contraste y tipografía pública | 6/8 visual tests fallan WCAG | Axe verde desktop/mobile |
| P0 | QA-001 | Corregir frontend lint/type/build | 313 errores de lint observados | cero errores en CI |
| P0 | EXEC-001 | Paper-default y bloqueo live verificable | Evitar órdenes reales inseguras | sandbox E2E + kill switch |
| P0 | FEAT-001 | Unificar registro 15D con SSOT activo 20D | Hay drift verificable entre contratos | generation test + parity verde |
| P0 | DATA-001 | Restaurar/retirar suite de reporte L2 | 32 tests fallan por módulo ausente | suite L2 verde y módulo trazable |
| P0 | STAT-001 | Endurecer gates económicos | Aceptan retorno -20% y Sharpe -3 | OOS positivo + CI/costos |
| P1 | QA-002 | Harness único + artefactos | QA actual está fragmentado | ejecución reproducible + manifest |
| P1 | QA-003 | Playwright journeys por rol | La UI no demuestra aislamiento | traces/screenshots/videos en CI |
| P1 | SEC-002 | SAST/SCA/secrets/IaC/container/DAST | Cobertura OWASP y supply chain | SARIF/SBOM sin High abiertos |
| P1 | BILL-001 | Suscripción/refund/chargeback | Modelo actual solo suma 30 días | máquina de estados + pruebas reloj |
| P1 | AUD-001 | Auditoría fail-closed para grants | Un grant financiero sin ledger no es aceptable | rollback transaccional probado |
| P1 | UX-001 | Responsive/a11y/visual baselines | Terminal debe funcionar móvil/teclado | WCAG 2.2 AA + diffs aprobados |
| P1 | AI-001 | Evals de prompt injection/tool abuse | Noticias/chat son entrada no confiable | corpus adversarial sin acción/fuga |
| P1 | SEC-004 | Eliminar SQL/command/eval inseguros | checklist pentest marca 2 critical | allowlists/AST + checklist verde |
| P1 | ARCH-001 | Fuente canónica file/DB | Dos verdades operativas crean drift | ADR + contract tests |
| P1 | STAT-002 | Prerregistro y ledger de trials | Evitar HARKing/multiplicidad invisible | hipótesis inmutable + todos los trials |
| P1 | STAT-003 | Manifest decision-grade | No exige DSR/OOS/benchmark/costos | schema + negativas + migración |
| P1 | MKT-001 | Definir SKU antes de vender modelos | Hoy solo vende acceso por activo | SKU/licencia/disclosure/versionado |
| P1 | PERF-001 | SLO y budgets por journey | Faltan carga, colas y degradación | p95/p99 + error budgets |
| P2 | OPS-001 | Retirar servicios/DAGs sin uso | Reduce superficie y misconfiguración | inventario reconciliado |
| P2 | DOC-001 | Generar conteos/estados en specs | README y specs divergen | CI detecta drift |

## Definition of Done

- spec SSOT actualizada;
- threat model y casos negativos incluidos;
- implementación revisada;
- unit/integration/E2E verdes;
- lint/typecheck/build verdes;
- evidencia y manifest generados;
- seguridad sin Critical/High o excepción vigente;
- observabilidad y rollback verificados;
- sin datos sensibles en artefactos.
