---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/plans/BACKLOG.md
  - .claude/codex/harness/run-quality-gates.ps1
  - .claude/codex/harness/tests/test_assurance_contracts.py
---

# Matriz de trazabilidad

| Requisito | Riesgo | Gate | Evidencia actual | Estado |
|---|---|---|---|---|
| Docs válidas | spec drift | knowledge tests | 433 passed | GREEN |
| Contratos TS/Python | schema drift | contract mirrors | suite dirigida verde | GREEN |
| Frontend lint | defecto/type safety | ESLint | 313 errores | RED |
| Billing cobra items | pérdida financiera | assurance + E2E webhook | no tests dedicados | RED |
| Billing idempotente | replay | DB/integration | ledger ausente | RED |
| RBAC deny-by-default | escalación | `qa:gate` + BOLA matrix | parcial | AMBER |
| Evidencia visual | regresión UX | Playwright | config existe; artefactos dispersos | AMBER |
| Mobile/WCAG | exclusión/error operativo | axe + viewport matrix | 6/8 visual tests fallan contraste | RED |
| OWASP App/API | vulnerabilidad | SAST/SCA/DAST | 3 critical + 17 high npm; pentest 2 critical | RED |
| IA resistente a injection | exfiltración/tool abuse | corpus adversarial | no gate observado | RED |
| Trading no-live por defecto | pérdida real | sandbox E2E | documentado/parcial | AMBER |
| Alpha demostrado | sobreafirmación | DSR/OOS/forward | no demostrado | RED |
| Contrato features único | training-serving skew | SSOT dimension gate | registry 15D vs experimento 20D | RED |
| Perfil estadístico L2 | datos no observados | suite descriptiva | 32 tests bloqueados: módulo ausente | RED |
| Promoción económica | desplegar pérdidas | thresholds OOS | acepta retorno -20% y Sharpe -3 | RED |
| Evidencia en manifest | cherry-picking | schema contract | faltan DSR/OOS/costos/benchmark | RED |
| Marketplace modelos | venta no reproducible | ModelSKU contract | solo add-ons por activo | RED |
| Paridad/selection/OHLCV | skew/fuga temporal | pytest dirigido | 65 passed, 1 skipped | GREEN |
| Harness agregado | gates fragmentados | harness_engine.py | PASS técnico; OOS/provider BLOCKED | RED |
| Release productivo | riesgo no cerrado | production-unblock-plan | plan definido, evidencia pendiente | RED |

Esta matriz se actualiza únicamente con artefactos reproducibles. Un documento o screenshot sin assertions no
cambia un estado a GREEN.
