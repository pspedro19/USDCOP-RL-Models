---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/app/api/admin
  - usdcop-trading-dashboard/components/admin
  - usdcop-trading-dashboard/lib/contracts/rbac.contract.ts
  - database/migrations/055_rbac_monetization.sql
  - database/migrations/056_rbac_dynamic_roles.sql
---

# Auditoría de administración y operaciones

## Operaciones críticas

Administrar usuarios, aprobar registros, cambiar roles/overrides, impersonar, recuperar sistema, gestionar
llaves, promover modelos, habilitar live y operar kill switches son acciones privilegiadas. Todas requieren
autorización server-side fresca, motivo, auditoría, protección CSRF, rate limit y respuesta idempotente.

## Backlog justificado

| ID | Riesgo | Requisito |
|---|---|---|
| ADM-P0-001 | self-lockout o eliminación del último admin | invariant DB + transacción + prueba concurrente |
| ADM-P0-002 | impersonación usada para escalar/mutar | downgrade/read-only visible; mutaciones con identidad real |
| ADM-P0-003 | recuperación del sistema abusada | step-up auth + confirmación + allowlist + auditoría fail-closed |
| ADM-P0-004 | cambio de role/override sin revocación | invalidar sesiones/tokens o revalidar operación sensible |
| ADM-P1-005 | auditoría incompleta | before/after, actor, target, request id, IP confiable, motivo |
| ADM-P1-006 | aprobación duplicada/concurrente | máquina de estados + optimistic lock/idempotency key |
| ADM-P1-007 | datos de test mezclados con producción | `is_test` visible y exclusión de KPIs/billing/live |
| ADM-P1-008 | acciones masivas accidentales | preview, count, confirmación y rollback/compensación |

## UX operativa

La consola debe mostrar quién actúa realmente y “viendo como” quién. Acciones destructivas se separan
espacialmente, usan verbo específico, explican impacto y requieren motivo. Toast no basta para operaciones
irreversibles; debe existir estado final verificable y enlace al evento de auditoría.

