---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Auditoría de dependencias frontend

Se instaló `jest-axe` para habilitar la suite de accesibilidad; la suite quedó
en 32/32 PASS. `npm audit` sigue reportando vulnerabilidades transitivas y
directas. Los paquetes directos críticos/altos incluyen `next`, `vitest`,
`@vitest/coverage-v8`, `@vitest/ui`, `concurrently`, `happy-dom`, `jspdf`,
`prisma`, `xlsx`, `fabric`, `lighthouse` y `@axe-core/cli`.

No se ejecutó `npm audit fix --force`: varias correcciones implican cambios
major o afectan Next/React/Prisma y podrían romper rutas o contratos. La acción
correcta es actualizar por grupos, ejecutar build, TypeScript, RBAC, QA
funcional y accesibilidad después de cada grupo, y revisar `xlsx`/Next de forma
manual.

Estado: frontend funcional y accesibilidad aprobados; seguridad de supply-chain
pendiente de remediación controlada.
