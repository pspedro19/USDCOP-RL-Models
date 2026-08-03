---
kind: roadmap
status: PLANNED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/playwright.config.ts
  - usdcop-trading-dashboard/vitest.config.ts
  - usdcop-trading-dashboard/scripts/functional-qa.mjs
  - usdcop-trading-dashboard/scripts/visual-qa.mjs
  - .github/workflows/security.yml
  - .github/workflows/security-scan.yml
---

# Quality Harness — especificación

## Objetivo

Un único harness debe producir una decisión reproducible de calidad y un paquete de evidencia. “Pasó en mi
máquina” no es criterio de cierre. Ninguna screenshot ni video reemplaza assertions; son evidencia adicional.

## Pirámide de gates

| Gate | Herramientas | Bloquea merge | Evidencia |
|---|---|---:|---|
| G0 Spec/contratos | pytest knowledge + mirrors + inventario | Sí | JUnit + resumen |
| G1 Estática | Ruff, mypy/pyright, ESLint, `tsc --noEmit` | Sí | SARIF/JUnit |
| G2 Unit/component | pytest, Vitest, Testing Library | Sí | cobertura LCOV/XML |
| G3 DB/contract | migraciones limpias, schema diff, API contract | Sí | logs + OpenAPI diff |
| G4 Integration | servicios reales en compose de test | Sí | JUnit + logs sanitizados |
| G5 E2E | Playwright por rol y viewport | Sí | trace + screenshot en fallo |
| G6 Visual/a11y | snapshots, axe, WCAG 2.2 AA | Sí para regresión aprobada | diff + reporte axe |
| G7 Security | SAST, SCA, secrets, IaC, container scan, DAST | Sí según severidad | SARIF + SBOM |
| G8 Performance | Lighthouse + k6/Locust | presupuesto | HTML/JSON |
| G9 Trading integrity | leakage, parity, DSR/OOS, costs | Sí para promoción | reporte firmado/hash |

## Matriz E2E mínima

Cada flujo corre como `anonymous`, `free`, `subscriber-signals`, `subscriber-auto`, `developer`, `admin` y
usuario suspendido/expirado:

- registro, aprobación, rechazo, reset obligatorio y login;
- navegación y API deny-by-default;
- catálogo → watchlist → carrito → checkout aprobado/declinado/replay;
- billing/account, expiración y revocación;
- forecasting/analysis con delay por plan;
- configuración de exchange en sandbox y validación anti-withdraw;
- paper signal → fan-out → ejecución → reconciliación → kill switch;
- Vote 1/Vote 2/promoción con negativas por permisos;
- admin roles, overrides, impersonación downgrade-only y auditoría.

## Evidencia visual

- Screenshots: baseline solo para estados deterministas; ocultar reloj, IDs, cotizaciones variables y PII.
- Video: conservar en fallo y para journeys críticos; no usar como assertion.
- Trace Playwright: conservar en fallo con DOM/network/console, sanitizando tokens y cookies.
- Viewports: 390×844, 768×1024, 1440×900 y 1920×1080.
- Temas: dark/light cuando existan; zoom 100% y 200%; teclado-only.
- Retención: 30 días en CI, release evidence permanente por versión.
- Manifest: commit, fecha, entorno, seed, navegador, viewport, hashes y resultado.

## Criterio de “perfecto” operativo

No es posible garantizar software sin defectos o vulnerabilidades. El criterio verificable es: cero fallos en
gates obligatorios, cero vulnerabilidades conocidas Critical/High sin excepción aprobada, cobertura de riesgos
trazable, evidencia reproducible y rollback ensayado.
