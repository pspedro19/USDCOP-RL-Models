---
kind: audit
status: PARTIAL
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - .claude/codex/evidence/playwright/results.json
  - .claude/codex/evidence/playwright/html/index.html
  - .claude/codex/harness/visual-evidence.spec.ts
  - .claude/codex/harness/tests/test_assurance_contracts.py
---

# Informe de evidencia — 2026-07-20

## Resumen

| Suite | Resultado | Veredicto |
|---|---:|---|
| Knowledge frontmatter + inventory | 433 passed | GREEN |
| Assurance + knowledge combinados | 449 passed, 3 failed | RED |
| Visual/Axe desktop+mobile | 2 passed, 6 failed | RED |
| Frontend ESLint | 313 errors, 2397 warnings | RED |
| npm audit producción | 3 critical, 17 high, 27 moderate, 3 low | RED |
| Pentest checklist | 16 pass, 2 critical fail, 1 warning | RED |

## Assurance failures

1. No hay tests dedicados de billing/Wompi/checkout/cart.
2. No existe ledger `checkout_orders` + `billing_events` con evento único.
3. El contexto LLM no declara explícitamente noticias/contenido recuperado como datos no confiables.

## Evidencia visual

Se ejecutaron `/login`, `/register` y `/pricing` en Chromium desktop 1440×900 y Pixel 5, con reduced-motion.
El test verificó status HTTP, heading, overflow horizontal, Axe WCAG A/AA, consola y evidencia.

- Foco de teclado: PASS desktop y mobile.
- Las seis combinaciones ruta×viewport: FAIL por contraste serio.
- Ratios observados: 3.81–4.08, requisito 4.5:1.
- Texto afectado observado entre ~9.2px y 12.5px, por debajo del criterio mobile de 16px para body.
- Se generaron 12 PNG, 24 WEBM, 12 traces ZIP, reporte HTML y JSON (~49 MB).

Artefactos: `evidence/playwright/`. Videos y traces existen porque los tests fallaron; no constituyen por sí
solos aprobación. No se capturaron journeys autenticados de carrito/admin porque requieren fixtures de usuarios
y DB deterministas todavía no integrados en este harness.

## Release decision

**NO-GO.** No es honesto afirmar que el sistema está production-ready. Los bloqueos son objetivos y
reproducibles: pagos, IA, accesibilidad, lint y dependencias. El harness está preparado para convertirlos en
GREEN después de corregir la implementación.

## Próxima evidencia requerida

- Unit/integration/E2E de orden, firma, monto, moneda, replay, concurrencia, refund y chargeback.
- Matriz BOLA/BFLA multiusuario.
- Journeys autenticados por rol con MailHog y DB sembrada.
- Baselines visuales después de corregir contraste/tamaño.
- Repetición de npm audit + SBOM + reachability y DAST autenticado.
- Corpus adversarial IA con cero fuga/acción no autorizada.
