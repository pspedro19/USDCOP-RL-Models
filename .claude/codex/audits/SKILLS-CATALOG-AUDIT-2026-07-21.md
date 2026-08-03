---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Auditoría de skills

SkillsMP es un catálogo externo de skills para Claude/Codex; la página reporta
millones de entradas y recomienda instalar carpetas completas basadas en el
formato `SKILL.md` [catálogo SkillsMP](https://skillsmp.com/skills). No se deben
instalar todas indiscriminadamente: cada skill comunitaria requiere revisión de
provenance, permisos, dependencias y compatibilidad con la constitución
cuantitativa del repositorio.

## Skills locales disponibles

Actualmente `.claude/skills` contiene 24 skills auditables para quant trading,
datos, estadísticas, riesgo, DAGs, contratos, RBAC, release, operaciones y
frontend. Las skills frontend instaladas globalmente incluyen `frontend-design` y
`ui-ux-pro-max`; esta última generó el design system persistido en
`design-system/usdcop-trading-platform/MASTER.md`.

## Skills candidatas a incorporar (previa revisión)

- Testing/browser automation para screenshots y videos.
- Next.js/React performance y accesibilidad.
- OWASP/API security y threat modeling.
- Observability/SLO y incident response.
- Mobile/responsive QA.

## Priorización recomendada

1. **webapp-testing / Playwright**: máxima prioridad; cubre flujos de dashboard,
   checkout, RBAC, screenshots, consola y regresión visual. [Referencia](https://skillsmp.com/skills/interstellar-code-claud-skills-generic-claude-framework-skills-webapp-testing-skill-md)
2. **security-testing OWASP**: máxima prioridad por los dos hallazgos críticos
   actuales de SQL/command injection. [Referencia](https://skillsmp.com/skills/proffesor-for-testing-sentinel-api-testing-claude-skills-security-testing-skill-md)
3. **Next.js optimization / React performance**: prioridad alta para Core Web
   Vitals, caching, RSC, bundle y accesibilidad. [Referencia](https://skillsmp.com/skills/slurpyb-registry-files-skills-nextjs-optimization-skill-md)
4. **browser-use o Chrome DevTools**: prioridad alta si se dispone de una
   sesión de navegador autenticada para capturar evidencia; requiere revisión
   estricta de credenciales y ejecución headed/headless. [Referencia](https://skillsmp.com/creators/browser-use/browser-use/browser-use-skills-browser-use)
5. **quant-trader**: prioridad media como material educativo; no debe sustituir
   `quant-algo-trading`, DSR/PBO, PIT ni los gates del repositorio. [Referencia](https://skillsmp.com/skills/theneoai-awesome-skills-skills-finance-quant-trader-skill-md)

No recomiendo instalar motion, scraping o skills de ejecución financiera antes
de cerrar seguridad, provenance y PIT; aportarían superficie de riesgo sin
resolver los bloqueos actuales.

## Regla de instalación

Antes de instalar una skill se debe registrar repositorio, commit/tag, licencia,
permisos, scripts ejecutables, dependencias, datos externos y resultado de
pruebas. Las skills que sugieran desactivar gates, usar datos restatados,
iterar sobre OOS o ejecutar órdenes reales se rechazan. Toda skill adoptada debe
tener un `SKILL.md`, una prueba smoke y una entrada en este inventario.
