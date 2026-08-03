---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Skills instaladas en el repositorio

Se instalaron con el instalador oficial de skills:

- `.claude/skills/webapp-testing` desde `Interstellar-code/claud-skills`, ruta
  `generic-claude-framework/skills/webapp-testing`.
- `.claude/skills/security-testing` desde
  `proffesor-for-testing/sentinel-api-testing`, ruta
  `.claude/skills/security-testing`.

Smoke check: ambos directorios existen y sus `SKILL.md` son legibles. No se
instalaron dependencias de navegador ni se ejecutó un navegador autenticado.
Las skills deben usarse como apoyo; no pueden saltarse los gates PIT/OOS,
RBAC, aprobación humana ni seguridad fail-closed.
