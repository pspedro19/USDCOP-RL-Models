---
kind: as-built
status: IMPLEMENTED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---

# Gobierno de skills externas

## Skills incorporadas

| Skill | Fuente | Instalación | Uso | Estado |
|---|---|---|---|---|
| `frontend-design` | `anthropics/skills/skills/frontend-design` | helper oficial de Codex | dirección visual e implementación frontend | instalada globalmente; activa desde próximo turno |
| `ui-ux-pro-max` | `nextlevelbuilder/ui-ux-pro-max-skill/.claude/skills/ui-ux-pro-max` | descarga directa, sin ejecutar npm | tokens, estilos, tipografía, charts, responsive/mobile | instalada globalmente; activa desde próximo turno |

AI UX Playground fue usado como catálogo/procedencia para localizar `frontend-design`; la fuente instalada es
el repositorio de Anthropic, no una copia intermediaria.

## Política

1. Instalar el conjunto mínimo aplicable; “todas las skills” sin revisión aumenta supply-chain y conflictos.
2. Registrar repo, path, versión/commit, licencia y hash antes de usar en cambios productivos.
3. Leer `SKILL.md` completo en el turno donde se aplique.
4. No ejecutar instaladores o scripts externos por defecto; revisar primero y usar sandbox.
5. Una skill aconseja; no reemplaza contratos, threat model, pruebas ni revisión humana.
6. Actualizaciones requieren diff de contenido y nueva validación.
7. Skills de diseño no pueden debilitar accesibilidad, seguridad, rendimiento o coherencia del terminal.

## Skills/capacidades a evaluar, no instalar ciegamente

- WCAG/accessibility y web-design-guidelines.
- Design engineering y design handoff.
- Arquitectura Next.js/FastAPI/PostgreSQL y API contract testing.
- Mobile/responsive y touch ergonomics.
- Threat modeling, secure coding, IaC/container security y AI evals.

Cada incorporación futura debe resolver una brecha concreta del backlog y pasar revisión de procedencia.
