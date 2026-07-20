---
name: spec-sync
description: Verify a spec against the code it claims to describe, then refresh its last_verified date — or archive it if it has gone stale. Use when a spec looks out of date, after changing a subsystem, or when the specs gate flags a stale document.
---

# Spec sync

El antídoto operativo al drift. Una spec envejece en silencio: sigue leyéndose bien mientras
describe un sistema que ya no existe.

## Paso 1 — Leer el front matter

```yaml
kind: as-built | rule | roadmap | adr | audit | historical
status: IMPLEMENTED | PARTIAL | PLANNED | PAUSED | DEPRECATED | SUPERSEDED | HISTORICAL | ARCHIVED
last_verified: YYYY-MM-DD
code_anchors: [...]     # los archivos que esta spec describe
```

## Paso 2 — Verificar los anchors

Cada `code_anchor` debe existir. Si uno desapareció: o el código se movió (actualiza el anchor)
o la spec quedó obsoleta (archívala). **No la dejes apuntando al vacío.**

## Paso 3 — Verificar las afirmaciones, no solo los paths

Es el paso que de verdad importa. Toma las 3-5 afirmaciones concretas de la spec y compruébalas
contra el código:

- ¿Los conteos coinciden? → deben venir de `.claude/generated/inventory.json`, no de prosa.
- ¿Lo que declara "pendiente" sigue pendiente? Este es el modo de fallo más peligroso: Gold decía
  que `config/assets/xauusd.yaml` "aún no existe" cuando existía; BTC decía que no había extractor
  de derivados cuando `ingest_btc_derivatives.py` ya corría en el pipeline.
- ¿Los nombres de tabla, rutas de config y IDs de contrato existen?

## Paso 4 — Actuar

| Hallazgo | Acción |
|---|---|
| Todo correcto | `last_verified: <hoy>` |
| Detalles desfasados | corregir + `last_verified: <hoy>` |
| Describe algo ya construido como pendiente | corregir el estado + `last_verified` |
| Es un plan/snapshot ya ejecutado | `git mv` a `.claude/specs/archive/<año-mes>/` + `status: ARCHIVED` + actualizar backlinks **en el mismo commit** |
| Lo superó otro doc | `status: SUPERSEDED` + rellenar `supersedes` en el nuevo |

## Paso 5 — Comprobar

```bash
python scripts/diagnostics/generate_inventory.py --check
python -m pytest tests/regression/test_knowledge_frontmatter.py -q
```

## Constraints

- **`last_verified` significa "verifiqué esto contra el código hoy"**, no "toqué el archivo".
  Actualizarlo sin verificar destruye la única señal de frescura que tenemos.
- Archivar **preserva historia**: `git mv`, nunca copiar+borrar.
- Ningún documento activo puede depender normativamente de uno archivado.
- Los conteos no se escriben a mano: van en bloques `<!-- inv:key -->`.
