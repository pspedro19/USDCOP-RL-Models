---
kind: historical
status: ARCHIVED
version: 1.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors: []
---

# Archivo — julio 2026

> **Estos documentos ya no son referencia vigente.** Se conservan porque registran *por qué* se
> tomaron decisiones, no *cómo funciona el sistema hoy*. Nada aquí debe citarse como as-built.

## Criterio de archivado

Un documento se archiva cuando cumple alguna de estas condiciones:

1. **Es un artefacto point-in-time** (plan, roadmap, snapshot de estado, ledger de auditoría
   cerrado) que quedó mezclado con la referencia vigente.
2. **Afirma como pendiente algo que ya está construido** — el caso más peligroso, porque un
   agente que lo lee planifica trabajo ya hecho.
3. **Fue superado por otro documento** que cubre lo mismo mejor.

## Qué hay aquí y por qué

| Archivo | Razón |
|---|---|
| `QA-100-PLAN.md` | Ledger de una iteración de QA ya cerrada ("iter-1 COMPLETADA"); los harnesses que planeaba ya existen |
| `admin-ui-polish.md` | Change-order completado (2026-07-07); su contenido pertenece a `platform/admin-console.md` |
| `PLAN-binance-derivatives-2026-07.md` | Plan **ya ejecutado**: afirmaba "no hay extractor que las llene" y `scripts/data/ingest_btc_derivatives.py` existe y corre en el pipeline |
| `btcusdt-IMPLEMENTATION_ROADMAP.md` | Mismo claim muerto sobre el extractor de derivados |
| `btcusdt-IMPLEMENTATION_STATUS.md` | Snapshot fechado; decía que el funding crypto-native seguía pendiente |
| `xauusd-IMPLEMENTATION_ROADMAP.md` | Roadmap de fases ya recorridas |
| `xauusd-IMPLEMENTATION_STATUS.md` | Snapshot fechado (2 activos / 5 estrategias); hoy hay más |
| `11_IMPLEMENTATION_ROADMAP.md` | Plan de construcción del News Engine, que lleva tiempo operativo |
| `_strategy-history.md` | Superado por `assets/_ds-cycle-asbuilt.md`, que es el indexado en `CLAUDE.md` |

## Reglas

- **No actualizar** estos archivos. Si algo aquí sigue siendo cierto y útil, muévelo al documento
  as-built que corresponda; no lo revivas aquí.
- **Ningún documento activo debe depender normativamente de uno archivado.** El specs-gate lo
  verifica.
- Todo lo de aquí lleva `status: ARCHIVED` en su front-matter y queda fuera de la navegación.
