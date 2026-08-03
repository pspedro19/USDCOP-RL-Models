---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-27
supersedes: []
code_anchors: []
---
# Auditoría frontend, noticias y carrito

Se endurecieron los enlaces provenientes de feeds de noticias: solo se renderizan URLs absolutas `http`/`https`; esquemas `javascript:`, `data:` y URLs malformadas quedan como texto no clicable. Los enlaces externos conservan `target=_blank` con `noopener noreferrer`.

El carrito y panel de noticias ya exponen estados de carga/error y etiquetas ARIA; mantener pruebas E2E de checkout/RBAC en CI. Próximo control: axe, teclado (focus trap del drawer) y pruebas de autorización server-side para checkout.
