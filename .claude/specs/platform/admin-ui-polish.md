# Admin Console — UI Polish + Lógica de Interacciones

> **Estado: IMPLEMENTADO (2026-07-07)** · Aplica sobre lo ya construido
> (`admin-console.md` y CTR-RBAC-001). Alcance: solo presentación e interacción — cero
> cambios de permisos/datos/endpoints salvo los marcados (URL-state de filtros; servicios
> completos en `/api/admin/system` §2.2).
>
> Contract: CTR-ADMIN-UI-001 · Tokens: `usdcop-trading-dashboard/lib/ui/tokens.ts`

---

## 1. Sistema visual (design tokens)

### 1.1 Layout
```
container:   max-width 1440px, centrado, padding-x 24px
grid:        12 columnas, gap 16px
spacing:     escala 8pt → 4 / 8 / 12 / 16 / 24 / 32 / 48
cards:       radius 12px, borde 1px rgba(148,163,184,.12), fondo superficie +1
densidad:    filas de tabla 40px (compacta), celdas px-12
```

### 1.2 Tipografía
```
page-title:    20/28 semibold           (no 32+)
section-title: 13/16 semibold, uppercase, tracking .06em, color text-secondary
kpi-value:     30/36 semibold, font-variant-numeric: tabular-nums
body:          14/20
meta:          12/16 text-secondary
mono:          timestamps, ids, montos → tabular-nums siempre
```

### 1.3 Color (semántico — cada color significa UNA cosa)
```
texto:   primario #E2E8F0 · secundario #94A3B8 (AA sobre fondo oscuro)
ok:      #10B981   solo estado (dots, badges de estado)
warn:    #F59E0B   estados degradados + badge TEST
error:   #F43F5E   solo errores y acciones destructivas confirmadas
info:    #38BDF8   enlaces, acento informativo
accent:  #22D3EE   CTAs primarios, tab activa

roles:   admin violeta #A78BFA · developer azul #60A5FA ·
         subscriber teal #2DD4BF · free gris #94A3B8
         (rojo ≠ jerarquía; rojo = peligro)

reglas:  el DOT lleva el estado; el valor numérico va neutro. Verde nunca en botones
         de acción rutinaria y texto de estado a la vez dentro del mismo card.
```

### 1.4 Componentes base
KpiTile (label/valor/delta/nota; "—" = no aplica con nota de fase, "0" = medido cero) ·
StatusDot (pulso solo en error) · Badge (rol/plan/TEST/fase) · Card (header con meta
"hace Xs"; prosa → tooltip ⓘ) · Table (hover, fila clickeable con chevron, números a la
derecha, header sticky, zebra OFF) · Drawer (480px, Esc, focus-trap) · Toast (inferior
derecha, aria-live, variante Deshacer 5 s, pausa al hover, máx 3) · EmptyState (icono +
causa + acción; prohibido texto solo).

## 2. Cambios por vista — ver historial del prompt original (§2.1–§2.6). Resumen as-built:

- **Shell**: app bar compacta (título 20px, StatusDot global = peor estado
  freshness+servicios, "Actualizado hace Xs" + ⟳), tabs con estado en URL (`?tab=`),
  tabs de fase opacas .45 con tooltip.
- **Overview**: 4 KpiTiles clickeables (Activos con toggle 7d/30d; Pendientes con
  "(N test)"), 3 cards de sistema iguales (Frescura con barra hacia umbral, Vote 2
  estado-dependiente, Servicios completos), alertas nunca mudas ("reglas armadas: N").
- **Registros**: checkboxes + barra pegajosa bulk, Aprobar primario / Rechazar ghost /
  overflow ⋯ para test-real, aprobar optimista con **commit diferido** (ver decisión),
  rechazar modal con motivo, "Esperando" vivo (>4 h warn, >24 h error), fila → drawer.
- **Usuarios**: segmentado `Reales | Test | Todos`, búsqueda debounce 250 ms con
  resaltado, badges ○ PAPER / ● LIVE, fila → drawer de ficha (read-only; editor de
  entitlements = incremento Bloque 2 de admin-console.md).
- **Sistema**: Vote 2 estado-dependiente + tooltip ⓘ, freshness con barra + botón
  copiar-DAG de recuperación, tabla completa de servicios.
- **Auditoría**: filtros → URL (deep-link), chips removibles, empty state inteligente
  ("0 visibles — N ocultas por filtro → [Incluir test]"), fila expandible con JSON
  formateado, export CSV con conteo filtrado.

## 3. Patrones de interacción — los 10 del prompt original. Decisiones as-built:

- **Undo de Aprobar = commit diferido (documentado, GATE 2):** el POST a SignalBridge
  dispara el correo con temp-password (irreversible server-side), así que el "optimismo"
  es client-side: la fila sale al instante, el POST se difiere 5 s y **Deshacer cancela
  antes de enviar** → el usuario sigue `pending` y NO se genera fila de auditoría ni
  correo. Si el POST diferido falla, la fila vuelve + toast de error.
- **Auto-refresh por widget:** freshness/servicios 30–60 s · KPIs 5 min · cola 60 s;
  cada card muestra "hace Xs"; fetch fallido ⇒ card atenuada con el último dato
  (stale-while-error), nunca tumba el resto.
- **Estado en URL:** `?tab=` + filtros de Auditoría/Usuarios/Registros + `?user=` del
  drawer.
- Timestamps relativos en vivo (absoluto en title/hover).
- Fase 2 (no ahora): ⌘K command palette.

## 4. Accesibilidad
AA en labels · focus ring accent 2px en todo interactivo · teclado en tablas/drawers ·
`prefers-reduced-motion` (motion-safe) · iconos con aria-label (el matraz "Test" lleva
label textual).

---

*Criterio de terminado: un desconocido con rol admin entiende el estado del sistema en
10 segundos desde Overview, ejecuta la acción de la cola en 2 clics con posibilidad de
deshacer, y ningún estado vacío lo deja sin saber por qué ni qué hacer.*
