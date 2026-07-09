# SFC Legal Gate — Checklist operativo (pre-`auto` con dinero real de terceros)

> **Regla dura** (CTR-RBAC-001 §9 / `.claude/rules/rbac.md`): el tier `auto` (auto-ejecución
> de señales para terceros) puede constituir actividad regulada en Colombia — órbita SFC
> (asesoría / administración de recursos de terceros). **Hasta cerrar este gate, `auto` es
> paper-only.** Este documento es la lista de verificación operativa, NO asesoría legal.
>
> Estado del enforcement técnico (verificado 2026-07-07): `trading_mode = PAPER` global
> (PreTradeGate SIMULATE antes de cualquier check de usuario, fail-safe BLOCK), todos los
> `user_risk_limits_v2.mode = 'paper'`, kill switches por usuario + global admin, llaves con
> permiso de retiro rechazadas al registro, fan-out solo produce filas PENDING paper.

## A. Consulta legal (bloqueante — decisión del operador)

- [ ] Concepto jurídico escrito (abogado con práctica en mercado de valores colombiano) sobre si
      el modelo de negocio (venta de señales + auto-ejecución en cuenta DEL CLIENTE con llaves
      propias, sin custodia) constituye: (a) asesoría de inversión, (b) administración de
      recursos de terceros, (c) actividad exclusiva de entidades vigiladas.
- [ ] Determinar si aplica registro/licencia ante la SFC o si el modelo "cliente ejecuta en su
      propio exchange con sus propias llaves, nosotros solo transmitimos señales" queda fuera
      del perímetro (el concepto debe cubrir el fan-out automático, no solo señales manuales).
- [ ] Revisar tratamiento de criptoactivos (MEXC/Binance no son valores; circulares BanRep/SFC
      sobre exposición cripto de terceros).
- [ ] Estructura societaria + jurisdicción de facturación (Wompi = pesos colombianos ⇒ actividad
      en Colombia; evaluar si el cobro por "software/señales" vs "gestión" cambia el análisis).

## B. Requisitos de producto ya implementados (verificar antes de activar)

- [x] Disclaimer persistente en toda superficie con señales ("contenido informativo y educativo;
      no es asesoría financiera; rendimientos pasados no garantizan resultados; riesgo de
      pérdida total") — `/login`, `/metodologia`, `/pricing`, vistas de señales.
- [x] Paper-first: `PreTradeGate` fail-safe (PAPER ⇒ SIMULATE, error ⇒ BLOCK); kill por usuario
      (default OFF) + kill global admin que domina; caps de notional/día techo del sistema.
- [x] Llaves del cliente cifradas (Vault AES-256-GCM), scoped a su `user_id`; **llaves con
      permiso de retiro se rechazan** (fail-closed a `pending` si no se puede probar).
- [x] `audit_log` append-only (trigger anti UPDATE/DELETE) con kills, fan-outs, aprobaciones.
- [ ] `paper_weeks_required` enforcement en la transición paper→live (campo existe en el
      entitlement; el check del paso a live aún no está cableado — bloquear activación).
- [ ] Aceptación registrada de ToS + risk disclosure ANTES de habilitar ejecución (guardar en
      `audit_log` con timestamp + IP).

## C. Activación (solo cuando A y B estén completos)

1. Concepto legal archivado en `docs/legal/` (privado si es necesario — NO commitear si contiene
   información privilegiada del negocio).
2. ADR formal que levanta el paper-only (referenciando el concepto legal).
3. `trading_mode` global pasa de PAPER solo tras: semana de testnet + cuenta fondeada propia
   (system account) + confirmación explícita del operador — NUNCA por defecto.
4. Por usuario: `mode='live'` requiere `paper_weeks_required` cumplidas + risk disclosure
   aceptado + límites configurados por debajo de los techos.

## D. Recordatorios de seguridad pendientes

- [ ] **ROTAR las llaves MEXC y Binance** que fueron pegadas en chat (expuestas 2026-07-05/06).
- [ ] Verificación anti-retiro contra exchange real cuando existan llaves vivas rotadas.
