# Visual Spec Checklist (BDD) — cada screenshot se interroga contra esto

> Derivado de `ux-navigation.md` (P1-P3, §4, §6) + mejores prácticas UI. Uso: por cada
> captura de `scripts/visual-qa.mjs`, responder cada ítem; todo "NO" = hallazgo numerado
> en `FINDINGS-iter-N.md` con severidad (BLOQUEANTE / MAYOR / MENOR).
> Convergencia = iteración con 0 bloqueantes+mayores y crawl de navegación 100% verde.

## A. Honestidad de métricas (P1/P2 — BLOQUEANTE si falla)
- [ ] ¿Toda cifra de rendimiento lleva badge ● LIVE / ◆ BACKTEST / ○ PAPER?
- [ ] ¿El titular es SIEMPRE el LIVE (backtest visible pero secundario)?
- [ ] ¿Tooltip/nota de procedencia (estrategia · versión · bundle · fecha)?
- [ ] ¿Cero números fabricados/hardcodeados? (¿de dónde sale cada cifra visible?)
- [ ] ¿N<20 trades ⇒ solo conteo y P&L (sin Sharpe/p-value)?
- [ ] ¿Cero recompute del frontend (todo del bundle publicado)?

## B. Divulgación por rol (P3 — BLOQUEANTE)
- [ ] ¿El rol ve SOLO su mundo? (subscriber sin gates/votos/jerga L4/PENDING)
- [ ] ¿Los módulos denegados a clientes aparecen como 🔒 con contenido detrás (blur) + CTA, no como ausencia?
- [ ] ¿Subscriber ve "Señales" (no "Producción")? ¿Sin botón "Promover" para developer?
- [ ] ¿Cero mecanismos internos en copy de cliente (RL/PPO/L4/votos)?

## C. Estados (§4 — MAYOR)
- [ ] ¿Skeletons con la forma del contenido (nunca "Loading…" colgado)?
- [ ] ¿Errores con causa + acción ("reintentar"), sin pantallas mudas?
- [ ] ¿Mercado cerrado = pausa comunicada, no apariencia de roto?
- [ ] ¿Sin flash de hidratación (vista de otro rol por un instante)?

## D. Diseño sobrio profesional (MAYOR/MENOR)
- [ ] ¿Contraste AA (texto secundario ≥ slate-300 sobre fondo oscuro)?
- [ ] ¿Jerarquía tipográfica clara (1 h1, tamaños consistentes)?
- [ ] ¿Espaciado/alineación consistentes (UI_TOKENS)? ¿contenido centrado, sin pegarse a bordes?
- [ ] ¿Sin overflow horizontal? ¿tablas densas con scroll propio?
- [ ] ¿Números tabulares (tabular-nums) en cifras?
- [ ] ¿CTAs: 1 primario claro por vista? ¿foco visible?
- [ ] ¿Sin promesas de retorno / lenguaje de hype ("maximiza tu alpha", escasez fake)?

## E. Navegación (BLOQUEANTE)
- [ ] ¿Todos los links del navbar/hub llegan a una vista viva (no 404/500/timeout)?
- [ ] ¿Deny-by-default: anónimo redirige a /login; API 401; rol sin permiso → bounce?
- [ ] ¿Login de cada rol aterriza donde corresponde?

## F. Performance percibida (MENOR)
- [ ] ¿Primer paint < ~3s en vistas clave? ¿landing sin gráficos vivos bloqueantes?

## Registro de iteraciones
| Iter | Capturas | Bloqueantes | Mayores | Menores | Estado |
|---|---|---|---|---|---|
| 0 (manual, hoy) | 18 | 9 | 3 | 2 | ✅ todos corregidos |
| 1 | — | — | — | — | en curso |
