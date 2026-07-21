---
kind: audit
status: PLANNED
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11_2025.json
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - config/execution/smart_simple_v1.yaml
---

# HANDOFF → Codex: diagnóstico P&L USD/COP — revisar adversarialmente y mejorar

> De: Claude · Para: Codex (revisor independiente) · 2026-07-21
> Contexto completo en el registry (secciones "EVENTO DE REVISIÓN DE DATOS" y
> "DIAGNÓSTICO DE INGENIERÍA INVERSA", N=59). Datos: trades publicados del re-run
> sobre el seed reparado (+57 días, 31 cierres corregidos).

## Los hallazgos a refutar (números exactos)

1. **Descomposición OOS-2025 reparado** (+13.05%, p=0.1151, 34 trades, 2L/32S):
   TP 19/+30.26pp · HS 5/−17.50pp (cada uno −3.5% exacto) · week_end 10/+0.09pp.
2. **HS = sizing, no mala suerte**: los 5 hard stops con leverage 2.00 (máximo);
   TP promedia 1.70; week_end 1.24. Tesis: el vol-targeter estaba en máxima agresión
   en las semanas que gapearon (em-fx §2 "ceiling, not recommendation").
3. **Celda única pre-declarada cap-1.5** (trial pagado, N=59): +12.85→+11.51pp,
   maxDD −7.74→−5.21%, Calmar-proxy 1.66→2.21. Candidata v12 PLANNED, juez=forward.
4. **Carry no cobrado**: IBR 8.80% − (prime−3)≈4.37% = 4.43pp; 103 días-posición cortos
   × lev 1.67 → +2.01pp/año teórico (100% pass-through), +1.00pp (50%).
5. **Timing intra-sesión M5 descartado**: media +0.7bps vs open de sesión, ±1pp ruido.

## Lo que te pido refutar/mejorar (en orden)

- **R1 — La muestra del hallazgo HS es N=5.** Calcula bajo el nulo: si K de los 34 trades
  tenían lev=2.0, ¿cuál es P(los 5 HS caigan todos en lev-max por azar)? (hipergeométrica).
  Si no es concluyente, dilo — la candidata v12 se sostiene igual por el prior em-fx,
  pero el claim "confirmación" debería rebajarse a "consistente".
- **R2 — Verifica la aritmética del cap** con COMPOSICIÓN (yo usé suma simple de pnl_pct y
  un DD por racha de trades; el equity real compone). ¿Cambia el signo del veredicto Calmar?
- **R3 — El carry**: revisa day-count (yo usé ACT/365 sobre días calendario de posición),
  la aproximación FFR=prime−3, triple-swap de miércoles/viernes según convención FX, y si
  el diferencial IBR−FFR es el proxy correcto para el swap de un CFD USDCOP retail
  (em-fx prefiere forward-implícito). Da tu banda honesta de pp/año.
- **R4 — ¿Qué se me escapó en la mecánica de salidas?** p.ej.: ¿los 10 week_end comparten
  algo observable t−1 (vol, confianza del modelo, día de entrada)? Solo DESCRIPTIVO —
  cualquier regla nueva = otra hipótesis pre-registrada, no la corras.
- **R5 — Diseña el arranque de v12 en paper** paralelo a v11 sin contaminar su Corte A
  (2026-09-16): mismo feed, mismo ledger, strategy_id nuevo, protocolo propio.

## Reglas de la casa (recordatorio)

Cero grid sobre OOS; 2025 está doblemente contaminado — cualquier cifra 2025 es contexto,
no juez; cada variante mirada = +1 trial en el registry (N hoy = 59); v11 FROZEN.
Deja tu respuesta en `.claude/codex/` como siempre.
