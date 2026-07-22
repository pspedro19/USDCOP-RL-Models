---
kind: roadmap
status: ACTIVE
contract: CTR-QUANT-CONSTITUTION-001
version: 1.0.0
date: 2026-07-22
last_verified: 2026-07-22
supersedes: []
code_anchors:
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - .claude/specs/assets/usdcop/BOOK-LEVERAGE-GOVERNOR.md
  - config/book/book_v1.yaml
---

# Plan siguiente fase (post-enjambre 2026-07-22)

> Estado de partida: N=73 trials · v11 FROZEN (+7.35% OOS-2025 / +3.36% YTD) · v12/v14
> congeladas rumbo a paper 2026-07-27 · v13 APRUEBA diseño (instrumento validado por
> Codex) · libro ERC + gobernador Kelly + stress MC listos (0 trials) · H-ENTRY-01
> pre-registrado sin ejecutar · forwards/remesas BanRep ingestados PIT.
> Regla transversal: nada de re-tocar 2025; el juez de todo es el forward.

## FASE 0 — Firmas y decisiones del OPERADOR (bloquean lo demás)

| # | Decisión | Desbloquea | Costo |
|---|---|---|---|
| 0.1 | Firmar `BOOK-LEVERAGE-GOVERNOR.md` (escalera 7/10/12 validada por stress MC) | operación del libro con reglas selladas | 0 trials |
| 0.2 | Firmar los 3 protocolos de retiro pendientes (BTC v2, XAU, SPX) | sleeves XAU/BTC en el libro con retiro pre-firmado | 0 |
| 0.3 | **Venue con colateral remunerado** (42.6% cash medio + margen ocioso) | +4.3-4.5 pp/año deterministas | 0 |
| 0.4 | Solicitar/acumular **≥20 statements de swap** del broker | gate H-COP-CARRY-00 → +1.0-1.9 pp/año | 0 (el estudio ya está pre-firmado) |
| 0.5 | ¿Congelar v13 para paper? (SSOT yaml + manifest propio + strategy_id migración) | tercera candidata al forward | 0 (el trial ya se pagó) |
| 0.6 | ¿Ejecutar H-ENTRY-01? (`cop_entry_compare.py --confirm-trial`) | posible mejora de bps por entrada TWAP | **+1 trial** |

## FASE 1 — Semana del 2026-07-27 (Claude, operativo)

1. **Lunes 27**: verificación del arranque de paper v12/v14 (+v13 si 0.5=sí):
   señales separadas por `strategy_id` (migración 064), artefactos aislados (A2),
   ledger semanal. Skill: `weekly-verify`.
2. **Ledger del libro** (0 trials): job semanal que consolida los retornos realizados de
   los 3 sleeves con pesos ERC de `book_v1.yaml` → una serie del libro en paper,
   registrada junto a las patas. Sin claims; alimenta al gobernador cuando se firme.
3. Si 0.6=sí: ejecutar H-ENTRY-01 sobre diseño 2020-24 (una pasada, +1 trial, N→74),
   registrar veredicto. Si PASA el bar (IC95 excluye 0 y supera costo incremental):
   montar medición shadow forward (ambas entradas registradas en paper) — el cambio de
   producción exige confirmación forward + Vote 2, nunca solo diseño.

## FASE 2 — Agosto 2026 (estudios que ya tienen datos)

1. **H-COP-CARRY-00** en cuanto existan ≥20 accruals (0.4): medir signo y pass-through
   del swap real vs forward-implícita BanRep (ya ingestada). Gate pre-firmado: IC95
   bootstrap ≥50% pass-through. Si pasa → carry devengado en el motor (flag, re-medición).
2. **Primer estudio de reloj MENSUAL** (pre-registro nuevo, +1 trial cuando se abra):
   devaluación implícita BanRep (2005→, ~250 obs) como predictor del retorno mensual
   siguiente — el único candidato direccional con N suficiente y prior económico
   (paridad cubierta). Diseño: expanding causal por published_at, bar = IC con IC95
   block-bootstrap vs cero Y baselines B1/B1′ mensuales; diseño ≤2024, un disparo 2025,
   juez forward. NO se abre sin pre-registro sellado.
3. **Carry cross-asset (datos, 0 trials)**: ingestar las patas que E1 dejó listadas
   (SOFR/Fed funds diaria, TIIE 28d, Selic/DI, curva GC o lease rates, div yield SPX)
   → llenar `carry_z` del harness. Solo ingesta PIT; ningún estudio.

## FASE 3 — Jueces y cortes (calendario ya sellado, solo cumplirlo)

| Fecha | Evento | Regla |
|---|---|---|
| Semanal (lun) | ledger v11/v12/v14(/v13) + libro | descriptivo, sin Sharpe hasta N≥20 |
| Mensual | corte descriptivo del forward | sin decisiones intra-corte |
| **2026-09-16** | **Corte A de v11** | WITHDRAWAL-PROTOCOL; umbrales no se relajan en DD |
| ~2027-01 (corte-26 v12) | safety/futility de v12 | protocolo sellado, reloj propio |
| 2027-03-17 | Corte B de v11 | ídem |
| ~2027-07 (corte-52) | **único test confirmatorio v12** (ΔCalmar α=0.05, b=4) | y graduación del libro/leverage si pasa |

## FASE 4 — Condicionales (solo si sus gates abren)

- **Leverage 1.5× del libro**: SOLO post-graduación corte-52 + Kelly forward (N≥20
  semanas forward) positivo; regla `clip(0.25·f*_fwd, 0, 1.5)` ya sellada.
- **Libro cross-asset trend/carry/value**: pre-registro de familia completa sobre el
  harness E1 cuando el carry esté poblado — presupuesto estimado +3-5 trials; se
  contabiliza por celda. No antes de que FASE 2.3 termine.
- **Revisita fundacionales (Chronos)**: solo con NUEVO pre-registro sobre 2026 completo
  juzgado en 2027 (ADR + operador). La puerta 2025 quedó cerrada (N=72, por 0.0004).

## Higiene pendiente (no bloqueante, decisión del operador)

- ~50 docs `.claude/codex/**` + ADR-0021 sin front-matter (otra sesión) → 47 fallos del
  test de knowledge; basura en repo root (`Microsoft/`, `audit*.json`, `design-system/`).
- Skill `webapp-testing` promovida sin tests (`test_quant_library_gate`).

## Qué NO está en el plan (cerrado con evidencia, no reabrir)

Más modelos/transformers sobre 2025 · features direccionales semanales · leverage por
estrategia · LATAM TSMOM · grid sobre cualquier OOS. Ver registry N=73.
