---
kind: roadmap
status: PLANNED
contract: CTR-QUANT-CONSTITUTION-001
version: 2.0.0
date: 2026-07-22
last_verified: 2026-07-22
supersedes: []
code_anchors:
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - .claude/specs/assets/usdcop/BOOK-LEVERAGE-GOVERNOR.md
  - config/book/book_v1.yaml
---

# Plan siguiente fase v2 (post-validación Codex 2026-07-22)

> v1 fue RECHAZADO por Codex (`codex exec -p audit`, 12 ajustes obligatorios —
> `.claude/evidence/swarm_codex_reviews/2026-07-22/codex_validate_plan.txt`). Esta v2
> incorpora los 12. Bloqueadores técnicos ya ejecutados el mismo día: gate semana-15
> neutralizado en los 3 YAML + monitor L6 (`gates.enabled:false` ⇒ solo integridad;
> el juez sellado es la única autoridad) y `cop_entry_compare.py` re-congelado con
> costo incremental pre-firmado (0.5 bp) y alcance = estudio de PRECIO (el PASS
> económico exige shadow forward).
> Estado de partida: N=73 · v11 FROZEN · v12/v14 rumbo a paper 2026-07-27 ·
> v13 APRUEBA diseño · libro/gobernador/stress listos · H-ENTRY-01 pre-registrado.

## META — waterfall A NIVEL LIBRO (el 20-30% es objetivo ASPIRACIONAL, no aritmética)

La v1 mezclaba escalas (sumaba palancas de COP al 100% y multiplicaba el libro entero).
Corrección: toda palanca de un sleeve entra al libro **ponderada por su peso escalado**
(book_v1: COP 0.5787 · XAU 0.5275 · BTC 0.2699, bruto 1.376).

| Componente | A nivel del track COP | × peso COP → a nivel LIBRO | Estado |
|---|---|---|---|
| Señal COP (forward) | +3.36% YTD (≈ **+3.12%** con carry bidireccional devengado — registry) | por medir en el ledger del libro | forward real |
| Sleeves XAU/BTC | sin claim (juez = sus protocolos) | por medir | forward |
| Colateral remunerado | +4.3-4.5 pp (estimación del track, NO neto) | ~+2.5-2.6 pp, **pendiente de**: yield neto del venue, haircuts, impuestos, geometría de margen | FASE 0.1 |
| Carry medido | +1.0-1.9 pp (signo por confirmar) | ~+0.6-1.1 pp | FASE 2.1 |
| Multiplicador Kelly | — | **realista 1.17×** (0.25×f*_shrunk 4.69); 1.5× es el CAP, no la proyección; IC del Kelly incluye cero | FASE 4 |

**Lectura honesta**: los sumandos cuantificables hoy dan señal-del-libro (desconocida,
forward) + ~3.1-3.7 pp deterministas ponderados, ×~1.17 solo si gradúa. **No existe
puente cuantificado al 20-30%** — se mantiene como aspiración que exigiría: forward que
gradúe + palancas capturadas + breadth adicional (FASE 4 cross-asset) demostrada.

### Ruta v13 (mejor diseño: −4.42% / DD 10.0 / Calmar −0.149, HS −2)

Constitucional (diseño ≤2024, juez = forward desde freeze; 2025 no se corre por
disciplina). Ruta: decisión 0.4 → freeze (SSOT + manifest + strategy_id) → paper →
**reloj y comparador propios definidos en FASE 3** (la v1 los omitía).

## FASE 0 — Decisiones del OPERADOR (re-ordenadas: venue PRIMERO)

| # | Decisión | Nota (ajustes Codex) | Costo |
|---|---|---|---|
| 0.1 | **Venue** (colateral remunerado + cuenta real de operación) | va PRIMERO: swaps, costos TWAP y colateral se miden en el venue que se va a operar | 0 trials |
| 0.2 | Acumular **≥20 accruals válidos provenientes de statements** (no "20 statements") en ese venue | gate pre-firmado H-COP-CARRY-00; abrir su resultado = +1 trial (FASE 2.1) | 0 al acumular |
| 0.3 | Firmar `BOOK-LEVERAGE-GOVERNOR.md` (propuesta de registry ya corregida a N=73 + aplicabilidad v13) | los 3 protocolos de retiro (el de BTC es de `btc_hodl_b1`; **SPX NO es sleeve de book_v1**) deben quedar inequívocamente `AWAITING_SIGNATURE` o `SIGNED` en su front-matter | 0 |
| 0.4 | ¿Congelar v13 para paper? | su reloj/corte se fija AL FREEZE (FASE 3) | 0 (trial ya pagado) |
| 0.5 | ¿Ejecutar H-ENTRY-01? | instrumento ya re-congelado (costo 0.5 bp pre-firmado, estudio de PRECIO); primera ejecución = **+1 trial (N 73→74)** | +1 al abrir |
| 0.6 | Coherencia de graduación: **el leverage del libro depende de la graduación del sleeve que EFECTIVAMENTE ocupa book_v1 (hoy v12)** — graduar v11 no autoriza nada sobre un sleeve v12 | regla escrita en el gobernador | 0 |

## FASE 1 — Semana del 2026-07-27 (Claude, operativo)

0. ~~Bloqueadores pre-paper~~ **HECHOS 2026-07-22**: gate semana-15 neutralizado
   (3 YAML + monitor, `sealed_judge_only`); `cop_entry_compare.py` re-congelado.
1. **Lunes 27**: verificación del arranque de paper v12/v14 (+v13 si 0.4=sí):
   `strategy_id` separados (migración 064), artefactos aislados, skill `weekly-verify`.
   El monitor semanal SOLO reporta integridad (verificar `gate_status=sealed_judge_only`).
2. **Ledger del libro** (0 trials SOLO si): registra integridad, semanas faltantes = NA
   (ITT del juez sellado), pesos ERC CONGELADOS de book_v1 — sin re-optimización, sin
   métricas de decisión.
3. Si 0.5=sí: ejecutar H-ENTRY-01 (una pasada, +1 trial, N→74). Aun con PASS del bar
   pre-firmado, es estudio de PRECIO: el gate económico vive en la medición shadow
   forward (ambas entradas registradas en paper) y el cambio de producción exige
   confirmación forward + Vote 2.

## FASE 2 — Agosto 2026

1. **H-COP-CARRY-00** cuando existan ≥20 accruals (0.2): **abrir el resultado = +1
   trial** (la constitución no regala gates pre-firmados: cada gate MIRADO = 1 trial).
   Antes de abrir: copiar LITERALMENTE al runner el gate del AMENDMENT del registry
   (mediana neta ≥50% de pass-through, manejo exacto del IC95, triple-swap miércoles,
   feriados) — sin parafrasear umbrales. Si pasa → carry devengado en el motor como
   **reconciliación contable** (re-medición): PROHIBIDO usar el PnL 2025 revisado para
   promover o modificar estrategia alguna.
2. **Estudio mensual (forwards BanRep)** — pre-registro en 4 etapas SEPARADAS, cada
   transición de N declarada ANTES de abrir:
   (i) gate predictivo ÚNICO (una señal, un tenor, una métrica — se fijan en el
   pre-registro; +1 trial al abrir diseño ≤2024);
   (ii) traducción económica CONGELADA (exposición, costos, baselines B1/B1′ mensuales
   — +1 trial al abrir);
   (iii) OOS-2025 un disparo (+1 trial);
   (iv) juez forward posterior (cada gate decisorio que se abra = +1).
   Nada se abre sin las 4 etapas selladas en el registry.
3. **Carry cross-asset (datos, 0 trials)**: ingesta PIT de las patas listadas por E1
   (SOFR, TIIE 28d, Selic/DI, curva GC, div yield SPX). 0 trials MIENTRAS no se miren
   resultados para elegir fuentes/ventanas/activos.

## FASE 3 — Jueces y cortes (fechas EXACTAS del registry; cada corte decisorio abierto = +1 trial)

| Config | Corte | Fecha | Regla |
|---|---|---|---|
| v11 | Corte A / Corte B | **2026-09-16 / 2027-03-17** | WITHDRAWAL-PROTOCOL; umbrales no se relajan en DD |
| v12 | corte-26 safety/futility | **2027-01-22** | sellado (registry §juez v12) |
| v12 | corte-52 confirmatorio ÚNICO | **2027-07-23** | ΔCalmar v12−v11 unilateral α (ver multiplicidad), block b=4; **requiere N_bind≥12 semanas cap-vinculantes — si no, INCONCLUSO y se EXTIENDE, no gradúa** |
| v14 | corte-26 / corte-52 | mismas fechas que v12 (freeze 2026-07-21) | mismo protocolo; comparador abajo |
| v13 | corte-26 / corte-52 | **se fijan al freeze** (si congela 2026-07-27: ≈2027-01-29 / ≈2027-07-30) | ídem |

**Comparador y multiplicidad (PROPUESTA a firmar ANTES del primer corte — la v1 lo
omitía y elegir "la mejor" ex-post sería un grid forward):**
- Comparador común pre-declarado: **cada candidata vs v11** (el test sellado de v12 ya
  es v12−v11; v13/v14 se declaran igual).
- Corrección de multiplicidad: **α = 0.05/3 (Bonferroni)** en los cortes-52 de las 3
  candidatas.
- Selección si pasa más de una: **orden pre-declarado de congelamiento (v12 → v14 →
  v13)** — la primera que pase su test corregido ocupa el slot; PROHIBIDO elegir por el
  forward observado.
- Semanal/mensual = descriptivo puro (0 trials); solo los cortes decisorios cuentan.
- **Graduación del libro**: la habilita únicamente la graduación de la config que ocupa
  el sleeve COP de book_v1 en ese momento.

## FASE 4 — Condicionales (solo si sus gates abren)

- **Leverage del libro**: multiplicador = `clip(0.25 × f*_shrunk_forward, 0, 1.5)` —
  **1.5× es el CAP**, la proyección realista con el f* actual es ~1.17×. Requiere:
  graduación del sleeve efectivo (FASE 3) + **N≥20 trade-semanas forward de la config
  graduada** (no semanas calendario) + firma del gobernador.
- **Libro cross-asset trend/carry/value**: su pre-registro debe declarar el número
  EXACTO de celdas, baselines, sensibilidades y regla de familia (nada de "3-5 trials
  estimados"). No se pre-registra hasta que FASE 2.3 esté completa.
- **Revisita fundacionales (Chronos)**: 2026 completo = DISEÑO del nuevo pre-registro;
  el juez limpio empieza en 2027 post-freeze. 2026 no puede ser confirmación si sus
  resultados influyen en el diseño. Requiere ADR + operador.

## Higiene (inventario CORREGIDO por Codex; no bloqueante)

- `.claude/codex/`: **36 de 71** markdown sin front-matter (no "~50") + ADR-0021 —
  reproducir el reporte exacto del test antes de limpiar; decisión del operador.
- `design-system/` está referenciado por docs del repo — revisar propiedad/consumidores
  antes de etiquetarlo basura. No hay `audit*.json` en la raíz actualmente.
- `Microsoft/` (ModuleAnalysisCache de PowerShell en el root) — artefacto accidental.
- Skill `webapp-testing`: shippea `scripts/with_server.py` sin tests ni `--verify`
  (`test_quant_library_gate`).

## Qué NO está en el plan (cerrado con evidencia, no reabrir)

Más modelos/transformers sobre 2025 · features direccionales semanales · leverage por
estrategia · LATAM TSMOM · grid sobre cualquier OOS · re-tocar el backtest 2025.
Ver registry N=73.
