---
kind: protocol
status: DRAFT_AWAITING_OPERATOR_SIGNATURE
contract: CTR-QUANT-CONSTITUTION-001
version: 0.1.0
date: 2026-07-22
last_verified: 2026-07-22
trials_consumed: 0
supersedes: []
code_anchors:
  - scripts/analysis/book_kelly_governor.py
  - .claude/evidence/book_kelly/2026-07-22/book_kelly.json
---
# BOOK-LEVERAGE-GOVERNOR — Gobernador de leverage fraccional-Kelly del LIBRO (pre-firmado)

> **TAREA A2 · 0 trials.** Este documento NO prueba una hipótesis: sella una regla de
> apalancamiento del libro ANTES de que el forward gradúe, para eliminar la tentación de
> ajustarla ex-post (quant-constitution §1 y §5). Es un BORRADOR que espera la firma del
> operador; hasta la firma, la regla vigente es implícitamente 1.0x (no hay leverage de
> libro en producción).
>
> **"Leverage del libro" = multiplicador GLOBAL sobre la configuración congelada**, por
> encima del leverage interno de la estrategia (v11 ya usa 0.5–2.0x por trade). f* aquí
> siempre es multiplicador de libro, jamás leverage absoluto de exchange.

---

## 0. Evidencia (única fuente válida: la serie honesta)

Datos: v11 2025 **purgado** (+7.35%, 32 trades, `trades/smart_simple_v11_2025.json`) +
2026 YTD forward (+3.36%, 11 trades, `cop_monitor_2025_2026/2026-07-21/monitor.json`).
Los números pre-reparación (+26.05/+13.05) están invalidados y el script generador
**aborta** si el bundle no coincide con la serie purgada. 2026 solo: N=11 < 20 ⇒ se cita
únicamente conteo y PnL. Serie combinada N=43 trade-semanas (≥20: los ratios se permiten).

Computado por `scripts/analysis/book_kelly_governor.py` (seed 42, B=10.000, block
bootstrap circular b=4 — el mismo prior del juez sellado de v12). Artefacto:
`.claude/evidence/book_kelly/2026-07-22/book_kelly.json` (+ `generator_script.py`).

| Cantidad | Valor |
|---|---|
| N (trade-semanas 2025+2026) | 43 |
| μ semanal / σ semanal | +0.255% / 1.649% |
| **f\* Kelly continuo** (μ/σ²) | **9.39** |
| **f\* Kelly discreto** (argmax E[log(1+f·r)] sobre la empírica) | **7.99** |
| IC95 de f\* (block bootstrap b=4) | **[−6.14, +48.71]** — **INCLUYE CERO** |
| IC95 de f\* (iid, referencia) | [−7.15, +50.62] |
| IC95 f\* discreto (block b=4) | [0.00, 21.79] |
| P(f\* ≤ 0) (block b=4) | **14.5%** |
| f\* con media real = mitad de la observada (shrinkage) | 4.69 (continuo) / 4.36 (discreto) |
| Banda 0.25–0.5 × Kelly (observado) | [2.35, 4.69] |
| **Banda 0.25–0.5 × Kelly (shrunk ×½)** | **[1.17, 2.35]** |

**Haircut de POLÍTICA ×½ (no un estimador de shrinkage)**: el recorte a la mitad es una
decisión conservadora declarada, NO un estimador formal derivado de Lo/Mertens. Lo que
Lo (2002, caso normal-iid) y Mertens (2002, extensión de la varianza con skew/kurtosis)
aportan es la MOTIVACIÓN: con N=43 el SE del Sharpe semanal (0.155) es 0.153 iid (Lo) y
0.168 con ajuste skew/kurtosis (Mertens; skew −1.27) — una media real igual a la MITAD
de la observada está a **0.46 SE**, estadísticamente indistinguible del dato, así que
operar al Kelly observado sería fingir una precisión que la serie no tiene (corrección
Codex verify A2 #1). Por eso ningún cálculo de este protocolo usa el Kelly sin
shrink: la fracción operable se toma de la banda shrunk, y aun así el IC95 del f\*
incluye cero — **esta serie no puede probar ni siquiera que el Kelly óptimo sea
positivo**. Además el tramo 2025 está contaminado por selección (DSR 0.72 < 0.95,
`HYPOTHESIS-REGISTRY.md` §2/σ_trials): motivo adicional para que el f\* de decisión se
recompute con datos forward, nunca con el backtest.

Distribución de maxDD forward simulada (52 trade-semanas densas — conservador: ~28
trades/año observados — block b=4, por multiplicador de libro):

| Mult. libro | maxDD p50 | maxDD p95 | P(DD>7%) | P(DD>10%) | P(DD>12%) |
|---|---|---|---|---|---|
| 0.5x | −4.2% | −8.0% | 9.3% | 1.2% | 0.3% |
| 1.0x | −8.4% | −15.6% | 67.0% | 33.1% | 17.5% |
| 1.5x | −12.5% | −22.6% | — | — | 55.4% |
| 2.0x | −16.5% | −29.7% | — | — | 82.7% |

(Simulación estática, SIN la escalera de §3; con la escalera activa la cola real es
menor porque el multiplicador cae antes de llegar a ella. Cifras exactas en el JSON.)

---

## 1. Regla (a) — Pre-graduación: leverage del libro = **1.0x**, sin excepciones

- Mientras el forward NO haya graduado, el libro opera a **1.0x** (la config congelada
  tal cual, con su leverage interno; ningún multiplicador global).
- **Criterio de graduación: el ya firmado — NO se duplica aquí.** Gradúa quien cumpla:
  - v11: Corte B de `WITHDRAWAL-PROTOCOL.md` §4 (52 semanas desde 2026-03-18; retorno,
    Calmar, MaxDD y N mínimos definidos ALLÍ; retiros §3 intactos).
  - v12/v14: el **juez sellado** del panel Codex (`HYPOTHESIS-REGISTRY.md`, "PANEL CODEX
    (4º auditor, 2026-07-21)"): reloj propio desde su freeze 2026-07-27, corte-52 único
    test confirmatorio, gates económicos conjuntos, N_bind ≥ 12.
- Si esos documentos cambian (vía ADR), este protocolo hereda el cambio sin editarse:
  la referencia es normativa, el número vive allí.

## 2. Regla (b) — Post-graduación: **leverage = min(1.5x, fracción-Kelly forward)**

- Al graduar una configuración, el leverage del libro puede subir a
  **clip(0.25 × f\*_shrunk_forward, 0, 1.5)** — con piso CERO explícito (corrección
  Codex verify A2 #2): si el Kelly forward sale ≤ 0 (posibilidad coherente con el IC
  actual), el multiplicador NO se vuelve negativo ni invierte el libro — la conducta
  operativa es **libro flat + revisión** (un Kelly forward ≤ 0 significa que el forward
  no evidencia edge; nunca se "compensa" con el backtest). Si 0 < resultado < 1.0, se
  opera a esa exposición reducida (el Kelly manda des-riesgar). Donde:
  - `f*_shrunk_forward` = ½ × (μ_fwd/σ²_fwd) computado **EXCLUSIVAMENTE con las
    trade-semanas del período forward** (post-freeze de la config graduada). El
    backtest 2025 queda prohibido en este cálculo para siempre (doblemente
    contaminado: selección + este mismo análisis lo miró).
  - Requiere **N_fwd ≥ 20 trade-semanas forward**; con menos, el libro permanece en
    1.0x aunque haya graduado (constitución §6: sin N no hay ratios).
  - La fracción es **0.25 de Kelly** (borde inferior de la banda 0.25–0.5). Subir a
    0.5×Kelly requiere ADR + ≥52 trade-semanas forward. Nunca más de 0.5×Kelly.
- **El cap 1.5x es techo duro** aunque el Kelly forward diga más. Contexto de por qué:
  con los datos actuales 0.25×Kelly sin shrink (2.35) lo excede, pero la celda
  S7-MOTOR (registry, trials 68→69) ya demostró que el leverage alto sobre esta
  familia amplifica la parte sin edge; y a 1.5x estático la simulación pone
  P(DD>12%) ≈ 55% en un año denso. 1.5x es lo máximo defendible.
- Recompute trimestral (calendario, no discrecional): cada recompute usa TODAS las
  trade-semanas forward acumuladas. Si el f\* recomputado cae por debajo del
  multiplicador vigente, se baja al nuevo valor en la siguiente semana operativa
  (bajar nunca requiere ADR; subir solo en el recompute trimestral).

## 3. Regla (c) — Presupuesto de DD del libro y escalera de des-apalancamiento — **PROPUESTA**

Presupuesto de drawdown del libro: **12%** (coherente con W1 del
`WITHDRAWAL-PROTOCOL.md`: a ese daño el sleeve único actual dispara retiro; el libro
no debe seguir apalancado camino a ese evento). DD medido sobre la equity del LIBRO,
peak-to-trough, marcada al cierre semanal (viernes 12:55 COT).

| Escalón (DD del libro desde el pico) | Multiplicador máximo | Derivación (PROPUESTA) |
|---|---|---|
| DD ≤ 7% | el de §1/§2 (1.0x o min(1.5x, Kelly)) | zona normal: 7.84% fue el maxDD observado 2025 a 1.0x; p50 del año denso simulado = 8.4% |
| **DD > 7%** | **1.0x** | a 1.5x la p50 simulada (12.5%) ya invade el presupuesto; volver a 1.0x corta la amplificación antes de eso |
| **DD > 10%** | **0.5x** | p(DD>10%) a 1.0x ≈ 33% denso: zona de cola; a 0.5x la P(DD>12%) residual ≈ 0.3% — frena la sangría casi con certeza |
| **DD > 12%** | **0 (flat) + revisión formal** | presupuesto agotado = umbral W1; la revisión decide retiro/continuación por los protocolos de cada sleeve, jamás re-tunea este documento en caliente |

- Nota: la tarea sugería 15% para el escalón flat; se propone **12%** porque es el
  número ya firmado en W1 — un libro apalancado no puede tener un presupuesto de DD
  más laxo que el retiro de su único sleeve. (Si el operador prefiere 15%, es una
  edición legítima ANTES de firmar; después, solo por ADR fuera de drawdown.)
- **Re-armado (histéresis)**: se recupera el escalón superior solo cuando la equity
  del libro cierra ≥ 4 semanas consecutivas con DD < (umbral del escalón − 2pp)
  (p.ej. vuelve de 1.0x→1.5x cuando DD < 5% durante 4 semanas). Sin re-armado
  intradía/intra-semana.
- La escalera gobierna el MULTIPLICADOR DE LIBRO; no toca el sizing interno de la
  estrategia (congelado) ni sustituye los criterios de retiro W1–W6, que siguen
  vigentes y dominan.

## 4. Regla (d) — Anti-relajación

- **Ningún umbral de este protocolo se relaja estando el libro en drawdown**
  (quant-constitution §5). Cambios solo por **ADR**, con el libro en DD < 7%, y
  nunca en la misma semana en que un escalón haya disparado.
- Endurecer (bajar leverage, bajar umbrales) está siempre permitido sin ADR.
- El multiplicador aplicado cada semana se registra en el ledger append-only junto
  al `strategy_id` y la semana ISO (mismo mecanismo del juez v12, punto 5).

## 5. Firma

- **Operador**: ____________________ Fecha: ____________
- Al firmar, el estado pasa a `SIGNED` y los números de §3 dejan de ser PROPUESTA.

---

## REGISTRY-PROPOSAL (evento 0-trials — texto listo para que el OPERADOR lo pegue en `HYPOTHESIS-REGISTRY.md`; este task NO edita el registro)

> ## GOBERNADOR KELLY DEL LIBRO PRE-FIRMADO (2026-07-22, 0 trials — N sigue en 72)
>
> Regla de capital sellada ANTES del veredicto forward (`BOOK-LEVERAGE-GOVERNOR.md`,
> borrador esperando firma). No es un trial: no se evaluó ninguna variante ni se
> seleccionó nada — se computó el guard-rail sobre la serie honesta ya pagada
> (2025 purgado +7.35%/32 tr + 2026 YTD +3.36%/11 tr; N=43 trade-semanas).
> Números (evidencia `.claude/evidence/book_kelly/2026-07-22/`): f\* continuo 9.39,
> discreto 7.99, **IC95 block-b4 [−6.14, +48.71] INCLUYE CERO** (P(f\*≤0)=14.5%);
> shrunk ×½ (Lo-Mertens, media/2 = 0.46 SE): 4.69; banda 0.25–0.5×Kelly shrunk
> [1.17, 2.35]. Regla: 1.0x hasta graduación (criterios de WITHDRAWAL-PROTOCOL §4 /
> juez v12 sellado, por referencia); después min(1.5x, 0.25×f\*_shrunk **forward-only**,
> N_fwd≥20); escalera DD 7%→1.0x / 10%→0.5x / 12%→flat+revisión (PROPUESTA, alineada
> con W1); umbrales no se relajan en drawdown (constitución §5). Consistente con el
> Kelly-guard del auditor fin-math (riesgo actual 0.23×Kelly): no hay upside legal en
> subir tamaño hoy — la decisión de capital queda pre-firmada para DESPUÉS del forward.
