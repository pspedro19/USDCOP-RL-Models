# WITHDRAWAL-PROTOCOL — USD/COP `smart_simple_v11` (firmado ex-ante)

> **Firmado 2026-07-06 (G3 del plan maestro).** BTC lo tenía constitucional (SPEC-12); el
> track en producción no. Este protocolo fija ANTES del veredicto los criterios de
> éxito/fracaso del forward 2026 — el único juez limpio de v11 tras la contaminación del
> OOS 2025 (ver `HYPOTHESIS-REGISTRY.md` §2: DSR < 0.95 en todos los escenarios).
> **Los umbrales NO se relajan mientras el sistema esté en drawdown**
> (`.claude/rules/quant-constitution.md` §5).

---

## 1. Congelamiento (efectivo YA)

- **`smart_simple_v11` v2.0 queda CONGELADA**: `config/execution/smart_simple_v1.yaml` no se
  toca (parámetros de gate, TP/HS, sizing, leverage). El retraining semanal del **modelo**
  (expanding window, L3) NO es un cambio de estrategia y continúa.
- Un diagnóstico sobre 2025/2026-pasado solo genera **hipótesis para el período siguiente**
  (registradas en `HYPOTHESIS-REGISTRY.md`), jamás cambios evaluados en el mismo período.
- Cualquier cambio de parámetro = **nueva versión** (v2.1+), nuevo trial en el registro, y
  su evaluación empieza en forward desde su fecha de congelamiento.

## 2. Ventana de evaluación

- **Inicio del reloj:** 2026-03-18 (congelamiento v2.0). Evaluación formal en dos cortes:
  - **Corte A — 26 semanas** (≈ 2026-09-16): evaluación intermedia.
  - **Corte B — 52 semanas** (≈ 2027-03-17): veredicto de graduación.
- Con **N < 20 trades** en el corte, solo se reporta conteo y PnL (sin Sharpe/p — regla §6
  de la constitución). El regime gate puede legítimamente producir pocos trades: pocas
  operaciones NO es fracaso por sí mismo.

## 3. Criterios de RETIRO (cualquiera dispara; pre-firmados)

| # | Condición | Umbral | Acción |
|---|---|---|---|
| W1 | Drawdown de equity forward desde inicio | **> 12%** | RETIRO inmediato (backtest MaxDD fue ~3.8%; 12% ≈ 3× ese daño) |
| W2 | Pérdida acumulada forward en el corte A (26 sem) | **< −8%** | RETIRO |
| W3 | Hard stops consecutivos | **3 seguidos** | PAUSA + revisión (no re-tunear: decidir retiro o continuar tal cual) |
| W4 | Circuit breaker del sistema activado | 2 veces en 8 semanas | PAUSA + revisión |
| W5 | Evidencia de fallo del gate: opera ≥3 semanas seguidas con Hurst en zona bloqueante (bug) | — | PAUSA técnica (bug ≠ estrategia) |
| W6 | H-COP-V11-01 concluye que NULL-A ≥ v11 en Calmar (2025+forward) | IC 95% no favorece a v11 | **NULL-A ES la estrategia**: retirar la capa ML, conservar mecánica |

## 4. Criterios de GRADUACIÓN (corte B, 52 semanas)

- Retorno forward positivo **y** Calmar forward > 1.0 **y** MaxDD < 12% **y** N ≥ 20 trades
  para que las métricas cuenten.
- Si gradúa: v11 gana estatus de evidencia limpia (1 trial forward, sin deflación).
- Si NO gradúa pero tampoco dispara retiro (p.ej. flat con pocos trades): se extiende 26
  semanas más UNA sola vez; a la segunda extensión sin veredicto, RETIRO por agotamiento.

## 5. Riesgo declarado (con la misma prominencia que el Sharpe)

- v11 es, en parte estructural, una **apuesta corta sobre COP validada en años bajistas**
  (B&H 2025 = −12%). El modo de fallo es una depreciación violenta del peso (2020-03,
  2021-22: −15/−30% en semanas contra el corto). STRESS-2122 (OLA 4) mide ese daño; su
  resultado NO cambia este protocolo, solo informa sizing futuro.
- El carry implícito del corto no está descompuesto (H-COP-CARRY-01); parte del retorno
  puede ser carry, no skill.

## 6. Gobernanza

- Cambiar cualquier umbral de este protocolo requiere **ADR** y NO puede ocurrir con el
  sistema en drawdown.
- El veredicto de cada corte se registra en `HYPOTHESIS-REGISTRY.md` y en
  `.claude/experiments/EXPERIMENT_LOG.md`.
