# Rule: Quant Constitution (TRANSVERSAL — todos los tracks)

> **SSOT de disciplina anti-selección para TODOS los activos y tracks** (USD/COP, XAU/USD,
> BTC/USDT, RL, futuros). Promovida desde la constitución BTC
> (`../specs/assets/btcusdt/design/constitution-modeling.md`) el 2026-07-06 tras la auditoría
> que encontró que el track en producción (COP H5) violó en la práctica las reglas que solo
> gobernaban BTC (grid search sobre el OOS 2025 en FC-H5-SIMPLE-001/FC-SIZE-001).
> Ante conflicto entre una spec, el código o una opinión, **gana esta regla**. Cambiarla
> requiere un ADR.
>
> Contract: CTR-QUANT-CONSTITUTION-001 · Version: 1.0.0 · Date: 2026-07-06

---

## 1. Cero magia numérica (la regla que COP violó)

- **Ningún parámetro se obtiene por grid search sobre el test/OOS.** Todos los parámetros son
  **priors económicos declarados ex-ante**. Las sensibilidades se **reportan completas**
  (cada celda = 1 trial en el DSR); **elegir la mejor celda está prohibido** — eso es empezar
  la v-siguiente con el test contaminado.
- **Un diagnóstico sobre el OOS solo genera hipótesis para el período SIGUIENTE**, jamás
  cambios que se evalúan en el mismo período. Si miraste 2025 para decidir un cambio, 2025 ya
  no es evidencia de ese cambio; el juez es el forward.

## 2. Registro de trials + Deflated Sharpe (obligatorio para claims)

- **Cada versión, cada grid, cada gate mirado = 1 trial.** Se registra en el
  `HYPOTHESIS-REGISTRY` del activo (BTC: `specs/assets/btcusdt/design/HYPOTHESIS-REGISTRY.md`;
  COP: `specs/assets/usdcop/HYPOTHESIS-REGISTRY.md`).
- **Ningún claim de edge sin Deflated Sharpe** (`services/common/metrics.py::deflated_sharpe_ratio`)
  recomputado con el conteo de trials actualizado. Bar por defecto: **DSR > 0.95**; cambiarlo
  requiere ADR. p<0.05 entre varios intentos es fácil; el DSR es el bar real.
- Métrica primaria de graduación: **Calmar** (y Sortino). Sharpe es secundario.

## 3. Baselines obligatorios (sin ellos no hay PROMOTE)

1. **B1** — buy&hold / exposición 1× del activo.
2. **B1′ — exposición emparejada:** "exposición constante = exposición promedio realizada de
   la estrategia". Separa timing genuino de simplemente tener menos beta.
3. **Baseline tonto del track** (p.ej. COP: siempre-short 1× con la misma mecánica de salidas).
   Si la estrategia no lo bate, **el baseline ES la estrategia** (honest gate, ya ejecutado
   con el RL — se aplica igual a la estrella).
4. **Stress de costos ×1/×2/×3:** si muere con costos al doble ⇒ REJECT.

## 4. Anti-look-ahead en tres capas

| Capa | Fuga típica | Defensa |
|---|---|---|
| Datos | ffill, normalización global, disponibilidad futura | macro `shift(1)`/T−1; `merge_asof(backward)`; `published_at ≤ bar` |
| Modelos | re-ajustar y re-etiquetar el pasado | fit congelado walk-forward; nunca re-etiquetar |
| Clasificadores/LLM | el corpus ya sabe cómo terminó la historia | recall histórico = cota superior; test insesgado solo post-cutoff |

## 5. Protocolo de retiro pre-firmado

- **Ningún track opera en producción sin protocolo de retiro firmado ex-ante** (semanas de
  evaluación, Sharpe/Calmar mínimo, DD máximo). Sus umbrales **no se relajan en drawdown**.
- COP: `../specs/assets/usdcop/WITHDRAWAL-PROTOCOL.md` · BTC: SPEC-12.

## 6. Desconfianza de la magia

- Sharpe > 4–5, DD < 1%, retornos de 3-4 cifras ⇒ look-ahead o costos ignorados hasta
  demostrar lo contrario.
- Métricas sin sentido estadístico no se reportan con solemnidad: con **N < 20 trades** se
  reporta solo conteo y PnL (nada de "Sharpe 19, p=0.000, 3 trades").

## 7. Decisiones sobre números publicados

- El **Vote 2 humano se emite sobre los números del bundle publicado** (`summary_*.json`),
  nunca sobre métricas recomputadas por el frontend. La UI puede animar/preview, pero los
  KPIs de decisión vienen del bundle.

## DO NOT

- Do NOT grid-search sobre el test/OOS — priors ex-ante, sensibilidades reportadas completas.
- Do NOT evaluar en el mismo OOS que motivó el cambio — el juez es el período siguiente.
- Do NOT reclamar edge sin DSR trial-aware recomputado (bar 0.95 salvo ADR).
- Do NOT promover sin B1′ + baseline tonto + stress de costos ×2.
- Do NOT reportar Sharpe/p-value con N<20 trades.
- Do NOT relajar umbrales de retiro estando en drawdown.
- Do NOT dejar que un LLM toque la cuenta — clasifica/resume; las reglas deciden.
