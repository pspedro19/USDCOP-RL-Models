---
kind: as-built
status: IMPLEMENTED
contract: CTR-WITHDRAWAL-BTC-001
version: 2.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/strategy_manifests/btcusdt.yaml
  - src/btc_strategy/strategies.py
---
# Protocolo de Retiro — btc_hodl_b1 (producción-paper) — PARA FIRMA DEL OPERADOR

> **Regla de la constitución (§5):** ningún track opera en producción sin protocolo de retiro
> firmado **ex-ante**. Los umbrales de abajo **no se relajan en drawdown** (ADR-0012).
> Cambiarlos después requiere ADR + reinicio del reloj.
>
> Contract: CTR-WITHDRAWAL-BTC-001 · **v2.0.0 (2026-07-21): reescrito para la CAMPEONA
> `btc_hodl_b1`** — la v1 estaba redactada para `btc_trend_b2`, destronada por OOS-2025
> (auditoría multi-agente + plan consolidado G0.1). La v1 nunca se firmó; ningún reloj corría.
> Estrategia: `btc_hodl_b1` (manifest hash `3b9b3c6dd1b9a70d`, CONGELADA, retrain NUNCA)
> Modo: **producción-PAPER** (PreTradeGate simula; cero dinero real)
> Metodología: diseño ≤ dic-2024 · OOS = todo 2025 · forward 2026+ = juez

## 1. Evidencia al momento de la firma (por qué entra a paper)

| Métrica | Valor | Fuente |
|---|---|---|
| OOS-2025 retorno neto | **+4.70%** (toda variante activa: ≈ −1.37%) | manifest `frozen` |
| Mecánica | HODL spot × vol-target 0.30/rv20, clip [0.30, 1.0], shift(1) | `strategies.py` |
| Claim de alfa | **NINGUNO** — es beta gestionada; DSR no aplica como claim | constitución §6 |
| Por qué campeona | El baseline brutal ganó: momentum b2 (DSR 0.9987 full-history) hizo −1.4% en OOS-2025 | registry BTC |

**Tesis del paper**: no hay señal que probar — hay una MECÁNICA de exposición que verificar
en vivo (tracking, costos, fills). El forward decide si la beta gestionada entrega el perfil
Calmar/DD del diseño, neto de costos reales.

## 2. Ventana de evaluación (elegir UNA y firmar)

- [ ] 16 semanas
- [ ] **26 semanas** (recomendado: medio año cubre ≥1 rotación de régimen cripto)

Inicio del reloj: fecha del **primer snapshot paper válido** publicado (no la fecha de este
documento). El reloj NO se reinicia por pausas del operador; solo por ADR.

## 3. Criterios de graduación a conversación-de-dinero-real (TODOS, al cierre)

1. **Calmar forward ≥ 0.75.**
2. **Max drawdown ≤ 15%** en la ventana.
3. **Costos ×2 positivos** (el retorno sobrevive con el doble de fees/slippage medidos).
4. **Tracking**: |paper − replay| de retorno semanal < 2pp promedio, mismo `strategy_id`/hash.
5. **≥ 30 oportunidades** de rebalanceo efectivo y más de un régimen de vol observado.
6. Cero violaciones de PreTradeGate / kill-switch.

Con N < 20 trades solo se reportan conteo y PnL (constitución §6); los días flat no son trades.

## 4. Criterios de RETIRO INMEDIATO (cualquiera, en cualquier momento)

1. **Drawdown > 20%** desde el inicio del paper → flat + post-mortem antes de reanudar.
2. **8 semanas consecutivas** con Calmar acumulado < 0 → revisión formal sin retune.
3. Error de custodia/venue, cambio regulatorio estructural del exchange → pausa + ADR.
4. Bug de datos/ejecución que invalide la serie → reinicio del reloj tras el fix (las filas
   forward no se reescriben; se marcan inválidas con motivo).

## 5. Reglas operativas

- **Spot-only, exposición ∈ [0.30, 1.0]**, sin apalancamiento, 24/7, reloj **365**.
- Señal diaria al cierre UTC; ejecución paper al open siguiente con fee/slippage del manifest
  (13 bps) hasta sustituirlos por medición del venue.
- `btc_hodl_b1` queda **CONGELADA** durante toda la ventana: cambiar estimador de vol, clips
  o intent = nueva versión = nuevo protocolo + nuevo reloj (H-VOL-02 del plan crea candidata
  aparte; NO toca este track).
- El publisher/normalizador mantiene UN solo campeón visible (`normalize_champions.py --check`).
- Ledger semanal: `paper_week_pct` derivado de snapshots YTD consecutivos + replay comparable;
  missing se registra como missing, jamás como 0.0.

## 6. Firma

```
Operador: ________________________        Fecha: ______________
Ventana elegida: [ ] 16 semanas   [ ] 26 semanas   (marcar una — recomendada: 26)
```

> Al firmar, este archivo se congela: ediciones posteriores solo por ADR referenciado aquí.
