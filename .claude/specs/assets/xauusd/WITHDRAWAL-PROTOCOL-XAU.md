---
kind: as-built
status: IMPLEMENTED
contract: CTR-WITHDRAWAL-XAU-001
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/strategy_manifests/xauusd.yaml
  - src/gold_rl/strategies.py
---
# Protocolo de Retiro — gold_trend_simple (paper-shadow MT5) — PARA FIRMA DEL OPERADOR

> Constitución §5: ningún track opera sin protocolo firmado **ex-ante**. Umbrales NO se
> relajan en drawdown. Creado por el plan consolidado G0.2 (2026-07-21): Oro no tenía
> protocolo y su LIVE-2026 ya es negativo (Calmar −0.217) — esta es la puerta que dice
> cuándo se retira la pata, decidida ANTES de mirar más forward.
>
> Estrategia: `gold_trend_simple` (manifest hash `4fd3574fbfbe615f`, CONGELADA, retrain NUNCA)
> Instrumento: XAUUSD spot/CFD vía MT5 (SPEC-11) — broker/contrato se anexan al homologar.

## 1. Evidencia al momento de la firma

| Métrica | Valor |
|---|---|
| Mecánica | voto ≥2 de {SMA63,126,252} long/flat × vol-target 0.10/rv20 clip[0.06,1.5], t−1 |
| OOS-2025 | Calmar 10.75 — **observado durante el desarrollo; NO es expectativa forward** |
| LIVE-2026 | Calmar **−0.217** |
| DSR (N=75) | 0.0033 — **sin claim de alfa**; se opera como beta de tendencia gestionada |
| Costo estructural | swap long ~2.5%/año supuesto; ×2 costos: Calmar full 0.242 → 0.142 |

## 2. Ventana y arranque

- Ventana mínima: **26 semanas** desde el **primer fill paper válido en MT5** (no desde hoy).
- Pausas del broker/feed se registran; el reloj solo se reinicia por ADR.

## 3. Graduación a conversación-de-dinero-real (TODOS)

1. Calmar forward > **0.75** · 2. MaxDD < **12%** · 3. Costos ×2 positivos con **swap real
medido** (no el supuesto 2.5%) · 4. Tracking |paper − replay| < 2pp promedio semanal ·
5. ≥ 30 oportunidades y más de un régimen · 6. Cero violaciones de gate.

## 4. RETIRO INMEDIATO (cualquiera)

1. DD > **15%** desde inicio del paper → flat + post-mortem.
2. 8 semanas consecutivas con Calmar acumulado < 0 → revisión formal, **sin retune**.
3. Divergencia de ejecución inexplicable (>2pp/sem sostenida) o violación de datos → pausa.
4. H-ROBUST-DECADES-XAU en FAIL (mayoría de décadas pre-2004 con Calmar negativo o
   pierde vs B1′) → el campeón se degrada a contexto y el live queda BLOQUEADO.

## 5. Reglas operativas

- Solo daily decide; 1h/4h diagnósticos; M5 únicamente fills/slippage del broker homologado.
- Flat de fin de semana permitido solo si estaba en la mecánica congelada (no lo está →
  cualquier cambio = nueva versión + nuevo protocolo).
- Swap/fees/spread del statement real del broker alimentan el ledger; el supuesto del
  manifest se retira al tener 20 accruals medidos.

## 6. Firma

```
Operador: ________________________        Fecha: ______________
Broker/cuenta MT5: ________________________
```
