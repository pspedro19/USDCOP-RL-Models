---
kind: as-built
status: IMPLEMENTED
contract: CTR-WITHDRAWAL-SPX-001
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/strategy_manifests/spx500.yaml
  - src/strategies/spx500_regime_gated_v1/policies.py
---
# Protocolo de Retiro — SPX500 (paper next-open) — PARA FIRMA DEL OPERADOR

> Constitución §5 + plan consolidado G0.2/S4. **Este protocolo cubre DOS bundles congelados
> a la misma fecha** para que el forward decida H-SIMP-SPX-02 sin looks adicionales:
> `spx500_regime_gated_v1` (campeón) y `spx500_ma200_b0` (baseline causal corregido —
> erratum 2026-07-21: el Calmar 1.641 del tonto era look-ahead; corregido = 0.5393, el
> gated 0.6482 sí lo bate en histórico; el forward es el juez).
>
> Research = SPY total-return · Ejecución = instrumento TBD por operador (rol `execution`
> de `dim_asset_symbol`; NO se opera el índice). Paper hasta homologar.

## 1. Estado al momento de la firma

Investigación-solo: DSR 0.8813 < 0.95, `net_return_gt_b1` FAIL, drivers del gate en modo
`proxy_drivers` (vol realizada en lugar de VIX; sin NFCI/HY-OAS) hasta cerrar S1.
LIVE-2026 YTD del gated ≈ plano. Nada de capital real bajo este protocolo.

## 2. Ventana y arranque

**≥26 semanas** desde el primer fill paper next-open válido con el runner REAL (fail-closed,
sin `datagen`). Ambos bundles corren en paralelo con el mismo reloj.

## 3. Graduación (TODOS, por bundle)

1. Calmar forward > 0.75 · 2. MaxDD < 12% · 3. Costos ×2 del instrumento real positivos ·
4. Tracking < 2pp/sem · 5. ≥30 oportunidades y >1 régimen · 6. Drivers reales (VIX/NFCI
con vintage) o etiqueta `proxy_drivers` mantenida — un bundle proxy NO gradúa a dinero real.

**Regla H-SIMP-SPX-02 pre-firmada**: si al cierre `spx500_ma200_b0` ≥ gated en Calmar
forward con DD comparable, **el baseline ES la estrategia** — el gate se retira sin apelación.

## 4. RETIRO INMEDIATO

1. DD > 15% → flat + post-mortem. 2. 8 semanas Calmar acumulado < 0 → revisión sin retune.
3. Fallback sintético detectado en cualquier artefacto publicado → serie inválida, reinicio.
4. Ruptura del alias (mezcla SPY/SPX en una misma serie) → pausa + corrección + ADR.

## 5. Firma

```
Operador: ________________________        Fecha: ______________
Instrumento de ejecución elegido: ________________________
```
