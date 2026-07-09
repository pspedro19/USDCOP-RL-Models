# Protocolo de Retiro — btc_trend_b2 (producción-paper) — PARA FIRMA DEL OPERADOR

> **Regla de la constitución (§5):** ningún track opera en producción sin protocolo de retiro
> firmado **ex-ante** — antes de ver un solo dato forward. Los umbrales de abajo **no se
> relajan en drawdown** (ADR-0012). Cambiarlos después requiere ADR + reinicio del reloj.
>
> Contract: CTR-WITHDRAWAL-BTC-001 · Estrategia: `btc_trend_b2` v1.2.1 (CONGELADA)
> Modo: **producción-PAPER** (PreTradeGate simula; cero dinero real)
> Metodología: entrenado/diseñado ≤ dic-2024 · OOS = todo 2025 · forward 2026+ = juez

## 1. Evidencia al momento de la firma (por qué entra a paper)

| Métrica | Historia completa (2018→2026) | OOS-2025 verdadero |
|---|---|---|
| Calmar | **1.83** | −0.16 |
| Sharpe | 1.40 | −0.05 |
| DSR (trial-aware) | **0.9987 ✓** (único >0.95 del sistema) | PSR 0.48 |
| vs B1′ (exposición emparejada, Calmar) | **1.83 vs 0.36 ✓** | — |
| Stress de costos ×2 / ×3 (Calmar) | **1.62 / 1.36 ✓** | — |
| p-value | 0.0 | 0.62 (IC95 [−19.9, +18.8] — no concluyente) |

**Tesis del paper**: el edge de tendencia de 8 años es real tras deflación, pero 2025 fue plano.
El forward paper decide si 2025 fue whipsaw pasajero o el fin del edge. **Nada de dinero real
hasta cumplir §3.**

## 2. Ventana de evaluación (elegir UNA y firmar)

- [ ] **16 semanas** (mínimo estadístico para una estrategia diaria; ~112 obs)
- [ ] **26 semanas** (recomendado: medio año cubre ≥1 rotación de régimen cripto)

Inicio del reloj: fecha del primer signal publicado en `/production` (paper).
El reloj NO se reinicia por pausas del operador; solo por ADR.

## 3. Criterios de graduación a conversación-de-dinero-real (TODOS, al cierre de la ventana)

1. **Calmar forward ≥ 0.75** (≈40% del histórico — exigente pero alcanzable si el edge vive).
2. **Sharpe forward > 0.5.**
3. **Max drawdown ≤ 15%** en la ventana (histórico DD implícito del Calmar 1.83 ≈ tolerable).
4. **Tracking**: divergencia |paper − backtest-replay| de retorno semanal < 2pp promedio
   (valida que la ejecución paper reproduce la mecánica del backtest).
5. Cero violaciones de PreTradeGate / kill-switch en la ventana.

## 4. Criterios de RETIRO INMEDIATO (cualquiera, en cualquier momento)

1. **Drawdown > 20%** desde el inicio del paper → flat + post-mortem antes de reanudar.
2. **8 semanas consecutivas** con Calmar acumulado < 0 → retiro (el edge no está).
3. Cambio estructural declarado del mercado (p.ej. delisting spot, cambio de régimen
   regulatorio de exchange) → pausa + ADR.
4. Cualquier bug de datos/ejecución que invalide la serie → reinicio del reloj tras el fix.

## 5. Reglas operativas

- **Spot-only, exposición ∈ [0,1]**, sin apalancamiento, sin forced-close (24/7, √365).
- Señal diaria al cierre UTC 00:00; ejecución paper simulada al open siguiente (+1 bps slip).
- La versión v1.2.1 queda **CONGELADA** durante toda la ventana: cualquier cambio de
  parámetros/función = nueva versión = nuevo protocolo + nuevo reloj.
- Publicación: bundle registry + `/production` selector (paper badge); fan-out a tenants
  solo PENDING-paper.
- Reporte semanal automático: L6 monitor (cuando se cablee el DAG BTC L6-forward) o manual
  `python scripts/pipeline/run_btc_pipeline.py --phase production`.

## 6. Firma

```
Operador: ________________________        Fecha: ______________
Ventana elegida: [ ] 16 semanas   [ ] 26 semanas   (marcar una — recomendada: 26)
```

> Al firmar (reemplazar la línea de arriba con nombre + fecha + ventana), este archivo se
> congela: ediciones posteriores solo por ADR referenciado aquí.
