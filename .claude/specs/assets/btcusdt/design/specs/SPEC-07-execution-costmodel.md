# SPEC-07 — Ejecución & CostModel

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | SPEC-06 (`exposure_final`, `rebalance_flag`), `btc_price_1h` (vol de barra, ½spread) |
| Materializa | `strategy/execution.py` |
| Pre-registro | Guía v3 §3.2, §5.7, §7.3 |
| Propósito | Que "con costos" **signifique algo** numérico en TODO backtest |

## 1. Propósito y alcance

Especificar el modelo de costos (fees + spread + slippage + impacto), la definición canónica
de barra, las reglas de stitching de precio y la mecánica de órdenes. Sin este componente,
"backtest con costos" es una frase vacía. **El funding NO entra** (núcleo spot-only, ADR-0008).

## 2. Entradas (contrato)

| Entrada | Tipo | Unidad | Fuente |
|---|---|---|---|
| `order_size` | float | fracción de NAV | SPEC-06 (`|target − prev|`) |
| `bar_volume` | float | USD | `btc_price_1h`/`btc_daily` |
| `half_spread` | float | bps | estimado de `btc_price_1h` |
| `price_close` | float | USD | barra canónica UTC 00:00 |

## 3. Salidas (contrato)

| Salida | Tipo | Unidad | Descripción |
|---|---|---|---|
| `cost_bps_t` | float | bps | costo total del rebalanceo de ida |
| `fill_price` | float | USD | precio de ejecución simulado |
| `turnover_contribution` | float | — | para el presupuesto de turnover |

## 4. Interfaz (API)

```python
def cost_model(order_size: float, bar_volume: float, half_spread_bps: float = 1.0,
               fee_bps: float = 10.0, slippage_bps: float = 2.0,
               impact_k: float = ...) -> float:
    """cost = fee + half_spread + slippage + impact·(order/bar_volume if order>0.1% vol else 0). En bps."""

def canonical_bar(ts: datetime) -> datetime:
    """Normaliza a cierre UTC 00:00. Nunca mezcla con cierres de sesión US."""

def stitch_price(sources: dict[str, pd.Series]) -> pd.Series:
    """Jerarquía fija; valida discrepancia <0.5% en solapes (QUALITY_FLAG si no)."""
```

## 5. Algoritmo / lógica

```
# CostModel (§5.7) — todos priors conservadores para tamaño retail/boutique
fee        = 10 bps por lado (taker spot)
half_spread= ~1 bp (BTC/USDT; estimado de datos H1)
slippage   = 2 bps fijos + impact_k·(order_size/bar_volume)  SI order_size > 0.1% del vol
cost_total ≈ 13–15 bps por rebalanceo de ida

# Barra canónica (§3.2)
bar = cierre UTC 00:00  (declarado en SSOT; jamás cierres US)

# Stitching de precio (§3.2) — jerarquía fija
#   CryptoCompare (primaria) → Bitstamp (2013–2017) → Binance spot (2017-08+)
#   en cada empalme: registrar fecha de corte; discrepancia de cierre en solape < 0.5%
#   si ≥ 0.5% ⇒ QUALITY_FLAG, barra excluida del entrenamiento (no del histórico)
#   hueco de precio ⇒ cubrir con secundaria del MISMO timestamp, JAMÁS ffill

# Órdenes (§5.7)
#   límite pasivas con timeout → market residual
#   tamaño mínimo de ajuste: 2% NAV (menores se acumulan al siguiente cruce)
```

## 6. Invariantes y post-condiciones

- Todo rebalanceo del backtest paga `cost_bps` (no hay trades gratis).
- El precio nunca se rellena con ffill (regla de oro heredada).
- Empalmes con discrepancia ≥ 0.5 % ⇒ `QUALITY_FLAG` (test I9).
- El funding no aparece en `cost_model` (spot-only).

## 7. Tests unitarios

- [ ] `cost_model` con `order_size` pequeño (< 0.1 % vol) ⇒ sin término de impacto.
- [ ] `cost_model` con orden grande ⇒ impacto proporcional a `order/bar_volume`.
- [ ] `canonical_bar` mapea cualquier timestamp intradía a 00:00 UTC.
- [ ] `stitch_price` con solape discrepante > 0.5 % ⇒ marca `QUALITY_FLAG`.
- [ ] Hueco de precio ⇒ usa secundaria del mismo ts, nunca ffill.
- [ ] Tamaño < 2 % NAV ⇒ no ejecuta (se acumula).

## 8. Tests de integración

- [ ] Backtest de B1 con y sin CostModel: el CostModel reduce el retorno de forma coherente.
- [ ] Turnover anualizado del motor (SPEC-06) × `cost_bps` = drag reportado; contra
      presupuesto (< 25 % del exceso de Calmar sobre B1).
- [ ] Serie de precio stitched es continua, sin saltos > 0.5 % en empalmes válidos.

## 9. Test de hipótesis

**H-COST-01 (soporte, no gate de graduación) — ¿el CostModel simulado ≈ ejecución real?**
- **H0:** la divergencia entre costo simulado y realizado en shadow trading es ≤ tolerancia.
- **Estadístico:** divergencia shadow-vs-real de NAV atribuible a costos, 30 d.
- **Criterio:** > 2 % NAV en 30 d ⇒ el CostModel miente ⇒ **gatilla suspensión** (SPEC-12
  §14.1.2). Es el vínculo directo entre esta spec y el protocolo de retiro.

## 10. Protocolo de backtest / validación

- El CostModel se aplica en **todos** los backtests de todas las specs (no es opcional).
- Sensibilidad del `fee` (tier base vs. peor) reportada.
- El stitching se audita con el test I9; su fecha de corte vive en el SSOT.

## 11. Criterios de aceptación (DoD)

- [ ] Todo backtest del repo pasa por `cost_model` (grep-check en CI).
- [ ] Tests I9 (empalme) e impacto en verde.
- [ ] Presupuesto de turnover computado y reportado.
- [ ] Vínculo con SPEC-12 (divergencia shadow) implementado.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.3 turnover:** el drag se mide y se presupuesta aquí.
- **Capa datos §3.2:** barra canónica, stitching con QUALITY_FLAG, no-ffill.
- **ADR-0008:** funding excluido del costo (spot-only).

## 13. Parámetros pre-registrados

Fee 10 bps/lado; ½spread ~1 bp; slippage 2 bps + impacto; barra UTC 00:00; jerarquía de
stitching; discrepancia de empalme < 0.5 %; tamaño mínimo 2 % NAV; presupuesto de turnover
< 25 % del exceso de Calmar.
