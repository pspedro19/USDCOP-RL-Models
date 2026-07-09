# SPEC-12 — Protocolo de Retiro del Sistema en Vivo ★

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) — **se firma ANTES del go-live** |
| Versión | 1.0 |
| Depende de | métricas en vivo, shadow trading (SPEC-07), Drift Monitor, B1 (SPEC-10) |
| Materializa | `ops/retirement.py` (Retirement Monitor) |
| Pre-registro | Guía v3 §14 |
| Regla | **No existe go-live sin este protocolo firmado.** Sus umbrales no se relajan en drawdown |

## 1. Propósito y alcance

Definir, **antes de encender el sistema**, las condiciones bajo las cuales se **pausa** o se
**apaga definitivamente**. Sin criterios pre-registrados, la decisión de apagar se tomaría en
pleno drawdown, con el peor estado emocional y el máximo incentivo a "darle una semana más".
El certificado de defunción se firma antes del go-live.

## 2. Entradas (contrato)

| Entrada | Tipo | Fuente |
|---|---|---|
| DD en vivo | float | métricas de la cuenta |
| max DD del backtest OOS | float | SPEC-11 (referencia congelada) |
| Divergencia shadow-vs-real | float | SPEC-07 §9 |
| PSI de features | float | Drift Monitor (SPEC-05/8.4) |
| Calmar 12m del sistema vs B1 | float | SPEC-10/11 |
| Estado de QA de datos | enum | pipeline |

## 3. Salidas (contrato)

| Salida | Tipo | Efecto |
|---|---|---|
| `action` | enum{RUN, SUSPEND, HALF, DEATH} | SUSPEND ⇒ exposición→0; HALF ⇒ media exposición; DEATH ⇒ capital→B1/cash |
| `reason` | str | condición disparada (auditoría) |
| `alert` | señal | al operador (el Retirement Monitor solo alerta; no ejecuta) |

## 4. Interfaz (API)

```python
class RetirementMonitor(Protocol):
    def evaluate(self, live_metrics: LiveMetrics, refs: FrozenRefs) -> RetirementAction:
        """Evalúa a diario los criterios de §14. Solo alerta/recomienda; no toca la cuenta."""
```

## 5. Algoritmo / lógica (los tres niveles)

**5.1 Suspensión automática (pausa, no muerte)** — cualquiera ⇒ exposición → 0 + revisión:
```
1. DD en vivo > 1.25 × max DD del backtest OOS        # el modelo del mundo está roto o el mundo cambió
2. Divergencia shadow-vs-real > 2% NAV acum. en 30d   # el CostModel miente
3. PSI > 0.25 sostenido 2 semanas en feature del núcleo
4. QA de datos rojo sin resolución en 48h
```

**5.2 Revisión de decaimiento (fría, trimestral):**
```
Calmar 12m del sistema < Calmar 12m de B1 durante 2 trimestres consecutivos
   ⇒ media exposición + revisión de tesis
   ⇒ si no hay causa identificable y corregible (pre-registrada, no ad-hoc) ⇒ retiro
```

**5.3 Muerte definitiva:**
```
3 suspensiones automáticas en 12 meses
   O  Calmar 12m < B1 durante 4 trimestres consecutivos
   O  quiebre de un supuesto estructural nombrado
      (p. ej. divergencia absolutos-vs-percentiles on-chain sostenida > 1 año, SPEC-01)
⇒ capital → B1 (HODL vol-targeted) o cash
⇒ re-encender exige una v4 con NUEVO pre-registro y NUEVO DSR desde cero (no un "ajuste")
```

**5.4 Regla de humildad operativa:** ninguna condición se relaja mientras el sistema está en
drawdown. Los cambios al protocolo solo se firman con el sistema en **máximos de equity** o
apagado.

## 6. Invariantes y post-condiciones

- Los umbrales de referencia (max DD OOS, etc.) están **congelados** desde el go-live.
- El Retirement Monitor **solo alerta**; nunca ejecuta órdenes (constitución §8).
- Ninguna condición se relaja en drawdown (regla de humildad).
- Una DEATH exige v4 (pre-registro + DSR nuevos), no un parche.

## 7. Tests unitarios

- [ ] DD en vivo = 1.26× max DD OOS ⇒ SUSPEND; 1.2× ⇒ RUN.
- [ ] Divergencia shadow 2.1 % en 30d ⇒ SUSPEND.
- [ ] PSI 0.26 sostenido 2 semanas ⇒ SUSPEND; pico de 1 día ⇒ no.
- [ ] Calmar < B1 por 2 trimestres ⇒ HALF; por 4 ⇒ DEATH.
- [ ] 3.ª suspensión en 12 meses ⇒ DEATH.
- [ ] El monitor no emite ninguna acción de ejecución (solo alerta).

## 8. Tests de integración

- [ ] Simulación de un drawdown sintético que cruza 1.25× ⇒ el monitor dispara SUSPEND en la
      fecha correcta.
- [ ] Un escenario de decaimiento lento de Calmar ⇒ HALF y luego DEATH en los trimestres
      correctos.

## 9. Test de hipótesis

Esta spec no tiene un test de hipótesis de edge (no es un componente de alpha). Su
"validación" es de **cobertura**: verificar que cada condición de §14 tiene una prueba
unitaria que la dispara y una que no (completitud del árbol de decisión).

## 10. Protocolo de backtest / validación

- Backtest histórico del protocolo: ¿habría apagado en los drawdowns conocidos? ¿habría
  sobre-reaccionado en correcciones normales? Se reporta la tasa de suspensiones que en
  retrospectiva fueron innecesarias (calibración de umbrales, pero **sin optimizarlos** — son
  priors de gestión de riesgo).
- Shadow trading ≥ 3 meses antes del go-live (Fase 11).

## 11. Criterios de aceptación (DoD)

- [ ] Protocolo **firmado antes del go-live** (Fase 11).
- [ ] Cada condición de §5 tiene test que la dispara y test que no.
- [ ] Retirement Monitor implementado como alerta-only (sin permisos de ejecución).
- [ ] Referencias congeladas (max DD OOS, etc.) registradas.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **Sesgo emocional en drawdown:** criterios pre-registrados + regla de humildad.
- **§7.6 supervivencia:** el quiebre estructural on-chain es condición de muerte.
- **Constitución §8:** el monitor no toca la cuenta.

## 13. Parámetros pre-registrados

Suspensión: DD > 1.25× OOS · shadow > 2 % NAV/30d · PSI > 0.25 · QA rojo 48h. Decaimiento:
Calmar < B1 2 trim. Muerte: 3 suspensiones/12m · Calmar < B1 4 trim. · quiebre estructural.
Re-encendido: v4 desde cero. Ninguno se relaja en drawdown.
