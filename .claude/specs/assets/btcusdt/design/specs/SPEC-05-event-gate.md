# SPEC-05 — Gate de Eventos (Clasificador LLM + Vol-Spike Breaker)

| Campo | Valor |
|---|---|
| Estado | Aprobada (pre-registrada) |
| Versión | 1.0 |
| Depende de | `headlines_stream` (texto tiempo-de-titular + timestamp), `btc_price_1h` (ATR) |
| Materializa | `strategy/events.py` (`classifier`, `vol_spike_breaker`) |
| Pre-registro | Guía v3 §5.5, §7.5, §7.8, §8.2, §8.3; ADR-0011 |
| Rol | **Multiplicativo** (ortogonal); el único componente que puede llevar exposición a 0 |

## 1. Propósito y alcance

Producir `G_eventos ∈ {1.0, 0.5, 0.25, 0}`, el gate que reacciona a shocks no calendarizados
(quiebras de exchange, depegs, shocks regulatorios). Dos subsistemas independientes: (a) un
**clasificador LLM** con taxonomía cerrada y acción determinista, y (b) un **vol-spike
breaker** sin LLM (a prueba de contaminación). **El LLM clasifica; las reglas deciden.**

## 2. Entradas (contrato)

| Entrada | Tipo | Fuente | Disponibilidad | Contrato |
|---|---|---|---|---|
| `headline` | str + timestamp | RSS/X archivado | tiempo real | **texto tiempo-de-titular** (nunca resumen retrospectivo, §7.8) |
| `ATR_1h` | float | `btc_price_1h` | barra horaria | para el breaker |
| `llm_model_version` | str (config) | spec | fijo | proveedor+versión+cutoff (auditoría §7.8) |

## 3. Salidas (contrato)

| Salida | Tipo | Rango | Invariante |
|---|---|---|---|
| `G_eventos_t` | float | {1.0, 0.5, 0.25, 0} | discreto; nunca > 1.0 |
| `event_class` | enum (taxonomía cerrada) | 7 clases | el LLM SOLO puede emitir estas |
| `reentry_timer` | duración/None | — | por clase |
| `breaker_active` | bool | — | del vol-spike, independiente del LLM |

## 4. Interfaz (API)

```python
class EventClassifier(Protocol):
    def classify(self, headline: str, ts: datetime) -> EventClass:
        """LLM temp=0, JSON schema estricto, taxonomía CERRADA. Solo texto tiempo-de-titular."""

def action_for(event_class: EventClass, confirmations: int) -> GAction:
    """Tabla DETERMINISTA clase→acción (versionada). El LLM NUNCA decide la acción."""

def vol_spike_breaker(atr_1h: pd.Series) -> bool:
    """ATR(H1) > 3×EWMA(ATR) ⇒ True. Sin LLM, sin corpus, sin memoria."""
```

## 5. Algoritmo / lógica

**Tabla determinista clase → acción (versionada, backtesteable):**

| Clase | 1.ª señal | Confirmación | Re-entrada |
|---|---|---|---|
| `EXCHANGE_FAILURE` | G=0.5 | G=0 | Manual + vol normalizada |
| `STABLECOIN_STRESS` | G=0.5 | G=0 | Manual |
| `REGULATORY_SHOCK` | G=0.25 | — | Timer 48 h + revisión |
| `MACRO_SURPRISE` | G=0.5 | — | Timer 24 h |
| `ETF_STRUCTURAL` | G=0.5 | — | Timer 12 h |
| `WHALE_MOVEMENT` | G=0.75 | — | Timer 12 h |
| `NOISE` (~99 %) | G=1.0 | — | — |

**Acción en dos tiempos (§7.5) — resuelve la tensión recall/confirmación:**
```
1.ª señal de clase catastrófica → G=0.5 INMEDIATO (des-exposición parcial)
confirmación (2.ª fuente independiente en 15 min, o vol-spike) → G=0 (flat total)
⇒ nunca plenamente expuesto mientras se confirma; nunca flat total por un solo titular
```

**Vol-spike breaker (independiente, §8.3):**
```
if ATR_1h_t > 3 × EWMA(ATR_1h):  G = min(G, 0.5); pausa
```
No lee noticias, no tiene corpus, no recuerda FTX. Es la red que **no se contamina** (§7.8.5).

**Fail-safe asimétrico:** feed caído ⇒ G mantiene último valor + alerta; salida LLM no
parseable ⇒ tratar como `NOISE` + alerta.

## 6. Invariantes y post-condiciones

- `G_eventos ≤ 1.0` siempre (nunca amplifica).
- El LLM solo emite clases de la taxonomía cerrada (validación de schema; fallo ⇒ NOISE).
- `G=0` requiere confirmación (nunca por un solo titular).
- El breaker es funcionalmente independiente del clasificador (test: con LLM apagado, el
  breaker sigue operando).
- `llm_model_version` queda registrada en cada clasificación.

## 7. Tests unitarios

- [ ] Cada clase mapea a su acción determinista exacta (tabla).
- [ ] Un solo titular catastrófico ⇒ G=0.5, NO G=0 (requiere confirmación).
- [ ] Dos fuentes en 15 min ⇒ G=0.
- [ ] Salida LLM no-JSON ⇒ NOISE + alerta (no crashea).
- [ ] `vol_spike_breaker`: ATR = 3.1×EWMA ⇒ True; 2.9× ⇒ False.
- [ ] Breaker opera con clasificador mockeado a error (independencia).
- [ ] Taxonomía cerrada: una clase inventada por el LLM se rechaza.

## 8. Tests de integración

- [ ] Corpus histórico etiquetado a mano (Terra, FTX, hack Bybit, aprobación ETF, halvings,
      sorpresas Fed) + ruido: el pipeline produce las acciones esperadas.
- [ ] El breaker se activa en las cascadas conocidas (2020-03, 2021-05, 2022-05/11) desde ATR
      solo, sin noticias.

## 9. Test de hipótesis

**H-EVT-01 — ¿el clasificador detecta eventos catastróficos? (recall es cota superior, §7.8)**
- **H0:** el recall del clasificador sobre clases catastróficas no supera una base trivial.
- **Estadístico:** recall + FP/año sobre el corpus etiquetado, **declarado como cota
  superior** (el LLM ya conoce estos eventos → contaminación).
- **Criterio:** recall ≈ 100 % en `EXCHANGE_FAILURE`/`STABLECOIN_STRESS` **y** FP/año
  aceptable. La estimación **insesgada** llega solo del shadow trading sobre eventos
  **post-cutoff** (§7.8.3).

**H-EVT-02 — ¿el gate mejora el riesgo? (¿las aplanadas salvan más de lo que cuestan?)**
- **H0:** la suma del PnL diferencial de las aplanadas ≤ 0 (el gate destruye valor).
- **Estadístico:** PnL diferencial por aplanada (con vs. sin gate) sobre el histórico,
  bootstrap.
- **Criterio:** suma de PnL diferencial > 0 e IC 95 % que no cruza 0; mejora de CVaR/Calmar.

## 10. Protocolo de backtest / validación

- **Corpus tiempo-de-titular exclusivamente** (§7.8.1): nada de resúmenes retrospectivos.
- **Recall histórico = cota superior** (§7.8.2); la métrica que cuenta es post-cutoff en
  shadow (§7.8.3).
- Versión del LLM fijada en la spec; cambiarla = **nuevo trial en el DSR** (§7.8.4).
- El vol-spike breaker se valida por separado (no contaminado).

## 11. Criterios de aceptación (DoD)

- [ ] H-EVT-01: recall ≈ 100 % en clases catastróficas (cota superior) + FP/año aceptable.
- [ ] H-EVT-02: PnL diferencial de aplanadas > 0.
- [ ] Acción en dos tiempos y fail-safe implementados y testeados.
- [ ] Breaker independiente verificado.
- [ ] `llm_model_version` versionada; archivo de titulares crudos activo desde el día 1.

## 12. Anti-look-ahead / modos de fallo cubiertos

- **§7.8 contaminación LLM (RESUELTO AQUÍ):** texto tiempo-de-titular; recall = cota superior;
  test insesgado post-cutoff; versión de LLM como trial; breaker no contaminado.
- **§7.5 asimetría:** recall sobre precisión; acción en dos tiempos.
- **Seguridad:** doble confirmación para flat = defensa anti fake-news / prompt-injection.

## 13. Parámetros pre-registrados

Taxonomía y acciones (§5, tabla); breaker `ATR>3×EWMA`; confirmación 2 fuentes en 15 min;
recall objetivo ~100 % clases catastróficas; timers por clase. Ninguno se optimiza sobre el
test.
