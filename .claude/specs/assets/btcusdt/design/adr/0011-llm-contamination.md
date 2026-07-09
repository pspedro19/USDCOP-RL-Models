# ADR-0011 — Contaminación del clasificador LLM; ningún LLM ejecuta

**Estado:** Aceptado · **Fecha:** pre-registro · **Spec:** SPEC-05 · **Modo de fallo:** §7.8 (anti-look-ahead capa clasificadores)

## Contexto

El backtest del gate de eventos usa un LLM para clasificar titulares de Terra, FTX, hack de
Bybit. Pero **cualquier LLM moderno ya sabe cómo terminaron esas historias** (están en su
corpus). Cuando clasifica "Binance abandona la compra de FTX" como `EXCHANGE_FAILURE` con
confianza perfecta, está **recordando**, no detectando. El recall histórico así medido es una
**cota superior inflada** — look-ahead a nivel del clasificador. Además, un LLM con permisos
de ejecución sería no reproducible y no backtesteable.

## Decisión

1. **Texto tiempo-de-titular exclusivamente** en el corpus (nunca resúmenes retrospectivos).
2. **Recall histórico = cota superior**, nunca estimación insesgada.
3. **Estimación insesgada solo post-cutoff** en shadow trading (por eso se archiva texto crudo
   con timestamp desde el día 1).
4. **Versión del LLM fijada en la spec**; cambiarla = nuevo trial en el DSR.
5. **El vol-spike breaker (sin LLM, sin corpus) es la red que no se contamina.**
6. **Ningún LLM/agente tiene permisos de ejecución.** El LLM clasifica; las reglas deciden; el
   único camino a una orden es el motor determinista.

## Alternativas descartadas

- **Tomar el recall histórico como real:** el fallo (§7.8).
- **Trading autónomo por LLM:** no reproducible, no backtesteable, viola toda la validación.
- **Sentimiento LLM como señal primaria:** edge marginal que decae al arbitrarse; solo feature
  secundaria.

## Consecuencias

El gate de eventos se valida honestamente (sabiendo que su número histórico es optimista) y el
sistema no delega decisiones a un componente no auditable. El breaker garantiza cobertura aun
si el LLM falla o se contamina.
