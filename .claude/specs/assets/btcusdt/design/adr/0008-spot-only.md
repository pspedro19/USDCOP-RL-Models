# ADR-0008 — Instrumento spot-only (exposición ∈ [0, 1])

**Estado:** Aceptado · **Fecha:** pre-registro · **Spec:** SPEC-06, SPEC-07 · **Modo de fallo:** §7.2

## Contexto

La v2 nunca especificó *dónde* opera el sistema (spot vs. perpetuos). De eso depende medio
diseño: si el funding es costo o señal, si existe riesgo de liquidación, si la exposición
puede superar 1.0. El modo de fallo §7.2 (apalancamiento → ruina) es catastrófico en cripto:
un −50 % rutinario a 2× = ruina, y "apalancar en capitulación" es leverage hacia un cuchillo
que cae.

## Decisión

El núcleo opera **spot** (BTC/USDT–USDC). **Exposición ∈ [0, 1]. Sin apalancamiento, sin
cortos.** Consecuencias:
- No existe riesgo de liquidación (el §7.2 se resuelve por **diseño**, no por disciplina).
- El **funding es señal, no costo** (no entra al CostModel).
- Los perpetuos se usan **solo como fuente de datos** (funding, OI, basis).

## Alternativas descartadas

- **Perps con leverage acotado (≤ 1.25×):** reintroduce liquidación y complejidad de margen;
  el beneficio marginal no compensa el riesgo de cola. Sería **otro sistema** con su propia
  spec, backtest y DSR — no un parámetro de este.
- **Spot + cortos:** los cortos en un activo con sesgo alcista secular tienen carry negativo
  estructural y riesgo ilimitado; fuera del alcance de un overlay de riesgo.

## Consecuencias

Diseño más simple y robusto; se renuncia al (dudoso) upside de apalancar en suelos a cambio
de eliminar la ruina por liquidación. El "1.5× en capitulación" de diseños previos queda
**eliminado**, no acotado.
