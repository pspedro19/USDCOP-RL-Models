# Constitución — Capa de Modelado BTC

> **SSOT de reglas para el modelado.** Extiende (no reemplaza) la constitución del pipeline
> de datos. Ante conflicto entre una spec, el código o una opinión, **gana esta
> constitución**. Cambiarla requiere un ADR.

---

## 0. Propósito

Especificar cómo se construye la **estrategia** de Bitcoin sobre el dataset del pipeline:
un motor de **exposición ∈ [0,1] spot-only** modulada por gates auditables, con validación
honesta y un protocolo de retiro pre-firmado. **Nada predice el precio; todo modula cuánto
estar expuesto.**

## 1. Principios no negociables (heredados de la Guía v3)

1. **La dirección de BTC es ~ruido.** No se construyen predictores de precio. El alpha vive
   en la gestión de exposición adaptativa (ciclo, liquidez, posicionamiento, eventos).
2. **HODL es el baseline brutal.** El objetivo declarado es ganar en **Calmar**, no en
   retorno absoluto. Es válido y honesto que el resultado sea "S3 con reglas ES la estrategia".
3. **Desconfianza de la magia.** Sharpe > 4–5, drawdown < 1 %, retornos de tres/cuatro cifras
   ⇒ look-ahead o costos ignorados hasta demostrar lo contrario.
4. **Pre-registro total.** Gates, mapeos, umbrales, pesos, bandas, vol objetivo, cap de
   portafolio y criterios de retiro se fijan **antes** de ver un solo resultado OOS
   (`PRE-REGISTRATION.md`). Todo lo decidido después de mirar el OOS es feature selection
   disfrazada.
5. **El default es descartar.** Un componente entra al producto solo si demuestra aporte en
   Calmar **en aislamiento** y su test de hipótesis rechaza H0 tras corrección múltiple.

## 2. Anti-look-ahead en TRES capas (regla central)

El anti-look-ahead no vive solo en los datos. Se audita en tres niveles y cada uno tiene su
test automatizado:

| Capa | Fuga posible | Defensa | Spec |
|---|---|---|---|
| **Datos** | ffill del precio, normalización global, disponibilidad futura | Reglas de oro del pipeline (heredadas) + test I10 de lag | pipeline `04-quality` |
| **Modelos** | re-ajustar el HMM sobre histórico completo y re-etiquetar el pasado | **Fit congelado walk-forward**: el modelo solo etiqueta hacia adelante; jamás se re-etiqueta el pasado | SPEC-01 |
| **Clasificadores** | el LLM ya conoce cómo terminó FTX/Terra (está en su corpus) | Texto tiempo-de-titular; recall histórico = **cota superior**; test insesgado solo post-cutoff en shadow | SPEC-05, ADR-0011 |

## 3. Instrumento: SPOT-ONLY (decisión constitucional)

- El núcleo opera **spot** (BTC/USDT–USDC). **Exposición ∈ [0, 1]. Sin apalancamiento, sin
  cortos.** No existe riesgo de liquidación por diseño.
- El **funding es señal, no costo.** El `CostModel` incluye fees + spread + slippage; **no**
  funding.
- Los perpetuos se usan **exclusivamente como fuente de datos** (funding, OI, basis).
- Cualquier variante apalancada es **otro sistema** con su propia spec, backtest y DSR.
  Ver ADR-0008.

## 4. Combinación de señales: en riesgo, no a ciegas (decisión constitucional)

- Los componentes **correlacionados** (`|ρ|` rodante 90d > **0.4** sostenido) **se combinan
  aditivamente en espacio de riesgo**, no se multiplican (evita doble conteo). Ver SPEC-03,
  ADR-0009.
- Solo los componentes **~ortogonales** se multiplican.
- La ortogonalidad se **mide** (matriz de correlación rodante trimestral), no se asume. Un
  cambio de combinación por violación del umbral **cuenta como trial en el DSR**.

## 5. Rol asimétrico de las señales de riesgo

- El gate de posicionamiento (`z_funding`) y el meta-modelo **casi solo reducen o frenan**;
  no crean ni amplían exposición. Su trabajo es esquivar cascadas y trades de baja calidad,
  no predecir rallies.
- El gate de eventos usa **acción en dos tiempos** (1.ª señal reduce parcial; confirmación
  aplana): nunca plenamente expuesto mientras se confirma, nunca flat total por un titular.

## 6. Cero magia numérica

- Ningún parámetro se obtiene por **grid search sobre el test**. Todos son **priors
  económicos** declarados. La sensibilidad (p. ej. σ_objetivo {25/30/35 %}) se **reporta
  completa**; elegir la mejor celda es empezar la v1 otra vez y se prohíbe.

## 7. Métricas de juicio

- Métrica primaria de graduación: **Calmar** (y Sortino). Sharpe es secundario (penaliza la
  asimetría deseada). **Deflated Sharpe** con registro real de trials es obligatorio para
  cualquier claim de edge.

## 8. Ningún LLM toca la cuenta

- Los LLMs **clasifican o resumen**; las **reglas deciden**; el único camino a una orden es
  el motor determinista. Ningún agente tiene permisos de ejecución. Ver ADR-0011.

## 9. No hay go-live sin certificado de retiro

- El **protocolo de retiro** (SPEC-12) se **firma antes** del go-live. Sus umbrales no se
  relajan mientras el sistema está en drawdown.

## 10. Definition of Done (por componente)

Un componente está terminado cuando: (1) tiene spec aprobada; (2) tests unitarios + de
integración en verde; (3) su **test de hipótesis rechaza H0** tras corrección múltiple en
walk-forward OOS; (4) demostró aporte en **Calmar en aislamiento**; (5) pasa `ruff`+`mypy`;
(6) sus parámetros están en `PRE-REGISTRATION.md`; (7) no introduce look-ahead en ninguna de
las tres capas; (8) su contribución quedó **registrada como trial en el DSR** (aporte o
descarte).
