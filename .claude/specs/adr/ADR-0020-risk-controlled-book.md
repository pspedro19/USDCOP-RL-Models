---
kind: adr
status: PROPOSED
contract: CTR-PRODUCT-CLASS-001
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - scripts/analysis/portfolio_daily.py
  - .claude/rules/quant-constitution.md
  - .claude/specs/audit/STRATEGIC-ASSESSMENT-2026-07.md
---

# ADR-0020 — "Libro de riesgo controlado" como clase de producto

## Contexto

Cuatro activos, cuatro veredictos negativos: DSR < 0.95 en todos, baselines tontos ganando en
el año held-out, forecasting direccional con p ajustado 0.657. Bajo la lectura actual de la
constitución, **nada de este sistema puede promoverse jamás**, porque la única barra escrita es
la barra del alfa.

Pero el propio `STRATEGIC-ASSESSMENT-2026-07.md:73-76` ya concluyó otra cosa:

> *"llama a la cosa por su nombre — una **estrategia de control de riesgo** (gate de régimen ×
> vol-target × stops), no un sistema de forecasting. Eso es un edge legítimo y defendible."*

Esa conclusión se escribió y nunca se ejecutó: sus prioridades P1-P6 no contienen ni un ítem
de cartera. La razón estructural es que **no existe una clase de producto para ella**, así que
`portfolio_daily.py` se auto-estampa `promotion_eligible: False` y ahí muere.

## Decisión

Se declaran **dos clases de producto con barras distintas**. Una afirmación más débil merece
una barra más débil — y merece que se le prohíba explícitamente hablar como la fuerte.

| | **Claim de alfa** | **Libro de riesgo controlado** |
|---|---|---|
| Afirma | "predice" / "tiene edge" | "beta diversificada con riesgo acotado" |
| Barra | **DSR > 0.95** (constitución §2) | **B1′ + costos ×2 + freno de DD + comportamiento en crisis** |
| Evidencia | OOS limpio + forward | forward + ventanas de estrés medidas |
| Se vende como | señal | exposición gestionada |
| Estado máximo | `production` | `production` (con las prohibiciones de abajo) |

**Justificación de la barra.** La constitución §2 exige DSR > 0.95 para *"ningún claim de
edge"* — un claim de alfa. Un libro que solo afirma beta gestionada no reclama edge. Pero §3
**sigue vinculándolo**, y su baseline **B1′ de exposición emparejada** es exactamente la prueba
correcta: no *"¿bato a cero?"* sino *"¿bato a exposición constante de mi propio tamaño medio?"*.
Sin B1′, "tener menos exposición" se disfrazaría de habilidad — que es el error que este
sistema ya cometió una vez.

## Criterios de graduación de un libro

Todos obligatorios, ninguno relajable en drawdown:

1. **Bate a B1′** (exposición constante = exposición bruta media realizada) en Calmar.
2. **Sobrevive costos ×2** con retorno neto y Calmar positivos.
3. **Freno de drawdown agregado activo**, con umbrales declarados ex-ante.
4. **Correlaciones reportadas condicionalmente** (ver prohibición 3).
5. **Comportamiento medido en ≥2 ventanas de estrés** históricas, con las correlaciones de
   esas ventanas, no las de la ventana completa.
6. **Contabilidad completa**: el efectivo no invertido rinde la tasa libre de riesgo, con su
   fuente y su lag documentados. Un libro plano el 43% del tiempo que credita 0% se está
   reportando mal, en la dirección que parece conservadora.
7. **Forward acumulado** en el ledger, con protocolo de retiro firmado ex-ante.

## Prohibiciones — sin esto, la clase es una puerta trasera

1. **No puede presentarse con lenguaje de alfa.** Ni "predice", ni "señal", ni "modelo", ni
   "edge". Es exposición gestionada.
2. **El Sharpe no puede ser el titular.** El titular es Calmar y el comportamiento en
   drawdown. El Sharpe va con su error estándar o no va.
3. **No puede publicarse una correlación incondicional sin su condicional al lado.** Medido
   aquí: ρ incondicional máxima 0.086, **ρ co-activa máxima 0.196**, y las tres piernas están
   activas a la vez solo el **12.9%** de los días. Buena parte de la diversificación aparente
   es **no-solapamiento de presencia**, no riesgo que se compense. Publicar solo el 0.086 es el
   error que este ADR existe para impedir.
4. **No puede omitir que su retorno es beta gestionada.** La correlación del libro con un largo
   pasivo de sus propios constituyentes es **0.536**.
5. **No exime del forward.** Un libro sin protocolo de retiro firmado no opera, igual que una
   estrategia.

## Estado medido hoy (2026-07-21)

| | sin efectivo | **con efectivo** |
|---|---|---|
| Retorno anual | 10.07% | **11.20%** |
| MaxDD | −7.92% | **−7.41%** |
| Calmar | 1.271 | **1.511** |

Costos ~0.9%/año, ~1.8% a ×2 — no es frágil a costos. COVID-2020: **−0.49%**. Bajista 2022:
**−0.92%**. Ratio de diversificación 1.614.

**Pero H-PORT-D-01 no pasa**: ΔCalmar contra el mejor sleeve individual = −0.049, IC95
[−1.526, +1.238], **incluye cero**. Y el libro va **−4.02% en 2026** con las tres piernas
negativas.

⇒ **El libro cumple los criterios 1-6 y NO el 7.** Su estado sigue siendo
`research_validated`, no `production`. Este ADR no lo promueve: **crea la vía por la que
podría promoverse si el forward lo respalda**, que es distinto y es lo que faltaba.

## Consecuencias

- `portfolio_daily.py` puede emitir `product_class: risk_controlled_book` en vez de solo
  `promotion_eligible: false`.
- El dashboard necesitará una superficie distinta para un libro (pesos, exposición bruta,
  distancia al freno de DD) que para una estrategia (trades, señales).
- **Riesgo asumido**: esta clase puede usarse para vender beta como si fuera habilidad. Las
  cinco prohibiciones son la mitigación, y son verificables por test — no dependen de la buena
  fe de quien redacte la ficha del producto.
- Revertir este ADR significa volver a "nada se promueve nunca", que es donde estábamos.
