---
kind: analysis
status: diagnostic
version: 1.0.0
last_verified: 2026-09-15
supersedes: docs/analysis/exp-tesis-rl-02-brazos-rl-llm-hibrido.md
code_anchors:
  - scripts/analysis/settle_thesis_llm.py
  - scripts/analysis/thesis_hybrid.py
  - scripts/presentation/generate_llm_figures.py
  - scripts/presentation/generate_hybrid_figures.py
---

# LLM e híbrido: diagnóstico retrospectivo

Este bloque usa el conjunto de selección 2023 (226 sesiones) y por diseño no es
confirmatorio. Los ledgers se validaron estructuralmente: 13.334/13.334 decisiones por
proveedor, 226/226 sesiones completas, sin JSON inválido ni campos secretos. El portable
histórico tiene SHA-256 `7f332df17a2492a0533f713bc143316fec9b9eadedcfbdad59d5eb0a36b7b0b5`.

La liquidación se ejecutó con `--allow-retrospective` porque el portable histórico no
coincide con la identidad calculada por el código actual. Ese indicador queda grabado en
los JSON y prohíbe interpretar estas cifras como resultados v4.

| Brazo | Sesiones | Retorno compuesto | Sharpe anualizado (sqrt 221) | Max drawdown |
|---|---:|---:|---:|---:|
| DeepSeek | 226 | −78,93 % | −22,78 | −78,82 % |
| Azure | 226 | −82,89 % | −14,84 | −82,82 % |
| Híbrido + DeepSeek | 226 | −69,24 % | −14,33 | −69,14 % |
| Híbrido + Azure | 226 | −70,59 % | −11,01 | −70,57 % |

El resultado es evidencia de que estas ejecuciones concretas no produjeron señal neta
positiva. No es evidencia general contra los LLM: los identificadores históricos
`deepseek-chat` y `gpt-4o-mini` no fijan una versión servida reproducible, y el prompt
recibía features normalizadas. Además, el bloque 2023 fue observado antes de las
correcciones v4.

## Figuras

- [Capital DeepSeek](../../outputs/thesis-repair/llm_figures/deepseek_selection/llm_curva_capital.png)
- [Drawdown DeepSeek](../../outputs/thesis-repair/llm_figures/deepseek_selection/llm_drawdown.png)
- [Capital Azure](../../outputs/thesis-repair/llm_figures/azure_selection/llm_curva_capital.png)
- [Drawdown Azure](../../outputs/thesis-repair/llm_figures/azure_selection/llm_drawdown.png)
- [Capital híbrido DeepSeek](../../outputs/thesis-repair/llm_figures/hybrid_deepseek/hybrid_curva_capital.png)
- [Capital híbrido Azure](../../outputs/thesis-repair/llm_figures/hybrid_azure/hybrid_curva_capital.png)

Para presentar un resultado confirmatorio de LLM habría que fijar un modelo versionado,
registrar el hash del dataset en cada fila, firmar un preregistro específico y ejecutar
el juez post-freeze/forward. Estas cifras no sustituyen ese paso.
