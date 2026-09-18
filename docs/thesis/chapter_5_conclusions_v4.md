---
kind: thesis
status: ready_for_review
version: 1.0.0
last_verified: 2026-09-15
supersedes: docs/thesis/chapters_1_2_5.md
code_anchors:
  - docs/thesis/confirmatory_results_v4.md
  - docs/thesis/llm_hybrid_diagnostic_results_v2.md
  - scripts/analysis/check_confirmatory_artifacts.py
---

# Capítulo 5. Conclusiones y límites

## 5.1 Conclusión principal

En la partición confirmatoria 2024--2025, ninguna de las diez corridas PPO v4 produjo
rentabilidad neta positiva. El resultado fue 0/5 semillas positivas tanto para la
configuración con régimen como para el backbone. Por ello la tesis no debe presentar
un sistema operativo rentable; debe presentar una evaluación reproducible que no
encuentra edge neto bajo este diseño, datos y contrato de costes.

## 5.2 Qué sí se estableció

El pipeline causal y la identidad del dataset son auditables; el entorno reproduce
soluciones conocidas en datos sintéticos; el entrenamiento y la liquidación comparten
contabilidad; y la variabilidad entre semillas está publicada. El fracaso del PPO no se
atribuye a que el espacio de acciones carezca de la acción flat: se atribuye a esta
receta y representación.

Los ledgers retrospectivos muestran además que los dos proveedores LLM probados en
2023 tampoco superaron la abstención. Esa evidencia es diagnóstica, porque los modelos
históricos no quedaron fijados a una versión servida y el bloque ya había sido
observado.

## 5.3 Limitaciones que impiden una afirmación universal

1. El spread intradía es un límite implícito del proveedor, no una serie histórica de
   bid/ask y fills observados.
2. La disponibilidad histórica de algunas macroseries no está probada mediante
   vintages completos; la regla T−1 evita la fuga directa, pero no resuelve por sí sola
   revisiones retrospectivas.
3. El hold-out 2024--2025 fue observado durante el ciclo de reparación y por eso el
   juez realmente nuevo es el forward posterior al freeze.
4. Los resultados LLM de selección son retrospectivos y no deben combinarse con el
   contraste confirmatorio PPO.

## 5.4 Trabajo futuro válido

El siguiente experimento autorizado es el forward 2026 con código, costes, modelo LLM,
prompt y ledger congelados. Solo después de alcanzar el tamaño muestral preregistrado
se podrá evaluar si existe evidencia de rentabilidad neta. Si vuelve a fallar, la
conclusión más fuerte será que el diseño intradía de cinco minutos no supera su peaje
de ejecución; si mejora, deberá replicarse en otro período y activo antes de llamarse
generalizable.

La lectura parcial disponible hasta 2026-09-10 ya fue ejecutada con el mismo freeze: 162
sesiones, 0/5 semillas positivas en régimen y backbone. Los baselines del mismo motor
también fueron liquidados: `always_flat` obtuvo 0,00 %, `always_short` −10,57 % y la
política de dos reglas por régimen −20,84 %. Se conserva como evidencia exploratoria
porque el periodo había sido inspeccionado por diagnósticos previos y todavía no se ha
cerrado un manifiesto de apertura confirmatoria con el ledger completo de trials.
