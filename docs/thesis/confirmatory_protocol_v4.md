---
kind: protocol
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-14
supersedes: []
code_anchors:
  - config/research/thesis_confirmatory_v4.yaml
  - scripts/diagnostics/validate_thesis_confirmatory_protocol.py
---

# Protocolo confirmatorio v4 de la tesis

Este documento implementa la mejora metodológica acordada para una nueva evaluación. No reetiqueta como confirmatorios los resultados retrospectivos de 2023 y no modifica el `partition.yaml` histórico.

## Partición

La versión nueva usa 2020–2022 para entrenamiento, 2023 para selección, 2024–2025 como hold-out confirmatorio de una sola mirada y 2026 como forward posterior al congelamiento. Los conteos efectivos se calculan desde el dataset congelado; no se escriben cifras antes de construir la máscara. El año 2023 sigue siendo retrospectivo para la versión actual porque ya fue observado.

## Datos y disponibilidad

Cada serie debe conservar identidad, frecuencia, unidad, fecha de publicación y primera disponibilidad. La macro del mismo día, el `bfill`, la interpolación no declarada y el cero silencioso están prohibidos. Investing puede comprobar coherencia diaria de DXY o USD/COP, pero no equivale a independencia de fuente ni a validación de fills intradía. Las velas planas se reportan y las features de rango degeneradas se eliminan globalmente antes del entrenamiento.

## Modelos

La comparación confirmatoria incluye flat, reglas fijas, LogReg y PPO con y sin régimen. PPO usa cinco semillas por configuración y se reporta por semilla. DeepSeek/Azure y oro son exploratorios: no definen el ganador del hold-out ni se elige retrospectivamente el proveedor con mejor resultado.

## Evidencia y decisión

El endpoint primario es el retorno diario neto pareado contra always-flat. Se calculan bootstrap jerárquico por semilla, intervalos, DSR con el registro reconciliado, stress de costos y análisis por trimestre y régimen. Un retorno bruto positivo no se llama alfa. La afirmación de alfa requiere mejora neta, incertidumbre favorable, DSR trial-aware y supervivencia al contrato de costos predefinido.

El validador `scripts/diagnostics/validate_thesis_confirmatory_protocol.py` es fail-closed: el protocolo solo admite ejecución después de una firma explícita (`SIGNED`) con responsable y fecha. La firma conversacional del operador quedó registrada en el YAML el 14 de septiembre de 2026. Validar el YAML no entrena, no lee secretos, no llama proveedores y no abre el hold-out; antes de cada trial aún debe reconciliarse el registro.
