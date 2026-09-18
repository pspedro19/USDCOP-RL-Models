---
kind: as-built
status: PARTIAL
version: 1.0.0
last_verified: 2026-09-14
supersedes: []
code_anchors:
  - scripts/presentation/build_thesis_manuscript.py
  - tests/regression/test_thesis_manuscript.py
---

# Memoria actualizada con evidencia retrospectiva

Resultados ejecutables: [PPO v4 confirmatorio](confirmatory_results_v4.md) y [LLM/híbrido retrospectivo](llm_hybrid_diagnostic_results_v2.md).

Para redacción final se dispone del [addendum de capítulos 3--4](chapters_3_4_v4_addendum.md)
y del [capítulo 5 v4](chapter_5_conclusions_v4.md).

La lectura post-freeze disponible de 2026 está separada en [forward v4](forward_results_2026_v4.md)
y permanece exploratoria hasta cerrar su apertura confirmatoria.

La operación pendiente de proveedores está descrita en el [runbook LLM v4](llm_execution_runbook_v4.md).
El estado máquina-auditable de cierre se guarda en
`outputs/thesis-repair/confirmatory_v4_stable/completion_audit_v4.json`; actualmente
reporta `READY_WITH_EXTERNAL_LLM_PENDING` y no oculta ese bloqueo.

Fuentes de escritura: [capítulos 1, 2 y 5](chapters_1_2_5.md), [capítulos 3 y 4](chapters_3_4.md) y [protocolo de diagnósticos](diagnostic_protocol.json). Los marcadores de tablas e intervalos en la fuente son sustituidos por el generador; no aparecen en la entrega.

El nuevo diseño confirmatorio está descrito en el [protocolo v4](confirmatory_protocol_v4.md) y declarado en [su configuración](../../config/research/thesis_confirmatory_v4.yaml). El validador se ejecuta antes de cualquier firma o entrenamiento.

La entrega editable de los capítulos actualizados se genera con
`python scripts/presentation/build_confirmatory_v4_addendum.py --output <directorio-nuevo>`.
La ejecución verificada del 15-sep-2026 produjo
`outputs/thesis-delivery/confirmatory_v4_addendum_20260915/capitulos_3_4_5_v4.docx`
y su PDF, separados del manuscrito histórico.

La entrega con las seis figuras v4 separadas y a página completa está en
`outputs/thesis-delivery/confirmatory_v4_addendum_20260915_with_figures/capitulos_3_4_5_v4.docx`
y su PDF. Las figuras se regeneran con
`scripts/presentation/generate_confirmatory_v4_figure_pack.py` desde los artefactos
congelados, sin retornos sintéticos.

La entrega utiliza el paquete histórico research_grade_20260912_v5 y conserva su clasificación retrospectiva. No entrena modelos, no llama a proveedores, no reabre el hold-out y no modifica fuentes ni registros de trials. Una revisión documental no certifica disponibilidad macro histórica, fills o rentabilidad. El protocolo v4 ya está firmado para preparar la siguiente fase, pero no se considera evidencia hasta ejecutar los gates de datos y trials.

## Reproducción

Instalar numpy, pandas, matplotlib, python-docx, reportlab, pypdfium2 y pypdf en el entorno de ejecución. En este workspace las dependencias documentales adicionales están aisladas en outputs/thesis-delivery/_deps; sus versiones exactas se exportan con cada entrega. Se requiere acceso a los objetos preservados del snapshot histórico.

```powershell
python scripts/presentation/build_thesis_manuscript.py --output outputs/thesis-delivery/entrega_nueva
python -m pytest tests/regression/test_thesis_manuscript.py -q
```

El destino debe ser nuevo: el generador rechaza sobrescritura. Produce tesis completa y capítulos 3–4 en DOCX/PDF/Markdown, seis paneles y figuras individuales, CSV, diagnósticos, copia de evidencia, páginas renderizadas y manifiesto de hashes. El generador rechaza discrepancias de identidad y contabilidad. Las páginas se renderizan para inspección visual; el archivo de QA automático no se confunde con una aprobación humana.

## Interpretación del resultado principal

PPO de pesos medianos: bruto compuesto +10,08 %, neto compuesto −36,67 %; el intervalo exploratorio del bruto compuesto incluye cero. No se etiqueta como alfa demostrado. Se publica también el bruto positivo de LogReg y siempre corto. El equilibrio de costos es contrafactual para posiciones congeladas, no una oferta de ejecución real. La tasa de éxito de llamadas originales no se deduce de una cohorte final de decisiones válidas.

## Alcance del cierre

Estado PARTIAL se refiere a revisión académica y restricciones científicas pendientes, no a columnas inventadas. El documento es una versión integral para el director; no una certificación de edge ni una aprobación del jurado. Se mantienen separados el ciclo de 38 entradas sin entrenar, el piloto futuro y los experimentos históricos.

La habilidad ui-ux-pro-max orientó colores diferenciables, estilos de línea, leyendas y separación de la lámina-resumen de los gráficos individuales. No se generaron curvas con un modelo de imágenes. Los originales entregados por el autor permanecen intactos.
