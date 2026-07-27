---
kind: roadmap
status: SUPERSEDED
version: 1.0.0
supersedes: []
last_verified: 2026-07-27
code_anchors:
  - scripts/pipeline/train_and_export_smart_simple.py
  - config/strategy_manifests/usdcop.yaml
  - .claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md
  - .claude/rules/quant-constitution.md
---

# PLAN — El doble rol de USD/COP y la contabilidad de trials por superficie

> **SUPERSEDED 2026-07-27**: absorbido por `04-CTR-QLAB-FABRIC-004.md` (§10, §16). Se
> conserva como detalle de referencia; las tareas pendientes viven en `backlog/`.

> Tercera pieza del plan de superficies. El caso más importante: en USD/COP el MISMO
> tipo de modelo (Ridge/BR) vive en AMBAS superficies — como componente interno de la
> estrategia y como panel público de forecasting. **No son el mismo producto.**

## 1. El predictor interno de la estrategia (superficie ACTION)

Es un **componente congelado de la política**, no un producto de predicción:

```text
Ridge + Bayesian Ridge → score débil → gate Hurst → sizing → TP/HS → salida viernes → PnL
```

Debe aparecer dentro del **Passport** de la estrategia:

```yaml
components:
  - component_id: usdcop_ridge_br_v5
    role: decision_input
    model_snapshot_id: ...
    code_hash: ...
    data_snapshot_id: ...
```

La estrategia completa se evalúa por: **Calmar · drawdown · DSR · PnL forward ·
slippage · consistencia paper/live**.

> Aunque el predictor tenga `R² < 0`, la estrategia puede ser útil si la combinación de
> gate, sizing y salidas agrega valor. **Esto ya está demostrado empíricamente en COP**:
> R²<0 en 2025 y 2026 y aun así +7.35%/+3.36% — el alfa es del ciclo de decisión.

## 2. El forecasting público (superficie DIAGNOSTIC)

El zoo semanal se registra como `role: diagnostic_panel`. Produce predicciones,
gráficos, métricas y comparaciones contra baseline. **No actualiza** el modelo
congelado de v11, **no reemplaza** su señal y **no toca** el allocator.

## 3. Cómo cobrar los trials (la distinción fina)

No todo forecasting diagnóstico es automáticamente gratuito.

### NO cobra trials nuevos

- Regenerar predicciones de un modelo congelado.
- Calcular métricas previamente declaradas.
- Actualizar el dashboard.
- Monitorear drift.
- Publicar nuevos períodos forward.
- Repetir la misma evaluación sin cambiar criterios.

### SÍ cobra trials

- Probar un modelo nuevo.
- Probar otro horizonte porque el anterior falló.
- Cambiar features después de ver resultados.
- **Escoger el mejor de nueve modelos.**
- Cambiar el target.
- Cambiar el cutoff.
- Buscar la métrica en la que el modelo se vea mejor.
- **Probar si un forecast se convierte en señal económica.**

## 4. Dos linajes de trials

```text
forecast_family:
  pregunta predictiva · modelos · horizontes · métricas predictivas · forecast_trial_ids

action_family:
  pregunta económica · predictor + gate + sizing + salidas · action_trial_ids
```

Cuando un predictor se incorpora a una estrategia, la **provenance** conecta ambos
linajes sin borrar la historia:

```yaml
provenance:
  forecast_trial_ids: [FT-0041, FT-0042]
  action_trial_id: AT-0113
  research_cluster: usdcop_directional_models
```

No se borra su historia predictiva. Pero **la combinación económica constituye otra
hipótesis** y debe evaluarse como tal (juez = forward de la estrategia, no el DA del
predictor).

## 5. Estado actual vs objetivo (as-built 2026-07-27)

| Pieza | Hoy | Acción |
|---|---|---|
| Predictor interno congelado con hash | ✅ parcial: `code_hash` del manifiesto cubre los fuentes | añadir bloque `components:` con model/data snapshot ids |
| Zoo como diagnostic sin tocar v11 | ✅ (pipelines separados; el zoo jamás escribe señal) | formalizar `role: diagnostic_panel` en config |
| Contabilidad de trials | ✅ un `HYPOTHESIS-REGISTRY` por activo, N=88 COP con TODO mezclado | **separar en dos ledgers** (`forecast_trial_ids` FT-xxxx / `action_trial_ids` AT-xxxx) manteniendo N total; los ~88 existentes se etiquetan retroactivamente por familia SIN cambiar el conteo |
| Regla "convertir forecast en señal = +1 trial action" | ✅ ya se practica (H-META-01 se cobró así) | escribirla como regla dura en `quant-constitution.md` §2 (requiere ADR menor) |
| Provenance FT→AT | ❌ | campo `provenance:` en pre-registros nuevos |

**Nota de compatibilidad con la constitución**: esta separación NO reduce el conteo
que deflacta el DSR — el `n_trials_total` del activo sigue siendo la suma de ambos
linajes (mirar una celda predictiva sigue quemando presupuesto del activo). Lo que
añade es TRAZABILIDAD: saber qué parte del presupuesto se gastó en predicción pura y
qué parte en hipótesis económicas, y exigir el `action_trial_id` explícito cuando un
forecast cruza la muralla.
