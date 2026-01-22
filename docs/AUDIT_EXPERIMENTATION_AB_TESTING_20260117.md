# Auditoría de Experimentación y A/B Testing
## USD/COP RL Trading System

**Fecha:** 2026-01-17
**Versión:** 1.0
**Total Preguntas:** 100
**Score Final:** 65/100 (65%)

---

## Resumen Ejecutivo

Esta auditoría evalúa la capacidad del sistema para rastrear experimentos de manera completa, desde la configuración hasta las métricas, y comparar entre experimentos para A/B testing.

### Scores por Categoría

| Categoría | Preguntas | Score | Porcentaje | Estado |
|-----------|-----------|-------|------------|--------|
| EXP: Estructura de Experimentos | 20 | 8/20 | 40% | 🔴 Crítico |
| DST: Configuración de Dataset | 15 | 10/15 | 67% | 🟡 Parcial |
| HYP: Hiperparámetros | 15 | 14/15 | 93% | 🟢 Excelente |
| TRACE: Trazabilidad | 20 | 17/20 | 85% | 🟢 Bueno |
| COMP: Comparación A/B | 20 | 8/20 | 40% | 🔴 Crítico |
| REPRO: Reproducibilidad | 10 | 8/10 | 80% | 🟢 Bueno |
| **TOTAL** | **100** | **65/100** | **65%** | 🟡 Parcial |

### Nivel de Cumplimiento

```
████████████████░░░░░░░░  65% - PARCIALMENTE IMPLEMENTADO
```

---

## Parte 1: Estructura del Experimento (EXP) - 8/20 (40%)

### Fortalezas Identificadas
- ✅ **EXP-01**: `params.yaml` como configuración central
- ✅ **EXP-02**: Incluye dataset, features, hyperparams, training config
- ✅ **EXP-03**: Formato YAML versionable en git
- ⚠️ **EXP-04**: Schemas Pydantic parciales (TrainingConfig, pero no ExperimentConfig completo)
- ⚠️ **EXP-05**: experiment_name generado con timestamp (ppo_YYYYMMDD)

### Brechas Críticas
- ❌ **EXP-06**: experiment_id NO incluye hash del config
- ❌ **EXP-07**: NO hay experiment_hash determinístico
- ❌ **EXP-08**: Configs idénticos NO producen mismo hash
- ❌ **EXP-09**: NO hay parent_experiment_id para lineage
- ❌ **EXP-10**: NO se puede ver árbol de experimentos derivados
- ❌ **EXP-11**: NO existe directorio `experiments/`
- ❌ **EXP-12**: NO hay estructura `experiments/{id}/config.yaml`
- ❌ **EXP-13**: NO existe `experiments/baseline/config.yaml`
- ❌ **EXP-16-17**: NO hay tags git para experimentos
- ❌ **EXP-18-19**: NO hay README de experimentos
- ❌ **EXP-20**: NO hay proceso de archivar experimentos

### Archivos Relevantes
- `params.yaml` - Configuración central
- `src/config/ppo_config.py` - Configuración PPO
- `src/training/train_ssot.py` - TrainingConfig dataclass

---

## Parte 2: Configuración de Dataset (DST) - 10/15 (67%)

### Fortalezas Identificadas
- ✅ **DST-05**: Lista explícita de 13 features en params.yaml
- ✅ **DST-07**: Orden de features enforced por FEATURE_ORDER
- ✅ **DST-08**: train_ratio: 0.7, val_ratio: 0.15 definidos
- ✅ **DST-09**: Splits temporales (no random) implementados
- ✅ **DST-10**: normalize.method: zscore configurado
- ✅ **DST-12**: Stats normalization calculados SOLO en train
- ✅ **DST-13**: norm_stats.json guardado como artifact

### Brechas Identificadas
- ❌ **DST-01**: NO hay sección `dataset:` explícita (usa `prepare:`)
- ❌ **DST-02**: NO hay dataset_version o dataset_hash en config
- ❌ **DST-03**: NO hay dvc_tag en config
- ⚠️ **DST-04**: Date range generado en runtime, no en config
- ⚠️ **DST-06**: Features definidos en 3 lugares (no SSOT único)
- ⚠️ **DST-14**: Validación silenciosa (no error si feature falta)
- ❌ **DST-15**: NO hay documentación de dataset versions

### Archivos Relevantes
- `params.yaml:19-46` - Sección prepare
- `config/norm_stats.json` - Stats de normalización
- `src/core/contracts/feature_contract.py` - FEATURE_ORDER

---

## Parte 3: Hiperparámetros (HYP) - 14/15 (93%)

### Implementación Excelente
- ✅ **HYP-01**: Sección hyperparameters completa en PPO_CONFIG
- ✅ **HYP-02-10**: Todos los hiperparámetros PPO presentes:
  - learning_rate: 3e-4
  - batch_size: 64
  - n_epochs: 10
  - gamma: 0.90/0.99
  - clip_range: 0.2
  - ent_coef: 0.05
  - vf_coef: 0.5
  - gae_lambda: 0.95
  - max_grad_norm: 0.5
- ✅ **HYP-11**: Network architecture: pi=[256,256], vf=[256,256]
- ✅ **HYP-12**: activation_fn: "tanh"
- ✅ **HYP-13**: random_seed: 42 con set_reproducible_seeds()
- ✅ **HYP-14**: Todos tienen valores por defecto documentados
- ✅ **HYP-15**: validate_config() con validación de rangos

### Única Brecha Menor
- ⚠️ Inconsistencia de valores entre configs (learning_rate 3e-4 vs 1e-4)

### Archivos Relevantes
- `src/config/ppo_config.py` - PPO_CONFIG, POLICY_KWARGS
- `src/training/train_ssot.py` - TrainingConfig, validate_config()
- `params.yaml:63-114` - Sección train

---

## Parte 4: Trazabilidad (TRACE) - 17/20 (85%)

### Fortalezas (Trazabilidad Hacia Abajo)
- ✅ **TRACE-01**: Config recuperable vía MLflow artifacts
- ✅ **TRACE-02**: dataset_hash logueado como parámetro
- ✅ **TRACE-03**: feature_order_hash logueado
- ✅ **TRACE-04**: Todos hyperparams logueados (hp_*)
- ✅ **TRACE-05**: norm_stats.json como artifact
- ✅ **TRACE-06**: Modelo .zip en MLflow artifacts
- ✅ **TRACE-07**: Todas las métricas logueadas
- ⚠️ **TRACE-08**: Logs de training parciales (callback-based)
- ⚠️ **TRACE-09**: Git commit NO logueado automáticamente

### Fortalezas (Trazabilidad Hacia Arriba)
- ✅ **TRACE-11**: model_version → experiment_id vía registry
- ✅ **TRACE-12**: Búsqueda por dataset_hash en MLflow
- ✅ **TRACE-13**: Búsqueda por feature_order_hash
- ✅ **TRACE-14**: Búsqueda por hiperparámetro
- ✅ **TRACE-15**: Búsqueda por métrica (sharpe > 1.5)
- ⚠️ **TRACE-16**: NO hay script trace_experiment.py dedicado
- ✅ **TRACE-17**: API lineage.py muestra árbol completo
- ❌ **TRACE-18**: NO hay exportación visual (mermaid, graphviz)
- ✅ **TRACE-19**: MLflow tiene todos los params
- ✅ **TRACE-20**: config.yaml como artifact

### Archivos Relevantes
- `scripts/train_with_mlflow.py` - Logging completo
- `scripts/reproduce_dataset_from_run.py` - Reproducción dataset
- `services/inference_api/routers/lineage.py` - API lineage

---

## Parte 5: Comparación A/B (COMP) - 8/20 (40%)

### Lo Que Existe (Fundamentos)
- ✅ **COMP-10**: MLflow UI permite comparar runs
- ✅ **COMP-18**: Tests estadísticos en ab_statistics.py:
  - Chi-square para win rates
  - Welch's t-test para Sharpe
  - Bootstrap confidence intervals
  - Cohen's d effect size
  - Bayesian A/B testing
- ✅ **COMP-19**: Tests consideran varianza (equal_var=False)
- ⚠️ **COMP-13**: relative_difference calculado pero no expuesto
- ⚠️ **COMP-17**: to_dict() exporta pero no markdown/HTML

### Lo Que Falta (Crítico)
- ❌ **COMP-01-09**: NO existe compare_experiments.py
- ❌ **COMP-02-07**: NO hay diff de configs lado a lado
- ❌ **COMP-11-12**: NO hay tabla comparativa de métricas
- ❌ **COMP-14**: NO se pueden comparar 3+ experimentos
- ❌ **COMP-15-16**: NO hay gráficas de equity/drawdown
- ❌ **COMP-20**: NO hay documentación de interpretación

### Infraestructura Existente No Expuesta
- `src/inference/ab_statistics.py` (538 líneas) - Módulo completo
- `src/inference/shadow_pnl.py` - Shadow mode para comparación
- `src/inference/model_router.py` - Champion/shadow execution

---

## Parte 6: Reproducibilidad (REPRO) - 8/10 (80%)

### Implementación Robusta
- ✅ **REPRO-01**: reproduce_dataset_from_run.py existe
- ✅ **REPRO-02**: Descarga config original vía MLflow
- ✅ **REPRO-03**: DVC checkout implementado
- ⚠️ **REPRO-04**: requirements.txt sin lock file exacto
- ✅ **REPRO-05**: Training ejecutable con config original
- ❌ **REPRO-06**: NO hay validación ±5% de métricas
- ✅ **REPRO-07**: tests/integration/test_determinism.py
- ✅ **REPRO-08**: Weekly CI validation (cron Sunday 2AM)
- ⚠️ **REPRO-09**: Alertas solo en GitHub (no Slack/email)
- ✅ **REPRO-10**: docs/REPRODUCIBILITY.md completo

### Archivos Relevantes
- `scripts/reproduce_dataset_from_run.py` (1,381 líneas)
- `tests/integration/test_determinism.py` (336 líneas)
- `.github/workflows/dvc-validate.yml` - Weekly validation
- `docs/REPRODUCIBILITY.md` (469 líneas)

---

## Las 10 Preguntas Más Críticas

| # | ID | Pregunta | Estado | Impacto |
|---|-----|----------|--------|---------|
| 1 | EXP-01 | ¿Config ÚNICO define experimento? | ✅ YES | - |
| 2 | TRACE-01 | ¿Puedo obtener config de experiment_id? | ✅ YES | - |
| 3 | TRACE-03 | ¿Puedo obtener lista exacta de features? | ✅ YES | - |
| 4 | DST-06 | ¿Features config es SSOT? | ⚠️ PARTIAL | Medio |
| 5 | COMP-01 | ¿Existe comando comparar experimentos? | ❌ NO | **Alto** |
| 6 | COMP-03 | ¿Diff resalta QUÉ cambió? | ❌ NO | **Alto** |
| 7 | COMP-13 | ¿Se calcula % mejora/degradación? | ⚠️ EXISTS | Medio |
| 8 | REPRO-01 | ¿Comando reproducir experimento? | ✅ YES | - |
| 9 | TRACE-11 | ¿model_version → experiment_id? | ✅ YES | - |
| 10 | TRACE-09 | ¿Puedo obtener git commit exacto? | ⚠️ PARTIAL | Medio |

**Críticos sin implementar: 2/10**

---

## Plan de Remediación

### Prioridad 1: Comparación de Experimentos (Alto Impacto)

```bash
# Crear script de comparación
scripts/compare_experiments.py --exp-a X --exp-b Y
```

**Componentes necesarios:**
1. CLI wrapper para ab_statistics.py
2. Diff de configs lado a lado
3. Tabla comparativa de métricas
4. Cálculo de % mejora/degradación
5. Exportación a markdown

**Estimación:** 4-6 horas

### Prioridad 2: Estructura de Experimentos (Fundacional)

```
experiments/
├── baseline/
│   └── config.yaml
├── exp_20260117_15features/
│   └── config.yaml
└── README.md  # Índice y resultados
```

**Componentes necesarios:**
1. Directorio experiments/
2. Schema ExperimentConfig Pydantic
3. experiment_hash determinístico
4. parent_experiment_id para lineage
5. Git tags para experimentos importantes

**Estimación:** 3-4 horas

### Prioridad 3: Dataset SSOT

```yaml
# params.yaml - Nueva sección dataset:
dataset:
  version: "v2.0.0"
  dvc_tag: "dataset-v2.0.0"
  hash: "sha256:abc123..."
  date_range:
    train_start: "2020-01-01"
    train_end: "2024-06-30"
```

**Estimación:** 2-3 horas

### Prioridad 4: Mejoras Menores

1. Loguear git commit automáticamente en MLflow
2. requirements.lock con versiones exactas
3. Validación ±5% de métricas en reproducción
4. Alertas Slack para fallos de reproducibilidad

**Estimación:** 2-3 horas

---

## Archivos a Crear

| Archivo | Propósito | Prioridad |
|---------|-----------|-----------|
| `scripts/compare_experiments.py` | Comparación A/B | P1 |
| `experiments/README.md` | Índice de experimentos | P2 |
| `experiments/baseline/config.yaml` | Experimento base | P2 |
| `src/core/schemas/experiment_config.py` | Schema Pydantic | P2 |
| `scripts/trace_experiment.py` | Trazabilidad completa | P2 |

## Archivos a Modificar

| Archivo | Cambio | Prioridad |
|---------|--------|-----------|
| `params.yaml` | Agregar sección dataset: | P3 |
| `scripts/train_with_mlflow.py` | Loguear git commit | P4 |
| `requirements.txt` → `requirements.lock` | Pinear versiones | P4 |

---

## Conclusión

El sistema tiene **fundamentos sólidos** para experimentación:
- ✅ Hiperparámetros bien gestionados (93%)
- ✅ Trazabilidad MLflow robusta (85%)
- ✅ Reproducibilidad documentada (80%)

Pero tiene **brechas críticas** en:
- ❌ Comparación de experimentos (40%) - No hay herramientas
- ❌ Estructura de experimentos (40%) - No hay organización

**Recomendación:** Implementar compare_experiments.py y estructura experiments/ para habilitar A/B testing efectivo.

---

*Auditoría completada: 2026-01-17*
*Metodología: 6 agentes paralelos analizando 100 preguntas*
*Auditor: Claude Code Assistant*
