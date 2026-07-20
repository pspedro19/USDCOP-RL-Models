---
kind: as-built
status: IMPLEMENTED
contract: CTR-SSOT-LIFECYCLE-001
version: 2.0.0
last_verified: 2026-07-20
supersedes: []
code_anchors:
  - config/pipeline_ssot.yaml
  - config/execution/smart_simple_v1.yaml
  - scripts/pipeline/run_ssot_pipeline.py
---
# SDD Spec: Ciclo de vida de configs SSOT (referencia)

> **Responsibility**: estructura de `config/experiments/`, campos `_meta`, cómo crear un config,
> y qué cambia vs qué permanece constante. Las **invariantes** están en
> `../../rules/ssot-versioning.md`.

---

## 1. Estructura

```
config/
├── pipeline_ssot.yaml            <- config ACTIVO (copia del experimento en curso)
├── macro_variables_ssot.yaml     <- L0: definiciones macro (compartido)
└── experiments/                  <- configs congelados, uno por experimento
    ├── v215b_baseline.yaml       <- baseline de REFERENCIA (nunca se modifica)
    ├── exp_asym_001.yaml
    └── ...
```

> **Nota (2026-04-06)**: la estrategia H5 Smart Simple usa `config/execution/smart_simple_v1.yaml`
> como su SSOT (no `pipeline_ssot.yaml`, que es del track RL). Añadidos v2.0: `regime_gate`,
> `dynamic_leverage`, `effective_portfolio_cap_pct`, `retraining.frequency: weekly`.

## 2. Campos `_meta` requeridos

```yaml
_meta:
  version: "4.1.0"
  experiment_id: "EXP-ASYM-001"          # debe coincidir con EXPERIMENT_QUEUE.md
  contract_id: "CTR-PIPELINE-SSOT-001"
  based_on: "v215b_baseline.yaml"
  based_on_performance: {total_return: 2.51, sharpe_ratio: 0.321, seeds_positive: "4/5"}
  variable_changed: "SL/TP ratio: SL -4% -> -2.5%, TP +4% -> +6%"
  hypothesis: "SL más ajustado + TP más amplio mejora PF vía avg_win >> avg_loss"
  created_at: "2026-02-11"
  status: "running"                       # pending | running | completed | failed
  results: {mean_return: null, sharpe: null, seeds_positive: null, decision: null}
```

## 3. Crear un experimento

```bash
cp config/experiments/v215b_baseline.yaml config/experiments/exp_new_001.yaml
# editar SOLO la variable bajo test + la sección _meta
# registrar en EXPERIMENT_QUEUE.md
python scripts/pipeline/run_ssot_pipeline.py --config config/experiments/exp_new_001.yaml
```

## 4. Ciclo de vida

```
PENDING ── config creado, registrado en la cola
   │
RUNNING ── training arrancó, el config queda CONGELADO
   │
COMPLETED ── resultados L4 logueados, _meta.results poblado
   │
   └── PROMOTED (opcional) ── se convierte en el nuevo pipeline_ssot.yaml
```

## 5. Qué cambia vs qué permanece

**Cambia** (la ÚNICA variable): la sección `_meta` y el parámetro bajo test.

**Permanece** (heredado del baseline): definiciones de features, splits de fechas, rutas de datos,
`aux_pairs` (salvo que el experimento SEA sobre cross-pair), y todo lo no listado explícitamente
como variable bajo test.

### `aux_pairs`

Define pares auxiliares (MXN, BRL) para experimentos cross-pair. `enabled: false` por defecto.
La ventana de sesión es compartida entre pares. Al crear un config, **copia la sección entera**
del baseline. Ver `../../rules/data-governance.md` para ingesta y timezones.

## 6. Integración con el runner

`scripts/pipeline/run_ssot_pipeline.py` acepta `--config` (default `config/pipeline_ssot.yaml`),
loguea `_meta.experiment_id` al arrancar y guarda el config junto a los artefactos del modelo para
reproducibilidad total.

---

## Cross-References

| Concern | Doc |
|---------|-----|
| Invariantes SSOT (auto-cargadas) | `../../rules/ssot-versioning.md` |
| Protocolo de experimentos | `../../rules/experiment-protocol.md` |
| Anti-selección / trials / DSR | `../../rules/quant-constitution.md` |
