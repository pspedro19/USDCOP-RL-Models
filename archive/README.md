# Archive Directory - USD/COP Trading System

This directory contains archived files, deprecated code, and historical documentation that is no longer actively used but preserved for reference.

**Last cleanup**: 2026-01-12 (V20 Consolidation - Clean Code Initiative)

---

## Recent Cleanup (2026-01-12)

### V16/V17 Legacy (`v16_v17_legacy/`)
Dataset builders and documentation from pre-V20:
- `02_build_v16_datasets.py`
- `03_build_v17_multifreq.py`
- `04_build_v17_raw.py`
- `dataset_builder_v17.py`
- `generate_dataset_v17.py`
- `test_colombia_features.py`
- Documentation (`docs/`)

### V19 Config (`v19_config/`)
Configuration files superseded by V20:
- `feature_config_v19.json`
- `v19_norm_stats.json`
- `feature_registry_v19.json`
- `normalization_stats_v19.json`

### Paper Trading Legacy (`paper_trading_legacy/`)
Older paper trading implementations:
- `paper_trading_simulation.py` (v1)
- `paper_trading_simulation_v2.py` (v2)
- `paper_trading_simulation_v3_real_macro.py` (v3)

---

## Structure

```
archive/
├── airflow-dags/           # Deprecated DAGs and factories
│   ├── deprecated/         # 15+ DAGs from v1/v2 (1.1MB)
│   ├── v3/                 # Duplicate DAGs replaced by current
│   └── legacy-factories/   # base_pipeline.py, dag_factory.py
├── configs/                # Old configuration files
├── data/                   # Archived data files
├── data-legacy/            # Legacy data formats
├── docs/                   # Deprecated documentation
│   ├── FASE_*.md           # Phase completion docs
│   ├── ADDENDUM_*.md       # Feature addendums
│   └── PLAN_ESTRATEGICO_*  # Strategic planning docs
├── infrastructure-legacy/  # Removed infrastructure
│   ├── app/                # Old FastAPI (replaced by services/)
│   ├── nginx/              # Nginx config
│   ├── pgadmin/            # PgAdmin config
│   ├── postgres/           # PostgreSQL config
│   └── prometheus/         # Prometheus config
├── init-scripts/           # Replaced SQL scripts
│   ├── 02-macro-data-schema.sql      # → database/01_core_tables.sql
│   ├── 11-realtime-inference-tables.sql  # → database/01_core_tables.sql
│   └── 12-unified-inference-schema.sql   # → database/02_inference_view.sql
├── investipy/              # InvestPy library (inactive)
├── notebooks/              # Deprecated Jupyter notebooks
├── reports/                # Historical validation reports
├── root-files/             # Files moved from project root
├── scripts/                # Deprecated scripts
└── services/               # Old service implementations
```

---

## What Was Archived

### From init-scripts/
SQL scripts replaced by new `database/schemas/` (SSOT v3.1):
- `02-macro-data-schema.sql` - Old macro_ohlcv table
- `11-realtime-inference-tables.sql` - Old inference tables
- `12-unified-inference-schema.sql` - Old unified schema

### From airflow/dags/
- **deprecated/**: 15 DAGs from pipeline v1/v2
- **v3/**: 5 duplicate DAGs (consolidation artifacts)
- **legacy-factories/**: Factory patterns no longer used

### From Project Root
- Old FastAPI app (`app/`)
- Infrastructure configs (nginx, pgadmin, postgres, prometheus)
- Validation reports (GOLD_VALIDATION_*)

### From docs/
- Phase completion documentation (FASE_0 to FASE_5)
- Old strategic plans and addendums

---

## Current Active Locations

| Purpose | Current Location |
|---------|-----------------|
| **DAGs** | `airflow/dags/` (5 active DAGs) |
| **APIs** | `services/` (6 production APIs) |
| **Database Schema** | `database/schemas/` (SSOT v3.1) |
| **Documentation** | `docs/ARQUITECTURA_INTEGRAL_V3.md` |
| **Feature Config** | `config/feature_config.json` |

---

## Policy

- Files here are **NOT actively maintained**
- Do NOT delete without verification that production works
- For **historical reference only**
- Git history preserves full file history if needed

---

## Cleanup Candidates

These can be permanently deleted after production verification:
- `airflow-dags/deprecated/` (~1.1MB)
- `infrastructure-legacy/` (configs for removed services)
- `notebooks/` (old exploration notebooks)

**Estimated savings**: ~50MB
