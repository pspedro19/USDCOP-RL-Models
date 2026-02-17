# Plan Arquitectura L0-L5: Pipeline Completo con A/B Testing

**Contract: CTR-ARCH-FINAL-001**
**Version: 1.0.0**
**Date: 2026-01-31**
**Author: Trading Team**

---

## 1. Resumen Ejecutivo

Este documento define la arquitectura completa del sistema de trading USDCOP, desde la adquisición de datos (L0) hasta la inferencia en producción (L5), incluyendo:

- **Contratos versionados** generados desde YAMLs SSOT
- **Lineage completo** de datos en cada etapa
- **A/B Testing** con backtest out-of-sample
- **Promoción de doble voto**: L4 propone, Dashboard aprueba

---

## 2. Arquitectura General

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           YAML SSOT (Single Source of Truth)                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  config/macro_variables_ssot.yaml     → Define 37 variables macro + extractors      │
│  config/date_ranges.yaml              → Define períodos train/val/test              │
│  config/experiments/*.yaml            → Define experimentos A/B                     │
│  src/core/contracts/feature_contract.py → Define 15 features (orden fijo)           │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
          │   L0: DATA      │   │   L1: FEATURES  │   │   L2: DATASET   │
          │   ACQUISITION   │   │   (Inference)   │   │   (Training)    │
          └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
                   │                     │                     │
                   │                     │                     ▼
                   │                     │            ┌─────────────────┐
                   │                     │            │   L3: TRAINING  │
                   │                     │            └────────┬────────┘
                   │                     │                     │
                   │                     │                     ▼
                   │                     │            ┌─────────────────┐
                   │                     │            │   L4: BACKTEST  │
                   │                     │            │   + PROMOTION   │
                   │                     │            │   (Primer Voto) │
                   │                     │            └────────┬────────┘
                   │                     │                     │
                   │                     │                     ▼
                   │                     │            ┌─────────────────┐
                   │                     │            │   DASHBOARD     │
                   │                     │            │   (Segundo Voto)│
                   │                     │            └────────┬────────┘
                   │                     │                     │
                   │                     ▼                     │
                   │            ┌─────────────────┐            │
                   └───────────▶│   L5: INFERENCE │◀───────────┘
                                │   (Producción)  │   modelo promovido
                                └─────────────────┘
```

---

## 3. Flujo de Datos y Lineage

### 3.1 Tabla de Lineage por Capa

| Capa | Input | Output | Hash Tracking |
|------|-------|--------|---------------|
| L0 | APIs externas | macro_daily, macro_monthly, macro_quarterly | `source_hash`, `extraction_timestamp` |
| L1 | OHLCV + macro_* | inference_features_5m | `feature_order_hash`, `norm_stats_hash` |
| L2 | OHLCV + macro_* + experiment.yaml | train/val/test.parquet | `dataset_hash`, `config_hash` |
| L3 | train.parquet + val.parquet | model.zip | `model_hash`, `dataset_hash`, `config_hash` |
| L4 | model + test.parquet | backtest_results + promotion_proposal | `backtest_hash`, `model_hash` |
| L5 | model + inference_features_5m | signal | `inference_hash`, `model_hash` |

### 3.2 Diagrama de Lineage Completo

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LINEAGE CHAIN                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

 macro_variables_ssot.yaml                    experiment.yaml
 (CTR-L0-SSOT-001)                           (CTR-EXP-XXX)
         │                                          │
         ▼                                          │
┌─────────────────┐                                 │
│ L0 EXTRACTION   │                                 │
│                 │                                 │
│ Tables:         │                                 │
│ ├─ macro_daily  │                                 │
│ ├─ macro_monthly│                                 │
│ └─ macro_quarterly                               │
│                 │                                 │
│ Hashes:         │                                 │
│ └─ extraction_hash: sha256(data + timestamp)     │
└────────┬────────┘                                 │
         │                                          │
         │              date_ranges.yaml            │
         │              (CTR-DATE-001)              │
         │                     │                    │
         ▼                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ L2 DATASET BUILDER                                              │
│                                                                 │
│ Config Hash = sha256(experiment.yaml)                           │
│                                                                 │
│ Process:                                                        │
│ 1. Load OHLCV (2023-01-01 → 2024-12-31) for TRAIN               │
│ 2. Load macro with T-1 shift (anti-leakage)                     │
│ 3. Calculate 13 features (Wilder's EMA)                         │
│ 4. Compute norm_stats on TRAIN only                             │
│ 5. Apply normalization to train/val/test                        │
│ 6. Save .parquet files                                          │
│                                                                 │
│ Output:                                                         │
│ ├─ DS_{exp_name}_train.parquet  (train: 2023-01-01 → 2024-12-31)│
│ ├─ DS_{exp_name}_val.parquet    (val: 2025-01-01 → 2025-06-30)  │
│ ├─ DS_{exp_name}_test.parquet   (test: 2025-07-01 → HOY)        │
│ ├─ norm_stats.json                                              │
│ └─ lineage.json                                                 │
│                                                                 │
│ Lineage Record:                                                 │
│ {                                                               │
│   "dataset_hash": "abc123...",                                  │
│   "config_hash": "def456...",       ← sha256(experiment.yaml)   │
│   "feature_order_hash": "ghi789...", ← from feature_contract.py │
│   "norm_stats_hash": "jkl012...",                               │
│   "date_ranges": {...},                                         │
│   "ohlcv_rows": 150000,                                         │
│   "macro_sources": ["fred", "investing", "suameca", ...]        │
│ }                                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ L3 MODEL TRAINING                                               │
│                                                                 │
│ Input:                                                          │
│ ├─ train.parquet (via XCom L2Output)                            │
│ ├─ val.parquet (for early stopping)                             │
│ └─ experiment.yaml (hyperparameters + reward config)            │
│                                                                 │
│ Process:                                                        │
│ 1. Load dataset from L2 XCom                                    │
│ 2. Create PPO environment with reward weights from YAML         │
│ 3. Train with curriculum learning (phases from YAML)            │
│ 4. Validate on val.parquet (early stopping)                     │
│ 5. Save model + artifacts                                       │
│                                                                 │
│ Output:                                                         │
│ ├─ model_{exp_name}_v{version}.zip                              │
│ ├─ config.yaml (frozen config snapshot)                         │
│ ├─ reward_config.json                                           │
│ └─ training_metrics.json                                        │
│                                                                 │
│ Lineage Record:                                                 │
│ {                                                               │
│   "model_hash": "mno345...",                                    │
│   "dataset_hash": "abc123...",      ← from L2                   │
│   "config_hash": "def456...",       ← from L2                   │
│   "reward_config_hash": "pqr678...",                            │
│   "curriculum_final_phase": "phase_3",                          │
│   "training_duration_seconds": 3600,                            │
│   "best_val_reward": 150.5,                                     │
│   "mlflow_run_id": "abc123"                                     │
│ }                                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ L4 BACKTEST + PROMOTION PROPOSAL                                │
│                                                                 │
│ Input:                                                          │
│ ├─ model.zip (from L3)                                          │
│ ├─ test.parquet (OUT-OF-SAMPLE, 2025-07-01 → HOY)               │
│ └─ success_criteria from experiment.yaml                        │
│                                                                 │
│ Process:                                                        │
│ 1. Load model from L3 XCom                                      │
│ 2. Run backtest on test.parquet (NUNCA VISTO en training)       │
│ 3. Calculate metrics: Sharpe, MaxDD, Win Rate, Profit Factor    │
│ 4. Compare vs baseline (si existe)                              │
│ 5. Evaluate success_criteria from YAML                          │
│ 6. Generate promotion_proposal                                  │
│                                                                 │
│ Success Criteria (from experiment.yaml):                        │
│ ├─ min_sharpe: 0.5                                              │
│ ├─ max_drawdown: 0.15                                           │
│ ├─ min_win_rate: 0.45                                           │
│ ├─ min_trades: 50                                               │
│ └─ improvement_threshold: 0.05  (5% mejor que baseline)         │
│                                                                 │
│ Output:                                                         │
│ ├─ backtest_results.json                                        │
│ ├─ comparison_report.json (vs baseline)                         │
│ └─ promotion_proposal.json                                      │
│                                                                 │
│ Promotion Proposal:                                             │
│ {                                                               │
│   "model_id": "exp1_curriculum_aggressive_v1_20260131",         │
│   "experiment_name": "exp1_curriculum_aggressive_v1",           │
│   "recommendation": "PROMOTE",  // or "REJECT" or "REVIEW"      │
│   "confidence": 0.85,                                           │
│   "reason": "Sharpe 1.2 > baseline 0.9, all criteria passed",   │
│   "metrics": {                                                  │
│     "sharpe_ratio": 1.2,                                        │
│     "max_drawdown": 0.12,                                       │
│     "win_rate": 0.55,                                           │
│     "profit_factor": 1.8,                                       │
│     "total_trades": 120                                         │
│   },                                                            │
│   "vs_baseline": {                                              │
│     "sharpe_improvement": "+33%",                               │
│     "drawdown_improvement": "-20%"                              │
│   },                                                            │
│   "criteria_results": {                                         │
│     "min_sharpe": "PASS (1.2 > 0.5)",                           │
│     "max_drawdown": "PASS (0.12 < 0.15)",                       │
│     "min_win_rate": "PASS (0.55 > 0.45)",                       │
│     "min_trades": "PASS (120 > 50)"                             │
│   },                                                            │
│   "requires_human_approval": true,                              │
│   "status": "PENDING_APPROVAL"                                  │
│ }                                                               │
│                                                                 │
│ Lineage Record:                                                 │
│ {                                                               │
│   "backtest_hash": "stu901...",                                 │
│   "model_hash": "mno345...",        ← from L3                   │
│   "test_dataset_hash": "vwx234...",                             │
│   "test_period": "2025-07-01 to 2026-01-31",                    │
│   "baseline_model_id": "ppo_production_v3",                     │
│   "promotion_decision": "PENDING_APPROVAL"                      │
│ }                                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ DASHBOARD (Segundo Voto - Aprobación Humana)                    │
│                                                                 │
│ UI Components:                                                  │
│ ├─ /experiments → Lista de experimentos pendientes de aprobación│
│ ├─ /experiments/{id}/review → Detalle con métricas + charts     │
│ └─ /experiments/{id}/approve → Botón de aprobación              │
│                                                                 │
│ Review Page Shows:                                              │
│ ├─ Backtest equity curve                                        │
│ ├─ Drawdown chart                                               │
│ ├─ Trade distribution                                           │
│ ├─ Metrics comparison table (vs baseline)                       │
│ ├─ L4 recommendation + confidence                               │
│ └─ Complete lineage chain (clickable)                           │
│                                                                 │
│ Actions:                                                        │
│ ├─ APPROVE → model_registry.stage = "staging" → "production"    │
│ ├─ REJECT → model stays in "staging", marked as rejected        │
│ └─ REQUEST_MORE_TESTS → trigger additional backtest periods     │
│                                                                 │
│ API Endpoint:                                                   │
│ POST /api/experiments/{id}/approve                              │
│ {                                                               │
│   "decision": "APPROVE",                                        │
│   "reviewer": "pedro@trading.com",                              │
│   "notes": "Approved after reviewing equity curve",             │
│   "promote_to_production": true                                 │
│ }                                                               │
│                                                                 │
│ On Approval:                                                    │
│ 1. Update model_registry: stage = "production"                  │
│ 2. Archive previous production model                            │
│ 3. Update L5 to use new model                                   │
│ 4. Log approval in audit_log table                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ L5 INFERENCE (Producción)                                       │
│                                                                 │
│ Loads:                                                          │
│ ├─ model from model_registry WHERE stage = 'production'         │
│ ├─ norm_stats.json (linked to model)                            │
│ └─ feature_order from feature_contract.py                       │
│                                                                 │
│ Every 5 minutes:                                                │
│ 1. L1 calculates features → inference_features_5m               │
│ 2. L5 reads inference_features_5m                               │
│ 3. L5 applies norm_stats (from approved model)                  │
│ 4. L5 runs model.predict()                                      │
│ 5. L5 outputs signal to trading_signals table                   │
│                                                                 │
│ Lineage per inference:                                          │
│ {                                                               │
│   "inference_id": "inf_20260131_120500",                        │
│   "model_id": "exp1_curriculum_aggressive_v1_20260131",         │
│   "model_hash": "mno345...",                                    │
│   "feature_hash": "xyz789...",   ← hash of input features       │
│   "signal": 0.75,                                               │
│   "confidence": 0.82,                                           │
│   "latency_ms": 15                                              │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Contratos Versionados desde YAML

### 4.1 Estructura de Versionamiento

Cada YAML genera un **contrato versionado** con hash único:

```
experiment.yaml
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ CONTRACT GENERATOR                                              │
│                                                                 │
│ Input: config/experiments/exp1_curriculum_aggressive_v1.yaml    │
│                                                                 │
│ Generated Contract:                                             │
│ {                                                               │
│   "contract_id": "CTR-EXP-exp1_curriculum_aggressive_v1",       │
│   "contract_version": "1.0.0",                                  │
│   "config_hash": "sha256(yaml_content)",                        │
│   "created_at": "2026-01-31T12:00:00Z",                         │
│   "components": {                                               │
│     "feature_contract": "v2.1.0",                               │
│     "reward_contract": "v1.0.0",                                │
│     "date_ranges_version": "1.0.0"                              │
│   },                                                            │
│   "immutable": true  // Once created, cannot be modified        │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Tabla de Contratos

| YAML Source | Contract ID | What it Defines |
|-------------|-------------|-----------------|
| `feature_contract.py` | CTR-FEAT-001 | 15 features, order, FEATURE_ORDER_HASH |
| `date_ranges.yaml` | CTR-DATE-001 | train/val/test periods |
| `macro_variables_ssot.yaml` | CTR-L0-SSOT-001 | 37 macro variables, extractors |
| `experiments/*.yaml` | CTR-EXP-{name} | Hyperparams, reward weights, curriculum |
| `quality_thresholds.yaml` | CTR-QUAL-001 | Success criteria for promotion |

### 4.3 Cómo el YAML Genera una Versión Única

```python
# src/contracts/experiment_contract.py

@dataclass
class ExperimentContract:
    """Contract generated from experiment YAML."""

    # Identity
    contract_id: str
    experiment_name: str
    experiment_version: str

    # Hashes for lineage
    config_hash: str        # sha256(yaml_content)
    feature_order_hash: str # from feature_contract.py
    reward_config_hash: str # sha256(reward section)

    # References to other contracts
    feature_contract_version: str  # e.g., "v2.1.0"
    date_ranges_version: str       # e.g., "1.0.0"

    # Frozen config snapshot
    frozen_config: Dict[str, Any]

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentContract":
        """Create contract from YAML file."""
        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        # Compute hashes
        yaml_content = yaml_path.read_bytes()
        config_hash = hashlib.sha256(yaml_content).hexdigest()[:16]

        reward_section = json.dumps(config.get("reward", {}), sort_keys=True)
        reward_hash = hashlib.sha256(reward_section.encode()).hexdigest()[:16]

        return cls(
            contract_id=f"CTR-EXP-{config['experiment']['name']}",
            experiment_name=config['experiment']['name'],
            experiment_version=config['experiment']['version'],
            config_hash=config_hash,
            feature_order_hash=FEATURE_ORDER_HASH,  # from SSOT
            reward_config_hash=reward_hash,
            feature_contract_version=config['environment'].get('feature_contract_id', 'v1.0.0'),
            date_ranges_version="1.0.0",
            frozen_config=config,
        )
```

---

## 5. Flujo Completo Paso a Paso

### 5.1 Fase 1: Data Acquisition (L0)

```
┌─────────────────────────────────────────────────────────────────┐
│ L0: DATA ACQUISITION                                            │
│ DAG: core_l0_04_macro_daily (diario 6 AM COT)                   │
│                                                                 │
│ Input:                                                          │
│ └─ config/macro_variables_ssot.yaml                             │
│                                                                 │
│ Process:                                                        │
│ 1. Para cada variable en variable_groups.daily:                 │
│    a. Obtener extractor según primary_source                    │
│    b. Extraer datos desde API                                   │
│    c. Validar rangos esperados (validation.expected_range)      │
│    d. Aplicar ffill según ffill.max_days                        │
│    e. Insertar en macro_indicators_daily                        │
│                                                                 │
│ 2. Para monthly/quarterly: similar pero con diferentes tables   │
│                                                                 │
│ Output Tables:                                                  │
│ ├─ macro_indicators_daily (18 variables)                        │
│ ├─ macro_indicators_monthly (18 variables)                      │
│ └─ macro_indicators_quarterly (4 variables)                     │
│                                                                 │
│ Anti-Leakage:                                                   │
│ - Usar schedule.publication.delay_days para T-1 shift           │
│ - Variables con leakage_risk=HIGH tienen mayor delay            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Fase 2: Dataset Building (L2)

```
┌─────────────────────────────────────────────────────────────────┐
│ L2: DATASET BUILDER                                             │
│ DAG: rl_l2_01_dataset_build (trigger manual o desde L4)         │
│                                                                 │
│ Input:                                                          │
│ ├─ dag_run.conf.experiment_name = "exp1_curriculum_aggressive"  │
│ ├─ config/experiments/exp1_curriculum_aggressive_v1.yaml        │
│ └─ config/date_ranges.yaml                                      │
│                                                                 │
│ Process:                                                        │
│ 1. Load experiment YAML → create ExperimentContract             │
│ 2. Load date_ranges.yaml → get train/val/test periods           │
│ 3. Query OHLCV: usdcop_m5_ohlcv (2023-01-01 → HOY)              │
│ 4. Query Macro: macro_indicators_daily (con T-1 shift)          │
│ 5. Merge OHLCV + Macro (ffill SOLO dentro de sesión)            │
│ 6. Calculate 13 features (CanonicalFeatureBuilder)              │
│ 7. Drop NaN rows                                                │
│ 8. Split by date_ranges:                                        │
│    ├─ train: 2023-01-01 → 2024-12-31                            │
│    ├─ val:   2025-01-01 → 2025-06-30                            │
│    └─ test:  2025-07-01 → HOY                                   │
│ 9. Compute norm_stats on TRAIN ONLY                             │
│ 10. Apply normalization to all splits                           │
│ 11. Save .parquet + lineage.json                                │
│                                                                 │
│ Output:                                                         │
│ ├─ data/pipeline/07_output/5min/                                │
│ │   ├─ DS_exp1_curriculum_aggressive_v1_train.parquet           │
│ │   ├─ DS_exp1_curriculum_aggressive_v1_val.parquet             │
│ │   ├─ DS_exp1_curriculum_aggressive_v1_test.parquet            │
│ │   ├─ DS_exp1_curriculum_aggressive_v1_norm_stats.json         │
│ │   └─ DS_exp1_curriculum_aggressive_v1_lineage.json            │
│                                                                 │
│ XCom Push (L2Output):                                           │
│ {                                                               │
│   "dataset_path": ".../DS_exp1_..._train.parquet",              │
│   "dataset_hash": "abc123",                                     │
│   "config_hash": "def456",                                      │
│   "feature_order_hash": "ghi789",                               │
│   "row_count": 150000,                                          │
│   "experiment_name": "exp1_curriculum_aggressive_v1"            │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Fase 3: Model Training (L3)

```
┌─────────────────────────────────────────────────────────────────┐
│ L3: MODEL TRAINING                                              │
│ DAG: rl_l3_01_model_training                                    │
│                                                                 │
│ Input:                                                          │
│ ├─ L2Output from XCom (dataset_path, hashes)                    │
│ └─ experiment.yaml (hyperparameters, reward config)             │
│                                                                 │
│ Process:                                                        │
│ 1. Pull L2Output via contracts                                  │
│ 2. Validate dataset_hash matches                                │
│ 3. Load experiment YAML                                         │
│ 4. Create PPO environment:                                      │
│    ├─ Load train.parquet                                        │
│    ├─ Apply reward weights from YAML                            │
│    └─ Configure curriculum phases from YAML                     │
│ 5. Train PPO:                                                   │
│    ├─ Phase 1: 75k steps (PnL focus)                            │
│    ├─ Phase 2: 100k steps (add risk penalties)                  │
│    └─ Phase 3: 225k steps (full constraints)                    │
│ 6. Validate on val.parquet (early stopping)                     │
│ 7. Save model + artifacts                                       │
│ 8. Log to MLflow                                                │
│                                                                 │
│ Output:                                                         │
│ ├─ models/exp1_curriculum_aggressive_v1/                        │
│ │   ├─ model.zip                                                │
│ │   ├─ norm_stats.json (copy from L2)                           │
│ │   ├─ config.yaml (frozen snapshot)                            │
│ │   └─ reward_config.json                                       │
│                                                                 │
│ XCom Push (L3Output):                                           │
│ {                                                               │
│   "model_path": ".../model.zip",                                │
│   "model_hash": "mno345",                                       │
│   "dataset_hash": "abc123",   ← inherited from L2               │
│   "config_hash": "def456",    ← inherited from L2               │
│   "best_reward": 150.5,                                         │
│   "mlflow_run_id": "run_abc123"                                 │
│ }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Fase 4: Backtest + Promotion Proposal (L4)

```
┌─────────────────────────────────────────────────────────────────┐
│ L4: BACKTEST + PROMOTION PROPOSAL                               │
│ DAG: rl_l4_01_backtest_promotion                                │
│                                                                 │
│ Input:                                                          │
│ ├─ L3Output from XCom (model_path, hashes)                      │
│ ├─ test.parquet (OUT-OF-SAMPLE)                                 │
│ └─ success_criteria from experiment.yaml                        │
│                                                                 │
│ Process:                                                        │
│                                                                 │
│ STEP 1: BACKTEST (Out-of-Sample)                                │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ 1. Load model from L3                                     │   │
│ │ 2. Load test.parquet (2025-07-01 → HOY)                   │   │
│ │    ⚠️  NUNCA VISTO durante training                       │   │
│ │ 3. Run simulation:                                        │   │
│ │    - Initial capital: $100,000                            │   │
│ │    - Transaction costs: 75 bps (USDCOP spread)            │   │
│ │    - Position sizing: 1.0                                 │   │
│ │ 4. Calculate metrics:                                     │   │
│ │    - Sharpe Ratio                                         │   │
│ │    - Max Drawdown                                         │   │
│ │    - Win Rate                                             │   │
│ │    - Profit Factor                                        │   │
│ │    - Total Trades                                         │   │
│ │    - Avg Trade PnL                                        │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ STEP 2: COMPARE VS BASELINE (si existe)                         │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ 1. Load baseline model (model_registry.stage='production')│   │
│ │ 2. Run same backtest on baseline                          │   │
│ │ 3. Calculate improvement %                                │   │
│ │    - sharpe_improvement = (new - old) / old               │   │
│ │    - drawdown_improvement = (old - new) / old             │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ STEP 3: EVALUATE CRITERIA                                       │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ From experiment.yaml → evaluation.success_criteria:       │   │
│ │                                                           │   │
│ │ Criteria        Threshold    Result      Status           │   │
│ │ ─────────────────────────────────────────────────────     │   │
│ │ min_sharpe      0.5          1.2         ✅ PASS          │   │
│ │ max_drawdown    0.15         0.12        ✅ PASS          │   │
│ │ min_win_rate    0.45         0.55        ✅ PASS          │   │
│ │ min_trades      50           120         ✅ PASS          │   │
│ │ improvement     5%           33%         ✅ PASS          │   │
│ │ ─────────────────────────────────────────────────────     │   │
│ │ OVERALL: ALL CRITERIA PASSED                              │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ STEP 4: GENERATE PROMOTION PROPOSAL                             │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ Decision Logic:                                           │   │
│ │                                                           │   │
│ │ IF all criteria passed AND improvement > threshold:       │   │
│ │   recommendation = "PROMOTE"                              │   │
│ │   confidence = 0.85                                       │   │
│ │                                                           │   │
│ │ ELIF all criteria passed BUT improvement < threshold:     │   │
│ │   recommendation = "REVIEW"                               │   │
│ │   confidence = 0.60                                       │   │
│ │                                                           │   │
│ │ ELSE:                                                     │   │
│ │   recommendation = "REJECT"                               │   │
│ │   confidence = 0.90                                       │   │
│ │                                                           │   │
│ │ ALWAYS:                                                   │   │
│ │   requires_human_approval = true                          │   │
│ │   status = "PENDING_APPROVAL"                             │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Output:                                                         │
│ ├─ backtest_results_{exp_name}.json                             │
│ ├─ comparison_report_{exp_name}.json                            │
│ └─ promotion_proposal_{exp_name}.json                           │
│                                                                 │
│ Database Insert:                                                │
│ INSERT INTO promotion_proposals (                               │
│   model_id, experiment_name, recommendation, confidence,        │
│   metrics, vs_baseline, criteria_results, status,               │
│   created_at, lineage                                           │
│ )                                                               │
│                                                                 │
│ Notification:                                                   │
│ → Slack: "New model ready for review: exp1_curriculum_..."      │
│ → Dashboard: Badge shows "1 pending approval"                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Fase 5: Dashboard Approval (Segundo Voto)

```
┌─────────────────────────────────────────────────────────────────┐
│ DASHBOARD: HUMAN APPROVAL                                       │
│ Route: /experiments/pending                                     │
│                                                                 │
│ UI Layout:                                                      │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ PENDING APPROVALS                                     [1] │   │
│ ├───────────────────────────────────────────────────────────┤   │
│ │                                                           │   │
│ │ ┌─────────────────────────────────────────────────────┐   │   │
│ │ │ exp1_curriculum_aggressive_v1                       │   │   │
│ │ │                                                     │   │   │
│ │ │ L4 Recommendation: PROMOTE (85% confidence)         │   │   │
│ │ │                                                     │   │   │
│ │ │ Metrics:                                            │   │   │
│ │ │ ├─ Sharpe: 1.2 (+33% vs baseline)                   │   │   │
│ │ │ ├─ Max DD: 12% (-20% vs baseline)                   │   │   │
│ │ │ ├─ Win Rate: 55%                                    │   │   │
│ │ │ └─ Trades: 120                                      │   │   │
│ │ │                                                     │   │   │
│ │ │ [View Details] [View Lineage]                       │   │   │
│ │ │                                                     │   │   │
│ │ │ [✅ APPROVE]  [❌ REJECT]  [🔄 REQUEST MORE TESTS]  │   │   │
│ │ └─────────────────────────────────────────────────────┘   │   │
│ │                                                           │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Detail View (/experiments/{id}/review):                         │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ EXPERIMENT REVIEW                                         │   │
│ │                                                           │   │
│ │ [Equity Curve]        [Drawdown Chart]                    │   │
│ │ ████████████████      ████████████████                    │   │
│ │                                                           │   │
│ │ [Trade Distribution]  [Monthly Returns]                   │   │
│ │ ████████████████      ████████████████                    │   │
│ │                                                           │   │
│ │ LINEAGE:                                                  │   │
│ │ ┌─────────────────────────────────────────────────────┐   │   │
│ │ │ L0 Macro → L2 Dataset → L3 Model → L4 Backtest      │   │   │
│ │ │                                                     │   │   │
│ │ │ Dataset:  DS_exp1_..._train.parquet                 │   │   │
│ │ │ Hash:     abc123def456                              │   │   │
│ │ │ Rows:     150,000                                   │   │   │
│ │ │ Period:   2023-01-01 → 2024-12-31                   │   │   │
│ │ │                                                     │   │   │
│ │ │ Model:    model_exp1_..._v1.zip                     │   │   │
│ │ │ Hash:     mno345pqr678                              │   │   │
│ │ │ MLflow:   run_abc123                                │   │   │
│ │ │                                                     │   │   │
│ │ │ Test:     2025-07-01 → 2026-01-31 (OOS)             │   │   │
│ │ │ Trades:   120                                       │   │   │
│ │ └─────────────────────────────────────────────────────┘   │   │
│ │                                                           │   │
│ │ [✅ APPROVE TO PRODUCTION]                                │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ On APPROVE:                                                     │
│ 1. API Call: POST /api/experiments/{id}/approve                 │
│ 2. Backend:                                                     │
│    a. UPDATE model_registry SET stage='production'              │
│       WHERE model_id = '{new_model_id}'                         │
│    b. UPDATE model_registry SET stage='archived'                │
│       WHERE stage='production' AND model_id != '{new_model_id}' │
│    c. INSERT INTO audit_log (action, model_id, reviewer, ...)   │
│ 3. Notify L5 to reload model                                    │
│ 4. Slack: "Model promoted to production by {reviewer}"          │
└─────────────────────────────────────────────────────────────────┘
```

### 5.6 Fase 6: Production Inference (L5)

```
┌─────────────────────────────────────────────────────────────────┐
│ L5: PRODUCTION INFERENCE                                        │
│ DAG: rl_l5_01_production_inference (cada 5 minutos)             │
│                                                                 │
│ Startup:                                                        │
│ 1. Query model_registry WHERE stage = 'production'              │
│ 2. Load model.zip + norm_stats.json                             │
│ 3. Verify feature_order_hash matches                            │
│ 4. Initialize inference service                                 │
│                                                                 │
│ Every 5 minutes:                                                │
│ ┌───────────────────────────────────────────────────────────┐   │
│ │ 1. L1 calculates features:                                │   │
│ │    - Query latest OHLCV from usdcop_m5_ohlcv              │   │
│ │    - Query macro from macro_indicators_daily (T-1)        │   │
│ │    - Calculate 13 features (same as L2)                   │   │
│ │    - Write to inference_features_5m                       │   │
│ │                                                           │   │
│ │ 2. L5 reads inference_features_5m                         │   │
│ │    - Verify feature_order matches contract                │   │
│ │    - Apply norm_stats from approved model                 │   │
│ │                                                           │   │
│ │ 3. L5 runs inference:                                     │   │
│ │    - model.predict(normalized_features)                   │   │
│ │    - Get action + confidence                              │   │
│ │                                                           │   │
│ │ 4. L5 outputs signal:                                     │   │
│ │    INSERT INTO trading_signals (                          │   │
│ │      timestamp, model_id, signal, confidence,             │   │
│ │      feature_hash, inference_latency_ms                   │   │
│ │    )                                                      │   │
│ └───────────────────────────────────────────────────────────┘   │
│                                                                 │
│ Model Hot Reload:                                               │
│ - Watch model_registry for changes                              │
│ - On new production model: reload without restart               │
│ - Log: "Model reloaded: {old_id} → {new_id}"                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Tablas de Base de Datos

### 6.1 Nuevas Tablas Requeridas

```sql
-- Promotion proposals from L4
CREATE TABLE promotion_proposals (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    recommendation VARCHAR(20) NOT NULL, -- 'PROMOTE', 'REJECT', 'REVIEW'
    confidence DECIMAL(5,4),
    metrics JSONB NOT NULL,
    vs_baseline JSONB,
    criteria_results JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'PENDING_APPROVAL', -- 'PENDING_APPROVAL', 'APPROVED', 'REJECTED'
    reviewer VARCHAR(255),
    reviewer_notes TEXT,
    approved_at TIMESTAMPTZ,
    lineage JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit log for all approvals
CREATE TABLE approval_audit_log (
    id SERIAL PRIMARY KEY,
    action VARCHAR(50) NOT NULL, -- 'APPROVE', 'REJECT', 'REQUEST_MORE_TESTS'
    model_id VARCHAR(255) NOT NULL,
    proposal_id INTEGER REFERENCES promotion_proposals(id),
    reviewer VARCHAR(255) NOT NULL,
    notes TEXT,
    previous_production_model VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model registry with stages
CREATE TABLE model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    model_path VARCHAR(512) NOT NULL,
    model_hash VARCHAR(64) NOT NULL,
    norm_stats_path VARCHAR(512),
    norm_stats_hash VARCHAR(64),
    config_hash VARCHAR(64),
    feature_order_hash VARCHAR(64),
    dataset_hash VARCHAR(64),
    stage VARCHAR(20) DEFAULT 'staging', -- 'staging', 'production', 'archived'
    metrics JSONB,
    lineage JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ
);

-- Experiment contracts (immutable)
CREATE TABLE experiment_contracts (
    id SERIAL PRIMARY KEY,
    contract_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    experiment_version VARCHAR(50) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    feature_order_hash VARCHAR(64) NOT NULL,
    reward_config_hash VARCHAR(64),
    frozen_config JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 7. API Endpoints del Dashboard

```yaml
# Dashboard API for experiment management

/api/experiments:
  GET:
    description: List all experiments with their status
    response:
      - id: 1
        experiment_name: exp1_curriculum_aggressive_v1
        status: PENDING_APPROVAL
        recommendation: PROMOTE
        confidence: 0.85
        created_at: 2026-01-31T12:00:00Z

/api/experiments/{id}:
  GET:
    description: Get experiment details including lineage
    response:
      id: 1
      experiment_name: exp1_curriculum_aggressive_v1
      metrics:
        sharpe_ratio: 1.2
        max_drawdown: 0.12
        win_rate: 0.55
      lineage:
        dataset_hash: abc123
        model_hash: mno345
        config_hash: def456
      backtest_results:
        equity_curve: [...]
        drawdown_series: [...]
        trades: [...]

/api/experiments/{id}/approve:
  POST:
    description: Approve experiment for production
    request:
      decision: APPROVE  # or REJECT
      notes: "Approved after reviewing equity curve"
      promote_to_production: true
    response:
      success: true
      model_id: exp1_curriculum_aggressive_v1_20260131
      new_stage: production

/api/experiments/{id}/backtest:
  POST:
    description: Request additional backtest period
    request:
      start_date: 2024-06-01
      end_date: 2024-12-31
    response:
      backtest_id: bt_123
      status: running
```

---

## 8. Resumen de Cambios Necesarios

### 8.1 DAGs a Modificar/Crear

| DAG | Acción | Cambios |
|-----|--------|---------|
| `l2_dataset_builder.py` | Ya existe | Añadir soporte para múltiples experimentos, mejorar lineage |
| `l3_model_training.py` | Ya existe | Verificar integración con contracts |
| `l4_experiment_runner.py` | **REEMPLAZAR** | Convertir en L4 Backtest + Promotion |
| `l4_backtest_validation.py` | **FUSIONAR** | Integrar en nuevo L4 |

### 8.2 Archivos Nuevos

```
airflow/dags/
├── l4_backtest_promotion.py      # NUEVO: Fusión de backtest + promotion
├── contracts/
│   └── experiment_contract.py    # NUEVO: Contract generator from YAML
│
src/
├── contracts/
│   └── experiment_contract.py    # NUEVO: ExperimentContract class
├── services/
│   └── promotion_service.py      # NUEVO: Promotion proposal logic
│
usdcop-trading-dashboard/
├── app/api/experiments/
│   ├── route.ts                  # NUEVO: List experiments
│   └── [id]/
│       ├── route.ts              # NUEVO: Get experiment details
│       └── approve/
│           └── route.ts          # NUEVO: Approve endpoint
├── components/experiments/
│   ├── PendingApprovalsList.tsx  # NUEVO: Pending approvals UI
│   ├── ExperimentReview.tsx      # NUEVO: Review page
│   └── LineageViewer.tsx         # NUEVO: Lineage visualization
```

### 8.3 Migraciones de Base de Datos

```
database/migrations/
├── 034_promotion_proposals.sql   # NUEVO: promotion_proposals table
├── 035_approval_audit_log.sql    # NUEVO: audit log table
├── 036_model_registry_stages.sql # NUEVO: Add stage column
└── 037_experiment_contracts.sql  # NUEVO: experiment_contracts table
```

---

## 9. Cronograma de Implementación

1. **Fase 1: Contracts** (2-3 días)
   - Crear `ExperimentContract` class
   - Crear migraciones de BD
   - Actualizar L2 para generar contracts

2. **Fase 2: L4 Backtest + Promotion** (3-4 días)
   - Fusionar `l4_experiment_runner.py` y `l4_backtest_validation.py`
   - Implementar `PromotionService`
   - Crear `promotion_proposals` table

3. **Fase 3: Dashboard** (3-4 días)
   - API endpoints para experiments
   - UI para pending approvals
   - Lineage viewer component

4. **Fase 4: Integration Testing** (2 días)
   - Test E2E: YAML → L2 → L3 → L4 → Dashboard → Production
   - Verify lineage chain is complete

---

## 10. Próximos Pasos

¿Deseas que proceda con:
1. Implementar `ExperimentContract` y las migraciones de BD?
2. Crear el nuevo `l4_backtest_promotion.py`?
3. Implementar los endpoints del dashboard?
4. Todo lo anterior en orden?
