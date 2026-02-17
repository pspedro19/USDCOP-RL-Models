# Rule: SSOT Versioning & Experiment Configs

> Each experiment is a frozen SSOT configuration file.
> This rule defines how experiment configs are created, stored, and consumed.

---

## Architecture

```
config/
├── pipeline_ssot.yaml                  <- ACTIVE config (points to current experiment)
│   ├── Section 1A: paths.sources       <- Primary data paths (OHLCV, macro)
│   ├── Section 1B: aux_pairs           <- Multi-pair config (MXN, BRL seeds + session)
│   ├── features / environment / ppo    <- Training config
│   └── _meta                           <- Experiment identity
├── macro_variables_ssot.yaml           <- L0: Macro variable definitions (shared, not per-experiment)
└── experiments/                        <- Frozen configs, one per experiment
    ├── v215b_baseline.yaml             <- REFERENCE baseline (NEVER modify)
    ├── exp_asym_001.yaml               <- EXP-ASYM-001: Asymmetric SL/TP
    ├── exp_v215c_001.yaml              <- EXP-V215c-001: 1M steps
    ├── exp_reward_asym_001.yaml        <- EXP-REWARD-ASYM-001: Reward asymmetry
    └── ...
```

---

## Rules

### 1. Every experiment = a complete SSOT file
- NOT a diff, NOT a partial override, NOT a YAML anchor
- A full, standalone `pipeline_ssot.yaml` that L2/L3/L4 can consume directly
- Reason: reproducibility. Anyone can re-run any experiment with just the config file.

### 2. Naming convention
- Filename: `{experiment_id_lowercase}.yaml` (e.g., `exp_asym_001.yaml`)
- `_meta.experiment_id`: Must match EXPERIMENT_QUEUE.md ID (e.g., `EXP-ASYM-001`)
- `_meta.version`: Semantic version (e.g., `4.1.0`)
- `_meta.based_on`: Parent config filename (e.g., `v215b_baseline.yaml`)

### 3. The baseline is sacred
- `v215b_baseline.yaml` is the reference config for V21.5b (best model to date)
- It is NEVER modified after creation
- All experiments derive from it by copying and changing ONE variable
- If a new baseline is established, create a new file (e.g., `v215c_baseline.yaml`)

### 4. Required `_meta` fields for experiment configs
```yaml
_meta:
  version: "4.1.0"
  experiment_id: "EXP-ASYM-001"
  contract_id: "CTR-PIPELINE-SSOT-001"
  based_on: "v215b_baseline.yaml"
  based_on_model: "v21.5b_temporal_features"
  based_on_performance:
    total_return: 2.51
    sharpe_ratio: 0.321
    seeds_positive: "4/5"
  variable_changed: "SL/TP ratio: SL -4% -> -2.5%, TP +4% -> +6%"
  hypothesis: "Tighter SL + wider TP improves PF via avg_win >> avg_loss"
  created_at: "2026-02-11"
  status: "running"           # pending | running | completed | failed
  results:                    # Populated AFTER L4
    mean_return: null
    sharpe: null
    seeds_positive: null
    decision: null
```

### 5. How to create a new experiment config
```bash
# 1. Copy baseline
cp config/experiments/v215b_baseline.yaml config/experiments/exp_new_001.yaml

# 2. Edit ONLY the variable under test + _meta section
# 3. Register in EXPERIMENT_QUEUE.md with config path
# 4. Run: python scripts/run_ssot_pipeline.py --config config/experiments/exp_new_001.yaml
```

### 6. Config lifecycle
```
PENDING ─── config created, registered in queue
   │
RUNNING ─── training started, config is FROZEN
   │
COMPLETED ── L4 results logged, _meta.results populated
   │
   └── PROMOTED (optional) ── becomes new pipeline_ssot.yaml if it wins
```

### 7. What changes vs what stays constant

**Changes per experiment** (the ONE variable):
- `_meta` section (experiment ID, hypothesis, etc.)
- The specific parameter being tested (e.g., `environment.stop_loss_pct`)

**Stays constant** (inherited from baseline):
- Feature definitions (`features.market_features`, `features.state_features`)
- Data splits (`date_ranges`)
- Data sources (`paths.sources`)
- Auxiliary pairs config (`aux_pairs`) — unless the experiment IS about cross-pair features
- Everything NOT explicitly listed as the variable under test

### 7b. Multi-pair data section (`aux_pairs`)

The `aux_pairs` section in `pipeline_ssot.yaml` defines auxiliary FX pairs (MXN, BRL)
for cross-pair feature experiments. Key rules:

- `aux_pairs.enabled: false` by default (only enabled for EXP-CROSS-PAIR experiments)
- Each pair specifies its seed file path and symbol name
- Session window (timezone, start, end) is shared across all pairs
- When creating experiment configs, COPY the entire `aux_pairs` section from baseline
- See `.claude/rules/l0-data-governance.md` for data ingestion rules and timezone handling

### 8. Active config (`pipeline_ssot.yaml`)
- The root `config/pipeline_ssot.yaml` is a COPY of the currently running experiment
- When an experiment is promoted (wins), its config becomes the new `pipeline_ssot.yaml`
- `pipeline_ssot.yaml` is what `run_ssot_pipeline.py` reads by DEFAULT
- To run a specific experiment: `--config config/experiments/exp_asym_001.yaml`

---

## Integration with pipeline runner

The pipeline runner (`scripts/run_ssot_pipeline.py`) should:
1. Accept `--config` argument (default: `config/pipeline_ssot.yaml`)
2. Load the specified SSOT config
3. Log the experiment ID from `_meta.experiment_id` at startup
4. Save the config alongside model artifacts for full reproducibility
