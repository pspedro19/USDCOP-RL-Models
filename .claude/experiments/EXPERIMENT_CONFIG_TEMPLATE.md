# Experiment Config Template

> How to create a new experiment SSOT config file.
> Each experiment = a frozen, complete `pipeline_ssot.yaml` in `config/experiments/`.

---

## Step 1: Copy baseline

```bash
cp config/experiments/v215b_baseline.yaml config/experiments/exp_{name}_{seq}.yaml
```

Naming convention: `exp_{short_name}_{sequential_number}.yaml`
- `exp_asym_001.yaml` — Asymmetric SL/TP, first attempt
- `exp_v215c_001.yaml` — V21.5c variant, first attempt
- `exp_reward_asym_001.yaml` — Reward asymmetry, first attempt

---

## Step 2: Update `_meta` section

Replace the `_meta` block with experiment-specific metadata:

```yaml
_meta:
  # Identification
  version: "X.Y.Z"                    # Increment from baseline
  experiment_id: "EXP-{NAME}-{SEQ}"   # Must match EXPERIMENT_QUEUE.md
  contract_id: "CTR-PIPELINE-SSOT-001"

  # Lineage
  based_on: "v215b_baseline.yaml"     # Parent config filename
  based_on_model: "v21.5b_temporal_features"
  based_on_performance:
    total_return: 2.51                 # V21.5b mean across 5 seeds
    sharpe_ratio: 0.321
    max_drawdown_pct: 14.33
    profit_factor: 1.009
    trades: 349
    win_rate: 51.3
    seeds_positive: "4/5"

  # Experiment definition
  variable_changed: "Describe the SINGLE variable changed and its old -> new values"
  hypothesis: "State the hypothesis: what you expect and why"
  created_at: "YYYY-MM-DD"

  # Status tracking (updated as experiment progresses)
  status: "pending"                    # pending | running | completed | failed

  # Results (populated AFTER L4 completes)
  results:
    mean_return: null
    std_return: null
    sharpe: null
    profit_factor: null
    seeds_positive: null
    bootstrap_ci_lower: null
    bootstrap_ci_upper: null
    p_value: null
    decision: null                     # SUCCESS | FAIL | INCONCLUSIVE
    completed_at: null
```

---

## Step 3: Change the ONE variable

Find the specific parameter in the config and change it. Examples:

| Experiment | Section | Parameter | Old | New |
|------------|---------|-----------|-----|-----|
| EXP-ASYM-001 | `environment` + `backtest` | `stop_loss_pct`, `take_profit_pct` | -0.04, +0.04 | -0.025, +0.06 |
| EXP-V215c-001 | `training` | `total_timesteps` | 2000000 | 1000000 |
| EXP-REWARD-ASYM-001 | `reward.pnl_transform` | `asymmetric.win_multiplier` | 1.0 | 1.5 |
| EXP-V215d-001 | `features` | Remove `rsi_21`, update `observation_dim` | 27 | 26 |

**DO NOT** change anything else. The entire point is isolating one variable.

---

## Step 4: Register in EXPERIMENT_QUEUE.md

Add the experiment to the queue with:
- Experiment ID
- Config file path
- Variable changed
- Hypothesis
- Decision framework (success/fail criteria)

---

## Step 5: Run

```bash
# Option A: Direct config path
python scripts/run_ssot_pipeline.py --config config/experiments/exp_{name}_{seq}.yaml

# Option B: Copy to active config first
cp config/experiments/exp_{name}_{seq}.yaml config/pipeline_ssot.yaml
python scripts/run_ssot_pipeline.py
```

---

## Step 6: Log results

After L4 completes:
1. Update `_meta.results` in the frozen config (only this section)
2. Update `_meta.status` to `completed` or `failed`
3. Append full results to `.claude/experiments/EXPERIMENT_LOG.md`
4. Update `.claude/experiments/EXPERIMENT_QUEUE.md` (move to COMPLETED)

---

## Checklist (before launching)

- [ ] Config file created in `config/experiments/`
- [ ] `_meta.experiment_id` matches queue
- [ ] `_meta.variable_changed` describes the ONE change
- [ ] `_meta.based_on` points to parent config
- [ ] ONLY the target variable differs from baseline (diff check)
- [ ] Registered in `EXPERIMENT_QUEUE.md`
- [ ] 5 seeds configured: [42, 123, 456, 789, 1337]
- [ ] Device matches rule (CPU for MLP, GPU for LSTM only)
