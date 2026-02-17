# Rule: Experiment Protocol

> Governs ALL experiment execution. No exceptions.
> Each experiment = a frozen SSOT config + hypothesis + statistical validation.

---

## Phase 1: BEFORE Starting an Experiment

1. **CHECK QUEUE**: Is there a matching experiment ID in `EXPERIMENT_QUEUE.md`?
   - If NO: Ask the user to define the experiment hypothesis first
   - If YES: Follow the experiment spec exactly

2. **VERIFY SSOT CONFIG**: Every experiment MUST have a frozen config file
   - Location: `config/experiments/{experiment_id}.yaml`
   - Naming: lowercase, e.g., `exp_asym_001.yaml`, `exp_v215c_001.yaml`
   - The config MUST be a complete `pipeline_ssot.yaml` — not a diff or partial override
   - The baseline config (`v215b_baseline.yaml`) is the reference; all experiments derive from it

3. **STATE THE PROTOCOL** before training:
   - Experiment ID (e.g., "Running EXP-ASYM-001")
   - Single variable being changed (e.g., "SL/TP ratio")
   - What is held constant (e.g., "Everything else = V21.5b baseline")
   - Config file path (e.g., `config/experiments/exp_asym_001.yaml`)
   - Confirm config matches the experiment spec in EXPERIMENT_QUEUE.md

---

## Phase 2: DURING Training

4. **RUN 5 SEEDS**: [42, 123, 456, 789, 1337]. No exceptions.
   - Use `MultiSeedTrainer` or sequential `run_ssot_pipeline.py` calls
   - Pass config via `--config config/experiments/{experiment_id}.yaml`

5. **NO SCOPE CREEP**:
   - Never "add X while we're at it"
   - If you discover a bug, fix it as a separate commit — do NOT re-run the experiment
   - If the config needs adjustment mid-experiment, ABORT and start a new experiment ID

---

## Phase 3: AFTER L4 Results

6. **REPORT** in standard format — append to `EXPERIMENT_LOG.md`:

   **Per-seed table (MANDATORY)**:
   ```
   | Seed | Return% | Sharpe | WR% | PF | MaxDD% | Trades | AvgBars |
   ```

   **Aggregate metrics (MANDATORY)**:
   - Mean return +/- std across seeds
   - Seeds with positive return: X/5
   - Bootstrap 95% CI of mean return (10,000 samples)
   - t-test: H0 = mean return is 0, report p-value
   - Comparison vs V21.5b baseline (+2.51% mean, 4/5 seeds)
   - Comparison vs buy-and-hold (-14.66%)
   - Comparison vs random agent mean (-4.12%)

7. **STATISTICAL VALIDATION** before declaring success:
   - >= 3/5 seeds with positive return
   - Bootstrap 95% CI excludes zero
   - Profit factor > 1.05 (1.00-1.02 is noise)
   - If p > 0.05: "NOT statistically significant"

8. **UPDATE QUEUE**: After logging results, update `EXPERIMENT_QUEUE.md`:
   - Move completed experiment to COMPLETED section
   - Evaluate decision framework to determine next experiment
   - Update the frozen SSOT config's `_meta.results` section

---

## Hard Rules (NEVER violate)

### Rule 1: ONE variable per experiment
Never change more than ONE major variable at a time. **Major variables**:
- Action space (continuous vs discrete)
- Model architecture (MLP vs LSTM)
- Feature set (adding/removing features)
- Reward function (new components or weights)
- Hyperparameters (ent_coef, lr, gamma, etc.)
- Stop levels (SL/TP percentages)
- Position sizing (fixed vs Kelly)

If the user asks to change multiple: push back with "That's N variables. Which ONE should we test first?" and propose splitting into N experiments. If user insists, log as "Rule 1 violated" in the experiment log.

### Rule 2: ONE experiment = ONE frozen SSOT
- Each experiment gets its own complete `pipeline_ssot.yaml` in `config/experiments/`
- Once training starts, the config is FROZEN — no modifications allowed
- The `_meta` section must identify the experiment and the single variable changed
- Baseline config (`v215b_baseline.yaml`) is never modified

### Rule 3: Eval reward != OOS performance
NEVER select "best model" by eval reward alone. Proven: seed 456 (eval=131) lost -20.6%, seed 1337 (eval=111) gained +9.6%.

### Rule 4: CPU for PPO MlpPolicy, GPU for RecurrentPPO only
RTX 3050 laptop throttles heavily. PPO MlpPolicy is faster on CPU (~300 FPS vs 65-187 GPU).
