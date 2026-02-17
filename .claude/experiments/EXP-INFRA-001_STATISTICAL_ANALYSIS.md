# EXP-INFRA-001: Statistical Baselines & Benchmarks

## RUN THIS BEFORE ANY MORE TRAINING

This experiment requires NO training. It analyzes existing data and results to establish
whether the project is even on the right track.

---

## DELIVERABLE: `scripts/statistical_analysis.py`

Create a single script that produces ALL of the following analyses and saves results to
`results/statistical_analysis/`. The script should run in under 5 minutes.

### ANALYSIS 1: Buy-and-Hold Benchmark

```python
"""
Calculate buy-and-hold return for USDCOP over the test period.
This is the simplest possible "strategy" — just hold USDCOP.

Load test data from: data/pipeline/07_output/5min/DS_production_test.parquet
Use the 'close' column for USDCOP price.
"""

def buy_and_hold_benchmark(test_df):
    """
    Returns:
        dict with: return_pct, start_price, end_price, period
    """
    start_price = test_df['close'].iloc[0]
    end_price = test_df['close'].iloc[-1]
    return_pct = (end_price - start_price) / start_price * 100
    
    # Also compute max drawdown of buy-and-hold
    cummax = test_df['close'].cummax()
    drawdown = (cummax - test_df['close']) / cummax
    max_dd = drawdown.max() * 100
    
    return {
        "strategy": "buy_and_hold_long",
        "return_pct": return_pct,
        "max_dd_pct": max_dd,
        "start_price": start_price,
        "end_price": end_price,
    }
    # Also compute sell-and-hold (short USDCOP = long COP):
    # return_pct_short = -return_pct
```

### ANALYSIS 2: SMA Crossover Baseline

```python
"""
Simple Moving Average crossover on 5-min bars.
SMA(20) > SMA(50) → Long USDCOP
SMA(20) < SMA(50) → Short USDCOP
Apply same SL/TP/costs as the RL agent.

This answers: "Can a 3-line strategy beat our RL agent?"
"""

def sma_crossover_benchmark(test_df, sma_fast=20, sma_slow=50, 
                             sl_pct=0.04, tp_pct=0.04, 
                             cost_bps=1.0, min_hold_bars=25):
    """
    Simulate SMA crossover strategy with same risk parameters as RL agent.
    
    Returns:
        dict with: return_pct, sharpe, win_rate, profit_factor, n_trades, max_dd
    """
    # Calculate SMAs
    sma_fast_series = test_df['close'].rolling(sma_fast).mean()
    sma_slow_series = test_df['close'].rolling(sma_slow).mean()
    
    # Generate signals
    signal = np.where(sma_fast_series > sma_slow_series, 1, -1)
    
    # Simulate with same SL/TP/min_hold as RL agent
    # ... (implement trade-by-trade simulation)
    pass
```

### ANALYSIS 3: Random Agent Baseline

```python
"""
Random agent: enters random long/short with same probability as observed in RL agent.
Same SL/TP/costs/min_hold. Run 1000 simulations.

This answers: "Is the RL agent better than random?"
"""

def random_agent_baseline(test_df, n_simulations=1000, 
                           entry_prob=0.05,  # ~380 trades per year at this rate
                           sl_pct=0.04, tp_pct=0.04,
                           cost_bps=1.0, min_hold_bars=25,
                           seed=42):
    """
    Run N simulations of random entry agent.
    
    Returns:
        dict with: 
            mean_return, std_return, p5_return, p50_return, p95_return,
            pct_positive (what % of random simulations are profitable)
    """
    np.random.seed(seed)
    results = []
    
    for sim in range(n_simulations):
        # Random entries: at each bar, with probability entry_prob, 
        # enter long or short (50/50)
        # Apply same SL/TP/costs/min_hold
        sim_return = simulate_random_trades(test_df, entry_prob, sl_pct, tp_pct, cost_bps, min_hold_bars)
        results.append(sim_return)
    
    results = np.array(results)
    return {
        "strategy": "random_agent",
        "n_simulations": n_simulations,
        "mean_return_pct": np.mean(results),
        "std_return_pct": np.std(results),
        "p5_return_pct": np.percentile(results, 5),
        "p50_return_pct": np.percentile(results, 50),
        "p95_return_pct": np.percentile(results, 95),
        "pct_profitable": np.mean(results > 0) * 100,
    }
```

### ANALYSIS 4: Bootstrap Confidence Intervals

```python
"""
Bootstrap 95% CI on trade-level PnL for each model version.
Load trades from L4 results.

This answers: "Is the return statistically different from zero?"
"""

def bootstrap_confidence_interval(trade_pnls, n_bootstrap=10000, confidence=0.95, seed=42):
    """
    Compute bootstrap CI for mean trade PnL.
    
    Args:
        trade_pnls: array of individual trade PnL percentages
    
    Returns:
        dict with: mean, ci_lower, ci_upper, p_value_vs_zero
    """
    np.random.seed(seed)
    n_trades = len(trade_pnls)
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(trade_pnls, size=n_trades, replace=True)
        boot_means.append(np.mean(sample))
    
    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_means, alpha/2 * 100)
    ci_upper = np.percentile(boot_means, (1 - alpha/2) * 100)
    
    # P-value: what fraction of bootstrap means are on the opposite side of zero?
    if np.mean(trade_pnls) > 0:
        p_value = np.mean(boot_means <= 0)
    else:
        p_value = np.mean(boot_means >= 0)
    
    return {
        "mean_pnl_pct": float(np.mean(trade_pnls)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_trades": n_trades,
        "n_bootstrap": n_bootstrap,
    }
```

### ANALYSIS 5: Long vs Short Breakdown

```python
"""
Separate trades by direction and compute metrics for each.

This answers: "Does the model predict direction or just follow regime?"
"""

def long_short_breakdown(trades_df):
    """
    Split trades into long and short, compute metrics for each.
    
    Args:
        trades_df: DataFrame with columns [direction, pnl_pct, bars_held, ...]
    
    Returns:
        dict with: long_metrics, short_metrics
    """
    long_trades = trades_df[trades_df['direction'] == 1]
    short_trades = trades_df[trades_df['direction'] == -1]
    
    def compute_metrics(trades):
        if len(trades) == 0:
            return {"n_trades": 0}
        pnls = trades['pnl_pct'].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        return {
            "n_trades": len(trades),
            "win_rate": len(wins) / len(trades) * 100,
            "avg_win_pct": np.mean(wins) if len(wins) > 0 else 0,
            "avg_loss_pct": np.mean(losses) if len(losses) > 0 else 0,
            "profit_factor": abs(np.sum(wins) / np.sum(losses)) if len(losses) > 0 and np.sum(losses) != 0 else float('inf'),
            "total_return_pct": np.sum(pnls),
            "avg_bars_held": trades['bars_held'].mean() if 'bars_held' in trades else 0,
        }
    
    return {
        "long": compute_metrics(long_trades),
        "short": compute_metrics(short_trades),
        "long_pct": len(long_trades) / max(len(trades_df), 1) * 100,
        "short_pct": len(short_trades) / max(len(trades_df), 1) * 100,
    }
```

### ANALYSIS 6: Trade Duration Distribution

```python
"""
Histogram of trade durations (bars held).
Are trades clustered at min_hold_bars (25)?
"""

def trade_duration_analysis(trades_df, min_hold_bars=25):
    """
    Analyze the distribution of trade durations.
    
    Returns:
        dict with: mean, median, std, pct_at_minimum (trades closed at exactly min_hold)
    """
    durations = trades_df['bars_held'].values
    return {
        "mean_bars": float(np.mean(durations)),
        "median_bars": float(np.median(durations)),
        "std_bars": float(np.std(durations)),
        "min_bars": int(np.min(durations)),
        "max_bars": int(np.max(durations)),
        "pct_at_min_hold": float(np.mean(durations <= min_hold_bars + 2) * 100),  # Within 2 bars of min
        "pct_under_50": float(np.mean(durations < 50) * 100),
        "pct_over_100": float(np.mean(durations > 100) * 100),
    }
```

---

## OUTPUT FORMAT

The script should print a clear summary and save detailed results:

```
================================================================
STATISTICAL ANALYSIS — USDCOP RL TRADING
================================================================

BENCHMARKS
  Buy-and-Hold (long USDCOP):  -22.3% return
  Buy-and-Hold (short USDCOP): +22.3% return
  SMA(20/50) crossover:        +X.X% return, Sharpe X.XX
  Random agent (median):        X.X% return
  Random agent (% profitable):  X.X%

V21.5 STATISTICAL SIGNIFICANCE
  Mean trade PnL:     +0.006% per trade
  Bootstrap 95% CI:   [-0.05%, +0.08%]  ← INCLUDES ZERO = NOT SIGNIFICANT
  P-value vs zero:    0.32
  Verdict:            ❌ NOT statistically significant at 95%

V22 SEED 1337 STATISTICAL SIGNIFICANCE
  Mean trade PnL:     +0.025% per trade
  Bootstrap 95% CI:   [-0.02%, +0.07%]
  P-value vs zero:    0.08
  Verdict:            ❌ NOT statistically significant at 95%

LONG vs SHORT BREAKDOWN (V21.5)
  Long trades:  120 (56%), WR 54%, PF 0.98
  Short trades: 93 (44%), WR 59%, PF 1.05
  Verdict:      Slight short bias advantage (consistent with bearish test period)

TRADE DURATION (V21.5)
  Mean: 109 bars | Median: 87 bars | At min_hold: 12%
  Verdict:      ✅ Healthy distribution (not clustered at minimum)

TRADE DURATION (V22 seed 1337)
  Mean: 37 bars | Median: 28 bars | At min_hold: 68%
  Verdict:      ❌ Clustered at min_hold (agent wants to exit ASAP)

================================================================
OVERALL ASSESSMENT
================================================================
[ ] Model return is statistically significant
[ ] Model beats buy-and-hold
[ ] Model beats SMA crossover
[ ] Model beats random agent (p50)
[ ] Trades not clustered at min_hold
[ ] Balanced long/short performance

If 0-2 checks pass: Project needs fundamental rethink (better features or market)
If 3-4 checks pass: On right track, focus on identified weaknesses
If 5-6 checks pass: Strong model, proceed to walk-forward validation
================================================================
```

Save detailed results to:
- `results/statistical_analysis/benchmarks.json`
- `results/statistical_analysis/bootstrap_ci.json`
- `results/statistical_analysis/long_short_breakdown.json`
- `results/statistical_analysis/trade_duration.json`
- `results/statistical_analysis/summary.txt`

---

## HOW TO LOAD TRADE DATA

Trade data should be extractable from L4 results. Check:
- `results/backtests/l4_test_*.json` — may contain trade list
- Equity curve parquets — can derive trades from position changes
- If neither has individual trades, re-run L4 with verbose output

For V21.5: model at `models/ppo_ssot_20260206_204424/`
For V22 seed 1337: model at `models/ppo_ssot_20260209_155510/`

---

## AFTER RUNNING

1. Update `EXPERIMENT_LOG.md` with EXP-INFRA-001 results
2. Re-evaluate EXPERIMENT_QUEUE based on findings
3. If model is NOT statistically significant: focus shifts to features/data, not architecture
4. If model IS significant but doesn't beat SMA: the RL agent is under-performing simplicity
5. Share results with Pedro for strategic discussion
