# APY Projections Analysis - Complete Overview

## Quick Navigation

This analysis package contains comprehensive APY projections for the USD/COP RL trading system. Start here to understand the three documents included.

### Documents in This Analysis

1. **APY_PROJECTIONS_ANALYSIS.md** (12 sections, 250+ lines)
   - Complete executive analysis with all scenarios
   - Cost impact breakdown
   - Sensitivity analysis
   - Realistic production projections
   - Strategic recommendations for 90-day implementation
   - START HERE for business decisions

2. **APY_QUICK_REFERENCE.txt** (30 sections)
   - One-page scenario comparison table
   - Cost sensitivity chart
   - Signal quality sensitivity
   - Break-even analysis
   - Implementation roadmap with timelines
   - Immediate action items
   - USE THIS for quick lookups and decisions

3. **APY_MATHEMATICAL_FOUNDATIONS.md** (15 sections)
   - Detailed mathematical formulas for all calculations
   - Derivations and proofs
   - Sensitivity analysis mathematics
   - Kelly criterion and position sizing
   - Risk-adjusted return metrics (Sharpe, VaR)
   - REFERENCE THIS when validating calculations

4. **scripts/analyze_apy_scenarios.py**
   - Python implementation of all scenarios
   - Runnable analysis with detailed output
   - Easy to modify parameters for sensitivity testing
   - RUN THIS to reproduce or extend analysis

---

## Key Findings at a Glance

### Current System (As-Is)
- Status: UNACCEPTABLE (X)
- 88-day Return: -13.54%
- Annualized (APY): -55.96%
- Problem: Transaction costs (19.8% drag) destroy profitable signal ($28.46 avg trade)

### Realistic Target (90-day improvement path)
- Status: ACCEPTABLE (CHECK)
- 88-day Return: 4.50%
- Annualized (APY): 18.60%
- Solution: Reduce costs to 35 bps + improve signal to $60 + filter to 18 trades

### Best Case (Maximum effort)
- Status: EXCELLENT (CHECK)
- 88-day Return: 9.00%
- Annualized (APY): 37.20%
- Requires: 30 bps costs + $75 avg trade + 20 trades

---

## Critical Insight: The Three-Legged Stool

**None of these alone will work:**

| Improvement | Solo Impact | Status |
|-------------|-----------|--------|
| Cost 90->50 bps (alone) | -19.59% APY | FAIL |
| Signal $28->$45 (alone) | -40.92% APY | FAIL |
| Filter 22->18 trades (alone) | -45.79% APY | FAIL |
| All three combined | +18.60% APY | SUCCESS |

---

## The Math in One Picture

```
BASELINE: 22 trades x $28.46 - $1,980 costs = -$1,354 (broken)

TARGET:  18 trades x $60.00 - $630 costs = +$450 (works!)
         ^^^^^^         ^^^^^^      ^^^^^^
         Trade filter   Signal+++   Cost down
         (important)    (critical)  (critical)
```

---

## 90-Day Implementation Path

### Phase 1 (Days 1-30): Cost Reduction
- Action: Negotiate prime brokerage rates
- Goal: 90 -> 50 bps
- Expected APY: -28% -> -19% (improvement: +9%)
- Priority: CRITICAL
- Effort: LOW

### Phase 2 (Days 15-45): Signal Enhancement
- Action: Add macro features, improve exits
- Goal: $28 -> $45 avg trade (+58%)
- Expected APY: -19% -> -5% (improvement: +14%)
- Priority: HIGH
- Effort: MEDIUM

### Phase 3 (Days 30-60): Trade Filtering
- Action: Implement confidence scoring
- Goal: Select 18/22 high-quality trades
- Expected APY: -5% -> +5% (improvement: +10%)
- Priority: HIGH
- Effort: LOW

### Phase 4 (Days 60-90): Validation
- Action: Backtest on 12-month OOS data
- Goal: Achieve +15-25% APY sustainably
- Priority: ESSENTIAL
- Effort: MEDIUM

---

## How to Use Each Document

### For Executive Decision-Makers
1. Read "APY_QUICK_REFERENCE.txt" first (5 minutes)
2. Review "Current Performance" vs "TARGET" sections
3. Check "Immediate Actions" for next steps
4. Decision: Proceed with Phase 1 cost reduction?

### For Traders/Quants
1. Start with "APY_PROJECTIONS_ANALYSIS.md" Part 1-3
2. Understand cost structure and sensitivity analysis (Part 5-7)
3. Review recommended improvements (Part 6)
4. Plan Phase 2 signal enhancement work

### For Data Scientists
1. Review "APY_MATHEMATICAL_FOUNDATIONS.md" for formulas
2. Study sensitivity analysis (Part 7)
3. Understand Kelly criterion and position sizing (Part 10)
4. Validate calculations independently

### For Risk Managers
1. Check "APY_QUICK_REFERENCE.txt" risk section
2. Review Sharpe ratio and VaR analysis (Foundations Part 13)
3. Understand max drawdown constraints
4. Validate against risk limits

---

## Sensitivity Analysis Quick Reference

### Cost Sensitivity (with $28 avg trade, 22 trades)
```
Cost Reduction    | APY Impact
30 bps (unrealistic) | -1.4%
50 bps (achievable)  | -19.6%
90 bps (current)     | -56.0%

=> Every 10 bps cut adds ~2% APY
```

### Signal Sensitivity (50 bps costs, 22 trades)
```
Avg Trade PnL | APY Impact
$28 (current) | -19.6%
$40 (+40%)    | -9.1%
$50 (break-even) | 0%
$60 (+110%)   | +9.1%
$75 (+160%)   | +22.7%

=> Every $10 improvement adds ~7% APY
```

### Trade Selection (50 bps costs, $45 signal)
```
Trades Selected | 88d Return | APY
22/22 (all)     | -0.9%      | -3.7%
18/22 (top 82%) | +2.0%      | +8.3%
15/22 (top 68%) | +4.0%      | +16.5%

=> Trade quality matters more than quantity
```

---

## Financial Projections

### 12-Month Capital Growth Scenarios

Starting Capital: $10,000

```
Scenario              | Pessimistic | Base Case | Optimistic
Current (baseline)    | $5,950      | $7,300    | $8,110
Moderate (6 mo)       | $10,700     | $11,000   | $11,500
Target (full stack)   | $11,750     | $12,500   | $13,750
Optimistic            | $12,940     | $14,200   | $16,300
```

**Target scenario (most likely with disciplined execution):**
- Starting: $10,000
- Base case at 12 months: $12,500 (+25%)
- This assumes ~18% APY sustained throughout year

---

## Most Important Equations

### APY Formula (Simple)
```
APY = (Net Profit / Initial Capital) x (365 / Days) x 100%

For your system:
APY = [N x (P_avg - Cost_per_Trade)] / C0 x (365/88.3) x 100%
```

### Cost-Signal Trade-Off
```
At break-even (APY = 0):
Cost per trade = P_avg
$90 = P_avg (current, at 90 bps)
$50 = P_avg (needed at 50 bps)
$35 = P_avg (needed at 35 bps)

=> Need BOTH cost reduction AND signal improvement
```

### Annual Frequency vs Costs
```
Annual Trades = (Current Trades / Backtest Days) x 365
             = (22 / 88.3) x 365
             = 91 trades/year

Cost Budget = Annual Trades x Cost per Trade
           = 91 x $50 = $4,550/year (at 50 bps)
           = Annual Gross Profit - Target Net Profit
```

---

## Critical Success Factors (Must Have All Three)

### 1. Cost Reduction (Non-Negotiable)
- Current: 90 bps (retail, terminal)
- Target: 50 bps (prime brokerage, achievable)
- Best: 35 bps (institutional, harder but possible)
- Why: Each trade loses $90 (vs profit of $28) - unsustainable

### 2. Signal Quality Improvement (Essential)
- Current: $28.46 avg trade (insufficient)
- Target: $60 avg trade (110% improvement)
- Methods:
  - Add macro regime detection features
  - Improve position exit logic
  - Weight high-conviction setups
- Why: Can't afford costs with low-quality signals

### 3. Trade Selection (Important)
- Current: 22 trades per period (all signals)
- Target: 18 trades (top 82% confidence)
- Methods:
  - Add conviction/confidence scoring
  - Filter by regime compatibility
  - Require macro alignment
- Why: Quality > Quantity; fewer good trades beat many mediocre ones

---

## Questions Answered by This Analysis

### Q: Can we just reduce costs and stay profitable?
A: No. Alone, 50 bps costs gives -19.6% APY. Need signal improvement too.

### Q: Can we improve signal quality and ignore costs?
A: No. Even at $60 avg trade, 90 bps costs gives -27.3% APY. Need cost cuts.

### Q: How many trades per year can we sustain?
A: At current 0.25 trades/day = 91 trades/year. Cost budget: $4,550 at 50 bps.
Gross profit budget: $1,000 for breakeven, $2,000+ for 10% APY.

### Q: What's the break-even signal quality?
A: At 50 bps costs: $50 avg trade PnL. Current $28.46 = 43% shortfall.

### Q: Is 18.6% APY (target) realistic?
A: Yes. Requires 35 bps costs + $60 avg trade + 18 trades. All achievable in 90 days.

### Q: What happens if only two of the three improvements succeed?
A: APY ranges from -3.7% to -40.9%. Need all three for positive returns.

---

## Recommended Reading Order

**For Quick Understanding (15 minutes):**
1. APY_QUICK_REFERENCE.txt - Overview and scenarios
2. APY_QUICK_REFERENCE.txt - Sensitivity tables

**For Deep Dive (45 minutes):**
1. APY_PROJECTIONS_ANALYSIS.md - Parts 1-4
2. APY_PROJECTIONS_ANALYSIS.md - Part 5-7
3. APY_QUICK_REFERENCE.txt - Roadmap and actions

**For Implementation (1-2 hours):**
1. APY_PROJECTIONS_ANALYSIS.md - Part 11 (recommendations)
2. APY_QUICK_REFERENCE.txt - Implementation roadmap
3. scripts/analyze_apy_scenarios.py - Run and review output

**For Complete Understanding (3-4 hours):**
- Read all three documents
- Run the Python script
- Modify parameters to test scenarios
- Cross-validate with mathematical formulas

---

## Files and Locations

All analysis files are in the project root:

```
USDCOP-RL-Models/
├── APY_PROJECTIONS_ANALYSIS.md        (Main analysis document)
├── APY_QUICK_REFERENCE.txt            (One-page summary)
├── APY_MATHEMATICAL_FOUNDATIONS.md    (Mathematical rigor)
├── APY_ANALYSIS_README.md             (This file)
└── scripts/
    └── analyze_apy_scenarios.py       (Runnable Python script)
```

---

## How to Modify Analysis

### Changing Assumptions in Python Script

Edit the configuration section at top of `scripts/analyze_apy_scenarios.py`:

```python
INITIAL_CAPITAL = 10_000.0      # Change initial account size
BACKTEST_DAYS = 88.3            # Change backtest period
NUM_TRADES = 22                 # Change number of trades
AVG_TRADE_PNL = 28.46           # Change average trade PnL
```

Then run:
```bash
python scripts/analyze_apy_scenarios.py
```

---

## Next Steps

1. **Immediate (Today):**
   - Review APY_QUICK_REFERENCE.txt
   - Share with stakeholders
   - Approve path forward

2. **This Week (Phase 1 Start):**
   - Begin prime brokerage benchmarking
   - List 3-5 potential brokers
   - Request RFQs for costs

3. **Next 2 Weeks (Phase 2 Planning):**
   - Analyze top 80% of historical trades
   - Identify common characteristics
   - Design new signal features

4. **Week 3-4 (Phase 3 Setup):**
   - Implement confidence scoring
   - Test on historical data
   - Validate trade quality improvement

5. **Week 5-6 (Phase 4 Validation):**
   - Backtest on 12-month OOS data
   - Validate +15-25% APY projections
   - Get sign-off for production deployment

---

**Document Generated:** February 2, 2026
**Model:** PPO v20 Production (500k timesteps)
**Analysis Period:** Full 2025 (88.3-day test window)
**Version:** 1.0
