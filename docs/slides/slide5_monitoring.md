# Slide 5/7 — L6 MONITORING OPS: Evaluate and Protect

> 7 DAGs | 1 ACTIVE (H5 monitor) + 6 PAUSED | Guardrails, drift, reconciliation
> "Friday: was this week profitable? Are guardrails triggered? Should we keep trading?"

```mermaid
flowchart TB
    subgraph INPUTS["FROM L7 EXECUTION"]
        EXEC[("forecast_h5_executions<br/>Weekly trade results")]
        SUBS[("forecast_h5_subtrades<br/>Individual entries/exits")]
    end

    subgraph L6_ACTIVE["L6 ACTIVE — H5 Weekly Monitor"]
        H5MON["<b>forecast_h5_l6_weekly_monitor</b><br/>Friday 14:30 COT | ACTIVE<br/><br/>1. Calculate weekly metrics:<br/>   DA, Sharpe, MaxDD, WR, PF<br/><br/>2. Check 4 guardrails:<br/>   Circuit breaker<br/>   Long insistence alarm<br/>   Rolling DA SHORT<br/>   Rolling DA LONG<br/><br/>3. Promotion gates (week 15):<br/>   DA overall >55% AND SHORT >60%<br/><br/>4. Write paper trading record"]
    end

    subgraph GUARDRAILS["4 GUARDRAILS"]
        CB["CIRCUIT BREAKER<br/>5 consecutive losses<br/>OR 12% drawdown<br/>Action: PAUSE TRADING"]
        LI["LONG INSISTENCE<br/>>60% LONGs in 8 weeks<br/>Action: ALERT ONLY"]
        RDS["ROLLING DA SHORT<br/>SHORT DA <55% in 16 weeks<br/>Action: PAUSE SHORTS"]
        RDL["ROLLING DA LONG<br/>LONG DA <45% in 16 weeks<br/>Action: PAUSE LONGS"]
    end

    subgraph L6_PAUSED["L6 PAUSED TRACKS"]
        H1MON["<b>forecast_h1_l6_paper_monitor</b><br/>Mon-Fri 19:00 COT | PAUSED<br/>H1 paper trading daily log"]
        RL_PROD["<b>rl_l6_01_production_monitor</b><br/>Manual | PAUSED<br/>RL production eval"]
        RL_DRIFT["<b>rl_l6_02_drift_monitor</b><br/>Post-trading | PAUSED<br/>Feature + prediction drift<br/>KS test per feature, p=0.01<br/>MMD + Wasserstein multivariate"]
        ALERT["<b>core_l6_01_alert_monitor</b><br/>every 15 min | PAUSED<br/>Latency, data integrity"]
        WEEKLY_RPT["<b>core_l6_02_weekly_report</b><br/>Weekly | PAUSED<br/>Production summary report"]
        RECON["<b>reconciliation_daily</b><br/>20:00 COT | PAUSED<br/>Internal vs exchange fills<br/>Slippage computation"]
    end

    subgraph OUTPUT["MONITORING OUTPUT"]
        PAPER[("forecast_h5_paper_trading<br/>Weekly evaluation record")]
        VIEWS["v_h5_performance_summary<br/>v_h5_collapse_monitor"]
    end

    EXEC --> H5MON
    SUBS --> H5MON
    H5MON --> GUARDRAILS
    H5MON --> PAPER
    H5MON --> VIEWS

    style L6_ACTIVE fill:#064e3b,stroke:#10b981,color:#d1fae5
    style GUARDRAILS fill:#78350f,stroke:#f59e0b,color:#fef3c7
    style L6_PAUSED fill:#374151,stroke:#6b7280,color:#9ca3af
```

## Risk Management Architecture (3 Layers)

```
Signal arrives
    |
    v
LAYER 1: Chain of Responsibility (9 checks in order)
    0. HOLD signal      --> short-circuit, no trade
    10. Trading hours   --> 8:00-16:00 COT Mon-Fri
    20. Circuit breaker --> CB active? block all
    30. Cooldown        --> cooldown expired?
    40. Confidence      --> min 0.6
    50. Daily loss      --> -2% daily? trigger CB
    55. Drawdown        --> -1% max? trigger kill switch
    60. Consecutive     --> 3+ losses? activate cooldown 300s
    70. Max trades      --> 10/day limit
    |
    v APPROVED
LAYER 2: Risk Enforcer (7 pluggable rules)
    1. Kill switch      --> BLOCK (drawdown >= 15%)
    2. Daily loss       --> BLOCK (P&L <= -5%)
    3. Trade limit      --> BLOCK (>= 20/day)
    4. Cooldown         --> BLOCK (period active)
    5. Short permission --> BLOCK (if shorts disabled)
    6. Position size    --> REDUCE (cap at max)
    7. Confidence       --> BLOCK (below threshold)
    |
    v ALLOW / REDUCE / BLOCK
LAYER 3: Kill Switch (highest priority)
    Active? ALL trades blocked
    Only exit signals (CLOSE, FLAT) pass through
    Reset requires confirm=True + audit trail
```

## Promotion Gates (Week 15)

| Gate | Pass | Fail |
|------|------|------|
| DA overall > 55% AND SHORT > 60% | Promote to live | Check keep conditions |
| DA overall < 50% | -- | Discard strategy |
| SHORT DA > 60% but LONG DA < 45% | -- | Switch to SHORT-only |
