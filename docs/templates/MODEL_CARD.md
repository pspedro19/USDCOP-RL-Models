# Model Card: {MODEL_ID}

## Basic Information

| Field | Value |
|-------|-------|
| **Model ID** | {model_id} |
| **Version** | {version} |
| **Created Date** | {created_date} |
| **Owner** | {owner} |
| **Backup Owner** | {backup_owner} |
| **Current Stage** | {stage} |
| **MLflow Run ID** | {mlflow_run_id} |
| **MLflow Experiment** | {mlflow_experiment} |

---

## Training Details

| Field | Value |
|-------|-------|
| **Training Start** | {training_start_date} |
| **Training End** | {training_end_date} |
| **Training Duration** | {training_hours} hours |
| **Total Timesteps** | {total_timesteps} |
| **Dataset Version** | {dataset_hash} |
| **Dataset Period** | {dataset_start} to {dataset_end} |
| **Feature Count** | 15 (CTR-FEAT-001) |
| **Reward Function** | {reward_function_version} |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | {learning_rate} |
| N Steps | {n_steps} |
| Batch Size | {batch_size} |
| N Epochs | {n_epochs} |
| Gamma | {gamma} |
| GAE Lambda | {gae_lambda} |
| Clip Range | {clip_range} |
| Entropy Coef | {ent_coef} |
| Value Function Coef | {vf_coef} |
| Max Grad Norm | {max_grad_norm} |

---

## Artifact Hashes

| Artifact | Hash (SHA256) | Verified |
|----------|---------------|----------|
| Model (.zip) | `{model_hash}` | {Yes/No} |
| Model (.onnx) | `{onnx_hash}` | {Yes/No} |
| norm_stats.json | `{norm_stats_hash}` | {Yes/No} |
| Dataset | `{dataset_hash}` | {Yes/No} |
| Feature Order | `{feature_order_hash}` | {Yes/No} |

**Hash Verification Command:**
```bash
sha256sum models/ppo_primary.zip
sha256sum config/norm_stats.json
```

---

## Feature Set (CTR-FEAT-001)

| # | Feature Name | Type | Range | Description |
|---|--------------|------|-------|-------------|
| 1 | log_ret_5m | Numeric | [-0.1, 0.1] | 5-minute log return |
| 2 | log_ret_1h | Numeric | [-0.2, 0.2] | 1-hour log return |
| 3 | log_ret_4h | Numeric | [-0.3, 0.3] | 4-hour log return |
| 4 | rsi_9 | Numeric | [0, 100] | 9-period RSI (Wilder's) |
| 5 | atr_pct | Numeric | [0, 0.1] | ATR as % of price |
| 6 | adx_14 | Numeric | [0, 100] | 14-period ADX |
| 7 | dxy_z | Numeric | [-3, 3] | DXY z-score |
| 8 | dxy_change_1d | Numeric | [-0.05, 0.05] | DXY 1-day change |
| 9 | vix_z | Numeric | [-3, 3] | VIX z-score |
| 10 | embi_z | Numeric | [-3, 3] | EMBI Colombia z-score |
| 11 | brent_change_1d | Numeric | [-0.1, 0.1] | Brent 1-day change |
| 12 | rate_spread | Numeric | [-0.05, 0.15] | Interest rate spread |
| 13 | usdmxn_change_1d | Numeric | [-0.05, 0.05] | USD/MXN 1-day change |
| 14 | position | Numeric | [-1, 1] | Current position |
| 15 | time_normalized | Numeric | [0, 1] | Normalized trading time |

---

## Performance Metrics

### Backtest Performance

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Sharpe Ratio | {backtest_sharpe} | >= 1.0 | {Pass/Fail} |
| Win Rate | {backtest_win_rate}% | >= 50% | {Pass/Fail} |
| Max Drawdown | {backtest_max_dd}% | <= 10% | {Pass/Fail} |
| Total Trades | {backtest_trades} | >= 100 | {Pass/Fail} |
| Profit Factor | {profit_factor} | >= 1.5 | {Pass/Fail} |
| Total Return | {total_return}% | N/A | - |
| Sortino Ratio | {sortino_ratio} | N/A | - |
| Calmar Ratio | {calmar_ratio} | N/A | - |

### Action Distribution (Backtest)

| Action | Count | Percentage |
|--------|-------|------------|
| HOLD (0) | {hold_count} | {hold_pct}% |
| BUY (1) | {buy_count} | {buy_pct}% |
| SELL (2) | {sell_count} | {sell_pct}% |

### Staging Performance (if applicable)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Sharpe Ratio | {staging_sharpe} | >= 1.0 | {Pass/Fail} |
| Win Rate | {staging_win_rate}% | >= 50% | {Pass/Fail} |
| Agreement Rate | {agreement_rate}% | >= 85% | {Pass/Fail} |
| Days in Staging | {staging_days} | >= 7 | {Pass/Fail} |
| Total Trades | {staging_trades} | >= 100 | {Pass/Fail} |

### Production Performance (if applicable)

| Metric | Value | As of Date |
|--------|-------|------------|
| Sharpe Ratio (30d) | {prod_sharpe} | {date} |
| Win Rate (30d) | {prod_win_rate}% | {date} |
| Total P&L | {total_pnl} | {date} |
| Days in Production | {prod_days} | {date} |

---

## Known Limitations

1. **Timeframe Dependency**
   - Model trained on 5-minute bars only
   - Not suitable for other timeframes without retraining

2. **Warmup Period**
   - Requires 14 bars of data before first valid prediction
   - During warmup, model outputs HOLD

3. **Volatility Sensitivity**
   - Performance may degrade when VIX > 40
   - Consider pausing in extreme volatility regimes

4. **Market Hours**
   - Trained on Colombia trading hours (8:00-16:00 COT)
   - May not perform well outside these hours

5. **Data Dependency**
   - Dependent on TwelveData for OHLCV data
   - Dependent on external macro data sources

---

## Risk Factors

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| High volatility regime | Medium | Medium | Circuit breaker at 5% daily drawdown |
| DXY correlation breakdown | Medium | Low | Daily macro indicator monitoring |
| Data source outage | High | Low | Auto-pause on stale data |
| Model drift | Medium | Medium | Hourly PSI monitoring |
| Low liquidity periods | Medium | Medium | Trading hours restriction |
| {custom_risk_1} | {severity} | {likelihood} | {mitigation} |

---

## Ethical Considerations

### Intended Use
- Automated trading signal generation for USD/COP currency pair
- Decision support for human traders
- Risk-managed algorithmic trading

### Out-of-Scope Uses
- Live trading without human oversight
- Trading other currency pairs without retraining
- High-frequency trading (sub-second decisions)
- Use as sole decision maker without risk controls

### Potential Biases
- Training data from 2020-2025 may overweight COVID recovery period
- Model may have learned patterns specific to Colombian market structure
- Macro indicators weighted toward US economic data

---

## Deployment Information

### Inference Requirements

| Requirement | Specification |
|-------------|---------------|
| Runtime | ONNX Runtime 1.15+ |
| Python | 3.9+ |
| Memory | ~500MB |
| CPU | Single core sufficient |
| GPU | Not required |

### Latency SLAs

| Metric | Target | Measured |
|--------|--------|----------|
| P50 | < 20ms | {p50_latency}ms |
| P95 | < 50ms | {p95_latency}ms |
| P99 | < 100ms | {p99_latency}ms |

### Required Files

| File | Location | Required |
|------|----------|----------|
| model.onnx | models/{model_id}/ | Yes |
| norm_stats.json | config/ | Yes |
| trading_config.yaml | config/ | Yes |

---

## Monitoring

### Dashboards
- Grafana: `https://grafana.internal/d/{dashboard_id}`
- MLflow: `https://mlflow.internal/#/experiments/{experiment_id}`

### Alerts Configured
- [ ] Sharpe ratio < 0.8 (warning)
- [ ] Sharpe ratio < 0.5 (critical)
- [ ] Error rate > 1% (warning)
- [ ] Error rate > 5% (critical)
- [ ] Latency P99 > 1s (warning)
- [ ] Feature drift PSI > 0.2 (warning)

---

## Change History

| Date | Change | Author | Approved By |
|------|--------|--------|-------------|
| {created_date} | Initial training complete | {author} | - |
| {staging_date} | Promoted to staging | {author} | {approver} |
| {prod_date} | Promoted to production | {author} | {approver} |
| {change_date} | {change_description} | {author} | {approver} |

---

## Review Schedule

| Review Type | Frequency | Next Due | Owner |
|-------------|-----------|----------|-------|
| Performance Review | Weekly | {next_weekly} | Model Owner |
| Drift Check | Daily (automated) | - | System |
| Comprehensive Review | Monthly | {next_monthly} | Model Owner + Eng Lead |
| Policy Compliance | Quarterly | {next_quarterly} | Trading Ops |

---

## Appendix

### A. Training Curves

{Link to or embed training loss curves, reward curves from MLflow}

### B. Feature Importance

| Feature | Importance Score | Rank |
|---------|------------------|------|
| {feature_1} | {score} | 1 |
| {feature_2} | {score} | 2 |
| ... | ... | ... |

### C. Confusion Matrix (Discretized Actions)

|  | Predicted HOLD | Predicted BUY | Predicted SELL |
|--|----------------|---------------|----------------|
| **Actual HOLD** | {value} | {value} | {value} |
| **Actual BUY** | {value} | {value} | {value} |
| **Actual SELL** | {value} | {value} | {value} |

---

**Generated**: {generation_date}
**Generator**: scripts/generate_model_card.py
**Next Review**: {review_date}

---

*This model card provides transparency about the model's capabilities, limitations, and appropriate use cases. It should be updated whenever significant changes are made to the model or its deployment.*
