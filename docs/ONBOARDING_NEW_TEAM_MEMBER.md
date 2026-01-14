# Onboarding Guide for New Team Members

> **Version**: 1.0.0
> **Last Updated**: 2026-01-14

## Welcome!

Welcome to the USDCOP RL Trading Team. This guide will help you get set up and productive.

## Day 1 Checklist

### Access Setup

- [ ] GitHub access to `usdcop-rl-models` repository
- [ ] Development environment credentials
- [ ] Slack/Teams channel access (#trading-dev)
- [ ] Grafana dashboard viewer access
- [ ] Database read-only credentials

### Environment Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/usdcop-rl-models.git
   cd usdcop-rl-models
   ```

2. **Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

3. **Docker Setup**
   ```bash
   docker-compose up -d
   ```

4. **Verify Installation**
   ```bash
   pytest tests/unit/ -v --tb=short
   ```

## Week 1 Goals

### Understand the System

1. **Read Documentation**
   - [ ] README.md - Project overview
   - [ ] ARCHITECTURE.md - System design
   - [ ] config/trading_config.yaml - SSOT for parameters

2. **Explore Codebase**
   - [ ] `src/` - Core library
   - [ ] `services/` - API services
   - [ ] `airflow/dags/` - Data pipeline

3. **Run Local Demo**
   ```bash
   # Start services
   docker-compose up -d

   # Run backtest
   python scripts/backtest.py --model models/ppo_primary

   # View dashboard
   open http://localhost:3000
   ```

### Key Concepts to Learn

| Concept | Resource |
|---------|----------|
| PPO Algorithm | [SB3 Docs](https://stable-baselines3.readthedocs.io/) |
| Feature Engineering | `docs/FEATURE_ENGINEERING.md` |
| SSOT Pattern | `src/config/trading_config.py` |
| Data Pipeline | `airflow/dags/README.md` |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline (Airflow)              │
│  L0: Ingestion → L1: Features → L3: Training → L5: Inf  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Inference API (FastAPI)               │
│  - /v1/backtest - Run model on historical data          │
│  - /v1/health - Service health check                    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Dashboard (Next.js)                   │
│  - Real-time trading signals                            │
│  - Performance metrics                                  │
└─────────────────────────────────────────────────────────┘
```

## Code Standards

### Commit Messages

```
feat: Add new feature
fix: Bug fix
docs: Documentation update
refactor: Code refactoring
test: Add tests
```

### Pull Request Template

1. Summary of changes
2. Test plan
3. Screenshots (if UI changes)

### Code Style

- Python: Black formatter, isort
- TypeScript: Prettier, ESLint
- Run `pre-commit install` for automatic checks

## Key Contacts

| Area | Contact |
|------|---------|
| Trading Logic | trading@example.com |
| Infrastructure | infra@example.com |
| Frontend | frontend@example.com |

## Resources

### Internal

- [Architecture Decisions](docs/adr/)
- [Model Cards](docs/model_cards/)
- [API Documentation](http://localhost:8000/docs)

### External

- [Stable Baselines 3](https://stable-baselines3.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Airflow](https://airflow.apache.org/docs/)
- [TimescaleDB](https://docs.timescale.com/)

## FAQ

**Q: How do I run a backtest?**

```bash
python scripts/backtest.py --model models/ppo_primary
```

**Q: Where are model parameters defined?**

All parameters are in `config/trading_config.yaml` (SSOT).

**Q: How do I add a new feature?**

1. Add to `config/feature_config.json`
2. Update `src/features/` computation
3. Regenerate `norm_stats.json`
4. Retrain model

**Q: Who do I ask for help?**

Post in #trading-dev or tag @trading-team.
