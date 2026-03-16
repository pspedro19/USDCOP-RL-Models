# SDD Spec: CI/CD & Testing Infrastructure

> **Responsibility**: Authoritative source for GitHub Actions workflows, Makefile automation,
> test categories, code quality tooling, coverage gates, and deployment pipelines.
> Other specs reference this file for CI/CD pipeline details — do not duplicate workflow definitions elsewhere.
>
> Contract: CTR-CICD-001
> Version: 1.0.0
> Date: 2026-03-14

---

## Architecture Overview

```
Developer Workstation                    GitHub Actions (9 Workflows)
    |                                         |
    ├── make lint                             ├── ci.yml (main CI pipeline)
    ├── make format                           ├── deploy.yml (CD with staging gates)
    ├── make typecheck                        ├── security.yml (Bandit, Safety, pip-audit)
    ├── make test                             ├── security-scan.yml (Gitleaks, Trivy, env check)
    ├── make coverage (70% gate)              ├── contracts-check.yml (SSOT compliance)
    ├── make validate-ssot                    ├── drift-check.yml (feature drift detection)
    ├── make validate-contracts               ├── dvc-validate.yml (DVC pipeline validation)
    ├── make pre-commit                       ├── experiment.yml (ML experiment workflow)
    └── make ci (composite)                   └── canary-promote.yml (gradual canary rollout)
```

### Tool Stack

| Tool | Version | Purpose | Config Location |
|------|---------|---------|-----------------|
| Python | 3.11 | Runtime | `pyproject.toml` |
| Ruff | latest | Linter (E/W/F/I/B/C4/UP/ARG/SIM/TCH/RUF) | `pyproject.toml [tool.ruff]` |
| Black | latest | Formatter (line 100, trailing commas) | `pyproject.toml [tool.black]` |
| isort | latest | Import sorter (profile=black) | `pyproject.toml [tool.isort]` |
| MyPy | latest | Type checker (strict mode) | `pyproject.toml [tool.mypy]` |
| Pytest | latest | Test runner | `pyproject.toml [tool.pytest]` |
| Coverage | latest | Code coverage (70% gate) | `pyproject.toml [tool.coverage]` |
| Bandit | latest | Security linter (skip B101) | `pyproject.toml [tool.bandit]` |
| Playwright | latest | Dashboard E2E tests | `usdcop-trading-dashboard/playwright.config.ts` |

---

## GitHub Actions Workflows (9 Files)

### Workflow Inventory

| # | Workflow | File | Trigger | Purpose |
|---|----------|------|---------|---------|
| 1 | CI Pipeline | `ci.yml` | Push main/develop, PRs | Lint, typecheck, test, integration, build |
| 2 | CD Pipeline | `deploy.yml` | Manual dispatch | Build, security scan, staging, production deploy |
| 3 | Security Scanning | `security.yml` | Push main, PRs, weekly Sun 3AM UTC, manual | Bandit, Safety, pip-audit |
| 4 | Security Scan | `security-scan.yml` | Push main/develop, PRs, daily 2AM UTC | Gitleaks, Trivy, env check, hardcoded secrets |
| 5 | Contracts & SSOT Check | `contracts-check.yml` | Changes in `src/**`, `services/**`, `airflow/**` | FEATURE_ORDER, hash validation, hardcoded scan |
| 6 | Feature Drift Check | `drift-check.yml` | Daily 6AM UTC, manual, workflow_call | Univariate + multivariate drift detection |
| 7 | DVC Pipeline Validation | `dvc-validate.yml` | Changes to DVC/data/scripts, weekly Sun 2AM UTC | YAML, params, data, lock consistency |
| 8 | ML Experiment Workflow | `experiment.yml` | Manual dispatch | Experiment lifecycle: setup, validate, train, report |
| 9 | Canary Promotion | `canary-promote.yml` | Manual dispatch | Gradual traffic promotion: 10% -> 25% -> 50% -> 100% |

---

### Workflow 1: CI Pipeline (`ci.yml`)

The main continuous integration pipeline. Runs on every push to `main`/`develop` and on all PRs.

```
push/PR to main/develop
    |
    v
┌─────────┐     ┌──────────────┐
│  lint    │     │  type-check  │    (parallel, no dependency)
└────┬────┘     └──────────────┘
     |
     v
┌─────────┐
│  test   │    (needs: lint)
│  70%    │    PostgreSQL 15 + Redis 7-alpine services
└────┬────┘
     |
     ├──────────────────┬──────────────────┐
     v                  v                  v
┌────────────┐   ┌──────────────┐   ┌──────────┐
│integration │   │feature-parity│   │ load-test│    (optional)
│   -test    │   │              │   │          │
└────────────┘   └──────────────┘   └──────────┘
     |                  |                  |
     v                  v                  v
┌─────────┐   ┌──────────┐   ┌─────────────┐
│  build  │   │  notify  │   │   summary   │
│ Docker  │   │(on fail) │   │             │
└─────────┘   └──────────┘   └─────────────┘
```

**Jobs**:

| Job | Needs | Services | Key Actions |
|-----|-------|----------|-------------|
| `lint` | none | none | `ruff check`, `black --check`, `isort --check-only` on `src/ services/ tests/` |
| `type-check` | none | none | `mypy src/` with `--ignore-missing-imports` |
| `test` | lint | PostgreSQL 15, Redis 7-alpine | `pytest tests/unit/ --cov-fail-under=70 -x` |
| `integration-test` | test | PostgreSQL 15, Redis 7-alpine | `pytest tests/integration/ -x` (ignores `test_database_schema.py`) |
| `feature-parity` | test | none | `pytest test_feature_parity.py test_observation_parity.py` |
| `load-test` | test | none | `pytest tests/load/ -m "load and not slow"` (schedule or `[load-test]` commit only) |
| `build` | test | none | Docker Buildx `services/Dockerfile.api` (push=false) |
| `notify` | test, integration, parity | none | Error annotation on failure |
| `summary` | all | none | GitHub Step Summary table |

**CI Services** (available to `test` and `integration-test`):

| Service | Image | Credentials | Port |
|---------|-------|-------------|------|
| PostgreSQL | `postgres:15` | `test:test@localhost:5432/test_db` | 5432 |
| Redis | `redis:7-alpine` | `redis://localhost:6379` | 6379 |

**Environment Variables**:

| Variable | Value | Used By |
|----------|-------|---------|
| `PYTHON_VERSION` | `3.11` | All jobs |
| `COVERAGE_THRESHOLD` | `70` | test job |
| `DATABASE_URL` | `postgresql://test:test@localhost:5432/test_db` | test, integration |
| `REDIS_URL` | `redis://localhost:6379` | test, integration |

---

### Workflow 2: CD Pipeline (`deploy.yml`)

Manual deployment workflow with staging validation and production approval gates.

**Inputs**:

| Input | Type | Options | Required |
|-------|------|---------|----------|
| `environment` | choice | staging, production | Yes |
| `deployment_type` | choice | blue-green, canary | Yes |
| `model_version` | string | (optional, uses latest) | No |
| `skip_staging` | boolean | default: false | No |

**Jobs**:

| Job | Needs | Environment | Key Actions |
|-----|-------|-------------|-------------|
| `build` | none | none | Docker build + push to GHCR (sha/branch/semver tags), SBOM generation |
| `security-scan` | build | none | Trivy container scan (CRITICAL fails build, HIGH to SARIF) |
| `staging-deploy` | build, security-scan | staging | Health check + smoke tests (skippable via input) |
| `production-approval` | staging-deploy | production | GitHub environment protection gate (human approval) |
| `production-deploy` | production-approval | production | Blue-green OR canary deployment, health verification |
| `rollback` | production-deploy | none | Auto-triggers on production-deploy failure |
| `summary` | all | none | Deployment summary with timestamps |

**Docker Image Tags** (from `docker/metadata-action`):

| Tag Pattern | Example | When |
|-------------|---------|------|
| `type=sha` | `abc1234` | Always |
| `type=ref,event=branch` | `main` | Always |
| `type=semver` | `1.2.3` | On semver tags |
| `type=raw,value=latest` | `latest` | Default branch only |

---

### Workflow 3: Security Scanning (`security.yml`)

Python-focused security scanning with artifact reports.

| Tool | Scope | Output | Severity Filter |
|------|-------|--------|-----------------|
| Bandit | `src/`, `services/` | JSON + text reports | medium+ confidence, medium+ severity |
| Safety | `requirements.txt` + `services/requirements.txt` | JSON + text reports | All known CVEs |
| pip-audit | Installed packages + `requirements.txt` | JSON + columns reports | PyPI vulnerability DB |

**Schedule**: Weekly Sun 3AM UTC + push to main + PRs to main + manual dispatch.
**Artifacts**: `security-reports/` directory (30-day retention) with combined `SUMMARY.md`.

---

### Workflow 4: Security Scan (`security-scan.yml`)

Broader security scan with secret detection and filesystem analysis. This is a separate workflow
from `security.yml` covering different concerns (secrets + container + env files vs Python packages).

| Job | Tool | Purpose |
|-----|------|---------|
| `gitleaks` | Gitleaks v2 | Secret detection in git history (SARIF upload) |
| `trivy-scan` | Trivy | Filesystem vulnerability scan (CRITICAL + HIGH) |
| `dependency-check` | Safety + pip-audit | Python dependency CVE check |
| `env-file-check` | Shell scripts | Detect committed `.env` files |
| `hardcoded-secrets-check` | grep patterns | Scan for password/api_key/secret/token/private key patterns |

**Schedule**: Daily 2AM UTC + push to main/develop + PRs.

---

### Workflow 5: Contracts & SSOT Check (`contracts-check.yml`)

Validates that SSOT contracts are consistent and importable. Triggered only when Python files change
in `src/`, `services/`, or `airflow/`.

**Job 1: `contracts-check`** (SSOT Compliance)

| Check | What It Validates |
|-------|-------------------|
| Deprecated imports | `scripts/migrate_imports.py --check` |
| Contract imports | `FEATURE_ORDER`, `OBSERVATION_DIM`, `FEATURE_CONTRACT`, `FEATURE_CONTRACT_VERSION` |
| SSOT constants | `CLIP_MIN/MAX`, `RSI_PERIOD`, `ATR_PERIOD`, `ADX_PERIOD`, `THRESHOLD_LONG/SHORT`, `BARS_PER_SESSION` |
| Hardcoded values | Grep for `log_ret_5m.*log_ret_1h.*rsi_9` outside `feature_contract.py` |
| Feature hash | Recomputes `SHA256(FEATURE_ORDER)[:16]` and asserts == `FEATURE_ORDER_HASH` |

**Job 2: `contract-consistency`** (Round-Trip Tests)

| Check | What It Validates |
|-------|-------------------|
| Round-trip | `features_dict_to_array()` -> `validate_feature_vector()` with correct shape + dtype |
| Cross-module | `src.core.contracts.FEATURE_ORDER` == `src.feature_store.core.FEATURE_ORDER` |

---

### Workflow 6: Feature Drift Check (`drift-check.yml`)

Daily automated drift detection for ML model monitoring.

**Schedule**: Daily 6AM UTC (1AM COT). Also callable via `workflow_call` on model deployment.

**Inputs**:

| Input | Default | Purpose |
|-------|---------|---------|
| `drift_threshold` | `medium` | Severity filter (none/low/medium/high) |
| `check_multivariate` | `true` | Enable multivariate checks (MMD, Wasserstein, PCA) |

**Detection Methods**:

| Type | Class | Methods |
|------|-------|---------|
| Univariate | `FeatureDriftDetector` | KS test per feature, p-value threshold (0.01) |
| Multivariate | `MultivariateDriftDetector` | MMD (Maximum Mean Discrepancy), Wasserstein distance, PCA reconstruction error |

**Module**: `src/monitoring/drift_detector.py`

**Jobs**: `drift-detection` -> `drift-alert` (if drifted) + `unit-tests` (always).

---

### Workflow 7: DVC Pipeline Validation (`dvc-validate.yml`)

Validates DVC pipeline configuration and data integrity.

**Triggers**: Changes to `dvc.yaml`, `dvc.lock`, `params.yaml`, `data/**`, `scripts/**`, `src/**`.
Also weekly Sun 2AM UTC.

| Job | Needs | What It Validates |
|-----|-------|-------------------|
| `validate-config` | none | YAML syntax, stage dependencies, DAG validity |
| `check-params` | validate-config | `lr > 0`, `batch_size > 0`, `0 < gamma <= 1`, `0 < clip_range <= 1`, `train_ratio + val_ratio < 1` |
| `validate-data` | validate-config | `.dvc` file format, hash presence (PRs only) |
| `check-lock` | validate-config | `dvc.lock` stages match `dvc.yaml` stages |
| `pipeline-dry-run` | validate-config, check-params | `dvc repro --dry` (schedule/manual only) |

---

### Workflow 8: ML Experiment Workflow (`experiment.yml`)

Manual workflow for running ML experiments with tracking.

**Inputs**:

| Input | Type | Required | Options / Default |
|-------|------|----------|-------------------|
| `experiment_name` | string | Yes | -- |
| `experiment_type` | choice | Yes | hyperparameter_search, architecture_comparison, feature_ablation, baseline_comparison |
| `learning_rate` | string | No | `3e-4` |
| `total_timesteps` | string | No | `100000` |
| `n_epochs` | string | No | `10` |
| `run_validation` | boolean | No | `true` |

**Jobs**: `setup` (timestamped ID) -> `validate-config` (bounds + SSOT) -> `train` (120min timeout) -> `validate` (60min timeout) -> `report` (markdown) -> `cleanup`.

**Artifacts**: Model files (30d retention), experiment report (90d retention).

**Note**: The `train` job currently contains a simulated placeholder. In production, it calls
the actual training script. The workflow structure is fully implemented.

---

### Workflow 9: Canary Promotion (`canary-promote.yml`)

Gradual canary deployment with automatic health monitoring and rollback.

**Inputs**:

| Input | Type | Options | Required |
|-------|------|---------|----------|
| `canary_deployment_id` | string | -- | Yes |
| `action` | choice | promote, rollback, status | Yes |
| `target_percentage` | choice | 25, 50, 75, 100 | No |

**Promotion Progression**: 10% -> 25% -> 50% -> 100% (automatic if healthy).

**Health Thresholds**:

| Metric | Threshold | Breach Action |
|--------|-----------|---------------|
| Error rate | 1% (`0.01`) | Auto-rollback |
| Latency P99 | 500ms | Auto-rollback |
| Observation period | 5 minutes (300s) | Wait between promotions |

**Jobs**:

| Job | Condition | Action |
|-----|-----------|--------|
| `check-canary-status` | Always | Query Prometheus for error_rate + latency_p99 |
| `auto-rollback` | Unhealthy + action != status | Roll back to stable version |
| `promote-canary` | Healthy + action == promote | Promote to next percentage tier |
| `manual-rollback` | action == rollback | Immediate rollback |
| `status-report` | action == status | Generate health summary |

---

## Makefile (268 Lines)

### Target Inventory

| Category | Target | Command | Description |
|----------|--------|---------|-------------|
| **Install** | `install` | `pip install -e .` | Production dependencies |
| | `install-dev` | `pip install -e ".[all]"` + `pre-commit install` | Dev dependencies + pre-commit hooks |
| **Test** | `test` | `pytest tests/ -v --tb=short` | All tests |
| | `test-unit` | `pytest tests/unit/ -m "unit or not (...)"` | Unit tests only |
| | `test-contracts` | `pytest tests/contracts/ tests/unit/test_contracts.py test_gtr_contracts.py test_all_layer_contracts.py` | Contract tests |
| | `test-regression` | `pytest tests/regression/` | Regression tests |
| | `test-integration` | `pytest tests/integration/ -m "integration"` | Integration tests (requires services) |
| | `coverage` | `pytest tests/ --cov=src --cov=services --cov-fail-under=70` | Tests with 70% coverage gate |
| **Quality** | `lint` | `ruff check src/ services/ tests/` | Linter |
| | `format` | `black` + `isort` + `ruff --fix` on `src/ services/ tests/` | Auto-format |
| | `typecheck` | `mypy src/ services/ --config-file pyproject.toml` | Type checking |
| **Validate** | `validate` | validate-ssot + validate-contracts | All validations |
| | `validate-ssot` | `pytest test_ssot_consistency.py test_config_ssot.py test_config_consistency.py` | SSOT checks |
| | `validate-contracts` | `pytest tests/contracts/ test_contracts.py test_feature_contract.py` | Contract checks |
| **Docker** | `docker-up` | `docker compose up -d` | Start all services |
| | `docker-down` | `docker compose down` | Stop all services |
| | `docker-logs` | `docker compose logs -f [SERVICE]` | View logs |
| **Database** | `db-migrate` | `python scripts/db_migrate.py` | Run pending migrations |
| | `db-status` | `python scripts/db_migrate.py --status` | Check migration status |
| | `db-validate` | `python scripts/db_migrate.py --validate` | Validate required tables |
| | `db-reset` | `docker compose down -v` + restart + migrate | Full reset (destructive) |
| | `migrate-create` | Creates timestamped SQL file | New migration (requires `NAME=`) |
| **Maintenance** | `clean` | Remove build/, dist/, *.egg-info, caches, .pyc | Clean artifacts |
| | `pre-commit` | `pre-commit run --all-files` | Run all pre-commit hooks |
| **Composite** | `ci` | lint + typecheck + test | CI pipeline |
| | `check` | lint + typecheck + validate | All checks |
| | `all` | install-dev + check + test + coverage | Full setup + verification |

### Service Endpoints (from `docker-up`)

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:5000 |
| Airflow | http://localhost:8080 |
| Grafana | http://localhost:3002 |
| MinIO Console | http://localhost:9001 |
| MLflow | http://localhost:5001 |
| pgAdmin | http://localhost:5050 |
| Prometheus | http://localhost:9090 |

---

## Test Categories

### Python Tests

| Category | Directory | Marker | CI Job | Trigger |
|----------|-----------|--------|--------|---------|
| Unit | `tests/unit/` | `@pytest.mark.unit` | ci.yml `test` | Every push/PR |
| Integration | `tests/integration/` | `@pytest.mark.integration` | ci.yml `integration-test` | Every push/PR |
| Contracts | `tests/contracts/` + `tests/unit/test_contracts.py` | N/A | contracts-check.yml | Python file changes |
| Regression | `tests/regression/` | N/A | ci.yml `test` | Every push/PR |
| Load | `tests/load/` | `@pytest.mark.load` | ci.yml `load-test` | Schedule or `[load-test]` commit |
| Feature Parity | `tests/integration/test_feature_parity.py`, `test_observation_parity.py` | `@pytest.mark.feature_parity` | ci.yml `feature-parity` | Every push/PR |

### Dashboard Tests (Playwright)

| Category | Directory | Runner | Trigger |
|----------|-----------|--------|---------|
| E2E | `usdcop-trading-dashboard/tests/e2e/` | Playwright | Manual (`npx playwright test`) |

**Playwright Config** (`playwright.config.ts`):

| Setting | Value |
|---------|-------|
| `testDir` | `./tests/e2e` |
| `baseURL` | `http://localhost:5000` (or `$BASE_URL`) |
| `timeout` | 60s per test, 10s per expect |
| `retries` | 2 on CI, 0 locally |
| `workers` | 1 on CI, auto locally |
| `fullyParallel` | true |
| `trace` | on-first-retry |
| `screenshot` | only-on-failure |
| `video` | retain-on-failure |

**Browser Projects**: Chromium, Firefox, WebKit, Mobile Chrome (Pixel 5), Mobile Safari (iPhone 12), Edge, Chrome.

**Web Server**: `npm run dev` on `http://localhost:5000` (reuses existing if running).

**Reporters**: HTML, JSON (`results.json`), JUnit (`results.xml`).

---

## Code Quality Configuration (`pyproject.toml`)

### MyPy (Strict Mode)

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
strict_optional = true
no_implicit_optional = true
show_error_codes = true
```

Third-party imports ignored via `[[tool.mypy.overrides]]` for: numpy, pandas, torch,
stable_baselines3, gymnasium, sklearn, xgboost, lightgbm, catboost, etc.

Excluded directories: `build/`, `dist/`, `__pycache__/`, `venv/`, `notebooks/`, `scripts/`.

### Ruff (Linter)

| Setting | Value |
|---------|-------|
| `target-version` | `py311` |
| `line-length` | 100 |
| `fix` | true |

**Selected rules**: E (pycodestyle errors), W (warnings), F (pyflakes), I (isort),
B (bugbear), C4 (comprehensions), UP (pyupgrade), ARG (unused args), SIM (simplify),
TCH (type checking), RUF (Ruff-specific).

**Ignored**: E501 (line length, handled by formatter), B008 (function call in default),
B904 (raise from), ARG001/ARG002 (unused args in overrides).

**Test overrides**: Allow ARG (unused args in fixtures) and S101 (assert usage).

### Black (Formatter)

| Setting | Value |
|---------|-------|
| `line-length` | 100 |
| `target-version` | `["py311"]` |
| `include` | `\.pyi?$` |

### isort (Import Sorter)

| Setting | Value |
|---------|-------|
| `profile` | `black` |
| `line_length` | 100 |
| `multi_line_output` | 3 (vertical hanging indent) |
| `include_trailing_comma` | true |
| `known_first_party` | `["src", "services", "tests"]` |

### Pytest

| Setting | Value |
|---------|-------|
| `testpaths` | `["tests"]` |
| `addopts` | `-v --tb=short --strict-markers -ra` |
| `asyncio_mode` | `auto` |

**Markers**: `unit`, `integration`, `load`, `slow`, `feature_parity`.

### Coverage

| Setting | Value |
|---------|-------|
| `source` | `["src", "services"]` |
| `branch` | `true` |
| `fail_under` | **70** |
| `omit` | `*/tests/*`, `*/__init__.py`, `*/migrations/*` |

**Excluded patterns**: `pragma: no cover`, `TYPE_CHECKING`, `__repr__`, `@abstractmethod`,
`raise NotImplementedError`, `if __name__`.

### Bandit (Security Linter)

| Setting | Value |
|---------|-------|
| `exclude_dirs` | `["tests", "scripts"]` |
| `skips` | `["B101"]` (assert_used, used in tests) |

---

## Coverage Gate

The 70% minimum coverage threshold is enforced at three levels:

| Level | Where | Setting |
|-------|-------|---------|
| pyproject.toml | `[tool.coverage.report]` | `fail_under = 70` |
| Makefile | `coverage` target | `--cov-fail-under=$(COVERAGE_MIN)` where `COVERAGE_MIN := 70` |
| CI Pipeline | `ci.yml` `test` job | `--cov-fail-under=${{ env.COVERAGE_THRESHOLD }}` where `COVERAGE_THRESHOLD: 70` |

**Coverage sources**: `src/` and `services/` (with branch coverage enabled).

**Reports generated in CI**: XML (Codecov upload), HTML (artifact, 7d retention), terminal.

---

## Security Scanning Summary

Two security workflows cover complementary concerns:

| Concern | `security.yml` | `security-scan.yml` |
|---------|-----------------|---------------------|
| Python static analysis | Bandit (src/ + services/) | -- |
| Dependency CVEs | Safety + pip-audit | Safety + pip-audit |
| Secret detection | -- | Gitleaks (git history) |
| Container scanning | -- | Trivy (filesystem) |
| Environment files | -- | `.env` file check |
| Hardcoded secrets | -- | Pattern grep (password/api_key/token) |
| Schedule | Weekly Sun 3AM UTC | Daily 2AM UTC |

**CD Security** (`deploy.yml`): Trivy container image scan (CRITICAL blocks deploy, HIGH to SARIF).

---

## Drift Monitoring

Feature drift detection runs daily and integrates with the model lifecycle.

### Detection Pipeline

```
Daily 6AM UTC
    |
    v
drift-detection job
    |
    ├── Verify imports (FeatureDriftDetector, MultivariateDriftDetector)
    ├── Univariate check (KS test per feature, p=0.01)
    ├── Multivariate check (MMD, Wasserstein, PCA reconstruction error)
    └── Production check (if config/reference_stats.json exists)
         |
         v
    ┌────┴────┐
    |         |
drift-alert  unit-tests
(if drifted)  (always)
```

### Module: `src/monitoring/drift_detector.py`

| Class | Purpose | Key Parameters |
|-------|---------|----------------|
| `FeatureDriftDetector` | Per-feature KS test | `p_value_threshold=0.01`, `window_size=100`, `min_samples=50` |
| `MultivariateDriftDetector` | Joint distribution drift | `n_features=15`, `window_size=200`, `min_samples=100` |
| `DriftResult` | Single feature result | `feature_name`, `is_drifted`, `drift_severity` |
| `MultivariateDriftResult` | Multi-method result | `score`, `is_drifted` |

---

## SSOT Contract Enforcement

The `contracts-check.yml` workflow prevents contract drift by validating on every Python change.

### What It Checks

| Check | Pass Criteria | Impact if Failed |
|-------|---------------|------------------|
| `FEATURE_ORDER` importable | `len(FEATURE_ORDER) == OBSERVATION_DIM` | Feature mismatch between training and inference |
| `FEATURE_ORDER_HASH` | Recomputed SHA256 == stored hash | Model produces garbage predictions |
| Constants importable | All 8 SSOT constants load correctly | Hardcoded fallbacks may diverge |
| Hardcoded values | No FEATURE_ORDER tuples outside `feature_contract.py` | SSOT violation |
| Cross-module consistency | `src.core.contracts.FEATURE_ORDER == src.feature_store.core.FEATURE_ORDER` | Feature order drift |
| Round-trip | `features_dict_to_array()` -> validate shape + dtype | Observation space broken |

---

## DVC Pipeline Validation

DVC tracks the ML pipeline reproducibility. The workflow validates configuration integrity.

### DVC Version

`dvc[s3]==3.42.0` (pinned in workflow).

### Validated Stages

From `dvc.yaml`: `prepare`, `train`, `evaluate`, `backtest`.

### Parameter Bounds (`params.yaml`)

| Parameter | Valid Range | Checked By |
|-----------|------------|------------|
| `learning_rate` | > 0 | `check-params` |
| `batch_size` | > 0 | `check-params` |
| `gamma` | (0, 1] | `check-params` |
| `clip_range` | (0, 1] | `check-params` |
| `train_ratio + val_ratio` | < 1.0 | `check-params` |
| `initial_balance` | > 0 | `check-params` |

---

## Local Development Workflow

### Recommended Pre-Push Checklist

```bash
# Quick check (< 2 min)
make ci                  # lint + typecheck + test

# Full check (< 5 min)
make check               # lint + typecheck + validate (SSOT + contracts)

# Full verification (< 10 min)
make all                 # install-dev + check + test + coverage

# Pre-commit hooks (< 1 min)
make pre-commit          # All registered pre-commit hooks
```

### Running Specific Test Categories

```bash
make test-unit           # Unit tests only
make test-contracts      # Contract tests (SSOT compliance)
make test-regression     # Regression tests
make test-integration    # Integration tests (requires docker-up)
make coverage            # All tests with 70% gate
```

### Dashboard E2E Tests

```bash
cd usdcop-trading-dashboard
npx playwright test                           # All browsers
npx playwright test --project=chromium        # Chromium only
npx playwright test tests/e2e/health-check    # Specific test
npx playwright show-report                    # View HTML report
```

---

## Dependency Tiers (`pyproject.toml`)

| Tier | Extra | Key Packages |
|------|-------|-------------|
| Base | (none) | numpy, pandas, pyyaml, pydantic, requests |
| ML | `[ml]` | torch, stable-baselines3, gymnasium, sb3-contrib |
| Forecasting | `[forecasting]` | scikit-learn, xgboost, lightgbm, catboost |
| API | `[api]` | fastapi, uvicorn, httpx |
| Database | `[database]` | psycopg2-binary, sqlalchemy, alembic |
| Dev | `[dev]` | pytest, pytest-cov, pytest-asyncio, ruff, black, isort, mypy, pre-commit, bandit |
| Agents | `[agents]` | langchain, openai, anthropic |
| All | `[all]` | All of the above |

Install for development: `pip install -e ".[all]"` or `make install-dev`.

---

## Artifact Retention

| Workflow | Artifact | Retention |
|----------|----------|-----------|
| ci.yml | `coverage-report` (HTML) | 7 days |
| security.yml | `security-reports/` | 30 days |
| experiment.yml | `model-*` (model artifacts) | 30 days |
| experiment.yml | `experiment-report-*` | 90 days |

---

## Related Specs

- `sdd-pipeline-lifecycle.md` -- 8-stage lifecycle (CI/CD integrates at Stage 2 train + Stage 6 deploy)
- `sdd-approval-spec.md` -- Approval gates (CI validates gates, CD deploys after approval)
- `sdd-strategy-spec.md` -- Strategy contracts validated by `contracts-check.yml`
- `experiment-protocol.md` -- Experiment rules enforced by `experiment.yml` workflow
- `ssot-versioning.md` -- SSOT config files validated by `contracts-check.yml`
- `data-freshness-enforcement.md` -- Data gates run in Airflow, drift monitored by `drift-check.yml`

---

## DO NOT

### CI Pipeline
- Do NOT lower the coverage gate below 70% -- it protects against untested code paths
- Do NOT skip the `feature-parity` job -- feature mismatches between training and inference cause silent model failures
- Do NOT remove `--strict-markers` from pytest -- unmarked tests should fail explicitly
- Do NOT add `continue-on-error: true` to the `lint` or `test` jobs -- these are blocking quality gates

### CD Pipeline
- Do NOT skip staging validation for production deployments without explicit `skip_staging: true`
- Do NOT bypass the production environment approval gate -- it requires human review
- Do NOT deploy without a passing security scan (Trivy CRITICAL blocks deploy)
- Do NOT manually edit Docker image tags in GHCR -- tags are generated from git metadata

### Security
- Do NOT commit `.env` files -- `security-scan.yml` will catch and fail the build
- Do NOT hardcode API keys, passwords, or tokens in Python files -- use environment variables
- Do NOT disable Bandit B101 skip outside of `pyproject.toml` -- it is scoped to test assertions
- Do NOT ignore CRITICAL findings from Trivy -- they indicate exploitable vulnerabilities

### Contracts
- Do NOT modify `FEATURE_ORDER` without updating `FEATURE_ORDER_HASH` -- hash mismatch breaks inference
- Do NOT duplicate FEATURE_ORDER outside `src/core/contracts/feature_contract.py` -- use imports
- Do NOT skip `contracts-check.yml` on Python changes -- SSOT drift causes silent model corruption

### Testing
- Do NOT run integration tests without PostgreSQL + Redis services available
- Do NOT add tests without a pytest marker (unit, integration, load, feature_parity)
- Do NOT run Playwright E2E tests without the dashboard dev server running on port 5000
- Do NOT use `test.only` in Playwright specs -- `forbidOnly` is enabled on CI
