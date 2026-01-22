# =============================================================================
# USD/COP Trading System - Makefile
# =============================================================================
# Centralized build, test, and deployment automation.
#
# Usage: make <target>
# Example: make help
# =============================================================================

.PHONY: help install install-dev test test-unit test-contracts test-regression \
        test-integration coverage lint format typecheck validate validate-ssot \
        validate-contracts docker-up docker-down docker-logs db-migrate db-status \
        db-validate db-reset migrate migrate-status migrate-create migrate-rollback \
        clean pre-commit

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project configuration
PYTHON := python
PIP := pip
PYTEST := pytest
DOCKER_COMPOSE := docker compose
COVERAGE_MIN := 70

# =============================================================================
# HELP
# =============================================================================

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)USD/COP Trading System - Available Commands$(RESET)"
	@echo "=============================================="
	@echo ""
	@echo "$(GREEN)Installation:$(RESET)"
	@grep -E '^(install|install-dev):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@grep -E '^(test|test-unit|test-contracts|test-regression|test-integration|coverage):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@grep -E '^(lint|format|typecheck):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Validation:$(RESET)"
	@grep -E '^(validate|validate-ssot|validate-contracts):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@grep -E '^(docker-up|docker-down|docker-logs):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Database:$(RESET)"
	@grep -E '^(db-migrate|db-status|db-validate|db-reset|migrate-create):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Maintenance:$(RESET)"
	@grep -E '^(clean|pre-commit):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# INSTALLATION
# =============================================================================

install: ## Install production dependencies
	@echo "$(CYAN)Installing production dependencies...$(RESET)"
	$(PIP) install -e .
	@echo "$(GREEN)Production dependencies installed successfully!$(RESET)"

install-dev: ## Install development dependencies (includes testing, linting tools)
	@echo "$(CYAN)Installing development dependencies...$(RESET)"
	$(PIP) install -e ".[all]"
	pre-commit install
	@echo "$(GREEN)Development dependencies installed successfully!$(RESET)"

# =============================================================================
# TESTING
# =============================================================================

test: ## Run all tests (unit, integration, contracts, regression)
	@echo "$(CYAN)Running all tests...$(RESET)"
	$(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)All tests completed!$(RESET)"

test-unit: ## Run unit tests only
	@echo "$(CYAN)Running unit tests...$(RESET)"
	$(PYTEST) tests/unit/ -v --tb=short -m "unit or not (integration or contracts or regression or load or chaos)"
	@echo "$(GREEN)Unit tests completed!$(RESET)"

test-contracts: ## Run contract tests (API contracts, feature contracts)
	@echo "$(CYAN)Running contract tests...$(RESET)"
	$(PYTEST) tests/contracts/ tests/unit/test_contracts.py tests/unit/test_gtr_contracts.py tests/unit/test_all_layer_contracts.py -v --tb=short
	@echo "$(GREEN)Contract tests completed!$(RESET)"

test-regression: ## Run regression tests
	@echo "$(CYAN)Running regression tests...$(RESET)"
	$(PYTEST) tests/regression/ -v --tb=short
	@echo "$(GREEN)Regression tests completed!$(RESET)"

test-integration: ## Run integration tests (requires running services)
	@echo "$(CYAN)Running integration tests...$(RESET)"
	$(PYTEST) tests/integration/ -v --tb=short -m "integration"
	@echo "$(GREEN)Integration tests completed!$(RESET)"

coverage: ## Run tests with coverage report
	@echo "$(CYAN)Running tests with coverage...$(RESET)"
	$(PYTEST) tests/ -v --cov=src --cov=services --cov-report=term-missing --cov-report=html --cov-fail-under=$(COVERAGE_MIN)
	@echo "$(GREEN)Coverage report generated in htmlcov/$(RESET)"

# =============================================================================
# CODE QUALITY
# =============================================================================

lint: ## Run linter (ruff) on source code
	@echo "$(CYAN)Running linter...$(RESET)"
	ruff check src/ services/ tests/
	@echo "$(GREEN)Linting completed!$(RESET)"

format: ## Format code with black and isort
	@echo "$(CYAN)Formatting code...$(RESET)"
	black src/ services/ tests/
	isort src/ services/ tests/
	ruff check --fix src/ services/ tests/
	@echo "$(GREEN)Code formatting completed!$(RESET)"

typecheck: ## Run type checker (mypy)
	@echo "$(CYAN)Running type checker...$(RESET)"
	mypy src/ services/ --config-file pyproject.toml
	@echo "$(GREEN)Type checking completed!$(RESET)"

# =============================================================================
# VALIDATION
# =============================================================================

validate: validate-ssot validate-contracts ## Run all validations
	@echo "$(GREEN)All validations passed!$(RESET)"

validate-ssot: ## Validate Single Source of Truth (feature registry, configs)
	@echo "$(CYAN)Validating SSOT consistency...$(RESET)"
	$(PYTEST) tests/unit/test_ssot_consistency.py tests/unit/test_config_ssot.py tests/unit/test_config_consistency.py -v
	@echo "$(GREEN)SSOT validation completed!$(RESET)"

validate-contracts: ## Validate all data contracts
	@echo "$(CYAN)Validating data contracts...$(RESET)"
	$(PYTEST) tests/contracts/ tests/unit/test_contracts.py tests/unit/test_feature_contract.py -v
	@echo "$(GREEN)Contract validation completed!$(RESET)"

# =============================================================================
# DOCKER
# =============================================================================

docker-up: ## Start all Docker services
	@echo "$(CYAN)Starting Docker services...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Docker services started!$(RESET)"
	@echo "$(YELLOW)Services available at:$(RESET)"
	@echo "  Dashboard:     http://localhost:5000"
	@echo "  Airflow:       http://localhost:8080"
	@echo "  Grafana:       http://localhost:3002"
	@echo "  MinIO Console: http://localhost:9001"
	@echo "  MLflow:        http://localhost:5001"
	@echo "  pgAdmin:       http://localhost:5050"
	@echo "  Prometheus:    http://localhost:9090"

docker-down: ## Stop all Docker services
	@echo "$(CYAN)Stopping Docker services...$(RESET)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Docker services stopped!$(RESET)"

docker-logs: ## View Docker service logs (use SERVICE=name for specific service)
	@echo "$(CYAN)Viewing Docker logs...$(RESET)"
ifdef SERVICE
	$(DOCKER_COMPOSE) logs -f $(SERVICE)
else
	$(DOCKER_COMPOSE) logs -f
endif

# =============================================================================
# DATABASE MIGRATIONS (Python-based, idempotent)
# =============================================================================

db-migrate: ## Run all pending database migrations (init-scripts)
	@echo "$(CYAN)Running database migrations...$(RESET)"
	$(PYTHON) scripts/db_migrate.py
	@echo "$(GREEN)Migrations completed!$(RESET)"

db-status: ## Show migration status
	@echo "$(CYAN)Checking migration status...$(RESET)"
	$(PYTHON) scripts/db_migrate.py --status

db-validate: ## Validate all required tables exist
	@echo "$(CYAN)Validating database schema...$(RESET)"
	$(PYTHON) scripts/db_migrate.py --validate

db-reset: ## Reset database (DESTRUCTIVE - deletes all data)
	@echo "$(RED)WARNING: This will delete ALL data!$(RESET)"
	@echo "Press Ctrl+C to cancel, or wait 5 seconds..."
	@sleep 5
	$(DOCKER_COMPOSE) down -v
	$(DOCKER_COMPOSE) up -d postgres
	@echo "Waiting for PostgreSQL to start..."
	@sleep 10
	$(PYTHON) scripts/db_migrate.py

# Legacy aliases for backwards compatibility
migrate: db-migrate ## (Legacy) Alias for db-migrate
migrate-status: db-status ## (Legacy) Alias for db-status

migrate-create: ## Create a new migration file (use NAME=description)
	@echo "$(CYAN)Creating new migration...$(RESET)"
ifndef NAME
	$(error NAME is required. Usage: make migrate-create NAME=add_new_table)
endif
	@TIMESTAMP=$$(date +%Y%m%d%H%M%S); \
	FILENAME="init-scripts/$${TIMESTAMP}_$(NAME).sql"; \
	echo "-- Migration: $(NAME)" > "$$FILENAME"; \
	echo "-- Created: $$(date -Iseconds)" >> "$$FILENAME"; \
	echo "" >> "$$FILENAME"; \
	echo "-- Add your migration SQL here" >> "$$FILENAME"; \
	echo "" >> "$$FILENAME"; \
	echo "$(GREEN)Created migration: $$FILENAME$(RESET)"

migrate-rollback: ## Rollback last migration (requires manual implementation)
	@echo "$(YELLOW)Migration rollback must be done manually.$(RESET)"
	@echo "Check init-scripts/ for recent migrations."
	@echo "Create a rollback script or use psql directly."

# =============================================================================
# MAINTENANCE
# =============================================================================

clean: ## Clean build artifacts, caches, and temporary files
	@echo "$(CYAN)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup completed!$(RESET)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(CYAN)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit checks completed!$(RESET)"

# =============================================================================
# COMPOSITE TARGETS
# =============================================================================

ci: lint typecheck test ## Run CI pipeline (lint, typecheck, test)
	@echo "$(GREEN)CI pipeline completed successfully!$(RESET)"

check: lint typecheck validate ## Run all checks (lint, typecheck, validate)
	@echo "$(GREEN)All checks passed!$(RESET)"

all: install-dev check test coverage ## Full development setup and verification
	@echo "$(GREEN)Full setup and verification completed!$(RESET)"
