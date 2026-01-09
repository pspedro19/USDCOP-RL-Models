@echo off
REM USD/COP Trading System - Test Suite Runner
REM ==========================================
REM
REM Quick test execution commands
REM Author: Pedro @ Lean Tech Solutions
REM Date: 2025-12-16

echo.
echo ============================================================
echo  USD/COP Trading System - Test Suite
echo ============================================================
echo.

if "%1"=="all" goto :run_all
if "%1"=="unit" goto :run_unit
if "%1"=="integration" goto :run_integration
if "%1"=="parity" goto :run_parity
if "%1"=="quick" goto :run_quick
if "%1"=="coverage" goto :run_coverage
if "%1"=="" goto :show_help

:show_help
echo Usage: RUN_TESTS.bat [command]
echo.
echo Commands:
echo   all          - Run all tests (unit + integration)
echo   unit         - Run only unit tests (fast)
echo   integration  - Run only integration tests
echo   parity       - Run CRITICAL feature parity test
echo   quick        - Run quick smoke tests
echo   coverage     - Run all tests with coverage report
echo.
echo Examples:
echo   RUN_TESTS.bat unit
echo   RUN_TESTS.bat parity
echo   RUN_TESTS.bat coverage
echo.
goto :eof

:run_all
echo Running all tests...
pytest tests/ -v
goto :eof

:run_unit
echo Running unit tests only...
pytest tests/unit/ -v
goto :eof

:run_integration
echo Running integration tests...
pytest tests/integration/ -v
goto :eof

:run_parity
echo Running CRITICAL feature parity test...
pytest tests/integration/test_feature_parity.py::TestFeatureParityLegacy::test_features_match_legacy -v
goto :eof

:run_quick
echo Running quick smoke tests...
pytest tests/unit/test_config_loader.py tests/unit/test_feature_builder.py::TestObservationSpace -v
goto :eof

:run_coverage
echo Running all tests with coverage...
pytest tests/ --cov=services --cov-report=html --cov-report=term
echo.
echo Coverage report generated in htmlcov/index.html
goto :eof
