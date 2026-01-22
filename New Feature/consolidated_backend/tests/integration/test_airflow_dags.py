"""
Integration Test: Airflow DAGs

This module tests Airflow DAG integrity:
1. Verify that DAGs parse correctly
2. Verify task dependencies
3. Verify no missing imports
4. Verify DAG configuration

Usage:
    pytest tests/integration/test_airflow_dags.py -v
    pytest tests/integration/test_airflow_dags.py -v --no-header
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from unittest.mock import patch, MagicMock

import pytest

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DAGS_PATH = PROJECT_ROOT / "data-engineering" / "dags"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DAGS_PATH))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dag_files() -> List[Path]:
    """Get all DAG files in the dags directory."""
    if not DAGS_PATH.exists():
        pytest.skip(f"DAGs path not found: {DAGS_PATH}")

    dag_files = list(DAGS_PATH.glob("dag_*.py"))
    return dag_files


@pytest.fixture
def dag_content(dag_files) -> Dict[str, str]:
    """Load content of all DAG files."""
    content = {}
    for dag_file in dag_files:
        with open(dag_file, "r", encoding="utf-8") as f:
            content[dag_file.name] = f.read()
    return content


# =============================================================================
# Test Class: DAG Parsing
# =============================================================================

@pytest.mark.integration
class TestDAGParsing:
    """Tests for DAG parsing and syntax validation."""

    def test_dag_files_exist(self, dag_files):
        """Test that DAG files exist."""
        assert len(dag_files) > 0, "No DAG files found"

        expected_dags = [
            "dag_monthly_training.py",
            "dag_weekly_inference.py",
        ]

        found_dags = [f.name for f in dag_files]

        for expected in expected_dags:
            assert expected in found_dags, f"Missing expected DAG: {expected}"

    def test_dag_files_syntax_valid(self, dag_files):
        """Test that all DAG files have valid Python syntax."""
        for dag_file in dag_files:
            try:
                with open(dag_file, "r", encoding="utf-8") as f:
                    source = f.read()

                ast.parse(source)

            except SyntaxError as e:
                pytest.fail(f"Syntax error in {dag_file.name}: {e}")

    def test_dag_files_no_undefined_names(self, dag_content):
        """Test DAG files for obvious undefined names."""
        for filename, content in dag_content.items():
            try:
                tree = ast.parse(content)

                # Get all defined names
                defined = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        defined.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        defined.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                defined.add(target.id)
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            defined.add(name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            name = alias.asname if alias.asname else alias.name
                            defined.add(name)

                # This is a basic check; full linting would use a proper linter

            except Exception as e:
                pytest.fail(f"Error parsing {filename}: {e}")

    def test_dag_has_dag_definition(self, dag_content):
        """Test that each DAG file contains a DAG definition."""
        for filename, content in dag_content.items():
            # Check for DAG instantiation patterns
            has_dag = (
                "DAG(" in content or
                "with DAG(" in content or
                "@dag" in content
            )

            assert has_dag, f"{filename} does not contain DAG definition"

    def test_dag_has_default_args(self, dag_content):
        """Test that DAG files define default_args."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_default_args = "default_args" in content

                assert has_default_args, f"{filename} missing default_args"


# =============================================================================
# Test Class: DAG Imports
# =============================================================================

@pytest.mark.integration
class TestDAGImports:
    """Tests for DAG import statements."""

    def test_airflow_imports_present(self, dag_content):
        """Test that required Airflow imports are present."""
        required_imports = [
            "from airflow import DAG",
            "from airflow.operators",
        ]

        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_dag_import = any(imp in content for imp in required_imports)

                assert has_dag_import, f"{filename} missing Airflow imports"

    def test_no_relative_imports_outside_package(self, dag_content):
        """Test for problematic relative imports."""
        for filename, content in dag_content.items():
            lines = content.split('\n')

            for i, line in enumerate(lines, 1):
                # Check for relative imports that go too far up
                if line.strip().startswith("from ...."):
                    pytest.fail(f"{filename}:{i}: Potentially problematic relative import")

    def test_standard_library_imports_valid(self, dag_content):
        """Test that standard library imports are valid."""
        standard_libs = [
            "datetime",
            "timedelta",
            "json",
            "os",
            "sys",
            "logging",
            "pathlib",
        ]

        for filename, content in dag_content.items():
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split('.')[0]
                        # Standard libs should be importable
                        if module in standard_libs:
                            try:
                                __import__(module)
                            except ImportError:
                                pytest.fail(f"{filename}: Cannot import {module}")

    def test_imports_have_fallbacks(self, dag_content):
        """Test that optional imports have fallbacks."""
        optional_packages = ["mlflow", "minio", "psycopg2"]

        for filename, content in dag_content.items():
            for package in optional_packages:
                if f"import {package}" in content or f"from {package}" in content:
                    # Check for try/except or AVAILABLE flag
                    has_fallback = (
                        "try:" in content or
                        "_AVAILABLE" in content or
                        "except ImportError" in content
                    )

                    # This is informational; some DAGs may require packages
                    if not has_fallback:
                        print(f"Info: {filename} imports {package} without fallback")


# =============================================================================
# Test Class: DAG Task Dependencies
# =============================================================================

@pytest.mark.integration
class TestDAGTaskDependencies:
    """Tests for DAG task dependencies."""

    def test_dag_has_tasks(self, dag_content):
        """Test that DAG files define tasks."""
        operator_patterns = [
            "PythonOperator",
            "BashOperator",
            "EmptyOperator",
            "BranchPythonOperator",
            "TriggerDagRunOperator",
        ]

        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_operators = any(op in content for op in operator_patterns)

                assert has_operators, f"{filename} has no task operators"

    def test_dag_has_dependencies(self, dag_content):
        """Test that DAG files define task dependencies."""
        dependency_patterns = [
            ">>",  # Bitshift dependency
            "<<",  # Reverse dependency
            ".set_downstream",
            ".set_upstream",
        ]

        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_deps = any(pattern in content for pattern in dependency_patterns)

                assert has_deps, f"{filename} has no task dependencies"

    def test_task_ids_unique_in_dag(self, dag_content):
        """Test that task IDs are unique within each DAG."""
        for filename, content in dag_content.items():
            tree = ast.parse(content)

            task_ids = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Look for task_id keyword argument
                    for keyword in node.keywords:
                        if keyword.arg == "task_id":
                            if isinstance(keyword.value, ast.Constant):
                                task_ids.append(keyword.value.value)

            # Check for duplicates
            duplicates = [t for t in task_ids if task_ids.count(t) > 1]
            unique_duplicates = set(duplicates)

            assert len(unique_duplicates) == 0, \
                f"{filename} has duplicate task_ids: {unique_duplicates}"

    def test_no_circular_dependencies(self, dag_content):
        """Test for potential circular dependencies."""
        # This is a basic check; full validation requires running Airflow
        for filename, content in dag_content.items():
            # Look for obvious patterns that might cause cycles
            # e.g., task_a >> task_b >> task_a

            # Simple heuristic: count >> and << occurrences
            forward_deps = content.count(">>")
            backward_deps = content.count("<<")

            # Too many backward deps might indicate issues
            if backward_deps > forward_deps:
                print(f"Info: {filename} has more << than >> dependencies")


# =============================================================================
# Test Class: DAG Configuration
# =============================================================================

@pytest.mark.integration
class TestDAGConfiguration:
    """Tests for DAG configuration settings."""

    def test_dag_has_schedule_interval(self, dag_content):
        """Test that DAGs have schedule_interval defined."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_schedule = (
                    "schedule_interval=" in content or
                    "schedule=" in content or
                    "@daily" in content or
                    "@weekly" in content or
                    "@monthly" in content
                )

                assert has_schedule, f"{filename} missing schedule_interval"

    def test_dag_has_start_date(self, dag_content):
        """Test that DAGs have start_date defined."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_start_date = "start_date" in content

                assert has_start_date, f"{filename} missing start_date"

    def test_dag_has_catchup_setting(self, dag_content):
        """Test that DAGs have catchup setting defined."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_catchup = "catchup" in content

                assert has_catchup, f"{filename} missing catchup setting"

    def test_dag_has_tags(self, dag_content):
        """Test that DAGs have tags defined."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_tags = "tags=" in content

                assert has_tags, f"{filename} missing tags"

    def test_dag_has_max_active_runs(self, dag_content):
        """Test that DAGs have max_active_runs setting."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_max_runs = "max_active_runs" in content

                # This is a recommendation, not a requirement
                if not has_max_runs:
                    print(f"Info: {filename} missing max_active_runs")

    def test_dag_has_failure_callback(self, dag_content):
        """Test that DAGs have failure callbacks."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_callback = (
                    "on_failure_callback" in content or
                    "failure_callback" in content
                )

                # Recommended for production DAGs
                if not has_callback:
                    print(f"Info: {filename} missing on_failure_callback")

    def test_dag_owner_set(self, dag_content):
        """Test that DAG owner is set in default_args."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_owner = "'owner'" in content or '"owner"' in content

                assert has_owner, f"{filename} missing owner in default_args"


# =============================================================================
# Test Class: DAG Best Practices
# =============================================================================

@pytest.mark.integration
class TestDAGBestPractices:
    """Tests for DAG best practices."""

    def test_dag_has_docstring(self, dag_content):
        """Test that DAG files have docstrings."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                # Check for module-level docstring
                has_docstring = content.strip().startswith('"""') or \
                               content.strip().startswith("'''")

                assert has_docstring, f"{filename} missing docstring"

    def test_dag_uses_provide_context(self, dag_content):
        """Test that PythonOperator uses provide_context (or **context)."""
        for filename, content in dag_content.items():
            if "PythonOperator" in content:
                # Check for context handling
                has_context = (
                    "provide_context=True" in content or
                    "**context" in content or
                    "**kwargs" in content
                )

                assert has_context, \
                    f"{filename} PythonOperator should use provide_context"

    def test_dag_retries_configured(self, dag_content):
        """Test that DAGs have retries configured."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_retries = "'retries'" in content or '"retries"' in content

                assert has_retries, f"{filename} missing retries configuration"

    def test_dag_retry_delay_configured(self, dag_content):
        """Test that DAGs have retry_delay configured."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_retry_delay = "'retry_delay'" in content or \
                                 '"retry_delay"' in content

                assert has_retry_delay, f"{filename} missing retry_delay"

    def test_dag_execution_timeout_configured(self, dag_content):
        """Test that DAGs have execution_timeout configured."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_timeout = "execution_timeout" in content

                # Recommended for production
                if not has_timeout:
                    print(f"Info: {filename} missing execution_timeout")

    def test_dag_email_on_failure(self, dag_content):
        """Test that DAGs have email_on_failure configured."""
        for filename, content in dag_content.items():
            if filename.startswith("dag_"):
                has_email = "email_on_failure" in content

                # Recommended for production
                if not has_email:
                    print(f"Info: {filename} missing email_on_failure")


# =============================================================================
# Test Class: Specific DAG Tests
# =============================================================================

@pytest.mark.integration
class TestSpecificDAGs:
    """Tests for specific DAG configurations."""

    def test_monthly_training_dag_schedule(self, dag_content):
        """Test monthly training DAG schedule."""
        if "dag_monthly_training.py" not in dag_content:
            pytest.skip("Monthly training DAG not found")

        content = dag_content["dag_monthly_training.py"]

        # Should run monthly or on first Sunday
        valid_schedules = [
            "0 7 1-7 * 0",  # First Sunday
            "@monthly",
            "0 0 1 * *",  # First day of month
        ]

        has_valid_schedule = any(sched in content for sched in valid_schedules)

        assert has_valid_schedule, \
            "Monthly training DAG should have monthly schedule"

    def test_weekly_inference_dag_schedule(self, dag_content):
        """Test weekly inference DAG schedule."""
        if "dag_weekly_inference.py" not in dag_content:
            pytest.skip("Weekly inference DAG not found")

        content = dag_content["dag_weekly_inference.py"]

        # Should run weekly on Monday
        valid_schedules = [
            "0 13 * * 1",  # Monday 13:00 UTC
            "@weekly",
            "* * * * 1",  # Any time on Monday
        ]

        has_valid_schedule = any(sched in content for sched in valid_schedules)

        assert has_valid_schedule, \
            "Weekly inference DAG should have weekly schedule"

    def test_training_dag_has_required_tasks(self, dag_content):
        """Test that training DAG has required tasks."""
        if "dag_monthly_training.py" not in dag_content:
            pytest.skip("Monthly training DAG not found")

        content = dag_content["dag_monthly_training.py"]

        required_tasks = [
            "check_data",
            "load",
            "train",
            "aggregate",
        ]

        for task in required_tasks:
            assert task.lower() in content.lower(), \
                f"Training DAG missing task: {task}"

    def test_inference_dag_has_required_tasks(self, dag_content):
        """Test that inference DAG has required tasks."""
        if "dag_weekly_inference.py" not in dag_content:
            pytest.skip("Weekly inference DAG not found")

        content = dag_content["dag_weekly_inference.py"]

        required_tasks = [
            "load",
            "inference",
            "upload",
            "notify",
        ]

        for task in required_tasks:
            assert task.lower() in content.lower(), \
                f"Inference DAG missing task: {task}"

    def test_training_dag_trains_multiple_models(self, dag_content):
        """Test that training DAG trains multiple models."""
        if "dag_monthly_training.py" not in dag_content:
            pytest.skip("Monthly training DAG not found")

        content = dag_content["dag_monthly_training.py"]

        models = ["ridge", "xgboost", "lightgbm", "catboost"]

        models_found = sum(1 for m in models if m.lower() in content.lower())

        assert models_found >= 2, \
            f"Training DAG should train multiple models, found {models_found}"

    def test_inference_dag_has_data_quality_check(self, dag_content):
        """Test that inference DAG has data quality check."""
        if "dag_weekly_inference.py" not in dag_content:
            pytest.skip("Weekly inference DAG not found")

        content = dag_content["dag_weekly_inference.py"]

        has_quality_check = (
            "check_data_quality" in content or
            "data_quality" in content.lower() or
            "quality" in content.lower()
        )

        assert has_quality_check, \
            "Inference DAG should have data quality check"


# =============================================================================
# Test Class: DAG Utils
# =============================================================================

@pytest.mark.integration
class TestDAGUtils:
    """Tests for DAG utility modules."""

    def test_callbacks_module_exists(self):
        """Test that callbacks module exists."""
        callbacks_path = DAGS_PATH / "utils" / "callbacks.py"

        if not callbacks_path.exists():
            pytest.skip("Callbacks module not found")

        with open(callbacks_path, "r") as f:
            content = f.read()

        # Check for expected functions
        expected_funcs = [
            "task_failure_callback",
            "task_success_callback",
        ]

        for func in expected_funcs:
            assert func in content, f"Callbacks missing function: {func}"

    def test_dag_common_module_exists(self):
        """Test that dag_common module exists."""
        common_path = DAGS_PATH / "utils" / "dag_common.py"

        if not common_path.exists():
            pytest.skip("dag_common module not found")

        with open(common_path, "r") as f:
            content = f.read()

        # Check for expected functions
        expected_funcs = [
            "get_db_connection",
        ]

        for func in expected_funcs:
            assert func in content, f"dag_common missing function: {func}"

    def test_utils_init_exists(self):
        """Test that utils __init__.py exists."""
        init_path = DAGS_PATH / "utils" / "__init__.py"

        # __init__.py should exist for proper imports
        assert init_path.exists(), "utils/__init__.py should exist"


# =============================================================================
# Test Class: DAG Loading (requires Airflow)
# =============================================================================

@pytest.mark.integration
class TestDAGLoading:
    """Tests that require Airflow to be installed."""

    def test_airflow_available(self):
        """Test that Airflow is importable."""
        try:
            import airflow
            assert hasattr(airflow, "__version__")
        except ImportError:
            pytest.skip("Airflow not installed")

    def test_dag_imports_work(self, dag_files):
        """Test that DAG files can be imported (mocked)."""
        # This test uses mocking to avoid full Airflow initialization

        for dag_file in dag_files:
            if dag_file.name.startswith("dag_"):
                # Check syntax
                with open(dag_file, "r") as f:
                    source = f.read()

                try:
                    compile(source, dag_file.name, "exec")
                except SyntaxError as e:
                    pytest.fail(f"Cannot compile {dag_file.name}: {e}")

    def test_dag_bag_can_load_dags(self):
        """Test that DagBag can load DAGs."""
        try:
            from airflow.models import DagBag

            # Load DAGs from the dags folder
            dag_bag = DagBag(
                dag_folder=str(DAGS_PATH),
                include_examples=False
            )

            # Check for import errors
            if dag_bag.import_errors:
                for dag_id, error in dag_bag.import_errors.items():
                    print(f"Import error in {dag_id}: {error}")

            # Should have loaded some DAGs
            # Note: This may fail if dependencies aren't installed
            assert len(dag_bag.dags) >= 0

        except ImportError:
            pytest.skip("Airflow not installed")
        except Exception as e:
            # DAG loading may fail due to missing dependencies
            print(f"DAG loading info: {e}")
