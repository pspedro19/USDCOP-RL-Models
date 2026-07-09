"""
Regression: scripts/ purpose-based layout (2026-07 reorg).

Stale-proofs the reorganization — if a load-bearing script is moved again without
updating its automation refs, or a loose script is dropped back at scripts/ root,
these assertions fail. Paths are derived from the reorg contract, not a snapshot.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

# Load-bearing operational entrypoints and where automation now expects them.
# (These paths are wired into airflow DAGs / deploy manifest / Makefile / dvc / CI.)
LOAD_BEARING = {
    "pipeline/train_and_export_smart_simple.py",
    "pipeline/generate_weekly_forecasts.py",
    "pipeline/generate_weekly_analysis.py",
    "pipeline/run_ssot_pipeline.py",
    "pipeline/export_to_onnx.py",
    "pipeline/promote_model.py",
    "pipeline/backtest.py",
    "data/prepare_training_data.py",
    "data/build_forecasting_dataset_aligned.py",
    "data/generate_validation_datasets.py",
    "ops/db_migrate.py",
    "ops/log_training_to_mlflow.py",
    "ops/migrate_imports.py",
    "analysis/backtest_2026_production.py",
    "lib/vol_target_backtest.py",
}

PURPOSE_SUBDIRS = [
    "pipeline", "data", "ops", "analysis", "diagnostics",
    "validation", "presentation", "tools", "lib", "archive",
]


@pytest.mark.parametrize("rel", sorted(LOAD_BEARING))
def test_load_bearing_script_exists_at_new_path(rel):
    assert (SCRIPTS / rel).is_file(), f"load-bearing script missing: scripts/{rel}"


def test_scripts_root_has_no_loose_files_except_init():
    """Root of scripts/ must hold only the package marker — every runnable script
    (.py/.sh/.sql/.ps1) lives in a purpose subdir.

    Untracked WIP (e.g. Gold `run_gold_pipeline.py`) is ignored — this guards the
    tracked tree only, using git to enumerate.
    """
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "scripts/"], cwd=ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    loose = [
        p for p in out
        if p.startswith("scripts/")
        and p.count("/") == 1
        and Path(p).name != "__init__.py"
    ]
    assert not loose, f"loose files at scripts/ root (should be in a subdir): {loose}"


def test_purpose_subdirs_are_python_packages():
    """scripts is an importable package (from scripts.pipeline.X import ...)."""
    for d in PURPOSE_SUBDIRS:
        assert (SCRIPTS / d / "__init__.py").is_file(), f"missing scripts/{d}/__init__.py"


def test_archive_has_readme_and_legacy_scripts():
    assert (SCRIPTS / "archive" / "README.md").is_file()
    for f in ("cron_m5_fetch.py", "ensure_db_tables.py", "build_course_pptx.py"):
        assert (SCRIPTS / "archive" / f).is_file(), f"expected archived: {f}"
    # and they must NOT still be at root
    assert not (SCRIPTS / "cron_m5_fetch.py").exists()


def test_deploy_manifest_points_at_new_pipeline_path():
    """The embedded deploy_manifest must reference the relocated script."""
    src = (SCRIPTS / "pipeline" / "train_and_export_smart_simple.py").read_text(encoding="utf-8")
    assert '"script": "scripts/pipeline/train_and_export_smart_simple.py"' in src


# A depth-2 script (scripts/<subdir>/x.py) that anchors on the REPO ROOT must reach it with
# parents[2] — NOT parents[1] / parent.parent (which land in scripts/). The 2026-07 reorg moved
# scripts down one level and silently broke several PROJECT_ROOT calcs (config/src/.env resolved
# under scripts/). This guards against that class of regression recurring.
_ROOT_VAR = re.compile(
    r"^\s*(?:PROJECT_ROOT|REPO_ROOT)\s*=\s*Path\(__file__\)\.resolve\(\)\."
    r"(parents\[1\]|parent\.parent)(?!\.)",
    re.MULTILINE,
)


def test_repo_root_vars_in_subdir_scripts_use_parents_2():
    offenders = []
    for py in SCRIPTS.glob("*/*.py"):
        if py.name == "__init__.py":
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        if _ROOT_VAR.search(text):
            offenders.append(py.relative_to(ROOT).as_posix())
    assert not offenders, (
        "depth-2 script(s) compute PROJECT_ROOT/REPO_ROOT with parents[1]/parent.parent "
        f"(resolves to scripts/, not the repo root) — use parents[2]: {offenders}"
    )
