"""
CTR-DEPLOY-CMD-001 — el manifiesto de deploy no puede elegir QUÉ se ejecuta.
============================================================================

El hallazgo original vive en el dashboard (``/api/production/deploy`` lanzaba
``spawn('python3', [manifest.script, ...manifest.args], {shell: true})`` ⇒ inyección de
comandos). Al cerrarlo aparece la OTRA vía de ejecución del mismo artefacto: el DAG
``forecast_h5_l4b_production_deploy`` — que además es el camino PREFERIDO en producción
(el contenedor de Node no tiene Python, así que el deploy real lo corre Airflow).

Ese DAG hacía:

    script = PROJECT_ROOT / plan['script']          # plan['script'] sale del manifiesto
    cmd = [sys.executable, str(script), *plan['args']]
    subprocess.run(cmd, ...)                        # sin shell → no hay inyección…

…pero ``pathlib`` tiene una semántica que aquí es una vulnerabilidad: **si el operando
derecho es absoluto, el izquierdo se DESCARTA**. ``Path('/opt/airflow') / '/etc/x.py'``
es ``/etc/x.py``. Es decir: el manifiesto podía elegir cualquier fichero del contenedor
como "script de deploy" (traversal con ``..`` incluido), y sus argumentos iban sin
validar. No es ejecución de shell arbitraria, pero sí ejecución de un programa arbitrario
en la ruta que promueve a producción.

Este fichero fija el contrato compartido (espejo de
``usdcop-trading-dashboard/lib/security/deploy-command.ts``):

  1. Allowlist de ``script``: relativo POSIX, bajo ``scripts/``, ``.py``, existente y con
     su ``realpath`` DENTRO del árbol permitido.
  2. Allowlist de forma para ``args`` (bandera o valor simple).
  3. Fail-closed: cualquier duda ⇒ excepción, jamás ejecución.
  4. El DAG usa ESTE validador (y ya no construye la ruta a mano).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.contracts.deploy_manifest import (
    LEGACY_ARGS,
    LEGACY_SCRIPT,
    DeployManifestRejected,
    resolve_deploy_command,
)

REPO = Path(__file__).resolve().parents[2]
DAG_FILE = REPO / "airflow" / "dags" / "forecast_h5_l4b_production_deploy.py"

VALID = {
    "script": "scripts/pipeline/train_and_export_smart_simple.py",
    "args": ["--phase", "production", "--no-png", "--seed-db"],
}


# ───────────────────────────────────────────────── 1 · el camino legítimo sigue vivo


def test_valid_manifest_resolves_inside_the_scripts_tree():
    cmd = resolve_deploy_command(REPO, VALID)
    assert cmd.script_path == (REPO / "scripts" / "pipeline" / "train_and_export_smart_simple.py").resolve()
    assert cmd.args == ["--phase", "production", "--no-png", "--seed-db"]
    assert cmd.legacy is False


def test_every_manifest_published_by_the_pipeline_is_accepted():
    """Los manifiestos REALES en data/approvals/ deben pasar: si el contrato los
    rechazara, este remedio habría roto el deploy en vez de asegurarlo."""
    import json

    manifests = []
    for p in sorted((REPO / "data" / "approvals").glob("approval_state*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if doc.get("deploy_manifest"):
            manifests.append((p.name, doc["deploy_manifest"]))
    assert manifests, "no hay manifiestos publicados que verificar"
    for name, m in manifests:
        cmd = resolve_deploy_command(REPO, m)
        assert cmd.script_path.is_file(), name


def test_missing_manifest_falls_back_to_the_legacy_command():
    cmd = resolve_deploy_command(REPO, None)
    assert cmd.script_rel == LEGACY_SCRIPT
    assert cmd.args == list(LEGACY_ARGS)
    assert cmd.legacy is True


def test_interpreter_is_explicit_never_a_shell():
    cmd = resolve_deploy_command(REPO, VALID, interpreter="/usr/local/bin/python3.12")
    assert cmd.interpreter == "/usr/local/bin/python3.12"


# ───────────────────────────────────────────── 2 · allowlist de ubicación (traversal)


@pytest.mark.parametrize(
    "script",
    [
        "/etc/evil.py",                                   # absoluto POSIX (el bug de pathlib)
        "C:\\Windows\\Temp\\evil.py",                     # absoluto Windows
        "../../../etc/evil.py",                           # traversal
        "scripts/../airflow/dags/forecast_h5_l4b_production_deploy.py",
        "airflow/dags/forecast_h5_l4b_production_deploy.py",  # dentro del repo, fuera de scripts/
        "scripts/ops/backup/restore.sh",                  # no es .py
        "scripts/pipeline/__no_existe__.py",              # no existe
        "scripts/pipeline/train_and_export_smart_simple.py; whoami",
        "scripts/pipeline/x.py && whoami",
        "$(whoami).py",
        "",
        None,
        42,
    ],
)
def test_script_outside_the_allowlist_is_rejected(script):
    with pytest.raises(DeployManifestRejected):
        resolve_deploy_command(REPO, {"script": script, "args": []})


@pytest.mark.parametrize(
    "args",
    [
        ["--phase", "production; whoami"],
        ["--phase", "production && whoami"],
        ["--phase", "$HOME"],
        ["--phase", "`whoami`"],
        ["--config", "/etc/shadow"],
        ["--phase", "produ'ction"],
        ["--phase", 42],
        "no-soy-una-lista",
        ["--x"] * 50,
    ],
)
def test_args_outside_the_declared_shape_are_rejected(args):
    with pytest.raises(DeployManifestRejected):
        resolve_deploy_command(REPO, {**VALID, "args": args})


def test_rejection_carries_field_and_reason_for_the_audit_trail():
    with pytest.raises(DeployManifestRejected) as exc:
        resolve_deploy_command(REPO, {"script": "/etc/evil.py", "args": []})
    assert exc.value.field == "script"
    assert exc.value.reason


# ─────────────────────────────────────── 3 · el DAG usa el validador, no la ruta cruda


def test_dag_resolves_the_manifest_through_the_shared_validator():
    src = DAG_FILE.read_text(encoding="utf-8")
    assert "resolve_deploy_command" in src, "el DAG H5-L4b no usa el validador compartido"


def test_dag_no_longer_builds_the_script_path_by_hand():
    """``PROJECT_ROOT / plan['script']`` es exactamente el patrón que un ``script``
    absoluto secuestra (pathlib descarta la raíz)."""
    src = DAG_FILE.read_text(encoding="utf-8")
    offenders = re.findall(r"PROJECT_ROOT\s*/\s*plan\[", src)
    assert offenders == [], offenders


def test_dag_does_not_spawn_through_a_shell():
    src = DAG_FILE.read_text(encoding="utf-8")
    assert "shell=True" not in src
