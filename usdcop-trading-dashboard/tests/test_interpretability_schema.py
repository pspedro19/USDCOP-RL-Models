"""C-006/BL-20-UI (CXD-040 #3) — el productor Python valida contra el MISMO schema que la API TS.

El JSON Schema compartido vive en
`usdcop-trading-dashboard/app/api/admin/interpretability/_schema/interp-summary.schema.json`:
  - la API TS lo aplica en runtime (unknown-field STRIP + finitos + size-cap);
  - este test valida ESTRICTO los artefactos publicados por
    `scripts/analysis/generate_interpretability.py` en `<repo>/data/interpretability/**`
    (additionalProperties: false ⇒ el productor no puede emitir campos desconocidos).

JSON Schema no puede expresar finitud numerica: aqui se impone por codigo (math.isfinite),
igual que en `_lib/artifact-schema.ts`.

Run:  pytest usdcop-trading-dashboard/tests/test_interpretability_schema.py -q
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

DASHBOARD = Path(__file__).resolve().parents[1]
REPO = DASHBOARD.parent
SCHEMA_PATH = (
    DASHBOARD / "app" / "api" / "admin" / "interpretability" / "_schema" / "interp-summary.schema.json"
)
ARTIFACT_ROOT = REPO / "data" / "interpretability"
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024  # espejo de MAX_ARTIFACT_BYTES en _lib/artifacts.ts

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
ARTIFACTS = sorted(ARTIFACT_ROOT.glob("*/*/*/*/summary.json"))


def _walk_numbers(node, path="$"):
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        assert math.isfinite(node), f"numero no finito en {path}: {node!r}"
    elif isinstance(node, dict):
        for k, v in node.items():
            _walk_numbers(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk_numbers(v, f"{path}[{i}]")


def test_schema_file_is_valid_draft7():
    jsonschema.Draft7Validator.check_schema(SCHEMA)


def test_artifacts_moved_out_of_public():
    """C-006 #1: los artefactos viven en <repo>/data/interpretability, NO en public/."""
    assert not (DASHBOARD / "public" / "data" / "interpretability").exists(), (
        "public/data/interpretability reaparecio: los estaticos de public/ bypassean el "
        "gate admin:all (middleware solo exige sesion). Mover a <repo>/data/interpretability."
    )
    assert ARTIFACTS, f"sin artefactos bajo {ARTIFACT_ROOT} — ¿se movieron a otro sitio?"


@pytest.mark.parametrize("artifact", ARTIFACTS, ids=lambda p: str(p.relative_to(ARTIFACT_ROOT)))
def test_artifact_validates_against_shared_schema(artifact: Path):
    assert artifact.stat().st_size <= MAX_ARTIFACT_BYTES, "artefacto excede el size-cap de la API"
    doc = json.loads(artifact.read_text(encoding="utf-8"))
    jsonschema.validate(doc, SCHEMA)  # estricto: unknown fields ⇒ ValidationError
    _walk_numbers(doc)
    # coherencia path ↔ payload (surface/asset/model_id/version)
    version_dir, model_dir, asset_dir, surface_dir = (
        artifact.parent,
        artifact.parents[1],
        artifact.parents[2],
        artifact.parents[3],
    )
    assert doc["version"] == version_dir.name
    assert doc["model_id"] == model_dir.name
    assert doc["asset"] == asset_dir.name
    assert doc["surface"] == surface_dir.name


def test_schema_rejects_unknown_fields():
    """Prueba negativa: additionalProperties:false es real (el TS los STRIPea, aqui fallan)."""
    doc = json.loads(ARTIFACTS[0].read_text(encoding="utf-8"))
    doc["___campo_desconocido"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)


def test_schema_rejects_missing_required():
    doc = json.loads(ARTIFACTS[0].read_text(encoding="utf-8"))
    doc.pop("nota")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, SCHEMA)


def test_finiteness_guard_catches_nonfinite():
    with pytest.raises(AssertionError):
        _walk_numbers({"base_value": float("inf")})
