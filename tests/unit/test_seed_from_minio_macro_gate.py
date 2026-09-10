"""Causal gates for the canonical Docker cold-start macro path."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "init-scripts" / "04-seed-from-minio.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("seed_from_minio", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_artifact_hashes_and_parses_the_same_bytes(tmp_path):
    module = _load_module()
    path = tmp_path / "macro.csv"
    payload = b"fecha,value\n2026-01-01,1\n"
    path.write_bytes(payload)

    artifact = module.load_from_local([str(path)])

    assert artifact.frame.to_dict("records") == [{"fecha": "2026-01-01", "value": 1}]
    assert artifact.sha256 == module.hashlib.sha256(payload).hexdigest()
    assert artifact.source_uri == str(path.resolve())


def test_macro_seed_uses_shared_fail_closed_validator(monkeypatch):
    module = _load_module()
    manifest = module.manifiesto_backup_2026_06()
    frame = pd.DataFrame(
        {
            "fecha": pd.to_datetime(["2026-06-28", "2026-06-29"]),
            manifest.columnas_vigiladas[0]: [17.5, 175000.0],
            manifest.columnas_vigiladas[1]: [920.0, 921.0],
        }
    )

    with pytest.raises(Exception, match="empalme|escala|salto"):
        module.validate_and_repair_macro_scale(frame, manifest)


def test_table_probe_propagates_database_errors():
    module = _load_module()

    class BrokenCursor:
        def execute(self, _query):
            raise RuntimeError("database unavailable")

        def close(self):
            pass

    with pytest.raises(RuntimeError, match="database unavailable"):
        module.table_has_data(SimpleNamespace(cursor=lambda: BrokenCursor()), "macro")


def test_insert_rejects_missing_required_database_column():
    module = _load_module()

    class SchemaCursor:
        def execute(self, *_args):
            pass

        def fetchall(self):
            return [("fecha",)]

        def close(self):
            pass

    conn = SimpleNamespace(cursor=lambda: SchemaCursor())
    with pytest.raises(RuntimeError, match="Required columns absent"):
        module.insert_dataframe(
            conn,
            pd.DataFrame({"fecha": ["2026-01-01"], "fx": [1.0]}),
            "macro_indicators_daily",
            required_target_columns={"fecha", "fx"},
        )


def test_required_seed_failure_returns_nonzero_and_never_touches_marker(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "SEED_SOURCES", [{
        "name": "macro", "table": "macro_indicators_daily", "required": True,
        "minio_path": "missing.parquet", "local_paths": [], "legacy_paths": [],
        "min_rows": 1, "date_column": "fecha",
    }])
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(module, "get_db_connection", lambda: conn)
    monkeypatch.setattr(module, "table_has_data", lambda *_: (False, 0))
    monkeypatch.setattr(module, "load_seed_data", lambda *_: None)

    assert module.main() == 1
    assert ".seeding_complete" not in SCRIPT.read_text(encoding="utf-8")


def test_macro_bootstrap_locks_rechecks_then_commits(monkeypatch):
    module = _load_module()
    events = []
    watched = module.manifiesto_backup_2026_06().columnas_vigiladas
    frame = pd.DataFrame({"fecha": ["2026-01-01"], watched[0]: [17.0], watched[1]: [900.0]})

    class Cursor:
        def execute(self, query):
            events.append("lock" if query.startswith("LOCK") else "recheck")

        def fetchone(self):
            return (0,)

        def close(self):
            events.append("cursor_close")

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            events.append("commit")

        def rollback(self):
            events.append("rollback")

        def close(self):
            events.append("connection_close")

    source = {
        "name": "macro", "table": "macro_indicators_daily", "required": True,
        "minio_path": "macro.parquet", "local_paths": [], "legacy_paths": [],
        "min_rows": 1, "date_column": "fecha",
    }
    monkeypatch.setattr(module, "SEED_SOURCES", [source])
    monkeypatch.setattr(module, "get_db_connection", Connection)
    monkeypatch.setattr(module, "table_has_data", lambda *_: (False, 0))
    monkeypatch.setattr(module, "load_seed_data", lambda *_: module.SeedArtifact(frame, "test", "abc"))
    monkeypatch.setattr(module, "validate_and_repair_macro_scale", lambda value, _manifest: (value, {}))
    monkeypatch.setattr(module, "validate_data", lambda *_: True)
    monkeypatch.setattr(
        module,
        "insert_dataframe",
        lambda *_args, **kwargs: events.append(f"insert:{kwargs['commit']}") or 1,
    )

    assert module.main() == 0
    assert events.index("lock") < events.index("recheck") < events.index("insert:False") < events.index("commit")
    assert "rollback" not in events


def test_docker_image_copies_shared_leaf_before_running_canonical_seeder():
    dockerfile = (ROOT / "docker" / "Dockerfile.data-seeder").read_text(encoding="utf-8")
    assert "COPY src/data_quality/macro_scale.py /app/src/data_quality/macro_scale.py" in dockerfile
    assert "python /app/seed_from_minio.py &&" in dockerfile


def test_docker_image_must_not_copy_package_init_or_whole_src():
    """El leaf-copy funciona SOLO porque `src/data_quality/__init__.py` NO esta en la imagen.

    Sin el, Python trata el directorio como namespace package (PEP 420) y el leaf importa.
    En el repo ese `__init__.py` SI existe y hace
    `from src.data_quality.rules import QualityDecision, QualityRuleSet`, y `rules.py` no se
    copia: el dia que alguien anada `COPY src/data_quality/` o `COPY src/` a secas, la imagen
    dejara de arrancar con ModuleNotFoundError **en runtime dentro del contenedor**, que es el
    peor sitio para enterarse.

    Hallazgo de CLAUDE en CLD-681, opcion (b). Es un assert de TEXTO sobre el Dockerfile, no
    una construccion de imagen: convierte un acuerdo tacito en candado, no prueba el build.
    """
    dockerfile = (ROOT / "docker" / "Dockerfile.data-seeder").read_text(encoding="utf-8")
    assert "src/data_quality/__init__.py" not in dockerfile, (
        "copiar el __init__ del paquete rompe el leaf-copy: arrastra src.data_quality.rules"
    )
    for line in dockerfile.splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        assert not re.match(r"COPY\s+(--\S+\s+)*src/?\s", stripped), (
            f"COPY de src/ completo rompe el namespace package: {stripped!r}"
        )


def test_seeding_marker_stays_chained_behind_a_successful_required_seed():
    """`.seeding_complete` es lo que mira el HEALTHCHECK, y su fail-closed vive en un `&&`.

    El script ya no toca el marker (por eso el otro test exige su ausencia en el fuente); quien
    lo crea es el CMD. Lo unico que impide que un seed required fallido marque la imagen como
    sana es el `&&` entre el seeder y el `touch`. Cambiarlo por `;` —o partir el CMD— dejaria
    el healthcheck verde sobre una DB vacia: exactamente el patron "mecanismo que no puede
    fallar por lo que no mira".

    Hallazgo MIO al revisar el WIP heredado; nada lo vigilaba. Tambien es assert de texto.
    """
    dockerfile = (ROOT / "docker" / "Dockerfile.data-seeder").read_text(encoding="utf-8")
    assert re.search(r"python /app/seed_from_minio\.py\s*&&", dockerfile), (
        "el seeder canonico debe encadenar con && lo que venga despues"
    )
    start = dockerfile.index("python /app/seed_from_minio.py")
    end = dockerfile.index("touch /app/.seeding_complete")
    assert start < end, "el marker debe crearse DESPUES del seeder"
    assert ";" not in dockerfile[start:end], (
        "un `;` entre el seeder y el marker desacopla el fallo: el touch correria igual"
    )
