# -*- coding: utf-8 -*-
"""Candados del cargador de `macro_indicators_daily`.

QUÉ SE PUEDE Y QUÉ NO SE PUEDE PROBAR AQUÍ, dicho antes que nada
----------------------------------------------------------------
El camino **puro** —leer, reparar, hashear, exportar— se ejercita de verdad: se llama al
script y se comprueba el resultado.

El camino que **toca la base** no. Desde este entorno Postgres no es alcanzable (los tests
que lo intentan reportan `postgres unreachable`), así que sus tres garantías —LOCK antes
del conteo, recheck bajo el lock y filtro por `table_schema`— se fijan **leyendo el
código**, no ejecutándolo.

Eso es una comprobación de forma y no de comportamiento, y conviene no confundirlas: estos
tests impiden que alguien **borre** esas líneas, no demuestran que Postgres las respete.
La prueba real de esas tres es una corrida contra una base, y hoy no la tenemos.

Se dice aquí porque un fichero de tests que no declara su alcance acaba leyéndose como una
garantía mayor de la que da — el defecto que este repo lleva un día entero corrigiendo.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/ops/load_macro_daily_repaired.py"
BACKUP = REPO / "data/backups/seeds/macro_indicators_daily_backup.parquet"
FUENTE = SCRIPT.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Camino puro: se ejecuta de verdad
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BACKUP.is_file(), reason="backup ausente en este checkout")
def test_export_is_reproducible_and_its_hash_matches_what_is_reported(tmp_path) -> None:
    """El hash que el script publica es el del CSV que realmente escribe.

    Sin esto la provenance sería decorativa: diría un sha y el fichero podría ser otro.
    Se comprueba además que dos corridas den byte-idéntico — una exportación que cambia
    sola no sirve para verificar nada aguas abajo.
    """
    destinos = []
    hashes_reportados = []
    for i in range(2):
        destino = tmp_path / f"salida_{i}.csv"
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--export-csv", str(destino)],
            cwd=str(REPO), capture_output=True, text=True, timeout=300,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        m = re.search(r"\[reparado\] sha256=([0-9a-f]{16})", proc.stdout)
        assert m, f"el script no publicó el hash del frame reparado:\n{proc.stdout}"
        destinos.append(destino)
        hashes_reportados.append(m.group(1))

    reales = [hashlib.sha256(d.read_bytes()).hexdigest()[:16] for d in destinos]
    assert reales[0] == reales[1], "dos exportaciones del mismo origen difieren"
    assert hashes_reportados[0] == reales[0], (
        f"el script reportó {hashes_reportados[0]} y el fichero es {reales[0]}: la "
        f"provenance no identifica lo que se carga"
    )


@pytest.mark.skipif(not BACKUP.is_file(), reason="backup ausente en este checkout")
def test_the_export_path_never_touches_the_database(tmp_path) -> None:
    """`--export-csv` no abre conexión: es la vía que evita manejar credenciales.

    Se comprueba sin base disponible a propósito. Si el script intentara conectar, aquí
    fallaría — y ese fallo es justamente la señal que se busca.
    """
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--export-csv", str(tmp_path / "x.csv")],
        cwd=str(REPO), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "la base NO se ha tocado" in proc.stdout


def test_a_missing_source_fails_instead_of_loading_nothing() -> None:
    """Sin origen, el script para. No «carga cero filas» con éxito.

    Es la forma de vacuidad que este repo se ha encontrado hoy dos veces: un proceso que
    no hace nada y devuelve 0.
    """
    assert 'if not ORIGEN.is_file():' in FUENTE
    assert 'return 1' in FUENTE.split('if not ORIGEN.is_file():')[1][:200]


# ---------------------------------------------------------------------------
# Camino con base: comprobación ESTÁTICA. Ver el docstring del módulo.
# ---------------------------------------------------------------------------


def test_the_source_is_read_once_for_both_hash_and_parse() -> None:
    """Un solo `read_bytes`, y el parseo va sobre esos mismos bytes.

    Hashear el fichero y luego volver a leerlo con `read_parquet(ruta)` deja una ventana
    en la que el fichero cambia y publicaríamos la provenance de algo que no cargamos.
    Mismo TOCTOU que se cerró en el lock de aprobaciones.
    """
    assert FUENTE.count("ORIGEN.read_bytes()") == 1, (
        "el origen se lee más de una vez: hash y parseo pueden divergir"
    )
    assert "pd.read_parquet(io.BytesIO(crudo))" in FUENTE, (
        "el parseo no usa los bytes ya leídos y hasheados"
    )


def test_the_empty_guard_is_taken_under_a_lock() -> None:
    """El LOCK va ANTES del conteo, y hay recheck antes de escribir.

    Sin lock, un backfill concurrente puede poblar la tabla entre el `count` y el
    `INSERT`: la guarda habría mirado un estado que ya no existe.
    """
    lock = FUENTE.find("LOCK TABLE")
    count = FUENTE.find("SELECT count(*) FROM {TABLA}")
    assert lock != -1, "no se toma LOCK sobre la tabla"
    assert lock < count, "el conteo ocurre antes del LOCK: la guarda es una foto"
    assert FUENTE.count("SELECT count(*) FROM {TABLA}") >= 3, (
        "falta el recheck bajo el lock justo antes del INSERT"
    )


def test_the_schema_lookup_is_qualified_and_required_columns_are_enforced() -> None:
    """La consulta de columnas filtra por esquema, y faltar columnas es error.

    `table_name` a secas casaría con una tabla homónima de otro esquema. Y descartar en
    silencio `fecha` o las columnas FX vigiladas dejaría datos que ningún gate revisa.
    """
    assert "table_schema = current_schema()" in FUENTE
    assert "obligatorias" in FUENTE and "faltan" in FUENTE
    assert "ON CONFLICT (fecha) DO NOTHING" in FUENTE


def test_the_destructive_scripts_are_named_as_forbidden() -> None:
    """El docstring dice por qué no se reusa `seed_database.py`.

    Ese script hace `DELETE ... WHERE TRUE` sobre OHLCV. La razón vive en el fichero para
    que el siguiente que busque «un cargador» no lo encuentre antes que la advertencia.
    """
    assert "seed_database.py" in FUENTE and "DELETE" in FUENTE
    assert "restore_master.py" in FUENTE
