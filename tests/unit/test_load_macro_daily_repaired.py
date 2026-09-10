# -*- coding: utf-8 -*-
"""Candados del cargador de `macro_indicators_daily`.

QUÉ SE PUEDE Y QUÉ NO SE PUEDE PROBAR AQUÍ, dicho antes que nada
----------------------------------------------------------------
El camino **puro** —leer, validar (reparando lo que el manifiesto declare, hoy nada),
hashear, exportar— se ejercita de verdad: se llama al script y se comprueba el resultado.

El camino que **toca la base** se ejercita con una conexión falsa que registra los eventos
en orden. Eso prueba el **comportamiento** —que el LOCK va antes del conteo, que hay
recheck bajo el lock y que el INSERT ocurre después— y no sólo que las líneas existan.

La primera versión de este fichero se conformaba con leer el fuente y lo declaraba como
limitación: «impide que alguien borre esas líneas, no demuestra que Postgres las respete».
Codex enseñó en `test_seed_from_minio_macro_gate.py` que la mitad de esa limitación era
pereza mía: el orden sí se puede probar sin base. Se cierra aquí.

**Lo que sigue sin probarse**, y conviene no perderlo de vista: que PostgreSQL respete la
semántica de `SHARE ROW EXCLUSIVE` es cosa de PostgreSQL; estos tests fijan que el script
la pida en el momento correcto, no que el motor la honre. Para eso hace falta una corrida
real contra una base, y desde este entorno no la hay.
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
# Camino con base: comprobación de COMPORTAMIENTO con una conexión falsa
# ---------------------------------------------------------------------------


class _CursorEspia:
    """Cursor que anota qué SQL recibe, en orden, y responde lo justo."""

    def __init__(self, eventos: list[str], filas_iniciales: int) -> None:
        self._eventos = eventos
        self._filas = filas_iniciales
        self._ultimo = None

    def execute(self, sql, params=None):  # noqa: D401
        texto = " ".join(str(sql).split())
        self._ultimo = texto
        if "LOCK TABLE" in texto:
            self._eventos.append("lock")
        elif "count(*)" in texto:
            self._eventos.append("count")
        elif "information_schema.columns" in texto:
            self._eventos.append("schema")
        elif texto.startswith("INSERT INTO"):
            self._eventos.append("insert")

    def executemany(self, sql, filas):
        self._eventos.append("insert")

    def fetchone(self):
        return (self._filas,)

    def fetchall(self):
        # Todas las columnas que el cargador pueda pedir: el objetivo de este test es el
        # ORDEN, no el mapeo de columnas, que ya se cubre aparte.
        import pandas as pd
        columnas = pd.read_parquet(BACKUP).columns if BACKUP.is_file() else []
        return [(c,) for c in list(columnas) + ["fecha"]]

    def close(self):
        self._eventos.append("close_cursor")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _ConexionFalsa:
    """Conexión mínima que registra el orden de las operaciones."""

    def __init__(self, filas_iniciales: int = 0) -> None:
        self.eventos: list[str] = []
        self._filas = filas_iniciales

    def cursor(self):
        return _CursorEspia(self.eventos, self._filas)

    def commit(self):
        self.eventos.append("commit")

    def rollback(self):
        self.eventos.append("rollback")

    def close(self):
        self.eventos.append("close_conn")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.commit()
        return False


def _cargar_modulo():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_loader_bajo_prueba", SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["_loader_bajo_prueba"] = modulo
    spec.loader.exec_module(modulo)
    return modulo


@pytest.mark.skipif(not BACKUP.is_file(), reason="backup ausente en este checkout")
def test_the_lock_is_taken_before_counting_and_the_insert_comes_after(monkeypatch) -> None:
    """EL orden real: `lock` → `count` → … → `insert` → `commit`.

    Antes esto se comprobaba leyendo el fuente, y lo dije en el docstring: eso impide que
    alguien borre las líneas, no que se ejecuten en el orden correcto. Alguien podía mover
    el `LOCK` después del conteo y los asserts estáticos —que sólo miran posiciones de
    texto— podían seguir contentos si el orden textual no cambiaba.

    Aquí se ejercita el `main()` con una conexión falsa que anota cada SQL. El primer
    evento tiene que ser `lock`, y el `insert` posterior al segundo `count` (el recheck).
    """
    modulo = _cargar_modulo()
    falsa = _ConexionFalsa(filas_iniciales=0)
    monkeypatch.setattr(modulo, "_conexion", lambda: falsa)
    monkeypatch.setattr(sys, "argv", ["load_macro_daily_repaired.py"])

    assert modulo.main() == 0, "la carga simulada debía terminar bien"

    eventos = [e for e in falsa.eventos if e in {"lock", "count", "insert", "commit"}]
    assert eventos[0] == "lock", (
        f"el primer evento contra la base fue {eventos[0]!r}: si se cuenta antes de "
        f"bloquear, la guarda empty-only es una foto"
    )
    assert "insert" in eventos, f"no hubo INSERT: {eventos}"

    # Se cuentan los conteos ANTERIORES al insert, no todos. El script hace un `count`
    # final para reportar el total, así que un `count(*) >= 2` a secas se cumpliría sin
    # recheck — medido: la mutación que borra el recheck dejaba este test en verde.
    antes_del_insert = eventos[: eventos.index("insert")]
    assert antes_del_insert.count("count") >= 2, (
        f"antes del INSERT sólo hubo {antes_del_insert.count('count')} conteo(s) "
        f"({eventos}): falta el recheck bajo el lock, y sin él la guarda mira un estado "
        f"que puede haber cambiado cuando se escribe"
    )
    assert eventos[-1] == "commit", f"la secuencia no termina en commit: {eventos}"


@pytest.mark.skipif(not BACKUP.is_file(), reason="backup ausente en este checkout")
def test_a_table_that_is_not_empty_aborts_without_inserting(monkeypatch) -> None:
    """Con filas ya presentes, aborta y NO inserta.

    El complemento del anterior: sin esto, una implementación que ignorase el conteo
    pasaría el test de orden igual, porque el orden seguiría siendo el mismo.
    """
    modulo = _cargar_modulo()
    falsa = _ConexionFalsa(filas_iniciales=26_326)
    monkeypatch.setattr(modulo, "_conexion", lambda: falsa)
    monkeypatch.setattr(sys, "argv", ["load_macro_daily_repaired.py"])

    assert modulo.main() == 1, "una tabla con filas debe abortar con código 1"
    assert "insert" not in falsa.eventos, (
        f"insertó sobre una tabla no vacía: {falsa.eventos}"
    )


# ---------------------------------------------------------------------------
# Comprobaciones ESTÁTICAS que siguen valiendo (forma, no comportamiento)
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
