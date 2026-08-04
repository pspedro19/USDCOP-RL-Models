"""C026 — el DAG de realtime publica por Fabric y escribe SOLO lo aceptado.

Es el cableado que BL-40 pide en la ruta de tiempo real: `publish_provider_rows` antes
del UPSERT legado, en la misma transacción, y la tabla de mercado recibiendo únicamente
`accepted`. Una barra en cuarentena que aterriza igualmente en `usdcop_m5_ohlcv` anula
la cuarentena entera — se registra el rechazo *y* el dato malo entra.

Los candados atacan los tres modos de fallo que ya cazamos en este ciclo, dos de ellos
en código de Codex y uno en el mío:

1. **presencia en vez de orden** — comprobar que la llamada aparece no prueba que ocurra
   antes del UPSERT (CXD-442, CLD-444);
2. **conteo de llamadas en vez de dataflow** — una rama muerta conserva el nodo `Call`, y
   pasar el frame sin filtrar deja el conteo intacto (CLD-449);
3. **cobertura declarada en una lista** — una lista de símbolos cubiertos se desincroniza
   de la espina en silencio; aquí la cobertura **se mide** contra
   `reference.provider_symbol`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DAG = ROOT / "airflow" / "dags" / "l0_ohlcv_realtime.py"


def _writer_body() -> ast.FunctionDef:
    arbol = ast.parse(DAG.read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.FunctionDef) and nodo.name == "fetch_and_store_symbol":
            return nodo
    pytest.fail("no existe 'fetch_and_store_symbol' en el DAG de realtime")


def _values_source(cuerpo: ast.FunctionDef) -> str | None:
    """Nombre del DataFrame del que se construyen los `values` del INSERT.

    Se ancla a la ASIGNACION `values = [...]`, no a "cualquier `iterrows` del cuerpo":
    construir el payload de Fabric desde `df_filtered` es correcto y necesario, y una
    asercion mas gruesa lo confundiria con el defecto — me paso al escribir este test.
    """
    for nodo in ast.walk(cuerpo):
        if not (isinstance(nodo, ast.Assign) and len(nodo.targets) == 1):
            continue
        destino = nodo.targets[0]
        if not (isinstance(destino, ast.Name) and destino.id == "values"):
            continue
        for interno in ast.walk(nodo.value):
            if (
                isinstance(interno, ast.comprehension)
                and isinstance(interno.iter, ast.Call)
                and isinstance(interno.iter.func, ast.Attribute)
                and interno.iter.func.attr == "iterrows"
                and isinstance(interno.iter.func.value, ast.Name)
            ):
                return interno.iter.func.value.id
    return None


def test_the_legacy_insert_receives_only_the_accepted_frame() -> None:
    """El UPSERT itera sobre `aceptadas`, nunca sobre el frame sin filtrar.

    Es el ataque que dejó verde el writer de Codex (CLD-449): cambiar el argumento sin
    tocar el orden ni el conteo de llamadas. Aquí se fija el **nombre del iterable** que
    alimenta los `values` del INSERT.
    """
    fuente_iterable = _values_source(_writer_body())

    assert fuente_iterable == "aceptadas", (
        f"los `values` del INSERT se construyen desde '{fuente_iterable}' y no desde "
        "'aceptadas': una barra en cuarentena aterrizaría en la tabla de mercado"
    )


def test_fabric_publication_happens_before_the_legacy_insert() -> None:
    """Orden real: publicar y luego escribir. Al revés, el gate llega tarde."""
    cuerpo = _writer_body()

    publicacion = [
        nodo.lineno
        for nodo in ast.walk(cuerpo)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "_publish_symbol_rows"
    ]
    inserts = [
        nodo.lineno
        for nodo in ast.walk(cuerpo)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == "execute_values"
    ]

    assert publicacion and inserts, "faltan la publicación Fabric o el INSERT legado"
    assert max(publicacion) < min(inserts), (
        "la publicación Fabric no precede al INSERT: el gate correría después de "
        "escribir, que es no correr"
    )


def test_coverage_is_measured_against_the_spine_not_hardcoded() -> None:
    """No hay lista de símbolos cubiertos: se resuelve el alias contra la espina.

    Una lista se desincroniza en silencio — un instrumento entraría en la espina y
    seguiría sin gate, o saldría y el gate lo bloquearía sin motivo. Resolviendo el
    alias, la cobertura sigue al catálogo sola.
    """
    import inspect

    from src.data_quality import ingest_guard

    # La cobertura la decide el helper COMPARTIDO, no cada DAG por su cuenta. Este
    # candado miraba antes el fichero del DAG y se puso rojo al extraer el helper —
    # correcto: estaba fijando la capa equivocada.
    decisor = inspect.getsource(ingest_guard.publish_or_declare_gap)
    assert "registry_from_spine" in decisor and "resolve(" in decisor, (
        "la cobertura no resuelve el alias contra la espina: quedaría fijada en código"
    )

    for fuente in (DAG.read_text(encoding="utf-8"), inspect.getsource(ingest_guard)):
        for sospechoso in ("COVERED_SYMBOLS", "FABRIC_SYMBOLS", "SYMBOLS_WITH_GATE"):
            assert sospechoso not in fuente, (
                f"'{sospechoso}' parece una lista de cobertura fija: mídela contra "
                "reference.provider_symbol en su lugar"
            )


def test_both_dags_share_one_definition_of_coverage() -> None:
    """Los dos DAGs usan el MISMO helper: dos copias divergirían en silencio.

    Es el defecto que este ciclo persiguió en varias formas —helpers duplicados,
    criterios paralelos— y no tenía sentido reintroducirlo aquí.
    """
    backfill = (ROOT / "airflow" / "dags" / "l0_ohlcv_backfill.py").read_text(
        encoding="utf-8"
    )
    realtime = DAG.read_text(encoding="utf-8")

    for nombre, fuente in (("realtime", realtime), ("backfill", backfill)):
        assert "publish_or_declare_gap" in fuente, (
            f"el DAG de {nombre} no usa el helper compartido de cobertura"
        )


def test_an_uncovered_symbol_is_reported_and_never_silently_screened() -> None:
    """Sin alias canónico se AVISA y se escribe sin filtrar — no se finge cobertura.

    Es la decisión incómoda de este cableado y por eso está fijada: hoy `USD/BRL` no
    tiene `AssetProfile`. Filtrar sus barras las mandaría todas a cuarentena y apagaría
    su ingesta; escribirlas calladamente haría creer que pasaron un gate que nunca
    corrió, y la ausencia de eventos parecería "todo limpio". El aviso es lo que
    distingue un hueco declarado de un bypass.
    """
    cuerpo = _writer_body()
    fuente = ast.unparse(cuerpo)

    assert "fuera de la cobertura Fabric" in fuente, (
        "el símbolo sin identidad canónica pasa sin dejar rastro: un hueco silencioso"
    )
    assert "publicacion is None" in fuente, (
        "no se distingue 'sin cobertura' de 'publicado': son casos distintos"
    )


def test_the_order_lock_fails_when_publication_is_moved_after_the_insert() -> None:
    """Fail-first versionado: mover la publicación después del INSERT debe romper.

    Sin esta demostración el candado de orden nunca se prueba capaz de fallar. La
    mutación se **verifica antes de creerla**, porque ya me pasó que un `replace`
    dejara de casar tras un cambio ajeno y el candado quedara verde por vacío.
    """
    original = DAG.read_text(encoding="utf-8")

    # Se añade una publicación DESPUÉS del INSERT y se retira la de antes: el conteo de
    # llamadas queda idéntico (1), sólo cambia el orden. Es la mutación que un candado
    # basado en `count(...) == 1` no vería.
    mutado = original.replace(
        "            publicacion = _publish_symbol_rows(\n"
        "                conn, symbol=symbol, provider_id='twelvedata_multi', rows=filas_fabric\n"
        "            )",
        "            publicacion = None",
        1,
    ).replace(
        "            conn.commit()\n            cur.close()",
        "            publicacion = _publish_symbol_rows(\n"
        "                conn, symbol=symbol, provider_id='twelvedata_multi', rows=filas_fabric\n"
        "            )\n            conn.commit()\n            cur.close()",
        1,
    )
    assert mutado != original, "la mutación de prueba no se aplicó"
    assert mutado.count("_publish_symbol_rows(\n") == original.count(
        "_publish_symbol_rows(\n"
    ), "la mutación cambió el CONTEO de llamadas: entonces no prueba el orden"

    cuerpo = next(
        n for n in ast.walk(ast.parse(mutado))
        if isinstance(n, ast.FunctionDef) and n.name == "fetch_and_store_symbol"
    )
    publicacion = [
        n.lineno for n in ast.walk(cuerpo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_publish_symbol_rows"
    ]
    inserts = [
        n.lineno for n in ast.walk(cuerpo)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "execute_values"
    ]

    assert not (max(publicacion) < min(inserts)), (
        "con la publicación movida después del INSERT el candado de orden sigue verde: "
        "estaría midiendo presencia, no precedencia"
    )


def test_publication_uses_the_declared_provider_not_the_job_name() -> None:
    """Se publica bajo el PROVEEDOR declarado, no bajo el nombre del job que escribe.

    Defecto medido al cablear el backfill, y era un apagón: las reglas escalonadas se
    declaran para el proveedor (`twelvedata`), mientras los DAGs escribían como
    `twelvedata_multi` / `twelvedata_backfill`. El alias resuelve en los tres casos, así
    que la barra superaba la identidad y moría después en `bar.range_scope`:

        USD/MXN por 'twelvedata'          -> accepted
        USD/MXN por 'twelvedata_multi'    -> bar.range_scope
        USD/MXN por 'twelvedata_backfill' -> bar.range_scope

    El 100% de las barras USD/MXN habría acabado en cuarentena — control de calidad que
    en realidad apaga una ingesta. La separación ya estaba en el modelo: sólo los
    proveedores **declarados** tienen `authoritative_for`; los jobs quedaron como
    `observed_writer` sin autoridad. El job no se pierde: viaja en `source_uri`.
    """
    from src.data_quality.ingest_guard import declared_provider_for

    assert declared_provider_for("USD/MXN") == "twelvedata"
    assert declared_provider_for("USD/COP") == "twelvedata"
    assert declared_provider_for("BTC/USDT") == "binance"
    assert declared_provider_for("USD/BRL") is None, (
        "USD/BRL no tiene activo declarado: debe quedar fuera de cobertura, no recibir "
        "un proveedor inventado"
    )

    import inspect

    from src.data_quality import ingest_guard

    decisor = inspect.getsource(ingest_guard.publish_or_declare_gap)
    assert "declared_provider_for" in decisor, (
        "la publicación no resuelve el proveedor declarado: volvería a publicar bajo el "
        "nombre del job y las reglas escalonadas no casarían nunca"
    )
