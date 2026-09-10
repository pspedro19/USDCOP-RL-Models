"""
Regression: un valor macro fuera de su rango declarado no llega a la BD.

Contract: CTR-L0-QUARANTINE-001 · Date: 2026-08-25

## El incidente que este fichero existe para que no se repita

Entre **2025-09-25 y 2025-12-19** entraron **59 días de Brent a 21-23 USD** (real: 60-70) en
`macro_indicators_daily` y en el parquet macro, dentro del hold-out de la tesis.

`config/l0_macro_sources.yaml:509` ya declaraba `comm_oil_brent_glb_d_brent: [30, 150]` con
`validation: {enabled: true}`. El rango estaba bien escrito. El dato entró igual.

## Por qué, y por qué el primer diagnóstico fue incorrecto

La hipótesis inicial —*"`RangeValidator` solo se instancia en tests"*— **era falsa**:
`data_validators.py` lo pone en la lista por defecto de `ValidationPipeline` y
`l0_macro_backfill.py` construye esa pipeline.

Los huecos reales eran cuatro, y todos tienen la misma forma: **la validación vivía en un
sitio y la escritura en otro**.

| | Hueco |
|---|---|
| A | `l0_macro_update` (ingesta DIARIA) no tenía tarea de validación — por ahí entró |
| B | `validate_data` del backfill nunca lanzaba (`fail_fast=False` + `logger.warning`) |
| C | La rama `restore_from_seeds` se saltaba la validación entera |
| D | Sin el YAML, el validador cargaba `{}` reglas **en silencio** |

Por eso el arreglo **no** fue añadir una quinta tarea al grafo —una tarea se puede limpiar,
saltar, o quedar fuera de una rama nueva— sino filtrar en el punto donde las filas se
escriben, que es por donde pasan todas esas rutas.

Estos tests comprueban el **comportamiento** en ese punto, no la existencia del validador.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "airflow" / "dags"
if str(DAGS) not in sys.path:
    sys.path.insert(0, str(DAGS))


def _load(module_name: str, relative: str):
    """Carga por RUTA, no por nombre de paquete.

    `tests/conftest.py` mete `<repo>/services` en `sys.path`, asi que un
    `import services.range_quarantine` resuelve al paquete `services/` de la raiz del repo
    —que no tiene ese modulo— en vez de al de `airflow/dags/`. Es la misma colision de
    rutas que ya rompe seis imports de `tests/unit`. Importar por ruta la esquiva sin
    depender del orden de `sys.path`.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, DAGS / relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_rq = _load("_dags_range_quarantine", "services/range_quarantine.py")
QuarantineBlocked = _rq.QuarantineBlocked
filter_frame = _rq.filter_frame
filter_out_of_range = _rq.filter_out_of_range
reset_config_cache = _rq.reset_config_cache

BRENT = "comm_oil_brent_glb_d_brent"
WTI = "comm_oil_wti_glb_d_wti"


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    """El registro de cuarentena va a un tmp: un test no ensucia el ledger real."""
    monkeypatch.setenv("MACRO_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    reset_config_cache()
    yield
    reset_config_cache()


def brent_incident_frame() -> pd.DataFrame:
    """Las cuatro filas de la frontera real del incidente, con sus valores reales."""
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2025-09-24", "2025-09-25", "2025-09-26", "2025-12-22"]),
        BRENT: [69.31, 23.50, 23.00, 61.58],
    })


# ---------------------------------------------------------------------------
# El replay del incidente
# ---------------------------------------------------------------------------

def test_the_brent_incident_no_longer_gets_through():
    """Las filas de 23,50 y 23,00 NO se escriben; las de 69,31 y 61,58 sí."""
    clean, res = filter_out_of_range(BRENT, brent_incident_frame(), date_col="fecha")

    assert res.rows_quarantined == 2, f"esperaba apartar 2 filas, apartó {res.rows_quarantined}"
    assert res.rows_kept == 2
    assert sorted(clean[BRENT].tolist()) == [61.58, 69.31]
    assert 23.50 not in clean[BRENT].tolist()
    assert res.expected_range == (30.0, 150.0)


def test_the_good_rows_survive_untouched():
    """Apartar lo malo no puede alterar lo bueno: mismos valores, mismas fechas."""
    df = brent_incident_frame()
    clean, _ = filter_out_of_range(BRENT, df, date_col="fecha")
    kept = df[df[BRENT] > 30]
    assert clean[BRENT].tolist() == kept[BRENT].tolist()
    assert clean["fecha"].tolist() == kept["fecha"].tolist()


def test_quarantined_rows_are_recorded_with_their_reason():
    """Un dato apartado sin registro es un dato perdido. Tiene que quedar el motivo."""
    import json

    _, res = filter_out_of_range(BRENT, brent_incident_frame(), date_col="fecha")
    assert res.ledger_path, "no se escribió registro de cuarentena"

    lines = [json.loads(x) for x in Path(res.ledger_path).read_text(
        encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2
    for entry in lines:
        assert entry["variable"] == BRENT
        assert entry["expected_range"] == [30.0, 150.0]
        assert entry["value"] < 30.0
        assert entry["quarantined_at"]
        assert entry["date"]


def test_a_clean_frame_is_returned_untouched_and_writes_no_ledger():
    df = pd.DataFrame({"fecha": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                       BRENT: [78.0, 79.5]})
    clean, res = filter_out_of_range(BRENT, df, date_col="fecha")
    assert res.clean and res.rows_quarantined == 0
    assert res.ledger_path is None
    assert clean.equals(df)


# ---------------------------------------------------------------------------
# Las reglas de convivencia: no romper la ingesta por un dato malo
# ---------------------------------------------------------------------------

def test_one_bad_variable_does_not_drop_the_others():
    """El motivo de elegir cuarentena y no `error`.

    Con `on_invalid: error` una variable mala entre 40 tumbaría el DAG, y macro con más de
    7 días de retraso **bloquea el training** (`data-freshness.md`). Cambiar un dato malo por
    ningún dato no es una mejora.
    """
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2025-09-25", "2025-09-26"]),
        BRENT: [23.50, 23.00],     # corruptos
        WTI: [64.98, 65.72],       # sanos
    })
    clean, results = filter_frame(df, [BRENT, WTI], date_col="fecha")

    assert clean.empty, "las dos filas tenían Brent malo: no queda ninguna que escribir"
    quarantined = {r.variable for r in results if r.rows_quarantined}
    assert quarantined == {BRENT}, f"WTI no debía apartarse: {quarantined}"


def test_rows_are_evaluated_independently_not_all_or_nothing():
    """Una fila mala no puede llevarse por delante a las buenas del mismo lote."""
    df = pd.DataFrame({
        "fecha": pd.to_datetime(["2025-09-24", "2025-09-25", "2025-09-29"]),
        BRENT: [69.31, 23.50, 69.00],
    })
    clean, res = filter_out_of_range(BRENT, df, date_col="fecha")
    assert res.rows_quarantined == 1 and len(clean) == 2


def test_missing_values_are_not_quarantined():
    """NaN es AUSENCIA, no un valor fuera de rango. Lo gestiona el gate de frescura.

    Apartar NaN además sería contraproducente: dejaría de escribirse la fila entera y con
    ella las variables sanas que la acompañan.
    """
    df = pd.DataFrame({"fecha": pd.to_datetime(["2026-01-05", "2026-01-06"]),
                       BRENT: [None, 78.0]})
    clean, res = filter_out_of_range(BRENT, df, date_col="fecha")
    assert res.rows_quarantined == 0
    assert len(clean) == 2


def test_a_variable_without_a_declared_range_passes_through():
    """No tener regla no es motivo para apartar datos."""
    df = pd.DataFrame({"fecha": pd.to_datetime(["2026-01-05"]),
                       "una_variable_sin_rango_declarado": [999999.0]})
    clean, res = filter_out_of_range("una_variable_sin_rango_declarado", df, date_col="fecha")
    assert res.rows_quarantined == 0 and len(clean) == 1


# ---------------------------------------------------------------------------
# La política es la del YAML, y se consume de verdad
# ---------------------------------------------------------------------------

def test_the_declared_policy_is_quarantine_not_warn():
    """`warn` fue lo que permitió los 59 días. El YAML tiene que decir `quarantine`.

    Si alguien lo revierte, que sea una decisión visible en un diff, no un descubrimiento
    tras el siguiente incidente.
    """
    import yaml

    cfg = yaml.safe_load((ROOT / "config" / "l0_macro_sources.yaml").read_text(
        encoding="utf-8"))
    validation = cfg["validation"]
    assert validation["enabled"] is True
    assert validation["on_invalid"] == "quarantine", (
        f"on_invalid={validation['on_invalid']!r}: con `warn` los datos fuera de rango se "
        "escriben igual, que es exactamente lo que pasó con Brent"
    )


def test_warn_policy_still_writes_everything():
    """Contraprueba: si `warn` filtrara, el test anterior no probaría nada."""
    clean, res = filter_out_of_range(BRENT, brent_incident_frame(), date_col="fecha",
                                     policy="warn")
    assert res.rows_quarantined == 0 and len(clean) == 4


def test_error_policy_raises_instead_of_writing():
    with pytest.raises(QuarantineBlocked, match="fuera de"):
        filter_out_of_range(BRENT, brent_incident_frame(), date_col="fecha", policy="error")


# ---------------------------------------------------------------------------
# Hueco D: un validador sin reglas tiene que ser RUIDOSO
# ---------------------------------------------------------------------------

def test_range_validator_raises_when_it_cannot_find_its_config():
    """Antes devolvía `{}` con un `logger.warning` y seguía vivo validando CERO variables.

    Ese modo de fallo es el peor de todos: los informes salen en verde sobre datos que nadie
    miró.
    """
    dv = _load("_dags_data_validators", "validators/data_validators.py")
    RangeValidator, ValidationConfigError = dv.RangeValidator, dv.ValidationConfigError

    real_exists = Path.exists
    try:
        Path.exists = lambda self: False        # noqa: E731 - se restaura en el finally
        with pytest.raises(ValidationConfigError, match="No se encontro"):
            RangeValidator()
    finally:
        Path.exists = real_exists


def test_explicit_ranges_bypass_the_config_entirely():
    """Pasar `ranges=` tiene que seguir funcionando sin tocar disco."""
    RangeValidator = _load("_dv2", "validators/data_validators.py").RangeValidator

    v = RangeValidator(ranges={"x": (0.0, 1.0)})
    assert v._ranges == {"x": (0.0, 1.0)}


def test_the_real_config_loads_the_brent_range():
    """Comprobación de extremo a extremo de que las reglas llegan cargadas."""
    RangeValidator = _load("_dv3", "validators/data_validators.py").RangeValidator

    assert RangeValidator()._ranges[BRENT] == (30.0, 150.0)


# ---------------------------------------------------------------------------
# Cobertura: la cuarentena está en el punto de ESCRITURA, no en una tarea
# ---------------------------------------------------------------------------

def test_both_write_funnels_apply_the_quarantine():
    """La lección del incidente: validar donde se escribe, no en un nodo del grafo.

    Se comprueba por AST que los dos embudos de `upsert_service.py` llaman al filtro. Si
    alguien añade una ruta de escritura nueva que no pase por ellos, este test no la caza —
    pero sí caza que alguien quite el filtro de los que ya existen.
    """
    import ast

    src = (DAGS / "services" / "upsert_service.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    funnels = {"upsert_variable": False, "_execute_upsert": False}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in funnels:
            body = ast.dump(node)
            funnels[node.name] = ("_quarantine_filter" in body or "filter_frame" in body)

    missing = [k for k, v in funnels.items() if not v]
    assert not missing, (
        f"estos puntos de escritura ya no filtran fuera de rango: {missing}. "
        "El incidente de Brent ocurrió exactamente porque la validación no estaba en el "
        "camino de la escritura."
    )


# ---------------------------------------------------------------------------
# El hueco MAS profundo: el umbral del 10%
# ---------------------------------------------------------------------------

def test_a_single_impossible_value_is_an_error_not_a_warning():
    """Antes, una violación de rango solo era error si superaba el **10%** de los valores.

    Ese umbral es la razón más profunda de que este guard no protegiera nada: los 59 días de
    Brent corrupto nunca llegaron al 10% de un lote de ingesta, así que un `RangeValidator`
    perfectamente cableado **y bloqueante** los habría dejado pasar igual como aviso.

    Un porcentaje sirve para «esta serie está derivando». No sirve para «este valor es
    posible»: un Brent a 23 USD es imposible haya o no otros 999 valores correctos al lado.
    """
    dv = _load("_dv_threshold", "validators/data_validators.py")

    values = [80.0] * 1000
    values[500] = 23.50                      # 1 sola fila mala = 0,1%
    df = pd.DataFrame({"fecha": pd.date_range("2024-01-01", periods=1000, freq="D"),
                       BRENT: values})

    result = dv.RangeValidator(ranges={BRENT: (30.0, 150.0)}).validate(df, BRENT)

    assert result.passed is False, (
        "una sola fila imposible entre 1.000 tiene que fallar la validación; con el umbral "
        "del 10% pasaba como aviso — y así entraron los 59 días de Brent"
    )
    assert result.severity == dv.ValidationSeverity.CRITICAL
    assert result.metadata["out_of_range_count"] == 1


def test_explicit_empty_ranges_means_no_ranges_not_load_from_config():
    """`RangeValidator(ranges={})` cargaba las 24 reglas del YAML por el `or`.

    El llamante pedía un validador sin reglas y recibía uno con todas. Misma familia de
    fallo que el resto del incidente: el código hacía algo distinto de lo pedido, callado.
    """
    dv = _load("_dv_empty", "validators/data_validators.py")

    assert dv.RangeValidator(ranges={})._ranges == {}
    assert dv.RangeValidator()._ranges, "sin argumento sí debe cargar del config"


def test_the_ledger_directory_resolves_under_airflow_home_in_the_container(monkeypatch,
                                                                          tmp_path):
    """En el contenedor, `parents[3]` da `/opt`, no la raíz del repo.

    El síntoma real fue `Permission denied: '/opt/data'`: la cuarentena filtraba bien pero
    **se quedaba sin constancia**, y una fila apartada sin registro es una fila perdida —
    justo lo que otro test de este fichero prohíbe.

    `AIRFLOW_HOME` desambigua las dos disposiciones de directorio.
    """
    monkeypatch.delenv("MACRO_QUARANTINE_DIR", raising=False)

    fake_home = tmp_path / "opt" / "airflow"
    fake_home.mkdir(parents=True)
    monkeypatch.setenv("AIRFLOW_HOME", str(fake_home))
    assert _rq._quarantine_dir() == fake_home / "data" / "quarantine" / "macro"

    # Sin AIRFLOW_HOME se cae a la raíz del repo, que es la disposición del host.
    monkeypatch.delenv("AIRFLOW_HOME", raising=False)
    assert _rq._quarantine_dir() == ROOT / "data" / "quarantine" / "macro"
