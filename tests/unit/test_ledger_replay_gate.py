"""BL-17 — el gate de replay: mutar el ledger debe romper la reproducción.

BL-17 pide literalmente *"un gate que reconstruya desde cero el `semantic_hash` de un
paper ledger anclado. Mutar una fila del ledger debe romper la reproducción y nombrar
ambos hashes"*. Estos candados exigen las tres propiedades por separado, porque un gate
de integridad falla de tres maneras distintas:

1. **Falso negativo**: no detecta la mutación (el fallo obvio).
2. **Falso positivo**: se pone rojo por operación normal —el ledger es append-only y
   crece cada lunes—, con lo que alguien lo desactiva y vuelve al caso 1.
3. **Diagnóstico inútil**: detecta pero no dice qué esperaba, y nadie puede investigar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.identity.ledger_replay import (  # noqa: E402
    LEDGER_EXCLUDED_FIELDS,
    LEDGER_SEMANTIC_FIELDS,
    LedgerReproductionError,
    anchor_payload,
    assert_anchor_holds,
    ledger_semantic_hash,
)

from tests.support.dag_graph import by_task_id, dag_source, task_runs_before  # noqa: E402

ANCHOR = ROOT / "data" / "anchors" / "paper_ledger_h5.json"


def _fila(semana: int, pnl: float, **extra):
    base = {
        "strategy_id": "smart_simple_v11",
        "signal_date": f"2026-01-{semana:02d}",
        "inference_year": 2026,
        "inference_week": semana,
        "direction": 1,
        "leverage": 1.0,
        "week_pnl_pct": pnl,
        "n_subtrades": 1,
        "cumulative_pnl_pct": pnl,
        "gate_status": None,
        "circuit_breaker": False,
        "running_da_pct": 55.0,
        "running_da_short_pct": 60.0,
        "running_da_long_pct": 45.0,
        "running_sharpe": 1.2,
        "running_max_dd_pct": -3.4,
        "n_weeks": 8,
        "n_long": 5,
        "n_short": 3,
        "long_pct_8w": 62.5,
        "consecutive_losses": 1,
        "notes": None,
    }
    base.update(extra)
    return base


@pytest.fixture
def ledger():
    return [_fila(2, 0.006), _fila(14, -0.0017), _fila(15, -0.0049)]


# ------------------------------------------------------- 1. detecta la mutación


def test_mutating_a_row_breaks_the_reproduction_and_names_both_hashes(ledger) -> None:
    """La propiedad central de BL-17, con las dos mitades que pide la ficha."""
    ancla = anchor_payload(ledger, until_year=2026, until_week=52)

    mutado = [dict(f) for f in ledger]
    mutado[0]["week_pnl_pct"] = 0.99

    with pytest.raises(LedgerReproductionError) as exc:
        assert_anchor_holds(ancla, mutado)

    mensaje = str(exc.value)
    assert ancla["semantic_hash"] in mensaje, "no nombra el hash anclado"
    assert exc.value.obtenido in mensaje, "no nombra el hash reconstruido"
    assert exc.value.obtenido != exc.value.esperado


def test_deleting_a_row_from_the_closed_past_is_detected(ledger) -> None:
    """Borrar una fila también rompe: el conteo entra en el ancla a propósito.

    Sin `n_rows`, borrar una fila y ajustar otra podría recomponer el hash por
    casualidad de la agregación.
    """
    ancla = anchor_payload(ledger, until_year=2026, until_week=52)

    with pytest.raises(LedgerReproductionError, match="filas del pasado cerrado"):
        assert_anchor_holds(ancla, ledger[:-1])


def test_changing_the_semantic_field_set_invalidates_the_anchor(ledger) -> None:
    """Si cambia QUÉ se hashea, el hash viejo deja de ser comparable — y se dice.

    Comparar contra un ancla calculada sobre otro conjunto de campos daría un rojo
    incomprensible; aquí el gate explica que hay que re-anclar de forma auditada.
    """
    ancla = anchor_payload(ledger, until_year=2026, until_week=52)
    ancla = {**ancla, "semantic_fields": ["strategy_id", "week_pnl_pct"]}

    with pytest.raises(LedgerReproductionError, match="re-anclar"):
        assert_anchor_holds(ancla, ledger)


# ---------------------------------------------- 2. NO se pone rojo por lo normal


def test_appending_a_new_week_does_not_break_the_anchor(ledger) -> None:
    """El ledger crece cada lunes; el ancla cubre el PREFIJO cerrado, no el total.

    Es la propiedad que evita que el gate se desactive por gritar cada semana. Sin
    ella, este candado sería técnicamente correcto y operativamente inservible.
    """
    ancla = anchor_payload(ledger, until_year=2026, until_week=15)

    futuro = ledger + [_fila(31, 0.42), _fila(32, -0.10)]
    assert assert_anchor_holds(ancla, futuro) == ancla["semantic_hash"]


def test_row_order_does_not_change_the_hash(ledger) -> None:
    """Dos lecturas de la misma tabla deben dar el mismo hash aunque el motor
    devuelva las filas en otro orden: el orden lo impone el módulo, no el `SELECT`."""
    assert ledger_semantic_hash(ledger) == ledger_semantic_hash(list(reversed(ledger)))


def test_write_metadata_is_excluded_so_a_restore_does_not_look_like_corruption(
    ledger,
) -> None:
    """`id`/`created_at` fuera del hash: restaurar un backup los reasigna.

    Si entraran, una restauración legítima produciría un hash distinto para
    exactamente los mismos trades y el gate gritaría "corrupción" — el falso positivo
    que termina con el candado apagado.
    """
    con_metadatos = [
        {**f, "id": 100 + i, "created_at": "2026-08-04T00:00:00Z"}
        for i, f in enumerate(ledger)
    ]
    assert ledger_semantic_hash(con_metadatos) == ledger_semantic_hash(ledger)
    assert not set(LEDGER_EXCLUDED_FIELDS) & set(LEDGER_SEMANTIC_FIELDS)
    # Sólo el surrogate técnico se excluye. `notes` NO: es evidencia auditada, y sacarla
    # exigiría un contrato explícito que hoy no existe (CXD-449 §1).
    assert set(LEDGER_EXCLUDED_FIELDS) == {"id", "created_at"}
    assert "notes" in LEDGER_SEMANTIC_FIELDS


def test_an_incomplete_row_is_refused_instead_of_hashed(ledger) -> None:
    """Una fila sin campos semánticos NO se hashea con huecos: se rechaza.

    Hashear un `None` implícito produciría un hash válido para un ledger incompleto —
    un "todo bien" que no significa nada.
    """
    incompleta = {k: v for k, v in ledger[0].items() if k != "week_pnl_pct"}
    with pytest.raises(LedgerReproductionError, match="week_pnl_pct"):
        ledger_semantic_hash([incompleta])


# ------------------------------------------------------ 3. el ancla real, viva


def test_the_monitor_dag_verifies_the_anchor_after_writing_the_ledger() -> None:
    """El gate debe correr DESPUÉS del append semanal, no existir suelto.

    Mismo aprendizaje que BL-16 (CXD-442): comprobar que la tarea aparece en el fichero
    no prueba nada — una tarea huérfana pasa ese test. Aquí se resuelve alcanzabilidad
    sobre el grafo `>>`, y el sentido importa: verificar ANTES de escribir dejaría sin
    cubrir justo el momento en que el ledger se toca.
    """
    fuente = dag_source("forecast_h5_l6_weekly_monitor.py")

    assert task_runs_before(
        fuente, by_task_id("paper_ledger_2026"), by_task_id("verify_ledger_anchor")
    ), (
        "verify_ledger_anchor no corre después de paper_ledger_2026: el gate no cubre "
        "el momento en que el ledger se modifica"
    )


def test_the_order_lock_here_also_fails_when_unlinked() -> None:
    """El par fail-first, igual que en BL-16: desenlazar debe poner esto en rojo.

    Se aplica sobre la fuente real, así que la demostración se ejecuta en cada corrida
    en vez de haber ocurrido una sola vez cuando se escribió el candado.

    La mutación se **verifica antes de creerla**: la primera versión de este test hacía
    un `replace` de una cadena que había dejado de existir —Codex insertó su tarea de
    métricas en medio y la cadena pasó de `t_persist >> t_paper_ledger` a
    `t_metric_event >> t_paper_ledger`—, así que no mutaba nada y el candado quedaba
    verde por vacío. Un fail-first que no comprueba haber roto algo no demuestra nada.
    """
    original = dag_source("forecast_h5_l6_weekly_monitor.py")
    huerfano = original.replace(" >> t_verify_anchor", "")

    assert huerfano != original, (
        "la mutación no se aplicó: la cadena cambió de forma y este test estaría "
        "midiendo el DAG intacto"
    )
    assert not task_runs_before(
        huerfano, by_task_id("paper_ledger_2026"), by_task_id("verify_ledger_anchor")
    ), "con la verificación desenlazada el candado sigue verde: mide presencia, no orden"


def test_mutating_a_decision_metric_breaks_the_hash(ledger) -> None:
    """`running_da_pct` es decisoria y DEBE entrar en el hash.

    Este candado nace de un falso verde propio que refutó Codex (CXD-449): la primera
    versión hasheaba once campos "económicos" y omitía las métricas acumuladas, así que
    `running_da_pct 55.0 -> 99.0` daba **hashes idénticos**. Y esas columnas no son
    decoración: `control_system_health` lee `running_sharpe`, y el DA y el drawdown
    gobiernan gates y circuit breaker.

    La lección de fondo va más allá de añadir campos: la frontera correcta no era "lo
    económico" —un juicio mío sobre qué importa— sino **lo persistido**, que se puede
    comprobar contra el esquema en vez de argumentar.
    """
    for metrica, nuevo in (
        ("running_da_pct", 99.0),
        ("running_sharpe", 42.0),
        ("running_max_dd_pct", -80.0),
        ("consecutive_losses", 9),
        ("notes", "editado a mano"),
    ):
        ancla = anchor_payload(ledger, until_year=2026, until_week=52)
        mutado = [dict(f) for f in ledger]
        mutado[0][metrica] = nuevo

        with pytest.raises(LedgerReproductionError) as exc:
            assert_anchor_holds(ancla, mutado)
        assert exc.value.obtenido != exc.value.esperado, (
            f"mutar '{metrica}' no cambió el hash: queda fuera del compromiso"
        )


def test_the_semantic_fields_cover_every_persisted_column() -> None:
    """El conjunto hasheado debe ser TODA columna persistida menos el surrogate.

    Comprobado contra el DDL real, no contra mi memoria: es lo que impide que la tupla
    vuelva a quedarse corta cuando alguien añada una columna al ledger.
    """
    try:
        from scripts.data.ingest_asset_ohlcv import _db_conn

        conn = _db_conn()
    except Exception:  # pragma: no cover - CI sin base de datos
        pytest.skip("sin base de datos: la cobertura de columnas se verifica en entorno con DB")

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'forecast_h5_paper_trading'"
            )
            columnas = {fila[0] for fila in cur.fetchall()}
    finally:
        conn.close()

    assert columnas, "la tabla del ledger no existe: no hay nada que hashear"
    faltantes = columnas - set(LEDGER_SEMANTIC_FIELDS) - set(LEDGER_EXCLUDED_FIELDS)
    assert not faltantes, (
        "columnas persistidas que ni entran en el hash ni se excluyen explícitamente: "
        f"{sorted(faltantes)}. Toda columna debe estar en un lado o en el otro; el "
        "silencio es como el hash se quedó corto la primera vez"
    )


def test_the_committed_anchor_matches_a_hash_recomputed_from_its_own_fields() -> None:
    """El ancla versionada existe, y su hash NO se valida contra sí mismo.

    La versión anterior de este candado era **circular** (CXD-449): comparaba el ancla
    contra la misma constante y por eso daba verde con un hash que ignoraba la mitad del
    ledger. Ahora se exige que declare exactamente los campos que el código hashea hoy,
    que es lo único verificable sin base de datos — y basta para detectar que el ancla
    quedó vieja tras un cambio de contrato.
    """
    assert ANCHOR.is_file(), f"falta el ancla del paper ledger: {ANCHOR}"
    ancla = json.loads(ANCHOR.read_text(encoding="utf-8"))

    assert ancla["table"] == "forecast_h5_paper_trading"
    assert ancla["semantic_hash"].startswith("sha256:")
    assert ancla["n_rows"] > 0
    assert list(ancla["semantic_fields"]) == list(LEDGER_SEMANTIC_FIELDS), (
        "el ancla se calculó sobre otro conjunto de campos que el que el código usa hoy: "
        "hay que re-anclar contra la base y volver a firmar el hash"
    )
