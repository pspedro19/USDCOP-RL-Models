# -*- coding: utf-8 -*-
"""La reparación de escala repara lo declarado y **falla** ante todo lo demás.

Un transformador que "limpia datos sucios" es peligroso por defecto: en cuanto acepta
decidir solo, se convierte en una licencia para dividir números por 10.000 sin que nadie
mire. Por eso lo que se fija aquí no es que repare —eso es lo fácil— sino **cada una de sus
formas de negarse**:

  * un empalme que nadie declaró             → error
  * una declaración que ya no hace falta     → error
  * una reparación que no arregla la serie   → error
  * un valor no finito, <= 0 o no numérico   → error
  * la misma (columna, fecha) declarada dos veces → error
  * un factor <=0, ==1 o no finito, o una celda sin evidencia → error
  * una fecha duplicada en el frame          → error

Los tres últimos grupos llegaron en R2, tras la revisión CXD-771. El caso del manifiesto
obsoleto es el que suele faltar: sin él, la lista de excepciones sobrevive a su motivo y
nadie se entera de que dejó de ser necesaria.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from data_quality.macro_scale import (  # noqa: E402
    CeldaDeclarada,
    MacroScaleError,
    ManifiestoEscala,
    manifiesto_backup_2026_06,
    validate_and_repair_macro_scale,
)

COL = "fxrt_spot_usdmxn_mex_d_usdmxn"
BACKUP = REPO / "data/backups/seeds/macro_indicators_daily_backup.parquet"


def _serie_limpia(n: int = 20) -> pd.DataFrame:
    fechas = pd.date_range("2026-06-01", periods=n, freq="D")
    valores = 17.5 + np.arange(n) * 0.01
    return pd.DataFrame({"fecha": fechas, COL: valores})


def _con_empalme(desde: int, hasta: int, factor: float = 1e4) -> pd.DataFrame:
    df = _serie_limpia()
    df.loc[desde:hasta, COL] = df.loc[desde:hasta, COL] * factor
    return df


def _manifiesto(fechas: list[str]) -> ManifiestoEscala:
    return ManifiestoEscala(
        celdas=tuple(
            CeldaDeclarada(COL, date.fromisoformat(f), 1e4, "prueba") for f in fechas
        ),
        columnas_vigiladas=(COL,),
    )


def test_repairs_exactly_the_declared_cells() -> None:
    """El camino feliz: repara el tramo declarado y deja el resto intacto."""
    df = _con_empalme(5, 8)
    fechas = ["2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09"]
    out, reporte = validate_and_repair_macro_scale(df, _manifiesto(fechas))

    assert reporte["n_celdas_reparadas"] == 4
    reparado = out.sort_values("fecha")[COL].to_numpy()
    original = _serie_limpia()[COL].to_numpy()
    assert np.allclose(reparado, original), (
        "tras reparar, la serie debe coincidir con la sana: si no, el factor no explica "
        "el daño"
    )


def test_an_undeclared_splice_is_refused() -> None:
    """Lo que nadie declaró NO se toca, y se para.

    Es la aserción central. Sin ella el transformador «arreglaría» cualquier salto que
    encuentre — incluido uno que fuese un evento real de mercado.
    """
    df = _con_empalme(5, 8)
    with pytest.raises(MacroScaleError, match="NO declarado"):
        validate_and_repair_macro_scale(df, _manifiesto(["2026-06-06"]))


def test_a_stale_declaration_is_refused() -> None:
    """Declarar una reparación que ya no hace falta también es error.

    Si la fuente se arregla y el manifiesto se queda, esa entrada pasa a autorizar una
    división sobre un valor sano. Vencer una excepción tiene que doler igual que crearla.
    """
    limpia = _serie_limpia()
    with pytest.raises(MacroScaleError, match="ya NO presentan empalme"):
        validate_and_repair_macro_scale(limpia, _manifiesto(["2026-06-06"]))


def test_a_wrong_factor_is_refused_by_the_post_check() -> None:
    """Si el factor declarado no explica el daño, no se publica media reparación.

    Aquí el empalme es ×10⁴ y se declara ×10²: la división deja la serie todavía rota, y
    el post-check lo ve. Sin ese paso final, el frame saldría «reparado» con un salto de
    100× dentro.
    """
    df = _con_empalme(5, 8)
    fechas = ["2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09"]
    mal = ManifiestoEscala(
        celdas=tuple(
            CeldaDeclarada(COL, date.fromisoformat(f), 1e2, "factor equivocado")
            for f in fechas
        ),
        columnas_vigiladas=(COL,),
    )
    with pytest.raises(MacroScaleError, match="SIGUEN quedando empalmes"):
        validate_and_repair_macro_scale(df, mal)


def test_a_watched_column_that_vanished_is_refused() -> None:
    """Vigilar una columna que ya no existe es vigilar nada; se dice en voz alta."""
    df = _serie_limpia().rename(columns={COL: "otra_cosa"})
    with pytest.raises(MacroScaleError, match="no está en el frame"):
        validate_and_repair_macro_scale(df, ManifiestoEscala((), (COL,)))


def test_the_function_does_not_mutate_its_input() -> None:
    """Pura: el frame que entra no se toca.

    Importa porque el llamador es un loader de ingesta; si mutáramos in-place, un fallo a
    mitad dejaría el frame original a medio reparar y nadie sabría en qué estado quedó.
    """
    df = _con_empalme(5, 8)
    antes = df[COL].copy()
    validate_and_repair_macro_scale(
        df, _manifiesto(["2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09"])
    )
    pd.testing.assert_series_equal(df[COL], antes)


@pytest.mark.skipif(not BACKUP.is_file(), reason="backup ausente en este checkout")
def test_the_real_backup_needs_no_repair_and_the_manifest_says_so() -> None:
    """El backup vigente está LIMPIO, y el manifiesto no declara reparaciones.

    Este test exigía lo contrario: que las 15 celdas declaradas fueran las que de verdad
    estaban rotas. Cambió por un hecho, no por conveniencia — el 2026-08-06
    `core_l0_05_seed_backup` regeneró el backup desde la base ya reparada
    (`430582f768e2b6b5` → `02d8bea07128f1da`) y el daño desapareció de la fuente.

    Lo que fija ahora son las dos mitades de esa situación:

      * el manifiesto no declara celdas —si volviera a declararlas sobre un fichero sano,
        la guarda de obsolescencia lo pararía, y este test lo dice antes—;
      * y el fichero **de verdad** está limpio, para que «cero reparaciones» signifique
        «no hace falta» y no «dejamos de mirar».
    """
    manifiesto = manifiesto_backup_2026_06()
    assert manifiesto.celdas == (), (
        f"el manifiesto declara {len(manifiesto.celdas)} celdas sobre un backup que ya no "
        f"las necesita: eso autoriza dividir valores sanos"
    )
    assert manifiesto.columnas_vigiladas, (
        "sin columnas vigiladas nadie audita las series: vaciar reparaciones no es dejar "
        "de mirar"
    )

    df = pd.read_parquet(BACKUP)
    out, reporte = validate_and_repair_macro_scale(df, manifiesto)
    assert reporte["n_celdas_reparadas"] == 0
    assert reporte["n_filas_entrada"] == reporte["n_filas_salida"]

    for columna in reporte["columnas_auditadas"]:
        s = out[["fecha", columna]].dropna().copy()
        s[columna] = pd.to_numeric(s[columna], errors="coerce")
        s = s.dropna().sort_values("fecha")
        r = (np.log(s[columna].astype(float)) - np.log(s[columna].astype(float).shift(1))).abs()
        assert int((r > 0.7).sum()) == 0, f"{columna} trae empalmes sin declarar"

    mxn = pd.to_numeric(out["fxrt_spot_usdmxn_mex_d_usdmxn"], errors="coerce").dropna()
    clp = pd.to_numeric(out["fxrt_spot_usdclp_chl_d_usdclp"], errors="coerce").dropna()
    assert 10 < mxn.max() < 40, f"USD/MXN fuera de rango plausible: max={mxn.max()}"
    assert 400 < clp.max() < 1500, f"USD/CLP fuera de rango plausible: max={clp.max()}"


# ---------------------------------------------------------------------------
# R2 (CXD-771): bypasses del fail-closed que la primera version dejaba pasar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valor, motivo",
    [
        (-5.0, "negativo: log da NaN y NaN > umbral es False -> NO se detectaba"),
        (0.0, "cero: log da -inf"),
        (float("inf"), "infinito"),
    ],
)
def test_non_positive_or_non_finite_values_are_refused(valor, motivo) -> None:
    """Un valor imposible para un precio para la función, no la atraviesa.

    El caso negativo es el que de verdad se colaba: `np.log(-5)` es NaN y toda comparación
    con NaN es False, así que la fila pasaba el detector como si fuese sana. Medido antes
    de escribir esto, no supuesto.
    """
    df = _serie_limpia()
    df.loc[7, COL] = valor
    with pytest.raises(MacroScaleError, match="no finitos o <= 0"):
        validate_and_repair_macro_scale(df, ManifiestoEscala((), (COL,)))


def test_non_numeric_values_are_refused_instead_of_dropped() -> None:
    """Una cotización que llegó como texto se denuncia; no se descarta en silencio.

    `to_numeric(errors="coerce")` + `dropna()` la haría desaparecer y la serie parecería
    completa — la misma familia de silencio que este módulo existe para cerrar.
    """
    df = _serie_limpia().astype({COL: object})
    df.loc[7, COL] = "17,53"
    with pytest.raises(MacroScaleError, match="no numéricos"):
        validate_and_repair_macro_scale(df, ManifiestoEscala((), (COL,)))


def test_a_duplicated_declaration_is_refused() -> None:
    """Declarar dos veces la misma (columna, fecha) es ambigüedad, no redundancia.

    Un dict dejaría ganar a la última y la otra —quizá con otro factor— desaparecería.
    """
    dup = ManifiestoEscala(
        celdas=(
            CeldaDeclarada(COL, date(2026, 6, 6), 1e4, "a"),
            CeldaDeclarada(COL, date(2026, 6, 6), 1e2, "b"),
        ),
        columnas_vigiladas=(COL,),
    )
    with pytest.raises(MacroScaleError, match="declarada dos veces"):
        validate_and_repair_macro_scale(_con_empalme(5, 8), dup)


@pytest.mark.parametrize("factor", [0.0, -10.0, 1.0, float("nan")])
def test_an_unusable_factor_is_refused(factor) -> None:
    """Factor <=0, ==1 o no finito: no repara nada o produce imposibles."""
    mal = ManifiestoEscala(
        celdas=(CeldaDeclarada(COL, date(2026, 6, 6), factor, "prueba"),),
        columnas_vigiladas=(COL,),
    )
    with pytest.raises(MacroScaleError):
        validate_and_repair_macro_scale(_con_empalme(5, 8), mal)


def test_a_declaration_without_evidence_is_refused() -> None:
    """Sin motivo escrito no es una excepción declarativa, es una heredada."""
    sin_ev = ManifiestoEscala(
        celdas=(CeldaDeclarada(COL, date(2026, 6, 6), 1e4, "   "),),
        columnas_vigiladas=(COL,),
    )
    with pytest.raises(MacroScaleError, match="SIN evidencia"):
        validate_and_repair_macro_scale(_con_empalme(5, 8), sin_ev)


def test_a_duplicated_date_in_the_frame_is_refused() -> None:
    """Si la fecha aparece dos veces, se repararían dos filas y el reporte contaría una.

    La provenance mentiría sobre su propio alcance, que es peor que no tenerla.
    """
    df = _con_empalme(5, 8)
    df = pd.concat([df, df.iloc[[6]]], ignore_index=True)
    fechas = ["2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09"]
    with pytest.raises(MacroScaleError, match="debe afectar exactamente a 1"):
        validate_and_repair_macro_scale(df, _manifiesto(fechas))


def test_the_report_carries_row_counts() -> None:
    """La provenance declara cardinalidad de entrada y salida, no sólo celdas."""
    df = _con_empalme(5, 8)
    fechas = ["2026-06-06", "2026-06-07", "2026-06-08", "2026-06-09"]
    _, reporte = validate_and_repair_macro_scale(df, _manifiesto(fechas))
    assert reporte["n_filas_entrada"] == len(df)
    assert reporte["n_filas_salida"] == len(df)
    assert all(c["n_filas_afectadas"] == 1 for c in reporte["celdas_reparadas"])
