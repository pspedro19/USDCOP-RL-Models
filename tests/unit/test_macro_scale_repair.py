# -*- coding: utf-8 -*-
"""La reparación de escala repara lo declarado y **falla** ante todo lo demás.

Un transformador que "limpia datos sucios" es peligroso por defecto: en cuanto acepta
decidir solo, se convierte en una licencia para dividir números por 10.000 sin que nadie
mire. Por eso lo que se fija aquí no es que repare —eso es lo fácil— sino sus tres formas
de negarse:

  * un empalme que nadie declaró          → error
  * una declaración que ya no hace falta  → error
  * una reparación que no arregla la serie → error

El caso del manifiesto obsoleto es el que suele faltar. Sin él, la lista de excepciones
sobrevive a su motivo y nadie se entera de que dejó de ser necesaria.
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
def test_the_declared_manifest_matches_the_real_backup() -> None:
    """El manifiesto no es teórico: se ejerce contra el fichero real.

    Los tests de arriba usan series sintéticas, que prueban la lógica pero no que las 15
    celdas declaradas sean las que de verdad están mal. Esto último sólo lo demuestra el
    backup, y si algún día se regenera con otro daño, este test lo dice.
    """
    df = pd.read_parquet(BACKUP)
    out, reporte = validate_and_repair_macro_scale(df, manifiesto_backup_2026_06())

    assert reporte["n_celdas_reparadas"] == 15, (
        f"se repararon {reporte['n_celdas_reparadas']} celdas y el manifiesto declara 15"
    )
    for columna in reporte["columnas_auditadas"]:
        s = out[["fecha", columna]].dropna().copy()
        s[columna] = pd.to_numeric(s[columna], errors="coerce")
        s = s.dropna().sort_values("fecha")
        r = (np.log(s[columna].astype(float)) - np.log(s[columna].astype(float).shift(1))).abs()
        assert int((r > 0.7).sum()) == 0, f"{columna} conserva empalmes tras reparar"

    # Rangos plausibles: sin esto, dividir por 10^6 también daría «cero saltos».
    mxn = pd.to_numeric(out["fxrt_spot_usdmxn_mex_d_usdmxn"], errors="coerce").dropna()
    clp = pd.to_numeric(out["fxrt_spot_usdclp_chl_d_usdclp"], errors="coerce").dropna()
    assert 10 < mxn.max() < 40, f"USD/MXN fuera de rango plausible: max={mxn.max()}"
    assert 400 < clp.max() < 1500, f"USD/CLP fuera de rango plausible: max={clp.max()}"
