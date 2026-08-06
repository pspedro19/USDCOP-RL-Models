# -*- coding: utf-8 -*-
"""Reparación DECLARADA de empalmes de escala en series FX macro.

QUÉ PASÓ
--------
Cotizaciones de USD/MXN y USD/CLP llegaron sin separador decimal: el valor se multiplicó
por 10^4 y 10^2 respectivamente —el número de decimales de cada par—. En
`data/backups/seeds/macro_indicators_daily_backup.parquet` el daño está **acotado**: entra
el 2026-06-29, sale el 2026-07-08, más un día suelto (2026-07-26 en MXN). Quince celdas de
1.729 filas.

No es la primera vez: el mismo bug ocurrió el 2026-01-27 y se "cerró" el 2026-07-21
reparando el artefacto derivado (`MACRO_DAILY_CLEAN`) sin tocar la fuente. Una regeneración
posterior lo trajo de vuelta. Por eso esto vive aquí, en el camino de ingesta, y no en un
script que arregla un parquet.

POR QUÉ NO SE LIMPIA "AUTOMÁTICAMENTE"
-------------------------------------
Detectar un salto es fácil; decidir que un salto es un bug y no un evento de mercado es una
afirmación sobre el mundo. Un devaluación real de un 40% existe; una de 9.965× no. Esta
función **no adivina**: sólo repara lo que un manifiesto declara célula por célula, con su
factor y su evidencia. Todo lo demás que parezca un empalme la hace **fallar**.

Esa asimetría es deliberada:

  * celda declarada y efectivamente rota  → se repara
  * salto NO declarado                    → `MacroScaleError` (fail-closed)
  * celda declarada que NO está rota      → `MacroScaleError` (el manifiesto envejeció)
  * valor no numérico, no finito o <= 0   → `MacroScaleError` (R2, ver `_serie`)
  * manifiesto ambiguo o sin evidencia    → `MacroScaleError` (R2, ver `validar`)

El tercer caso importa tanto como el segundo. Un manifiesto que sigue declarando
reparaciones ya innecesarias es una licencia abierta para dividir números por 10.000, y
nadie se enteraría de que dejó de hacer falta.

Contract: CTR-DQ-MACRO-001 (mismo que `tests/regression/test_macro_clean_fx_scale.py`)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

#: Umbral de detección: |log-ret| > 0.7 es un movimiento diario de +100%. En un par FX
#: mayor eso no es un evento de mercado, es un bug decimal disfrazado. Mismo umbral que
#: usa `tests/regression/test_macro_clean_fx_scale.py`, a propósito: si aquí y allí
#: divergieran, una serie podría pasar la reparación y fallar el gate.
UMBRAL_LOG_RET = 0.7


class MacroScaleError(RuntimeError):
    """La serie no coincide con lo declarado: se para en vez de adivinar."""


@dataclass(frozen=True)
class CeldaDeclarada:
    """Una celda concreta que se repara, con por qué se cree que está rota."""

    columna: str
    fecha: date
    factor: float
    evidencia: str


@dataclass(frozen=True)
class ManifiestoEscala:
    """El conjunto cerrado de reparaciones autorizadas."""

    celdas: tuple[CeldaDeclarada, ...]
    #: Columnas que se auditan aunque no tengan celdas declaradas. Sin esto, retirar la
    #: última celda de una columna la sacaría también de la vigilancia — y entonces un
    #: empalme nuevo en esa serie pasaría inadvertido.
    columnas_vigiladas: tuple[str, ...] = field(default=())

    def columnas(self) -> tuple[str, ...]:
        declaradas = {c.columna for c in self.celdas}
        return tuple(sorted(declaradas | set(self.columnas_vigiladas)))

    def para(self, columna: str) -> dict[date, CeldaDeclarada]:
        """Celdas de una columna, indexadas por fecha, rechazando duplicados.

        Un dict silenciaría una segunda declaración de la misma `(columna, fecha)`: la
        última ganaría y la otra —quizá con otro factor— desaparecería sin aviso.
        """
        propias = [c for c in self.celdas if c.columna == columna]
        indexadas: dict[date, CeldaDeclarada] = {}
        for celda in propias:
            if celda.fecha in indexadas:
                raise MacroScaleError(
                    f"{columna}: la fecha {celda.fecha} está declarada dos veces "
                    f"(factores {indexadas[celda.fecha].factor} y {celda.factor}). Un "
                    f"manifiesto ambiguo no se resuelve eligiendo uno"
                )
            indexadas[celda.fecha] = celda
        return indexadas

    def validar(self) -> None:
        """Metadata utilizable: factor finito, > 0 y != 1; evidencia no vacía.

        Un factor 1 no repara nada y haría creer que sí; uno <= 0 o no finito produce
        valores imposibles. Y una entrada sin evidencia es una excepción sin motivo, que
        es como empiezan las allowlist que nadie sabe por qué existen.
        """
        for celda in self.celdas:
            f = celda.factor
            if not isinstance(f, (int, float)) or isinstance(f, bool) or not np.isfinite(f):
                raise MacroScaleError(f"{celda.columna} {celda.fecha}: factor no numérico/finito: {f!r}")
            if f <= 0 or f == 1:
                raise MacroScaleError(
                    f"{celda.columna} {celda.fecha}: factor {f} inservible (<=0 o ==1: no repara nada)"
                )
            if not str(celda.evidencia).strip():
                raise MacroScaleError(
                    f"{celda.columna} {celda.fecha}: celda declarada SIN evidencia. Una "
                    f"excepción sin motivo escrito no es declarativa, es heredada"
                )


def _serie(frame: pd.DataFrame, columna: str, col_fecha: str) -> pd.DataFrame:
    """Serie numérica ordenada, con las filas inservibles RECHAZADAS, no descartadas.

    `pd.to_numeric(errors="coerce")` convierte en NaN lo que no es número y un `dropna()`
    lo hace desaparecer sin ruido — una cotización que llegó como `"17,53"` o vacía se
    esfumaría y la serie parecería sana. Peor aún con valores ≤ 0: `np.log(-5)` da NaN, y
    `NaN > 0.7` es **False**, así que un precio negativo NO dispara el detector y pasa
    como bueno. Medido, no supuesto (CXD-771).

    Aquí las filas presentes tienen que ser numéricas, finitas y positivas; si no, se para.
    """
    s = frame[[col_fecha, columna]].copy()
    presentes = s[columna].notna()
    crudo = s.loc[presentes, columna]
    numerico = pd.to_numeric(crudo, errors="coerce")
    no_numericas = crudo[numerico.isna()]
    if len(no_numericas):
        muestra = [repr(v) for v in no_numericas.head(3)]
        raise MacroScaleError(
            f"{columna}: {len(no_numericas)} valores no numéricos (p.ej. {muestra}). "
            f"Descartarlos en silencio dejaría una serie que parece sana"
        )
    malos = numerico[~np.isfinite(numerico) | (numerico <= 0)]
    if len(malos):
        muestra = [float(v) for v in malos.head(3)]
        raise MacroScaleError(
            f"{columna}: {len(malos)} valores no finitos o <= 0 (p.ej. {muestra}). Un "
            f"precio negativo produce log-ret NaN y NaN > umbral es False: pasaría el "
            f"detector sin que nadie lo viera"
        )
    s = s.loc[presentes].copy()
    s[columna] = numerico
    return s.sort_values(col_fecha).reset_index(drop=True)


def _saltos(serie: pd.Series) -> list[int]:
    """Índices donde |log-ret| supera el umbral."""
    valores = serie.astype(float)
    if len(valores) < 2:
        return []
    r = np.log(valores) - np.log(valores.shift(1))
    return [int(i) for i in np.flatnonzero(np.abs(r.to_numpy()) > UMBRAL_LOG_RET)]


def validate_and_repair_macro_scale(
    frame: pd.DataFrame,
    manifiesto: ManifiestoEscala,
    *,
    col_fecha: str = "fecha",
) -> tuple[pd.DataFrame, dict]:
    """Devuelve `(frame_reparado, reporte)` o lanza `MacroScaleError`.

    Función **pura**: no lee ficheros, no toca la base de datos, no escribe nada. Recibe el
    frame ya cargado y devuelve otro. Así el mismo código sirve para el loader de ingesta,
    para un test y para una comprobación manual, sin que ninguno arrastre efectos del otro.

    El `reporte` lleva la provenance que pidió la revisión: qué columnas se auditaron, qué
    celdas se tocaron, con qué factor y qué valor tenían antes y después.
    """
    if col_fecha not in frame.columns:
        raise MacroScaleError(f"el frame no trae la columna de fecha {col_fecha!r}")

    manifiesto.validar()
    out = frame.copy()
    out[col_fecha] = pd.to_datetime(out[col_fecha])
    reporte: dict = {
        "n_filas_entrada": int(len(frame)),
        "columnas_auditadas": [],
        "celdas_reparadas": [],
        "sin_reparar": [],
    }

    for columna in manifiesto.columnas():
        if columna not in out.columns:
            raise MacroScaleError(
                f"la columna vigilada {columna!r} no está en el frame: el manifiesto "
                f"describe una serie que ya no existe, así que no puede vigilarla"
            )
        declaradas = manifiesto.para(columna)
        serie = _serie(out, columna, col_fecha)
        reporte["columnas_auditadas"].append(columna)

        # 1) Toda celda declarada tiene que estar EFECTIVAMENTE rota.
        indices_por_fecha = {d.date(): i for i, d in enumerate(serie[col_fecha])}
        # Un salto marca una FRONTERA, no una celda rota. El tramo dañado va desde el
        # salto de entrada (inclusive) hasta el de vuelta (EXCLUSIVE): el día del retorno
        # es el primer día sano, no el último malo. Si nunca vuelve, el tramo llega al
        # final de la serie.
        #
        # Una versión anterior marcaba además cada índice de salto por separado, lo que
        # incluía el día del retorno entre los rotos. Lo destapó el propio fail-closed al
        # correr contra el backup real: acusó un "empalme no declarado" el 2026-07-08, que
        # es justamente el día en que CLP vuelve a 934.50. El gate encontró el bug de su
        # propio autor, que es exactamente para lo que sirve fallar cerrado.
        rotas: set[date] = set()
        saltos = _saltos(serie[columna])
        for k in range(0, len(saltos), 2):
            ini = saltos[k]
            fin = saltos[k + 1] if k + 1 < len(saltos) else len(serie)
            for j in range(ini, fin):
                rotas.add(serie[col_fecha].iloc[j].date())

        obsoletas = sorted(f for f in declaradas if f not in rotas)
        if obsoletas:
            raise MacroScaleError(
                f"{columna}: el manifiesto declara reparación para {obsoletas} pero esas "
                f"celdas ya NO presentan empalme. Un manifiesto que envejece es una "
                f"licencia abierta para dividir valores sanos: retirar esas entradas"
            )

        # 2) Todo lo roto tiene que estar declarado.
        no_declaradas = sorted(f for f in rotas if f not in declaradas)
        if no_declaradas:
            raise MacroScaleError(
                f"{columna}: empalme de escala NO declarado en {no_declaradas[:5]}"
                f"{' …' if len(no_declaradas) > 5 else ''} ({len(no_declaradas)} celdas). "
                f"La fuente se rompió de una forma que nadie ha revisado; parar es lo "
                f"correcto — declarar la celda con su factor y su evidencia es un acto "
                f"humano, no automático"
            )

        # 3) Reparar exactamente lo declarado.
        for fecha, celda in sorted(declaradas.items()):
            fila = indices_por_fecha[fecha]
            antes = float(serie[columna].iloc[fila])
            despues = antes / celda.factor
            mascara = (out[col_fecha].dt.date == fecha) & out[columna].notna()
            n_filas = int(mascara.sum())
            if n_filas != 1:
                # Una fecha que aparece dos veces repararía dos filas mientras el reporte
                # cuenta una: la provenance mentiría sobre su propio alcance. Y una fecha
                # que no aparece señala un manifiesto desalineado con el frame.
                raise MacroScaleError(
                    f"{columna} {fecha}: la reparación afectaría a {n_filas} filas y debe "
                    f"afectar exactamente a 1. Con fechas duplicadas el reporte contaría "
                    f"menos celdas de las que toca"
                )
            out.loc[mascara, columna] = despues
            reporte["celdas_reparadas"].append(
                {
                    "columna": columna,
                    "fecha": fecha.isoformat(),
                    "factor": celda.factor,
                    "antes": antes,
                    "despues": despues,
                    "n_filas_afectadas": n_filas,
                    "evidencia": celda.evidencia,
                }
            )

    # 4) Post-check: tras reparar no puede quedar ningún empalme.
    for columna in manifiesto.columnas():
        serie = _serie(out, columna, col_fecha)
        restantes = _saltos(serie[columna])
        if restantes:
            fechas = [serie[col_fecha].iloc[i].date().isoformat() for i in restantes]
            raise MacroScaleError(
                f"{columna}: tras aplicar el manifiesto SIGUEN quedando empalmes en "
                f"{fechas}. El factor declarado no explica el daño; no se publica una "
                f"serie a medio reparar"
            )

    reporte["n_celdas_reparadas"] = len(reporte["celdas_reparadas"])
    reporte["n_filas_salida"] = int(len(out))
    if reporte["n_filas_salida"] != reporte["n_filas_entrada"]:
        # La reparación cambia VALORES, nunca la cardinalidad. Si el conteo se movió,
        # algo perdió o duplicó filas y el frame ya no es el que entró.
        raise MacroScaleError(
            f"la reparación cambió el número de filas: "
            f"{reporte['n_filas_entrada']} -> {reporte['n_filas_salida']}"
        )
    return out, reporte


def manifiesto_backup_2026_06() -> ManifiestoEscala:
    """El manifiesto medido sobre `macro_indicators_daily_backup.parquet`.

    Quince celdas: 8 en MXN (×10⁴) y 7 en CLP (×10²). La ventana 2026-06-29..07-07 más el
    día suelto 2026-07-26 en MXN — que importa porque prueba que la fuente **seguía**
    produciendo el bug a finales de julio: reparar el pasado sin arreglar el extractor lo
    repetiría.

    Evidencia del factor, y es lo que lo hace declarable en vez de adivinado: los valores
    reparados caen entre los vecinos sanos. MXN pasa a 17.39–17.55 con 17.5326 el día antes
    y 17.5814 el día después; CLP a 921–930 con 922.70 antes y 934.50 después.
    """
    mxn = "fxrt_spot_usdmxn_mex_d_usdmxn"
    clp = "fxrt_spot_usdclp_chl_d_usdclp"
    ventana = ["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
               "2026-07-03", "2026-07-06", "2026-07-07"]
    ev_mxn = ("separador decimal perdido (4 decimales); reparado cae entre 17.5326 del "
              "2026-06-26 y 17.5814 del 2026-07-08")
    ev_clp = ("separador decimal perdido (2 decimales); reparado cae entre 922.70 del "
              "2026-06-26 y 934.50 del 2026-07-08")
    celdas: list[CeldaDeclarada] = []
    for f in ventana:
        celdas.append(CeldaDeclarada(mxn, date.fromisoformat(f), 1e4, ev_mxn))
        celdas.append(CeldaDeclarada(clp, date.fromisoformat(f), 1e2, ev_clp))
    celdas.append(
        CeldaDeclarada(
            mxn, date(2026, 7, 26), 1e4,
            "pico aislado de un solo día, mismo factor; prueba que el extractor seguía "
            "produciendo el defecto un mes después del primer tramo",
        )
    )
    return ManifiestoEscala(celdas=tuple(celdas), columnas_vigiladas=(mxn, clp))
